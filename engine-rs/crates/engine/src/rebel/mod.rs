//! ReBeL-style public-belief search.
//!
//! This module is deliberately separate from ordinary perfect-information
//! MCTS. The search runs from one public information set and returns a single
//! distribution over the acting (observer) side's public legal actions.
//!
//! ## What the search actually computes
//!
//! For each candidate root action it estimates the action's value by a
//! **turn-aware policy-improvement rollout**: apply the action to each weighted
//! belief particle, then continue the modeled game forward under the network's
//! own greedy policy (each mover acting on *its own* observation) for up to
//! `max_depth` modeled-turn steps, and bootstrap the leaf with the network
//! value, sign-converted to the observer's perspective. Averaging those
//! returns over the belief particles gives one public action-value vector; the
//! root policy is a temperature softmax over it.
//!
//! This is one step of policy iteration: evaluate `Q^pi(s, a)` for each root
//! action under the current (network) policy, then act greedily. That makes the
//! search a genuine improvement operator over the raw policy net (the property
//! the earlier one-ply value read did *not* have), and it removes the old
//! degenerate leaf that scored every turn-ending action exactly 0 (a
//! non-terminal state with no observer actions used to fall back to a 0 leaf;
//! the rollout now continues through the turn handover and evaluates the
//! opponent's resulting position instead).
//!
//! ## Honest scope
//!
//! Rollouts run over *determinized* particles, so the opponent reasons about a
//! fully-instantiated world rather than its own belief. This remains PIMC-style
//! (strategy fusion is not eliminated); it is **not** equilibrium CFR. Full
//! public-belief CFR with regret tables keyed by public information set is the
//! tracked long-term target (see `docs/ai-research/scoping/rebel-e2e-scoping.md`).
//! The network/observation value is the only leaf source — the
//! information-incorrect heuristic rollout leaf has been removed entirely.

use serde::{Deserialize, Serialize};

use crate::belief::PublicBeliefState;
use crate::core::constants::SideId;
use crate::core::random::{with_rng, Rng};
use crate::core::state::{CurrentSide, GameState, Phase};
use crate::dispatcher::{
    advance_modeled_turn_step, advance_opponent_turn_step, advance_player_ai_turn_step,
    get_forced_attack_coin_results, state_fingerprint,
};
use crate::inference::InferenceSession;
use crate::mcts::math::mcts_terminal_value;
use crate::policy::actions::enumerate_legal_ai_actions;
use crate::policy::observation::build_public_observation;
use crate::policy::types::{LegalAiAction, PublicObservation};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RebelSearchConfig {
    /// Independent chance re-samples per (particle, root action). Each rollout
    /// re-draws coin flips / shuffles, so >1 averages over chance. Clamped to
    /// at least 1.
    pub iterations: u32,
    /// Lookahead horizon, in modeled-turn steps taken *after* the root action,
    /// before the network value bootstraps the leaf. Clamped to at least 1.
    /// This is the field the previous implementation declared but never read.
    pub max_depth: u32,
    /// Weight in [0, 1] for blending the network's root policy prior into the
    /// search policy. 0 = pure search policy.
    pub neural_policy_weight: f64,
    /// Weight in [0, 1] for blending the network's root value into
    /// `public_belief_value`. 0 = pure policy-weighted search value.
    pub neural_value_weight: f64,
    /// Softmax temperature for turning root action values into the root policy
    /// target. Smaller = peakier. Must be > 0 (non-positive falls back to 1.0).
    pub policy_temperature: f64,
    /// On-wire label recorded with each row. Describes the algorithm honestly.
    pub algorithm: String,
}

impl Default for RebelSearchConfig {
    fn default() -> Self {
        RebelSearchConfig {
            iterations: 2,
            max_depth: 8,
            neural_policy_weight: 0.0,
            neural_value_weight: 0.0,
            policy_temperature: 0.5,
            algorithm: "public-belief-policy-improvement-v2".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RebelSearchDiagnostics {
    pub information_set_key: String,
    pub particle_count: usize,
    pub legal_action_count: usize,
    /// Total network evaluations performed for this decision (rollout steps +
    /// leaf bootstraps + optional root prior). Named for schema continuity with
    /// the binary's timing counters.
    pub particle_action_evaluations: usize,
    /// Always 0: the information-incorrect heuristic rollout leaf was removed.
    /// Retained so existing timing/serialization consumers keep working.
    pub rollout_leaf_calls: usize,
    /// Lookahead horizon actually used (`max(1, config.max_depth)`).
    pub max_depth: u32,
    /// Number of rollouts that hit a non-terminal dead-end before any network
    /// leaf could be evaluated (and therefore fell back to 0). Should be 0 in
    /// normal play; a non-zero count flags states the modeled forward step
    /// cannot advance.
    pub degenerate_leaf_count: usize,
    pub neural_policy_weight: f64,
    pub neural_value_weight: f64,
    pub neural_value: Option<f64>,
    pub search_iterations: u32,
    pub policy_entropy: f64,
    pub particle_action_agreement: f64,
    pub aggregate_q_variance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BeliefSearchResult {
    pub root_policy: Vec<f64>,
    pub root_action_values: Vec<f64>,
    pub public_belief_value: f64,
    pub private_state_values: Vec<f64>,
    pub sampled_action_index: usize,
    /// Always false now: the heuristic rollout leaf was removed, so no row can
    /// be contaminated by it. Retained for downstream schema/replay continuity
    /// (`rolloutLeafUsed`).
    pub rollout_leaf_used: bool,
    pub diagnostics: RebelSearchDiagnostics,
    pub legal_actions: Vec<LegalAiAction>,
    pub search_algorithm: String,
}

pub fn run_public_belief_search(
    belief: &PublicBeliefState,
    config: &RebelSearchConfig,
    rng: &mut Rng,
) -> BeliefSearchResult {
    let action_count = belief.legal_actions.len();
    let observer = belief.observer_side;
    if action_count == 0 {
        return empty_result(belief, config);
    }

    // The network/observation value is the only leaf source. There is no
    // rollout fallback any more, so refuse to run without a model rather than
    // silently emitting zeroed targets into the training data.
    let session = match crate::inference::global() {
        Some(session) => session,
        None => panic!(
            "ReBeL public-belief search requires a loaded inference model; the \
             information-incorrect heuristic rollout leaf was removed, so there is \
             no fallback value source. Load an ONNX session before searching."
        ),
    };

    let iterations = config.iterations.max(1) as usize;
    let max_depth = config.max_depth.max(1) as usize;
    let particle_count = belief.particles.len();
    let belief_features = belief.belief_features.vector.clone();

    // One rollout per (particle, root action, chance iteration). All rollouts of
    // this decision advance in LOCKSTEP so the network decision/leaf queries can
    // be issued as one batched call per depth wave — restoring the large-batch
    // GPU utilization the throughput sweep tuned for (a per-row sequential
    // pattern cannot fill batches). Each rollout carries its own RNG seeded from
    // the particle's content-stable label, so results are deterministic and
    // invariant to particle ordering, independent of how chance resolves.
    let mut rollouts: Vec<Rollout> =
        Vec::with_capacity(particle_count * action_count * iterations);
    let mut degenerate = 0usize;
    for (particle_index, particle) in belief.particles.iter().enumerate() {
        let weight = particle.weight.max(0.0);
        for action_index in 0..action_count {
            let action = &belief.legal_actions[action_index];
            for iter_index in 0..iterations {
                let mut roll_rng = Rng::from_seed(
                    format!("{}:a{}:i{}", particle.rng_seed_label, action_index, iter_index),
                    "rebel-rollout",
                );
                let state = with_rng_borrow(&mut roll_rng, || {
                    let forced = get_forced_attack_coin_results(&particle.game_state);
                    advance_modeled_turn_step(&particle.game_state, observer, action, forced)
                });
                let value = if state.game_over {
                    Some(mcts_terminal_value(&state, observer))
                } else {
                    None
                };
                rollouts.push(Rollout {
                    particle: particle_index,
                    action: action_index,
                    weight,
                    state,
                    rng: roll_rng,
                    value,
                    last_value: None,
                });
            }
        }
    }

    let mut total_evals = 0usize;
    // Lockstep: advance every active rollout one modeled step per wave, batching
    // the wave's decision/leaf predicts. Stop early once all rollouts resolved.
    for _ in 0..max_depth {
        advance_rollout_wave(
            &mut rollouts,
            observer,
            &session,
            &belief_features,
            &mut total_evals,
            &mut degenerate,
            false,
        );
        if rollouts.iter().all(|r| r.value.is_some()) {
            break;
        }
    }
    // Horizon pass: value any still-active rollout by the network leaf.
    advance_rollout_wave(
        &mut rollouts,
        observer,
        &session,
        &belief_features,
        &mut total_evals,
        &mut degenerate,
        true,
    );

    // Aggregate: per (particle, action) mean over chance iterations, then a
    // particle-weighted mean over particles. Canonical particle/action order
    // keeps the float reduction stable (and identical under particle
    // permutation when per-particle values match).
    let mut pa_sum = vec![vec![0.0f64; action_count]; particle_count];
    for r in &rollouts {
        pa_sum[r.particle][r.action] += r.value.unwrap_or(0.0);
    }
    let mut action_values = vec![0.0f64; action_count];
    let mut total_weight = 0.0f64;
    let mut private_state_values = Vec::with_capacity(particle_count);
    let mut particle_best = Vec::with_capacity(particle_count);
    for (particle_index, particle) in belief.particles.iter().enumerate() {
        let weight = particle.weight.max(0.0);
        let per_particle: Vec<f64> = pa_sum[particle_index]
            .iter()
            .map(|sum| sum / iterations as f64)
            .collect();
        for action_index in 0..action_count {
            action_values[action_index] += weight * per_particle[action_index];
        }
        total_weight += weight;
        let best = argmax_f64(&per_particle);
        particle_best.push(best);
        private_state_values.push(per_particle.get(best).copied().unwrap_or(0.0));
    }
    if total_weight > 0.0 {
        for value in action_values.iter_mut() {
            *value /= total_weight;
        }
    }

    // Root policy = temperature softmax over the improved (lookahead) action
    // values. Unlike the previous regret-matching-over-a-constant-vector
    // collapse to a near one-hot argmax, this keeps the relative value ordering
    // in the distilled target.
    let search_policy = softmax_policy(&action_values, config.policy_temperature);

    let mut neural_value = None;
    let root_policy = if config.neural_policy_weight > 0.0 || config.neural_value_weight > 0.0 {
        match session.predict_v3_with_belief(
            &belief.public_observation,
            &belief.legal_actions,
            Some(&belief_features),
        ) {
            Ok(prediction) => {
                total_evals += 1;
                neural_value = Some(prediction.value as f64);
                if config.neural_policy_weight > 0.0 {
                    mix_policy(&search_policy, &prediction.probs, config.neural_policy_weight)
                } else {
                    search_policy.clone()
                }
            }
            Err(_) => search_policy.clone(),
        }
    } else {
        search_policy.clone()
    };

    let sampled_action_index = sample_policy(&root_policy, rng);
    let policy_weighted_value: f64 = root_policy
        .iter()
        .zip(action_values.iter())
        .map(|(p, q)| p * q)
        .sum();
    let public_belief_value = match neural_value {
        Some(v) if config.neural_value_weight > 0.0 => {
            let w = config.neural_value_weight.clamp(0.0, 1.0);
            (1.0 - w) * policy_weighted_value + w * v
        }
        _ => policy_weighted_value,
    };

    let top = argmax_f64(&action_values);
    let agreement = if particle_best.is_empty() {
        0.0
    } else {
        particle_best.iter().filter(|&&i| i == top).count() as f64 / particle_best.len() as f64
    };

    BeliefSearchResult {
        root_policy: root_policy.clone(),
        root_action_values: action_values.clone(),
        public_belief_value,
        private_state_values,
        sampled_action_index,
        rollout_leaf_used: false,
        diagnostics: RebelSearchDiagnostics {
            information_set_key: belief.public_history_digest.clone(),
            particle_count: belief.particles.len(),
            legal_action_count: action_count,
            particle_action_evaluations: total_evals,
            rollout_leaf_calls: 0,
            max_depth: max_depth as u32,
            degenerate_leaf_count: degenerate,
            neural_policy_weight: config.neural_policy_weight,
            neural_value_weight: config.neural_value_weight,
            neural_value,
            search_iterations: iterations as u32,
            policy_entropy: entropy(&root_policy),
            particle_action_agreement: agreement,
            aggregate_q_variance: variance(&action_values),
        },
        legal_actions: belief.legal_actions.clone(),
        search_algorithm: config.algorithm.clone(),
    }
}

/// A single in-flight rollout: a particle's world after one root action,
/// advanced under the network's greedy policy. All rollouts of a decision are
/// stepped in lockstep so their network queries batch (see
/// `advance_rollout_wave`). `value` is the resolved observer-perspective return
/// (terminal or network-leaf); `last_value` carries the most recent network
/// value as a fallback for non-evaluable leaves.
struct Rollout {
    particle: usize,
    action: usize,
    weight: f64,
    state: GameState,
    rng: Rng,
    value: Option<f64>,
    last_value: Option<f64>,
}

/// One lockstep wave over all still-active rollouts. Classifies each rollout's
/// current node, batches the decision/leaf network queries into a single
/// `predict_v3_batch_with_belief` call, then advances each rollout one modeled
/// step (greedy network action, or engine auto-resolve for non-decision nodes).
///
/// When `bootstrap` is true this is the horizon pass: active rollouts are
/// *valued* by their network leaf (sign-converted to the observer) and finalized
/// rather than advanced. Every per-rollout engine call that consumes the ambient
/// RNG runs inside a `with_rng` scope keyed by that rollout's own RNG, so the
/// result is deterministic and order-independent.
#[allow(clippy::too_many_arguments)]
fn advance_rollout_wave(
    rollouts: &mut [Rollout],
    observer: SideId,
    session: &InferenceSession,
    belief_features: &[f32],
    evals: &mut usize,
    degenerate: &mut usize,
    bootstrap: bool,
) {
    // Phase 1 — classify; gather batched predict requests + auto-resolve nodes.
    let mut req_obs: Vec<PublicObservation> = Vec::new();
    let mut req_legal: Vec<Vec<LegalAiAction>> = Vec::new();
    let mut req_rollout: Vec<usize> = Vec::new();
    let mut req_mover: Vec<SideId> = Vec::new();
    let mut auto_nodes: Vec<(usize, SideId)> = Vec::new();

    for (index, r) in rollouts.iter_mut().enumerate() {
        if r.value.is_some() {
            continue;
        }
        if r.state.game_over {
            r.value = Some(mcts_terminal_value(&r.state, observer));
            continue;
        }
        let mover = match current_side(&r.state) {
            Some(side) => side,
            None => {
                r.value = Some(
                    r.last_value
                        .unwrap_or_else(|| mcts_terminal_value(&r.state, observer)),
                );
                continue;
            }
        };
        let (legal, used) = with_rng(r.rng.clone(), || enumerate_legal_ai_actions(&r.state, mover));
        r.rng = used;

        if bootstrap {
            // Horizon: value the leaf with the network where there is a decision
            // set to feed it; otherwise carry forward the last network value.
            if legal.is_empty() {
                if r.last_value.is_none() {
                    *degenerate += 1;
                }
                r.value = Some(r.last_value.unwrap_or(0.0));
            } else {
                req_obs.push(build_public_observation(&r.state, mover));
                req_legal.push(legal);
                req_rollout.push(index);
                req_mover.push(mover);
            }
            continue;
        }

        let is_decision = r.state.phase == Phase::Play
            && r.state.pending_player_choice.is_none()
            && legal.len() > 1;
        if is_decision {
            req_obs.push(build_public_observation(&r.state, mover));
            req_legal.push(legal);
            req_rollout.push(index);
            req_mover.push(mover);
        } else {
            auto_nodes.push((index, mover));
        }
    }

    // Phase 2 — one batched network call for every UNIQUE decision/leaf node this
    // wave. Many requests share byte-identical model inputs: at observer decision
    // nodes every particle sees the same masked public observation (the
    // public-belief info-set property), and chance-iteration copies coincide until
    // chance diverges. Deduplicating collapses the redundant forward passes
    // (~60% of leaves at the measured relational config). Identical inputs
    // deterministically yield identical outputs, so every distilled target is
    // byte-for-byte unchanged. belief_features is constant across the wave, so the
    // dedup key is just (observation, legal_actions); first-occurrence order is
    // canonical because req_obs is built in canonical rollout order, keeping the
    // unique batch deterministic and invariant to particle ordering.
    if !req_obs.is_empty() {
        let mut key_to_slot: std::collections::HashMap<String, usize> =
            std::collections::HashMap::with_capacity(req_obs.len());
        let mut unique_req: Vec<usize> = Vec::new();
        let mut slot_of_req: Vec<usize> = Vec::with_capacity(req_obs.len());
        for k in 0..req_obs.len() {
            let key = serde_json::to_string(&(&req_obs[k], &req_legal[k]))
                .expect("serialize rebel leaf dedup key");
            let slot = *key_to_slot.entry(key).or_insert_with(|| {
                unique_req.push(k);
                unique_req.len() - 1
            });
            slot_of_req.push(slot);
        }
        let batch: Vec<(&PublicObservation, &[LegalAiAction], Option<&[f32]>)> = unique_req
            .iter()
            .map(|&k| (&req_obs[k], req_legal[k].as_slice(), Some(belief_features)))
            .collect();
        let predictions = match session.predict_v3_batch_with_belief(&batch) {
            Ok(predictions) => predictions,
            // Loud-fail: silently zeroing leaves corrupts both action values and
            // the derived policy/value targets.
            Err(e) => panic!("ReBeL batched leaf inference failed: {e}"),
        };
        // Count LOGICAL leaf evaluations (one per request) so the recorded
        // search-work diagnostic is identical with or without dedup; dedup only
        // reduces the count of actual network forward passes (predictions.len()).
        *evals += req_obs.len();
        for k in 0..req_obs.len() {
            let prediction = &predictions[slot_of_req[k]];
            let index = req_rollout[k];
            let mover = req_mover[k];
            let observer_value = to_observer(prediction.value as f64, mover, observer);
            if bootstrap {
                rollouts[index].value = Some(observer_value);
                continue;
            }
            let chosen = argmax_f32(&prediction.probs).min(req_legal[k].len() - 1);
            let action = req_legal[k][chosen].clone();
            let r = &mut rollouts[index];
            r.last_value = Some(observer_value);
            let before = state_fingerprint(&r.state);
            let (next, used) = with_rng(r.rng.clone(), || {
                let forced = get_forced_attack_coin_results(&r.state);
                advance_modeled_turn_step(&r.state, mover, &action, forced)
            });
            r.rng = used;
            if state_fingerprint(&next) != before {
                r.state = next;
                continue;
            }
            // The modeled step could not apply the chosen action; auto-resolve.
            let (auto, used) = with_rng(r.rng.clone(), || auto_resolve(&r.state, mover));
            r.rng = used;
            if state_fingerprint(&auto) == before {
                r.value = Some(r.last_value.unwrap());
            } else {
                r.state = auto;
            }
        }
    }

    // Phase 3 — auto-resolve non-decision nodes (advance waves only).
    for (index, mover) in auto_nodes {
        let r = &mut rollouts[index];
        let before = state_fingerprint(&r.state);
        let (auto, used) = with_rng(r.rng.clone(), || auto_resolve(&r.state, mover));
        r.rng = used;
        if state_fingerprint(&auto) == before {
            if r.last_value.is_none() {
                *degenerate += 1;
            }
            r.value = Some(r.last_value.unwrap_or(0.0));
        } else {
            r.state = auto;
        }
    }
}

/// Advance one engine AI step for `mover` using the built-in auto-resolver
/// (chance is consumed from the ambient RNG). Used for setup / pending-choice /
/// single-action nodes where there is no multi-way decision to search.
fn auto_resolve(state: &GameState, mover: SideId) -> GameState {
    let mut next = state.clone();
    let forced = get_forced_attack_coin_results(&next);
    match mover {
        SideId::Player => advance_player_ai_turn_step(&mut next, forced),
        SideId::Opponent => advance_opponent_turn_step(&mut next, forced),
    }
    next
}

fn current_side(state: &GameState) -> Option<SideId> {
    match state.current_side {
        CurrentSide::Player => Some(SideId::Player),
        CurrentSide::Opponent => Some(SideId::Opponent),
        CurrentSide::Done => None,
    }
}

/// Convert a value expressed from `mover`'s perspective into the `observer`'s
/// perspective (two-player zero-sum sign flip).
#[inline]
fn to_observer(value: f64, mover: SideId, observer: SideId) -> f64 {
    if mover == observer {
        value
    } else {
        -value
    }
}

fn empty_result(belief: &PublicBeliefState, config: &RebelSearchConfig) -> BeliefSearchResult {
    BeliefSearchResult {
        root_policy: Vec::new(),
        root_action_values: Vec::new(),
        public_belief_value: 0.0,
        private_state_values: Vec::new(),
        sampled_action_index: 0,
        rollout_leaf_used: false,
        diagnostics: RebelSearchDiagnostics {
            information_set_key: belief.public_history_digest.clone(),
            particle_count: belief.particles.len(),
            legal_action_count: 0,
            particle_action_evaluations: 0,
            rollout_leaf_calls: 0,
            max_depth: config.max_depth.max(1),
            degenerate_leaf_count: 0,
            neural_policy_weight: config.neural_policy_weight,
            neural_value_weight: config.neural_value_weight,
            neural_value: None,
            search_iterations: config.iterations.max(1),
            policy_entropy: 0.0,
            particle_action_agreement: 0.0,
            aggregate_q_variance: 0.0,
        },
        legal_actions: Vec::new(),
        search_algorithm: config.algorithm.clone(),
    }
}

/// Temperature softmax over action values. Falls back to uniform on a
/// degenerate value vector, and treats a non-positive/non-finite temperature
/// as 1.0. Always sums to 1.
fn softmax_policy(values: &[f64], temperature: f64) -> Vec<f64> {
    let k = values.len();
    if k == 0 {
        return Vec::new();
    }
    let t = if temperature > 0.0 && temperature.is_finite() {
        temperature
    } else {
        1.0
    };
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return vec![1.0 / k as f64; k];
    }
    let exps: Vec<f64> = values.iter().map(|v| ((v - max) / t).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum > 0.0 && sum.is_finite() {
        exps.iter().map(|e| e / sum).collect()
    } else {
        vec![1.0 / k as f64; k]
    }
}

fn mix_policy(search_policy: &[f64], neural_policy: &[f32], neural_weight: f64) -> Vec<f64> {
    let w = neural_weight.clamp(0.0, 1.0);
    let mut mixed = Vec::with_capacity(search_policy.len());
    for (i, search_p) in search_policy.iter().enumerate() {
        let neural_p = neural_policy.get(i).copied().unwrap_or(0.0).max(0.0) as f64;
        mixed.push((1.0 - w) * *search_p + w * neural_p);
    }
    let total: f64 = mixed.iter().sum();
    if total > 0.0 && total.is_finite() {
        for p in mixed.iter_mut() {
            *p /= total;
        }
        mixed
    } else {
        search_policy.to_vec()
    }
}

fn sample_policy(policy: &[f64], rng: &mut Rng) -> usize {
    if policy.is_empty() {
        return 0;
    }
    let mut r = rng.next_f64();
    for (index, p) in policy.iter().enumerate() {
        r -= *p;
        if r <= 0.0 {
            return index;
        }
    }
    policy.len() - 1
}

fn argmax_f64(values: &[f64]) -> usize {
    let mut best = 0usize;
    let mut best_value = f64::NEG_INFINITY;
    for (index, value) in values.iter().enumerate() {
        if *value > best_value {
            best_value = *value;
            best = index;
        }
    }
    best
}

fn argmax_f32(values: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, value) in values.iter().enumerate() {
        if *value > best_value {
            best_value = *value;
            best = index;
        }
    }
    best
}

fn with_rng_borrow<T>(rng: &mut Rng, f: impl FnOnce() -> T) -> T {
    let taken = rng.clone();
    let (out, used) = with_rng(taken, f);
    *rng = used;
    out
}

fn entropy(policy: &[f64]) -> f64 {
    policy
        .iter()
        .map(|p| if *p > 0.0 { -p * p.ln() } else { 0.0 })
        .sum()
}

fn variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief::{build_public_belief_state, BeliefBuildConfig, PublicHistory};
    use crate::core::random::{with_rng, Rng};
    use crate::headless_setup::setup_ai_vs_ai_game;
    use crate::inference::{self, InferenceSession};
    use std::sync::Once;

    /// Deterministic, public-only value oracle: hash the serialized public
    /// observation to a stable scalar in [-1, 1]. Crucially this depends ONLY
    /// on the public observation — never on particle identity, ordering, or
    /// belief features — which is the invariant the search must preserve.
    fn public_only_value(obs: &PublicObservation) -> f32 {
        let json = serde_json::to_string(obs).expect("serialize observation");
        let mut h: u64 = 1469598103934665603; // FNV-1a offset basis
        for b in json.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(1099511628211);
        }
        ((h % 20001) as f64 / 10000.0 - 1.0) as f32
    }

    static INSTALL_STUB: Once = Once::new();

    fn install_public_only_stub() {
        INSTALL_STUB.call_once(|| {
            inference::set_global(InferenceSession::new_test_stub(public_only_value));
        });
    }

    fn build_initial_belief(seed: &str, particle_count: usize) -> crate::belief::PublicBeliefState {
        let rng = Rng::from_seed(format!("rebel-test:{seed}"), "rebel-test");
        let (state, mut rng) = with_rng(rng, setup_ai_vs_ai_game);
        let history =
            PublicHistory::from_state(&state, "matikanetannhauser", "matikanetannhauser", 0);
        let config = BeliefBuildConfig {
            particle_count,
            seed_label: format!("rebel-test:{seed}"),
        };
        build_public_belief_state(&state, SideId::Player, history, &config, &mut rng)
    }

    fn test_config() -> RebelSearchConfig {
        RebelSearchConfig {
            iterations: 2,
            max_depth: 4,
            neural_policy_weight: 0.0,
            neural_value_weight: 0.0,
            policy_temperature: 0.5,
            algorithm: "public-belief-policy-improvement-v2".to_string(),
        }
    }

    /// Info-set invariance: with a value oracle that depends ONLY on the public
    /// observation, permuting the particle set must yield a bit-identical
    /// `root_policy`. Per-rollout RNG is seeded from each particle's
    /// content-stable label, so chance resolution does not depend on ordering.
    #[test]
    fn info_set_invariant_under_particle_permutation() {
        install_public_only_stub();
        let config = test_config();

        let belief = build_initial_belief("perm", 8);

        let mut search_rng_a = Rng::from_seed("rebel-test:search", "rebel-test");
        let result_a = run_public_belief_search(&belief, &config, &mut search_rng_a);

        let mut permuted = belief.clone();
        permuted.particles.reverse();
        permuted.particle_weights.reverse();

        let mut search_rng_b = Rng::from_seed("rebel-test:search", "rebel-test");
        let result_b = run_public_belief_search(&permuted, &config, &mut search_rng_b);

        assert_eq!(
            result_a.root_policy.len(),
            result_b.root_policy.len(),
            "policy length must be stable across particle permutation"
        );
        for (i, (a, b)) in result_a
            .root_policy
            .iter()
            .zip(result_b.root_policy.iter())
            .enumerate()
        {
            assert_eq!(
                a, b,
                "root_policy[{i}] diverged under particle permutation: {a} vs {b}"
            );
        }
        // The heuristic rollout leaf is gone: no row can ever be flagged as
        // rollout-contaminated.
        assert!(!result_a.rollout_leaf_used);
        assert_eq!(result_a.diagnostics.rollout_leaf_calls, 0);
    }

    /// Anti-regression for the old zero-leaf bias: every rollout must actually
    /// evaluate the network (so turn-ending actions are valued by their
    /// continuation, not silently scored 0), the root policy must be a valid
    /// distribution, and no rollout should dead-end before a leaf is seen.
    #[test]
    fn rollouts_evaluate_leaves_and_produce_valid_policy() {
        install_public_only_stub();
        let config = test_config();
        let belief = build_initial_belief("leaf", 6);

        let mut search_rng = Rng::from_seed("rebel-test:leaf", "rebel-test");
        let result = run_public_belief_search(&belief, &config, &mut search_rng);

        assert_eq!(result.root_policy.len(), belief.legal_actions.len());
        let sum: f64 = result.root_policy.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "root policy must sum to 1, got {sum}"
        );
        assert!(
            result.root_action_values.iter().all(|v| v.is_finite()),
            "all action values must be finite"
        );
        assert!(
            result.diagnostics.particle_action_evaluations > 0,
            "the search must perform network leaf evaluations (rollouts ran)"
        );
        assert_eq!(
            result.diagnostics.degenerate_leaf_count, 0,
            "no rollout should dead-end before evaluating a leaf at the initial position"
        );
        assert_eq!(result.diagnostics.max_depth, 4);
    }

    /// Softmax policy orders by value (argmax of the value vector carries the
    /// most probability mass) and sums to 1.
    #[test]
    fn softmax_policy_orders_by_value() {
        let values = vec![0.1, 0.9, 0.3, 0.2];
        let policy = softmax_policy(&values, 0.5);
        let sum: f64 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "policy must sum to 1, got {sum}");
        let argmax = argmax_f64(&values);
        assert_eq!(argmax, 1);
        for (i, p) in policy.iter().enumerate() {
            if i != argmax {
                assert!(
                    *p < policy[argmax],
                    "argmax index must carry the most mass; index {i} had {p}"
                );
            }
        }
    }
}
