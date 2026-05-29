//! ReBeL-style public-belief search.
//!
//! This module is deliberately separate from ordinary perfect-information
//! MCTS. The current search is a bounded public-information-set evaluator:
//! it returns one root distribution over public legal actions, aggregates
//! values over weighted particles, and records diagnostics needed by the
//! ReBeL E2E audits. It is the integration path for later CFR regret tables.

use serde::{Deserialize, Serialize};

use crate::belief::PublicBeliefState;
use crate::core::constants::SideId;
use crate::core::random::{with_rng, Rng};
use crate::dispatcher::{advance_modeled_turn_step, get_forced_attack_coin_results};
use crate::mcts::driver::rollout_leaf_value_for_state;
use crate::mcts::math::mcts_terminal_value;
use crate::policy::actions::enumerate_legal_ai_actions;
use crate::policy::observation::build_public_observation;
use crate::policy::types::{LegalAiAction, PublicObservation};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RebelSearchConfig {
    pub iterations: u32,
    pub max_depth: u32,
    pub rollout_steps: u32,
    pub neural_policy_weight: f64,
    pub neural_value_weight: f64,
    pub neural_leaf_weight: f64,
    /// Opt-in gate for the determinized heuristic rollout leaf
    /// (`rollout_leaf_value_for_state`). That leaf is info-INCORRECT for this
    /// imperfect-information game (strategy fusion): it evaluates each
    /// determinized particle as if its private state were common knowledge.
    /// The neural/observation leaf is the only sound production leaf, so the
    /// rollout path is OFF by default and reachable only as an explicit
    /// ablation (`--allow-rollout-leaf`).
    pub allow_rollout_leaf: bool,
    pub algorithm: String,
}

impl Default for RebelSearchConfig {
    fn default() -> Self {
        RebelSearchConfig {
            iterations: 64,
            max_depth: 1,
            rollout_steps: 120,
            neural_policy_weight: 0.0,
            neural_value_weight: 0.0,
            neural_leaf_weight: 0.0,
            allow_rollout_leaf: false,
            algorithm: "public-belief-cfr-v1".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RebelSearchDiagnostics {
    pub information_set_key: String,
    pub particle_count: usize,
    pub legal_action_count: usize,
    pub particle_action_evaluations: usize,
    pub rollout_leaf_calls: usize,
    pub neural_leaf_calls: usize,
    pub neural_leaf_batch_rows: usize,
    pub neural_policy_weight: f64,
    pub neural_value_weight: f64,
    pub neural_leaf_weight: f64,
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
    /// True iff this decision evaluated at least one determinized rollout leaf,
    /// i.e. the (info-incorrect) rollout ablation was both enabled and actually
    /// exercised. Serialized downstream as `rolloutLeafUsed`.
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
    if action_count == 0 {
        return BeliefSearchResult {
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
                neural_leaf_calls: 0,
                neural_leaf_batch_rows: 0,
                neural_policy_weight: config.neural_policy_weight,
                neural_value_weight: config.neural_value_weight,
                neural_leaf_weight: config.neural_leaf_weight,
                neural_value: None,
                search_iterations: 0,
                policy_entropy: 0.0,
                particle_action_agreement: 0.0,
                aggregate_q_variance: 0.0,
            },
            legal_actions: Vec::new(),
            search_algorithm: config.algorithm.clone(),
        };
    }

    let mut action_values = vec![0.0; action_count];
    let mut action_weight = vec![0.0; action_count];
    let mut private_state_values = Vec::with_capacity(belief.particles.len());
    let mut particle_best = Vec::with_capacity(belief.particles.len());
    let mut rollout_leaf_calls = 0usize;
    let mut neural_leaf_calls = 0usize;
    let mut neural_leaf_batch_rows = 0usize;
    let search_iterations = config.iterations.max(1) as usize;
    let neural_leaf_weight = config.neural_leaf_weight.clamp(0.0, 1.0);
    // Rollouts are info-incorrect; they NEVER run unless the operator explicitly
    // opts into the ablation via `allow_rollout_leaf`. Even then they only fire
    // when the neural leaf cannot fully cover the leaf value (weight < 1.0 or no
    // model loaded).
    let use_rollouts = config.allow_rollout_leaf
        && (neural_leaf_weight < 1.0 || crate::inference::global().is_none());
    let use_neural_leaf = neural_leaf_weight > 0.0;
    // Loud-fail: with the rollout leaf disabled (the production default), the
    // neural leaf is the ONLY value source. Refusing to silently emit zeroed
    // leaf values when no model is loaded keeps corrupted targets out of the
    // training data.
    if use_neural_leaf && !use_rollouts && crate::inference::global().is_none() {
        panic!(
            "ReBeL leaf evaluation has no value source: no inference model loaded \
             and the rollout leaf is disabled; pass --allow-rollout-leaf to opt \
             into the (info-incorrect) rollout ablation"
        );
    }
    let mut base_leaf_values: Vec<Vec<f64>> = vec![vec![0.0; action_count]; belief.particles.len()];
    let mut leaf_observations: Vec<PublicObservation> = Vec::new();
    let mut leaf_legal_actions: Vec<Vec<LegalAiAction>> = Vec::new();
    let mut leaf_targets: Vec<(usize, usize)> = Vec::new();
    let mut neural_leaf_sums: Vec<Vec<f64>> = vec![vec![0.0; action_count]; belief.particles.len()];
    let mut neural_leaf_counts: Vec<Vec<usize>> =
        vec![vec![0; action_count]; belief.particles.len()];

    for (particle_index, particle) in belief.particles.iter().enumerate() {
        for (action_index, action) in belief.legal_actions.iter().enumerate() {
            for sample_index in 0..search_iterations {
                let forced =
                    with_rng_borrow(rng, || get_forced_attack_coin_results(&particle.game_state));
                let next = with_rng_borrow(rng, || {
                    advance_modeled_turn_step(
                        &particle.game_state,
                        belief.observer_side,
                        action,
                        forced,
                    )
                });
                let rollout_value = if use_rollouts {
                    rollout_leaf_calls += 1;
                    rollout_leaf_value_for_state(
                        &next,
                        belief.observer_side,
                        1,
                        config.rollout_steps,
                        format!(
                            "{}:particle:{}:action:{}:sample:{}",
                            belief.public_history_digest,
                            particle_index,
                            action_index,
                            sample_index
                        )
                        .as_str(),
                    )
                } else if next.game_over {
                    mcts_terminal_value(&next, belief.observer_side)
                } else {
                    0.0
                };
                base_leaf_values[particle_index][action_index] += rollout_value;
                if use_neural_leaf {
                    if next.game_over {
                        neural_leaf_sums[particle_index][action_index] +=
                            mcts_terminal_value(&next, belief.observer_side);
                        neural_leaf_counts[particle_index][action_index] += 1;
                    } else {
                        let legal = enumerate_legal_ai_actions(&next, belief.observer_side);
                        if legal.is_empty() {
                            // Non-terminal leaf with no legal actions and no model
                            // input. Do NOT increment the count with an implicit 0
                            // value: that biased the per-(particle, action) neural
                            // average toward 0 for no defensible reason. Skipping it
                            // leaves the average over the genuinely-evaluated leaves;
                            // if NO leaf is ever evaluated for this slot, the
                            // aggregation below falls back to the rollout value.
                        } else {
                            leaf_observations
                                .push(build_public_observation(&next, belief.observer_side));
                            leaf_legal_actions.push(legal);
                            leaf_targets.push((particle_index, action_index));
                        }
                    }
                }
            }
            base_leaf_values[particle_index][action_index] /= search_iterations as f64;
        }
    }

    if use_neural_leaf && !leaf_targets.is_empty() {
        if let Some(session) = crate::inference::global() {
            let belief_features = belief.belief_features.vector.as_slice();
            let batch: Vec<(&PublicObservation, &[LegalAiAction], Option<&[f32]>)> =
                leaf_observations
                    .iter()
                    .zip(leaf_legal_actions.iter())
                    .map(|(obs, legal)| (obs, legal.as_slice(), Some(belief_features)))
                    .collect();
            match session.predict_v3_batch_with_belief(&batch) {
                Ok(predictions) => {
                    neural_leaf_batch_rows += predictions.len();
                    for ((particle_index, action_index), prediction) in
                        leaf_targets.iter().copied().zip(predictions.into_iter())
                    {
                        neural_leaf_calls += 1;
                        neural_leaf_sums[particle_index][action_index] += prediction.value as f64;
                        neural_leaf_counts[particle_index][action_index] += 1;
                    }
                }
                Err(e) => {
                    // Hard failure: silently zeroing the leaf sums/counts here
                    // injected zero-valued leaves into the search aggregation,
                    // which corrupts both the action values and the derived
                    // policy/value targets. Fail loudly with the underlying
                    // inference error instead.
                    panic!("ReBeL neural leaf inference failed: {e}");
                }
            }
        }
    }

    for (particle_index, particle) in belief.particles.iter().enumerate() {
        let particle_weight = particle.weight.max(0.0);
        let mut per_particle = vec![0.0; action_count];
        for action_index in 0..action_count {
            let rollout_value = base_leaf_values[particle_index][action_index];
            let neural_count = neural_leaf_counts[particle_index][action_index];
            let value = if use_neural_leaf && neural_count > 0 {
                let neural_value =
                    neural_leaf_sums[particle_index][action_index] / neural_count as f64;
                (1.0 - neural_leaf_weight) * rollout_value + neural_leaf_weight * neural_value
            } else {
                rollout_value
            };
            per_particle[action_index] = value;
            action_values[action_index] += value * particle_weight;
            action_weight[action_index] += particle_weight;
        }
        let best = argmax_f64(&per_particle);
        particle_best.push(best);
        private_state_values.push(per_particle.get(best).copied().unwrap_or(0.0));
    }

    for (value, weight) in action_values.iter_mut().zip(action_weight.iter()) {
        if *weight > 0.0 {
            *value /= *weight;
        }
    }
    // Regret-matching over the empirical action_values (replaces the prior
    // softmax(Q)/T=1 target, which produced near-uniform searchPolicy whenever
    // rollouts saturated).
    //
    // HONEST LABELING: this is NOT equilibrium CFR. CFR's regret is computed
    // against a counterfactual value vector that itself depends on the opponent's
    // current strategy and is recomputed every iteration. Here action_values is a
    // FIXED vector for the whole loop, so regret-matching just concentrates the
    // average strategy on argmax(action_values) (with ties split). It is a smooth
    // argmax over a single value estimate, exposed as the policy target so the
    // head learns the same expert action under policy distillation. The on-wire
    // `algorithm` string is left as "public-belief-cfr-v1" for schema continuity;
    // do not read it as a claim of CFR equilibrium convergence.
    let rollout_policy = average_strategy_from_values(&action_values);
    let mut neural_value = None;
    let root_policy = if config.neural_policy_weight > 0.0 || config.neural_value_weight > 0.0 {
        if let Some(session) = crate::inference::global() {
            match session.predict_v3_with_belief(
                &belief.public_observation,
                &belief.legal_actions,
                Some(&belief.belief_features.vector),
            ) {
                Ok(prediction) => {
                    neural_value = Some(prediction.value as f64);
                    if config.neural_policy_weight > 0.0 {
                        mix_policy(
                            &rollout_policy,
                            &prediction.probs,
                            config.neural_policy_weight,
                        )
                    } else {
                        rollout_policy.clone()
                    }
                }
                Err(_) => rollout_policy.clone(),
            }
        } else {
            rollout_policy.clone()
        }
    } else {
        rollout_policy.clone()
    };
    let sampled_action_index = sample_policy(&root_policy, rng);
    let rollout_belief_value: f64 = root_policy
        .iter()
        .zip(action_values.iter())
        .map(|(p, q)| p * q)
        .sum();
    let public_belief_value = match neural_value {
        Some(v) if config.neural_value_weight > 0.0 => {
            let w = config.neural_value_weight.clamp(0.0, 1.0);
            (1.0 - w) * rollout_belief_value + w * v
        }
        _ => rollout_belief_value,
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
        rollout_leaf_used: use_rollouts && rollout_leaf_calls > 0,
        diagnostics: RebelSearchDiagnostics {
            information_set_key: belief.public_history_digest.clone(),
            particle_count: belief.particles.len(),
            legal_action_count: action_count,
            particle_action_evaluations: belief.particles.len() * action_count * search_iterations,
            rollout_leaf_calls,
            neural_leaf_calls,
            neural_leaf_batch_rows,
            neural_policy_weight: config.neural_policy_weight,
            neural_value_weight: config.neural_value_weight,
            neural_leaf_weight: config.neural_leaf_weight,
            neural_value,
            // Report the EFFECTIVE per-(particle, action) sampling budget that
            // actually ran (`config.iterations.max(1)`), not the raw configured
            // value, so diagnostics and the emitted `searchIterations` match the
            // work performed even when `iterations == 0`.
            search_iterations: search_iterations as u32,
            policy_entropy: entropy(&root_policy),
            particle_action_agreement: agreement,
            aggregate_q_variance: variance(&action_values),
        },
        legal_actions: belief.legal_actions.clone(),
        search_algorithm: config.algorithm.clone(),
    }
}

/// Regret-matching average strategy over a FIXED value vector.
///
/// See the call site for the honest-labeling note: because `action_values`
/// does not change across iterations, the cumulative regret accumulates
/// monotonically in favor of `argmax(action_values)`, and the time-averaged
/// strategy concentrates its mass there. The returned distribution always sums
/// to 1 (falling back to uniform on a degenerate/empty value vector). Extracted
/// from `run_public_belief_search` so it can be unit-tested directly.
fn average_strategy_from_values(action_values: &[f64]) -> Vec<f64> {
    let k = action_values.len();
    if k == 0 {
        return Vec::new();
    }
    let cfr_iterations: usize = 128;
    let mut cumulative_regret = vec![0.0f64; k];
    let mut strategy_sum = vec![0.0f64; k];
    let mut sigma = vec![1.0 / k as f64; k];
    for _ in 0..cfr_iterations {
        let mean_value: f64 = sigma
            .iter()
            .zip(action_values.iter())
            .map(|(s, v)| s * v)
            .sum();
        for a in 0..k {
            cumulative_regret[a] += action_values[a] - mean_value;
        }
        let positive_sum: f64 = cumulative_regret.iter().map(|r| r.max(0.0)).sum();
        if positive_sum > 0.0 {
            for a in 0..k {
                sigma[a] = cumulative_regret[a].max(0.0) / positive_sum;
            }
        } else {
            for a in 0..k {
                sigma[a] = 1.0 / k as f64;
            }
        }
        for a in 0..k {
            strategy_sum[a] += sigma[a];
        }
    }
    let total: f64 = strategy_sum.iter().sum();
    if total > 0.0 && total.is_finite() {
        strategy_sum.iter().map(|s| s / total).collect()
    } else {
        vec![1.0 / k as f64; k]
    }
}

fn mix_policy(rollout_policy: &[f64], neural_policy: &[f32], neural_weight: f64) -> Vec<f64> {
    let w = neural_weight.clamp(0.0, 1.0);
    let mut mixed = Vec::with_capacity(rollout_policy.len());
    for (i, rollout_p) in rollout_policy.iter().enumerate() {
        let neural_p = neural_policy.get(i).copied().unwrap_or(0.0).max(0.0) as f64;
        mixed.push((1.0 - w) * *rollout_p + w * neural_p);
    }
    let total: f64 = mixed.iter().sum();
    if total > 0.0 && total.is_finite() {
        for p in mixed.iter_mut() {
            *p /= total;
        }
        mixed
    } else {
        rollout_policy.to_vec()
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

#[allow(dead_code)]
fn _side_key(side: SideId) -> &'static str {
    match side {
        SideId::Player => "player",
        SideId::Opponent => "opponent",
    }
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
        // Map to (-1, 1) deterministically.
        ((h % 20001) as f64 / 10000.0 - 1.0) as f32
    }

    static INSTALL_STUB: Once = Once::new();

    /// Install the public-only stub into the process-global inference slot.
    /// `set_global` is a `OnceLock`, so we guard with `Once` and tolerate the
    /// (benign) case where another test installed first — we only rely on the
    /// global being *some* public-only oracle. No engine test other than this
    /// module calls `set_global`, so the installed session is ours.
    fn install_public_only_stub() {
        INSTALL_STUB.call_once(|| {
            inference::set_global(InferenceSession::new_test_stub(public_only_value));
        });
    }

    /// Build a belief state at the game's initial position for the player side.
    fn build_initial_belief(seed: &str, particle_count: usize) -> crate::belief::PublicBeliefState {
        let rng = Rng::from_seed(format!("rebel-test:{seed}"), "rebel-test");
        let (state, mut rng) = with_rng(rng, setup_ai_vs_ai_game);
        let history = PublicHistory::from_state(&state, "matikanetannhauser", "matikanetannhauser", 0);
        let config = BeliefBuildConfig {
            particle_count,
            seed_label: format!("rebel-test:{seed}"),
        };
        build_public_belief_state(&state, SideId::Player, history, &config, &mut rng)
    }

    fn neural_leaf_config() -> RebelSearchConfig {
        RebelSearchConfig {
            iterations: 4,
            max_depth: 1,
            rollout_steps: 8,
            neural_policy_weight: 0.0,
            neural_value_weight: 0.0,
            neural_leaf_weight: 1.0,
            allow_rollout_leaf: false,
            algorithm: "public-belief-cfr-v1".to_string(),
        }
    }

    /// Info-set invariance: with the neural leaf as the only value source
    /// (`neural_leaf_weight = 1.0`, `allow_rollout_leaf = false`) and a value
    /// oracle that depends ONLY on the public observation, permuting the
    /// particle set must yield a bit-identical `root_policy`, and zero rollout
    /// leaves must be evaluated.
    ///
    /// Coverage note: this asserts the strongest *exactly checkable* property —
    /// invariance under particle PERMUTATION (the brief's clause (a)) — because
    /// the per-particle leaf observation can legitimately differ across distinct
    /// particle SAMPLES once an action advances and reveals previously-hidden
    /// cards. Permutation invariance isolates the aggregation order from the
    /// result, which is exactly the strategy-fusion guard the production leaf
    /// must satisfy.
    #[test]
    fn info_set_invariant_under_particle_permutation() {
        install_public_only_stub();
        let config = neural_leaf_config();

        let belief = build_initial_belief("perm", 8);

        let mut search_rng_a = Rng::from_seed("rebel-test:search", "rebel-test");
        let result_a = run_public_belief_search(&belief, &config, &mut search_rng_a);

        // Reverse the particle ordering (a permutation of the SAME particle
        // set, with matching weights) and re-run with the SAME search RNG seed.
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

        // (b) No determinized rollout leaf was evaluated.
        assert_eq!(
            result_a.diagnostics.rollout_leaf_calls, 0,
            "rollout leaf must never run with allow_rollout_leaf=false"
        );
        assert_eq!(result_b.diagnostics.rollout_leaf_calls, 0);
        assert!(!result_a.rollout_leaf_used);
        assert!(!result_b.rollout_leaf_used);
    }

    /// CFR average strategy over a FIXED value vector concentrates mass on the
    /// argmax and sums to 1.
    #[test]
    fn cfr_average_strategy_concentrates_on_argmax() {
        let action_values = vec![0.1, 0.9, 0.3, 0.2];
        let policy = average_strategy_from_values(&action_values);

        let sum: f64 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "policy must sum to 1, got {sum}");

        let argmax = super::argmax_f64(&action_values);
        assert_eq!(argmax, 1, "argmax of the fixed value vector is index 1");
        for (i, p) in policy.iter().enumerate() {
            if i == argmax {
                assert!(
                    *p > 0.99,
                    "argmax index must carry nearly all mass, got {p}"
                );
            } else {
                assert!(
                    *p < 0.01,
                    "non-argmax index {i} should carry near-zero mass, got {p}"
                );
            }
        }
    }
}
