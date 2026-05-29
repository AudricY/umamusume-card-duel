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
    let use_rollouts = neural_leaf_weight < 1.0 || crate::inference::global().is_none();
    let use_neural_leaf = neural_leaf_weight > 0.0;
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
                            neural_leaf_counts[particle_index][action_index] += 1;
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
                Err(_) => {
                    neural_leaf_sums = vec![vec![0.0; action_count]; belief.particles.len()];
                    neural_leaf_counts = vec![vec![0; action_count]; belief.particles.len()];
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
    // Honest CFR over the empirical action_values (replaces the prior softmax(Q)/T=1
    // target, which produced near-uniform searchPolicy whenever rollouts saturated).
    // Regret-matching over fixed v(a) converges to argmax; the average strategy is
    // exposed as the policy target so the head learns the same expert action under
    // policy distillation.
    let rollout_policy = {
        let cfr_iterations: usize = 128;
        let k = action_count;
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
    };
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
            search_iterations: config.iterations,
            policy_entropy: entropy(&root_policy),
            particle_action_agreement: agreement,
            aggregate_q_variance: variance(&action_values),
        },
        legal_actions: belief.legal_actions.clone(),
        search_algorithm: config.algorithm.clone(),
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

fn softmax(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp = values.iter().map(|v| (v - max).exp()).collect::<Vec<_>>();
    let total = exp.iter().sum::<f64>();
    if total <= 0.0 || !total.is_finite() {
        return vec![1.0 / values.len() as f64; values.len()];
    }
    exp.into_iter().map(|v| v / total).collect()
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
