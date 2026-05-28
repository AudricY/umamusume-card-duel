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
use crate::policy::types::LegalAiAction;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RebelSearchConfig {
    pub iterations: u32,
    pub max_depth: u32,
    pub rollout_steps: u32,
    pub algorithm: String,
}

impl Default for RebelSearchConfig {
    fn default() -> Self {
        RebelSearchConfig {
            iterations: 64,
            max_depth: 1,
            rollout_steps: 120,
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

    for (particle_index, particle) in belief.particles.iter().enumerate() {
        let particle_weight = particle.weight.max(0.0);
        let mut per_particle = vec![0.0; action_count];
        for (action_index, action) in belief.legal_actions.iter().enumerate() {
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
            rollout_leaf_calls += 1;
            let value = rollout_leaf_value_for_state(
                &next,
                belief.observer_side,
                1,
                config.rollout_steps,
                format!(
                    "{}:particle:{}:action:{}",
                    belief.public_history_digest, particle_index, action_index
                )
                .as_str(),
            );
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
    let root_policy = softmax(&action_values);
    let sampled_action_index = sample_policy(&root_policy, rng);
    let public_belief_value = root_policy
        .iter()
        .zip(action_values.iter())
        .map(|(p, q)| p * q)
        .sum();
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
            particle_action_evaluations: belief.particles.len() * action_count,
            rollout_leaf_calls,
            search_iterations: config.iterations,
            policy_entropy: entropy(&root_policy),
            particle_action_agreement: agreement,
            aggregate_q_variance: variance(&action_values),
        },
        legal_actions: belief.legal_actions.clone(),
        search_algorithm: config.algorithm.clone(),
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
