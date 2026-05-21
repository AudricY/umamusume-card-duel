//! `MctsConfig` + defaults — bit-identical to `mcts.ts:143` `defaultMctsConfig`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MctsLeaf {
    ValueHead,
    Rollout,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MctsPrior {
    Uniform,
    Policy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MctsConfig {
    pub simulations: u32,
    #[serde(rename = "cPuct")]
    pub c_puct: f64,
    pub leaf: MctsLeaf,
    pub prior: MctsPrior,
    pub rollout_crn_samples: u32,
    pub rollout_steps: u32,
    pub add_root_dirichlet: bool,
    pub dirichlet_alpha: f64,
    pub dirichlet_epsilon: f64,
    pub max_nodes: u32,
    pub collapse_max_steps: u32,
    pub adaptive_ratio: f64,
    pub adaptive_min_sims: u32,
}

impl Default for MctsConfig {
    /// Mirror of `defaultMctsConfig(undefined)` in `mcts.ts:143`.
    fn default() -> Self {
        MctsConfig {
            simulations: 100,
            c_puct: 1.5,
            leaf: MctsLeaf::ValueHead,
            prior: MctsPrior::Uniform,
            rollout_crn_samples: 3,
            rollout_steps: 200,
            add_root_dirichlet: false,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            max_nodes: 5_000,
            collapse_max_steps: 64,
            adaptive_ratio: 0.0,
            adaptive_min_sims: 20,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MctsDiagnostics {
    pub root_value: f64,
    pub root_visit_distribution: Vec<f64>,
    pub root_mean_q: Vec<f64>,
    pub root_priors: Vec<f64>,
    pub expansions: u32,
    pub leaf_evaluations: u32,
    pub terminal_leafs: u32,
    pub visited_hashes: u32,
    pub root_prior_entropy: f64,
    pub root_prior_argmax: usize,
    pub simulations_run: u32,
    pub halted_early: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MctsResult {
    pub selected_index: usize,
    pub visits: Vec<u32>,
    pub diagnostics: MctsDiagnostics,
}
