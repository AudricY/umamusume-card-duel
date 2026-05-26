//! `MctsConfig` + defaults — bit-identical to `mcts.ts:143` `defaultMctsConfig`.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::core::constants::SideId;
use crate::core::state::GameState;
use crate::policy::types::{LegalAiAction, PublicObservation};

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

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MctsRootActionSelection {
    MaxVisits,
    MaxMeanQ,
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
    /// Opt-in diagnostic/training capture: when non-zero and
    /// `leaf=rollout`, `run_mcts` returns an evenly-spaced sample of up to
    /// this many non-terminal leaf states evaluated by rollout, with the
    /// rollout scalar attached.
    /// This is off by default so normal MCTS calls keep the historical
    /// result shape and memory profile.
    #[serde(default)]
    pub record_rollout_leaf_samples: u32,
    /// When `leaf=value-head`, optionally blend rollout evaluation into
    /// leaf scoring: `(1 - blend) * value_head + blend * rollout`.
    /// Defaults to 0.0, preserving pure value-head behavior.
    #[serde(default)]
    pub value_head_rollout_blend: f64,
    pub add_root_dirichlet: bool,
    pub dirichlet_alpha: f64,
    pub dirichlet_epsilon: f64,
    pub max_nodes: u32,
    pub collapse_max_steps: u32,
    pub adaptive_ratio: f64,
    pub adaptive_min_sims: u32,
    /// Final root action readout. `MaxVisits` is standard MCTS and the
    /// historical default. `MaxMeanQ` is an opt-in diagnostic for learned
    /// leaves, where visit counts can be dominated by the policy prior while
    /// the leaf scalar may carry the better action ranking.
    #[serde(default = "default_root_action_selection")]
    pub root_action_selection: MctsRootActionSelection,
    /// AlphaZero-style two-sided search (default false). When true,
    /// opponent decision points become real tree nodes, the policy/value
    /// net is queried for whoever is to move, and backup uses sign-flips
    /// per ply. Root is still constructed as a modelSide decision node so
    /// `root_value` / `root_mean_q` / `root_visit_distribution` stay in
    /// modelSide frame and the downstream JSONL schema is unaffected.
    /// See `docs/ai-research/scoping/two-sided-mcts-scoping.md`.
    #[serde(default)]
    pub two_sided: bool,
    /// LEGACY: was the `/predict` HTTP server URL when the engine called
    /// out to `serve_onnx.py`. R16-P3 spike Option A landed in-process
    /// ORT (see `crate::inference`); this field is retained for one
    /// release so orchestrator argument plumbing doesn't break in
    /// lockstep, but it is now IGNORED at predict time. Deprecate-and-
    /// remove in the follow-up r12_orchestrator-wiring slice.
    #[serde(default)]
    pub model_url: String,
    /// B6 intra-tree wave-batching (see
    /// `docs/ai-research/scoping/gpu-batched-inference-throughput.md`).
    /// `1` (default) keeps the historical serial MCTS loop bit-identical.
    /// `>1` selects `wave_size` leaves per cycle using virtual loss,
    /// batches the per-leaf inference calls into a single
    /// `predict_v3_batch`, then backs them all up. Intended to sustain
    /// higher dispatcher fill than inter-game-only batching can hit
    /// (B5 showed fill capped at ~46-48 because each game's MCTS is
    /// serial-per-tree). Mutually exclusive with the cross-thread
    /// batched dispatcher (`InferenceSession::load_on_with_batching`
    /// with `max_batch > 1`); `sim-eval-gate` rejects the combination
    /// at flag-parse time.
    #[serde(default = "default_wave_size")]
    pub wave_size: u32,
    /// B6 virtual-loss pessimization applied during wave-member
    /// selection. AlphaGo-standard `1.0`. Wave member k adds
    /// `virtual_loss` to `visits[a]` and subtracts `virtual_loss` from
    /// `wsum[a]` along its selected path; backup of wave member k undoes
    /// the virtual loss first, then applies the real `visits[a] += 1` /
    /// `wsum[a] += leaf_scalar`. Net effect at wave-end is identical to
    /// serial backup. Only consulted when `wave_size > 1`.
    #[serde(default = "default_virtual_loss")]
    pub virtual_loss: f64,
    /// R16-P3 spike Option A: path to the ONNX policy file. When set,
    /// `crate::inference::set_global` should have been invoked with a
    /// session loaded from this path before MCTS launches (typically by
    /// the sim-cli binary's `main()`). Carried here so callers can pin
    /// the source-of-truth on the config object alongside the URL field
    /// during the transition window.
    #[serde(default)]
    pub onnx_path: Option<PathBuf>,
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
            record_rollout_leaf_samples: 0,
            value_head_rollout_blend: 0.0,
            add_root_dirichlet: false,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            max_nodes: 5_000,
            collapse_max_steps: 64,
            adaptive_ratio: 0.0,
            adaptive_min_sims: 20,
            root_action_selection: MctsRootActionSelection::MaxVisits,
            two_sided: false,
            model_url: String::new(),
            onnx_path: None,
            wave_size: 1,
            virtual_loss: 1.0,
        }
    }
}

fn default_root_action_selection() -> MctsRootActionSelection {
    MctsRootActionSelection::MaxVisits
}

fn default_wave_size() -> u32 {
    1
}

fn default_virtual_loss() -> f64 {
    1.0
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
pub struct MctsLeafSample {
    pub model_side: SideId,
    pub turn_number: u32,
    pub state: GameState,
    pub observation: PublicObservation,
    pub legal_actions: Vec<LegalAiAction>,
    pub rollout_value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MctsResult {
    pub selected_index: usize,
    pub visits: Vec<u32>,
    pub diagnostics: MctsDiagnostics,
    #[serde(default)]
    pub rollout_leaf_samples: Vec<MctsLeafSample>,
}
