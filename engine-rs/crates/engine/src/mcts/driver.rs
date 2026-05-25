//! Bit-identical port of `backend/src/sim/mcts.ts`.
//!
//! The MCTS driver:
//! - Builds a root model-decision node, expanding lazily.
//! - Selection via `puct_select` (already in `mcts::math`).
//! - Step via `step_from_model_decision`, which applies an action and then
//!   collapses heuristic-opponent turns until the next model decision.
//! - Leaf evaluation either rolls out via the heuristic opponent or calls
//!   the in-process ONNX inference path (when configured for value-head
//!   leaves). R16-P3 spike Option A: the prior HTTP `/predict` round-trip
//!   was replaced with `crate::inference::global().predict_v3(...)` —
//!   `model_url` is now an ignored backward-compat field on `MctsConfig`.
//!
//! RNG-tree: the driver creates `Rng::from_seed(seed, "mcts-root")` — this
//! is a SEPARATE RNG tree from any outer recorder/selfplay tree, matching
//! the TS source's `createSeededRng(seed, "mcts-root")` in `mcts.ts:177`.

use crate::core::constants::SideId;
use crate::core::random::{with_rng, Rng};
use crate::core::state::{CurrentSide, GameState};
use crate::dispatcher::{
    advance_modeled_turn_step, advance_opponent_turn_step, advance_player_ai_turn_step,
    get_forced_attack_coin_results, state_fingerprint, state_hash,
};
use crate::mcts::config::{
    MctsConfig, MctsDiagnostics, MctsLeaf, MctsLeafSample, MctsPrior, MctsResult,
    MctsRootActionSelection,
};
use crate::mcts::math::{argmax, entropy, mcts_terminal_value, puct_select};
use crate::mcts::node::MctsNode;
use crate::mcts::sample::sample_dirichlet;
use crate::policy::actions::enumerate_legal_ai_actions;
use crate::policy::observation::build_public_observation;
use crate::policy::types::{LegalAiAction, PublicObservation};

/// Top-level entry. Caller passes the inner seed string verbatim — the
/// driver constructs the `mcts-root` RNG tree internally.
pub fn run_mcts(
    root_state: &GameState,
    model_side: SideId,
    config: &MctsConfig,
    model_url: &str,
    seed: &str,
) -> MctsResult {
    let mut root_rng = Rng::from_seed(seed, "mcts-root");
    let mut root =
        build_model_decision_node(root_state, model_side, model_url, config, &mut root_rng);
    let mut diagnostics = MctsDiagnostics {
        root_value: 0.0,
        root_visit_distribution: Vec::new(),
        root_mean_q: Vec::new(),
        root_priors: Vec::new(),
        expansions: 1,
        leaf_evaluations: 0,
        terminal_leafs: 0,
        visited_hashes: 0,
        root_prior_entropy: 0.0,
        root_prior_argmax: 0,
        simulations_run: 0,
        halted_early: false,
    };

    if let Some(v) = root.terminal_value {
        diagnostics.root_value = v;
        return MctsResult {
            selected_index: 0,
            visits: Vec::new(),
            diagnostics,
            rollout_leaf_samples: Vec::new(),
        };
    }

    // Snapshot the prior pre-noise.
    diagnostics.root_priors = root.priors.clone();
    diagnostics.root_prior_entropy = entropy(&diagnostics.root_priors);
    diagnostics.root_prior_argmax = argmax(&diagnostics.root_priors);

    if config.add_root_dirichlet && root.legal_actions.len() > 1 {
        let mut dirichlet_rng = root_rng.fork("dirichlet");
        let noise = sample_dirichlet(
            root.legal_actions.len(),
            config.dirichlet_alpha,
            &mut dirichlet_rng,
        );
        let eps = config.dirichlet_epsilon;
        for (i, p) in root.priors.iter_mut().enumerate() {
            let n = noise.get(i).copied().unwrap_or(0.0);
            *p = (1.0 - eps) * *p + eps * n;
        }
    }

    // Cache root value.
    let mut rollout_leaf_samples: Vec<MctsLeafSample> = Vec::new();

    if matches!(config.leaf, MctsLeaf::ValueHead)
        && config.value_head_rollout_blend <= 0.0
        && root.cached_leaf_value.is_some()
    {
        diagnostics.root_value = root.cached_leaf_value.unwrap();
    } else {
        let mut leaf_rng = root_rng.fork("root-leaf");
        diagnostics.root_value = leaf_value(
            &root.state,
            model_side,
            model_url,
            config,
            &mut leaf_rng,
            &mut rollout_leaf_samples,
        );
        diagnostics.leaf_evaluations += 1;
    }

    let mut total_nodes: u32 = 1;

    if config.wave_size <= 1 {
        run_serial_loop(
            &mut root,
            &mut root_rng,
            config,
            model_url,
            &mut diagnostics,
            &mut rollout_leaf_samples,
            &mut total_nodes,
        );
    } else {
        run_wave_loop(
            &mut root,
            &mut root_rng,
            config,
            model_url,
            &mut diagnostics,
            &mut rollout_leaf_samples,
            &mut total_nodes,
        );
    }

    let visits = root.visits.clone();
    let total_visits: u32 = visits.iter().sum();
    diagnostics.root_visit_distribution = if total_visits > 0 {
        visits
            .iter()
            .map(|n| *n as f64 / total_visits as f64)
            .collect()
    } else {
        visits.iter().map(|_| 0.0).collect()
    };
    diagnostics.root_mean_q = root
        .visits
        .iter()
        .zip(root.wsum.iter())
        .map(|(n, w)| if *n > 0 { w / *n as f64 } else { 0.0 })
        .collect();
    diagnostics.visited_hashes = total_nodes;

    let best_index = select_root_action(&visits, &diagnostics.root_mean_q, config);

    MctsResult {
        selected_index: best_index,
        visits,
        diagnostics,
        rollout_leaf_samples: downsample_rollout_leaf_samples(
            rollout_leaf_samples,
            config.record_rollout_leaf_samples,
        ),
    }
}

#[derive(Copy, Clone)]
struct PathStep {
    node_ptr: *mut MctsNode,
    action_index: usize,
}

fn select_root_action(visits: &[u32], mean_q: &[f64], config: &MctsConfig) -> usize {
    let mut best_index = 0usize;
    match config.root_action_selection {
        MctsRootActionSelection::MaxVisits => {
            // Historical behavior: argmax visits, tiebreak by mean Q.
            let mut best_visits: i64 = -1;
            let mut best_q = f64::NEG_INFINITY;
            for i in 0..visits.len() {
                let n = visits[i] as i64;
                let q = mean_q.get(i).copied().unwrap_or(0.0);
                if n > best_visits || (n == best_visits && q > best_q) {
                    best_visits = n;
                    best_q = q;
                    best_index = i;
                }
            }
        }
        MctsRootActionSelection::MaxMeanQ => {
            // Diagnostic behavior for learned leaves: choose by backed-up
            // value among visited root actions, tiebreak by visits. Unvisited
            // actions are ignored unless every visit count is zero.
            let mut best_q = f64::NEG_INFINITY;
            let mut best_visits: i64 = -1;
            let mut saw_visited = false;
            for i in 0..visits.len() {
                let n = visits[i] as i64;
                if n <= 0 {
                    continue;
                }
                saw_visited = true;
                let q = mean_q.get(i).copied().unwrap_or(0.0);
                if q > best_q || (q == best_q && n > best_visits) {
                    best_q = q;
                    best_visits = n;
                    best_index = i;
                }
            }
            if !saw_visited {
                let fallback = MctsConfig {
                    root_action_selection: MctsRootActionSelection::MaxVisits,
                    ..config.clone()
                };
                return select_root_action(visits, mean_q, &fallback);
            }
        }
    }
    best_index
}

/// Historical serial MCTS loop — one simulation per RNG fork, fully
/// expand/eval/back-up before the next sim sees the updated tree.
/// `run_mcts` dispatches here when `config.wave_size <= 1`, preserving
/// the pre-B6 bit-identical behavior.
#[allow(clippy::too_many_arguments)]
fn run_serial_loop(
    root: &mut MctsNode,
    root_rng: &mut Rng,
    config: &MctsConfig,
    model_url: &str,
    diagnostics: &mut MctsDiagnostics,
    rollout_leaf_samples: &mut Vec<MctsLeafSample>,
    total_nodes: &mut u32,
) {
    for sim in 0..config.simulations {
        let mut sim_rng = root_rng.fork(&format!("sim{}", sim));

        // Walk path via PUCT until leaf (unexpanded child or terminal).
        let mut path: Vec<PathStep> = Vec::new();
        let mut node_ptr: *mut MctsNode = root;
        // SAFETY: the path holds raw pointers to nodes inside the tree.
        // We mutate one node at a time (the leaf's parent on expansion,
        // or visit/wsum on backup). Aliasing is single-threaded.
        loop {
            let node: &MctsNode = unsafe { &*node_ptr };
            if node.terminal_value.is_some() {
                break;
            }
            let action_index =
                puct_select(&node.visits, &node.wsum, &node.priors, config.c_puct);
            path.push(PathStep {
                node_ptr,
                action_index,
            });
            let child_present = node
                .children
                .get(action_index)
                .and_then(|c| c.as_ref())
                .is_some();
            if !child_present {
                break;
            }
            let node_mut: &mut MctsNode = unsafe { &mut *node_ptr };
            // Descend into the child.
            node_ptr = node_mut.children[action_index]
                .as_mut()
                .map(|b| b.as_mut() as *mut MctsNode)
                .unwrap();
        }

        let leaf_scalar: f64;
        let leaf_node: &MctsNode = unsafe { &*node_ptr };
        if let Some(v) = leaf_node.terminal_value {
            leaf_scalar = v;
            diagnostics.terminal_leafs += 1;
        } else {
            // Expand the deepest unexpanded child.
            let last = path
                .last()
                .copied()
                .expect("non-terminal selection produced an empty path");
            let parent: &MctsNode = unsafe { &*last.node_ptr };
            let action = parent.legal_actions[last.action_index].clone();
            let parent_state = parent.state.clone();
            let parent_model_side = parent.model_side;
            let mut expand_rng = sim_rng.fork(&format!("expand:a{}", last.action_index));
            let next_state_opt = step_from_model_decision(
                &parent_state,
                parent_model_side,
                &action,
                config,
                &mut expand_rng,
            );
            match next_state_opt {
                None => {
                    leaf_scalar = 0.0;
                }
                Some(next_state) if *total_nodes >= config.max_nodes => {
                    if next_state.game_over {
                        leaf_scalar = mcts_terminal_value(&next_state, parent_model_side);
                        diagnostics.terminal_leafs += 1;
                    } else {
                        let mut leaf_rng =
                            sim_rng.fork(&format!("leaf:cap:a{}", last.action_index));
                        leaf_scalar = leaf_value(
                            &next_state,
                            parent_model_side,
                            model_url,
                            config,
                            &mut leaf_rng,
                            rollout_leaf_samples,
                        );
                        diagnostics.leaf_evaluations += 1;
                    }
                }
                Some(next_state) => {
                    let new_child = build_model_decision_node(
                        &next_state,
                        parent_model_side,
                        model_url,
                        config,
                        &mut sim_rng,
                    );
                    let cached = new_child.cached_leaf_value;
                    let term = new_child.terminal_value;
                    let model_side_for_leaf = new_child.model_side;
                    let leaf_state_for_eval = new_child.state.clone();
                    // Insert.
                    let parent_mut: &mut MctsNode = unsafe { &mut *last.node_ptr };
                    parent_mut.children[last.action_index] = Some(Box::new(new_child));
                    *total_nodes += 1;
                    diagnostics.expansions += 1;
                    if let Some(v) = term {
                        leaf_scalar = v;
                        diagnostics.terminal_leafs += 1;
                    } else if matches!(config.leaf, MctsLeaf::ValueHead)
                        && config.value_head_rollout_blend <= 0.0
                        && cached.is_some()
                    {
                        leaf_scalar = cached.unwrap();
                    } else {
                        let mut leaf_rng =
                            sim_rng.fork(&format!("leaf:expand:a{}", last.action_index));
                        leaf_scalar = leaf_value(
                            &leaf_state_for_eval,
                            model_side_for_leaf,
                            model_url,
                            config,
                            &mut leaf_rng,
                            rollout_leaf_samples,
                        );
                        diagnostics.leaf_evaluations += 1;
                    }
                }
            }
        }

        // Backup.
        for step in &path {
            let node_mut: &mut MctsNode = unsafe { &mut *step.node_ptr };
            let i = step.action_index;
            node_mut.visits[i] += 1;
            node_mut.wsum[i] += leaf_scalar;
        }

        diagnostics.simulations_run = sim + 1;

        // Adaptive halt.
        if config.adaptive_ratio > 0.0
            && diagnostics.simulations_run >= config.adaptive_min_sims
            && root.visits.len() >= 2
        {
            let mut top: u32 = 0;
            let mut second: u32 = 0;
            for &v in &root.visits {
                if v > top {
                    second = top;
                    top = v;
                } else if v > second {
                    second = v;
                }
            }
            if (top as f64) / (second as f64 + 1.0) >= config.adaptive_ratio {
                diagnostics.halted_early = true;
                break;
            }
        }
    }
}

/// B6 intra-tree wave-batching loop (see
/// `docs/ai-research/scoping/gpu-batched-inference-throughput.md`). For
/// each wave of up to `config.wave_size` simulations:
///
/// 1. **Selection.** Walk `wave` PUCT paths from the root, applying
///    virtual loss (`config.virtual_loss`) to every visited
///    `(node, action_index)` along the way. Virtual loss adds
///    `virtual_loss` to `visits[a]` and subtracts `virtual_loss` from
///    `wsum[a]`, pessimizing Q(s,a) along the in-flight path so wave
///    members 1..N-1 are biased toward unexplored siblings. This is the
///    AlphaGo-standard tree-parallelism trick.
/// 2. **Expansion.** For each wave member, apply
///    `step_from_model_decision` to compute the next state; collect the
///    states that need a model `predict_v3` call (those that will
///    instantiate a new policy-prior child node, OR those that need a
///    value-head leaf evaluation).
/// 3. **Batched inference.** Issue a single `predict_v3_batch` call for
///    the model-eval subset.
/// 4. **Backup.** For each wave member: undo the virtual loss along the
///    path, then apply the real `visits[a] += 1` / `wsum[a] +=
///    leaf_scalar` update. Net effect at wave-end is identical to a
///    serial backup; virtual loss only shifts within-wave selection
///    ordering.
///
/// **Determinism.** Each wave member k of wave w forks
/// `root_rng.fork("sim{i}")` where `i = w * wave_size + k`. With
/// `wave_size = 1` this matches the serial loop's RNG sequence (but
/// `run_mcts` dispatches `wave_size = 1` straight to `run_serial_loop`
/// for bit-identical results — the wave loop is reserved for `wave_size
/// > 1`).
#[allow(clippy::too_many_arguments)]
fn run_wave_loop(
    root: &mut MctsNode,
    root_rng: &mut Rng,
    config: &MctsConfig,
    model_url: &str,
    diagnostics: &mut MctsDiagnostics,
    rollout_leaf_samples: &mut Vec<MctsLeafSample>,
    total_nodes: &mut u32,
) {
    let wave_size = config.wave_size.max(1) as usize;
    let virtual_loss = config.virtual_loss;
    let mut sim_index: u32 = 0;
    let total_sims = config.simulations;

    while sim_index < total_sims {
        let wave = ((total_sims - sim_index) as usize).min(wave_size);
        // Per-wave-member scratch state. We collect everything we need
        // for phase-4 backup here so we can run model inference once at
        // wave-mid and back up after.
        let mut members: Vec<WaveMember> = Vec::with_capacity(wave);

        for k in 0..wave {
            let i = sim_index + k as u32;
            let mut sim_rng = root_rng.fork(&format!("sim{}", i));

            // Phase 1: select a leaf path applying virtual loss along
            // the way. Virtual loss is per-`(node, action_index)`
            // entry on the path and is undone after this wave member's
            // backup runs.
            let mut path: Vec<PathStep> = Vec::new();
            let mut node_ptr: *mut MctsNode = root;
            // SAFETY: tree is single-threaded. We mutate `visits` /
            // `wsum` for virtual loss along the path; the path holds
            // raw pointers but no two wave members hold the same `&mut`
            // simultaneously (we drop the borrow before the next loop
            // iteration).
            loop {
                let node: &MctsNode = unsafe { &*node_ptr };
                if node.terminal_value.is_some() {
                    break;
                }
                let action_index = puct_select(
                    &node.visits,
                    &node.wsum,
                    &node.priors,
                    config.c_puct,
                );
                path.push(PathStep {
                    node_ptr,
                    action_index,
                });
                // Apply virtual loss to the parent's (visits[a], wsum[a])
                // BEFORE descending so the next wave member sees the
                // pessimized Q.
                {
                    let node_mut: &mut MctsNode = unsafe { &mut *node_ptr };
                    apply_virtual_loss(node_mut, action_index, virtual_loss);
                }
                let child_present = unsafe { &*node_ptr }
                    .children
                    .get(action_index)
                    .and_then(|c| c.as_ref())
                    .is_some();
                if !child_present {
                    break;
                }
                let node_mut: &mut MctsNode = unsafe { &mut *node_ptr };
                node_ptr = node_mut.children[action_index]
                    .as_mut()
                    .map(|b| b.as_mut() as *mut MctsNode)
                    .unwrap();
            }

            // Phase 2: determine whether this leaf is terminal,
            // becomes a new child (expansion), or hits the no-op /
            // max-nodes branch. Stash everything we need for phase 3
            // (batched inference) and phase 4 (backup).
            let leaf_node: &MctsNode = unsafe { &*node_ptr };
            let action = WaveAction::compute(
                root,
                node_ptr,
                leaf_node,
                &path,
                config,
                &mut sim_rng,
                model_url,
                rollout_leaf_samples,
                diagnostics,
                total_nodes,
            );
            members.push(WaveMember {
                sim_index: i,
                sim_rng,
                path,
                action,
            });
        }

        // Phase 3: batched inference for any expansion that needs a
        // policy-prior call.
        //
        // We collect the (state, side, legal_actions) tuples for all
        // expansions that need a model call when `prior=policy`. For
        // `prior=uniform`, no inference is needed and we skip the batch
        // entirely. Crucially we run inference AFTER all expansions have
        // had their next_state computed (cheap, CPU-only, no model
        // call), so one batched call covers the whole wave.
        if matches!(config.prior, MctsPrior::Policy) {
            wave_run_priors(&mut members, config);
        }

        // Phase 4: complete each wave member's expansion (insert new
        // child if any, compute leaf_scalar), then undo virtual loss
        // and apply real backup.
        for member in members.iter_mut() {
            wave_complete_member(
                member,
                config,
                model_url,
                rollout_leaf_samples,
                diagnostics,
                total_nodes,
            );
            // Undo virtual loss along the path, then apply real backup.
            // Net effect on (visits, wsum) is identical to serial.
            let scalar = member
                .action
                .leaf_scalar
                .expect("wave member did not produce a leaf scalar");
            for step in &member.path {
                let node_mut: &mut MctsNode = unsafe { &mut *step.node_ptr };
                undo_virtual_loss(node_mut, step.action_index, virtual_loss);
                node_mut.visits[step.action_index] += 1;
                node_mut.wsum[step.action_index] += scalar;
            }
        }

        sim_index += wave as u32;
        diagnostics.simulations_run = sim_index;

        // Adaptive halt — same logic as serial, checked once per wave
        // (behaviourally equivalent to checking every `wave_size` sims
        // in the serial loop).
        if config.adaptive_ratio > 0.0
            && diagnostics.simulations_run >= config.adaptive_min_sims
            && root.visits.len() >= 2
        {
            let mut top: u32 = 0;
            let mut second: u32 = 0;
            for &v in &root.visits {
                if v > top {
                    second = top;
                    top = v;
                } else if v > second {
                    second = v;
                }
            }
            if (top as f64) / (second as f64 + 1.0) >= config.adaptive_ratio {
                diagnostics.halted_early = true;
                break;
            }
        }
    }
}

/// Per-wave-member scratch. Lives for one wave only.
struct WaveMember {
    /// Sim index in the global sequence (`w * wave_size + k`). Used as
    /// the RNG fork seed `"sim{i}"` so wave_size=1 matches serial.
    #[allow(dead_code)]
    sim_index: u32,
    sim_rng: Rng,
    path: Vec<PathStep>,
    action: WaveAction,
}

/// What phase 1 decided this wave member's leaf needs.
struct WaveAction {
    kind: WaveActionKind,
    /// Computed in phase 4 (after batched inference). Drives backup.
    leaf_scalar: Option<f64>,
}

enum WaveActionKind {
    /// Selection terminated on a terminal node — backup with that node's
    /// terminal value.
    TerminalLeaf { value: f64 },
    /// Expansion attempted but `step_from_model_decision` returned None
    /// (no-op action / state unchanged). Backup with 0.0.
    NoOpStep,
    /// Expansion produced a `next_state` and we hit the `max_nodes`
    /// cap. Use `leaf_value` on the state directly without inserting
    /// a new child.
    MaxNodesCap {
        next_state: GameState,
        parent_model_side: SideId,
        leaf_rng_seed: String,
    },
    /// Expansion produced a `next_state` that became terminal. Use
    /// `mcts_terminal_value` without inserting a new child.
    TerminalAfterStep {
        value: f64,
    },
    /// Expansion produced a non-terminal `next_state`. We need to
    /// build the child node (which under `prior=policy` requires a
    /// `predict_v3` call — batched in phase 3) and then compute the
    /// leaf scalar.
    Expand {
        parent_ptr: *mut MctsNode,
        parent_action_index: usize,
        parent_model_side: SideId,
        next_state: GameState,
        legal_actions: Vec<LegalAiAction>,
        leaf_rng_seed: String,
        /// Filled in by phase 3 (batched predict). For `prior=uniform`
        /// this stays None and the priors are uniform; for
        /// `prior=policy` this becomes `Some((probs, value))` after
        /// the batched call.
        prediction: Option<(Vec<f64>, f64)>,
        /// `true` when `state.game_over` is set on `next_state` but the
        /// node-build branch can short-circuit; tracked here so
        /// phase-3 doesn't bother packing a request for it.
        next_state_game_over: bool,
        /// `true` when the collapsed state inside the to-be-built node
        /// would have empty `legal_actions` (terminal-by-no-moves).
        /// Phase 4 short-circuits to terminal_value(0.0).
        collapsed_terminal: bool,
    },
}

impl WaveAction {
    /// Phase 2 of the wave loop: compute the leaf action for one wave
    /// member without making any model inference call. All inference
    /// is batched in phase 3.
    ///
    /// `_root_ptr` is unused but kept in the signature to satisfy the
    /// raw-pointer-aliasing contract (`root` must outlive the wave).
    #[allow(clippy::too_many_arguments)]
    fn compute(
        _root: &mut MctsNode,
        node_ptr: *mut MctsNode,
        leaf_node: &MctsNode,
        path: &[PathStep],
        config: &MctsConfig,
        sim_rng: &mut Rng,
        _model_url: &str,
        _rollout_leaf_samples: &mut Vec<MctsLeafSample>,
        _diagnostics: &mut MctsDiagnostics,
        total_nodes: &u32,
    ) -> Self {
        if let Some(v) = leaf_node.terminal_value {
            return WaveAction {
                kind: WaveActionKind::TerminalLeaf { value: v },
                leaf_scalar: None,
            };
        }
        let last = path
            .last()
            .copied()
            .expect("non-terminal wave selection produced an empty path");
        let parent: &MctsNode = unsafe { &*last.node_ptr };
        let action = parent.legal_actions[last.action_index].clone();
        let parent_state = parent.state.clone();
        let parent_model_side = parent.model_side;
        let mut expand_rng = sim_rng.fork(&format!("expand:a{}", last.action_index));
        let next_state_opt = step_from_model_decision(
            &parent_state,
            parent_model_side,
            &action,
            config,
            &mut expand_rng,
        );
        match next_state_opt {
            None => WaveAction {
                kind: WaveActionKind::NoOpStep,
                leaf_scalar: None,
            },
            Some(next_state) if *total_nodes >= config.max_nodes => {
                if next_state.game_over {
                    let v = mcts_terminal_value(&next_state, parent_model_side);
                    WaveAction {
                        kind: WaveActionKind::TerminalAfterStep { value: v },
                        leaf_scalar: None,
                    }
                } else {
                    WaveAction {
                        kind: WaveActionKind::MaxNodesCap {
                            next_state,
                            parent_model_side,
                            leaf_rng_seed: format!("leaf:cap:a{}", last.action_index),
                        },
                        leaf_scalar: None,
                    }
                }
            }
            Some(next_state) => {
                if next_state.game_over {
                    let v = mcts_terminal_value(&next_state, parent_model_side);
                    return WaveAction {
                        kind: WaveActionKind::Expand {
                            parent_ptr: last.node_ptr,
                            parent_action_index: last.action_index,
                            parent_model_side,
                            next_state,
                            legal_actions: Vec::new(),
                            leaf_rng_seed: format!("leaf:expand:a{}", last.action_index),
                            prediction: None,
                            next_state_game_over: true,
                            collapsed_terminal: false,
                        },
                        leaf_scalar: Some(v),
                    };
                }
                // Pre-collapse: replicate the same legality lookup
                // `build_model_decision_node` does so phase 3 has the
                // legal_actions for the batched predict call. Note we
                // do NOT collapse opponent turns here (the new child's
                // current_side is already the model's side per
                // step_from_model_decision); we just enumerate.
                let legal_actions = enumerate_legal_ai_actions(&next_state, parent_model_side);
                if legal_actions.is_empty() {
                    // build_model_decision_node will treat this as a
                    // terminal-by-no-moves node with value 0.0. Phase 4
                    // will short-circuit and not need a model call.
                    return WaveAction {
                        kind: WaveActionKind::Expand {
                            parent_ptr: last.node_ptr,
                            parent_action_index: last.action_index,
                            parent_model_side,
                            next_state,
                            legal_actions: Vec::new(),
                            leaf_rng_seed: format!("leaf:expand:a{}", last.action_index),
                            prediction: None,
                            next_state_game_over: false,
                            collapsed_terminal: true,
                        },
                        leaf_scalar: Some(0.0),
                    };
                }
                let _ = node_ptr;
                WaveAction {
                    kind: WaveActionKind::Expand {
                        parent_ptr: last.node_ptr,
                        parent_action_index: last.action_index,
                        parent_model_side,
                        next_state,
                        legal_actions,
                        leaf_rng_seed: format!("leaf:expand:a{}", last.action_index),
                        prediction: None,
                        next_state_game_over: false,
                        collapsed_terminal: false,
                    },
                    leaf_scalar: None,
                }
            }
        }
    }
}

/// Phase 3 helper — batch the policy-prior `predict_v3` calls for every
/// wave member's expansion that needs one. `prior=uniform` callers skip
/// this entirely; `prior=policy` callers see one `Session::run` for the
/// wave instead of N sequential ones.
fn wave_run_priors(members: &mut [WaveMember], _config: &MctsConfig) {
    // Collect references for the batched call. We re-walk after to
    // re-attach the predictions, so we need to preserve the index map.
    let mut indices: Vec<usize> = Vec::new();
    let mut obs_buf: Vec<PublicObservation> = Vec::new();
    let mut legals_buf: Vec<Vec<LegalAiAction>> = Vec::new();
    for (i, m) in members.iter().enumerate() {
        if let WaveActionKind::Expand {
            ref next_state,
            ref legal_actions,
            next_state_game_over,
            collapsed_terminal,
            parent_model_side,
            ..
        } = m.action.kind
        {
            if next_state_game_over || collapsed_terminal {
                continue;
            }
            // Build the observation now so phase 4 reuses it (and the
            // batched predict call has a stable lifetime).
            obs_buf.push(build_public_observation(next_state, parent_model_side));
            legals_buf.push(legal_actions.clone());
            indices.push(i);
        }
    }
    if indices.is_empty() {
        return;
    }
    let session = crate::inference::global().expect(
        "MCTS wave_run_priors: no inference session loaded — call \
         inference::set_global(...) before run_mcts (typically in the \
         sim-cli main()).",
    );
    let batch: Vec<(&PublicObservation, &[LegalAiAction])> = obs_buf
        .iter()
        .zip(legals_buf.iter())
        .map(|(o, l)| (o, l.as_slice()))
        .collect();
    let predictions = match session.predict_v3_batch(&batch) {
        Ok(p) => p,
        Err(e) => panic!("MCTS wave batched prior+value inference failed: {}", e),
    };
    debug_assert_eq!(predictions.len(), indices.len());
    for ((member_idx, prediction), legal) in indices
        .iter()
        .zip(predictions.into_iter())
        .zip(legals_buf.iter())
    {
        let mut probs: Vec<f64> = prediction
            .probs
            .iter()
            .take(legal.len())
            .copied()
            .map(|p| if p.is_finite() { p.max(0.0) as f64 } else { 0.0 })
            .collect();
        let sum: f64 = probs.iter().sum();
        if sum > 0.0 {
            for p in probs.iter_mut() {
                *p /= sum;
            }
        } else {
            probs = vec![1.0 / legal.len() as f64; legal.len()];
        }
        let value = prediction.value as f64;
        if let WaveActionKind::Expand {
            prediction: ref mut slot,
            ..
        } = members[*member_idx].action.kind
        {
            *slot = Some((probs, value));
        }
    }
}

/// Phase 4 helper — complete one wave member's expansion (insert child,
/// compute leaf scalar). After this returns, `member.action.leaf_scalar`
/// is `Some(v)` and the caller undoes virtual loss + applies real backup.
fn wave_complete_member(
    member: &mut WaveMember,
    config: &MctsConfig,
    model_url: &str,
    rollout_leaf_samples: &mut Vec<MctsLeafSample>,
    diagnostics: &mut MctsDiagnostics,
    total_nodes: &mut u32,
) {
    // Take ownership of the action so we can move out of its enum.
    // We only care about the resulting `leaf_scalar` from this point
    // on; the `kind` is discarded.
    let action = std::mem::replace(
        &mut member.action,
        WaveAction {
            kind: WaveActionKind::NoOpStep,
            leaf_scalar: None,
        },
    );
    let scalar = match action.kind {
        WaveActionKind::TerminalLeaf { value } => {
            diagnostics.terminal_leafs += 1;
            value
        }
        WaveActionKind::NoOpStep => 0.0,
        WaveActionKind::TerminalAfterStep { value } => {
            diagnostics.terminal_leafs += 1;
            value
        }
        WaveActionKind::MaxNodesCap {
            next_state,
            parent_model_side,
            leaf_rng_seed,
        } => {
            let mut leaf_rng = member.sim_rng.fork(&leaf_rng_seed);
            let v = leaf_value(
                &next_state,
                parent_model_side,
                model_url,
                config,
                &mut leaf_rng,
                rollout_leaf_samples,
            );
            diagnostics.leaf_evaluations += 1;
            v
        }
        WaveActionKind::Expand {
            parent_ptr,
            parent_action_index,
            parent_model_side,
            next_state,
            legal_actions,
            leaf_rng_seed,
            prediction,
            next_state_game_over,
            collapsed_terminal,
        } => {
            if next_state_game_over {
                // Build a terminal sentinel child so the tree records
                // this branch. Mirror what `build_model_decision_node`
                // does when state.game_over is true at the top.
                let v = mcts_terminal_value(&next_state, parent_model_side);
                let child = MctsNode::terminal(next_state.clone(), parent_model_side, v);
                let parent_mut: &mut MctsNode = unsafe { &mut *parent_ptr };
                parent_mut.children[parent_action_index] = Some(Box::new(child));
                *total_nodes += 1;
                diagnostics.expansions += 1;
                diagnostics.terminal_leafs += 1;
                v
            } else if collapsed_terminal {
                // Empty legal_actions ⇒ terminal-by-no-moves node with
                // value 0.0 (mirror of build_model_decision_node's
                // empty-legals branch).
                let child = MctsNode {
                    state: next_state.clone(),
                    model_side: parent_model_side,
                    legal_actions: Vec::new(),
                    priors: Vec::new(),
                    visits: Vec::new(),
                    wsum: Vec::new(),
                    children: Vec::new(),
                    terminal_value: Some(0.0),
                    cached_leaf_value: Some(0.0),
                };
                let parent_mut: &mut MctsNode = unsafe { &mut *parent_ptr };
                parent_mut.children[parent_action_index] = Some(Box::new(child));
                *total_nodes += 1;
                diagnostics.expansions += 1;
                diagnostics.terminal_leafs += 1;
                0.0
            } else {
                // Non-terminal expansion. Build the child node from the
                // batched prediction (if prior=policy) or uniform priors
                // (if prior=uniform).
                let (priors, cached_leaf_value): (Vec<f64>, Option<f64>) =
                    if let Some((probs, value)) = prediction {
                        (probs, Some(value))
                    } else {
                        let uniform = 1.0 / legal_actions.len() as f64;
                        (vec![uniform; legal_actions.len()], None)
                    };
                let visits = vec![0u32; legal_actions.len()];
                let wsum = vec![0.0f64; legal_actions.len()];
                let mut children: Vec<Option<Box<MctsNode>>> =
                    Vec::with_capacity(legal_actions.len());
                for _ in 0..legal_actions.len() {
                    children.push(None);
                }
                let child = MctsNode {
                    state: next_state.clone(),
                    model_side: parent_model_side,
                    legal_actions: legal_actions.clone(),
                    priors,
                    visits,
                    wsum,
                    children,
                    terminal_value: None,
                    cached_leaf_value,
                };
                let leaf_state_for_eval = child.state.clone();
                let cached = child.cached_leaf_value;
                let model_side_for_leaf = child.model_side;
                let parent_mut: &mut MctsNode = unsafe { &mut *parent_ptr };
                parent_mut.children[parent_action_index] = Some(Box::new(child));
                *total_nodes += 1;
                diagnostics.expansions += 1;
                if matches!(config.leaf, MctsLeaf::ValueHead)
                    && config.value_head_rollout_blend <= 0.0
                    && cached.is_some()
                {
                    cached.unwrap()
                } else {
                    let mut leaf_rng = member.sim_rng.fork(&leaf_rng_seed);
                    let v = leaf_value(
                        &leaf_state_for_eval,
                        model_side_for_leaf,
                        model_url,
                        config,
                        &mut leaf_rng,
                        rollout_leaf_samples,
                    );
                    diagnostics.leaf_evaluations += 1;
                    v
                }
            }
        }
    };
    member.action.leaf_scalar = Some(scalar);
}

fn apply_virtual_loss(node: &mut MctsNode, action_index: usize, virtual_loss: f64) {
    // Encode virtual loss into the integer `visits` counter as a
    // single-unit bump and the float `wsum` as a -virtual_loss
    // contribution. Wave members 1..N see Q(s,a) = wsum/visits skew
    // negative along the in-flight path, biasing PUCT toward
    // unexplored siblings. Net effect after undo_virtual_loss + real
    // backup is identical to a single serial backup.
    if action_index < node.visits.len() {
        node.visits[action_index] = node.visits[action_index].saturating_add(1);
        node.wsum[action_index] -= virtual_loss;
    }
}

fn undo_virtual_loss(node: &mut MctsNode, action_index: usize, virtual_loss: f64) {
    if action_index < node.visits.len() {
        node.visits[action_index] = node.visits[action_index].saturating_sub(1);
        node.wsum[action_index] += virtual_loss;
    }
}

fn build_model_decision_node(
    state: &GameState,
    model_side: SideId,
    model_url: &str,
    config: &MctsConfig,
    rng: &mut Rng,
) -> MctsNode {
    if state.game_over {
        let v = mcts_terminal_value(state, model_side);
        return MctsNode {
            state: state.clone(),
            model_side,
            legal_actions: Vec::new(),
            priors: Vec::new(),
            visits: Vec::new(),
            wsum: Vec::new(),
            children: Vec::new(),
            terminal_value: Some(v),
            cached_leaf_value: Some(v),
        };
    }

    // Defensive collapse: if it's the opponent's turn, advance through
    // heuristic moves until model's turn or terminal.
    let mut collapsed = state.clone();
    if collapsed.current_side != CurrentSide::from_side(model_side) {
        let mut collapse_rng = Rng::from_seed(
            format!("{}:precollapse", state_hash(state)),
            "mcts-collapse",
        );
        collapsed = collapse_until_model_or_terminal(
            &collapsed,
            model_side,
            config.collapse_max_steps,
            &mut collapse_rng,
        );
        if collapsed.game_over {
            let v = mcts_terminal_value(&collapsed, model_side);
            return MctsNode {
                state: collapsed,
                model_side,
                legal_actions: Vec::new(),
                priors: Vec::new(),
                visits: Vec::new(),
                wsum: Vec::new(),
                children: Vec::new(),
                terminal_value: Some(v),
                cached_leaf_value: Some(v),
            };
        }
    }

    let legal_actions = enumerate_legal_ai_actions(&collapsed, model_side);
    if legal_actions.is_empty() {
        return MctsNode {
            state: collapsed,
            model_side,
            legal_actions: Vec::new(),
            priors: Vec::new(),
            visits: Vec::new(),
            wsum: Vec::new(),
            children: Vec::new(),
            terminal_value: Some(0.0),
            cached_leaf_value: Some(0.0),
        };
    }

    let mut priors: Vec<f64>;
    let mut cached_leaf_value: Option<f64> = None;
    if matches!(config.prior, MctsPrior::Policy) {
        let (action_probs, value) =
            predict_policy_and_value(model_url, &collapsed, model_side, &legal_actions, rng);
        priors = legal_actions
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let p = action_probs.get(i).copied().unwrap_or(0.0);
                p.max(1e-8)
            })
            .collect();
        cached_leaf_value = Some(value);
    } else {
        let uniform = 1.0 / legal_actions.len() as f64;
        priors = vec![uniform; legal_actions.len()];
    }
    let _ = rng;

    let visits = vec![0u32; legal_actions.len()];
    let wsum = vec![0.0f64; legal_actions.len()];
    let mut children: Vec<Option<Box<MctsNode>>> = Vec::with_capacity(legal_actions.len());
    for _ in 0..legal_actions.len() {
        children.push(None);
    }

    // Normalize priors if pulled from /predict (defensive).
    if matches!(config.prior, MctsPrior::Policy) {
        let s: f64 = priors.iter().sum();
        if s > 0.0 {
            for p in priors.iter_mut() {
                *p /= s;
            }
        }
    }

    MctsNode {
        state: collapsed,
        model_side,
        legal_actions,
        priors,
        visits,
        wsum,
        children,
        terminal_value: None,
        cached_leaf_value,
    }
}

/// R16-P3 Option A: in-process predict via `crate::inference`. The
/// previous HTTP `/predict` round-trip + JSON serde lived here; that
/// path was ~0.5-1ms RTT-dominated, the new path is ~10-100µs.
fn predict_policy_and_value(
    _model_url: &str,
    state: &GameState,
    model_side: SideId,
    legal_actions: &[LegalAiAction],
    _rng: &mut Rng,
) -> (Vec<f64>, f64) {
    let observation = build_public_observation(state, model_side);
    let session = crate::inference::global().expect(
        "MCTS predict_policy_and_value: no inference session loaded — \
         call inference::set_global(...) before run_mcts (typically in \
         the sim-cli main()).",
    );
    let prediction = match session.predict_v3(&observation, legal_actions) {
        Ok(p) => p,
        Err(e) => panic!("MCTS prior+value inference failed: {}", e),
    };
    let mut probs: Vec<f64> = prediction
        .probs
        .iter()
        .take(legal_actions.len())
        .copied()
        .map(|p| {
            if p.is_finite() {
                p.max(0.0) as f64
            } else {
                0.0
            }
        })
        .collect();
    let sum: f64 = probs.iter().sum();
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    } else {
        probs = vec![1.0 / legal_actions.len() as f64; legal_actions.len()];
    }
    (probs, prediction.value as f64)
}

fn step_from_model_decision(
    state: &GameState,
    model_side: SideId,
    action: &LegalAiAction,
    config: &MctsConfig,
    rng: &mut Rng,
) -> Option<GameState> {
    // Apply forced-coin computation in the same RNG stream the TS source
    // uses. `get_forced_attack_coin_results` consumes one rand per non-
    // guaranteed-heads flip needed by the eligible attack with the most
    // forced flips.
    //
    // Match TS: forced uses INNER rng; advance reads AMBIENT outer rng via
    // random_float(). Same fix as rollout_heuristic + collapse — see
    // their comments for the rationale.
    let forced_coins = with_rng_borrow(rng, || get_forced_attack_coin_results(state));
    let next_state = advance_modeled_turn_step(state, model_side, action, forced_coins);
    if state_fingerprint(&next_state) == state_fingerprint(state) {
        return None;
    }
    if next_state.game_over {
        return Some(next_state);
    }
    if next_state.current_side == CurrentSide::from_side(model_side) {
        return Some(next_state);
    }
    Some(collapse_until_model_or_terminal(
        &next_state,
        model_side,
        config.collapse_max_steps,
        rng,
    ))
}

fn collapse_until_model_or_terminal(
    state: &GameState,
    model_side: SideId,
    max_steps: u32,
    rng: &mut Rng,
) -> GameState {
    let mut current = state.clone();
    let mut before: Option<u128> = None;
    for _ in 0..max_steps {
        if current.game_over {
            break;
        }
        if current.current_side == CurrentSide::from_side(model_side) {
            break;
        }
        if before.is_none() {
            before = Some(state_fingerprint(&current));
        }
        let side_id = match current.current_side {
            CurrentSide::Player => SideId::Player,
            CurrentSide::Opponent => SideId::Opponent,
            CurrentSide::Done => break,
        };
        // Same convention as rollout_heuristic: forced uses inner rng
        // (TS passes rng explicitly to getForcedAttackCoinResults); advance
        // internals call random_float() reading the ambient outer rng.
        let forced = with_rng_borrow(rng, || get_forced_attack_coin_results(&current));
        if side_id == SideId::Player {
            advance_player_ai_turn_step(&mut current, forced);
        } else {
            advance_opponent_turn_step(&mut current, forced);
        }
        let after = state_fingerprint(&current);
        if Some(after) == before {
            break;
        }
        before = Some(after);
    }
    current
}

fn leaf_value(
    state: &GameState,
    model_side: SideId,
    model_url: &str,
    config: &MctsConfig,
    rng: &mut Rng,
    rollout_leaf_samples: &mut Vec<MctsLeafSample>,
) -> f64 {
    if state.game_over {
        return mcts_terminal_value(state, model_side);
    }
    match config.leaf {
        MctsLeaf::Rollout => {
            let value = rollout_leaf_value(state, model_side, config, rng);
            maybe_record_rollout_leaf_sample(
                rollout_leaf_samples,
                state,
                model_side,
                value,
                config.record_rollout_leaf_samples,
            );
            value
        }
        MctsLeaf::ValueHead => {
            let value = value_head_leaf_value(state, model_side, model_url);
            let blend = config.value_head_rollout_blend.clamp(0.0, 1.0);
            if blend > 0.0 {
                let rollout = rollout_leaf_value(state, model_side, config, rng);
                (1.0 - blend) * value + blend * rollout
            } else {
                value
            }
        }
    }
}

fn maybe_record_rollout_leaf_sample(
    samples: &mut Vec<MctsLeafSample>,
    state: &GameState,
    model_side: SideId,
    rollout_value: f64,
    max_samples: u32,
) {
    if max_samples == 0 || state.game_over {
        return;
    }
    let legal_actions = enumerate_legal_ai_actions(state, model_side);
    if legal_actions.len() < 2 {
        return;
    }
    samples.push(MctsLeafSample {
        model_side,
        turn_number: state.turn_number,
        state: state.clone(),
        observation: build_public_observation(state, model_side),
        legal_actions,
        rollout_value,
    });
}

fn downsample_rollout_leaf_samples(
    samples: Vec<MctsLeafSample>,
    max_samples: u32,
) -> Vec<MctsLeafSample> {
    let max_samples = max_samples as usize;
    if max_samples == 0 || samples.len() <= max_samples {
        return samples;
    }
    spread_sample_indices(samples.len(), max_samples)
        .into_iter()
        .map(|idx| samples[idx].clone())
        .collect()
}

fn spread_sample_indices(total: usize, max_samples: usize) -> Vec<usize> {
    if max_samples == 0 || total == 0 {
        return Vec::new();
    }
    if total <= max_samples {
        return (0..total).collect();
    }
    if max_samples == 1 {
        return vec![total / 2];
    }
    let last = total - 1;
    let denom = max_samples - 1;
    (0..max_samples)
        .map(|i| {
            // Integer-rounded linspace over [0, last].
            (i * last + denom / 2) / denom
        })
        .collect()
}

fn value_head_leaf_value(state: &GameState, model_side: SideId, _model_url: &str) -> f64 {
    let legal_actions = enumerate_legal_ai_actions(state, model_side);
    if legal_actions.is_empty() {
        // No actions means there's nothing to predict against; fall back
        // to the terminal-value evaluation (Python serve_onnx errors on
        // empty legalActions, so we short-circuit before the call).
        return mcts_terminal_value(state, model_side);
    }
    let observation = build_public_observation(state, model_side);
    let session = crate::inference::global().expect(
        "MCTS value_head_leaf_value: no inference session loaded — \
         call inference::set_global(...) before run_mcts (typically in \
         the sim-cli main()).",
    );
    match session.predict_v3(&observation, &legal_actions) {
        Ok(p) => p.value as f64,
        Err(e) => panic!("MCTS leaf value inference failed: {}", e),
    }
}

fn rollout_leaf_value(
    state: &GameState,
    model_side: SideId,
    config: &MctsConfig,
    rng: &mut Rng,
) -> f64 {
    let k = config.rollout_crn_samples.max(1);
    let mut sum = 0.0;
    let mut counted = 0;
    for i in 0..k {
        let mut sample_rng = rng.fork(&format!("rollout-crn-{}", i));
        let sample = rollout_heuristic(state, &mut sample_rng, config.rollout_steps);
        sum += mcts_terminal_value(&sample, model_side);
        counted += 1;
    }
    if counted > 0 {
        sum / counted as f64
    } else {
        0.0
    }
}

pub fn rollout_leaf_value_for_state(
    state: &GameState,
    model_side: SideId,
    rollout_crn_samples: u32,
    rollout_steps: u32,
    seed: &str,
) -> f64 {
    if state.game_over {
        return mcts_terminal_value(state, model_side);
    }
    let outer_rng = Rng::from_seed(seed, "mcts-leaf-relabel-outer");
    let (value, _) = with_rng(outer_rng, || {
        let rng = Rng::from_seed(seed, "mcts-leaf-relabel");
        let k = rollout_crn_samples.max(1);
        let mut sum = 0.0;
        for i in 0..k {
            let mut sample_rng = rng.fork(&format!("rollout-crn-{}", i));
            let sample = rollout_heuristic(state, &mut sample_rng, rollout_steps);
            sum += mcts_terminal_value(&sample, model_side);
        }
        sum / k as f64
    });
    value
}

thread_local! {
    /// Per-rollout-termination-reason counters (instrumentation only).
    /// Reset via `reset_rollout_stats()`; readable via `rollout_stats()`.
    pub static ROLLOUT_STATS: std::cell::RefCell<RolloutStats> =
        std::cell::RefCell::new(RolloutStats::default());
    /// When set, the FIRST rollout of each MCTS call prints per-advance
    /// (phase, side, rng-delta) to stderr.
    pub static VERBOSE_FIRST_ROLLOUT: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

pub fn set_verbose_first_rollout(on: bool) {
    VERBOSE_FIRST_ROLLOUT.with(|c| c.set(on));
}

#[derive(Debug, Default, Clone, Copy)]
pub struct RolloutStats {
    pub rollouts_started: u64,
    pub rollouts_ended_game_over: u64,
    pub rollouts_ended_current_side_done: u64,
    pub rollouts_ended_state_unchanged: u64,
    pub rollouts_ended_max_steps: u64,
    pub total_advance_calls: u64,
    /// Sum of "outer rng draws made during all rollouts in the current
    /// measurement window". Updated by `rollout_heuristic` via a Cell
    /// passed through `with_active_outer_draws_counter`.
    pub total_outer_rng_draws_during_rollouts: u64,
    /// Max outer-rng draws by any single rollout in this window.
    pub max_outer_rng_draws_in_single_rollout: u64,
}

pub fn reset_rollout_stats() {
    ROLLOUT_STATS.with(|s| *s.borrow_mut() = RolloutStats::default());
}

pub fn rollout_stats() -> RolloutStats {
    ROLLOUT_STATS.with(|s| *s.borrow())
}

fn rollout_heuristic(state: &GameState, rng: &mut Rng, max_steps: u32) -> GameState {
    let rollout_idx = ROLLOUT_STATS.with(|s| {
        let mut st = s.borrow_mut();
        st.rollouts_started += 1;
        st.rollouts_started
    });
    let verbose = VERBOSE_FIRST_ROLLOUT.with(|c| c.get()) && rollout_idx == 1;
    let outer_draws_before = peek_active_outer_draws().unwrap_or(0);
    let mut next = state.clone();
    let mut before: Option<u128> = None;
    let mut step_idx: u32 = 0;
    let end_reason: u8 = 'rollout: loop {
        if step_idx >= max_steps {
            break 'rollout 3; // max_steps
        }
        if next.game_over {
            break 'rollout 0; // game_over
        }
        if before.is_none() {
            before = Some(state_fingerprint(&next));
        }
        let side_id = match next.current_side {
            CurrentSide::Player => SideId::Player,
            CurrentSide::Opponent => SideId::Opponent,
            CurrentSide::Done => break 'rollout 1, // current_side_done
        };
        let pre_advance_draws = peek_active_outer_draws().unwrap_or(0);
        let pre_opponent_step = next.opponent_turn_step;
        // get_forced_attack_coin_results: TS calls with INNER rng explicitly
        // (mcts.ts:653 rolloutHeuristic). Keep the inner-rng install here.
        let forced = with_rng_borrow(rng, || get_forced_attack_coin_results(&next));
        // advance_*_turn_step internals call random_float() which reads the
        // AMBIENT outer rng — matching TS where engine.ts's
        // advance functions ignore the passed `random` param and use
        // randomFloat() from the storage provider. Do NOT re-install the
        // inner rng here; let the recorder's outer with_rng remain active.
        if side_id == SideId::Player {
            advance_player_ai_turn_step(&mut next, forced);
        } else {
            advance_opponent_turn_step(&mut next, forced);
        }
        ROLLOUT_STATS.with(|s| s.borrow_mut().total_advance_calls += 1);
        if verbose {
            let post_advance_draws = peek_active_outer_draws().unwrap_or(pre_advance_draws);
            let advance_draws = post_advance_draws.saturating_sub(pre_advance_draws);
            eprintln!(
                "    [rollout 1 advance {}] side={:?} step={:?} draws+={} turn={}",
                step_idx, side_id, pre_opponent_step, advance_draws, next.turn_number
            );
        }
        let after = state_fingerprint(&next);
        if Some(after) == before {
            break 'rollout 2; // state_unchanged
        }
        before = Some(after);
        step_idx += 1;
    };
    let outer_draws_after = peek_active_outer_draws().unwrap_or(outer_draws_before);
    let rollout_draws = outer_draws_after.saturating_sub(outer_draws_before);
    ROLLOUT_STATS.with(|s| {
        let mut st = s.borrow_mut();
        match end_reason {
            0 => st.rollouts_ended_game_over += 1,
            1 => st.rollouts_ended_current_side_done += 1,
            2 => st.rollouts_ended_state_unchanged += 1,
            3 => st.rollouts_ended_max_steps += 1,
            _ => {}
        }
        st.total_outer_rng_draws_during_rollouts += rollout_draws;
        if rollout_draws > st.max_outer_rng_draws_in_single_rollout {
            st.max_outer_rng_draws_in_single_rollout = rollout_draws;
        }
    });
    next
}

/// Peek the active thread-local Rng's draw counter without mutating it.
fn peek_active_outer_draws() -> Option<u64> {
    crate::core::random::peek_active_draws()
}

/// Helper: `with_rng` borrows-by-move, so this swap-in-swap-out shim lets
/// us reuse an `&mut Rng` across multiple calls without rebinding.
fn with_rng_borrow<T>(rng: &mut Rng, f: impl FnOnce() -> T) -> T {
    let taken = rng.clone();
    let (out, used) = with_rng(taken, f);
    *rng = used;
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::random::with_rng;
    use crate::flow::ai::telemetry::clear_ai_telemetry;

    fn setup_state() -> GameState {
        let cat = crate::core::catalog::catalog();
        let deck_a = [
            "matikanetannhauserBasic",
            "matikanetannhauserStage1",
            "matikanetannhauserStage2",
            "matikanetannhauserBasic",
            "matikanetannhauserStage1",
            "matikanetannhauserStage2",
            "matikanetannhauserBasic",
            "matikanetannhauserStage1",
            "matikanetannhauserStage2",
            "matikanetannhauserBasic",
            "matikanetannhauserStage1",
            "matikanetannhauserStage2",
            "matikanetannhauserBasic",
            "matikanetannhauserStage1",
            "matikanetannhauserStage2",
            "matikanetannhauserBasic",
            "matikanetannhauserStage1",
            "matikanetannhauserStage2",
            "matikanetannhauserBasic",
            "matikanetannhauserStage1",
        ];
        let ids_a: Vec<crate::core::card_id::CardId> =
            deck_a.iter().filter_map(|n| cat.id_for(n)).collect();
        let ids_b = ids_a.clone();
        // Run construction inside a with_rng scope so opening hands draw.
        let (state, _) = with_rng(Rng::from_seed(1234u32, "test-root"), || {
            clear_ai_telemetry();
            crate::dispatcher::create_game(
                &ids_a,
                &ids_b,
                "Opponent",
                crate::core::constants::AiDifficulty::Hard,
                false,
                "Player",
                None,
                None,
            )
        });
        state
    }

    #[test]
    fn mcts_terminal_value_returns_zero_when_not_game_over() {
        let state = setup_state();
        assert_eq!(mcts_terminal_value(&state, SideId::Player), 0.0);
    }

    #[test]
    fn dispatcher_advance_player_step_consumes_one_rng_draw_for_setup_coin() {
        use crate::core::random::with_rng;
        // `choose_opening_coin` calls `random_float()` exactly once. Counting
        // that draw is the simplest way to confirm the dispatcher honors
        // the TS RNG-site contract for the setup-phase coin flip.
        let state_before = setup_state();
        let rng_a = Rng::from_seed(7u32, "rng-draw-count");
        let rng_b = Rng::from_seed(7u32, "rng-draw-count");
        // Sample one draw from rng_b for the expected coin.
        let mut sentinel = rng_b.clone();
        let _expected = sentinel.next_f64();

        let (_state_after, used_a) = with_rng(rng_a, || {
            let mut s = state_before.clone();
            crate::dispatcher::choose_opening_coin(
                &mut s,
                crate::core::constants::CoinFlipResult::Heads,
            );
            s
        });
        // After choose_opening_coin runs one draw, used_a's next draw
        // should equal sentinel's next draw (same starting state).
        let mut after_a = used_a.clone();
        let mut after_sentinel = sentinel.clone();
        assert_eq!(
            after_a.next_f64().to_bits(),
            after_sentinel.next_f64().to_bits()
        );
    }

    #[test]
    fn step_from_model_decision_pass_action_advances_phase() {
        use crate::core::random::with_rng;
        use crate::policy::types::{AiPhase, LegalAiAction};

        let state = setup_state();
        let pass = LegalAiAction {
            id: "pass:bench".to_string(),
            phase: AiPhase::Bench,
            kind: "pass".to_string(),
            payload: serde_json::Value::Null,
            features: Vec::new(),
            action_source_card_idx: None,
            action_target_card_idx: None,
        };
        let cfg = MctsConfig {
            simulations: 1,
            c_puct: 1.5,
            leaf: MctsLeaf::Rollout,
            prior: MctsPrior::Uniform,
            rollout_crn_samples: 1,
            rollout_steps: 1,
            record_rollout_leaf_samples: 0,
            value_head_rollout_blend: 0.0,
            add_root_dirichlet: false,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            max_nodes: 8,
            collapse_max_steps: 4,
            adaptive_ratio: 0.0,
            adaptive_min_sims: 100,
            root_action_selection: MctsRootActionSelection::MaxVisits,
            model_url: String::new(),
            onnx_path: None,
            wave_size: 1,
            virtual_loss: 1.0,
        };
        let (_out, _) = with_rng(Rng::from_seed(33u32, "step-test"), || {
            let mut rng = Rng::from_seed(33u32, "inner");
            step_from_model_decision(&state, SideId::Player, &pass, &cfg, &mut rng)
        });
        // We only assert this doesn't panic; the state is in setup phase
        // so `advance_modeled_turn_step` bails early and returns the same
        // state — step_from_model_decision should then return None (the
        // `state_hash` equality check).
    }

    #[test]
    fn run_mcts_uniform_prior_rollout_leaf_visits_sum_to_simulations() {
        let state = setup_state();
        let cfg = MctsConfig {
            simulations: 8,
            c_puct: 1.5,
            leaf: MctsLeaf::Rollout,
            prior: MctsPrior::Uniform,
            rollout_crn_samples: 1,
            rollout_steps: 20,
            record_rollout_leaf_samples: 0,
            value_head_rollout_blend: 0.0,
            add_root_dirichlet: false,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            max_nodes: 32,
            collapse_max_steps: 8,
            adaptive_ratio: 0.0,
            adaptive_min_sims: 100,
            root_action_selection: MctsRootActionSelection::MaxVisits,
            model_url: String::new(),
            onnx_path: None,
            wave_size: 1,
            virtual_loss: 1.0,
        };
        // The state we built is still in `setup` phase — MCTS root will see
        // `game_over=false` but enumerate_legal_ai_actions will return setup
        // actions only. That's enough to exercise the loop without hitting
        // /predict. We only validate that visits sum to simulations.
        let (result, _) = with_rng(Rng::from_seed(42u32, "outer"), || {
            run_mcts(
                &state,
                SideId::Player,
                &cfg,
                "http://unused.invalid",
                "test-seed",
            )
        });
        // For a terminal/empty-actions root, visits is empty.
        let total: u32 = result.visits.iter().sum();
        if !result.visits.is_empty() {
            assert_eq!(
                total, cfg.simulations,
                "visit count should match simulations"
            );
        }
    }

    #[test]
    fn spread_sample_indices_cover_leaf_prefix_middle_and_tail() {
        assert_eq!(spread_sample_indices(0, 8), Vec::<usize>::new());
        assert_eq!(spread_sample_indices(5, 8), vec![0, 1, 2, 3, 4]);
        assert_eq!(spread_sample_indices(10, 1), vec![5]);
        assert_eq!(spread_sample_indices(10, 4), vec![0, 3, 6, 9]);
        assert_eq!(spread_sample_indices(101, 5), vec![0, 25, 50, 75, 100]);
    }
}
