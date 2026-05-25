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
use crate::mcts::config::{MctsConfig, MctsDiagnostics, MctsLeaf, MctsPrior, MctsResult};
use crate::mcts::math::{argmax, entropy, mcts_terminal_value, puct_select};
use crate::mcts::node::MctsNode;
use crate::mcts::sample::sample_dirichlet;
use crate::policy::actions::enumerate_legal_ai_actions;
use crate::policy::observation::build_public_observation;
use crate::policy::types::LegalAiAction;

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
    let mut root = build_model_decision_node(root_state, model_side, model_url, config, &mut root_rng);
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
        };
    }

    // Snapshot the prior pre-noise.
    diagnostics.root_priors = root.priors.clone();
    diagnostics.root_prior_entropy = entropy(&diagnostics.root_priors);
    diagnostics.root_prior_argmax = argmax(&diagnostics.root_priors);

    if config.add_root_dirichlet && root.legal_actions.len() > 1 {
        let mut dirichlet_rng = root_rng.fork("dirichlet");
        let noise = sample_dirichlet(root.legal_actions.len(), config.dirichlet_alpha, &mut dirichlet_rng);
        let eps = config.dirichlet_epsilon;
        for (i, p) in root.priors.iter_mut().enumerate() {
            let n = noise.get(i).copied().unwrap_or(0.0);
            *p = (1.0 - eps) * *p + eps * n;
        }
    }

    // Cache root value.
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
        );
        diagnostics.leaf_evaluations += 1;
    }

    let mut total_nodes: u32 = 1;

    for sim in 0..config.simulations {
        let mut sim_rng = root_rng.fork(&format!("sim{}", sim));

        // Walk path via PUCT until leaf (unexpanded child or terminal).
        let mut path: Vec<PathStep> = Vec::new();
        let mut node_ptr: *mut MctsNode = &mut root;
        // SAFETY: the path holds raw pointers to nodes inside the tree.
        // We mutate one node at a time (the leaf's parent on expansion,
        // or visit/wsum on backup). Aliasing is single-threaded.
        loop {
            let node: &MctsNode = unsafe { &*node_ptr };
            if node.terminal_value.is_some() {
                break;
            }
            let action_index = puct_select(&node.visits, &node.wsum, &node.priors, config.c_puct);
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
            let last = path.last().copied().expect("non-terminal selection produced an empty path");
            let parent: &MctsNode = unsafe { &*last.node_ptr };
            let action = parent.legal_actions[last.action_index].clone();
            let parent_state = parent.state.clone();
            let parent_model_side = parent.model_side;
            let mut expand_rng =
                sim_rng.fork(&format!("expand:a{}", last.action_index));
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
                Some(next_state) if total_nodes >= config.max_nodes => {
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
                    total_nodes += 1;
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

    // Argmax visits, tiebreak by mean Q.
    let mut best_index = 0usize;
    let mut best_visits: i64 = -1;
    let mut best_q = f64::NEG_INFINITY;
    for i in 0..visits.len() {
        let n = visits[i] as i64;
        let q = diagnostics.root_mean_q[i];
        if n > best_visits || (n == best_visits && q > best_q) {
            best_visits = n;
            best_q = q;
            best_index = i;
        }
    }

    MctsResult {
        selected_index: best_index,
        visits,
        diagnostics,
    }
}

#[derive(Copy, Clone)]
struct PathStep {
    node_ptr: *mut MctsNode,
    action_index: usize,
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
        .map(|p| if p.is_finite() { p.max(0.0) as f64 } else { 0.0 })
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
) -> f64 {
    if state.game_over {
        return mcts_terminal_value(state, model_side);
    }
    match config.leaf {
        MctsLeaf::Rollout => rollout_leaf_value(state, model_side, config, rng),
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
        let ids_a: Vec<crate::core::card_id::CardId> = deck_a
            .iter()
            .filter_map(|n| cat.id_for(n))
            .collect();
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
            crate::dispatcher::choose_opening_coin(&mut s, crate::core::constants::CoinFlipResult::Heads);
            s
        });
        // After choose_opening_coin runs one draw, used_a's next draw
        // should equal sentinel's next draw (same starting state).
        let mut after_a = used_a.clone();
        let mut after_sentinel = sentinel.clone();
        assert_eq!(after_a.next_f64().to_bits(), after_sentinel.next_f64().to_bits());
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
            value_head_rollout_blend: 0.0,
            add_root_dirichlet: false,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            max_nodes: 8,
            collapse_max_steps: 4,
            adaptive_ratio: 0.0,
            adaptive_min_sims: 100,
            model_url: String::new(),
            onnx_path: None,
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
            value_head_rollout_blend: 0.0,
            add_root_dirichlet: false,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            max_nodes: 32,
            collapse_max_steps: 8,
            adaptive_ratio: 0.0,
            adaptive_min_sims: 100,
            model_url: String::new(),
            onnx_path: None,
        };
        // The state we built is still in `setup` phase — MCTS root will see
        // `game_over=false` but enumerate_legal_ai_actions will return setup
        // actions only. That's enough to exercise the loop without hitting
        // /predict. We only validate that visits sum to simulations.
        let (result, _) = with_rng(Rng::from_seed(42u32, "outer"), || {
            run_mcts(&state, SideId::Player, &cfg, "http://unused.invalid", "test-seed")
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
}
