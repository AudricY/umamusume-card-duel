//! MCTS tree node — port of `MctsNode` from `mcts.ts:126`.
//!
//! Phase 1g scaffold: defines the node shape and a few helpers. The tree
//! expansion / leaf-evaluation loop is deferred until the action
//! enumerator (Phase 1e) and the heuristic rollout policy (Phase 1f)
//! land. Once they do, this module gets the `build_model_decision_node`
//! and `mcts_decide` driver functions ported.

use crate::core::constants::SideId;
use crate::core::state::GameState;
use crate::policy::types::LegalAiAction;

#[derive(Debug, Clone)]
pub struct MctsNode {
    pub state: GameState,
    pub model_side: SideId,
    pub legal_actions: Vec<LegalAiAction>,
    pub priors: Vec<f64>,
    pub visits: Vec<u32>,
    pub wsum: Vec<f64>,
    /// One entry per legal action. Children populated lazily on first
    /// expansion of that action slot.
    pub children: Vec<Option<Box<MctsNode>>>,
    /// `None` for non-terminal interior nodes; `Some(v)` for terminal
    /// states with value already determined in `modelSide`'s frame.
    pub terminal_value: Option<f64>,
    /// Cached leaf value from the value-head `/predict` response, so
    /// backup doesn't need a second round trip. Mirrors
    /// `cachedLeafValue` in TS.
    pub cached_leaf_value: Option<f64>,
}

impl MctsNode {
    /// Construct a terminal sentinel node (TS: the `state.gameOver`
    /// fast path in `buildModelDecisionNode`).
    pub fn terminal(state: GameState, model_side: SideId, value: f64) -> Self {
        MctsNode {
            state,
            model_side,
            legal_actions: Vec::new(),
            priors: Vec::new(),
            visits: Vec::new(),
            wsum: Vec::new(),
            children: Vec::new(),
            terminal_value: Some(value),
            cached_leaf_value: None,
        }
    }

    pub fn is_terminal(&self) -> bool {
        self.terminal_value.is_some()
    }

    pub fn total_visits(&self) -> u32 {
        self.visits.iter().sum()
    }
}
