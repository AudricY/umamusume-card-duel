//! Port of `frontend/src/game/engine/flow/`.
//!
//! Only the pure helpers and predicates land in this commit. The big
//! state-mutating files (turn.rs, combat.rs, trainers.rs, setup.rs,
//! playRules.rs, evolution.rs, board.rs) are tracked in Phase 1d of the
//! hand-off doc.

pub mod ability_rules;
pub mod board;
pub mod eligibility;
pub mod energy;
pub mod evolution;
pub mod play_rules;
pub mod retreat;
pub mod setup;
pub mod special_conditions;
pub mod turn;
