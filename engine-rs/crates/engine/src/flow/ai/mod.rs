//! Port of `frontend/src/game/engine/flow/ai/` — the heuristic opponent.
//!
//! See `docs/ai-research/scoping/rust-engine-port-handoff.md` Phase 1f.
//!
//! Bit-identity rules for this subtree:
//! - **Float math is in scope.** AI scoring uses `* 0.5`, `* 0.4`,
//!   `(a + b) / 2`, etc. Preserve TS source-text operator order
//!   exactly. IEEE 754 round-to-nearest-even is deterministic given
//!   identical op-order.
//! - `Number(score.toFixed(2))` ports as
//!   `(score * 100.0).round() / 100.0`.
//! - **Map/Set iteration order is load-bearing.** Use `IndexMap` (not
//!   `HashMap`) or `Vec<(K, V)>` for `usefulCapByUid` / `beforeByUid`.
//! - **Module-global `seenTurnGoalKeys`** in `telemetry.rs` is a
//!   thread-local `RefCell<BTreeSet<String>>` to mirror the TS contract
//!   while staying single-threaded.
//! - **No log calls** — `state.log` is not in the fingerprint contract.

pub mod ability_utils;
pub mod attach_utils;
pub mod combat_planner;
pub mod combat_utils;
pub mod core;
pub mod deck_inference;
pub mod energy_awareness;
pub mod mid_level;
pub mod opponent_heuristics;
pub mod public_info;
pub mod setup_selection;
pub mod telemetry;
pub mod trainer_utils;
pub mod turn_plan;
pub mod types;
