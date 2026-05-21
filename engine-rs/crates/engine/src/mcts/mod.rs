//! Bit-identical port of `backend/src/sim/mcts.ts`.
//!
//! Phase 1g progress:
//! - Config + diagnostics + result types — done.
//! - Pure math (puct_select, entropy, argmax, sample_dirichlet,
//!   sample_gamma, mcts_terminal_value) — done.
//! - Tree expansion + leaf evaluation + game-stepping — pending Phase 1e
//!   (action enumerator) and Phase 1f (heuristic rollout policy).
//!
//! The inner RNG tree split called out in the scoping doc's Open Q5 is
//! preserved: the MCTS driver creates `Rng::from_seed(seed_str,
//! "mcts-root")` separately from any outer recorder/selfplay RNG.

pub mod config;
pub mod math;
pub mod node;
pub mod sample;
