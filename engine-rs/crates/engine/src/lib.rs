//! Bit-identical Rust port of `frontend/src/game/engine/`.
//!
//! Scope and conformance rules live in
//! `docs/ai-research/scoping/rust-engine-port-plan.md`.
//!
//! Module layout mirrors the TS source tree:
//! - `core` ↔ `frontend/src/game/engine/core/`
//! - `flow` ↔ `frontend/src/game/engine/flow/` (excluding ai/)
//! - `ai`   ↔ `frontend/src/game/engine/flow/ai/`
//! - `policy` ↔ `frontend/src/game/engine/ai-policy/`
//! - `fingerprint` ↔ `backend/src/sim/stateFingerprint.ts`

pub mod core;
pub mod fingerprint;
pub mod flow;
pub mod policy;

pub use crate::core::random;
