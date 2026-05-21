//! Port of `frontend/src/game/engine/ai-policy/`.
//!
//! Phase 1e scope: legal-action enumeration + small support types. The
//! observation types (`PublicObservation`, `PublicSideObservation`, etc.)
//! are model-interface concerns and land in a later phase only if the
//! Rust binary needs to talk to `serve_onnx`.

pub mod card_vocab;
pub mod phase;
pub mod types;
// pub mod actions; — Phase 1e remaining work (666 LOC enumerator).
