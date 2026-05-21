//! Port of `frontend/src/game/engine/ai-policy/`.
//!
//! Phase 1e scope: legal-action enumeration + small support types. The
//! observation types (`PublicObservation`, `PublicSideObservation`, etc.)
//! are model-interface concerns and land in a later phase only if the
//! Rust binary needs to talk to `serve_onnx`.

pub mod actions;
pub mod card_vocab;
/// R16-P3 throughput-spike Option A: in-process v3.0 featurizer.
pub mod featurize;
pub mod observation;
pub mod phase;
pub mod types;
