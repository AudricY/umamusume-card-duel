//! Phase 2 NAPI scaffolding. Exposes a minimal Rust-engine surface to
//! Node so the existing TS pipeline can call the bit-identical Rust
//! port without spawning the sim-cli binaries.
//!
//! Status: starter set — `setup_ai_vs_ai_game` + `state_hash`. Used to
//! prove the bridge works end-to-end before expanding the surface.

use napi::Result;
use napi_derive::napi;

use engine::core::random::{with_rng, Rng};
use engine::dispatcher::state_hash;
use engine::headless_setup::setup_ai_vs_ai_game;

/// Seed a fresh AI-vs-AI game and return the serialized initial state
/// as a JSON string (the same camelCase shape sim-export-training
/// emits). Node decodes with `JSON.parse`.
///
/// Returning a string instead of a JS object keeps the binding simple
/// and avoids replicating the full GameState typescript surface here.
#[napi]
pub fn create_game_json(seed: String) -> Result<String> {
    let rng = Rng::from_seed(format!("{}:selfplay", seed).as_str(), "selfplay");
    let (state, _used) = with_rng(rng, || setup_ai_vs_ai_game());
    serde_json::to_string(&state).map_err(|e| napi::Error::from_reason(format!("serialize: {e}")))
}

/// Return the structural fingerprint hash of the JSON-encoded game
/// state. Mirrors `engine::dispatcher::state_hash` exactly, so a TS
/// caller that parses → re-serializes can verify bit-identical engine
/// behavior end-to-end.
#[napi]
pub fn state_hash_for_json(state_json: String) -> Result<String> {
    let state: engine::core::state::GameState = serde_json::from_str(&state_json)
        .map_err(|e| napi::Error::from_reason(format!("parse: {e}")))?;
    Ok(state_hash(&state))
}

/// Sentinel function — Node can call this to confirm the bridge is
/// loaded and that engine::core::catalog initializes correctly.
#[napi]
pub fn engine_version() -> String {
    format!(
        "rust-port v0.1.0 — catalog={} cards",
        engine::core::catalog::catalog().len()
    )
}
