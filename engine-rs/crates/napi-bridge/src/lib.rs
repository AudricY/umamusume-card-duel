//! Phase 2 NAPI scaffolding. Exposes a minimal Rust-engine surface to
//! Node so the existing TS pipeline can call the bit-identical Rust
//! port without spawning the sim-cli binaries.
//!
//! Status: setup + advance + enumerate legal actions + state-hash.
//! Lets a TS caller drive a full heuristic-vs-heuristic game without
//! subprocesses. MCTS surface lands in a follow-up.

use napi::Result;
use napi_derive::napi;
use serde::{Deserialize, Serialize};

use engine::core::constants::SideId;
use engine::core::random::{with_rng, Rng};
use engine::core::state::{CurrentSide, GameState};
use engine::dispatcher::{
    advance_opponent_turn_step, advance_player_ai_turn_step, get_forced_attack_coin_results,
    state_hash,
};
use engine::headless_setup::setup_ai_vs_ai_game;
use engine::policy::actions::enumerate_legal_ai_actions;

/// Result of a successful setup or step call: serialized state plus
/// the RNG snapshot Node should pass back into the next call to keep
/// the trace deterministic.
#[derive(Serialize, Deserialize)]
struct StepResult {
    #[serde(rename = "stateJson")]
    state_json: String,
    #[serde(rename = "rngStateJson")]
    rng_state_json: String,
}

fn parse_state(state_json: &str) -> Result<GameState> {
    serde_json::from_str(state_json)
        .map_err(|e| napi::Error::from_reason(format!("parse state: {e}")))
}

fn parse_rng(rng_state_json: &str) -> Result<Rng> {
    serde_json::from_str(rng_state_json)
        .map_err(|e| napi::Error::from_reason(format!("parse rng: {e}")))
}

fn pack_step(state: &GameState, rng: &Rng) -> Result<String> {
    let state_json = serde_json::to_string(state)
        .map_err(|e| napi::Error::from_reason(format!("serialize state: {e}")))?;
    let rng_state_json = serde_json::to_string(rng)
        .map_err(|e| napi::Error::from_reason(format!("serialize rng: {e}")))?;
    serde_json::to_string(&StepResult {
        state_json,
        rng_state_json,
    })
    .map_err(|e| napi::Error::from_reason(format!("serialize step: {e}")))
}

/// Seed a fresh AI-vs-AI game. Returns `{stateJson, rngStateJson}` as
/// a JSON string Node `JSON.parse`s. The returned rngStateJson must
/// be passed back into subsequent calls so step sequencing matches
/// the Rust-side driver.
#[napi]
pub fn create_game_json(seed: String) -> Result<String> {
    let rng = Rng::from_seed(format!("{}:selfplay", seed).as_str(), "selfplay");
    let (state, used_rng) = with_rng(rng, || setup_ai_vs_ai_game());
    pack_step(&state, &used_rng)
}

/// Drive one heuristic-AI step for the side whose turn it is. Returns
/// `{stateJson, rngStateJson}`. Caller is responsible for terminating
/// when state.gameOver flips true.
#[napi]
pub fn advance_step_json(state_json: String, rng_state_json: String) -> Result<String> {
    let state = parse_state(&state_json)?;
    let rng = parse_rng(&rng_state_json)?;
    let side = match state.current_side {
        CurrentSide::Player => SideId::Player,
        CurrentSide::Opponent => SideId::Opponent,
        CurrentSide::Done => {
            return Err(napi::Error::from_reason("game is already over"));
        }
    };
    let (next, used_rng) = with_rng(rng, || {
        let forced = get_forced_attack_coin_results(&state);
        let mut s = state.clone();
        match side {
            SideId::Player => advance_player_ai_turn_step(&mut s, forced),
            SideId::Opponent => advance_opponent_turn_step(&mut s, forced),
        }
        s
    });
    pack_step(&next, &used_rng)
}

/// Return the JSON-encoded list of legal AI actions at the current
/// state. Side defaults to `state.currentSide`; callers wanting the
/// non-active side must enumerate before stepping (this is rarely
/// useful — the AI never decides for the side not on the clock).
#[napi]
pub fn legal_actions_json(state_json: String, rng_state_json: String) -> Result<String> {
    let state = parse_state(&state_json)?;
    let rng = parse_rng(&rng_state_json)?;
    let side = match state.current_side {
        CurrentSide::Player => SideId::Player,
        CurrentSide::Opponent => SideId::Opponent,
        CurrentSide::Done => {
            return Err(napi::Error::from_reason("game is already over"));
        }
    };
    let (legal, _) = with_rng(rng, || enumerate_legal_ai_actions(&state, side));
    serde_json::to_string(&legal)
        .map_err(|e| napi::Error::from_reason(format!("serialize legal: {e}")))
}

/// Return the structural fingerprint hash of the JSON-encoded game
/// state. Mirrors `engine::dispatcher::state_hash` exactly.
#[napi]
pub fn state_hash_for_json(state_json: String) -> Result<String> {
    let state = parse_state(&state_json)?;
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
