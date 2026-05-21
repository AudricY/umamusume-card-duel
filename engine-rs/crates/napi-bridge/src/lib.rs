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
    advance_modeled_turn_step, advance_opponent_turn_step, advance_player_ai_turn_step,
    get_forced_attack_coin_results, state_hash,
};
use engine::headless_setup::setup_ai_vs_ai_game;
use engine::mcts::config::{MctsConfig, MctsLeaf, MctsPrior};
use engine::mcts::driver::run_mcts;
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

/// Drive a full heuristic-vs-heuristic game in pure Rust and return
/// just the summary. Avoids JS↔Rust roundtrip overhead per step.
/// Returns `{finalStateHash, steps, winner, gameOver, terminalReason}`
/// where terminalReason is one of "gameOver" | "stalled" | "maxSteps".
#[napi]
pub fn drive_heuristic_game_json(seed: String, max_steps: u32) -> Result<String> {
    let rng = Rng::from_seed(format!("{}:selfplay", seed).as_str(), "selfplay");
    let (mut state, mut step_rng) = with_rng(rng, || setup_ai_vs_ai_game());
    let mut prior_hash = state_hash(&state);
    let mut terminal = "maxSteps";
    let mut steps = 0u32;
    for s in 0..max_steps {
        steps = s;
        if state.game_over {
            terminal = "gameOver";
            break;
        }
        let side = match state.current_side {
            CurrentSide::Player => SideId::Player,
            CurrentSide::Opponent => SideId::Opponent,
            CurrentSide::Done => {
                terminal = "gameOver";
                break;
            }
        };
        let (next, used_rng) = with_rng(step_rng.clone(), || {
            let forced = get_forced_attack_coin_results(&state);
            let mut s2 = state.clone();
            match side {
                SideId::Player => advance_player_ai_turn_step(&mut s2, forced),
                SideId::Opponent => advance_opponent_turn_step(&mut s2, forced),
            }
            s2
        });
        step_rng = used_rng;
        let next_hash = state_hash(&next);
        if next_hash == prior_hash {
            terminal = "stalled";
            break;
        }
        prior_hash = next_hash;
        state = next;
    }
    let winner = state.winner.map(|s| match s {
        SideId::Player => "player",
        SideId::Opponent => "opponent",
    });
    let summary = serde_json::json!({
        "finalStateHash": state_hash(&state),
        "steps": steps,
        "winner": winner,
        "gameOver": state.game_over,
        "terminalReason": terminal,
    });
    serde_json::to_string(&summary)
        .map_err(|e| napi::Error::from_reason(format!("serialize summary: {e}")))
}

/// Drive a full MCTS-vs-heuristic game in pure Rust and return just
/// the summary. Pattern matches sim-mcts-selfplay exactly:
/// enumerate → MCTS (when legal.len > 1 and model_side) → advance,
/// otherwise fall through to heuristic step.
///
/// Avoids the per-step JSON serialize-deserialize roundtrip the
/// composable bridge functions go through, so a TS caller gets
/// bit-identical output to a subprocess sim-mcts-selfplay invocation.
///
/// `model_side`: "player" or "opponent" — which side runs MCTS.
#[napi]
pub fn drive_mcts_game_json(
    seed: String,
    model_side: String,
    max_steps: u32,
    mcts_args_json: String,
) -> Result<String> {
    let args: McTsArgs = serde_json::from_str(&mcts_args_json)
        .map_err(|e| napi::Error::from_reason(format!("parse mcts args: {e}")))?;
    let config = build_config(&args);
    let model_side = match model_side.as_str() {
        "player" => SideId::Player,
        "opponent" => SideId::Opponent,
        other => {
            return Err(napi::Error::from_reason(format!(
                "model_side must be 'player' or 'opponent', got '{other}'"
            )))
        }
    };

    let rng = Rng::from_seed(format!("{}:selfplay", seed).as_str(), "selfplay");
    let (mut state, mut step_rng) = with_rng(rng, || setup_ai_vs_ai_game());
    let mut prior_hash = state_hash(&state);
    let mut terminal = "maxSteps";
    let mut steps = 0u32;
    let mut model_decisions = 0u32;

    for step in 0..max_steps {
        steps = step;
        if state.game_over {
            terminal = "gameOver";
            break;
        }
        let side = match state.current_side {
            CurrentSide::Player => SideId::Player,
            CurrentSide::Opponent => SideId::Opponent,
            CurrentSide::Done => {
                terminal = "gameOver";
                break;
            }
        };
        let (legal, used_after_legal) = with_rng(step_rng.clone(), || {
            enumerate_legal_ai_actions(&state, side)
        });
        step_rng = used_after_legal;

        let (next, used) = if side == model_side && legal.len() > 1 {
            let mcts_seed = format!(
                "{}:{}:{}:mcts",
                seed,
                if model_side == SideId::Player { "player" } else { "opponent" },
                step
            );
            let (result, used_after_mcts) = with_rng(step_rng.clone(), || {
                run_mcts(&state, side, &config, "", mcts_seed.as_str())
            });
            let idx = result.selected_index.min(legal.len() - 1);
            let chosen = legal[idx].clone();
            model_decisions += 1;
            with_rng(used_after_mcts, || {
                let forced = get_forced_attack_coin_results(&state);
                advance_modeled_turn_step(&state, side, &chosen, forced)
            })
        } else {
            with_rng(step_rng.clone(), || {
                let forced = get_forced_attack_coin_results(&state);
                let mut s = state.clone();
                match side {
                    SideId::Player => advance_player_ai_turn_step(&mut s, forced),
                    SideId::Opponent => advance_opponent_turn_step(&mut s, forced),
                }
                s
            })
        };
        step_rng = used;
        let nh = state_hash(&next);
        if nh == prior_hash {
            terminal = "stalled";
            break;
        }
        prior_hash = nh;
        state = next;
    }
    let winner = state.winner.map(|s| match s {
        SideId::Player => "player",
        SideId::Opponent => "opponent",
    });
    let summary = serde_json::json!({
        "finalStateHash": state_hash(&state),
        "steps": steps,
        "modelDecisions": model_decisions,
        "winner": winner,
        "gameOver": state.game_over,
        "terminalReason": terminal,
    });
    serde_json::to_string(&summary)
        .map_err(|e| napi::Error::from_reason(format!("serialize summary: {e}")))
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

/// MCTS config the TS caller passes in. Mirrors the subset of
/// MctsConfig orchestrators tune; defaults match sim-mcts-selfplay.
#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase", default)]
struct McTsArgs {
    simulations: Option<u32>,
    c_puct: Option<f64>,
    /// "rollout" | "value-head"
    leaf: Option<String>,
    /// "uniform" | "policy"
    prior: Option<String>,
    rollout_crn_samples: Option<u32>,
    rollout_steps: Option<u32>,
    add_root_dirichlet: Option<bool>,
    dirichlet_alpha: Option<f64>,
    dirichlet_epsilon: Option<f64>,
    max_nodes: Option<u32>,
    collapse_max_steps: Option<u32>,
    model_url: Option<String>,
}

fn build_config(args: &McTsArgs) -> MctsConfig {
    let leaf = match args.leaf.as_deref() {
        Some("value-head") => MctsLeaf::ValueHead,
        _ => MctsLeaf::Rollout,
    };
    let prior = match args.prior.as_deref() {
        Some("policy") => MctsPrior::Policy,
        _ => MctsPrior::Uniform,
    };
    MctsConfig {
        simulations: args.simulations.unwrap_or(100),
        c_puct: args.c_puct.unwrap_or(1.5),
        leaf,
        prior,
        rollout_crn_samples: args.rollout_crn_samples.unwrap_or(3),
        rollout_steps: args.rollout_steps.unwrap_or(200),
        add_root_dirichlet: args.add_root_dirichlet.unwrap_or(false),
        dirichlet_alpha: args.dirichlet_alpha.unwrap_or(0.3),
        dirichlet_epsilon: args.dirichlet_epsilon.unwrap_or(0.25),
        max_nodes: args.max_nodes.unwrap_or(5_000),
        collapse_max_steps: args.collapse_max_steps.unwrap_or(64),
        adaptive_ratio: 0.0,
        adaptive_min_sims: 100,
        model_url: args.model_url.clone().unwrap_or_default(),
    }
}

/// Run MCTS at the given state. Returns `{stateJson, rngStateJson, mctsResult}`
/// where mctsResult is the serialized MctsResult (selectedIndex,
/// visits, diagnostics). State is unchanged; this is search-only.
///
/// `mcts_seed` is the fork label (TS parity: `format!("{seed}:{side}:{step}:mcts")`).
#[napi]
pub fn run_mcts_json(
    state_json: String,
    rng_state_json: String,
    mcts_args_json: String,
    mcts_seed: String,
) -> Result<String> {
    let state = parse_state(&state_json)?;
    let rng = parse_rng(&rng_state_json)?;
    let args: McTsArgs = serde_json::from_str(&mcts_args_json)
        .map_err(|e| napi::Error::from_reason(format!("parse mcts args: {e}")))?;
    let config = build_config(&args);
    let side = match state.current_side {
        CurrentSide::Player => SideId::Player,
        CurrentSide::Opponent => SideId::Opponent,
        CurrentSide::Done => {
            return Err(napi::Error::from_reason("game is already over"));
        }
    };
    let model_url = config.model_url.clone();
    let (result, used_rng) = with_rng(rng, || {
        run_mcts(&state, side, &config, model_url.as_str(), mcts_seed.as_str())
    });
    let result_json = serde_json::to_string(&result)
        .map_err(|e| napi::Error::from_reason(format!("serialize mcts result: {e}")))?;
    let state_json2 = serde_json::to_string(&state)
        .map_err(|e| napi::Error::from_reason(format!("serialize state: {e}")))?;
    let rng_state_json2 = serde_json::to_string(&used_rng)
        .map_err(|e| napi::Error::from_reason(format!("serialize rng: {e}")))?;
    let wrapper = serde_json::json!({
        "stateJson": state_json2,
        "rngStateJson": rng_state_json2,
        "mctsResult": result_json,
    });
    serde_json::to_string(&wrapper)
        .map_err(|e| napi::Error::from_reason(format!("serialize wrapper: {e}")))
}

/// Run MCTS, take the most-visited action, apply it. The full
/// "MCTS decision → state advance" cycle. Returns `{stateJson,
/// rngStateJson, mctsResult, chosenActionIndex, chosenActionJson}`.
#[napi]
pub fn mcts_step_json(
    state_json: String,
    rng_state_json: String,
    mcts_args_json: String,
    mcts_seed: String,
) -> Result<String> {
    let state = parse_state(&state_json)?;
    let rng = parse_rng(&rng_state_json)?;
    let args: McTsArgs = serde_json::from_str(&mcts_args_json)
        .map_err(|e| napi::Error::from_reason(format!("parse mcts args: {e}")))?;
    let config = build_config(&args);
    let side = match state.current_side {
        CurrentSide::Player => SideId::Player,
        CurrentSide::Opponent => SideId::Opponent,
        CurrentSide::Done => {
            return Err(napi::Error::from_reason("game is already over"));
        }
    };
    let model_url = config.model_url.clone();
    let mcts_seed_owned = mcts_seed.clone();
    // RNG-call order MATCHES the pure-Rust drivers (sim-mcts-selfplay,
    // headless_drive_smoke): enumerate → run_mcts → advance. Previously
    // this fn did run_mcts → enumerate → advance, which is also valid
    // but produces a different rng advance schedule than production
    // sim-cli. Aligning them means goldens captured one path apply to
    // the other.
    let (legal, used_rng_after_legal) = with_rng(rng, || {
        enumerate_legal_ai_actions(&state, side)
    });
    if legal.is_empty() {
        return Err(napi::Error::from_reason("no legal actions"));
    }
    let (result, used_rng_after_mcts) = with_rng(used_rng_after_legal, || {
        run_mcts(&state, side, &config, model_url.as_str(), mcts_seed_owned.as_str())
    });
    let chosen_idx = result.selected_index.min(legal.len() - 1);
    let chosen = legal[chosen_idx].clone();
    let (next, used_rng_after_step) = with_rng(used_rng_after_mcts, || {
        let forced = get_forced_attack_coin_results(&state);
        advance_modeled_turn_step(&state, side, &chosen, forced)
    });
    let wrapper = serde_json::json!({
        "stateJson": serde_json::to_string(&next)
            .map_err(|e| napi::Error::from_reason(format!("serialize state: {e}")))?,
        "rngStateJson": serde_json::to_string(&used_rng_after_step)
            .map_err(|e| napi::Error::from_reason(format!("serialize rng: {e}")))?,
        "mctsResult": serde_json::to_string(&result)
            .map_err(|e| napi::Error::from_reason(format!("serialize mcts result: {e}")))?,
        "chosenActionIndex": chosen_idx,
        "chosenActionJson": serde_json::to_string(&chosen)
            .map_err(|e| napi::Error::from_reason(format!("serialize action: {e}")))?,
    });
    serde_json::to_string(&wrapper)
        .map_err(|e| napi::Error::from_reason(format!("serialize wrapper: {e}")))
}
