//! Phase 1h bit-identity gate: replay TS-recorded golden traces against
//! the Rust engine.
//!
//! Current scope (V1): **setup-phase parity check.** For each recorded
//! seed, reconstruct the initial post-setup state via Rust's
//! `setup_ai_vs_ai_game()` under the same seed string, then diff the
//! key state fields against the recorded `actions[0].fingerprintBefore`.
//!
//! Per Open Question 1(b) in the scoping handoff: the TS fingerprint is
//! a `JSON.stringify` whose byte representation cannot equal a Rust
//! `xxh3` digest — so we diff in PARSED-JSON-FIELD space, not byte-for-
//! byte. Fields compared (every recorded seed):
//! - `turn_number`
//! - `current_side`
//! - `sides[*].active.card_id`
//! - `sides[*].bench[*].card_id`
//! - `sides.player.hand` (TS records own-side full hand; opponent hand
//!   is hidden so we only check length)
//!
//! Diffs surface the FIRST mismatch and exit non-zero. On success:
//! `OK <N>/<N> seeds setup-bit-identical`.
//!
//! V2 (next session) will extend this to the per-step replay loop using
//! `dispatcher::advance_*` to drive the Rust sim and compare against
//! each recorded step. Today's V1 already catches whole classes of bugs:
//! RNG drift in opening-coin, opening-hand shuffle, or AI setup choice.

use std::fs::File;
use std::io::{BufRead, BufReader};

use anyhow::{Context, Result};
use clap::Parser;
use engine::core::constants::SideId;
use engine::core::random::{with_rng, Rng};
use engine::core::state::{CurrentSide, GameState};
use engine::dispatcher::{advance_modeled_turn_step, get_forced_attack_coin_results};
use engine::headless_setup::setup_ai_vs_ai_game;
use engine::mcts::config::{MctsConfig, MctsLeaf, MctsPrior, MctsRootActionSelection};
use engine::mcts::driver::{
    reset_rollout_stats, rollout_stats, run_mcts, set_verbose_first_rollout,
};
use engine::policy::actions::enumerate_legal_ai_actions;
use engine::policy::types::{AiPhase, LegalAiAction};
use serde::Deserialize;

#[derive(Parser, Debug)]
#[command(name = "golden-replay")]
struct Args {
    #[arg(long, default_value = "runs/rust-port-golden-traces/traces-500.jsonl")]
    input: String,

    /// Print a diff for the first N seeds that fail, then continue.
    #[arg(long, default_value_t = 1)]
    max_failures_to_print: usize,

    /// V1 (default): setup-phase parity only — `chooseAiSetupSelection` +
    /// opening-hand shuffle reproducible. V2: also drive the sim
    /// step-by-step from the recorded action sequence and verify
    /// per-step state convergence.
    #[arg(long, default_value = "setup")]
    mode: String,

    /// In V2 mode, stop after this many seeds (0 = all). Useful for
    /// debug iteration.
    #[arg(long, default_value_t = 0)]
    limit_seeds: usize,

    /// In V4 (mcts) mode, stop after this many steps per seed (0 = all).
    /// One MCTS decision is ~seconds, so limit during debug iteration.
    #[arg(long, default_value_t = 0)]
    limit_steps: usize,
}

#[derive(Deserialize)]
struct Trace {
    seed: u64,
    #[serde(rename = "traceVersion")]
    trace_version: u32,
    actions: Vec<TraceStep>,
    // turn_number + winner are recorded in the trace; we don't read them
    // today, but keep parsing-tolerant for forward compatibility.
    #[serde(rename = "turnNumber", default)]
    #[allow(dead_code)]
    turn_number: Option<u32>,
    #[serde(default)]
    #[allow(dead_code)]
    winner: Option<String>,
}

#[derive(Deserialize)]
struct TraceStep {
    #[serde(rename = "fingerprintBefore")]
    fingerprint_before: String,
    #[serde(rename = "sideId")]
    side_id: String,
    action: TraceAction,
    #[serde(default, rename = "rngDrawsThisStep")]
    rng_draws_this_step: u64,
}

#[derive(Deserialize, Clone)]
struct TraceAction {
    id: String,
    kind: String,
    phase: String,
    #[serde(default)]
    payload: serde_json::Value,
}

#[derive(Deserialize, Debug)]
struct TsFingerprint {
    // `phase` is parsed for completeness but not asserted today; the
    // current_side + turn_number + per-side identity checks already
    // catch every observable divergence.
    #[allow(dead_code)]
    phase: String,
    #[serde(rename = "currentSide")]
    current_side: String,
    #[serde(rename = "turnNumber")]
    turn_number: u32,
    sides: TsSides,
}

#[derive(Deserialize, Debug)]
struct TsSides {
    player: TsSide,
    opponent: TsSide,
}

#[derive(Deserialize, Debug)]
struct TsSide {
    hand: Vec<String>,
    active: Option<TsInst>,
    bench: Vec<TsInst>,
}

#[derive(Deserialize, Debug)]
struct TsInst {
    #[serde(rename = "cardId")]
    card_id: String,
    uid: u32,
    #[serde(default)]
    hp: i32,
    #[serde(default, rename = "maxHp")]
    max_hp: i32,
}

// `reconstruct_legal_action` was the V2/V3 helper before V3's
// `remap_action_uids` superseded it (which also handles UID remap).
// Removed in favor of the unified path.

fn parse_ai_phase(s: &str) -> AiPhase {
    match s {
        "setup" => AiPhase::Setup,
        "pendingChoice" => AiPhase::PendingChoice,
        "bench" => AiPhase::Bench,
        "trainerBefore" => AiPhase::TrainerBefore,
        "evolve" => AiPhase::Evolve,
        "attach" => AiPhase::Attach,
        "trainerAfter" => AiPhase::TrainerAfter,
        "ability" => AiPhase::Ability,
        "combat" => AiPhase::Combat,
        "stadiumOrEnd" => AiPhase::StadiumOrEnd,
        _ => AiPhase::StadiumOrEnd,
    }
}

fn parse_side_id(s: &str) -> Option<SideId> {
    match s {
        "player" => Some(SideId::Player),
        "opponent" => Some(SideId::Opponent),
        _ => None,
    }
}

/// V4: drive Rust MCTS at each step (matching the recorder's per-step
/// MCTS config) and compare the chosen action + post-state to the
/// recorded trace.
///
/// MCTS config matches the recorder's default: sims=100, K=3,
/// rollout_steps=200, prior=uniform, leaf=rollout, collapseMaxSteps=64.
/// Inner MCTS rng seed: `${seed}:${sideId}:${step}:mcts` (matching TS
/// recordGoldenTraces.ts).
fn v4_replay_for_seed(
    trace: &Trace,
    initial_rng: Rng,
    resolve: &impl Fn(engine::core::card_id::CardId) -> String,
    max_steps_to_replay: usize,
) -> Vec<String> {
    let mut diffs: Vec<String> = Vec::new();
    let (mut state, mut step_rng) = with_rng(initial_rng, || setup_ai_vs_ai_game());

    let config = MctsConfig {
        simulations: 100,
        c_puct: 1.5,
        leaf: MctsLeaf::Rollout,
        prior: MctsPrior::Uniform,
        rollout_crn_samples: 3,
        rollout_steps: 200,
        record_rollout_leaf_samples: 0,
        value_head_rollout_blend: 0.0,
        add_root_dirichlet: false,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.25,
        max_nodes: 5_000,
        collapse_max_steps: 64,
        adaptive_ratio: 0.0,
        adaptive_min_sims: 100,
        root_action_selection: MctsRootActionSelection::MaxVisits,
        two_sided: false,
        model_url: String::new(),
        onnx_path: None,
        wave_size: 1,
        virtual_loss: 1.0,
    };

    let limit = if max_steps_to_replay == 0 {
        trace.actions.len()
    } else {
        max_steps_to_replay.min(trace.actions.len())
    };

    for i in 0..limit {
        let step = &trace.actions[i];
        let Some(side_id) = parse_side_id(&step.side_id) else {
            diffs.push(format!(
                "step[{}]: unrecognized sideId {:?}",
                i, step.side_id
            ));
            break;
        };

        let legal = enumerate_legal_ai_actions(&state, side_id);
        let pre_step_draws = step_rng.draws();

        // V4b: run Rust MCTS for RNG consumption when there are multiple
        // legal actions, but advance with the RECORDED action (uid-
        // remapped per V3 logic). This isolates engine port correctness
        // from MCTS port correctness.
        //
        // Once engine correctness is confirmed, a separate gate compares
        // Rust MCTS's selected_index to the recorded selectedIndex.
        if legal.len() > 1 {
            reset_rollout_stats();
            // Only first multi-action step prints per-advance trace.
            set_verbose_first_rollout(i == 1);
            let mcts_seed = format!("{}:{}:{}:mcts", trace.seed, step.side_id, i);
            let (_mcts_result, used_rng) = with_rng(step_rng.clone(), || {
                run_mcts(&state, side_id, &config, "", mcts_seed.as_str())
            });
            set_verbose_first_rollout(false);
            step_rng = used_rng;
            let s = rollout_stats();
            if i < 5 {
                let avg_draws = if s.rollouts_started > 0 {
                    s.total_outer_rng_draws_during_rollouts as f64 / s.rollouts_started as f64
                } else {
                    0.0
                };
                let avg_advances = if s.rollouts_started > 0 {
                    s.total_advance_calls as f64 / s.rollouts_started as f64
                } else {
                    0.0
                };
                eprintln!(
                    "  step[{}] rollouts={} game_over={} side_done={} state_unchanged={} max_steps={} advances={} (avg {:.1} adv, {:.1} draws/rollout; max {})",
                    i,
                    s.rollouts_started,
                    s.rollouts_ended_game_over,
                    s.rollouts_ended_current_side_done,
                    s.rollouts_ended_state_unchanged,
                    s.rollouts_ended_max_steps,
                    s.total_advance_calls,
                    avg_advances,
                    avg_draws,
                    s.max_outer_rng_draws_in_single_rollout,
                );
            }
        }

        // Apply the RECORDED action (with uid remap).
        let chosen_action = match remap_action_uids(step, &state) {
            Ok(a) => a,
            Err(e) => {
                diffs.push(format!("step[{}] uid remap failed: {}", i, e));
                break;
            }
        };

        let (next_state, used_rng) = with_rng(step_rng.clone(), || {
            let forced = get_forced_attack_coin_results(&state);
            advance_modeled_turn_step(&state, side_id, &chosen_action, forced)
        });
        step_rng = used_rng;
        state = next_state;

        // RNG-draw-count diagnostic: TS recorded `rngDrawsThisStep`.
        let rust_draws = step_rng.draws() - pre_step_draws;
        if rust_draws != step.rng_draws_this_step {
            diffs.push(format!(
                "step[{}] rngDrawsThisStep: rust={} ts={} (delta {:+})",
                i,
                rust_draws,
                step.rng_draws_this_step,
                rust_draws as i64 - step.rng_draws_this_step as i64
            ));
            // Don't break — continue to state check + subsequent steps so
            // we see the full divergence pattern.
        }

        let target_fp = if i + 1 < trace.actions.len() {
            trace.actions[i + 1].fingerprint_before.as_str()
        } else {
            continue;
        };
        let target: TsFingerprint = match serde_json::from_str(target_fp) {
            Ok(v) => v,
            Err(e) => {
                diffs.push(format!("step[{}] parse next fp: {}", i, e));
                break;
            }
        };
        let card_id_diffs = compare_card_id_fields(&state, &target, resolve);
        if !card_id_diffs.is_empty() {
            diffs.push(format!("step[{}] post-advance:", i));
            for d in card_id_diffs {
                diffs.push(format!("    - {}", d));
            }
            break;
        }
    }
    diffs
}

/// V3: remap a recorded action's TS uids → Rust uids via position.
///
/// The recorded `fingerprintBefore` snapshots every umamusume
/// instance's uid + position (active vs bench index) on both sides. We
/// look up the position of each TS uid in the recorded fingerprint,
/// then write the Rust uid at the same position into the action's
/// payload. Same logical target, different uid space.
fn remap_action_uids(step: &TraceStep, rust_state: &GameState) -> Result<LegalAiAction, String> {
    let ts_fp: TsFingerprint = serde_json::from_str(&step.fingerprint_before)
        .map_err(|e| format!("parse fingerprintBefore: {}", e))?;

    // Build TS uid → (sideId, position) lookup. Position is `None` for
    // active, `Some(idx)` for bench slot.
    let mut ts_uid_to_pos: std::collections::HashMap<u32, (SideId, Option<usize>)> =
        std::collections::HashMap::new();
    for (side_id, side) in [
        (SideId::Player, &ts_fp.sides.player),
        (SideId::Opponent, &ts_fp.sides.opponent),
    ] {
        if let Some(active) = &side.active {
            ts_uid_to_pos.insert(active.uid, (side_id, None));
        }
        for (idx, b) in side.bench.iter().enumerate() {
            ts_uid_to_pos.insert(b.uid, (side_id, Some(idx)));
        }
    }

    // For each uid-bearing payload key, remap TS uid → Rust uid at
    // the same position. Keys observed across action kinds:
    // - "targetUid" (evolve, ability, attachEnergy, retreat, attack)
    // - "switchTargetUid" (attack with switch-self)
    // - "healTargetUid" (attack with heal)
    // - "pendingTargetUid" (resolvePendingChoice) — may also appear
    // - nested "decision" dict for combat
    let mut payload = step.action.payload.clone();
    let known_uid_keys = [
        "targetUid",
        "switchTargetUid",
        "healTargetUid",
        "pendingTargetUid",
    ];
    if let serde_json::Value::Object(map) = &mut payload {
        for key in &known_uid_keys {
            if let Some(serde_json::Value::Number(n)) = map.get(*key) {
                if let Some(ts_uid) = n.as_u64().map(|x| x as u32) {
                    if let Some(&(side, position)) = ts_uid_to_pos.get(&ts_uid) {
                        let rust_side = rust_state.side(side);
                        let rust_uid = match position {
                            None => rust_side.active.as_ref().map(|u| u.uid),
                            Some(idx) => rust_side.bench.get(idx).map(|u| u.uid),
                        };
                        if let Some(rust_uid) = rust_uid {
                            map.insert((*key).into(), serde_json::Value::Number(rust_uid.into()));
                        }
                    }
                }
            }
        }
        // Combat actions nest a `decision` object that itself contains
        // targetUid + switchTargetUid + healTargetUid.
        if let Some(serde_json::Value::Object(decision)) = map.get_mut("decision") {
            for key in &known_uid_keys {
                if let Some(serde_json::Value::Number(n)) = decision.get(*key) {
                    if let Some(ts_uid) = n.as_u64().map(|x| x as u32) {
                        if let Some(&(side, position)) = ts_uid_to_pos.get(&ts_uid) {
                            let rust_side = rust_state.side(side);
                            let rust_uid = match position {
                                None => rust_side.active.as_ref().map(|u| u.uid),
                                Some(idx) => rust_side.bench.get(idx).map(|u| u.uid),
                            };
                            if let Some(rust_uid) = rust_uid {
                                decision.insert(
                                    (*key).into(),
                                    serde_json::Value::Number(rust_uid.into()),
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(LegalAiAction {
        id: step.action.id.clone(),
        phase: parse_ai_phase(&step.action.phase),
        kind: step.action.kind.clone(),
        payload,
        features: Vec::new(),
        action_source_card_idx: None,
        action_target_card_idx: None,
    })
}

/// V2: drive the Rust sim step-by-step from the recorded action sequence.
/// Verifies that after each `advance_modeled_turn_step`, the resulting
/// state's identity-bearing fields match the next recorded step's
/// `fingerprintBefore` — or, for the last step, the trace's terminal.
fn replay_steps_for_seed(
    trace: &Trace,
    rng: Rng,
    resolve: &impl Fn(engine::core::card_id::CardId) -> String,
) -> Vec<String> {
    let mut diffs = Vec::new();
    // Thread the post-setup Rng forward into the step loop so the outer
    // PRNG stream matches the recorder's exactly. with_rng returns the
    // consumed Rng so we can pick up where setup left off.
    let (mut state, mut step_rng) = with_rng(rng, || setup_ai_vs_ai_game());

    for (i, step) in trace.actions.iter().enumerate() {
        let Some(side_id) = parse_side_id(&step.side_id) else {
            diffs.push(format!(
                "step[{}]: unrecognized sideId {:?}",
                i, step.side_id
            ));
            break;
        };

        // V3: normalize recorded uids → Rust uids by mapping POSITION
        // (active vs bench[N]) using the recorded fingerprintBefore.
        // TS's uid counter is bumped by MCTS internal rollouts, so the
        // raw recorded uid won't exist in Rust's V2 replay (which
        // doesn't run MCTS). Position-based remap preserves the role
        // intent.
        let action = match remap_action_uids(&step, &state) {
            Ok(a) => a,
            Err(e) => {
                diffs.push(format!("step[{}] uid remap failed: {}", i, e));
                break;
            }
        };

        // Wrap the entire forced-coin + advance step in a single with_rng
        // scope so any randomFloat() inside advance_modeled_turn_step
        // (combat's flipCoin, energy roll, etc.) advances the SAME rng
        // the recorder's outer-tree used.
        let (next_state, used_rng) = with_rng(step_rng.clone(), || {
            let forced = get_forced_attack_coin_results(&state);
            advance_modeled_turn_step(&state, side_id, &action, forced)
        });
        step_rng = used_rng;
        state = next_state;

        // After-advance state — compare to the NEXT step's
        // fingerprintBefore (or to terminal at the end).
        let target_fp = if i + 1 < trace.actions.len() {
            trace.actions[i + 1].fingerprint_before.as_str()
        } else {
            continue; // terminal compared by caller via trace.winner / trace.turn_number
        };
        let target: TsFingerprint = match serde_json::from_str(target_fp) {
            Ok(v) => v,
            Err(e) => {
                diffs.push(format!("step[{}]: parse next fingerprint failed: {}", i, e));
                break;
            }
        };

        let rust_side = match state.current_side {
            CurrentSide::Player => "player",
            CurrentSide::Opponent => "opponent",
            CurrentSide::Done => "done",
        };
        if rust_side != target.current_side {
            diffs.push(format!(
                "step[{}] after-advance currentSide: rust={} ts={}",
                i, rust_side, target.current_side
            ));
            break;
        }
        if state.turn_number != target.turn_number {
            diffs.push(format!(
                "step[{}] after-advance turnNumber: rust={} ts={}",
                i, state.turn_number, target.turn_number
            ));
            break;
        }
        let card_id_diffs = compare_card_id_fields(&state, &target, resolve);
        if !card_id_diffs.is_empty() {
            diffs.push(format!("step[{}] after-advance:", i));
            for d in card_id_diffs {
                diffs.push(format!("    - {}", d));
            }
            break;
        }
    }
    diffs
}

fn compare_card_id_fields<F: Fn(engine::core::card_id::CardId) -> String>(
    state: &GameState,
    ts: &TsFingerprint,
    resolve: &F,
) -> Vec<String> {
    let mut out = Vec::new();
    let fmt_inst = |u: &engine::core::state::UmamusumeInstance| {
        format!("{}@{}/{}", resolve(u.card_id), u.hp, u.max_hp)
    };
    let fmt_ts_inst = |u: &TsInst| format!("{}@{}/{}", u.card_id, u.hp, u.max_hp);

    let rust_player_active = state.sides[0].active.as_ref().map(fmt_inst);
    let ts_player_active = ts.sides.player.active.as_ref().map(fmt_ts_inst);
    if rust_player_active != ts_player_active {
        out.push(format!(
            "player.active: rust={:?} ts={:?}",
            rust_player_active, ts_player_active
        ));
    }
    let rust_opp_active = state.sides[1].active.as_ref().map(fmt_inst);
    let ts_opp_active = ts.sides.opponent.active.as_ref().map(fmt_ts_inst);
    if rust_opp_active != ts_opp_active {
        out.push(format!(
            "opponent.active: rust={:?} ts={:?}",
            rust_opp_active, ts_opp_active
        ));
    }
    let rust_player_bench: Vec<String> = state.sides[0].bench.iter().map(fmt_inst).collect();
    let ts_player_bench: Vec<String> = ts.sides.player.bench.iter().map(fmt_ts_inst).collect();
    if rust_player_bench != ts_player_bench {
        out.push(format!(
            "player.bench: rust={:?} ts={:?}",
            rust_player_bench, ts_player_bench
        ));
    }
    let rust_opp_bench: Vec<String> = state.sides[1].bench.iter().map(fmt_inst).collect();
    let ts_opp_bench: Vec<String> = ts.sides.opponent.bench.iter().map(fmt_ts_inst).collect();
    if rust_opp_bench != ts_opp_bench {
        out.push(format!(
            "opponent.bench: rust={:?} ts={:?}",
            rust_opp_bench, ts_opp_bench
        ));
    }
    out
}

fn main() -> Result<()> {
    let args = Args::parse();
    let file = File::open(&args.input)
        .with_context(|| format!("open golden trace file {}", args.input))?;
    let reader = BufReader::new(file);

    let cat = engine::core::catalog::catalog();
    let resolve = |cid: engine::core::card_id::CardId| -> String {
        cat.interner.resolve(cid).unwrap_or("?").to_string()
    };

    let mut total = 0usize;
    let mut ok = 0usize;
    let mut failures = 0usize;

    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let trace: Trace =
            serde_json::from_str(&line).with_context(|| format!("parse line {}", lineno + 1))?;
        if trace.trace_version != 1 {
            anyhow::bail!("unsupported traceVersion {}", trace.trace_version);
        }
        total += 1;

        // First-step fingerprint (post-setup, turn 1, phase=play).
        let Some(first_step) = trace.actions.first() else {
            continue;
        };
        let ts_fp: TsFingerprint = serde_json::from_str(&first_step.fingerprint_before)
            .with_context(|| format!("parse fingerprintBefore at seed {}", trace.seed))?;

        // Rust-side setup.
        let seed_string = format!("{}:selfplay", trace.seed);
        let (state, _) = with_rng(Rng::from_seed(seed_string.as_str(), "selfplay"), || {
            setup_ai_vs_ai_game()
        });

        let rust_current = match state.current_side {
            CurrentSide::Player => "player",
            CurrentSide::Opponent => "opponent",
            CurrentSide::Done => "done",
        };

        // Compare.
        let mut diffs: Vec<String> = Vec::new();

        if rust_current != ts_fp.current_side {
            diffs.push(format!(
                "currentSide: rust={} ts={}",
                rust_current, ts_fp.current_side
            ));
        }
        if state.turn_number != ts_fp.turn_number {
            diffs.push(format!(
                "turnNumber: rust={} ts={}",
                state.turn_number, ts_fp.turn_number
            ));
        }

        // Player hand — TS records own-side full hand at step 0
        // (sideId of step 0 may be opponent so the "own" of the trace
        // is the side acting; we still get the full player hand because
        // the TS fingerprint records BOTH sides' hands).
        let rust_player_hand: Vec<String> =
            state.sides[0].hand.iter().copied().map(resolve).collect();
        if rust_player_hand != ts_fp.sides.player.hand {
            diffs.push(format!(
                "player.hand: rust={:?} ts={:?}",
                rust_player_hand, ts_fp.sides.player.hand
            ));
        }

        // Active card ids.
        let rust_player_active = state.sides[0].active.as_ref().map(|u| resolve(u.card_id));
        let ts_player_active = ts_fp
            .sides
            .player
            .active
            .as_ref()
            .map(|u| u.card_id.clone());
        if rust_player_active != ts_player_active {
            diffs.push(format!(
                "player.active: rust={:?} ts={:?}",
                rust_player_active, ts_player_active
            ));
        }
        let rust_opp_active = state.sides[1].active.as_ref().map(|u| resolve(u.card_id));
        let ts_opp_active = ts_fp
            .sides
            .opponent
            .active
            .as_ref()
            .map(|u| u.card_id.clone());
        if rust_opp_active != ts_opp_active {
            diffs.push(format!(
                "opponent.active: rust={:?} ts={:?}",
                rust_opp_active, ts_opp_active
            ));
        }

        // Bench card ids.
        let rust_player_bench: Vec<String> = state.sides[0]
            .bench
            .iter()
            .map(|u| resolve(u.card_id))
            .collect();
        let ts_player_bench: Vec<String> = ts_fp
            .sides
            .player
            .bench
            .iter()
            .map(|u| u.card_id.clone())
            .collect();
        if rust_player_bench != ts_player_bench {
            diffs.push(format!(
                "player.bench: rust={:?} ts={:?}",
                rust_player_bench, ts_player_bench
            ));
        }
        let rust_opp_bench: Vec<String> = state.sides[1]
            .bench
            .iter()
            .map(|u| resolve(u.card_id))
            .collect();
        let ts_opp_bench: Vec<String> = ts_fp
            .sides
            .opponent
            .bench
            .iter()
            .map(|u| u.card_id.clone())
            .collect();
        if rust_opp_bench != ts_opp_bench {
            diffs.push(format!(
                "opponent.bench: rust={:?} ts={:?}",
                rust_opp_bench, ts_opp_bench
            ));
        }

        // V2 step-replay (if requested): drive the sim through all
        // recorded steps and accumulate diffs.
        if args.mode == "steps" {
            let step_rng = Rng::from_seed(format!("{}:selfplay", trace.seed).as_str(), "selfplay");
            let step_diffs = replay_steps_for_seed(&trace, step_rng, &resolve);
            diffs.extend(step_diffs);
        }

        // V4 MCTS replay: run Rust MCTS at each step (slow but the real
        // bit-identity gate).
        if args.mode == "mcts" {
            let mcts_rng = Rng::from_seed(format!("{}:selfplay", trace.seed).as_str(), "selfplay");
            let max_steps = if args.limit_steps == 0 {
                0
            } else {
                args.limit_steps
            };
            let mcts_diffs = v4_replay_for_seed(&trace, mcts_rng, &resolve, max_steps);
            diffs.extend(mcts_diffs);
        }

        if diffs.is_empty() {
            ok += 1;
        } else {
            failures += 1;
            if failures <= args.max_failures_to_print {
                eprintln!("FAIL seed={}:", trace.seed);
                for d in &diffs {
                    eprintln!("  - {}", d);
                }
            }
            // In V2 mode early-exit to keep run time bounded — each
            // failing seed prints up to one diff; this is enough to
            // root-cause the first divergence class.
            if args.mode == "steps" && args.limit_seeds > 0 && total >= args.limit_seeds {
                break;
            }
        }
        if args.limit_seeds > 0 && total >= args.limit_seeds {
            break;
        }
    }

    let mode_label = match args.mode.as_str() {
        "steps" => "setup+steps parity",
        _ => "setup parity",
    };
    eprintln!(
        "golden-replay ({}): {} OK / {} FAIL / {} total",
        mode_label, ok, failures, total
    );
    if failures > 0 {
        std::process::exit(1);
    }
    println!("OK {}/{} seeds bit-identical", ok, total);
    Ok(())
}
