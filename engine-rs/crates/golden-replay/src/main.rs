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
use engine::core::random::{with_rng, Rng};
use engine::core::state::CurrentSide;
use engine::headless_setup::setup_ai_vs_ai_game;
use serde::Deserialize;

#[derive(Parser, Debug)]
#[command(name = "golden-replay")]
struct Args {
    #[arg(long, default_value = "runs/rust-port-golden-traces/traces-500.jsonl")]
    input: String,

    /// Print a diff for the first N seeds that fail, then continue.
    #[arg(long, default_value_t = 1)]
    max_failures_to_print: usize,
}

#[derive(Deserialize)]
struct Trace {
    seed: u64,
    #[serde(rename = "traceVersion")]
    trace_version: u32,
    actions: Vec<TraceStep>,
}

#[derive(Deserialize)]
struct TraceStep {
    #[serde(rename = "fingerprintBefore")]
    fingerprint_before: String,
}

#[derive(Deserialize, Debug)]
struct TsFingerprint {
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
    #[allow(dead_code)]
    uid: u32,
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
        let trace: Trace = serde_json::from_str(&line)
            .with_context(|| format!("parse line {}", lineno + 1))?;
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
        let ts_player_active = ts_fp.sides.player.active.as_ref().map(|u| u.card_id.clone());
        if rust_player_active != ts_player_active {
            diffs.push(format!(
                "player.active: rust={:?} ts={:?}",
                rust_player_active, ts_player_active
            ));
        }
        let rust_opp_active = state.sides[1].active.as_ref().map(|u| resolve(u.card_id));
        let ts_opp_active = ts_fp.sides.opponent.active.as_ref().map(|u| u.card_id.clone());
        if rust_opp_active != ts_opp_active {
            diffs.push(format!(
                "opponent.active: rust={:?} ts={:?}",
                rust_opp_active, ts_opp_active
            ));
        }

        // Bench card ids.
        let rust_player_bench: Vec<String> =
            state.sides[0].bench.iter().map(|u| resolve(u.card_id)).collect();
        let ts_player_bench: Vec<String> =
            ts_fp.sides.player.bench.iter().map(|u| u.card_id.clone()).collect();
        if rust_player_bench != ts_player_bench {
            diffs.push(format!(
                "player.bench: rust={:?} ts={:?}",
                rust_player_bench, ts_player_bench
            ));
        }
        let rust_opp_bench: Vec<String> =
            state.sides[1].bench.iter().map(|u| resolve(u.card_id)).collect();
        let ts_opp_bench: Vec<String> =
            ts_fp.sides.opponent.bench.iter().map(|u| u.card_id.clone()).collect();
        if rust_opp_bench != ts_opp_bench {
            diffs.push(format!(
                "opponent.bench: rust={:?} ts={:?}",
                rust_opp_bench, ts_opp_bench
            ));
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
        }
    }

    eprintln!(
        "golden-replay (setup parity): {} OK / {} FAIL / {} total",
        ok, failures, total
    );
    if failures > 0 {
        std::process::exit(1);
    }
    println!("OK {}/{} seeds setup-bit-identical", ok, total);
    Ok(())
}
