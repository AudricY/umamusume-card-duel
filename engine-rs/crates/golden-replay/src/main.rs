//! Phase 1h bit-identity gate: replay TS-recorded golden traces against the
//! Rust engine.
//!
//! Pairs with `backend/src/sim/recordGoldenTraces.ts` (Phase 0). The schema
//! is `traceVersion: 1` and currently records:
//!   { seed, configHash, actions: [{ turn, fingerprintBefore, action,
//!     rngDrawsThisTurn }], terminalFingerprint, winner, turnNumber,
//!     totalRngDraws }
//!
//! This binary is a scaffold until the engine flow modules land
//! (Phases 1d–1f). It can read the JSONL and report shape statistics today;
//! the actual replay check waits on a working Rust sim loop.

use std::fs::File;
use std::io::{BufRead, BufReader};

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;

#[derive(Parser, Debug)]
#[command(name = "golden-replay")]
struct Args {
    #[arg(long, default_value = "runs/rust-port-golden-traces/traces.jsonl")]
    input: String,

    /// When set, do not attempt the replay — just sanity-check schema.
    #[arg(long, default_value_t = false)]
    schema_only: bool,
}

#[derive(Deserialize)]
struct Trace {
    seed: u64,
    #[serde(rename = "traceVersion")]
    trace_version: u32,
    #[serde(rename = "turnNumber")]
    turn_number: u32,
    actions: Vec<serde_json::Value>,
    winner: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let file = File::open(&args.input)
        .with_context(|| format!("open golden trace file {}", args.input))?;
    let reader = BufReader::new(file);

    let mut count = 0usize;
    let mut total_turns = 0usize;
    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let trace: Trace = serde_json::from_str(&line)
            .with_context(|| format!("parse line {}", lineno + 1))?;
        if trace.trace_version != 1 {
            anyhow::bail!(
                "line {}: unsupported traceVersion {}",
                lineno + 1,
                trace.trace_version
            );
        }
        count += 1;
        total_turns += trace.actions.len();
        if !args.schema_only {
            // TODO(Phase 1h): re-run seed through Rust sim and assert
            // per-turn fingerprint + per-turn rngDrawsThisTurn + terminal
            // fingerprint + winner + turnNumber. Currently the Rust sim
            // loop isn't ready, so we accept schema-only.
            let _ = (&trace.seed, &trace.turn_number, &trace.winner);
        }
    }

    eprintln!(
        "golden-replay: parsed {} traces ({} turns total){}",
        count,
        total_turns,
        if args.schema_only {
            " [schema-only]"
        } else {
            " [replay not yet wired — see Phase 1d–1f]"
        }
    );
    Ok(())
}
