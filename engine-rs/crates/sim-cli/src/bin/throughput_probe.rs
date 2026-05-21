//! Rust port of `backend/src/sim/throughputProbe.ts`.
//!
//! Phase 1g placeholder. Wires up CLI parsing now so the runner script
//! shapes match; the per-call micro-benchmarks (cloneNsPerCall,
//! fingerprintNsPerCall, enumerateNsPerCall) will be populated once
//! Phases 1b–1e land.

use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "sim-throughput-probe",
    about = "Rust port of backend/src/sim/throughputProbe.ts (Phase 1 — scaffold)"
)]
struct Args {
    #[arg(long, default_value = "default")]
    profile: String,

    #[arg(long, default_value_t = 1)]
    legal_action_count: u32,

    #[arg(long)]
    out: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    eprintln!(
        "sim-throughput-probe: scaffold only (profile={} legal_action_count={} out={:?}). \
         Engine modules not yet ported — see Phase 1 task list.",
        args.profile, args.legal_action_count, args.out
    );
    Ok(())
}
