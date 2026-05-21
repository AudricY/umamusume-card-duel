//! Rust port of `backend/src/sim/evalGate.ts` (Phase 1 scaffold).

use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "sim-eval-gate")]
struct Args {
    #[arg(long)]
    challenger: Option<String>,
    #[arg(long)]
    baseline: Option<String>,
    #[arg(long, default_value_t = 100)]
    seeds: u32,
    #[arg(long)]
    manifest_out: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    eprintln!(
        "sim-eval-gate: scaffold only (challenger={:?} baseline={:?} seeds={} manifest_out={:?})",
        args.challenger, args.baseline, args.seeds, args.manifest_out
    );
    Ok(())
}
