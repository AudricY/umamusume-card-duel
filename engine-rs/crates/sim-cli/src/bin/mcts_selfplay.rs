//! Rust port of `backend/src/sim/mctsSelfPlay.ts` (Phase 1 scaffold).

use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "sim-mcts-selfplay")]
struct Args {
    #[arg(long, default_value_t = 100)]
    sims: u32,
    #[arg(long, default_value_t = 3)]
    k: u32,
    #[arg(long, default_value_t = 200)]
    rollout_steps: u32,
    #[arg(long, default_value_t = 64)]
    collapse_max: u32,
    #[arg(long, default_value_t = 1)]
    seeds: u32,
    #[arg(long, default_value_t = 0)]
    seed_base: u32,
    #[arg(long)]
    out: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    eprintln!(
        "sim-mcts-selfplay: scaffold only (sims={} k={} rollout_steps={} collapse_max={} seeds={} seed_base={} out={:?})",
        args.sims, args.k, args.rollout_steps, args.collapse_max, args.seeds, args.seed_base, args.out
    );
    Ok(())
}
