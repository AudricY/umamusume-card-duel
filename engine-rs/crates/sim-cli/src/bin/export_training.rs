//! Rust port of `backend/src/sim/exportTrainingExamples.ts` (Phase 1 scaffold).

use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "sim-export-training")]
struct Args {
    #[arg(long)]
    input: Option<String>,
    #[arg(long)]
    out: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    eprintln!(
        "sim-export-training: scaffold only (input={:?} out={:?})",
        args.input, args.out
    );
    Ok(())
}
