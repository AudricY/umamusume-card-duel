//! Card-catalog codegen — reads `shared/src/cards/*.ts` (or its compiled
//! JSON form, depending on what the build pipeline emits) and writes a
//! `const` Rust table consumed by the engine crate.
//!
//! Phase 1b scaffold. The full implementation depends on what the catalog's
//! authoritative on-disk representation is; see the audit work tracked in
//! task #3.

use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "codegen")]
struct Args {
    /// Path to the catalog source (TS or JSON).
    #[arg(long, default_value = "shared/src/types.ts")]
    input: String,

    /// Output Rust file (will be `include!`'d by the engine crate).
    #[arg(long, default_value = "engine-rs/crates/engine/src/core/catalog_gen.rs")]
    output: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    eprintln!(
        "codegen: scaffold only (input={} output={}). \
         Catalog source-of-truth audit pending; see scoping doc §7.",
        args.input, args.output
    );
    Ok(())
}
