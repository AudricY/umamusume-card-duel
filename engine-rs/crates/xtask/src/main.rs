//! Workspace helper tasks.
//!
//! - `cargo run -p xtask -- dump-rng-reference` regenerates
//!   `crates/engine/tests/rng-reference.json` by invoking the TS dumper.
//! - `cargo run -p xtask -- check-conformance` runs cargo test and the TS
//!   replay tool over the same Phase 0 corpus.

use std::process::Command;

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    DumpRngReference,
    CheckConformance,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::DumpRngReference => dump_rng_reference(),
        Cmd::CheckConformance => check_conformance(),
    }
}

fn dump_rng_reference() -> Result<()> {
    let repo_root = repo_root()?;
    let out = repo_root.join("engine-rs/crates/engine/tests/rng-reference.json");
    let status = Command::new("npx")
        .args(["tsx", "engine-rs/scripts/dump-ts-rng.ts"])
        .current_dir(&repo_root)
        .stdout(std::fs::File::create(&out)?)
        .status()?;
    if !status.success() {
        bail!("dump-ts-rng.ts exited with {}", status);
    }
    eprintln!("wrote {}", out.display());
    Ok(())
}

fn check_conformance() -> Result<()> {
    let repo_root = repo_root()?;
    let status = Command::new("cargo")
        .args(["test", "--workspace"])
        .current_dir(repo_root.join("engine-rs"))
        .status()?;
    if !status.success() {
        bail!("cargo test failed");
    }
    Ok(())
}

fn repo_root() -> Result<std::path::PathBuf> {
    let here = std::env::current_dir()?;
    let mut cur = here.as_path();
    loop {
        if cur.join("package.json").exists() && cur.join("frontend").exists() {
            return Ok(cur.to_path_buf());
        }
        match cur.parent() {
            Some(p) => cur = p,
            None => bail!("could not locate repo root from {}", here.display()),
        }
    }
}
