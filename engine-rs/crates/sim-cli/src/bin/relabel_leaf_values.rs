//! Relabel rollout-leaf rows that carry exact hidden `leafState` snapshots.
//!
//! Input is the flat JSONL emitted by `sim-mcts-selfplay
//! --record-rollout-leaf-rows` after the `leafState` snapshot slice. For
//! each row with `leafState`, this binary reruns rollout evaluation from the
//! exact hidden state and overwrites `rootValue`, `rootMeanQ`, and
//! `valueTarget` with the lower-noise scalar label.

use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use engine::core::constants::SideId;
use engine::core::state::GameState;
use engine::dispatcher::state_hash;
use engine::mcts::driver::rollout_leaf_value_for_state;
use serde::Serialize;
use serde_json::Value;

#[derive(Parser, Debug)]
#[command(
    name = "sim-relabel-leaf-values",
    about = "Relabel leafState JSONL rows with higher-K exact-state rollout values"
)]
struct Args {
    #[arg(long)]
    input: String,
    #[arg(long)]
    out: String,
    #[arg(long, default_value_t = 31)]
    k: u32,
    #[arg(long, default_value_t = 200)]
    rollout_steps: u32,
    #[arg(long, default_value = "leaf-relabel")]
    seed_salt: String,
    /// Reuse one relabel value for duplicate exact `(sideId, state_hash)`
    /// states. This is on by default because exact-state duplicate labels
    /// should not differ only because the row appeared in a different game.
    #[arg(long, default_value_t = true)]
    cache_exact_states: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Summary {
    input: String,
    out: String,
    rows_read: u64,
    rows_written: u64,
    relabeled_rows: u64,
    missing_leaf_state_rows: u64,
    bad_rows: u64,
    unique_state_labels: usize,
    cache_hits: u64,
    k: u32,
    rollout_steps: u32,
    elapsed_secs: f64,
}

fn parse_side(value: &Value) -> Option<SideId> {
    match value.get("sideId").and_then(Value::as_str) {
        Some("player") => Some(SideId::Player),
        Some("opponent") => Some(SideId::Opponent),
        _ => None,
    }
}

fn update_row_value(row: &mut Value, value: f64) {
    let n_actions = row
        .get("legalActions")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);
    if let Some(obj) = row.as_object_mut() {
        obj.insert("rootValue".to_string(), Value::from(value));
        obj.insert("valueTarget".to_string(), Value::from(value));
        if n_actions > 0 {
            obj.insert(
                "rootMeanQ".to_string(),
                Value::Array((0..n_actions).map(|_| Value::from(value)).collect()),
            );
        }
        obj.insert(
            "relabel".to_string(),
            serde_json::json!({
                "kind": "exact-leaf-state-rollout",
                "target": "rootValue",
            }),
        );
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let start = Instant::now();
    let input = fs::File::open(&args.input).with_context(|| format!("open {}", args.input))?;
    let mut writer = {
        let p = PathBuf::from(&args.out);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).with_context(|| format!("mkdir {}", parent.display()))?;
        }
        fs::File::create(&p).with_context(|| format!("create {}", args.out))?
    };

    let mut rows_read = 0u64;
    let mut rows_written = 0u64;
    let mut relabeled_rows = 0u64;
    let mut missing_leaf_state_rows = 0u64;
    let mut bad_rows = 0u64;
    let mut cache_hits = 0u64;
    let mut cache: HashMap<(String, SideId), f64> = HashMap::new();

    for (line_idx, line) in BufReader::new(input).lines().enumerate() {
        let line = line.with_context(|| format!("read {}:{}", args.input, line_idx + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        rows_read += 1;
        let mut row: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => {
                bad_rows += 1;
                continue;
            }
        };
        let Some(side) = parse_side(&row) else {
            bad_rows += 1;
            continue;
        };
        let Some(leaf_state_value) = row.get("leafState").cloned() else {
            missing_leaf_state_rows += 1;
            writeln!(writer, "{}", serde_json::to_string(&row)?)?;
            rows_written += 1;
            continue;
        };
        let state: GameState = match serde_json::from_value(leaf_state_value) {
            Ok(s) => s,
            Err(_) => {
                bad_rows += 1;
                continue;
            }
        };
        let hash = state_hash(&state);
        let key = (hash.clone(), side);
        let value = if args.cache_exact_states {
            if let Some(v) = cache.get(&key).copied() {
                cache_hits += 1;
                v
            } else {
                let seed = format!("{}:{}:{:?}", args.seed_salt, hash, side);
                let v = rollout_leaf_value_for_state(
                    &state,
                    side,
                    args.k,
                    args.rollout_steps,
                    seed.as_str(),
                );
                cache.insert(key, v);
                v
            }
        } else {
            let seed = format!("{}:{}:{:?}:{}", args.seed_salt, hash, side, line_idx + 1);
            rollout_leaf_value_for_state(&state, side, args.k, args.rollout_steps, seed.as_str())
        };
        update_row_value(&mut row, value);
        relabeled_rows += 1;
        writeln!(writer, "{}", serde_json::to_string(&row)?)?;
        rows_written += 1;
    }

    let summary = Summary {
        input: args.input,
        out: args.out,
        rows_read,
        rows_written,
        relabeled_rows,
        missing_leaf_state_rows,
        bad_rows,
        unique_state_labels: cache.len(),
        cache_hits,
        k: args.k,
        rollout_steps: args.rollout_steps,
        elapsed_secs: start.elapsed().as_secs_f64(),
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
