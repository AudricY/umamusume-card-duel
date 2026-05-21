//! R16-P3 throughput-spike Option A parity smoke.
//!
//! Validates that the in-process Rust ORT path (`engine::inference`)
//! agrees numerically with the legacy HTTP `serve_onnx.py` server on N
//! random `(state, legal_actions)` snapshots. This is the EXTERNAL
//! stopping gate for the inference slice: a green run here unblocks the
//! orchestrator-wiring follow-up.
//!
//! Protocol
//! --------
//! 1. Caller passes `--onnx-path` (v3.0) and `--serve-onnx-url` (a
//!    running serve_onnx instance). The smoke does NOT spawn serve_onnx
//!    itself — that's the caller's responsibility (typically via
//!    `python -m uma_ai.scripts.serve_onnx --model ... --ort-threads 1`
//!    pinned to single-threaded so the FP determinism contract holds
//!    on the HTTP side too).
//! 2. For each of N random selfplay seeds, drive the heuristic game
//!    forward until it hits a model-decision point with `>1` legal
//!    actions. Build a v3.0 `PublicObservation` + `legalActions` JSON
//!    body and POST to serve_onnx; in parallel run the same observation
//!    through the in-process ORT path.
//! 3. Assert `(actionProbs[0], value[0])` between the two paths agree
//!    elementwise to `<= --tol abs` (default 1e-5). Any failure aborts
//!    immediately with a per-slot diagnostic.
//! 4. Print a per-seed summary (max abs diff over probs, value diff)
//!    and a final aggregate.
//!
//! See `docs/ai-research/scoping/throughput-optimization-spike.md` for
//! the acceptance criteria this smoke implements.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use engine::core::constants::SideId;
use engine::core::random::{with_rng, Rng};
use engine::core::state::CurrentSide;
use engine::dispatcher::{
    advance_opponent_turn_step, advance_player_ai_turn_step, get_forced_attack_coin_results,
    state_hash,
};
use engine::headless_setup::setup_ai_vs_ai_game;
use engine::inference::InferenceSession;
use engine::policy::actions::enumerate_legal_ai_actions;
use engine::policy::observation::build_public_observation;
use engine::policy::types::{LegalAiAction, PublicObservation};

#[derive(Parser, Debug)]
#[command(
    name = "sim-inference-parity",
    about = "R16-P3 spike Option A parity smoke: in-process ORT vs serve_onnx HTTP."
)]
struct Args {
    /// Path to the v3.0 ONNX policy file (with `.meta.json` sidecar).
    #[arg(long)]
    onnx_path: PathBuf,
    /// URL of an already-running serve_onnx instance (e.g.
    /// `http://127.0.0.1:8765`). The smoke does NOT launch serve_onnx
    /// itself — the caller must pin `--ort-threads 1` for parity.
    #[arg(long)]
    serve_onnx_url: String,
    /// Number of random game-state snapshots to sample.
    #[arg(long, default_value_t = 50)]
    n_states: u32,
    /// Per-element absolute tolerance for prob + value parity.
    #[arg(long, default_value_t = 1.0e-5)]
    tol: f64,
    /// First seed in the sample range.
    #[arg(long, default_value_t = 7_000_000)]
    seed_base: u32,
    /// Max heuristic-AI steps allowed before bailing on a seed that
    /// doesn't produce a branching decision (rare; ~all seeds branch
    /// within ~5 steps).
    #[arg(long, default_value_t = 300)]
    max_steps: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    eprintln!(
        "sim-inference-parity: n_states={} tol={} onnx={} url={}",
        args.n_states,
        args.tol,
        args.onnx_path.display(),
        args.serve_onnx_url,
    );

    let session = InferenceSession::load(&args.onnx_path)
        .map_err(|e| anyhow::anyhow!("load ONNX session: {}", e))?;

    let mut http = HttpClient::new(&args.serve_onnx_url);

    let mut max_prob_diff_all = 0.0f64;
    let mut max_value_diff_all = 0.0f64;
    let mut total_rust_us: u128 = 0;
    let mut total_http_us: u128 = 0;
    let mut samples_taken = 0u32;
    let mut seed = args.seed_base;

    let start = Instant::now();

    while samples_taken < args.n_states {
        let snapshot = sample_decision_point(seed, args.max_steps);
        seed += 1;
        let Some((obs, legal)) = snapshot else { continue; };

        // In-process ORT path.
        let t_rust = Instant::now();
        let rust_pred = session
            .predict_v3(&obs, &legal)
            .map_err(|e| anyhow::anyhow!("rust predict: {}", e))?;
        let rust_us = t_rust.elapsed().as_micros();
        total_rust_us += rust_us;

        // serve_onnx HTTP path.
        let t_http = Instant::now();
        let http_pred = http
            .predict(&obs, &legal)
            .with_context(|| format!("HTTP predict at sample {}", samples_taken))?;
        let http_us = t_http.elapsed().as_micros();
        total_http_us += http_us;

        // Per-slot diff.
        if rust_pred.probs.len() != http_pred.probs.len() {
            anyhow::bail!(
                "sample {}: probs length mismatch rust={} http={}",
                samples_taken,
                rust_pred.probs.len(),
                http_pred.probs.len()
            );
        }
        let mut max_prob_diff = 0.0f64;
        for (i, (&a, &b)) in rust_pred.probs.iter().zip(http_pred.probs.iter()).enumerate() {
            let d = (a as f64 - b as f64).abs();
            if d > max_prob_diff {
                max_prob_diff = d;
            }
            if d > args.tol {
                anyhow::bail!(
                    "sample {} (seed {}): action[{}] prob mismatch rust={:.9e} http={:.9e} diff={:.3e} > tol {:.3e}",
                    samples_taken, seed - 1, i, a, b, d, args.tol
                );
            }
        }
        let value_diff = (rust_pred.value as f64 - http_pred.value as f64).abs();
        if value_diff > args.tol {
            anyhow::bail!(
                "sample {} (seed {}): value mismatch rust={:.9e} http={:.9e} diff={:.3e} > tol {:.3e}",
                samples_taken, seed - 1, rust_pred.value, http_pred.value, value_diff, args.tol
            );
        }
        if max_prob_diff > max_prob_diff_all {
            max_prob_diff_all = max_prob_diff;
        }
        if value_diff > max_value_diff_all {
            max_value_diff_all = value_diff;
        }

        if samples_taken % 10 == 0 {
            eprintln!(
                "  sample {}/{}: actions={} max_prob_diff={:.3e} value_diff={:.3e} rust={}us http={}us",
                samples_taken + 1, args.n_states, legal.len(), max_prob_diff, value_diff, rust_us, http_us,
            );
        }
        samples_taken += 1;
    }

    let elapsed = start.elapsed();
    let rust_mean_us = if samples_taken > 0 { total_rust_us / samples_taken as u128 } else { 0 };
    let http_mean_us = if samples_taken > 0 { total_http_us / samples_taken as u128 } else { 0 };

    println!(
        "{}",
        serde_json::json!({
            "status": "PASS",
            "n_states": samples_taken,
            "tol": args.tol,
            "max_prob_diff": max_prob_diff_all,
            "max_value_diff": max_value_diff_all,
            "rust_mean_us": rust_mean_us,
            "http_mean_us": http_mean_us,
            "speedup": (http_mean_us as f64) / (rust_mean_us.max(1) as f64),
            "elapsed_secs": elapsed.as_secs_f64(),
        })
    );

    Ok(())
}

/// Drive heuristic-vs-heuristic selfplay starting at `seed` until the
/// first model decision (defined as `>1` legal AI actions for the
/// current side). Returns `None` if the game terminates without one.
fn sample_decision_point(
    seed: u32,
    max_steps: u32,
) -> Option<(PublicObservation, Vec<LegalAiAction>)> {
    let rng = Rng::from_seed(format!("{}:parity", seed).as_str(), "parity");
    let (mut state, mut step_rng) = with_rng(rng, || setup_ai_vs_ai_game());

    for _ in 0..max_steps {
        if state.game_over {
            return None;
        }
        let side = match state.current_side {
            CurrentSide::Player => SideId::Player,
            CurrentSide::Opponent => SideId::Opponent,
            CurrentSide::Done => return None,
        };

        let (legal, used_rng) = with_rng(step_rng.clone(), || {
            enumerate_legal_ai_actions(&state, side)
        });
        step_rng = used_rng;

        if legal.len() > 1 {
            // Found a branching decision — sample it for parity.
            let obs = build_public_observation(&state, side);
            return Some((obs, legal));
        }

        // Single-action or zero-action — advance the heuristic AI and
        // keep looking.
        let pre_hash = state_hash(&state);
        let (ns, used_rng) = with_rng(step_rng.clone(), || {
            let forced = get_forced_attack_coin_results(&state);
            let mut s = state.clone();
            if side == SideId::Player {
                advance_player_ai_turn_step(&mut s, forced);
            } else {
                advance_opponent_turn_step(&mut s, forced);
            }
            s
        });
        step_rng = used_rng;
        if state_hash(&ns) == pre_hash {
            // Stalled — no progress possible.
            return None;
        }
        state = ns;
    }
    None
}

/// Thin POST client for `serve_onnx`. Reuses a single TCP connection
/// across calls (HTTP/1.1 keep-alive via `ureq`).
struct HttpClient {
    url: String,
    agent: ureq::Agent,
}

impl HttpClient {
    fn new(base: &str) -> Self {
        let trimmed = base.trim_end_matches('/').to_string();
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(std::time::Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(60))
            .build();
        Self {
            url: format!("{}/predict", trimmed),
            agent,
        }
    }

    fn predict(
        &mut self,
        observation: &PublicObservation,
        legal_actions: &[LegalAiAction],
    ) -> Result<HttpPrediction> {
        let body = serde_json::json!({
            "observation": observation,
            "legalActions": legal_actions,
            "sampling": "greedy",
        });
        let body_str = serde_json::to_string(&body)?;
        let resp = self
            .agent
            .post(&self.url)
            .set("Content-Type", "application/json")
            .send_string(&body_str)
            .with_context(|| format!("POST {}", self.url))?;
        let text = resp.into_string().context("serve_onnx response body")?;
        let parsed: ParsedResponse = serde_json::from_str(&text)
            .with_context(|| format!("serve_onnx response parse: {}", text))?;
        let probs_row = parsed
            .action_probs
            .as_ref()
            .and_then(|m| m.first())
            .cloned()
            .unwrap_or_default();
        let value = parsed
            .value
            .as_ref()
            .and_then(|v| v.first())
            .copied()
            .unwrap_or(0.0);
        Ok(HttpPrediction {
            probs: probs_row.into_iter().map(|p| p as f32).collect(),
            value: value as f32,
        })
    }
}

#[derive(Debug, serde::Deserialize)]
struct ParsedResponse {
    #[serde(default, rename = "actionProbs")]
    action_probs: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    value: Option<Vec<f64>>,
}

struct HttpPrediction {
    probs: Vec<f32>,
    value: f32,
}

