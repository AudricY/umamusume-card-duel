//! Rust port of `backend/src/sim/mctsSelfPlay.ts`.
//!
//! Phase 1g — drives N games with Rust MCTS for the modeled side and
//! heuristic AI for the opponent. Writes per-game JSONL summaries +
//! a top-level summary. Per-decision trajectory recording is the next
//! increment (matches the recorder's selfplay row schema).

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use engine::core::constants::SideId;
use engine::core::random::{with_rng, Rng};
use engine::core::state::CurrentSide;
use engine::dispatcher::{
    advance_modeled_turn_step, advance_opponent_turn_step, advance_player_ai_turn_step,
    get_forced_attack_coin_results, state_hash,
};
use engine::headless_setup::setup_ai_vs_ai_game;
use engine::mcts::config::{MctsConfig, MctsLeaf, MctsPrior};
use engine::mcts::driver::run_mcts;
use engine::policy::actions::enumerate_legal_ai_actions;
use engine::policy::observation::build_public_observation;
use engine::policy::types::{LegalAiAction, PublicObservation};
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(
    name = "sim-mcts-selfplay",
    about = "Rust port of backend/src/sim/mctsSelfPlay.ts"
)]
struct Args {
    /// MCTS simulations per decision. Orchestrator alias: --mcts-simulations.
    #[arg(long, alias = "mcts-simulations", default_value_t = 100)]
    sims: u32,
    /// CRN rollout samples. Orchestrator alias: --mcts-rollout-crn-samples.
    #[arg(long, alias = "mcts-rollout-crn-samples", default_value_t = 3)]
    k: u32,
    /// Rollout step cap. Orchestrator alias: --mcts-rollout-steps.
    #[arg(long, alias = "mcts-rollout-steps", default_value_t = 200)]
    rollout_steps: u32,
    /// Heuristic-collapse cap. Orchestrator alias: --mcts-collapse-max-steps.
    #[arg(long, alias = "mcts-collapse-max-steps", default_value_t = 64)]
    collapse_max: u32,
    /// Number of seeds. Orchestrator alias: --games.
    #[arg(long, alias = "games", default_value_t = 100)]
    seeds: u32,
    /// First seed. Orchestrator alias: --seed-start.
    #[arg(long, alias = "seed-start", default_value_t = 0)]
    seed_base: u32,
    /// Prior mode: "uniform" or "policy". Orchestrator alias: --mcts-prior.
    #[arg(long, alias = "mcts-prior", default_value = "uniform")]
    prior: String,
    /// Leaf mode. Orchestrator alias: --mcts-leaf.
    #[arg(long, alias = "mcts-leaf", default_value = "rollout")]
    leaf: String,
    /// Worker fan-out (ignored — Rust runs single-process; orchestrator
    /// parallelises by spawning multiple Rust binaries).
    #[arg(long, default_value_t = 1)]
    workers: u32,
    /// PUCT exploration constant. Orchestrator alias: --mcts-c-puct.
    #[arg(long, alias = "mcts-c-puct", default_value_t = 1.5)]
    c_puct: f64,
    /// MCTS tree-node cap. Orchestrator alias: --mcts-max-nodes.
    #[arg(long, alias = "mcts-max-nodes", default_value_t = 5_000)]
    max_nodes: u32,
    /// Root Dirichlet noise alpha. Orchestrator alias: --mcts-dirichlet-alpha.
    #[arg(long, alias = "mcts-dirichlet-alpha", default_value_t = 0.3)]
    dirichlet_alpha: f64,
    /// Root Dirichlet noise epsilon. Orchestrator alias: --mcts-dirichlet-epsilon.
    #[arg(long, alias = "mcts-dirichlet-epsilon", default_value_t = 0.25)]
    dirichlet_epsilon: f64,
    /// Temperature-sampling cutoff in moves (per side).
    #[arg(long, default_value_t = 6)]
    temperature_moves: u32,
    #[arg(long, default_value_t = 1.0)]
    temperature_value: f64,
    /// Optional manifest JSON output (mirrors TS --manifest-out).
    #[arg(long)]
    manifest_out: Option<String>,
    #[arg(long, default_value_t = 1000)]
    max_steps: u32,
    #[arg(long)]
    out: Option<String>,
    /// Record per-decision trajectory rows (observation + legal actions
    /// + visit distribution) inside each game record. Off by default
    /// since rows can be large.
    #[arg(long, default_value_t = false)]
    record_rows: bool,
    /// `/predict` server URL. When set with --leaf=value-head, MCTS
    /// queries it for leaf values + policy priors.
    #[arg(long, default_value = "")]
    model_url: String,
}

impl Args {
    /// Serialize the args set into the JSON shape the TS sim:mcts-selfplay
    /// writes to --manifest-out: camelCase keys, all flags present.
    fn manifest_args_json(&self) -> serde_json::Value {
        serde_json::json!({
            "sims": self.sims,
            "k": self.k,
            "rolloutSteps": self.rollout_steps,
            "collapseMax": self.collapse_max,
            "seeds": self.seeds,
            "seedBase": self.seed_base,
            "prior": self.prior,
            "leaf": self.leaf,
            "workers": self.workers,
            "cPuct": self.c_puct,
            "maxNodes": self.max_nodes,
            "dirichletAlpha": self.dirichlet_alpha,
            "dirichletEpsilon": self.dirichlet_epsilon,
            "temperatureMoves": self.temperature_moves,
            "temperatureValue": self.temperature_value,
            "manifestOut": self.manifest_out,
            "maxSteps": self.max_steps,
            "out": self.out,
            "recordRows": self.record_rows,
            "modelUrl": self.model_url,
        })
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GameRecord {
    seed: u32,
    terminal_reason: String,
    turn_number: u32,
    winner: Option<String>,
    total_steps: u32,
    model_decisions: u32,
    /// Per-MCTS-decision trajectory rows. Empty when no MCTS decisions
    /// fired (e.g., game ended in single-action-only steps).
    rows: Vec<SelfPlayRow>,
}

/// Mirror of TS `SelfPlayRow` in `backend/src/sim/mctsSelfPlay.ts:485`.
/// Subset of fields populated today; full parity in a follow-up.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SelfPlayRow {
    schema_version: u32,
    kind: &'static str,
    seed: u32,
    side_id: String,
    step: u32,
    turn_number: u32,
    observation: PublicObservation,
    legal_actions: Vec<LegalAiAction>,
    selected_action_index: usize,
    visit_distribution: Vec<f64>,
    root_value: f64,
    expansions: u32,
    leaf_evaluations: u32,
}

const ROW_SCHEMA_VERSION: u32 = 1;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RunSummary {
    games: u32,
    elapsed_secs: f64,
    games_per_sec: f64,
    player_wins: u32,
    opponent_wins: u32,
    draws: u32,
    terminal_reasons: TerminalReasons,
}

#[derive(Serialize, Default)]
#[serde(rename_all = "camelCase")]
struct TerminalReasons {
    game_over: u32,
    max_steps: u32,
    stalled: u32,
}

fn drive_one_game(
    seed: u32,
    max_steps: u32,
    config: &MctsConfig,
    model_side: SideId,
    record_rows: bool,
) -> GameRecord {
    let seed_str = seed.to_string();
    let rng = Rng::from_seed(format!("{}:selfplay", seed_str).as_str(), "selfplay");
    let (mut state, mut step_rng) = with_rng(rng, || setup_ai_vs_ai_game());
    let mut step = 0u32;
    let mut terminal = "max_steps";
    let mut model_decisions = 0u32;
    let mut rows: Vec<SelfPlayRow> = Vec::new();

    for s in 0..max_steps {
        step = s;
        if state.game_over {
            terminal = "game_over";
            break;
        }
        let side = match state.current_side {
            CurrentSide::Player => SideId::Player,
            CurrentSide::Opponent => SideId::Opponent,
            CurrentSide::Done => {
                terminal = "game_over";
                break;
            }
        };
        let (legal, used_rng) = with_rng(step_rng.clone(), || {
            enumerate_legal_ai_actions(&state, side)
        });
        step_rng = used_rng;

        let pre_hash = state_hash(&state);
        let next_state = if side == model_side && legal.len() > 1 {
            let mcts_seed = format!("{}:{:?}:{}:mcts", seed_str, side, s);
            let model_url = config.model_url.clone();
            let (mcts_result, used_rng) = with_rng(step_rng.clone(), || {
                run_mcts(&state, side, config, model_url.as_str(), mcts_seed.as_str())
            });
            step_rng = used_rng;
            let idx = mcts_result.selected_index.min(legal.len() - 1);
            let chosen = legal[idx].clone();
            model_decisions += 1;

            if record_rows {
                let total_visits: u32 = mcts_result.visits.iter().sum();
                let visit_distribution: Vec<f64> = if total_visits > 0 {
                    mcts_result.visits.iter().map(|&n| n as f64 / total_visits as f64).collect()
                } else {
                    let n = legal.len();
                    vec![1.0 / n.max(1) as f64; n]
                };
                rows.push(SelfPlayRow {
                    schema_version: ROW_SCHEMA_VERSION,
                    kind: "mcts-selfplay",
                    seed,
                    side_id: match side {
                        SideId::Player => "player".into(),
                        SideId::Opponent => "opponent".into(),
                    },
                    step: s,
                    turn_number: state.turn_number,
                    observation: build_public_observation(&state, side),
                    legal_actions: legal.clone(),
                    selected_action_index: idx,
                    visit_distribution,
                    root_value: mcts_result.diagnostics.root_value,
                    expansions: mcts_result.diagnostics.expansions,
                    leaf_evaluations: mcts_result.diagnostics.leaf_evaluations,
                });
            }

            let (ns, used_rng) = with_rng(step_rng.clone(), || {
                let forced = get_forced_attack_coin_results(&state);
                advance_modeled_turn_step(&state, side, &chosen, forced)
            });
            step_rng = used_rng;
            ns
        } else {
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
            ns
        };

        if state_hash(&next_state) == pre_hash {
            terminal = "stalled";
            break;
        }
        state = next_state;
    }

    let winner = state.winner.map(|s| match s {
        SideId::Player => "player".to_string(),
        SideId::Opponent => "opponent".to_string(),
    });

    GameRecord {
        seed,
        terminal_reason: terminal.to_string(),
        turn_number: state.turn_number,
        winner,
        total_steps: step + 1,
        model_decisions,
        rows,
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let leaf = match args.leaf.as_str() {
        "value-head" => MctsLeaf::ValueHead,
        _ => MctsLeaf::Rollout,
    };
    let prior = match args.prior.as_str() {
        "policy" => MctsPrior::Policy,
        _ => MctsPrior::Uniform,
    };
    let config = MctsConfig {
        simulations: args.sims,
        c_puct: args.c_puct,
        leaf,
        prior,
        rollout_crn_samples: args.k,
        rollout_steps: args.rollout_steps,
        add_root_dirichlet: false,
        dirichlet_alpha: args.dirichlet_alpha,
        dirichlet_epsilon: args.dirichlet_epsilon,
        max_nodes: args.max_nodes,
        collapse_max_steps: args.collapse_max,
        adaptive_ratio: 0.0,
        adaptive_min_sims: 100,
        model_url: args.model_url.clone(),
    };
    let _ = args.temperature_moves;
    let _ = args.temperature_value;
    let _ = args.workers;

    let model_side = SideId::Player;
    eprintln!(
        "sim-mcts-selfplay: sims={} K={} rollout_steps={} prior={} leaf={} seeds={} (base={}) model_side={:?}",
        args.sims, args.k, args.rollout_steps, args.prior, args.leaf, args.seeds, args.seed_base, model_side,
    );

    let mut writer: Option<fs::File> = match args.out.as_ref() {
        Some(path) => {
            let p = PathBuf::from(path);
            if let Some(parent) = p.parent() {
                fs::create_dir_all(parent).with_context(|| format!("mkdir {}", parent.display()))?;
            }
            Some(fs::File::create(&p).with_context(|| format!("create {}", path))?)
        }
        None => None,
    };

    let start = Instant::now();
    let mut player_wins = 0u32;
    let mut opponent_wins = 0u32;
    let mut draws = 0u32;
    let mut terminal_reasons = TerminalReasons::default();

    for i in 0..args.seeds {
        let seed = args.seed_base + i;
        let record = drive_one_game(seed, args.max_steps, &config, model_side, args.record_rows);
        match record.terminal_reason.as_str() {
            "game_over" => terminal_reasons.game_over += 1,
            "stalled" => terminal_reasons.stalled += 1,
            _ => terminal_reasons.max_steps += 1,
        }
        match record.winner.as_deref() {
            Some("player") => player_wins += 1,
            Some("opponent") => opponent_wins += 1,
            _ => draws += 1,
        }
        if let Some(w) = writer.as_mut() {
            let line = serde_json::to_string(&record)?;
            writeln!(w, "{}", line)?;
        }
    }

    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();
    let summary = RunSummary {
        games: args.seeds,
        elapsed_secs,
        games_per_sec: args.seeds as f64 / elapsed_secs.max(1e-9),
        player_wins,
        opponent_wins,
        draws,
        terminal_reasons,
    };
    let output = serde_json::json!({
        "args": args.manifest_args_json(),
        "summary": summary,
    });
    if let Some(path) = args.manifest_out.as_ref() {
        let p = PathBuf::from(path);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("mkdir {}", parent.display()))?;
        }
        let body = serde_json::to_string_pretty(&output)? + "\n";
        fs::write(&p, body).with_context(|| format!("write {}", p.display()))?;
        eprintln!("manifest-out: wrote {}", p.display());
    }
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
