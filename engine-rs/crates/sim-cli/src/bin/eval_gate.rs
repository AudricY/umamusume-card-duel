//! Rust port of `backend/src/sim/evalGate.ts`.
//!
//! Phase 1g — runs N games of MCTS-player vs heuristic-opponent
//! (both sides as model rotating across half the seeds), aggregates
//! win-rate with Wilson 95% confidence intervals, emits a summary
//! JSON matching the TS schema's top-level fields.
//!
//! Without /predict integration the "model" path uses Rust MCTS
//! (uniform prior, rollout leaf) as a stand-in. When /predict lands,
//! swap MctsLeaf::Rollout → MctsLeaf::ValueHead + thread through
//! --challenger / --baseline URLs.

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
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(
    name = "sim-eval-gate",
    about = "Rust port of backend/src/sim/evalGate.ts"
)]
struct Args {
    /// Model URL (also serves as the value-head /predict endpoint when
    /// --leaf=value-head). Orchestrators pass this here.
    #[arg(long)]
    challenger: Option<String>,
    #[arg(long)]
    baseline: Option<String>,
    /// Number of seeds. Orchestrator alias: --games.
    #[arg(long, alias = "games", default_value_t = 100)]
    seeds: u32,
    /// First seed. Orchestrator alias: --seed-start.
    #[arg(long, alias = "seed-start", default_value_t = 0)]
    seed_base: u32,
    /// MCTS simulations. Orchestrator alias: --mcts-simulations.
    #[arg(long, alias = "mcts-simulations", default_value_t = 100)]
    sims: u32,
    /// CRN rollout samples. Orchestrator alias: --mcts-rollout-crn-samples.
    #[arg(long, alias = "mcts-rollout-crn-samples", default_value_t = 3)]
    k: u32,
    /// Rollout step cap. Orchestrator alias: --mcts-rollout-steps.
    #[arg(long, alias = "mcts-rollout-steps", default_value_t = 200)]
    rollout_steps: u32,
    #[arg(long, default_value_t = 1000)]
    max_steps: u32,
    /// Manifest output (mirrors TS --manifest-out).
    #[arg(long)]
    manifest_out: Option<String>,
    /// Leaf mode: "rollout" (default) or "value-head" (queries /predict
    /// via --challenger URL). Orchestrator alias: --mcts-leaf.
    #[arg(long, alias = "mcts-leaf", default_value = "rollout")]
    leaf: String,
    /// Prior mode. Orchestrator alias: --mcts-prior.
    #[arg(long, alias = "mcts-prior", default_value = "uniform")]
    prior: String,
    /// PUCT constant. Orchestrator alias: --mcts-c-puct.
    #[arg(long, alias = "mcts-c-puct", default_value_t = 1.5)]
    c_puct: f64,
    /// MCTS tree-node cap. Orchestrator alias: --mcts-max-nodes.
    #[arg(long, alias = "mcts-max-nodes", default_value_t = 5_000)]
    max_nodes: u32,
    /// Heuristic-collapse cap. Orchestrator alias: --mcts-collapse-max-steps.
    #[arg(long, alias = "mcts-collapse-max-steps", default_value_t = 64)]
    collapse_max: u32,
    /// Selection mode (mcts | rollout | planner). Currently only mcts
    /// is wired in this binary; accepted for orchestrator-flag parity.
    #[arg(long, default_value = "mcts")]
    selection: String,
    /// Model-side rotation: "both" (default — alternate per-seed),
    /// "player", or "opponent". Accepted for orchestrator-flag parity.
    #[arg(long, default_value = "both")]
    model_side: String,
    /// Early-stop Wilson-lower threshold. If overall WR's Wilson lower
    /// bound crosses this AFTER --min-games, the gate halts and returns
    /// the partial summary. 0.0 disables early-stop.
    #[arg(long, default_value_t = 0.0)]
    min_ci_lower: f64,
    /// Minimum games before early-stop is allowed.
    #[arg(long, default_value_t = 0)]
    min_games: u32,
    /// Per-game progress JSONL output.
    #[arg(long)]
    progress_out: Option<String>,
    /// Worker fan-out (no-op — Rust runs single-process; orchestrators
    /// parallelise by spawning multiple Rust binaries).
    #[arg(long, default_value_t = 1)]
    workers: u32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SideStat {
    games: u32,
    wins: u32,
    win_rate: f64,
    wilson_lower: f64,
    wilson_upper: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GateSummary {
    challenger: Option<String>,
    baseline: Option<String>,
    config: ConfigEcho,
    elapsed_secs: f64,
    games_per_sec: f64,
    overall: SideStat,
    player_side: SideStat,
    opponent_side: SideStat,
    terminal_game_over: u32,
    terminal_stalled: u32,
    terminal_max_steps: u32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ConfigEcho {
    sims: u32,
    k: u32,
    rollout_steps: u32,
    seeds: u32,
    seed_base: u32,
}

/// Wilson score interval for a binomial proportion at 95% confidence.
/// Mirror of `wilsonInterval` in `evalGate.ts:440`.
fn wilson_interval(successes: u32, total: u32) -> (f64, f64) {
    if total == 0 {
        return (0.0, 0.0);
    }
    let z = 1.96_f64;
    let n = total as f64;
    let p = successes as f64 / n;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denom;
    let half = (z * ((p * (1.0 - p) + z2 / (4.0 * n)) / n).sqrt()) / denom;
    ((center - half).max(0.0), (center + half).min(1.0))
}

fn drive_one_game(
    seed: u32,
    model_side: SideId,
    max_steps: u32,
    config: &MctsConfig,
) -> (Option<SideId>, &'static str) {
    let seed_str = seed.to_string();
    let rng = Rng::from_seed(format!("{}:selfplay", seed_str).as_str(), "selfplay");
    let (mut state, mut step_rng) = with_rng(rng, || setup_ai_vs_ai_game());
    let mut terminal = "max_steps";
    for s in 0..max_steps {
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
        let _ = s;
    }
    (state.winner, terminal)
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
    let model_url = args.challenger.clone().unwrap_or_default();
    let config = MctsConfig {
        simulations: args.sims,
        c_puct: args.c_puct,
        leaf,
        prior,
        rollout_crn_samples: args.k,
        rollout_steps: args.rollout_steps,
        add_root_dirichlet: false,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.25,
        max_nodes: args.max_nodes,
        collapse_max_steps: args.collapse_max,
        adaptive_ratio: 0.0,
        adaptive_min_sims: 100,
        model_url: model_url.clone(),
    };
    // Accept for orchestrator-flag parity (no-op stubs for now).
    let _ = args.selection;
    let _ = args.model_side;
    let _ = args.min_ci_lower;
    let _ = args.min_games;
    let _ = args.progress_out;
    let _ = args.workers;

    eprintln!(
        "sim-eval-gate: sims={} K={} rollout_steps={} seeds={} (base={}) challenger={:?} baseline={:?}",
        args.sims, args.k, args.rollout_steps, args.seeds, args.seed_base, args.challenger, args.baseline,
    );

    let start = Instant::now();
    // Rotate model side across seeds: even seed → Player, odd → Opponent.
    let mut player_games = 0u32;
    let mut player_wins = 0u32;
    let mut opp_games = 0u32;
    let mut opp_wins = 0u32;
    let mut terminal_game_over = 0u32;
    let mut terminal_stalled = 0u32;
    let mut terminal_max_steps = 0u32;

    for i in 0..args.seeds {
        let seed = args.seed_base + i;
        let model_side = if i % 2 == 0 { SideId::Player } else { SideId::Opponent };
        let (winner, terminal) = drive_one_game(seed, model_side, args.max_steps, &config);
        match terminal {
            "game_over" => terminal_game_over += 1,
            "stalled" => terminal_stalled += 1,
            _ => terminal_max_steps += 1,
        }
        let model_won = winner == Some(model_side);
        if model_side == SideId::Player {
            player_games += 1;
            if model_won {
                player_wins += 1;
            }
        } else {
            opp_games += 1;
            if model_won {
                opp_wins += 1;
            }
        }
    }
    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();

    let total_games = player_games + opp_games;
    let total_wins = player_wins + opp_wins;
    let mk_stat = |w: u32, g: u32| {
        let (lo, hi) = wilson_interval(w, g);
        SideStat {
            games: g,
            wins: w,
            win_rate: if g > 0 { w as f64 / g as f64 } else { 0.0 },
            wilson_lower: lo,
            wilson_upper: hi,
        }
    };

    let summary = GateSummary {
        challenger: args.challenger.clone(),
        baseline: args.baseline.clone(),
        config: ConfigEcho {
            sims: args.sims,
            k: args.k,
            rollout_steps: args.rollout_steps,
            seeds: args.seeds,
            seed_base: args.seed_base,
        },
        elapsed_secs,
        games_per_sec: total_games as f64 / elapsed_secs.max(1e-9),
        overall: mk_stat(total_wins, total_games),
        player_side: mk_stat(player_wins, player_games),
        opponent_side: mk_stat(opp_wins, opp_games),
        terminal_game_over,
        terminal_stalled,
        terminal_max_steps,
    };

    let json = serde_json::to_string_pretty(&summary)?;
    println!("{}", json);

    if let Some(path) = args.manifest_out.as_ref() {
        let p = PathBuf::from(path);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).with_context(|| format!("mkdir {}", parent.display()))?;
        }
        let mut f = fs::File::create(&p).with_context(|| format!("create {}", path))?;
        writeln!(f, "{}", json)?;
    }

    Ok(())
}
