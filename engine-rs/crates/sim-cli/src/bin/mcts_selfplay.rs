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
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(
    name = "sim-mcts-selfplay",
    about = "Rust port of backend/src/sim/mctsSelfPlay.ts"
)]
struct Args {
    #[arg(long, default_value_t = 100)]
    sims: u32,
    #[arg(long, default_value_t = 3)]
    k: u32,
    #[arg(long, default_value_t = 200)]
    rollout_steps: u32,
    #[arg(long, default_value_t = 64)]
    collapse_max: u32,
    #[arg(long, default_value_t = 100)]
    seeds: u32,
    #[arg(long, default_value_t = 0)]
    seed_base: u32,
    /// Prior mode: "uniform" or "policy". Only uniform supported until
    /// /predict integration is wired.
    #[arg(long, default_value = "uniform")]
    prior: String,
    /// Leaf mode: "rollout" or "value-head". Only rollout supported
    /// until /predict integration is wired.
    #[arg(long, default_value = "rollout")]
    leaf: String,
    #[arg(long, default_value_t = 1000)]
    max_steps: u32,
    #[arg(long)]
    out: Option<String>,
}

#[derive(Serialize)]
struct GameRecord {
    seed: u32,
    terminal_reason: String,
    turn_number: u32,
    winner: Option<String>,
    total_steps: u32,
}

#[derive(Serialize)]
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
struct TerminalReasons {
    game_over: u32,
    max_steps: u32,
    stalled: u32,
}

fn drive_one_game(seed: u32, max_steps: u32, config: &MctsConfig, model_side: SideId) -> GameRecord {
    let seed_str = seed.to_string();
    let rng = Rng::from_seed(format!("{}:selfplay", seed_str).as_str(), "selfplay");
    let (mut state, mut step_rng) = with_rng(rng, || setup_ai_vs_ai_game());
    let mut step = 0u32;
    let mut terminal = "max_steps";

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
            let (mcts_result, used_rng) = with_rng(step_rng.clone(), || {
                run_mcts(&state, side, config, "", mcts_seed.as_str())
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
        c_puct: 1.5,
        leaf,
        prior,
        rollout_crn_samples: args.k,
        rollout_steps: args.rollout_steps,
        add_root_dirichlet: false,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.25,
        max_nodes: 5_000,
        collapse_max_steps: args.collapse_max,
        adaptive_ratio: 0.0,
        adaptive_min_sims: 100,
        model_url: String::new(),
    };

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
        let record = drive_one_game(seed, args.max_steps, &config, model_side);
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
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
