//! Rust port of `backend/src/sim/exportTrainingExamples.ts`.
//!
//! Phase 1g — flag-for-flag CLI surface matching the TS script. Drives
//! N games of rule-bot AI-vs-AI selfplay, collects per-decision
//! TrainingExample entries, writes JSONL + a manifest.
//!
//! Current status: CLI parsing + game loop scaffold are in place.
//! Per-decision observation + features serialization waits on a few
//! more pieces (PublicObservation → JSON shape matching TS exactly),
//! tracked in the hand-off doc.

use std::fs;
use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use engine::core::random::{with_rng, Rng};
use engine::core::state::CurrentSide;
use engine::dispatcher::{advance_opponent_turn_step, advance_player_ai_turn_step, get_forced_attack_coin_results};
use engine::headless_setup::setup_ai_vs_ai_game;
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(
    name = "sim-export-training",
    about = "Rust port of backend/src/sim/exportTrainingExamples.ts"
)]
struct Args {
    /// JSONL output path for training examples.
    #[arg(long, default_value = "training/runs/smoke/examples.jsonl")]
    out: String,

    /// First seed (seeds are `seed_start + 0..games-1`).
    #[arg(long, default_value_t = 1000)]
    seed_start: u32,

    /// Number of games to run.
    #[arg(long, default_value_t = 16)]
    games: u32,

    /// Hard cap on engine advance steps per game (stall guard).
    #[arg(long, default_value_t = 360)]
    max_steps: u32,
}

#[derive(Serialize)]
struct RunSummary {
    out: String,
    games: u32,
    examples: u32,
    terminal_reasons: TerminalReasons,
}

#[derive(Serialize, Default)]
struct TerminalReasons {
    game_over: u32,
    max_steps: u32,
    stalled: u32,
}

#[derive(Serialize)]
struct TrainingExample {
    seed: String,
    step: u32,
    turn_number: u32,
    side_id: String,
    phase: String,
    // Placeholder for downstream consumers. Populated in a follow-up when
    // the observation builder's JSON shape is finalized to match TS.
    // legal_actions: Vec<LegalAiActionJson>,
    // selected_action_index: usize,
    // observation: PublicObservation,
}

fn drive_one_game(seed: &str, max_steps: u32) -> (Vec<TrainingExample>, &'static str) {
    let rng = Rng::from_seed(format!("{}:selfplay", seed).as_str(), "selfplay");
    let (mut state, mut step_rng) = with_rng(rng, || setup_ai_vs_ai_game());
    let examples: Vec<TrainingExample> = Vec::new();

    let mut terminal = "max_steps";
    for _step in 0..max_steps {
        if state.game_over {
            terminal = "game_over";
            break;
        }
        let side = match state.current_side {
            CurrentSide::Player => Some(engine::core::constants::SideId::Player),
            CurrentSide::Opponent => Some(engine::core::constants::SideId::Opponent),
            CurrentSide::Done => {
                terminal = "game_over";
                break;
            }
        };
        let Some(side) = side else {
            terminal = "game_over";
            break;
        };

        let (next_state, used_rng) = with_rng(step_rng.clone(), || {
            let forced = get_forced_attack_coin_results(&state);
            let mut s = state.clone();
            match side {
                engine::core::constants::SideId::Player => {
                    advance_player_ai_turn_step(&mut s, forced)
                }
                engine::core::constants::SideId::Opponent => {
                    advance_opponent_turn_step(&mut s, forced)
                }
            }
            s
        });
        step_rng = used_rng;

        // Detect stall via identical state (mirrors TS recorder's
        // stalled-fingerprint check).
        if engine::dispatcher::state_hash(&next_state) == engine::dispatcher::state_hash(&state) {
            terminal = "stalled";
            break;
        }
        state = next_state;
    }
    (examples, terminal)
}

fn main() -> Result<()> {
    let args = Args::parse();
    eprintln!(
        "sim-export-training: games={} seed_start={} max_steps={} out={}",
        args.games, args.seed_start, args.max_steps, args.out
    );

    let mut all_examples: Vec<TrainingExample> = Vec::new();
    let mut terminal_reasons = TerminalReasons::default();

    for i in 0..args.games {
        let seed = (args.seed_start + i).to_string();
        let (examples, terminal) = drive_one_game(&seed, args.max_steps);
        all_examples.extend(examples);
        match terminal {
            "game_over" => terminal_reasons.game_over += 1,
            "stalled" => terminal_reasons.stalled += 1,
            _ => terminal_reasons.max_steps += 1,
        }
    }

    let out_path = PathBuf::from(&args.out);
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("mkdir {}", parent.display()))?;
    }
    let mut file = fs::File::create(&out_path).with_context(|| format!("create {}", args.out))?;
    for ex in &all_examples {
        let line = serde_json::to_string(ex)?;
        writeln!(file, "{}", line)?;
    }

    let summary = RunSummary {
        out: args.out.clone(),
        games: args.games,
        examples: all_examples.len() as u32,
        terminal_reasons,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
