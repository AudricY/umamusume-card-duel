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
use engine::dispatcher::{advance_opponent_turn_step, advance_player_ai_turn_step, get_forced_attack_coin_results, state_hash};
use engine::headless_setup::setup_ai_vs_ai_game;
use engine::policy::actions::{choose_highest_scored_action, enumerate_legal_ai_actions};
use engine::policy::observation::build_public_observation;
use engine::policy::types::{LegalAiAction, PublicObservation};
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

/// Mirror of TS `TrainingExample` (`backend/src/sim/headlessAiVsAi.ts:151`).
/// Per-AI-decision row produced by the heuristic-candidate-v1 policy
/// (`chooseHighestScoredAction` over enumerated legal actions).
#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct TrainingExample {
    schema_version: u32,
    episode_id: String,
    step: u32,
    seed: String,
    side_id: String,
    phase: String,
    observation: PublicObservation,
    legal_actions: Vec<LegalAiAction>,
    selected_action_id: String,
    selected_action_index: usize,
    policy: &'static str,
    result: ExampleResult,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct ExampleResult {
    winner: Option<String>,
    points: PointsByside,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct PointsByside {
    player: u8,
    opponent: u8,
}

fn drive_one_game(seed: &str, max_steps: u32) -> (Vec<TrainingExample>, &'static str) {
    let rng = Rng::from_seed(format!("{}:selfplay", seed).as_str(), "selfplay");
    let (mut state, mut step_rng) = with_rng(rng, || setup_ai_vs_ai_game());
    let mut examples: Vec<TrainingExample> = Vec::new();
    let episode_id = format!("ep-{}", seed);

    let mut terminal = "max_steps";
    for step in 0..max_steps {
        if state.game_over {
            terminal = "game_over";
            break;
        }
        let side = match state.current_side {
            CurrentSide::Player => engine::core::constants::SideId::Player,
            CurrentSide::Opponent => engine::core::constants::SideId::Opponent,
            CurrentSide::Done => {
                terminal = "game_over";
                break;
            }
        };

        // Heuristic-candidate-v1 policy: enumerate legal actions, pick
        // highest-scored. Record the example BEFORE advancing.
        let (legal, used_rng) = with_rng(step_rng.clone(), || {
            enumerate_legal_ai_actions(&state, side)
        });
        step_rng = used_rng;
        if legal.is_empty() {
            // Some phases (e.g. setup) may have zero AI actions; just
            // advance without recording.
        } else {
            let selected = choose_highest_scored_action(&legal);
            let selected_idx = legal
                .iter()
                .position(|a| a.id == selected.id)
                .unwrap_or(0);
            let phase_str = match selected.phase {
                engine::policy::types::AiPhase::Setup => "setup",
                engine::policy::types::AiPhase::PendingChoice => "pendingChoice",
                engine::policy::types::AiPhase::Bench => "bench",
                engine::policy::types::AiPhase::TrainerBefore => "trainerBefore",
                engine::policy::types::AiPhase::Evolve => "evolve",
                engine::policy::types::AiPhase::Attach => "attach",
                engine::policy::types::AiPhase::TrainerAfter => "trainerAfter",
                engine::policy::types::AiPhase::Ability => "ability",
                engine::policy::types::AiPhase::Combat => "combat",
                engine::policy::types::AiPhase::StadiumOrEnd => "stadiumOrEnd",
            };
            examples.push(TrainingExample {
                schema_version: 1,
                episode_id: episode_id.clone(),
                step,
                seed: seed.to_string(),
                side_id: match side {
                    engine::core::constants::SideId::Player => "player".to_string(),
                    engine::core::constants::SideId::Opponent => "opponent".to_string(),
                },
                phase: phase_str.to_string(),
                observation: build_public_observation(&state, side),
                legal_actions: legal,
                selected_action_id: selected.id.clone(),
                selected_action_index: selected_idx,
                policy: "heuristic-candidate-v1",
                result: ExampleResult {
                    winner: None,
                    points: PointsByside {
                        player: state.sides[0].points,
                        opponent: state.sides[1].points,
                    },
                },
            });
        }

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

        if state_hash(&next_state) == state_hash(&state) {
            terminal = "stalled";
            break;
        }
        state = next_state;
    }

    // Backfill winner + final points across all examples (matches TS pattern).
    let final_winner = state.winner.map(|s| match s {
        engine::core::constants::SideId::Player => "player".to_string(),
        engine::core::constants::SideId::Opponent => "opponent".to_string(),
    });
    let final_points = PointsByside {
        player: state.sides[0].points,
        opponent: state.sides[1].points,
    };
    for ex in &mut examples {
        ex.result.winner = final_winner.clone();
        ex.result.points = final_points.clone();
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
