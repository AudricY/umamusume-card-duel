//! Integration test: drive a full headless AI-vs-AI game via the Rust
//! dispatcher for several seeds and assert each one terminates cleanly.
//!
//! Catches regressions in the dispatcher / heuristic-opponent / engine
//! flow that would let a game stall or loop forever. Each game uses the
//! default deck registry and the same setup the golden-trace recorder
//! uses, so any regression that breaks the standard self-play sequence
//! shows up here.

use engine::core::constants::SideId;
use engine::core::random::{with_rng, Rng};
use engine::core::state::CurrentSide;
use engine::dispatcher::{
    advance_opponent_turn_step, advance_player_ai_turn_step, get_forced_attack_coin_results,
    state_hash,
};
use engine::headless_setup::setup_ai_vs_ai_game;

#[derive(Debug)]
struct GameOutcome {
    seed: String,
    terminal_reason: &'static str,
    steps_taken: u32,
    final_turn_number: u32,
}

fn drive_one_game(seed: &str, max_steps: u32) -> GameOutcome {
    let rng = Rng::from_seed(format!("{}:selfplay", seed).as_str(), "selfplay");
    let (mut state, mut step_rng) = with_rng(rng, || setup_ai_vs_ai_game());

    let mut terminal_reason = "max_steps";
    let mut steps_taken = 0u32;
    for step in 0..max_steps {
        steps_taken = step;
        if state.game_over {
            terminal_reason = "game_over";
            break;
        }
        let side = match state.current_side {
            CurrentSide::Player => SideId::Player,
            CurrentSide::Opponent => SideId::Opponent,
            CurrentSide::Done => {
                terminal_reason = "game_over";
                break;
            }
        };

        let pre_hash = state_hash(&state);
        let (next_state, used_rng) = with_rng(step_rng.clone(), || {
            let forced = get_forced_attack_coin_results(&state);
            let mut s = state.clone();
            match side {
                SideId::Player => advance_player_ai_turn_step(&mut s, forced),
                SideId::Opponent => advance_opponent_turn_step(&mut s, forced),
            }
            s
        });
        step_rng = used_rng;
        if state_hash(&next_state) == pre_hash {
            terminal_reason = "stalled";
            break;
        }
        state = next_state;
    }
    GameOutcome {
        seed: seed.to_string(),
        terminal_reason,
        steps_taken,
        final_turn_number: state.turn_number,
    }
}

#[test]
fn drives_games_for_a_handful_of_seeds_without_stalling() {
    // Five varied seeds covering different opening-coin outcomes and
    // different deck shuffles. If any of these stall, the dispatcher or
    // heuristic-opponent has a regression.
    let seeds = ["0", "1", "7", "42", "12345"];
    let mut outcomes = Vec::new();
    for seed in &seeds {
        let outcome = drive_one_game(seed, 500);
        eprintln!(
            "seed={:>6} terminal={} steps={} turn={}",
            outcome.seed, outcome.terminal_reason, outcome.steps_taken, outcome.final_turn_number,
        );
        outcomes.push(outcome);
    }
    // All seeds should terminate via game_over (the production seeds
    // recorded all do — verified via runs/rust-port-golden-traces).
    for o in &outcomes {
        assert_ne!(
            o.terminal_reason, "stalled",
            "seed {} stalled at step {} turn {}",
            o.seed, o.steps_taken, o.final_turn_number
        );
        assert_ne!(
            o.terminal_reason, "max_steps",
            "seed {} hit max_steps at turn {}; expected game_over",
            o.seed, o.final_turn_number
        );
        assert!(
            o.final_turn_number > 0,
            "seed {} ended at turn 0",
            o.seed
        );
    }
}

#[test]
fn determinism_same_seed_same_outcome() {
    let a = drive_one_game("7", 500);
    let b = drive_one_game("7", 500);
    assert_eq!(a.terminal_reason, b.terminal_reason);
    assert_eq!(a.steps_taken, b.steps_taken);
    assert_eq!(a.final_turn_number, b.final_turn_number);
}
