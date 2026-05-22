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
    winner: Option<SideId>,
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
        winner: state.winner,
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
            "seed={:>6} terminal={} steps={} turn={} winner={:?}",
            outcome.seed,
            outcome.terminal_reason,
            outcome.steps_taken,
            outcome.final_turn_number,
            outcome.winner,
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
    assert_eq!(a.winner, b.winner);
}

/// MCTS-driven throughput: drives N games where the modeled side uses
/// MCTS at each decision (sims=100, K=3, prior=uniform, leaf=rollout)
/// — matching the recorder's config — and the opponent uses heuristic
/// AI. Reports games/sec.
#[test]
#[ignore]
fn throughput_mcts_games() {
    use engine::core::constants::SideId;
    use engine::dispatcher::{
        advance_modeled_turn_step, advance_opponent_turn_step, get_forced_attack_coin_results,
    };
    use engine::mcts::config::{MctsConfig, MctsLeaf, MctsPrior};
    use engine::mcts::driver::run_mcts;
    use engine::policy::actions::enumerate_legal_ai_actions;

    let config = MctsConfig {
        simulations: 100,
        c_puct: 1.5,
        leaf: MctsLeaf::Rollout,
        prior: MctsPrior::Uniform,
        rollout_crn_samples: 3,
        rollout_steps: 200,
        value_head_rollout_blend: 0.0,
        add_root_dirichlet: false,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.25,
        max_nodes: 5_000,
        collapse_max_steps: 64,
        adaptive_ratio: 0.0,
        adaptive_min_sims: 100,
        model_url: String::new(),
        onnx_path: None,
    };

    let model_side = SideId::Player;
    let n = 3u32;
    let start = std::time::Instant::now();
    let mut total_advances = 0u32;
    for seed_num in 0..n {
        let seed = seed_num.to_string();
        let rng = Rng::from_seed(format!("{}:selfplay", seed).as_str(), "selfplay");
        let (mut state, mut step_rng) = with_rng(rng, || setup_ai_vs_ai_game());
        let mut step = 0u32;
        for _ in 0..1000 {
            if state.game_over { break; }
            let side = match state.current_side {
                CurrentSide::Player => SideId::Player,
                CurrentSide::Opponent => SideId::Opponent,
                CurrentSide::Done => break,
            };
            // enumerate_legal_ai_actions may invoke RNG indirectly via the
            // ai/* scoring path — wrap in the active rng.
            let (legal, used_rng) = with_rng(step_rng.clone(), || {
                enumerate_legal_ai_actions(&state, side)
            });
            step_rng = used_rng;
            if side == model_side && legal.len() > 1 {
                let mcts_seed = format!("{}:{:?}:{}:mcts", seed, side, step);
                let (mcts_result, used_rng) = with_rng(step_rng.clone(), || {
                    run_mcts(&state, side, &config, "", mcts_seed.as_str())
                });
                step_rng = used_rng;
                let action_idx = mcts_result.selected_index.min(legal.len() - 1);
                let chosen = legal[action_idx].clone();
                let (next_state, used_rng) = with_rng(step_rng.clone(), || {
                    let forced = get_forced_attack_coin_results(&state);
                    advance_modeled_turn_step(&state, side, &chosen, forced)
                });
                step_rng = used_rng;
                state = next_state;
            } else {
                let (next_state, used_rng) = with_rng(step_rng.clone(), || {
                    let forced = get_forced_attack_coin_results(&state);
                    let mut s = state.clone();
                    if side == SideId::Player {
                        engine::dispatcher::advance_player_ai_turn_step(&mut s, forced);
                    } else {
                        advance_opponent_turn_step(&mut s, forced);
                    }
                    s
                });
                step_rng = used_rng;
                state = next_state;
            }
            step += 1;
            total_advances += 1;
        }
    }
    let elapsed = start.elapsed();
    let games_per_sec = n as f64 / elapsed.as_secs_f64();
    eprintln!(
        "MCTS throughput: {} games in {:?} = {:.2} games/s; total_advances={} (avg {:.1}/game)",
        n, elapsed, games_per_sec, total_advances, total_advances as f64 / n as f64,
    );
}

/// Throughput benchmark — drive 100 games and report games/sec.
/// Marked `#[ignore]` so it doesn't run by default. Enable with
/// `cargo test --release -- --ignored throughput`.
#[test]
#[ignore]
fn throughput_100_headless_games() {
    let n = 100u32;
    let start = std::time::Instant::now();
    let mut total_steps = 0u32;
    let mut player_wins = 0u32;
    let mut opponent_wins = 0u32;
    for seed_num in 0..n {
        let seed = seed_num.to_string();
        let outcome = drive_one_game(&seed, 1000);
        total_steps += outcome.steps_taken + 1;
        match outcome.winner {
            Some(SideId::Player) => player_wins += 1,
            Some(SideId::Opponent) => opponent_wins += 1,
            None => {}
        }
    }
    let elapsed = start.elapsed();
    let games_per_sec = n as f64 / elapsed.as_secs_f64();
    let steps_per_sec = total_steps as f64 / elapsed.as_secs_f64();
    let avg_steps = total_steps as f64 / n as f64;
    eprintln!(
        "100-game throughput: {:.1} games/s, {:.1} steps/s, avg {:.1} steps/game; player={}, opponent={}, elapsed={:?}",
        games_per_sec, steps_per_sec, avg_steps, player_wins, opponent_wins, elapsed,
    );
    assert!(games_per_sec > 50.0, "expected >50 games/s, got {:.1}", games_per_sec);
}

/// Broader sweep — first 32 seeds — to flush out any stall a single
/// seed might miss. Marked `#[ignore]` by default to keep the standard
/// `cargo test` fast; enable with `cargo test -- --ignored`.
#[test]
#[ignore]
fn no_stalls_across_first_32_seeds() {
    let mut player_wins = 0u32;
    let mut opponent_wins = 0u32;
    let mut draws = 0u32;
    for seed_num in 0..32 {
        let seed = seed_num.to_string();
        let outcome = drive_one_game(&seed, 1000);
        assert_ne!(
            outcome.terminal_reason, "stalled",
            "seed {} stalled at step {} turn {}",
            outcome.seed, outcome.steps_taken, outcome.final_turn_number,
        );
        assert_ne!(
            outcome.terminal_reason, "max_steps",
            "seed {} hit max_steps at turn {}",
            outcome.seed, outcome.final_turn_number,
        );
        match outcome.winner {
            Some(SideId::Player) => player_wins += 1,
            Some(SideId::Opponent) => opponent_wins += 1,
            None => draws += 1,
        }
    }
    eprintln!(
        "32-seed sweep: player={} opponent={} draws={}",
        player_wins, opponent_wins, draws
    );
    // Sanity: at least SOME games end with a winner.
    assert!(
        player_wins + opponent_wins > 0,
        "32 games and no winners — heuristic AI is misbehaving",
    );
}
