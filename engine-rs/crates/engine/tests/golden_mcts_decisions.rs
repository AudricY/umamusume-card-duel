//! Golden-decision regression test for MCTS.
//!
//! Pins MCTS's selectedIndex + visit distribution + rootValue at the
//! FIRST branching decision (the earliest step with legal_actions > 1)
//! for 4 seeds. This locks in the entire search pipeline end-to-end:
//!
//!   * setup (RNG ordering)
//!   * heuristic AI step (RNG advance schedule)
//!   * legal-action enumeration
//!   * MCTS expansion / selection / rollout / backpropagation
//!   * leaf-value rollout heuristic
//!
//! Any of those drifting silently changes the search result. The
//! existing MctsResult-invariants test catches structural issues
//! (visits length, argmax-matches-selected, etc.) but doesn't catch
//! "MCTS still produces a structurally-valid result, just a different
//! one". This test does.

use engine::core::constants::SideId;
use engine::core::random::{with_rng, Rng};
use engine::core::state::CurrentSide;
use engine::dispatcher::{
    advance_opponent_turn_step, advance_player_ai_turn_step, get_forced_attack_coin_results,
};
use engine::headless_setup::setup_ai_vs_ai_game;
use engine::mcts::config::{MctsConfig, MctsLeaf, MctsPrior};
use engine::mcts::driver::run_mcts;
use engine::policy::actions::enumerate_legal_ai_actions;

/// Captured 2026-05-21 via the NAPI bridge:
///   simulations=50, cPuct=1.5, rolloutCrnSamples=2, rolloutSteps=100,
///   prior=uniform, leaf=rollout, maxNodes=1000.
///
/// Format: (seed, branch_step, legal_count, selected_index,
///          visits_string ('|'-separated), total_visits, root_value).
const GOLDEN: &[(&str, u32, usize, usize, &str, u32, f64)] = &[
    ("0", 1, 2, 1, "21|29", 50, -1.000000),
    ("1", 1, 10, 5, "1|1|1|1|1|39|1|1|1|3", 50, 0.000000),
    ("7", 1, 3, 2, "14|14|22", 50, 0.000000),
    ("42", 3, 5, 3, "1|6|12|17|14", 50, 1.000000),
];

fn first_mcts_decision(seed: &str) -> (u32, usize, usize, String, u32, f64) {
    let config = MctsConfig {
        simulations: 50,
        c_puct: 1.5,
        leaf: MctsLeaf::Rollout,
        prior: MctsPrior::Uniform,
        rollout_crn_samples: 2,
        rollout_steps: 100,
        add_root_dirichlet: false,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.25,
        max_nodes: 1_000,
        collapse_max_steps: 64,
        adaptive_ratio: 0.0,
        adaptive_min_sims: 50,
        model_url: String::new(),
    };

    let rng = Rng::from_seed(format!("{}:selfplay", seed).as_str(), "selfplay");
    let (mut state, mut step_rng) = with_rng(rng, || setup_ai_vs_ai_game());

    for step in 0..30u32 {
        let side = match state.current_side {
            CurrentSide::Player => SideId::Player,
            CurrentSide::Opponent => SideId::Opponent,
            CurrentSide::Done => panic!("game over before first branching decision"),
        };
        let (legal, used) = with_rng(step_rng.clone(), || {
            enumerate_legal_ai_actions(&state, side)
        });
        step_rng = used;
        if legal.len() > 1 {
            let mcts_seed = format!("{}:P:{}:mcts", seed, step);
            let (result, _) = with_rng(step_rng.clone(), || {
                run_mcts(&state, side, &config, "", mcts_seed.as_str())
            });
            let visits_str = result
                .visits
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join("|");
            let total: u32 = result.visits.iter().sum();
            return (
                step,
                legal.len(),
                result.selected_index,
                visits_str,
                total,
                result.diagnostics.root_value,
            );
        }
        // No branching at this step — advance heuristically and try again.
        let (next, used) = with_rng(step_rng.clone(), || {
            let forced = get_forced_attack_coin_results(&state);
            let mut s = state.clone();
            match side {
                SideId::Player => advance_player_ai_turn_step(&mut s, forced),
                SideId::Opponent => advance_opponent_turn_step(&mut s, forced),
            }
            s
        });
        step_rng = used;
        state = next;
    }
    panic!("no branching decision in first 30 steps");
}

#[test]
fn mcts_first_branching_decisions_match_golden() {
    let mut drift = Vec::new();
    for (
        seed,
        expected_step,
        expected_legal,
        expected_selected,
        expected_visits,
        expected_total,
        expected_root_value,
    ) in GOLDEN
    {
        let (got_step, got_legal, got_selected, got_visits, got_total, got_root_value) =
            first_mcts_decision(seed);
        let root_drift = (got_root_value - *expected_root_value).abs() > 1e-6;
        if got_step != *expected_step
            || got_legal != *expected_legal
            || got_selected != *expected_selected
            || got_visits != *expected_visits
            || got_total != *expected_total
            || root_drift
        {
            drift.push(format!(
                "seed {}: expected (step={}, legal={}, sel={}, visits='{}', total={}, rv={:.6}), got (step={}, legal={}, sel={}, visits='{}', total={}, rv={:.6})",
                seed,
                expected_step,
                expected_legal,
                expected_selected,
                expected_visits,
                expected_total,
                expected_root_value,
                got_step,
                got_legal,
                got_selected,
                got_visits,
                got_total,
                got_root_value,
            ));
        }
    }
    assert!(
        drift.is_empty(),
        "MCTS first-branching-decisions drifted from recorded golden values:\n{}",
        drift.join("\n"),
    );
}
