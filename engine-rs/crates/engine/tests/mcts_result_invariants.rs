//! Invariants the `MctsResult` returned from `run_mcts` must satisfy.
//!
//! Downstream consumers (visit-distribution sampling, training-row
//! recording, /predict policy targets) implicitly trust these. A
//! regression that leaves `visits` shorter than `legal_actions`, or
//! `selected_index` out of bounds, can index past the end of the
//! action list and panic — or worse, silently pick the wrong action.

use engine::core::constants::SideId;
use engine::core::random::{with_rng, Rng};
use engine::core::state::CurrentSide;
use engine::headless_setup::setup_ai_vs_ai_game;
use engine::mcts::config::{MctsConfig, MctsLeaf, MctsPrior};
use engine::mcts::driver::run_mcts;
use engine::policy::actions::enumerate_legal_ai_actions;

fn default_config(sims: u32) -> MctsConfig {
    MctsConfig {
        simulations: sims,
        c_puct: 1.5,
        leaf: MctsLeaf::Rollout,
        prior: MctsPrior::Uniform,
        rollout_crn_samples: 2,
        rollout_steps: 100,
        record_rollout_leaf_samples: 0,
        value_head_rollout_blend: 0.0,
        add_root_dirichlet: false,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.25,
        max_nodes: 1_000,
        collapse_max_steps: 64,
        adaptive_ratio: 0.0,
        adaptive_min_sims: 50,
        two_sided: false,
        model_url: String::new(),
        onnx_path: None,
    }
}

/// Drive setup, find a state with multiple legal actions, then run
/// MCTS once at that decision point.
fn mcts_at_first_branching_decision(
    seed: &str,
    config: &MctsConfig,
) -> (
    Vec<engine::policy::types::LegalAiAction>,
    engine::mcts::config::MctsResult,
) {
    let setup_rng = Rng::from_seed(format!("{}:selfplay", seed).as_str(), "selfplay");
    let (state, mut step_rng) = with_rng(setup_rng, || setup_ai_vs_ai_game());

    let side = match state.current_side {
        CurrentSide::Player => SideId::Player,
        CurrentSide::Opponent => SideId::Opponent,
        CurrentSide::Done => panic!("game over on fresh state"),
    };

    let (legal, used) = with_rng(step_rng.clone(), || {
        enumerate_legal_ai_actions(&state, side)
    });
    step_rng = used;

    // run_mcts called outside an active rng scope panics — wrap.
    let (result, _) = with_rng(step_rng.clone(), || {
        run_mcts(
            &state,
            side,
            config,
            "",
            format!("{}:mcts-test", seed).as_str(),
        )
    });

    (legal, result)
}

#[test]
fn visits_length_matches_legal_actions() {
    let config = default_config(64);
    let (legal, result) = mcts_at_first_branching_decision("0", &config);
    assert_eq!(
        result.visits.len(),
        legal.len(),
        "visits[{}] != legal_actions[{}] — downstream consumers will index past end",
        result.visits.len(),
        legal.len()
    );
}

#[test]
fn selected_index_is_within_bounds() {
    let config = default_config(64);
    for seed in &["0", "1", "7", "42"] {
        let (legal, result) = mcts_at_first_branching_decision(seed, &config);
        assert!(
            result.selected_index < legal.len(),
            "seed {}: selected_index {} >= legal.len {}",
            seed,
            result.selected_index,
            legal.len()
        );
    }
}

#[test]
fn visits_are_nonnegative_and_at_least_one_is_positive() {
    // u32 makes "non-negative" trivial; the substantive guarantee is
    // that MCTS actually visited *something*, not just returned all
    // zeros. (All zeros would mean rollout/expansion never fired.)
    let config = default_config(64);
    let (_legal, result) = mcts_at_first_branching_decision("0", &config);
    let total: u64 = result.visits.iter().map(|&v| v as u64).sum();
    assert!(
        total > 0,
        "total visits across all actions is 0 — MCTS did not search anything"
    );
}

#[test]
fn argmax_of_visits_matches_selected_index() {
    // MCTS picks the most-visited action at the root, per AlphaZero
    // protocol. (Temperature sampling is layered on top of this in
    // the sim binary; the raw result.selected_index is argmax.)
    let config = default_config(128);
    let (_legal, result) = mcts_at_first_branching_decision("0", &config);
    let argmax = result
        .visits
        .iter()
        .enumerate()
        .max_by_key(|(_, &v)| v)
        .map(|(i, _)| i)
        .expect("at least one visit");
    assert_eq!(
        result.selected_index, argmax,
        "selected_index {} differs from visit-argmax {}",
        result.selected_index, argmax
    );
}

#[test]
fn deterministic_given_same_seed() {
    // Same state + same MCTS seed → same MctsResult.
    let config = default_config(64);
    let (legal_a, result_a) = mcts_at_first_branching_decision("42", &config);
    let (legal_b, result_b) = mcts_at_first_branching_decision("42", &config);
    assert_eq!(legal_a.len(), legal_b.len());
    assert_eq!(result_a.selected_index, result_b.selected_index);
    assert_eq!(result_a.visits, result_b.visits);
}

#[test]
fn root_value_is_finite_and_bounded() {
    // root_value is consumed as a training target; if it's NaN/Inf or
    // outside [-1, 1] it poisons any model that reads it.
    let config = default_config(64);
    for seed in &["0", "1", "7", "42"] {
        let (_legal, result) = mcts_at_first_branching_decision(seed, &config);
        let v = result.diagnostics.root_value;
        assert!(
            v.is_finite(),
            "seed {}: root_value is non-finite: {}",
            seed,
            v
        );
        assert!(
            (-1.001..=1.001).contains(&v),
            "seed {}: root_value {} outside [-1, 1] training-target range",
            seed,
            v
        );
    }
}
