//! Golden full-game test for MCTS pipeline end-to-end.
//!
//! Pure-Rust drive loop, matching sim-mcts-selfplay and the NAPI
//! bridge's `driveMctsGameJson`. (These three paths now produce
//! bit-identical traces; the composable NAPI functions
//! mctsStepJson + advanceStepJson do NOT — they drift due to
//! per-step JSON roundtrip of GameState, where `log: VecDeque` is
//! `#[serde(skip)]` and resets. See memory
//! project_napi_vs_rust_rng_order.)
//!
//! Sibling to:
//!   - golden_full_game_hashes: heuristic-vs-heuristic full games
//!   - golden_mcts_decisions: MCTS's first branching decision

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

/// Captured 2026-05-21 via NAPI driveMctsGameJson (which uses the
/// same pure-Rust loop pattern as this test):
///   simulations=30, cPuct=1.5, K=2, rolloutSteps=100,
///   prior=uniform, leaf=rollout, maxNodes=1000, modelSide=Player.
/// Format: (seed, final_state_hash, steps, winner).
const GOLDEN: &[(&str, &str, u32, &str)] = &[
    ("0", "d81c37747400442e79ed50e2a4e14867", 51, "player"),
    ("1", "9ebcfec8f72f98a75f72cb7403481d67", 53, "opponent"),
    ("7", "fae9bf5657dc1db88cc2bfc2bd65d471", 71, "player"),
    ("42", "0d3f4d96d091896628b98c82456772f8", 71, "player"),
];

fn drive_mcts_game(seed: &str) -> (String, u32, Option<SideId>) {
    let config = MctsConfig {
        simulations: 30,
        c_puct: 1.5,
        leaf: MctsLeaf::Rollout,
        prior: MctsPrior::Uniform,
        rollout_crn_samples: 2,
        rollout_steps: 100,
        value_head_rollout_blend: 0.0,
        add_root_dirichlet: false,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.25,
        max_nodes: 1_000,
        collapse_max_steps: 64,
        adaptive_ratio: 0.0,
        adaptive_min_sims: 30,
        model_url: String::new(),
        onnx_path: None,
    };
    let model_side = SideId::Player;

    let rng = Rng::from_seed(format!("{}:selfplay", seed).as_str(), "selfplay");
    let (mut state, mut step_rng) = with_rng(rng, || setup_ai_vs_ai_game());
    let mut prior_hash = state_hash(&state);

    let mut steps = 0u32;
    for step in 0..500u32 {
        steps = step;
        if state.game_over {
            break;
        }
        let side = match state.current_side {
            CurrentSide::Player => SideId::Player,
            CurrentSide::Opponent => SideId::Opponent,
            CurrentSide::Done => break,
        };
        let (legal, used_after_legal) = with_rng(step_rng.clone(), || {
            enumerate_legal_ai_actions(&state, side)
        });
        step_rng = used_after_legal;

        let (next, used) = if side == model_side && legal.len() > 1 {
            let mcts_seed = format!("{}:player:{}:mcts", seed, step);
            let (result, used_after_mcts) = with_rng(step_rng.clone(), || {
                run_mcts(&state, side, &config, "", mcts_seed.as_str())
            });
            let idx = result.selected_index.min(legal.len() - 1);
            let chosen = legal[idx].clone();
            with_rng(used_after_mcts, || {
                let forced = get_forced_attack_coin_results(&state);
                advance_modeled_turn_step(&state, side, &chosen, forced)
            })
        } else {
            let _ = legal;
            with_rng(step_rng.clone(), || {
                let forced = get_forced_attack_coin_results(&state);
                let mut s = state.clone();
                match side {
                    SideId::Player => advance_player_ai_turn_step(&mut s, forced),
                    SideId::Opponent => advance_opponent_turn_step(&mut s, forced),
                }
                s
            })
        };
        step_rng = used;
        let nh = state_hash(&next);
        if nh == prior_hash {
            break; // stalled
        }
        prior_hash = nh;
        state = next;
    }
    (state_hash(&state), steps, state.winner)
}

#[test]
fn four_seeds_mcts_full_games_match_golden() {
    let mut drift = Vec::new();
    for (seed, expected_hash, expected_steps, expected_winner) in GOLDEN {
        let (got_hash, got_steps, got_winner) = drive_mcts_game(seed);
        let winner_str = got_winner
            .map(|s| match s {
                SideId::Player => "player",
                SideId::Opponent => "opponent",
            })
            .unwrap_or("none");
        if &got_hash != expected_hash
            || got_steps != *expected_steps
            || winner_str != *expected_winner
        {
            drift.push(format!(
                "seed {}: expected ({}, {}, {}), got ({}, {}, {})",
                seed,
                expected_hash,
                expected_steps,
                expected_winner,
                got_hash,
                got_steps,
                winner_str
            ));
        }
    }
    assert!(
        drift.is_empty(),
        "MCTS-driven full games drifted from recorded golden values:\n{}\n\nRegenerate via:\n  node --input-type=module -e \"\
import {{createRequire}} from 'module'; const req = createRequire('file://' + process.cwd() + '/_.mjs'); \
const b = req('./engine-rs/crates/napi-bridge/index.js'); \
const cfg = JSON.stringify({{simulations: 30, cPuct: 1.5, rolloutCrnSamples: 2, rolloutSteps: 100, prior: 'uniform', leaf: 'rollout', maxNodes: 1000}}); \
for (const s of ['0','1','7','42']) {{ const r = JSON.parse(b.driveMctsGameJson(s, 'player', 500, cfg)); console.log(r); }}\"",
        drift.join("\n"),
    );
}
