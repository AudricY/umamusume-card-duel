//! Golden-hash regression test for full heuristic-vs-heuristic games.
//!
//! These hashes pin the END-TO-END engine behavior — setup +
//! heuristic-AI driving + every flow rule + packed-buffer
//! fingerprint — for 5 seeds. If ANY of these change, something
//! semantically meaningful drifted in the engine.
//!
//! What this catches that other tests do NOT:
//!   - A flow rule that subtly mis-orders RNG draws (would change
//!     downstream shuffle outcomes → different terminal state)
//!   - A heuristic-AI scoring change that picks different actions
//!   - A packed-buffer layout change (different bytes packed →
//!     different fingerprint)
//!   - Any combination of the above that doesn't show up in
//!     single-step state-hash tests
//!
//! Regenerating golden values:
//!   node --input-type=module -e "
//!     import {createRequire} from 'module';
//!     const req = createRequire('file://' + process.cwd() + '/_.mjs');
//!     const b = req('./engine-rs/crates/napi-bridge/index.js');
//!     for (const seed of ['0','1','7','42','123']) {
//!       const r = JSON.parse(b.driveHeuristicGameJson(seed, 1000));
//!       console.log(\`(\\\"\${seed}\\\", \\\"\${r.finalStateHash}\\\", \${r.steps}, \\\"\${r.winner}\\\"),\`);
//!     }
//!   "

use engine::core::constants::SideId;
use engine::core::random::{with_rng, Rng};
use engine::core::state::CurrentSide;
use engine::dispatcher::{
    advance_opponent_turn_step, advance_player_ai_turn_step, get_forced_attack_coin_results,
    state_hash,
};
use engine::headless_setup::setup_ai_vs_ai_game;

/// Captured 2026-05-21 against engine HEAD on engine-rust-port branch.
/// Format: (seed, expected_final_hash, expected_steps, expected_winner).
const GOLDEN: &[(&str, &str, u32, &str)] = &[
    ("0", "8aeb5b409c927963c14be1b7145286f6", 50, "player"),
    ("1", "de339ebe08376058bc217469d821f2e8", 48, "opponent"),
    ("7", "00ec9a9bd85406e9adf191b02c57bc9b", 44, "player"),
    ("42", "e658530bdb37790af2bd546dea967009", 62, "player"),
    ("123", "61f0642a2a983f4ea4594d4eb3c0020b", 64, "opponent"),
];

fn drive_one(seed: &str) -> (String, u32, Option<SideId>) {
    let rng = Rng::from_seed(format!("{}:selfplay", seed).as_str(), "selfplay");
    let (mut state, mut step_rng) = with_rng(rng, || setup_ai_vs_ai_game());
    let mut prior_hash = state_hash(&state);
    let mut steps = 0u32;
    for s in 0..1000 {
        steps = s;
        if state.game_over {
            break;
        }
        let side = match state.current_side {
            CurrentSide::Player => SideId::Player,
            CurrentSide::Opponent => SideId::Opponent,
            CurrentSide::Done => break,
        };
        let (next, used) = with_rng(step_rng.clone(), || {
            let forced = get_forced_attack_coin_results(&state);
            let mut s2 = state.clone();
            match side {
                SideId::Player => advance_player_ai_turn_step(&mut s2, forced),
                SideId::Opponent => advance_opponent_turn_step(&mut s2, forced),
            }
            s2
        });
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
fn five_seeds_match_recorded_golden_state() {
    let mut drift = Vec::new();
    for (seed, expected_hash, expected_steps, expected_winner) in GOLDEN {
        let (got_hash, got_steps, got_winner) = drive_one(seed);
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
        "engine behavior drifted from recorded golden values:\n{}\n\nIf this drift is INTENTIONAL, regenerate the GOLDEN table per the doc-comment instructions.",
        drift.join("\n"),
    );
}
