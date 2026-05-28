//! Integration tests for `state_hash` — the cross-cutting fingerprint
//! that golden-trace replay, MCTS no-progress detection, and the
//! headless-drive stall detector all depend on.
//!
//! A regression here typically manifests as either:
//!   * spurious "stalled" reports (hash changes when nothing changed)
//!   * silent infinite loops (hash doesn't change when something did)
//! Both are subtle enough to escape coarser test signal.

use engine::core::constants::SideId;
use engine::core::random::{with_rng, Rng};
use engine::core::state::CurrentSide;
use engine::dispatcher::{
    advance_opponent_turn_step, advance_player_ai_turn_step, get_forced_attack_coin_results,
    state_hash,
};
use engine::headless_setup::setup_ai_vs_ai_game;

fn fresh_state(seed: &str) -> engine::core::state::GameState {
    let rng = Rng::from_seed(format!("{}:selfplay", seed).as_str(), "selfplay");
    let (state, _used) = with_rng(rng, || setup_ai_vs_ai_game());
    state
}

#[test]
fn state_hash_is_deterministic() {
    // Same seed → same setup → same hash. If this drifts, golden-trace
    // replay can never succeed even with bit-identical engine output.
    let a = state_hash(&fresh_state("0"));
    let b = state_hash(&fresh_state("0"));
    assert_eq!(a, b);
}

#[test]
fn state_hash_differs_across_seeds() {
    // Different seeds → different setups → different hashes (with very
    // high probability, given different decks and coin outcomes).
    let h0 = state_hash(&fresh_state("0"));
    let h1 = state_hash(&fresh_state("1"));
    let h2 = state_hash(&fresh_state("42"));
    assert_ne!(h0, h1);
    assert_ne!(h0, h2);
    assert_ne!(h1, h2);
}

#[test]
fn state_hash_format_is_stable() {
    // Format is "{:032x}" — exactly 32 lower-case hex chars. Any change
    // breaks consumers parsing the hash string (e.g., the golden-trace
    // gate's chunk-fingerprint check).
    let h = state_hash(&fresh_state("99"));
    assert_eq!(h.len(), 32, "hash was {}", h);
    assert!(
        h.chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
        "hash {} has non-lowercase-hex chars",
        h
    );
}

#[test]
fn state_hash_changes_when_state_advances() {
    // After a heuristic step on a fresh game, the hash MUST change.
    // (If it doesn't, headless drive's stall detector falsely fires.)
    let mut state = fresh_state("0");
    let pre = state_hash(&state);
    let rng = Rng::from_seed("0:selfplay-advance", "selfplay-advance");
    let side = match state.current_side {
        CurrentSide::Player => SideId::Player,
        CurrentSide::Opponent => SideId::Opponent,
        CurrentSide::Done => panic!("fresh game is already done"),
    };
    let (next_state, _) = with_rng(rng, || {
        let forced = get_forced_attack_coin_results(&state);
        let mut s = state.clone();
        match side {
            SideId::Player => advance_player_ai_turn_step(&mut s, forced),
            SideId::Opponent => advance_opponent_turn_step(&mut s, forced),
        }
        s
    });
    state = next_state;
    let post = state_hash(&state);
    assert_ne!(
        pre, post,
        "state_hash unchanged after a heuristic step — stall detector would false-fire"
    );
}

#[test]
fn state_hash_unchanged_after_clone() {
    // Cloning the state must not change its hash. (This guards against
    // packed-state-buffer codepaths that read process-local memory
    // addresses or RNG draws as part of the fingerprint.)
    let state = fresh_state("0");
    let h1 = state_hash(&state);
    let clone = state.clone();
    let h2 = state_hash(&clone);
    assert_eq!(h1, h2);
}
