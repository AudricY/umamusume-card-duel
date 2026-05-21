//! Integration tests for `enumerate_legal_ai_actions` against real
//! post-setup game states. These guard the cross-cutting invariants
//! every MCTS / heuristic / training consumer relies on:
//!
//!   1. Result is non-empty during play phase (endTurn at minimum).
//!   2. Result is deterministic given the same state.
//!   3. Enumeration does not mutate the state (state_hash unchanged).
//!   4. Action `id` strings are unique within a single enumeration
//!      (otherwise visit distributions collide on the model side).

use std::collections::HashSet;

use engine::core::constants::SideId;
use engine::core::random::{with_rng, Rng};
use engine::core::state::CurrentSide;
use engine::dispatcher::state_hash;
use engine::headless_setup::setup_ai_vs_ai_game;
use engine::policy::actions::enumerate_legal_ai_actions;

fn fresh_state(seed: &str) -> engine::core::state::GameState {
    let rng = Rng::from_seed(format!("{}:selfplay", seed).as_str(), "selfplay");
    let (state, _used) = with_rng(rng, || setup_ai_vs_ai_game());
    state
}

fn current_side(s: &engine::core::state::GameState) -> SideId {
    match s.current_side {
        CurrentSide::Player => SideId::Player,
        CurrentSide::Opponent => SideId::Opponent,
        CurrentSide::Done => panic!("game over — no current side"),
    }
}

#[test]
fn legal_actions_nonempty_on_fresh_setup() {
    for seed in &["0", "1", "7", "42", "123"] {
        let state = fresh_state(seed);
        let side = current_side(&state);
        let (legal, _) = with_rng(Rng::from_seed(*seed, "legal-enum"), || {
            enumerate_legal_ai_actions(&state, side)
        });
        assert!(
            !legal.is_empty(),
            "seed {} yielded zero legal actions for {:?}",
            seed,
            side
        );
    }
}

#[test]
fn legal_actions_deterministic_given_state() {
    let state = fresh_state("0");
    let side = current_side(&state);
    let (a, _) = with_rng(Rng::from_seed("0", "legal-det"), || {
        enumerate_legal_ai_actions(&state, side)
    });
    let (b, _) = with_rng(Rng::from_seed("0", "legal-det"), || {
        enumerate_legal_ai_actions(&state, side)
    });
    assert_eq!(a.len(), b.len());
    for (l, r) in a.iter().zip(b.iter()) {
        assert_eq!(l.id, r.id, "action ids differ");
        assert_eq!(l.kind, r.kind, "action kinds differ");
    }
}

#[test]
fn enumeration_does_not_mutate_state() {
    let state = fresh_state("42");
    let side = current_side(&state);
    let pre = state_hash(&state);
    let _ = with_rng(Rng::from_seed("42", "no-mutate"), || {
        enumerate_legal_ai_actions(&state, side)
    });
    let post = state_hash(&state);
    assert_eq!(
        pre, post,
        "enumerate_legal_ai_actions mutated the state — visible to caller"
    );
}

#[test]
fn legal_action_ids_unique_within_enumeration() {
    // Visit distributions are indexed positionally by legal-action
    // index — but training/MCTS code joins by `action.id`. Duplicate
    // ids collapse visits to one slot in any code that joins by id.
    for seed in &["0", "1", "7", "42", "123"] {
        let state = fresh_state(seed);
        let side = current_side(&state);
        let (legal, _) = with_rng(Rng::from_seed(*seed, "id-unique"), || {
            enumerate_legal_ai_actions(&state, side)
        });
        let mut ids: HashSet<&str> = HashSet::new();
        for a in &legal {
            assert!(
                ids.insert(a.id.as_str()),
                "seed {} had duplicate action id {:?}",
                seed,
                a.id
            );
        }
    }
}

#[test]
fn progress_action_is_always_legal_during_play() {
    // The game's liveness guarantee: there is ALWAYS a forward-progress
    // action available — either "endTurn" (normal turn end) or "pass"
    // (the only legal action on the first turn, since attacks are
    // restricted T1). Without one of these the side deadlocks.
    for seed in &["0", "1", "7", "42", "123"] {
        let state = fresh_state(seed);
        let side = current_side(&state);
        let (legal, _) = with_rng(Rng::from_seed(*seed, "progress"), || {
            enumerate_legal_ai_actions(&state, side)
        });
        let has_progress = legal.iter().any(|a| a.kind == "endTurn" || a.kind == "pass");
        assert!(
            has_progress,
            "seed {} for {:?} has no endTurn/pass action (kinds: {:?})",
            seed,
            side,
            legal.iter().map(|a| a.kind.as_str()).collect::<Vec<_>>()
        );
    }
}
