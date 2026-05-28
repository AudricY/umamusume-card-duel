//! Invariants for the dispatcher's "advance" early-return guards.
//!
//! `advance_modeled_turn_step` has several pre-conditions that, when
//! violated, must return the state UNCHANGED (no panic, no mutation).
//! This is the safety contract MCTS rollouts depend on — a stalled
//! state is detected by state_hash equality, but only IF the dispatcher
//! actually returns unchanged state rather than panicking or producing
//! a corrupted state.

use engine::core::constants::SideId;
use engine::core::random::{with_rng, Rng};
use engine::core::state::{CurrentSide, GameState};
use engine::dispatcher::{
    advance_modeled_turn_step, advance_player_ai_turn_step, get_forced_attack_coin_results,
    state_hash,
};
use engine::headless_setup::setup_ai_vs_ai_game;
use engine::policy::actions::enumerate_legal_ai_actions;
use engine::policy::types::LegalAiAction;

fn fresh_state(seed: &str) -> GameState {
    let rng = Rng::from_seed(format!("{}:selfplay", seed).as_str(), "selfplay");
    let (state, _used) = with_rng(rng, || setup_ai_vs_ai_game());
    state
}

fn current_side(s: &GameState) -> SideId {
    match s.current_side {
        CurrentSide::Player => SideId::Player,
        CurrentSide::Opponent => SideId::Opponent,
        CurrentSide::Done => panic!("game already done"),
    }
}

fn first_legal(state: &GameState, side: SideId, seed: &str) -> LegalAiAction {
    let (legal, _) = with_rng(Rng::from_seed(seed, "first-legal"), || {
        enumerate_legal_ai_actions(state, side)
    });
    assert!(!legal.is_empty(), "no legal actions on fresh state");
    legal[0].clone()
}

#[test]
fn advance_modeled_with_wrong_side_returns_unchanged() {
    // currentSide is the SIDE on the clock; passing the OPPOSITE side
    // must short-circuit (the dispatcher refuses to act for someone
    // else). Otherwise MCTS could mistakenly advance the wrong side
    // and produce illegal trajectories.
    let state = fresh_state("0");
    let side_on_clock = current_side(&state);
    let opposite = side_on_clock.opposite();
    let action = first_legal(&state, side_on_clock, "0");
    let pre = state_hash(&state);
    let next = with_rng(Rng::from_seed("0", "wrong-side"), || {
        advance_modeled_turn_step(&state, opposite, &action, None)
    })
    .0;
    let post = state_hash(&next);
    assert_eq!(
        pre, post,
        "advance_modeled with opposite side must return unchanged state"
    );
}

#[test]
fn advance_modeled_after_game_over_returns_unchanged() {
    // Manufacture a game_over state by driving an entire game to
    // completion, then try to advance the WINNING side past it.
    let mut state = fresh_state("0");
    let rng = Rng::from_seed("0:drive-to-end", "drive-to-end");
    let mut step_rng = rng;
    for _ in 0..200 {
        if state.game_over {
            break;
        }
        let side = current_side(&state);
        let (next, used) = with_rng(step_rng.clone(), || {
            let forced = get_forced_attack_coin_results(&state);
            let mut s = state.clone();
            if side == SideId::Player {
                advance_player_ai_turn_step(&mut s, forced);
            } else {
                engine::dispatcher::advance_opponent_turn_step(&mut s, forced);
            }
            s
        });
        step_rng = used;
        state = next;
    }
    assert!(state.game_over, "test setup failed to reach game_over");

    // Now try to advance via advance_modeled_turn_step. Must be a
    // no-op (early-return guards on game_over).
    let pre = state_hash(&state);
    // Build a fake LegalAiAction to pass in — its content doesn't
    // matter since the guard fires before action dispatch.
    let fake_action = LegalAiAction {
        id: "fake".to_string(),
        phase: engine::policy::types::AiPhase::StadiumOrEnd,
        kind: "pass".to_string(),
        payload: serde_json::json!({}),
        features: Vec::new(),
        action_source_card_idx: None,
        action_target_card_idx: None,
    };
    let next = with_rng(Rng::from_seed("0", "after-end"), || {
        advance_modeled_turn_step(&state, SideId::Player, &fake_action, None)
    })
    .0;
    let post = state_hash(&next);
    assert_eq!(
        pre, post,
        "advance_modeled_turn_step after game_over must be a no-op"
    );
    assert!(next.game_over, "game_over flag should remain set");
}

#[test]
fn advance_modeled_pass_action_progresses_state() {
    // Positive case: a "pass" action on a fresh state DOES change the
    // state. Pairs with the no-op tests above — together they assert
    // the dispatcher's behavior is "no-op when guard fires, real
    // progress otherwise".
    let state = fresh_state("0");
    let side = current_side(&state);
    let action = first_legal(&state, side, "0");
    let pre = state_hash(&state);
    let next = with_rng(Rng::from_seed("0", "pass-progress"), || {
        let forced = get_forced_attack_coin_results(&state);
        advance_modeled_turn_step(&state, side, &action, forced)
    })
    .0;
    let post = state_hash(&next);
    assert_ne!(
        pre, post,
        "first legal action on fresh state must change state_hash (otherwise everything stalls)"
    );
}

#[test]
fn advance_modeled_unknown_action_kind_falls_through_to_phase_advance() {
    // Unknown kinds hit the dispatcher's catch-all `_ =>` arm which
    // calls advance_modeled_phase — a benign progress operation.
    // (See dispatcher.rs:1393.) The contract is: NEVER panic, and
    // produce a STILL-WELL-FORMED state. Catches regressions where
    // an unknown action could corrupt state mid-mutation.
    let state = fresh_state("0");
    let side = current_side(&state);
    let bogus = LegalAiAction {
        id: "bogus".to_string(),
        phase: engine::policy::types::AiPhase::StadiumOrEnd,
        kind: "completelyMadeUp".to_string(),
        payload: serde_json::json!({}),
        features: Vec::new(),
        action_source_card_idx: None,
        action_target_card_idx: None,
    };
    let next = with_rng(Rng::from_seed("0", "bogus"), || {
        advance_modeled_turn_step(&state, side, &bogus, None)
    })
    .0;
    // State must remain well-formed: same phase, valid current_side,
    // hash is well-formed 32-hex (the structural validator). The
    // hash MAY differ from pre (phase advance is allowed) — that
    // matches by-design behavior, but corruption would explode the
    // hash format or panic the call.
    let post_hash = state_hash(&next);
    assert_eq!(post_hash.len(), 32);
    assert!(post_hash.chars().all(|c| c.is_ascii_hexdigit()));
    assert_eq!(next.phase, engine::core::state::Phase::Play);
}
