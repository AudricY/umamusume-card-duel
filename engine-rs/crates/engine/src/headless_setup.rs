//! Bit-identical port of `backend/src/sim/headlessAiVsAi.ts:setupAiVsAiGame`.
//!
//! Drives a fresh AI-vs-AI `GameState` to `phase=play, turn=1`. All RNG
//! draws (opening-coin flip + opening-hand shuffles) happen inside the
//! caller's `with_rng` scope so they're sequenced against the same
//! outer PRNG the recorder used.

use crate::core::card_id::CardId;
use crate::core::constants::{AiDifficulty, CoinFlipResult, SideId};
use crate::core::decks::{default_ai_opponent_deck, default_player_deck};
use crate::core::random::random_float;
use crate::core::state::{GameState, Phase};
use crate::dispatcher::{
    auto_complete_opponent_setup, choose_opening_coin, complete_pregame_setup, create_game,
    deal_opening_hands, tick_setup_countdown,
};
use crate::flow::ai::setup_selection::choose_ai_setup_selection;

/// Mirror of `setupAiVsAiGame()` in `backend/src/sim/headlessAiVsAi.ts:96`.
///
/// Must be called inside a `with_rng` scope. Returns the post-setup state
/// in `Phase::Play`.
///
/// Panics if the AI cannot pick a basic for the player's active slot
/// (mirrors the TS `throw new Error("Unable to choose AI setup …")`).
pub fn setup_ai_vs_ai_game() -> GameState {
    setup_ai_vs_ai_game_with_decks(None, None)
}

/// Variant of `setup_ai_vs_ai_game()` that lets the caller override the
/// player / opponent deck card lists.
///
/// Either side accepts an `Option<&[CardId]>`: `None` keeps the default
/// (matches `setup_ai_vs_ai_game()` exactly — same bytes for player and
/// opponent), `Some(slice)` uses the explicit list. Used by:
///
/// - `--deck-sampling=uniform|pair=...` plumbing in the sim-cli binaries.
/// - The deck-pair legality smoke
///   (`engine-rs/crates/engine/tests/deck_pair_legality.rs`).
///
/// Calling `setup_ai_vs_ai_game_with_decks(None, None)` is byte-identical
/// to `setup_ai_vs_ai_game()`; this is the seam Slice 1 of the
/// `deck-pair-sampling` scoping doc plumbs through the CLI.
pub fn setup_ai_vs_ai_game_with_decks(
    player_deck: Option<&[CardId]>,
    opponent_deck: Option<&[CardId]>,
) -> GameState {
    let default_player = default_player_deck();
    let default_opp = default_ai_opponent_deck();
    let player_slice = player_deck.unwrap_or(default_player);
    let opp_slice = opponent_deck.unwrap_or(default_opp);
    let player_deck_vec = player_slice.to_vec();
    let opp_deck_vec = opp_slice.to_vec();
    let mut state = create_game(
        &player_deck_vec,
        &opp_deck_vec,
        "Opponent",
        AiDifficulty::Hard,
        false,
        "Player AI",
        None,
        None,
    );
    state.human_by_side[SideId::Player as usize] = false;
    state.human_by_side[SideId::Opponent as usize] = false;

    let coin = if random_float() >= 0.5 {
        CoinFlipResult::Heads
    } else {
        CoinFlipResult::Tails
    };
    choose_opening_coin(&mut state, coin);
    deal_opening_hands(&mut state);

    let selection = {
        let player = state.side(SideId::Player);
        choose_ai_setup_selection(player).expect("Unable to choose AI setup for player.")
    };

    complete_pregame_setup(&mut state, selection.active_index, &selection.bench_indexes);
    auto_complete_opponent_setup(&mut state);

    for _ in 0..5 {
        if state.phase != Phase::Setup {
            break;
        }
        tick_setup_countdown(&mut state);
    }
    if state.phase != Phase::Play {
        panic!("Headless setup did not enter play phase.");
    }
    state
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::random::{with_rng, Rng};
    use crate::core::state::CurrentSide;

    #[test]
    fn setup_ai_vs_ai_game_reaches_play_phase() {
        let (state, _) = with_rng(Rng::from_seed("0:selfplay", "selfplay"), || {
            setup_ai_vs_ai_game()
        });
        assert_eq!(state.phase, Phase::Play);
        assert!(state.sides[0].active.is_some(), "player should have active");
        assert!(
            state.sides[1].active.is_some(),
            "opponent should have active"
        );
        assert!(state.turn_number >= 1);
        assert!(matches!(
            state.current_side,
            CurrentSide::Player | CurrentSide::Opponent
        ));
    }

    #[test]
    fn setup_is_deterministic_given_seed() {
        let (a, _) = with_rng(Rng::from_seed("42:selfplay", "selfplay"), || {
            setup_ai_vs_ai_game()
        });
        let (b, _) = with_rng(Rng::from_seed("42:selfplay", "selfplay"), || {
            setup_ai_vs_ai_game()
        });
        // Hand cards should match.
        assert_eq!(a.sides[0].hand.as_slice(), b.sides[0].hand.as_slice());
        assert_eq!(a.sides[1].hand.as_slice(), b.sides[1].hand.as_slice());
        // Active uid matches.
        assert_eq!(
            a.sides[0].active.as_ref().map(|u| u.uid),
            b.sides[0].active.as_ref().map(|u| u.uid),
        );
    }
}
