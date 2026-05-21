//! Bit-identical port of `backend/src/sim/headlessAiVsAi.ts:setupAiVsAiGame`.
//!
//! Drives a fresh AI-vs-AI `GameState` to `phase=play, turn=1`. All RNG
//! draws (opening-coin flip + opening-hand shuffles) happen inside the
//! caller's `with_rng` scope so they're sequenced against the same
//! outer PRNG the recorder used.

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
    let player_deck = default_player_deck().to_vec();
    let opp_deck = default_ai_opponent_deck().to_vec();
    let mut state = create_game(
        &player_deck,
        &opp_deck,
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
        assert!(state.sides[1].active.is_some(), "opponent should have active");
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
        assert_eq!(
            a.sides[0].hand.as_slice(),
            b.sides[0].hand.as_slice()
        );
        assert_eq!(
            a.sides[1].hand.as_slice(),
            b.sides[1].hand.as_slice()
        );
        // Active uid matches.
        assert_eq!(
            a.sides[0].active.as_ref().map(|u| u.uid),
            b.sides[0].active.as_ref().map(|u| u.uid),
        );
    }
}
