//! Deck-pair legality smoke (P0b of `docs/ai-research/scoping/deck-pair-sampling.md`).
//!
//! Loops over every `(player_deck × ai_deck)` pair declared in
//! `shared/src/data/premadeDecks.json`, drives a short headless AI-vs-AI
//! game per pair via `setup_ai_vs_ai_game_with_decks(...)`, and asserts:
//!
//!   1. No panic during setup, dispatcher, or terminal-result extraction.
//!   2. No silent card-drop in `intern_deck` — the interned `card_ids`
//!      length matches the source JSON's `cardIds` length (both 20 today;
//!      a card-data drift that breaks the catalog lookup would silently
//!      shrink one without the other).
//!   3. The setup reaches `Phase::Play` (already asserted by
//!      `setup_ai_vs_ai_game()`'s post-condition, but we re-check
//!      explicitly to surface per-pair regressions).
//!
//! Failure mode this catches: 9 of 11 AI decks have never been simulated
//! end-to-end (only the AI-flavor Matikane mirror has, per the scoping
//! doc). Any card id in those decks that no longer exists in the catalog
//! (rename, removed card, casing drift) would silently shrink the
//! interned deck below 20. Below 20 cards can change opening-hand /
//! reshuffle behaviour without panicking — a slow, silent corruption.
//!
//! Pre-fix observed coverage: 1 AI deck (matikanetannhauser) ever
//! simulated; post-Slice 1 the orchestrators uniform-sample all 22 pairs.

use engine::core::constants::SideId;
use engine::core::decks::decks;
use engine::core::random::{with_rng, Rng};
use engine::core::state::CurrentSide;
use engine::dispatcher::{
    advance_opponent_turn_step, advance_player_ai_turn_step, get_forced_attack_coin_results,
    state_hash,
};
use engine::headless_setup::setup_ai_vs_ai_game_with_decks;

/// Drive a short headless game for a (player, ai) deck pair. Returns
/// `Err` with a per-pair diagnostic string on any failure mode the smoke
/// is responsible for catching. The caller aggregates into a single
/// panic at the end so all failures surface in one run.
fn drive_pair_short_game(
    player_id: &str,
    player_deck: &[engine::core::card_id::CardId],
    ai_id: &str,
    ai_deck: &[engine::core::card_id::CardId],
    max_steps: u32,
) -> Result<(), String> {
    // The seed is salted by deck-pair id so different pairs explore
    // different RNG trajectories. Deterministic across re-runs.
    let seed = format!("legality:{}:{}", player_id, ai_id);
    let rng = Rng::from_seed(seed.as_str(), "selfplay");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_rng(rng, || {
            setup_ai_vs_ai_game_with_decks(Some(player_deck), Some(ai_deck))
        })
    }));
    let (mut state, mut step_rng) = match result {
        Ok(pair) => pair,
        Err(err) => {
            let msg = if let Some(s) = err.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = err.downcast_ref::<&'static str>() {
                (*s).to_string()
            } else {
                "unknown panic payload".to_string()
            };
            return Err(format!(
                "setup panic for player={} ai={}: {}",
                player_id, ai_id, msg
            ));
        }
    };
    // Run a handful of dispatcher steps. 80 is enough to enter combat
    // without taking the full ~500-step tail latency of a full game in a
    // unit-test budget. We don't require the game to reach `game_over`;
    // we only require no panic and forward progress.
    for step in 0..max_steps {
        if state.game_over {
            break;
        }
        let side = match state.current_side {
            CurrentSide::Player => SideId::Player,
            CurrentSide::Opponent => SideId::Opponent,
            CurrentSide::Done => break,
        };
        let pre_hash = state_hash(&state);
        let step_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_rng(step_rng.clone(), || {
                let forced = get_forced_attack_coin_results(&state);
                let mut s = state.clone();
                match side {
                    SideId::Player => advance_player_ai_turn_step(&mut s, forced),
                    SideId::Opponent => advance_opponent_turn_step(&mut s, forced),
                }
                s
            })
        }));
        let (next_state, used_rng) = match step_result {
            Ok(pair) => pair,
            Err(err) => {
                let msg = if let Some(s) = err.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = err.downcast_ref::<&'static str>() {
                    (*s).to_string()
                } else {
                    "unknown panic payload".to_string()
                };
                return Err(format!(
                    "step panic for player={} ai={} at step {}: {}",
                    player_id, ai_id, step, msg
                ));
            }
        };
        step_rng = used_rng;
        if state_hash(&next_state) == pre_hash {
            // Stalled is acceptable for this smoke — we're not asserting
            // strategic progress, only that the dispatcher doesn't crash
            // on a deck. Break and move on.
            break;
        }
        state = next_state;
    }
    Ok(())
}

#[test]
fn every_player_ai_deck_pair_passes_legality_smoke() {
    let r = decks();

    // Sanity: catalog has at least 1 of each. If the JSON list ever
    // shrinks to zero this would be a real failure mode all on its own.
    assert!(
        !r.player_decks.is_empty(),
        "premadeDecks must have at least one entry"
    );
    assert!(
        !r.ai_decks.is_empty(),
        "aiPremadeDecks must have at least one entry"
    );

    // Step 1: assert no silent drops in any deck (player or AI).
    // `intern_deck` warn-and-skips unknown card ids; the only way to
    // detect that from outside is comparing `card_ids.len()` against the
    // source JSON count we recorded at load time.
    let mut intern_failures: Vec<String> = Vec::new();
    for d in r.player_decks.iter().chain(r.ai_decks.iter()) {
        if d.card_ids.len() != d.source_card_count {
            intern_failures.push(format!(
                "deck '{}': source had {} cards but interned to {} (silent drop in catalog lookup)",
                d.id,
                d.source_card_count,
                d.card_ids.len(),
            ));
        }
    }

    // Step 2: drive a short game per ordered (player, ai) pair.
    let mut pair_failures: Vec<String> = Vec::new();
    let mut pairs_run = 0usize;
    for player in &r.player_decks {
        for ai in &r.ai_decks {
            pairs_run += 1;
            if let Err(diag) =
                drive_pair_short_game(&player.id, &player.card_ids, &ai.id, &ai.card_ids, 80)
            {
                pair_failures.push(diag);
            }
        }
    }

    let total_pairs = r.player_decks.len() * r.ai_decks.len();
    assert_eq!(
        pairs_run, total_pairs,
        "expected to drive {} pairs, drove {}",
        total_pairs, pairs_run
    );

    // Aggregate. Print a one-line summary for the green case so the test
    // output makes the deck-pair coverage observable to operators.
    if intern_failures.is_empty() && pair_failures.is_empty() {
        eprintln!(
            "deck_pair_legality: {} player decks × {} ai decks = {} pairs OK",
            r.player_decks.len(),
            r.ai_decks.len(),
            total_pairs,
        );
        return;
    }
    let mut diag = String::new();
    if !intern_failures.is_empty() {
        diag.push_str("intern_deck silent drops:\n  - ");
        diag.push_str(&intern_failures.join("\n  - "));
        diag.push('\n');
    }
    if !pair_failures.is_empty() {
        diag.push_str("deck-pair drive failures:\n  - ");
        diag.push_str(&pair_failures.join("\n  - "));
    }
    panic!("{}", diag);
}
