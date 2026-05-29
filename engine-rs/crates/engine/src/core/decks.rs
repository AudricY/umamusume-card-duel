//! Default deck loader.
//!
//! Bit-identical port of the `premadeDecks` plumbing in
//! `shared/src/gameData.ts:101-159`. Embeds
//! `shared/src/data/premadeDecks.json` via `include_str!`, returns the
//! resolved default-player and default-AI-opponent decks as interned
//! `Vec<CardId>` slices.
//!
//! Used by the golden-replay binary (Phase 1h gate) and by the upcoming
//! `sim-mcts-selfplay` CLI's `setup_ai_vs_ai_game` equivalent.

use std::sync::OnceLock;

use serde::Deserialize;

use super::card_id::CardId;
use super::catalog::catalog;

const DECKS_JSON: &str = include_str!("../../../../../shared/src/data/premadeDecks.json");

#[derive(Debug, Deserialize)]
struct PremadeDecksFile {
    #[serde(rename = "defaultPlayerDeckId")]
    default_player_deck_id: String,
    #[serde(rename = "defaultAiOpponentDeckId")]
    default_ai_opponent_deck_id: String,
    #[serde(rename = "premadeDecks")]
    premade_decks: Vec<PremadeDeckJson>,
    #[serde(rename = "aiPremadeDecks", default)]
    ai_premade_decks: Vec<PremadeDeckJson>,
}

#[derive(Debug, Deserialize)]
struct PremadeDeckJson {
    id: String,
    #[serde(rename = "cardIds")]
    card_ids: Vec<String>,
}

#[derive(Debug)]
pub struct DeckRegistry {
    pub default_player_deck_id: String,
    pub default_ai_opponent_deck_id: String,
    pub player_decks: Vec<Deck>,
    pub ai_decks: Vec<Deck>,
}

#[derive(Debug, Clone)]
pub struct Deck {
    pub id: String,
    pub card_ids: Vec<CardId>,
    /// Number of card ids in the source JSON entry, before catalog
    /// interning. Used by the deck-legality smoke
    /// (`deck_pair_legality.rs`) to detect silent drops where
    /// `intern_deck` warned-and-skipped an unknown card id. For a
    /// healthy deck this equals `card_ids.len()`.
    pub source_card_count: usize,
}

pub fn decks() -> &'static DeckRegistry {
    static R: OnceLock<DeckRegistry> = OnceLock::new();
    R.get_or_init(load)
}

fn load() -> DeckRegistry {
    let parsed: PremadeDecksFile =
        serde_json::from_str(DECKS_JSON).expect("premadeDecks.json must parse");
    let cat = catalog();
    let intern_deck = |d: PremadeDeckJson| -> Deck {
        let source_card_count = d.card_ids.len();
        let card_ids: Vec<CardId> = d
            .card_ids
            .iter()
            .filter_map(|cid| {
                cat.id_for(cid).or_else(|| {
                    eprintln!(
                        "warning: premade deck {} references unknown card {}",
                        d.id, cid
                    );
                    None
                })
            })
            .collect();
        Deck {
            id: d.id,
            card_ids,
            source_card_count,
        }
    };
    DeckRegistry {
        default_player_deck_id: parsed.default_player_deck_id,
        default_ai_opponent_deck_id: parsed.default_ai_opponent_deck_id,
        player_decks: parsed.premade_decks.into_iter().map(intern_deck).collect(),
        ai_decks: parsed
            .ai_premade_decks
            .into_iter()
            .map(intern_deck)
            .collect(),
    }
}

/// Lookup helper. Mirror of `gameData.ts:131` `getDeckListById` searching
/// the player-side decks first.
pub fn deck_by_id(deck_id: &str) -> Option<&'static [CardId]> {
    let r = decks();
    if let Some(d) = r.player_decks.iter().find(|d| d.id == deck_id) {
        return Some(&d.card_ids);
    }
    r.ai_decks
        .iter()
        .find(|d| d.id == deck_id)
        .map(|d| d.card_ids.as_slice())
}

/// All deck lists registered under `deck_id`, across both the player-side
/// (`premadeDecks`) and AI-side (`aiPremadeDecks`) registries, player-side
/// first.
///
/// The same id can appear in BOTH registries with DIFFERENT card lists —
/// e.g. `"matikanetannhauser"` is a 20-card single-uma list under
/// `premadeDecks` but a mixed list (with `matikanefukukitaru*`) under
/// `aiPremadeDecks`. `deck_by_id` only returns the player-side match, which
/// is ambiguous for the AI opponent. Callers that need to reconcile a deck
/// id against an *observed* state (belief complement) should consider every
/// candidate and pick the one consistent with the state.
pub fn deck_lists_by_id(deck_id: &str) -> Vec<&'static [CardId]> {
    let r = decks();
    let mut out: Vec<&'static [CardId]> = Vec::new();
    for d in r.player_decks.iter().filter(|d| d.id == deck_id) {
        out.push(d.card_ids.as_slice());
    }
    for d in r.ai_decks.iter().filter(|d| d.id == deck_id) {
        out.push(d.card_ids.as_slice());
    }
    out
}

/// Default player deck — first match in `premadeDecks`, with the same
/// `mihonoBourbon` / `premadeDecks[0]` cascading fallback as
/// `gameData.ts:149-152`.
pub fn default_player_deck() -> &'static [CardId] {
    let r = decks();
    if let Some(d) = r
        .player_decks
        .iter()
        .find(|d| d.id == r.default_player_deck_id)
    {
        if !d.card_ids.is_empty() {
            return &d.card_ids;
        }
    }
    if let Some(d) = r
        .ai_decks
        .iter()
        .find(|d| d.id == "mihonoBourbonNishinoFlower")
    {
        if !d.card_ids.is_empty() {
            return &d.card_ids;
        }
    }
    r.player_decks
        .first()
        .map(|d| d.card_ids.as_slice())
        .unwrap_or(&[])
}

/// Default AI-opponent deck — analogous to `gameData.ts:155-158`.
pub fn default_ai_opponent_deck() -> &'static [CardId] {
    let r = decks();
    if let Some(d) = r
        .ai_decks
        .iter()
        .find(|d| d.id == r.default_ai_opponent_deck_id)
    {
        if !d.card_ids.is_empty() {
            return &d.card_ids;
        }
    }
    if let Some(d) = r.ai_decks.iter().find(|d| d.id == "riceShowerHaruUrara") {
        if !d.card_ids.is_empty() {
            return &d.card_ids;
        }
    }
    r.ai_decks
        .first()
        .map(|d| d.card_ids.as_slice())
        .unwrap_or(&[])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_player_deck_is_20_cards() {
        let d = default_player_deck();
        assert_eq!(d.len(), 20, "default player deck should be 20 cards");
    }

    #[test]
    fn default_ai_opponent_deck_is_20_cards() {
        let d = default_ai_opponent_deck();
        assert_eq!(d.len(), 20, "default AI opponent deck should be 20 cards");
    }

    #[test]
    fn deck_registry_loads_known_default_ids() {
        let r = decks();
        assert_eq!(r.default_player_deck_id, "matikanetannhauser");
        assert_eq!(r.default_ai_opponent_deck_id, "matikanetannhauser");
    }

    #[test]
    fn deck_by_id_finds_either_side() {
        // Decks live in both `premadeDecks` (player-side) and
        // `aiPremadeDecks` (AI). Look up should find a deck regardless of
        // which list it's in.
        assert!(deck_by_id("matikanetannhauser").is_some());
        assert!(deck_by_id("riceShower").is_some());
        // `defaultAiOpponentDeckId` round-trips cleanly now that it
        // matches the AI Matikane id (P0a defaults fix, 2026-05-22). Prior
        // to the fix, the JSON shipped `"riceShowerHaruUrara"` — an id no
        // deck carries — so callers silently fell through to
        // `ai_decks.first()` via `default_ai_opponent_deck()`'s cascade.
        let r = decks();
        assert!(
            deck_by_id(&r.default_ai_opponent_deck_id).is_some(),
            "defaultAiOpponentDeckId must round-trip; current value {} has no matching deck",
            r.default_ai_opponent_deck_id,
        );
        assert!(deck_by_id("nope-not-a-real-deck-id").is_none());
    }
}
