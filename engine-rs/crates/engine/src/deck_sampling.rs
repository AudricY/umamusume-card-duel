//! Deck-pair sampler for sim-cli binaries.
//!
//! Implements Slice 1 of `docs/ai-research/scoping/deck-pair-sampling.md`.
//! Parses `--deck-sampling=fixed|uniform|pair=<P>:<O>` and produces a
//! per-game `(player_deck_id, opponent_deck_id)` resolution that the
//! caller passes to `setup_ai_vs_ai_game_with_decks(...)`.
//!
//! Determinism contract:
//!   - `fixed`: returns `(None, None)` for every game so the caller falls
//!     through to the existing defaults (current behavior, byte-identical
//!     to pre-Slice-1 code paths).
//!   - `uniform`: `index = (seed_start + game_index) % (n_player_decks *
//!     n_ai_decks)`; row-major decompose to `(player_deck_index,
//!     ai_deck_index)`. Same `(seed_start, game_index)` → same pair
//!     regardless of `--workers`.
//!   - `pair=<P>:<O>`: literal pair, validates both ids at parse time;
//!     bails with a per-id diagnostic on miss.
//!
//! Per-game manifest emission is the caller's responsibility (the
//! binaries already own the JSONL writer); this module only resolves the
//! deck pair from CLI args.

use crate::core::card_id::CardId;
use crate::core::decks::{decks, Deck};

/// Parsed sampling mode (the result of parsing `--deck-sampling=...`).
#[derive(Debug, Clone)]
pub enum DeckSampling {
    /// Default — every game uses `setup_ai_vs_ai_game()` defaults.
    Fixed,
    /// Uniform over `player_decks × ai_decks`, indexed by seed.
    Uniform,
    /// Literal pair (both ids must round-trip in the deck registry).
    Pair {
        player_deck_id: String,
        ai_deck_id: String,
    },
}

/// Resolved deck-pair for a single game. `id` strings are kept for
/// manifest emission; `card_ids` slices are passed to
/// `setup_ai_vs_ai_game_with_decks(...)`.
#[derive(Debug, Clone)]
pub struct ResolvedDeckPair<'a> {
    pub player_deck_id: &'a str,
    pub player_deck: &'a [CardId],
    pub opponent_deck_id: &'a str,
    pub opponent_deck: &'a [CardId],
}

impl DeckSampling {
    /// Parse a `--deck-sampling=...` value. Accepted forms:
    ///   - `fixed` (default)
    ///   - `uniform`
    ///   - `pair=<player_deck_id>:<ai_deck_id>`
    ///
    /// Returns `Err(String)` with a CLI-friendly diagnostic on bad input
    /// (including unknown deck ids for `pair=...`).
    pub fn parse(raw: &str) -> Result<Self, String> {
        let trimmed = raw.trim();
        if trimmed.eq_ignore_ascii_case("fixed") {
            return Ok(DeckSampling::Fixed);
        }
        if trimmed.eq_ignore_ascii_case("uniform") {
            return Ok(DeckSampling::Uniform);
        }
        if let Some(rest) = trimmed.strip_prefix("pair=") {
            let parts: Vec<&str> = rest.splitn(2, ':').collect();
            if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty() {
                return Err(format!(
                    "deck-sampling pair must be in the form pair=<player>:<opponent> (got '{}')",
                    raw
                ));
            }
            let player_deck_id = parts[0].to_string();
            let ai_deck_id = parts[1].to_string();
            // Validate against the registry now so bad ids fail at arg-parse
            // time, not on the first game.
            let r = decks();
            if !r.player_decks.iter().any(|d| d.id == player_deck_id) {
                return Err(format!(
                    "deck-sampling pair player id '{}' not found in premadeDecks (known: {})",
                    player_deck_id,
                    r.player_decks
                        .iter()
                        .map(|d| d.id.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            if !r.ai_decks.iter().any(|d| d.id == ai_deck_id) {
                return Err(format!(
                    "deck-sampling pair opponent id '{}' not found in aiPremadeDecks (known: {})",
                    ai_deck_id,
                    r.ai_decks
                        .iter()
                        .map(|d| d.id.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            return Ok(DeckSampling::Pair {
                player_deck_id,
                ai_deck_id,
            });
        }
        Err(format!(
            "deck-sampling must be one of: fixed, uniform, pair=<player>:<opponent> (got '{}')",
            raw
        ))
    }

    /// Whether this sampler is anything other than `Fixed`. Used by the
    /// per-game progress emitter to know whether to populate
    /// `perMatchup` in the aggregate manifest.
    pub fn is_active(&self) -> bool {
        !matches!(self, DeckSampling::Fixed)
    }

    /// Resolve the per-game deck-pair. Inputs are the seed-relative game
    /// index (`game_index >= 0`; for a binary like `sim-eval-gate` with
    /// `--seeds N --model-side both`, this is the 0..2N task index) and
    /// the seed-base.
    ///
    /// Returns `None` for `Fixed` so the caller falls through to
    /// `setup_ai_vs_ai_game()` defaults exactly as today.
    pub fn resolve(&self, seed_start: u32, game_index: u32) -> Option<ResolvedDeckPair<'static>> {
        let r = decks();
        match self {
            DeckSampling::Fixed => None,
            DeckSampling::Uniform => {
                let n_player = r.player_decks.len();
                let n_ai = r.ai_decks.len();
                if n_player == 0 || n_ai == 0 {
                    return None;
                }
                // Row-major decompose `(seed_start + game_index) %
                // (n_player * n_ai)` into `(player_idx, ai_idx)`. Stable
                // under reordering of `(--workers, game_index)` since the
                // index is purely seed-derived.
                let total = (n_player as u64) * (n_ai as u64);
                let raw = (seed_start as u64).wrapping_add(game_index as u64);
                let combined = (raw % total) as usize;
                let player_idx = combined / n_ai;
                let ai_idx = combined % n_ai;
                let player = &r.player_decks[player_idx];
                let ai = &r.ai_decks[ai_idx];
                Some(resolved_from(player, ai))
            }
            DeckSampling::Pair {
                player_deck_id,
                ai_deck_id,
            } => {
                let player = r
                    .player_decks
                    .iter()
                    .find(|d| d.id == *player_deck_id)
                    .expect("pair player id validated at parse time");
                let ai = r
                    .ai_decks
                    .iter()
                    .find(|d| d.id == *ai_deck_id)
                    .expect("pair ai id validated at parse time");
                Some(resolved_from(player, ai))
            }
        }
    }
}

fn resolved_from(player: &'static Deck, ai: &'static Deck) -> ResolvedDeckPair<'static> {
    ResolvedDeckPair {
        player_deck_id: player.id.as_str(),
        player_deck: player.card_ids.as_slice(),
        opponent_deck_id: ai.id.as_str(),
        opponent_deck: ai.card_ids.as_slice(),
    }
}

/// Resolve the *manifest-facing* deck ids for a single game even when
/// the sampler is `Fixed`. This is what the per-game progress JSONL
/// uses: every game record carries `playerDeckId` + `opponentDeckId`
/// regardless of sampling mode, so consumers can stratify post-hoc.
///
/// For `Fixed`, returns the registry defaults
/// (`defaultPlayerDeckId` / `defaultAiOpponentDeckId`).
pub fn manifest_pair_for(sampling: &DeckSampling, seed_start: u32, game_index: u32) -> (&'static str, &'static str) {
    if let Some(pair) = sampling.resolve(seed_start, game_index) {
        return (pair.player_deck_id, pair.opponent_deck_id);
    }
    let r = decks();
    // Mirror `default_player_deck()` / `default_ai_opponent_deck()`
    // resolution, but return the *id* the caller would have used (after
    // the cascade) so the manifest is still meaningful when the default
    // id is broken.
    let player_id = r
        .player_decks
        .iter()
        .find(|d| d.id == r.default_player_deck_id)
        .or_else(|| r.player_decks.first())
        .map(|d| d.id.as_str())
        .unwrap_or("<unknown>");
    let ai_id = r
        .ai_decks
        .iter()
        .find(|d| d.id == r.default_ai_opponent_deck_id)
        .or_else(|| r.ai_decks.iter().find(|d| d.id == "riceShowerHaruUrara"))
        .or_else(|| r.ai_decks.first())
        .map(|d| d.id.as_str())
        .unwrap_or("<unknown>");
    (player_id, ai_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_fixed_uniform_pair() {
        match DeckSampling::parse("fixed").unwrap() {
            DeckSampling::Fixed => {}
            other => panic!("expected Fixed, got {:?}", other),
        }
        match DeckSampling::parse("uniform").unwrap() {
            DeckSampling::Uniform => {}
            other => panic!("expected Uniform, got {:?}", other),
        }
        let pair =
            DeckSampling::parse("pair=matikanetannhauser:riceShower").unwrap();
        match pair {
            DeckSampling::Pair {
                player_deck_id,
                ai_deck_id,
            } => {
                assert_eq!(player_deck_id, "matikanetannhauser");
                assert_eq!(ai_deck_id, "riceShower");
            }
            other => panic!("expected Pair, got {:?}", other),
        }
    }

    #[test]
    fn pair_rejects_unknown_ids() {
        assert!(DeckSampling::parse("pair=notADeck:riceShower").is_err());
        assert!(DeckSampling::parse("pair=matikanetannhauser:alsoNotADeck").is_err());
        assert!(DeckSampling::parse("pair=:").is_err());
        assert!(DeckSampling::parse("pair=matikanetannhauser").is_err());
    }

    #[test]
    fn uniform_is_deterministic_and_covers_all_pairs() {
        let s = DeckSampling::parse("uniform").unwrap();
        let r = decks();
        let n_player = r.player_decks.len();
        let n_ai = r.ai_decks.len();
        let total = n_player * n_ai;
        // Distinct game indices in [0, total) must cover every pair
        // exactly once when seed_start=0.
        let mut seen: std::collections::HashSet<(String, String)> =
            std::collections::HashSet::new();
        for i in 0..(total as u32) {
            let p = s.resolve(0, i).expect("uniform must resolve");
            seen.insert((p.player_deck_id.to_string(), p.opponent_deck_id.to_string()));
        }
        assert_eq!(
            seen.len(),
            total,
            "expected {} unique pairs in one cycle, got {}",
            total,
            seen.len()
        );
        // Determinism: same (seed_start, game_index) → same pair.
        let a = s.resolve(9000, 17).unwrap();
        let b = s.resolve(9000, 17).unwrap();
        assert_eq!(a.player_deck_id, b.player_deck_id);
        assert_eq!(a.opponent_deck_id, b.opponent_deck_id);
        // seed-shift: (seed_start + game_index) is the key, so
        // (9000, 17) == (9017, 0).
        let c = s.resolve(9017, 0).unwrap();
        assert_eq!(a.player_deck_id, c.player_deck_id);
        assert_eq!(a.opponent_deck_id, c.opponent_deck_id);
    }

    #[test]
    fn fixed_resolves_to_none_pair_is_some() {
        let fixed = DeckSampling::parse("fixed").unwrap();
        assert!(fixed.resolve(0, 0).is_none());
        let pair = DeckSampling::parse("pair=matikanetannhauser:riceShower").unwrap();
        let r = pair.resolve(0, 0).expect("pair must resolve");
        assert_eq!(r.player_deck_id, "matikanetannhauser");
        assert_eq!(r.opponent_deck_id, "riceShower");
    }

    #[test]
    fn manifest_pair_for_fixed_returns_defaults() {
        let fixed = DeckSampling::parse("fixed").unwrap();
        let (p, o) = manifest_pair_for(&fixed, 0, 0);
        // Post-P0a the AI default id round-trips cleanly.
        assert_eq!(p, "matikanetannhauser");
        assert_eq!(o, "matikanetannhauser");
    }
}
