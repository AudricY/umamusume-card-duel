//! Runtime card catalog loader.
//!
//! Bit-identical port of `frontend/src/game/engine/core/catalog.ts` plus
//! the variant-expansion logic from `shared/src/gameData.ts` lines 52,
//! 109-128. Reads `shared/src/data/cards.json` at compile time via
//! `include_str!`, applies the same variant expansions TS does, and
//! interns every resulting card id into `CardId(u16)`.
//!
//! The full Attack / Ability / TrainerEffect shape is preserved as
//! `serde_json::Value` for now — flow modules that need a specific field
//! pull it out via typed accessors as they're ported. This keeps the
//! parser surface tight while letting the engine reference the catalog by
//! interned id from day one.

use std::sync::OnceLock;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use super::card_id::{CardId, CardIdInterner};
use super::constants::{TrainerType, UmamusumeType};
use super::effects::{Ability, Attack, TrainerEffect, Weakness};

const CARDS_JSON: &str = include_str!("../../../../../shared/src/data/cards.json");

#[derive(Debug, Deserialize)]
struct CardsDataFile {
    #[serde(rename = "weaknessBonus")]
    weakness_bonus: i32,
    #[serde(rename = "fullArtSuffix")]
    _full_art_suffix: String,
    #[serde(rename = "fullArtBaseCardIds")]
    full_art_base_card_ids: Vec<String>,
    #[serde(default, rename = "goldFullArtBaseCardIds")]
    gold_full_art_base_card_ids: Vec<String>,
    #[serde(rename = "baseCards")]
    base_cards: IndexMap<String, serde_json::Value>,
}

/// One card in the catalog, post variant-expansion. Identity-bearing
/// fields are typed; effect payloads stay as `serde_json::Value`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Card {
    Umamusume(UmamusumeCard),
    Trainer(TrainerCard),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UmamusumeCard {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub label: String,
    pub species: String,
    pub r#type: UmamusumeType,
    pub stage: u8,
    pub hp: i32,
    pub weakness: Weakness,
    pub retreat: String,
    #[serde(default)]
    pub attacks: Vec<Attack>,
    #[serde(default)]
    pub ability: Option<Ability>,
    #[serde(default, rename = "evolvesFrom")]
    pub evolves_from: Option<String>,
    /// Variant flag — set during expansion. Catalog-level only.
    #[serde(default)]
    pub variant: Variant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerCard {
    pub id: String,
    #[serde(rename = "trainerType")]
    pub trainer_type: TrainerType,
    pub name: String,
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub effect: TrainerEffect,
    #[serde(default)]
    pub variant: Variant,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Variant {
    #[default]
    Base,
    FullArt,
    FullArtGold,
    UncommonPlus,
}

#[derive(Debug)]
pub struct Catalog {
    pub weakness_bonus: i32,
    pub interner: CardIdInterner,
    /// Indexed by `CardId.0 as usize`.
    pub cards: Vec<Card>,
    /// Per-CardId fast-classify table — built once at catalog init so
    /// hot loops (`count_consumed_basics`, `get_known_remaining_deck_counts`)
    /// can avoid the full `Card` variant match + struct field load on
    /// every iteration. `count_consumed_basics` showed up at ~7%
    /// inclusive in the post-3h rollout flamegraph at sims=800.
    pub is_basic_umamusume: Vec<bool>,
}

impl Catalog {
    pub fn get(&self, id: CardId) -> Option<&Card> {
        self.cards.get(id.index())
    }

    pub fn get_by_str(&self, id: &str) -> Option<&Card> {
        let cid = self.interner.get(id)?;
        self.get(cid)
    }

    pub fn id_for(&self, id: &str) -> Option<CardId> {
        self.interner.get(id)
    }

    /// Number of catalog entries (base + all variants).
    pub fn len(&self) -> usize {
        self.cards.len()
    }
    pub fn is_empty(&self) -> bool {
        self.cards.is_empty()
    }

    /// Mirror of `core/catalog.ts:22` `isUmamusumeInDeck`.
    pub fn is_umamusume(&self, id: CardId) -> bool {
        matches!(self.get(id), Some(Card::Umamusume(_)))
    }

    /// Mirror of `core/catalog.ts:26` `isBasicUmamusumeInDeck`. O(1) via
    /// the precomputed `is_basic_umamusume` table — bypasses the
    /// `Card` variant tag-check + struct field load that drove
    /// `count_consumed_basics` up to ~7% inclusive in the post-3h
    /// rollout flamegraph.
    pub fn is_basic_umamusume(&self, id: CardId) -> bool {
        self.is_basic_umamusume
            .get(id.index())
            .copied()
            .unwrap_or(false)
    }
}

/// Singleton catalog, initialized on first call.
pub fn catalog() -> &'static Catalog {
    static C: OnceLock<Catalog> = OnceLock::new();
    C.get_or_init(load_catalog)
}

fn load_catalog() -> Catalog {
    let raw: CardsDataFile = serde_json::from_str(CARDS_JSON).expect("cards.json must parse");
    let mut expanded: IndexMap<String, Card> = IndexMap::new();

    // Base cards first, in source-declaration order.
    for (id, value) in &raw.base_cards {
        let card = parse_card(value).unwrap_or_else(|e| {
            panic!("failed to parse base card {}: {} (raw: {})", id, e, value)
        });
        expanded.insert(id.clone(), card);
    }

    // Full-art variants — clone base, set variant flag, append suffix.
    for base_id in &raw.full_art_base_card_ids {
        if let Some(base) = expanded.get(base_id).cloned() {
            let variant_id = format!("{base_id}FullArt");
            let card = with_id_and_variant(base, &variant_id, Variant::FullArt);
            expanded.insert(variant_id, card);
        }
    }

    // Gold full-art (trainer-only). Mirror of withGoldFullArtVariants.
    for base_id in &raw.gold_full_art_base_card_ids {
        if let Some(base) = expanded.get(base_id).cloned() {
            if !matches!(base, Card::Trainer(_)) {
                continue;
            }
            let variant_id = format!("{base_id}FullArtGold");
            let card = with_id_and_variant(base, &variant_id, Variant::FullArtGold);
            expanded.insert(variant_id, card);
        }
    }

    // Uncommon-plus variants — mirror of isUncommonPlusBaseCard
    // (cardRarity.ts:57 + gameData.ts:109).
    let snapshot: Vec<(String, Card)> = expanded
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    for (id, card) in snapshot {
        if !is_uncommon_plus_base(&id, &card) {
            continue;
        }
        let variant_id = format!("{id}UncommonPlus");
        let new_card = with_id_and_variant(card, &variant_id, Variant::UncommonPlus);
        expanded.insert(variant_id, new_card);
    }

    // Intern in iteration order so CardId.0 is stable run-to-run.
    let mut interner = CardIdInterner::new();
    let mut cards = Vec::with_capacity(expanded.len());
    let mut is_basic_umamusume = Vec::with_capacity(expanded.len());
    for (id, card) in expanded.into_iter() {
        let cid = interner.intern(&id);
        debug_assert_eq!(cid.index(), cards.len());
        let is_basic = matches!(&card, Card::Umamusume(u) if u.stage == 0);
        cards.push(card);
        is_basic_umamusume.push(is_basic);
    }

    Catalog {
        weakness_bonus: raw.weakness_bonus,
        interner,
        cards,
        is_basic_umamusume,
    }
}

fn parse_card(value: &serde_json::Value) -> Result<Card, serde_json::Error> {
    serde_json::from_value(value.clone())
}

fn with_id_and_variant(card: Card, new_id: &str, variant: Variant) -> Card {
    match card {
        Card::Umamusume(mut u) => {
            u.id = new_id.to_string();
            u.variant = variant;
            Card::Umamusume(u)
        }
        Card::Trainer(mut t) => {
            t.id = new_id.to_string();
            t.variant = variant;
            Card::Trainer(t)
        }
    }
}

fn is_uncommon_plus_base(id: &str, card: &Card) -> bool {
    // Mirror of cardRarity.ts:57 `isUncommonPlusBaseCard`.
    if id.ends_with("FullArt") || id.ends_with("FullArtGold") || id.ends_with("UncommonPlus") {
        return false;
    }
    if is_ex_card(id, card) {
        return false;
    }
    match card {
        Card::Umamusume(u) => u.stage == 1,
        Card::Trainer(t) => t.trainer_type == TrainerType::Tool,
    }
}

fn is_ex_card(id: &str, card: &Card) -> bool {
    // Mirror of cardRarity.ts:61 `isExCard`.
    matches!(card, Card::Umamusume(_)) && id.ends_with("Ex")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_loads_and_includes_known_cards() {
        let c = catalog();
        // Sanity: should have base 64 + variants. Floor: > 64.
        assert!(
            c.len() > 64,
            "expected variant expansion to grow catalog beyond base count, got {}",
            c.len()
        );
        // matikanetannhauserBasic is the first base card in cards.json — must exist.
        let id = c.id_for("matikanetannhauserBasic").expect("base umamusume present");
        let card = c.get(id).expect("lookup by interned id");
        match card {
            Card::Umamusume(u) => {
                assert_eq!(u.stage, 0);
                assert_eq!(u.hp, 60);
                assert_eq!(u.r#type, UmamusumeType::Psychic);
                assert_eq!(u.variant, Variant::Base);
            }
            Card::Trainer(_) => panic!("expected umamusume"),
        }
    }

    #[test]
    fn weakness_bonus_matches_json() {
        let c = catalog();
        assert_eq!(c.weakness_bonus, 20);
    }

    #[test]
    fn fullart_variants_exist_for_listed_base_ids() {
        let c = catalog();
        // "aoiKiryuin" is in fullArtBaseCardIds (cards.json:4).
        assert!(c.id_for("aoiKiryuin").is_some(), "base aoiKiryuin present");
        assert!(
            c.id_for("aoiKiryuinFullArt").is_some(),
            "fullart variant must exist"
        );
    }

    #[test]
    fn interned_ids_are_dense_and_under_u16_max() {
        let c = catalog();
        assert!(c.len() < u16::MAX as usize);
        for (idx, _card) in c.cards.iter().enumerate() {
            let cid = CardId(idx as u16);
            assert!(c.get(cid).is_some());
        }
    }

    #[test]
    fn every_umamusume_has_positive_hp_and_at_least_one_attack() {
        // Codegen-regression guard. An umamusume with hp <= 0 is
        // unplayable (instantly knocked out); zero attacks means it
        // can never deal damage. Both conditions silently break
        // gameplay and were not caught by cross-language identity
        // comparisons, since hp == 0 would propagate from TS too.
        let c = catalog();
        let mut umas = 0u32;
        for card in c.cards.iter() {
            if let Card::Umamusume(u) = card {
                umas += 1;
                assert!(
                    u.hp > 0,
                    "umamusume {} has non-positive hp {}",
                    u.id,
                    u.hp
                );
                assert!(
                    !u.attacks.is_empty(),
                    "umamusume {} has zero attacks — unplayable",
                    u.id
                );
                assert!(
                    !u.name.is_empty(),
                    "umamusume {} has empty name",
                    u.id
                );
                assert!(
                    !u.species.is_empty(),
                    "umamusume {} has empty species",
                    u.id
                );
            }
        }
        assert!(umas > 0, "catalog has zero umamusume — codegen broken");
    }

    #[test]
    fn every_trainer_card_has_nonempty_name() {
        let c = catalog();
        let mut trainers = 0u32;
        for card in c.cards.iter() {
            if let Card::Trainer(t) = card {
                trainers += 1;
                assert!(
                    !t.name.is_empty(),
                    "trainer {} has empty name",
                    t.id
                );
            }
        }
        assert!(trainers > 0, "catalog has zero trainer cards");
    }

    #[test]
    fn no_duplicate_card_ids() {
        let c = catalog();
        let mut seen: std::collections::HashSet<&str> =
            std::collections::HashSet::with_capacity(c.len());
        for card in c.cards.iter() {
            let id = match card {
                Card::Umamusume(u) => u.id.as_str(),
                Card::Trainer(t) => t.id.as_str(),
            };
            assert!(
                seen.insert(id),
                "duplicate catalog id: {} (interner would also collapse this — silent bug)",
                id
            );
        }
    }

    #[test]
    fn evolution_chains_resolve_to_a_basic_umamusume() {
        // `evolves_from` is matched by SPECIES, not card id (see
        // flow/evolution.rs:47). So for every umamusume that claims
        // an evolves_from, some basic umamusume in the catalog must
        // have that species — otherwise the evolve action is unreachable.
        let c = catalog();
        let mut species_in_catalog: std::collections::HashSet<&str> =
            std::collections::HashSet::new();
        for card in c.cards.iter() {
            if let Card::Umamusume(u) = card {
                species_in_catalog.insert(u.species.as_str());
            }
        }
        for card in c.cards.iter() {
            if let Card::Umamusume(u) = card {
                if let Some(parent_species) = u.evolves_from.as_ref() {
                    assert!(
                        species_in_catalog.contains(parent_species.as_str()),
                        "{} (stage {}) evolves from species {:?}, but no catalog card has that species — chain is broken",
                        u.id,
                        u.stage,
                        parent_species,
                    );
                }
            }
        }
    }
}
