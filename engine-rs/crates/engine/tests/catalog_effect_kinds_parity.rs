//! Catalog-coverage golden test for v3.7 Channels 5 + 6 classifiers.
//!
//! Enumerates every BASE catalog card (variant expansion would
//! duplicate ability payloads via clone — see `core/catalog.rs:168-202`)
//! and asserts:
//!   1. `classify_tool_effect` / `classify_active_ability` never panic.
//!   2. Per-kind card counts match the v3.7 scope vocab exactly. If the
//!      counts drift, the scope vocab needs a fresh scoping doc — NOT
//!      a silent vocab patch.
//!   3. A sorted `(card_id, kind_idx)` golden table is printed to
//!      stdout; the Python smoke independently asserts the same counts
//!      against `shared/src/data/cards.json` so cross-language parity
//!      is captured at the count level (Phase A). Byte-level parity on
//!      a per-card table comes in Phase C.
//!
//! See `docs/ai-research/scoping/v37-combat-arith-and-catalog-scoping.md`
//! §3.4.1 + §4.5 Channels 5/6 + §13.5.

use std::collections::BTreeMap;

use engine::core::catalog::{catalog, Card, Variant};
use engine::core::effect_kinds::{
    classify_active_ability, classify_tool_effect, AbilityEffectKind, ToolEffectKind,
};

#[test]
fn catalog_effect_kinds_parity() {
    let cat = catalog();

    // Per-kind tallies (BTreeMap for stable ordering in error messages).
    let mut tool_counts: BTreeMap<u8, usize> = BTreeMap::new();
    let mut ability_counts: BTreeMap<u8, usize> = BTreeMap::new();
    let mut tool_table: Vec<(String, u8)> = Vec::new();
    let mut ability_table: Vec<(String, u8)> = Vec::new();

    for card in cat.cards.iter() {
        let variant = match card {
            Card::Umamusume(u) => u.variant,
            Card::Trainer(t) => t.variant,
        };
        if variant != Variant::Base {
            continue;
        }
        match card {
            Card::Trainer(t) => {
                // Tools only — `is_tool` proxy via classifier on the
                // effect struct works for ANY trainer, but the
                // tool-effect fields fire ONLY on tool trainers in the
                // current catalog. We still want to classify only
                // tools to avoid lumping supporter cards under `Other`
                // and inflating the bucket.
                if matches!(t.trainer_type, engine::core::constants::TrainerType::Tool) {
                    let kind = classify_tool_effect(&t.effect);
                    *tool_counts.entry(kind.as_u8()).or_insert(0) += 1;
                    tool_table.push((t.id.clone(), kind.as_u8()));
                }
            }
            Card::Umamusume(u) => {
                if let Some(ability) = u.ability.as_ref() {
                    let kind = classify_active_ability(ability);
                    *ability_counts.entry(kind.as_u8()).or_insert(0) += 1;
                    ability_table.push((u.id.clone(), kind.as_u8()));
                }
            }
        }
    }

    tool_table.sort();
    ability_table.sort();

    // Print the golden tables (cross-language reference for the
    // Python smoke; surfaced in `cargo test -- --nocapture`).
    println!("# Tool effect-kind golden table (base variants only)");
    for (id, k) in &tool_table {
        println!("{}\t{}", id, k);
    }
    println!("# Ability effect-kind golden table (base variants only)");
    for (id, k) in &ability_table {
        println!("{}\t{}", id, k);
    }

    // Expected per-kind counts (v3.7 scope §4.5 Channels 5 + 6).
    let expected_tool: BTreeMap<u8, usize> = BTreeMap::from([
        (ToolEffectKind::HealAtTurnEnd.as_u8(), 1usize),
        (ToolEffectKind::DamageReduction.as_u8(), 1),
        (ToolEffectKind::CounterDamage.as_u8(), 1),
    ]);
    let expected_ability: BTreeMap<u8, usize> = BTreeMap::from([
        (AbilityEffectKind::HealOnTurnStart.as_u8(), 1usize),
        (AbilityEffectKind::DamageReduction.as_u8(), 4),
        (AbilityEffectKind::HpBonus.as_u8(), 2),
        (AbilityEffectKind::EnergyAcceleration.as_u8(), 1),
        (AbilityEffectKind::ConditionalAttackBonus.as_u8(), 2),
        (AbilityEffectKind::DirectDamage.as_u8(), 1),
        (AbilityEffectKind::RetreatModifier.as_u8(), 3),
        (AbilityEffectKind::Other.as_u8(), 3),
    ]);

    assert_eq!(
        tool_counts, expected_tool,
        "tool effect-kind counts drifted from v3.7 scope vocab; see §4.5 Channel 5. \
         Actual: {:?} Expected: {:?}",
        tool_counts, expected_tool
    );
    assert_eq!(
        ability_counts, expected_ability,
        "ability effect-kind counts drifted from v3.7 scope vocab; see §4.5 Channel 6. \
         Actual: {:?} Expected: {:?}",
        ability_counts, expected_ability
    );

    // Catalog floor checks.
    let tool_total: usize = tool_counts.values().sum();
    let ability_total: usize = ability_counts.values().sum();
    assert_eq!(tool_total, 3, "expected exactly 3 tool cards at v3.7 scope time");
    assert_eq!(
        ability_total, 17,
        "expected exactly 17 ability cards at v3.7 scope time"
    );
}
