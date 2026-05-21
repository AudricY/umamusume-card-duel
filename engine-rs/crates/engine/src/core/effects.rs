//! Typed Attack / Ability / TrainerEffect payloads.
//!
//! Bit-identical port of the `Attack`, `Ability`, and `TrainerCard.effect`
//! shapes in `shared/src/types.ts:22-156`. JSON field names match TS
//! verbatim via `serde(rename)` / `serde(rename_all = "camelCase")`.
//!
//! `EnergyCost` is `Record<EnergyType, number>` with all keys optional —
//! port as a fixed `[u8; EnergyType::COUNT + 1]` array indexed by
//! `EnergyType as usize`, with the "colorless" requirement at index
//! `EnergyType::Colorless as usize` (already present in the enum).

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use super::constants::{EnergyType, SpecialCondition, UmamusumeType};

/// Mirror of `EnergyCost = Partial<Record<EnergyRequirement, number>>`.
///
/// Stored as a fixed `[u8; 10]` indexed by `EnergyType as usize`. The
/// `colorless` field doubles as the "colorless requirement" channel
/// (TS allows it on the cost side; engine logic treats it as
/// any-energy).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(from = "EnergyCostJson", into = "EnergyCostJson")]
pub struct EnergyCost {
    pub by_type: [u8; EnergyType::COUNT],
}

impl EnergyCost {
    pub fn get(&self, t: EnergyType) -> u8 {
        self.by_type[t as usize]
    }
    pub fn iter_typed(&self) -> impl Iterator<Item = (EnergyType, u8)> + '_ {
        EnergyType::ALL
            .iter()
            .copied()
            .filter(|&t| t != EnergyType::Colorless)
            .map(move |t| (t, self.by_type[t as usize]))
            .filter(|&(_, n)| n > 0)
    }
    pub fn colorless(&self) -> u8 {
        self.by_type[EnergyType::Colorless as usize]
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct EnergyCostJson {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    grass: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    fire: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    water: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    lightning: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    psychic: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    fighting: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    darkness: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    steel: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    colorless: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    dragon: Option<u8>,
}

impl From<EnergyCostJson> for EnergyCost {
    fn from(j: EnergyCostJson) -> Self {
        let mut by = [0u8; EnergyType::COUNT];
        by[EnergyType::Grass as usize] = j.grass.unwrap_or(0);
        by[EnergyType::Fire as usize] = j.fire.unwrap_or(0);
        by[EnergyType::Water as usize] = j.water.unwrap_or(0);
        by[EnergyType::Lightning as usize] = j.lightning.unwrap_or(0);
        by[EnergyType::Psychic as usize] = j.psychic.unwrap_or(0);
        by[EnergyType::Fighting as usize] = j.fighting.unwrap_or(0);
        by[EnergyType::Darkness as usize] = j.darkness.unwrap_or(0);
        by[EnergyType::Steel as usize] = j.steel.unwrap_or(0);
        by[EnergyType::Colorless as usize] = j.colorless.unwrap_or(0);
        by[EnergyType::Dragon as usize] = j.dragon.unwrap_or(0);
        EnergyCost { by_type: by }
    }
}

impl From<EnergyCost> for EnergyCostJson {
    fn from(c: EnergyCost) -> Self {
        let some_if = |i: usize| {
            let v = c.by_type[i];
            if v == 0 {
                None
            } else {
                Some(v)
            }
        };
        EnergyCostJson {
            grass: some_if(EnergyType::Grass as usize),
            fire: some_if(EnergyType::Fire as usize),
            water: some_if(EnergyType::Water as usize),
            lightning: some_if(EnergyType::Lightning as usize),
            psychic: some_if(EnergyType::Psychic as usize),
            fighting: some_if(EnergyType::Fighting as usize),
            darkness: some_if(EnergyType::Darkness as usize),
            steel: some_if(EnergyType::Steel as usize),
            colorless: some_if(EnergyType::Colorless as usize),
            dragon: some_if(EnergyType::Dragon as usize),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AttackTarget {
    Active,
    Any,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum HealTarget {
    Self_,
    Any,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum DamagePerUmamusumeSide {
    Own,
    All,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Attack {
    pub name: String,
    #[serde(default)]
    pub cost: EnergyCost,
    pub damage: i32,
    #[serde(default)]
    pub text: String,

    // All-optional behavioral toggles below — mirror TS field-for-field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_opponent: Option<AttackTarget>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bench_damage: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coin_bonus: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draw_on_heads: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draw: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heal: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heal_target: Option<HealTarget>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub discard_energy: Option<EnergyCost>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evolve_from_deck: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recover_special_conditions: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shuffle_self_into_deck: Option<ShuffleSelfIntoDeck>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub switch_self_after_attack: Option<SwitchSelfAfterAttack>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prevent_damage_next_turn: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bonus_if_took_damage_last_turn: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub damage_per_attached_energy: Option<DamagePerAttachedEnergy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub damage_per_unique_attached_energy: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub damage_per_umamusume_in_play: Option<DamagePerUmamusumeInPlay>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attack_damage_bonus_if_tool_attached: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attack_damage_bonus_if_discard_hand_card: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attack_damage_bonus_per_discarded_hand_card: Option<AttackBonusPerDiscarded>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub discard_random_opponent_hand_on_heads: Option<DiscardRandomOnHeads>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shuffle_random_discard_into_deck: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guarantee_next_coin_flip_heads: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub knock_out_active_if_all_coin_heads: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cannot_attack_next_turn: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inflict_special_condition: Option<SpecialCondition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ShuffleSelfIntoDeck {
    #[serde(default)]
    pub discard_energy: EnergyCost,
    pub requires_bench: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SwitchSelfAfterAttack {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bonus_damage: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DamagePerAttachedEnergy {
    pub types: Vec<EnergyType>,
    pub amount: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DamagePerUmamusumeInPlay {
    pub side: DamagePerUmamusumeSide,
    pub amount: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AttackBonusPerDiscarded {
    pub max_discard: i32,
    pub bonus_per_card: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiscardRandomOnHeads {
    pub self_damage: i32,
}

/// Mirror of `shared/src/types.ts:71` `Ability`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Ability {
    pub name: String,
    #[serde(default)]
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heal: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub damage_opponent: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub damage_opponent_target: Option<AttackTarget>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub discard_energy: Option<EnergyCost>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_hp_bonus: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub damage_reduction: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub move_benched_energy_to_active: Option<MoveEnergyTypes>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attack_damage_bonus_if_attached_energy: Option<AttackBonusIfAttachedEnergy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attack_damage_bonus_if_evolved_last_turn: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disable_other_umamusume_abilities_while_active: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub once_per_game: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shuffle_random_discard_into_deck: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub discard_to_draw: Option<DiscardToDraw>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coin_flip_draw_or_active_damage_counter: Option<CoinFlipDrawOrDamage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retreat_cost_zero_if_took_damage_last_turn: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retreat_cost_zero_if_has_energy: Option<bool>,
}

/// `moveBenchedEnergyToActive` can be a single `EnergyType` or an array.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MoveEnergyTypes {
    One(EnergyType),
    Many(Vec<EnergyType>),
}

impl MoveEnergyTypes {
    pub fn as_slice(&self) -> &[EnergyType] {
        match self {
            MoveEnergyTypes::One(e) => std::slice::from_ref(e),
            MoveEnergyTypes::Many(v) => v.as_slice(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AttackBonusIfAttachedEnergy {
    pub r#type: EnergyType,
    pub min: i32,
    pub amount: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiscardToDraw {
    pub discard: i32,
    pub draw: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CoinFlipDrawOrDamage {
    pub draw: i32,
    pub damage_on_tails: i32,
}

/// Mirror of `TrainerCard.effect`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainerEffect {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draw: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heal: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heal_target: Option<TrainerHealTarget>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub discard_other_card: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_umamusume: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_evolution_umamusume: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_random_basic_umamusume: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reveal_searched_card: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retreat_cost_reduction: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gust_opponent: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_attack_damage_bonus: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_energy_attach: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attach_energy_from_zone_to_bench: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_retreat_cost_reduction: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub basic_hp_bonus: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shuffle_hand_into_deck_draw: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub random_basic_umamusume_from_discard: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rainbow_uncap_crystal: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub discard_random_opponent_active_energy: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recover_active_special_conditions: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_damage_reduction: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_counter_damage: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_end_turn_heal_active: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disable_tools: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TrainerHealTarget {
    Active,
    Any,
}

/// `Weakness` is here so the catalog parser can refer to it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Weakness {
    pub r#type: UmamusumeType,
    pub amount: i32,
}

/// Container for typed effect lookups by name, used by AI heuristics that
/// scan abilities across a side's board. Not currently populated — left as
/// a forward-declaration. (`IndexMap` to preserve catalog source order.)
#[derive(Debug, Default, Clone)]
pub struct EffectIndex {
    pub abilities: IndexMap<String, Ability>,
    pub attacks: IndexMap<String, Attack>,
}
