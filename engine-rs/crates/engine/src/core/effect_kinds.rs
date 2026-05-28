//! Frozen effect-kind enums + pure classifiers for v3.7 Channels 5 + 6.
//!
//! `ToolEffectKind` (4 variants) and `AbilityEffectKind` (8 variants) are
//! locked at v3.7 scope time. Any new card whose effect-shape lands in
//! `Other` requires a fresh scoping doc (v3.7.1 or v3.8); the
//! `catalog_effect_kinds_parity.rs` integration test enforces the
//! per-kind count breakdown so silent vocab drift is caught at CI.
//!
//! Vocab + precedence are mirrored bit-for-bit in
//! `training/uma_ai/effect_kinds.py`. The two languages must agree on
//! the integer discriminant for each variant; `#[repr(u8)]` + explicit
//! discriminants make the mapping byte-stable.
//!
//! See `docs/ai-research/scoping/v37-combat-arith-and-catalog-scoping.md`
//! §3.4.1 + §4.5 Channels 5/6 + §13.5/§13.7 for the locked vocab
//! definitions and recon corrections.

use serde::{Deserialize, Serialize};

use super::effects::{Ability, TrainerEffect};

/// Tool effect-kind one-hot vocabulary (v3.7 Channel 5).
///
/// Today's catalog has 3 tools across the first 3 variants. `Other` is
/// a buffer for future tool additions that fall outside this vocab.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ToolEffectKind {
    /// `TrainerEffect.toolEndTurnHealActive: Option<i32>` set.
    /// Today: `leftoverCarrot`.
    HealAtTurnEnd = 0,
    /// `TrainerEffect.toolDamageReduction: Option<i32>` set.
    /// Today: `trainingHelmet`.
    DamageReduction = 1,
    /// `TrainerEffect.toolCounterDamage: Option<i32>` set.
    /// Today: `boxingGloves`.
    CounterDamage = 2,
    /// Catch-all for cards that don't map to the above variants.
    /// Today the tool catalog produces zero `Other` cards.
    Other = 3,
}

/// Active ability effect-kind one-hot vocabulary (v3.7 Channel 6).
///
/// Catalog inventory at v3.7 scope time: 17 ability cards across 7
/// distinct effect classes + 3 genuine one-offs in `Other`. The
/// `Other` floor of 18% is irreducible at the current vocab.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum AbilityEffectKind {
    /// `Ability.heal: Option<i32>`. Today: `matikanetannhauserStage2`.
    HealOnTurnStart = 0,
    /// `Ability.damageReduction: Option<i32>`. Today: `mihonoBourbon`
    /// line (Basic, Stage1, Stage2, Stage2Ex).
    DamageReduction = 1,
    /// `Ability.activeHpBonus: Option<i32>`. Today: `niceNature` Basic
    /// + BasicEx. Persistent stat buff, semantically distinct from
    /// healing.
    HpBonus = 2,
    /// `Ability.moveBenchedEnergyToActive: Option<_>` set. Today:
    /// `haruUraraBasic`.
    EnergyAcceleration = 3,
    /// `Ability.attackDamageBonusIfAttachedEnergy: Option<_>` OR
    /// `attackDamageBonusIfEvolvedLastTurn: Option<_>` set. Today:
    /// `agnesDigitalStage1`, `tamamoCrossStage2`.
    ConditionalAttackBonus = 4,
    /// `Ability.damageOpponent: Option<_>` set. Today:
    /// `manhattanCafeStage1`.
    DirectDamage = 5,
    /// `Ability.retreatCostZeroIfTookDamageLastTurn` OR
    /// `retreatCostZeroIfHasEnergy` set. Today: `twinTurboBasicEx`,
    /// `daiwaScarletStage1`, `daiwaScarletStage2`.
    RetreatModifier = 6,
    /// Catch-all for genuine one-offs that don't yet warrant their own
    /// variant. Today: `agnesTachyonStage1` (coin-flip card draw),
    /// `agnesTachyonStage1Ex` (disable-other-abilities aura),
    /// `symboliRudolfStage2Ex` (shuffle-discard).
    Other = 7,
}

impl ToolEffectKind {
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

impl AbilityEffectKind {
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Pure classifier: `TrainerEffect` -> `ToolEffectKind`.
///
/// Precedence (highest first; first match wins):
///   HealAtTurnEnd > DamageReduction > CounterDamage > Other.
///
/// Today's catalog has no multi-flag tools; precedence is preventative
/// against silent class drift on future cards.
pub fn classify_tool_effect(effect: &TrainerEffect) -> ToolEffectKind {
    if effect.tool_end_turn_heal_active.is_some() {
        return ToolEffectKind::HealAtTurnEnd;
    }
    if effect.tool_damage_reduction.is_some() {
        return ToolEffectKind::DamageReduction;
    }
    if effect.tool_counter_damage.is_some() {
        return ToolEffectKind::CounterDamage;
    }
    ToolEffectKind::Other
}

/// Pure classifier: `Ability` -> `AbilityEffectKind`.
///
/// Precedence (highest first; first match wins) mirrors the variant
/// declaration order:
///   HealOnTurnStart > DamageReduction > HpBonus > EnergyAcceleration
///   > ConditionalAttackBonus > DirectDamage > RetreatModifier > Other.
pub fn classify_active_ability(ability: &Ability) -> AbilityEffectKind {
    if ability.heal.is_some() {
        return AbilityEffectKind::HealOnTurnStart;
    }
    if ability.damage_reduction.is_some() {
        return AbilityEffectKind::DamageReduction;
    }
    if ability.active_hp_bonus.is_some() {
        return AbilityEffectKind::HpBonus;
    }
    if ability.move_benched_energy_to_active.is_some() {
        return AbilityEffectKind::EnergyAcceleration;
    }
    if ability.attack_damage_bonus_if_attached_energy.is_some()
        || ability.attack_damage_bonus_if_evolved_last_turn.is_some()
    {
        return AbilityEffectKind::ConditionalAttackBonus;
    }
    if ability.damage_opponent.is_some() {
        return AbilityEffectKind::DirectDamage;
    }
    if ability.retreat_cost_zero_if_took_damage_last_turn.is_some()
        || ability.retreat_cost_zero_if_has_energy.is_some()
    {
        return AbilityEffectKind::RetreatModifier;
    }
    AbilityEffectKind::Other
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_ability() -> Ability {
        Ability {
            name: String::new(),
            text: String::new(),
            heal: None,
            damage_opponent: None,
            damage_opponent_target: None,
            discard_energy: None,
            active_hp_bonus: None,
            damage_reduction: None,
            move_benched_energy_to_active: None,
            attack_damage_bonus_if_attached_energy: None,
            attack_damage_bonus_if_evolved_last_turn: None,
            disable_other_umamusume_abilities_while_active: None,
            once_per_game: None,
            shuffle_random_discard_into_deck: None,
            discard_to_draw: None,
            coin_flip_draw_or_active_damage_counter: None,
            retreat_cost_zero_if_took_damage_last_turn: None,
            retreat_cost_zero_if_has_energy: None,
        }
    }

    #[test]
    fn empty_trainer_effect_classifies_as_other() {
        let e = TrainerEffect::default();
        assert_eq!(classify_tool_effect(&e), ToolEffectKind::Other);
    }

    #[test]
    fn empty_ability_classifies_as_other() {
        let a = empty_ability();
        assert_eq!(classify_active_ability(&a), AbilityEffectKind::Other);
    }

    #[test]
    fn tool_heal_at_turn_end_classifies() {
        let mut e = TrainerEffect::default();
        e.tool_end_turn_heal_active = Some(10);
        assert_eq!(classify_tool_effect(&e), ToolEffectKind::HealAtTurnEnd);
    }

    #[test]
    fn tool_damage_reduction_classifies() {
        let mut e = TrainerEffect::default();
        e.tool_damage_reduction = Some(10);
        assert_eq!(classify_tool_effect(&e), ToolEffectKind::DamageReduction);
    }

    #[test]
    fn tool_counter_damage_classifies() {
        let mut e = TrainerEffect::default();
        e.tool_counter_damage = Some(20);
        assert_eq!(classify_tool_effect(&e), ToolEffectKind::CounterDamage);
    }

    #[test]
    fn tool_precedence_heal_beats_damage_reduction() {
        let mut e = TrainerEffect::default();
        e.tool_end_turn_heal_active = Some(10);
        e.tool_damage_reduction = Some(10);
        assert_eq!(classify_tool_effect(&e), ToolEffectKind::HealAtTurnEnd);
    }

    #[test]
    fn ability_heal_classifies() {
        let mut a = empty_ability();
        a.heal = Some(10);
        assert_eq!(
            classify_active_ability(&a),
            AbilityEffectKind::HealOnTurnStart
        );
    }

    #[test]
    fn ability_damage_reduction_classifies() {
        let mut a = empty_ability();
        a.damage_reduction = Some(10);
        assert_eq!(
            classify_active_ability(&a),
            AbilityEffectKind::DamageReduction
        );
    }

    #[test]
    fn ability_hp_bonus_classifies() {
        let mut a = empty_ability();
        a.active_hp_bonus = Some(10);
        assert_eq!(classify_active_ability(&a), AbilityEffectKind::HpBonus);
    }

    #[test]
    fn ability_energy_acceleration_classifies() {
        use crate::core::constants::EnergyType;
        use crate::core::effects::MoveEnergyTypes;
        let mut a = empty_ability();
        a.move_benched_energy_to_active = Some(MoveEnergyTypes::Many(vec![
            EnergyType::Psychic,
            EnergyType::Darkness,
        ]));
        assert_eq!(
            classify_active_ability(&a),
            AbilityEffectKind::EnergyAcceleration
        );
    }

    #[test]
    fn ability_conditional_attack_bonus_attached_energy_classifies() {
        use crate::core::constants::EnergyType;
        use crate::core::effects::AttackBonusIfAttachedEnergy;
        let mut a = empty_ability();
        a.attack_damage_bonus_if_attached_energy = Some(AttackBonusIfAttachedEnergy {
            r#type: EnergyType::Fire,
            min: 4,
            amount: 30,
        });
        assert_eq!(
            classify_active_ability(&a),
            AbilityEffectKind::ConditionalAttackBonus
        );
    }

    #[test]
    fn ability_conditional_attack_bonus_evolved_classifies() {
        let mut a = empty_ability();
        a.attack_damage_bonus_if_evolved_last_turn = Some(20);
        assert_eq!(
            classify_active_ability(&a),
            AbilityEffectKind::ConditionalAttackBonus
        );
    }

    #[test]
    fn ability_direct_damage_classifies() {
        let mut a = empty_ability();
        a.damage_opponent = Some(20);
        assert_eq!(classify_active_ability(&a), AbilityEffectKind::DirectDamage);
    }

    #[test]
    fn ability_retreat_modifier_took_damage_classifies() {
        let mut a = empty_ability();
        a.retreat_cost_zero_if_took_damage_last_turn = Some(true);
        assert_eq!(
            classify_active_ability(&a),
            AbilityEffectKind::RetreatModifier
        );
    }

    #[test]
    fn ability_retreat_modifier_has_energy_classifies() {
        let mut a = empty_ability();
        a.retreat_cost_zero_if_has_energy = Some(true);
        assert_eq!(
            classify_active_ability(&a),
            AbilityEffectKind::RetreatModifier
        );
    }

    #[test]
    fn ability_precedence_heal_beats_damage_reduction() {
        let mut a = empty_ability();
        a.heal = Some(10);
        a.damage_reduction = Some(10);
        assert_eq!(
            classify_active_ability(&a),
            AbilityEffectKind::HealOnTurnStart
        );
    }

    #[test]
    fn discriminants_are_stable() {
        // Byte-stable contract with Python `IntEnum` mirror.
        assert_eq!(ToolEffectKind::HealAtTurnEnd.as_u8(), 0);
        assert_eq!(ToolEffectKind::DamageReduction.as_u8(), 1);
        assert_eq!(ToolEffectKind::CounterDamage.as_u8(), 2);
        assert_eq!(ToolEffectKind::Other.as_u8(), 3);

        assert_eq!(AbilityEffectKind::HealOnTurnStart.as_u8(), 0);
        assert_eq!(AbilityEffectKind::DamageReduction.as_u8(), 1);
        assert_eq!(AbilityEffectKind::HpBonus.as_u8(), 2);
        assert_eq!(AbilityEffectKind::EnergyAcceleration.as_u8(), 3);
        assert_eq!(AbilityEffectKind::ConditionalAttackBonus.as_u8(), 4);
        assert_eq!(AbilityEffectKind::DirectDamage.as_u8(), 5);
        assert_eq!(AbilityEffectKind::RetreatModifier.as_u8(), 6);
        assert_eq!(AbilityEffectKind::Other.as_u8(), 7);
    }
}
