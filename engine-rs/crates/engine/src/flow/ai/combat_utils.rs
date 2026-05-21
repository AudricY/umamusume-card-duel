//! Bit-identical port of `frontend/src/game/engine/flow/ai/combatUtils.ts`.
//!
//! Heuristic helpers for combat scoring. Float arithmetic preserved from
//! TS exactly: weights `* 1.2`, ratio thresholds, etc.
//!
//! **Iteration order**: `beforeByUid` in `getHealingGained` /
//! `getDamageDealt` (TS uses `new Map(...)` whose insertion order matches
//! `getAllUmamusume`). The Rust port uses `Vec<(uid, hp)>` with linear
//! scan — equivalent for the small board sizes (≤4 per side).

use crate::core::catalog::{catalog, Card};
use crate::core::constants::{AiDifficulty, SideId};
use crate::core::state::{GameState, SideState, UmamusumeInstance};
use crate::core::umamusume::{attached_energy_count, get_all_umamusume};
use crate::flow::energy::has_enough_energy;

use super::types::{AiCombatDecision, AttackDecision, CombatCandidate};

/// `combatUtils.ts:7` `getHealingGained`.
pub fn get_healing_gained(before_side: &SideState, after_side: &SideState) -> i32 {
    let before_by_uid: Vec<(u32, i32)> = get_all_umamusume(before_side)
        .into_iter()
        .map(|u| (u.uid, u.hp))
        .collect();
    let mut sum = 0i32;
    for u in get_all_umamusume(after_side) {
        let before_hp = before_by_uid
            .iter()
            .find(|(uid, _)| *uid == u.uid)
            .map(|(_, hp)| *hp);
        let Some(before_hp) = before_hp else {
            continue;
        };
        sum += (u.hp - before_hp).max(0);
    }
    sum
}

/// `combatUtils.ts:16` `pickCandidateByDifficulty`. RNG-driven choice for
/// the "easy"/"normal" difficulties. Returns `None` only when the
/// candidate list is empty.
pub fn pick_candidate_by_difficulty(
    candidates: &[CombatCandidate],
    ai_difficulty: AiDifficulty,
    mut random: impl FnMut() -> f64,
    immediate_ko_threat: bool,
) -> Option<CombatCandidate> {
    if candidates.is_empty() {
        return None;
    }
    let mut sorted: Vec<CombatCandidate> = candidates.to_vec();
    sorted.sort_by(compare_candidates);

    if ai_difficulty == AiDifficulty::Hard {
        return sorted.into_iter().next();
    }

    if ai_difficulty == AiDifficulty::Easy {
        let easy_pool: Vec<CombatCandidate> = if immediate_ko_threat {
            sorted.clone()
        } else {
            sorted
                .iter()
                .filter(|c| match &c.decision {
                    AiCombatDecision::Attack(a) => a.retreat_target_uid.is_none(),
                    AiCombatDecision::EndTurn => true,
                })
                .cloned()
                .collect()
        };
        if !easy_pool.is_empty() && random() < 0.35 {
            let r = random();
            let idx = (r * easy_pool.len() as f64).floor() as usize;
            return Some(
                easy_pool
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| easy_pool[0].clone()),
            );
        }
        return pick_normal(&sorted, random);
    }

    pick_normal(&sorted, random)
}

fn pick_normal(
    sorted_candidates: &[CombatCandidate],
    mut random: impl FnMut() -> f64,
) -> Option<CombatCandidate> {
    let first = sorted_candidates.first()?;
    let Some(second) = sorted_candidates.get(1) else {
        return Some(first.clone());
    };
    Some(if random() < 0.85 {
        first.clone()
    } else {
        second.clone()
    })
}

/// `combatUtils.ts:45` `compareCandidates`.
///
/// Comparison order:
/// 1. Higher score wins.
/// 2. `lethalTarget` true wins.
/// 3. Higher `targetValue` wins.
/// 4. `targetIsActive` true wins.
/// 5. `attack` over `endTurn`.
/// 6. Lexicographic id (`localeCompare`).
pub fn compare_candidates(a: &CombatCandidate, b: &CombatCandidate) -> std::cmp::Ordering {
    if b.score != a.score {
        return b
            .score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal);
    }
    if a.lethal_target != b.lethal_target {
        return if a.lethal_target {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Greater
        };
    }
    if b.target_value != a.target_value {
        return b
            .target_value
            .partial_cmp(&a.target_value)
            .unwrap_or(std::cmp::Ordering::Equal);
    }
    if a.target_is_active != b.target_is_active {
        return if a.target_is_active {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Greater
        };
    }
    let a_end = matches!(a.decision, AiCombatDecision::EndTurn);
    let b_end = matches!(b.decision, AiCombatDecision::EndTurn);
    if a_end && !b_end {
        return std::cmp::Ordering::Greater;
    }
    if b_end && !a_end {
        return std::cmp::Ordering::Less;
    }
    a.id.cmp(&b.id)
}

/// `combatUtils.ts:55` `getDamageDealt`.
///
/// Sum HP losses on still-present umamusume plus the full HP of those that
/// were removed entirely. Iteration matches TS source.
pub fn get_damage_dealt(before_side: &SideState, after_side: &SideState) -> i32 {
    let before_by_uid: Vec<(u32, i32, i32)> = get_all_umamusume(before_side)
        .into_iter()
        .map(|u| (u.uid, u.hp, u.max_hp))
        .collect();

    let mut sum = 0i32;
    for u in get_all_umamusume(after_side) {
        if let Some((_, before_hp, _)) = before_by_uid.iter().find(|(uid, _, _)| *uid == u.uid) {
            sum += (before_hp - u.hp).max(0);
        }
    }
    for u in get_all_umamusume(before_side) {
        let still_exists = get_all_umamusume(after_side)
            .into_iter()
            .any(|e| e.uid == u.uid);
        if !still_exists {
            sum += u.hp.max(0);
        }
    }
    sum
}

/// `combatUtils.ts:67` `getTargetValue`.
pub fn get_target_value(defender: &SideState, target_uid: Option<u32>) -> f64 {
    let target: Option<&UmamusumeInstance> = match target_uid {
        Some(uid) => get_all_umamusume(defender)
            .into_iter()
            .find(|u| u.uid == uid),
        None => defender.active.as_ref(),
    };
    let Some(target) = target else {
        return 0.0;
    };
    let cat = catalog();
    let damage = match cat.get(target.card_id) {
        Some(Card::Umamusume(u)) => u
            .attacks
            .first()
            .map(|a| a.damage as f64)
            .unwrap_or(0.0),
        _ => 0.0,
    };
    damage + (attached_energy_count(target) as f64) * 12.0 + (target.stage as f64) * 18.0
}

/// `combatUtils.ts:76` `didCandidateKoTarget`.
pub fn did_candidate_ko_target(
    before_defender: &SideState,
    after_defender: &SideState,
    target_uid: Option<u32>,
) -> bool {
    let effective_uid = target_uid.or_else(|| before_defender.active.as_ref().map(|a| a.uid));
    let Some(effective_uid) = effective_uid else {
        return false;
    };
    let alive = get_all_umamusume(after_defender)
        .into_iter()
        .any(|u| u.uid == effective_uid && u.hp > 0);
    !alive
}

/// `combatUtils.ts:83` `canImmediateOpponentKo` (risk = expected).
pub fn can_immediate_opponent_ko(state: &GameState, side_id: SideId) -> bool {
    can_immediate_opponent_ko_by_risk(state, side_id, RiskMode::Expected)
}

/// `combatUtils.ts:87` `canImmediateOpponentKoConservative` (risk = max).
pub fn can_immediate_opponent_ko_conservative(state: &GameState, side_id: SideId) -> bool {
    can_immediate_opponent_ko_by_risk(state, side_id, RiskMode::Max)
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum RiskMode {
    Min,
    Expected,
    Max,
}

fn can_immediate_opponent_ko_by_risk(
    state: &GameState,
    side_id: SideId,
    risk_mode: RiskMode,
) -> bool {
    let side = state.side(side_id);
    let opponent = state.side(side_id.opposite());
    let Some(active) = &side.active else {
        return false;
    };
    let Some(attacker) = &opponent.active else {
        return false;
    };
    let cat = catalog();
    let attack = match cat.get(attacker.card_id) {
        Some(Card::Umamusume(u)) => match u.attacks.first() {
            Some(a) => a,
            None => return false,
        },
        _ => return false,
    };
    if !has_enough_energy(attacker, &attack.cost) {
        return false;
    }
    let own_in_play_count = 1 + opponent.bench.len() as i32;
    let all_in_play_count = own_in_play_count + 1 + side.bench.len() as i32;
    let predicted = predict_attack_damage_with_risk(
        attacker,
        active,
        opponent.active_attack_damage_bonus as i32,
        own_in_play_count,
        all_in_play_count,
        Some(state.turn_number),
        risk_mode,
        are_tool_effects_disabled(state),
    );
    predicted >= active.hp
}

/// `combatUtils.ts:104` `predictAttackDamage`.
pub fn predict_attack_damage(
    attacker: &UmamusumeInstance,
    defender: &UmamusumeInstance,
    bonus_damage: i32,
    own_in_play_count: i32,
    all_in_play_count: i32,
    turn_number: Option<u32>,
    tools_disabled: bool,
) -> i32 {
    let cat = catalog();
    let attacker_card = match cat.get(attacker.card_id) {
        Some(Card::Umamusume(u)) => u,
        _ => return 0,
    };
    let attack = match attacker_card.attacks.first() {
        Some(a) => a,
        None => return 0,
    };
    let defender_card = match cat.get(defender.card_id) {
        Some(Card::Umamusume(u)) => u,
        _ => return 0,
    };
    let mut damage: i32 = attack.damage + bonus_damage;
    if let Some(bonus) = attack.bonus_if_took_damage_last_turn {
        if attacker.took_damage_last_turn {
            damage += bonus;
        }
    }
    if let Some(d) = &attack.damage_per_attached_energy {
        let bonus_energy_count: i32 = d
            .types
            .iter()
            .map(|t| attacker.energies[*t as usize] as i32)
            .sum();
        damage += bonus_energy_count * d.amount;
    }
    if let Some(per_unique) = attack.damage_per_unique_attached_energy {
        let unique_energy_count = attacker
            .energies
            .iter()
            .filter(|&&c| c > 0)
            .count() as i32;
        damage += unique_energy_count * per_unique;
    }
    if let Some(d) = &attack.damage_per_umamusume_in_play {
        use crate::core::effects::DamagePerUmamusumeSide;
        let count = match d.side {
            DamagePerUmamusumeSide::All => all_in_play_count,
            DamagePerUmamusumeSide::Own => own_in_play_count,
        };
        damage += count * d.amount;
    }
    if let Some(bonus) = attack.attack_damage_bonus_if_tool_attached {
        if attacker.tool_card_id.is_some() && !tools_disabled {
            damage += bonus;
        }
    }
    if let Some(per_discard) = &attack.attack_damage_bonus_per_discarded_hand_card {
        damage += per_discard.max_discard * per_discard.bonus_per_card;
    }
    if let Some(ability) = &attacker_card.ability {
        if let Some(cond) = &ability.attack_damage_bonus_if_attached_energy {
            if (attacker.energies[cond.r#type as usize] as i32) >= cond.min {
                damage += cond.amount;
            }
        }
        let evolved_last_turn_bonus = ability.attack_damage_bonus_if_evolved_last_turn.unwrap_or(0);
        if evolved_last_turn_bonus > 0 {
            if let Some(turn_number) = turn_number {
                let target = turn_number.checked_sub(1);
                if attacker.evolved_turn == target {
                    damage += evolved_last_turn_bonus;
                }
            }
        }
    }
    if let Some(coin_bonus) = attack.coin_bonus {
        damage += coin_bonus / 2;
    }
    if let Some(amount) = attack.knock_out_active_if_all_coin_heads {
        // Math.floor(defender.hp / Math.pow(2, amount))
        let divisor = 2i32.saturating_pow(amount.max(0) as u32);
        if divisor > 0 {
            damage += defender.hp / divisor;
        }
    }
    if defender_card.weakness.r#type == attacker_card.r#type {
        damage += defender_card.weakness.amount;
    }
    damage.max(0)
}

/// `combatUtils.ts:141` `predictAttackDamageWithRisk`.
#[allow(clippy::too_many_arguments)]
pub fn predict_attack_damage_with_risk(
    attacker: &UmamusumeInstance,
    defender: &UmamusumeInstance,
    bonus_damage: i32,
    own_in_play_count: i32,
    all_in_play_count: i32,
    turn_number: Option<u32>,
    risk_mode: RiskMode,
    tools_disabled: bool,
) -> i32 {
    let base = predict_attack_damage(
        attacker,
        defender,
        bonus_damage,
        own_in_play_count,
        all_in_play_count,
        turn_number,
        tools_disabled,
    );
    let cat = catalog();
    let attack = match cat.get(attacker.card_id) {
        Some(Card::Umamusume(u)) => match u.attacks.first() {
            Some(a) => a,
            None => return base.max(0),
        },
        _ => return base.max(0),
    };
    let mut adjustment: i32 = 0;
    if let Some(coin_bonus) = attack.coin_bonus {
        // TS: Math.ceil(coinBonus / 2) for max; Math.floor for min.
        match risk_mode {
            RiskMode::Max => {
                adjustment += (coin_bonus + 1) / 2;
            }
            RiskMode::Min => {
                adjustment -= coin_bonus / 2;
            }
            RiskMode::Expected => {}
        }
    }
    if let Some(amount) = attack.knock_out_active_if_all_coin_heads {
        let divisor = 2i32.saturating_pow(amount.max(0) as u32);
        let ko_chance_damage_proxy = if divisor > 0 {
            defender.hp / divisor
        } else {
            0
        };
        match risk_mode {
            RiskMode::Max => adjustment += (defender.hp - base).max(0),
            RiskMode::Min => adjustment -= ko_chance_damage_proxy,
            RiskMode::Expected => {}
        }
    }
    (base + adjustment).max(0)
}

/// `combatUtils.ts:166` `areToolEffectsDisabled`.
pub fn are_tool_effects_disabled(state: &GameState) -> bool {
    let Some(stadium) = &state.stadium else {
        return false;
    };
    let cat = catalog();
    match cat.get(stadium.card_id) {
        Some(Card::Trainer(t)) => t.effect.disable_tools == Some(true),
        _ => false,
    }
}

/// `combatUtils.ts:172` `countDiscardedUmamusume`.
pub fn count_discarded_umamusume(
    card_ids: impl IntoIterator<Item = crate::core::card_id::CardId>,
) -> i32 {
    let cat = catalog();
    let mut count = 0i32;
    for cid in card_ids {
        if let Some(Card::Umamusume(_)) = cat.get(cid) {
            count += 1;
        }
    }
    count
}

/// `combatUtils.ts:176` `buildAttackDecision`. All optional choice fields
/// default to `None`.
#[allow(clippy::too_many_arguments)]
pub fn build_attack_decision(
    attack_index: usize,
    retreat_target_uid: Option<u32>,
    attack_target_uid: Option<u32>,
    heal_target_uid: Option<u32>,
    uses_coin_flip: bool,
    discard_hand_index: Option<usize>,
    evolution_deck_card_index: Option<usize>,
    random_discard_index: Option<usize>,
    switch_target_uid: Option<u32>,
    use_shuffle_self_into_deck: Option<bool>,
) -> AttackDecision {
    AttackDecision {
        retreat_target_uid,
        attack_target_uid,
        heal_target_uid,
        attack_index,
        uses_coin_flip,
        discard_hand_index,
        evolution_deck_card_index,
        random_discard_index,
        switch_target_uid,
        use_shuffle_self_into_deck,
    }
}
