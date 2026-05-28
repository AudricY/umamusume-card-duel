//! Bit-identical port of `frontend/src/game/engine/flow/ai/abilityUtils.ts`.
//!
//! Heuristic dispatchers for the three umamusume ability shapes:
//! `moveBenchedEnergyToActive`, `damageOpponent`, and
//! `coinFlipDrawOrActiveDamageCounter`.
//!
//! Float math: `* 1.2` weight in `aiUseMoveBenchedEnergyAbility`. Preserve
//! TS source-text exactly.

use crate::core::catalog::{catalog, Card};
use crate::core::constants::{AiDifficulty, CoinFlipResult, EnergyType, SideId, MAX_HAND};
use crate::core::state::{GameState, SideState, UmamusumeInstance};
use crate::core::umamusume::{attached_energy_count, get_all_umamusume};
use crate::flow::combat::{knock_out_umamusume, perform_attack, CombatDeps};
use crate::flow::eligibility::can_attack;
use crate::flow::energy::{get_ability_move_energy_types, has_enough_energy};
use crate::flow::turn::draw_cards;

use super::combat_utils::{can_immediate_opponent_ko, get_damage_dealt};

/// Function table for the move-energy heuristic. Mirrors TS
/// `AbilityHeuristicDeps`.
pub struct AbilityHeuristicDeps<'a> {
    pub estimate_attack_damage_output:
        &'a dyn Fn(&GameState, SideId, &UmamusumeInstance, &UmamusumeInstance) -> f64,
    pub with_energy_shift: &'a dyn Fn(&UmamusumeInstance, EnergyType, i32) -> UmamusumeInstance,
    pub mark_ability_used: &'a dyn Fn(&mut SideState, &UmamusumeInstance, &str),
}

/// `abilityUtils.ts:30` `aiUseMoveBenchedEnergyAbility`.
pub fn ai_use_move_benched_energy_ability(
    state: &GameState,
    side: &mut SideState,
    ability_umamusume_uid: u32,
    ai_difficulty: AiDifficulty,
    heuristic_deps: &AbilityHeuristicDeps<'_>,
) -> bool {
    let cat = catalog();
    // Snapshot the ability info before borrowing mutably.
    let (ability_name, wanted_energy_types) = {
        let owner = match find_in_side(side, ability_umamusume_uid) {
            Some(o) => o,
            None => return false,
        };
        let card = match cat.get(owner.card_id) {
            Some(Card::Umamusume(c)) => c,
            _ => return false,
        };
        let ability = match &card.ability {
            Some(a) => a,
            None => return false,
        };
        if ability.move_benched_energy_to_active.is_none() {
            return false;
        }
        (
            ability.name.clone(),
            get_ability_move_energy_types(Some(ability)),
        )
    };
    if side.active.is_none() {
        return false;
    }
    let active = side.active.as_ref().unwrap().clone();
    let active_card = match cat.get(active.card_id) {
        Some(Card::Umamusume(c)) => c.clone(),
        _ => return false,
    };
    let active_attack = match active_card.attacks.first() {
        Some(a) => a.clone(),
        None => return false,
    };

    let opposite_side_active_present = state.side(side.id.opposite()).active.is_some();

    let before_active_damage =
        (heuristic_deps.estimate_attack_damage_output)(state, side.id, &active, &active);
    let before_can_attack = has_enough_energy(&active, &active_attack.cost);

    // Build candidates in TS iteration order: bench × wanted-energy-types.
    let mut candidates: Vec<(u32, EnergyType, f64)> = Vec::new();
    for source in side.bench.iter() {
        for &energy_type in wanted_energy_types.iter() {
            if source.energies[energy_type as usize] == 0 {
                continue;
            }
            let simulated_active = (heuristic_deps.with_energy_shift)(&active, energy_type, 1);
            let simulated_source = (heuristic_deps.with_energy_shift)(source, energy_type, -1);
            let after_active_damage = (heuristic_deps.estimate_attack_damage_output)(
                state,
                side.id,
                &simulated_active,
                &active,
            );
            let after_can_attack = has_enough_energy(&simulated_active, &active_attack.cost);
            let source_before_damage =
                (heuristic_deps.estimate_attack_damage_output)(state, side.id, source, source);
            let source_after_damage = (heuristic_deps.estimate_attack_damage_output)(
                state,
                side.id,
                &simulated_source,
                source,
            );
            let mut score = 0.0f64;
            if after_can_attack && !before_can_attack {
                score += 220.0;
            }
            score += (after_active_damage - before_active_damage).max(0.0) * 3.0;
            if let Some(scale) = &active_attack.damage_per_attached_energy {
                if scale.types.iter().any(|&t| t == energy_type) {
                    score += 18.0;
                }
            }
            if let Some(threshold) = active_card
                .ability
                .as_ref()
                .and_then(|a| a.attack_damage_bonus_if_attached_energy.as_ref())
            {
                if threshold.r#type == energy_type
                    && (active.energies[energy_type as usize] as i32) < threshold.min
                    && (simulated_active.energies[energy_type as usize] as i32) >= threshold.min
                {
                    score += 60.0;
                }
            }
            let active_damage_gain = (after_active_damage - before_active_damage).max(0.0);
            if source_before_damage > source_after_damage && active_damage_gain <= 0.0 {
                score -= (source_before_damage - source_after_damage) * 1.2;
            }
            let source_card = match cat.get(source.card_id) {
                Some(Card::Umamusume(c)) => c,
                _ => continue,
            };
            let Some(source_attack) = source_card.attacks.first() else {
                continue;
            };
            if has_enough_energy(source, &source_attack.cost)
                && !has_enough_energy(&simulated_source, &source_attack.cost)
            {
                score -= if active_damage_gain > 0.0 { 8.0 } else { 60.0 };
            }
            if !opposite_side_active_present {
                score -= 20.0;
            }
            candidates.push((source.uid, energy_type, score));
        }
    }
    if candidates.is_empty() {
        return false;
    }
    // Sort by score desc — stable sort preserves enumeration order on ties
    // (matches JS Array.prototype.sort which is stable on V8 ≥ 7.0).
    candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    let best = candidates[0];
    if ai_difficulty == AiDifficulty::Hard && best.2 <= 20.0 {
        return false;
    }
    if ai_difficulty != AiDifficulty::Easy && best.2 <= 0.0 {
        return false;
    }

    // Apply mutation: decrement source bench energy, increment active energy.
    for u in side.bench.iter_mut() {
        if u.uid == best.0 {
            u.energies[best.1 as usize] = u.energies[best.1 as usize].saturating_sub(1);
            break;
        }
    }
    if let Some(active_mut) = side.active.as_mut() {
        active_mut.energies[best.1 as usize] =
            active_mut.energies[best.1 as usize].saturating_add(1);
    }
    // mark_ability_used wants the owner instance; capture a clone first.
    let owner_clone = find_in_side(side, ability_umamusume_uid).cloned();
    if let Some(owner) = owner_clone {
        (heuristic_deps.mark_ability_used)(side, &owner, &ability_name);
    }
    true
}

/// `abilityUtils.ts:87` `aiUseDamageAbility`.
pub fn ai_use_damage_ability(
    state: &mut GameState,
    side_id: SideId,
    ability_umamusume_uid: u32,
    deps: &mut CombatDeps<'_>,
) -> bool {
    let cat = catalog();
    let opponent_id = side_id.opposite();
    let (ability_name, damage_amount, target_kind, discard_energy) = {
        let side = state.side(side_id);
        let owner = match find_in_side(side, ability_umamusume_uid) {
            Some(o) => o,
            None => return false,
        };
        let card = match cat.get(owner.card_id) {
            Some(Card::Umamusume(c)) => c,
            _ => return false,
        };
        let ability = match &card.ability {
            Some(a) => a,
            None => return false,
        };
        let damage = match ability.damage_opponent {
            Some(d) => d,
            None => return false,
        };
        if let Some(cost) = &ability.discard_energy {
            // Check we can pay.
            let can_pay = EnergyType::ALL
                .iter()
                .all(|&t| (owner.energies[t as usize] as i32) >= cost.get(t) as i32);
            if !can_pay {
                return false;
            }
        }
        let target_kind = ability.damage_opponent_target;
        (
            ability.name.clone(),
            damage,
            target_kind,
            ability.discard_energy.clone(),
        )
    };

    // Build target list.
    let opponent = state.side(opponent_id);
    use crate::core::effects::AttackTarget;
    let potential_targets: Vec<UmamusumeInstance> = match target_kind {
        Some(AttackTarget::Any) => get_all_umamusume(opponent).into_iter().cloned().collect(),
        _ => opponent.active.as_ref().cloned().into_iter().collect(),
    };
    if potential_targets.is_empty() {
        return false;
    }

    let best_attack_damage_without_ability =
        estimate_best_attack_total_damage(state, side_id, deps);

    // Sort targets: lethal first, then higher "value" first.
    let mut sorted: Vec<UmamusumeInstance> = potential_targets;
    sorted.sort_by(|left, right| {
        let left_lethal = if left.hp <= damage_amount { 1 } else { 0 };
        let right_lethal = if right.hp <= damage_amount { 1 } else { 0 };
        if right_lethal != left_lethal {
            return right_lethal.cmp(&left_lethal);
        }
        let left_value = umamusume_value_for_ability_sort(left);
        let right_value = umamusume_value_for_ability_sort(right);
        right_value.cmp(&left_value)
    });
    let target = match sorted.first() {
        Some(t) => t.clone(),
        None => return false,
    };
    let direct_ability_damage = target.hp.min(damage_amount);

    // Simulate paying the discard energy and then estimate best attack after.
    let mut with_ability_state = state.clone();
    let simulated_ability_owner_present = {
        let simulated_side = with_ability_state.side_mut(side_id);
        let mut found = false;
        if let Some(active) = simulated_side.active.as_mut() {
            if active.uid == ability_umamusume_uid {
                if let Some(cost) = &discard_energy {
                    for t in EnergyType::ALL {
                        let amount = cost.get(t) as i32;
                        let cur = active.energies[t as usize] as i32;
                        active.energies[t as usize] = (cur - amount).max(0) as u16;
                    }
                }
                found = true;
            }
        }
        if !found {
            for u in simulated_side.bench.iter_mut() {
                if u.uid == ability_umamusume_uid {
                    if let Some(cost) = &discard_energy {
                        for t in EnergyType::ALL {
                            let amount = cost.get(t) as i32;
                            let cur = u.energies[t as usize] as i32;
                            u.energies[t as usize] = (cur - amount).max(0) as u16;
                        }
                    }
                    found = true;
                    break;
                }
            }
        }
        found
    };
    if !simulated_ability_owner_present {
        return false;
    }
    let best_attack_damage_after_ability_cost =
        estimate_best_attack_total_damage(&mut with_ability_state, side_id, deps);
    let total_damage_with_ability =
        direct_ability_damage as f64 + best_attack_damage_after_ability_cost;
    let total_damage_without_ability = best_attack_damage_without_ability;
    if total_damage_with_ability < total_damage_without_ability {
        return false;
    }

    // Apply the discard energy + damage on the real state.
    if let Some(cost) = &discard_energy {
        let side = state.side_mut(side_id);
        let owner = match find_in_side_mut(side, ability_umamusume_uid) {
            Some(o) => o,
            None => return false,
        };
        for t in EnergyType::ALL {
            let amount = cost.get(t) as i32;
            let cur = owner.energies[t as usize] as i32;
            owner.energies[t as usize] = (cur - amount).max(0) as u16;
        }
    }

    let target_uid = target.uid;
    let target_inst_snapshot: UmamusumeInstance;
    {
        let opp_side = state.side_mut(opponent_id);
        let target_ref = find_in_side_mut(opp_side, target_uid);
        let Some(target_ref) = target_ref else {
            return false;
        };
        target_ref.hp = (target_ref.hp - damage_amount).max(0);
        target_ref.took_damage_this_turn = damage_amount > 0;
        target_inst_snapshot = target_ref.clone();
    }
    {
        let side = state.side_mut(side_id);
        let owner_clone = find_in_side(side, ability_umamusume_uid).cloned();
        if let Some(owner) = owner_clone {
            heuristic_mark_ability_used(side, &owner, &ability_name);
        }
    }
    if target_inst_snapshot.hp <= 0 {
        let knocked_out = knock_out_umamusume(
            state,
            side_id,
            opponent_id,
            &target_inst_snapshot,
            deps.choose_preferred_active_index,
        );
        if knocked_out && !state.game_over {
            (deps.refresh_continuous_effects)(state);
        }
    }
    true
}

fn umamusume_value_for_ability_sort(u: &UmamusumeInstance) -> i32 {
    let cat = catalog();
    let damage = match cat.get(u.card_id) {
        Some(Card::Umamusume(c)) => c.attacks.first().map(|a| a.damage).unwrap_or(0),
        _ => 0,
    };
    damage + (attached_energy_count(u) as i32) * 12 + (u.stage as i32) * 18
}

/// `abilityUtils.ts:152` `aiUseCoinFlipDrawAbility`.
pub fn ai_use_coin_flip_draw_ability(
    state: &mut GameState,
    side_id: SideId,
    ability_umamusume_uid: u32,
    random: &mut dyn FnMut() -> f64,
    deps: &mut CombatDeps<'_>,
    ai_difficulty: AiDifficulty,
) -> bool {
    let cat = catalog();
    let (ability_name, damage_on_tails, draw_amount) = {
        let side = state.side(side_id);
        let owner = match find_in_side(side, ability_umamusume_uid) {
            Some(o) => o,
            None => return false,
        };
        let card = match cat.get(owner.card_id) {
            Some(Card::Umamusume(c)) => c,
            _ => return false,
        };
        let ability = match &card.ability {
            Some(a) => a,
            None => return false,
        };
        let coin = match &ability.coin_flip_draw_or_active_damage_counter {
            Some(c) => c,
            None => return false,
        };
        if side.active.is_none() {
            return false;
        }
        (ability.name.clone(), coin.damage_on_tails, coin.draw)
    };

    let active_uid = state.side(side_id).active.as_ref().unwrap().uid;
    let _ = active_uid;
    if state.side(side_id).hand.len() >= MAX_HAND {
        return false;
    }
    // Simulate the tails branch: would the opponent be able to KO the active
    // after the self-damage?
    let newly_ko_threatened_by_tails = {
        let mut simulated_after_tails = state.clone();
        if let Some(active) = simulated_after_tails.side_mut(side_id).active.as_mut() {
            active.hp = (active.hp - damage_on_tails).max(0);
        }
        can_immediate_opponent_ko(&simulated_after_tails, side_id)
            && !can_immediate_opponent_ko(state, side_id)
    };
    if ai_difficulty == AiDifficulty::Hard && newly_ko_threatened_by_tails {
        return false;
    }
    if ai_difficulty == AiDifficulty::Hard && state.side(side_id).hand.len() >= 6 {
        return false;
    }

    let heads = flip_coin(state.side_mut(side_id), random) == CoinFlipResult::Heads;
    // mark ability used
    {
        let side = state.side_mut(side_id);
        let owner_clone = find_in_side(side, ability_umamusume_uid).cloned();
        if let Some(owner) = owner_clone {
            heuristic_mark_ability_used(side, &owner, &ability_name);
        }
    }
    if heads {
        let side = state.side_mut(side_id);
        let _ = draw_cards(side, draw_amount as u32);
        return true;
    }

    // Tails: apply damage to own active.
    let active_snapshot: UmamusumeInstance;
    {
        let side = state.side_mut(side_id);
        let Some(active) = side.active.as_mut() else {
            return true;
        };
        active.hp = (active.hp - damage_on_tails).max(0);
        active.took_damage_this_turn = true;
        active_snapshot = active.clone();
    }
    if active_snapshot.hp <= 0 {
        let scoring_side_id: SideId = side_id.opposite();
        let knocked_out = knock_out_umamusume(
            state,
            scoring_side_id,
            side_id,
            &active_snapshot,
            deps.choose_preferred_active_index,
        );
        if knocked_out && !state.game_over {
            (deps.refresh_continuous_effects)(state);
        }
    }
    true
}

fn flip_coin(side: &mut SideState, random: &mut dyn FnMut() -> f64) -> CoinFlipResult {
    if side.guaranteed_coin_flip_heads > 0 {
        side.guaranteed_coin_flip_heads -= 1;
        return CoinFlipResult::Heads;
    }
    if random() >= 0.5 {
        CoinFlipResult::Heads
    } else {
        CoinFlipResult::Tails
    }
}

/// `abilityUtils.ts:204` `estimateBestAttackTotalDamage`.
pub fn estimate_best_attack_total_damage(
    state: &mut GameState,
    acting_side_id: SideId,
    deps: &mut CombatDeps<'_>,
) -> f64 {
    if !can_attack(state, state.side(acting_side_id)) {
        return 0.0;
    }
    let side = state.side(acting_side_id);
    let Some(active) = &side.active else {
        return 0.0;
    };
    let cat = catalog();
    let attack = match cat.get(active.card_id) {
        Some(Card::Umamusume(u)) => match u.attacks.first() {
            Some(a) => a.clone(),
            None => return 0.0,
        },
        _ => return 0.0,
    };
    let defending_id: SideId = acting_side_id.opposite();
    let defender = state.side(defending_id);
    use crate::core::effects::{AttackTarget, HealTarget};
    let attack_targets: Vec<Option<u32>> = match attack.target_opponent {
        Some(AttackTarget::Any) => {
            let v: Vec<Option<u32>> = get_all_umamusume(defender)
                .into_iter()
                .map(|u| Some(u.uid))
                .collect();
            if v.is_empty() {
                vec![None]
            } else {
                v
            }
        }
        _ => vec![None],
    };
    let heal_targets: Vec<Option<u32>> =
        if let (Some(_), Some(HealTarget::Any)) = (attack.heal, attack.heal_target) {
            let v: Vec<Option<u32>> = get_all_umamusume(side)
                .into_iter()
                .filter(|u| u.hp < u.max_hp)
                .map(|u| Some(u.uid))
                .collect();
            if v.is_empty() {
                vec![None]
            } else {
                v
            }
        } else {
            vec![None]
        };
    let mut best_damage: f64 = 0.0;
    for &attack_target_uid in attack_targets.iter() {
        for &heal_target_uid in heal_targets.iter() {
            if attack.coin_bonus.is_some() || attack.draw_on_heads.is_some() {
                let mut heads_state = state.clone();
                let mut tails_state = state.clone();
                let heads_before = state.clone();
                let tails_before = state.clone();
                perform_attack(
                    &mut heads_state,
                    acting_side_id,
                    deps,
                    attack_target_uid,
                    heal_target_uid,
                    Some(vec![CoinFlipResult::Heads]),
                    None,
                    0,
                    None,
                    None,
                    None,
                    None,
                );
                perform_attack(
                    &mut tails_state,
                    acting_side_id,
                    deps,
                    attack_target_uid,
                    heal_target_uid,
                    Some(vec![CoinFlipResult::Tails]),
                    None,
                    0,
                    None,
                    None,
                    None,
                    None,
                );
                let heads_damage = get_damage_dealt(
                    heads_before.side(defending_id),
                    heads_state.side(defending_id),
                );
                let tails_damage = get_damage_dealt(
                    tails_before.side(defending_id),
                    tails_state.side(defending_id),
                );
                let expected = (heads_damage as f64 + tails_damage as f64) / 2.0;
                if expected > best_damage {
                    best_damage = expected;
                }
                continue;
            }
            let mut simulated = state.clone();
            let before = state.clone();
            perform_attack(
                &mut simulated,
                acting_side_id,
                deps,
                attack_target_uid,
                heal_target_uid,
                None,
                None,
                0,
                None,
                None,
                None,
                None,
            );
            let dealt = get_damage_dealt(before.side(defending_id), simulated.side(defending_id));
            if (dealt as f64) > best_damage {
                best_damage = dealt as f64;
            }
        }
    }
    best_damage
}

fn heuristic_mark_ability_used(
    side: &mut SideState,
    umamusume: &UmamusumeInstance,
    ability_name: &str,
) {
    // The instance lives in the side itself; we mutate via uid match.
    if let Some(active) = side.active.as_mut() {
        if active.uid == umamusume.uid {
            active.used_ability_this_turn = true;
        }
    }
    for u in side.bench.iter_mut() {
        if u.uid == umamusume.uid {
            u.used_ability_this_turn = true;
        }
    }
    if !side
        .used_ability_names_this_turn
        .iter()
        .any(|n| n == ability_name)
    {
        let _ = side
            .used_ability_names_this_turn
            .try_push(ability_name.to_string());
    }
}

fn find_in_side(side: &SideState, uid: u32) -> Option<&UmamusumeInstance> {
    if let Some(a) = &side.active {
        if a.uid == uid {
            return Some(a);
        }
    }
    side.bench.iter().find(|u| u.uid == uid)
}

fn find_in_side_mut(side: &mut SideState, uid: u32) -> Option<&mut UmamusumeInstance> {
    if let Some(a) = side.active.as_mut() {
        if a.uid == uid {
            return Some(a);
        }
    }
    side.bench.iter_mut().find(|u| u.uid == uid)
}
