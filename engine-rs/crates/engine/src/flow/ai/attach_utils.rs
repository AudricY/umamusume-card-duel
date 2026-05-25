//! Bit-identical port of `frontend/src/game/engine/flow/ai/attachUtils.ts`.
//!
//! Picks the energy-attach target for the AI. Many float-weighted
//! scoring rules; preserve source-text op order exactly.
//!
//! `usefulCapByUid` uses `IndexMap` (TS `Map` preserves insertion order).
//! Bench/active iteration matches `getAllUmamusume` (active first).

use indexmap::IndexMap;

use crate::core::catalog::{catalog, Card, UmamusumeCard};
use crate::core::constants::{AiDeckStyle, EnergyType, SideId, MAX_BENCH};
use crate::core::effects::{Attack, EnergyCost};
use crate::core::state::{GameState, SideState, UmamusumeInstance};
use crate::core::umamusume::{attached_energy_count, get_all_umamusume};
use crate::flow::energy::{attach_energy, has_enough_energy};
use crate::flow::retreat::effective_retreat_cost;

use super::combat_utils::{are_tool_effects_disabled, predict_attack_damage};
use super::types::AiTurnGoal;

/// Source-declaration order of `ENERGY_TYPES` in TS:
/// `["grass","fire","water","lightning","psychic","fighting","darkness","steel","colorless","dragon"]`.
/// Matches `EnergyType::ALL`.
const ENERGY_TYPES: [EnergyType; 10] = EnergyType::ALL;

pub fn ai_attach_one_energy(state: &mut GameState, side_id: SideId, turn_goal: AiTurnGoal) -> bool {
    if state.side(side_id).active.is_none() {
        return false;
    }
    let next_energy_type = match state.side(side_id).energy_zone.first().copied() {
        Some(e) => e,
        None => return false,
    };
    let deck_style = state.ai_deck_style_by_side[side_id as usize];
    let future_demand = build_future_energy_demand(state.side(side_id));

    let energy_attachments_this_turn = state.side(side_id).energy_attachments_this_turn;
    let candidates_owned: Vec<UmamusumeInstance> = if energy_attachments_this_turn >= 1 {
        state
            .side(side_id)
            .active
            .as_ref()
            .cloned()
            .into_iter()
            .collect()
    } else {
        get_all_umamusume(state.side(side_id))
            .into_iter()
            .cloned()
            .collect()
    };

    let mut useful_cap_by_uid: IndexMap<u32, u32> = IndexMap::new();
    for umamusume in candidates_owned.iter() {
        let cap = get_useful_energy_cap(
            state,
            state.side(side_id),
            umamusume,
            next_energy_type,
            deck_style,
        );
        useful_cap_by_uid.insert(umamusume.uid, cap);
    }
    let has_undercharged_alternative = candidates_owned.iter().any(|umamusume| {
        let cap = useful_cap_by_uid
            .get(&umamusume.uid)
            .copied()
            .unwrap_or(1);
        attached_energy_count(umamusume) < cap
    });

    let mut scored: Vec<(UmamusumeInstance, f64)> = candidates_owned
        .iter()
        .map(|umamusume| {
            let cap = useful_cap_by_uid
                .get(&umamusume.uid)
                .copied()
                .unwrap_or(1);
            let s = score_attach_target(
                state,
                state.side(side_id),
                umamusume,
                next_energy_type,
                deck_style,
                &future_demand,
                cap,
                has_undercharged_alternative,
                turn_goal,
            );
            (umamusume.clone(), s)
        })
        .collect();

    let active_uid = state.side(side_id).active.as_ref().map(|a| a.uid);
    scored.sort_by(|left, right| {
        if right.1 != left.1 {
            return right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal);
        }
        let left_active = if Some(left.0.uid) == active_uid { 1 } else { 0 };
        let right_active = if Some(right.0.uid) == active_uid { 1 } else { 0 };
        if right_active != left_active {
            return right_active.cmp(&left_active);
        }
        if right.0.stage != left.0.stage {
            return right.0.stage.cmp(&left.0.stage);
        }
        let left_energy = attached_energy_count(&left.0) as i32;
        let right_energy = attached_energy_count(&right.0) as i32;
        left_energy.cmp(&right_energy)
    });

    let chosen = match scored.first() {
        Some(c) => c.0.clone(),
        None => return false,
    };
    attach_energy(state.side_mut(side_id), chosen.uid)
}

/// `attachUtils.ts:61` `markAbilityUsed`. Side-mutating helper used as a
/// callback by `aiUseMoveBenchedEnergyAbility`.
pub fn mark_ability_used(side: &mut SideState, umamusume: &UmamusumeInstance, ability_name: &str) {
    // Apply to whichever copy lives on the side.
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

/// `attachUtils.ts:66` `withEnergyShift`.
pub fn with_energy_shift(
    umamusume: &UmamusumeInstance,
    energy_type: EnergyType,
    delta: i32,
) -> UmamusumeInstance {
    let mut clone = umamusume.clone();
    let cur = clone.energies[energy_type as usize] as i32;
    let next = (cur + delta).max(0);
    clone.energies[energy_type as usize] = next as u16;
    clone
}

/// `attachUtils.ts:80` `estimateAttackDamageOutput`.
pub fn estimate_attack_damage_output(
    state: &GameState,
    attacking_side_id: SideId,
    attacker: &UmamusumeInstance,
    original_attacker: &UmamusumeInstance,
) -> f64 {
    let cat = catalog();
    let card = match cat.get(original_attacker.card_id) {
        Some(Card::Umamusume(u)) => u,
        _ => return 0.0,
    };
    let attack = match card.attacks.first() {
        Some(a) => a,
        None => return 0.0,
    };
    if !has_enough_energy(attacker, &attack.cost) {
        return 0.0;
    }
    let attacking_side = state.side(attacking_side_id);
    let defending_side = state.side(attacking_side_id.opposite());
    let own_in_play_count = 1 + attacking_side.bench.len() as i32;
    let all_in_play_count = own_in_play_count + 1 + defending_side.bench.len() as i32;

    use crate::core::effects::AttackTarget;
    let targets: crate::core::umamusume::AllUmamusume<'_> = match attack.target_opponent {
        Some(AttackTarget::Any) => get_all_umamusume(defending_side),
        _ => defending_side.active.as_ref().into_iter().collect(),
    };
    if targets.is_empty() {
        return 0.0;
    }
    let tools_disabled = are_tool_effects_disabled(state);
    let mut best = i32::MIN;
    for target in targets {
        let d = predict_attack_damage(
            attacker,
            target,
            attacking_side.active_attack_damage_bonus as i32,
            own_in_play_count,
            all_in_play_count,
            Some(state.turn_number),
            tools_disabled,
        );
        if d > best {
            best = d;
        }
    }
    best.max(i32::MIN + 1) as f64
}

/// `attachUtils.ts:97` `scoreAiAttachTarget`.
pub fn score_ai_attach_target(
    state: &GameState,
    side_id: SideId,
    target: &UmamusumeInstance,
    turn_goal: AiTurnGoal,
) -> f64 {
    let side = state.side(side_id);
    let next_energy_type = match side.energy_zone.first().copied() {
        Some(e) => e,
        None => return f64::NEG_INFINITY,
    };
    let deck_style = state.ai_deck_style_by_side[side_id as usize];
    let candidates: Vec<UmamusumeInstance> = if side.energy_attachments_this_turn >= 1 {
        side.active.as_ref().cloned().into_iter().collect()
    } else {
        get_all_umamusume(side).into_iter().cloned().collect()
    };
    let future_demand = build_future_energy_demand(side);
    let mut useful_cap_by_uid: IndexMap<u32, u32> = IndexMap::new();
    for u in candidates.iter() {
        let cap = get_useful_energy_cap(state, side, u, next_energy_type, deck_style);
        useful_cap_by_uid.insert(u.uid, cap);
    }
    let has_undercharged_alternative = candidates.iter().any(|u| {
        let cap = useful_cap_by_uid.get(&u.uid).copied().unwrap_or(1);
        attached_energy_count(u) < cap
    });
    let cap = useful_cap_by_uid.get(&target.uid).copied().unwrap_or(1);
    score_attach_target(
        state,
        side,
        target,
        next_energy_type,
        deck_style,
        &future_demand,
        cap,
        has_undercharged_alternative,
        turn_goal,
    )
}

fn should_attach_for_damage_scaling(
    umamusume: &UmamusumeInstance,
    next_energy_type: EnergyType,
) -> bool {
    let cat = catalog();
    let card = match cat.get(umamusume.card_id) {
        Some(Card::Umamusume(c)) => c,
        _ => return false,
    };
    let Some(attack) = card.attacks.first() else {
        return false;
    };
    if let Some(scale) = &attack.damage_per_attached_energy {
        if scale.types.iter().any(|&t| t == next_energy_type) {
            return true;
        }
    }
    if let Some(threshold) = card
        .ability
        .as_ref()
        .and_then(|a| a.attack_damage_bonus_if_attached_energy.as_ref())
    {
        if threshold.r#type == next_energy_type
            && (umamusume.energies[next_energy_type as usize] as i32) < threshold.min
        {
            return true;
        }
    }
    false
}

#[allow(clippy::too_many_arguments)]
fn score_attach_target(
    state: &GameState,
    side: &SideState,
    target: &UmamusumeInstance,
    energy_type: EnergyType,
    deck_style: AiDeckStyle,
    future_demand: &[f64; EnergyType::COUNT],
    useful_cap: u32,
    has_undercharged_alternative: bool,
    turn_goal: AiTurnGoal,
) -> f64 {
    let active = side.active.as_ref();
    let is_active = active.map(|a| a.uid == target.uid).unwrap_or(false);
    let cat = catalog();
    let card = match cat.get(target.card_id) {
        Some(Card::Umamusume(u)) => u,
        _ => return f64::NEG_INFINITY,
    };
    let attack = match card.attacks.first() {
        Some(a) => a,
        None => return f64::NEG_INFINITY,
    };
    let total_energy_before = attached_energy_count(target);
    let total_energy_after = total_energy_before + 1;
    let before_can_attack = has_enough_energy(target, &attack.cost);
    let before_damage = estimate_attack_damage_output(state, side.id, target, target);
    let simulated = with_energy_shift(target, energy_type, 1);
    let after_can_attack = has_enough_energy(&simulated, &attack.cost);
    let after_damage = estimate_attack_damage_output(state, side.id, &simulated, target);
    let typed_deficit_before = get_typed_energy_deficit(target, &attack.cost);
    let typed_deficit_after = get_typed_energy_deficit(&simulated, &attack.cost);
    let target_needs_this_type =
        (attack.cost.get(energy_type) as u16) > target.energies[energy_type as usize];
    let attack_uses_this_type = attack.cost.get(energy_type) > 0;
    let supports_scaling = should_attach_for_damage_scaling(target, energy_type);
    let target_type_demand: i32 = if target_needs_this_type { 1 } else { 0 };
    let future_demand_weight = future_demand[energy_type as usize];

    let mut score: f64 = 0.0;
    if after_can_attack && !before_can_attack {
        score += if is_active { 220.0 } else { 170.0 };
    }
    score += ((typed_deficit_before - typed_deficit_after) as f64) * 30.0;
    score += (after_damage - before_damage).max(0.0) * (if is_active { 2.5 } else { 1.8 });
    score += (target_type_demand as f64) * 24.0;
    score += (40.0_f64).min(future_demand_weight * 4.0);
    if is_active {
        score += 20.0;
    }
    if target.stage > 0 {
        score += (target.stage as f64) * 8.0;
    }
    if before_can_attack && after_damage <= before_damage && !supports_scaling {
        score -= 80.0;
    }
    if !attack_uses_this_type && !supports_scaling {
        score -= if is_active { 42.0 } else { 120.0 };
        if before_can_attack && after_can_attack && after_damage <= before_damage {
            score -= 60.0;
        }
    }
    if total_energy_before < useful_cap {
        score += 28.0;
    }
    if total_energy_after > useful_cap {
        score -= ((total_energy_after - useful_cap) as f64) * 140.0;
    }
    if has_undercharged_alternative && total_energy_after > useful_cap {
        score -= 180.0;
    }
    score += score_deck_style_attach_preference(
        side,
        target,
        deck_style,
        energy_type,
        before_can_attack,
        after_can_attack,
        before_damage,
        after_damage,
    );
    score += score_turn_goal_attach_preference(
        side,
        target,
        before_can_attack,
        after_can_attack,
        turn_goal,
    );
    score
}

fn score_turn_goal_attach_preference(
    side: &SideState,
    target: &UmamusumeInstance,
    before_can_attack: bool,
    after_can_attack: bool,
    turn_goal: AiTurnGoal,
) -> f64 {
    let is_active = side
        .active
        .as_ref()
        .map(|a| a.uid == target.uid)
        .unwrap_or(false);
    match turn_goal {
        AiTurnGoal::DenyOpponentLethal => {
            let mut score = if is_active { 30.0 } else { 10.0 };
            if !before_can_attack && after_can_attack {
                score += if is_active { 120.0 } else { 28.0 };
            }
            if target.max_hp >= 100 && is_active {
                score += 16.0;
            }
            score
        }
        AiTurnGoal::StabilizeBoard => {
            let mut score = 0.0;
            if !is_active {
                score += 22.0;
            }
            if !before_can_attack && after_can_attack {
                score += 34.0;
            }
            if target.max_hp >= 90 {
                score += 10.0;
            }
            score
        }
        AiTurnGoal::ProtectLoadedActive => {
            let mut score = if is_active { 10.0 } else { 18.0 };
            if !before_can_attack && after_can_attack {
                score += if is_active { 92.0 } else { 70.0 };
            }
            if !is_active && target.max_hp >= 90 {
                score += 18.0;
            }
            if is_active && before_can_attack {
                score -= 16.0;
            }
            score
        }
        AiTurnGoal::BuildBackupAttacker => {
            let mut score = if is_active { -34.0 } else { 40.0 };
            if !before_can_attack && after_can_attack {
                score += if is_active { 20.0 } else { 118.0 };
            }
            if !is_active && target.stage >= 1 {
                score += 18.0;
            }
            if !is_active && target.max_hp >= 90 {
                score += 12.0;
            }
            score
        }
        AiTurnGoal::DigForEvolution => {
            let mut score = 0.0;
            if target.stage <= 1 {
                score += if is_active { 22.0 } else { 28.0 };
            }
            if !before_can_attack && after_can_attack {
                score += if is_active { 40.0 } else { 50.0 };
            }
            if target.stage >= 1 {
                score += 12.0;
            }
            score
        }
        AiTurnGoal::ConvertPointLead => {
            let mut score = if is_active { 44.0 } else { 4.0 };
            if !before_can_attack && after_can_attack {
                score += if is_active { 100.0 } else { 44.0 };
            }
            if before_can_attack && is_active {
                score += 20.0;
            }
            score
        }
        AiTurnGoal::SecureLethalNow => {
            let mut score = if is_active { 26.0 } else { -8.0 };
            if !before_can_attack && after_can_attack && is_active {
                score += 84.0;
            }
            score
        }
        AiTurnGoal::SetUpTwoTurnLethal => {
            let mut score = if is_active { 34.0 } else { 6.0 };
            if !before_can_attack && after_can_attack && is_active {
                score += 96.0;
            }
            if target.stage >= 1 && is_active {
                score += 12.0;
            }
            score
        }
        AiTurnGoal::MaximizeProgress => 0.0,
    }
}

fn get_useful_energy_cap(
    state: &GameState,
    side: &SideState,
    target: &UmamusumeInstance,
    next_energy_type: EnergyType,
    deck_style: AiDeckStyle,
) -> u32 {
    let cat = catalog();
    let card = match cat.get(target.card_id) {
        Some(Card::Umamusume(c)) => c,
        _ => return 1,
    };
    let Some(attack) = card.attacks.first() else {
        return 1;
    };
    let base_attack_cost = total_attack_cost(&attack.cost);
    let attack_discard_cost = typed_energy_total(attack.discard_energy.as_ref());
    let ability_discard_cost = typed_energy_total(
        card.ability.as_ref().and_then(|a| a.discard_energy.as_ref()),
    );
    let total_discard_cost = attack_discard_cost + ability_discard_cost;
    let damage_scaling_types: &[EnergyType] = attack
        .damage_per_attached_energy
        .as_ref()
        .map(|d| d.types.as_slice())
        .unwrap_or(&[]);

    let mut cap = base_attack_cost.max(1);
    if total_discard_cost > 0 {
        cap += 1;
    }
    if damage_scaling_types.iter().any(|&t| t == next_energy_type) {
        cap += 1;
    }
    if !damage_scaling_types.is_empty()
        && side.active.as_ref().map(|a| a.uid == target.uid).unwrap_or(false)
    {
        cap += 1;
    }
    if deck_style == AiDeckStyle::Stall
        && side.active.as_ref().map(|a| a.uid == target.uid).unwrap_or(false)
    {
        cap += 1;
    }
    if let Some(threshold) = card
        .ability
        .as_ref()
        .and_then(|a| a.attack_damage_bonus_if_attached_energy.as_ref())
    {
        let required_in_attack = attack.cost.get(threshold.r#type) as i32;
        if threshold.min > required_in_attack {
            cap += 1;
        }
    }
    if side.active.as_ref().map(|a| a.uid == target.uid).unwrap_or(false) {
        let retreat_cost = effective_retreat_cost(state, side);
        if retreat_cost >= 2 {
            cap += 1;
        }
    }
    cap.max(1)
}

fn total_attack_cost(cost: &EnergyCost) -> u32 {
    let mut sum = 0u32;
    for t in EnergyType::ALL {
        sum += cost.get(t) as u32;
    }
    sum
}

fn typed_energy_total(cost: Option<&EnergyCost>) -> u32 {
    let Some(cost) = cost else {
        return 0;
    };
    let mut sum = 0u32;
    for t in EnergyType::ALL {
        sum += cost.get(t) as u32;
    }
    sum
}

#[allow(clippy::too_many_arguments)]
fn score_deck_style_attach_preference(
    side: &SideState,
    target: &UmamusumeInstance,
    deck_style: AiDeckStyle,
    energy_type: EnergyType,
    before_can_attack: bool,
    after_can_attack: bool,
    before_damage: f64,
    after_damage: f64,
) -> f64 {
    let active = side.active.as_ref();
    let is_active = active.map(|a| a.uid == target.uid).unwrap_or(false);
    let cat = catalog();
    let card = match cat.get(target.card_id) {
        Some(Card::Umamusume(u)) => u,
        _ => return 0.0,
    };
    let Some(attack) = card.attacks.first() else {
        return 0.0;
    };
    let has_scaling_by_board = attack.damage_per_umamusume_in_play.is_some();
    let has_damage_reduction = card
        .ability
        .as_ref()
        .and_then(|a| a.damage_reduction)
        .is_some();
    let high_hp = if target.max_hp >= 100 { 1 } else { 0 };
    let stage_bonus = (target.stage as i32).max(0);

    match deck_style {
        AiDeckStyle::Blitz => {
            let mut score = if is_active { 80.0 } else { 10.0 };
            if !before_can_attack && after_can_attack {
                score += if is_active { 120.0 } else { 55.0 };
            }
            if after_damage > before_damage {
                score += if is_active { 46.0 } else { 14.0 };
            }
            if !is_active {
                if let Some(active) = active {
                    if let Some(Card::Umamusume(c)) = cat.get(active.card_id) {
                        if let Some(active_attack) = c.attacks.first() {
                            if !has_enough_energy(active, &active_attack.cost) {
                                score -= 65.0;
                            }
                        }
                    }
                }
            }
            score
        }
        AiDeckStyle::ScaleBench => {
            let mut score = 0.0;
            if has_scaling_by_board {
                score += 48.0;
            }
            if !is_active {
                score += 28.0;
            }
            if side.bench.len() < MAX_BENCH && !is_active {
                score += 12.0;
            }
            if !before_can_attack && after_can_attack {
                score += if is_active { 60.0 } else { 42.0 };
            }
            if let Some(scale) = &attack.damage_per_attached_energy {
                if scale.types.iter().any(|&t| t == energy_type) {
                    score += 18.0;
                }
            }
            score += (stage_bonus as f64) * 10.0;
            score
        }
        AiDeckStyle::Stall => {
            let mut score = if is_active { 58.0 } else { 24.0 };
            if has_damage_reduction {
                score += 45.0;
            }
            score += (high_hp as f64) * 20.0;
            if !before_can_attack && after_can_attack {
                score += if is_active { 70.0 } else { 36.0 };
            }
            if after_damage > before_damage {
                score += 10.0;
            }
            score
        }
        AiDeckStyle::Balanced => {
            let mut score = if is_active { 24.0 } else { 8.0 };
            if !before_can_attack && after_can_attack {
                score += if is_active { 72.0 } else { 40.0 };
            }
            if after_damage > before_damage {
                score += if is_active { 16.0 } else { 9.0 };
            }
            score += (stage_bonus as f64) * 5.0;
            score
        }
    }
}

fn build_future_energy_demand(side: &SideState) -> [f64; EnergyType::COUNT] {
    let available_energy_types: Vec<EnergyType> = side.energy_pool.iter().copied().collect();
    let mut demand = [0.0f64; EnergyType::COUNT];
    let _ = ENERGY_TYPES;
    let cat = catalog();

    let add_cost_demand = |demand: &mut [f64; EnergyType::COUNT],
                            card_id: crate::core::card_id::CardId,
                            weight: f64| {
        let Some(Card::Umamusume(card)) = cat.get(card_id) else {
            return;
        };
        let Some(attack) = card.attacks.first() else {
            return;
        };
        for t in EnergyType::ALL {
            if t == EnergyType::Colorless {
                continue;
            }
            let required = attack.cost.get(t) as f64;
            if required <= 0.0 {
                continue;
            }
            if !available_energy_types.iter().any(|&e| e == t) {
                continue;
            }
            demand[t as usize] += required * weight;
        }
        if let Some(threshold) = card
            .ability
            .as_ref()
            .and_then(|a| a.attack_damage_bonus_if_attached_energy.as_ref())
        {
            if available_energy_types.iter().any(|&e| e == threshold.r#type) {
                demand[threshold.r#type as usize] += (threshold.min as f64) * weight * 0.4;
            }
        }
        if let Some(scale) = &attack.damage_per_attached_energy {
            for &t in scale.types.iter() {
                if !available_energy_types.iter().any(|&e| e == t) {
                    continue;
                }
                demand[t as usize] += weight * 0.9;
            }
        }
    };

    if let Some(active) = &side.active {
        add_cost_demand(&mut demand, active.card_id, 1.8);
    }
    for u in side.bench.iter() {
        add_cost_demand(&mut demand, u.card_id, 1.4);
    }
    for &cid in side.hand.iter() {
        add_cost_demand(&mut demand, cid, 1.0);
    }
    for &cid in side.discard.iter() {
        add_cost_demand(&mut demand, cid, 0.45);
    }
    for &cid in side.deck.iter() {
        add_cost_demand(&mut demand, cid, 0.2);
    }
    demand
}

fn get_typed_energy_deficit(umamusume: &UmamusumeInstance, cost: &EnergyCost) -> i32 {
    let mut sum = 0i32;
    for t in EnergyType::ALL {
        if t == EnergyType::Colorless {
            continue;
        }
        let required = cost.get(t) as i32;
        let have = umamusume.energies[t as usize] as i32;
        sum += (required - have).max(0);
    }
    sum
}

// Unused-import suppression for `Attack`.
#[allow(dead_code)]
fn _suppress(_a: Attack, _u: UmamusumeCard) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::card_id::CardId;
    use crate::core::constants::{AiDeckStyle, AiDifficulty, EnergyType, SideId};
    use crate::core::state::{CurrentSide, Phase, SideState, UmamusumeInstance};
    use arrayvec::ArrayVec;

    fn empty_side(id: SideId) -> SideState {
        SideState {
            id,
            title: String::new(),
            energy_pool: ArrayVec::new(),
            deck: ArrayVec::new(),
            discard: ArrayVec::new(),
            hand: ArrayVec::new(),
            active: None,
            bench: ArrayVec::new(),
            points: 0,
            energy_zone: ArrayVec::new(),
            energy_attachments_this_turn: 0,
            bonus_energy_attachments: 0,
            retreat_cost_reduction: 0,
            active_attack_damage_bonus: 0,
            used_supporter_this_turn: false,
            used_retreat_this_turn: false,
            used_stadium_this_turn: false,
            used_ability_names_this_turn: ArrayVec::new(),
            used_ability_names_this_game: ArrayVec::new(),
            guaranteed_coin_flip_heads: 0,
        }
    }

    fn empty_state() -> GameState {
        GameState {
            phase: Phase::Play,
            setup: None,
            pending_player_choice: None,
            sides: [empty_side(SideId::Player), empty_side(SideId::Opponent)],
            current_side: CurrentSide::Player,
            opponent_turn_step: None,
            stadium: None,
            turn_deadline_ms: None,
            turn_number: 2,
            first_player: SideId::Player,
            turns_taken_by_side: [1, 1],
            ai_difficulty: AiDifficulty::Normal,
            human_by_side: [false, false],
            ai_deck_style_by_side: [AiDeckStyle::Balanced, AiDeckStyle::Balanced],
            game_over: false,
            winner: None,
            log: std::collections::VecDeque::new(),
        }
    }

    #[test]
    fn score_ai_attach_target_returns_neg_inf_when_zone_empty() {
        // No energy in the zone → score should be NEG_INFINITY.
        let state = empty_state();
        let cat = crate::core::catalog::catalog();
        let cid = cat
            .id_for("matikanetannhauserBasic")
            .expect("base card present");
        let inst = UmamusumeInstance {
            uid: 1,
            card_id: cid,
            evolution_card_ids: ArrayVec::new(),
            stage: 0,
            hp: 60,
            max_hp: 60,
            energies: [0u16; EnergyType::COUNT],
            special_conditions: ArrayVec::new(),
            entered_turn: 0,
            evolved_turn: None,
            took_damage_last_turn: false,
            took_damage_this_turn: false,
            next_turn_damage_reduction: 0,
            used_ability_this_turn: false,
            attack_blocked_until_own_turn: None,
            paralysed_until_own_turn: None,
            tool_card_id: None,
        };
        let score = score_ai_attach_target(
            &state,
            SideId::Player,
            &inst,
            AiTurnGoal::MaximizeProgress,
        );
        assert_eq!(score, f64::NEG_INFINITY);
    }

    #[test]
    fn with_energy_shift_clamps_at_zero() {
        let mut inst = UmamusumeInstance {
            uid: 1,
            card_id: CardId(0),
            evolution_card_ids: ArrayVec::new(),
            stage: 0,
            hp: 60,
            max_hp: 60,
            energies: [0u16; EnergyType::COUNT],
            special_conditions: ArrayVec::new(),
            entered_turn: 0,
            evolved_turn: None,
            took_damage_last_turn: false,
            took_damage_this_turn: false,
            next_turn_damage_reduction: 0,
            used_ability_this_turn: false,
            attack_blocked_until_own_turn: None,
            paralysed_until_own_turn: None,
            tool_card_id: None,
        };
        inst.energies[EnergyType::Grass as usize] = 0;
        let result = with_energy_shift(&inst, EnergyType::Grass, -1);
        assert_eq!(result.energies[EnergyType::Grass as usize], 0);
        let bumped = with_energy_shift(&inst, EnergyType::Fire, 2);
        assert_eq!(bumped.energies[EnergyType::Fire as usize], 2);
    }
}
