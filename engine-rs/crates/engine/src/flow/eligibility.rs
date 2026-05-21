//! Bit-identical port of `frontend/src/game/engine/flow/eligibility.ts`.
//!
//! Predicate-only — read-only checks the engine uses to gate legal
//! actions. No state mutation here.

use crate::core::catalog::{catalog, Card};
use crate::core::constants::{EnergyType, SpecialCondition};
use crate::core::state::{CurrentSide, GameState, Phase, SideState};
use crate::core::umamusume::{attached_energy_count, find_own_umamusume_by_uid};
use crate::flow::ability_rules::get_umamusume_ability;
use crate::flow::energy::{get_ability_move_energy_types, has_enough_energy};
use crate::flow::retreat::effective_retreat_cost;

pub fn is_player_turn(state: &GameState) -> bool {
    state.phase == Phase::Play && !state.game_over && state.current_side == CurrentSide::Player
}

pub fn can_attach_energy(state: &GameState, side: &SideState) -> bool {
    state.phase == Phase::Play
        && state.pending_player_choice.is_none()
        && !state.game_over
        && state.current_side == CurrentSide::from_side(side.id)
        && !side.energy_zone.is_empty()
        && (side.energy_attachments_this_turn as u32)
            < 1 + side.bonus_energy_attachments as u32
}

pub fn can_attach_energy_to_umamusume(
    state: &GameState,
    side: &SideState,
    umamusume_uid: u32,
) -> bool {
    if !can_attach_energy(state, side) {
        return false;
    }
    if side.energy_attachments_this_turn < 1 {
        return true;
    }
    Some(umamusume_uid) == side.active.as_ref().map(|a| a.uid)
}

pub fn can_attack(state: &GameState, side: &SideState) -> bool {
    if state.phase != Phase::Play
        || state.pending_player_choice.is_some()
        || state.game_over
        || state.current_side != CurrentSide::from_side(side.id)
    {
        return false;
    }
    let Some(active) = &side.active else {
        return false;
    };
    if active
        .special_conditions
        .iter()
        .any(|&c| c == SpecialCondition::Paralysed)
    {
        return false;
    }
    let turns_taken = state.turns_taken_by_side[side.id as usize];
    if active.attack_blocked_until_own_turn == Some(turns_taken) {
        return false;
    }
    let cat = catalog();
    let card = match cat.get(active.card_id) {
        Some(Card::Umamusume(u)) => u,
        _ => return false,
    };
    let Some(primary) = card.attacks.first() else {
        return false;
    };
    has_enough_energy(active, &primary.cost)
}

pub fn can_retreat(state: &GameState, side: &SideState) -> bool {
    if state.phase != Phase::Play
        || state.pending_player_choice.is_some()
        || state.game_over
        || state.current_side != CurrentSide::from_side(side.id)
        || side.used_retreat_this_turn
    {
        return false;
    }
    let Some(active) = &side.active else {
        return false;
    };
    if active
        .special_conditions
        .iter()
        .any(|&c| c == SpecialCondition::Paralysed)
    {
        return false;
    }
    if side.bench.is_empty() {
        return false;
    }
    attached_energy_count(active) >= effective_retreat_cost(state, side)
}

pub fn can_use_umamusume_ability(
    state: &GameState,
    side: &SideState,
    ability_umamusume_uid: u32,
) -> bool {
    if state.phase != Phase::Play
        || state.pending_player_choice.is_some()
        || state.game_over
        || state.current_side != CurrentSide::from_side(side.id)
    {
        return false;
    }
    let Some(ability_umamusume) = find_own_umamusume_by_uid(side, ability_umamusume_uid) else {
        return false;
    };
    if ability_umamusume.used_ability_this_turn {
        return false;
    }
    let Some(ability) = get_umamusume_ability(state, side.id, ability_umamusume) else {
        return false;
    };
    if side
        .used_ability_names_this_turn
        .iter()
        .any(|n| n == &ability.name)
    {
        return false;
    }
    if ability.once_per_game == Some(true)
        && side
            .used_ability_names_this_game
            .iter()
            .any(|n| n == &ability.name)
    {
        return false;
    }
    if ability.move_benched_energy_to_active.is_some() {
        if side.active.is_none() {
            return false;
        }
        let types = get_ability_move_energy_types(Some(ability));
        if types.is_empty() {
            return false;
        }
        return side
            .bench
            .iter()
            .any(|u| types.iter().any(|&t| u.energies[t as usize] > 0));
    }
    if let Some(dtd) = &ability.discard_to_draw {
        return side.hand.len() as i32 >= dtd.discard;
    }
    if ability.coin_flip_draw_or_active_damage_counter.is_some() {
        return true;
    }
    if let Some(damage) = ability.damage_opponent {
        let _ = damage;
        if let Some(cost) = &ability.discard_energy {
            for t in EnergyType::ALL {
                let need = cost.get(t) as u16;
                if need > ability_umamusume.energies[t as usize] {
                    return false;
                }
            }
        }
        let opponent = state.side(side.id.opposite());
        return opponent.active.is_some() || !opponent.bench.is_empty();
    }
    if ability.shuffle_random_discard_into_deck.is_some() {
        return !side.discard.is_empty();
    }
    false
}
