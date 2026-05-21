//! Bit-identical port of `frontend/src/game/engine/flow/turn.ts`.
//!
//! RNG sites:
//! - `prepareUmamusumeForTurn`: no RNG.
//! - `startTurn`: one `rollEnergyFromPool(side.energyPool)` per call,
//!   skipped only on the first player's first turn.
//! - `endTurn`: no RNG.
//!
//! `startTurn` and `endTurn` accept the `refreshContinuousEffects`
//! callback by closure so the caller can wire `flow::board::refresh_continuous_hp`
//! without creating a circular module dep.

use crate::core::card_id::CardId;
use crate::core::catalog::{catalog, Card};
use crate::core::constants::{EnergyType, OpponentTurnStep, SideId, SpecialCondition, MAX_HAND};
use crate::core::random::random_int;
use crate::core::state::{CurrentSide, GameState, SideState};
use crate::flow::ability_rules::get_umamusume_ability;

/// `turn.ts:10` `prepareUmamusumeForTurn`. Resets the per-turn damage and
/// ability flags on every umamusume on `side`.
pub fn prepare_umamusume_for_turn(side: &mut SideState) {
    let touch = |u: &mut crate::core::state::UmamusumeInstance| {
        u.took_damage_last_turn = u.took_damage_this_turn;
        u.took_damage_this_turn = false;
        u.next_turn_damage_reduction = 0;
        u.used_ability_this_turn = false;
    };
    if let Some(active) = &mut side.active {
        touch(active);
    }
    for u in side.bench.iter_mut() {
        touch(u);
    }
}

/// `turn.ts:19` `drawCards`. Up to `amount` cards from deck → hand.
/// Returns the actual draw count (TS returns the array of drawn ids).
pub fn draw_cards(side: &mut SideState, amount: u32) -> Vec<CardId> {
    let mut drawn = Vec::with_capacity(amount as usize);
    for _ in 0..amount {
        if side.hand.len() >= MAX_HAND {
            return drawn;
        }
        let Some(card) = (if side.deck.is_empty() {
            None
        } else {
            Some(side.deck.remove(0))
        }) else {
            return drawn;
        };
        let _ = side.hand.try_push(card);
        drawn.push(card);
    }
    drawn
}

/// `turn.ts:37` `applyStartAbilities`. Takes `&mut GameState` so it can
/// look up ability data through the catalog. Two-phase to avoid an
/// immutable+mutable borrow conflict.
pub fn apply_start_abilities(state: &mut GameState, side_id: SideId) {
    let heal_amount: Option<i32> = {
        let side = state.side(side_id);
        side.active.as_ref().and_then(|active| {
            get_umamusume_ability(state, side_id, active)
                .and_then(|a| a.heal)
                .filter(|h| *h > 0)
        })
    };
    if let Some(heal) = heal_amount {
        if let Some(active) = state.side_mut(side_id).active.as_mut() {
            let new_hp = (active.hp + heal).min(active.max_hp);
            active.hp = new_hp;
        }
    }
}

/// `turn.ts:48` `startTurn`. RNG site: one `rollEnergyFromPool` unless
/// this is the first player's very first turn.
///
/// `refresh_continuous_effects` is taken as a callback so flow modules
/// can compose without circular deps.
pub fn start_turn(
    state: &mut GameState,
    side_id: SideId,
    mut refresh_continuous_effects: impl FnMut(&mut GameState),
    skip_draw: bool,
) {
    let turns_taken = state.turns_taken_by_side[side_id as usize];
    let is_side_first_turn = turns_taken == 0;
    state.turns_taken_by_side[side_id as usize] = turns_taken + 1;
    state.current_side = CurrentSide::from_side(side_id);
    state.opponent_turn_step = if side_id == SideId::Opponent {
        Some(OpponentTurnStep::Bench)
    } else {
        None
    };

    let energy_pool_snapshot: Vec<EnergyType> = {
        let side = state.side_mut(side_id);
        side.energy_attachments_this_turn = 0;
        side.bonus_energy_attachments = 0;
        side.retreat_cost_reduction = 0;
        side.active_attack_damage_bonus = 0;
        side.used_supporter_this_turn = false;
        side.used_retreat_this_turn = false;
        side.used_stadium_this_turn = false;
        side.used_ability_names_this_turn.clear();
        prepare_umamusume_for_turn(side);
        side.energy_zone.clear();
        side.energy_pool.iter().copied().collect()
    };

    let should_roll_energy = !(is_side_first_turn && state.first_player == side_id);
    if should_roll_energy && !energy_pool_snapshot.is_empty() {
        let idx = random_int(energy_pool_snapshot.len() as u32) as usize;
        let next_energy = EnergyType::from_pool_index(&energy_pool_snapshot, idx);
        let side = state.side_mut(side_id);
        let _ = side.energy_zone.try_push(next_energy);
    }

    refresh_continuous_effects(state);
    apply_start_abilities(state, side_id);
    if !skip_draw {
        let _ = draw_cards(state.side_mut(side_id), 1);
    }
}

/// `turn.ts:78` `endTurn`. Composition of `applyEndTurnToolTriggers`,
/// `processEndTurnStatusConditions`, then either `startTurn(nextSide)` or
/// a no-op if a pending choice / game over fires.
pub fn end_turn(
    state: &mut GameState,
    start_turn_impl: impl FnOnce(&mut GameState, SideId),
    mut refresh_continuous_effects: impl FnMut(&mut GameState),
) {
    if state.game_over || state.current_side == CurrentSide::Done {
        return;
    }
    let Some(current_side) = state.current_side.as_side() else {
        return;
    };
    apply_end_turn_tool_triggers(state, current_side);
    process_end_turn_status_conditions(state);
    refresh_continuous_effects(state);

    // The TS source mutates the pending choice's resume in place; mirror.
    if let Some(crate::core::state::PendingPlayerChoice::PromoteAfterKnockout {
        side_id,
        resume,
    }) = state.pending_player_choice.as_mut()
    {
        if *side_id == current_side {
            *resume = crate::core::state::PromoteResume::FinishOpponentTurn;
        }
    }

    if state.game_over || state.pending_player_choice.is_some() {
        return;
    }
    let next_side = current_side.opposite();
    if next_side == state.first_player {
        state.turn_number += 1;
    }
    start_turn_impl(state, next_side);
}

fn apply_end_turn_tool_triggers(state: &mut GameState, side_id: SideId) {
    if are_tools_disabled(state) {
        return;
    }
    let (tool_card_id, max_hp, hp) = {
        let side = state.side(side_id);
        let Some(active) = &side.active else { return };
        let Some(tool) = active.tool_card_id else {
            return;
        };
        (tool, active.max_hp, active.hp)
    };
    let cat = catalog();
    let Some(Card::Trainer(t)) = cat.get(tool_card_id) else {
        return;
    };
    let heal = t.effect.tool_end_turn_heal_active.unwrap_or(0);
    if heal <= 0 {
        return;
    }
    let new_hp = (hp + heal).min(max_hp);
    if let Some(active) = state.side_mut(side_id).active.as_mut() {
        active.hp = new_hp;
    }
}

fn are_tools_disabled(state: &GameState) -> bool {
    let Some(stadium) = &state.stadium else {
        return false;
    };
    let cat = catalog();
    matches!(
        cat.get(stadium.card_id),
        Some(Card::Trainer(t)) if t.effect.disable_tools == Some(true)
    )
}

fn process_end_turn_status_conditions(state: &mut GameState) {
    // Snapshot per-side turnsTaken to avoid an aliasing borrow later.
    let turns_taken: [u32; 2] = [
        state.turns_taken_by_side[0],
        state.turns_taken_by_side[1],
    ];
    for &side_id in &SideId::ALL {
        let side = state.side_mut(side_id);
        let touch = |u: &mut crate::core::state::UmamusumeInstance| {
            if u.special_conditions.iter().any(|&c| c == SpecialCondition::Poisoned) {
                u.hp = (u.hp - 10).max(0);
                u.took_damage_this_turn = true;
            }
            if !u.special_conditions.iter().any(|&c| c == SpecialCondition::Paralysed) {
                return;
            }
            let Some(recovery_turn) = u.paralysed_until_own_turn else {
                return;
            };
            if turns_taken[side_id as usize] < recovery_turn {
                return;
            }
            u.special_conditions.retain(|c| *c != SpecialCondition::Paralysed);
            u.paralysed_until_own_turn = None;
        };
        if let Some(active) = side.active.as_mut() {
            touch(active);
        }
        for b in side.bench.iter_mut() {
            touch(b);
        }
    }
}
