//! Bit-identical port of `frontend/src/game/engine/flow/evolution.ts`.

use crate::core::card_id::CardId;
use crate::core::catalog::{catalog, Card, UmamusumeCard};
use crate::core::constants::SideId;
use crate::core::state::{GameState, SideState, UmamusumeInstance};
use crate::flow::special_conditions::clear_special_conditions;

/// Iterates `[active, ...bench]` in order (active first), preserving TS
/// iteration semantics. Returns refs into the side state.
pub fn get_evolution_targets<'a>(
    state: &GameState,
    side: &'a SideState,
    evolution_card: &UmamusumeCard,
) -> Vec<&'a UmamusumeInstance> {
    let mut out = Vec::new();
    if let Some(a) = &side.active {
        if is_valid_evolution_target(state, side.id, a, evolution_card) {
            out.push(a);
        }
    }
    for u in &side.bench {
        if is_valid_evolution_target(state, side.id, u, evolution_card) {
            out.push(u);
        }
    }
    out
}

pub fn find_evolution_target<'a>(
    state: &GameState,
    side: &'a SideState,
    evolution_card: &UmamusumeCard,
) -> Option<&'a UmamusumeInstance> {
    get_evolution_targets(state, side, evolution_card)
        .into_iter()
        .next()
}

pub fn is_valid_evolution_target(
    state: &GameState,
    side_id: SideId,
    umamusume: &UmamusumeInstance,
    evolution_card: &UmamusumeCard,
) -> bool {
    if is_side_first_turn(state, side_id) {
        return false;
    }
    let evolves_from = match &evolution_card.evolves_from {
        Some(s) => s.as_str(),
        None => return false,
    };
    if umamusume.species() != evolves_from {
        return false;
    }
    if umamusume.stage as i32 != evolution_card.stage as i32 - 1 {
        return false;
    }
    if umamusume.entered_turn == state.turn_number {
        return false;
    }
    if umamusume.evolved_turn == Some(state.turn_number) {
        return false;
    }
    true
}

/// Mutate the umamusume instance in place. The caller must hold a `&mut`
/// to the right instance (active or one of bench). `turn_number` is
/// passed by value so the caller doesn't have to hold a `&GameState` and
/// a `&mut SideState` simultaneously.
pub fn evolve_umamusume(
    turn_number: u32,
    umamusume: &mut UmamusumeInstance,
    evolution_card_id: CardId,
    evolution_card: &UmamusumeCard,
) {
    let damage = umamusume.max_hp - umamusume.hp;
    let prev_card_id = umamusume.card_id;
    let _ = umamusume.evolution_card_ids.try_push(prev_card_id);
    umamusume.card_id = evolution_card_id;
    // species() now derives from card_id via the catalog — no
    // synchronization needed here. stage stays as a hot direct field.
    umamusume.stage = evolution_card.stage;
    umamusume.max_hp = evolution_card.hp;
    umamusume.hp = evolution_card.hp - damage;
    umamusume.evolved_turn = Some(turn_number);
    umamusume.entered_turn = umamusume.entered_turn.min(turn_number.saturating_sub(1));
    clear_special_conditions(umamusume);
}

fn is_side_first_turn(state: &GameState, side_id: SideId) -> bool {
    state.turns_taken_by_side[side_id as usize] <= 1
}

/// Convenience: resolve an evolution-card id from the catalog. Useful for
/// tests and callers that have a card id but not the card payload.
pub fn evolution_card_from_catalog(card_id: CardId) -> Option<&'static UmamusumeCard> {
    catalog().get(card_id).and_then(|c| match c {
        Card::Umamusume(u) => Some(u),
        _ => None,
    })
}
