//! Bit-identical port of `frontend/src/game/engine/flow/board.ts`.
//!
//! Notes:
//! - Log calls in TS are deliberately skipped here — `state.log` is
//!   excluded from `backend/src/sim/stateFingerprint.ts` (lines 32–75),
//!   so log content is not part of the fingerprint contract. If a future
//!   trace schema includes logs, this is the touchpoint to revisit.
//! - All HP / energy / max_hp arithmetic is pure-integer i32.
//! - `choosePreferredActiveIndex` uses the printed attack damage from the
//!   primary attack — sourced from the catalog, never mutated.

use arrayvec::ArrayVec;

use crate::core::catalog::{catalog, Card, UmamusumeCard};
use crate::core::constants::{SideId, MAX_BENCH};
use crate::core::state::{
    GameState, PendingPlayerChoice, SideState, SwitchResume, UmamusumeInstance,
};
use crate::core::umamusume::{attached_energy_count, get_all_umamusume};
use crate::flow::ability_rules::get_umamusume_ability;
use crate::flow::retreat::retreat_cost;
use crate::flow::special_conditions::clear_special_conditions;

pub fn get_opposing_side(state: &GameState, side_id: SideId) -> &SideState {
    state.side(side_id.opposite())
}

pub fn refresh_continuous_hp(state: &mut GameState) {
    normalize_board_state(state);
    let basic_hp_bonus = get_stadium_basic_hp_bonus(state);
    for &id in &SideId::ALL {
        refresh_side_continuous_effects(state, id, basic_hp_bonus);
    }
}

pub fn normalize_board_state(state: &mut GameState) {
    for &id in &SideId::ALL {
        normalize_side_board(state.side_mut(id));
    }
}

/// Mirror of `board.ts:29` `switchOutOpponentActive`. Used when an effect
/// gusts the opposing active into the bench. Mutates state.
///
/// Returns true if the switch happened; false if the swap was either
/// queued as a human player choice (gusted side is human) or impossible
/// (no bench candidate).
pub fn switch_out_opponent_active(
    state: &mut GameState,
    acting_side_id: SideId,
    pending_choice_resume: SwitchResume,
) -> bool {
    let opponent_id = acting_side_id.opposite();
    {
        let opponent = state.side(opponent_id);
        if opponent.active.is_none() {
            return false;
        }
        if state.human_by_side[opponent_id as usize] {
            if opponent.bench.is_empty() {
                return false;
            }
            state.pending_player_choice = Some(PendingPlayerChoice::SwitchAfterGust {
                side_id: opponent_id,
                resume: pending_choice_resume,
            });
            return false;
        }
    }

    let opponent = state.side_mut(opponent_id);
    let replacement_index = choose_preferred_active_index(opponent);
    if replacement_index < 0 {
        return false;
    }
    let replacement = opponent.bench.remove(replacement_index as usize);
    let mut switched_out = opponent.active.take().expect("active checked above");
    clear_special_conditions(&mut switched_out);
    // TS pushes to the END of bench; preserve that order.
    opponent.bench.push(switched_out);
    opponent.active = Some(replacement);
    true
}

/// Mirror of `board.ts:57` `choosePreferredActiveIndex`. Returns -1 if no
/// bench. Score formula is exact:
///
/// ```text
/// score = umamusume.hp
///       + attachedEnergyCount(umamusume) * 20
///       + attack.damage
///       - retreatCost(card.retreat) * 6
/// ```
///
/// Ties go to the **lower** bench index because the TS loop uses strict
/// `>` and proceeds in order.
pub fn choose_preferred_active_index(side: &SideState) -> i32 {
    let cat = catalog();
    let mut best_index: i32 = -1;
    let mut best_score: i32 = i32::MIN;
    for (idx, umamusume) in side.bench.iter().enumerate() {
        let Some(card) = cat.get(umamusume.card_id).and_then(|c| match c {
            Card::Umamusume(u) => Some(u),
            _ => None,
        }) else {
            continue;
        };
        let Some(primary) = card.attacks.first() else {
            continue;
        };
        let energy_count = attached_energy_count(umamusume) as i32;
        let cost = retreat_cost(&card.retreat) as i32;
        let score = umamusume.hp + energy_count * 20 + primary.damage - cost * 6;
        if score > best_score {
            best_score = score;
            best_index = idx as i32;
        }
    }
    best_index
}

fn normalize_side_board(side: &mut SideState) {
    let active_uid = side.active.as_ref().map(|a| a.uid);
    let mut clean_bench: ArrayVec<UmamusumeInstance, MAX_BENCH> = ArrayVec::new();
    let mut overflow: Vec<UmamusumeInstance> = Vec::new();
    let mut seen: Vec<u32> = Vec::with_capacity(8);

    for umamusume in side.bench.drain(..) {
        if Some(umamusume.uid) == active_uid || seen.iter().any(|&s| s == umamusume.uid) {
            continue;
        }
        seen.push(umamusume.uid);
        if clean_bench.len() < MAX_BENCH {
            clean_bench.push(umamusume);
        } else {
            overflow.push(umamusume);
        }
    }
    if !overflow.is_empty() {
        for u in overflow {
            // TS: discards the card and any evolution chain + tool card.
            // ArrayVec is bounded; cap at deck size.
            let _ = side.discard.try_push(u.card_id);
            for c in &u.evolution_card_ids {
                let _ = side.discard.try_push(*c);
            }
            if let Some(tool) = u.tool_card_id {
                let _ = side.discard.try_push(tool);
            }
        }
    }
    side.bench = clean_bench;
}

fn get_stadium_basic_hp_bonus(state: &GameState) -> i32 {
    let Some(stadium) = &state.stadium else {
        return 0;
    };
    let cat = catalog();
    let Some(Card::Trainer(t)) = cat.get(stadium.card_id) else {
        return 0;
    };
    t.effect.basic_hp_bonus.unwrap_or(0)
}

fn refresh_side_continuous_effects(state: &mut GameState, side_id: SideId, basic_hp_bonus: i32) {
    let cat = catalog();

    // First pass: compute the active-HP bonus (max over abilities on this side).
    let active_hp_bonus = {
        let side = state.side(side_id);
        let mut best = 0i32;
        for u in get_all_umamusume(side) {
            if let Some(a) = get_umamusume_ability(state, side_id, u) {
                if let Some(bonus) = a.active_hp_bonus {
                    if bonus > best {
                        best = bonus;
                    }
                }
            }
        }
        best
    };

    // Second pass: mutate each umamusume's max_hp + hp.
    let active_uid = state.side(side_id).active.as_ref().map(|a| a.uid);
    let side = state.side_mut(side_id);

    // We need to mutate both active and bench instances. Collect printed
    // HP / stage from catalog first to dodge the borrow conflict.
    let active_meta = side.active.as_ref().map(|a| (a.uid, a.card_id));
    let bench_meta: Vec<(u32, crate::core::card_id::CardId)> =
        side.bench.iter().map(|u| (u.uid, u.card_id)).collect();

    let lookup_printed = |cid: crate::core::card_id::CardId| -> Option<&UmamusumeCard> {
        cat.get(cid).and_then(|c| match c {
            Card::Umamusume(u) => Some(u),
            _ => None,
        })
    };

    let apply = |umamusume: &mut UmamusumeInstance, printed_hp: i32, printed_stage: u8| {
        let stadium_hp_bonus = if printed_stage == 0 {
            basic_hp_bonus
        } else {
            0
        };
        let is_active = Some(umamusume.uid) == active_uid;
        let target_max_hp =
            printed_hp + stadium_hp_bonus + if is_active { active_hp_bonus } else { 0 };
        let damage = umamusume.max_hp - umamusume.hp;
        let next_hp = (target_max_hp - damage).clamp(0, target_max_hp);
        umamusume.max_hp = target_max_hp;
        umamusume.hp = next_hp;
    };

    if let Some((_uid, cid)) = active_meta {
        let (printed_hp, printed_stage) = lookup_printed(cid)
            .map(|c| (c.hp, c.stage))
            .unwrap_or((0, 0));
        if let Some(a) = side.active.as_mut() {
            apply(a, printed_hp, printed_stage);
        }
    }
    for (idx, (_uid, cid)) in bench_meta.iter().enumerate() {
        let (printed_hp, printed_stage) = lookup_printed(*cid)
            .map(|c| (c.hp, c.stage))
            .unwrap_or((0, 0));
        if let Some(u) = side.bench.get_mut(idx) {
            apply(u, printed_hp, printed_stage);
        }
    }
}
