//! Bit-identical port of `frontend/src/game/engine/flow/playRules.ts`.
//!
//! Trainer dispatch and the rainbow uncap crystal evolution branch are
//! wired through `flow::trainers` and `use_rainbow_uncap_crystal` below
//! (Phase 1d).

use crate::core::card_id::CardId;
use crate::core::catalog::{catalog, Card, UmamusumeCard};
use crate::core::constants::{SideId, TrainerType, MAX_BENCH, MAX_HAND};
use crate::core::play_types::{PlayActionKind, PlayActionOutcome, PlayChoices};
use crate::core::state::{GameState, SideState, SwitchResume, UmamusumeInstance};
use crate::core::umamusume::{find_own_umamusume_by_uid, get_all_umamusume};
use crate::flow::evolution::{evolve_umamusume, find_evolution_target, is_valid_evolution_target};
use crate::flow::setup::create_umamusume;
use crate::flow::trainers;

/// Mirror of `getPlayableAction`.
pub fn get_playable_action(
    state: &GameState,
    side: &SideState,
    card_id: CardId,
) -> PlayActionOutcome {
    let cat = catalog();
    let Some(card) = cat.get(card_id) else {
        return PlayActionOutcome::CannotPlay {
            reason: "Unknown card.".into(),
        };
    };
    match card {
        Card::Trainer(t) => {
            use crate::core::constants::TrainerType;
            if t.trainer_type == TrainerType::Supporter && side.used_supporter_this_turn {
                return PlayActionOutcome::CannotPlay {
                    reason: "You already used a Supporter this turn.".into(),
                };
            }
            if t.trainer_type == TrainerType::Tool {
                let Some(target) = get_tool_targets(side).into_iter().next() else {
                    return PlayActionOutcome::CannotPlay {
                        reason: "No Umamusume can hold another Tool.".into(),
                    };
                };
                return PlayActionOutcome::CanPlay(PlayActionKind::AttachTool {
                    target_uid: target.uid,
                });
            }
            if t.trainer_type == TrainerType::Stadium {
                if let Some(active_stadium) = &state.stadium {
                    if let Some(Card::Trainer(existing)) = cat.get(active_stadium.card_id) {
                        if existing.name == t.name {
                            return PlayActionOutcome::CannotPlay {
                                reason: "That Stadium is already in play.".into(),
                            };
                        }
                    }
                }
            }
            let effect = &t.effect;
            if effect.discard_other_card == Some(true) && side.hand.len() < 2 {
                return PlayActionOutcome::CannotPlay {
                    reason: "You need another card to discard.".into(),
                };
            }
            if effect.attach_energy_from_zone_to_bench.is_some() && side.bench.is_empty() {
                return PlayActionOutcome::CannotPlay {
                    reason: "You need a benched Umamusume.".into(),
                };
            }
            if effect.random_basic_umamusume_from_discard == Some(true)
                && !has_basic_umamusume_in_discard(side)
            {
                return PlayActionOutcome::CannotPlay {
                    reason: "You need a Basic Umamusume in discard.".into(),
                };
            }
            if effect.random_basic_umamusume_from_discard == Some(true)
                && side.hand.len() >= MAX_HAND
            {
                return PlayActionOutcome::CannotPlay {
                    reason: "Your hand is full.".into(),
                };
            }
            if effect.search_evolution_umamusume == Some(true) && side.hand.len() >= MAX_HAND {
                return PlayActionOutcome::CannotPlay {
                    reason: "Your hand is full.".into(),
                };
            }
            if effect.search_evolution_umamusume == Some(true)
                && !has_evolution_umamusume_in_deck(side)
            {
                return PlayActionOutcome::CannotPlay {
                    reason: "You need an Evolution Umamusume in deck.".into(),
                };
            }
            if effect.discard_random_opponent_active_energy == Some(true)
                && !opponent_active_has_energy(state, side)
            {
                return PlayActionOutcome::CannotPlay {
                    reason: "Opponent's Active Umamusume has no Energy.".into(),
                };
            }
            if effect.rainbow_uncap_crystal == Some(true)
                && get_rainbow_uncap_targets(state, side).is_empty()
            {
                return PlayActionOutcome::CannotPlay {
                    reason: "No Basic Umamusume can skip to Stage 2.".into(),
                };
            }
            PlayActionOutcome::CanPlay(PlayActionKind::Trainer)
        }
        Card::Umamusume(u) => {
            if u.stage == 0 {
                if side.bench.len() >= MAX_BENCH {
                    return PlayActionOutcome::CannotPlay {
                        reason: "Bench is full.".into(),
                    };
                }
                return PlayActionOutcome::CanPlay(PlayActionKind::BenchBasic);
            }
            match find_evolution_target(state, side, u) {
                Some(t) => PlayActionOutcome::CanPlay(PlayActionKind::Evolve { target_uid: t.uid }),
                None => PlayActionOutcome::CannotPlay {
                    reason: "No eligible evolution target.".into(),
                },
            }
        }
    }
}

pub fn get_tool_targets(side: &SideState) -> Vec<&UmamusumeInstance> {
    get_all_umamusume(side)
        .into_iter()
        .filter(|u| u.tool_card_id.is_none())
        .collect()
}

pub fn get_rainbow_uncap_targets<'a>(
    state: &GameState,
    side: &'a SideState,
) -> Vec<&'a UmamusumeInstance> {
    if is_side_first_turn(state, side.id) {
        return Vec::new();
    }
    get_all_umamusume(side)
        .into_iter()
        .filter(|u| {
            u.stage == 0
                && u.entered_turn != state.turn_number
                && !get_rainbow_uncap_evolution_hand_options(side, u).is_empty()
        })
        .collect()
}

pub fn get_rainbow_uncap_evolution_hand_options<'a>(
    side: &SideState,
    umamusume: &UmamusumeInstance,
) -> Vec<(usize, &'a UmamusumeCard)> {
    let cat = catalog();
    let mut out = Vec::new();
    for (idx, cid) in side.hand.iter().copied().enumerate() {
        if let Some(Card::Umamusume(c)) = cat.get(cid) {
            if c.stage == 2 && c.evolves_from.as_deref() == Some(umamusume.species()) {
                out.push((idx, c));
            }
        }
    }
    out
}

/// Mirror of `resolveCardPlay`.
///
/// Trainer + rainbow-uncap branches dispatch into `flow::trainers` and
/// `use_rainbow_uncap_crystal` below.
#[allow(clippy::too_many_arguments)]
pub fn resolve_card_play(
    state: &mut GameState,
    side_id: crate::core::constants::SideId,
    card_id: CardId,
    card: &Card,
    play: &PlayActionKind,
    choices: &PlayChoices,
) {
    match (card, play) {
        (Card::Umamusume(u), PlayActionKind::BenchBasic) if u.stage == 0 => {
            let new_inst = create_umamusume(card_id, state.turn_number);
            let side = state.side_mut(side_id);
            let _ = side.bench.try_push(new_inst);
        }
        (Card::Umamusume(evolution_card), PlayActionKind::Evolve { target_uid }) => {
            let resolved_target_uid = choices.umamusume_target_uid.unwrap_or(*target_uid);
            let validated = {
                let side = state.side(side_id);
                find_own_umamusume_by_uid(side, resolved_target_uid)
                    .map(|u| is_valid_evolution_target(state, side_id, u, evolution_card))
                    .unwrap_or(false)
            };
            if !validated {
                return;
            }
            let turn_number = state.turn_number;
            let side = state.side_mut(side_id);
            if let Some(active) = side.active.as_mut() {
                if active.uid == resolved_target_uid {
                    evolve_umamusume(turn_number, active, card_id, evolution_card);
                    return;
                }
            }
            for u in side.bench.iter_mut() {
                if u.uid == resolved_target_uid {
                    evolve_umamusume(turn_number, u, card_id, evolution_card);
                    return;
                }
            }
        }
        (Card::Trainer(_t), PlayActionKind::AttachTool { target_uid }) => {
            let resolved_uid = choices.umamusume_target_uid.unwrap_or(*target_uid);
            let side = state.side_mut(side_id);
            if let Some(active) = side.active.as_mut() {
                if active.uid == resolved_uid && active.tool_card_id.is_none() {
                    active.tool_card_id = Some(card_id);
                    return;
                }
            }
            for u in side.bench.iter_mut() {
                if u.uid == resolved_uid && u.tool_card_id.is_none() {
                    u.tool_card_id = Some(card_id);
                    return;
                }
            }
        }
        (Card::Trainer(t), PlayActionKind::Trainer) => {
            // Stadium cards install on the board; they do NOT discard
            // the card (TS `playStadium` early-returns before
            // `side.discard.push`).
            if t.trainer_type == TrainerType::Stadium {
                trainers::play_stadium(state, side_id, card_id, t);
                return;
            }
            // Non-stadium trainers: rainbow-uncap evolves; everything else
            // dispatches into apply_trainer. Both paths still go to
            // discard, and supporters mark the once-per-turn flag.
            if t.effect.rainbow_uncap_crystal == Some(true) {
                use_rainbow_uncap_crystal(
                    state,
                    side_id,
                    choices.umamusume_target_uid,
                    choices.rainbow_evolution_hand_index,
                );
            } else {
                trainers::apply_trainer(state, side_id, card_id, t, choices, SwitchResume::None);
            }
            if t.trainer_type == TrainerType::Supporter {
                state.side_mut(side_id).used_supporter_this_turn = true;
            }
            let _ = state.side_mut(side_id).discard.try_push(card_id);
        }
        _ => {}
    }
}

/// Mirror of `playRules.ts:144` `useRainbowUncapCrystal`. Skips Stage 1 by
/// splicing a Stage 2 evolution card directly out of the hand. Returns
/// true if the evolution applied.
pub fn use_rainbow_uncap_crystal(
    state: &mut GameState,
    side_id: SideId,
    target_uid: Option<u32>,
    evolution_hand_index: Option<usize>,
) -> bool {
    // Resolve target uid from the eligible list.
    let target_uid_resolved: Option<u32> = {
        let side = state.side(side_id);
        let targets = get_rainbow_uncap_targets(state, side);
        if let Some(uid) = target_uid {
            targets.iter().find(|u| u.uid == uid).map(|u| u.uid)
        } else {
            targets.first().map(|u| u.uid)
        }
    };
    let Some(target_uid) = target_uid_resolved else {
        return false;
    };

    // Resolve the evolution-card hand index + card.
    let (hand_index, evolution_card): (usize, UmamusumeCard) = {
        let side = state.side(side_id);
        // We need to look up the target instance for species. The side may
        // currently hold it as active or bench.
        let target = get_all_umamusume(side)
            .into_iter()
            .find(|u| u.uid == target_uid);
        let Some(target) = target else {
            return false;
        };
        let options = get_rainbow_uncap_evolution_hand_options(side, target);
        let chosen = match evolution_hand_index {
            Some(i) => options.iter().find(|(idx, _)| *idx == i).cloned(),
            None => options.first().cloned(),
        };
        match chosen {
            Some((idx, card)) => (idx, card.clone()),
            None => return false,
        }
    };

    // Remove the evolution card from hand and apply the evolution to the
    // target instance in place.
    let cat = catalog();
    let evolution_card_id = match cat.id_for(&evolution_card.id) {
        Some(c) => c,
        None => return false,
    };
    let turn_number = state.turn_number;
    let side = state.side_mut(side_id);
    if hand_index >= side.hand.len() {
        return false;
    }
    let _ = side.hand.remove(hand_index);
    if let Some(active) = side.active.as_mut() {
        if active.uid == target_uid {
            evolve_umamusume(turn_number, active, evolution_card_id, &evolution_card);
            return true;
        }
    }
    for u in side.bench.iter_mut() {
        if u.uid == target_uid {
            evolve_umamusume(turn_number, u, evolution_card_id, &evolution_card);
            return true;
        }
    }
    false
}

fn has_basic_umamusume_in_discard(side: &SideState) -> bool {
    let cat = catalog();
    side.discard
        .iter()
        .any(|&cid| matches!(cat.get(cid), Some(Card::Umamusume(u)) if u.stage == 0))
}

fn has_evolution_umamusume_in_deck(side: &SideState) -> bool {
    let cat = catalog();
    side.deck
        .iter()
        .any(|&cid| matches!(cat.get(cid), Some(Card::Umamusume(u)) if u.stage > 0))
}

fn opponent_active_has_energy(state: &GameState, side: &SideState) -> bool {
    let opp = state.side(side.id.opposite());
    let Some(active) = &opp.active else {
        return false;
    };
    active.energies.iter().any(|&n| n > 0)
}

fn is_side_first_turn(state: &GameState, side_id: crate::core::constants::SideId) -> bool {
    state.turns_taken_by_side[side_id as usize] <= 1
}
