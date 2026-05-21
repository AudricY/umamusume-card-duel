//! Bit-identical port of `frontend/src/game/engine/flow/playRules.ts`.
//!
//! **PARTIAL**: trainer-card resolution and the rainbow uncap crystal
//! evolution branch call into `flow::trainers` (Phase 1d remaining work).
//! Both are stubbed below with `#[allow(unused_variables)]` placeholders
//! marked TODO so the next session can land them without restructuring.
//!
//! The bench/evolve/attach-tool branches are fully ported.

use crate::core::card_id::CardId;
use crate::core::catalog::{catalog, Card, UmamusumeCard};
use crate::core::constants::{MAX_BENCH, MAX_HAND};
use crate::core::play_types::{PlayActionKind, PlayActionOutcome, PlayChoices};
use crate::core::state::{GameState, SideState, UmamusumeInstance};
use crate::core::umamusume::{find_own_umamusume_by_uid, get_all_umamusume};
use crate::flow::evolution::{evolve_umamusume, find_evolution_target, is_valid_evolution_target};
use crate::flow::setup::create_umamusume;

/// Mirror of `getPlayableAction`.
pub fn get_playable_action(state: &GameState, side: &SideState, card_id: CardId) -> PlayActionOutcome {
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
            if effect.random_basic_umamusume_from_discard == Some(true) && side.hand.len() >= MAX_HAND
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
            if c.stage == 2 && c.evolves_from.as_deref() == Some(umamusume.species.as_str()) {
                out.push((idx, c));
            }
        }
    }
    out
}

/// Mirror of `resolveCardPlay`.
///
/// **PARTIAL**: trainer + rainbow-uncap branches require `flow::trainers`
/// (not yet ported). The bench/evolve/attachTool branches are full.
#[allow(clippy::too_many_arguments)]
pub fn resolve_card_play(
    state: &mut GameState,
    side_id: crate::core::constants::SideId,
    card_id: CardId,
    card: &Card,
    play: &PlayActionKind,
    choices: &PlayChoices,
    // TODO Phase 1d: thread a closure for trainers when that module lands.
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
        (Card::Trainer(_t), PlayActionKind::Trainer) => {
            // TODO Phase 1d: dispatch into flow::trainers::play_stadium or
            // flow::trainers::apply_trainer (with the rainbow-uncap branch).
            // For now, no-op + side.discard.push(card_id) to keep the deck
            // accounting roughly sane in tests. Production gate requires
            // the full trainer logic.
            let side = state.side_mut(side_id);
            let _ = side.discard.try_push(card_id);
        }
        _ => {}
    }
}

fn has_basic_umamusume_in_discard(side: &SideState) -> bool {
    let cat = catalog();
    side.discard.iter().any(|&cid| matches!(cat.get(cid), Some(Card::Umamusume(u)) if u.stage == 0))
}

fn has_evolution_umamusume_in_deck(side: &SideState) -> bool {
    let cat = catalog();
    side.deck.iter().any(|&cid| matches!(cat.get(cid), Some(Card::Umamusume(u)) if u.stage > 0))
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
