//! Bit-identical port of `frontend/src/game/engine/flow/trainers.ts`.
//!
//! RNG sites (must consume in exact source order):
//! - `useStadium`: one `shuffle(deck + hand)` per call.
//! - `applyTrainer.extraEnergyAttach`: one `rollEnergyFromPool` per
//!   attached energy.
//! - `attachEnergyFromZoneToBench`: one `rollEnergyFromPool` per attached
//!   energy.
//! - `searchRandomBasicUmamusumeFromDeck`: one `randomInt(candidates.len())`.
//! - `moveRandomBasicUmamusumeFromDiscardToHand`: one
//!   `randomInt(candidates.len())`.
//! - `discardRandomOpponentActiveEnergy`: one `randomInt(energyPool.len())`.
//!
//! Logging is skipped (fingerprint excludes `state.log`).

use crate::core::card_id::CardId;
use crate::core::catalog::{catalog, Card, TrainerCard};
use crate::core::constants::{
    EnergyType, SideId, TrainerType, MAX_HAND,
};
use crate::core::effects::TrainerHealTarget;
use crate::core::play_types::PlayChoices;
use crate::core::random::random_int;
use crate::core::random::shuffle;
use crate::core::state::{GameState, SwitchResume, UmamusumeInstance};
use crate::core::umamusume::{find_most_damaged_umamusume, find_own_umamusume_by_uid};
use crate::flow::board::switch_out_opponent_active;
use crate::flow::special_conditions::clear_special_conditions;
use crate::flow::turn::draw_cards;

/// `trainers.ts:16` `playStadium`. Discards the previous stadium card
/// (if any) into its owner's discard pile, then installs the new one.
pub fn play_stadium(state: &mut GameState, side_id: SideId, _card_id: CardId, stadium: &TrainerCard) {
    if let Some(previous) = state.stadium.clone() {
        let owner_side = state.side_mut(previous.owner);
        let _ = owner_side.discard.try_push(previous.card_id);
    }
    let new_card_id = catalog()
        .id_for(&stadium.id)
        .expect("stadium card must be interned");
    state.stadium = Some(crate::core::state::StadiumState {
        card_id: new_card_id,
        owner: side_id,
    });
}

/// `trainers.ts:28` `canUseStadium`.
pub fn can_use_stadium(state: &GameState, side_id: SideId) -> bool {
    if !matches!(state.phase, crate::core::state::Phase::Play)
        || state.pending_player_choice.is_some()
        || state.game_over
        || state.current_side != crate::core::state::CurrentSide::from_side(side_id)
    {
        return false;
    }
    let side = state.side(side_id);
    if side.used_stadium_this_turn {
        return false;
    }
    let Some(stadium) = &state.stadium else {
        return false;
    };
    let Some(Card::Trainer(t)) = catalog().get(stadium.card_id) else {
        return false;
    };
    if t.trainer_type != TrainerType::Stadium {
        return false;
    }
    matches!(t.effect.shuffle_hand_into_deck_draw, Some(n) if n > 0)
}

/// `trainers.ts:37` `useStadium`. RNG: one `shuffle`.
pub fn use_stadium(state: &mut GameState, side_id: SideId) -> bool {
    if !can_use_stadium(state, side_id) {
        return false;
    }
    let stadium_card_id = match &state.stadium {
        Some(s) => s.card_id,
        None => return false,
    };
    let draw_amount = match catalog().get(stadium_card_id) {
        Some(Card::Trainer(t)) => t.effect.shuffle_hand_into_deck_draw.unwrap_or(0),
        _ => return false,
    };
    if draw_amount <= 0 {
        return false;
    }
    {
        let side = state.side_mut(side_id);
        let mut all: Vec<CardId> = side.deck.iter().copied().collect();
        all.extend(side.hand.iter().copied());
        let shuffled = shuffle(&all);
        side.deck.clear();
        for c in shuffled {
            let _ = side.deck.try_push(c);
        }
        side.hand.clear();
    }
    let side = state.side_mut(side_id);
    let _ = draw_cards(side, draw_amount as u32);
    side.used_stadium_this_turn = true;
    true
}

/// `trainers.ts:57` `applyTrainer`. The big dispatch.
///
/// `switch_out_opponent_active_fn` callback intentionally NOT a separate
/// parameter — Rust ports `gust_opponent` directly to
/// `flow::board::switch_out_opponent_active`, the only implementation.
pub fn apply_trainer(
    state: &mut GameState,
    side_id: SideId,
    _card_id: CardId,
    trainer: &TrainerCard,
    choices: &PlayChoices,
    pending_choice_resume: SwitchResume,
) {
    // `discardOtherCard` runs FIRST in TS — captured for the trailing
    // log line in the original, but the side effect (splice + push to
    // discard) is what matters here.
    let _discarded_card_id: Option<CardId> = if trainer.effect.discard_other_card == Some(true) {
        discard_other_card_for_scout(state, side_id, choices.discard_hand_index)
    } else {
        None
    };

    if let Some(amount) = trainer.effect.retreat_cost_reduction {
        let side = state.side_mut(side_id);
        side.retreat_cost_reduction = (side.retreat_cost_reduction as i32 + amount).max(0) as u8;
    }
    if let Some(amount) = trainer.effect.active_attack_damage_bonus {
        let side = state.side_mut(side_id);
        side.active_attack_damage_bonus =
            (side.active_attack_damage_bonus as i32 + amount).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
    if let Some(extra) = trainer.effect.extra_energy_attach {
        // Snapshot the pool, generate `extra` rolls, then bulk-push.
        let side = state.side_mut(side_id);
        side.bonus_energy_attachments = side.bonus_energy_attachments.saturating_add(extra as u8);
        for _ in 0..extra {
            let pool: Vec<EnergyType> = side.energy_pool.iter().copied().collect();
            let idx = if pool.is_empty() {
                0
            } else {
                random_int(pool.len() as u32) as usize
            };
            let energy = EnergyType::from_pool_index(&pool, idx);
            let _ = side.energy_zone.try_push(energy);
        }
    }
    if let Some(count) = trainer.effect.attach_energy_from_zone_to_bench {
        attach_energy_from_zone_to_bench(state, side_id, count, choices.umamusume_target_uid);
    }
    if trainer.effect.gust_opponent == Some(true) {
        switch_out_opponent_active(state, side_id, pending_choice_resume);
    }
    if let Some(heal_amount) = trainer.effect.heal {
        // Resolve the heal target uid.
        let target_uid: Option<u32> = {
            let side = state.side(side_id);
            let chosen = choices
                .umamusume_target_uid
                .and_then(|uid| find_own_umamusume_by_uid(side, uid))
                .map(|u| u.uid);
            match trainer.effect.heal_target {
                Some(TrainerHealTarget::Any) => {
                    chosen.or_else(|| find_most_damaged_umamusume(side).map(|u| u.uid))
                }
                // healTarget == "active" or undefined → active only.
                _ => side.active.as_ref().map(|a| a.uid),
            }
        };
        if let Some(uid) = target_uid {
            let side = state.side_mut(side_id);
            let touch = |u: &mut UmamusumeInstance| {
                u.hp = (u.hp + heal_amount).min(u.max_hp);
            };
            if let Some(active) = side.active.as_mut() {
                if active.uid == uid {
                    touch(active);
                }
            }
            for b in side.bench.iter_mut() {
                if b.uid == uid {
                    touch(b);
                }
            }
        }
    }
    if let Some(n) = trainer.effect.draw {
        let side = state.side_mut(side_id);
        let _ = draw_cards(side, n as u32);
    }
    if trainer.effect.search_umamusume == Some(true) {
        search_umamusume_from_deck(state, side_id, choices.deck_card_index);
    }
    if trainer.effect.search_evolution_umamusume == Some(true) {
        search_evolution_umamusume_from_deck(state, side_id, choices.deck_card_index);
    }
    if trainer.effect.search_random_basic_umamusume == Some(true) {
        search_random_basic_umamusume_from_deck(state, side_id);
    }
    if trainer.effect.random_basic_umamusume_from_discard == Some(true) {
        move_random_basic_umamusume_from_discard_to_hand(state, side_id);
    }
    if trainer.effect.discard_random_opponent_active_energy == Some(true) {
        discard_random_opponent_active_energy(state, side_id);
    }
    if trainer.effect.recover_active_special_conditions == Some(true) {
        recover_active_special_conditions(state, side_id);
    }
}

/// `trainers.ts:117` `hasDamagedHealingTarget`.
pub fn has_damaged_healing_target(side: &crate::core::state::SideState, card: &TrainerCard) -> bool {
    let mut candidates: Vec<&UmamusumeInstance> = Vec::new();
    if let Some(a) = &side.active {
        candidates.push(a);
    }
    if card.effect.heal_target == Some(TrainerHealTarget::Any) {
        for b in &side.bench {
            candidates.push(b);
        }
    }
    candidates.iter().any(|u| u.hp < u.max_hp)
}

fn attach_energy_from_zone_to_bench(
    state: &mut GameState,
    side_id: SideId,
    count: i32,
    target_uid: Option<u32>,
) {
    if count <= 0 {
        return;
    }
    if state.side(side_id).bench.is_empty() {
        return;
    }
    let chosen_uid: u32 = {
        let side = state.side(side_id);
        target_uid
            .and_then(|uid| side.bench.iter().find(|u| u.uid == uid).map(|u| u.uid))
            .unwrap_or_else(|| side.bench[0].uid)
    };
    for _ in 0..count {
        let pool: Vec<EnergyType> = state.side(side_id).energy_pool.iter().copied().collect();
        let idx = if pool.is_empty() {
            0
        } else {
            random_int(pool.len() as u32) as usize
        };
        let energy = EnergyType::from_pool_index(&pool, idx);
        let side = state.side_mut(side_id);
        for u in side.bench.iter_mut() {
            if u.uid == chosen_uid {
                let cur = u.energies[energy as usize];
                u.energies[energy as usize] = cur.saturating_add(1);
                break;
            }
        }
    }
}

fn discard_other_card_for_scout(
    state: &mut GameState,
    side_id: SideId,
    discard_hand_index: Option<usize>,
) -> Option<CardId> {
    let side = state.side_mut(side_id);
    let idx = discard_hand_index.unwrap_or(0);
    if idx >= side.hand.len() {
        return None;
    }
    let discarded = side.hand.remove(idx);
    let _ = side.discard.try_push(discarded);
    Some(discarded)
}

fn search_umamusume_from_deck(
    state: &mut GameState,
    side_id: SideId,
    deck_card_index: Option<usize>,
) {
    let cat = catalog();
    let side = state.side(side_id);
    let index: i32 = match deck_card_index {
        Some(i) if side.deck.get(i).map(|&cid| cat.is_umamusume(cid)).unwrap_or(false) => {
            i as i32
        }
        _ => side
            .deck
            .iter()
            .position(|&cid| cat.is_umamusume(cid))
            .map(|i| i as i32)
            .unwrap_or(-1),
    };
    move_deck_card_to_hand(state, side_id, index);
}

fn search_evolution_umamusume_from_deck(
    state: &mut GameState,
    side_id: SideId,
    deck_card_index: Option<usize>,
) {
    let cat = catalog();
    let is_evolution = |cid: CardId| -> bool {
        matches!(cat.get(cid), Some(Card::Umamusume(u)) if u.stage > 0)
    };
    let side = state.side(side_id);
    let index: i32 = match deck_card_index {
        Some(i) if side.deck.get(i).copied().map(is_evolution).unwrap_or(false) => i as i32,
        _ => side
            .deck
            .iter()
            .position(|&cid| is_evolution(cid))
            .map(|i| i as i32)
            .unwrap_or(-1),
    };
    move_deck_card_to_hand(state, side_id, index);
}

fn search_random_basic_umamusume_from_deck(state: &mut GameState, side_id: SideId) {
    let cat = catalog();
    let candidates: Vec<usize> = state
        .side(side_id)
        .deck
        .iter()
        .enumerate()
        .filter(|(_, &cid)| cat.is_basic_umamusume(cid))
        .map(|(i, _)| i)
        .collect();
    let chosen_index: i32 = if candidates.is_empty() {
        // TS still calls randomInt(0) here — which is a no-op (returns 0)
        // but it does NOT advance the RNG (random_int(0) early-returns).
        // Mirror that exactly: do nothing, just return -1.
        // Then `chosen?.index ?? -1` yields -1.
        let _ = random_int(0);
        -1
    } else {
        let pick = random_int(candidates.len() as u32) as usize;
        candidates.get(pick).copied().map(|i| i as i32).unwrap_or(-1)
    };
    move_deck_card_to_hand(state, side_id, chosen_index);
}

fn move_random_basic_umamusume_from_discard_to_hand(state: &mut GameState, side_id: SideId) {
    if state.side(side_id).hand.len() >= MAX_HAND {
        return;
    }
    let cat = catalog();
    let candidates: Vec<usize> = state
        .side(side_id)
        .discard
        .iter()
        .enumerate()
        .filter(|(_, &cid)| cat.is_basic_umamusume(cid))
        .map(|(i, _)| i)
        .collect();
    if candidates.is_empty() {
        // TS: randomInt(0) — random_int(0) early-returns without
        // advancing the RNG, matching TS.
        let _ = random_int(0);
        return;
    }
    let pick = random_int(candidates.len() as u32) as usize;
    let Some(&deck_index) = candidates.get(pick) else {
        return;
    };
    let side = state.side_mut(side_id);
    if deck_index >= side.discard.len() {
        return;
    }
    let card = side.discard.remove(deck_index);
    let _ = side.hand.try_push(card);
}

fn discard_random_opponent_active_energy(state: &mut GameState, side_id: SideId) {
    let opponent_id = side_id.opposite();
    // Build the flattened energy pool. TS iterates `Object.entries(active.energies)`
    // which preserves source-declaration order — i.e. `EnergyType::ALL`.
    let energy_pool: Vec<EnergyType> = {
        let opp = state.side(opponent_id);
        let Some(active) = &opp.active else {
            return;
        };
        let mut out: Vec<EnergyType> = Vec::new();
        for energy_type in EnergyType::ALL.iter().copied() {
            let count = active.energies[energy_type as usize];
            for _ in 0..count {
                out.push(energy_type);
            }
        }
        out
    };
    if energy_pool.is_empty() {
        // TS: `energyPool[randomInt(0)]` — random_int(0) returns 0 without
        // advancing the RNG, then the lookup yields undefined.
        let _ = random_int(0);
        return;
    }
    let idx = random_int(energy_pool.len() as u32) as usize;
    let Some(energy) = energy_pool.get(idx).copied() else {
        return;
    };
    let opp = state.side_mut(opponent_id);
    if let Some(active) = opp.active.as_mut() {
        let cur = active.energies[energy as usize] as i32;
        active.energies[energy as usize] = (cur - 1).max(0) as u16;
    }
}

fn recover_active_special_conditions(state: &mut GameState, side_id: SideId) {
    let side = state.side_mut(side_id);
    let Some(active) = side.active.as_mut() else {
        return;
    };
    if active.special_conditions.is_empty() {
        return;
    }
    clear_special_conditions(active);
}

fn move_deck_card_to_hand(state: &mut GameState, side_id: SideId, deck_index: i32) {
    let side = state.side_mut(side_id);
    if deck_index < 0 || side.hand.len() >= MAX_HAND {
        return;
    }
    let idx = deck_index as usize;
    if idx >= side.deck.len() {
        return;
    }
    let card = side.deck.remove(idx);
    let _ = side.hand.try_push(card);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::card_id::CardId;
    use crate::core::constants::{
        AiDeckStyle, AiDifficulty, EnergyType, SideId,
    };
    use crate::core::random::{with_rng, Rng};
    use crate::core::state::{CurrentSide, Phase, SideState};
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
            turn_number: 1,
            first_player: SideId::Player,
            turns_taken_by_side: [0, 0],
            ai_difficulty: AiDifficulty::Normal,
            human_by_side: [false, false],
            ai_deck_style_by_side: [AiDeckStyle::Balanced, AiDeckStyle::Balanced],
            game_over: false,
            winner: None,
            log: std::collections::VecDeque::new(),
        }
    }

    fn dummy_instance(uid: u32) -> UmamusumeInstance {
        UmamusumeInstance {
            uid,
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
        }
    }

    #[test]
    fn discard_random_opponent_active_energy_consumes_one_rng_draw() {
        let mut state = empty_state();
        let mut active = dummy_instance(7);
        active.energies[EnergyType::Grass as usize] = 1;
        active.energies[EnergyType::Fire as usize] = 1;
        state.sides[SideId::Opponent as usize].active = Some(active);

        let rng = Rng::from_seed(1u32, "root");
        let (_, _used) = with_rng(rng, || {
            discard_random_opponent_active_energy(&mut state, SideId::Player);
        });
        let total: i32 = state
            .side(SideId::Opponent)
            .active
            .as_ref()
            .unwrap()
            .energies
            .iter()
            .map(|&n| n as i32)
            .sum();
        assert_eq!(total, 1, "one energy must be removed");
    }

    #[test]
    fn move_deck_card_to_hand_respects_max_hand() {
        let mut state = empty_state();
        // Fill hand to MAX_HAND with a sentinel id.
        for _ in 0..MAX_HAND {
            state.sides[SideId::Player as usize]
                .hand
                .push(CardId(99));
        }
        state.sides[SideId::Player as usize].deck.push(CardId(7));
        move_deck_card_to_hand(&mut state, SideId::Player, 0);
        // Hand is already full; the deck card must not move.
        assert_eq!(state.side(SideId::Player).deck.len(), 1);
        assert_eq!(state.side(SideId::Player).hand.len(), MAX_HAND);
    }
}
