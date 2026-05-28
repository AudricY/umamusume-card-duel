//! Bit-identical port of `frontend/src/game/engine/flow/ai/trainerUtils.ts`.
//!
//! Trainer-play heuristics + evolution-target / tool-target scoring.
//! Floats preserved (e.g. `* 0.7`, `* 0.5`, `* 0.45`, `* 0.35`).

use crate::core::catalog::{catalog, Card, TrainerCard, UmamusumeCard};
use crate::core::constants::{EnergyType, SideId, TrainerType, MAX_BENCH, MAX_HAND};
use crate::core::play_types::{PlayActionOutcome, PlayChoices};
use crate::core::state::{GameState, SideState, UmamusumeInstance};
use crate::core::umamusume::{attached_energy_count, get_all_umamusume};
use crate::flow::eligibility::{can_attack, can_retreat};
use crate::flow::energy::has_enough_energy;
use crate::flow::play_rules::{
    get_playable_action, get_rainbow_uncap_evolution_hand_options, get_rainbow_uncap_targets,
    get_tool_targets,
};
use crate::flow::retreat::{effective_retreat_cost, retreat_cost};
use crate::flow::trainers::has_damaged_healing_target;

use super::combat_utils::can_immediate_opponent_ko_conservative;
use super::deck_inference::{count_consumed_basics, get_known_remaining_deck_counts};
use super::energy_awareness::score_attack_energy_pool_fit;
use super::opponent_heuristics::{
    get_aoi_kiryuin_bonus_value_from_public, get_opponent_active_energy_count_from_public,
    get_yayoi_akikawa_value_from_public,
};
use super::public_info::get_public_opponent_view;
use super::turn_plan::has_consecutive_no_attack_turns;
use super::types::AiTurnGoal;

pub fn should_ai_play_trainer(
    state: &GameState,
    side: &SideState,
    card_id: crate::core::card_id::CardId,
    hand_index: usize,
    turn_goal: AiTurnGoal,
) -> bool {
    let cat = catalog();
    let Some(Card::Trainer(card)) = cat.get(card_id) else {
        return false;
    };
    if !matches!(
        get_playable_action(state, side, card_id),
        PlayActionOutcome::CanPlay(_)
    ) {
        return false;
    }
    let recovery_mode = has_consecutive_no_attack_turns(state, &side.title, 2);
    let bench_fragile = side.bench.len() <= 1;
    let under_threat = can_immediate_opponent_ko_conservative(state, side.id);

    if card.trainer_type == TrainerType::Stadium {
        return should_ai_play_stadium(state, side, card, turn_goal, recovery_mode);
    }
    if recovery_mode {
        if card.effect.search_umamusume == Some(true)
            || card.effect.search_random_basic_umamusume == Some(true)
            || card.effect.draw.is_some()
        {
            return side.hand.len() < MAX_HAND;
        }
        if card.effect.retreat_cost_reduction.is_some()
            || card.effect.active_attack_damage_bonus.is_some()
            || card.effect.gust_opponent == Some(true)
        {
            return false;
        }
    }
    if card.trainer_type == TrainerType::Tool {
        return !get_tool_targets(side).is_empty();
    }
    if card.effect.gust_opponent == Some(true) {
        return get_yayoi_akikawa_value(state, side) > 0.0;
    }
    if card.effect.active_attack_damage_bonus.is_some() {
        if turn_goal == AiTurnGoal::ConvertPointLead {
            return true;
        }
        return get_aoi_kiryuin_bonus_value(state, side) > 0.0;
    }
    if bench_fragile
        && under_threat
        && (card.effect.active_attack_damage_bonus.is_some()
            || card.effect.gust_opponent == Some(true))
    {
        return false;
    }
    if card.effect.discard_random_opponent_active_energy == Some(true) {
        return get_opponent_active_energy_count(state, side) > 0;
    }
    if card.effect.attach_energy_from_zone_to_bench.is_some() {
        return !side.bench.is_empty();
    }
    if card.effect.extra_energy_attach.is_some() {
        return true;
    }
    if let Some(reduction) = card.effect.retreat_cost_reduction {
        let Some(active) = &side.active else {
            return false;
        };
        if side.bench.is_empty() {
            return false;
        }
        if reduction <= 0 {
            return false;
        }
        let retreat_unlocks_now = !can_retreat(state, side)
            && (attached_energy_count(active) as i32) + reduction
                >= effective_retreat_cost(state, side) as i32;
        if !retreat_unlocks_now {
            return false;
        }
        if turn_goal == AiTurnGoal::DenyOpponentLethal {
            if !can_immediate_opponent_ko_conservative(state, side.id) {
                return false;
            }
            return is_retreat_likely_beneficial(state, side);
        }
        if turn_goal == AiTurnGoal::ProtectLoadedActive {
            if !can_immediate_opponent_ko_conservative(state, side.id) {
                return false;
            }
            return is_retreat_likely_beneficial(state, side);
        }
        return is_retreat_likely_beneficial(state, side);
    }
    if card.effect.heal.is_some() && !has_damaged_healing_target(side, card) {
        return card.effect.draw.unwrap_or(0) > 0 && side.hand.len() < MAX_HAND;
    }
    if card.effect.draw.is_some() && side.hand.len() >= MAX_HAND {
        return false;
    }
    if card.effect.draw.is_some()
        && (turn_goal == AiTurnGoal::DigForEvolution
            || turn_goal == AiTurnGoal::BuildBackupAttacker)
    {
        return true;
    }
    if card.effect.search_umamusume == Some(true)
        || card.effect.search_evolution_umamusume == Some(true)
        || card.effect.search_random_basic_umamusume == Some(true)
    {
        if side.hand.len() >= MAX_HAND {
            return false;
        }
        if turn_goal == AiTurnGoal::StabilizeBoard && card.effect.search_umamusume == Some(true) {
            return true;
        }
        if turn_goal == AiTurnGoal::DigForEvolution
            && (card.effect.search_evolution_umamusume == Some(true)
                || card.effect.search_umamusume == Some(true))
        {
            return true;
        }
        if turn_goal == AiTurnGoal::BuildBackupAttacker
            && (card.effect.search_umamusume == Some(true)
                || card.effect.search_random_basic_umamusume == Some(true))
        {
            return true;
        }
        if card.effect.discard_other_card == Some(true) {
            if turn_goal == AiTurnGoal::DenyOpponentLethal && side.hand.len() <= 2 {
                return false;
            }
            let discard_index = choose_ai_discard_hand_index(state, side, hand_index);
            let search_index = choose_ai_search_deck_index(
                state,
                side,
                SearchIndexOpts {
                    evolution_only: false,
                    prefer_basics: turn_goal == AiTurnGoal::StabilizeBoard
                        || turn_goal == AiTurnGoal::BuildBackupAttacker,
                    prefer_evolution_targets: turn_goal == AiTurnGoal::DigForEvolution,
                },
            );
            let (Some(di), Some(si)) = (discard_index, search_index) else {
                return false;
            };
            let Some(&discarded_cid) = side.hand.get(di) else {
                return false;
            };
            let Some(&searched_cid) = side.deck.get(si) else {
                return false;
            };
            let upgrade_gain = score_card_future_value(state, side, searched_cid)
                - score_card_future_value(state, side, discarded_cid);
            return upgrade_gain >= 10.0;
        }
        return true;
    }
    if card.effect.rainbow_uncap_crystal == Some(true) {
        return get_ai_rainbow_uncap_choice(state, side).is_some();
    }
    if card.effect.recover_active_special_conditions == Some(true) {
        return match &side.active {
            Some(a) => !a.special_conditions.is_empty(),
            None => false,
        };
    }
    true
}

fn should_ai_play_stadium(
    state: &GameState,
    side: &SideState,
    card: &TrainerCard,
    turn_goal: AiTurnGoal,
    recovery_mode: bool,
) -> bool {
    if card.trainer_type != TrainerType::Stadium {
        return false;
    }
    if (turn_goal == AiTurnGoal::StabilizeBoard || recovery_mode)
        && card.effect.shuffle_hand_into_deck_draw.is_some()
    {
        return true;
    }
    let candidate_value = evaluate_stadium_net_value(state, side.id, card);
    let cat = catalog();
    if state.stadium.is_none() {
        return candidate_value >= 8.0;
    }
    let active_stadium_id = state.stadium.as_ref().unwrap().card_id;
    let current_value = match cat.get(active_stadium_id) {
        Some(Card::Trainer(t)) if t.trainer_type == TrainerType::Stadium => {
            evaluate_stadium_net_value(state, side.id, t)
        }
        _ => 0.0,
    };
    let improvement = candidate_value - current_value;
    if improvement <= 0.0 {
        return false;
    }
    if state.stadium.as_ref().unwrap().owner == side.id {
        return candidate_value >= 8.0 && improvement >= 12.0;
    }
    candidate_value >= 6.0 && improvement >= 4.0
}

fn evaluate_stadium_net_value(
    state: &GameState,
    perspective_side_id: SideId,
    stadium: &TrainerCard,
) -> f64 {
    let opponent_side_id = perspective_side_id.opposite();
    let own_value = evaluate_stadium_side_value(state, perspective_side_id, stadium);
    let opponent_value = evaluate_stadium_side_value(state, opponent_side_id, stadium);
    own_value - opponent_value
}

fn evaluate_stadium_side_value(state: &GameState, side_id: SideId, stadium: &TrainerCard) -> f64 {
    let side = state.side(side_id);
    let opponent = state.side(side_id.opposite());
    let mut score = 0.0f64;
    let retreat_reduction = stadium.effect.global_retreat_cost_reduction.unwrap_or(0);
    if retreat_reduction > 0 && side.active.is_some() && !side.bench.is_empty() {
        let cat = catalog();
        let active_card = side.active.as_ref().unwrap();
        let printed_retreat = match cat.get(active_card.card_id) {
            Some(Card::Umamusume(u)) => retreat_cost(&u.retreat) as i32,
            _ => 0,
        };
        let retreat_relief = retreat_reduction.min(printed_retreat);
        score += (retreat_relief as f64) * 14.0;
        if !can_retreat(state, side)
            && (attached_energy_count(active_card) as i32) + retreat_reduction
                >= effective_retreat_cost(state, side) as i32
        {
            score += 36.0;
        }
    }
    let basic_hp_bonus = stadium.effect.basic_hp_bonus.unwrap_or(0);
    if basic_hp_bonus > 0 {
        score += (count_basics_in_play(side) as f64) * (basic_hp_bonus as f64) * 0.7;
    }
    if stadium.effect.shuffle_hand_into_deck_draw.is_some() {
        score += score_shuffle_hand_into_deck_draw_value(state, side);
    }
    if stadium.effect.disable_tools == Some(true) {
        score += (count_tools_in_play(opponent) as f64) * 22.0;
        score -= (count_tools_in_play(side) as f64) * 18.0;
    }
    score
}

fn count_basics_in_play(side: &SideState) -> i32 {
    let cat = catalog();
    let mut count = 0i32;
    for u in get_all_umamusume(side) {
        if let Some(Card::Umamusume(c)) = cat.get(u.card_id) {
            if c.stage == 0 {
                count += 1;
            }
        }
    }
    count
}

fn count_tools_in_play(side: &SideState) -> i32 {
    let mut count = 0i32;
    for u in get_all_umamusume(side) {
        if u.tool_card_id.is_some() {
            count += 1;
        }
    }
    count
}

fn score_shuffle_hand_into_deck_draw_value(state: &GameState, side: &SideState) -> f64 {
    let hand_size_score: f64 = if side.hand.len() <= 1 {
        34.0
    } else if side.hand.len() <= 3 {
        22.0
    } else if side.hand.len() <= 5 {
        12.0
    } else {
        4.0
    };
    let attack_penalty: f64 = if can_attack(state, side) { 0.45 } else { 1.0 };
    let already_used_penalty: f64 = if side.used_stadium_this_turn {
        0.35
    } else {
        1.0
    };
    let mut score = hand_size_score * attack_penalty * already_used_penalty;
    let has_bench = !side.bench.is_empty();
    let cat = catalog();
    let has_basic_in_hand = side
        .hand
        .iter()
        .any(|&cid| matches!(cat.get(cid), Some(Card::Umamusume(u)) if u.stage == 0));
    if !has_bench && !has_basic_in_hand {
        let deck_counts = get_known_remaining_deck_counts(side);
        let basics_in_deck = deck_counts.basic_umamusume;
        let consumed_basics = count_consumed_basics(side);
        if basics_in_deck > 0 {
            score += 44.0 + (22.0_f64).min((consumed_basics as f64) * 1.5);
        }
    }
    score
}

pub fn get_ai_trainer_choices(
    state: &GameState,
    side: &SideState,
    card: &TrainerCard,
    hand_index: usize,
    turn_goal: AiTurnGoal,
) -> PlayChoices {
    let mut choices = PlayChoices::default();
    if card.effect.attach_energy_from_zone_to_bench.is_some() {
        if let Some(target) = get_ai_bench_energy_attach_target(state, side) {
            choices.umamusume_target_uid = Some(target.uid);
        }
    }
    if card.effect.discard_other_card == Some(true) {
        if let Some(idx) = choose_ai_discard_hand_index(state, side, hand_index) {
            choices.discard_hand_index = Some(idx);
        }
    }
    if card.effect.search_umamusume == Some(true) {
        if let Some(idx) = choose_ai_search_deck_index(
            state,
            side,
            SearchIndexOpts {
                evolution_only: false,
                prefer_basics: turn_goal == AiTurnGoal::StabilizeBoard
                    || turn_goal == AiTurnGoal::BuildBackupAttacker,
                prefer_evolution_targets: turn_goal == AiTurnGoal::DigForEvolution,
            },
        ) {
            choices.deck_card_index = Some(idx);
        }
    }
    if card.effect.search_evolution_umamusume == Some(true) {
        if let Some(idx) = choose_ai_search_deck_index(
            state,
            side,
            SearchIndexOpts {
                evolution_only: true,
                prefer_basics: false,
                prefer_evolution_targets: true,
            },
        ) {
            choices.deck_card_index = Some(idx);
        }
    }
    if card.trainer_type == TrainerType::Tool {
        if let Some(tool_target) = choose_ai_tool_target(side) {
            choices.umamusume_target_uid = Some(tool_target.uid);
        }
    }
    choices
}

pub fn score_evolution_target(
    state: &GameState,
    side: &SideState,
    target: &UmamusumeInstance,
    evolution_card: &UmamusumeCard,
) -> f64 {
    let cat = catalog();
    let active = side.active.as_ref();
    let target_card = match cat.get(target.card_id) {
        Some(Card::Umamusume(u)) => u,
        _ => return f64::NEG_INFINITY,
    };
    let before_attack = match target_card.attacks.first() {
        Some(a) => a,
        None => return f64::NEG_INFINITY,
    };
    let after_attack = match evolution_card.attacks.first() {
        Some(a) => a,
        None => return f64::NEG_INFINITY,
    };
    let before_can_attack = has_enough_energy(target, &before_attack.cost);
    let after_can_attack = has_enough_energy(target, &after_attack.cost);
    let hp_gain = (evolution_card.hp - target_card.hp).max(0) as f64;
    let mut score: f64 = 0.0;
    if Some(target.uid) == active.map(|a| a.uid) {
        score += 80.0;
    }
    score += hp_gain * 1.1;
    score += ((after_attack.damage - before_attack.damage).max(0) as f64) * 1.8;
    if !before_can_attack && after_can_attack {
        score += 90.0;
    }
    score += (attached_energy_count(target) as f64) * 12.0;
    score += (evolution_card.stage as f64) * 20.0;
    score += score_attack_energy_pool_fit(side, &after_attack.cost);
    if state.ai_deck_style_by_side[side.id as usize] == crate::core::constants::AiDeckStyle::Stall {
        score += hp_gain * 0.5;
    }
    score
}

pub fn get_ai_rainbow_uncap_choice(
    state: &GameState,
    side: &SideState,
) -> Option<RainbowUncapChoice> {
    let targets = get_rainbow_uncap_targets(state, side);
    let mut entries: Vec<(UmamusumeInstance, Vec<(usize, &'static UmamusumeCard)>)> = Vec::new();
    for target in targets {
        let options = get_rainbow_uncap_evolution_hand_options(side, target);
        if !options.is_empty() {
            entries.push((target.clone(), options));
        }
    }
    if entries.is_empty() {
        return None;
    }
    entries.sort_by(|left, right| {
        if right.0.max_hp != left.0.max_hp {
            return right.0.max_hp.cmp(&left.0.max_hp);
        }
        right.1.len().cmp(&left.1.len())
    });
    let best = &entries[0];
    let option = best.1.first()?;
    Some(RainbowUncapChoice {
        target_uid: best.0.uid,
        evolution_hand_index: option.0,
    })
}

#[derive(Debug, Clone, Copy)]
pub struct RainbowUncapChoice {
    pub target_uid: u32,
    pub evolution_hand_index: usize,
}

fn get_ai_bench_energy_attach_target<'a>(
    state: &GameState,
    side: &'a SideState,
) -> Option<&'a UmamusumeInstance> {
    let next_energy_type = side.energy_zone.first().copied();
    let mut sorted: Vec<&UmamusumeInstance> = side.bench.iter().collect();
    sorted.sort_by(|left, right| {
        let l = score_bench_attach_target(state, side, left, next_energy_type);
        let r = score_bench_attach_target(state, side, right, next_energy_type);
        r.partial_cmp(&l).unwrap_or(std::cmp::Ordering::Equal)
    });
    let undercharged = sorted.iter().copied().find(|u| {
        let cat = catalog();
        let Some(Card::Umamusume(c)) = cat.get(u.card_id) else {
            return false;
        };
        let Some(attack) = c.attacks.first() else {
            return false;
        };
        !has_enough_energy(u, &attack.cost)
    });
    undercharged.or_else(|| side.bench.first())
}

fn score_bench_attach_target(
    state: &GameState,
    side: &SideState,
    umamusume: &UmamusumeInstance,
    next_energy_type: Option<EnergyType>,
) -> f64 {
    let cat = catalog();
    let card = match cat.get(umamusume.card_id) {
        Some(Card::Umamusume(c)) => c,
        _ => return f64::NEG_INFINITY,
    };
    let Some(attack) = card.attacks.first() else {
        return f64::NEG_INFINITY;
    };
    let before_can_attack = has_enough_energy(umamusume, &attack.cost);
    let mut after_can_attack = before_can_attack;
    if let Some(next_energy_type) = next_energy_type {
        let mut simulated = umamusume.clone();
        let cur = simulated.energies[next_energy_type as usize];
        simulated.energies[next_energy_type as usize] = cur.saturating_add(1);
        after_can_attack = has_enough_energy(&simulated, &attack.cost);
    }
    let mut score = (umamusume.stage as f64) * 24.0
        + (attack.damage as f64)
        + (attached_energy_count(umamusume) as f64) * 12.0;
    score += score_attack_energy_pool_fit(side, &attack.cost);
    if !before_can_attack && after_can_attack {
        score += 120.0;
    }
    if side.active.is_some() && !can_attack(state, side) && !before_can_attack && after_can_attack {
        score += 80.0;
    }
    score
}

fn choose_ai_tool_target(side: &SideState) -> Option<&UmamusumeInstance> {
    let targets = get_tool_targets(side);
    let mut sorted: Vec<&UmamusumeInstance> = targets;
    sorted.sort_by(|left, right| {
        let left_active = if Some(left.uid) == side.active.as_ref().map(|a| a.uid) {
            1
        } else {
            0
        };
        let right_active = if Some(right.uid) == side.active.as_ref().map(|a| a.uid) {
            1
        } else {
            0
        };
        if right_active != left_active {
            return right_active.cmp(&left_active);
        }
        if right.stage != left.stage {
            return right.stage.cmp(&left.stage);
        }
        (attached_energy_count(right) as i32).cmp(&(attached_energy_count(left) as i32))
    });
    sorted.into_iter().next()
}

fn choose_ai_discard_hand_index(
    state: &GameState,
    side: &SideState,
    excluding_hand_index: usize,
) -> Option<usize> {
    let options: Vec<(crate::core::card_id::CardId, usize)> = side
        .hand
        .iter()
        .enumerate()
        .filter_map(|(idx, &cid)| {
            if idx == excluding_hand_index {
                None
            } else {
                Some((cid, idx))
            }
        })
        .collect();
    if options.is_empty() {
        return None;
    }
    let mut sorted = options;
    sorted.sort_by(|left, right| {
        let l = score_card_future_value(state, side, left.0);
        let r = score_card_future_value(state, side, right.0);
        l.partial_cmp(&r).unwrap_or(std::cmp::Ordering::Equal)
    });
    sorted.first().map(|(_, idx)| *idx)
}

#[derive(Debug, Clone, Copy, Default)]
struct SearchIndexOpts {
    evolution_only: bool,
    prefer_basics: bool,
    prefer_evolution_targets: bool,
}

fn choose_ai_search_deck_index(
    state: &GameState,
    side: &SideState,
    opts: SearchIndexOpts,
) -> Option<usize> {
    let cat = catalog();
    let deck_options: Vec<(crate::core::card_id::CardId, usize)> = side
        .deck
        .iter()
        .enumerate()
        .filter_map(|(idx, &cid)| match cat.get(cid) {
            Some(Card::Umamusume(c)) => {
                if !opts.evolution_only || c.stage > 0 {
                    Some((cid, idx))
                } else {
                    None
                }
            }
            _ => None,
        })
        .collect();
    if deck_options.is_empty() {
        return None;
    }
    if opts.prefer_basics {
        let basic_options: Vec<(crate::core::card_id::CardId, usize)> = deck_options
            .iter()
            .filter(|(cid, _)| matches!(cat.get(*cid), Some(Card::Umamusume(c)) if c.stage == 0))
            .cloned()
            .collect();
        if !basic_options.is_empty() {
            let mut sorted = basic_options;
            sorted.sort_by(|left, right| {
                let l = score_card_future_value(state, side, left.0);
                let r = score_card_future_value(state, side, right.0);
                r.partial_cmp(&l).unwrap_or(std::cmp::Ordering::Equal)
            });
            return sorted.first().map(|(_, idx)| *idx);
        }
    }
    if opts.prefer_evolution_targets {
        let evolution_options: Vec<(crate::core::card_id::CardId, usize)> = deck_options
            .iter()
            .filter(|(cid, _)| is_useful_evolution_search_hit(side, *cid))
            .cloned()
            .collect();
        if !evolution_options.is_empty() {
            let mut sorted = evolution_options;
            sorted.sort_by(|left, right| {
                let l = score_card_future_value(state, side, left.0);
                let r = score_card_future_value(state, side, right.0);
                r.partial_cmp(&l).unwrap_or(std::cmp::Ordering::Equal)
            });
            return sorted.first().map(|(_, idx)| *idx);
        }
    }
    let mut sorted = deck_options;
    sorted.sort_by(|left, right| {
        let l = score_card_future_value(state, side, left.0);
        let r = score_card_future_value(state, side, right.0);
        r.partial_cmp(&l).unwrap_or(std::cmp::Ordering::Equal)
    });
    sorted.first().map(|(_, idx)| *idx)
}

fn is_useful_evolution_search_hit(side: &SideState, card_id: crate::core::card_id::CardId) -> bool {
    let cat = catalog();
    let Some(Card::Umamusume(card)) = cat.get(card_id) else {
        return false;
    };
    if card.stage == 0 {
        return false;
    }
    get_all_umamusume(side).into_iter().any(|target| {
        let evolves_from = card.evolves_from.as_deref();
        target.species() == evolves_from.unwrap_or("")
            && (target.stage as i32) == (card.stage as i32 - 1)
    })
}

fn score_card_future_value(
    state: &GameState,
    side: &SideState,
    card_id: crate::core::card_id::CardId,
) -> f64 {
    let cat = catalog();
    let deck_style = state.ai_deck_style_by_side[side.id as usize];
    let Some(card) = cat.get(card_id) else {
        return 0.0;
    };
    match card {
        Card::Trainer(t) => {
            let mut value: f64 = if t.trainer_type == TrainerType::Supporter {
                36.0
            } else {
                24.0
            };
            if t.effect.draw.is_some() {
                value += 24.0;
            }
            if t.effect.active_attack_damage_bonus.is_some() {
                value += get_aoi_kiryuin_bonus_value(state, side);
            }
            if t.effect.gust_opponent == Some(true) {
                value += get_yayoi_akikawa_value(state, side);
            }
            if t.effect.discard_random_opponent_active_energy == Some(true) {
                value += (get_opponent_active_energy_count(state, side) as f64) * 18.0;
            }
            if t.effect.rainbow_uncap_crystal == Some(true) {
                value += if get_ai_rainbow_uncap_choice(state, side).is_some() {
                    42.0
                } else {
                    0.0
                };
            }
            value
        }
        Card::Umamusume(u) => {
            let attack = match u.attacks.first() {
                Some(a) => a,
                None => return 0.0,
            };
            let mut value: f64 = 30.0 + (u.stage as f64) * 18.0 + (attack.damage as f64) * 0.7;
            value += score_attack_energy_pool_fit(side, &attack.cost);
            let evolution_targets: Vec<&UmamusumeInstance> = get_all_umamusume(side)
                .into_iter()
                .filter(|um| {
                    let evolves_from = u.evolves_from.as_deref().unwrap_or("");
                    um.species() == evolves_from && (um.stage as i32) == (u.stage as i32 - 1)
                })
                .collect();
            if u.stage > 0 && !evolution_targets.is_empty() {
                value += 70.0;
            }
            if let Some(active) = &side.active {
                if u.evolves_from.as_deref() == Some(active.species())
                    && (u.stage as i32) == (active.stage as i32 + 1)
                {
                    value += 45.0;
                }
            }
            if u.stage == 0 && side.bench.len() < MAX_BENCH {
                value += 26.0;
            }
            if deck_style == crate::core::constants::AiDeckStyle::ScaleBench
                && u.species == "Agnes Digital"
            {
                value += 85.0;
            }
            if deck_style == crate::core::constants::AiDeckStyle::ScaleBench && u.stage == 0 {
                value += 24.0;
            }
            value
        }
    }
}

fn get_opponent_active_energy_count(state: &GameState, side: &SideState) -> i32 {
    let opponent = get_public_opponent_view(state, side.id);
    get_opponent_active_energy_count_from_public(&opponent)
}

fn get_aoi_kiryuin_bonus_value(state: &GameState, side: &SideState) -> f64 {
    let Some(active) = &side.active else {
        return 0.0;
    };
    if !can_attack(state, side) {
        return 0.0;
    }
    let opponent = get_public_opponent_view(state, side.id);
    get_aoi_kiryuin_bonus_value_from_public(
        active,
        &opponent,
        side.active_attack_damage_bonus as i32,
        side.bench.len() as i32,
        state.turn_number,
    )
}

fn get_yayoi_akikawa_value(state: &GameState, side: &SideState) -> f64 {
    let Some(active) = &side.active else {
        return 0.0;
    };
    if !can_attack(state, side) {
        return 0.0;
    }
    let opponent = get_public_opponent_view(state, side.id);
    get_yayoi_akikawa_value_from_public(
        active,
        &opponent,
        side.active_attack_damage_bonus as i32,
        side.bench.len() as i32,
        state.turn_number,
    )
}

fn is_retreat_likely_beneficial(state: &GameState, side: &SideState) -> bool {
    let _ = state;
    let Some(active) = &side.active else {
        return false;
    };
    let cat = catalog();
    let active_attack = match cat.get(active.card_id) {
        Some(Card::Umamusume(u)) => match u.attacks.first() {
            Some(a) => a.clone(),
            None => return false,
        },
        _ => return false,
    };
    let active_can_attack = has_enough_energy(active, &active_attack.cost);
    side.bench.iter().any(|bench| {
        let Some(Card::Umamusume(bc)) = cat.get(bench.card_id) else {
            return false;
        };
        let Some(bench_attack) = bc.attacks.first() else {
            return false;
        };
        let bench_can_attack = has_enough_energy(bench, &bench_attack.cost);
        if !active_can_attack && bench_can_attack {
            return true;
        }
        if bench.max_hp - active.max_hp >= 20 {
            return true;
        }
        if bench.hp - active.hp >= 20 {
            return true;
        }
        bench.stage > active.stage && bench.hp >= active.hp
    })
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn should_ai_play_trainer_rejects_non_trainer_card() {
        // Pass a known basic umamusume card id; should return false because
        // the catalog entry is Card::Umamusume, not Card::Trainer.
        let state = empty_state();
        let side = state.side(SideId::Player);
        let cid = crate::core::catalog::catalog()
            .id_for("matikanetannhauserBasic")
            .expect("base card present");
        assert!(!should_ai_play_trainer(
            &state,
            side,
            cid,
            0,
            AiTurnGoal::MaximizeProgress,
        ));
    }

    #[test]
    fn score_evolution_target_active_target_gets_active_bonus() {
        // Score is purely a function of catalog data + the instance fields;
        // there is no RNG. With a fresh basic instance as the target, and
        // a Stage-1 card from the catalog, we expect a finite score.
        let cat = crate::core::catalog::catalog();
        let basic_cid = cat
            .id_for("matikanetannhauserBasic")
            .expect("basic present");
        let stage1_cid = cat
            .id_for("matikanetannhauserStage1")
            .expect("stage1 present");
        let basic_card = match cat.get(basic_cid).unwrap() {
            Card::Umamusume(u) => u,
            _ => panic!("expected umamusume"),
        };
        let stage1_card = match cat.get(stage1_cid).unwrap() {
            Card::Umamusume(u) => u,
            _ => panic!("expected umamusume"),
        };
        let mut state = empty_state();
        let target = UmamusumeInstance {
            uid: 1,
            card_id: basic_cid,
            evolution_card_ids: ArrayVec::new(),
            stage: basic_card.stage,
            hp: basic_card.hp,
            max_hp: basic_card.hp,
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
        state.sides[SideId::Player as usize].active = Some(target.clone());
        let side = state.side(SideId::Player);
        let score_active =
            score_evolution_target(&state, side, side.active.as_ref().unwrap(), stage1_card);
        // With an active-uid match and matching evolution, the score
        // should at least include the +80 active bonus contribution.
        assert!(
            score_active >= 80.0,
            "expected active bonus to lift score, got {}",
            score_active
        );
    }
}
