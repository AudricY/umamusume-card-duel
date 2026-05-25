//! Bit-identical port of `frontend/src/game/engine/flow/ai/combatPlanner.ts`.
//!
//! Builds the full list of combat candidates by cross-producting attack
//! targets × heal targets × shuffle/discard/evolution/random/switch
//! choices, then scores each via `score_candidate` (which clones the state
//! and replays the attack).
//!
//! Float math weights (`BASE_DAMAGE_DEALT_WEIGHT = 1.2`, etc.) preserved
//! verbatim. `(heads.score + tails.score) / 2` averaging mirrors TS
//! source-text order.

use crate::core::catalog::{catalog, Card};
use crate::core::constants::{CoinFlipResult, SideId};
use crate::core::state::{GameState, SideState};
use crate::core::umamusume::{attached_energy_count, get_all_umamusume};
use crate::flow::combat::{perform_attack, CombatDeps};
use crate::flow::eligibility::can_retreat;
use crate::flow::energy::has_enough_energy;
use crate::flow::retreat::{effective_retreat_cost, pay_retreat_cost};

use super::combat_utils::{
    build_attack_decision, can_immediate_opponent_ko, can_immediate_opponent_ko_conservative,
    count_discarded_umamusume, did_candidate_ko_target, get_damage_dealt, get_healing_gained,
    get_target_value,
};
use super::types::{AiCombatDecision, AttackDecision, CombatCandidate};

const BASE_POINTS_WEIGHT: f64 = 1000.0;
const BASE_KO_WEIGHT: f64 = 260.0;
const BASE_DAMAGE_DEALT_WEIGHT: f64 = 1.2;
const BASE_DAMAGE_TAKEN_WEIGHT: f64 = 1.0;
const BASE_HEAL_GAINED_WEIGHT: f64 = 0.8;
const BASE_ACTIVE_KO_BONUS: f64 = 35.0;

pub fn build_combat_candidates(
    state: &GameState,
    side_id: SideId,
    deps: &mut CombatDeps<'_>,
    forced_attack_coin_result: Option<Vec<CoinFlipResult>>,
) -> Vec<CombatCandidate> {
    let mut candidates: Vec<CombatCandidate> = Vec::new();

    if can_use_any_attack(state, state.side(side_id)) {
        candidates.extend(build_attack_candidates(
            state,
            side_id,
            deps,
            forced_attack_coin_result.clone(),
            None,
        ));
    }

    if can_retreat(state, state.side(side_id)) {
        let bench_uids: Vec<u32> = state
            .side(side_id)
            .bench
            .iter()
            .map(|u| u.uid)
            .collect();
        for retreat_uid in bench_uids {
            let mut simulated_retreat = state.clone();
            if !ai_retreat_to_target(&mut simulated_retreat, side_id, retreat_uid) {
                continue;
            }
            if !can_use_any_attack(&simulated_retreat, simulated_retreat.side(side_id)) {
                continue;
            }
            candidates.extend(build_attack_candidates(
                &simulated_retreat,
                side_id,
                deps,
                forced_attack_coin_result.clone(),
                Some(retreat_uid),
            ));
        }
    }

    if candidates.is_empty() {
        candidates.push(score_candidate(
            state,
            side_id,
            deps,
            AiCombatDecision::EndTurn,
            "end-turn".to_string(),
            None,
        ));
    }
    candidates
}

/// `combatPlanner.ts:58` `aiRetreatToTarget`. Mutates state.
pub fn ai_retreat_to_target(state: &mut GameState, side_id: SideId, target_uid: u32) -> bool {
    let retreat_cost = effective_retreat_cost(state, state.side(side_id));
    let side = state.side_mut(side_id);
    if side.active.is_none() {
        return false;
    }
    let target_index = side
        .bench
        .iter()
        .position(|u| u.uid == target_uid);
    let Some(target_index) = target_index else {
        return false;
    };
    if attached_energy_count(side.active.as_ref().unwrap()) < retreat_cost {
        return false;
    }
    {
        let active = side.active.as_mut().unwrap();
        pay_retreat_cost(active, retreat_cost);
    }
    let promoted = side.bench.remove(target_index);
    // TS: side.bench.push(active); side.active = promoted.
    let prev_active = side.active.take().unwrap();
    let _ = side.bench.try_push(prev_active);
    side.active = Some(promoted);
    side.used_retreat_this_turn = true;
    true
}

fn build_attack_candidates(
    state: &GameState,
    side_id: SideId,
    deps: &mut CombatDeps<'_>,
    forced_attack_coin_result: Option<Vec<CoinFlipResult>>,
    retreat_target_uid: Option<u32>,
) -> Vec<CombatCandidate> {
    let cat = catalog();
    let side = state.side(side_id);
    let Some(active) = &side.active else {
        return Vec::new();
    };
    let card = match cat.get(active.card_id) {
        Some(Card::Umamusume(u)) => u.clone(),
        _ => return Vec::new(),
    };
    let opponent = state.side(side_id.opposite());
    let mut out: Vec<CombatCandidate> = Vec::new();

    use crate::core::effects::{AttackTarget, HealTarget};

    for (attack_index, attack) in card.attacks.iter().enumerate() {
        if !has_enough_energy(active, &attack.cost) {
            continue;
        }
        let attack_target_uids: Vec<Option<u32>> = match attack.target_opponent {
            Some(AttackTarget::Any) => {
                let v: Vec<Option<u32>> = get_all_umamusume(opponent)
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
        let heal_target_uids: Vec<Option<u32>> = match (attack.heal, attack.heal_target) {
            (Some(_), Some(HealTarget::Any)) => {
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
            }
            _ => vec![None],
        };
        let uses_coin_flip = attack.coin_bonus.is_some()
            || attack.draw_on_heads.is_some()
            || attack.discard_random_opponent_hand_on_heads.is_some()
            || attack.knock_out_active_if_all_coin_heads.is_some();
        let shuffle_options: Vec<Option<bool>> = if attack.shuffle_self_into_deck.is_some() {
            vec![Some(false), Some(true)]
        } else {
            vec![None]
        };
        let discard_hand_indexes: Vec<Option<usize>> =
            if attack.attack_damage_bonus_if_discard_hand_card.is_some() {
                let mut v: Vec<Option<usize>> = vec![None];
                for i in choice_indexes(side.hand.len(), 4) {
                    v.push(Some(i));
                }
                v
            } else {
                vec![None]
            };
        let evolution_deck_indexes: Vec<Option<usize>> = if attack.evolve_from_deck == Some(true) {
            explicit_evolution_deck_indexes(side, attack)
        } else {
            vec![None]
        };
        let random_discard_indexes: Vec<Option<usize>> = if attack
            .shuffle_random_discard_into_deck
            .is_some()
            && !side.discard.is_empty()
        {
            choice_indexes(side.discard.len(), 4)
                .into_iter()
                .map(Some)
                .collect()
        } else {
            vec![None]
        };
        let switch_target_uids: Vec<Option<u32>> = if attack.switch_self_after_attack.is_some()
            && !side.bench.is_empty()
        {
            side.bench.iter().map(|u| Some(u.uid)).collect()
        } else {
            vec![None]
        };

        for &attack_target_uid in attack_target_uids.iter() {
            for &heal_target_uid in heal_target_uids.iter() {
                for &use_shuffle_self_into_deck in shuffle_options.iter() {
                    for &discard_hand_index in discard_hand_indexes.iter() {
                        for &evolution_deck_card_index in evolution_deck_indexes.iter() {
                            for &random_discard_index in random_discard_indexes.iter() {
                                for &switch_target_uid in switch_target_uids.iter() {
                                    let choice_tag = build_choice_tag(
                                        attack_index,
                                        retreat_target_uid,
                                        attack_target_uid,
                                        heal_target_uid,
                                        use_shuffle_self_into_deck,
                                        discard_hand_index,
                                        evolution_deck_card_index,
                                        random_discard_index,
                                        switch_target_uid,
                                    );
                                    let attack_decision = build_attack_decision(
                                        attack_index,
                                        retreat_target_uid,
                                        attack_target_uid,
                                        heal_target_uid,
                                        uses_coin_flip,
                                        discard_hand_index,
                                        evolution_deck_card_index,
                                        random_discard_index,
                                        switch_target_uid,
                                        use_shuffle_self_into_deck,
                                    );
                                    let decision = AiCombatDecision::Attack(attack_decision);
                                    if !uses_coin_flip || forced_attack_coin_result.is_some() {
                                        let suffix = match &forced_attack_coin_result {
                                            Some(v) => coin_results_label(v),
                                            None => "none".to_string(),
                                        };
                                        out.push(score_candidate(
                                            state,
                                            side_id,
                                            deps,
                                            decision.clone(),
                                            format!("{}-{}", choice_tag, suffix),
                                            forced_attack_coin_result.clone(),
                                        ));
                                        continue;
                                    }
                                    let heads = score_candidate(
                                        state,
                                        side_id,
                                        deps,
                                        decision.clone(),
                                        format!("{}-heads", choice_tag),
                                        Some(vec![CoinFlipResult::Heads]),
                                    );
                                    let tails = score_candidate(
                                        state,
                                        side_id,
                                        deps,
                                        decision.clone(),
                                        format!("{}-tails", choice_tag),
                                        Some(vec![CoinFlipResult::Tails]),
                                    );
                                    let expected_score = (heads.score + tails.score) / 2.0;
                                    out.push(CombatCandidate {
                                        id: format!("{}-expected", choice_tag),
                                        score: expected_score,
                                        ..heads
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    out
}

fn coin_results_label(results: &[CoinFlipResult]) -> String {
    if results.len() == 1 {
        return match results[0] {
            CoinFlipResult::Heads => "heads".to_string(),
            CoinFlipResult::Tails => "tails".to_string(),
        };
    }
    // TS uses `forcedAttackCoinResult ?? "none"` and string-coercion of an
    // array yields comma-joined ("heads,tails"). Mirror that.
    results
        .iter()
        .map(|r| match r {
            CoinFlipResult::Heads => "heads",
            CoinFlipResult::Tails => "tails",
        })
        .collect::<Vec<&str>>()
        .join(",")
}

#[allow(clippy::too_many_arguments)]
fn build_choice_tag(
    attack_index: usize,
    retreat_target_uid: Option<u32>,
    attack_target_uid: Option<u32>,
    heal_target_uid: Option<u32>,
    use_shuffle_self_into_deck: Option<bool>,
    discard_hand_index: Option<usize>,
    evolution_deck_card_index: Option<usize>,
    random_discard_index: Option<usize>,
    switch_target_uid: Option<u32>,
) -> String {
    let parts: Vec<String> = vec![
        format!("a{}", attack_index),
        retreat_target_uid
            .map(|u| u.to_string())
            .unwrap_or_else(|| "stay".to_string()),
        attack_target_uid
            .map(|u| u.to_string())
            .unwrap_or_else(|| "active".to_string()),
        heal_target_uid
            .map(|u| u.to_string())
            .unwrap_or_else(|| "auto".to_string()),
        match use_shuffle_self_into_deck {
            None => "auto".to_string(),
            Some(true) => "shuffle".to_string(),
            Some(false) => "keep".to_string(),
        },
        discard_hand_index
            .map(|i| i.to_string())
            .unwrap_or_else(|| "noDiscard".to_string()),
        evolution_deck_card_index
            .map(|i| i.to_string())
            .unwrap_or_else(|| "autoEvolve".to_string()),
        random_discard_index
            .map(|i| i.to_string())
            .unwrap_or_else(|| "randomDiscard".to_string()),
        switch_target_uid
            .map(|u| u.to_string())
            .unwrap_or_else(|| "autoSwitch".to_string()),
    ];
    parts.join("-")
}

fn can_use_any_attack(state: &GameState, side: &SideState) -> bool {
    use crate::core::state::{CurrentSide, Phase};
    use crate::core::constants::SpecialCondition;
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
    card.attacks
        .iter()
        .any(|attack| has_enough_energy(active, &attack.cost))
}

fn choice_indexes(length: usize, limit: usize) -> Vec<usize> {
    let n = length.min(limit);
    (0..n).collect()
}

fn explicit_evolution_deck_indexes(side: &SideState, attack: &crate::core::effects::Attack) -> Vec<Option<usize>> {
    if attack.evolve_from_deck != Some(true) || side.active.is_none() {
        return vec![None];
    }
    let active = side.active.as_ref().unwrap();
    let next_stage = active.stage + 1;
    let cat = catalog();
    let mut indexes: Vec<usize> = side
        .deck
        .iter()
        .enumerate()
        .filter_map(|(idx, &cid)| match cat.get(cid) {
            Some(Card::Umamusume(c))
                if c.evolves_from.as_deref() == Some(active.species())
                    && c.stage == next_stage =>
            {
                Some(idx)
            }
            _ => None,
        })
        .collect();
    indexes.truncate(6);
    if indexes.is_empty() {
        return vec![None];
    }
    indexes.into_iter().map(Some).collect()
}

fn score_candidate(
    base_state: &GameState,
    acting_side_id: SideId,
    deps: &mut CombatDeps<'_>,
    decision: AiCombatDecision,
    id: String,
    forced_coin_result: Option<Vec<CoinFlipResult>>,
) -> CombatCandidate {
    // The TS port translated `structuredClone(baseState)` into a full
    // GameState clone here, but `score_candidate` only ever READS `before`
    // (passes it as `&GameState` to the penalty helpers and reads
    // `side(...)` snapshots). Drop the redundant clone — we already hold
    // an immutable borrow. `simulated` retains its clone because
    // `perform_attack` mutates it in-place. Removing the `before` and
    // per-side clones eliminates ~4 SideState clones per candidate, which
    // is the dominant heuristic-rollout cost at sims=800 (flamegraph
    // `Slice 3e` — see r12-selfplay-gate-throughput.md).
    let before: &GameState = base_state;
    let mut simulated = base_state.clone();
    let defending_id: SideId = acting_side_id.opposite();
    let acting_before: &SideState = before.side(acting_side_id);
    let defending_before: &SideState = before.side(defending_id);

    if let AiCombatDecision::Attack(attack_decision) = &decision {
        if let Some(retreat_uid) = attack_decision.retreat_target_uid {
            ai_retreat_to_target(&mut simulated, acting_side_id, retreat_uid);
        }
        perform_attack(
            &mut simulated,
            acting_side_id,
            deps,
            attack_decision.attack_target_uid,
            attack_decision.heal_target_uid,
            forced_coin_result,
            attack_decision.evolution_deck_card_index,
            attack_decision.attack_index,
            attack_decision.discard_hand_index,
            attack_decision.random_discard_index,
            attack_decision.switch_target_uid,
            attack_decision.use_shuffle_self_into_deck,
        );
    }

    let points_gained = (simulated.side(acting_side_id).points as i32 - acting_before.points as i32) as f64;
    let ko_count = (count_discarded_umamusume(
        simulated.side(defending_id).discard.iter().copied(),
    ) - count_discarded_umamusume(defending_before.discard.iter().copied())) as f64;
    let damage_dealt = get_damage_dealt(
        before.side(defending_id),
        simulated.side(defending_id),
    ) as f64;
    let healing_gained = get_healing_gained(
        before.side(acting_side_id),
        simulated.side(acting_side_id),
    ) as f64;
    let start_active_uid = acting_before.active.as_ref().map(|a| a.uid);
    let start_active_hp = acting_before.active.as_ref().map(|a| a.hp).unwrap_or(0);
    let acting_after = simulated.side(acting_side_id);
    let after_start_active = match start_active_uid {
        Some(uid) => get_all_umamusume(acting_after)
            .into_iter()
            .find(|u| u.uid == uid),
        None => None,
    };
    let damage_taken_by_starting_active = (start_active_hp
        - after_start_active.map(|u| u.hp).unwrap_or(0))
    .max(0) as f64;
    let active_koed = start_active_uid.is_some() && after_start_active.is_none();

    let attack_target_uid = match &decision {
        AiCombatDecision::Attack(a) => a.attack_target_uid,
        AiCombatDecision::EndTurn => None,
    };
    let target_value = get_target_value(before.side(defending_id), attack_target_uid);
    let target_is_active = attack_target_uid.is_none()
        || attack_target_uid == defending_before.active.as_ref().map(|a| a.uid);
    let lethal_target = did_candidate_ko_target(
        before.side(defending_id),
        simulated.side(defending_id),
        attack_target_uid,
    );

    let score = points_gained * BASE_POINTS_WEIGHT
        + ko_count * BASE_KO_WEIGHT
        + damage_dealt * BASE_DAMAGE_DEALT_WEIGHT
        + healing_gained * BASE_HEAL_GAINED_WEIGHT
        - damage_taken_by_starting_active * BASE_DAMAGE_TAKEN_WEIGHT
        + (if active_koed { BASE_ACTIVE_KO_BONUS } else { 0.0 });
    let shuffle_risk_penalty = get_optional_self_shuffle_risk_penalty(
        &before,
        &simulated,
        acting_side_id,
        &decision,
        lethal_target,
    );
    let next_turn_lock_penalty = get_cannot_attack_next_turn_penalty(
        &before,
        &simulated,
        acting_side_id,
        &decision,
        lethal_target,
    );
    let bench_survival_penalty = get_bench_survival_floor_penalty(
        &before,
        &simulated,
        acting_side_id,
        lethal_target,
    );

    CombatCandidate {
        id,
        decision,
        score: score - shuffle_risk_penalty - next_turn_lock_penalty - bench_survival_penalty,
        keeps_safe: !can_immediate_opponent_ko(&simulated, acting_side_id),
        lethal_target,
        target_value,
        target_is_active,
    }
}

fn get_bench_survival_floor_penalty(
    before: &GameState,
    after: &GameState,
    acting_side_id: SideId,
    lethal_target: bool,
) -> f64 {
    if lethal_target {
        return 0.0;
    }
    let before_side = before.side(acting_side_id);
    let after_side = after.side(acting_side_id);
    let threatened = can_immediate_opponent_ko_conservative(after, acting_side_id);
    let bench_count = after_side.bench.len();
    if bench_count >= 2 && !threatened {
        return 0.0;
    }

    let mut penalty = 0.0f64;
    if bench_count == 0 {
        penalty += 260.0;
    } else if bench_count == 1 {
        penalty += if threatened { 180.0 } else { 90.0 };
    } else if threatened {
        penalty += 50.0;
    }
    if bench_count > 0 {
        let fragile_bench = after_side.bench.iter().filter(|u| u.hp <= 40).count();
        penalty += (fragile_bench as f64) * if threatened { 46.0 } else { 22.0 };
    }
    if after_side.bench.len() < before_side.bench.len() {
        penalty += ((before_side.bench.len() - after_side.bench.len()) as f64) * 35.0;
    }
    penalty
}

fn get_cannot_attack_next_turn_penalty(
    before: &GameState,
    after: &GameState,
    acting_side_id: SideId,
    decision: &AiCombatDecision,
    lethal_target: bool,
) -> f64 {
    if !matches!(decision, AiCombatDecision::Attack(_)) {
        return 0.0;
    }
    let before_active = before.side(acting_side_id).active.as_ref();
    let after_active = after.side(acting_side_id).active.as_ref();
    let (Some(before_active), Some(after_active)) = (before_active, after_active) else {
        return 0.0;
    };
    if before_active.uid != after_active.uid {
        return 0.0;
    }
    let gained_lock =
        after_active.attack_blocked_until_own_turn != before_active.attack_blocked_until_own_turn;
    if !gained_lock {
        return 0.0;
    }
    if lethal_target {
        return 0.0;
    }
    if after.side(acting_side_id).bench.is_empty() {
        120.0
    } else {
        55.0
    }
}

fn get_optional_self_shuffle_risk_penalty(
    before: &GameState,
    after: &GameState,
    acting_side_id: SideId,
    decision: &AiCombatDecision,
    lethal_target: bool,
) -> f64 {
    let attack_decision: &AttackDecision = match decision {
        AiCombatDecision::Attack(a) => a,
        _ => return 0.0,
    };
    if attack_decision.use_shuffle_self_into_deck != Some(true) {
        return 0.0;
    }
    let before_side = before.side(acting_side_id);
    let after_side = after.side(acting_side_id);
    let Some(before_active) = &before_side.active else {
        return 0.0;
    };
    let Some(after_active) = &after_side.active else {
        return 0.0;
    };
    let mut penalty = 0.0f64;
    if after_side.bench.is_empty() {
        penalty += 220.0;
    } else if after_side.bench.len() == 1 {
        penalty += 110.0;
    }
    let promoted_hp_loss = (before_active.hp - after_active.hp).max(0) as f64;
    penalty += promoted_hp_loss * 1.2;
    let invested_energy = attached_energy_count(before_active) as f64;
    if !lethal_target {
        penalty += invested_energy * 28.0;
    } else {
        penalty += invested_energy * 8.0;
    }
    if before_side.points >= 2 && !lethal_target {
        penalty += 160.0;
    }
    penalty
}

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

    fn dummy_instance(uid: u32, card_id: CardId) -> UmamusumeInstance {
        UmamusumeInstance {
            uid,
            card_id,
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
    fn build_combat_candidates_returns_endturn_when_no_active() {
        // No active anywhere → can_use_any_attack is false; can_retreat is
        // false; build_combat_candidates emits a single end-turn candidate.
        let state = empty_state();
        let mut refresh = |_state: &mut GameState| {};
        let choose: fn(&SideState) -> i32 = |_| -1;
        let mut deps = crate::flow::combat::CombatDeps {
            refresh_continuous_effects: &mut refresh,
            choose_preferred_active_index: &choose,
        };
        let candidates = build_combat_candidates(&state, SideId::Player, &mut deps, None);
        assert_eq!(candidates.len(), 1);
        assert!(matches!(candidates[0].decision, super::super::types::AiCombatDecision::EndTurn));
        assert_eq!(candidates[0].id, "end-turn");
    }

    #[test]
    fn ai_retreat_to_target_returns_false_when_no_active() {
        let mut state = empty_state();
        // Bench a candidate but no active.
        state.sides[SideId::Player as usize]
            .bench
            .push(dummy_instance(7, CardId(0)));
        assert!(!ai_retreat_to_target(&mut state, SideId::Player, 7));
    }
}
