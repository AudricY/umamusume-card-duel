//! Bit-identical port of `frontend/src/game/engine/flow/ai/core.rs`
//! (`frontend/.../ai/core.ts`). Top-level orchestration: bench, evolve,
//! attach energy, play trainer, resolve combat, use ability.
//!
//! Float math:
//! - `Number(score.toFixed(2))` ports as `(score * 100.0).round() / 100.0`.
//! - Weights `* 0.5`, `* 0.25`, `* 0.7` preserved.

use serde_json::{json, Map, Value};

use crate::core::catalog::{catalog, Card};
use crate::core::constants::{
    AiDeckStyle, CoinFlipResult, SideId, TrainerType, MAX_BENCH,
};
use crate::core::state::{GameState, SideState};
use crate::core::umamusume::get_all_umamusume;
use crate::flow::board::choose_preferred_active_index;
use crate::flow::combat::{perform_attack, CombatDeps};
use crate::flow::eligibility::{can_attach_energy, can_use_umamusume_ability};
use crate::flow::evolution::{evolve_umamusume, find_evolution_target};
use crate::flow::play_rules::{
    get_tool_targets, use_rainbow_uncap_crystal,
};
use crate::core::random::random_float;
use crate::flow::setup::create_umamusume;
use crate::flow::trainers::{apply_trainer, can_use_stadium, play_stadium};
use crate::flow::turn::draw_cards;

use super::ability_utils::{
    ai_use_coin_flip_draw_ability, ai_use_damage_ability, ai_use_move_benched_energy_ability,
    AbilityHeuristicDeps,
};
use super::attach_utils::{
    ai_attach_one_energy as execute_ai_attach_one_energy, estimate_attack_damage_output,
    mark_ability_used, with_energy_shift,
};
use super::combat_planner::{ai_retreat_to_target, build_combat_candidates};
use super::combat_utils::{
    can_immediate_opponent_ko_conservative, compare_candidates, pick_candidate_by_difficulty,
};
use super::energy_awareness::score_attack_energy_pool_fit;
use super::mid_level::choose_mid_level_combat_decision;
use super::telemetry::{emit_ai_telemetry, AiTelemetryEvent};
use super::trainer_utils::{
    get_ai_rainbow_uncap_choice, get_ai_trainer_choices, score_evolution_target,
    should_ai_play_trainer,
};
use super::turn_plan::{choose_ai_turn_goal, explain_ai_turn_goal, has_consecutive_no_attack_turns};
use super::types::{
    AiCombatDecision, AiCombatDecisionResult, AiTurnGoal, PendingSwitchAfterGustResume,
};

const BASE_THREAT_PENALTY: f64 = 120.0;

pub struct AiTrainerDeps<'a> {
    pub refresh_continuous_effects: &'a mut dyn FnMut(&mut GameState),
}

pub struct AiCombatDepsAi<'a> {
    pub refresh_continuous_effects: &'a mut dyn FnMut(&mut GameState),
    pub choose_preferred_active_index: &'a dyn Fn(&SideState) -> i32,
}

pub fn ai_play_one_basic(state: &mut GameState, side_id: SideId) -> bool {
    let cat = catalog();
    if state.side(side_id).bench.len() >= MAX_BENCH {
        return false;
    }
    let deck_style = state.ai_deck_style_by_side[side_id as usize];
    let mut scored: Vec<(usize, f64)> = Vec::new();
    let side_snapshot = state.side(side_id).clone();
    for (hand_index, &cid) in side_snapshot.hand.iter().enumerate() {
        let Some(Card::Umamusume(c)) = cat.get(cid) else {
            continue;
        };
        if c.stage != 0 {
            continue;
        }
        let score =
            score_basic_bench_candidate(state, &side_snapshot, cid, deck_style);
        scored.push((hand_index, score));
    }
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    let Some(&(best_idx, _)) = scored.first() else {
        return false;
    };
    let side = state.side_mut(side_id);
    if best_idx >= side.hand.len() {
        return false;
    }
    let cid = side.hand.remove(best_idx);
    let Some(Card::Umamusume(_)) = cat.get(cid) else {
        return false;
    };
    let inst = create_umamusume(cid, state.turn_number);
    let side = state.side_mut(side_id);
    let _ = side.bench.try_push(inst);
    true
}

pub fn ai_evolve_one(state: &mut GameState, side_id: SideId) -> bool {
    let cat = catalog();
    let side_snapshot = state.side(side_id).clone();
    let mut scored: Vec<(usize, u32, f64)> = Vec::new(); // (hand_index, target_uid, score)
    for (hand_index, &cid) in side_snapshot.hand.iter().enumerate() {
        let Some(Card::Umamusume(card)) = cat.get(cid) else {
            continue;
        };
        if card.stage == 0 {
            continue;
        }
        let target = find_evolution_target(state, &side_snapshot, card);
        let Some(target) = target else {
            continue;
        };
        let s = score_evolution_target(state, &side_snapshot, target, card);
        scored.push((hand_index, target.uid, s));
    }
    scored.sort_by(|a, b| {
        b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)
    });
    let Some(&(best_idx, target_uid, _)) = scored.first() else {
        return false;
    };
    let evolution_card = {
        let side = state.side(side_id);
        let Some(&cid) = side.hand.get(best_idx) else {
            return false;
        };
        match cat.get(cid) {
            Some(Card::Umamusume(c)) => (cid, c.clone()),
            _ => return false,
        }
    };
    // Pull the card out of hand.
    let removed_cid;
    {
        let side = state.side_mut(side_id);
        if best_idx >= side.hand.len() {
            return false;
        }
        removed_cid = side.hand.remove(best_idx);
    }
    if removed_cid != evolution_card.0 {
        return false;
    }
    // Apply evolution.
    let turn_number = state.turn_number;
    let side = state.side_mut(side_id);
    if let Some(active) = side.active.as_mut() {
        if active.uid == target_uid {
            evolve_umamusume(turn_number, active, evolution_card.0, &evolution_card.1);
            return true;
        }
    }
    for u in side.bench.iter_mut() {
        if u.uid == target_uid {
            evolve_umamusume(turn_number, u, evolution_card.0, &evolution_card.1);
            return true;
        }
    }
    false
}

pub fn ai_attach_one_energy(state: &mut GameState, side_id: SideId) -> bool {
    if state.side(side_id).active.is_none() || !can_attach_energy(state, state.side(side_id)) {
        return false;
    }
    let turn_goal = choose_ai_turn_goal(state, state.side(side_id));
    emit_turn_goal_telemetry(state, side_id, "attach", turn_goal);
    execute_ai_attach_one_energy(state, side_id, turn_goal)
}

pub fn ai_play_one_trainer(
    state: &mut GameState,
    side_id: SideId,
    pending_choice_resume: PendingSwitchAfterGustResume,
    deps: &mut AiTrainerDeps<'_>,
) -> bool {
    let turn_goal = choose_ai_turn_goal(state, state.side(side_id));
    emit_turn_goal_telemetry(state, side_id, "trainer", turn_goal);
    let cat = catalog();
    let side_snapshot = state.side(side_id).clone();
    let trainer_indexes: Vec<usize> = side_snapshot
        .hand
        .iter()
        .enumerate()
        .filter(|(handi, &cid)| should_ai_play_trainer(state, &side_snapshot, cid, *handi, turn_goal))
        .map(|(handi, _)| handi)
        .collect();
    let fallback_index = trainer_indexes.first().copied();
    let index = if turn_goal == AiTurnGoal::SetUpTwoTurnLethal {
        choose_trainer_index_for_two_turn_bundle(
            state,
            side_id,
            &trainer_indexes,
            pending_choice_resume,
            deps,
            turn_goal,
        )
        .or(fallback_index)
    } else {
        fallback_index
    };
    let Some(index) = index else {
        return false;
    };
    let cid = match state.side(side_id).hand.get(index).copied() {
        Some(c) => c,
        None => return false,
    };
    let card = match cat.get(cid) {
        Some(Card::Trainer(t)) => t.clone(),
        _ => return false,
    };
    let choices = get_ai_trainer_choices(state, state.side(side_id), &card, index, turn_goal);
    if card.effect.rainbow_uncap_crystal == Some(true) {
        let Some(rainbow_choice) = get_ai_rainbow_uncap_choice(state, state.side(side_id)) else {
            return false;
        };
        {
            let side = state.side_mut(side_id);
            if index >= side.hand.len() {
                return false;
            }
            let _ = side.hand.remove(index);
        }
        let shifted_evolution_hand_index = if rainbow_choice.evolution_hand_index > index {
            rainbow_choice.evolution_hand_index - 1
        } else {
            rainbow_choice.evolution_hand_index
        };
        let resolved = use_rainbow_uncap_crystal(
            state,
            side_id,
            Some(rainbow_choice.target_uid),
            Some(shifted_evolution_hand_index),
        ) || use_rainbow_uncap_crystal(state, side_id, Some(rainbow_choice.target_uid), None);
        if resolved {
            let _ = state.side_mut(side_id).discard.try_push(cid);
        }
        return true;
    }
    {
        let side = state.side_mut(side_id);
        if index >= side.hand.len() {
            return false;
        }
        let _ = side.hand.remove(index);
    }
    if card.trainer_type == TrainerType::Stadium {
        play_stadium(state, side_id, cid, &card);
        (deps.refresh_continuous_effects)(state);
        return true;
    }
    if card.trainer_type == TrainerType::Tool {
        let target_uid: Option<u32> = match choices.umamusume_target_uid {
            Some(uid) => Some(uid),
            None => {
                let side = state.side(side_id);
                get_tool_targets(side).into_iter().next().map(|u| u.uid)
            }
        };
        let Some(target_uid) = target_uid else {
            return false;
        };
        let side = state.side_mut(side_id);
        if let Some(active) = side.active.as_mut() {
            if active.uid == target_uid {
                active.tool_card_id = Some(cid);
                return true;
            }
        }
        for u in side.bench.iter_mut() {
            if u.uid == target_uid {
                u.tool_card_id = Some(cid);
                return true;
            }
        }
        return false;
    }
    apply_trainer(state, side_id, cid, &card, &choices, pending_choice_resume);
    if card.trainer_type == TrainerType::Supporter {
        state.side_mut(side_id).used_supporter_this_turn = true;
    }
    let _ = state.side_mut(side_id).discard.try_push(cid);
    true
}

fn choose_trainer_index_for_two_turn_bundle(
    state: &GameState,
    side_id: SideId,
    trainer_indexes: &[usize],
    pending_choice_resume: PendingSwitchAfterGustResume,
    deps: &mut AiTrainerDeps<'_>,
    turn_goal: AiTurnGoal,
) -> Option<usize> {
    if trainer_indexes.is_empty() {
        return None;
    }
    let cat = catalog();
    let side = state.side(side_id);
    let mut top: Vec<(usize, f64)> = trainer_indexes
        .iter()
        .map(|&handi| {
            let cid_opt = side.hand.get(handi).copied();
            let prio = score_two_turn_trainer_priority(cid_opt);
            (handi, prio)
        })
        .collect();
    top.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    top.truncate(3);

    let mut best_index: Option<usize> = None;
    let mut best_score: f64 = f64::NEG_INFINITY;
    // Trainer-bundle-scores telemetry is opt-in. When disabled (default)
    // skip the per-iteration Map allocs + post-loop payload assembly.
    // The score computation stays — it picks the actual best hand index.
    let telemetry_enabled = super::telemetry::is_enabled();
    let mut scored_payload: Vec<Value> = if telemetry_enabled {
        Vec::with_capacity(top.len())
    } else {
        Vec::new()
    };
    for (handi, _) in top.iter().copied() {
        let score = simulate_trainer_attach_combat_bundle(
            state,
            side_id,
            handi,
            pending_choice_resume,
            deps,
            turn_goal,
        );
        if telemetry_enabled {
            let card_id = side.hand.get(handi).copied();
            let mut record = Map::new();
            record.insert("handIndex".into(), json!(handi));
            record.insert(
                "cardId".into(),
                match card_id {
                    Some(c) => match cat.interner.resolve(c) {
                        Some(s) => json!(s),
                        None => Value::Null,
                    },
                    None => Value::Null,
                },
            );
            // Number(score.toFixed(2)) → round to 2 decimals.
            let rounded = round_to_2dp(score);
            record.insert("score".into(), json!(rounded));
            scored_payload.push(Value::Object(record));
        }
        if score > best_score {
            best_score = score;
            best_index = Some(handi);
        }
    }
    if telemetry_enabled {
        let mut payload = Map::new();
        payload.insert("side".into(), json!(side_id_tag(side_id)));
        payload.insert("turn".into(), json!(state.turn_number));
        payload.insert("goal".into(), json!(turn_goal.tag()));
        payload.insert("candidates".into(), Value::Array(scored_payload));
        payload.insert(
            "selectedHandIndex".into(),
            match best_index.or_else(|| trainer_indexes.first().copied()) {
                Some(idx) => json!(idx),
                None => Value::Null,
            },
        );
        emit_ai_telemetry(AiTelemetryEvent::TrainerBundleScores, payload);
    }
    best_index.or_else(|| trainer_indexes.first().copied())
}

fn score_two_turn_trainer_priority(card_id: Option<crate::core::card_id::CardId>) -> f64 {
    let Some(cid) = card_id else {
        return f64::NEG_INFINITY;
    };
    let cat = catalog();
    let Some(Card::Trainer(card)) = cat.get(cid) else {
        return f64::NEG_INFINITY;
    };
    let mut score = 0.0f64;
    if card.effect.active_attack_damage_bonus.is_some() {
        score += 60.0;
    }
    if card.effect.extra_energy_attach.is_some() {
        score += 54.0;
    }
    if card.effect.attach_energy_from_zone_to_bench.is_some() {
        score += 42.0;
    }
    if card.effect.search_evolution_umamusume == Some(true) {
        score += 36.0;
    }
    if card.effect.search_umamusume == Some(true) {
        score += 32.0;
    }
    if card.effect.retreat_cost_reduction.is_some() {
        score += 14.0;
    }
    if card.effect.draw.is_some() {
        score += 8.0;
    }
    score
}

fn simulate_trainer_attach_combat_bundle(
    state: &GameState,
    side_id: SideId,
    trainer_hand_index: usize,
    pending_choice_resume: PendingSwitchAfterGustResume,
    deps: &mut AiTrainerDeps<'_>,
    turn_goal: AiTurnGoal,
) -> f64 {
    let cat = catalog();
    let mut simulated = state.clone();
    let Some(&cid) = simulated.side(side_id).hand.get(trainer_hand_index) else {
        return f64::NEG_INFINITY;
    };
    let card = match cat.get(cid) {
        Some(Card::Trainer(t)) => t.clone(),
        _ => return f64::NEG_INFINITY,
    };
    let choices = get_ai_trainer_choices(
        &simulated,
        simulated.side(side_id),
        &card,
        trainer_hand_index,
        turn_goal,
    );
    if card.effect.rainbow_uncap_crystal == Some(true) {
        if let Some(rainbow_choice) = get_ai_rainbow_uncap_choice(&simulated, simulated.side(side_id))
        {
            let side = simulated.side_mut(side_id);
            if trainer_hand_index < side.hand.len() {
                let _ = side.hand.remove(trainer_hand_index);
            }
            let shifted = if rainbow_choice.evolution_hand_index > trainer_hand_index {
                rainbow_choice.evolution_hand_index - 1
            } else {
                rainbow_choice.evolution_hand_index
            };
            let resolved = use_rainbow_uncap_crystal(
                &mut simulated,
                side_id,
                Some(rainbow_choice.target_uid),
                Some(shifted),
            ) || use_rainbow_uncap_crystal(
                &mut simulated,
                side_id,
                Some(rainbow_choice.target_uid),
                None,
            );
            if resolved {
                let _ = simulated.side_mut(side_id).discard.try_push(cid);
            }
        }
    } else {
        {
            let side = simulated.side_mut(side_id);
            if trainer_hand_index >= side.hand.len() {
                return f64::NEG_INFINITY;
            }
            let _ = side.hand.remove(trainer_hand_index);
        }
        if card.trainer_type == TrainerType::Stadium {
            play_stadium(&mut simulated, side_id, cid, &card);
            (deps.refresh_continuous_effects)(&mut simulated);
            if card.effect.shuffle_hand_into_deck_draw.is_some()
                && can_use_stadium(&simulated, side_id)
            {
                return -250.0;
            }
        } else if card.trainer_type == TrainerType::Tool {
            let target_uid: Option<u32> = match choices.umamusume_target_uid {
                Some(uid) => Some(uid),
                None => get_tool_targets(simulated.side(side_id))
                    .into_iter()
                    .next()
                    .map(|u| u.uid),
            };
            if let Some(uid) = target_uid {
                let side = simulated.side_mut(side_id);
                if let Some(active) = side.active.as_mut() {
                    if active.uid == uid {
                        active.tool_card_id = Some(cid);
                    }
                }
                for u in side.bench.iter_mut() {
                    if u.uid == uid {
                        u.tool_card_id = Some(cid);
                    }
                }
            }
        } else {
            apply_trainer(
                &mut simulated,
                side_id,
                cid,
                &card,
                &choices,
                pending_choice_resume,
            );
            if card.trainer_type == TrainerType::Supporter {
                simulated.side_mut(side_id).used_supporter_this_turn = true;
            }
            let _ = simulated.side_mut(side_id).discard.try_push(cid);
        }
    }

    if can_attach_energy(&simulated, simulated.side(side_id)) {
        execute_ai_attach_one_energy(&mut simulated, side_id, AiTurnGoal::SetUpTwoTurnLethal);
    }

    let mut combat_deps = CombatDeps {
        refresh_continuous_effects: deps.refresh_continuous_effects,
        choose_preferred_active_index: &choose_preferred_active_index,
    };
    let combat_candidates = build_combat_candidates(&simulated, side_id, &mut combat_deps, None);
    let mut sorted = combat_candidates;
    sorted.sort_by(|a, b| {
        b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
    });
    let Some(best) = sorted.first() else {
        return 0.0;
    };
    let no_attack_penalty = if matches!(best.decision, AiCombatDecision::EndTurn) {
        220.0
    } else {
        0.0
    };
    best.score + (if best.lethal_target { 300.0 } else { 0.0 }) - no_attack_penalty
}

pub fn ai_resolve_combat_decision(
    state: &mut GameState,
    side_id: SideId,
    forced_attack_coin_result: Option<Vec<CoinFlipResult>>,
    deps: &mut AiCombatDepsAi<'_>,
) -> AiCombatDecisionResult {
    if state.side(side_id).active.is_none() {
        return AiCombatDecisionResult {
            resolved: true,
            used_attack: false,
            did_retreat: Some(false),
        };
    }
    let mut combat_deps = CombatDeps {
        refresh_continuous_effects: deps.refresh_continuous_effects,
        choose_preferred_active_index: deps.choose_preferred_active_index,
    };
    let mut candidates = build_combat_candidates(
        state,
        side_id,
        &mut combat_deps,
        forced_attack_coin_result.clone(),
    );
    if candidates.is_empty() {
        return AiCombatDecisionResult {
            resolved: true,
            used_attack: false,
            did_retreat: Some(false),
        };
    }
    let safe_exists = candidates.iter().any(|c| c.keeps_safe);
    if safe_exists {
        for c in candidates.iter_mut() {
            if !c.keeps_safe {
                c.score -= BASE_THREAT_PENALTY;
            }
        }
    }

    let tactical_choice = choose_mid_level_combat_decision(
        state,
        side_id,
        &candidates,
        &mut combat_deps,
        forced_attack_coin_result.clone(),
    );
    let immediate_ko_threat = can_immediate_opponent_ko_conservative(state, side_id);
    let mut rng = || random_float();
    let selected = tactical_choice
        .as_ref()
        .map(|m| m.candidate.clone())
        .or_else(|| {
            pick_candidate_by_difficulty(
                &candidates,
                state.ai_difficulty,
                &mut rng,
                immediate_ko_threat,
            )
        });

    // Combat-candidates telemetry: pure observer side-effect, but the
    // payload assembly clones the entire `candidates` Vec and sort-by-
    // score'd it just to keep the top 3. Skip the work when the sink is
    // disabled (the default; `__UMA_AI_TELEMETRY__` is opt-in). Same
    // payload shape preserved for telemetry-enabled callers.
    if super::telemetry::is_enabled() {
        let mut top3_payload: Vec<Value> = Vec::new();
        let mut sorted_top = candidates.clone();
        sorted_top.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });
        for c in sorted_top.iter().take(3) {
            let mut rec = Map::new();
            rec.insert("id".into(), json!(&c.id));
            rec.insert("score".into(), json!(round_to_2dp(c.score)));
            rec.insert("keepsSafe".into(), json!(c.keeps_safe));
            rec.insert("lethalTarget".into(), json!(c.lethal_target));
            rec.insert(
                "decision".into(),
                json!(match c.decision {
                    AiCombatDecision::Attack(_) => "attack",
                    AiCombatDecision::EndTurn => "endTurn",
                }),
            );
            top3_payload.push(Value::Object(rec));
        }
        let mut payload = Map::new();
        payload.insert("side".into(), json!(side_id_tag(side_id)));
        payload.insert("turn".into(), json!(state.turn_number));
        payload.insert(
            "tacticalGoal".into(),
            match tactical_choice.as_ref().map(|m| m.goal) {
                Some(super::types::AiTacticalGoal::SecureLethal) => json!("secure_lethal"),
                Some(super::types::AiTacticalGoal::DenyOpponentLethal) => {
                    json!("deny_opponent_lethal")
                }
                Some(super::types::AiTacticalGoal::MaximizeExpectedDamage) => {
                    json!("maximize_expected_damage")
                }
                None => Value::Null,
            },
        );
        payload.insert(
            "selectedId".into(),
            selected
                .as_ref()
                .map(|c| json!(&c.id))
                .unwrap_or(Value::Null),
        );
        payload.insert("top3".into(), Value::Array(top3_payload));
        emit_ai_telemetry(AiTelemetryEvent::CombatCandidates, payload);
    }

    let Some(selected) = selected else {
        return AiCombatDecisionResult {
            resolved: true,
            used_attack: false,
            did_retreat: Some(false),
        };
    };
    let attack_decision = match &selected.decision {
        AiCombatDecision::Attack(a) => a.clone(),
        AiCombatDecision::EndTurn => {
            return AiCombatDecisionResult {
                resolved: true,
                used_attack: false,
                did_retreat: Some(false),
            };
        }
    };

    if let Some(retreat_uid) = attack_decision.retreat_target_uid {
        let retreated = ai_retreat_to_target(state, side_id, retreat_uid);
        if !retreated {
            return AiCombatDecisionResult {
                resolved: true,
                used_attack: false,
                did_retreat: Some(false),
            };
        }
        return AiCombatDecisionResult {
            resolved: true,
            used_attack: false,
            did_retreat: Some(true),
        };
    }

    if attack_decision.uses_coin_flip && forced_attack_coin_result.is_none() {
        return AiCombatDecisionResult {
            resolved: false,
            used_attack: false,
            did_retreat: Some(false),
        };
    }

    perform_attack(
        state,
        side_id,
        &mut combat_deps,
        attack_decision.attack_target_uid,
        attack_decision.heal_target_uid,
        forced_attack_coin_result,
        attack_decision.evolution_deck_card_index,
        attack_decision.attack_index,
        attack_decision.discard_hand_index,
        attack_decision.random_discard_index,
        attack_decision.switch_target_uid,
        attack_decision.use_shuffle_self_into_deck,
    );
    AiCombatDecisionResult {
        resolved: true,
        used_attack: true,
        did_retreat: Some(false),
    }
}

pub fn ai_use_one_ability(
    state: &mut GameState,
    side_id: SideId,
    deps: &mut AiCombatDepsAi<'_>,
    turn_goal: AiTurnGoal,
) -> bool {
    emit_turn_goal_telemetry(state, side_id, "ability", turn_goal);
    let cat = catalog();
    let mut priorities: Vec<(u32, u8)> = get_all_umamusume(state.side(side_id))
        .into_iter()
        .filter(|u| can_use_umamusume_ability(state, state.side(side_id), u.uid))
        .map(|u| (u.uid, u.stage))
        .collect();
    priorities.sort_by(|a, b| b.1.cmp(&a.1));

    let mut combat_deps = CombatDeps {
        refresh_continuous_effects: deps.refresh_continuous_effects,
        choose_preferred_active_index: deps.choose_preferred_active_index,
    };

    for (ability_uid, _stage) in priorities {
        // Snapshot ability shape so we can branch without holding state immutably.
        let (
            has_damage_opponent,
            has_move_benched_energy,
            has_coin_flip,
            has_shuffle_random_discard,
            shuffle_random_discard_amount,
            ability_name,
            ability_once_per_game,
        ) = {
            let side = state.side(side_id);
            let owner = get_all_umamusume(side)
                .into_iter()
                .find(|u| u.uid == ability_uid);
            let owner = match owner {
                Some(o) => o,
                None => continue,
            };
            let Some(Card::Umamusume(card)) = cat.get(owner.card_id) else {
                continue;
            };
            let Some(ability) = &card.ability else {
                continue;
            };
            (
                ability.damage_opponent.is_some(),
                ability.move_benched_energy_to_active.is_some(),
                ability.coin_flip_draw_or_active_damage_counter.is_some(),
                ability.shuffle_random_discard_into_deck.is_some(),
                ability.shuffle_random_discard_into_deck.unwrap_or(0),
                ability.name.clone(),
                ability.once_per_game == Some(true),
            )
        };

        if has_damage_opponent
            && ai_use_damage_ability(state, side_id, ability_uid, &mut combat_deps)
        {
            return true;
        }
        if has_move_benched_energy {
            let ai_difficulty = state.ai_difficulty;
            let deps = AbilityHeuristicDeps {
                estimate_attack_damage_output: &estimate_attack_damage_output,
                with_energy_shift: &with_energy_shift,
                mark_ability_used: &mark_ability_used,
            };
            // We need a mutable side reference.
            let side = state.side_mut(side_id);
            // estimate_attack_damage_output takes &GameState; we can't hold
            // &state and &mut side at once. Snapshot state for read-only
            // queries inside the heuristic.
            let _ = (ai_difficulty, deps, side);
            // Use a clone-based path: clone the side, run heuristic on
            // mutable clone, then write back.
            let state_snapshot = state.clone();
            let mut side_clone = state.side(side_id).clone();
            let deps2 = AbilityHeuristicDeps {
                estimate_attack_damage_output: &estimate_attack_damage_output,
                with_energy_shift: &with_energy_shift,
                mark_ability_used: &mark_ability_used,
            };
            let used = ai_use_move_benched_energy_ability(
                &state_snapshot,
                &mut side_clone,
                ability_uid,
                ai_difficulty,
                &deps2,
            );
            if used {
                *state.side_mut(side_id) = side_clone;
                return true;
            }
        }
        if has_coin_flip {
            if turn_goal == AiTurnGoal::DenyOpponentLethal
                || turn_goal == AiTurnGoal::SecureLethalNow
                || turn_goal == AiTurnGoal::ProtectLoadedActive
                || turn_goal == AiTurnGoal::ConvertPointLead
            {
                continue;
            }
            let ai_difficulty = state.ai_difficulty;
            let mut rng = || random_float();
            if ai_use_coin_flip_draw_ability(
                state,
                side_id,
                ability_uid,
                &mut rng,
                &mut combat_deps,
                ai_difficulty,
            ) {
                return true;
            }
        }
        if has_shuffle_random_discard {
            let count = (shuffle_random_discard_amount as usize)
                .min(state.side(side_id).discard.len());
            if count == 0 {
                continue;
            }
            let mut shuffled_card_ids: Vec<crate::core::card_id::CardId> =
                Vec::with_capacity(count);
            for _ in 0..count {
                let discard_len = state.side(side_id).discard.len();
                if discard_len == 0 {
                    break;
                }
                let pick_index = {
                    let r = random_float();
                    ((r * discard_len as f64).floor() as usize).min(discard_len - 1)
                };
                let cid = state.side_mut(side_id).discard.remove(pick_index);
                shuffled_card_ids.push(cid);
            }
            // Concatenate deck + shuffled, then shuffle.
            {
                let side = state.side_mut(side_id);
                let mut combined: Vec<crate::core::card_id::CardId> =
                    side.deck.iter().copied().collect();
                combined.extend(shuffled_card_ids.iter().copied());
                let shuffled = crate::core::random::shuffle(&combined);
                side.deck.clear();
                for c in shuffled {
                    let _ = side.deck.try_push(c);
                }
            }
            // Mark ability used.
            {
                let side = state.side_mut(side_id);
                if let Some(active) = side.active.as_mut() {
                    if active.uid == ability_uid {
                        active.used_ability_this_turn = true;
                    }
                }
                for u in side.bench.iter_mut() {
                    if u.uid == ability_uid {
                        u.used_ability_this_turn = true;
                    }
                }
                if !side
                    .used_ability_names_this_turn
                    .iter()
                    .any(|n| n == &ability_name)
                {
                    let _ = side
                        .used_ability_names_this_turn
                        .try_push(ability_name.clone());
                }
                if ability_once_per_game
                    && !side
                        .used_ability_names_this_game
                        .iter()
                        .any(|n| n == &ability_name)
                {
                    let _ = side
                        .used_ability_names_this_game
                        .try_push(ability_name.clone());
                }
            }
            return true;
        }
    }
    false
}

fn score_basic_bench_candidate(
    state: &GameState,
    side: &SideState,
    card_id: crate::core::card_id::CardId,
    deck_style: AiDeckStyle,
) -> f64 {
    let cat = catalog();
    let Some(Card::Umamusume(card)) = cat.get(card_id) else {
        return f64::NEG_INFINITY;
    };
    if card.stage != 0 {
        return f64::NEG_INFINITY;
    }
    let attack = match card.attacks.first() {
        Some(a) => a,
        None => return f64::NEG_INFINITY,
    };
    let evolution_in_hand: i32 = side
        .hand
        .iter()
        .filter(|&&cid| {
            matches!(cat.get(cid), Some(Card::Umamusume(c))
                if c.stage > 0
                    && c.evolves_from.as_deref() == Some(card.species.as_str()))
        })
        .count() as i32;
    let immediate_evolution_in_hand = side.hand.iter().any(|&cid| {
        matches!(cat.get(cid), Some(Card::Umamusume(c))
            if c.stage == 1
                && c.evolves_from.as_deref() == Some(card.species.as_str()))
    });
    let evolution_in_deck: i32 = side
        .deck
        .iter()
        .filter(|&&cid| {
            matches!(cat.get(cid), Some(Card::Umamusume(c))
                if c.stage > 0
                    && c.evolves_from.as_deref() == Some(card.species.as_str()))
        })
        .count() as i32;
    let same_species_already_in_play = get_all_umamusume(side)
        .into_iter()
        .any(|u| u.species() == card.species.as_str());
    let mut score: f64 = (card.hp as f64) * 0.5 + (attack.damage as f64);
    score += score_attack_energy_pool_fit(side, &attack.cost);
    score += (84.0_f64)
        .min((evolution_in_hand as f64) * 36.0 + (evolution_in_deck as f64) * 12.0);
    if immediate_evolution_in_hand {
        score += if same_species_already_in_play { 22.0 } else { 64.0 };
    }
    if side.bench.len() < MAX_BENCH {
        score += 20.0;
    }
    if deck_style == AiDeckStyle::ScaleBench {
        score += 22.0;
    }
    if deck_style == AiDeckStyle::ScaleBench && card.species == "Agnes Digital" {
        score += 95.0;
    }
    if deck_style == AiDeckStyle::Blitz && attack.damage >= 30 {
        score += 18.0;
    }
    if deck_style == AiDeckStyle::Stall {
        score += (card.hp as f64) * 0.25;
    }
    if has_consecutive_no_attack_turns(state, &side.title, 2) {
        score += 34.0;
        if !same_species_already_in_play {
            score += 20.0;
        }
        if card.hp >= 80 {
            score += 12.0;
        }
    }
    score
}

fn emit_turn_goal_telemetry(state: &GameState, side_id: SideId, phase: &str, goal: AiTurnGoal) {
    // Telemetry is disabled by default (sink early-exits when
    // `__UMA_AI_TELEMETRY__` is unset). Building the payload — three
    // `json!` allocs + `explain_ai_turn_goal` Vec + per-tag `String::from`
    // — happens on every rollout-step heuristic AI decision (~200 steps ×
    // 3 CRN × 800 sims/decision). Skip the work when the sink will throw
    // it away. Same payload shape preserved for callers that opt in via
    // `set_enabled(true)`.
    if !super::telemetry::is_enabled() {
        return;
    }
    let mut payload = Map::new();
    payload.insert("phase".into(), json!(phase));
    payload.insert("side".into(), json!(side_id_tag(side_id)));
    payload.insert("goal".into(), json!(goal.tag()));
    payload.insert(
        "reasonTags".into(),
        Value::Array(
            explain_ai_turn_goal(state, state.side(side_id))
                .into_iter()
                .map(|s| Value::String(s.to_string()))
                .collect(),
        ),
    );
    payload.insert("turn".into(), json!(state.turn_number));
    emit_ai_telemetry(AiTelemetryEvent::TurnGoal, payload);
}

fn side_id_tag(side_id: SideId) -> &'static str {
    match side_id {
        SideId::Player => "player",
        SideId::Opponent => "opponent",
    }
}

fn round_to_2dp(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

// Compare candidates helper is imported but only used inside mid-level.
#[allow(dead_code)]
fn _suppress(c: super::types::CombatCandidate) {
    let _ = compare_candidates(&c, &c);
    let _ = draw_cards;
}
