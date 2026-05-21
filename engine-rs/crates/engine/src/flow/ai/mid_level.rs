//! Bit-identical port of `frontend/src/game/engine/flow/ai/midLevel.ts`.
//!
//! Picks the combat candidate via `chooseMidLevelCombatDecision`. Float
//! math (`* 0.35`) preserved in source-text operator order.

use crate::core::catalog::{catalog, Card};
use crate::core::constants::{CoinFlipResult, SideId};
use crate::core::state::{GameState, SideState};
use crate::core::umamusume::attached_energy_count;
use crate::flow::combat::{perform_attack, CombatDeps};
use crate::flow::energy::has_enough_energy;

use super::combat_planner::{ai_retreat_to_target, build_combat_candidates};
use super::combat_utils::{can_immediate_opponent_ko_conservative, compare_candidates};
use super::types::{
    AiCombatDecision, AiTacticalGoal, AttackDecision, CombatCandidate, MidLevelDecision,
};

pub fn choose_mid_level_combat_decision(
    state: &GameState,
    side_id: SideId,
    candidates: &[CombatCandidate],
    deps: &mut CombatDeps<'_>,
    forced_attack_coin_result: Option<Vec<CoinFlipResult>>,
) -> Option<MidLevelDecision> {
    if candidates.is_empty() {
        return None;
    }

    let lethal_candidates: Vec<CombatCandidate> = candidates
        .iter()
        .filter(|c| c.lethal_target)
        .cloned()
        .collect();
    if !lethal_candidates.is_empty() {
        return Some(MidLevelDecision {
            goal: AiTacticalGoal::SecureLethal,
            candidate: pick_best_candidate(&lethal_candidates),
        });
    }

    if can_immediate_opponent_ko_conservative(state, side_id) {
        let safe_candidates: Vec<CombatCandidate> = candidates
            .iter()
            .filter(|c| c.keeps_safe)
            .cloned()
            .collect();
        if !safe_candidates.is_empty() {
            return Some(MidLevelDecision {
                goal: AiTacticalGoal::DenyOpponentLethal,
                candidate: pick_best_candidate(&safe_candidates),
            });
        }
    }

    Some(MidLevelDecision {
        goal: AiTacticalGoal::MaximizeExpectedDamage,
        candidate: pick_best_lookahead_candidate(
            state,
            side_id,
            candidates,
            deps,
            forced_attack_coin_result,
        ),
    })
}

fn pick_best_candidate(candidates: &[CombatCandidate]) -> CombatCandidate {
    let mut sorted: Vec<CombatCandidate> = candidates.to_vec();
    sorted.sort_by(compare_candidates);
    sorted
        .into_iter()
        .next()
        .unwrap_or_else(|| candidates[0].clone())
}

fn pick_best_lookahead_candidate(
    state: &GameState,
    side_id: SideId,
    candidates: &[CombatCandidate],
    deps: &mut CombatDeps<'_>,
    forced_attack_coin_result: Option<Vec<CoinFlipResult>>,
) -> CombatCandidate {
    let mut sorted: Vec<CombatCandidate> = candidates.to_vec();
    sorted.sort_by(compare_candidates);
    let top_candidates: Vec<CombatCandidate> = sorted.iter().take(3).cloned().collect();
    let opponent_id = side_id.opposite();
    let mut best = top_candidates
        .first()
        .cloned()
        .unwrap_or_else(|| candidates[0].clone());
    let mut best_adjusted = f64::NEG_INFINITY;

    for candidate in top_candidates.iter() {
        let mut simulated = state.clone();
        apply_decision(
            &mut simulated,
            side_id,
            &candidate.decision,
            deps,
            forced_attack_coin_result.clone(),
        );
        let risk = estimate_opponent_best_score(&simulated, opponent_id, deps);
        let retreat_penalty =
            estimate_retreat_investment_penalty(state, state.side(side_id), &candidate.decision);
        let adjusted = candidate.score - risk * 0.35 - retreat_penalty;
        if adjusted > best_adjusted {
            best_adjusted = adjusted;
            best = candidate.clone();
        }
    }

    best
}

fn apply_decision(
    state: &mut GameState,
    side_id: SideId,
    decision: &AiCombatDecision,
    deps: &mut CombatDeps<'_>,
    forced_attack_coin_result: Option<Vec<CoinFlipResult>>,
) {
    let AiCombatDecision::Attack(attack_decision) = decision else {
        return;
    };
    if let Some(retreat_uid) = attack_decision.retreat_target_uid {
        if !ai_retreat_to_target(state, side_id, retreat_uid) {
            return;
        }
    }
    let coin_arg = if attack_decision.uses_coin_flip {
        forced_attack_coin_result
    } else {
        None
    };
    perform_attack(
        state,
        side_id,
        deps,
        attack_decision.attack_target_uid,
        attack_decision.heal_target_uid,
        coin_arg,
        None,
        0,
        None,
        None,
        None,
        attack_decision.use_shuffle_self_into_deck,
    );
}

fn estimate_opponent_best_score(
    state: &GameState,
    opponent_id: SideId,
    deps: &mut CombatDeps<'_>,
) -> f64 {
    if state.game_over {
        return if state.winner == Some(opponent_id) {
            5000.0
        } else {
            0.0
        };
    }
    let opponent = state.side(opponent_id);
    if opponent.active.is_none() {
        return 0.0;
    }
    let responses = build_combat_candidates(state, opponent_id, deps, None);
    let mut sorted = responses;
    sorted.sort_by(compare_candidates);
    sorted.first().map(|c| c.score).unwrap_or(0.0)
}

fn estimate_retreat_investment_penalty(
    state: &GameState,
    side: &SideState,
    decision: &AiCombatDecision,
) -> f64 {
    let AiCombatDecision::Attack(attack_decision) = decision else {
        return 0.0;
    };
    let Some(retreat_uid) = attack_decision.retreat_target_uid else {
        return 0.0;
    };
    let Some(active) = &side.active else {
        return 0.0;
    };
    let Some(retreat_target) = side.bench.iter().find(|u| u.uid == retreat_uid) else {
        return 0.0;
    };

    let cat = catalog();
    let active_card = match cat.get(active.card_id) {
        Some(Card::Umamusume(u)) => u,
        _ => return 0.0,
    };
    let active_attack = match active_card.attacks.first() {
        Some(a) => a,
        None => return 0.0,
    };
    let target_card = match cat.get(retreat_target.card_id) {
        Some(Card::Umamusume(u)) => u,
        _ => return 0.0,
    };
    let target_attack = match target_card.attacks.first() {
        Some(a) => a,
        None => return 0.0,
    };
    let active_energy = attached_energy_count(active) as i32;
    let target_energy = attached_energy_count(retreat_target) as i32;
    let active_ready = has_enough_energy(active, &active_attack.cost);
    let target_ready = has_enough_energy(retreat_target, &target_attack.cost);
    let threatened_now = can_immediate_opponent_ko_conservative(state, side.id);

    let mut penalty = 0.0f64;
    if active.stage >= 1 && active_energy >= 2 {
        penalty += (active_energy as f64) * 28.0 + (active.stage as f64) * 20.0;
    }
    if active.max_hp - retreat_target.max_hp >= 30 {
        penalty += 34.0;
    }
    if active_ready && !target_ready {
        penalty += 32.0;
    }
    if active_ready && target_attack.damage + 20 < active_attack.damage {
        penalty += 22.0;
    }
    if !threatened_now && active.hp >= 40 {
        penalty += 18.0;
    }
    if target_energy == 0 && retreat_target.max_hp <= 80 {
        penalty += 18.0;
    }
    penalty
}

// `AttackDecision` import suppress lint
#[allow(dead_code)]
fn _unused(_: AttackDecision) {}
