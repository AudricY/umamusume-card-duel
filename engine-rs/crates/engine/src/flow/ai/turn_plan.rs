//! Bit-identical port of `frontend/src/game/engine/flow/ai/turnPlan.ts`.
//!
//! Picks the high-level `AiTurnGoal` for the current side. All branching
//! is integer / boolean; no RNG.

use crate::core::catalog::{catalog, Card, UmamusumeCard};
use crate::core::constants::EnergyType;
use crate::core::state::{GameState, SideState, UmamusumeInstance};
use crate::core::umamusume::{attached_energy_count, get_all_umamusume};
use crate::flow::eligibility::can_attack;
use crate::flow::energy::has_enough_energy;

use super::combat_utils::{
    are_tool_effects_disabled, can_immediate_opponent_ko_conservative, predict_attack_damage,
};
use super::energy_awareness::is_attack_supported_by_energy_pool;
use super::types::AiTurnGoal;

fn umamusume_card(inst: &UmamusumeInstance) -> Option<&'static UmamusumeCard> {
    match catalog().get(inst.card_id) {
        Some(Card::Umamusume(u)) => Some(u),
        _ => None,
    }
}

fn primary_attack(card: &UmamusumeCard) -> Option<&crate::core::effects::Attack> {
    card.attacks.first()
}

fn total_attack_cost(cost: &crate::core::effects::EnergyCost) -> u32 {
    let mut sum = 0u32;
    for t in EnergyType::ALL {
        sum += cost.get(t) as u32;
    }
    sum
}

pub fn choose_ai_turn_goal(state: &GameState, side: &SideState) -> AiTurnGoal {
    if side.active.is_none() {
        return AiTurnGoal::MaximizeProgress;
    }
    if can_secure_immediate_lethal(state, side) {
        return AiTurnGoal::SecureLethalNow;
    }
    if can_set_up_two_turn_lethal(state, side) {
        return AiTurnGoal::SetUpTwoTurnLethal;
    }
    if should_protect_loaded_active(state, side) {
        return AiTurnGoal::ProtectLoadedActive;
    }
    if can_immediate_opponent_ko_conservative(state, side.id) {
        return AiTurnGoal::DenyOpponentLethal;
    }
    if should_convert_point_lead(state, side) {
        return AiTurnGoal::ConvertPointLead;
    }
    if should_dig_for_evolution(side) {
        return AiTurnGoal::DigForEvolution;
    }
    if should_build_backup_attacker(state, side) {
        return AiTurnGoal::BuildBackupAttacker;
    }
    if had_recent_no_attack(state, &side.title) {
        return AiTurnGoal::MaximizeProgress;
    }
    let has_bench = !side.bench.is_empty();
    let cat = catalog();
    let has_basic_in_hand = side.hand.iter().any(|&cid| {
        matches!(cat.get(cid), Some(Card::Umamusume(u)) if u.stage == 0)
    });
    if !has_bench && !has_basic_in_hand {
        if has_bench_recovery_option_in_hand(side) {
            return AiTurnGoal::StabilizeBoard;
        }
        return AiTurnGoal::MaximizeProgress;
    }
    AiTurnGoal::MaximizeProgress
}

pub fn explain_ai_turn_goal(state: &GameState, side: &SideState) -> Vec<&'static str> {
    if side.active.is_none() {
        return vec!["no_active"];
    }
    if can_secure_immediate_lethal(state, side) {
        return vec!["immediate_lethal"];
    }
    if can_set_up_two_turn_lethal(state, side) {
        return vec!["two_turn_setup_window"];
    }
    if should_protect_loaded_active(state, side) {
        return vec!["loaded_active_under_ko_threat"];
    }
    if can_immediate_opponent_ko_conservative(state, side.id) {
        return vec!["ko_threat_conservative"];
    }
    if should_convert_point_lead(state, side) {
        return vec!["point_lead_convert_to_pressure"];
    }
    if should_dig_for_evolution(side) {
        return vec!["evolution_available_for_board"];
    }
    if should_build_backup_attacker(state, side) {
        return vec!["bench_attacker_needs_energy"];
    }
    if has_consecutive_no_attack_turns(state, &side.title, 2) {
        return vec!["no_attack_recovery_mode"];
    }
    let cat = catalog();
    let has_bench = !side.bench.is_empty();
    let has_basic_in_hand = side.hand.iter().any(|&cid| {
        matches!(cat.get(cid), Some(Card::Umamusume(u)) if u.stage == 0)
    });
    if !has_bench && !has_basic_in_hand {
        return vec!["no_bench_no_basic_in_hand"];
    }
    vec!["default_progress"]
}

fn should_protect_loaded_active(state: &GameState, side: &SideState) -> bool {
    let Some(active) = &side.active else {
        return false;
    };
    if side.bench.is_empty() {
        return false;
    }
    if !can_immediate_opponent_ko_conservative(state, side.id) {
        return false;
    }
    let Some(active_card) = umamusume_card(active) else {
        return false;
    };
    let loaded =
        active.stage >= 1 || active.max_hp >= 100 || attached_energy_count(active) >= 2;
    if !loaded {
        return false;
    }
    let Some(active_attack) = primary_attack(active_card) else {
        return false;
    };
    if has_enough_energy(active, &active_attack.cost) {
        return true;
    }
    side.bench.iter().any(|bench| {
        let Some(card) = umamusume_card(bench) else {
            return false;
        };
        let Some(attack) = primary_attack(card) else {
            return false;
        };
        let attached = attached_energy_count(bench);
        has_enough_energy(bench, &attack.cost)
            || attached + 1 >= total_attack_cost(&attack.cost)
    })
}

fn should_convert_point_lead(state: &GameState, side: &SideState) -> bool {
    if side.active.is_none() {
        return false;
    }
    let opponent = state.side(side.id.opposite());
    let lead = side.points as i32 - opponent.points as i32;
    if (side.points as i32) < 2 || lead <= 0 {
        return false;
    }
    if can_immediate_opponent_ko_conservative(state, side.id) {
        return false;
    }
    if can_attack(state, side) {
        return true;
    }
    side.bench.iter().any(|bench| {
        let Some(card) = umamusume_card(bench) else {
            return false;
        };
        let Some(attack) = primary_attack(card) else {
            return false;
        };
        has_enough_energy(bench, &attack.cost)
    })
}

fn should_dig_for_evolution(side: &SideState) -> bool {
    let board = get_all_umamusume(side);
    if board.is_empty() {
        return false;
    }
    let cat = catalog();
    board.into_iter().any(|target| {
        if target.stage >= 2 {
            return false;
        }
        let next_stage = target.stage + 1;
        let pool = side.hand.iter().chain(side.deck.iter());
        pool.into_iter().any(|&cid| {
            matches!(cat.get(cid), Some(Card::Umamusume(c))
                if c.stage == next_stage
                    && c.evolves_from.as_deref() == Some(target.species()))
        })
    })
}

fn should_build_backup_attacker(state: &GameState, side: &SideState) -> bool {
    let Some(active) = &side.active else {
        return false;
    };
    if side.bench.is_empty() {
        return false;
    }
    let Some(active_card) = umamusume_card(active) else {
        return false;
    };
    let Some(active_attack) = primary_attack(active_card) else {
        return false;
    };
    if !has_enough_energy(active, &active_attack.cost) && !can_attack(state, side) {
        return false;
    }
    side.bench.iter().any(|bench| {
        let Some(card) = umamusume_card(bench) else {
            return false;
        };
        let Some(attack) = primary_attack(card) else {
            return false;
        };
        if has_enough_energy(bench, &attack.cost) {
            return false;
        }
        let attached = attached_energy_count(bench);
        let total_cost = total_attack_cost(&attack.cost);
        total_cost > 0 && attached + 1 >= total_cost
    })
}

fn can_set_up_two_turn_lethal(state: &GameState, side: &SideState) -> bool {
    let Some(attacker) = &side.active else {
        return false;
    };
    if can_secure_immediate_lethal(state, side) {
        return false;
    }
    let opponent = state.side(side.id.opposite());
    let Some(opp_active) = &opponent.active else {
        return false;
    };
    if !can_attack(state, side) {
        return false;
    }
    let Some(attacker_card) = umamusume_card(attacker) else {
        return false;
    };
    let Some(attack) = primary_attack(attacker_card) else {
        return false;
    };
    if !is_attack_supported_by_energy_pool(side, &attack.cost) {
        return false;
    }
    // typedCostMet: every non-colorless cost is satisfied by typed energy on attacker.
    let typed_cost_met = EnergyType::ALL
        .iter()
        .filter(|t| **t != EnergyType::Colorless)
        .all(|&t| (attacker.energies[t as usize] as u32) >= (attack.cost.get(t) as u32));
    if !typed_cost_met {
        return false;
    }
    let total_cost: u32 = total_attack_cost(&attack.cost);
    let attached: u32 = attacker.energies.iter().map(|&c| c as u32).sum();
    if attached + 1 < total_cost {
        return false;
    }
    let own_in_play_count = 1 + side.bench.len() as i32;
    let all_in_play_count = own_in_play_count + 1 + opponent.bench.len() as i32;
    let damage_now = predict_attack_damage(
        attacker,
        opp_active,
        side.active_attack_damage_bonus as i32,
        own_in_play_count,
        all_in_play_count,
        Some(state.turn_number),
        are_tool_effects_disabled(state),
    );
    let needed_to_ko = opp_active.hp - damage_now;
    if needed_to_ko <= 0 {
        return false;
    }
    if damage_now <= 0 {
        return false;
    }
    if needed_to_ko <= 30
        && damage_now >= 20
        && !can_immediate_opponent_ko_conservative(state, side.id)
    {
        return true;
    }
    false
}

fn can_secure_immediate_lethal(state: &GameState, side: &SideState) -> bool {
    let Some(attacker) = &side.active else {
        return false;
    };
    if !can_attack(state, side) {
        return false;
    }
    let opponent = state.side(side.id.opposite());
    let Some(opp_active) = &opponent.active else {
        return false;
    };
    let Some(attacker_card) = umamusume_card(attacker) else {
        return false;
    };
    let Some(attack) = primary_attack(attacker_card) else {
        return false;
    };
    if !has_enough_energy(attacker, &attack.cost) {
        return false;
    }
    let own_in_play_count = 1 + side.bench.len() as i32;
    let all_in_play_count = own_in_play_count + 1 + opponent.bench.len() as i32;
    let damage = predict_attack_damage(
        attacker,
        opp_active,
        side.active_attack_damage_bonus as i32,
        own_in_play_count,
        all_in_play_count,
        Some(state.turn_number),
        are_tool_effects_disabled(state),
    );
    damage >= opp_active.hp
}

fn has_bench_recovery_option_in_hand(side: &SideState) -> bool {
    let cat = catalog();
    side.hand.iter().any(|&cid| {
        let Some(Card::Trainer(t)) = cat.get(cid) else {
            return false;
        };
        if t.effect.search_umamusume == Some(true)
            || t.effect.search_random_basic_umamusume == Some(true)
        {
            return true;
        }
        if t.effect.draw.unwrap_or(0) > 0 || t.effect.shuffle_hand_into_deck_draw.unwrap_or(0) > 0 {
            return true;
        }
        false
    })
}

fn had_recent_no_attack(state: &GameState, side_title: &str) -> bool {
    has_consecutive_no_attack_turns(state, side_title, 1)
}

/// `turnPlan.ts:179` `hasConsecutiveNoAttackTurns`.
///
/// Walks the first 64 log entries (most-recent first, matching TS
/// `unshift` semantics) looking for "${sideTitle} did not attack." and
/// "${sideTitle} attacked with " streaks.
///
/// **TS quirk preserved**: the "attacked with" log line is emitted by
/// `combat.ts:124` using `actorName(attacker)` ("You" or "Opponent")
/// rather than `side.title`, so when the heuristic searches for
/// "${title} attacked with " on the player side (title="Player AI",
/// actorName="You"), the prefix NEVER matches and the streak does not
/// reset. This is the intended TS behavior — `state.log` is bounded at
/// 12 entries by `core/log.ts`, so the heuristic's window is small
/// enough that the bug rarely produces stuck-on-streak states.
pub fn has_consecutive_no_attack_turns(
    state: &GameState,
    side_title: &str,
    required_count: u32,
) -> bool {
    let no_attack = format!("{} did not attack.", side_title);
    let attack_prefix = format!("{} attacked with ", side_title);
    let mut streak: u32 = 0;
    for entry in state.log.iter().take(64) {
        if entry == &no_attack {
            streak += 1;
            if streak >= required_count {
                return true;
            }
            continue;
        }
        if entry.starts_with(&attack_prefix) {
            streak = 0;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::{AiDeckStyle, AiDifficulty, SideId};
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
    fn choose_ai_turn_goal_returns_maximize_progress_when_no_active() {
        let state = empty_state();
        let side = state.side(SideId::Player);
        assert_eq!(choose_ai_turn_goal(&state, side), AiTurnGoal::MaximizeProgress);
    }

    #[test]
    fn has_consecutive_no_attack_turns_always_false_no_log_buffer() {
        let state = empty_state();
        assert!(!has_consecutive_no_attack_turns(&state, "You", 1));
        assert!(!has_consecutive_no_attack_turns(&state, "You", 2));
    }

    #[test]
    fn explain_ai_turn_goal_no_active_returns_no_active_tag() {
        let state = empty_state();
        let side = state.side(SideId::Player);
        let tags = explain_ai_turn_goal(&state, side);
        assert_eq!(tags, vec!["no_active"]);
    }
}
