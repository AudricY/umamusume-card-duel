//! Bit-identical port of `frontend/src/game/engine/flow/ai/opponentHeuristics.ts`.
//!
//! Pure scoring functions over a `PublicSideView`. No state mutation.

use crate::core::state::UmamusumeInstance;

use super::combat_utils::predict_attack_damage;
use super::public_info::PublicSideView;

pub fn get_opponent_active_energy_count_from_public(opponent: &PublicSideView) -> i32 {
    match &opponent.active {
        None => 0,
        Some(active) => active.energies.iter().map(|&c| c as i32).sum(),
    }
}

pub fn get_aoi_kiryuin_bonus_value_from_public(
    own_active: &UmamusumeInstance,
    opponent: &PublicSideView,
    active_attack_damage_bonus: i32,
    own_bench_count: i32,
    turn_number: u32,
) -> f64 {
    let Some(target) = opponent.active.as_ref() else {
        return 0.0;
    };
    let own_in_play_count = 1 + own_bench_count;
    let all_in_play_count = own_in_play_count + 1 + opponent.bench.len() as i32;
    let without_bonus = predict_attack_damage(
        own_active,
        target,
        active_attack_damage_bonus,
        own_in_play_count,
        all_in_play_count,
        Some(turn_number),
        false,
    );
    let with_bonus = predict_attack_damage(
        own_active,
        target,
        active_attack_damage_bonus + 10,
        own_in_play_count,
        all_in_play_count,
        Some(turn_number),
        false,
    );
    let delta = (with_bonus - without_bonus).max(0);
    let ko_swing = if without_bonus < target.hp && with_bonus >= target.hp {
        90
    } else {
        0
    };
    (delta as f64) * 2.0 + ko_swing as f64
}

pub fn get_yayoi_akikawa_value_from_public(
    own_active: &UmamusumeInstance,
    opponent: &PublicSideView,
    active_attack_damage_bonus: i32,
    own_bench_count: i32,
    turn_number: u32,
) -> f64 {
    let Some(active) = opponent.active.as_ref() else {
        return 0.0;
    };
    if opponent.bench.is_empty() {
        return 0.0;
    }
    let own_in_play_count = 1 + own_bench_count;
    let all_in_play_count = own_in_play_count + 1 + opponent.bench.len() as i32;
    let damage_now = predict_attack_damage(
        own_active,
        active,
        active_attack_damage_bonus,
        own_in_play_count,
        all_in_play_count,
        Some(turn_number),
        false,
    );

    // Map each bench candidate to (damage, ko) then sort. TS:
    //   - sort puts non-ko first (smaller damage first)
    //   - the [0] is the "worst replacement for them" = AI prefers them
    //     swapping to a healthier bench, so we take the FIRST element of
    //     the post-sort list which is the most resilient option.
    let mut candidates: Vec<(i32, bool)> = opponent
        .bench
        .iter()
        .map(|bench_target| {
            let damage = predict_attack_damage(
                own_active,
                bench_target,
                active_attack_damage_bonus,
                own_in_play_count,
                all_in_play_count,
                Some(turn_number),
                false,
            );
            let ko = damage >= bench_target.hp;
            (damage, ko)
        })
        .collect();
    candidates.sort_by(|left, right| {
        if left.1 != right.1 {
            // !ko first → if left.ko { Greater } else { Less }
            return if left.1 {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Less
            };
        }
        left.0.cmp(&right.0)
    });
    let Some(opponent_best_replacement) = candidates.first() else {
        return 0.0;
    };
    let damage_after_gust_worst_case = opponent_best_replacement.0;
    let ko_now = damage_now >= active.hp;
    let ko_after_worst_case = opponent_best_replacement.1;
    if !ko_now && ko_after_worst_case {
        return 120.0;
    }
    (damage_after_gust_worst_case - damage_now).max(0) as f64
}
