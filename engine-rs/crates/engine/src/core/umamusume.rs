//! Bit-identical port of `frontend/src/game/engine/core/umamusume.ts`.
//!
//! Tiny helper module — `getAllUmamusume` iteration order is load-bearing
//! (active first, then bench in push-order), several flow files iterate
//! exactly this order to build hash keys.

use arrayvec::ArrayVec;

use crate::core::constants::MAX_UMA_IN_PLAY_PER_SIDE;
use crate::core::state::{SideState, UmamusumeInstance};

/// Inline-storage return type for `get_all_umamusume`. Capacity is
/// `MAX_UMA_IN_PLAY_PER_SIDE = MAX_BENCH + 1 = 4`, so the result stays on
/// the stack — eliminates a per-call `Vec` allocation that was visible at
/// ~11% inclusive in the rollout flamegraph at sims=800.
pub type AllUmamusume<'a> = ArrayVec<&'a UmamusumeInstance, MAX_UMA_IN_PLAY_PER_SIDE>;

/// `core/umamusume.ts:3` `attachedEnergyCount`. Sum of all attached
/// energies, regardless of type.
pub fn attached_energy_count(umamusume: &UmamusumeInstance) -> u32 {
    umamusume.energies.iter().map(|&n| n as u32).sum()
}

/// `core/umamusume.ts:7` `getAllUmamusume`. Active first (if any), then
/// bench in push-order. **Iteration order is part of the engine
/// contract.**
pub fn get_all_umamusume(side: &SideState) -> AllUmamusume<'_> {
    let mut out = AllUmamusume::new();
    if let Some(a) = &side.active {
        out.push(a);
    }
    for u in &side.bench {
        out.push(u);
    }
    out
}

/// `core/umamusume.ts:11` `getDamagedUmamusume`. Order preserved.
pub fn get_damaged_umamusume(side: &SideState) -> Vec<&UmamusumeInstance> {
    get_all_umamusume(side)
        .into_iter()
        .filter(|u| u.hp < u.max_hp)
        .collect()
}

/// `core/umamusume.ts:15` `findOwnUmamusumeByUid`.
pub fn find_own_umamusume_by_uid(side: &SideState, uid: u32) -> Option<&UmamusumeInstance> {
    get_all_umamusume(side).into_iter().find(|u| u.uid == uid)
}

/// `core/umamusume.ts:19` `findMostDamagedUmamusume`. Active is the
/// fallback "best" seed, ties broken by source iteration order (which
/// active wins because it's evaluated first).
pub fn find_most_damaged_umamusume(side: &SideState) -> Option<&UmamusumeInstance> {
    let mut iter = get_all_umamusume(side).into_iter();
    let first = iter.next()?;
    let mut best = first;
    let mut best_damage = best.max_hp - best.hp;
    for u in iter {
        let damage = u.max_hp - u.hp;
        if damage > best_damage {
            best = u;
            best_damage = damage;
        }
    }
    Some(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::card_id::CardId;
    use crate::core::constants::{EnergyType, SideId};
    use arrayvec::ArrayVec;

    fn empty_instance(uid: u32) -> UmamusumeInstance {
        UmamusumeInstance {
            uid,
            card_id: CardId(0),
            evolution_card_ids: ArrayVec::new(),
            stage: 0,
            hp: 50,
            max_hp: 60,
            energies: [0; EnergyType::COUNT],
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

    #[test]
    fn attached_energy_count_sums_all_types() {
        let mut u = empty_instance(1);
        u.energies[EnergyType::Grass as usize] = 2;
        u.energies[EnergyType::Fire as usize] = 1;
        u.energies[EnergyType::Colorless as usize] = 3;
        assert_eq!(attached_energy_count(&u), 6);
    }

    #[test]
    fn get_all_umamusume_active_first_then_bench() {
        let mut side = empty_side(SideId::Player);
        side.active = Some(empty_instance(1));
        side.bench.push(empty_instance(2));
        side.bench.push(empty_instance(3));
        let order: Vec<u32> = get_all_umamusume(&side)
            .into_iter()
            .map(|u| u.uid)
            .collect();
        assert_eq!(order, vec![1, 2, 3]);
    }

    #[test]
    fn find_most_damaged_picks_largest_damage_or_active_on_tie() {
        let mut side = empty_side(SideId::Player);
        let mut active = empty_instance(1);
        active.hp = 50;
        active.max_hp = 60;
        side.active = Some(active);
        let mut bench = empty_instance(2);
        bench.hp = 30;
        bench.max_hp = 60;
        side.bench.push(bench);
        let pick = find_most_damaged_umamusume(&side).unwrap();
        assert_eq!(pick.uid, 2, "bench has 30 damage vs active's 10");
    }
}
