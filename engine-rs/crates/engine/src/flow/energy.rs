//! Bit-identical port of `frontend/src/game/engine/flow/energy.ts`.

use crate::core::constants::EnergyType;
use crate::core::effects::{Ability, EnergyCost, MoveEnergyTypes};
use crate::core::state::{SideState, UmamusumeInstance};
use crate::core::umamusume::attached_energy_count;

/// `flow/energy.ts:6` `attachEnergy`. Pops the first item from
/// `energyZone`, increments the typed energy counter on the instance, and
/// bumps `energyAttachmentsThisTurn`. No-op if the zone is empty.
///
/// The TS version writes a log line; the Rust side leaves logging to the
/// caller — the engine's log buffer is out-of-band per the scoping doc.
pub fn attach_energy(side: &mut SideState, umamusume_uid: u32) -> bool {
    let Some(&next_energy) = side.energy_zone.first() else {
        return false;
    };
    side.energy_zone.remove(0);

    // Mutate the matching instance (active or bench-by-uid).
    if let Some(active) = &mut side.active {
        if active.uid == umamusume_uid {
            active.energies[next_energy as usize] += 1;
            side.energy_attachments_this_turn += 1;
            return true;
        }
    }
    for bench_u in side.bench.iter_mut() {
        if bench_u.uid == umamusume_uid {
            bench_u.energies[next_energy as usize] += 1;
            side.energy_attachments_this_turn += 1;
            return true;
        }
    }
    // The TS source doesn't validate the target; it just attaches to the
    // passed-in instance reference. Caller bug if we get here.
    false
}

/// `flow/energy.ts:14` `hasEnoughEnergy`. Typed-energy requirements are
/// satisfied if the instance has at least that many of the specified
/// type; colorless can be paid by any energy. Returns false if any typed
/// requirement is short.
pub fn has_enough_energy(umamusume: &UmamusumeInstance, cost: &EnergyCost) -> bool {
    let required_colorless = cost.colorless() as u32;
    let mut required_typed_total: u32 = 0;
    for t in EnergyType::ALL {
        if t == EnergyType::Colorless {
            continue;
        }
        let need = cost.get(t) as u32;
        if need == 0 {
            continue;
        }
        if (umamusume.energies[t as usize] as u32) < need {
            return false;
        }
        required_typed_total += need;
    }
    attached_energy_count(umamusume) >= required_typed_total + required_colorless
}

/// `flow/energy.ts:28` `getAbilityMoveEnergyTypes`. Normalize the
/// single-or-array shape into a slice.
pub fn get_ability_move_energy_types(ability: Option<&Ability>) -> Vec<EnergyType> {
    match ability.and_then(|a| a.move_benched_energy_to_active.as_ref()) {
        None => Vec::new(),
        Some(MoveEnergyTypes::One(e)) => vec![*e],
        Some(MoveEnergyTypes::Many(v)) => v.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::card_id::CardId;
    use crate::core::constants::SideId;
    use arrayvec::ArrayVec;

    fn inst() -> UmamusumeInstance {
        UmamusumeInstance {
            uid: 1,
            card_id: CardId(0),
            evolution_card_ids: ArrayVec::new(),
            stage: 0,
            hp: 60,
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

    #[test]
    fn cost_satisfied_when_typed_and_colorless_met() {
        let mut u = inst();
        u.energies[EnergyType::Psychic as usize] = 1;
        u.energies[EnergyType::Colorless as usize] = 2;
        let cost = EnergyCost {
            by_type: {
                let mut a = [0u8; EnergyType::COUNT];
                a[EnergyType::Psychic as usize] = 1;
                a[EnergyType::Colorless as usize] = 2;
                a
            },
        };
        assert!(has_enough_energy(&u, &cost));
    }

    #[test]
    fn cost_unmet_when_typed_short() {
        let mut u = inst();
        u.energies[EnergyType::Psychic as usize] = 0;
        u.energies[EnergyType::Colorless as usize] = 3;
        let cost = EnergyCost {
            by_type: {
                let mut a = [0u8; EnergyType::COUNT];
                a[EnergyType::Psychic as usize] = 1; // typed requirement unmet
                a
            },
        };
        assert!(!has_enough_energy(&u, &cost));
    }

    #[test]
    fn attach_energy_moves_from_zone_to_instance() {
        let mut side = SideState {
            id: SideId::Player,
            title: String::new(),
            energy_pool: ArrayVec::new(),
            deck: ArrayVec::new(),
            discard: ArrayVec::new(),
            hand: ArrayVec::new(),
            active: Some(inst()),
            bench: ArrayVec::new(),
            points: 0,
            energy_zone: ArrayVec::from_iter([EnergyType::Fire]),
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
        };
        let ok = attach_energy(&mut side, 1);
        assert!(ok);
        assert_eq!(
            side.active.as_ref().unwrap().energies[EnergyType::Fire as usize],
            1
        );
        assert_eq!(side.energy_zone.len(), 0);
        assert_eq!(side.energy_attachments_this_turn, 1);
    }
}
