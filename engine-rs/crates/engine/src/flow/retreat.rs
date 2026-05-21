//! Bit-identical port of `frontend/src/game/engine/flow/retreat.ts`.

use crate::core::catalog::{catalog, Card};
use crate::core::constants::EnergyType;
use crate::core::state::{GameState, SideState, UmamusumeInstance};
use crate::core::umamusume::attached_energy_count;
use crate::flow::ability_rules::get_umamusume_ability;

/// TS source: `retreatCost(retreat)`.
/// `"Empty"` → 0; matches `x(\d+)` → that number; otherwise 1.
pub fn retreat_cost(retreat: &str) -> u32 {
    if retreat == "Empty" {
        return 0;
    }
    // Match `x(\d+)` — same regex as TS.
    if let Some(pos) = retreat.find('x') {
        let after = &retreat[pos + 'x'.len_utf8()..];
        let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !digits.is_empty() {
            if let Ok(n) = digits.parse::<u32>() {
                return n;
            }
        }
    }
    1
}

pub fn get_global_retreat_cost_reduction(state: &GameState) -> u32 {
    let Some(stadium) = &state.stadium else {
        return 0;
    };
    let cat = catalog();
    let Some(Card::Trainer(t)) = cat.get(stadium.card_id) else {
        return 0;
    };
    t.effect.global_retreat_cost_reduction.unwrap_or(0).max(0) as u32
}

pub fn effective_retreat_cost(state: &GameState, side: &SideState) -> u32 {
    let Some(active) = &side.active else {
        return 0;
    };
    let ability = get_umamusume_ability(state, side.id, active);
    if let Some(a) = ability {
        if a.retreat_cost_zero_if_took_damage_last_turn == Some(true) && active.took_damage_last_turn {
            return 0;
        }
        if a.retreat_cost_zero_if_has_energy == Some(true) && attached_energy_count(active) > 0 {
            return 0;
        }
    }
    let cat = catalog();
    let printed_cost = match cat.get(active.card_id) {
        Some(Card::Umamusume(u)) => retreat_cost(&u.retreat),
        _ => 0,
    };
    let reduction = side.retreat_cost_reduction as u32 + get_global_retreat_cost_reduction(state);
    printed_cost.saturating_sub(reduction)
}

/// `payRetreatCost`: discard energies in `ALL_ENERGY_TYPES` order until
/// the cost is paid. Order is load-bearing because it's
/// `ALL_ENERGY_TYPES` declaration order.
pub fn pay_retreat_cost(umamusume: &mut UmamusumeInstance, cost: u32) {
    let mut remaining = cost;
    for t in EnergyType::ALL {
        if remaining == 0 {
            return;
        }
        let have = umamusume.energies[t as usize] as u32;
        let discarded = have.min(remaining);
        umamusume.energies[t as usize] -= discarded as u16;
        remaining -= discarded;
    }
}

/// `payRetreatCostBySelection`: caller picks exactly `cost` energies.
/// Returns true on success.
pub fn pay_retreat_cost_by_selection(
    umamusume: &mut UmamusumeInstance,
    selected: &[EnergyType],
    cost: u32,
) -> bool {
    if selected.len() as u32 != cost {
        return false;
    }
    let mut required = [0u32; EnergyType::COUNT];
    for &e in selected {
        required[e as usize] += 1;
    }
    for t in EnergyType::ALL {
        if required[t as usize] > umamusume.energies[t as usize] as u32 {
            return false;
        }
    }
    for t in EnergyType::ALL {
        let r = required[t as usize];
        if r > 0 {
            umamusume.energies[t as usize] -= r as u16;
        }
    }
    true
}

pub fn get_displayed_retreat_cost(
    state: &GameState,
    side: &SideState,
    umamusume: &UmamusumeInstance,
) -> u32 {
    if Some(umamusume.uid) == side.active.as_ref().map(|a| a.uid) {
        return effective_retreat_cost(state, side);
    }
    let cat = catalog();
    let printed = match cat.get(umamusume.card_id) {
        Some(Card::Umamusume(u)) => retreat_cost(&u.retreat),
        _ => 0,
    };
    printed.saturating_sub(get_global_retreat_cost_reduction(state))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retreat_cost_parses_x_prefix() {
        assert_eq!(retreat_cost("Empty"), 0);
        assert_eq!(retreat_cost("Colorless"), 1);
        assert_eq!(retreat_cost("Colorlessx2"), 2);
        assert_eq!(retreat_cost("Colorlessx10"), 10);
        // Edge: trailing junk after digits should still parse the leading number.
        assert_eq!(retreat_cost("Colorlessx3whatever"), 3);
    }
}
