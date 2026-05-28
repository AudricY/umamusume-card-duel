//! Bit-identical port of `frontend/src/game/engine/flow/ai/energyAwareness.ts`.
//!
//! Pool-fit scoring. TS uses `Object.entries(cost).reduce(...)` over a
//! `Partial<Record<EnergyType, number>>`. Rust iterates `EnergyType::ALL`
//! to preserve declaration order; only non-colorless typed entries with
//! a non-zero requirement contribute.

use crate::core::constants::EnergyType;
use crate::core::effects::EnergyCost;
use crate::core::state::SideState;

pub fn get_unsupported_typed_energy(state_side: &SideState, cost: &EnergyCost) -> i32 {
    // TS builds `new Set(stateSide.energyPool)` then iterates `cost`. We
    // walk the cost in declaration order and look up membership in the
    // pool — equivalent because the reducer sums missing amounts.
    let mut missing: i32 = 0;
    for energy_type in EnergyType::ALL {
        if energy_type == EnergyType::Colorless {
            continue;
        }
        let amount = cost.get(energy_type) as i32;
        if amount == 0 {
            continue;
        }
        let in_pool = state_side.energy_pool.iter().any(|&e| e == energy_type);
        if !in_pool {
            missing += amount;
        }
    }
    missing
}

pub fn is_attack_supported_by_energy_pool(state_side: &SideState, cost: &EnergyCost) -> bool {
    get_unsupported_typed_energy(state_side, cost) == 0
}

/// Mirror of `scoreAttackEnergyPoolFit`.
///
/// Returns 14.0 when fully supported, else `-45 - unsupportedTyped * 22`.
pub fn score_attack_energy_pool_fit(state_side: &SideState, cost: &EnergyCost) -> f64 {
    let unsupported_typed = get_unsupported_typed_energy(state_side, cost);
    if unsupported_typed <= 0 {
        return 14.0;
    }
    (-45 - unsupported_typed * 22) as f64
}
