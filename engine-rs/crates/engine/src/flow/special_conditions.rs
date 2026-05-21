//! Bit-identical port of `frontend/src/game/engine/flow/specialConditions.ts`.

use crate::core::state::UmamusumeInstance;

/// Mirror of `clearSpecialConditions` — also clears the paralysis-until
/// counter, per the TS source.
pub fn clear_special_conditions(umamusume: &mut UmamusumeInstance) {
    umamusume.special_conditions.clear();
    umamusume.paralysed_until_own_turn = None;
}
