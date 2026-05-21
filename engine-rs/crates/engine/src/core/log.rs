//! Bit-identical port of `frontend/src/game/engine/core/log.ts`.
//!
//! `log(state, msg)` unshifts msg onto `state.log` and caps at 12 entries.
//! Used only by the heuristic opponent's `has_consecutive_no_attack_turns`
//! today; the fingerprint excludes the log per
//! `backend/src/sim/stateFingerprint.ts`.

use crate::core::state::GameState;

const LOG_CAP: usize = 12;

pub fn log(state: &mut GameState, message: impl Into<String>) {
    state.log.push_front(message.into());
    while state.log.len() > LOG_CAP {
        state.log.pop_back();
    }
}
