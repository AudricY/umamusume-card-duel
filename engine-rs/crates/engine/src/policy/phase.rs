//! Bit-identical port of `frontend/src/game/engine/ai-policy/phase.ts`.

use crate::core::constants::{OpponentTurnStep, SideId};
use crate::core::state::{CurrentSide, GameState, PendingPlayerChoice, Phase};
use crate::policy::types::AiPhase;

pub fn get_ai_phase(state: &GameState, side_id: SideId) -> AiPhase {
    if state.phase == Phase::Setup {
        return AiPhase::Setup;
    }
    if let Some(pending) = &state.pending_player_choice {
        let pending_side = match pending {
            PendingPlayerChoice::PromoteAfterKnockout { side_id, .. } => *side_id,
            PendingPlayerChoice::SwitchAfterGust { side_id, .. } => *side_id,
        };
        if pending_side == side_id {
            return AiPhase::PendingChoice;
        }
    }
    if state.current_side != CurrentSide::from_side(side_id) {
        return AiPhase::StadiumOrEnd;
    }
    match state.opponent_turn_step.unwrap_or(OpponentTurnStep::Bench) {
        OpponentTurnStep::Bench => AiPhase::Bench,
        OpponentTurnStep::TrainerBefore => AiPhase::TrainerBefore,
        OpponentTurnStep::Evolve => AiPhase::Evolve,
        OpponentTurnStep::Attach => AiPhase::Attach,
        OpponentTurnStep::TrainerAfter => AiPhase::TrainerAfter,
        OpponentTurnStep::Ability => AiPhase::Ability,
        OpponentTurnStep::Attack => AiPhase::Combat,
        OpponentTurnStep::Finish => AiPhase::StadiumOrEnd,
    }
}
