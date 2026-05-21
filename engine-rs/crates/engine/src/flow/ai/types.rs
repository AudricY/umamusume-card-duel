//! Bit-identical port of `frontend/src/game/engine/flow/ai/types.ts`.
//!
//! Decision / candidate / goal types for the heuristic opponent. Float
//! fields (`score`, `target_value`) are `f64` to match TS `number`.

use serde::{Deserialize, Serialize};

/// Mirror of `AiCombatDecision` in TS. `endTurn` carries no payload; the
/// `attack` variant carries the full bundle of optional choice indices the
/// AI may need to forward to `performAttack`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum AiCombatDecision {
    EndTurn,
    Attack(AttackDecision),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AttackDecision {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retreat_target_uid: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attack_target_uid: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heal_target_uid: Option<u32>,
    pub attack_index: usize,
    pub uses_coin_flip: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub discard_hand_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evolution_deck_card_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub random_discard_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub switch_target_uid: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub use_shuffle_self_into_deck: Option<bool>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AiCombatDecisionResult {
    pub resolved: bool,
    pub used_attack: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub did_retreat: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombatCandidate {
    pub id: String,
    pub decision: AiCombatDecision,
    pub score: f64,
    pub keeps_safe: bool,
    pub lethal_target: bool,
    pub target_value: f64,
    pub target_is_active: bool,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AiTacticalGoal {
    SecureLethal,
    DenyOpponentLethal,
    MaximizeExpectedDamage,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AiTurnGoal {
    SecureLethalNow,
    SetUpTwoTurnLethal,
    DenyOpponentLethal,
    StabilizeBoard,
    ProtectLoadedActive,
    BuildBackupAttacker,
    DigForEvolution,
    ConvertPointLead,
    MaximizeProgress,
}

impl AiTurnGoal {
    /// Stable identifier used for telemetry keys.
    pub fn tag(self) -> &'static str {
        match self {
            AiTurnGoal::SecureLethalNow => "secure_lethal_now",
            AiTurnGoal::SetUpTwoTurnLethal => "set_up_two_turn_lethal",
            AiTurnGoal::DenyOpponentLethal => "deny_opponent_lethal",
            AiTurnGoal::StabilizeBoard => "stabilize_board",
            AiTurnGoal::ProtectLoadedActive => "protect_loaded_active",
            AiTurnGoal::BuildBackupAttacker => "build_backup_attacker",
            AiTurnGoal::DigForEvolution => "dig_for_evolution",
            AiTurnGoal::ConvertPointLead => "convert_point_lead",
            AiTurnGoal::MaximizeProgress => "maximize_progress",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MidLevelDecision {
    pub goal: AiTacticalGoal,
    pub candidate: CombatCandidate,
}

/// Convenience: the resume kind the AI threads through to
/// `apply_trainer` / `switch_out_opponent_active`. Mirrors TS
/// `Extract<PendingPlayerChoice, { kind: "switchAfterGust" }>["resume"]`.
pub use crate::core::state::SwitchResume as PendingSwitchAfterGustResume;
