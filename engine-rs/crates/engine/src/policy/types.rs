//! Bit-identical port of `frontend/src/game/engine/ai-policy/types.ts`
//! (enumeration-side types only — observation types deferred until the
//! Rust binary needs to talk to a Python model).

use serde::{Deserialize, Serialize};

/// `ai-policy/types.ts:3` `AiPhase`. Order matters wherever the heuristic
/// opponent or MCTS indexes by phase — the existing scoring functions
/// rely on `phaseIndex` which reads source-declaration order.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AiPhase {
    Setup,
    PendingChoice,
    Bench,
    TrainerBefore,
    Evolve,
    Attach,
    TrainerAfter,
    Ability,
    Combat,
    StadiumOrEnd,
}

impl AiPhase {
    pub const ALL: [AiPhase; 10] = [
        AiPhase::Setup,
        AiPhase::PendingChoice,
        AiPhase::Bench,
        AiPhase::TrainerBefore,
        AiPhase::Evolve,
        AiPhase::Attach,
        AiPhase::TrainerAfter,
        AiPhase::Ability,
        AiPhase::Combat,
        AiPhase::StadiumOrEnd,
    ];
    pub fn index(self) -> usize {
        self as usize
    }
}

/// `ai-policy/types.ts:18` `ZoneKey` — used by the observation embedding
/// table. Declaration order is load-bearing for the Python collator.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ZoneKey {
    OwnActive,
    OppActive,
    OwnBench,
    OppBench,
    OwnHand,
    OwnDiscard,
    OppDiscard,
    Stadium,
}

impl ZoneKey {
    pub const ALL: [ZoneKey; 8] = [
        ZoneKey::OwnActive,
        ZoneKey::OppActive,
        ZoneKey::OwnBench,
        ZoneKey::OppBench,
        ZoneKey::OwnHand,
        ZoneKey::OwnDiscard,
        ZoneKey::OppDiscard,
        ZoneKey::Stadium,
    ];
    pub fn index(self) -> usize {
        self as usize
    }
}

/// `ai-policy/types.ts:28` `LegalAiAction`. `payload` is a free-form
/// JSON object because TS heuristics stuff varied fields into it (hand
/// indices, target uids, evolution choices, etc.) — Rust port keeps the
/// same shape so per-action serialization matches the recorded golden
/// traces.
///
/// `actionSourceCardIdx` / `actionTargetCardIdx` are i32-able u16
/// values (-1 sentinel becomes Python `null` via `Option<u16>` →
/// `serde` skip-if-none + a custom serializer to JSON null).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegalAiAction {
    pub id: String,
    pub phase: AiPhase,
    pub kind: String,
    #[serde(default)]
    pub payload: serde_json::Value,
    #[serde(default)]
    pub features: Vec<f64>,
    #[serde(rename = "actionSourceCardIdx")]
    pub action_source_card_idx: Option<u16>,
    #[serde(rename = "actionTargetCardIdx")]
    pub action_target_card_idx: Option<u16>,
}
