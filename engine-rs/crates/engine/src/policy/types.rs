//! Bit-identical port of `frontend/src/game/engine/ai-policy/types.ts`
//! (enumeration + observation types).

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::core::constants::SideId;

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

// ---------------------------------------------------------------------------
// PublicObservation family — ports of `ai-policy/types.ts:51-143`.
//
// JSON shape stays byte-stable with TS so `/predict` request bodies and
// the recorded training examples match what existing Python servers see.
// `serde(rename_all = "camelCase")` is the default. Fields that should be
// emitted only when present use `Option` + `skip_serializing_if`.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PublicTemporalObservation {
    pub own_turns_taken: u32,
    pub opponent_turns_taken: u32,
    pub own_is_first_turn: bool,
    pub opponent_is_first_turn: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PublicSideTurnState {
    pub energy_attachments_this_turn: u8,
    pub bonus_energy_attachments: u8,
    pub retreat_cost_reduction: u8,
    pub effective_retreat_cost_reduction: u32,
    pub active_attack_damage_bonus: i16,
    pub used_ability_name_count_this_turn: usize,
    pub used_ability_name_count_this_game: usize,
    pub guaranteed_coin_flip_heads: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PublicUmaTurnState {
    pub turns_in_play: u32,
    pub entered_this_turn: bool,
    pub evolved_this_turn: bool,
    pub evolved_last_turn: bool,
    pub took_damage_last_turn: bool,
    pub took_damage_this_turn: bool,
    pub next_turn_damage_reduction: i32,
    pub attack_blocked_this_turn: bool,
    pub paralysis_recovery_pending: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PublicUmaObservation {
    pub uid: u32,
    pub card_id: String,
    pub species: String,
    pub stage: u8,
    pub hp: i32,
    pub max_hp: i32,
    pub energy_total: u32,
    /// `Record<string, number>` — TS keys are lowercase EnergyType strings.
    pub energies: IndexMap<String, u16>,
    pub special_conditions: Vec<String>,
    pub tool_card_id: Option<String>,
    pub used_ability_this_turn: bool,
    pub turn_state: PublicUmaTurnState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PublicSideObservation {
    pub id: SideId,
    pub points: u8,
    pub hand_count: usize,
    /// Present only when the side is `own` (model side).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hand_card_ids: Option<Vec<String>>,
    pub deck_count: usize,
    pub discard: Vec<String>,
    pub active: Option<PublicUmaObservation>,
    pub bench: Vec<Option<PublicUmaObservation>>,
    pub energy_zone: Vec<String>,
    /// v3.6 obs-contract extension (per
    /// `docs/ai-research/scoping/v36-priors-and-arithmetic-scoping.md`
    /// §3.4). Per-game-setup energy pool (≤3 colors), public from setup
    /// onward (`flow/setup.rs:143`). Serialized as `energyPool` (camelCase
    /// strings matching `EnergyType` rename_all="lowercase"), mirroring the
    /// `energy_zone: Vec<String>` precedent. `#[serde(default)]` so legacy
    /// v3.5 traces (no field) still deserialize — empty vec → v3.6
    /// builder reads no pool bits (safe additive default).
    #[serde(default)]
    pub energy_pool: Vec<String>,
    pub used_supporter_this_turn: bool,
    pub used_retreat_this_turn: bool,
    pub used_stadium_this_turn: bool,
    pub turn_state: PublicSideTurnState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PublicSharedObservation {
    pub stadium_card_id: Option<String>,
    /// Matches `state.currentSide` — `"player" | "opponent" | "done"`.
    pub current_side: String,
    pub game_over: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PublicObservation {
    pub schema_version: u32,
    pub side_to_act: SideId,
    pub phase: AiPhase,
    pub turn_number: u32,
    pub first_player: SideId,
    pub pending_choice_kind: Option<String>,
    pub temporal: PublicTemporalObservation,
    pub own: PublicSideObservation,
    pub opponent: PublicSideObservation,
    pub shared: PublicSharedObservation,
    /// Per-zone card-vocab idx arrays. Lengths vary; Python collator pads.
    /// Keyed by `ZoneKey` camelCase string (`ownActive`, `oppActive`, …).
    pub card_ids_by_zone: IndexMap<String, Vec<u32>>,
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
