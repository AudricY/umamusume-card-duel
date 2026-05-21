//! Port of the engine's GameState types.
//!
//! `shared/src/types.ts:160-236`. This module is the canonical Rust shape;
//! the **packed buffer** representation (`packed.rs`, Phase 1c) is a
//! separate, allocation-free serialization for MCTS clone + fingerprint.
//!
//! Boundedness audit (informs the packed buffer):
//! - `deck`: ≤20 (`DECK_CARD_COUNT`)
//! - `hand`: ≤10 (`MAX_HAND`)
//! - `discard`: ≤20 (deck size)
//! - `bench`: ≤3 (`MAX_BENCH`)
//! - `active + bench`: ≤4 in-play Umamusume per side
//! - `evolutionCardIds`: ≤2 (basic → stage1 → stage2)
//! - `specialConditions`: ≤5 (enum cardinality)
//! - `energyPool`, `energyZone`: bounded by use; cap at 16 for safety
//! - `points`: ≤3
//! - `log`: UNBOUNDED — explicitly excluded from clone path; the simulator
//!   writes to an out-of-band buffer
//! - `usedAbilityNamesThisTurn/Game`: bounded by # of ability instances in
//!   play; cap at 16
//!
//! Maps preserve TS source-declaration insertion order (`player` first,
//! then `opponent`) via `[T; 2]` indexed by `SideId as usize`.

use arrayvec::ArrayVec;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use super::card_id::{CardId, CardIdInterner};
use super::constants::{
    AiDeckStyle, AiDifficulty, CoinFlipResult, EnergyType, OpponentTurnStep, SideId,
    SpecialCondition, DECK_CARD_COUNT, MAX_BENCH, MAX_EVOLUTION_CHAIN, MAX_HAND, OPENING_HAND,
};

/// `shared/src/types.ts:160` `UmamusumeInstance`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UmamusumeInstance {
    pub uid: u32,
    pub card_id: CardId,
    pub evolution_card_ids: ArrayVec<CardId, MAX_EVOLUTION_CHAIN>,
    pub species: String,
    pub stage: u8,
    pub hp: i32,
    pub max_hp: i32,
    /// `Record<EnergyType, number>` — fixed 10 entries, indexed by
    /// `EnergyType as usize`, iteration order matches `EnergyType::ALL`.
    pub energies: [u16; EnergyType::COUNT],
    /// Push-order list of conditions currently in effect on this instance.
    pub special_conditions: ArrayVec<SpecialCondition, { SpecialCondition::COUNT }>,
    pub entered_turn: u32,
    pub evolved_turn: Option<u32>,
    pub took_damage_last_turn: bool,
    pub took_damage_this_turn: bool,
    pub next_turn_damage_reduction: i32,
    pub used_ability_this_turn: bool,
    pub attack_blocked_until_own_turn: Option<u32>,
    pub paralysed_until_own_turn: Option<u32>,
    pub tool_card_id: Option<CardId>,
}

/// `shared/src/types.ts:185` `SetupState`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetupState {
    pub coin_choice: Option<CoinFlipResult>,
    pub coin_flip_result: Option<CoinFlipResult>,
    /// `Record<SideId, string[]>` — indexed by `SideId as usize`.
    pub opening_hands: [ArrayVec<CardId, OPENING_HAND>; 2],
    pub opening_hands_dealt: bool,
    pub ready_by_side: [bool; 2],
    pub opponent_revealed: bool,
    pub countdown_seconds_remaining: Option<u32>,
}

/// `shared/src/types.ts:195` `SideState`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SideState {
    pub id: SideId,
    pub title: String,
    pub energy_pool: ArrayVec<EnergyType, { EnergyType::COUNT }>,
    pub deck: ArrayVec<CardId, DECK_CARD_COUNT>,
    pub discard: ArrayVec<CardId, DECK_CARD_COUNT>,
    pub hand: ArrayVec<CardId, MAX_HAND>,
    pub active: Option<UmamusumeInstance>,
    pub bench: ArrayVec<UmamusumeInstance, MAX_BENCH>,
    pub points: u8,
    /// Tiny queue (typically ≤2 in practice).
    pub energy_zone: ArrayVec<EnergyType, 4>,
    pub energy_attachments_this_turn: u8,
    pub bonus_energy_attachments: u8,
    pub retreat_cost_reduction: u8,
    pub active_attack_damage_bonus: i16,
    pub used_supporter_this_turn: bool,
    pub used_retreat_this_turn: bool,
    pub used_stadium_this_turn: bool,
    /// String-ability-name interning is intentionally NOT done — these are
    /// catalog-driven names; storage cost is acceptable.
    pub used_ability_names_this_turn: ArrayVec<String, 16>,
    pub used_ability_names_this_game: ArrayVec<String, 16>,
    pub guaranteed_coin_flip_heads: u8,
}

/// `shared/src/types.ts:181`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum PendingPlayerChoice {
    PromoteAfterKnockout {
        side_id: SideId,
        resume: PromoteResume,
    },
    SwitchAfterGust {
        side_id: SideId,
        resume: SwitchResume,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PromoteResume {
    FinishOpponentTurn,
    None,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SwitchResume {
    ResumeOpponentAfterFirstTrainerPass,
    ResumeOpponentAfterSecondTrainerPass,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    Setup,
    Play,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CurrentSide {
    Player,
    Opponent,
    Done,
}

impl CurrentSide {
    pub fn from_side(side: SideId) -> CurrentSide {
        match side {
            SideId::Player => CurrentSide::Player,
            SideId::Opponent => CurrentSide::Opponent,
        }
    }
    pub fn as_side(self) -> Option<SideId> {
        match self {
            CurrentSide::Player => Some(SideId::Player),
            CurrentSide::Opponent => Some(SideId::Opponent),
            CurrentSide::Done => None,
        }
    }
}

/// `shared/src/types.ts:218` `GameState`.
///
/// `log` is excluded — see module doc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameState {
    pub phase: Phase,
    pub setup: Option<SetupState>,
    pub pending_player_choice: Option<PendingPlayerChoice>,
    pub sides: [SideState; 2],
    pub current_side: CurrentSide,
    pub opponent_turn_step: Option<OpponentTurnStep>,
    pub stadium: Option<StadiumState>,
    pub turn_deadline_ms: Option<u64>,
    pub turn_number: u32,
    pub first_player: SideId,
    pub turns_taken_by_side: [u32; 2],
    pub ai_difficulty: AiDifficulty,
    pub human_by_side: [bool; 2],
    pub ai_deck_style_by_side: [AiDeckStyle; 2],
    pub game_over: bool,
    pub winner: Option<SideId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StadiumState {
    pub card_id: CardId,
    pub owner: SideId,
}

impl GameState {
    /// Helper to read a side without `state.sides[side as usize]` litter.
    pub fn side(&self, id: SideId) -> &SideState {
        &self.sides[id as usize]
    }
    pub fn side_mut(&mut self, id: SideId) -> &mut SideState {
        &mut self.sides[id as usize]
    }
}

/// Reference table for the engine: the catalog of card definitions, plus
/// the id interner used to map string card ids → `CardId`. Static for the
/// lifetime of a sim run.
#[derive(Debug, Clone)]
pub struct CardRegistry {
    pub interner: CardIdInterner,
    /// Indexed by `CardId.0 as usize`.
    pub cards: Vec<CardDef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum CardDef {
    Umamusume(UmamusumeCardDef),
    Trainer(TrainerCardDef),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UmamusumeCardDef {
    pub id: String,
    pub name: String,
    pub label: String,
    pub species: String,
    pub stage: u8,
    pub hp: i32,
    pub r#type: super::constants::UmamusumeType,
    pub weakness_type: super::constants::UmamusumeType,
    pub weakness_amount: i32,
    pub retreat: String,
    /// Catalog data is intentionally lightweight here — full Attack/Ability
    /// JSON is loaded by the codegen step (Phase 1b) and stored separately
    /// in `attacks_index` / `ability_index` for lookup by id.
    pub attacks_index: u32,
    pub ability_index: Option<u32>,
    pub evolves_from: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainerCardDef {
    pub id: String,
    pub name: String,
    pub label: String,
    pub trainer_type: super::constants::TrainerType,
    pub effect_index: u32,
}

/// Free-form attack/ability/effect tables. Loaded once at sim start; never
/// mutated. Use `IndexMap` so iteration matches TS source-declaration
/// order, which the heuristic opponent depends on (per §4 #3 audit).
#[derive(Debug, Default, Clone)]
pub struct CatalogTables {
    pub attacks: Vec<AttackDef>,
    pub abilities: Vec<AbilityDef>,
    pub trainer_effects: Vec<TrainerEffectDef>,
    pub by_id: IndexMap<String, CardId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackDef(pub serde_json::Value);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbilityDef(pub serde_json::Value);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerEffectDef(pub serde_json::Value);
