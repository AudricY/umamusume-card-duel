//! Bit-identical port of `frontend/src/game/engine/core/constants.ts` and
//! the shared enums in `shared/src/types.ts`.
//!
//! These enums are the small finite alphabets the engine pivots on. Iteration
//! order matters wherever it's user-visible (e.g. `ALL_ENERGY_TYPES`, used to
//! iterate the energy pool deterministically) — preserve TS source order
//! exactly.

use serde::{Deserialize, Serialize};

/// `shared/src/types.ts:1` — `SideId = "player" | "opponent"`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SideId {
    Player,
    Opponent,
}

impl SideId {
    pub fn opposite(self) -> SideId {
        match self {
            SideId::Player => SideId::Opponent,
            SideId::Opponent => SideId::Player,
        }
    }
    /// Stable canonical iteration order, matching how `Record<SideId, …>`
    /// fields are declared in TS source (`player` then `opponent`).
    pub const ALL: [SideId; 2] = [SideId::Player, SideId::Opponent];
}

/// `shared/src/types.ts:2`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AiDifficulty {
    Easy,
    Normal,
    Hard,
}

/// `shared/src/types.ts:3`. Note camelCase tag values.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AiDeckStyle {
    Blitz,
    ScaleBench,
    Stall,
    Balanced,
}

/// `shared/src/types.ts:4`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CoinFlipResult {
    Heads,
    Tails,
}

/// `shared/src/types.ts:5`. Order matches `ALL_ENERGY_TYPES` in
/// `frontend/src/game/engine/core/constants.ts:3`. **Do not reorder**: the
/// state fingerprint and any iteration over `energies: Record<EnergyType,
/// number>` depend on this insertion order.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EnergyType {
    Grass,
    Fire,
    Water,
    Lightning,
    Psychic,
    Fighting,
    Darkness,
    Steel,
    Colorless,
    Dragon,
}

impl EnergyType {
    pub const COUNT: usize = 10;

    /// Source-declaration order — the canonical iteration order.
    pub const ALL: [EnergyType; Self::COUNT] = [
        EnergyType::Grass,
        EnergyType::Fire,
        EnergyType::Water,
        EnergyType::Lightning,
        EnergyType::Psychic,
        EnergyType::Fighting,
        EnergyType::Darkness,
        EnergyType::Steel,
        EnergyType::Colorless,
        EnergyType::Dragon,
    ];

    pub fn label(self) -> &'static str {
        // Mirror of ENERGY_LABELS in `core/constants.ts:18`.
        match self {
            EnergyType::Grass => "Grass Energy",
            EnergyType::Fire => "Fire Energy",
            EnergyType::Water => "Water Energy",
            EnergyType::Lightning => "Lightning Energy",
            EnergyType::Psychic => "Psychic Energy",
            EnergyType::Fighting => "Fighting Energy",
            EnergyType::Darkness => "Darkness Energy",
            EnergyType::Steel => "Steel Energy",
            EnergyType::Colorless => "Colorless Energy",
            EnergyType::Dragon => "Dragon Energy",
        }
    }

    pub fn index(self) -> usize {
        self as usize
    }

    /// Used by `pool[rollEnergyFromPool]` (`core/random.ts:77`). Falls
    /// back to `Psychic` when the pool is empty — matches the TS
    /// `?? "psychic"`.
    pub fn from_pool_index(pool: &[EnergyType], idx: usize) -> EnergyType {
        pool.get(idx).copied().unwrap_or(EnergyType::Psychic)
    }
}

/// `shared/src/types.ts:6`. TitleCase tag values (different from
/// `EnergyType`) — these appear as `UmamusumeCard.type` in card JSON.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UmamusumeType {
    Grass,
    Fire,
    Water,
    Lightning,
    Psychic,
    Fighting,
    Darkness,
    Steel,
    Colorless,
    Dragon,
}

impl UmamusumeType {
    /// Mirror of `UMAMUSUME_TYPE_TO_ENERGY` in `core/constants.ts:5`.
    pub fn energy(self) -> EnergyType {
        match self {
            UmamusumeType::Grass => EnergyType::Grass,
            UmamusumeType::Fire => EnergyType::Fire,
            UmamusumeType::Water => EnergyType::Water,
            UmamusumeType::Lightning => EnergyType::Lightning,
            UmamusumeType::Psychic => EnergyType::Psychic,
            UmamusumeType::Fighting => EnergyType::Fighting,
            UmamusumeType::Darkness => EnergyType::Darkness,
            UmamusumeType::Steel => EnergyType::Steel,
            UmamusumeType::Colorless => EnergyType::Colorless,
            UmamusumeType::Dragon => EnergyType::Dragon,
        }
    }
}

/// `shared/src/types.ts:9`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrainerType {
    Supporter,
    Item,
    Stadium,
    Tool,
}

/// `shared/src/types.ts:10`. Iteration order (`asleep`, `burned`, `frozen`,
/// `paralysed`, `poisoned`) matters wherever the TS code iterates a
/// `specialConditions: SpecialCondition[]`. Most TS sites preserve push
/// order from the in-game effect, so we don't claim source-declaration
/// order is canonical for state — but it IS the canonical iteration order
/// for an `ALL`-style scan.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SpecialCondition {
    Asleep,
    Burned,
    Frozen,
    Paralysed,
    Poisoned,
}

impl SpecialCondition {
    pub const COUNT: usize = 5;
    pub const ALL: [SpecialCondition; Self::COUNT] = [
        SpecialCondition::Asleep,
        SpecialCondition::Burned,
        SpecialCondition::Frozen,
        SpecialCondition::Paralysed,
        SpecialCondition::Poisoned,
    ];
}

/// `shared/src/types.ts:13`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum OpponentTurnStep {
    Bench,
    TrainerBefore,
    Evolve,
    Attach,
    TrainerAfter,
    Ability,
    Attack,
    Finish,
}

// ---------------------------------------------------------------------------
// Game-data magic numbers. Source: shared/src/gameData.ts and
// shared/src/localDecks.ts as called out in the engine plan §2.
// ---------------------------------------------------------------------------

/// Total cards in a constructed deck.
pub const DECK_CARD_COUNT: usize = 20;

/// Hard cap on hand size (`MAX_HAND` in gameData.ts).
pub const MAX_HAND: usize = 10;

/// Hard cap on bench size (`MAX_BENCH` in gameData.ts).
pub const MAX_BENCH: usize = 3;

/// Active + bench cap.
pub const MAX_UMA_IN_PLAY_PER_SIDE: usize = MAX_BENCH + 1;

/// Opening hand size.
pub const OPENING_HAND: usize = 5;

/// Points-to-win cap.
pub const MAX_POINTS: u32 = 3;

/// Evolution chain depth (basic → stage1 → stage2).
pub const MAX_EVOLUTION_CHAIN: usize = 2;
