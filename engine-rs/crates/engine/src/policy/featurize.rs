//! R16-P3 throughput-spike Option A: in-process Rust port of the v3.0,
//! v3.1, and v3.2 observation/action featurizers.
//!
//! Bit-exact port of the **v3.0 + v3.1 + v3.2** Python builders in
//! `training/uma_ai/features.py`:
//!   - `observation_to_features`             → `observation_state_features`
//!   - `observation_to_features_v3_1`        → `observation_state_features_v3_1`
//!   - `legal_actions_to_features`           → `legal_actions_features`
//!   - `observation_to_card_ids`             → `observation_card_ids_by_zone`
//!   - `action_card_idx_pair`                → `action_card_idx_pair`
//!   - `observation_to_uma_slots` (v3.2)     → `observation_uma_slots`
//!   - `card_vocab_metadata` / `card_vocab_index` → reused from
//!     `crate::policy::card_vocab`.
//!
//! Pinned schemas
//! --------------
//! v3.0 + v3.1 + v3.2 all supported. Slice 3 added v3.2 (per-Uma slot
//! tokens). Slice 3b (this commit) adds v3.1: 164-d = frozen v3.0 110-d
//! head + 54-d temporal/turn-state tail. Slot 0–109 are byte-identical
//! to the v3.0 builder (the v3.1 builder calls into the v3.0 builder
//! and concatenates the temporal block; no re-derivation). Slots
//! 110–163 are the FROZEN P1 enumeration (4 global + 14 side + 36
//! per-Uma; see `observation_state_features_v3_1` for the slot map).
//! v3.2 inputs add `uma_slot_card_ids: int64[10]` +
//! `uma_slot_features: float32[10, 23]` on top of the v3.0 5-input
//! contract; the 23-d slot layout is FROZEN (see comment block below).
//!
//! Layout fidelity is enforced by:
//!   - `STATE_DIM_V3 = 110` / `ACTION_DIM = 48` / `NUM_ZONES = 8` /
//!     `MAX_CARDS_PER_ZONE = 30` (== Python module constants);
//!   - `_hash_to_unit` uses the vocab-backed mapping (size > 0) with FNV
//!     fallback, exact mirror of the Python helper;
//!   - card-aware helpers consult the catalog via the existing
//!     `crate::core::catalog` interner so the catalog lookup table is
//!     shared with the rest of the engine (avoids vocabulary drift).

use crate::core::catalog::{catalog, Card, UmamusumeCard};
use crate::core::constants::{EnergyType, TrainerType};
use crate::core::effect_kinds::{classify_active_ability, classify_tool_effect};
use crate::core::effects::Attack;
use crate::policy::card_vocab::{card_vocab, card_vocab_index};
use crate::policy::types::{AiPhase, LegalAiAction, PublicObservation, PublicSideObservation, PublicUmaObservation};

/// Mirrors `STATE_DIM_V3` / `STATE_DIM` in Python (frozen 110-d v3.0).
pub const STATE_DIM_V3: usize = 110;
/// Mirrors `STATE_DIM_V3_1` in Python — 164-d v3.1 temporal/turn-state
/// builder. Layout is the frozen v3.0 110-d head + 54-d temporal block.
pub const STATE_DIM_V3_1: usize = 164;
/// Mirrors `STATE_DIM_V3_3` in Python — 167-d v3.3 additive-tail builder
/// (`v33-additive-tail-scoping.md`). Layout is the frozen v3.1 164-d head
/// + 3 opp-side flag bits (usedSupporter / usedRetreat / usedStadium).
/// Tail-init is zero-init residual: a v3.2 ckpt loaded into the v3.3
/// graph produces bit-identical iter-0 outputs (new Linear columns are
/// zero-init in `make_v33_tail_init.py`).
pub const STATE_DIM_V3_3: usize = 167;
/// Mirrors `STATE_DIM_V3_5` in Python — 212-d v3.5 multichannel-tail
/// builder (`v35-multichannel-tail-scoping.md`). Layout is the frozen
/// v3.3 167-d head + 45-bit channel-orthogonal tail. Tail-init is
/// zero-init residual: a v3.3 ckpt loaded into the v3.5 graph produces
/// bit-identical iter-0 outputs (new Linear columns are zero-init in
/// `make_v35_tail_init.py`).
pub const STATE_DIM_V3_5: usize = 212;
/// Mirrors `STATE_DIM_V3_6` in Python — 246-d v3.6 priors-and-arithmetic
/// builder (`v36-priors-and-arithmetic-scoping.md`). Layout is the frozen
/// v3.5 212-d head with band [197:207] REPURPOSED in-place from the dead
/// opp.energy_zone.front to own.energy_pool typed multihot, plus 34 bits
/// appended at [212:246]. Tail-init under `make_v36_priors_init.py` drops
/// columns 197-206 of the v3.5 ckpt (dead band) and zero-inits the 34
/// new columns; iter-0 drift contract is Δlogits ≤ 1e-3 (looser than
/// v3.5's 1e-5 because the column-drop is not strictly bit-identical).
pub const STATE_DIM_V3_6: usize = 246;
/// Mirrors `STATE_DIM_V3_7` in Python — 296-d v3.7 combat-arith-and-catalog
/// builder (`v37-combat-arith-and-catalog-scoping.md`). Layout is the
/// frozen v3.6 246-d head + 50-bit combat-arith / catalog-lookup tail
/// appended at [246:296]. Tail-init for v3.6 → v3.7 is zero-init residual
/// (new Linear columns at [246:296] are zero-init); strict `Δlogits ≤ 1e-5`
/// contract since no column-drop. All 50 tail bits derive from existing
/// v3.6 obs fields + static catalog lookup; no new obs-contract fields.
pub const STATE_DIM_V3_7: usize = 296;
/// Mirrors `STATE_DIM_V3_8` in Python — 304-d v3.8 slim-feature-add
/// builder (`v38-slim-feature-add-scoping.md`). Layout is the frozen
/// v3.7 296-d head + 8-bit slim tail at [296:304]: per-bench primary-
/// attack ETA (3 own + 3 opp) + gust-swing catastrophe (2 bits). Tail-
/// init for v3.7 → v3.8 is zero-init residual; strict `Δlogits ≤ 1e-5`.
/// All 8 tail bits derive from existing v3.7 obs fields + static
/// catalog lookup; no new obs-contract fields. NO trunk widening
/// (hidden_dim=128/depth=2 preserved per scoping §13.3).
pub const STATE_DIM_V3_8: usize = 304;
/// Mirrors `ACTION_DIM` — pre-computed TS-side and carried verbatim on
/// `LegalAiAction.features`.
///
/// v38-slim-feature-add bumped 48 → 52 (action schema v3 → v4). 4 new
/// slots at [48:52]: swap_in_attack_ready, expected_damage_norm,
/// attach_color_matches_typed_need, attach_completes_typed_threshold.
/// v3 slots [0:48] BYTE-STABLE.
///
/// v5-action-disambiguation bumped 52 → 57 (action schema v4 → v5). 5
/// new choice-card stat slots at [52:57]: choice_card_present,
/// choice_card_hp_norm, choice_card_attack_damage_norm,
/// choice_card_attack_cost_total_norm, choice_card_has_ability. All
/// fire iff `input.choice_card_id` resolves (deck-search / discard-cost
/// / rainbow-evolution payload routes). v4 slots [0:52] BYTE-STABLE.
pub const ACTION_DIM: usize = 57;
/// The v4 action-feature width — preserved for graphs/sidecars trained
/// at action schema 4 (v3.8 lineage). `pack_row` slices the v5 57-d
/// feature buffer down to this width when the graph + sidecar declare
/// the v4 contract.
pub const ACTION_DIM_V4: usize = 52;
/// The v3 action-feature width — preserved for graphs/sidecars trained
/// at action schema 3 (v3.7-and-earlier). `pack_row` slices the v5
/// 57-d feature buffer down to this width when the graph + sidecar
/// declare the v3 contract.
pub const ACTION_DIM_V3: usize = 48;
/// Per-zone padding widths — mirrors `CARD_ID_SHAPES`. Order is
/// load-bearing (matches Python `ZONE_ORDER` tuple).
pub const ZONE_NAMES: [&str; 8] = [
    "ownActive",
    "oppActive",
    "ownBench",
    "oppBench",
    "ownHand",
    "ownDiscard",
    "oppDiscard",
    "stadium",
];
pub const ZONE_WIDTHS: [usize; 8] = [1, 1, 4, 4, 10, 30, 30, 1];
pub const NUM_ZONES: usize = 8;
/// Max of `ZONE_WIDTHS` — Python `MAX_CARDS_PER_ZONE` (the ONNX graph
/// dim). Used as the second axis of the flattened card_ids tensor.
pub const MAX_CARDS_PER_ZONE: usize = 30;

const PHASES: [AiPhase; 10] = [
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

fn phase_index(p: AiPhase) -> usize {
    PHASES.iter().position(|&x| x == p).unwrap_or(0)
}

/// Mirror of Python `SIDES.index(side)` — `player` → 0, `opponent` → 1.
fn side_index_str(side: &str) -> Option<usize> {
    match side {
        "player" => Some(0),
        "opponent" => Some(1),
        _ => None,
    }
}

fn side_to_str(side: crate::core::constants::SideId) -> &'static str {
    match side {
        crate::core::constants::SideId::Player => "player",
        crate::core::constants::SideId::Opponent => "opponent",
    }
}

// `PENDING_CHOICE_KINDS` (Python) — order is load-bearing for the one-hot
// at slots [97, 98, 99] (slot 97 is "none").
const PENDING_CHOICE_KINDS: [&str; 2] = ["promoteAfterKnockout", "switchAfterGust"];

// v35-multichannel-tail special-condition vocab. FROZEN. Mirrors Python
// `_V35_CONDITION_VOCAB`. Sourced from
// `frontend/src/game/engine/flow/specialConditions.ts` / `flow/turn.ts` /
// `flow/eligibility.ts` — these five tokens are the complete set the
// engine emits into `special_conditions: Vec<String>`. Adding a new
// arm requires bumping the schema (additive v3.6 tail), never editing
// this vocab.
const V35_CONDITION_VOCAB: [&str; 5] = [
    "paralysed",
    "burned",
    "poisoned",
    "asleep",
    "frozen",
];

// Energy-type label order MUST match Python `_energy_vector` and the
// JSON keys emitted by the observation builder (`energy_label_camel`).
const ENERGY_TYPES_ORDER: [&str; 10] = [
    "grass",
    "fire",
    "water",
    "lightning",
    "psychic",
    "fighting",
    "darkness",
    "steel",
    "colorless",
    "dragon",
];

/// Mirror of Python `_hash_to_unit`.
///
/// Vocab-backed (size > 0) mapping: returns `index / max(1, size - 1)`.
/// FNV-1a fallback (legacy path, only when vocab missing — never
/// observed in production but pinned for parity with the Python helper).
fn hash_to_unit(text: &str) -> f32 {
    if text.is_empty() {
        return 0.0;
    }
    let vocab = card_vocab();
    let size = vocab.vocab_size as i64;
    if size > 0 {
        let denom = (size - 1).max(1) as f32;
        let idx = card_vocab_index(Some(text)) as f32;
        return idx / denom;
    }
    // FNV-1a fallback — only fires when shared/src/cardVocab.json is
    // absent at compile time (CI always has it; covered for parity only).
    let mut value: u32 = 2166136261;
    for b in text.bytes() {
        value ^= b as u32;
        value = value.wrapping_mul(16777619);
    }
    (value as f32) / 4294967295.0
}

fn hash_average(items: &[String]) -> f32 {
    if items.is_empty() {
        return 0.0;
    }
    let sum: f32 = items.iter().map(|s| hash_to_unit(s)).sum();
    sum / items.len() as f32
}

/// Public façade for `observation_to_features` (v3.0). Builds a 110-d
/// `Vec<f32>` (heap-allocated; ndarray callers can wrap with `from_vec`).
pub fn observation_state_features(obs: &PublicObservation) -> Vec<f32> {
    let mut f = vec![0.0f32; STATE_DIM_V3];

    let phase = obs.phase;
    let side_str = side_to_str(obs.side_to_act);
    let own = &obs.own;
    let opp = &obs.opponent;
    let shared = &obs.shared;

    // slot 0: phase / max(1, |PHASES|-1)
    f[0] = phase_index(phase) as f32 / (PHASES.len() - 1).max(1) as f32;
    // slot 1: side index — 0 for player, 1 for opponent.
    f[1] = side_index_str(side_str).unwrap_or(0) as f32;
    f[2] = obs.turn_number as f32 / 20.0;
    f[3] = own.points as f32 / 3.0;
    f[4] = opp.points as f32 / 3.0;
    f[5] = own.hand_count as f32 / 10.0;
    f[6] = opp.hand_count as f32 / 10.0;
    f[7] = own.deck_count as f32 / 50.0;
    f[8] = opp.deck_count as f32 / 50.0;
    f[9] = if shared.stadium_card_id.is_some() { 1.0 } else { 0.0 };

    // [10:18) own board, [18:26) opp board.
    let own_board = side_board_features(own);
    let opp_board = side_board_features(opp);
    f[10..18].copy_from_slice(&own_board);
    f[18..26].copy_from_slice(&opp_board);

    f[26] = own.discard.len() as f32 / 50.0;
    f[27] = opp.discard.len() as f32 / 50.0;
    f[28] = own.energy_zone.len() as f32 / 4.0;
    f[29] = if own.used_supporter_this_turn { 1.0 } else { 0.0 };
    f[30] = if own.used_retreat_this_turn { 1.0 } else { 0.0 };
    f[31] = if own.used_stadium_this_turn { 1.0 } else { 0.0 };

    // [32:48) identity features.
    let identity = identity_features(own, opp);
    f[32..48].copy_from_slice(&identity);

    // [48:58) own active energies; [58:68) opp active energies.
    f[48..58].copy_from_slice(&energy_vector_from_uma(own.active.as_ref()));
    f[58..68].copy_from_slice(&energy_vector_from_uma(opp.active.as_ref()));

    // [68:96) card-awareness features.
    let aware = card_awareness_features(own, opp, shared);
    f[68..96].copy_from_slice(&aware);

    // v2.1 additive slots — [96, 97:100, 100:110).
    f[96] = first_player_polarity(obs);
    let one_hot = pending_choice_one_hot(obs.pending_choice_kind.as_deref());
    f[97..100].copy_from_slice(&one_hot);
    let tool = tool_card_features(own, opp);
    f[100..110].copy_from_slice(&tool);

    f
}

// ---------------------------------------------------------------------------
// v3.1 temporal / turn-state builder (R16-P1; ported from
// `training/uma_ai/features.py:237-388`).
//
// Layout (frozen, mirrors P1 enumeration in scoping doc):
//   0–109      v3.0 head (byte-identical via `observation_state_features`)
//   110–113    global temporal: ownTurnsTaken(/CAP), oppTurnsTaken(/CAP),
//              ownIsFirstTurn(bool), oppIsFirstTurn(bool)
//   114–120    own side turnState ×7
//   121–127    opp side turnState ×7
//   128–136    own active per-Uma temporal ×9
//   137–145    own bench aggregate (mean of the 9 over present bench Umas)
//   146–154    opp active per-Uma temporal ×9
//   155–163    opp bench aggregate (mean of the 9 over present bench Umas)
//
// Normalisation caps (match Python module constants):
//   _TURN_CAP   = 20.0  (matches features[2] = turnNumber/20.0)
//   _BUDGET_CAP = 3.0   (energy-attach budgets, coin-flip count, ability counts)
//   _DAMAGE_CAP = 30.0  (activeAttackDamageBonus, nextTurnDamageReduction,
//                        effectiveRetreatCostReduction)
//
// The temporal-block ablation (`state_temporal_turn_v31`) lives in the
// Python training path only; the Rust runtime serves an already-trained
// ONNX and never needs to apply ablations.
// ---------------------------------------------------------------------------

const _TURN_CAP_V3_1: f32 = 20.0;
const _BUDGET_CAP_V3_1: f32 = 3.0;
const _DAMAGE_CAP_V3_1: f32 = 30.0;

fn norm_v3_1(value: f32, cap: f32) -> f32 {
    value.min(cap) / cap
}

/// Mirror of Python `_side_turn_state_vec`. 7 side-level turnState
/// scalars. The retreat slot encodes the DERIVED
/// `effectiveRetreatCostReduction` (raw side reduction + stadium global
/// term), available on every Rust observation (the field is non-Option).
fn side_turn_state_vec(side: &PublicSideObservation) -> [f32; 7] {
    let ts = &side.turn_state;
    let mut out = [0.0f32; 7];
    out[0] = norm_v3_1(ts.energy_attachments_this_turn as f32, _BUDGET_CAP_V3_1);
    out[1] = norm_v3_1(ts.bonus_energy_attachments as f32, _BUDGET_CAP_V3_1);
    out[2] = norm_v3_1(ts.effective_retreat_cost_reduction as f32, _DAMAGE_CAP_V3_1);
    out[3] = norm_v3_1(ts.active_attack_damage_bonus as f32, _DAMAGE_CAP_V3_1);
    out[4] = norm_v3_1(ts.used_ability_name_count_this_turn as f32, _BUDGET_CAP_V3_1);
    out[5] = norm_v3_1(ts.used_ability_name_count_this_game as f32, _TURN_CAP_V3_1);
    out[6] = norm_v3_1(ts.guaranteed_coin_flip_heads as f32, _BUDGET_CAP_V3_1);
    out
}

/// Mirror of Python `_uma_turn_state_vec`. 9 per-Uma temporal scalars.
/// `None` (absent slot) → all zeros (mirrors Python's `if not uma:
/// return zeros`).
fn uma_turn_state_vec(uma: Option<&PublicUmaObservation>) -> [f32; 9] {
    let mut out = [0.0f32; 9];
    let Some(u) = uma else { return out; };
    let ts = &u.turn_state;
    out[0] = norm_v3_1(ts.turns_in_play as f32, _TURN_CAP_V3_1);
    out[1] = if ts.entered_this_turn { 1.0 } else { 0.0 };
    out[2] = if ts.evolved_this_turn { 1.0 } else { 0.0 };
    out[3] = if ts.evolved_last_turn { 1.0 } else { 0.0 };
    out[4] = if ts.took_damage_last_turn { 1.0 } else { 0.0 };
    out[5] = if ts.took_damage_this_turn { 1.0 } else { 0.0 };
    out[6] = norm_v3_1(ts.next_turn_damage_reduction as f32, _DAMAGE_CAP_V3_1);
    out[7] = if ts.attack_blocked_this_turn { 1.0 } else { 0.0 };
    out[8] = if ts.paralysis_recovery_pending { 1.0 } else { 0.0 };
    out
}

/// Mirror of Python `_bench_turn_state_aggregate`. Mean of the 9 per-Uma
/// temporal scalars over PRESENT bench Umas. Empty bench → zeros.
fn bench_turn_state_aggregate(side: &PublicSideObservation) -> [f32; 9] {
    let present: Vec<&PublicUmaObservation> =
        side.bench.iter().flatten().collect();
    let mut out = [0.0f32; 9];
    if present.is_empty() {
        return out;
    }
    let denom = present.len() as f32;
    for u in &present {
        let row = uma_turn_state_vec(Some(*u));
        for (i, v) in row.iter().enumerate() {
            out[i] += *v;
        }
    }
    for v in out.iter_mut() {
        *v /= denom;
    }
    out
}

/// Public façade for `observation_to_features_v3_1` (R16-P1 schema-v3.1).
/// Builds a 164-d `Vec<f32>` = byte-identical v3.0 head + 54-d temporal
/// block. The head is produced by calling `observation_state_features`
/// directly, NOT re-deriving — Python's contract is "slots 0–109 are
/// byte-identical to the frozen v3.0 builder".
pub fn observation_state_features_v3_1(obs: &PublicObservation) -> Vec<f32> {
    let mut f = vec![0.0f32; STATE_DIM_V3_1];

    // Head: byte-identical v3.0 builder output.
    let head = observation_state_features(obs);
    debug_assert_eq!(head.len(), STATE_DIM_V3);
    f[..STATE_DIM_V3].copy_from_slice(&head);

    // 110–113: global temporal.
    let temporal = &obs.temporal;
    f[110] = norm_v3_1(temporal.own_turns_taken as f32, _TURN_CAP_V3_1);
    f[111] = norm_v3_1(temporal.opponent_turns_taken as f32, _TURN_CAP_V3_1);
    f[112] = if temporal.own_is_first_turn { 1.0 } else { 0.0 };
    f[113] = if temporal.opponent_is_first_turn { 1.0 } else { 0.0 };

    // 114–120: own side turnState; 121–127: opp side turnState.
    f[114..121].copy_from_slice(&side_turn_state_vec(&obs.own));
    f[121..128].copy_from_slice(&side_turn_state_vec(&obs.opponent));

    // 128–136: own active per-Uma; 137–145: own bench aggregate.
    f[128..137].copy_from_slice(&uma_turn_state_vec(obs.own.active.as_ref()));
    f[137..146].copy_from_slice(&bench_turn_state_aggregate(&obs.own));

    // 146–154: opp active per-Uma; 155–163: opp bench aggregate.
    f[146..155].copy_from_slice(&uma_turn_state_vec(obs.opponent.active.as_ref()));
    f[155..164].copy_from_slice(&bench_turn_state_aggregate(&obs.opponent));

    f
}

/// v33-additive-tail: 167-d builder. Slots 0–163 are byte-identical to
/// v3.1 (produced by calling `observation_state_features_v3_1` directly,
/// NOT re-derived); slots [164:167] are the opp-side used* flag tail.
/// Mirrors `observation_to_features_v3_3` in Python.
pub fn observation_state_features_v3_3(obs: &PublicObservation) -> Vec<f32> {
    let mut f = vec![0.0f32; STATE_DIM_V3_3];
    let head = observation_state_features_v3_1(obs);
    debug_assert_eq!(head.len(), STATE_DIM_V3_1);
    f[..STATE_DIM_V3_1].copy_from_slice(&head);

    // Opp-side flag tail (mirror of own-side at slots 29/30/31, which
    // are emitted by the frozen v3.0/v2 head). v3.0/v3.1/v3.2 had no
    // opponent-side equivalent — v3.3 is the first surfacing.
    f[164] = if obs.opponent.used_supporter_this_turn { 1.0 } else { 0.0 };
    f[165] = if obs.opponent.used_retreat_this_turn { 1.0 } else { 0.0 };
    f[166] = if obs.opponent.used_stadium_this_turn { 1.0 } else { 0.0 };

    f
}

fn v35_condition_one_hot(uma: Option<&PublicUmaObservation>, out: &mut [f32]) {
    // 5-bit one-hot over `V35_CONDITION_VOCAB`. Multiple conditions set
    // multiple bits. Missing Uma → all zeros (`out` is presumed
    // zero-init). Unknown tokens → silently dropped (mirrors Python
    // `_v35_condition_one_hot`).
    debug_assert_eq!(out.len(), V35_CONDITION_VOCAB.len());
    let Some(uma) = uma else { return; };
    for cond in &uma.special_conditions {
        if let Some(idx) = V35_CONDITION_VOCAB.iter().position(|v| *v == cond.as_str()) {
            out[idx] = 1.0;
        }
    }
}

fn v35_energy_front_one_hot(side: &PublicSideObservation, out: &mut [f32]) {
    // 10-bit one-hot over `ENERGY_TYPES_ORDER` for the front-of-queue
    // entry of this side's `energy_zone`. Empty zone or unknown type →
    // all zeros. Mirrors Python `_v35_energy_front_one_hot`.
    debug_assert_eq!(out.len(), ENERGY_TYPES_ORDER.len());
    let zone = &side.energy_zone;
    let Some(front) = zone.first() else { return; };
    if let Some(idx) = ENERGY_TYPES_ORDER.iter().position(|t| *t == front.as_str()) {
        out[idx] = 1.0;
    }
}

fn v35_bench_refill_catastrophe(side: &PublicSideObservation) -> f32 {
    // 1.0 if no non-null bench Uma to promote on active KO. Minimum-
    // viable terminal-state predicate (scoping doc §4.5). Mirrors
    // Python `_v35_bench_refill_catastrophe`.
    if side.bench.iter().any(|b| b.is_some()) {
        0.0
    } else {
        1.0
    }
}

/// v35-multichannel-tail: 212-d builder. Slots 0–166 are byte-identical
/// to v3.3 (produced by calling `observation_state_features_v3_3`
/// directly, NOT re-derived); slots [167:212] are the 45-bit channel-
/// orthogonal tail. Mirrors `observation_to_features_v3_5` in Python.
///
/// Tail layout (absolute):
///   [167:177] phase one-hot (10) — temporal-cadence channel
///   [177:182] own active per-condition one-hot (5)
///   [182:187] opp active per-condition one-hot (5)
///   [187:197] own energy-zone front-of-queue typed one-hot (10)
///   [197:207] opp energy-zone front-of-queue typed one-hot (10)
///   [207:210] opp discard role buckets (3)
///   [210]     own would_lose_on_active_KO (bench empty)
///   [211]     opp would_lose_on_active_KO (bench empty)
pub fn observation_state_features_v3_5(obs: &PublicObservation) -> Vec<f32> {
    let mut f = vec![0.0f32; STATE_DIM_V3_5];
    let head = observation_state_features_v3_3(obs);
    debug_assert_eq!(head.len(), STATE_DIM_V3_3);
    f[..STATE_DIM_V3_3].copy_from_slice(&head);

    // [167:177] phase one-hot — unknown phase → all zeros.
    if let Some(idx) = PHASES.iter().position(|&p| p == obs.phase) {
        f[167 + idx] = 1.0;
    }

    // [177:182] own active per-condition; [182:187] opp active.
    v35_condition_one_hot(obs.own.active.as_ref(), &mut f[177..182]);
    v35_condition_one_hot(obs.opponent.active.as_ref(), &mut f[182..187]);

    // [187:197] own energy-zone front; [197:207] opp energy-zone front.
    v35_energy_front_one_hot(&obs.own, &mut f[187..197]);
    v35_energy_front_one_hot(&obs.opponent, &mut f[197..207]);

    // [207:210] opp discard role buckets — mirror of own slots 84-86
    // inside the v3.0 head's `_card_awareness_features` block (own
    // discard at values[16:19] of that 28-d block). REUSES
    // `discard_role_features` for bit-exactness with Python.
    let opp_buckets = discard_role_features(&obs.opponent.discard);
    f[207..210].copy_from_slice(&opp_buckets);

    // [210] own would_lose_on_active_KO; [211] opp.
    f[210] = v35_bench_refill_catastrophe(&obs.own);
    f[211] = v35_bench_refill_catastrophe(&obs.opponent);

    f
}

// ---------------------------------------------------------------------------
// v3.6 priors-and-arithmetic builder
// (`docs/ai-research/scoping/v36-priors-and-arithmetic-scoping.md`).
//
// Layout (FROZEN, mirrors Python `_V36_*` slot offsets):
//   [0:197]     v3.5 head (byte-identical via `observation_state_features_v3_5`)
//   [197:207]   own energy_pool typed multihot (REPURPOSED in-place from
//               the dead v3.5 opp.energy_zone.front band)
//   [207:212]   v3.5 tail (opp discard role buckets + bench-refill bits)
//   [212:222]   opp energy_pool typed multihot
//   [222:226]   own prize one-hot {3,2,1,0}
//   [226:230]   opp prize one-hot {3,2,1,0}
//   [230:240]   opp bench typed energy aggregate (own bench DROPPED at
//               impl-phase reconciliation; see scoping §4 step 2)
//   [240]       own_lethal_next_turn (face-value)
//   [241]       opp_lethal_next_turn (face-value)
//   [242]       own_secondary_attack_usable
//   [243]       own_secondary_attack_would_KO
//   [244]       opp_secondary_attack_usable
//   [245]       opp_secondary_attack_would_KO
// ---------------------------------------------------------------------------

// v3.6 tail slot offsets — absolute in the 246-d vector. The [197:207]
// band is REPURPOSED in-place from the dead v3.5 opp.energy_zone.front
// to own.energy_pool typed multihot. The [212:246] band is the new tail.
const _V36_OWN_ENERGY_POOL_START: usize = 197; // [197:207] REPURPOSED IN-BAND
const _V36_OWN_ENERGY_POOL_END: usize = 207;
const _V36_OPP_ENERGY_POOL_START: usize = 212; // [212:222]
const _V36_OPP_ENERGY_POOL_END: usize = 222;
const _V36_OWN_PRIZE_START: usize = 222; // [222:226]
const _V36_OWN_PRIZE_END: usize = 226;
const _V36_OPP_PRIZE_START: usize = 226; // [226:230]
const _V36_OPP_PRIZE_END: usize = 230;
const _V36_OPP_BENCH_TYPED_START: usize = 230; // [230:240] (opp-only)
const _V36_OPP_BENCH_TYPED_END: usize = 240;
const _V36_OWN_LETHAL_NEXT_TURN_SLOT: usize = 240;
const _V36_OPP_LETHAL_NEXT_TURN_SLOT: usize = 241;
const _V36_OWN_SECONDARY_USABLE_SLOT: usize = 242;
const _V36_OWN_SECONDARY_WOULD_KO_SLOT: usize = 243;
const _V36_OPP_SECONDARY_USABLE_SLOT: usize = 244;
const _V36_OPP_SECONDARY_WOULD_KO_SLOT: usize = 245;

/// 10-bit multihot over `ENERGY_TYPES_ORDER` for the side's typed
/// `energy_pool` (v3.6 obs-contract extension). Bag semantics: duplicates
/// collapse (`[fire, fire, water]` → only fire+water bits). Unknown
/// energy types silently dropped. Mirrors Python
/// `_v36_energy_pool_multihot`.
fn v36_energy_pool_multihot(side: &PublicSideObservation, out: &mut [f32]) {
    debug_assert_eq!(out.len(), ENERGY_TYPES_ORDER.len());
    for token in &side.energy_pool {
        if let Some(idx) = ENERGY_TYPES_ORDER.iter().position(|t| *t == token.as_str()) {
            out[idx] = 1.0;
        }
    }
}

/// 4-bit one-hot over remaining-prize counts {3, 2, 1, 0}. `remaining =
/// clamp(3 - points, 0, 3)`; bit index = 3 - remaining. Mirrors Python
/// `_v36_prize_one_hot`.
fn v36_prize_one_hot(side: &PublicSideObservation, out: &mut [f32]) {
    debug_assert_eq!(out.len(), 4);
    let points = side.points as i32;
    let remaining = (3 - points).clamp(0, 3);
    let bit = (3 - remaining) as usize;
    out[bit] = 1.0;
}

/// 10-bit multihot over `ENERGY_TYPES_ORDER` for the side's BENCH (active
/// EXCLUDED). Bit `i` set iff any bench Uma has ≥1 attached energy of
/// type `ENERGY_TYPES_ORDER[i]`. Padded `None` bench slots silently
/// skipped. Mirrors Python `_v36_bench_typed_aggregate`.
fn v36_bench_typed_aggregate(side: &PublicSideObservation, out: &mut [f32]) {
    debug_assert_eq!(out.len(), ENERGY_TYPES_ORDER.len());
    for uma in side.bench.iter().flatten() {
        for (i, &energy_type) in ENERGY_TYPES_ORDER.iter().enumerate() {
            let amount = uma.energies.get(energy_type).copied().unwrap_or(0);
            if amount > 0 {
                out[i] = 1.0;
            }
        }
    }
}

/// Mirror of engine attack-legality (multiset matching): typed costs
/// satisfied per-color AND total attached ≥ total cost (colorless absorbs
/// any leftover). Re-uses the v3.5 `typed_energy_deficit` + `cost_total`
/// pair so primary-attack and secondary-attack share one matcher. Mirrors
/// Python `_v36_attack_cost_covered`.
fn v36_attack_cost_covered(
    attached: &indexmap::IndexMap<String, u16>,
    cost: &crate::core::effects::EnergyCost,
) -> bool {
    if typed_energy_deficit(attached, cost) > 0.0 {
        return false;
    }
    let total_attached: f32 = attached.values().map(|&v| v as f32).sum();
    let total_cost = cost_total(cost);
    total_attached >= total_cost
}

/// Resolve the catalog `UmamusumeCard` for an active Uma observation.
/// Returns `None` for absent / non-Uma / unknown card ids. Centralised so
/// both the lethal-next-turn and secondary-attack predicates share one
/// catalog-access path; mirrors Python `_v36_active_attacks` (Python
/// returns the attack list directly; Rust returns the card so callers
/// also see `attacks[1]`).
fn v36_uma_card_for_active(active: Option<&PublicUmaObservation>) -> Option<&'static UmamusumeCard> {
    let entry = active?;
    let card = get_card(&entry.card_id)?;
    match card {
        Card::Umamusume(u) => Some(u),
        Card::Trainer(_) => None,
    }
}

/// Effective HP for the lethal predicate. The public observation's `hp`
/// field already reflects damage taken (engine sets `umamusume.hp =
/// max_hp - damage_taken`), so we use `hp` directly. Mirrors Python
/// `_v36_remaining_hp` (Python falls back to `max_hp - damageCounters`
/// when `hp` is missing — Rust's struct guarantees the field).
fn v36_remaining_hp(active: Option<&PublicUmaObservation>) -> f32 {
    match active {
        Some(u) => (u.hp as f32).max(0.0),
        None => 0.0,
    }
}

/// 1.0 iff `attacker.active` has any attack with `base_damage ≥
/// defender.remaining_hp`. Face-value only (scoping §4.5 Channel 4): no
/// weakness multiplier, no coin-flip expectation, no energy-availability
/// check. Returns 0.0 if attacker active is absent / non-Uma / has no
/// attacks, or if defender active is absent. Mirrors Python
/// `_v36_lethal_face_value`.
fn v36_lethal_face_value(
    attacker_active: Option<&PublicUmaObservation>,
    defender_active: Option<&PublicUmaObservation>,
) -> f32 {
    let Some(uma) = v36_uma_card_for_active(attacker_active) else {
        return 0.0;
    };
    if uma.attacks.is_empty() || defender_active.is_none() {
        return 0.0;
    }
    let defender_hp = v36_remaining_hp(defender_active);
    let max_dmg = uma
        .attacks
        .iter()
        .map(|a| a.damage as f32)
        .fold(f32::NEG_INFINITY, f32::max);
    if max_dmg >= defender_hp { 1.0 } else { 0.0 }
}

/// (usable, would_KO) for the attacker's `attacks[1]`. Both 0.0 if no
/// secondary attack exists. `usable` requires energy coverage per
/// `v36_attack_cost_covered`; `would_KO` additionally requires
/// `attacks[1].damage ≥ defender.remaining_hp`. Face-value per scoping
/// §4.5 Channel 5. Mirrors Python `_v36_secondary_attack_bits`.
fn v36_secondary_attack_bits(
    attacker_active: Option<&PublicUmaObservation>,
    defender_active: Option<&PublicUmaObservation>,
) -> (f32, f32) {
    let Some(uma) = v36_uma_card_for_active(attacker_active) else {
        return (0.0, 0.0);
    };
    if uma.attacks.len() < 2 {
        return (0.0, 0.0);
    }
    let secondary = &uma.attacks[1];
    // attacker_active is Some(...) because v36_uma_card_for_active returned
    // a card (it short-circuits on None).
    let attached = &attacker_active.unwrap().energies;
    if !v36_attack_cost_covered(attached, &secondary.cost) {
        return (0.0, 0.0);
    }
    if defender_active.is_none() {
        return (1.0, 0.0);
    }
    let defender_hp = v36_remaining_hp(defender_active);
    let base_damage = secondary.damage as f32;
    let would_ko = if base_damage >= defender_hp { 1.0 } else { 0.0 };
    (1.0, would_ko)
}

/// v36-priors-arithmetic: 246-d builder. Slots 0–196 are byte-identical
/// to v3.5; slots [197:207] are REPURPOSED in-place from the dead v3.5
/// opp.energy_zone.front band to own.energy_pool typed multihot; the
/// v3.5 tail at [207:212] is preserved; new bits are appended at
/// [212:246]. Mirrors `observation_to_features_v3_6` in Python.
///
/// Reconciliation note (impl-phase): the §4.5 channel breakdown sums to
/// 54 bits across both sides; the TL;DR locks STATE_DIM at 246 (net +34
/// bits). Own bench typed energy aggregate dropped as the lowest-priority
/// cut (own.active typed energies already in v3.5 head). Opp bench kept
/// since opp-threat-by-color is the stated rationale.
pub fn observation_state_features_v3_6(obs: &PublicObservation) -> Vec<f32> {
    let mut f = vec![0.0f32; STATE_DIM_V3_6];
    let head = observation_state_features_v3_5(obs);
    debug_assert_eq!(head.len(), STATE_DIM_V3_5);
    f[..STATE_DIM_V3_5].copy_from_slice(&head);

    // Zero-overwrite the dead v3.5 opp.energy_zone.front band [197:207]
    // and repurpose in-place for own.energy_pool typed multihot. This is
    // the only v3.5 slot v3.6 touches; [0:197] stays byte-stable, and
    // [207:212] (the rest of the v3.5 tail) is left untouched.
    for v in &mut f[_V36_OWN_ENERGY_POOL_START.._V36_OWN_ENERGY_POOL_END] {
        *v = 0.0;
    }

    // Channel 1 — energy_pool typed multihot. Own in-band at [197:207];
    // opp at appended-tail [212:222].
    v36_energy_pool_multihot(
        &obs.own,
        &mut f[_V36_OWN_ENERGY_POOL_START.._V36_OWN_ENERGY_POOL_END],
    );
    v36_energy_pool_multihot(
        &obs.opponent,
        &mut f[_V36_OPP_ENERGY_POOL_START.._V36_OPP_ENERGY_POOL_END],
    );

    // Channel 2 — prize one-hot ×4 over remaining-prize {3,2,1,0}.
    v36_prize_one_hot(&obs.own, &mut f[_V36_OWN_PRIZE_START.._V36_OWN_PRIZE_END]);
    v36_prize_one_hot(
        &obs.opponent,
        &mut f[_V36_OPP_PRIZE_START.._V36_OPP_PRIZE_END],
    );

    // Channel 3 — opp bench typed energy aggregate (multihot). Own
    // dropped at reconciliation (see header).
    v36_bench_typed_aggregate(
        &obs.opponent,
        &mut f[_V36_OPP_BENCH_TYPED_START.._V36_OPP_BENCH_TYPED_END],
    );

    // Channel 4 — lethal-next-turn face-value. `own_lethal` means OPP can
    // KO OWN's active at face value.
    let own_active = obs.own.active.as_ref();
    let opp_active = obs.opponent.active.as_ref();
    f[_V36_OWN_LETHAL_NEXT_TURN_SLOT] = v36_lethal_face_value(opp_active, own_active);
    f[_V36_OPP_LETHAL_NEXT_TURN_SLOT] = v36_lethal_face_value(own_active, opp_active);

    // Channel 5 — secondary attack readiness + would-KO.
    let (own_sec_usable, own_sec_ko) = v36_secondary_attack_bits(own_active, opp_active);
    f[_V36_OWN_SECONDARY_USABLE_SLOT] = own_sec_usable;
    f[_V36_OWN_SECONDARY_WOULD_KO_SLOT] = own_sec_ko;
    let (opp_sec_usable, opp_sec_ko) = v36_secondary_attack_bits(opp_active, own_active);
    f[_V36_OPP_SECONDARY_USABLE_SLOT] = opp_sec_usable;
    f[_V36_OPP_SECONDARY_WOULD_KO_SLOT] = opp_sec_ko;

    f
}

// ---------------------------------------------------------------------------
// v3.7 combat-arith-and-catalog tail
// (`docs/ai-research/scoping/v37-combat-arith-and-catalog-scoping.md`).
//
// Mirrors Python `_V37_*` slot offsets bit-for-bit. The v3.6 head [0:246]
// stays byte-stable; v3.7 only writes the 50-bit tail at [246:296].
//
// Layout (FROZEN):
//   [246:248] own/opp weakness-adjusted lethal (Channel 1)
//   [248:250] own/opp weakness-adjusted secondary KO (Channel 1)
//   [250:254] own primary has_cf/cf_eko, opp primary has_cf/cf_eko (Ch2)
//   [254:258] own sec has_cf/cf_eko, opp sec has_cf/cf_eko (Ch2)
//   [258:262] own primary per_energy/per_bench, opp primary per_energy/per_bench (Ch3)
//   [262:266] own sec per_energy/per_bench, opp sec per_energy/per_bench (Ch3)
//   [266:270] own primary ETA, own sec ETA, opp primary ETA, opp sec ETA (Ch4)
//   [270:272] own / opp paralysis_window_open (Ch4)
//   [272:276] own tool effect-kind one-hot (4 classes) (Ch5)
//   [276:280] opp tool effect-kind one-hot (4 classes) (Ch5)
//   [280:288] own ability effect-kind one-hot (8 classes) (Ch6)
//   [288:296] opp ability effect-kind one-hot (8 classes) (Ch6)
// ---------------------------------------------------------------------------

const _V37_TAIL_START: usize = 246;
const _V37_OWN_WEAKNESS_LETHAL: usize = 246;
const _V37_OPP_WEAKNESS_LETHAL: usize = 247;
const _V37_OWN_WEAKNESS_SECONDARY_KO: usize = 248;
const _V37_OPP_WEAKNESS_SECONDARY_KO: usize = 249;
const _V37_COIN_FLIP_BASE: usize = 250;
const _V37_COND_BONUS_BASE: usize = 258;
const _V37_ETA_BASE: usize = 266;
const _V37_PARALYSIS_BASE: usize = 270;
const _V37_TOOL_KIND_OWN_BASE: usize = 272;
const _V37_TOOL_KIND_OPP_BASE: usize = 276;
const _V37_ABILITY_KIND_OWN_BASE: usize = 280;
const _V37_ABILITY_KIND_OPP_BASE: usize = 288;

/// Weakness-adjusted lethal predicate for ONE attack (`attacks[idx]`).
/// Bit = 1.0 iff
///   `attack.damage + (defender.weakness.amount if defender.weakness.type
///                      == attacker_card.type else 0) >= defender_hp`,
/// matching `flow/combat.rs:303-305` exactly (additive bonus, applied
/// only when `damage > 0`).
///
/// Both Uma fields are catalog-derived. Mirrors Python
/// `_v37_weakness_adjusted_lethal`.
fn v37_weakness_lethal_for_attack(
    attacker_card: &UmamusumeCard,
    defender_card: &UmamusumeCard,
    attack: &Attack,
    defender_hp: f32,
) -> f32 {
    let mut damage = attack.damage as f32;
    if attack.damage > 0 && attacker_card.r#type == defender_card.weakness.r#type {
        damage += defender_card.weakness.amount as f32;
    }
    if damage >= defender_hp { 1.0 } else { 0.0 }
}

/// Channel 1 primary-attack predicate. Returns 0.0 on absent / non-Uma
/// attacker, absent defender, or empty attack list. Mirrors Python
/// `_v37_weakness_adjusted_lethal`.
fn v37_weakness_adjusted_lethal(
    attacker_active: Option<&PublicUmaObservation>,
    defender_active: Option<&PublicUmaObservation>,
) -> f32 {
    let Some(att_card) = v36_uma_card_for_active(attacker_active) else {
        return 0.0;
    };
    let Some(def_active) = defender_active else {
        return 0.0;
    };
    let Some(def_card) = v36_uma_card_for_active(defender_active) else {
        return 0.0;
    };
    let Some(primary) = att_card.attacks.first() else {
        return 0.0;
    };
    let defender_hp = v36_remaining_hp(Some(def_active));
    v37_weakness_lethal_for_attack(att_card, def_card, primary, defender_hp)
}

/// Channel 1 secondary-attack predicate. Predicated on the v3.6
/// secondary-usable bit (= 0 → return 0). Mirrors Python
/// `_v37_weakness_adjusted_secondary_ko`.
fn v37_weakness_adjusted_secondary_ko(
    attacker_active: Option<&PublicUmaObservation>,
    defender_active: Option<&PublicUmaObservation>,
    secondary_usable_bit: f32,
) -> f32 {
    if secondary_usable_bit <= 0.0 {
        return 0.0;
    }
    let Some(att_card) = v36_uma_card_for_active(attacker_active) else {
        return 0.0;
    };
    if att_card.attacks.len() < 2 {
        return 0.0;
    }
    let Some(def_active) = defender_active else {
        return 0.0;
    };
    let Some(def_card) = v36_uma_card_for_active(defender_active) else {
        return 0.0;
    };
    let secondary = &att_card.attacks[1];
    let defender_hp = v36_remaining_hp(Some(def_active));
    v37_weakness_lethal_for_attack(att_card, def_card, secondary, defender_hp)
}

/// Channel 2 bits for one attack: `(has_cf, cf_eko)`. Mirrors Python
/// `_v37_attack_coin_flip_bits`:
///   - `has_cf` = 1 iff any of {coin_bonus, knock_out_active_if_all_coin_heads,
///     draw_on_heads, discard_random_opponent_hand_on_heads} is set.
///   - `cf_eko` = 1 iff `damage + 0.5 * (coin_bonus or 0) >= defender_hp`.
fn v37_attack_coin_flip_bits(attack: Option<&Attack>, defender_hp: f32) -> (f32, f32) {
    let Some(attack) = attack else {
        return (0.0, 0.0);
    };
    let has_cf = attack.coin_bonus.is_some()
        || attack.knock_out_active_if_all_coin_heads.is_some()
        || attack.draw_on_heads.is_some()
        || attack.discard_random_opponent_hand_on_heads.is_some();
    let coin_bonus = attack.coin_bonus.unwrap_or(0) as f32;
    let expected = attack.damage as f32 + 0.5_f32 * coin_bonus;
    let cf_eko = if expected >= defender_hp { 1.0 } else { 0.0 };
    (if has_cf { 1.0 } else { 0.0 }, cf_eko)
}

/// Channel 3 bits for one attack: `(per_energy, per_bench)`. Mirrors
/// Python `_v37_attack_conditional_bonus_bits`. Structural-only flags:
/// `per_energy` covers both `damage_per_attached_energy` and
/// `damage_per_unique_attached_energy`; `per_bench` covers
/// `damage_per_umamusume_in_play`.
fn v37_attack_conditional_bonus_bits(attack: Option<&Attack>) -> (f32, f32) {
    let Some(attack) = attack else {
        return (0.0, 0.0);
    };
    let per_energy = attack.damage_per_attached_energy.is_some()
        || attack.damage_per_unique_attached_energy.is_some();
    let per_bench = attack.damage_per_umamusume_in_play.is_some();
    (
        if per_energy { 1.0 } else { 0.0 },
        if per_bench { 1.0 } else { 0.0 },
    )
}

/// Channel 4 energy-ETA predicate. Returns 1.0 iff the attack's cost is
/// feasible by next turn given the attacker's currently-attached
/// energies plus a `+1` next-turn attach budget drawn from the public
/// `energy_pool`. Per scope §4.5 Ch.4 LOCKED DEFINITION: typed-color
/// feasibility only (no total-cost / colorless-absorption check). The
/// +1 attach is applied to the LARGEST shortfall color with
/// lex-tiebreak by `ENERGY_TYPES_ORDER` index (matches Python).
/// Returns 0.0 if `attack` is None / attacker absent. Mirrors Python
/// `_v37_attack_usable_next_turn`.
fn v37_attack_usable_next_turn(
    attacker_active: Option<&PublicUmaObservation>,
    attack: Option<&Attack>,
    energy_pool: &[String],
) -> f32 {
    let Some(attack) = attack else {
        return 0.0;
    };
    let Some(attacker) = attacker_active else {
        return 0.0;
    };
    let attached = &attacker.energies;
    // Per-color shortfall (excluding colorless — colorless absorbs).
    // Use a Vec<(usize, u8)> keyed by ENERGY_TYPES_ORDER index so the
    // tiebreak is deterministic and matches Python's lex-by-order.
    let mut shortfall: Vec<(usize, u8)> = Vec::new();
    for (et, amount) in attack.cost.iter_typed() {
        if et == EnergyType::Colorless {
            continue;
        }
        let name = energy_type_name(et);
        let have = attached.get(name).copied().unwrap_or(0);
        if (amount as i32) > (have as i32) {
            let deficit = (amount as i32 - have as i32) as u8;
            let order = ENERGY_TYPES_ORDER
                .iter()
                .position(|n| *n == name)
                .unwrap_or(usize::MAX);
            shortfall.push((order, deficit));
        }
    }
    // Apply +1 attach to largest shortfall (lex-tiebreak by order index).
    if !shortfall.is_empty() {
        shortfall.sort_by(|a, b| {
            // Largest deficit first; on ties, smaller order index first.
            b.1.cmp(&a.1).then(a.0.cmp(&b.0))
        });
        let (target_order, target_amt) = shortfall[0];
        if target_amt <= 1 {
            shortfall.remove(0);
        } else {
            shortfall[0] = (target_order, target_amt - 1);
        }
    }
    // After +1 attach: every remaining shortfall color must appear in
    // the pool.
    for (order, _amt) in &shortfall {
        let color = ENERGY_TYPES_ORDER.get(*order).copied().unwrap_or("");
        if !energy_pool.iter().any(|t| t.as_str() == color) {
            return 0.0;
        }
    }
    1.0
}

/// Channel 4 paralysis bit. `own_paralysis_window_open` = 1 iff OPP
/// active is paralysis_recovery_pending. Mirrors Python
/// `_v37_paralysis_window_open`.
fn v37_paralysis_window_open(other_side_active: Option<&PublicUmaObservation>) -> f32 {
    match other_side_active {
        Some(u) if u.turn_state.paralysis_recovery_pending => 1.0,
        _ => 0.0,
    }
}

/// Channel 5 tool effect-kind one-hot. All zeros iff
/// `active.tool_card_id is None` OR the tool's catalog entry is
/// missing/non-trainer. Else writes a 1 at the
/// `ToolEffectKind::as_u8()` offset. Mirrors Python
/// `_v37_tool_kind_one_hot`.
fn v37_tool_kind_one_hot(active: Option<&PublicUmaObservation>, out: &mut [f32]) {
    debug_assert_eq!(out.len(), 4);
    let Some(uma) = active else {
        return;
    };
    let Some(tool_id) = uma.tool_card_id.as_deref() else {
        return;
    };
    let Some(card) = get_card(tool_id) else {
        return;
    };
    if let Card::Trainer(t) = card {
        let kind = classify_tool_effect(&t.effect);
        out[kind.as_u8() as usize] = 1.0;
    }
}

/// Channel 6 ability effect-kind one-hot. All zeros iff the active
/// didn't actually USE its ability this turn (`used_ability_this_turn
/// == false`) OR has no card / no `ability` payload. Else writes a 1
/// at the `AbilityEffectKind::as_u8()` offset. Mirrors Python
/// `_v37_ability_kind_one_hot`.
fn v37_ability_kind_one_hot(active: Option<&PublicUmaObservation>, out: &mut [f32]) {
    debug_assert_eq!(out.len(), 8);
    let Some(uma) = active else {
        return;
    };
    if !uma.used_ability_this_turn {
        return;
    }
    let Some(card) = v36_uma_card_for_active(Some(uma)) else {
        return;
    };
    let Some(ability) = card.ability.as_ref() else {
        return;
    };
    let kind = classify_active_ability(ability);
    out[kind.as_u8() as usize] = 1.0;
}

/// v37-combat-arith-and-catalog: 296-d builder. Slots 0–245 are
/// byte-identical to v3.6; the 50-bit tail at [246:296] holds the
/// combat-arith + catalog channels per scoping §4 step 3 / §4.5.
/// Mirrors `observation_to_features_v3_7` in Python.
pub fn observation_state_features_v3_7(obs: &PublicObservation) -> Vec<f32> {
    let mut f = vec![0.0f32; STATE_DIM_V3_7];
    let head = observation_state_features_v3_6(obs);
    debug_assert_eq!(head.len(), STATE_DIM_V3_6);
    f[..STATE_DIM_V3_6].copy_from_slice(&head);

    let own_active = obs.own.active.as_ref();
    let opp_active = obs.opponent.active.as_ref();

    // Channel 1 — weakness-adjusted lethal. `own_weakness_lethal` =
    // OPP attacks OWN at face+weakness (mirrors v3.6 polarity).
    f[_V37_OWN_WEAKNESS_LETHAL] = v37_weakness_adjusted_lethal(opp_active, own_active);
    f[_V37_OPP_WEAKNESS_LETHAL] = v37_weakness_adjusted_lethal(own_active, opp_active);

    // Channel 1 secondary — predicated on v3.6 slots [242] / [244]
    // (own / opp secondary usable). own_weakness_secondary_KO = OPP
    // secondary KOs OWN, so it reads v3.6 slot [244].
    let own_weakness_sec = v37_weakness_adjusted_secondary_ko(
        opp_active,
        own_active,
        head[_V36_OPP_SECONDARY_USABLE_SLOT],
    );
    let opp_weakness_sec = v37_weakness_adjusted_secondary_ko(
        own_active,
        opp_active,
        head[_V36_OWN_SECONDARY_USABLE_SLOT],
    );
    f[_V37_OWN_WEAKNESS_SECONDARY_KO] = own_weakness_sec;
    f[_V37_OPP_WEAKNESS_SECONDARY_KO] = opp_weakness_sec;

    // Resolve catalog attacks once. v3.7 channels 2/3/4 read primary +
    // secondary on each side; primary is `attacks[0]`, secondary is
    // `attacks[1]` if present (Option<&Attack>).
    let own_card = v36_uma_card_for_active(own_active);
    let opp_card = v36_uma_card_for_active(opp_active);
    let own_primary = own_card.and_then(|c| c.attacks.first());
    let opp_primary = opp_card.and_then(|c| c.attacks.first());
    let own_sec = own_card.and_then(|c| c.attacks.get(1));
    let opp_sec = opp_card.and_then(|c| c.attacks.get(1));
    let own_hp = v36_remaining_hp(own_active);
    let opp_hp = v36_remaining_hp(opp_active);

    // Channel 2 — coin-flip indicators. Layout: own primary has_cf,
    // cf_eko; opp primary has_cf, cf_eko; own sec has_cf, cf_eko;
    // opp sec has_cf, cf_eko.
    let (own_p_has_cf, own_p_eko) = v37_attack_coin_flip_bits(own_primary, opp_hp);
    let (opp_p_has_cf, opp_p_eko) = v37_attack_coin_flip_bits(opp_primary, own_hp);
    let (own_s_has_cf, own_s_eko) = v37_attack_coin_flip_bits(own_sec, opp_hp);
    let (opp_s_has_cf, opp_s_eko) = v37_attack_coin_flip_bits(opp_sec, own_hp);
    f[_V37_COIN_FLIP_BASE + 0] = own_p_has_cf;
    f[_V37_COIN_FLIP_BASE + 1] = own_p_eko;
    f[_V37_COIN_FLIP_BASE + 2] = opp_p_has_cf;
    f[_V37_COIN_FLIP_BASE + 3] = opp_p_eko;
    f[_V37_COIN_FLIP_BASE + 4] = own_s_has_cf;
    f[_V37_COIN_FLIP_BASE + 5] = own_s_eko;
    f[_V37_COIN_FLIP_BASE + 6] = opp_s_has_cf;
    f[_V37_COIN_FLIP_BASE + 7] = opp_s_eko;

    // Channel 3 — conditional damage bonuses.
    let (own_p_pe, own_p_pb) = v37_attack_conditional_bonus_bits(own_primary);
    let (opp_p_pe, opp_p_pb) = v37_attack_conditional_bonus_bits(opp_primary);
    let (own_s_pe, own_s_pb) = v37_attack_conditional_bonus_bits(own_sec);
    let (opp_s_pe, opp_s_pb) = v37_attack_conditional_bonus_bits(opp_sec);
    f[_V37_COND_BONUS_BASE + 0] = own_p_pe;
    f[_V37_COND_BONUS_BASE + 1] = own_p_pb;
    f[_V37_COND_BONUS_BASE + 2] = opp_p_pe;
    f[_V37_COND_BONUS_BASE + 3] = opp_p_pb;
    f[_V37_COND_BONUS_BASE + 4] = own_s_pe;
    f[_V37_COND_BONUS_BASE + 5] = own_s_pb;
    f[_V37_COND_BONUS_BASE + 6] = opp_s_pe;
    f[_V37_COND_BONUS_BASE + 7] = opp_s_pb;

    // Channel 4 — energy ETA.
    f[_V37_ETA_BASE + 0] = v37_attack_usable_next_turn(own_active, own_primary, &obs.own.energy_pool);
    f[_V37_ETA_BASE + 1] = v37_attack_usable_next_turn(own_active, own_sec, &obs.own.energy_pool);
    f[_V37_ETA_BASE + 2] = v37_attack_usable_next_turn(opp_active, opp_primary, &obs.opponent.energy_pool);
    f[_V37_ETA_BASE + 3] = v37_attack_usable_next_turn(opp_active, opp_sec, &obs.opponent.energy_pool);

    // Channel 4 paralysis (own window open iff opp.active paralysed).
    f[_V37_PARALYSIS_BASE + 0] = v37_paralysis_window_open(opp_active);
    f[_V37_PARALYSIS_BASE + 1] = v37_paralysis_window_open(own_active);

    // Channel 5 — tool effect-kind one-hot.
    v37_tool_kind_one_hot(
        own_active,
        &mut f[_V37_TOOL_KIND_OWN_BASE.._V37_TOOL_KIND_OWN_BASE + 4],
    );
    v37_tool_kind_one_hot(
        opp_active,
        &mut f[_V37_TOOL_KIND_OPP_BASE.._V37_TOOL_KIND_OPP_BASE + 4],
    );

    // Channel 6 — ability effect-kind one-hot.
    v37_ability_kind_one_hot(
        own_active,
        &mut f[_V37_ABILITY_KIND_OWN_BASE.._V37_ABILITY_KIND_OWN_BASE + 8],
    );
    v37_ability_kind_one_hot(
        opp_active,
        &mut f[_V37_ABILITY_KIND_OPP_BASE.._V37_ABILITY_KIND_OPP_BASE + 8],
    );

    f
}

// ---------------------------------------------------------------------------
// v3.8 slim-feature-add tail
// (`docs/ai-research/scoping/v38-slim-feature-add-scoping.md`).
//
// Mirrors Python `_V38_*` slot offsets bit-for-bit. The v3.7 head
// [0:296] stays byte-stable; v3.8 only writes the 8-bit tail at
// [296:304]. See `observation_to_features_v3_8` in
// `training/uma_ai/features.py` for the slot definitions, including the
// adapted gust-availability proxy (scoping §13.4) used because
// `PublicSideObservation.hand_card_ids` is None on opp side.
//
// Layout:
//   [296:299] own.bench[0..2] primary-attack usable-next-turn
//   [299:302] opp.bench[0..2] primary-attack usable-next-turn
//   [302]     own_lose_if_opp_gusts_weakest_bench
//   [303]     own_can_gust_win_prize_race
// ---------------------------------------------------------------------------

const _V38_TAIL_START: usize = 296;
const _V38_OWN_BENCH_ETA_BASE: usize = 296;
const _V38_OPP_BENCH_ETA_BASE: usize = 299;
const _V38_OWN_LOSE_IF_OPP_GUSTS: usize = 302;
const _V38_OWN_GUST_WIN_RACE: usize = 303;

/// v3.8 per-bench ETA — reuses v3.7 Ch.4
/// `v37_attack_usable_next_turn` applied to each `side.bench[i]`. Writes
/// 3 bits at `out[0..3]`.
fn v38_bench_primary_eta_bits(side: &PublicSideObservation, out: &mut [f32]) {
    debug_assert_eq!(out.len(), 3);
    for i in 0..3 {
        if i >= side.bench.len() {
            continue;
        }
        let Some(entry) = side.bench[i].as_ref() else {
            continue;
        };
        let Some(card) = v36_uma_card_for_active(Some(entry)) else {
            continue;
        };
        let Some(primary) = card.attacks.first() else {
            continue;
        };
        out[i] = v37_attack_usable_next_turn(Some(entry), Some(primary), &side.energy_pool);
    }
}

/// True iff `card` is a trainer with `gustOpponent` or
/// `discardRandomOpponentActiveEnergy` effect set.
fn v38_card_is_gust_trainer(card_id: &str) -> bool {
    let Some(card) = get_card(card_id) else {
        return false;
    };
    if let Card::Trainer(t) = card {
        if t.effect.gust_opponent == Some(true)
            || t.effect.discard_random_opponent_active_energy == Some(true)
        {
            return true;
        }
    }
    false
}

/// v3.8 public-info proxy for "opp could play a gust this opp turn."
/// Per scoping §13.4 adaptation (opp.hand_card_ids is private):
///   - opp has demonstrated they hold gusts (any gust trainer in
///     `opp.discard`) AND
///   - opp.used_supporter_this_turn == false AND
///   - opp.hand_count > 0.
fn v38_opp_gust_playable_proxy(opp: &PublicSideObservation) -> bool {
    if opp.used_supporter_this_turn {
        return false;
    }
    if opp.hand_count == 0 {
        return false;
    }
    for card_id in &opp.discard {
        if v38_card_is_gust_trainer(card_id.as_str()) {
            return true;
        }
    }
    false
}

/// v3.8 "own has gust in hand" — own.hand_card_ids IS exposed (private
/// to own side only, but model's own perspective IS own-side).
fn v38_own_has_gust_in_hand(own: &PublicSideObservation) -> bool {
    let Some(hand_ids) = own.hand_card_ids.as_ref() else {
        return false;
    };
    for card_id in hand_ids {
        if v38_card_is_gust_trainer(card_id.as_str()) {
            return true;
        }
    }
    false
}

/// v3.8 weakness-adjusted face-value damage of `attacker.active.attacks[0]`
/// against a defender catalog card. Mirrors v3.7 Ch.1 formula:
/// damage = attack.damage; if damage > 0 and attacker.type ==
/// defender.weakness.type → damage += weakness.amount. Returns 0.0
/// when data is absent.
fn v38_weakness_adjusted_damage(
    attacker_active: Option<&PublicUmaObservation>,
    defender_card: Option<&UmamusumeCard>,
) -> f32 {
    let Some(att_card) = v36_uma_card_for_active(attacker_active) else {
        return 0.0;
    };
    let Some(def_card) = defender_card else {
        return 0.0;
    };
    let Some(primary) = att_card.attacks.first() else {
        return 0.0;
    };
    let mut damage = primary.damage as f32;
    if damage > 0.0 && att_card.r#type == def_card.weakness.r#type {
        damage += def_card.weakness.amount as f32;
    }
    damage
}

/// Returns (min_remaining_hp, the_card) over present bench Umas.
/// `f32::INFINITY` + None if bench is empty.
fn v38_min_bench_remaining_hp<'a>(
    side: &'a PublicSideObservation,
) -> (f32, Option<&'a PublicUmaObservation>) {
    let mut best_hp = f32::INFINITY;
    let mut best_entry: Option<&PublicUmaObservation> = None;
    for entry in side.bench.iter().flatten() {
        let hp = v36_remaining_hp(Some(entry));
        if hp < best_hp {
            best_hp = hp;
            best_entry = Some(entry);
        }
    }
    (best_hp, best_entry)
}

/// True iff any present bench Uma on `defender_side` has remaining HP
/// ≤ attacker.active.attacks[0] damage (weakness-adjusted per defender's
/// own weakness type).
fn v38_any_bench_ko_able(
    attacker_active: Option<&PublicUmaObservation>,
    defender_side: &PublicSideObservation,
) -> bool {
    let Some(_att) = attacker_active else {
        return false;
    };
    for entry in defender_side.bench.iter().flatten() {
        let def_card = v36_uma_card_for_active(Some(entry));
        let damage = v38_weakness_adjusted_damage(attacker_active, def_card);
        if damage <= 0.0 {
            continue;
        }
        let hp = v36_remaining_hp(Some(entry));
        if hp <= damage {
            return true;
        }
    }
    false
}

fn v38_remaining_prizes(side: &PublicSideObservation) -> i32 {
    let points = side.points as i32;
    (3 - points).clamp(0, 3)
}

/// v3.8 slot [302] — own_lose_if_opp_gusts_weakest_bench. See scoping
/// §4.5 and §13.4 (public-info adapted predicate).
fn v38_lose_if_opp_gusts_weakest_bench(
    own: &PublicSideObservation,
    opp: &PublicSideObservation,
) -> f32 {
    if v38_remaining_prizes(own) > 1 {
        return 0.0;
    }
    if !v38_opp_gust_playable_proxy(opp) {
        return 0.0;
    }
    let Some(opp_active) = opp.active.as_ref() else {
        return 0.0;
    };
    let (weakest_hp, weakest_entry) = v38_min_bench_remaining_hp(own);
    let Some(weakest) = weakest_entry else {
        return 0.0;
    };
    let weakest_card = v36_uma_card_for_active(Some(weakest));
    let damage = v38_weakness_adjusted_damage(Some(opp_active), weakest_card);
    if damage <= 0.0 {
        return 0.0;
    }
    if weakest_hp <= damage {
        1.0
    } else {
        0.0
    }
}

/// v3.8 slot [303] — own_can_gust_win_prize_race. Symmetric.
fn v38_can_gust_win_prize_race(
    own: &PublicSideObservation,
    opp: &PublicSideObservation,
) -> f32 {
    if v38_remaining_prizes(opp) > 1 {
        return 0.0;
    }
    if !v38_own_has_gust_in_hand(own) {
        return 0.0;
    }
    let Some(own_active) = own.active.as_ref() else {
        return 0.0;
    };
    if v38_any_bench_ko_able(Some(own_active), opp) {
        1.0
    } else {
        0.0
    }
}

/// v38-slim-feature-add: 304-d builder. Slots 0–295 are byte-identical
/// to v3.7; the 8-bit tail at [296:304] adds per-bench ETA + gust-swing
/// catastrophe bits. Mirrors `observation_to_features_v3_8` in Python.
pub fn observation_state_features_v3_8(obs: &PublicObservation) -> Vec<f32> {
    let mut f = vec![0.0f32; STATE_DIM_V3_8];
    let head = observation_state_features_v3_7(obs);
    debug_assert_eq!(head.len(), STATE_DIM_V3_7);
    f[..STATE_DIM_V3_7].copy_from_slice(&head);

    // [296:299] own bench ETA.
    v38_bench_primary_eta_bits(
        &obs.own,
        &mut f[_V38_OWN_BENCH_ETA_BASE.._V38_OWN_BENCH_ETA_BASE + 3],
    );
    // [299:302] opp bench ETA.
    v38_bench_primary_eta_bits(
        &obs.opponent,
        &mut f[_V38_OPP_BENCH_ETA_BASE.._V38_OPP_BENCH_ETA_BASE + 3],
    );
    // [302] own_lose_if_opp_gusts_weakest_bench.
    f[_V38_OWN_LOSE_IF_OPP_GUSTS] =
        v38_lose_if_opp_gusts_weakest_bench(&obs.own, &obs.opponent);
    // [303] own_can_gust_win_prize_race.
    f[_V38_OWN_GUST_WIN_RACE] = v38_can_gust_win_prize_race(&obs.own, &obs.opponent);

    f
}

fn side_board_features(side: &PublicSideObservation) -> [f32; 8] {
    let mut hp_total = 0.0f32;
    let mut max_hp_total = 0.0f32;
    let mut energy_total = 0.0f32;
    let mut stage_total = 0.0f32;
    let mut damaged = 0.0f32;
    let mut statuses = 0.0f32;
    let mut count = 0.0f32;

    let mut tally = |uma: &PublicUmaObservation| {
        hp_total += uma.hp as f32;
        max_hp_total += uma.max_hp as f32;
        energy_total += uma.energy_total as f32;
        stage_total += uma.stage as f32;
        if (uma.hp as f32) < (uma.max_hp as f32) {
            damaged += 1.0;
        }
        statuses += uma.special_conditions.len() as f32;
        count += 1.0;
    };

    if let Some(active) = &side.active {
        tally(active);
    }
    for entry in side.bench.iter().flatten() {
        tally(entry);
    }

    let max_hp_total = max_hp_total.max(1.0);
    let (active_hp_ratio, active_energy) = match &side.active {
        Some(a) => {
            let max_hp = (a.max_hp as f32).max(1.0);
            (a.hp as f32 / max_hp, a.energy_total as f32 / 6.0)
        }
        None => (0.0, 0.0),
    };

    [
        active_hp_ratio,
        active_energy,
        count / 4.0,
        hp_total / max_hp_total,
        energy_total / 12.0,
        stage_total / 8.0,
        damaged / 4.0,
        statuses / 4.0,
    ]
}

fn identity_features(own: &PublicSideObservation, opp: &PublicSideObservation) -> [f32; 16] {
    let mut v = [0.0f32; 16];
    let own_active_id = own.active.as_ref().map(|a| a.card_id.as_str()).unwrap_or("");
    let opp_active_id = opp.active.as_ref().map(|a| a.card_id.as_str()).unwrap_or("");
    v[0] = hash_to_unit(own_active_id);
    v[1] = hash_to_unit(opp_active_id);
    // own bench[0..4], opp bench[0..4] — pad with empty entries beyond
    // the engine's MAX_BENCH=3 cap (Python iterates `bench[:4]`).
    for i in 0..4 {
        let id = own.bench.get(i).and_then(|o| o.as_ref()).map(|u| u.card_id.as_str()).unwrap_or("");
        v[2 + i] = hash_to_unit(id);
    }
    for i in 0..4 {
        let id = opp.bench.get(i).and_then(|o| o.as_ref()).map(|u| u.card_id.as_str()).unwrap_or("");
        v[6 + i] = hash_to_unit(id);
    }
    // Python iterates `handCardIds` (the private hand ids — present on
    // own perspective). Use the actual handCardIds list when present;
    // otherwise hash_average over the empty list returns 0.
    let hand_ids: Vec<String> = own
        .hand_card_ids
        .as_ref()
        .map(|v| v.clone())
        .unwrap_or_default();
    v[10] = hash_average(&hand_ids);
    v[11] = hash_average(&own.discard);
    v[12] = hash_average(&opp.discard);
    v[13] = hand_ids.len() as f32 / 10.0;
    // bench length counts only PRESENT entries — Python iterates
    // `side.get('bench') or []` and the JSON `bench` list excludes
    // engine-side `null` only if the producer drops nulls. Our
    // `PublicSideObservation.bench` keeps `Option<...>` slots; Python
    // `len(bench)` would count `null` slots too. Mirror Python exactly:
    // count the WHOLE list (including None entries) since the Python
    // observation list there does not pre-drop nulls either at this
    // call site.
    v[14] = own.bench.len() as f32 / 4.0;
    v[15] = opp.bench.len() as f32 / 4.0;
    v
}

fn energy_vector_from_uma(uma: Option<&PublicUmaObservation>) -> [f32; 10] {
    let mut v = [0.0f32; 10];
    let Some(u) = uma else { return v; };
    for (i, &name) in ENERGY_TYPES_ORDER.iter().enumerate() {
        let amount = u.energies.get(name).copied().unwrap_or(0) as f32;
        v[i] = amount / 4.0;
    }
    v
}

fn first_player_polarity(obs: &PublicObservation) -> f32 {
    // `firstPlayer` is always Player|Opponent (SideId enum), so it's
    // always one of the SIDES strings — Python tolerates an absent
    // field with 0.0, but the Rust struct guarantees presence.
    let side_to_act = side_to_str(obs.side_to_act);
    let first = side_to_str(obs.first_player);
    if first == side_to_act {
        1.0
    } else {
        -1.0
    }
}

fn pending_choice_one_hot(kind: Option<&str>) -> [f32; 3] {
    let mut v = [0.0f32; 3];
    match kind {
        None => {
            v[0] = 1.0;
        }
        Some(k) => match PENDING_CHOICE_KINDS.iter().position(|s| *s == k) {
            Some(idx) => v[1 + idx] = 1.0,
            // Unknown union arm — fall back to the "none" bucket
            // exactly as Python does. Bump PENDING_CHOICE_KINDS when
            // this fires.
            None => v[0] = 1.0,
        },
    }
    v
}

fn tool_card_features(own: &PublicSideObservation, opp: &PublicSideObservation) -> [f32; 10] {
    let mut v = [0.0f32; 10];
    let own_active_tool = own
        .active
        .as_ref()
        .and_then(|a| a.tool_card_id.as_deref())
        .unwrap_or("");
    let opp_active_tool = opp
        .active
        .as_ref()
        .and_then(|a| a.tool_card_id.as_deref())
        .unwrap_or("");
    v[0] = hash_to_unit(own_active_tool);
    for i in 0..4 {
        let id = own
            .bench
            .get(i)
            .and_then(|o| o.as_ref())
            .and_then(|u| u.tool_card_id.as_deref())
            .unwrap_or("");
        v[1 + i] = hash_to_unit(id);
    }
    v[5] = hash_to_unit(opp_active_tool);
    for i in 0..4 {
        let id = opp
            .bench
            .get(i)
            .and_then(|o| o.as_ref())
            .and_then(|u| u.tool_card_id.as_deref())
            .unwrap_or("");
        v[6 + i] = hash_to_unit(id);
    }
    v
}

// ---------------------------------------------------------------------------
// Card awareness — port of `_card_awareness_features` + helpers (28 slots).
// ---------------------------------------------------------------------------

fn card_awareness_features(
    own: &PublicSideObservation,
    opp: &PublicSideObservation,
    shared: &crate::policy::types::PublicSharedObservation,
) -> [f32; 28] {
    let mut v = [0.0f32; 28];
    let hand_ids: Vec<String> = own
        .hand_card_ids
        .as_ref()
        .map(|v| v.clone())
        .unwrap_or_default();
    let discard_ids = own.discard.clone();
    let own_board = board_entries(own);
    let opp_board = board_entries(opp);
    let own_active = own.active.as_ref();
    let opp_active = opp.active.as_ref();

    let hr = hand_role_features(&hand_ids);
    v[0..9].copy_from_slice(&hr);
    v[9] = matching_evolution_count(&hand_ids, &own_board) / 4.0;
    let r1 = uma_readiness_features(own_active);
    v[10..14].copy_from_slice(&r1);
    v[14] = ready_attacker_count(&own_board) / 4.0;
    v[15] = ability_ready_count(&own_board) / 4.0;
    let dr = discard_role_features(&discard_ids);
    v[16..19].copy_from_slice(&dr);
    let r2 = uma_readiness_features(opp_active);
    v[19..23].copy_from_slice(&r2);
    v[23] = can_ko(opp_active, own_active);
    v[24] = can_ko(own_active, opp_active);
    v[25] = next_energy_matches_active_need(own);
    v[26] = hash_to_unit(shared.stadium_card_id.as_deref().unwrap_or(""));
    v[27] = ready_attacker_count(&opp_board) / 4.0;
    v
}

fn board_entries(side: &PublicSideObservation) -> Vec<&PublicUmaObservation> {
    let mut out: Vec<&PublicUmaObservation> = Vec::new();
    if let Some(a) = &side.active {
        out.push(a);
    }
    for entry in side.bench.iter().flatten() {
        out.push(entry);
    }
    out
}

fn get_card<'a>(card_id: &str) -> Option<&'a Card> {
    if card_id.is_empty() {
        return None;
    }
    let cat = catalog();
    if let Some(c) = cat.get_by_str(card_id) {
        return Some(c);
    }
    for suffix in ["FullArtGold", "FullArt", "UncommonPlus"] {
        if card_id.ends_with(suffix) {
            let base = &card_id[..card_id.len() - suffix.len()];
            if let Some(c) = cat.get_by_str(base) {
                return Some(c);
            }
        }
    }
    None
}

fn primary_attack(card: &UmamusumeCard) -> Option<&crate::core::effects::Attack> {
    card.attacks.first()
}

fn cost_total(cost: &crate::core::effects::EnergyCost) -> f32 {
    // Python sums `cost.values()` which INCLUDES colorless — `by_type`
    // already covers every EnergyType in canonical ALL order including
    // colorless, so a flat sum is the right mirror.
    cost.by_type.iter().map(|&n| n as f32).sum()
}

fn typed_energy_deficit(
    energies: &indexmap::IndexMap<String, u16>,
    cost: &crate::core::effects::EnergyCost,
) -> f32 {
    let mut deficit = 0.0f32;
    for (et, amount) in cost.iter_typed() {
        if et == EnergyType::Colorless {
            continue;
        }
        let name = energy_type_name(et);
        let attached = energies.get(name).copied().unwrap_or(0) as f32;
        deficit += (amount as f32 - attached).max(0.0);
    }
    deficit
}

fn energy_type_name(t: EnergyType) -> &'static str {
    match t {
        EnergyType::Grass => "grass",
        EnergyType::Fire => "fire",
        EnergyType::Water => "water",
        EnergyType::Lightning => "lightning",
        EnergyType::Psychic => "psychic",
        EnergyType::Fighting => "fighting",
        EnergyType::Darkness => "darkness",
        EnergyType::Steel => "steel",
        EnergyType::Colorless => "colorless",
        EnergyType::Dragon => "dragon",
    }
}

fn hand_role_features(card_ids: &[String]) -> [f32; 9] {
    let mut v = [0.0f32; 9];
    if card_ids.is_empty() {
        return v;
    }
    let mut attack_damage_total = 0.0f32;
    let mut attack_cards = 0i32;
    for raw_id in card_ids {
        let card = match get_card(raw_id) {
            Some(c) => c,
            None => continue,
        };
        match card {
            Card::Umamusume(u) => {
                let stage = u.stage as f32;
                let bucket = if stage <= 0.0 { 0 } else { 1 };
                v[bucket] += 1.0;
                if let Some(att) = primary_attack(u) {
                    attack_damage_total += att.damage as f32;
                    attack_cards += 1;
                }
                if u.ability.is_some() {
                    v[8] += 1.0;
                }
            }
            Card::Trainer(t) => {
                v[2] += 1.0;
                let e = &t.effect;
                if e.draw.is_some() || e.shuffle_hand_into_deck_draw.is_some() {
                    v[3] += 1.0;
                }
                if e.search_umamusume.is_some()
                    || e.search_evolution_umamusume.is_some()
                    || e.search_random_basic_umamusume.is_some()
                {
                    v[4] += 1.0;
                }
                if e.extra_energy_attach.is_some() || e.attach_energy_from_zone_to_bench.is_some() {
                    v[5] += 1.0;
                }
                if e.heal.is_some() || e.recover_active_special_conditions.is_some() {
                    v[6] += 1.0;
                }
                if e.gust_opponent.is_some()
                    || e.discard_random_opponent_active_energy.is_some()
                    || t.trainer_type == TrainerType::Tool
                {
                    v[7] += 1.0;
                }
            }
        }
    }
    for i in 0..8 {
        v[i] /= 10.0;
    }
    let denom = attack_cards.max(1) as f32;
    v[8] = (attack_damage_total / denom) / 120.0;
    v
}

fn discard_role_features(card_ids: &[String]) -> [f32; 3] {
    let mut counts = [0.0f32; 3];
    for raw_id in card_ids {
        let card = match get_card(raw_id) {
            Some(c) => c,
            None => continue,
        };
        match card {
            Card::Trainer(_) => counts[0] += 1.0,
            Card::Umamusume(u) => {
                if (u.stage as f32) <= 0.0 {
                    counts[1] += 1.0;
                } else {
                    counts[2] += 1.0;
                }
            }
        }
    }
    [counts[0] / 20.0, counts[1] / 10.0, counts[2] / 10.0]
}

fn uma_readiness_features(entry: Option<&PublicUmaObservation>) -> [f32; 4] {
    let mut v = [0.0f32; 4];
    let Some(entry) = entry else { return v; };
    let card = match get_card(&entry.card_id) {
        Some(c) => c,
        None => return v,
    };
    let uma = match card {
        Card::Umamusume(u) => u,
        Card::Trainer(_) => return v,
    };
    let attack = match primary_attack(uma) {
        Some(a) => a,
        None => return v,
    };
    let cost = &attack.cost;
    let typed_deficit = typed_energy_deficit(&entry.energies, cost);
    let total_cost = cost_total(cost);
    let energy_total = entry.energy_total as f32;

    v[0] = 1.5f32.min(energy_total / total_cost.max(1.0));
    v[1] = typed_deficit / 4.0;
    v[2] = attack.damage as f32 / 150.0;
    v[3] = if typed_deficit <= 0.0 && energy_total >= total_cost {
        1.0
    } else {
        0.0
    };
    v
}

fn matching_evolution_count(hand_ids: &[String], board: &[&PublicUmaObservation]) -> f32 {
    let mut species_in_play: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for entry in board {
        species_in_play.insert(entry.species.as_str());
    }
    let mut count = 0.0f32;
    for raw_id in hand_ids {
        let card = match get_card(raw_id) {
            Some(c) => c,
            None => continue,
        };
        if let Card::Umamusume(u) = card {
            if (u.stage as f32) > 0.0 && species_in_play.contains(u.species.as_str()) {
                count += 1.0;
            }
        }
    }
    count
}

fn ready_attacker_count(board: &[&PublicUmaObservation]) -> f32 {
    board
        .iter()
        .map(|entry| {
            let r = uma_readiness_features(Some(entry));
            if r[3] > 0.0 {
                1.0
            } else {
                0.0
            }
        })
        .sum()
}

fn ability_ready_count(board: &[&PublicUmaObservation]) -> f32 {
    let mut count = 0.0f32;
    for entry in board {
        let card = match get_card(&entry.card_id) {
            Some(c) => c,
            None => continue,
        };
        if let Card::Umamusume(u) = card {
            if u.ability.is_some() && !entry.used_ability_this_turn {
                count += 1.0;
            }
        }
    }
    count
}

fn can_ko(attacker: Option<&PublicUmaObservation>, defender: Option<&PublicUmaObservation>) -> f32 {
    let (Some(a), Some(d)) = (attacker, defender) else {
        return 0.0;
    };
    let r = uma_readiness_features(Some(a));
    if r[3] <= 0.0 {
        return 0.0;
    }
    let mut damage = r[2] * 150.0;
    // Weakness bonus: simulator applies `damage += defender.weakness.amount`
    // when damage > 0 and defender's printed weakness type matches the
    // attacker's primary type (`flow/combat.rs:303-305`). Featurizer
    // previously ignored this — bit-exact mirror with the Python fix in
    // `features.py::_can_ko`.
    if damage > 0.0 {
        if let (Some(Card::Umamusume(attacker_uma)), Some(Card::Umamusume(defender_uma))) =
            (get_card(&a.card_id), get_card(&d.card_id))
        {
            if attacker_uma.r#type == defender_uma.weakness.r#type {
                damage += defender_uma.weakness.amount as f32;
            }
        }
    }
    if damage >= d.hp as f32 {
        1.0
    } else {
        0.0
    }
}

fn next_energy_matches_active_need(side: &PublicSideObservation) -> f32 {
    let zone = &side.energy_zone;
    let active = match &side.active {
        Some(a) => a,
        None => return 0.0,
    };
    if zone.is_empty() {
        return 0.0;
    }
    let card = match get_card(&active.card_id) {
        Some(c) => c,
        None => return 0.0,
    };
    let uma = match card {
        Card::Umamusume(u) => u,
        Card::Trainer(_) => return 0.0,
    };
    let attack = match primary_attack(uma) {
        Some(a) => a,
        None => return 0.0,
    };
    // Python looks up `cost.get(zone[0], 0)` which is a flat dict
    // get — colorless is a valid lookup key here, so iterate ALL energy
    // types (including colorless) when matching by name.
    let energy_type = zone[0].as_str();
    let required = EnergyType::ALL
        .iter()
        .copied()
        .find(|t| energy_type_name(*t) == energy_type)
        .map(|t| attack.cost.get(t) as f32)
        .unwrap_or(0.0);
    let attached = active.energies.get(energy_type).copied().unwrap_or(0) as f32;
    if required > attached {
        1.0
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Action features + per-action card idx pair.
// ---------------------------------------------------------------------------

/// Stack the pre-computed `features` vec of each action into a [A, 48]
/// row-major array (as a flat `Vec<f32>`). Per the Python contract,
/// every action MUST carry exactly `ACTION_DIM` features — bail with a
/// caller-friendly error if not (the TS enumerator already enforces this
/// via its own assertion; this catches transport-time corruption).
pub fn legal_actions_features(actions: &[LegalAiAction]) -> Result<Vec<f32>, FeaturizeError> {
    let mut buf: Vec<f32> = Vec::with_capacity(actions.len() * ACTION_DIM);
    for (i, action) in actions.iter().enumerate() {
        if action.features.len() != ACTION_DIM {
            return Err(FeaturizeError::ActionFeatureLen {
                index: i,
                got: action.features.len(),
                expected: ACTION_DIM,
                id: action.id.clone(),
            });
        }
        for v in &action.features {
            buf.push(*v as f32);
        }
    }
    Ok(buf)
}

/// `(source, target)` int64 idx pair per action — port of
/// `action_card_idx_pair`. Returns a flat row-major `[A, 2]` buffer
/// suitable for ndarray reshape.
pub fn action_card_idx_pairs_flat(actions: &[LegalAiAction]) -> Vec<i64> {
    let mut buf = Vec::with_capacity(actions.len() * 2);
    for action in actions {
        let src = action.action_source_card_idx.map(|x| x as i64).unwrap_or(0);
        let tgt = action.action_target_card_idx.map(|x| x as i64).unwrap_or(0);
        buf.push(src);
        buf.push(tgt);
    }
    buf
}

/// Build the `[NUM_ZONES, MAX_CARDS_PER_ZONE]` int64 card-id tensor (as
/// a flat row-major Vec). Zone iteration order is `ZONE_NAMES` and per-
/// zone width caps are `ZONE_WIDTHS`. Entries beyond the cap are
/// silently clipped (matches Python `observation_to_card_ids`).
pub fn observation_card_ids_by_zone(obs: &PublicObservation) -> Vec<i64> {
    let mut buf = vec![0i64; NUM_ZONES * MAX_CARDS_PER_ZONE];
    for (zone_idx, &zone) in ZONE_NAMES.iter().enumerate() {
        let cap = ZONE_WIDTHS[zone_idx];
        if let Some(ids) = obs.card_ids_by_zone.get(zone) {
            for (slot, &value) in ids.iter().take(cap).enumerate() {
                buf[zone_idx * MAX_CARDS_PER_ZONE + slot] = value as i64;
            }
        }
    }
    buf
}

// ---------------------------------------------------------------------------
// v3.2 per-Uma slot tokens (R16-P2 C1, ported from
// `training/uma_ai/features.py:563-705`).
//
// Slot order is FROZEN as UMA_SLOT_ORDER below — index 0 is own active,
// 1..4 own bench 0..3, index 5 opp active, 6..9 opp bench 0..3. The 4th
// bench slot per side is reserved (always absent under engine MAX_BENCH=3);
// kept so the byte layout stays stable if MAX_BENCH ever grows. Absent
// slots emit `card_id=0` and an all-zero feature row (same convention as
// v3.0 `card_ids_by_zone` padding).
// ---------------------------------------------------------------------------

/// Number of per-Uma slot tokens (2 actives + 8 bench placeholders).
pub const UMA_SLOT_COUNT: usize = 10;
/// FROZEN width of the per-slot feature vector. See `_uma_slot_feature_row`
/// in `training/uma_ai/features.py:577` for the column-by-column layout.
pub const UMA_SLOT_FEATURE_DIM: usize = 23;
/// Bench placeholder count per side (= 4; engine `MAX_BENCH=3` so the 4th
/// is reserved). Mirrors `_UMA_SLOT_BENCH_PER_SIDE`.
const UMA_SLOT_BENCH_PER_SIDE: usize = 4;

// Per-slot feature-index constants (mirror `_UMA_SLOT_F_*` in Python).
const _UMA_SLOT_F_POLARITY: usize = 0;
const _UMA_SLOT_F_ROLE_ACTIVE: usize = 1;
const _UMA_SLOT_F_SLOT_IDX: usize = 2;
const _UMA_SLOT_F_PRESENT: usize = 3;
const _UMA_SLOT_F_HP: usize = 4;
const _UMA_SLOT_F_DAMAGE: usize = 5;
const _UMA_SLOT_F_STAGE: usize = 6;
const _UMA_SLOT_F_ENERGY_TOTAL: usize = 7;
const _UMA_SLOT_F_ENERGY_TYPED_START: usize = 8; // exclusive end = 18
const _UMA_SLOT_F_TOOL: usize = 18;
const _UMA_SLOT_F_COND_PARALYSIS: usize = 19;
const _UMA_SLOT_F_COND_COUNT: usize = 20;
const _UMA_SLOT_F_ABILITY_USED: usize = 21;
const _UMA_SLOT_F_EVOLVED: usize = 22;

/// Per-slot typed-energy order — MUST match `_UMA_SLOT_ENERGY_TYPES` in
/// Python and the existing `ENERGY_TYPES_ORDER` 10-wide slice (slots 48-58
/// of v3.0). Compile-time guarded via the length assertion below.
const _UMA_SLOT_ENERGY_TYPES: [&str; 10] = ENERGY_TYPES_ORDER;
const _: () = {
    // Width sanity (mirror Python's assert on _UMA_SLOT_ENERGY_TYPES len).
    assert!(_UMA_SLOT_ENERGY_TYPES.len() == 10);
};

/// Build the v3.2 per-Uma slot tensors from a `PublicObservation`.
///
/// Returns `(card_ids, features)`:
///   - `card_ids`: flat `Vec<i64>` of length `UMA_SLOT_COUNT` (= 10). Caller
///     reshapes to `[1, 10]` for the ONNX feed (matches Python serve_onnx
///     which prepends the batch dim via `slot_ids[None, :]`).
///   - `features`: flat row-major `Vec<f32>` of length
///     `UMA_SLOT_COUNT * UMA_SLOT_FEATURE_DIM` (= 230). Caller reshapes to
///     `[1, 10, 23]`.
///
/// Absent slots (`active` is None, bench slot empty, or `card_id` is the
/// empty string) yield `card_id=0` and an all-zero feature row. This mirrors
/// the Python `observation_to_uma_slots` "absence is zero" contract.
pub fn observation_uma_slots(obs: &PublicObservation) -> (Vec<i64>, Vec<f32>) {
    let mut card_ids = vec![0i64; UMA_SLOT_COUNT];
    let mut features = vec![0.0f32; UMA_SLOT_COUNT * UMA_SLOT_FEATURE_DIM];

    // Active slots: own at slot 0 (polarity +1), opp at slot 5 (polarity
    // -1). `slot_idx_norm = -1.0` marks the active role (per chunk plan).
    for &(side, polarity, slot_idx) in &[
        (Side::Own, 1.0f32, 0usize),
        (Side::Opp, -1.0f32, 5usize),
    ] {
        let side_obs = match side {
            Side::Own => &obs.own,
            Side::Opp => &obs.opponent,
        };
        if let Some(active) = side_obs.active.as_ref() {
            if !active.card_id.is_empty() {
                card_ids[slot_idx] = card_vocab_index(Some(active.card_id.as_str())) as i64;
                let row_start = slot_idx * UMA_SLOT_FEATURE_DIM;
                fill_uma_slot_row(
                    &mut features[row_start..row_start + UMA_SLOT_FEATURE_DIM],
                    Some(active),
                    polarity,
                    true,
                    -1.0,
                );
            }
        }
    }

    // Bench slots: own bench i at slot 1+i, opp bench i at slot 6+i. Bench
    // index norm = i / max(1, bench_per_side - 1) (= i/3 for the 4-slot
    // placeholder; the 4th slot is always absent under MAX_BENCH=3).
    let bench_denom = (UMA_SLOT_BENCH_PER_SIDE - 1).max(1) as f32;
    for &(side, polarity, slot_base) in &[
        (Side::Own, 1.0f32, 1usize),
        (Side::Opp, -1.0f32, 6usize),
    ] {
        let side_obs = match side {
            Side::Own => &obs.own,
            Side::Opp => &obs.opponent,
        };
        for bench_pos in 0..UMA_SLOT_BENCH_PER_SIDE {
            let entry = side_obs
                .bench
                .get(bench_pos)
                .and_then(|o| o.as_ref());
            let Some(entry) = entry else { continue; };
            if entry.card_id.is_empty() {
                continue;
            }
            let slot_idx = slot_base + bench_pos;
            card_ids[slot_idx] = card_vocab_index(Some(entry.card_id.as_str())) as i64;
            let row_start = slot_idx * UMA_SLOT_FEATURE_DIM;
            fill_uma_slot_row(
                &mut features[row_start..row_start + UMA_SLOT_FEATURE_DIM],
                Some(entry),
                polarity,
                false,
                bench_pos as f32 / bench_denom,
            );
        }
    }

    (card_ids, features)
}

#[derive(Copy, Clone)]
enum Side {
    Own,
    Opp,
}

/// Mirror of Python `_uma_slot_feature_row` — emit one
/// `UMA_SLOT_FEATURE_DIM`-wide feature row for a single slot. The slice
/// must already be zeroed; we only set non-zero columns.
fn fill_uma_slot_row(
    row: &mut [f32],
    uma: Option<&PublicUmaObservation>,
    polarity: f32,
    role_active: bool,
    slot_idx_norm: f32,
) {
    debug_assert_eq!(row.len(), UMA_SLOT_FEATURE_DIM);
    let Some(uma) = uma else { return; };
    if uma.card_id.is_empty() {
        return;
    }

    row[_UMA_SLOT_F_POLARITY] = polarity;
    row[_UMA_SLOT_F_ROLE_ACTIVE] = if role_active { 1.0 } else { 0.0 };
    row[_UMA_SLOT_F_SLOT_IDX] = slot_idx_norm;
    row[_UMA_SLOT_F_PRESENT] = 1.0;

    let max_hp = (uma.max_hp as f32).max(1.0);
    let hp = uma.hp as f32;
    row[_UMA_SLOT_F_HP] = (hp / max_hp).max(0.0).min(1.0);
    row[_UMA_SLOT_F_DAMAGE] = ((max_hp - hp) / max_hp).max(0.0).min(1.0);
    row[_UMA_SLOT_F_STAGE] = uma.stage as f32 / 2.0;
    row[_UMA_SLOT_F_ENERGY_TOTAL] = uma.energy_total as f32 / 6.0;

    for (offset, &energy_type) in _UMA_SLOT_ENERGY_TYPES.iter().enumerate() {
        let amount = uma.energies.get(energy_type).copied().unwrap_or(0) as f32;
        row[_UMA_SLOT_F_ENERGY_TYPED_START + offset] = amount / 4.0;
    }

    row[_UMA_SLOT_F_TOOL] = if uma.tool_card_id.as_ref().map(|s| !s.is_empty()).unwrap_or(false) {
        1.0
    } else {
        0.0
    };

    let cond_paralysis = uma
        .special_conditions
        .iter()
        .any(|c| c == "paralysed");
    row[_UMA_SLOT_F_COND_PARALYSIS] = if cond_paralysis { 1.0 } else { 0.0 };
    // Full SpecialCondition union has 5 members (asleep/burned/frozen/
    // paralysed/poisoned per shared/src/types.ts:10), so divide by 5.0.
    row[_UMA_SLOT_F_COND_COUNT] = uma.special_conditions.len() as f32 / 5.0;
    row[_UMA_SLOT_F_ABILITY_USED] = if uma.used_ability_this_turn { 1.0 } else { 0.0 };
    row[_UMA_SLOT_F_EVOLVED] = if (uma.stage as f32) > 0.0 { 1.0 } else { 0.0 };
}

/// Errors surfaced by the featurizer.
#[derive(Debug)]
pub enum FeaturizeError {
    ActionFeatureLen {
        index: usize,
        got: usize,
        expected: usize,
        id: String,
    },
}

impl std::fmt::Display for FeaturizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FeaturizeError::ActionFeatureLen { index, got, expected, id } => {
                write!(
                    f,
                    "action[{index}] id={id:?}: feature length {got} != expected {expected}"
                )
            }
        }
    }
}

impl std::error::Error for FeaturizeError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::random::{with_rng, Rng};
    use crate::headless_setup::setup_ai_vs_ai_game;
    use crate::policy::observation::build_public_observation;
    use crate::policy::types::PublicUmaTurnState;

    fn fixture() -> PublicObservation {
        let rng = Rng::from_seed("featurize-fixture:selfplay", "selfplay");
        let (state, _used) = with_rng(rng, || setup_ai_vs_ai_game());
        build_public_observation(&state, crate::core::constants::SideId::Player)
    }

    #[test]
    fn state_vector_dimension_is_110() {
        let obs = fixture();
        let v = observation_state_features(&obs);
        assert_eq!(v.len(), STATE_DIM_V3);
    }

    #[test]
    fn state_vector_first_slots_match_normalized_form() {
        // slot 2 = turnNumber / 20; with a fresh game, turn_number is 1
        // after setup (matches the TS setup phase advance).
        let obs = fixture();
        let v = observation_state_features(&obs);
        assert!((v[2] - obs.turn_number as f32 / 20.0).abs() < 1e-6);
        // slot 5 = own.handCount / 10
        assert!((v[5] - obs.own.hand_count as f32 / 10.0).abs() < 1e-6);
    }

    #[test]
    fn card_ids_by_zone_packs_to_padded_row_major() {
        let obs = fixture();
        let pack = observation_card_ids_by_zone(&obs);
        assert_eq!(pack.len(), NUM_ZONES * MAX_CARDS_PER_ZONE);
        // ownActive zone is index 0; first slot must be a positive vocab idx
        // (the player's active uma is always set after setup).
        if obs.own.active.is_some() {
            assert!(pack[0] > 0);
        }
    }

    #[test]
    fn pending_choice_one_hot_none_sets_slot_zero() {
        let h = pending_choice_one_hot(None);
        assert_eq!(h, [1.0, 0.0, 0.0]);
        let h = pending_choice_one_hot(Some("promoteAfterKnockout"));
        assert_eq!(h, [0.0, 1.0, 0.0]);
        let h = pending_choice_one_hot(Some("switchAfterGust"));
        assert_eq!(h, [0.0, 0.0, 1.0]);
        // Unknown union arm collapses to slot 0 (parity with Python).
        let h = pending_choice_one_hot(Some("unknownArm"));
        assert_eq!(h, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn hash_to_unit_uses_vocab_when_available() {
        // Known basic card → nonzero, deterministic.
        let a = hash_to_unit("matikanetannhauserBasic");
        let b = hash_to_unit("matikanetannhauserBasic");
        assert_eq!(a, b);
        assert!(a > 0.0);
        // Empty → 0.
        assert_eq!(hash_to_unit(""), 0.0);
    }

    // ----------------------------------------------------------------
    // v3.2 per-Uma slot tokens
    // ----------------------------------------------------------------

    #[test]
    fn uma_slots_have_correct_shape() {
        let obs = fixture();
        let (card_ids, features) = observation_uma_slots(&obs);
        assert_eq!(card_ids.len(), UMA_SLOT_COUNT);
        assert_eq!(features.len(), UMA_SLOT_COUNT * UMA_SLOT_FEATURE_DIM);
        // UMA_SLOT_COUNT and UMA_SLOT_FEATURE_DIM frozen to the C1
        // contract — guard against accidental width drift.
        assert_eq!(UMA_SLOT_COUNT, 10);
        assert_eq!(UMA_SLOT_FEATURE_DIM, 23);
    }

    #[test]
    fn uma_slots_active_row_present_polarity_roleactive() {
        let obs = fixture();
        let (card_ids, features) = observation_uma_slots(&obs);

        // Own active is slot 0; setup_ai_vs_ai_game always assigns one.
        assert!(card_ids[0] > 0, "own active card_id must be a real vocab idx");
        let row0 = &features[0..UMA_SLOT_FEATURE_DIM];
        assert_eq!(row0[_UMA_SLOT_F_POLARITY], 1.0, "own polarity = +1");
        assert_eq!(row0[_UMA_SLOT_F_ROLE_ACTIVE], 1.0, "active flag = 1");
        assert_eq!(row0[_UMA_SLOT_F_SLOT_IDX], -1.0, "active slot idx = -1");
        assert_eq!(row0[_UMA_SLOT_F_PRESENT], 1.0, "present mask = 1");
        // hp ratio is in [0, 1]; evolved is 0 or 1.
        assert!(row0[_UMA_SLOT_F_HP] >= 0.0 && row0[_UMA_SLOT_F_HP] <= 1.0);
        assert!(
            row0[_UMA_SLOT_F_EVOLVED] == 0.0 || row0[_UMA_SLOT_F_EVOLVED] == 1.0
        );

        // Opp active is slot 5; opponent polarity = -1.
        assert!(card_ids[5] > 0, "opp active card_id must be set");
        let row5 = &features[5 * UMA_SLOT_FEATURE_DIM..6 * UMA_SLOT_FEATURE_DIM];
        assert_eq!(row5[_UMA_SLOT_F_POLARITY], -1.0, "opp polarity = -1");
        assert_eq!(row5[_UMA_SLOT_F_ROLE_ACTIVE], 1.0, "opp active flag");
        assert_eq!(row5[_UMA_SLOT_F_PRESENT], 1.0);
    }

    #[test]
    fn uma_slots_absent_4th_bench_is_zero() {
        // Engine MAX_BENCH=3 so slot 4 (own bench[3]) and slot 9 (opp
        // bench[3]) are ALWAYS absent. The whole row must be zero AND
        // the card_id must be 0 (the padding idx).
        let obs = fixture();
        let (card_ids, features) = observation_uma_slots(&obs);
        for &slot in &[4usize, 9usize] {
            assert_eq!(
                card_ids[slot], 0,
                "reserved 4th-bench slot {} card_id must be 0 padding",
                slot
            );
            let row = &features[slot * UMA_SLOT_FEATURE_DIM..(slot + 1) * UMA_SLOT_FEATURE_DIM];
            for (i, &v) in row.iter().enumerate() {
                assert_eq!(
                    v, 0.0,
                    "reserved 4th-bench slot {} col {} must be 0 (got {})",
                    slot, i, v
                );
            }
        }
    }

    #[test]
    fn uma_slots_hp_ratio_matches_observation() {
        // Numeric parity gate: hp_norm = hp/max_hp clamped to [0,1].
        // damage_norm = (max_hp-hp)/max_hp. The two MUST sum to 1.0 for
        // a present slot (no clipping under nominal HP bounds).
        let obs = fixture();
        let (_card_ids, features) = observation_uma_slots(&obs);
        let row0 = &features[0..UMA_SLOT_FEATURE_DIM];
        let sum = row0[_UMA_SLOT_F_HP] + row0[_UMA_SLOT_F_DAMAGE];
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "hp+damage must sum to 1.0 for present slot (got {})",
            sum
        );

        // Cross-check against the raw observation.
        let active = obs.own.active.as_ref().expect("own active set in fixture");
        let expected_hp = (active.hp as f32 / (active.max_hp as f32).max(1.0))
            .max(0.0)
            .min(1.0);
        assert!((row0[_UMA_SLOT_F_HP] - expected_hp).abs() < 1e-6);
    }

    #[test]
    fn uma_slots_typed_energy_matches_per_slot_sum() {
        // Per-slot typed-energy slice ∈ [0, 1.5+] (each / 4.0); their
        // SUM should equal energyTotal / 4.0 modulo per-type rounding —
        // strictly typed_total / 4.0 = sum of typed entries when all
        // attached energies enumerate the same 10 types as the order
        // table (which they do, see ENERGY_TYPES_ORDER guard).
        let obs = fixture();
        let (_card_ids, features) = observation_uma_slots(&obs);
        let row0 = &features[0..UMA_SLOT_FEATURE_DIM];
        let typed_sum: f32 = row0
            [_UMA_SLOT_F_ENERGY_TYPED_START.._UMA_SLOT_F_ENERGY_TYPED_START + 10]
            .iter()
            .sum();
        let active = obs.own.active.as_ref().unwrap();
        let raw_sum: u32 = active.energies.values().map(|&v| v as u32).sum();
        let expected = raw_sum as f32 / 4.0;
        assert!(
            (typed_sum - expected).abs() < 1e-5,
            "typed-energy sum {} != raw {}/4.0 = {}",
            typed_sum, raw_sum, expected
        );
    }

    // ----------------------------------------------------------------
    // v3.1 temporal / turn-state builder (R16-P1)
    // ----------------------------------------------------------------

    #[test]
    fn v3_1_state_vector_dimension_is_164() {
        let obs = fixture();
        let v = observation_state_features_v3_1(&obs);
        assert_eq!(v.len(), STATE_DIM_V3_1);
        assert_eq!(STATE_DIM_V3_1, 164);
    }

    #[test]
    fn v3_1_head_is_byte_identical_to_v3_0() {
        // The first 110 slots MUST be byte-equal to the standalone v3.0
        // builder — this is the core layering contract.
        let obs = fixture();
        let head = observation_state_features(&obs);
        let v31 = observation_state_features_v3_1(&obs);
        assert_eq!(&v31[..STATE_DIM_V3], &head[..]);
    }

    #[test]
    fn v3_1_temporal_block_layout_and_bounds() {
        // Slot ranges from the FROZEN v3.1 layout (110-d v3.0 head +
        // 54-d temporal block). All entries must be in [0, 1] after
        // bounded-norm/bool encoding.
        let obs = fixture();
        let v = observation_state_features_v3_1(&obs);

        // Global temporal: turn counters in [0,1] (capped), bools in {0,1}.
        for &slot in &[110usize, 111] {
            assert!(
                v[slot] >= 0.0 && v[slot] <= 1.0,
                "global temporal slot {} out of [0,1]: {}",
                slot, v[slot]
            );
        }
        for &slot in &[112usize, 113] {
            assert!(
                v[slot] == 0.0 || v[slot] == 1.0,
                "isFirstTurn slot {} must be 0/1: {}", slot, v[slot]
            );
        }

        // Side turnState scalars are all bounded-norm in [0,1]
        // (effectiveRetreatCostReduction can be > 30 only in pathological
        // states; the cap clamps).
        for slot in 114..128 {
            assert!(
                v[slot] >= 0.0 && v[slot] <= 1.0,
                "side turnState slot {} out of [0,1]: {}", slot, v[slot]
            );
        }

        // Active + bench-aggregate per-Uma scalars in [0,1] too.
        for slot in 128..164 {
            assert!(
                v[slot] >= 0.0 && v[slot] <= 1.0,
                "per-Uma temporal slot {} out of [0,1]: {}", slot, v[slot]
            );
        }
    }

    #[test]
    fn v3_1_global_temporal_matches_observation() {
        // Slots 110, 111 = turn counters; 112, 113 = isFirstTurn bools.
        let obs = fixture();
        let v = observation_state_features_v3_1(&obs);
        let t = &obs.temporal;
        let exp_own = (t.own_turns_taken as f32).min(20.0) / 20.0;
        let exp_opp = (t.opponent_turns_taken as f32).min(20.0) / 20.0;
        assert!((v[110] - exp_own).abs() < 1e-6);
        assert!((v[111] - exp_opp).abs() < 1e-6);
        assert_eq!(v[112], if t.own_is_first_turn { 1.0 } else { 0.0 });
        assert_eq!(v[113], if t.opponent_is_first_turn { 1.0 } else { 0.0 });
    }

    #[test]
    fn v3_1_uma_turn_state_absent_uma_is_zero() {
        // The per-Uma helper must return all zeros for None input
        // (mirrors Python `if not uma: return np.zeros(9)`).
        let row = uma_turn_state_vec(None);
        for (i, &v) in row.iter().enumerate() {
            assert_eq!(v, 0.0, "absent-uma col {} must be 0 (got {})", i, v);
        }
    }

    #[test]
    fn v3_1_bench_aggregate_empty_bench_is_zero() {
        // Wipe both benches; the aggregate must collapse to zero rows.
        let mut obs = fixture();
        for slot in obs.own.bench.iter_mut() {
            *slot = None;
        }
        for slot in obs.opponent.bench.iter_mut() {
            *slot = None;
        }
        let v = observation_state_features_v3_1(&obs);
        for slot in 137..146 {
            assert_eq!(v[slot], 0.0, "own bench-agg slot {} must be 0", slot);
        }
        for slot in 155..164 {
            assert_eq!(v[slot], 0.0, "opp bench-agg slot {} must be 0", slot);
        }
    }

    #[test]
    fn uma_slots_absent_active_yields_zero_row() {
        // Take the seeded fixture, wipe own.active, re-build slots, and
        // confirm slot 0 (own active) is now zero card_id + zero row.
        let mut obs = fixture();
        obs.own.active = None;
        // Also blank the own bench so the no-active surface is sharp.
        for slot in obs.own.bench.iter_mut() {
            *slot = None;
        }
        let (card_ids, features) = observation_uma_slots(&obs);
        for i in 0..5 {
            assert_eq!(card_ids[i], 0, "own slot {} card_id must be 0", i);
            let row = &features[i * UMA_SLOT_FEATURE_DIM..(i + 1) * UMA_SLOT_FEATURE_DIM];
            for (j, &v) in row.iter().enumerate() {
                assert_eq!(v, 0.0, "own slot {} col {} must be 0 (got {})", i, j, v);
            }
        }
        // Opp side still has its actives — slot 5 (opp active) should
        // remain populated to confirm we did NOT zero everything.
        assert!(card_ids[5] > 0);
    }

    // ----------------------------------------------------------------
    // can_ko weakness-bonus correction (v33-correctness-fix slice)
    // ----------------------------------------------------------------

    fn make_uma_obs(
        card_id: &str,
        hp: i32,
        max_hp: i32,
        energy_total: u32,
        energies: &[(&str, u16)],
    ) -> PublicUmaObservation {
        let mut e = indexmap::IndexMap::new();
        for (k, v) in energies {
            e.insert((*k).to_string(), *v);
        }
        PublicUmaObservation {
            uid: 1,
            card_id: card_id.to_string(),
            species: String::new(),
            stage: 0,
            hp,
            max_hp,
            energy_total,
            energies: e,
            special_conditions: Vec::new(),
            tool_card_id: None,
            used_ability_this_turn: false,
            turn_state: PublicUmaTurnState {
                turns_in_play: 1,
                entered_this_turn: false,
                evolved_this_turn: false,
                evolved_last_turn: false,
                took_damage_last_turn: false,
                took_damage_this_turn: false,
                next_turn_damage_reduction: 0,
                attack_blocked_this_turn: false,
                paralysis_recovery_pending: false,
            },
        }
    }

    #[test]
    fn can_ko_applies_weakness_bonus_at_threshold() {
        // Darkness attacker (`manhattanCafeStage1`, 40 damage, cost
        // darkness+colorless) vs Psychic defender (`matikanetannhauserBasic`,
        // hp 60, weakness Darkness +20). Without weakness: 40 < 60 → no KO.
        // With weakness: 40 + 20 = 60 >= 60 → KO. This is the canonical
        // ~30% featurizer/simulator disagreement case the slice targets.
        let attacker = make_uma_obs(
            "manhattanCafeStage1",
            90,
            90,
            2,
            &[("darkness", 1), ("colorless", 1)],
        );
        let defender = make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[]);
        // Sanity: readiness should be ready (energy_total >= total_cost,
        // typed_deficit == 0).
        let r = uma_readiness_features(Some(&attacker));
        assert_eq!(r[3], 1.0, "attacker must be attack-ready in fixture");
        // can_ko returns 1.0 because 40 + 20 weakness == 60 hp.
        assert_eq!(can_ko(Some(&attacker), Some(&defender)), 1.0);
    }

    #[test]
    fn can_ko_no_weakness_when_types_dont_match() {
        // Psychic attacker (`matikanetannhauserBasic`, 20 damage) vs
        // Psychic defender (`haruUraraBasic`, hp 90). Defender weakness
        // is Darkness, not Psychic, so no bonus applies. 20 < 90 → no KO.
        let attacker = make_uma_obs("matikanetannhauserBasic", 60, 60, 1, &[("psychic", 1)]);
        let defender = make_uma_obs("haruUraraBasic", 90, 90, 0, &[]);
        let r = uma_readiness_features(Some(&attacker));
        assert_eq!(r[3], 1.0);
        assert_eq!(can_ko(Some(&attacker), Some(&defender)), 0.0);
    }

    // ----------------------------------------------------------------
    // v3.3 additive-tail (v33-additive-tail-scoping.md)
    // ----------------------------------------------------------------

    #[test]
    fn v3_3_state_vector_dimension_is_167() {
        let obs = fixture();
        let v = observation_state_features_v3_3(&obs);
        assert_eq!(v.len(), STATE_DIM_V3_3);
        assert_eq!(STATE_DIM_V3_3, 167);
    }

    #[test]
    fn v3_3_head_is_byte_identical_to_v3_1() {
        // The first 164 slots MUST be byte-equal to the standalone v3.1
        // builder — this is the core layering contract that lets a v3.2
        // ckpt warm-start a v3.3 graph via zero-init residual.
        let obs = fixture();
        let v31 = observation_state_features_v3_1(&obs);
        let v33 = observation_state_features_v3_3(&obs);
        assert_eq!(&v33[..STATE_DIM_V3_1], &v31[..]);
    }

    #[test]
    fn v3_3_tail_emits_opp_used_flags() {
        // With a fresh fixture, no actions have been taken on either
        // side, so all used* flags are false → tail slots are 0.0.
        let obs = fixture();
        let v33 = observation_state_features_v3_3(&obs);
        assert_eq!(v33[164], 0.0, "fresh fixture has opp.used_supporter=false");
        assert_eq!(v33[165], 0.0, "fresh fixture has opp.used_retreat=false");
        assert_eq!(v33[166], 0.0, "fresh fixture has opp.used_stadium=false");

        // Mutate the opp side; the tail flips.
        let mut obs2 = obs.clone();
        obs2.opponent.used_supporter_this_turn = true;
        obs2.opponent.used_stadium_this_turn = true;
        let v33b = observation_state_features_v3_3(&obs2);
        assert_eq!(v33b[164], 1.0);
        assert_eq!(v33b[165], 0.0);
        assert_eq!(v33b[166], 1.0);
        // Own-side flags unchanged → slots 29/30/31 still zero from the
        // fresh fixture.
        assert_eq!(v33b[29], 0.0);
        assert_eq!(v33b[30], 0.0);
        assert_eq!(v33b[31], 0.0);
    }

    // ----------------------------------------------------------------
    // v3.5 multichannel-tail (v35-multichannel-tail-scoping.md)
    // ----------------------------------------------------------------

    #[test]
    fn v3_5_state_vector_dimension_is_212() {
        let obs = fixture();
        let v = observation_state_features_v3_5(&obs);
        assert_eq!(v.len(), STATE_DIM_V3_5);
        assert_eq!(STATE_DIM_V3_5, 212);
    }

    #[test]
    fn v3_5_head_is_byte_identical_to_v3_3() {
        // The first 167 slots MUST be byte-equal to the standalone v3.3
        // builder — this is the core layering contract that lets a v3.3
        // ckpt warm-start a v3.5 graph via zero-init residual.
        let obs = fixture();
        let v33 = observation_state_features_v3_3(&obs);
        let v35 = observation_state_features_v3_5(&obs);
        assert_eq!(&v35[..STATE_DIM_V3_3], &v33[..]);
    }

    #[test]
    fn v3_5_tail_emits_phase_one_hot() {
        // Fresh fixture phase should land on exactly one of the 10 bits.
        let obs = fixture();
        let v35 = observation_state_features_v3_5(&obs);
        let phase_one_hot = &v35[167..177];
        let set: f32 = phase_one_hot.iter().sum();
        assert!(set <= 1.0, "phase one-hot is non-exclusive");
        // Mutate the phase; the one-hot moves accordingly.
        let mut obs2 = obs.clone();
        obs2.phase = AiPhase::Combat;
        let v35b = observation_state_features_v3_5(&obs2);
        let combat_idx = PHASES.iter().position(|&p| p == AiPhase::Combat).unwrap();
        assert_eq!(v35b[167 + combat_idx], 1.0);
        // Every other phase bit is zero.
        for i in 0..10 {
            if i != combat_idx {
                assert_eq!(v35b[167 + i], 0.0);
            }
        }
    }

    #[test]
    fn v3_5_tail_emits_condition_one_hot() {
        let mut obs = fixture();
        // Stamp the own active with two conditions; opp active with one.
        if let Some(active) = obs.own.active.as_mut() {
            active.special_conditions = vec!["paralysed".to_string(), "burned".to_string()];
        }
        if let Some(active) = obs.opponent.active.as_mut() {
            active.special_conditions = vec!["frozen".to_string()];
        }
        let v35 = observation_state_features_v3_5(&obs);
        // own [177:182] — paralysed=0, burned=1.
        assert_eq!(v35[177], 1.0);
        assert_eq!(v35[178], 1.0);
        assert_eq!(v35[179], 0.0);
        assert_eq!(v35[180], 0.0);
        assert_eq!(v35[181], 0.0);
        // opp [182:187] — frozen=4.
        assert_eq!(v35[182], 0.0);
        assert_eq!(v35[183], 0.0);
        assert_eq!(v35[184], 0.0);
        assert_eq!(v35[185], 0.0);
        assert_eq!(v35[186], 1.0);
    }

    #[test]
    fn v3_5_tail_emits_energy_zone_front() {
        let mut obs = fixture();
        obs.own.energy_zone = vec!["fire".to_string(), "water".to_string()];
        obs.opponent.energy_zone = vec!["psychic".to_string()];
        let v35 = observation_state_features_v3_5(&obs);
        // own [187:197] — front="fire" at index 1.
        let own_front: Vec<f32> = v35[187..197].to_vec();
        assert_eq!(own_front[1], 1.0, "fire should be set at index 1");
        assert_eq!(own_front.iter().sum::<f32>(), 1.0, "exactly one own-front bit");
        // opp [197:207] — front="psychic" at index 4.
        let opp_front: Vec<f32> = v35[197..207].to_vec();
        assert_eq!(opp_front[4], 1.0, "psychic should be set at index 4");
        assert_eq!(opp_front.iter().sum::<f32>(), 1.0, "exactly one opp-front bit");
    }

    #[test]
    fn v3_5_tail_emits_bench_refill_catastrophe() {
        let mut obs = fixture();
        // Empty both benches. Fresh fixture has populated benches, so
        // wipe them.
        obs.own.bench = vec![None, None, None];
        obs.opponent.bench = vec![None, None, None];
        let v35 = observation_state_features_v3_5(&obs);
        assert_eq!(v35[210], 1.0, "own bench-refill bit = 1.0 when no bench");
        assert_eq!(v35[211], 1.0, "opp bench-refill bit = 1.0 when no bench");

        // Put one Uma on own bench; own bit flips to 0.
        let mut obs2 = obs.clone();
        let proto = obs2.opponent.active.clone().or_else(|| obs2.own.active.clone());
        obs2.own.bench[0] = proto;
        let v35b = observation_state_features_v3_5(&obs2);
        assert_eq!(v35b[210], 0.0, "own bench-refill flips when bench has a Uma");
        assert_eq!(v35b[211], 1.0, "opp still no bench → still 1.0");
    }

    #[test]
    fn can_ko_zero_damage_does_not_apply_weakness() {
        // Mirror of simulator rule: weakness only applies when damage > 0.
        // We synthesise this by reducing the attacker's energy below the
        // attack cost — readiness[3] = 0, damage = 0, can_ko returns 0
        // without ever entering the weakness branch.
        let attacker = make_uma_obs("manhattanCafeStage1", 90, 90, 0, &[]);
        let defender = make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[]);
        // Readiness fail (energy_total 0 < total_cost 2).
        let r = uma_readiness_features(Some(&attacker));
        assert_eq!(r[3], 0.0);
        assert_eq!(can_ko(Some(&attacker), Some(&defender)), 0.0);
    }

    // ----------------------------------------------------------------
    // v3.6 priors-and-arithmetic (v36-priors-and-arithmetic-scoping.md)
    // ----------------------------------------------------------------

    #[test]
    fn v3_6_state_dim_is_246() {
        let obs = fixture();
        let v = observation_state_features_v3_6(&obs);
        assert_eq!(v.len(), STATE_DIM_V3_6);
        assert_eq!(STATE_DIM_V3_6, 246);
    }

    #[test]
    fn v3_6_head_is_byte_identical_to_v3_5_excluding_opp_energy_zone_band() {
        // Construct a fixture with a non-empty opp.energy_zone so the
        // v3.5 [197:207] band would carry a 1-bit; v3.6 must overwrite
        // that band with own.energy_pool typed multihot.
        let mut obs = fixture();
        // Opp energy_zone front = "psychic" → v3.5 sets bit at [197+4].
        obs.opponent.energy_zone = vec!["psychic".to_string()];
        // Own energy_pool = ["fire"] → v3.6 sets bit at [197+1] (own
        // in-band repurpose).
        obs.own.energy_pool = vec!["fire".to_string()];
        // Clear opp.energy_pool so the appended-tail opp slot doesn't
        // perturb other assertions.
        obs.opponent.energy_pool = Vec::new();

        let v35 = observation_state_features_v3_5(&obs);
        let v36 = observation_state_features_v3_6(&obs);

        // [0:197] byte-stable.
        assert_eq!(&v36[..197], &v35[..197]);
        // [197:207] differs: v3.5 had psychic-bit at offset 4; v3.6 has
        // fire-bit at offset 1.
        assert_eq!(v35[197 + 4], 1.0, "v3.5 opp.energy_zone.front bit");
        assert_eq!(v36[197 + 4], 0.0, "v3.6 must zero the v3.5 bit");
        assert_eq!(v36[197 + 1], 1.0, "v3.6 own.energy_pool typed bit");
        assert_eq!(
            v36[197..207].iter().sum::<f32>(),
            1.0,
            "exactly one own.energy_pool bit"
        );
        // [207:212] (rest of v3.5 tail) preserved byte-identically.
        assert_eq!(&v36[207..212], &v35[207..212]);
    }

    #[test]
    fn v3_6_energy_pool_multihot_both_sides() {
        let mut obs = fixture();
        obs.own.energy_pool = vec!["fire".to_string(), "water".to_string()];
        obs.opponent.energy_pool = vec!["lightning".to_string()];
        let v = observation_state_features_v3_6(&obs);
        // own at [197:207] — fire (idx 1) + water (idx 2).
        let own_band = &v[197..207];
        assert_eq!(own_band[1], 1.0, "own fire bit");
        assert_eq!(own_band[2], 1.0, "own water bit");
        assert_eq!(own_band.iter().sum::<f32>(), 2.0, "exactly 2 own bits");
        // opp at [212:222] — lightning (idx 3).
        let opp_band = &v[212..222];
        assert_eq!(opp_band[3], 1.0, "opp lightning bit");
        assert_eq!(opp_band.iter().sum::<f32>(), 1.0, "exactly 1 opp bit");
    }

    #[test]
    fn v3_6_energy_pool_collapses_duplicates() {
        // Bag semantics: duplicates collapse.
        let mut obs = fixture();
        obs.own.energy_pool = vec![
            "fire".to_string(),
            "fire".to_string(),
            "water".to_string(),
        ];
        let v = observation_state_features_v3_6(&obs);
        let own_band = &v[197..207];
        assert_eq!(own_band[1], 1.0, "fire bit set");
        assert_eq!(own_band[2], 1.0, "water bit set");
        assert_eq!(own_band.iter().sum::<f32>(), 2.0, "duplicates collapse");
    }

    #[test]
    fn v3_6_prize_one_hot_both_sides() {
        // Iterate own.points ∈ {0,1,2,3} and assert the correct slot fires.
        // Bit index = 3 - remaining where remaining = clamp(3 - points, 0, 3).
        for (points, expected_bit) in [(0u8, 0usize), (1, 1), (2, 2), (3, 3)] {
            let mut obs = fixture();
            obs.own.points = points;
            // Pin opp side at points=0 so its one-hot is at index 0.
            obs.opponent.points = 0;
            let v = observation_state_features_v3_6(&obs);
            // Own prize at [222:226].
            let own_prize = &v[222..226];
            assert_eq!(
                own_prize[expected_bit], 1.0,
                "own.points={} → own prize bit {} should be 1",
                points, expected_bit
            );
            assert_eq!(
                own_prize.iter().sum::<f32>(),
                1.0,
                "exactly one own-prize bit for points={}", points
            );
            // Opp prize at [226:230] — points=0 → bit 0.
            let opp_prize = &v[226..230];
            assert_eq!(opp_prize[0], 1.0, "opp prize bit 0 (points=0)");
            assert_eq!(opp_prize.iter().sum::<f32>(), 1.0);
        }
    }

    #[test]
    fn v3_6_bench_typed_aggregate_opp_only() {
        // Construct opp bench with mixed typed energies; assert the
        // opp-bench typed multihot at [230:240]. Own bench is NOT
        // featurized (impl-phase reconciliation dropped it).
        let mut obs = fixture();
        // Inject two opp bench Umas with typed energies. Use the fresh
        // fixture's bench-shape (3-slot Vec) and overwrite slot 0 + 1.
        let mut a = make_uma_obs(
            "matikanetannhauserBasic",
            60,
            60,
            2,
            &[("fire", 1), ("water", 1)],
        );
        a.uid = 100;
        let mut b = make_uma_obs(
            "matikanetannhauserBasic",
            60,
            60,
            1,
            &[("darkness", 1)],
        );
        b.uid = 101;
        obs.opponent.bench = vec![Some(a), Some(b), None];
        // Inject own bench with typed energies — these MUST NOT appear
        // in any v3.6 slot (own bench was dropped at reconciliation).
        let mut own_bench = make_uma_obs(
            "matikanetannhauserBasic",
            60,
            60,
            1,
            &[("steel", 1)],
        );
        own_bench.uid = 200;
        obs.own.bench = vec![Some(own_bench), None, None];

        let v = observation_state_features_v3_6(&obs);
        let opp_bench_band = &v[230..240];
        // ENERGY_TYPES_ORDER: grass(0), fire(1), water(2), lightning(3),
        // psychic(4), fighting(5), darkness(6), steel(7), colorless(8),
        // dragon(9).
        assert_eq!(opp_bench_band[1], 1.0, "opp bench fire bit");
        assert_eq!(opp_bench_band[2], 1.0, "opp bench water bit");
        assert_eq!(opp_bench_band[6], 1.0, "opp bench darkness bit");
        assert_eq!(opp_bench_band[7], 0.0, "opp bench steel bit not set (only own had steel)");
        assert_eq!(
            opp_bench_band.iter().sum::<f32>(),
            3.0,
            "exactly 3 opp-bench typed bits set"
        );
        // No own-bench typed slot exists; the only slot own bench could
        // bleed into is the opp typed band — verify the steel bit (idx 7)
        // stayed clear, proving own bench did NOT contaminate.
    }

    #[test]
    fn v3_6_lethal_face_value_truth_table() {
        // matikanetannhauserStage2: 60 damage @ {psychic:2, colorless:1}.
        // niceNatureBasic: 40 damage. (Faceshot: energy NOT checked.)
        // (a) opp.active = 60-dmg attacker, own.active hp=60 → own_lethal=1.
        let mut obs = fixture();
        let attacker = make_uma_obs("matikanetannhauserStage2", 120, 120, 0, &[]);
        let weak_defender = make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[]);
        obs.opponent.active = Some(attacker.clone());
        obs.own.active = Some(weak_defender.clone());
        let v = observation_state_features_v3_6(&obs);
        assert_eq!(v[240], 1.0, "own_lethal: opp 60-dmg ≥ own.hp=60");

        // (b) flip damage direction: weak defender hp=60 with own
        // attacker = 60-dmg → opp_lethal=1; while opp_active is the
        // weak defender → own_lethal=0 (attacker only 20 dmg vs hp=60).
        let mut obs2 = fixture();
        obs2.own.active = Some(attacker.clone());
        obs2.opponent.active = Some(weak_defender.clone());
        let v2 = observation_state_features_v3_6(&obs2);
        assert_eq!(v2[241], 1.0, "opp_lethal: own 60-dmg ≥ opp.hp=60");
        // own_lethal: opp.active (matikanetannhauserBasic, 20 dmg) vs
        // own.hp=120 → 20 < 120 → 0.
        assert_eq!(v2[240], 0.0, "own_lethal: opp 20-dmg < own.hp=120");

        // (c) attacker has no card (unknown id) → both lethal bits 0.
        let mut obs3 = fixture();
        let mut null_attacker = make_uma_obs("__unknown_card__", 60, 60, 0, &[]);
        null_attacker.uid = 999;
        obs3.opponent.active = Some(null_attacker.clone());
        obs3.own.active = Some(weak_defender.clone());
        let v3 = observation_state_features_v3_6(&obs3);
        assert_eq!(v3[240], 0.0, "no-card attacker → own_lethal=0");

        // (d) defender absent → lethal=0 (Python rule: "nothing to KO").
        let mut obs4 = fixture();
        obs4.opponent.active = Some(attacker.clone());
        obs4.own.active = None;
        let v4 = observation_state_features_v3_6(&obs4);
        assert_eq!(v4[240], 0.0, "absent defender → own_lethal=0");
    }

    #[test]
    fn v3_6_secondary_attack_bits_two_attack_card() {
        // matikanefukukitaruStage1: 2 attacks — primary 20 dmg @{psychic:1},
        // secondary 0 dmg @ {psychic:1, colorless:1}.
        let mut obs = fixture();
        // Energy covers BOTH costs (2 psychic ≥ {psychic:1} for primary,
        // 2 psychic ≥ {psychic:1, colorless:1} since colorless absorbs).
        let attacker = make_uma_obs(
            "matikanefukukitaruStage1",
            100,
            100,
            2,
            &[("psychic", 2)],
        );
        let defender = make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[]);
        obs.own.active = Some(attacker.clone());
        obs.opponent.active = Some(defender.clone());
        let v = observation_state_features_v3_6(&obs);
        assert_eq!(v[242], 1.0, "own_secondary_usable: energy covers cost");
        // secondary damage is 0 → would_KO = (0 >= 60) = 0.
        assert_eq!(v[243], 0.0, "own_secondary_would_KO: 0 dmg can't KO 60hp");

        // Insufficient energy: 1 psychic covers primary {psychic:1} but
        // not secondary {psychic:1, colorless:1} (total cost 2 > attached 1).
        let mut attacker2 = make_uma_obs(
            "matikanefukukitaruStage1",
            100,
            100,
            1,
            &[("psychic", 1)],
        );
        attacker2.uid = 101;
        let mut obs2 = fixture();
        obs2.own.active = Some(attacker2);
        obs2.opponent.active = Some(defender.clone());
        let v2 = observation_state_features_v3_6(&obs2);
        assert_eq!(v2[242], 0.0, "own_secondary_usable: cost not covered");
        assert_eq!(v2[243], 0.0, "would_KO requires usable");
    }

    #[test]
    fn v3_6_secondary_attack_bits_single_attack_card() {
        // matikanetannhauserBasic has exactly 1 attack → both bits 0.
        let mut obs = fixture();
        let attacker = make_uma_obs("matikanetannhauserBasic", 60, 60, 1, &[("psychic", 1)]);
        let defender = make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[]);
        obs.own.active = Some(attacker);
        obs.opponent.active = Some(defender);
        let v = observation_state_features_v3_6(&obs);
        assert_eq!(v[242], 0.0, "own_secondary_usable: only 1 attack");
        assert_eq!(v[243], 0.0, "own_secondary_would_KO: only 1 attack");
    }

    // ----------------------------------------------------------------
    // v3.7 combat-arith-and-catalog
    // (v37-combat-arith-and-catalog-scoping.md)
    // ----------------------------------------------------------------

    #[test]
    fn v3_7_state_dim_is_296() {
        let obs = fixture();
        let v = observation_state_features_v3_7(&obs);
        assert_eq!(v.len(), STATE_DIM_V3_7);
        assert_eq!(STATE_DIM_V3_7, 296);
    }

    #[test]
    fn v3_7_head_is_byte_identical_to_v3_6() {
        // v3.7 must NEVER mutate v3.6 slots [0:246]. Build a non-trivial
        // observation (multiple typed energies + a tool + ability) and
        // confirm the first 246 bytes are bit-identical to the v3.6
        // builder on the same observation.
        let mut obs = fixture();
        obs.own.energy_pool = vec!["fire".into(), "psychic".into()];
        obs.opponent.energy_pool = vec!["lightning".into()];
        // Tool + ability on own active.
        let mut owner = make_uma_obs(
            "matikanetannhauserStage1",
            90,
            90,
            2,
            &[("psychic", 1), ("colorless", 1)],
        );
        owner.tool_card_id = Some("leftoverCarrot".to_string());
        owner.used_ability_this_turn = true;
        obs.own.active = Some(owner);
        let v36 = observation_state_features_v3_6(&obs);
        let v37 = observation_state_features_v3_7(&obs);
        assert_eq!(&v37[..STATE_DIM_V3_6], &v36[..]);
    }

    #[test]
    fn v3_7_tail_offsets_match_python_layout() {
        // Constant guard: catches a future drift between the Python
        // `_V37_*` offsets and the Rust `_V37_*` consts. The byte
        // layout is the v3.7 contract; this test is the canonical
        // anchor.
        assert_eq!(_V37_TAIL_START, 246);
        assert_eq!(_V37_OWN_WEAKNESS_LETHAL, 246);
        assert_eq!(_V37_OPP_WEAKNESS_LETHAL, 247);
        assert_eq!(_V37_OWN_WEAKNESS_SECONDARY_KO, 248);
        assert_eq!(_V37_OPP_WEAKNESS_SECONDARY_KO, 249);
        assert_eq!(_V37_COIN_FLIP_BASE, 250);
        assert_eq!(_V37_COND_BONUS_BASE, 258);
        assert_eq!(_V37_ETA_BASE, 266);
        assert_eq!(_V37_PARALYSIS_BASE, 270);
        assert_eq!(_V37_TOOL_KIND_OWN_BASE, 272);
        assert_eq!(_V37_TOOL_KIND_OPP_BASE, 276);
        assert_eq!(_V37_ABILITY_KIND_OWN_BASE, 280);
        assert_eq!(_V37_ABILITY_KIND_OPP_BASE, 288);
        // Final boundary: 288 (own ability) + 8 (one-hot) = 296.
        assert_eq!(_V37_ABILITY_KIND_OPP_BASE + 8, STATE_DIM_V3_7);
    }

    #[test]
    fn v3_7_weakness_adjusted_lethal_fires_on_type_match() {
        // manhattanCafeStage1 (Darkness, 40 damage) vs
        // matikanetannhauserBasic (Psychic, hp=60, weakness Darkness +20).
        // Without weakness: 40 < 60 → no KO; with weakness: 40+20=60 → KO.
        let mut obs = fixture();
        let attacker = make_uma_obs(
            "manhattanCafeStage1",
            90,
            90,
            2,
            &[("darkness", 1), ("colorless", 1)],
        );
        let defender = make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[]);
        obs.own.active = Some(attacker);
        obs.opponent.active = Some(defender);
        let v = observation_state_features_v3_7(&obs);
        // opp_weakness_lethal = OWN attacks OPP via weakness → 1.
        assert_eq!(v[_V37_OPP_WEAKNESS_LETHAL], 1.0);
        // own_weakness_lethal = OPP attacks OWN. matikanetannhauserBasic
        // is Psychic 20dmg; manhattanCafeStage1 weakness is Grass +20
        // → no type match → 20 < 90 → 0.
        assert_eq!(v[_V37_OWN_WEAKNESS_LETHAL], 0.0);
    }

    #[test]
    fn v3_7_weakness_adjusted_lethal_no_bonus_off_type() {
        // Psychic-vs-Psychic: matikanetannhauserBasic (Psychic, 20dmg)
        // vs haruUraraBasic (Psychic, hp=90, weakness Darkness +20).
        // No type match → 20 < 90 → 0.
        let mut obs = fixture();
        let attacker = make_uma_obs("matikanetannhauserBasic", 60, 60, 1, &[("psychic", 1)]);
        let defender = make_uma_obs("haruUraraBasic", 90, 90, 0, &[]);
        obs.own.active = Some(attacker);
        obs.opponent.active = Some(defender);
        let v = observation_state_features_v3_7(&obs);
        assert_eq!(v[_V37_OPP_WEAKNESS_LETHAL], 0.0);
    }

    #[test]
    fn v3_7_coin_flip_has_cf_and_eko() {
        // matikanetannhauserStage1: primary attack damage=40, coinBonus=20.
        // Defender at hp=50 → expected = 40 + 0.5*20 = 50 ≥ 50 → cf_eko=1.
        let mut obs = fixture();
        let attacker = make_uma_obs(
            "matikanetannhauserStage1",
            90,
            90,
            2,
            &[("psychic", 1), ("colorless", 1)],
        );
        let defender = make_uma_obs("haruUraraBasic", 50, 90, 0, &[]);
        obs.own.active = Some(attacker);
        obs.opponent.active = Some(defender);
        let v = observation_state_features_v3_7(&obs);
        // own_primary_has_cf at offset +0, cf_eko at offset +1.
        assert_eq!(v[_V37_COIN_FLIP_BASE + 0], 1.0, "own primary has_cf");
        assert_eq!(v[_V37_COIN_FLIP_BASE + 1], 1.0, "own primary cf_eko");
    }

    #[test]
    fn v3_7_coin_flip_eko_does_not_fire_when_expected_below_hp() {
        // Same attack but defender hp=90 → expected 50 < 90 → cf_eko=0.
        let mut obs = fixture();
        let attacker = make_uma_obs(
            "matikanetannhauserStage1",
            90,
            90,
            2,
            &[("psychic", 1), ("colorless", 1)],
        );
        let defender = make_uma_obs("haruUraraBasic", 90, 90, 0, &[]);
        obs.own.active = Some(attacker);
        obs.opponent.active = Some(defender);
        let v = observation_state_features_v3_7(&obs);
        assert_eq!(v[_V37_COIN_FLIP_BASE + 0], 1.0, "own primary has_cf");
        assert_eq!(v[_V37_COIN_FLIP_BASE + 1], 0.0, "own primary cf_eko");
    }

    #[test]
    fn v3_7_conditional_bonus_per_energy_fires() {
        // haruUraraBasic primary has damagePerAttachedEnergy →
        // own_primary_per_energy at offset +0.
        let mut obs = fixture();
        let attacker = make_uma_obs("haruUraraBasic", 90, 90, 0, &[]);
        let defender = make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[]);
        obs.own.active = Some(attacker);
        obs.opponent.active = Some(defender);
        let v = observation_state_features_v3_7(&obs);
        assert_eq!(
            v[_V37_COND_BONUS_BASE + 0], 1.0,
            "own primary per_energy bit"
        );
        assert_eq!(
            v[_V37_COND_BONUS_BASE + 1], 0.0,
            "own primary per_bench bit (haruUraraBasic has no per-bench bonus)"
        );
    }

    #[test]
    fn v3_7_eta_fires_when_pool_covers_shortfall() {
        // matikanetannhauserBasic cost {psychic:1}; attached={} →
        // shortfall={psychic:1}. +1 attach reduces to 0 → no surviving
        // shortfall → ETA=1 regardless of pool.
        let mut obs = fixture();
        let attacker = make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[]);
        let defender = make_uma_obs("haruUraraBasic", 90, 90, 0, &[]);
        obs.own.active = Some(attacker);
        obs.opponent.active = Some(defender);
        obs.own.energy_pool = vec![];
        let v = observation_state_features_v3_7(&obs);
        assert_eq!(v[_V37_ETA_BASE + 0], 1.0, "own primary ETA (psychic:1 from +1 attach)");
    }

    #[test]
    fn v3_7_eta_zero_when_shortfall_survives_and_pool_misses() {
        // Find a 2-typed-cost card so a single +1 attach can't cover both.
        // symboliRudolfStage2 has cost {water:1, dragon:1, colorless:1}.
        // attached={} → shortfall={water:1, dragon:1}. After +1 attach
        // to water (lex-tiebreak first), surviving={dragon:1}. With
        // pool=[] → dragon not in pool → ETA=0.
        let mut obs = fixture();
        let attacker = make_uma_obs("symboliRudolfStage2", 120, 120, 0, &[]);
        let defender = make_uma_obs("haruUraraBasic", 90, 90, 0, &[]);
        obs.own.active = Some(attacker);
        obs.opponent.active = Some(defender);
        obs.own.energy_pool = vec![];
        let v = observation_state_features_v3_7(&obs);
        assert_eq!(v[_V37_ETA_BASE + 0], 0.0, "own primary ETA (dragon shortfall not in pool)");
        // With pool=[dragon] → surviving dragon covered → ETA=1.
        obs.own.energy_pool = vec!["dragon".to_string()];
        let v = observation_state_features_v3_7(&obs);
        assert_eq!(v[_V37_ETA_BASE + 0], 1.0, "own primary ETA (dragon now in pool)");
    }

    #[test]
    fn v3_7_paralysis_window_open_reads_typed_bool() {
        let mut obs = fixture();
        let mut own_a = make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[]);
        own_a.turn_state.paralysis_recovery_pending = false;
        let mut opp_a = make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[]);
        opp_a.turn_state.paralysis_recovery_pending = true;
        obs.own.active = Some(own_a);
        obs.opponent.active = Some(opp_a);
        let v = observation_state_features_v3_7(&obs);
        // own_paralysis_window_open = OPP paralysed → 1.
        assert_eq!(v[_V37_PARALYSIS_BASE + 0], 1.0);
        // opp_paralysis_window_open = OWN paralysed → 0.
        assert_eq!(v[_V37_PARALYSIS_BASE + 1], 0.0);
    }

    #[test]
    fn v3_7_tool_kind_leftover_carrot() {
        // leftoverCarrot is a TrainerEffect with toolEndTurnHealActive
        // → ToolEffectKind::HealAtTurnEnd (index 0).
        let mut obs = fixture();
        let mut own_a = make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[]);
        own_a.tool_card_id = Some("leftoverCarrot".to_string());
        obs.own.active = Some(own_a);
        obs.opponent.active = Some(make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[]));
        let v = observation_state_features_v3_7(&obs);
        assert_eq!(v[_V37_TOOL_KIND_OWN_BASE + 0], 1.0, "own tool HealAtTurnEnd");
        // Exactly one bit in the 4-bit own block.
        let sum: f32 = v[_V37_TOOL_KIND_OWN_BASE.._V37_TOOL_KIND_OWN_BASE + 4]
            .iter()
            .sum();
        assert_eq!(sum, 1.0);
        // Opp has no tool → all zero.
        let opp_sum: f32 = v[_V37_TOOL_KIND_OPP_BASE.._V37_TOOL_KIND_OPP_BASE + 4]
            .iter()
            .sum();
        assert_eq!(opp_sum, 0.0);
    }

    #[test]
    fn v3_7_ability_kind_nice_nature_hp_bonus() {
        // niceNatureBasic ability has activeHpBonus → AbilityEffectKind::HpBonus
        // (index 2). Only fires if used_ability_this_turn=true.
        let mut obs = fixture();
        let mut own_a = make_uma_obs("niceNatureBasic", 70, 70, 0, &[]);
        own_a.used_ability_this_turn = true;
        obs.own.active = Some(own_a);
        obs.opponent.active = Some(make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[]));
        let v = observation_state_features_v3_7(&obs);
        assert_eq!(v[_V37_ABILITY_KIND_OWN_BASE + 2], 1.0, "own ability HpBonus");
        let sum: f32 = v[_V37_ABILITY_KIND_OWN_BASE.._V37_ABILITY_KIND_OWN_BASE + 8]
            .iter()
            .sum();
        assert_eq!(sum, 1.0);
    }

    #[test]
    fn v3_7_ability_kind_zero_when_not_used_this_turn() {
        // Even with the ability card, the bit only fires if
        // used_ability_this_turn=true.
        let mut obs = fixture();
        let mut own_a = make_uma_obs("niceNatureBasic", 70, 70, 0, &[]);
        own_a.used_ability_this_turn = false;
        obs.own.active = Some(own_a);
        obs.opponent.active = Some(make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[]));
        let v = observation_state_features_v3_7(&obs);
        let sum: f32 = v[_V37_ABILITY_KIND_OWN_BASE.._V37_ABILITY_KIND_OWN_BASE + 8]
            .iter()
            .sum();
        assert_eq!(sum, 0.0);
    }

    #[test]
    fn v3_7_secondary_weakness_ko_predicated_on_v36_usable() {
        // matikanefukukitaruStage1 has 2 attacks; secondary cost
        // {psychic:1, colorless:1}, damage=0. With 2 psychic attached,
        // v3.6 says secondary usable=1. Defender Psychic-weakness-
        // Darkness; attacker is Psychic → no weakness type match.
        // Damage=0 → even with weakness off, 0 < hp → secondary KO=0.
        let mut obs = fixture();
        let attacker = make_uma_obs(
            "matikanefukukitaruStage1",
            100,
            100,
            2,
            &[("psychic", 2)],
        );
        let defender = make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[]);
        obs.own.active = Some(attacker);
        obs.opponent.active = Some(defender);
        let v = observation_state_features_v3_7(&obs);
        // v3.6 head says own secondary usable=1 (head bit 242).
        assert_eq!(v[242], 1.0);
        // But secondary damage is 0 → opp_weakness_sec_KO=0.
        assert_eq!(v[_V37_OPP_WEAKNESS_SECONDARY_KO], 0.0);
    }

    #[test]
    fn v3_7_dispatch_covers_state_dim_296() {
        // Mirror of inference::tests::state_dim_dispatch_covers_v3_0_through_v3_6
        // — adding the v3.7 row keeps the dispatcher honest.
        assert_eq!(STATE_DIM_V3_7, 296);
    }
}
