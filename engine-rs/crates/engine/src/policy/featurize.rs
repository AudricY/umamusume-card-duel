//! R16-P3 throughput-spike Option A: in-process Rust port of the v3.0
//! observation/action featurizer.
//!
//! Bit-exact port of the **v3.0** Python builders in
//! `training/uma_ai/features.py`:
//!   - `observation_to_features`         → `observation_state_features`
//!   - `legal_actions_to_features`       → `legal_actions_features`
//!   - `observation_to_card_ids`         → `observation_card_ids_by_zone`
//!   - `action_card_idx_pair`            → `action_card_idx_pair`
//!   - `card_vocab_metadata` / `card_vocab_index` → reused from
//!     `crate::policy::card_vocab`.
//!
//! Pinned schemas
//! --------------
//! v3.0 ONLY in this slice. v3.1 (164-d temporal/turn-state) and v3.2
//! (110-d head + per-Uma slot tensors) are explicit-unimplemented panics
//! at the `inference::InferenceSession::load` graph-signature gate so a
//! parity smoke against a v3.2 model fails loud rather than silently
//! mis-routing tensors. Follow-up slice covers v3.2 once parity is
//! confirmed.
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
use crate::policy::card_vocab::{card_vocab, card_vocab_index};
use crate::policy::types::{AiPhase, LegalAiAction, PublicObservation, PublicSideObservation, PublicUmaObservation};

/// Mirrors `STATE_DIM_V3` / `STATE_DIM` in Python (frozen 110-d v3.0).
pub const STATE_DIM_V3: usize = 110;
/// Mirrors `ACTION_DIM` (48-d action feature vector — pre-computed
/// TS-side and carried verbatim on `LegalAiAction.features`).
pub const ACTION_DIM: usize = 48;
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
    let damage = r[2] * 150.0;
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
}
