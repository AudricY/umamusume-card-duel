from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

# R7.b.1 hygiene wire-unused-fields: schema v2.1 additive bump.
# Adds 14 scalar slots [96:110] surfacing three JSON fields the encoder
# previously ignored (see scoping doc § 3 "free signals" + § 6 step 1):
#   - slot 96: firstPlayer polarity vs sideToAct (own=+1 / opp=-1).
#   - slots 97-99: pendingChoiceKind one-hot (none / promoteAfterKnockout /
#     switchAfterGust). Hard-coded order in PENDING_CHOICE_KINDS so future
#     unions append without shifting existing slot meanings.
#   - slots 100-109: per-uma toolCardId hashed-float (own active, own bench
#     1-4, opp active, opp bench 1-4). Matches the 10-uma accounting in
#     `_identity_features` (slots 32-41). Empty / null → 0.0.
# Additive only. Existing slot meanings unchanged. Existing R7/R15.S1 trace
# JSONLs re-extract under v2.1 without resimulating (R7.b.0 verdict YES).
STATE_DIM = 110
ACTION_DIM = 48
# R7.b.2 Phase 2 bumps STATE_FEATURE_SCHEMA_VERSION 2.1 → 3.0 because the
# Python encoder now consumes the new TS-side fields landed in Phase 1
# (`PublicObservation.cardIdsByZone`, per-action `actionSourceCardIdx` /
# `actionTargetCardIdx`). STATE_DIM is unchanged — the embedding pass is
# ADDITIVE over the 110-d hygiene encoding (see model.py header for the
# additive-vs-replace rationale); Phase 6 will decide whether the 16
# identity-hash slots (32–47) can be retired in a future bump.
STATE_FEATURE_SCHEMA_VERSION = 3.0
ACTION_FEATURE_SCHEMA_VERSION = 2

# --- Serving-schema 96/110/164 freeze contract (r16 P1 prerequisite) -------
# `serve_onnx` resolves the feature builder from the loaded ONNX graph's
# `state_features` last dim. These named dim constants are the *contract*
# that resolution keys off. The frozen 110-d v3.0 builder
# (`observation_to_features`) and the frozen 96-d v2 builder
# (`observation_to_features_v2`) MUST keep emitting exactly these widths so
# that introducing a future STATE_DIM=164 v3.1 schema (P1 temporal/turn-state
# features) is an *additive new schema*, never an edit that shifts v2/v3
# slots. The module-load assertion below makes a stale constant a hard import
# failure, not a silent serving corruption.
#
# STATE_DIM is intentionally an alias of STATE_DIM_V3 (not the reverse) so a
# future P1 bump adds STATE_DIM_V3_1 = 164 as a NEW constant and a NEW builder
# rather than mutating STATE_DIM_V3 / `observation_to_features`.
STATE_DIM_V3 = 110  # frozen 110-d v3.0 builder (`observation_to_features`)
STATE_FEATURE_SCHEMA_VERSION_V3 = 3.0
# Disabled placeholder for the P1 STATE_DIM=164 v3.1 temporal/turn-state
# schema. Declared here so the serving guard's schema table has a named slot
# to reject against today (it maps to no builder until P1 lands). Implement
# NO 164-d feature logic against this — it is a contract anchor only.
STATE_DIM_V3_1 = 164  # placeholder; P1 work, intentionally unimplemented
assert STATE_DIM == STATE_DIM_V3, (
    f"STATE_DIM ({STATE_DIM}) must equal the frozen v3.0 dim "
    f"STATE_DIM_V3 ({STATE_DIM_V3}). The 110-d v3.0 builder is frozen for "
    f"serving-schema resolution; a 164-d v3.1 schema must be ADDITIVE "
    f"(new STATE_DIM_V3_1 + new builder), not an edit to STATE_DIM."
)

# R7.b.2 Phase 2: per-zone pad widths consumed by `observation_to_card_ids`
# AND the dataset collator (so packed `LongTensor[B, 8, max_cards_per_zone]`
# has stable per-zone slots) AND any future ONNX exporter (Phase 3) so the
# graph shape is fixed across producers. ORDER is load-bearing — it matches
# the 8-zone enumeration in `frontend/src/game/engine/ai-policy/types.ts`
# `ZoneKey` and the per-zone concat order in `CandidatePolicyNet.forward`.
# Caps chosen per scoping § 11 Phase 2 (active/stadium single, bench up to
# 4, hand up to 10, discards up to 30 for headroom).
CARD_ID_SHAPES: dict[str, int] = {
    "ownActive": 1,
    "oppActive": 1,
    "ownBench": 4,
    "oppBench": 4,
    "ownHand": 10,
    "ownDiscard": 30,
    "oppDiscard": 30,
    "stadium": 1,
}
ZONE_ORDER: tuple[str, ...] = tuple(CARD_ID_SHAPES.keys())

PHASES = [
    "setup",
    "pendingChoice",
    "bench",
    "trainerBefore",
    "evolve",
    "attach",
    "trainerAfter",
    "ability",
    "combat",
    "stadiumOrEnd",
]

SIDES = ["player", "opponent"]

# Hard-coded ordering so adding a new union arm doesn't shift v2.1 slot
# meanings. Mirrors `PublicObservation.pendingChoiceKind` union in
# `frontend/src/game/engine/ai-policy/types.ts`.
PENDING_CHOICE_KINDS = ["promoteAfterKnockout", "switchAfterGust"]


FeatureAblation = str


def observation_to_features(observation: dict[str, Any], ablations: set[FeatureAblation] | None = None) -> np.ndarray:
    features = np.zeros(STATE_DIM, dtype=np.float32)
    phase = observation.get("phase", "stadiumOrEnd")
    side = observation.get("sideToAct", "player")
    own = observation.get("own", {})
    opponent = observation.get("opponent", {})
    shared = observation.get("shared", {})

    features[0] = PHASES.index(phase) / max(1, len(PHASES) - 1) if phase in PHASES else 0.0
    features[1] = SIDES.index(side) if side in SIDES else 0.0
    features[2] = float(observation.get("turnNumber", 0)) / 20.0
    features[3] = float(own.get("points", 0)) / 3.0
    features[4] = float(opponent.get("points", 0)) / 3.0
    features[5] = float(own.get("handCount", 0)) / 10.0
    features[6] = float(opponent.get("handCount", 0)) / 10.0
    features[7] = float(own.get("deckCount", 0)) / 50.0
    features[8] = float(opponent.get("deckCount", 0)) / 50.0
    features[9] = 1.0 if shared.get("stadiumCardId") else 0.0
    features[10:18] = _side_board_features(own)
    features[18:26] = _side_board_features(opponent)
    features[26] = float(len(own.get("discard", []))) / 50.0
    features[27] = float(len(opponent.get("discard", []))) / 50.0
    features[28] = float(len(own.get("energyZone", []))) / 4.0
    features[29] = 1.0 if own.get("usedSupporterThisTurn") else 0.0
    features[30] = 1.0 if own.get("usedRetreatThisTurn") else 0.0
    features[31] = 1.0 if own.get("usedStadiumThisTurn") else 0.0
    features[32:48] = _identity_features(own, opponent)
    features[48:58] = _energy_vector((own.get("active") or {}).get("energies", {}))
    features[58:68] = _energy_vector((opponent.get("active") or {}).get("energies", {}))
    features[68:96] = _card_awareness_features(own, opponent, shared)
    # R7.b.1 hygiene additive slots — see schema v2.1 header note.
    features[96] = _first_player_polarity(observation, side)
    features[97:100] = _pending_choice_one_hot(observation.get("pendingChoiceKind"))
    features[100:110] = _tool_card_features(own, opponent)
    apply_state_ablations(features, ablations or set())
    # FROZEN v3.0 contract: this builder MUST emit exactly STATE_DIM_V3 (110)
    # slots. If a future change shifts the layout the serving guard would
    # silently pair a 110-d graph with a wrong-width vector — fail here first.
    assert features.shape == (STATE_DIM_V3,), (
        f"observation_to_features (frozen v3.0) emitted {features.shape}, "
        f"expected ({STATE_DIM_V3},). The 110-d v3.0 slot layout is frozen."
    )
    return features


# ---------------------------------------------------------------------------
# FROZEN schema-v2 state builder (vendored verbatim from commit bc6db85).
#
# This is the production serving encoding used by the 96-dim model
# `runs/R13-W6-phase-d/iter-2/policy.onnx` (schema v2). It is pinned here so
# HEAD (which trains/serves the additive 110-d v2.1/v3 encoding above) can
# still serve the 96-d production artifact byte-for-byte.
#
# DO NOT MODIFY. Slots 0–95 only. Mechanically equal to bc6db85's
# `observation_to_features` body (bc6db85 features.py lines ~33–65), reusing
# HEAD's shared helpers (`_side_board_features`, `_identity_features`,
# `_energy_vector`, `_card_awareness_features`) which are byte-identical
# between bc6db85 and HEAD (verified). It deliberately omits the v2.1 hygiene
# slots [96:110] and the v3 embedding inputs.
#
# "Promote later" path: load a 110-d model and serve it under
# `serve_onnx --feature-schema v3` (or `auto`, which selects v3 from the ONNX
# graph signature). Switching the production model is the ONLY supported way
# to graduate off this frozen builder — never edit slots 0–95 here.
STATE_DIM_V2 = 96
STATE_FEATURE_SCHEMA_VERSION_V2 = 2


def observation_to_features_v2(
    observation: dict[str, Any], ablations: set[FeatureAblation] | None = None
) -> np.ndarray:
    """FROZEN bc6db85 schema-v2 builder. Slots 0–95 only. Do not modify.

    Serving (the only caller) passes no ablations. To keep this correct on a
    96-len array we only apply ablations that are meaningful to slots 0–95;
    the v2.1 `state_hygiene_v21` branch (which indexes [96:110]) is skipped
    here by construction because we never pass it the v2.1 ablation.
    """
    features = np.zeros(STATE_DIM_V2, dtype=np.float32)
    phase = observation.get("phase", "stadiumOrEnd")
    side = observation.get("sideToAct", "player")
    own = observation.get("own", {})
    opponent = observation.get("opponent", {})
    shared = observation.get("shared", {})

    features[0] = PHASES.index(phase) / max(1, len(PHASES) - 1) if phase in PHASES else 0.0
    features[1] = SIDES.index(side) if side in SIDES else 0.0
    features[2] = float(observation.get("turnNumber", 0)) / 20.0
    features[3] = float(own.get("points", 0)) / 3.0
    features[4] = float(opponent.get("points", 0)) / 3.0
    features[5] = float(own.get("handCount", 0)) / 10.0
    features[6] = float(opponent.get("handCount", 0)) / 10.0
    features[7] = float(own.get("deckCount", 0)) / 50.0
    features[8] = float(opponent.get("deckCount", 0)) / 50.0
    features[9] = 1.0 if shared.get("stadiumCardId") else 0.0
    features[10:18] = _side_board_features(own)
    features[18:26] = _side_board_features(opponent)
    features[26] = float(len(own.get("discard", []))) / 50.0
    features[27] = float(len(opponent.get("discard", []))) / 50.0
    features[28] = float(len(own.get("energyZone", []))) / 4.0
    features[29] = 1.0 if own.get("usedSupporterThisTurn") else 0.0
    features[30] = 1.0 if own.get("usedRetreatThisTurn") else 0.0
    features[31] = 1.0 if own.get("usedStadiumThisTurn") else 0.0
    features[32:48] = _identity_features(own, opponent)
    features[48:58] = _energy_vector((own.get("active") or {}).get("energies", {}))
    features[58:68] = _energy_vector((opponent.get("active") or {}).get("energies", {}))
    features[68:96] = _card_awareness_features(own, opponent, shared)
    # FROZEN: only apply ablations meaningful to slots 0–95. The v2.1
    # `state_hygiene_v21` branch indexes [96:110] and would raise on this
    # 96-len array; serving never passes ablations so this is a no-op there.
    v2_safe = {a for a in (ablations or set()) if a != "state_hygiene_v21"}
    if v2_safe:
        apply_state_ablations(features, v2_safe)
    # FROZEN v2 contract: this builder MUST emit exactly STATE_DIM_V2 (96)
    # slots — it is the production-pinned encoding. Same rationale as the
    # v3.0 assertion above.
    assert features.shape == (STATE_DIM_V2,), (
        f"observation_to_features_v2 (frozen v2) emitted {features.shape}, "
        f"expected ({STATE_DIM_V2},). The 96-d v2 slot layout is frozen."
    )
    return features


def legal_actions_to_features(actions: list[dict[str, Any]], ablations: set[FeatureAblation] | None = None) -> np.ndarray:
    rows = []
    for action in actions:
        raw = action.get("features", [])
        if len(raw) != ACTION_DIM:
            action_id = action.get("id", "<unknown>")
            raise ValueError(f"Action feature length mismatch for {action_id}: got {len(raw)}, expected {ACTION_DIM}")
        row = np.zeros(ACTION_DIM, dtype=np.float32)
        row[:] = np.asarray(raw, dtype=np.float32)
        apply_action_ablations(row, ablations or set())
        rows.append(row)
    if not rows:
        return np.zeros((0, ACTION_DIM), dtype=np.float32)
    return np.stack(rows, axis=0)


def observation_to_card_ids(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    """R7.b.2 Phase 2: extract per-zone packed card-vocab indices.

    Reads the new TS-side `cardIdsByZone` field on `PublicObservation`
    (landed Phase 1, see `frontend/src/game/engine/ai-policy/types.ts:62`).
    Returns a `{zone: np.ndarray(int64)}` dict whose array shape matches
    `CARD_ID_SHAPES[zone]`; entries beyond the variable-length TS list are
    zero-padded (index 0 doubles as `unknownIndex` AND the embedding
    `padding_idx`, per `shared/src/cardVocab.json:3-4`). Entries that exceed
    the cap are clipped silently — `WARNING` logs would spam under heavy
    discards; the cap of 30 already covers 95th-percentile observed sizes.

    Fail-loud guard: if the observation lacks `cardIdsByZone`, raise
    `ValueError`. This rejects pre-Phase-1 trace JSONLs at load time so the
    embedding pass never silently zero-tensors a corpus that should have
    been re-extracted by Phase 4. Older corpora must run through
    `r7b2_extract_features.py` (Phase 4, separate slot) to gain the field.
    """

    if "cardIdsByZone" not in observation:
        raise ValueError(
            "observation_to_card_ids: missing 'cardIdsByZone' — this row predates R7.b.2 Phase 1 "
            "TS schema bump. Re-extract via Phase 4 (`r7b2_extract_features.py`) before training. "
            "The Phase 2 embedding pass refuses to silently zero-tensor pre-v3 corpora."
        )
    raw = observation.get("cardIdsByZone") or {}
    out: dict[str, np.ndarray] = {}
    for zone, width in CARD_ID_SHAPES.items():
        arr = np.zeros((width,), dtype=np.int64)
        ids = raw.get(zone) or []
        if not isinstance(ids, list):
            raise ValueError(
                f"observation_to_card_ids: cardIdsByZone[{zone!r}] must be a list, got {type(ids).__name__}"
            )
        # Clip to the per-zone cap; pre-Phase-2 reextractors are expected to
        # already respect these widths, but TS-side is variable-length so we
        # cannot rely on it.
        for slot, value in enumerate(ids[:width]):
            try:
                idx = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"observation_to_card_ids: cardIdsByZone[{zone!r}][{slot}] must be int-coercible; got {value!r}"
                ) from exc
            arr[slot] = idx
        out[zone] = arr
    return out


def action_card_idx_pair(action: dict[str, Any]) -> np.ndarray:
    """R7.b.2 Phase 2: pull the (source, target) card-vocab idx pair off a
    LegalAiAction. Phase 1 added these fields on the TS side
    (`frontend/src/game/engine/ai-policy/types.ts:38-39`); `null` means
    "no clear source/target" → maps to 0 (the shared embedding pad/unknown).

    Pre-Phase-1 actions lack the fields; fall back to 0/0 with NO error here
    because the row-level missing-cardIdsByZone guard in
    `observation_to_card_ids` is the canonical fail point — duplicating it
    per-action would obscure the error.
    """

    src = action.get("actionSourceCardIdx")
    tgt = action.get("actionTargetCardIdx")
    return np.asarray(
        [0 if src is None else int(src), 0 if tgt is None else int(tgt)],
        dtype=np.int64,
    )


def apply_state_ablations(features: np.ndarray, ablations: set[FeatureAblation]) -> None:
    if "state_identity_hashes" in ablations:
        features[32:48] = 0
    if "state_energy_vectors" in ablations:
        features[48:68] = 0
    if "state_card_awareness" in ablations:
        features[68:96] = 0
    if "state_semantic" in ablations:
        features[10:32] = 0
        features[48:96] = 0
    # R7.b.1 v2.1 ablation hook: zero the additive hygiene slots so callers
    # can isolate the contribution of `firstPlayer` / `pendingChoiceKind` /
    # per-uma `toolCardId` against the legacy 96-d encoding without
    # touching slot meanings elsewhere.
    if "state_hygiene_v21" in ablations:
        features[96:110] = 0


def apply_action_ablations(features: np.ndarray, ablations: set[FeatureAblation]) -> None:
    if "action_source_target" in ablations:
        features[3:10] = 0
        features[13:18] = 0
        features[25:29] = 0
        features[37:48] = 0
    if "action_trainer_semantic" in ablations:
        features[18:25] = 0
        features[32:37] = 0
        features[42:46] = 0
    if "action_tactical" in ablations:
        features[10:12] = 0
        features[29:32] = 0
        features[46:48] = 0
    if "action_semantic" in ablations:
        features[12:48] = 0


def _side_board_features(side: dict[str, Any]) -> np.ndarray:
    active = side.get("active")
    bench = [entry for entry in side.get("bench", []) if entry]
    board = [active] if active else []
    board.extend(bench)
    hp_total = sum(float(entry.get("hp", 0)) for entry in board)
    max_hp_total = max(1.0, sum(float(entry.get("maxHp", 0)) for entry in board))
    energy_total = sum(float(entry.get("energyTotal", 0)) for entry in board)
    stage_total = sum(float(entry.get("stage", 0)) for entry in board)
    damaged = sum(1.0 for entry in board if float(entry.get("hp", 0)) < float(entry.get("maxHp", 0)))
    statuses = sum(float(len(entry.get("specialConditions", []))) for entry in board)
    active_hp_ratio = float(active.get("hp", 0)) / max(1.0, float(active.get("maxHp", 0))) if active else 0.0
    active_energy = float(active.get("energyTotal", 0)) / 6.0 if active else 0.0
    return np.asarray(
        [
            active_hp_ratio,
            active_energy,
            float(len(board)) / 4.0,
            hp_total / max_hp_total,
            energy_total / 12.0,
            stage_total / 8.0,
            damaged / 4.0,
            statuses / 4.0,
        ],
        dtype=np.float32,
    )


def _identity_features(own: dict[str, Any], opponent: dict[str, Any]) -> np.ndarray:
    values = np.zeros(16, dtype=np.float32)
    own_active = own.get("active") or {}
    opponent_active = opponent.get("active") or {}
    values[0] = _hash_to_unit(str(own_active.get("cardId", "")))
    values[1] = _hash_to_unit(str(opponent_active.get("cardId", "")))
    for offset, entry in enumerate((own.get("bench") or [])[:4], start=2):
        values[offset] = _hash_to_unit(str((entry or {}).get("cardId", "")))
    for offset, entry in enumerate((opponent.get("bench") or [])[:4], start=6):
        values[offset] = _hash_to_unit(str((entry or {}).get("cardId", "")))
    values[10] = _hash_average(own.get("handCardIds") or [])
    values[11] = _hash_average(own.get("discard") or [])
    values[12] = _hash_average(opponent.get("discard") or [])
    values[13] = float(len(own.get("handCardIds") or [])) / 10.0
    values[14] = float(len(own.get("bench") or [])) / 4.0
    values[15] = float(len(opponent.get("bench") or [])) / 4.0
    return values


def _first_player_polarity(observation: dict[str, Any], side_to_act: str) -> float:
    """Encode first-player as own/opp polarity vs the side currently acting.

    `+1` when the side currently acting also went first this game; `-1` when
    the acting side is the second player. Polarity matches the +1/-1
    convention used by the side-of-active scalar at slot 1 (which encodes the
    SIDES index 0/1 directly, kept unchanged to avoid disturbing legacy slot
    meanings). Returns 0.0 if the field is absent (legacy rows pre-dating the
    field — no v2.1 trace has been seen without it, but be permissive).
    """

    first = observation.get("firstPlayer")
    if first not in SIDES or side_to_act not in SIDES:
        return 0.0
    return 1.0 if first == side_to_act else -1.0


def _pending_choice_one_hot(kind: Any) -> np.ndarray:
    """Stable one-hot over the union in `PublicObservation.pendingChoiceKind`.

    Layout: [none, promoteAfterKnockout, switchAfterGust]. Order is hard-coded
    in `PENDING_CHOICE_KINDS` so new union arms append (lifting STATE_DIM)
    rather than shifting existing positions.
    """

    values = np.zeros(1 + len(PENDING_CHOICE_KINDS), dtype=np.float32)
    if kind is None or not isinstance(kind, str):
        values[0] = 1.0
        return values
    if kind in PENDING_CHOICE_KINDS:
        values[1 + PENDING_CHOICE_KINDS.index(kind)] = 1.0
    else:
        # Unknown kind (new union arm in TS not yet mirrored here) → fall back
        # to the "none" bucket so we never silently corrupt one of the named
        # slots. Bump `PENDING_CHOICE_KINDS` + STATE_DIM when this fires.
        values[0] = 1.0
    return values


def _tool_card_features(own: dict[str, Any], opponent: dict[str, Any]) -> np.ndarray:
    """Per-uma `toolCardId` hashed-float; 10 slots in fixed positions.

    Layout matches the 10-uma accounting in `_identity_features`:
        [own.active, own.bench[0..3], opp.active, opp.bench[0..3]]
    Uses the same `_hash_to_unit` mapping (vocab-indexed when available) so
    a tool card's float position is stable across runs and shares the vocab
    hash recorded on the checkpoint. Empty / null → 0.0.
    """

    values = np.zeros(10, dtype=np.float32)
    own_active = own.get("active") or {}
    opp_active = opponent.get("active") or {}
    values[0] = _hash_to_unit(str(own_active.get("toolCardId") or ""))
    for offset, entry in enumerate((own.get("bench") or [])[:4], start=1):
        values[offset] = _hash_to_unit(str((entry or {}).get("toolCardId") or ""))
    values[5] = _hash_to_unit(str(opp_active.get("toolCardId") or ""))
    for offset, entry in enumerate((opponent.get("bench") or [])[:4], start=6):
        values[offset] = _hash_to_unit(str((entry or {}).get("toolCardId") or ""))
    return values


def _energy_vector(energies: dict[str, Any]) -> np.ndarray:
    types = ["grass", "fire", "water", "lightning", "psychic", "fighting", "darkness", "steel", "colorless", "dragon"]
    return np.asarray([float(energies.get(energy_type, 0)) / 4.0 for energy_type in types], dtype=np.float32)


def _card_awareness_features(own: dict[str, Any], opponent: dict[str, Any], shared: dict[str, Any]) -> np.ndarray:
    values = np.zeros(28, dtype=np.float32)
    hand_ids = own.get("handCardIds") or []
    discard_ids = own.get("discard") or []
    own_board = _board_entries(own)
    opponent_board = _board_entries(opponent)
    own_active = own.get("active") or {}
    opponent_active = opponent.get("active") or {}

    values[0:9] = _hand_role_features(hand_ids)
    values[9] = _matching_evolution_count(hand_ids, own_board) / 4.0
    values[10:14] = _uma_readiness_features(own_active)
    values[14] = _ready_attacker_count(own_board) / 4.0
    values[15] = _ability_ready_count(own_board) / 4.0
    values[16:19] = _discard_role_features(discard_ids)
    values[19:23] = _uma_readiness_features(opponent_active)
    values[23] = _can_ko(opponent_active, own_active)
    values[24] = _can_ko(own_active, opponent_active)
    values[25] = _next_energy_matches_active_need(own)
    values[26] = _hash_to_unit(str(shared.get("stadiumCardId", "")))
    values[27] = _ready_attacker_count(opponent_board) / 4.0
    return values


def _hand_role_features(card_ids: list[Any]) -> np.ndarray:
    values = np.zeros(9, dtype=np.float32)
    if not card_ids:
        return values
    attack_damage_total = 0.0
    attack_cards = 0
    for raw_id in card_ids:
        card = _get_card(str(raw_id))
        if not card:
            continue
        if card.get("kind") == "umamusume":
            stage = float(card.get("stage", 0))
            values[0 if stage <= 0 else 1] += 1
            attack = _primary_attack(card)
            attack_damage_total += float(attack.get("damage", 0))
            attack_cards += 1
            if card.get("ability"):
                values[8] += 1
            continue
        if card.get("kind") == "trainer":
            effect = card.get("effect") or {}
            values[2] += 1
            if effect.get("draw") or effect.get("shuffleHandIntoDeckDraw"):
                values[3] += 1
            if effect.get("searchUmamusume") or effect.get("searchEvolutionUmamusume") or effect.get("searchRandomBasicUmamusume"):
                values[4] += 1
            if effect.get("extraEnergyAttach") or effect.get("attachEnergyFromZoneToBench"):
                values[5] += 1
            if effect.get("heal") or effect.get("recoverActiveSpecialConditions"):
                values[6] += 1
            if effect.get("gustOpponent") or effect.get("discardRandomOpponentActiveEnergy") or card.get("trainerType") == "tool":
                values[7] += 1
    values[:8] /= 10.0
    values[8] = (attack_damage_total / max(1, attack_cards)) / 120.0
    return values


def _discard_role_features(card_ids: list[Any]) -> np.ndarray:
    values = np.zeros(3, dtype=np.float32)
    for raw_id in card_ids:
        card = _get_card(str(raw_id))
        if not card:
            continue
        if card.get("kind") == "trainer":
            values[0] += 1
        elif float(card.get("stage", 0)) <= 0:
            values[1] += 1
        else:
            values[2] += 1
    return np.asarray([values[0] / 20.0, values[1] / 10.0, values[2] / 10.0], dtype=np.float32)


def _uma_readiness_features(entry: dict[str, Any]) -> np.ndarray:
    values = np.zeros(4, dtype=np.float32)
    card = _get_card(str(entry.get("cardId", "")))
    if not card or card.get("kind") != "umamusume":
        return values
    attack = _primary_attack(card)
    cost = attack.get("cost") or {}
    energies = entry.get("energies") or {}
    typed_deficit = _typed_energy_deficit(energies, cost)
    total_cost = _total_cost(cost)
    energy_total = float(entry.get("energyTotal", 0))
    values[0] = min(1.5, energy_total / max(1.0, total_cost))
    values[1] = typed_deficit / 4.0
    values[2] = float(attack.get("damage", 0)) / 150.0
    values[3] = 1.0 if typed_deficit <= 0 and energy_total >= total_cost else 0.0
    return values


def _matching_evolution_count(hand_ids: list[Any], board: list[dict[str, Any]]) -> float:
    species_in_play = {str(entry.get("species", "")) for entry in board if entry}
    count = 0.0
    for raw_id in hand_ids:
        card = _get_card(str(raw_id))
        if card and card.get("kind") == "umamusume" and float(card.get("stage", 0)) > 0 and str(card.get("species", "")) in species_in_play:
            count += 1.0
    return count


def _ready_attacker_count(board: list[dict[str, Any]]) -> float:
    return sum(float(_uma_readiness_features(entry)[3] > 0) for entry in board)


def _ability_ready_count(board: list[dict[str, Any]]) -> float:
    count = 0.0
    for entry in board:
        card = _get_card(str(entry.get("cardId", "")))
        if card and card.get("ability") and not entry.get("usedAbilityThisTurn"):
            count += 1.0
    return count


def _can_ko(attacker: dict[str, Any], defender: dict[str, Any]) -> float:
    if not attacker or not defender:
        return 0.0
    readiness = _uma_readiness_features(attacker)
    if readiness[3] <= 0:
        return 0.0
    damage = readiness[2] * 150.0
    return 1.0 if damage >= float(defender.get("hp", 0)) else 0.0


def _next_energy_matches_active_need(side: dict[str, Any]) -> float:
    zone = side.get("energyZone") or []
    active = side.get("active") or {}
    if not zone or not active:
        return 0.0
    card = _get_card(str(active.get("cardId", "")))
    if not card:
        return 0.0
    attack = _primary_attack(card)
    energy_type = str(zone[0])
    required = float((attack.get("cost") or {}).get(energy_type, 0))
    attached = float((active.get("energies") or {}).get(energy_type, 0))
    return 1.0 if required > attached else 0.0


def _board_entries(side: dict[str, Any]) -> list[dict[str, Any]]:
    board = []
    active = side.get("active")
    if active:
        board.append(active)
    board.extend(entry for entry in side.get("bench", []) if entry)
    return board


def _typed_energy_deficit(energies: dict[str, Any], cost: dict[str, Any]) -> float:
    deficit = 0.0
    for energy_type, amount in cost.items():
        if energy_type == "colorless":
            continue
        deficit += max(0.0, float(amount or 0) - float(energies.get(energy_type, 0)))
    return deficit


def _total_cost(cost: dict[str, Any]) -> float:
    return float(sum(float(amount or 0) for amount in cost.values()))


def _primary_attack(card: dict[str, Any]) -> dict[str, Any]:
    attacks = card.get("attacks") or []
    return attacks[0] if attacks else {}


@lru_cache(maxsize=1)
def _card_catalog() -> dict[str, dict[str, Any]]:
    path = Path(__file__).resolve().parents[2] / "shared" / "src" / "data" / "cards.json"
    raw = json.loads(path.read_text(encoding="utf8"))
    return raw.get("baseCards", {})


def _get_card(card_id: str) -> dict[str, Any] | None:
    cards = _card_catalog()
    if card_id in cards:
        return cards[card_id]
    for suffix in ("FullArtGold", "FullArt", "UncommonPlus"):
        if card_id.endswith(suffix):
            base_id = card_id[: -len(suffix)]
            if base_id in cards:
                return cards[base_id]
    return None


@lru_cache(maxsize=1)
def _card_vocab() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[2] / "shared" / "src" / "cardVocab.json"
    if not path.exists():
        return {"schemaVersion": 0, "vocabSize": 0, "indexById": {}, "hash": "missing"}
    return json.loads(path.read_text(encoding="utf8"))


def card_vocab_metadata() -> dict[str, Any]:
    vocab = _card_vocab()
    return {
        "schemaVersion": vocab.get("schemaVersion", 0),
        "vocabSize": vocab.get("vocabSize", 0),
        "hash": vocab.get("hash", "missing"),
    }


def card_vocab_index(card_id: str) -> int:
    if not card_id:
        return 0
    vocab = _card_vocab()
    indices = vocab.get("indexById", {})
    if card_id in indices:
        return int(indices[card_id])
    for suffix in ("FullArtGold", "FullArt", "UncommonPlus", "Ex"):
        if card_id.endswith(suffix):
            base_id = card_id[: -len(suffix)]
            if base_id in indices:
                return int(indices[base_id])
    return int(vocab.get("unknownIndex", 0))


def _hash_average(items: list[Any]) -> float:
    if not items:
        return 0.0
    return float(sum(_hash_to_unit(str(item)) for item in items) / len(items))


def _hash_to_unit(text: str) -> float:
    """Deterministic [0, 1] mapping for a card id.

    Schema version 2: backed by the canonical ``shared/src/cardVocab.json``
    vocabulary so a card id's float position is stable across runs and tied
    to the recorded vocab hash. Falls back to the legacy FNV hash only when
    the vocab file is missing — mainly to keep the unit smoke usable in
    environments that ran ``npm run sim:build-card-vocab`` once.
    """

    if not text:
        return 0.0
    vocab = _card_vocab()
    size = int(vocab.get("vocabSize", 0))
    if size > 0:
        return float(card_vocab_index(text)) / float(max(1, size - 1))
    value = 2166136261
    for char in text:
        value ^= ord(char)
        value = (value * 16777619) & 0xFFFFFFFF
    return float(value) / 4294967295.0
