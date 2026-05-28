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
# v38-slim-feature-add bumped 48 → 52 (4 new cross-bit action slots at
# [48:52]). v3 (48-d) action vectors are NOT loadable under v4 — call
# sites that compare `action_features.shape[-1]` against ACTION_DIM
# (dataset loader, serve_onnx guard, export_onnx ckpt-vs-runtime gate)
# now require 52-d. v3.7 ckpts are no longer re-exportable without
# downgrading this constant; v3.7 is SOFT-SHIPPED and not in further
# rotation.
# v5-action-disambiguation bumped 52 → 57 (5 new choice-card stat slots
# at [52:57]: present flag + hp / attack damage / cost total / ability
# bit for the choice card resolved via getChoiceCardId). v4 slots [0:52]
# BYTE-STABLE. v3.8 ckpts (action schema 4) require the v5 expander
# (`make_v5_action_init.py`) to widen the action input projection 52→57
# before re-export; legacy v3.8 ONNX graphs continue to serve via the
# sidecar-aware slicing path in `engine-rs/.../inference/mod.rs`.
ACTION_DIM = 57
ACTION_DIM_V4 = 52
ACTION_DIM_V3 = 48
# R7.b.2 Phase 2 bumped STATE_FEATURE_SCHEMA_VERSION 2.1 → 3.0 (Python
# encoder consumes `cardIdsByZone` + per-action idx). R16-P1 bumps the
# latest-schema marker 3.0 → 3.1 (164-d temporal/turn-state builder). This
# constant is the LATEST available schema version; the version actually
# emitted by a run is derived from its state dim via
# `schema_version_for_state_dim` so a 110-d (v3.0) run is still tagged 3.0
# even though 3.1 exists. STATE_DIM stays an alias of STATE_DIM_V3 (110) so
# the default builder (and default manifest version) remain v3.0; v3.1 is
# opt-in by state dim.
STATE_FEATURE_SCHEMA_VERSION = 3.1
# v33-correctness-fix Fix 2-4 bumped 2 → 3. Action-vector slot 10 was
# polysemic (setup/attach/combat overloads on `amount`); slot 26 was an
# exact duplicate of slot 8; slot 28 compared `target.uid` to
# `targetSlot` (different ID spaces — near-always 0). New layout
# documented at `frontend/src/game/engine/ai-policy/actions.ts:21-32`
# and `engine-rs/crates/engine/src/policy/actions.rs:46-58`. This
# constant is metadata only — Python does not compute action features;
# it's consumed by `train_bc.py` / `train_ppo.py` manifest writers.
# v38-slim-feature-add bumped 3 → 4. v4 extends v3 with 4 new slots at
# [48:52]: swap_in_attack_ready (slot 48, retreat/retreatAttack),
# expected_damage_norm (slot 49, base+activeAttackDamageBonus+weakness,
# RESTRICTED — no conditional bonuses, no coin-flip, no discard),
# attach_color_matches_typed_need (slot 50, attachEnergy cross-bit),
# attach_completes_typed_threshold (slot 51, attachEnergy threshold-cross
# predicate). v3 slots [0:48] BYTE-STABLE. See
# `docs/ai-research/scoping/v38-slim-feature-add-scoping.md` §4.5.
# v5-action-disambiguation bumped 4 → 5. v5 extends v4 with 5 new
# choice-card stat slots at [52:57]: choice_card_present (52),
# choice_card_hp_norm (53), choice_card_attack_damage_norm (54),
# choice_card_attack_cost_total_norm (55), choice_card_has_ability (56).
# All fire iff getChoiceCardId(side, choices) resolves a catalog card
# (deck-search / discard-cost / rainbow-evolution payload routes). v4
# slots [0:52] BYTE-STABLE. See
# `docs/ai-research/scoping/v5-action-disambiguation-scoping.md` §4.5.
ACTION_FEATURE_SCHEMA_VERSION = 5

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
# R16-P1: STATE_DIM_V3_1 is now a REAL builder (`observation_to_features_v3_1`)
# emitting 164-d = frozen v3.0 110-d (slots 0–109, byte-stable) + 54 temporal
# / turn-state scalar slots [110:164]. STATE_DIM stays an alias of
# STATE_DIM_V3 (110) — v3.1 is opt-in (selected by graph state dim in
# serve_onnx, or by an explicit builder choice in training).
STATE_DIM_V3_1 = 164  # R16-P1: real v3.1 temporal/turn-state builder
STATE_FEATURE_SCHEMA_VERSION_V3_1 = 3.1
# v33-additive-tail (`v33-additive-tail-scoping.md`): v3.3 extends v3.1's
# 164-d head with 3 opp-side flag bits at slots [164:167]. The slots mirror
# the own-side bits already emitted at slots 114-120's
# usedSupporter/Retreat/Stadium positions. v3.3 inherits v3.2's slot-token
# 7-input ONNX contract (v3.2 set `uses_uma_slot_tokens=True` orthogonally
# without bumping state_dim; v3.3 stacks on top of v3.2).
STATE_DIM_V3_3 = 167  # v33-additive-tail: v3.1 head + 3-bit opp-side flag tail
STATE_FEATURE_SCHEMA_VERSION_V3_3 = 3.3
# v34-compound-axis: v3.3 state vector (state_dim=167) PLUS v3.2 per-Uma
# slot tokens (`uses_uma_slot_tokens=True`). The state-vector dim itself
# is unchanged from v3.3; v3.4 differs only by the slot-token ONNX input
# pair. Labeled 3.4 in meta.json when both axes are active so downstream
# tooling can distinguish the compound ckpt from a plain v3.3 ckpt. v3.4
# FALSIFIED 2026-05-25 (progress/r110.md §4h); kept here for dispatch
# completeness, not as a recommended schema.
STATE_FEATURE_SCHEMA_VERSION_V3_4 = 3.4
# v35-multichannel-tail (`v35-multichannel-tail-scoping.md`): v3.5 extends
# v3.3's 167-d head with 45 channel-orthogonal bits in the tail. Layout:
#   [167:177] phase one-hot (10) — temporal-cadence channel
#   [177:182] own active per-condition one-hot (5) — uma-condition channel
#   [182:187] opp active per-condition one-hot (5) — uma-condition channel
#   [187:197] own energy-zone front-of-queue typed one-hot (10) — energy-color
#   [197:207] opp energy-zone front-of-queue typed one-hot (10) — energy-color
#   [207:210] opp discard role buckets (3) — zone-residual symmetry (mirror
#             of own slots 84-86 inside _card_awareness_features [16:19])
#   [210]     own would_lose_on_active_KO (bench empty) — terminal-state synth
#   [211]     opp would_lose_on_active_KO (bench empty) — terminal-state synth
# Channels chosen so each new bit touches signal v3.3 cannot express.
# Slot-tokens (v3.2/v3.4) EXCLUDED — proven channel-overlap with opp-flag
# tail at progress/r110.md §4h.
STATE_DIM_V3_5 = 212  # v35-multichannel-tail: v3.3 head + 45-bit tail
STATE_FEATURE_SCHEMA_VERSION_V3_5 = 3.5
# v36-priors-arithmetic (`v36-priors-and-arithmetic-scoping.md`): v3.6 extends
# v3.5's 212-d state vector by trimming the dead opp.energy_zone.front band at
# [197:207] and adding 44 channel-orthogonal bits (net +34 → 246):
#   [197:207] own.energy_pool typed multihot (10) — IN-BAND repurpose of the
#             dead opp.energy_zone slot; future-roll color distribution.
#   [212:222] opp.energy_pool typed multihot (10) — same channel as own.
#   [222:226] own prize one-hot ×4 over remaining-prize {3,2,1,0}.
#   [226:230] opp prize one-hot ×4 over remaining-prize {3,2,1,0}.
#   [230:240] opp bench typed energy aggregate (10) — multihot over the 10
#             energy types; bit set iff any opp bench Uma has ≥1 attached
#             energy of that type. Own bench dropped at impl reconciliation
#             (see scoping §4 step 2 — own.active typed energies already in
#             v3.5 head; own bench partly redundant).
#   [240]     own_lethal_next_turn (face-value, §4.5 Channel 4).
#   [241]     opp_lethal_next_turn (face-value, §4.5 Channel 4).
#   [242]     own_secondary_attack_usable (§4.5 Channel 5).
#   [243]     own_secondary_attack_would_KO (§4.5 Channel 5).
#   [244]     opp_secondary_attack_usable (§4.5 Channel 5).
#   [245]     opp_secondary_attack_would_KO (§4.5 Channel 5).
# Slot-tokens (v3.2/v3.4) still EXCLUDED. v3.5's [0:197] byte-stable; v3.6's
# [197:207] is the repurposed band (own.energy_pool replaces the dead
# opp.energy_zone.front), and [212:246] is the truly-new tail.
STATE_DIM_V3_6 = 246
STATE_FEATURE_SCHEMA_VERSION_V3_6 = 3.6
# v37-combat-arith-and-catalog (`v37-combat-arith-and-catalog-scoping.md`): v3.7
# extends v3.6's 246-d state vector with 50 channel-orthogonal bits at indices
# [246:296] — combat-arithmetic + catalog-lookup channels v3.x has never
# encoded. Net +50 bits, v3.6 head [0:246] BYTE-STABLE (no field re-writes).
#   [246:248] own/opp weakness-adjusted lethal (2)            — Channel 1.
#   [248:250] own/opp weakness-adjusted secondary KO (2)      — Channel 1.
#   [250:254] own/opp primary has_cf / cf_eko (4)             — Channel 2.
#   [254:258] own/opp secondary has_cf / cf_eko (4)           — Channel 2.
#   [258:262] own/opp primary per-energy / per-bench (4)      — Channel 3.
#   [262:266] own/opp secondary per-energy / per-bench (4)    — Channel 3.
#   [266:270] own_primary_eta / own_sec_eta /
#             opp_primary_eta / opp_sec_eta (4)               — Channel 4.
#   [270:272] own/opp paralysis_window_open (2)               — Channel 4.
#   [272:276] own active tool effect-kind one-hot (4)         — Channel 5.
#   [276:280] opp active tool effect-kind one-hot (4)         — Channel 5.
#   [280:288] own active ability effect-kind one-hot (8)      — Channel 6.
#   [288:296] opp active ability effect-kind one-hot (8)      — Channel 6.
# All catalog data derived from card_id / tool_card_id via the v3.6 lookup
# helpers. No new obs-contract fields. Slot-tokens still EXCLUDED. See scoping
# §3 (freeze contract) + §4.5 (LOCKED bit definitions) + §13 (recon
# corrections — weakness ADDITIVE, Attack flat-struct, paralysis already
# typed, vocab tightened to 4-tool / 8-ability).
STATE_DIM_V3_7 = 296
STATE_FEATURE_SCHEMA_VERSION_V3_7 = 3.7
# v38-slim-feature-add (`v38-slim-feature-add-scoping.md`): v3.8 extends
# v3.7's 296-d state vector with 8 bits at [296:304] — per-bench primary-
# attack ETA (3 own + 3 opp = 6 bits) + gust-swing catastrophe (2 bits).
# Net +8 bits, v3.7 head [0:296] BYTE-STABLE. NO trunk widening.
#   [296:299] own.bench[0..2] primary-attack usable-next-turn (3 bits)
#   [299:302] opp.bench[0..2] primary-attack usable-next-turn (3 bits)
#   [302]     own_lose_if_opp_gusts_weakest_bench
#   [303]     own_can_gust_win_prize_race
# Per-bench bits reuse the v3.7 Ch.4 ETA structural-feasibility helper
# (`_v37_attack_usable_next_turn`) applied to `side.bench[i]` (with
# `side.energyPool` as the +1 attach budget pool). Gust-swing predicates
# are pure-function cross-bits over public state — see scoping doc §4.5
# and §13.4 for the adapted opp-side predicate (opp's hand_card_ids are
# private, so we use public proxies: opp gust appears in opp.discard +
# opp.usedSupporterThisTurn=False + opp.hand_count>0).
STATE_DIM_V3_8 = 304
STATE_FEATURE_SCHEMA_VERSION_V3_8 = 3.8
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


# ---------------------------------------------------------------------------
# R16-P1 schema-v3.1 temporal / turn-state builder.
#
# Emits 164-d = frozen v3.0 110-d (slots 0–109, byte-IDENTICAL — produced by
# reusing `observation_to_features` verbatim, never re-deriving them) + 54
# temporal/turn-state scalar slots [110:164]. Slot layout is the FROZEN
# enumeration from
# docs/ai-research/scoping/r16-model-feature-backlog-refinement.md § P1:
#
#   110–113  global temporal: ownTurnsTaken(/CAP), oppTurnsTaken(/CAP),
#            ownIsFirstTurn(bool), oppIsFirstTurn(bool)
#   114–120  own side turnState ×7
#   121–127  opp side turnState ×7
#   128–136  own active per-Uma temporal ×9
#   137–145  own bench aggregate (mean of the 9 over present bench Umas)
#   146–154  opp active per-Uma temporal ×9
#   155–163  opp bench aggregate (mean of the 9 over present bench Umas)
#
# Encoding (overfit guard, per scope § "Encoding"):
#   - never emit raw turn stamps; the TS observation already derives the
#     sickness/memory booleans (enteredThisTurn/evolvedThisTurn/
#     evolvedLastTurn/attackBlockedThisTurn/paralysisRecoveryPending).
#   - turn counters: bounded-norm min(x, _TURN_CAP)/_TURN_CAP (CAP=20,
#     matching the existing turnNumber/20 normalisation at slot 2).
#   - small int budgets: /3 (energy attach budgets, coin flips,
#     ability-name counts), /30 (damage-ish values: activeAttackDamageBonus,
#     nextTurnDamageReduction, retreat reduction).
#   - damage memory / sickness predicates: bool as-is (0.0/1.0).
# The bench block is the *mean* of each per-Uma field over present bench
# Umas (0 if the bench is empty) — a max variant was considered but mean
# keeps the 9-wide block (table says "mean/max"; mean chosen as the single
# aggregate that does not double the block width beyond the 54-slot budget).
STATE_DIM_V3_1 = 164  # (re-stated post-helpers for locality; == module const)
_TURN_CAP = 20.0  # matches features[2] = turnNumber / 20.0
_BUDGET_CAP = 3.0  # energy-attach budgets, coin flips, ability-name counts
_DAMAGE_CAP = 30.0  # activeAttackDamageBonus / nextTurnDamageReduction / retreat


def _norm(value: float, cap: float) -> float:
    return min(float(value), cap) / cap


def _side_turn_state_vec(side: dict[str, Any]) -> np.ndarray:
    """7 side-level turnState scalars. The retreat slot encodes the DERIVED
    `effectiveRetreatCostReduction` (raw side reduction + stadium global
    term) per the omission resolution — falling back to the raw
    `retreatCostReduction` only if the TS observation predates the v3.1
    schema bump (it should not, schemaVersion 3 carries the derived key).
    Ability NAME strings are never read — the TS layer emits counts only.
    """

    ts = side.get("turnState") or {}
    effective_retreat = ts.get("effectiveRetreatCostReduction")
    if effective_retreat is None:
        effective_retreat = ts.get("retreatCostReduction", 0)
    out = np.zeros(7, dtype=np.float32)
    out[0] = _norm(ts.get("energyAttachmentsThisTurn", 0), _BUDGET_CAP)
    out[1] = _norm(ts.get("bonusEnergyAttachments", 0), _BUDGET_CAP)
    out[2] = _norm(effective_retreat, _DAMAGE_CAP)
    out[3] = _norm(ts.get("activeAttackDamageBonus", 0), _DAMAGE_CAP)
    out[4] = _norm(ts.get("usedAbilityNameCountThisTurn", 0), _BUDGET_CAP)
    out[5] = _norm(ts.get("usedAbilityNameCountThisGame", 0), _TURN_CAP)
    out[6] = _norm(ts.get("guaranteedCoinFlipHeads", 0), _BUDGET_CAP)
    return out


def _uma_turn_state_vec(uma: dict[str, Any] | None) -> np.ndarray:
    """9 per-Uma temporal scalars. Absent Uma (None / empty slot) → zeros
    (the present-mask is already carried by the v3.0 board-feature slots)."""

    out = np.zeros(9, dtype=np.float32)
    if not uma:
        return out
    ts = uma.get("turnState") or {}
    out[0] = _norm(ts.get("turnsInPlay", 0), _TURN_CAP)
    out[1] = 1.0 if ts.get("enteredThisTurn") else 0.0
    out[2] = 1.0 if ts.get("evolvedThisTurn") else 0.0
    out[3] = 1.0 if ts.get("evolvedLastTurn") else 0.0
    out[4] = 1.0 if ts.get("tookDamageLastTurn") else 0.0
    out[5] = 1.0 if ts.get("tookDamageThisTurn") else 0.0
    out[6] = _norm(ts.get("nextTurnDamageReduction", 0), _DAMAGE_CAP)
    out[7] = 1.0 if ts.get("attackBlockedThisTurn") else 0.0
    out[8] = 1.0 if ts.get("paralysisRecoveryPending") else 0.0
    return out


def _bench_turn_state_aggregate(side: dict[str, Any]) -> np.ndarray:
    """Mean of the 9 per-Uma temporal scalars over present bench Umas.
    Empty bench → zeros."""

    bench = side.get("bench") or []
    present = [u for u in bench if u]
    if not present:
        return np.zeros(9, dtype=np.float32)
    stacked = np.stack([_uma_turn_state_vec(u) for u in present], axis=0)
    return stacked.mean(axis=0).astype(np.float32)


def observation_to_features_v3_1(
    observation: dict[str, Any], ablations: set[FeatureAblation] | None = None
) -> np.ndarray:
    """R16-P1 schema-v3.1: 164-d. Slots 0–109 are byte-identical to the
    frozen v3.0 builder (produced by calling it directly, NOT re-derived);
    slots [110:164] are the temporal/turn-state block. The
    `state_temporal_turn_v31` ablation zeroes [110:164]."""

    base_ablations = set(ablations or set())
    # v3.0 owns ablations that touch slots 0–109; the temporal-block
    # ablation is applied below after the tail is written so it cannot be
    # silently dropped by the v3.0 builder's slice asserts.
    v30 = {a for a in base_ablations if a != "state_temporal_turn_v31"}
    head = observation_to_features(observation, ablations=v30)
    assert head.shape == (STATE_DIM_V3,), (
        f"v3.1 head reuse expected ({STATE_DIM_V3},), got {head.shape}"
    )

    features = np.zeros(STATE_DIM_V3_1, dtype=np.float32)
    features[0:STATE_DIM_V3] = head

    own = observation.get("own", {}) or {}
    opponent = observation.get("opponent", {}) or {}
    temporal = observation.get("temporal", {}) or {}

    # 110–113 global temporal
    features[110] = _norm(temporal.get("ownTurnsTaken", 0), _TURN_CAP)
    features[111] = _norm(temporal.get("opponentTurnsTaken", 0), _TURN_CAP)
    features[112] = 1.0 if temporal.get("ownIsFirstTurn") else 0.0
    features[113] = 1.0 if temporal.get("opponentIsFirstTurn") else 0.0

    # 114–120 own side turnState ×7 ; 121–127 opp side turnState ×7
    features[114:121] = _side_turn_state_vec(own)
    features[121:128] = _side_turn_state_vec(opponent)

    # 128–136 own active ×9 ; 137–145 own bench aggregate ×9
    features[128:137] = _uma_turn_state_vec(own.get("active"))
    features[137:146] = _bench_turn_state_aggregate(own)
    # 146–154 opp active ×9 ; 155–163 opp bench aggregate ×9
    features[146:155] = _uma_turn_state_vec(opponent.get("active"))
    features[155:164] = _bench_turn_state_aggregate(opponent)

    # Temporal-block ablation lives in `apply_state_ablations` (canonical
    # home, mirrors `state_hygiene_v21`). Slots 0–109 ablations were already
    # applied by the v3.0 head call; apply ONLY the temporal key here so the
    # head stays byte-identical to the standalone v3.0 builder.
    if "state_temporal_turn_v31" in base_ablations:
        apply_state_ablations(features, {"state_temporal_turn_v31"})

    assert features.shape == (STATE_DIM_V3_1,), (
        f"observation_to_features_v3_1 emitted {features.shape}, "
        f"expected ({STATE_DIM_V3_1},)."
    )
    return features


# v33-additive-tail: v3.3 = v3.1 head + 3 opp-side flag bits at slots
# [164:167]. The flags are the opponent-side mirror of own slots 29/30/31
# (usedSupporterThisTurn / usedRetreatThisTurn / usedStadiumThisTurn). The
# own-side booleans are emitted in the FROZEN v2 head at slots 29/30/31
# (carried verbatim into v3.0/v3.1). The opp-side mirror is GENUINELY
# MISSING from v3.0/v3.1/v3.2 — the v3.1 opp turnState block (slots
# 121-127) covers resource state (energy attachments, retreat reduction,
# damage bonus, ability counts, coin flips) but NOT the used* booleans.
# So v3.3 is the first time opponent's "have they already burned a
# supporter / retreat / stadium this turn" signal is surfaced to the
# model. Per the v33-feature-gap-brainstorm-handoff.md §2.A item 2
# hypothesis, this gives the policy head threat-window awareness.
_OPP_USED_SUPPORTER_SLOT = 164
_OPP_USED_RETREAT_SLOT = 165
_OPP_USED_STADIUM_SLOT = 166


def observation_to_features_v3_3(
    observation: dict[str, Any], ablations: set[FeatureAblation] | None = None
) -> np.ndarray:
    """v33-additive-tail: 167-d. Slots 0–163 are byte-identical to v3.1
    (produced by calling `observation_to_features_v3_1` directly, NOT
    re-derived); slots [164:167] are the opp-side used* flag tail."""

    base = observation_to_features_v3_1(observation, ablations=ablations)
    assert base.shape == (STATE_DIM_V3_1,), (
        f"v3.3 base reuse expected ({STATE_DIM_V3_1},), got {base.shape}"
    )

    features = np.zeros(STATE_DIM_V3_3, dtype=np.float32)
    features[0:STATE_DIM_V3_1] = base

    opp = observation.get("opponent", {}) or {}
    features[_OPP_USED_SUPPORTER_SLOT] = 1.0 if opp.get("usedSupporterThisTurn") else 0.0
    features[_OPP_USED_RETREAT_SLOT] = 1.0 if opp.get("usedRetreatThisTurn") else 0.0
    features[_OPP_USED_STADIUM_SLOT] = 1.0 if opp.get("usedStadiumThisTurn") else 0.0

    assert features.shape == (STATE_DIM_V3_3,), (
        f"observation_to_features_v3_3 emitted {features.shape}, "
        f"expected ({STATE_DIM_V3_3},)."
    )
    return features


# v35-multichannel-tail: vocabularies and slot offsets for the 45-bit tail.
# FROZEN — adding a new vocab arm in the future must be an additive bump
# (v3.6 or later), never a mutation of these tuples or the slot offsets.

# Special-condition vocab. Sourced from
# `frontend/src/game/engine/flow/specialConditions.ts` / `flow/turn.ts` /
# `flow/eligibility.ts` — these five tokens are the complete set the engine
# emits into `specialConditions: string[]` as of 2026-05-25. Matches the
# /5 cap at v3.0 slot 20 (`condition_count_norm`) so the one-hot is
# information-complete vs the legacy "count cap = 5" encoding.
_V35_CONDITION_VOCAB: tuple[str, ...] = (
    "paralysed",
    "burned",
    "poisoned",
    "asleep",
    "frozen",
)

# Energy-zone front-of-queue vocab. REUSES `_UMA_SLOT_ENERGY_TYPES` order
# verbatim (10 entries: grass, fire, water, lightning, psychic, fighting,
# darkness, steel, colorless, dragon). Defined later in this module; we
# index into it by string at v3.5 emit time.

# v3.5 tail slot offsets (absolute, in the 212-d vector).
_V35_PHASE_ONE_HOT_START = 167          # [167:177] 10 bits
_V35_PHASE_ONE_HOT_END = 177
_V35_OWN_COND_START = 177               # [177:182] 5 bits
_V35_OWN_COND_END = 182
_V35_OPP_COND_START = 182               # [182:187] 5 bits
_V35_OPP_COND_END = 187
_V35_OWN_ENERGY_FRONT_START = 187       # [187:197] 10 bits
_V35_OWN_ENERGY_FRONT_END = 197
_V35_OPP_ENERGY_FRONT_START = 197       # [197:207] 10 bits
_V35_OPP_ENERGY_FRONT_END = 207
_V35_OPP_DISCARD_BUCKETS_START = 207    # [207:210] 3 bits
_V35_OPP_DISCARD_BUCKETS_END = 210
_V35_OWN_BENCH_REFILL_SLOT = 210        # 1 bit
_V35_OPP_BENCH_REFILL_SLOT = 211        # 1 bit


def _v35_condition_one_hot(uma: dict[str, Any] | None) -> np.ndarray:
    """5-bit one-hot over `_V35_CONDITION_VOCAB` for the given Uma's
    `specialConditions` list. Multiple conditions set multiple bits. Missing
    / null Uma → all zeros. Unknown tokens (vocab drift) → silently dropped
    (the count/5 legacy slot still surfaces them via cardinality)."""

    values = np.zeros(len(_V35_CONDITION_VOCAB), dtype=np.float32)
    if not uma:
        return values
    conditions = uma.get("specialConditions") or []
    for cond in conditions:
        try:
            idx = _V35_CONDITION_VOCAB.index(str(cond))
        except ValueError:
            continue
        values[idx] = 1.0
    return values


def _v35_energy_front_one_hot(side: dict[str, Any]) -> np.ndarray:
    """10-bit one-hot over `_UMA_SLOT_ENERGY_TYPES` for the front-of-queue
    entry of this side's `energyZone`. Empty / missing zone → all zeros.
    Unknown energy type → all zeros."""

    values = np.zeros(len(_UMA_SLOT_ENERGY_TYPES), dtype=np.float32)
    zone = side.get("energyZone") or []
    if not zone:
        return values
    front = str(zone[0])
    try:
        idx = _UMA_SLOT_ENERGY_TYPES.index(front)
    except ValueError:
        return values
    values[idx] = 1.0
    return values


def _v35_bench_refill_catastrophe(side: dict[str, Any]) -> float:
    """1.0 if this side has no non-null bench Uma to promote on active KO.
    Minimum-viable terminal-state predicate; HP threshold deferred (see
    scoping doc §4.5)."""

    bench = side.get("bench") or []
    has_promotable = any(u for u in bench)
    return 0.0 if has_promotable else 1.0


def observation_to_features_v3_5(
    observation: dict[str, Any], ablations: set[FeatureAblation] | None = None
) -> np.ndarray:
    """v35-multichannel-tail: 212-d. Slots 0–166 are byte-identical to v3.3
    (produced by calling `observation_to_features_v3_3` directly, NOT
    re-derived); slots [167:212] are the 45-bit channel-orthogonal tail."""

    base = observation_to_features_v3_3(observation, ablations=ablations)
    assert base.shape == (STATE_DIM_V3_3,), (
        f"v3.5 base reuse expected ({STATE_DIM_V3_3},), got {base.shape}"
    )

    features = np.zeros(STATE_DIM_V3_5, dtype=np.float32)
    features[0:STATE_DIM_V3_3] = base

    own = observation.get("own", {}) or {}
    opponent = observation.get("opponent", {}) or {}

    # Phase one-hot [167:177]. Unknown phase → all zeros (the legacy
    # ordinal at slot 0 already encodes "unknown→0" as stadiumOrEnd index 0
    # by default; the one-hot here is strictly additive).
    phase = observation.get("phase")
    if phase in PHASES:
        features[_V35_PHASE_ONE_HOT_START + PHASES.index(phase)] = 1.0

    # Per-condition one-hot, own [177:182] and opp [182:187].
    features[_V35_OWN_COND_START:_V35_OWN_COND_END] = _v35_condition_one_hot(
        own.get("active")
    )
    features[_V35_OPP_COND_START:_V35_OPP_COND_END] = _v35_condition_one_hot(
        opponent.get("active")
    )

    # Energy-zone front-of-queue typed one-hot, own [187:197] and opp [197:207].
    features[_V35_OWN_ENERGY_FRONT_START:_V35_OWN_ENERGY_FRONT_END] = (
        _v35_energy_front_one_hot(own)
    )
    features[_V35_OPP_ENERGY_FRONT_START:_V35_OPP_ENERGY_FRONT_END] = (
        _v35_energy_front_one_hot(opponent)
    )

    # Opp-side discard role buckets [207:210] — mirror of own slots 84-86
    # inside _card_awareness_features (own discard at values[16:19] of that
    # 28-d block). REUSES `_discard_role_features` for bit-exactness.
    features[_V35_OPP_DISCARD_BUCKETS_START:_V35_OPP_DISCARD_BUCKETS_END] = (
        _discard_role_features(opponent.get("discard") or [])
    )

    # Bench-refill catastrophe bits [210] own, [211] opp.
    features[_V35_OWN_BENCH_REFILL_SLOT] = _v35_bench_refill_catastrophe(own)
    features[_V35_OPP_BENCH_REFILL_SLOT] = _v35_bench_refill_catastrophe(opponent)

    assert features.shape == (STATE_DIM_V3_5,), (
        f"observation_to_features_v3_5 emitted {features.shape}, "
        f"expected ({STATE_DIM_V3_5},)."
    )
    return features


# v3.6 tail slot offsets (absolute, in the 246-d vector). The [197:207] band
# is REPURPOSED in-place from the dead v3.5 opp.energy_zone.front to the new
# own.energy_pool typed multihot. The [212:246] band is the truly-new tail.
_V36_OWN_ENERGY_POOL_START = 197       # [197:207] 10 bits — REPURPOSED IN-BAND
_V36_OWN_ENERGY_POOL_END = 207
_V36_OPP_ENERGY_POOL_START = 212       # [212:222] 10 bits
_V36_OPP_ENERGY_POOL_END = 222
_V36_OWN_PRIZE_START = 222             # [222:226] 4 bits
_V36_OWN_PRIZE_END = 226
_V36_OPP_PRIZE_START = 226             # [226:230] 4 bits
_V36_OPP_PRIZE_END = 230
_V36_OPP_BENCH_TYPED_START = 230       # [230:240] 10 bits (opp-only — see header)
_V36_OPP_BENCH_TYPED_END = 240
_V36_OWN_LETHAL_NEXT_TURN_SLOT = 240   # 1 bit
_V36_OPP_LETHAL_NEXT_TURN_SLOT = 241   # 1 bit
_V36_OWN_SECONDARY_USABLE_SLOT = 242   # 1 bit
_V36_OWN_SECONDARY_WOULD_KO_SLOT = 243 # 1 bit
_V36_OPP_SECONDARY_USABLE_SLOT = 244   # 1 bit
_V36_OPP_SECONDARY_WOULD_KO_SLOT = 245 # 1 bit


def _v36_energy_pool_multihot(side: dict[str, Any]) -> np.ndarray:
    """10-bit multihot over `_UMA_SLOT_ENERGY_TYPES` for the side's typed
    `energyPool` (added by the v3.6 obs-contract extension). Bag semantics:
    duplicates collapse (`[fire, fire, water]` → only fire+water bits).
    Missing / empty / unknown energy types → silently 0. Camel-case key
    matches the Rust JSON `rename_all = "camelCase"` rule
    (`engine-rs/crates/engine/src/policy/types.rs:140`)."""

    values = np.zeros(len(_UMA_SLOT_ENERGY_TYPES), dtype=np.float32)
    pool = side.get("energyPool") or []
    for token in pool:
        try:
            idx = _UMA_SLOT_ENERGY_TYPES.index(str(token))
        except ValueError:
            continue
        values[idx] = 1.0
    return values


def _v36_prize_one_hot(side: dict[str, Any]) -> np.ndarray:
    """4-bit one-hot over remaining-prize counts {3, 2, 1, 0}.

    `remaining = clamp(3 - points, 0, 3)`. Index 0 → 3 prizes left (game
    start); index 3 → 0 prizes left (terminal frame; should not fire at a
    player-decision point but included for completeness per scoping §4.5)."""

    values = np.zeros(4, dtype=np.float32)
    points = int(side.get("points", 0) or 0)
    remaining = max(0, min(3, 3 - points))
    # Map remaining → bit index: 3→0, 2→1, 1→2, 0→3.
    bit = 3 - remaining
    values[bit] = 1.0
    return values


def _v36_bench_typed_aggregate(side: dict[str, Any]) -> np.ndarray:
    """10-bit multihot over `_UMA_SLOT_ENERGY_TYPES` for the side's bench
    (EXCLUDES the active). Bit `i` set iff any bench Uma has ≥1 attached
    energy of type `_UMA_SLOT_ENERGY_TYPES[i]`. Padded-None bench slots
    silently skipped. Unknown energy keys silently dropped."""

    values = np.zeros(len(_UMA_SLOT_ENERGY_TYPES), dtype=np.float32)
    for uma in (side.get("bench") or []):
        if not uma:
            continue
        energies = uma.get("energies") or {}
        for idx, energy_type in enumerate(_UMA_SLOT_ENERGY_TYPES):
            amt = energies.get(energy_type, 0) or 0
            if float(amt) > 0:
                values[idx] = 1.0
    return values


def _v36_active_attacks(active: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the attack list for an active Uma via the card catalog.

    Public observation does NOT carry per-active attack data
    (`PublicUmaObservation` only emits hp/max_hp/energies). v3.6 lethal +
    secondary-attack channels read attack base damage and energy cost from
    `_get_card(card_id).attacks`, which is the same lookup
    `_card_progress_features` and `_attack_readiness_features` already use
    elsewhere in this module."""

    if not active:
        return []
    card = _get_card(str(active.get("cardId") or ""))
    if not card:
        return []
    attacks = card.get("attacks") or []
    return [a for a in attacks if isinstance(a, dict)]


def _v36_attack_base_damage(attack: dict[str, Any]) -> float:
    """Static base damage of an attack (`Attack.damage` in cards.json;
    `Attack.base_damage` in engine-rs). Face-value only — NOT adjusted for
    weakness, energy availability, status conditions, damage reduction,
    coin flips, or any conditional bonus. Per scoping §4.5 Channel 4."""

    val = attack.get("damage", attack.get("baseDamage", 0)) or 0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _v36_remaining_hp(active: dict[str, Any] | None) -> float:
    """Effective HP for the lethal predicate. `hp` in the public
    observation already reflects damage taken (engine sets
    `umamusume.hp = max_hp - damage_taken`), so we use `hp` directly. If
    `hp` is missing we fall back to `max_hp - damageCounters` to stay
    forward-compatible with any future schema that exposes raw
    damage_counters separately."""

    if not active:
        return 0.0
    if "hp" in active and active.get("hp") is not None:
        try:
            return float(active["hp"])
        except (TypeError, ValueError):
            return 0.0
    max_hp = float(active.get("maxHp", 0) or 0)
    damage = float(active.get("damageCounters", 0) or 0)
    return max(0.0, max_hp - damage)


def _v36_attack_cost_covered(
    attached: dict[str, Any], cost: dict[str, Any]
) -> bool:
    """Mirror of engine attack-legality: typed costs satisfied per-color
    AND total attached ≥ total cost (colorless absorbs leftover energy of
    any type). `_typed_energy_deficit` already encodes the typed half;
    re-use it to avoid a second copy of the multiset matcher."""

    if _typed_energy_deficit(attached, cost) > 0.0:
        return False
    total_attached = float(sum(float(v or 0) for v in attached.values()))
    total_cost = _total_cost(cost)
    return total_attached >= total_cost


def _v36_lethal_face_value(
    attacker_active: dict[str, Any] | None,
    defender_active: dict[str, Any] | None,
) -> float:
    """Return 1.0 iff the attacker's active has any attack with
    `base_damage ≥ defender.remaining_hp`. Face-value only (scoping §4.5
    Channel 4): no weakness multiplier, no coin-flip expectation, no
    energy-availability check. Returns 0.0 if attacker active is null or
    has no attacks, or if defender active is null (a null defender means
    nothing to KO — but in practice this fires during pending-promote
    states; conservative 0)."""

    attacks = _v36_active_attacks(attacker_active)
    if not attacks or not defender_active:
        return 0.0
    defender_hp = _v36_remaining_hp(defender_active)
    max_dmg = max((_v36_attack_base_damage(a) for a in attacks), default=0.0)
    return 1.0 if max_dmg >= defender_hp else 0.0


def _v36_secondary_attack_bits(
    attacker_active: dict[str, Any] | None,
    defender_active: dict[str, Any] | None,
) -> tuple[float, float]:
    """Return (usable, would_KO) for the attacker's `attacks[1]`. Both
    bits 0.0 if no secondary attack exists. `usable` requires energy
    coverage per `_v36_attack_cost_covered`; `would_KO` additionally
    requires `usable AND base_damage ≥ defender.remaining_hp`. Face-value
    per scoping §4.5 Channel 5."""

    attacks = _v36_active_attacks(attacker_active)
    if len(attacks) < 2:
        return 0.0, 0.0
    secondary = attacks[1]
    attached = (attacker_active or {}).get("energies") or {}
    cost = secondary.get("cost") or secondary.get("energyCost") or {}
    usable = _v36_attack_cost_covered(attached, cost)
    if not usable:
        return 0.0, 0.0
    if not defender_active:
        return 1.0, 0.0
    defender_hp = _v36_remaining_hp(defender_active)
    base_damage = _v36_attack_base_damage(secondary)
    would_ko = 1.0 if base_damage >= defender_hp else 0.0
    return 1.0, would_ko


def observation_to_features_v3_6(
    observation: dict[str, Any], ablations: set[FeatureAblation] | None = None
) -> np.ndarray:
    """v36-priors-arithmetic: 246-d. Layered on top of v3.5 — calls
    `observation_to_features_v3_5(...)`, then zero-overwrites the dead
    [197:207] opp.energy_zone.front band and repurposes it for own
    energy_pool typed multihot, then appends 34 bits at [212:246] per the
    layout locked in `docs/ai-research/scoping/v36-priors-and-arithmetic-
    scoping.md` §4 step 2.

    Reconciliation note (impl-phase): the §4.5 channel breakdown sums to
    54 bits across both sides; TL;DR locks STATE_DIM at 246 (net +34 bits).
    Own bench typed energy aggregate dropped as the lowest-priority cut
    (own.active typed energies already in v3.5 head). Opp bench kept since
    opp-threat-by-color is the stated rationale."""

    base = observation_to_features_v3_5(observation, ablations=ablations)
    assert base.shape == (STATE_DIM_V3_5,), (
        f"v3.6 base reuse expected ({STATE_DIM_V3_5},), got {base.shape}"
    )

    features = np.zeros(STATE_DIM_V3_6, dtype=np.float32)
    features[0:STATE_DIM_V3_5] = base

    # Zero-overwrite the dead v3.5 opp.energy_zone.front band [197:207] —
    # this band is repurposed in-place for own.energy_pool typed multihot.
    # This is the only v3.5 slot v3.6 touches; [0:197] stays byte-stable.
    features[_V36_OWN_ENERGY_POOL_START:_V36_OWN_ENERGY_POOL_END] = 0.0

    own = observation.get("own", {}) or {}
    opponent = observation.get("opponent", {}) or {}
    own_active = own.get("active") if isinstance(own, dict) else None
    opp_active = opponent.get("active") if isinstance(opponent, dict) else None

    # Channel 1 — energy_pool typed multihot. Own in-band at [197:207];
    # opp at appended-tail [212:222].
    features[_V36_OWN_ENERGY_POOL_START:_V36_OWN_ENERGY_POOL_END] = (
        _v36_energy_pool_multihot(own)
    )
    features[_V36_OPP_ENERGY_POOL_START:_V36_OPP_ENERGY_POOL_END] = (
        _v36_energy_pool_multihot(opponent)
    )

    # Channel 2 — prize one-hot ×4 over remaining-prize {3,2,1,0}.
    features[_V36_OWN_PRIZE_START:_V36_OWN_PRIZE_END] = _v36_prize_one_hot(own)
    features[_V36_OPP_PRIZE_START:_V36_OPP_PRIZE_END] = _v36_prize_one_hot(
        opponent
    )

    # Channel 3 — opp bench typed energy aggregate (multihot). Own dropped
    # at reconciliation; see header.
    features[_V36_OPP_BENCH_TYPED_START:_V36_OPP_BENCH_TYPED_END] = (
        _v36_bench_typed_aggregate(opponent)
    )

    # Channel 4 — lethal-next-turn face-value. `own_lethal` means OPP can
    # KO OWN's active at face value.
    features[_V36_OWN_LETHAL_NEXT_TURN_SLOT] = _v36_lethal_face_value(
        attacker_active=opp_active, defender_active=own_active
    )
    features[_V36_OPP_LETHAL_NEXT_TURN_SLOT] = _v36_lethal_face_value(
        attacker_active=own_active, defender_active=opp_active
    )

    # Channel 5 — secondary attack readiness + would-KO.
    own_sec_usable, own_sec_ko = _v36_secondary_attack_bits(
        attacker_active=own_active, defender_active=opp_active
    )
    features[_V36_OWN_SECONDARY_USABLE_SLOT] = own_sec_usable
    features[_V36_OWN_SECONDARY_WOULD_KO_SLOT] = own_sec_ko
    opp_sec_usable, opp_sec_ko = _v36_secondary_attack_bits(
        attacker_active=opp_active, defender_active=own_active
    )
    features[_V36_OPP_SECONDARY_USABLE_SLOT] = opp_sec_usable
    features[_V36_OPP_SECONDARY_WOULD_KO_SLOT] = opp_sec_ko

    assert features.shape == (STATE_DIM_V3_6,), (
        f"observation_to_features_v3_6 emitted {features.shape}, "
        f"expected ({STATE_DIM_V3_6},)."
    )
    return features


# ---------------------------------------------------------------------------
# v3.7 combat-arith-and-catalog tail
# (`docs/ai-research/scoping/v37-combat-arith-and-catalog-scoping.md`).
#
# Layout (FROZEN; mirrored bit-for-bit in
# `engine-rs/crates/engine/src/policy/featurize.rs`):
#   [246:248] own/opp weakness-adjusted lethal (Channel 1).
#   [248:250] own/opp weakness-adjusted secondary KO (Channel 1).
#   [250:254] own primary has_cf, cf_eko ; opp primary has_cf, cf_eko
#             (Channel 2).
#   [254:258] own secondary has_cf, cf_eko ; opp secondary has_cf, cf_eko
#             (Channel 2).
#   [258:262] own primary per-energy, per-bench ; opp primary per-energy,
#             per-bench (Channel 3).
#   [262:266] own secondary per-energy, per-bench ; opp secondary
#             per-energy, per-bench (Channel 3).
#   [266:270] own primary ETA, own secondary ETA, opp primary ETA, opp
#             secondary ETA (Channel 4).
#   [270:272] own paralysis_window_open, opp paralysis_window_open
#             (Channel 4).
#   [272:276] own tool effect-kind one-hot, 4 classes (Channel 5).
#   [276:280] opp tool effect-kind one-hot, 4 classes (Channel 5).
#   [280:288] own ability effect-kind one-hot, 8 classes (Channel 6).
#   [288:296] opp ability effect-kind one-hot, 8 classes (Channel 6).
# ---------------------------------------------------------------------------

_V37_TAIL_START = 246
# Channel 1 — weakness-adjusted lethal / secondary KO.
_V37_OWN_WEAKNESS_LETHAL = 246
_V37_OPP_WEAKNESS_LETHAL = 247
_V37_OWN_WEAKNESS_SECONDARY_KO = 248
_V37_OPP_WEAKNESS_SECONDARY_KO = 249
# Channel 2 — coin-flip indicators (base + 8 offsets).
_V37_COIN_FLIP_BASE = 250  # +0 own_primary_has_cf, +1 own_primary_cf_eko,
                            # +2 opp_primary_has_cf, +3 opp_primary_cf_eko,
                            # +4 own_sec_has_cf,     +5 own_sec_cf_eko,
                            # +6 opp_sec_has_cf,     +7 opp_sec_cf_eko.
# Channel 3 — conditional-damage-bonus indicators (base + 8 offsets).
_V37_COND_BONUS_BASE = 258  # +0 own_primary_per_energy, +1 own_primary_per_bench,
                             # +2 opp_primary_per_energy, +3 opp_primary_per_bench,
                             # +4 own_sec_per_energy,     +5 own_sec_per_bench,
                             # +6 opp_sec_per_energy,     +7 opp_sec_per_bench.
# Channel 4 — energy ETA (4 bits) + paralysis windows (2 bits).
_V37_ETA_BASE = 266  # +0 own_primary_eta, +1 own_sec_eta,
                      # +2 opp_primary_eta, +3 opp_sec_eta.
_V37_PARALYSIS_BASE = 270  # +0 own_para_window, +1 opp_para_window.
# Channel 5 — tool effect-kind one-hot (4 classes per active).
_V37_TOOL_KIND_OWN_BASE = 272
_V37_TOOL_KIND_OPP_BASE = 276
# Channel 6 — ability effect-kind one-hot (8 classes per active).
_V37_ABILITY_KIND_OWN_BASE = 280
_V37_ABILITY_KIND_OPP_BASE = 288


def _v37_card_type(card: dict[str, Any] | None) -> str:
    """Read the printed `type` (UmamusumeType) of an Uma card. Returns
    empty string for absent / non-Uma cards. Catalog convention: the
    `type` JSON field is TitleCase (`"Psychic"`, `"Darkness"`); the
    weakness `type` field uses the same casing. Direct string equality
    matches engine `defender_weakness_match_type == attacker_card.r#type`
    semantics at `flow/combat.rs:303-305`."""

    if not card:
        return ""
    return str(card.get("type", "") or "")


def _v37_weakness_adjusted_lethal(
    attacker_active: dict[str, Any] | None,
    defender_active: dict[str, Any] | None,
) -> float:
    """Channel 1 bit: 1.0 iff
        attack.damage + (weakness.amount if attacker.type == defender.weakness.type else 0)
            >= defender.remaining_hp,
    using the PRIMARY attack (`attacks[0]`). Weakness bonus matches
    `flow/combat.rs:303-305` exactly: ADDITIVE, applied ONLY when
    `damage > 0` (TS-equivalent guard). Face-value cost — no energy
    coverage check; this is a "can the attacker face-KO if they can
    attack" predicate, parallel to v3.6's lethal-face-value."""

    if not attacker_active or not defender_active:
        return 0.0
    attacks = _v36_active_attacks(attacker_active)
    if not attacks:
        return 0.0
    primary = attacks[0]
    damage = _v36_attack_base_damage(primary)
    if damage > 0:
        attacker_card = _get_card(str(attacker_active.get("cardId", "") or ""))
        defender_card = _get_card(str(defender_active.get("cardId", "") or ""))
        attacker_type = _v37_card_type(attacker_card)
        weakness = (defender_card or {}).get("weakness") or {}
        weakness_type = str(weakness.get("type", "") or "")
        if attacker_type and weakness_type and attacker_type == weakness_type:
            try:
                damage += float(weakness.get("amount", 0) or 0)
            except (TypeError, ValueError):
                pass
    defender_hp = _v36_remaining_hp(defender_active)
    return 1.0 if damage >= defender_hp else 0.0


def _v37_weakness_adjusted_secondary_ko(
    attacker_active: dict[str, Any] | None,
    defender_active: dict[str, Any] | None,
    secondary_usable_bit: float,
) -> float:
    """Channel 1 secondary bit: predicated on the v3.6 secondary-readiness
    bit. If `secondary_usable_bit` == 0 (v3.6 says secondary isn't
    usable), return 0 — match scoping §4.5 Ch.1 "predicated on
    [slot 244]" semantics. Else compute the weakness-adjusted KO check
    on `attacks[1]`."""

    if secondary_usable_bit <= 0.0:
        return 0.0
    if not attacker_active or not defender_active:
        return 0.0
    attacks = _v36_active_attacks(attacker_active)
    if len(attacks) < 2:
        return 0.0
    secondary = attacks[1]
    damage = _v36_attack_base_damage(secondary)
    if damage > 0:
        attacker_card = _get_card(str(attacker_active.get("cardId", "") or ""))
        defender_card = _get_card(str(defender_active.get("cardId", "") or ""))
        attacker_type = _v37_card_type(attacker_card)
        weakness = (defender_card or {}).get("weakness") or {}
        weakness_type = str(weakness.get("type", "") or "")
        if attacker_type and weakness_type and attacker_type == weakness_type:
            try:
                damage += float(weakness.get("amount", 0) or 0)
            except (TypeError, ValueError):
                pass
    defender_hp = _v36_remaining_hp(defender_active)
    return 1.0 if damage >= defender_hp else 0.0


def _v37_attack_coin_flip_bits(
    attack: dict[str, Any] | None,
    defender_hp: float,
) -> tuple[float, float]:
    """Channel 2 bits for one attack: (has_cf, cf_eko).

    `has_cf` = 1 iff any of {coinBonus, knockOutActiveIfAllCoinHeads,
    drawOnHeads, discardRandomOpponentHandOnHeads} is set on the attack
    (non-None / non-zero per catalog JSON convention).

    `cf_eko` = 1 iff `attack.damage + 0.5 * (coinBonus or 0) >=
    defender_hp`. Single-flip expected-damage formula only —
    `knockOutActiveIfAllCoinHeads` contributes 0 (its KO is boolean),
    other heads-conditional fields contribute 0 to damage.
    """

    if not attack:
        return 0.0, 0.0
    coin_bonus = attack.get("coinBonus")
    knock_out_heads = attack.get("knockOutActiveIfAllCoinHeads")
    draw_heads = attack.get("drawOnHeads")
    discard_heads = attack.get("discardRandomOpponentHandOnHeads")
    has_cf = (
        coin_bonus is not None
        or knock_out_heads is not None
        or draw_heads is not None
        or discard_heads is not None
    )
    damage = _v36_attack_base_damage(attack)
    try:
        coin_bonus_val = float(coin_bonus) if coin_bonus is not None else 0.0
    except (TypeError, ValueError):
        coin_bonus_val = 0.0
    expected = damage + 0.5 * coin_bonus_val
    cf_eko = 1.0 if expected >= defender_hp else 0.0
    return (1.0 if has_cf else 0.0), cf_eko


def _v37_attack_conditional_bonus_bits(
    attack: dict[str, Any] | None,
) -> tuple[float, float]:
    """Channel 3 bits for one attack: (per_energy, per_bench).

    Structural-only flags. `per_energy` covers both
    `damagePerAttachedEnergy` and `damagePerUniqueAttachedEnergy`.
    `per_bench` covers `damagePerUmamusumeInPlay` (engine side scope is
    own-or-all; v3.7 treats both as 'per-bench' since the model already
    has bench counts at v3.6).
    """

    if not attack:
        return 0.0, 0.0
    per_energy = (
        attack.get("damagePerAttachedEnergy") is not None
        or attack.get("damagePerUniqueAttachedEnergy") is not None
    )
    per_bench = attack.get("damagePerUmamusumeInPlay") is not None
    return (1.0 if per_energy else 0.0), (1.0 if per_bench else 0.0)


def _v37_attack_usable_next_turn(
    attacker_active: dict[str, Any] | None,
    attack: dict[str, Any] | None,
    energy_pool: list[Any],
) -> float:
    """Channel 4 ETA bit: 1.0 iff the attack's cost becomes feasible by
    next turn, given the attacker's currently-attached energies plus a
    next-turn attach budget of 1 from the public `energyPool`.

    Per scoping §4.5 Ch.4 LOCKED DEFINITION (structural feasibility, no
    expected value):
      1. Compute per-color shortfall = cost - attached, clamped at 0,
         for each EnergyType (Colorless is handled below).
      2. The next-turn attach budget = 1; apply it to the LARGEST
         shortfall color (deterministic tiebreak: lexicographic order on
         the energy-type name, matching `_UMA_SLOT_ENERGY_TYPES`).
      3. After the +1 attach, every still-shortfall color (>0) must
         appear in `energyPool` (set-difference check). Total attached
         must also cover total cost (colorless absorbs leftover energy
         of any type).

    Returns 0.0 if `attack` is None / costless / attacker missing.
    """

    if not attack or not attacker_active:
        return 0.0
    cost = attack.get("cost") or attack.get("energyCost") or {}
    if not cost:
        return 1.0
    attached = attacker_active.get("energies") or {}
    # Per-color shortfall (excluding colorless — colorless absorbs any).
    shortfall: dict[str, int] = {}
    for energy_type, amount in cost.items():
        if energy_type == "colorless":
            continue
        need = float(amount or 0)
        have = float(attached.get(energy_type, 0) or 0)
        deficit = need - have
        if deficit > 0:
            shortfall[energy_type] = int(deficit)
    # Apply the +1 attach budget to the largest (lex-tiebreak: pick the
    # earliest key in `_UMA_SLOT_ENERGY_TYPES` order on ties).
    if shortfall:
        order_index = {et: i for i, et in enumerate(_UMA_SLOT_ENERGY_TYPES)}
        candidates = sorted(
            shortfall.items(),
            key=lambda kv: (-kv[1], order_index.get(kv[0], len(_UMA_SLOT_ENERGY_TYPES))),
        )
        target_type, target_amt = candidates[0]
        if target_amt <= 1:
            del shortfall[target_type]
        else:
            shortfall[target_type] = target_amt - 1
    # After the +1 attach, every remaining shortfall color must appear
    # in the pool. Per scope §4.5 Ch.4 LOCKED DEFINITION: "single
    # set-difference operation per attack" — TYPED-color feasibility
    # only; no total-cost / colorless-absorption check. This is a
    # STRUCTURAL feasibility predicate (the model already has
    # `energy_total` and the cost slots elsewhere in the head; the bit
    # signals "all typed colors reachable next turn").
    pool_set = {str(token) for token in (energy_pool or [])}
    for color in shortfall.keys():
        if color not in pool_set:
            return 0.0
    return 1.0


def _v37_paralysis_window_open(other_side_active: dict[str, Any] | None) -> float:
    """Channel 4 paralysis bit. `own_paralysis_window_open` = 1 iff
    OPP's active has `turn_state.paralysisRecoveryPending == True` —
    paralysed opp can't attack next turn, so own has a free-attack
    window. Direct typed bool read (per §13.4 recon: no
    special_conditions string-matching)."""

    if not other_side_active:
        return 0.0
    turn_state = other_side_active.get("turnState") or {}
    return 1.0 if bool(turn_state.get("paralysisRecoveryPending", False)) else 0.0


def _v37_tool_kind_one_hot(active: dict[str, Any] | None) -> np.ndarray:
    """Channel 5 one-hot (4 classes). All zeros iff
    `active.toolCardId is None`; else look up the tool card's
    `TrainerEffect` and apply `classify_tool_effect()` from Phase A.
    The Other class (index 3) fires when a tool's effect shape doesn't
    match the locked vocab (e.g. a future card)."""

    from .effect_kinds import ToolEffectKind, classify_tool_effect

    out = np.zeros(4, dtype=np.float32)
    if not active:
        return out
    tool_id = active.get("toolCardId")
    if not tool_id:
        return out
    card = _get_card(str(tool_id))
    if not card:
        return out
    # The JSON catalog stores trainer effect under `effect` for kind=trainer.
    effect = card.get("effect") or {}
    kind = classify_tool_effect(effect)
    out[int(ToolEffectKind(kind))] = 1.0
    return out


def _v37_ability_kind_one_hot(active: dict[str, Any] | None) -> np.ndarray:
    """Channel 6 one-hot (8 classes). All zeros iff the active didn't
    actually USE its ability this turn (`usedAbilityThisTurn == False`)
    or has no card / no `ability` payload; else look up the ability
    payload and apply `classify_active_ability()` from Phase A. The
    Other class (index 7) fires for genuine catalog one-offs (Tachyon
    coin-draw / disable-aura, Rudolf shuffle)."""

    from .effect_kinds import AbilityEffectKind, classify_active_ability

    out = np.zeros(8, dtype=np.float32)
    if not active:
        return out
    if not bool(active.get("usedAbilityThisTurn", False)):
        return out
    card = _get_card(str(active.get("cardId", "") or ""))
    if not card:
        return out
    ability = card.get("ability")
    if not ability:
        return out
    kind = classify_active_ability(ability)
    out[int(AbilityEffectKind(kind))] = 1.0
    return out


def observation_to_features_v3_7(
    observation: dict[str, Any], ablations: set[FeatureAblation] | None = None
) -> np.ndarray:
    """v37-combat-arith-and-catalog: 296-d. Layered on top of v3.6 —
    calls `observation_to_features_v3_6(...)` to seed slots [0:246]
    (byte-stable), then writes the 50-bit combat-arith + catalog tail at
    [246:296] per the LOCKED layout in scoping §4 step 3 / §4.5.

    All v3.7 channels derive from existing v3.6 obs fields + static
    catalog lookup (`shared/src/data/cards.json` via `_get_card`). No
    new obs-contract fields per §13 recon."""

    base = observation_to_features_v3_6(observation, ablations=ablations)
    assert base.shape == (STATE_DIM_V3_6,), (
        f"v3.7 base reuse expected ({STATE_DIM_V3_6},), got {base.shape}"
    )

    features = np.zeros(STATE_DIM_V3_7, dtype=np.float32)
    features[0:STATE_DIM_V3_6] = base

    own = observation.get("own", {}) or {}
    opponent = observation.get("opponent", {}) or {}
    own_active = own.get("active") if isinstance(own, dict) else None
    opp_active = opponent.get("active") if isinstance(opponent, dict) else None

    # Channel 1 — weakness-adjusted lethal. `own_weakness_lethal` means
    # OPP can KO OWN at face-value-plus-weakness (mirrors v3.6
    # `own_lethal_next_turn` polarity).
    features[_V37_OWN_WEAKNESS_LETHAL] = _v37_weakness_adjusted_lethal(
        attacker_active=opp_active, defender_active=own_active
    )
    features[_V37_OPP_WEAKNESS_LETHAL] = _v37_weakness_adjusted_lethal(
        attacker_active=own_active, defender_active=opp_active
    )

    # Channel 1 secondary — predicated on v3.6 slots [242] (own
    # secondary usable) and [244] (opp secondary usable). v3.6 secondary
    # bits are computed from "own_active attacks opp_active" semantics
    # (so [242] = own.active.attacks[1] usable, [244] = opp.active
    # ditto). The v3.7 secondary-KO predicate uses the SAME polarity as
    # the lethal bit above: `own_weakness_secondary_KO` = OPP secondary
    # KOs OWN, so it must read v3.6 slot [244] (opp secondary usable),
    # not [242].
    own_weakness_sec = _v37_weakness_adjusted_secondary_ko(
        attacker_active=opp_active,
        defender_active=own_active,
        secondary_usable_bit=float(base[_V36_OPP_SECONDARY_USABLE_SLOT]),
    )
    opp_weakness_sec = _v37_weakness_adjusted_secondary_ko(
        attacker_active=own_active,
        defender_active=opp_active,
        secondary_usable_bit=float(base[_V36_OWN_SECONDARY_USABLE_SLOT]),
    )
    features[_V37_OWN_WEAKNESS_SECONDARY_KO] = own_weakness_sec
    features[_V37_OPP_WEAKNESS_SECONDARY_KO] = opp_weakness_sec

    # Channel 2 — coin-flip indicators. The "own_primary_*" bits read
    # OWN's primary attack on OWN's active (mirrors action-side
    # polarity: "what coin-flips MY active has"). Layout: own_primary
    # has_cf, cf_eko; opp_primary has_cf, cf_eko; own_sec has_cf,
    # cf_eko; opp_sec has_cf, cf_eko.
    own_attacks = _v36_active_attacks(own_active)
    opp_attacks = _v36_active_attacks(opp_active)
    own_hp = _v36_remaining_hp(own_active)
    opp_hp = _v36_remaining_hp(opp_active)
    own_primary_attack = own_attacks[0] if own_attacks else None
    opp_primary_attack = opp_attacks[0] if opp_attacks else None
    own_sec_attack = own_attacks[1] if len(own_attacks) >= 2 else None
    opp_sec_attack = opp_attacks[1] if len(opp_attacks) >= 2 else None

    own_p_has_cf, own_p_eko = _v37_attack_coin_flip_bits(own_primary_attack, opp_hp)
    opp_p_has_cf, opp_p_eko = _v37_attack_coin_flip_bits(opp_primary_attack, own_hp)
    own_s_has_cf, own_s_eko = _v37_attack_coin_flip_bits(own_sec_attack, opp_hp)
    opp_s_has_cf, opp_s_eko = _v37_attack_coin_flip_bits(opp_sec_attack, own_hp)
    features[_V37_COIN_FLIP_BASE + 0] = own_p_has_cf
    features[_V37_COIN_FLIP_BASE + 1] = own_p_eko
    features[_V37_COIN_FLIP_BASE + 2] = opp_p_has_cf
    features[_V37_COIN_FLIP_BASE + 3] = opp_p_eko
    features[_V37_COIN_FLIP_BASE + 4] = own_s_has_cf
    features[_V37_COIN_FLIP_BASE + 5] = own_s_eko
    features[_V37_COIN_FLIP_BASE + 6] = opp_s_has_cf
    features[_V37_COIN_FLIP_BASE + 7] = opp_s_eko

    # Channel 3 — conditional damage bonuses. Same own-active-primary
    # / opp-active-primary / own-secondary / opp-secondary polarity.
    own_p_pe, own_p_pb = _v37_attack_conditional_bonus_bits(own_primary_attack)
    opp_p_pe, opp_p_pb = _v37_attack_conditional_bonus_bits(opp_primary_attack)
    own_s_pe, own_s_pb = _v37_attack_conditional_bonus_bits(own_sec_attack)
    opp_s_pe, opp_s_pb = _v37_attack_conditional_bonus_bits(opp_sec_attack)
    features[_V37_COND_BONUS_BASE + 0] = own_p_pe
    features[_V37_COND_BONUS_BASE + 1] = own_p_pb
    features[_V37_COND_BONUS_BASE + 2] = opp_p_pe
    features[_V37_COND_BONUS_BASE + 3] = opp_p_pb
    features[_V37_COND_BONUS_BASE + 4] = own_s_pe
    features[_V37_COND_BONUS_BASE + 5] = own_s_pb
    features[_V37_COND_BONUS_BASE + 6] = opp_s_pe
    features[_V37_COND_BONUS_BASE + 7] = opp_s_pb

    # Channel 4 — energy ETA. Each side's own ETA reads that side's own
    # active + own energy_pool.
    own_pool = own.get("energyPool") or []
    opp_pool = opponent.get("energyPool") or []
    features[_V37_ETA_BASE + 0] = _v37_attack_usable_next_turn(
        own_active, own_primary_attack, own_pool
    )
    features[_V37_ETA_BASE + 1] = _v37_attack_usable_next_turn(
        own_active, own_sec_attack, own_pool
    )
    features[_V37_ETA_BASE + 2] = _v37_attack_usable_next_turn(
        opp_active, opp_primary_attack, opp_pool
    )
    features[_V37_ETA_BASE + 3] = _v37_attack_usable_next_turn(
        opp_active, opp_sec_attack, opp_pool
    )

    # Channel 4 paralysis. `own_paralysis_window_open` = 1 iff OPP
    # active is paralysis_recovery_pending. Symmetric for opp.
    features[_V37_PARALYSIS_BASE + 0] = _v37_paralysis_window_open(opp_active)
    features[_V37_PARALYSIS_BASE + 1] = _v37_paralysis_window_open(own_active)

    # Channel 5 — tool effect-kind one-hot.
    features[_V37_TOOL_KIND_OWN_BASE : _V37_TOOL_KIND_OWN_BASE + 4] = (
        _v37_tool_kind_one_hot(own_active)
    )
    features[_V37_TOOL_KIND_OPP_BASE : _V37_TOOL_KIND_OPP_BASE + 4] = (
        _v37_tool_kind_one_hot(opp_active)
    )

    # Channel 6 — ability effect-kind one-hot.
    features[_V37_ABILITY_KIND_OWN_BASE : _V37_ABILITY_KIND_OWN_BASE + 8] = (
        _v37_ability_kind_one_hot(own_active)
    )
    features[_V37_ABILITY_KIND_OPP_BASE : _V37_ABILITY_KIND_OPP_BASE + 8] = (
        _v37_ability_kind_one_hot(opp_active)
    )

    assert features.shape == (STATE_DIM_V3_7,), (
        f"observation_to_features_v3_7 emitted {features.shape}, "
        f"expected ({STATE_DIM_V3_7},)."
    )
    return features


# ---------------------------------------------------------------------------
# v3.8 slim-feature-add tail
# (`docs/ai-research/scoping/v38-slim-feature-add-scoping.md`).
#
# Layout (FROZEN; mirrored bit-for-bit in
# `engine-rs/crates/engine/src/policy/featurize.rs`):
#   [296:299] own.bench[0..2] primary-attack usable-next-turn (3 bits).
#             Same structural-feasibility predicate as v3.7 Ch.4 ETA
#             (`_v37_attack_usable_next_turn`) applied to `bench[i]` with
#             `side.energyPool` as the +1 attach budget pool. 0 if bench
#             slot is absent / non-Uma / has no attack.
#   [299:302] opp.bench[0..2] primary-attack usable-next-turn (3 bits).
#             Symmetric.
#   [302]     own_lose_if_opp_gusts_weakest_bench.
#             1 iff ALL of:
#               (a) opp could play a gust this opp turn — proxied by
#                   public state: gust-trainer appears in opp.discard
#                   (i.e. opp has demonstrated they hold/play gusts) AND
#                   opp.usedSupporterThisTurn == False AND
#                   opp.handCount > 0 (per scoping §13.4 adapted
#                   predicate — opp.handCardIds are private, so the
#                   "opp has gust in hand" check uses public proxies).
#               (b) min(bench HP - damage) ≤ opp.active.attacks[0]
#                   damage, weakness-adjusted via v3.7 Ch.1 formula
#                   (additive bonus when attacker.type ==
#                   defender.weakness.type AND damage > 0).
#               (c) own.points >= 2  (taking one more prize loses;
#                   `remaining = clamp(3 - points, 0, 3) ≤ 1`).
#   [303]     own_can_gust_win_prize_race.
#             Symmetric: 1 iff ALL of (a') own has gust in own.hand
#             (own.handCardIds IS public — slot uses the existing
#             `_hand_role_features` gust-count derivation directly),
#             (b') ANY opp.bench[i] has hp - damage ≤ own.active
#             attacks[0] damage weakness-adjusted, (c') opp.points >= 2.
#
# Defaults: all 8 bits = 0 when the predicate's prerequisite state is
# absent (e.g. bench slot empty, opp.active absent, no gust signal,
# prize state not in winning range).
# ---------------------------------------------------------------------------

_V38_TAIL_START = 296
_V38_OWN_BENCH_ETA_BASE = 296   # +0 own.bench[0], +1 bench[1], +2 bench[2]
_V38_OPP_BENCH_ETA_BASE = 299   # +0 opp.bench[0], +1 bench[1], +2 bench[2]
_V38_OWN_LOSE_IF_OPP_GUSTS = 302
_V38_OWN_GUST_WIN_RACE = 303


def _v38_bench_primary_eta_bits(side: dict[str, Any]) -> np.ndarray:
    """3-bit vector: per-bench primary-attack usable-next-turn for the
    first 3 bench slots. Reuses the v3.7 Ch.4 structural-feasibility
    helper (`_v37_attack_usable_next_turn`) applied to each bench[i].

    The next-turn attach budget pool is `side.energyPool` (same source
    the v3.7 active-side ETA uses)."""

    out = np.zeros(3, dtype=np.float32)
    bench = side.get("bench") or []
    pool = side.get("energyPool") or []
    for i in range(3):
        if i >= len(bench):
            continue
        entry = bench[i]
        if not entry:
            continue
        attacks = _v36_active_attacks(entry)
        primary = attacks[0] if attacks else None
        if primary is None:
            continue
        out[i] = _v37_attack_usable_next_turn(entry, primary, pool)
    return out


def _v38_opp_gust_playable_proxy(opponent: dict[str, Any]) -> bool:
    """Public-info proxy for "opp could play a gust this opp turn."

    Adapted per scoping §13.4: opp.handCardIds is PRIVATE, so we can't
    read the opp hand-role gust bit directly. Instead we require:
      - opp has demonstrated gust play historically — any gust-trainer
        appears in opp.discard (catalog flags `gustOpponent` or
        `discardRandomOpponentActiveEnergy`); AND
      - opp.usedSupporterThisTurn == False (supporter slot still open
        for opp's next turn-start, gust trainers are typically
        supporters); AND
      - opp.handCount > 0 (opp has SOMETHING in hand — necessary but
        not sufficient).

    Strictly weaker than "opp has gust in hand right now" — this is
    public-info-only fallback per scoping §13.4 documentation. Returns
    bool (False if any prereq fails)."""

    if bool(opponent.get("usedSupporterThisTurn", False)):
        return False
    if int(opponent.get("handCount", 0) or 0) <= 0:
        return False
    for raw_id in opponent.get("discard") or []:
        card = _get_card(str(raw_id))
        if not card:
            continue
        if card.get("kind") != "trainer":
            continue
        effect = card.get("effect") or {}
        if effect.get("gustOpponent") or effect.get(
            "discardRandomOpponentActiveEnergy"
        ):
            return True
    return False


def _v38_own_has_gust_in_hand(own: dict[str, Any]) -> bool:
    """True iff `own.handCardIds` contains any trainer with `gustOpponent`
    or `discardRandomOpponentActiveEnergy`. Reads the public own hand
    (handCardIds is exposed only for own side, mirroring the v3.7
    `_hand_role_features` gust-bit semantics at slot [75])."""

    for raw_id in own.get("handCardIds") or []:
        card = _get_card(str(raw_id))
        if not card:
            continue
        if card.get("kind") != "trainer":
            continue
        effect = card.get("effect") or {}
        if effect.get("gustOpponent") or effect.get(
            "discardRandomOpponentActiveEnergy"
        ):
            return True
    return False


def _v38_weakness_adjusted_damage(
    attacker_active: dict[str, Any] | None,
    defender_card: dict[str, Any] | None,
) -> float:
    """Weakness-adjusted face-value damage of `attacker.active.attacks[0]`
    against `defender_card`. Mirrors v3.7 Ch.1 formula:
      damage = attack.damage
      if damage > 0 and attacker.type == defender.weakness.type:
        damage += defender.weakness.amount
    Returns 0.0 if attacker / attack / defender data is absent."""

    if not attacker_active or not defender_card:
        return 0.0
    attacks = _v36_active_attacks(attacker_active)
    if not attacks:
        return 0.0
    primary = attacks[0]
    damage = _v36_attack_base_damage(primary)
    if damage > 0:
        attacker_card = _get_card(str(attacker_active.get("cardId", "") or ""))
        attacker_type = _v37_card_type(attacker_card)
        weakness = (defender_card or {}).get("weakness") or {}
        weakness_type = str(weakness.get("type", "") or "")
        if attacker_type and weakness_type and attacker_type == weakness_type:
            try:
                damage += float(weakness.get("amount", 0) or 0)
            except (TypeError, ValueError):
                pass
    return damage


def _v38_min_bench_remaining_hp(side: dict[str, Any]) -> tuple[float, dict[str, Any] | None]:
    """Returns (min_hp_among_present_bench, the_card_entry). If no bench
    is present, returns (inf, None). Uses `_v36_remaining_hp` so the
    semantics match every other v3.x HP read."""

    best_hp = float("inf")
    best_entry: dict[str, Any] | None = None
    for entry in side.get("bench") or []:
        if not entry:
            continue
        hp = _v36_remaining_hp(entry)
        if hp < best_hp:
            best_hp = hp
            best_entry = entry
    return best_hp, best_entry


def _v38_any_bench_ko_able(
    attacker_active: dict[str, Any] | None,
    defender_side: dict[str, Any],
) -> bool:
    """True iff any present bench on `defender_side` has remaining HP
    ≤ attacker.active.attacks[0] damage (weakness-adjusted per bench
    card's weakness type). Used by [303] own_can_gust_win_prize_race."""

    if not attacker_active:
        return False
    for entry in defender_side.get("bench") or []:
        if not entry:
            continue
        card = _get_card(str(entry.get("cardId", "") or ""))
        damage = _v38_weakness_adjusted_damage(attacker_active, card)
        if damage <= 0.0:
            continue
        hp = _v36_remaining_hp(entry)
        if hp <= damage:
            return True
    return False


def _v38_remaining_prizes(side: dict[str, Any]) -> int:
    """`remaining = clamp(3 - points, 0, 3)` mirror of v3.6 prize one-hot."""

    points = int(side.get("points", 0) or 0)
    return max(0, min(3, 3 - points))


def _v38_lose_if_opp_gusts_weakest_bench(
    own: dict[str, Any],
    opponent: dict[str, Any],
) -> float:
    """[302] predicate. Adapted per scoping §13.4 because opp.handCardIds
    is private — we substitute a public-info proxy for "opp has gust"."""

    # (c) own.points >= 2 → remaining prize ≤ 1 → losing one more prize
    # ends the game.
    if _v38_remaining_prizes(own) > 1:
        return 0.0
    # (a) Public proxy for "opp could play a gust this opp turn."
    if not _v38_opp_gust_playable_proxy(opponent):
        return 0.0
    # (b) Weakest own bench KO-able by opp.active primary, weakness-
    # adjusted per the weakest-bencher's own weakness type.
    opp_active = opponent.get("active") if isinstance(opponent, dict) else None
    if not opp_active:
        return 0.0
    weakest_hp, weakest_entry = _v38_min_bench_remaining_hp(own)
    if weakest_entry is None:
        return 0.0
    weakest_card = _get_card(str(weakest_entry.get("cardId", "") or ""))
    damage = _v38_weakness_adjusted_damage(opp_active, weakest_card)
    if damage <= 0.0:
        return 0.0
    return 1.0 if weakest_hp <= damage else 0.0


def _v38_can_gust_win_prize_race(
    own: dict[str, Any],
    opponent: dict[str, Any],
) -> float:
    """[303] predicate. Symmetric to [302] but on the WINNING side: own
    has gust in hand (own.handCardIds is PUBLIC — direct lookup), any
    opp bench is KO-able by own.active primary, and opp is at one
    prize from losing."""

    # (c') opp.points >= 2 → remaining prize ≤ 1.
    if _v38_remaining_prizes(opponent) > 1:
        return 0.0
    # (a') Direct read of own.handCardIds for gust trainers.
    if not _v38_own_has_gust_in_hand(own):
        return 0.0
    # (b') Any opp bench KO-able by own active primary.
    own_active = own.get("active") if isinstance(own, dict) else None
    if not own_active:
        return 0.0
    if not _v38_any_bench_ko_able(own_active, opponent):
        return 0.0
    return 1.0


def observation_to_features_v3_8(
    observation: dict[str, Any], ablations: set[FeatureAblation] | None = None
) -> np.ndarray:
    """v38-slim-feature-add: 304-d. Layered on top of v3.7 — calls
    `observation_to_features_v3_7(...)` to seed slots [0:296]
    (byte-stable), then writes the 8-bit slim tail at [296:304] per the
    LOCKED layout in scoping §4.5.

    All v3.8 bits derive from existing v3.7 obs fields + static catalog
    lookup. NO new obs-contract fields. Gust-swing catastrophe uses a
    public-info-only proxy for opp's gust availability (scoping §13.4)
    because opp.handCardIds is private."""

    base = observation_to_features_v3_7(observation, ablations=ablations)
    assert base.shape == (STATE_DIM_V3_7,), (
        f"v3.8 base reuse expected ({STATE_DIM_V3_7},), got {base.shape}"
    )

    features = np.zeros(STATE_DIM_V3_8, dtype=np.float32)
    features[0:STATE_DIM_V3_7] = base

    own = observation.get("own", {}) or {}
    opponent = observation.get("opponent", {}) or {}

    # [296:299] own bench primary ETA bits.
    own_bench_eta = _v38_bench_primary_eta_bits(own)
    features[_V38_OWN_BENCH_ETA_BASE : _V38_OWN_BENCH_ETA_BASE + 3] = own_bench_eta
    # [299:302] opp bench primary ETA bits.
    opp_bench_eta = _v38_bench_primary_eta_bits(opponent)
    features[_V38_OPP_BENCH_ETA_BASE : _V38_OPP_BENCH_ETA_BASE + 3] = opp_bench_eta

    # [302] own_lose_if_opp_gusts_weakest_bench.
    features[_V38_OWN_LOSE_IF_OPP_GUSTS] = _v38_lose_if_opp_gusts_weakest_bench(
        own, opponent
    )
    # [303] own_can_gust_win_prize_race.
    features[_V38_OWN_GUST_WIN_RACE] = _v38_can_gust_win_prize_race(own, opponent)

    assert features.shape == (STATE_DIM_V3_8,), (
        f"observation_to_features_v3_8 emitted {features.shape}, "
        f"expected ({STATE_DIM_V3_8},)."
    )
    return features


# Builder selector keyed off the state dim. Mirrors the existing serve_onnx
# `_SCHEMA_BY_STATE_DIM` discrimination (graph dim -> builder) so training /
# dataset code can opt into v3.1/v3.3/v3.5/v3.6/v3.7 without a new framework:
# pass the state dim and get the matching frozen builder. v3.0 (110) stays
# the default everywhere `STATE_DIM` is referenced.
_BUILDER_BY_STATE_DIM = {
    STATE_DIM_V2: observation_to_features_v2,
    STATE_DIM_V3: observation_to_features,
    STATE_DIM_V3_1: observation_to_features_v3_1,
    STATE_DIM_V3_3: observation_to_features_v3_3,
    STATE_DIM_V3_5: observation_to_features_v3_5,
    STATE_DIM_V3_6: observation_to_features_v3_6,
    STATE_DIM_V3_7: observation_to_features_v3_7,
    STATE_DIM_V3_8: observation_to_features_v3_8,
}


_SCHEMA_VERSION_BY_STATE_DIM = {
    STATE_DIM_V2: STATE_FEATURE_SCHEMA_VERSION_V2,
    STATE_DIM_V3: STATE_FEATURE_SCHEMA_VERSION_V3,
    STATE_DIM_V3_1: STATE_FEATURE_SCHEMA_VERSION_V3_1,
    STATE_DIM_V3_3: STATE_FEATURE_SCHEMA_VERSION_V3_3,
    STATE_DIM_V3_5: STATE_FEATURE_SCHEMA_VERSION_V3_5,
    STATE_DIM_V3_6: STATE_FEATURE_SCHEMA_VERSION_V3_6,
    STATE_DIM_V3_7: STATE_FEATURE_SCHEMA_VERSION_V3_7,
    STATE_DIM_V3_8: STATE_FEATURE_SCHEMA_VERSION_V3_8,
}


def feature_builder_for_state_dim(state_dim: int):
    """Return the frozen observation->features builder for a state dim.

    96 -> v2, 110 -> v3.0, 164 -> v3.1, 167 -> v3.3, 212 -> v3.5,
    246 -> v3.6, 296 -> v3.7, 304 -> v3.8. Unknown dims raise (never
    silently fall back) — same fail-loud contract the serve_onnx guard
    enforces."""

    try:
        return _BUILDER_BY_STATE_DIM[state_dim]
    except KeyError as exc:
        known = ", ".join(str(d) for d in sorted(_BUILDER_BY_STATE_DIM))
        raise ValueError(
            f"no feature builder for state_dim={state_dim} "
            f"(known: {known}). A new schema must add a NEW builder + "
            f"constant, never mutate a frozen one."
        ) from exc


def schema_version_for_state_dim(state_dim: int) -> float:
    """The state-feature schema version a given state dim actually emits.

    Manifest/metadata writers pair this with the run's state dim so a 110-d
    v3.0 run is tagged 3.0 even though the module's latest-schema constant
    (`STATE_FEATURE_SCHEMA_VERSION`) is 3.1. Unknown dims raise."""

    try:
        return _SCHEMA_VERSION_BY_STATE_DIM[state_dim]
    except KeyError as exc:
        known = ", ".join(str(d) for d in sorted(_SCHEMA_VERSION_BY_STATE_DIM))
        raise ValueError(
            f"no schema version for state_dim={state_dim} (known: {known})."
        ) from exc


# ---------------------------------------------------------------------------
# R16-P2 C1: per-Uma slot-token feature builder (additive, no v3.0/v3.1 churn).
#
# This is the foundation chunk for r16-p2-per-uma-slot-tokens (8-chunk
# initiative, kickoff 2026-05-21). v3.2 keeps STATE_DIM=110 — it is
# distinguished from v3.0/v3.1 by ONNX input-set, not state-vector width. The
# per-Uma slot tensors are a NEW auxiliary tensor pair consumed alongside the
# existing 110-d state vector by a future `uma_slot_encoder` branch (C2).
#
# UMA_SLOT_ORDER is a FROZEN enumeration. The 4th bench slot per side
# (`own_bench_3`, `opp_bench_3`) is currently always absent in TS observations
# because `MAX_BENCH = 3` in `shared/src/gameData.ts`; the slot is reserved
# so that the layout stays byte-stable if MAX_BENCH ever grows. This mirrors
# the existing 4-bench-slot over-provisioning in `_identity_features` /
# `_tool_card_features` (slots 32–48 and 100–110 of the v3.0 builder both
# already iterate `bench[:4]`). Absent slots emit card_id=0 and an all-zero
# feature row — the same "absence is zero" contract that drives v3.0
# `card_ids_by_zone` padding (index 0 = padding_idx in the card vocab).
UMA_SLOT_ORDER: tuple[str, ...] = (
    "own_active",
    "own_bench_0",
    "own_bench_1",
    "own_bench_2",
    "own_bench_3",
    "opp_active",
    "opp_bench_0",
    "opp_bench_1",
    "opp_bench_2",
    "opp_bench_3",
)
UMA_SLOT_COUNT = len(UMA_SLOT_ORDER)  # = 10 (2 active + 8 bench placeholders)

# UMA_SLOT_FEATURE_DIM = 23 — FROZEN this commit. Downstream chunks (C2 model
# branch, C4 dataset packing, C5 ONNX graph) depend on this width being
# stable. Per-slot layout (in column order; helper indices match):
#
#   [ 0]  polarity                 own=+1, opp=-1  (mirrors slot 96 v2.1)
#   [ 1]  role_active              active=1, bench=0
#   [ 2]  slot_idx_norm            active=-1; bench i ∈ {0..MAX_BENCH-1}
#                                  encoded as i / max(1, MAX_BENCH-1) so
#                                  the 4th reserved bench placeholder
#                                  (index 3) stays in [0,1] under MAX_BENCH=3
#                                  by clamping to 1.0 if ever instantiated.
#   [ 3]  present_mask             1 if the slot is occupied, else 0
#   [ 4]  hp_norm                  hp / max(1, maxHp), clipped [0, 1]
#   [ 5]  damage_norm              (maxHp - hp) / max(1, maxHp), clipped
#   [ 6]  stage_norm               stage / 2.0
#   [ 7]  energyTotal_norm         energyTotal / 6.0
#   [ 8:18]  typed_energy ×10      grass, fire, water, lightning, psychic,
#                                  fighting, darkness, steel, colorless,
#                                  dragon (each / 4.0; order matches
#                                  `_energy_vector` and EnergyType in
#                                  `shared/src/types.ts:5`)
#   [18]  tool_attached            1 if toolCardId is set, else 0
#   [19]  condition_paralysis      1 if "paralysed" ∈ specialConditions
#                                  (legality-affecting: attack-blocking)
#   [20]  condition_count_norm     |specialConditions| / 5.0  (5 = the full
#                                  SpecialCondition union: asleep/burned/
#                                  frozen/paralysed/poisoned)
#   [21]  ability_used_this_turn   1 if usedAbilityThisTurn, else 0
#   [22]  evolved                  1 if stage > 0, else 0  (hard-threshold
#                                  redundant-with-stage_norm bit, useful for
#                                  "is this an evolution stage Uma?"
#                                  predicates that the network would
#                                  otherwise have to learn from stage_norm)
#
# Deviation from the chunk-plan recommendation (F=23 enumeration in
# docs/ai-research/scoping/r16-model-feature-backlog-refinement.md §
# "P2 - Chunk Plan"): the enumerated list there sums to 22, not 23. I added
# `condition_count_norm` (slot 20) to make the count match while keeping
# the legality-load-bearing `condition_paralysis` bit. Documented here so
# C2/C4/C5 freeze against the 23-wide layout.
UMA_SLOT_FEATURE_DIM = 23
STATE_FEATURE_SCHEMA_VERSION_V3_2 = 3.2

# Internal feature-index constants (used by both the builder and any future
# C2 model-side mask / ablation logic). Keep these as module-level names so
# external consumers (e.g. zone-zeroing in C2) can reference them by name
# instead of magic numbers.
_UMA_SLOT_F_POLARITY = 0
_UMA_SLOT_F_ROLE_ACTIVE = 1
_UMA_SLOT_F_SLOT_IDX = 2
_UMA_SLOT_F_PRESENT = 3
_UMA_SLOT_F_HP = 4
_UMA_SLOT_F_DAMAGE = 5
_UMA_SLOT_F_STAGE = 6
_UMA_SLOT_F_ENERGY_TOTAL = 7
_UMA_SLOT_F_ENERGY_TYPED = slice(8, 18)
_UMA_SLOT_F_TOOL = 18
_UMA_SLOT_F_COND_PARALYSIS = 19
_UMA_SLOT_F_COND_COUNT = 20
_UMA_SLOT_F_ABILITY_USED = 21
_UMA_SLOT_F_EVOLVED = 22

# Per-Uma typed-energy order — MUST match `_energy_vector` (slots 48–58 of
# v3.0) and the EnergyType union in `shared/src/types.ts:5`. Asserted at
# import time so a future re-order of `_energy_vector` does not silently
# desync the per-slot encoding.
_UMA_SLOT_ENERGY_TYPES: tuple[str, ...] = (
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
)
assert len(_UMA_SLOT_ENERGY_TYPES) == (
    _UMA_SLOT_F_ENERGY_TYPED.stop - _UMA_SLOT_F_ENERGY_TYPED.start
), "uma slot typed-energy width must match the 10-wide energy slice"

# Bench placeholder cap = 4 (matches existing v3.0 `bench[:4]` slicing in
# `_identity_features` and `_tool_card_features`). The real MAX_BENCH is 3
# today (`shared/src/gameData.ts:7`); the 4th slot is reserved.
_UMA_SLOT_BENCH_PER_SIDE = 4


def _uma_slot_feature_row(
    uma: dict[str, Any] | None,
    *,
    polarity: float,
    role_active: bool,
    slot_idx_norm: float,
) -> np.ndarray:
    """Emit one [UMA_SLOT_FEATURE_DIM]-wide feature row for a single slot.

    Absent slot (`uma is None` or `card_id` empty) → all zeros (polarity and
    role bits zeroed too, so the present-mask is the sole "this slot exists"
    signal and downstream pooling on absent slots is exactly null).
    """

    row = np.zeros(UMA_SLOT_FEATURE_DIM, dtype=np.float32)
    if not uma:
        return row
    card_id = str(uma.get("cardId") or "")
    if not card_id:
        # Treat a missing/blank cardId as an absent slot — consistent with
        # the card-vocab `padding_idx=0` convention.
        return row

    row[_UMA_SLOT_F_POLARITY] = polarity
    row[_UMA_SLOT_F_ROLE_ACTIVE] = 1.0 if role_active else 0.0
    row[_UMA_SLOT_F_SLOT_IDX] = slot_idx_norm
    row[_UMA_SLOT_F_PRESENT] = 1.0

    max_hp = max(1.0, float(uma.get("maxHp", 0) or 0))
    hp = float(uma.get("hp", 0) or 0)
    row[_UMA_SLOT_F_HP] = max(0.0, min(1.0, hp / max_hp))
    row[_UMA_SLOT_F_DAMAGE] = max(0.0, min(1.0, (max_hp - hp) / max_hp))
    row[_UMA_SLOT_F_STAGE] = float(uma.get("stage", 0) or 0) / 2.0
    row[_UMA_SLOT_F_ENERGY_TOTAL] = float(uma.get("energyTotal", 0) or 0) / 6.0

    energies = uma.get("energies") or {}
    for offset, energy_type in enumerate(_UMA_SLOT_ENERGY_TYPES):
        row[_UMA_SLOT_F_ENERGY_TYPED.start + offset] = (
            float(energies.get(energy_type, 0) or 0) / 4.0
        )

    row[_UMA_SLOT_F_TOOL] = 1.0 if uma.get("toolCardId") else 0.0

    conditions = uma.get("specialConditions") or []
    row[_UMA_SLOT_F_COND_PARALYSIS] = 1.0 if "paralysed" in conditions else 0.0
    # Full SpecialCondition union has 5 members (asleep/burned/frozen/
    # paralysed/poisoned per shared/src/types.ts:10), so divide by 5.0.
    row[_UMA_SLOT_F_COND_COUNT] = float(len(conditions)) / 5.0

    row[_UMA_SLOT_F_ABILITY_USED] = 1.0 if uma.get("usedAbilityThisTurn") else 0.0
    row[_UMA_SLOT_F_EVOLVED] = 1.0 if float(uma.get("stage", 0) or 0) > 0 else 0.0

    return row


def observation_to_uma_slots(
    observation: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """R16-P2 C1: extract per-Uma slot tokens from a `PublicObservation`.

    Returns `(card_ids: int64[UMA_SLOT_COUNT], features:
    float32[UMA_SLOT_COUNT, UMA_SLOT_FEATURE_DIM])`. The slot iteration order
    is FROZEN in `UMA_SLOT_ORDER` — index 0 is own active, indices 1..4 are
    own bench 0..3, index 5 is opp active, indices 6..9 are opp bench 0..3.

    Absent slots (engine `bench[i]` is `null`, side has no `active`, or
    `cardId` is empty) emit `card_id=0` and an all-zero feature row. This
    matches the existing v3.0 `cardIdsByZone` convention (index 0 doubles as
    `unknownIndex` / `padding_idx`).

    Fail-loud guard (mirroring `observation_to_card_ids` ~L478): if the
    observation lacks the v3-era nested `own` / `opponent` side dicts, raise
    `ValueError`. The PublicObservation schema has carried these since v1
    (predates the R7.b.2 v3 bump), so a missing key indicates the caller is
    feeding a non-`PublicObservation` payload, not just an old corpus.
    """

    if "own" not in observation or "opponent" not in observation:
        raise ValueError(
            "observation_to_uma_slots: observation is missing 'own' and/or "
            "'opponent' side dicts — payload is not a PublicObservation. "
            "The per-Uma slot builder requires the nested side schema."
        )

    own = observation.get("own") or {}
    opponent = observation.get("opponent") or {}

    card_ids = np.zeros(UMA_SLOT_COUNT, dtype=np.int64)
    features = np.zeros((UMA_SLOT_COUNT, UMA_SLOT_FEATURE_DIM), dtype=np.float32)

    # Active slots: own at slot 0, opp at slot 5. slot_idx_norm=-1.0 marks
    # the active role (per chunk-plan recommendation).
    for side_dict, polarity, slot_idx in (
        (own, 1.0, 0),
        (opponent, -1.0, 5),
    ):
        active = side_dict.get("active")
        if active:
            card_ids[slot_idx] = card_vocab_index(str(active.get("cardId") or ""))
            features[slot_idx] = _uma_slot_feature_row(
                active,
                polarity=polarity,
                role_active=True,
                slot_idx_norm=-1.0,
            )

    # Bench slots: own bench i at slot 1+i, opp bench i at slot 6+i.
    # bench_idx_norm = i / max(1, _UMA_SLOT_BENCH_PER_SIDE - 1) (= i/3 for
    # the 4-slot placeholder; the 4th slot is always absent under
    # MAX_BENCH=3 today). Empty bench entries (null) → zero card_id + row.
    bench_denom = max(1.0, float(_UMA_SLOT_BENCH_PER_SIDE - 1))
    for side_dict, polarity, slot_base in (
        (own, 1.0, 1),
        (opponent, -1.0, 6),
    ):
        bench = side_dict.get("bench") or []
        for bench_pos in range(_UMA_SLOT_BENCH_PER_SIDE):
            entry = bench[bench_pos] if bench_pos < len(bench) else None
            if not entry:
                continue
            card_ids[slot_base + bench_pos] = card_vocab_index(
                str(entry.get("cardId") or "")
            )
            features[slot_base + bench_pos] = _uma_slot_feature_row(
                entry,
                polarity=polarity,
                role_active=False,
                slot_idx_norm=float(bench_pos) / bench_denom,
            )

    # Frozen-shape guards (paranoia parity with the v3.0 / v3.1 builder
    # asserts — any future refactor that perturbs the slot count or
    # feature width fails here before reaching downstream collators / ONNX).
    assert card_ids.shape == (UMA_SLOT_COUNT,), (
        f"observation_to_uma_slots: card_ids shape {card_ids.shape} != "
        f"({UMA_SLOT_COUNT},). UMA_SLOT_ORDER layout is frozen."
    )
    assert features.shape == (UMA_SLOT_COUNT, UMA_SLOT_FEATURE_DIM), (
        f"observation_to_uma_slots: features shape {features.shape} != "
        f"({UMA_SLOT_COUNT}, {UMA_SLOT_FEATURE_DIM}). The 23-wide per-slot "
        f"layout is frozen this commit; downstream chunks depend on it."
    )
    return card_ids, features


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
    # R16-P1 v3.1 ablation hook: zero the 54 temporal/turn-state slots so
    # callers can isolate the temporal signal against the frozen v3.0
    # encoding. No-op on a 96/110-d array (numpy slice past the end is
    # empty) — only bites the 164-d v3.1 vector. Mirrors the
    # `state_hygiene_v21` slice-zero pattern above.
    if "state_temporal_turn_v31" in ablations:
        features[110:164] = 0


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
    # Weakness bonus: simulator applies `damage += defender.weakness.amount`
    # when damage > 0 and defender's printed weakness type matches the
    # attacker's primary type (engine-rs flow/combat.rs:303-305). The
    # featurizer previously ignored this, causing can_ko to under-report
    # KOs on ~30% of matchups (any type-disadvantaged defender).
    if damage > 0:
        attacker_card = _get_card(str(attacker.get("cardId", "")))
        defender_card = _get_card(str(defender.get("cardId", "")))
        if attacker_card and defender_card:
            attacker_type = str(attacker_card.get("type", ""))
            weakness = defender_card.get("weakness") or {}
            weakness_type = str(weakness.get("type", ""))
            if attacker_type and weakness_type and attacker_type == weakness_type:
                damage += float(weakness.get("amount", 0))
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
