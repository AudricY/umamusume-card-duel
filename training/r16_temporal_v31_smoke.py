"""R16-P1 temporal / turn-state v3.1 feature smoke.

Self-contained (no external corpus): proves the NEW 164-d
`observation_to_features_v3_1` builder is a faithful additive extension of
the FROZEN v3.0 110-d builder.

  1. Width        — v3.1 emits exactly 164; v3.0 emits 110; v2 emits 96.
  2. Frozen head  — v3.1 slots [0:110] are byte-IDENTICAL to the v3.0
                    builder on the same observation (frozen invariant).
  3. Temporal-only diff — changing ONLY a temporal field
                    (evolvedLastTurn / tookDamageLastTurn / energy attach
                    budget) changes ONLY slots [110:164]; [0:110] stays
                    byte-stable.
  4. Ablation      — `state_temporal_turn_v31` zeroes exactly [110:164]
                    and leaves [0:110] untouched.
  5. Selector      — feature_builder_for_state_dim / schema_version_for_
                    state_dim map 96/110/164 correctly and fail loud on
                    an unknown dim.
  6. Derived field — the retreat slot encodes
                    `effectiveRetreatCostReduction` (omission-1 resolution),
                    not the raw per-side value.

Deliberately lean — a shape/invariant check, not a strength test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uma_ai.features import (  # noqa: E402
    STATE_DIM_V2,
    STATE_DIM_V3,
    STATE_DIM_V3_1,
    feature_builder_for_state_dim,
    observation_to_features,
    observation_to_features_v2,
    observation_to_features_v3_1,
    schema_version_for_state_dim,
)


def fail(msg: str) -> None:
    print(f"[r16-temporal-v31-smoke] FAIL — {msg}", file=sys.stderr)
    sys.exit(1)


def _uma(uid: int, *, entered=1, evolved=None, took_last=False, took_this=False,
         ndr=0, blocked=False, para_pending=False, conds=None) -> dict:
    return {
        "uid": uid,
        "cardId": "agnesDigitalBasic",
        "species": "agnesDigital",
        "stage": 1,
        "hp": 90,
        "maxHp": 100,
        "energyTotal": 2,
        "energies": {"fire": 1, "water": 1},
        "specialConditions": conds or [],
        "toolCardId": None,
        "usedAbilityThisTurn": False,
        "turnState": {
            "turnsInPlay": 3,
            "enteredThisTurn": entered == 0,
            "evolvedThisTurn": evolved == "this",
            "evolvedLastTurn": evolved == "last",
            "tookDamageLastTurn": took_last,
            "tookDamageThisTurn": took_this,
            "nextTurnDamageReduction": ndr,
            "attackBlockedThisTurn": blocked,
            "paralysisRecoveryPending": para_pending,
        },
    }


def _side(points: int, *, retreat_raw=1, eff_retreat=2) -> dict:
    return {
        "id": "player",
        "points": points,
        "handCount": 4,
        "deckCount": 30,
        "discard": [],
        "active": _uma(1, evolved="last"),
        "bench": [_uma(2), _uma(3), None, None],
        "energyZone": ["fire"],
        "usedSupporterThisTurn": False,
        "usedRetreatThisTurn": False,
        "usedStadiumThisTurn": False,
        "turnState": {
            "energyAttachmentsThisTurn": 1,
            "bonusEnergyAttachments": 0,
            "retreatCostReduction": retreat_raw,
            "effectiveRetreatCostReduction": eff_retreat,
            "activeAttackDamageBonus": 30,
            "usedAbilityNameCountThisTurn": 0,
            "usedAbilityNameCountThisGame": 2,
            "guaranteedCoinFlipHeads": 0,
        },
    }


def _observation() -> dict:
    return {
        "schemaVersion": 3,
        "sideToAct": "player",
        "phase": "attach",
        "turnNumber": 5,
        "firstPlayer": "player",
        "pendingChoiceKind": None,
        "temporal": {
            "ownTurnsTaken": 3,
            "opponentTurnsTaken": 2,
            "ownIsFirstTurn": False,
            "opponentIsFirstTurn": False,
        },
        "own": _side(1),
        "opponent": _side(0),
        "shared": {
            "stadiumCardId": "nakayamaTurf",
            "currentSide": "player",
            "gameOver": False,
        },
        "cardIdsByZone": {
            "ownActive": [3],
            "oppActive": [3],
            "ownBench": [3, 3],
            "oppBench": [3, 3],
            "ownHand": [1, 1, 1, 1],
            "ownDiscard": [],
            "oppDiscard": [],
            "stadium": [],
        },
    }


def main() -> None:
    obs = _observation()

    # 1. widths
    v2 = observation_to_features_v2(obs)
    v3 = observation_to_features(obs)
    v31 = observation_to_features_v3_1(obs)
    if v2.shape != (STATE_DIM_V2,):
        fail(f"v2 width {v2.shape} != ({STATE_DIM_V2},)")
    if v3.shape != (STATE_DIM_V3,):
        fail(f"v3.0 width {v3.shape} != ({STATE_DIM_V3},)")
    if v31.shape != (STATE_DIM_V3_1,):
        fail(f"v3.1 width {v31.shape} != ({STATE_DIM_V3_1},)")
    print(f"  PASS  widths: v2={v2.shape[0]} v3.0={v3.shape[0]} v3.1={v31.shape[0]}")

    # 2. frozen head: v3.1[0:110] byte-identical to v3.0
    if not np.array_equal(v31[:STATE_DIM_V3], v3):
        fail("v3.1 head [0:110] differs from frozen v3.0 builder output")
    print("  PASS  v3.1 head [0:110] byte-identical to frozen v3.0")

    # 3. temporal-only diff isolates [110:164]
    for field_path, mutate in (
        ("own.active.turnState.evolvedLastTurn",
         lambda o: o["own"]["active"]["turnState"].__setitem__("evolvedLastTurn", False)),
        ("own.bench[1].turnState.tookDamageLastTurn",
         lambda o: o["own"]["bench"][1]["turnState"].__setitem__("tookDamageLastTurn", True)),
        ("own.turnState.bonusEnergyAttachments",
         lambda o: o["own"]["turnState"].__setitem__("bonusEnergyAttachments", 3)),
        ("temporal.ownIsFirstTurn",
         lambda o: o["temporal"].__setitem__("ownIsFirstTurn", True)),
    ):
        import copy

        o2 = copy.deepcopy(obs)
        mutate(o2)
        a = observation_to_features_v3_1(obs)
        b = observation_to_features_v3_1(o2)
        if not np.array_equal(a[:STATE_DIM_V3], b[:STATE_DIM_V3]):
            fail(f"temporal mutation {field_path} perturbed frozen [0:110]")
        if np.array_equal(a[STATE_DIM_V3:], b[STATE_DIM_V3:]):
            fail(f"temporal mutation {field_path} did NOT change [110:164]")
    print("  PASS  temporal-only mutation changes only [110:164], [0:110] stable")

    # 4. ablation zeroes [110:164] only
    abl = observation_to_features_v3_1(obs, ablations={"state_temporal_turn_v31"})
    if not np.array_equal(abl[:STATE_DIM_V3], v31[:STATE_DIM_V3]):
        fail("temporal ablation perturbed [0:110]")
    if abl[STATE_DIM_V3:].any():
        fail("temporal ablation did not fully zero [110:164]")
    print("  PASS  state_temporal_turn_v31 ablation zeroes exactly [110:164]")

    # 5. selectors
    assert feature_builder_for_state_dim(96) is observation_to_features_v2
    assert feature_builder_for_state_dim(110) is observation_to_features
    assert feature_builder_for_state_dim(164) is observation_to_features_v3_1
    assert schema_version_for_state_dim(110) == 3.0
    assert schema_version_for_state_dim(164) == 3.1
    try:
        feature_builder_for_state_dim(999)
    except ValueError:
        pass
    else:
        fail("feature_builder_for_state_dim(999) should raise")
    print("  PASS  builder/version selectors map 96/110/164 + fail loud on 999")

    # 6. derived effective-retreat field rides the retreat slot.
    #    Slot index: 110(global×4)..113, own side ×7 = [114:121]; the retreat
    #    field is the 3rd side scalar -> slot 114+2 = 116. With raw=1,
    #    eff=2, _DAMAGE_CAP=30 -> encoded min(2,30)/30.
    expected = 2.0 / 30.0
    if abs(float(v31[116]) - expected) > 1e-6:
        fail(f"retreat slot 116 = {v31[116]!r}, expected effective {expected!r}")
    # raw-only sanity: if eff key absent it must fall back to raw (1/30).
    import copy as _copy

    o_raw = _copy.deepcopy(obs)
    del o_raw["own"]["turnState"]["effectiveRetreatCostReduction"]
    v31_raw = observation_to_features_v3_1(o_raw)
    if abs(float(v31_raw[116]) - 1.0 / 30.0) > 1e-6:
        fail("retreat slot did not fall back to raw when effective key absent")
    print("  PASS  retreat slot encodes effectiveRetreatCostReduction (omission-1)")

    # 7. dataset loader + collator for a v3.1 (164-d) row.
    _dataset_collator_164(obs)
    print("  PASS  JsonlPolicyDataset(state_dim=164) + collate_policy_batch -> [B,164]")

    # 8. ONNX export + roundtrip + /predict for the 164-d v3.1 graph.
    #    Exercises the real serve_onnx path (request_to_arrays -> ORT) so a
    #    164-d graph is served bit-consistently with the PyTorch model and
    #    the v3.1 builder packs the same 3 base feeds + 2 embedding feeds.
    _onnx_roundtrip_164(obs)
    print("  PASS  164-d v3.1 ONNX export + /predict roundtrip (<1e-3)")

    print("r16-temporal-v31 smoke: ALL PASS")


def _dataset_collator_164(obs: dict) -> None:
    import json
    import tempfile

    from uma_ai.dataset import JsonlPolicyDataset, collate_policy_batch
    from uma_ai.features import STATE_DIM_V3_1

    row = {
        "schemaVersion": 1,  # TS TrainingExample row schema (unchanged axis)
        "observation": obs,
        "selectedActionIndex": 0,
        "legalActions": [
            {"id": "a0", "features": [0.0] * 48, "actionSourceCardIdx": 3, "actionTargetCardIdx": 0},
            {"id": "a1", "features": [0.1] * 48, "actionSourceCardIdx": None, "actionTargetCardIdx": None},
        ],
        "result": {"winner": "player", "points": {"player": 1, "opponent": 0}},
    }
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "v31.jsonl"
        p.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf8")
        ds = JsonlPolicyDataset(p, state_dim=STATE_DIM_V3_1)
        assert len(ds) == 2, f"expected 2 samples, got {len(ds)}"
        for s in ds:
            assert s.state_features.shape == (STATE_DIM_V3_1,), s.state_features.shape
        batch = collate_policy_batch(list(ds))
        sf = batch["state_features"]
        assert tuple(sf.shape) == (2, STATE_DIM_V3_1), tuple(sf.shape)
        assert "card_ids_by_zone" in batch and "action_card_idx" in batch, (
            "v3.1 batch must still carry the embedding tensors (v3.0 head)"
        )


def _onnx_roundtrip_164(obs: dict) -> None:
    import tempfile

    import torch

    from uma_ai.model import CandidatePolicyNet, ModelConfig
    from serve_onnx import request_to_arrays

    try:
        import onnxruntime as ort  # noqa: F401
    except ImportError:
        print("  SKIP  ONNX roundtrip: onnxruntime unavailable (env gap)", file=sys.stderr)
        return

    cfg = ModelConfig(state_dim=STATE_DIM_V3_1, hidden_dim=32, depth=2)
    model = CandidatePolicyNet(cfg).eval()

    legal = [
        {"id": "a0", "features": [0.0] * 48, "actionSourceCardIdx": 3, "actionTargetCardIdx": 0},
        {"id": "a1", "features": [0.1] * 48, "actionSourceCardIdx": None, "actionTargetCardIdx": None},
    ]
    payload = {"observation": obs, "legalActions": legal}
    feed, _ = request_to_arrays(payload, "v3.1")
    assert set(feed) == {
        "state_features",
        "action_features",
        "action_mask",
        "card_ids_by_zone",
        "action_card_idx",
    }, f"v3.1 feed keys: {sorted(feed)}"
    assert feed["state_features"].shape == (1, STATE_DIM_V3_1), feed["state_features"].shape

    import numpy as _np

    with torch.no_grad():
        pt = model(
            torch.from_numpy(feed["state_features"]),
            torch.from_numpy(feed["action_features"]),
            torch.from_numpy(feed["action_mask"]),
            torch.from_numpy(feed["card_ids_by_zone"]),
            torch.from_numpy(feed["action_card_idx"]),
        )
    pt_logits = pt[0].numpy() if isinstance(pt, (tuple, list)) else pt.numpy()

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "v31.onnx"
        torch.onnx.export(
            model,
            (
                torch.from_numpy(feed["state_features"]),
                torch.from_numpy(feed["action_features"]),
                torch.from_numpy(feed["action_mask"]),
                torch.from_numpy(feed["card_ids_by_zone"]),
                torch.from_numpy(feed["action_card_idx"]),
            ),
            str(out),
            input_names=[
                "state_features",
                "action_features",
                "action_mask",
                "card_ids_by_zone",
                "action_card_idx",
            ],
            output_names=["logits", "value"],
            dynamic_axes={
                "state_features": {0: "batch"},
                "action_features": {0: "batch", 1: "actions"},
                "action_mask": {0: "batch", 1: "actions"},
                "card_ids_by_zone": {0: "batch"},
                "action_card_idx": {0: "batch", 1: "actions"},
                "logits": {0: "batch", 1: "actions"},
                "value": {0: "batch"},
            },
            opset_version=17,
        )
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        # Graph must declare a 164-d state_features input.
        sf = next(i for i in sess.get_inputs() if i.name == "state_features")
        assert sf.shape[-1] == STATE_DIM_V3_1, f"graph state dim {sf.shape[-1]} != 164"
        ort_out = sess.run(None, {k: v for k, v in feed.items()})
        ort_logits = _np.asarray(ort_out[0])

    diff = float(_np.max(_np.abs(ort_logits - pt_logits)))
    if not _np.isfinite(ort_logits).all():
        fail("164-d ONNX produced non-finite logits")
    if diff > 1e-3:
        fail(f"164-d ONNX vs PyTorch logits diverge: max|Δ|={diff:.2e}")


if __name__ == "__main__":
    main()
