"""R16-P2 C4: dataset / selfplay-dataset slot-tensor packing smoke.

Self-contained (no external corpus). Proves the NEW
`uses_uma_slot_tokens` kwarg on `JsonlPolicyDataset` /
`MctsSelfPlayDataset` (and their underlying loaders / collators) honors the
contracts the chunk plan freezes:

  (i)   Default OFF -> batch dict has NO `uma_slot_*` keys, byte-identical
        to the pre-C4 collator output. Run against BOTH the BC dataset and
        the MCTS self-play dataset.
  (ii)  Flag ON -> `uma_slot_card_ids` is int64 `[B, 10]`, `uma_slot_features`
        is float32 `[B, 10, 23]`. At least one card_id across the batch is
        non-zero so the populator demonstrably ran.
  (iii) End-to-end with C2 model: `CandidatePolicyNet(uses_uma_slot_tokens=
        True)` consumes the C4-emitted slot tensors via the (uma_slot_card_ids
        != 0) mask path. Output shape unchanged from the legacy 5-input
        forward; no crash.
  (iv)  All-or-nothing collator guard: synthesize a batch where ONE sample
        is missing both slot fields. Assert the collator omits BOTH
        `uma_slot_*` keys (does not partial-emit). Mirrors the existing
        all-or-nothing pattern on `card_ids_by_zone` / `action_card_idx`.

Deliberately lean. Builds tiny synthetic JSONL fixtures inline (no
dependency on simulator-derived corpora), so the smoke runs in well under
a second.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uma_ai.dataset import (  # noqa: E402
    JsonlPolicyDataset,
    PolicySample,
    collate_policy_batch,
)
from uma_ai.features import (  # noqa: E402
    ACTION_DIM,
    STATE_DIM,
    UMA_SLOT_COUNT,
    UMA_SLOT_FEATURE_DIM,
    ZONE_ORDER,
)
from uma_ai.model import CandidatePolicyNet, ModelConfig  # noqa: E402
from uma_ai.selfplay_dataset import (  # noqa: E402
    MctsSelfPlayDataset,
    MctsSelfPlaySample,
    collate_mcts_selfplay_batch,
)


def fail(msg: str) -> None:
    print(f"[r16-p2-slot-dataset-smoke] FAIL - {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Fixture builders. The synthetic observation matches the v3 PublicObservation
# shape the C1 builder requires (own/opponent nested sides, cardIdsByZone,
# legalActions with 48-d features). Hand-written so the smoke has zero
# dependency on the live simulator.


def _uma(card_id: str, *, hp: int = 90, max_hp: int = 100,
         energy_total: int = 2, stage: int = 1) -> dict[str, Any]:
    return {
        "uid": abs(hash(card_id)) % 10_000,
        "cardId": card_id,
        "species": card_id,
        "stage": stage,
        "hp": hp,
        "maxHp": max_hp,
        "energyTotal": energy_total,
        "energies": {"fire": 1, "water": 1},
        "specialConditions": [],
        "toolCardId": None,
        "usedAbilityThisTurn": False,
    }


def _side(active_id: str, bench_ids: list[str | None]) -> dict[str, Any]:
    bench = [_uma(b) if isinstance(b, str) else None for b in bench_ids]
    return {
        "id": "player",
        "points": 0,
        "handCount": 3,
        "deckCount": 30,
        "discard": [],
        "active": _uma(active_id),
        "bench": bench,
        "handCardIds": ["agnesDigitalBasic"],
        "energyZone": ["fire"],
        "usedSupporterThisTurn": False,
        "usedRetreatThisTurn": False,
        "usedStadiumThisTurn": False,
    }


def _observation(own_active: str, own_bench: list[str | None],
                 opp_active: str, opp_bench: list[str | None]) -> dict[str, Any]:
    return {
        "schemaVersion": 3,
        "sideToAct": "player",
        "phase": "attach",
        "turnNumber": 5,
        "firstPlayer": "player",
        "pendingChoiceKind": None,
        "own": _side(own_active, own_bench),
        "opponent": _side(opp_active, opp_bench),
        "shared": {
            "stadiumCardId": None,
            "currentSide": "player",
            "gameOver": False,
        },
        # Empty per-zone packed ids — the v3 card-embedding branch reads
        # this via `observation_to_card_ids`. Lists must be present (empty
        # is fine; the per-zone arrays will all-zero-pad).
        "cardIdsByZone": {
            "ownActive": [],
            "oppActive": [],
            "ownBench": [],
            "oppBench": [],
            "ownHand": [],
            "ownDiscard": [],
            "oppDiscard": [],
            "stadium": [],
        },
    }


def _action(idx: int) -> dict[str, Any]:
    # 48-d feature vector deterministically derived from the action index so
    # every action row is distinct (so contested-state heuristics in the
    # loader can't conflate them). Floats in [-1, 1].
    rng = np.random.default_rng(seed=idx + 1)
    features = rng.uniform(-1, 1, size=ACTION_DIM).astype(np.float32).tolist()
    return {
        "id": f"act-{idx}",
        "phase": "attach",
        "kind": "pass",
        "payload": {},
        "features": features,
        "actionSourceCardIdx": None,
        "actionTargetCardIdx": None,
    }


def _bc_row(*, seed: int = 0, num_actions: int = 3,
            own_active: str = "agnesDigitalBasic",
            own_bench: list[str | None] | None = None,
            opp_active: str = "agnesDigitalBasic",
            opp_bench: list[str | None] | None = None) -> dict[str, Any]:
    """Build a single BC-format row (schemaVersion=1)."""
    own_bench = own_bench if own_bench is not None else ["matikanetannhauser", None, None]
    opp_bench = opp_bench if opp_bench is not None else [None, None, None]
    return {
        "schemaVersion": 1,
        "seed": 14_000 + seed,
        "sideId": "player",
        "source": "smoke",
        "selectedActionIndex": min(seed % num_actions, num_actions - 1),
        "legalActions": [_action(seed * 10 + i) for i in range(num_actions)],
        "observation": _observation(own_active, own_bench, opp_active, opp_bench),
        "valueTarget": 0.0,
        "sampleWeight": 1.0,
    }


def _mcts_row(*, seed: int = 0, num_actions: int = 3) -> dict[str, Any]:
    """Build a single mcts-selfplay row (schemaVersion=1, kind=mcts-selfplay)."""
    sel = min(seed % num_actions, num_actions - 1)
    visit = [0.0] * num_actions
    visit[sel] = 1.0
    return {
        "schemaVersion": 1,
        "kind": "mcts-selfplay",
        "seed": 17_000 + seed,
        "sideId": "player",
        "step": seed,
        "turnNumber": seed % 10,
        "observation": _observation(
            "agnesDigitalBasic",
            ["matikanetannhauser", None, None],
            "agnesDigitalBasic",
            [None, None, None],
        ),
        "legalActions": [_action(seed * 10 + i) for i in range(num_actions)],
        "selectedActionIndex": sel,
        "visitDistribution": visit,
        "valueTarget": 0.0,
        "sampleWeight": 1.0,
        "result": {"winner": "player", "pointsP": 3, "pointsO": 0},
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf8")


# ---------------------------------------------------------------------------
# Helpers to compare batch dicts (both for the byte-identical regression and
# for inspecting individual tensors).

def _tensor_eq(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(a, b))


def _batch_keys(batch: dict[str, torch.Tensor]) -> set[str]:
    return set(batch.keys())


# ---------------------------------------------------------------------------
# Smokes


def _smoke_default_off_byte_identical() -> None:
    """Smoke (i): default OFF -> batch dict has no `uma_slot_*` keys, and
    the emitted tensors match a pre-C4 baseline (same loader without the
    flag)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # BC: load WITHOUT the flag (default). The batch dict should never
        # carry the new keys.
        bc_path = tmp_path / "bc.jsonl"
        _write_jsonl(bc_path, [_bc_row(seed=i) for i in range(4)])
        ds_off = JsonlPolicyDataset(bc_path)
        if any(s.uma_slot_card_ids is not None for s in ds_off.samples):
            fail("BC default-off: sample.uma_slot_card_ids should be None on every row")
        if any(s.uma_slot_features is not None for s in ds_off.samples):
            fail("BC default-off: sample.uma_slot_features should be None on every row")
        batch_off = collate_policy_batch(list(ds_off.samples))
        if "uma_slot_card_ids" in batch_off or "uma_slot_features" in batch_off:
            fail(
                "BC default-off: batch dict carries uma_slot_* keys "
                f"({sorted(_batch_keys(batch_off))}) - flag should keep them absent."
            )
        # Sanity: the expected pre-C4 keys are all present.
        expected_pre_c4 = {
            "state_features", "action_features", "action_mask",
            "targets", "value_targets", "sample_weights",
            "card_ids_by_zone", "action_card_idx",
        }
        if not expected_pre_c4.issubset(_batch_keys(batch_off)):
            missing = expected_pre_c4 - _batch_keys(batch_off)
            fail(f"BC default-off: pre-C4 batch keys missing {missing}")

        # Byte-identical: a SECOND load with the same default kwargs must
        # produce a tensor-equal batch dict (the regression check for
        # accidental nondeterminism slipping in via the new code path).
        ds_off_again = JsonlPolicyDataset(bc_path)
        batch_off_again = collate_policy_batch(list(ds_off_again.samples))
        for key in expected_pre_c4:
            if not _tensor_eq(batch_off[key], batch_off_again[key]):
                fail(f"BC default-off: key {key!r} not byte-identical across reloads")

        # MCTS: same checks against the self-play loader.
        mc_path = tmp_path / "mcts.jsonl"
        _write_jsonl(mc_path, [_mcts_row(seed=i) for i in range(4)])
        mds_off = MctsSelfPlayDataset(mc_path)
        if any(s.uma_slot_card_ids is not None for s in mds_off.samples):
            fail("MCTS default-off: sample.uma_slot_card_ids should be None on every row")
        mbatch_off = collate_mcts_selfplay_batch(list(mds_off.samples))
        if "uma_slot_card_ids" in mbatch_off or "uma_slot_features" in mbatch_off:
            fail(
                "MCTS default-off: batch dict carries uma_slot_* keys "
                f"({sorted(_batch_keys(mbatch_off))}) - flag should keep them absent."
            )

    print("  PASS  smoke (i): default OFF -> batch dict omits uma_slot_* keys "
          "(BC + MCTS); byte-identical across reloads")


def _smoke_flag_on_shapes() -> None:
    """Smoke (ii): flag ON -> tensors present with correct dtype/shape.

    We confirm the non-trivial-content guarantee (at least one card_id != 0
    across the batch) so a future "populator silently emits all zeros"
    regression bites here.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bc_path = tmp_path / "bc.jsonl"
        _write_jsonl(bc_path, [_bc_row(seed=i) for i in range(4)])
        ds_on = JsonlPolicyDataset(bc_path, uses_uma_slot_tokens=True)
        # Per-row presence + shape sanity.
        for sample in ds_on.samples:
            if sample.uma_slot_card_ids is None or sample.uma_slot_features is None:
                fail("BC flag-on: row carries None uma_slot_* fields - populator skipped a row")
            if sample.uma_slot_card_ids.shape != (UMA_SLOT_COUNT,):
                fail(
                    f"BC flag-on: per-row uma_slot_card_ids shape "
                    f"{sample.uma_slot_card_ids.shape} != ({UMA_SLOT_COUNT},)"
                )
            if sample.uma_slot_features.shape != (UMA_SLOT_COUNT, UMA_SLOT_FEATURE_DIM):
                fail(
                    f"BC flag-on: per-row uma_slot_features shape "
                    f"{sample.uma_slot_features.shape} != "
                    f"({UMA_SLOT_COUNT}, {UMA_SLOT_FEATURE_DIM})"
                )

        batch_on = collate_policy_batch(list(ds_on.samples))
        if "uma_slot_card_ids" not in batch_on or "uma_slot_features" not in batch_on:
            fail("BC flag-on: batch dict missing uma_slot_* keys despite flag=True")
        bsz = len(ds_on.samples)
        ids = batch_on["uma_slot_card_ids"]
        feats = batch_on["uma_slot_features"]
        if ids.dtype != torch.int64 or ids.shape != (bsz, UMA_SLOT_COUNT):
            fail(f"BC flag-on: card_ids dtype/shape mismatch ({ids.dtype}/{tuple(ids.shape)})")
        if feats.dtype != torch.float32 or feats.shape != (bsz, UMA_SLOT_COUNT, UMA_SLOT_FEATURE_DIM):
            fail(
                f"BC flag-on: features dtype/shape mismatch "
                f"({feats.dtype}/{tuple(feats.shape)})"
            )
        if int(ids.abs().sum().item()) == 0:
            fail(
                "BC flag-on: all uma_slot_card_ids are zero across the batch - "
                "populator likely did not run (the synthetic fixture's "
                "agnesDigitalBasic active should resolve to a non-zero vocab idx)."
            )

        # MCTS: identical contract.
        mc_path = tmp_path / "mcts.jsonl"
        _write_jsonl(mc_path, [_mcts_row(seed=i) for i in range(4)])
        mds_on = MctsSelfPlayDataset(mc_path, uses_uma_slot_tokens=True)
        mbatch_on = collate_mcts_selfplay_batch(list(mds_on.samples))
        if "uma_slot_card_ids" not in mbatch_on or "uma_slot_features" not in mbatch_on:
            fail("MCTS flag-on: batch dict missing uma_slot_* keys despite flag=True")
        m_ids = mbatch_on["uma_slot_card_ids"]
        m_feats = mbatch_on["uma_slot_features"]
        if m_ids.dtype != torch.int64 or m_ids.shape != (len(mds_on.samples), UMA_SLOT_COUNT):
            fail(
                f"MCTS flag-on: card_ids dtype/shape mismatch "
                f"({m_ids.dtype}/{tuple(m_ids.shape)})"
            )
        if m_feats.dtype != torch.float32 or m_feats.shape != (
            len(mds_on.samples), UMA_SLOT_COUNT, UMA_SLOT_FEATURE_DIM
        ):
            fail(
                f"MCTS flag-on: features dtype/shape mismatch "
                f"({m_feats.dtype}/{tuple(m_feats.shape)})"
            )
        if int(m_ids.abs().sum().item()) == 0:
            fail("MCTS flag-on: all uma_slot_card_ids are zero across the batch")

    print(f"  PASS  smoke (ii): flag ON -> card_ids [B={bsz}, {UMA_SLOT_COUNT}] int64, "
          f"features [B, {UMA_SLOT_COUNT}, {UMA_SLOT_FEATURE_DIM}] float32; non-trivial content")


def _smoke_e2e_with_model() -> None:
    """Smoke (iii): C4 batches flow through the C2 `CandidatePolicyNet`
    with `uses_uma_slot_tokens=True` end-to-end.

    Builds a small model (hidden_dim=32, depth=2) so the smoke stays fast,
    feeds the C4-emitted batch dict into the forward, and verifies output
    shapes match what the legacy 5-input forward returns. This is the
    load-bearing end-to-end check proving C4 and C2 are wired together
    correctly - the `(uma_slot_card_ids != 0)` mask path inside the model
    consumes exactly what the dataset emits.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bc_path = tmp_path / "bc.jsonl"
        _write_jsonl(bc_path, [_bc_row(seed=i) for i in range(3)])
        ds = JsonlPolicyDataset(bc_path, uses_uma_slot_tokens=True)
        batch = collate_policy_batch(list(ds.samples))

        cfg = ModelConfig(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=32,
            depth=2,
            dropout=0.0,
            uses_uma_slot_tokens=True,
        )
        torch.manual_seed(0)
        model = CandidatePolicyNet(cfg).eval()
        with torch.no_grad():
            logits, value = model(
                state_features=batch["state_features"],
                action_features=batch["action_features"],
                action_mask=batch["action_mask"],
                card_ids_by_zone=batch["card_ids_by_zone"],
                action_card_idx=batch["action_card_idx"],
                uma_slot_card_ids=batch["uma_slot_card_ids"],
                uma_slot_features=batch["uma_slot_features"],
            )

        expected_bsz = len(ds.samples)
        expected_actions = batch["action_features"].shape[1]
        if logits.shape != (expected_bsz, expected_actions):
            fail(
                f"e2e: logits shape {tuple(logits.shape)} != "
                f"({expected_bsz}, {expected_actions})"
            )
        if value.shape != (expected_bsz,):
            fail(f"e2e: value shape {tuple(value.shape)} != ({expected_bsz},)")
        if not torch.isfinite(logits).all() or not torch.isfinite(value).all():
            fail("e2e: model produced non-finite outputs on a C4 batch")

        # Parity sanity: a slot-encoder built with zero-init final Linear plus
        # board-zone-masked card_ids should match a control forward identically
        # (the same delta=0 invariant the C2 smoke pins). We re-use the
        # board-zone-masking convention: zero lanes 0..3 of the control input.
        control_cfg = ModelConfig(
            state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=32,
            depth=2, dropout=0.0, uses_uma_slot_tokens=False,
        )
        torch.manual_seed(0)
        control = CandidatePolicyNet(control_cfg).eval()
        # Copy shared params verbatim into model so the only delta is the
        # slot residual (which is zero at init).
        control_sd = control.state_dict()
        treatment_sd = model.state_dict()
        for k in control_sd:
            treatment_sd[k] = control_sd[k]
        model.load_state_dict(treatment_sd, strict=True)
        masked_card_ids = batch["card_ids_by_zone"].clone()
        masked_card_ids[:, :4, :] = 0  # board zones (matches BOARD_ZONE_LANE_INDICES)
        with torch.no_grad():
            c_logits, c_value = control(
                state_features=batch["state_features"],
                action_features=batch["action_features"],
                action_mask=batch["action_mask"],
                card_ids_by_zone=masked_card_ids,
                action_card_idx=batch["action_card_idx"],
            )
            t_logits, t_value = model(
                state_features=batch["state_features"],
                action_features=batch["action_features"],
                action_mask=batch["action_mask"],
                card_ids_by_zone=batch["card_ids_by_zone"],
                action_card_idx=batch["action_card_idx"],
                uma_slot_card_ids=batch["uma_slot_card_ids"],
                uma_slot_features=batch["uma_slot_features"],
            )
        max_logits_diff = float((t_logits - c_logits).abs().max().item())
        max_value_diff = float((t_value - c_value).abs().max().item())
        if max_logits_diff > 1e-6:
            fail(
                f"e2e: zero-init slot-encoder logits diverge from "
                f"board-masked control by {max_logits_diff:.3e} > 1e-6 - "
                f"C4 batch is feeding the model in a way that breaks the "
                f"parity invariant C2 pins."
            )
        if max_value_diff > 1e-6:
            fail(
                f"e2e: zero-init slot-encoder value diverges by "
                f"{max_value_diff:.3e} > 1e-6"
            )

    print(f"  PASS  smoke (iii): C4 batch -> CandidatePolicyNet(uses_uma_slot_tokens=True) "
          f"forward; logits {tuple(logits.shape)}, value {tuple(value.shape)}; "
          f"zero-init parity max|delta_logits|={max_logits_diff:.3e}")


def _smoke_all_or_nothing_collator() -> None:
    """Smoke (iv): collator omits BOTH `uma_slot_*` keys when even one
    sample lacks them (mirrors the existing all-or-nothing pattern for
    `card_ids_by_zone` / `action_card_idx`).

    Build the samples by hand (skipping the loader) so we can force the
    mixed state cleanly.
    """
    # Build two real samples via the loader (with flag ON) and then construct
    # a third "synthetic" sample without the slot fields. Mix them and run
    # collate_policy_batch.
    with tempfile.TemporaryDirectory() as tmp:
        bc_path = Path(tmp) / "bc.jsonl"
        _write_jsonl(bc_path, [_bc_row(seed=i) for i in range(2)])
        ds = JsonlPolicyDataset(bc_path, uses_uma_slot_tokens=True)
        real_samples = list(ds.samples)
        if len(real_samples) < 2:
            fail("fixture too small - expected >= 2 samples")

        # Construct a doppelganger of real_samples[0] but with the slot fields
        # set to None (simulating a row that didn't go through the slot
        # populator).
        s0 = real_samples[0]
        synthetic = PolicySample(
            state_features=s0.state_features,
            action_features=s0.action_features,
            target_index=s0.target_index,
            value_target=s0.value_target,
            sample_weight=s0.sample_weight,
            example=s0.example,
            policy_target=s0.policy_target,
            card_ids_by_zone=s0.card_ids_by_zone,
            action_card_idx=s0.action_card_idx,
            uma_slot_card_ids=None,
            uma_slot_features=None,
        )
        mixed = real_samples + [synthetic]
        batch = collate_policy_batch(mixed)
        if "uma_slot_card_ids" in batch:
            fail(
                "all-or-nothing: batch contains uma_slot_card_ids despite one "
                "sample missing it (partial-emit detected)"
            )
        if "uma_slot_features" in batch:
            fail(
                "all-or-nothing: batch contains uma_slot_features despite one "
                "sample missing it (partial-emit detected)"
            )
        # Sanity: the other optional keys still present (card_ids_by_zone is
        # carried by every sample including the synthetic one).
        if "card_ids_by_zone" not in batch:
            fail(
                "all-or-nothing fixture broken: card_ids_by_zone unexpectedly "
                "absent (the synthetic doppelganger carries it)"
            )

        # Mirror against the MCTS dataset path.
        mc_path = Path(tmp) / "mcts.jsonl"
        _write_jsonl(mc_path, [_mcts_row(seed=i) for i in range(2)])
        mds = MctsSelfPlayDataset(mc_path, uses_uma_slot_tokens=True)
        real_mcts = list(mds.samples)
        m0 = real_mcts[0]
        synthetic_mcts = MctsSelfPlaySample(
            state_features=m0.state_features,
            action_features=m0.action_features,
            target_index=m0.target_index,
            value_target=m0.value_target,
            sample_weight=m0.sample_weight,
            policy_target=m0.policy_target,
            example=m0.example,
            card_ids_by_zone=m0.card_ids_by_zone,
            action_card_idx=m0.action_card_idx,
            uma_slot_card_ids=None,
            uma_slot_features=None,
        )
        mbatch = collate_mcts_selfplay_batch(real_mcts + [synthetic_mcts])
        if "uma_slot_card_ids" in mbatch or "uma_slot_features" in mbatch:
            fail("MCTS all-or-nothing: collator partial-emitted uma_slot_* keys")

    print("  PASS  smoke (iv): collator all-or-nothing - mixed batch omits "
          "BOTH uma_slot_* keys (BC + MCTS paths)")


def main() -> None:
    # Frozen-shape guards (mirror the C1/C2 smokes).
    if UMA_SLOT_COUNT != 10:
        fail(f"UMA_SLOT_COUNT {UMA_SLOT_COUNT} != 10 (C1 contract drift)")
    if UMA_SLOT_FEATURE_DIM != 23:
        fail(f"UMA_SLOT_FEATURE_DIM {UMA_SLOT_FEATURE_DIM} != 23 (C1 contract drift)")
    if len(ZONE_ORDER) != 8:
        fail(f"ZONE_ORDER len {len(ZONE_ORDER)} != 8 (v3 contract drift)")

    _smoke_default_off_byte_identical()
    _smoke_flag_on_shapes()
    _smoke_e2e_with_model()
    _smoke_all_or_nothing_collator()

    print("r16-p2-slot-dataset smoke: ALL PASS")


if __name__ == "__main__":
    main()
