"""Smoke for training/pair_corpus.py — R16-TD 3b chunk 2.

Asserts on a synthetic 5-row fixture (3 explicit pairs + 2 legacy
outcome-v2 rows) AND a sanity load of the R16-TD 3a prod-corpus pair
file if present:
  (i)   explicit `kind: preference-pair` rows parse via the new path;
  (ii)  legacy `policy: rollout-outcome-v2` rows still parse (no
        regression);
  (iii) `collate_preference_batch` emits all chunk-2 tensors with the
        correct shapes and dtypes;
  (iv)  `card_ids_by_zone_w` / `card_ids_by_zone_l` match the per-row
        `cardIdsByZone` content;
  (v)   `action_card_idx_w` / `action_card_idx_l` match the per-pair
        action-card-index lookup on the winner / loser action;
  (vi)  the `_EMBEDDING_FALLBACK_WARNINGS` counter increments exactly
        once when a row is missing `cardIdsByZone` (legacy path);
  (vii) the manifest log line fires with the right counts.

Runtime budget <5s. Pure stdlib + torch + numpy.

Wire: `npm run test:pair-corpus`.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import pair_corpus  # noqa: E402
from pair_corpus import (  # noqa: E402
    PreferencePairDataset,
    collate_preference_batch,
    load_preference_pairs,
)
from uma_ai.features import (  # noqa: E402
    ACTION_DIM,
    CARD_ID_SHAPES,
    STATE_DIM,
    ZONE_ORDER,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
PROD_PAIRS = REPO_ROOT / "runs" / "R16-TD-3a-prod-corpus" / "pairs" / "pairs.jsonl"


def _mk_action(action_id: str, src_idx: int | None = None, tgt_idx: int | None = None) -> dict:
    return {
        "id": action_id,
        "kind": "pass",
        "phase": "main",
        "payload": {},
        "features": [0.0] * ACTION_DIM,
        "actionSourceCardIdx": src_idx,
        "actionTargetCardIdx": tgt_idx,
    }


def _mk_observation(card_id_seed: int) -> dict:
    """Build a v3-shaped observation with deterministic per-zone card ids.

    The `cardIdsByZone` lists are intentionally shorter than the per-zone
    cap so the collator's per-zone padding stride is exercised. Card ids
    use `card_id_seed * 100 + slot` so each row's tensor is uniquely
    identifiable in the assertion.
    """
    card_ids = {}
    for zone, width in CARD_ID_SHAPES.items():
        # Pack at most `width // 2 + 1` ids so the zero-padded tail is
        # also covered; minimum 1 entry so a zone always has *some* id.
        n = max(1, width // 2 + 1) if width > 1 else 1
        card_ids[zone] = [card_id_seed * 100 + slot + 1 for slot in range(n)]
    return {
        "phase": "main",
        "sideToAct": "player",
        "turnNumber": 1,
        "own": {"points": 0, "handCount": 0, "deckCount": 0, "discard": [], "energyZone": []},
        "opponent": {"points": 0, "handCount": 0, "deckCount": 0, "discard": []},
        "shared": {},
        "cardIdsByZone": card_ids,
    }


def _mk_observation_legacy_no_card_ids() -> dict:
    """Legacy outcome-v2 observation that lacks `cardIdsByZone`."""
    return {
        "phase": "main",
        "sideToAct": "player",
        "turnNumber": 1,
        "own": {"points": 0, "handCount": 0, "deckCount": 0, "discard": [], "energyZone": []},
        "opponent": {"points": 0, "handCount": 0, "deckCount": 0, "discard": []},
        "shared": {},
    }


def _mk_explicit_pair_row(
    seed: str,
    step: int,
    n_actions: int,
    winner_idx: int,
    loser_idx: int,
    margin: float,
    card_id_seed: int,
) -> dict:
    """Build a `kind: preference-pair` row matching pair_builder.py."""
    actions = [
        _mk_action(f"a{i}", src_idx=card_id_seed * 10 + i, tgt_idx=card_id_seed * 10 + i + 1)
        for i in range(n_actions)
    ]
    return {
        "schemaVersion": 1,
        "kind": "preference-pair",
        "sourceKind": "mcts-relabel",
        "sourceEpisodeId": seed,
        "sourceStep": step,
        "seed": seed,
        "sideId": "player",
        "phase": "main",
        "observation": _mk_observation(card_id_seed),
        "legalActions": actions,
        "winner": {
            "index": winner_idx,
            "actionId": actions[winner_idx]["id"],
            "score": 0.6,
        },
        "loser": {
            "index": loser_idx,
            "actionId": actions[loser_idx]["id"],
            "score": 0.3,
            "negativeKind": "runner_up",
        },
        "margin": margin,
        "confidence": {
            "visitShareWinner": 0.6,
            "visitShareLoser": 0.3,
            "sampleCount": 100,
        },
        "sampleWeight": 1.0,
    }


def _mk_legacy_outcome_row(
    n_actions: int,
    winner_idx: int,
    loser_idx: int,
    margin: float,
    with_card_ids: bool = True,
    card_id_seed: int = 0,
) -> dict:
    """Build a legacy `rollout-outcome-v2` row (oracle.candidates path)."""
    actions = [
        _mk_action(f"la{i}", src_idx=card_id_seed * 10 + i if with_card_ids else None,
                   tgt_idx=card_id_seed * 10 + i + 1 if with_card_ids else None)
        for i in range(n_actions)
    ]
    if with_card_ids:
        observation = _mk_observation(card_id_seed)
    else:
        observation = _mk_observation_legacy_no_card_ids()
    return {
        "schemaVersion": 1,
        "policy": "rollout-outcome-v2",
        "observation": observation,
        "legalActions": actions,
        "selectedActionIndex": winner_idx,
        "sampleWeight": 1.0,
        "oracle": {
            "sampleCount": 3,
            "selectedVsRunnerUpMargin": margin,
            "candidates": [
                {"index": winner_idx, "rewardMean": 0.5, "rewardVariance": 0.01},
                {"index": loser_idx, "rewardMean": 0.5 - margin, "rewardVariance": 0.01},
            ],
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf8") as h:
        for r in rows:
            h.write(json.dumps(r))
            h.write("\n")


def test_synthetic_dual_path() -> None:
    """Synthetic 5-row fixture: 3 explicit pairs + 2 legacy outcome-v2 rows."""
    tmp = Path(tempfile.mkdtemp(prefix="pair-corpus-smoke-"))
    in_path = tmp / "mixed.jsonl"

    rows = [
        # Explicit pairs (chunk-1 schema). Card-id seeds 1, 2, 3 so each
        # row's `cardIdsByZone` tensor is uniquely identifiable.
        _mk_explicit_pair_row("seed-A", 1, n_actions=5, winner_idx=2, loser_idx=0, margin=0.30, card_id_seed=1),
        _mk_explicit_pair_row("seed-B", 2, n_actions=3, winner_idx=0, loser_idx=2, margin=0.15, card_id_seed=2),
        _mk_explicit_pair_row("seed-C", 3, n_actions=4, winner_idx=3, loser_idx=1, margin=0.10, card_id_seed=3),
        # Legacy outcome-v2 rows (with cardIdsByZone present, mirroring
        # a v3-extracted legacy corpus).
        _mk_legacy_outcome_row(n_actions=4, winner_idx=0, loser_idx=2, margin=0.20, with_card_ids=True, card_id_seed=4),
        _mk_legacy_outcome_row(n_actions=3, winner_idx=1, loser_idx=0, margin=0.10, with_card_ids=True, card_id_seed=5),
    ]
    _write_jsonl(in_path, rows)

    # Capture the manifest log line.
    log_buf = io.StringIO()
    pairs = list(
        load_preference_pairs(
            in_path,
            tau=0.02,
            strict_schema_version=True,
            log_manifest=True,
            log_stream=log_buf,
        )
    )

    # (i) explicit pairs parse and carry the right metadata.
    explicit = [p for p in pairs if p.pair_source_kind == "mcts-relabel"]
    legacy = [p for p in pairs if p.pair_source_kind == "rollout-outcome-v2"]
    assert len(explicit) == 3, f"expected 3 explicit pairs, got {len(explicit)}"
    assert len(legacy) == 2, f"expected 2 legacy pairs, got {len(legacy)}"

    # Per-row spot-checks (winner/loser indices match the explicit schema).
    assert explicit[0].y_w_index == 2 and explicit[0].y_l_index == 0
    assert abs(explicit[0].margin - 0.30) < 1e-6
    assert explicit[1].y_w_index == 0 and explicit[1].y_l_index == 2
    assert legacy[0].y_w_index == 0 and legacy[0].y_l_index == 2

    # (vii) manifest log line.
    log = log_buf.getvalue()
    assert "[pair_corpus] loaded 3 explicit-pair rows + 2 legacy outcome-v2 rows" in log, log

    # Reset the fallback counter for the deterministic check below.
    pair_corpus._EMBEDDING_FALLBACK_WARNINGS = 0

    # (iii) collate all 5 and verify shapes/dtypes.
    batch = collate_preference_batch(pairs)
    expected_keys = {
        "state_features",
        "action_features",
        "action_mask",
        "y_w_index",
        "y_l_index",
        "margins",
        "sample_weights",
        "card_ids_by_zone",
        "card_ids_by_zone_w",
        "card_ids_by_zone_l",
        "action_card_idx",
        "action_card_idx_w",
        "action_card_idx_l",
    }
    assert set(batch.keys()) == expected_keys, (
        f"unexpected keys: extra={set(batch.keys())-expected_keys}, "
        f"missing={expected_keys-set(batch.keys())}"
    )
    B = len(pairs)
    max_actions = max(p.action_features.shape[0] for p in pairs)
    num_zones = len(ZONE_ORDER)
    max_cards_per_zone = max(CARD_ID_SHAPES.values())

    # Shape/dtype contract.
    assert batch["state_features"].shape == (B, STATE_DIM)
    assert batch["action_features"].shape == (B, max_actions, ACTION_DIM)
    assert batch["action_mask"].shape == (B, max_actions)
    assert batch["action_mask"].dtype.is_floating_point is False
    assert batch["y_w_index"].shape == (B,) and batch["y_w_index"].dtype.is_floating_point is False
    assert batch["card_ids_by_zone"].shape == (B, num_zones, max_cards_per_zone)
    assert batch["card_ids_by_zone_w"].shape == (B, num_zones, max_cards_per_zone)
    assert batch["card_ids_by_zone_l"].shape == (B, num_zones, max_cards_per_zone)
    assert batch["card_ids_by_zone"].dtype.is_floating_point is False
    assert batch["action_card_idx"].shape == (B, max_actions, 2)
    assert batch["action_card_idx_w"].shape == (B, 2)
    assert batch["action_card_idx_l"].shape == (B, 2)
    assert batch["action_card_idx_w"].dtype.is_floating_point is False

    # `_w` and `_l` zone tensors should be identical (same observation).
    assert (batch["card_ids_by_zone_w"] == batch["card_ids_by_zone_l"]).all().item()
    assert (batch["card_ids_by_zone_w"] == batch["card_ids_by_zone"]).all().item()

    # (iv) per-row zone content matches the original row's `cardIdsByZone`.
    for row_idx, pair in enumerate(pairs):
        zones = pair.card_ids_by_zone
        assert zones is not None, (
            f"row {row_idx} ({pair.pair_source_kind}) should have card_ids_by_zone"
        )
        emitted = batch["card_ids_by_zone"][row_idx].numpy()
        for zone_index, zone in enumerate(ZONE_ORDER):
            width = zones[zone].shape[0]
            assert np.array_equal(emitted[zone_index, :width], zones[zone]), (
                f"row {row_idx} zone {zone!r}: emitted={emitted[zone_index, :width]} "
                f"!= source={zones[zone]}"
            )
            # Padding tail must be zero.
            assert (emitted[zone_index, width:] == 0).all()

    # (v) winner/loser action_card_idx match the chosen action's pair.
    for row_idx, pair in enumerate(pairs):
        action_idx = pair.action_card_idx
        assert action_idx is not None
        w_pair = action_idx[pair.y_w_index]
        l_pair = action_idx[pair.y_l_index]
        emitted_w = batch["action_card_idx_w"][row_idx].numpy()
        emitted_l = batch["action_card_idx_l"][row_idx].numpy()
        assert np.array_equal(emitted_w, w_pair), (
            f"row {row_idx}: action_card_idx_w {emitted_w} != source {w_pair}"
        )
        assert np.array_equal(emitted_l, l_pair), (
            f"row {row_idx}: action_card_idx_l {emitted_l} != source {l_pair}"
        )
        # Full padded action_card_idx[row, :count, :] must match the source.
        count = pair.action_features.shape[0]
        emitted_full = batch["action_card_idx"][row_idx, :count, :].numpy()
        assert np.array_equal(emitted_full, action_idx), (
            f"row {row_idx}: full action_card_idx mismatch"
        )

    # (vi) fallback counter should still be 0 — every row has cardIdsByZone.
    assert pair_corpus._EMBEDDING_FALLBACK_WARNINGS == 0, (
        f"expected 0 fallback warnings, got {pair_corpus._EMBEDDING_FALLBACK_WARNINGS}"
    )

    print(
        f"[ok] synthetic dual-path: explicit={len(explicit)} legacy={len(legacy)} "
        f"batch_shapes verified, zero fallback warnings"
    )


def test_legacy_row_without_card_ids_falls_back() -> None:
    """A legacy outcome-v2 row that lacks `cardIdsByZone` must still load.

    Asserts the embedding tensors are zero for that row's slot and the
    `_EMBEDDING_FALLBACK_WARNINGS` counter increments exactly once.
    """
    tmp = Path(tempfile.mkdtemp(prefix="pair-corpus-smoke-legacy-"))
    in_path = tmp / "legacy_no_cardids.jsonl"
    rows = [
        # One legacy row WITH cardIdsByZone.
        _mk_legacy_outcome_row(n_actions=3, winner_idx=0, loser_idx=1, margin=0.20, with_card_ids=True, card_id_seed=10),
        # One legacy row WITHOUT cardIdsByZone (the pre-Phase-1 fallback case).
        _mk_legacy_outcome_row(n_actions=3, winner_idx=1, loser_idx=2, margin=0.20, with_card_ids=False),
    ]
    _write_jsonl(in_path, rows)

    pairs = list(load_preference_pairs(in_path, tau=0.02))
    assert len(pairs) == 2, len(pairs)
    assert pairs[0].card_ids_by_zone is not None
    assert pairs[0].action_card_idx is not None
    assert pairs[1].card_ids_by_zone is None, (
        "legacy row without cardIdsByZone should have card_ids_by_zone=None"
    )
    assert pairs[1].action_card_idx is None

    pair_corpus._EMBEDDING_FALLBACK_WARNINGS = 0
    batch = collate_preference_batch(pairs)
    # Row 0 should be populated, row 1 zero-tensored.
    assert (batch["card_ids_by_zone"][1] == 0).all().item()
    assert (batch["action_card_idx"][1] == 0).all().item()
    assert (batch["action_card_idx_w"][1] == 0).all().item()
    assert (batch["action_card_idx_l"][1] == 0).all().item()
    # Row 0 has nonzero card ids (card_id_seed=10 → all entries > 0).
    assert batch["card_ids_by_zone"][0].sum().item() > 0
    # Fallback counter: exactly one row (row 1) lacked cardIdsByZone.
    assert pair_corpus._EMBEDDING_FALLBACK_WARNINGS == 1, (
        f"expected 1 fallback warning, got {pair_corpus._EMBEDDING_FALLBACK_WARNINGS}"
    )
    print(
        "[ok] legacy-no-cardids fallback: row populated for cardIds-present row, "
        "zero-tensored for missing-cardIds row, fallback counter == 1"
    )


def test_dataset_class_loads_explicit_only_corpus() -> None:
    """`PreferencePairDataset` should handle an explicit-pair-only corpus
    (no `oracle` field anywhere) without exploding on tau probing."""
    tmp = Path(tempfile.mkdtemp(prefix="pair-corpus-smoke-explicit-only-"))
    in_path = tmp / "explicit_only.jsonl"
    rows = [
        _mk_explicit_pair_row("seed-X", 1, n_actions=4, winner_idx=2, loser_idx=0, margin=0.25, card_id_seed=7),
        _mk_explicit_pair_row("seed-Y", 2, n_actions=3, winner_idx=0, loser_idx=1, margin=0.15, card_id_seed=8),
    ]
    _write_jsonl(in_path, rows)

    ds = PreferencePairDataset(in_path, log_manifest=False)
    assert len(ds) == 2, len(ds)
    assert ds.source_counts.get("mcts-relabel") == 2, ds.source_counts
    batch = collate_preference_batch([ds[0], ds[1]])
    assert "card_ids_by_zone_w" in batch
    assert "action_card_idx_w" in batch
    print(
        f"[ok] explicit-only PreferencePairDataset: len={len(ds)} "
        f"source_counts={ds.source_counts}"
    )


def test_sanity_load_prod_pairs() -> None:
    """Sanity-load the real 3a prod-corpus pair file (8093 rows)."""
    if not PROD_PAIRS.is_file():
        print(f"[skip] prod pairs not found at {PROD_PAIRS}")
        return
    t0 = time.monotonic()
    ds = PreferencePairDataset(PROD_PAIRS)
    elapsed = time.monotonic() - t0
    # Sanity bounds — the corpus is documented at 8093 pairs.
    n = len(ds)
    assert n > 0, "expected non-empty prod corpus"
    assert ds.source_counts.get("mcts-relabel", 0) == n, (
        f"expected all rows tagged mcts-relabel, got {ds.source_counts}"
    )

    # First 4-row batch.
    batch = collate_preference_batch([ds[i] for i in range(4)])
    print(
        f"[ok] sanity load: {n} rows from {PROD_PAIRS.name} in {elapsed:.2f}s; "
        f"first-4 batch shapes: "
        f"state_features={tuple(batch['state_features'].shape)} "
        f"action_features={tuple(batch['action_features'].shape)} "
        f"card_ids_by_zone_w={tuple(batch['card_ids_by_zone_w'].shape)} "
        f"action_card_idx_w={tuple(batch['action_card_idx_w'].shape)} "
        f"action_card_idx={tuple(batch['action_card_idx'].shape)}"
    )


def main_smoke() -> int:
    t0 = time.monotonic()
    test_synthetic_dual_path()
    test_legacy_row_without_card_ids_falls_back()
    test_dataset_class_loads_explicit_only_corpus()
    test_sanity_load_prod_pairs()
    elapsed = time.monotonic() - t0
    print(f"[pair_corpus_smoke] OK in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main_smoke())
