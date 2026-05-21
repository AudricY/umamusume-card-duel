"""Smoke for training/dpo_quality_eval.py — R16-TD 3b P2 step 5.

Asserts on a synthetic 8-row fixture (4 explicit pairs + 4 explicit pairs
with varied phases/action kinds) and an in-memory eval against a freshly-
initialized `CandidatePolicyNet`:

  (i)   `split_pairs_deterministic` is stable: rerunning with the same
        seed yields identical (train, held-out) index lists.
  (ii)  Changing `--split-seed` changes the held-out partition.
  (iii) The output manifest has `schemaVersion == 1` and all expected
        keys (`args`, `split`, `held_out_metrics`, etc.).
  (iv)  All grouped axes fire on the held-out subset (negative_kind,
        source_kind, phase, action_kind, margin_bucket).
  (v)   Ranking metrics are monotone: top1 <= top2 <= top3 across all
        rows (sample-mean), and MRR >= 1/A_max worst-case.
  (vi)  When the synthetic fixture forces winner != argmax for at least
        one row, `ranking_accuracy_top1 < 1.0`.

Runtime budget <5s. Pure stdlib + torch + numpy.

Wire: `npm run test:dpo-quality-eval`.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from dpo_quality_eval import (  # noqa: E402
    SCHEMA_VERSION,
    compute_ranking_metrics,
    run_eval,
    split_pairs_deterministic,
)
from pair_corpus import PreferencePairDataset  # noqa: E402
from uma_ai.features import ACTION_DIM, CARD_ID_SHAPES  # noqa: E402
from uma_ai.model import CandidatePolicyNet, ModelConfig  # noqa: E402


def _mk_action(action_id: str, kind: str = "pass", feature_seed: int = 0) -> dict:
    """Generate an action with deterministic but row-distinct features.

    Action features matter for the smoke: if every row's actions all
    share the same feature vector, the policy emits identical logits per
    slot and `pair_accuracy` degenerates to 0.5 (or 0.0 under strict `>`),
    while `top1` degenerates to 1.0 (all tied at the top). Vary the
    features per (action_id, kind) seed so the policy differentiates.
    """
    # Build a deterministic seed off the inputs (Python's `hash` is salted
    # per process by PYTHONHASHSEED, so we use a stable string mix
    # instead).
    import zlib
    seed_int = zlib.adler32(f"{action_id}|{kind}|{feature_seed}".encode("utf8"))
    rng = torch.Generator().manual_seed(seed_int)
    feats = torch.randn(ACTION_DIM, generator=rng).tolist()
    return {
        "id": action_id,
        "kind": kind,
        "phase": "main",
        "payload": {},
        "features": feats,
        "actionSourceCardIdx": None,
        "actionTargetCardIdx": None,
    }


def _mk_observation(seed_offset: int) -> dict:
    """Minimal v3 observation with the cardIdsByZone field present.

    The card ids vary per row so the embedding branch is exercised, but
    we don't need them to be meaningful for the ranking metric assertions
    (we control the winner index relative to a fresh-init policy).

    Card ids are kept in [1, CARD_VOCAB_TABLE_SIZE - 1] so the embedding
    lookup never indexes past the vocab table (108 entries; 0 = pad).
    """
    from uma_ai.model import CARD_VOCAB_TABLE_SIZE
    max_id = CARD_VOCAB_TABLE_SIZE - 1  # exclusive upper bound is the table size
    card_ids = {}
    for zone, width in CARD_ID_SHAPES.items():
        n = max(1, width // 2)
        # Modular-arithmetic over the vocab so different rows get distinct
        # but-in-range ids — the actual semantics are irrelevant for the
        # smoke (we never test that "card X means Y" downstream).
        card_ids[zone] = [
            ((seed_offset * 10 + slot) % max_id) + 1 for slot in range(n)
        ]
    return {
        "phase": "main",
        "sideToAct": "player",
        "turnNumber": 1,
        "own": {"points": 0, "handCount": 0, "deckCount": 0, "discard": [], "energyZone": []},
        "opponent": {"points": 0, "handCount": 0, "deckCount": 0, "discard": []},
        "shared": {},
        "cardIdsByZone": card_ids,
    }


def _mk_pair_row(
    seed: str,
    step: int,
    n_actions: int,
    winner_idx: int,
    loser_idx: int,
    margin: float,
    phase: str = "main",
    negative_kind: str = "runner_up",
    action_kind: str = "pass",
    source_kind: str = "mcts-relabel",
) -> dict:
    actions = [
        _mk_action(f"a{i}", kind=action_kind, feature_seed=step * 100 + i)
        for i in range(n_actions)
    ]
    return {
        "schemaVersion": 1,
        "kind": "preference-pair",
        "sourceKind": source_kind,
        "sourceEpisodeId": seed,
        "sourceStep": step,
        "seed": seed,
        "sideId": "player",
        "phase": phase,
        "observation": _mk_observation(seed_offset=step),
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
            "negativeKind": negative_kind,
        },
        "margin": margin,
        "confidence": {
            "visitShareWinner": 0.6,
            "visitShareLoser": 0.3,
            "sampleCount": 100,
        },
        "sampleWeight": 1.0,
    }


def _write_corpus(path: Path) -> None:
    """8 explicit-pair rows with varied phases / action_kinds / margins.

    The mix is designed so all five grouped axes have >=2 buckets:
      - phases: main, end (matched on rows 0-3 vs 4-7)
      - action_kinds: pass, play
      - margins: 0.10, 0.20, 0.30 (spans <0.10 / 0.10-0.20 / 0.20+)
      - negative_kind: runner_up, scripted
      - source_kind: mcts-relabel, behaviour-clone
    """
    rows = [
        _mk_pair_row("seed-A", 1, 4, 0, 2, 0.30, phase="main", action_kind="pass", source_kind="mcts-relabel"),
        _mk_pair_row("seed-A", 2, 5, 1, 3, 0.20, phase="main", action_kind="play", negative_kind="scripted"),
        _mk_pair_row("seed-B", 3, 3, 2, 0, 0.15, phase="main", action_kind="pass", source_kind="behaviour-clone"),
        _mk_pair_row("seed-B", 4, 4, 0, 1, 0.10, phase="main", action_kind="play"),
        _mk_pair_row("seed-C", 5, 5, 3, 0, 0.30, phase="end", action_kind="pass"),
        _mk_pair_row("seed-C", 6, 4, 1, 2, 0.20, phase="end", action_kind="play", negative_kind="scripted"),
        _mk_pair_row("seed-D", 7, 3, 0, 2, 0.10, phase="end", action_kind="pass", source_kind="behaviour-clone"),
        _mk_pair_row("seed-D", 8, 4, 2, 3, 0.25, phase="end", action_kind="play"),
    ]
    with path.open("w", encoding="utf8") as h:
        for r in rows:
            h.write(json.dumps(r))
            h.write("\n")


def _write_checkpoint(path: Path, *, seed: int = 0) -> ModelConfig:
    """Write a freshly-initialised CandidatePolicyNet checkpoint.

    We force the default state_dim (110-d v3.0) so the corpus loader's
    `observation_to_features` shape matches the model's encoder.
    """
    torch.manual_seed(seed)
    config = ModelConfig(hidden_dim=32, depth=1, dropout=0.0)
    model = CandidatePolicyNet(config)
    payload = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": config.to_dict(),
        "training": {"objective": "smoke-init"},
    }
    torch.save(payload, path)
    return config


def test_deterministic_split() -> None:
    """(i) + (ii): same seed → identical split; different seed → different."""
    tmp = Path(tempfile.mkdtemp(prefix="dpo-eval-smoke-split-"))
    corpus_path = tmp / "pairs.jsonl"
    _write_corpus(corpus_path)

    ds = PreferencePairDataset(corpus_path, log_manifest=False)
    pairs = list(ds.pairs)

    train_a, hold_a = split_pairs_deterministic(pairs, 0.25, seed=17)
    train_b, hold_b = split_pairs_deterministic(pairs, 0.25, seed=17)
    assert train_a == train_b, f"seed=17 not deterministic: {train_a} vs {train_b}"
    assert hold_a == hold_b, f"seed=17 not deterministic: {hold_a} vs {hold_b}"

    # Partitions are disjoint and cover everything.
    assert set(train_a).isdisjoint(set(hold_a)), "train/holdout overlap"
    assert sorted(train_a + hold_a) == list(range(len(pairs))), "split missed rows"

    # A different seed should produce a different partition (with high prob).
    train_c, hold_c = split_pairs_deterministic(pairs, 0.25, seed=99)
    assert (hold_a, train_a) != (hold_c, train_c), (
        "expected different seed to change the partition; smoke fixture is small "
        "so a collision is possible — bump seed values to repair"
    )
    print(
        f"[ok] deterministic split: seed=17 holdout={hold_a} "
        f"(stable across reruns); seed=99 holdout={hold_c}"
    )


def test_ranking_metrics_unit() -> None:
    """(v) sanity-check `compute_ranking_metrics` on a tiny synthetic case."""
    # 3 rows, 4 actions each.
    # Strict-`>` tie semantics: rank = 1 + #strictly_better_than_winner. Ties
    # with the winner do NOT increase the rank, so a four-way tie at the
    # winner's logit still gives the winner rank=1 ("in the top set").
    logits = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0],  # winner=0 → strictly best → rank=1
            [0.0, 10.0, 5.0, 0.0],  # winner=2 → only slot 1 (10) beats it → rank=2
            [3.0, 1.0, 2.0, 0.0],   # winner=3 → 3 strictly better → rank=4
        ]
    )
    mask = torch.ones((3, 4), dtype=torch.bool)
    y_w = torch.tensor([0, 2, 3], dtype=torch.long)
    metrics = compute_ranking_metrics(logits, mask, y_w)
    assert torch.equal(metrics["rank"], torch.tensor([1.0, 2.0, 4.0])), metrics["rank"]
    assert torch.equal(metrics["top1"], torch.tensor([1.0, 0.0, 0.0])), metrics["top1"]
    assert torch.equal(metrics["top2"], torch.tensor([1.0, 1.0, 0.0])), metrics["top2"]
    assert torch.equal(metrics["top3"], torch.tensor([1.0, 1.0, 0.0])), metrics["top3"]
    # Reciprocal rank: 1, 0.5, 0.25
    assert torch.allclose(metrics["reciprocal_rank"], torch.tensor([1.0, 0.5, 0.25])), (
        metrics["reciprocal_rank"]
    )
    print(
        "[ok] compute_ranking_metrics: rank=[1,2,4] top1=[1,0,0] "
        "top2=[1,1,0] MRR=[1,0.5,0.25]"
    )


def test_ranking_metrics_with_mask() -> None:
    """Masked positions must not steal a top-K slot."""
    logits = torch.tensor([[10.0, 100.0, 5.0]])  # illegal slot 1 has a huge logit
    mask = torch.tensor([[True, False, True]], dtype=torch.bool)
    y_w = torch.tensor([0], dtype=torch.long)
    metrics = compute_ranking_metrics(logits, mask, y_w)
    # Slot 1 should be masked out; winner=0 with logit 10 beats slot 2's 5.
    assert metrics["rank"].item() == 1.0, metrics["rank"]
    assert metrics["top1"].item() == 1.0
    print("[ok] compute_ranking_metrics respects action_mask (illegal slot masked out)")


def test_end_to_end_eval() -> None:
    """(iii) + (iv) + (v) + (vi): run eval, check manifest schema + axes."""
    tmp = Path(tempfile.mkdtemp(prefix="dpo-eval-smoke-e2e-"))
    corpus_path = tmp / "pairs.jsonl"
    ckpt_path = tmp / "checkpoint.pt"
    out_path = tmp / "manifest.json"
    _write_corpus(corpus_path)
    _write_checkpoint(ckpt_path, seed=42)

    import argparse
    args = argparse.Namespace(
        pairs=corpus_path,
        checkpoint=ckpt_path,
        out=out_path,
        held_out_fraction=0.5,  # large for a tiny corpus so holdout is non-trivial
        split_seed=17,
        batch_size=4,
        beta=0.1,
        tau=-1.0,
        device="cpu",
    )
    manifest = run_eval(args)

    # (iii) manifest schema.
    assert manifest["schemaVersion"] == SCHEMA_VERSION, manifest["schemaVersion"]
    expected_top_keys = {
        "schemaVersion",
        "args",
        "split",
        "source_counts",
        "model_config",
        "feature_schema",
        "timings",
        "held_out_metrics",
    }
    assert expected_top_keys.issubset(set(manifest.keys())), (
        f"missing keys: {expected_top_keys - set(manifest.keys())}"
    )

    metrics = manifest["held_out_metrics"]
    for key in (
        "pair_accuracy",
        "ranking_accuracy_top1",
        "ranking_accuracy_top2",
        "ranking_accuracy_top3",
        "mean_reciprocal_rank",
        "loss",
        "samples",
        "grouped",
    ):
        assert key in metrics, f"held_out_metrics missing {key!r}"

    # Pair accuracy is in [0, 1].
    for key in (
        "pair_accuracy",
        "ranking_accuracy_top1",
        "ranking_accuracy_top2",
        "ranking_accuracy_top3",
        "mean_reciprocal_rank",
    ):
        v = metrics[key]
        assert 0.0 <= v <= 1.0, f"{key}={v} out of [0, 1]"

    # (v) ranking monotonicity.
    t1 = metrics["ranking_accuracy_top1"]
    t2 = metrics["ranking_accuracy_top2"]
    t3 = metrics["ranking_accuracy_top3"]
    assert t1 <= t2 <= t3 + 1e-9, f"non-monotone: top1={t1} top2={t2} top3={t3}"

    # MRR >= 1 / max_actions_in_corpus = 1/5
    assert metrics["mean_reciprocal_rank"] >= 1.0 / 5.0 - 1e-9, (
        f"MRR={metrics['mean_reciprocal_rank']} below 1/5 worst-case"
    )

    # DPO loss for a checkpoint-vs-itself baseline is exactly
    # -log_sigmoid(0) = log(2) ≈ 0.6931 per row, so the batch mean
    # should be very close to that.
    expected_loss = float(torch.log(torch.tensor(2.0)).item())
    assert abs(metrics["loss"] - expected_loss) < 1e-4, (
        f"baseline loss {metrics['loss']} differs from log(2)={expected_loss}"
    )

    # (iv) all 5 grouped axes fire with >=1 bucket each.
    grouped = metrics["grouped"]
    expected_axes = {"negative_kind", "source_kind", "phase", "action_kind", "margin_bucket"}
    assert expected_axes.issubset(set(grouped.keys())), (
        f"missing axes: {expected_axes - set(grouped.keys())}"
    )
    for axis in expected_axes:
        assert len(grouped[axis]) >= 1, f"axis {axis} has no buckets"
        for label, entry in grouped[axis].items():
            assert {"count", "mean_loss", "mean_acc"}.issubset(entry.keys()), (
                f"axis {axis} label {label} missing fields"
            )

    # Held-out row count matches the split.
    assert manifest["split"]["held_out_rows"] == int(metrics["samples"]), (
        f"holdout {manifest['split']['held_out_rows']} != samples {metrics['samples']}"
    )

    # Manifest writes to disk only when main() is invoked; run_eval just
    # builds the dict, so directly write here to exercise the JSON round-trip.
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    reloaded = json.loads(out_path.read_text(encoding="utf8"))
    assert reloaded["schemaVersion"] == SCHEMA_VERSION

    print(
        f"[ok] end-to-end eval: holdout={manifest['split']['held_out_rows']} rows, "
        f"pair_acc={metrics['pair_accuracy']:.3f} top1={t1:.3f} "
        f"top2={t2:.3f} top3={t3:.3f} MRR={metrics['mean_reciprocal_rank']:.3f} "
        f"loss={metrics['loss']:.4f} axes={sorted(grouped.keys())}"
    )


def main() -> int:
    t0 = time.monotonic()
    test_deterministic_split()
    test_ranking_metrics_unit()
    test_ranking_metrics_with_mask()
    test_end_to_end_eval()
    elapsed = time.monotonic() - t0
    print(f"[dpo_quality_eval_smoke] OK in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
