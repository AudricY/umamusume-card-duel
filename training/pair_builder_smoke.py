"""Smoke for training/pair_builder.py — fixture-driven, no GPU, no network.

Asserts on a synthetic 5-row fixture:
  (i)   zero out-of-range indices on emitted pairs;
  (ii)  winner != loser on every emitted pair;
  (iii) every emitted pair's margin >= --min-margin;
  (iv)  every emitted pair's JSON parses with all required schema fields;
  (v)   manifest counters add up: pairs_emitted + sum(drops) == rows_in;
  (vi)  manifest histogram counts sum to pairs_emitted;
  (vii) bucket assignment matches the documented edges.

Runtime budget <2s. Uses tempfile.mkdtemp for isolation; no cleanup.

Wire: `npm run test:pair-builder`.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pair_builder import (  # noqa: E402
    DEFAULT_MIN_MARGIN,
    HISTOGRAM_BUCKETS,
    _bucket_for_margin,
    main,
    process_inputs,
)


REQUIRED_PAIR_KEYS = (
    "schemaVersion",
    "kind",
    "sourceKind",
    "sourceEpisodeId",
    "sourceStep",
    "seed",
    "sideId",
    "phase",
    "observation",
    "legalActions",
    "winner",
    "loser",
    "margin",
    "confidence",
    "sampleWeight",
)
REQUIRED_WINNER_KEYS = ("index", "actionId", "score")
REQUIRED_LOSER_KEYS = ("index", "actionId", "score", "negativeKind")
REQUIRED_CONF_KEYS = ("visitShareWinner", "visitShareLoser", "sampleCount")


def _mk_legal(n: int) -> list[dict]:
    return [{"id": f"a{i}", "kind": "pass", "phase": "main"} for i in range(n)]


def _mk_row(
    seed: str,
    step: int,
    legal_n: int,
    visit_dist: list[float],
    side_id: str = "player",
) -> dict:
    return {
        "schemaVersion": 1,
        "seed": seed,
        "step": step,
        "sideId": side_id,
        "observation": {"phase": "main", "turnNumber": step},
        "legalActions": _mk_legal(legal_n),
        "policyTargets": [1.0 / legal_n] * legal_n,
        "oracle": {
            "simulationsRun": 100,
            "visitDistribution": visit_dist,
            "rolloutCrnSamples": 3,
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf8") as h:
        for r in rows:
            h.write(json.dumps(r))
            h.write("\n")


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf8") as h:
        for line in h:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def test_fixture_end_to_end() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="pair-builder-smoke-"))
    in_path = tmp / "traces.jsonl"
    out_path = tmp / "pairs.jsonl"
    manifest_path = tmp / "pairs.manifest.json"

    rows = [
        # 1. Happy-path big margin: vd=[8,4,1,0,0]/13 -> winner=0 (8/13≈0.615),
        #    loser=1 (4/13≈0.308), margin≈0.308 -> bucket ">=0.20".
        _mk_row("seed-A", 1, 5, [8.0, 4.0, 1.0, 0.0, 0.0]),
        # 2. Just-above-floor: vd=[0.53,0.47] -> margin=0.06 -> bucket "0.05-0.10".
        _mk_row("seed-B", 2, 2, [0.53, 0.47]),
        # 3. Just-below-floor: vd=[0.52,0.48] -> margin=0.04 -> dropped.
        _mk_row("seed-C", 3, 2, [0.52, 0.48]),
        # 4. Forced state: only one legal action.
        _mk_row("seed-D", 4, 1, [1.0]),
        # 5. Zero-visit oracle.
        _mk_row("seed-E", 5, 2, [0.0, 0.0]),
    ]
    _write_jsonl(in_path, rows)

    manifest = process_inputs(
        [in_path],
        out_path,
        min_margin=DEFAULT_MIN_MARGIN,
        mode="runner-up",
    )

    totals = manifest["totals"]
    drops = totals["drops_by_reason"]
    hist = totals["margin_histogram"]

    # (v) manifest counters add up.
    assert totals["rows_in"] == 5, totals
    assert totals["pairs_emitted"] == 2, totals
    drop_sum = sum(drops.values())
    assert totals["pairs_emitted"] + drop_sum == totals["rows_in"], (totals, drop_sum)
    # Per-input check too.
    in0 = manifest["inputs"][0]
    assert in0["rows_in"] == 5 and in0["pairs_emitted"] == 2
    assert in0["pairs_emitted"] + sum(in0["drops_by_reason"].values()) == in0["rows_in"]

    # Expected drop reasons.
    assert drops["margin_below_floor"] == 1, drops
    assert drops["too_few_legal_actions"] == 1, drops
    assert drops["oracle_zero_visits"] == 1, drops
    # The rest must be zero on this fixture.
    for k in ("no_oracle", "oracle_shape_mismatch", "winner_equals_loser", "index_out_of_range"):
        assert drops[k] == 0, (k, drops)

    # (vi) histogram counts sum to pairs_emitted.
    assert sum(hist.values()) == totals["pairs_emitted"], hist
    # (vii) bucket assignment.
    assert hist["0.05-0.10"] == 1, hist
    assert hist[">=0.20"] == 1, hist
    assert hist["0.10-0.20"] == 0, hist

    # Manifest schema sanity.
    assert manifest["mode"] == "runner-up"
    assert manifest["min_margin"] == DEFAULT_MIN_MARGIN
    assert totals["pair_source_counts"].get("mcts-relabel") == 2

    # (i)-(iv): inspect emitted pairs.
    emitted = _read_jsonl(out_path)
    assert len(emitted) == 2, len(emitted)
    for pair in emitted:
        for key in REQUIRED_PAIR_KEYS:
            assert key in pair, (key, sorted(pair.keys()))
        for key in REQUIRED_WINNER_KEYS:
            assert key in pair["winner"], (key, sorted(pair["winner"].keys()))
        for key in REQUIRED_LOSER_KEYS:
            assert key in pair["loser"], (key, sorted(pair["loser"].keys()))
        for key in REQUIRED_CONF_KEYS:
            assert key in pair["confidence"], (key, sorted(pair["confidence"].keys()))

        assert pair["schemaVersion"] == 1
        assert pair["kind"] == "preference-pair"
        assert pair["sourceKind"] == "mcts-relabel"
        assert pair["loser"]["negativeKind"] == "runner_up"

        # (i) indices in range.
        n_legal = len(pair["legalActions"])
        assert 0 <= pair["winner"]["index"] < n_legal, pair["winner"]
        assert 0 <= pair["loser"]["index"] < n_legal, pair["loser"]
        # (ii) winner != loser.
        assert pair["winner"]["index"] != pair["loser"]["index"], pair
        # (iii) margin floor honored.
        assert pair["margin"] >= DEFAULT_MIN_MARGIN, pair["margin"]
        # actionId consistency.
        assert pair["winner"]["actionId"] == pair["legalActions"][pair["winner"]["index"]]["id"]
        assert pair["loser"]["actionId"] == pair["legalActions"][pair["loser"]["index"]]["id"]
        # Visit-share bookkeeping.
        assert pair["confidence"]["visitShareWinner"] == pair["winner"]["score"]
        assert pair["confidence"]["visitShareLoser"] == pair["loser"]["score"]
        # Bucket sanity.
        assert _bucket_for_margin(pair["margin"]) in HISTOGRAM_BUCKETS

    # Spot-check happy-path numerics.
    p0 = next(p for p in emitted if p["seed"] == "seed-A")
    assert p0["winner"]["index"] == 0 and p0["loser"]["index"] == 1
    expected_margin = (8.0 - 4.0) / 13.0
    assert abs(p0["margin"] - expected_margin) < 1e-9, p0["margin"]
    p1 = next(p for p in emitted if p["seed"] == "seed-B")
    assert p1["winner"]["index"] == 0 and p1["loser"]["index"] == 1
    assert abs(p1["margin"] - 0.06) < 1e-9, p1["margin"]

    print(f"[ok] fixture end-to-end: emitted=2/5 drops={drops} histogram={hist}")


def test_cli_entrypoint() -> None:
    """Sanity-run main() to ensure argparse wiring + manifest write work."""
    tmp = Path(tempfile.mkdtemp(prefix="pair-builder-smoke-cli-"))
    in_path = tmp / "traces.jsonl"
    out_path = tmp / "pairs.jsonl"
    manifest_path = tmp / "pairs.manifest.json"
    rows = [_mk_row("seed-cli", 1, 3, [6.0, 3.0, 1.0])]
    _write_jsonl(in_path, rows)

    rc = main(
        [
            "--in",
            str(in_path),
            "--out",
            str(out_path),
            "--manifest-out",
            str(manifest_path),
        ]
    )
    assert rc == 0, rc
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    assert manifest["totals"]["pairs_emitted"] == 1
    assert manifest["totals"]["rows_in"] == 1
    emitted = _read_jsonl(out_path)
    assert len(emitted) == 1
    pair = emitted[0]
    # 6/10 - 3/10 = 0.30 -> bucket ">=0.20".
    assert abs(pair["margin"] - 0.30) < 1e-9, pair["margin"]
    assert manifest["totals"]["margin_histogram"][">=0.20"] == 1

    print("[ok] CLI entrypoint round-trip via main()")


def main_smoke() -> int:
    test_fixture_end_to_end()
    test_cli_entrypoint()
    print("[pair_builder_smoke] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main_smoke())
