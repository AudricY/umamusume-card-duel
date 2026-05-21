"""Smoke for training/pair_builder.py — fixture-driven, no GPU, no network.

Coverage:
  Chunk-1 baseline (runner-up):
    (i)   zero out-of-range indices on emitted pairs;
    (ii)  winner != loser on every emitted pair;
    (iii) every emitted pair's margin >= --min-margin;
    (iv)  every emitted pair's JSON parses with all required schema fields;
    (v)   manifest counters add up: pairs_emitted + sum(drops) == rows_in
          (runner-up only — each row emits at most one pair);
    (vi)  manifest histogram counts sum to pairs_emitted;
    (vii) bucket assignment matches the documented edges.

  Chunk-5d extensions:
    (viii) ``--modes runner-up,top-k --top-k 3`` emits strictly more
           pairs than runner-up alone, with ``top_k_rank_2`` /
           ``top_k_rank_3`` negative-kind counts populated.
    (ix)   ``--modes runner-up,rule-bot`` emits ``rule_bot`` pairs when
           heuristic != winner and counts ``rule_bot_matches_winner``
           when heuristic == winner.
    (x)    ``--modes runner-up,top-k,rule-bot --max-pairs-per-state 3``
           truncates correctly when a single state has 4+ candidate
           pairs; ``pairs_capped`` reports the truncation count.
    (xi)   Backward-compat: default args (no ``--modes``) produce a
           bit-identical pairs JSONL to ``--modes runner-up``.

Runtime budget <2s. Uses tempfile.mkdtemp for isolation; no cleanup.

Wire: ``npm run test:pair-builder``.
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
    heuristic_idx: int | None = None,
) -> dict:
    row = {
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
    if heuristic_idx is not None:
        row["heuristicSelectedActionIndex"] = heuristic_idx
        row["heuristicSelectedActionId"] = f"a{heuristic_idx}"
    return row


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


def _baseline_fixture_rows() -> list[dict]:
    return [
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


def test_fixture_end_to_end() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="pair-builder-smoke-"))
    in_path = tmp / "traces.jsonl"
    out_path = tmp / "pairs.jsonl"

    rows = _baseline_fixture_rows()
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
    for k in (
        "no_oracle",
        "oracle_shape_mismatch",
        "winner_equals_loser",
        "index_out_of_range",
        "rule_bot_matches_winner",
    ):
        assert drops[k] == 0, (k, drops)

    # (vi) histogram counts sum to pairs_emitted.
    assert sum(hist.values()) == totals["pairs_emitted"], hist
    # (vii) bucket assignment.
    assert hist["0.05-0.10"] == 1, hist
    assert hist[">=0.20"] == 1, hist
    assert hist["0.10-0.20"] == 0, hist

    # Manifest schema sanity.
    assert manifest["mode"] == "runner-up"
    assert manifest["modes"] == ["runner-up"]
    assert manifest["min_margin"] == DEFAULT_MIN_MARGIN
    assert totals["pair_source_counts"].get("mcts-relabel") == 2
    assert totals["negative_kind_counts"].get("runner_up") == 2
    assert totals["pairs_capped"] == 0

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


def test_top_k_mode() -> None:
    """(viii) ``--modes runner-up,top-k --top-k 3`` adds ranks 3+ pairs."""
    tmp = Path(tempfile.mkdtemp(prefix="pair-builder-smoke-topk-"))
    in_path = tmp / "traces.jsonl"
    out_runner = tmp / "pairs-runner.jsonl"
    out_topk = tmp / "pairs-topk.jsonl"
    rows = [
        # Big-spread row: vd=[10,4,2,1,0]/17. Ranks: 0,1,2,3,4. Shares:
        #   0.588, 0.235, 0.118, 0.059, 0.0
        # Margins from winner (0.588): rank2 0.353, rank3 0.470, rank4 0.529.
        # All above floor 0.05.
        _mk_row("seed-topk-A", 1, 5, [10.0, 4.0, 2.0, 1.0, 0.0]),
        # Tight tail: vd=[10,4,3.5,3.4,0]/20.9 winner share 0.478, rank2
        # 0.191 (margin 0.287), rank3 0.167 (margin 0.310), rank4 0.163
        # (margin 0.315). All above floor.
        _mk_row("seed-topk-B", 2, 5, [10.0, 4.0, 3.5, 3.4, 0.0]),
        # Below-floor tail row (rank2 above, rank3 below):
        # vd=[10,5,4.7,0,0]/19.7 -> winner 0.508, rank2 0.254 (margin
        # 0.254 ok), rank3 0.239 (margin 0.269 ok). All above floor.
        # Construct one where rank3 dips: vd=[100,30,29,28,0]/187 ->
        # winner 0.535, rank2 0.160 (margin 0.375), rank3 0.155 (margin
        # 0.380), rank4 0.150 (margin 0.385). Margins are well above
        # floor here too. For our test we just need top-k to emit MORE
        # pairs than runner-up — these fixtures already do.
    ]
    _write_jsonl(in_path, rows)

    runner_manifest = process_inputs(
        [in_path], out_runner, min_margin=DEFAULT_MIN_MARGIN, modes=("runner-up",)
    )
    topk_manifest = process_inputs(
        [in_path],
        out_topk,
        min_margin=DEFAULT_MIN_MARGIN,
        modes=("runner-up", "top-k"),
        top_k=3,
    )

    runner_pairs = _read_jsonl(out_runner)
    topk_pairs = _read_jsonl(out_topk)

    # Runner-up alone: one pair per row (2 rows, both contested).
    assert len(runner_pairs) == 2, len(runner_pairs)
    # runner-up + top-k 3: per row, rank-2 (runner-up label) + rank-3
    # (top_k_rank_3 label) = 2 pairs/row × 2 rows = 4 pairs.
    assert len(topk_pairs) == 4, len(topk_pairs)
    assert len(topk_pairs) > len(runner_pairs), (len(topk_pairs), len(runner_pairs))

    nk = topk_manifest["totals"]["negative_kind_counts"]
    assert nk.get("runner_up", 0) == 2, nk
    assert nk.get("top_k_rank_3", 0) == 2, nk
    # rank-2 label should NOT appear separately when runner-up is enabled —
    # it's already covered as "runner_up".
    assert "top_k_rank_2" not in nk, nk

    # When runner-up is NOT enabled, top-k owns rank-2 itself.
    out_topk_only = tmp / "pairs-topk-only.jsonl"
    topk_only_manifest = process_inputs(
        [in_path],
        out_topk_only,
        min_margin=DEFAULT_MIN_MARGIN,
        modes=("top-k",),
        top_k=3,
    )
    nk2 = topk_only_manifest["totals"]["negative_kind_counts"]
    assert nk2.get("top_k_rank_2", 0) == 2, nk2
    assert nk2.get("top_k_rank_3", 0) == 2, nk2

    print(
        f"[ok] top-k mode: runner-up={len(runner_pairs)} "
        f"runner+topk={len(topk_pairs)} negative_kinds={nk}"
    )


def test_rule_bot_mode() -> None:
    """(ix) ``--modes runner-up,rule-bot`` emits/skips per heuristic."""
    tmp = Path(tempfile.mkdtemp(prefix="pair-builder-smoke-rb-"))
    in_path = tmp / "traces.jsonl"
    out_path = tmp / "pairs.jsonl"
    rows = [
        # Row 1: winner=0, heuristic=2 (different). Both share margins
        # above floor: rank2 margin 0.353, rule_bot vs idx 2 margin 0.470.
        _mk_row("seed-rb-A", 1, 5, [10.0, 4.0, 2.0, 1.0, 0.0], heuristic_idx=2),
        # Row 2: winner=0, heuristic=0 (same as winner) -> rule_bot
        # candidate dropped (rule_bot_matches_winner). Runner-up still emits.
        _mk_row("seed-rb-B", 2, 3, [6.0, 3.0, 1.0], heuristic_idx=0),
        # Row 3: no heuristic field at all -> silently skip rule-bot
        # candidate (runner-up still emits).
        _mk_row("seed-rb-C", 3, 3, [6.0, 3.0, 1.0]),
    ]
    _write_jsonl(in_path, rows)

    manifest = process_inputs(
        [in_path],
        out_path,
        min_margin=DEFAULT_MIN_MARGIN,
        modes=("runner-up", "rule-bot"),
    )
    pairs = _read_jsonl(out_path)

    nk = manifest["totals"]["negative_kind_counts"]
    assert nk.get("runner_up", 0) == 3, nk
    assert nk.get("rule_bot", 0) == 1, nk

    rb_pair = next(p for p in pairs if p["loser"]["negativeKind"] == "rule_bot")
    assert rb_pair["seed"] == "seed-rb-A"
    assert rb_pair["loser"]["index"] == 2, rb_pair["loser"]

    drops = manifest["totals"]["drops_by_reason"]
    assert drops["rule_bot_matches_winner"] == 1, drops
    # No row-level drops on this fixture.
    for k in (
        "no_oracle",
        "too_few_legal_actions",
        "oracle_zero_visits",
        "oracle_shape_mismatch",
        "winner_equals_loser",
        "index_out_of_range",
        "margin_below_floor",
    ):
        assert drops[k] == 0, (k, drops)

    print(
        f"[ok] rule-bot mode: pairs={len(pairs)} "
        f"rule_bot_matches_winner={drops['rule_bot_matches_winner']}"
    )


def test_cap_truncation() -> None:
    """(x) Per-state cap truncates when 4+ candidates would emit."""
    tmp = Path(tempfile.mkdtemp(prefix="pair-builder-smoke-cap-"))
    in_path = tmp / "traces.jsonl"
    out_cap3 = tmp / "pairs-cap3.jsonl"
    out_cap2 = tmp / "pairs-cap2.jsonl"
    # 5-action row where all 4 candidates (runner-up + rank-3 + rank-4 +
    # rule_bot) survive the margin floor and the rule-bot pick is rank-4
    # (i.e. distinct from runner-up and top-k ranks 3..4 for a different
    # index). vd=[20,5,4,3,2]/34 -> winner=0 (0.588), rank2=1 (0.147),
    # rank3=2 (0.118), rank4=3 (0.088). All margins >= 0.05. Heuristic
    # index 4 -> share 0.059, margin 0.529.
    rows = [
        _mk_row("seed-cap", 1, 5, [20.0, 5.0, 4.0, 3.0, 2.0], heuristic_idx=4),
    ]
    _write_jsonl(in_path, rows)

    manifest3 = process_inputs(
        [in_path],
        out_cap3,
        min_margin=DEFAULT_MIN_MARGIN,
        modes=("runner-up", "top-k", "rule-bot"),
        top_k=4,
        max_pairs_per_state=3,
    )
    pairs3 = _read_jsonl(out_cap3)
    # Candidates in priority order: runner_up (idx 1), top_k_rank_3
    # (idx 2), top_k_rank_4 (idx 3), rule_bot (idx 4) -> 4 candidates,
    # cap=3 -> emit first 3, drop the last (rule_bot here).
    assert len(pairs3) == 3, [p["loser"] for p in pairs3]
    assert manifest3["totals"]["pairs_capped"] == 1, manifest3["totals"]
    kinds3 = [p["loser"]["negativeKind"] for p in pairs3]
    assert kinds3 == ["runner_up", "top_k_rank_3", "top_k_rank_4"], kinds3

    # Cap=2 truncates two pairs (rank_4 and rule_bot).
    manifest2 = process_inputs(
        [in_path],
        out_cap2,
        min_margin=DEFAULT_MIN_MARGIN,
        modes=("runner-up", "top-k", "rule-bot"),
        top_k=4,
        max_pairs_per_state=2,
    )
    pairs2 = _read_jsonl(out_cap2)
    assert len(pairs2) == 2, [p["loser"] for p in pairs2]
    assert manifest2["totals"]["pairs_capped"] == 2, manifest2["totals"]
    kinds2 = [p["loser"]["negativeKind"] for p in pairs2]
    assert kinds2 == ["runner_up", "top_k_rank_3"], kinds2

    print(
        f"[ok] cap truncation: cap=3 emit=3 capped=1; cap=2 emit=2 capped=2"
    )


def test_backward_compat_bit_identical() -> None:
    """(xi) Default args produce bit-identical pairs JSONL to runner-up.

    Compares the raw bytes of the output file written by:
      A) CLI invocation with no ``--modes`` flag, vs.
      B) Direct process_inputs(modes=("runner-up",)).
    Both must yield the same pairs JSONL byte-for-byte.
    """
    tmp = Path(tempfile.mkdtemp(prefix="pair-builder-smoke-bc-"))
    in_path = tmp / "traces.jsonl"
    out_default = tmp / "pairs-default.jsonl"
    out_explicit = tmp / "pairs-explicit.jsonl"
    manifest_default = tmp / "pairs-default.manifest.json"

    _write_jsonl(in_path, _baseline_fixture_rows())

    # A) Default CLI: no --modes flag.
    rc = main(
        [
            "--in",
            str(in_path),
            "--out",
            str(out_default),
            "--manifest-out",
            str(manifest_default),
        ]
    )
    assert rc == 0, rc
    # B) Explicit single-mode call.
    process_inputs(
        [in_path],
        out_explicit,
        min_margin=DEFAULT_MIN_MARGIN,
        modes=("runner-up",),
    )

    bytes_a = out_default.read_bytes()
    bytes_b = out_explicit.read_bytes()
    assert bytes_a == bytes_b, (
        f"backward-compat broken: default-CLI bytes != runner-up bytes "
        f"({len(bytes_a)} vs {len(bytes_b)})"
    )

    # Manifest backward-compat: legacy 'mode' field set, 'modes' list
    # contains exactly ['runner-up'].
    manifest = json.loads(manifest_default.read_text(encoding="utf8"))
    assert manifest["mode"] == "runner-up", manifest["mode"]
    assert manifest["modes"] == ["runner-up"], manifest["modes"]

    print(f"[ok] backward-compat: {len(bytes_a)} bytes identical (default == runner-up)")


def main_smoke() -> int:
    test_fixture_end_to_end()
    test_cli_entrypoint()
    test_top_k_mode()
    test_rule_bot_mode()
    test_cap_truncation()
    test_backward_compat_bit_identical()
    print("[pair_builder_smoke] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main_smoke())
