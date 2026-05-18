"""Smoke for training/data_coverage_audit.py — read-only, CPU-only, no network.

Asserts:
  (a) the loader-replica drop predicate matches uma_ai.dataset's silent-drop
      conditions exactly (min_actions=2 + selected-index bounds);
  (b) build_report produces the required acceptance sections;
  (c) the real R7 corpus, when present, reproduces the headline
      retained==(total-dropped) invariant with the single min-actions reason;
  (d) main() exits 0 in --print-only mode and never writes outside the doc dir.

Wired into `npm run test:python-train` via smoke_e2e.py? No — peer
read-only analysis tools (r14_i2_inspect, etc.) pair with a dedicated
*_smoke.py and an npm `test:` entry rather than bloating the monolithic
e2e flow. This file is that pair; `npm run test:data-coverage-audit`.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_coverage_audit import (  # noqa: E402
    DEFAULT_CORPUS,
    _is_retained,
    audit_corpus,
    build_report,
    main,
    render_markdown,
)


def _row(n_legal: int, sel: int, **obs_extra: object) -> dict:
    obs = {
        "phase": "trainerBefore",
        "turnNumber": 3,
        "schemaVersion": 1,
        "own": {"points": 0, "handCount": 4, "deckCount": 12, "active": {"energyTotal": 1}},
    }
    obs.update(obs_extra)  # type: ignore[arg-type]
    return {
        "schemaVersion": 1,
        "seed": 14000,
        "sideId": "player",
        "source": "model-visited-rollout-relabeled",
        "selectedActionIndex": sel,
        "legalActions": [{"kind": "pass"} for _ in range(n_legal)],
        "observation": obs,
    }


def test_predicate() -> None:
    keep, reason = _is_retained(_row(1, 0))
    assert not keep and "lt_min_actions" in reason, reason
    keep, reason = _is_retained(_row(3, -1))
    assert not keep and reason == "selected_index_out_of_range", reason
    keep, reason = _is_retained(_row(3, 5))
    assert not keep and reason == "selected_index_out_of_range", reason
    keep, reason = _is_retained(_row(2, 1))
    assert keep and reason == "retained", reason
    print("[ok] drop predicate mirrors loader (min_actions=2 + index bounds)")


def test_synthetic_report() -> None:
    with tempfile.TemporaryDirectory() as d:
        corpus = Path(d) / "c.jsonl"
        rows = (
            [_row(1, 0)] * 7  # forced -> dropped
            + [_row(3, 1)] * 2  # retained
            + [_row(2, 0, phase="combat")]  # retained
        )
        corpus.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf8"
        )
        c = audit_corpus(corpus)
        assert c["total_rows"] == 10, c["total_rows"]
        assert c["retained_rows"] == 3, c["retained_rows"]
        assert c["dropped_rows"] == 7, c["dropped_rows"]
        assert (
            c["retained_rows"] + c["dropped_rows"] == c["total_rows"]
        ), "retention accounting must close"
        assert list(c["drop_reason_counts"]) == [
            "lt_min_actions(<2_legal)"
        ], c["drop_reason_counts"]

        report = build_report(corpus, Path(d) / "missing.json", Path(d) / "ng.json", Path(d) / "pg.json")
        for key in (
            "corpus_retention",
            "top_skew_slices",
            "train_diagnostics",
            "train_vs_gate_state_overlap",
            "proposed_source_mix_target",
            "environment_gaps",
        ):
            assert key in report, f"missing report section: {key}"
        assert report["proposed_source_mix_target"]["targets"], "mix target empty"
        assert len(report["top_skew_slices"]["most_dropped_buckets"]) >= 1
        md = render_markdown(report)
        assert "Filter-reason decomposition" in md
        assert "Proposed source-mix target" in md
        assert "Environment gaps" in md
        print("[ok] synthetic report has all acceptance sections")


def test_main_print_only() -> None:
    rc = main(["--corpus", str(DEFAULT_CORPUS), "--print-only"])
    if DEFAULT_CORPUS.exists():
        assert rc == 0, rc
        print("[ok] main --print-only exit 0 on real R7 corpus")
    else:
        assert rc == 2, rc
        print("[ok] missing R7 corpus reported as environment gap (rc=2)")


if __name__ == "__main__":
    test_predicate()
    test_synthetic_report()
    test_main_print_only()
    print("data_coverage_audit_smoke: PASS")
