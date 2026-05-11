"""Smoke test for training/observability_app.py.

Plants a fake run dir with hand-crafted orchestrator-state.json + events.jsonl,
spins the Flask app via test_client(), and asserts the index, run dashboard,
and the two JSON/JSONL API routes return expected content.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from observability_app import create_app  # noqa: E402


def _plant(runs: Path) -> None:
    run = runs / "test-run"
    run.mkdir(parents=True)
    now = time.time() - 30  # recent so status is "running"-ish
    state = {
        "promoted_checkpoint": str(run / "iter-001/model/checkpoint.pt"),
        "promoted_wilson_lower": 0.30,
        "iterations": [
            {"iteration": 0, "wilson_lower": 0.30, "promoted": True,
             "decision_reason": "wilson 0.30 >= floor 0.0", "consecutive_failures": 0,
             "matchup_floor_violations": [], "cycling_alarm": [],
             "halted": False, "halt_reason": None, "pool_evals": []},
            {"iteration": 1, "wilson_lower": 0.20, "promoted": False,
             "decision_reason": "wilson 0.20 < floor 0.30", "consecutive_failures": 1,
             "matchup_floor_violations": [], "cycling_alarm": [],
             "halted": False, "halt_reason": None, "pool_evals": []},
        ],
        "halted": False, "halt_reason": None,
    }
    (run / "orchestrator-state.json").write_text(json.dumps(state))
    events = [
        {"ts": now, "run_id": "test-run", "iteration": -1, "stage": "orchestrator",
         "event_type": "run_started", "schema_version": 1, "data": {"iterations": 2}},
    ]
    for it in (0, 1):
        for stage in ("trace-gen", "relabel", "mix", "train", "gate"):
            events.append({"ts": now + it + 0.1, "run_id": "test-run", "iteration": it,
                           "stage": stage, "event_type": "started", "schema_version": 1, "data": {}})
            events.append({"ts": now + it + 0.2, "run_id": "test-run", "iteration": it,
                           "stage": stage, "event_type": "completed", "schema_version": 1, "data": {}})
        for ep in (1, 2):
            events.append({"ts": now + it + 0.3, "run_id": "test-run", "iteration": it,
                           "stage": "train", "event_type": "epoch", "schema_version": 1,
                           "data": {"epoch": ep, "train_loss": 1.0 - ep * 0.1, "train_kl_loss": 0.0}})
        et = "promoted" if it == 0 else "rejected"
        events.append({"ts": now + it + 0.4, "run_id": "test-run", "iteration": it,
                       "stage": "decision", "event_type": et, "schema_version": 1,
                       "data": {"wilson_lower": 0.30 if it == 0 else 0.20}})
    # one malformed line — must be skipped
    lines = [json.dumps(e) for e in events] + ["{not json"]
    (run / "events.jsonl").write_text("\n".join(lines) + "\n")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        runs = Path(tmp) / "runs"
        runs.mkdir()
        _plant(runs)
        app = create_app(runs)
        c = app.test_client()

        r = c.get("/")
        assert r.status_code == 200, r.status_code
        body = r.get_data(as_text=True)
        assert "test-run" in body, "index missing run name"

        r = c.get("/run/test-run")
        assert r.status_code == 200, r.status_code
        body = r.get_data(as_text=True)
        for needle in ("iteration 0", "iteration 1", "promoted", "rejected",
                       "cdn.jsdelivr.net/npm/chart.js", "trace-gen", "train", "gate"):
            assert needle in body, f"missing {needle!r} in run page"

        r = c.get("/run/test-run/api/state.json")
        assert r.status_code == 200
        st = r.get_json()
        assert st["promoted_wilson_lower"] == 0.30
        assert len(st["iterations"]) == 2

        r = c.get("/run/test-run/api/events.jsonl")
        assert r.status_code == 200
        text = r.get_data(as_text=True)
        # planted events + 1 malformed sentinel
        assert "run_started" in text
        assert "{not json" in text  # passthrough preserves raw content

        r = c.get("/run/missing-run")
        assert r.status_code == 404

        print(json.dumps({"status": "PASS", "routes_checked": 5,
                          "asserts": "index, run, state, events, 404"}))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
