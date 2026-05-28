"""Unit-style smoke coverage for ReBeL orchestrator loop plumbing.

No self-play, training, export, or gate subprocesses are launched here. The
test guards the AlphaZero-parity orchestration pieces that are easy to regress:
bounded cross-iteration replay, fixed KL-anchor resolution, and release-binary
preflight failures.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from events import EventWriter  # noqa: E402
from rebel_orchestrator import (  # noqa: E402
    RebelLoopState,
    materialize_replay_mix,
    preflight_release_binaries,
    read_jsonl_lines,
    resolve_kl_anchor,
    snapshot_promoted_artifacts,
)


def _args(**overrides: object) -> argparse.Namespace:
    values = {
        "cross_iter_replay": True,
        "replay_window": 3,
        "replay_old_fraction": 0.4,
        "fixed_kl_anchor": True,
        "kl_anchor_checkpoint": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _write_rows(path: Path, prefix: str, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as fh:
        for index in range(count):
            fh.write(json.dumps({"row": f"{prefix}-{index}"}) + "\n")


def test_replay_mix_materializes_bounded_old_fraction() -> None:
    work = Path(tempfile.mkdtemp(prefix="uma-rebel-replay-smoke-"))
    try:
        loop = work / "loop"
        iter_dir = loop / "iter-2"
        current = iter_dir / "rebel-selfplay.jsonl"
        _write_rows(loop / "iter-0" / "rebel-selfplay.jsonl", "old0", 4)
        _write_rows(loop / "iter-1" / "rebel-selfplay.jsonl", "old1", 4)
        _write_rows(current, "cur", 6)

        mixed, summary = materialize_replay_mix(
            args=_args(replay_window=2, replay_old_fraction=0.4),
            loop_dir=loop,
            iter_dir=iter_dir,
            iteration=2,
            current_selfplay=current,
            events=EventWriter(loop),
        )

        assert mixed == iter_dir / "rebel-train-mixed.jsonl"
        assert summary["status"] == "materialized", summary
        assert summary["current_rows"] == 6, summary
        assert summary["old_rows"] == 4, summary
        assert summary["total_rows"] == 10, summary
        assert abs(summary["old_fraction_actual"] - 0.4) < 1e-9, summary
        lines = read_jsonl_lines(mixed)
        assert len(lines) == 10, lines
        assert [json.loads(line)["row"] for line in lines[:6]] == [f"cur-{i}" for i in range(6)]
        assert any("old1-" in line for line in lines[6:]), lines

        events = [json.loads(line) for line in (loop / "events.jsonl").read_text().splitlines()]
        replay_events = [e for e in events if e["stage"] == "replay"]
        assert replay_events and replay_events[-1]["event_type"] == "materialized", events
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_replay_passthrough_when_disabled() -> None:
    work = Path(tempfile.mkdtemp(prefix="uma-rebel-replay-disabled-"))
    try:
        loop = work / "loop"
        current = loop / "iter-0" / "rebel-selfplay.jsonl"
        _write_rows(current, "cur", 3)
        mixed, summary = materialize_replay_mix(
            args=_args(cross_iter_replay=False),
            loop_dir=loop,
            iter_dir=current.parent,
            iteration=0,
            current_selfplay=current,
            events=EventWriter(loop),
        )
        assert mixed == current
        assert summary == {
            "status": "disabled",
            "data": str(current),
            "current_rows": 3,
            "old_rows": 0,
            "total_rows": 3,
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_fixed_kl_anchor_stays_pinned() -> None:
    state = RebelLoopState(kl_anchor_checkpoint="/tmp/iter-0/checkpoint.pt")
    assert (
        resolve_kl_anchor(_args(), state, "/tmp/iter-2/checkpoint.pt")
        == "/tmp/iter-0/checkpoint.pt"
    )
    assert (
        resolve_kl_anchor(_args(fixed_kl_anchor=False), state, "/tmp/iter-2/checkpoint.pt")
        == "/tmp/iter-2/checkpoint.pt"
    )
    explicit = Path("/tmp/manual-anchor.pt").resolve()
    assert resolve_kl_anchor(_args(kl_anchor_checkpoint=str(explicit)), state, None) == str(explicit)


def test_release_binary_preflight_reports_missing_and_stale() -> None:
    work = Path(tempfile.mkdtemp(prefix="uma-rebel-preflight-"))
    try:
        release = work / "engine-rs" / "target" / "release"
        release.mkdir(parents=True, exist_ok=True)
        events = EventWriter(work / "loop")

        try:
            preflight_release_binaries(work, events)
        except SystemExit as exc:
            assert "sim-rebel-selfplay" in str(exc), exc
            assert "sim-eval-gate" in str(exc), exc
        else:
            raise AssertionError("missing binaries should fail preflight")

        selfplay = release / "sim-rebel-selfplay"
        gate = release / "sim-eval-gate"
        selfplay.write_text("bin", encoding="utf8")
        gate.write_text("bin", encoding="utf8")
        old = time.time() - 100
        os.utime(selfplay, (old, old))
        os.utime(gate, (old, old))

        source = work / "engine-rs" / "crates" / "engine" / "src" / "rebel" / "mod.rs"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("// newer source\n", encoding="utf8")
        new = time.time()
        os.utime(source, (new, new))

        try:
            preflight_release_binaries(work, events)
        except SystemExit as exc:
            assert "older than ReBeL/eval Rust sources" in str(exc), exc
        else:
            raise AssertionError("stale binaries should fail preflight")

        events_rows = [json.loads(line) for line in (work / "loop" / "events.jsonl").read_text().splitlines()]
        failures = [
            e for e in events_rows
            if e["stage"] == "rebel-orchestrator"
            and e["event_type"] == "release_binary_preflight_failed"
        ]
        assert len(failures) == 2, failures
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_promoted_artifacts_snapshot_to_pool() -> None:
    work = Path(tempfile.mkdtemp(prefix="uma-rebel-pool-snapshot-"))
    try:
        loop = work / "loop"
        iter_dir = loop / "iter-3"
        train_dir = iter_dir / "train"
        train_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = train_dir / "checkpoint.pt"
        onnx = iter_dir / "policy.onnx"
        meta = iter_dir / "policy.onnx.meta.json"
        checkpoint.write_bytes(b"checkpoint")
        onnx.write_bytes(b"onnx")
        meta.write_text(json.dumps({"meta": True}), encoding="utf8")

        snapshot = snapshot_promoted_artifacts(
            loop,
            {
                "iteration": 3,
                "checkpoint": str(checkpoint),
                "onnx": str(onnx),
                "wilson_lower": 0.42,
            },
        )

        pool_dir = loop / "pool" / "iter-003"
        assert snapshot["dir"] == str(pool_dir), snapshot
        assert Path(snapshot["checkpoint"]).read_bytes() == b"checkpoint"
        assert Path(snapshot["onnx"]).read_bytes() == b"onnx"
        assert json.loads(Path(snapshot["onnx_meta"]).read_text(encoding="utf8")) == {"meta": True}
        assert snapshot["wilson_lower"] == 0.42
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> None:
    test_replay_mix_materializes_bounded_old_fraction()
    test_replay_passthrough_when_disabled()
    test_fixed_kl_anchor_stays_pinned()
    test_release_binary_preflight_reports_missing_and_stale()
    test_promoted_artifacts_snapshot_to_pool()
    print(json.dumps({"status": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
