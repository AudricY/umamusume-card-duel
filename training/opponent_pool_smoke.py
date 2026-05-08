"""Unit-style smoke for ``training/opponent_pool.py``.

Exercises the pool data structure in isolation: retention, PFSP weight
collapse and skew, cycling-alarm strict-monotone detection, and JSON
round-trip persistence. No subprocesses, no orchestrator pipeline; the
goal is fast, deterministic plumbing coverage so item 12's data layer
doesn't drift silently when downstream callers change.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from opponent_pool import OpponentPool, PoolEntry  # noqa: E402  (path bootstrap)


def _make_entry(iteration: int, *, base: Path, wilson: float = 0.5) -> PoolEntry:
    cp_dir = base / "pool" / f"iter-{iteration:03d}"
    cp_dir.mkdir(parents=True, exist_ok=True)
    cp = cp_dir / "checkpoint.pt"
    cp.write_bytes(b"\0" * 16)
    return PoolEntry(
        iteration=iteration,
        checkpoint_path=str(cp),
        wilson_lower_at_promotion=wilson,
        value_mean_drift_from_prior=0.0,
    )


def test_retention_caps_at_24() -> None:
    work = Path(tempfile.mkdtemp(prefix="uma-pool-retain-"))
    try:
        pool = OpponentPool()
        for i in range(30):
            pool.add(_make_entry(i, base=work))
        pool.retain()
        assert len(pool.entries) <= 24, f"retain should cap at 24, got {len(pool.entries)}"
        # The last 8 promoted entries should always survive.
        kept_iters = {entry.iteration for entry in pool.entries}
        for tail in range(22, 30):
            assert tail in kept_iters, f"recent iteration {tail} should survive retention: {sorted(kept_iters)}"
        # Confirm pruned files were cleaned up under pool/.
        survivors = sorted(p.name for p in (work / "pool").iterdir() if p.is_dir())
        assert len(survivors) == len(pool.entries), (
            f"on-disk pool entries should match in-memory survivors: {survivors} vs {kept_iters}"
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_pfsp_uniform_when_no_win_rates() -> None:
    work = Path(tempfile.mkdtemp(prefix="uma-pool-pfsp-uniform-"))
    try:
        pool = OpponentPool()
        for i in range(4):
            pool.add(_make_entry(i, base=work))
        weights = pool.pfsp_weights({})
        assert len(weights) == 4
        for value in weights.values():
            assert abs(value - 0.25) < 1e-9, f"uniform weight should be 0.25, got {value}"
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_pfsp_skews_toward_losing_matchups() -> None:
    work = Path(tempfile.mkdtemp(prefix="uma-pool-pfsp-skew-"))
    try:
        pool = OpponentPool()
        for i in range(3):
            pool.add(_make_entry(i, base=work))
        # Pretend the current model wins 90% vs iter 0, 50% vs iter 1, 10% vs iter 2.
        # PFSP should weight iter 2 (lossy matchup) the highest.
        weights = pool.pfsp_weights({0: 0.9, 1: 0.5, 2: 0.1})
        assert weights[2] > weights[1] > weights[0], (
            f"weights should skew toward losing matchups, got {weights}"
        )
        # Sum to 1 (probability distribution).
        assert abs(sum(weights.values()) - 1.0) < 1e-9, weights
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_cycling_alarm_strict_monotone() -> None:
    # Plant a strict-decline opponent at iteration 7 and a non-cycling
    # opponent at iteration 4 (one tie breaks strict monotonicity).
    history = {
        4: [0.4, 0.5, 0.5, 0.45],  # last 3: 0.5, 0.5, 0.45 — not strict
        7: [0.7, 0.6, 0.5, 0.4],   # last 3: 0.6, 0.5, 0.4 — strict decline
        9: [0.5, 0.4, 0.5],        # not strictly declining
    }
    flagged = OpponentPool.cycling_alarm(history)
    assert flagged == [7], f"only opponent 7 should trip cycling alarm, got {flagged}"


def test_json_roundtrip() -> None:
    work = Path(tempfile.mkdtemp(prefix="uma-pool-json-"))
    try:
        pool = OpponentPool()
        for i in range(3):
            pool.add(_make_entry(i, base=work, wilson=0.4 + i * 0.05))
        out = work / "opponent-pool.json"
        pool.to_json(out)
        loaded = OpponentPool.from_json(out)
        assert len(loaded.entries) == 3
        for original, recovered in zip(pool.entries, loaded.entries):
            assert original.iteration == recovered.iteration
            assert original.checkpoint_path == recovered.checkpoint_path
            assert abs(original.wilson_lower_at_promotion - recovered.wilson_lower_at_promotion) < 1e-9
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> None:
    test_retention_caps_at_24()
    test_pfsp_uniform_when_no_win_rates()
    test_pfsp_skews_toward_losing_matchups()
    test_cycling_alarm_strict_monotone()
    test_json_roundtrip()
    print(json.dumps({"status": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
