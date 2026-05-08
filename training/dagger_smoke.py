"""3-iteration DAgger orchestrator smoke.

Drives ``dagger_orchestrator.py`` with tiny configs (1-3 games per
iteration, baseline-as-gate to skip the ONNX server bring-up) so the
chained pipeline is exercised end-to-end and a per-iteration manifest
tree is produced under a temp directory.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from opponent_pool import OpponentPool  # noqa: E402  (path bootstrap)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    work_dir = Path(tempfile.mkdtemp(prefix="uma-dagger-orchestrator-"))
    try:
        cmd = [
            sys.executable,
            str(repo_root / "training" / "dagger_orchestrator.py"),
            "--out-dir",
            str(work_dir),
            "--iterations",
            "3",
            "--games",
            "1",
            "--max-steps",
            "120",
            "--teacher",
            "rollout",
            "--rollout-steps",
            "40",
            "--rollout-crn-samples",
            "2",
            "--replay-games",
            "2",
            "--epochs",
            "2",
            "--batch-size",
            "16",
            "--hidden-dim",
            "32",
            "--depth",
            "1",
            "--eval-games",
            "1",
            "--skip-policy-gate",
            # Item 12 escape hatch: pool matchups need two simultaneous
            # serve_onnx instances, which would dominate the smoke wall
            # clock. The unit-style smoke in opponent_pool_smoke.py
            # exercises the data-structure invariants directly.
            "--pool-eval-games",
            "0",
        ]
        subprocess.run(cmd, cwd=repo_root, check=True)
        state = json.loads((work_dir / "orchestrator-state.json").read_text(encoding="utf8"))
        if len(state.get("iterations", [])) != 3:
            raise AssertionError(f"Expected 3 iterations recorded, got {state.get('iterations')}")
        seen = set()
        for iteration in state["iterations"]:
            seen.add(int(iteration["iteration"]))
            iter_dir = work_dir / f"iter-{int(iteration['iteration']):03d}"
            for required in ("trace.jsonl", "relabeled.jsonl", "mixed.jsonl", "iteration-manifest.json", "model"):
                if not (iter_dir / required).exists():
                    raise AssertionError(f"Missing {required} under {iter_dir}")
            if int(iteration["mixed_rows"]) <= 0:
                raise AssertionError(f"Iteration {iteration['iteration']} produced no mixed rows")
            for required_field in ("pool_evals", "pool_aggregate_wilson_lower", "cycling_alarm"):
                if required_field not in iteration:
                    raise AssertionError(
                        f"Iteration {iteration['iteration']} missing {required_field}: {iteration}"
                    )
            if iteration["pool_evals"]:
                # --pool-eval-games 0 should skip pool matchups cleanly.
                raise AssertionError(
                    f"--pool-eval-games 0 should produce no pool eval rows, got {iteration['pool_evals']}"
                )
        if seen != {0, 1, 2}:
            raise AssertionError(f"Iteration ids unexpected: {sorted(seen)}")
        promoted_count = sum(1 for it in state["iterations"] if it.get("promoted"))
        # First iteration always seeds the promoted checkpoint when the
        # gate passes; later iterations may reject without breaking the
        # smoke contract. We just demand at least one promotion so the
        # gate->promote path is exercised.
        if promoted_count < 1:
            raise AssertionError(f"Expected at least one promoted iteration, got {state}")

        # Item 12: opponent-pool persistence + per-iteration snapshot.
        pool_path = work_dir / "opponent-pool.json"
        if not pool_path.exists():
            raise AssertionError(f"Missing opponent-pool.json under {work_dir}")
        pool = OpponentPool.from_json(pool_path)
        # Each promoted iteration snapshots one entry. Iteration 2 must
        # have at least 1 promoted entry behind it; iteration 3, at
        # least 2. We assert the floor since the rule-bot gate is
        # noisy at this scale and may promote all three.
        promoted_iters = [int(it["iteration"]) for it in state["iterations"] if it.get("promoted")]
        expected_pool_size = len(promoted_iters)
        if len(pool.entries) != expected_pool_size:
            raise AssertionError(
                f"Pool size {len(pool.entries)} != promoted-iteration count {expected_pool_size}"
            )
        if expected_pool_size >= 2:
            # When 2 or more iterations promote, the pool must reflect that.
            assert len(pool.entries) >= 2, f"expected >=2 pool entries, got {len(pool.entries)}"
        for entry in pool.entries:
            if not Path(entry.checkpoint_path).exists():
                raise AssertionError(f"Pool entry checkpoint missing on disk: {entry}")

        # Item 12: cycling alarm unit-style assertion on a planted
        # rock-paper-scissors regression. Independent of the smoke run
        # so it doesn't depend on the noisy pipeline.
        planted_history = {
            4: [0.4, 0.5, 0.5, 0.45],  # not strictly declining (tie at 0.5,0.5)
            7: [0.7, 0.6, 0.5, 0.4],   # strict decline on last 3 → must trip
        }
        flagged = OpponentPool.cycling_alarm(planted_history)
        if flagged != [7]:
            raise AssertionError(
                f"planted RPS regression should flag opponent 7 only, got {flagged}"
            )
        print(json.dumps({
            "status": "PASS",
            "work_dir": str(work_dir),
            "promoted_count": promoted_count,
            "pool_size": len(pool.entries),
        }, indent=2))
    finally:
        # Keep work_dir for inspection on failure; clean only on success.
        if "PASS" in str(sys.exc_info()[1] or ""):
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
