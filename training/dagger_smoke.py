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
        if seen != {0, 1, 2}:
            raise AssertionError(f"Iteration ids unexpected: {sorted(seen)}")
        promoted_count = sum(1 for it in state["iterations"] if it.get("promoted"))
        # First iteration always seeds the promoted checkpoint when the
        # gate passes; later iterations may reject without breaking the
        # smoke contract. We just demand at least one promotion so the
        # gate->promote path is exercised.
        if promoted_count < 1:
            raise AssertionError(f"Expected at least one promoted iteration, got {state}")
        print(json.dumps({"status": "PASS", "work_dir": str(work_dir), "promoted_count": promoted_count}, indent=2))
    finally:
        # Keep work_dir for inspection on failure; clean only on success.
        if "PASS" in str(sys.exc_info()[1] or ""):
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
