"""Smoke for ``f1_hp_sweep.py``.

Trains a tiny warm-start checkpoint via the existing python-train smoke
flow, then runs ``f1_hp_sweep.py --quick`` (2 cells × 1 seed × 1 PPO
iteration) and verifies the resulting ``sweep-summary.json`` carries a
winner + runner-up + per-cell Wilson lower bounds. The smoke does NOT
verify that the sweep picked a "good" cell — only that the plumbing
runs end-to-end and the per-cell aggregation is computed correctly.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    work_dir = Path(tempfile.mkdtemp(prefix="uma-hp-sweep-"))
    try:
        # 1. Produce a tiny warm-start checkpoint via the existing examples
        # exporter + train_bc path. This is the same data spine smoke_e2e
        # uses, just with the minimal flags.
        examples = work_dir / "examples.jsonl"
        subprocess.run(
            [
                "npm", "--workspace", "backend", "run", "sim:export-training", "--",
                "--out", str(examples),
                "--games", "4",
                "--max-steps", "60",
            ],
            cwd=repo_root,
            check=True,
            env={**os.environ, "TMPDIR": str(work_dir)},
        )
        warm_dir = work_dir / "warm"
        subprocess.run(
            [
                sys.executable, str(repo_root / "training" / "train_bc.py"),
                "--data", str(examples),
                "--out-dir", str(warm_dir),
                "--epochs", "2",
                "--batch-size", "16",
                "--hidden-dim", "32",
                "--depth", "1",
            ],
            cwd=repo_root,
            check=True,
        )
        warm_ckpt = warm_dir / "checkpoint.pt"
        if not warm_ckpt.exists():
            raise AssertionError(f"Warm-start checkpoint missing: {warm_ckpt}")

        # 2. Run the HP sweep in --quick mode against the warm-start.
        sweep_dir = work_dir / "sweep"
        cmd = [
            sys.executable, str(repo_root / "training" / "f1_hp_sweep.py"),
            "--warm-start", str(warm_ckpt),
            "--out-dir", str(sweep_dir),
            "--quick",
            "--seeds-per-cell", "1",
            "--iterations", "1",
            "--games-per-update", "2",
            "--max-steps", "60",
            "--rollout-steps", "40",
            "--eval-games", "2",
            "--skip-policy-gate",
        ]
        subprocess.run(cmd, cwd=repo_root, check=True)

        # 3. Verify the sweep summary.
        summary_path = sweep_dir / "sweep-summary.json"
        if not summary_path.exists():
            raise AssertionError(f"sweep-summary.json missing under {sweep_dir}")
        summary = json.loads(summary_path.read_text(encoding="utf8"))

        if summary["grid_size"] != 2:
            raise AssertionError(f"expected grid_size=2 in --quick; got {summary['grid_size']}")
        if summary["seeds_per_cell"] != 1:
            raise AssertionError(f"expected seeds_per_cell=1; got {summary['seeds_per_cell']}")
        if not summary.get("winner") or not summary.get("runner_up"):
            raise AssertionError(f"winner/runner_up missing: {summary}")
        if len(summary["cells"]) != 2:
            raise AssertionError(f"expected 2 cell records; got {len(summary['cells'])}")

        # Each cell must report a Wilson lower bound and at least one
        # completed trial.
        for cell in summary["cells"]:
            if cell["trials_completed"] < 1:
                raise AssertionError(f"cell {cell['cell_id']} had zero completed trials: {cell}")
            if not isinstance(cell["mean_wilson_lower"], (int, float)):
                raise AssertionError(f"cell {cell['cell_id']} mean_wilson_lower not numeric: {cell}")
            if not (0.0 <= cell["mean_wilson_lower"] <= 1.0):
                raise AssertionError(f"cell {cell['cell_id']} mean_wilson_lower out of [0,1]: {cell['mean_wilson_lower']}")

        # Ranking must be a permutation of cell_ids.
        ranking = summary["ranking"]
        if sorted(ranking) != [c["cell_id"] for c in sorted(summary["cells"], key=lambda c: c["cell_id"])]:
            raise AssertionError(f"ranking is not a permutation of cell_ids: {ranking}")

        # Winner must be the head of the ranking.
        if summary["winner"]["cell_id"] != ranking[0]:
            raise AssertionError(f"winner cell_id {summary['winner']['cell_id']} != ranking[0] {ranking[0]}")

        print(json.dumps({
            "status": "PASS",
            "work_dir": str(work_dir),
            "winner_cell": summary["winner"]["cell_id"],
            "winner_hp": summary["winner"]["hp"],
            "cells_complete": sum(1 for c in summary["cells"] if c["trials_completed"] > 0),
        }, indent=2))
    finally:
        # Keep work_dir on failure for forensics.
        if "PASS" in str(sys.exc_info()[1] or ""):
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
