"""F1 PPO hyperparameter sweep.

Drives ``ppo_orchestrator.py`` across the grid defined in
``docs/archive/ai-research/f1-design.md``. Each cell × seed becomes a separate sweep
subdirectory containing the orchestrator's full output tree (events,
manifests, state). The sweep collects the final iteration's Wilson
lower bound per trial and ranks cells by mean Wilson lower across seeds.

Grid (after dropping the KL-coef axis since adaptive KL isn't enabled
yet — see docs/archive/ai-research/f1-design.md "Stability controls"):

    lr           ∈ {3e-5, 1e-4}
    entropy_coef ∈ {0.001, 0.005}
    gae_lambda   ∈ {0.9, 0.95}

Default: 2³ = 8 cells × 3 seeds = 24 trials, ~10 min/trial on the user's
box at smoke scale, fits the 24 cell-hour spec budget.

``--quick`` runs a 2-cell × 1-seed × 1-iter version (4 trials) for the
smoke. ``f1_hp_sweep_smoke.py`` invokes that variant.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Production grid from docs/archive/ai-research/f1-design.md.
LR_VALUES = [3e-5, 1e-4]
ENTROPY_VALUES = [0.001, 0.005]
GAE_LAMBDA_VALUES = [0.9, 0.95]

# Reduced grid for --quick. One default-vs-non-default cell so the smoke
# exercises sweep aggregation across at least two cells.
QUICK_GRID: list[dict[str, float]] = [
    {"lr": 3e-5, "entropy_coef": 0.005, "gae_lambda": 0.95},  # defaults
    {"lr": 1e-4, "entropy_coef": 0.001, "gae_lambda": 0.9},   # opposite corner
]


@dataclass
class CellResult:
    cell_id: int
    cell_index: tuple[int, int, int]
    hp: dict[str, float]
    seeds: list[int]
    seed_wilson_lower: list[float]  # parallel to seeds
    seed_failed: list[bool]
    out_dirs: list[Path]

    @property
    def mean_wilson_lower(self) -> float:
        valid = [w for w, f in zip(self.seed_wilson_lower, self.seed_failed) if not f]
        return statistics.mean(valid) if valid else 0.0

    @property
    def stdev_wilson_lower(self) -> float:
        valid = [w for w, f in zip(self.seed_wilson_lower, self.seed_failed) if not f]
        return statistics.stdev(valid) if len(valid) >= 2 else 0.0

    @property
    def trials_completed(self) -> int:
        return sum(1 for f in self.seed_failed if not f)


def build_grid(quick: bool) -> list[dict[str, float]]:
    if quick:
        return list(QUICK_GRID)
    grid: list[dict[str, float]] = []
    for lr, ent, lam in itertools.product(LR_VALUES, ENTROPY_VALUES, GAE_LAMBDA_VALUES):
        grid.append({"lr": lr, "entropy_coef": ent, "gae_lambda": lam})
    return grid


def run_trial(
    repo_root: Path,
    out_dir: Path,
    *,
    warm_start: Path,
    hp: dict[str, float],
    seed: int,
    iterations: int,
    games_per_update: int,
    max_steps: int,
    eval_games: int,
    rollout_steps: int,
    extra_args: list[str],
) -> tuple[float, bool]:
    """Run one ppo_orchestrator trial. Returns (final_wilson_lower, failed)."""

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(repo_root / "training" / "ppo_orchestrator.py"),
        "--out-dir", str(out_dir),
        "--init-from", str(warm_start),
        "--iterations", str(iterations),
        "--games-per-update", str(games_per_update),
        "--max-steps", str(max_steps),
        "--rollout-steps", str(rollout_steps),
        "--eval-games", str(eval_games),
        "--lr", str(hp["lr"]),
        "--entropy-coef", str(hp["entropy_coef"]),
        "--gae-lambda", str(hp["gae_lambda"]),
        "--trace-seed-start", str(14000 + seed * 1000),
        "--eval-seed-start", str(9000 + seed * 1000),
        *extra_args,
    ]
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    state_path = out_dir / "orchestrator-state.json"
    if proc.returncode != 0 or not state_path.exists():
        (out_dir / "trial-error.txt").write_text(
            f"returncode={proc.returncode}\nstdout-tail:\n{proc.stdout[-2000:]}\nstderr-tail:\n{proc.stderr[-2000:]}\n",
            encoding="utf8",
        )
        return 0.0, True
    state = json.loads(state_path.read_text(encoding="utf8"))
    iters = state.get("iterations", [])
    if not iters:
        return 0.0, True
    # Use the last iteration's wilson lower (final policy after all PPO updates).
    return float(iters[-1].get("wilson_lower", 0.0)), False


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root or Path(__file__).resolve().parents[1])
    sweep_dir = Path(args.out_dir).resolve()
    sweep_dir.mkdir(parents=True, exist_ok=True)

    grid = build_grid(args.quick)
    seeds = list(range(args.seeds_per_cell))
    extra_args: list[str] = []
    if args.skip_policy_gate:
        extra_args.append("--skip-policy-gate")

    print(f"[hp-sweep] {len(grid)} cells × {len(seeds)} seeds = {len(grid) * len(seeds)} trials")

    cells: list[CellResult] = []
    for cell_id, hp in enumerate(grid):
        cell_dir = sweep_dir / f"cell-{cell_id:02d}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        # Encode cell_index for reference (LR, entropy, lambda axis positions).
        cell_index = (
            LR_VALUES.index(hp["lr"]) if hp["lr"] in LR_VALUES else -1,
            ENTROPY_VALUES.index(hp["entropy_coef"]) if hp["entropy_coef"] in ENTROPY_VALUES else -1,
            GAE_LAMBDA_VALUES.index(hp["gae_lambda"]) if hp["gae_lambda"] in GAE_LAMBDA_VALUES else -1,
        )
        wilson_lowers: list[float] = []
        failed: list[bool] = []
        out_dirs: list[Path] = []
        for seed in seeds:
            trial_dir = cell_dir / f"seed-{seed}"
            print(f"[hp-sweep] cell {cell_id} seed {seed} hp={hp}")
            wl, fail = run_trial(
                repo_root,
                trial_dir,
                warm_start=Path(args.warm_start),
                hp=hp,
                seed=seed,
                iterations=args.iterations,
                games_per_update=args.games_per_update,
                max_steps=args.max_steps,
                eval_games=args.eval_games,
                rollout_steps=args.rollout_steps,
                extra_args=extra_args,
            )
            wilson_lowers.append(wl)
            failed.append(fail)
            out_dirs.append(trial_dir)
        cells.append(CellResult(
            cell_id=cell_id,
            cell_index=cell_index,
            hp=hp,
            seeds=seeds,
            seed_wilson_lower=wilson_lowers,
            seed_failed=failed,
            out_dirs=out_dirs,
        ))

    cells_ranked = sorted(cells, key=lambda c: c.mean_wilson_lower, reverse=True)
    winner = cells_ranked[0] if cells_ranked else None
    runner_up = cells_ranked[1] if len(cells_ranked) >= 2 else None

    summary: dict[str, Any] = {
        "warm_start": str(args.warm_start),
        "grid_size": len(grid),
        "seeds_per_cell": len(seeds),
        "iterations": args.iterations,
        "games_per_update": args.games_per_update,
        "quick": args.quick,
        "cells": [
            {
                "cell_id": c.cell_id,
                "cell_index": list(c.cell_index),
                "hp": c.hp,
                "mean_wilson_lower": c.mean_wilson_lower,
                "stdev_wilson_lower": c.stdev_wilson_lower,
                "trials_completed": c.trials_completed,
                "seed_wilson_lower": c.seed_wilson_lower,
                "seed_failed": c.seed_failed,
                "out_dirs": [str(p) for p in c.out_dirs],
            }
            for c in cells
        ],
        "ranking": [c.cell_id for c in cells_ranked],
        "winner": {
            "cell_id": winner.cell_id,
            "hp": winner.hp,
            "mean_wilson_lower": winner.mean_wilson_lower,
        } if winner else None,
        "runner_up": {
            "cell_id": runner_up.cell_id,
            "hp": runner_up.hp,
            "mean_wilson_lower": runner_up.mean_wilson_lower,
        } if runner_up else None,
    }
    summary_path = sweep_dir / "sweep-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf8")
    all_cells_have_a_trial = all(c.trials_completed > 0 for c in cells)
    print(json.dumps({
        "status": "PASS" if all_cells_have_a_trial else "FAIL",
        "sweep_summary": str(summary_path),
        "cells_complete": sum(1 for c in cells if c.trials_completed > 0),
        "cells_total": len(cells),
        "winner_cell": winner.cell_id if winner else None,
        "winner_hp": winner.hp if winner else None,
        "winner_wilson_lower": winner.mean_wilson_lower if winner else None,
    }, indent=2))
    if not all_cells_have_a_trial:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-start", required=True,
                        help="Path to DAgger-produced checkpoint.pt to warm-start every trial from.")
    parser.add_argument("--out-dir", required=True,
                        help="Sweep output dir; subtree: cell-NN/seed-SS/<orchestrator output>.")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--quick", action="store_true",
                        help="2-cell grid for smoke runs.")
    parser.add_argument("--seeds-per-cell", type=int, default=3,
                        help="Trials per cell. Prod: 3. Smoke: 1.")
    parser.add_argument("--iterations", type=int, default=3,
                        help="PPO iterations per trial.")
    parser.add_argument("--games-per-update", type=int, default=50,
                        help="Games collected per PPO update. Prod 800 (docs/archive/ai-research/f1-design.md); smoke uses much less.")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--rollout-steps", type=int, default=200)
    parser.add_argument("--eval-games", type=int, default=100,
                        help="Side-balanced games for the gate eval per iter.")
    parser.add_argument("--skip-policy-gate", action="store_true",
                        help="Use baseline-mode gate to avoid ONNX server bring-up (smoke).")
    return parser.parse_args()


if __name__ == "__main__":
    main()
