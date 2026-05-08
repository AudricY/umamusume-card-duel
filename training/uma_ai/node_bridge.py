from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable


def _backend_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("TMPDIR", "/tmp")
    return env


def export_training_examples(
    repo_root: str | Path,
    out: str | Path,
    *,
    seed_start: int,
    games: int,
    max_steps: int,
) -> None:
    repo_root = Path(repo_root)
    out = Path(out)
    command = [
        "npm",
        "--workspace",
        "backend",
        "run",
        "sim:export-training",
        "--",
        "--out",
        str(out),
        "--seed-start",
        str(seed_start),
        "--games",
        str(games),
        "--max-steps",
        str(max_steps),
    ]
    subprocess.run(command, cwd=repo_root, env=_backend_env(), check=True)


def run_evaluator(
    repo_root: str | Path,
    *,
    selection: str,
    games: int,
    seed_start: int = 8000,
    model_side: str = "both",
    max_steps: int = 500,
    rollout_steps: int = 500,
    decision_trace_out: str | Path | None = None,
    trace_teacher: str = "none",
    rollout_crn_samples: int = 1,
    planner_crn_samples: int = 3,
    model_url: str | None = None,
    opponent_model_url: str | None = None,
    manifest_out: str | Path | None = None,
    extra: Iterable[str] = (),
) -> None:
    repo_root = Path(repo_root)
    cmd = [
        "npm",
        "--workspace",
        "backend",
        "run",
        "sim:evaluate-model",
        "--",
        "--selection",
        selection,
        "--games",
        str(games),
        "--seed-start",
        str(seed_start),
        "--model-side",
        model_side,
        "--max-steps",
        str(max_steps),
        "--rollout-steps",
        str(rollout_steps),
        "--rollout-crn-samples",
        str(rollout_crn_samples),
        "--planner-crn-samples",
        str(planner_crn_samples),
    ]
    if decision_trace_out is not None:
        cmd += ["--decision-trace-out", str(decision_trace_out)]
    if trace_teacher and trace_teacher != "none":
        cmd += ["--trace-teacher", trace_teacher]
    if model_url is not None:
        cmd += ["--model-url", model_url]
    if opponent_model_url is not None:
        cmd += ["--opponent-model-url", opponent_model_url]
    if manifest_out is not None:
        cmd += ["--manifest-out", str(manifest_out)]
    cmd.extend(extra)
    subprocess.run(cmd, cwd=repo_root, env=_backend_env(), check=True)


def relabel_decision_trace(
    repo_root: str | Path,
    trace_in: str | Path,
    relabeled_out: str | Path,
    *,
    source: str,
    label_source: str,
    manifest_out: str | Path | None = None,
) -> None:
    repo_root = Path(repo_root)
    cmd = [
        "npx",
        "tsx",
        "src/sim/dagger/relabelDecisionTrace.ts",
        "--in",
        str(trace_in),
        "--out",
        str(relabeled_out),
        "--source",
        source,
        "--label-source",
        label_source,
    ]
    if manifest_out is not None:
        cmd += ["--manifest-out", str(manifest_out)]
    subprocess.run(cmd, cwd=repo_root / "backend", env=_backend_env(), check=True)


def mix_sources(
    repo_root: str | Path,
    *,
    out: str | Path,
    seed: str,
    components: Iterable[tuple[str, float, str]],
    manifest_out: str | Path | None = None,
    total_rows: int | None = None,
) -> None:
    repo_root = Path(repo_root)
    cmd = [
        "npx",
        "tsx",
        "src/sim/dagger/mixSources.ts",
        "--out",
        str(out),
        "--seed",
        seed,
    ]
    for path, weight, tag in components:
        cmd += ["--source", f"{path}:{weight}:{tag}"]
    if manifest_out is not None:
        cmd += ["--manifest-out", str(manifest_out)]
    if total_rows is not None:
        cmd += ["--total-rows", str(total_rows)]
    subprocess.run(cmd, cwd=repo_root / "backend", env=_backend_env(), check=True)


def run_eval_gate(
    repo_root: str | Path,
    *,
    selection: str,
    games: int,
    seed_start: int = 9000,
    min_games: int,
    min_ci_lower: float,
    manifest_out: str | Path,
    require_zero_no_ops: bool = True,
    rollout_crn_samples: int = 1,
    planner_crn_samples: int = 3,
    opponent_model_url: str | None = None,
    extra: Iterable[str] = (),
) -> int:
    repo_root = Path(repo_root)
    cmd = [
        "npm",
        "--workspace",
        "backend",
        "run",
        "sim:eval-gate",
        "--",
        "--selection",
        selection,
        "--games",
        str(games),
        "--seed-start",
        str(seed_start),
        "--min-games",
        str(min_games),
        "--min-ci-lower",
        str(min_ci_lower),
        "--manifest-out",
        str(manifest_out),
        "--require-manifest",
        "--rollout-crn-samples",
        str(rollout_crn_samples),
        "--planner-crn-samples",
        str(planner_crn_samples),
    ]
    if not require_zero_no_ops:
        cmd.append("--allow-no-ops")
    if opponent_model_url is not None:
        cmd += ["--opponent-model-url", opponent_model_url]
    cmd.extend(extra)
    completed = subprocess.run(cmd, cwd=repo_root, env=_backend_env())
    return int(completed.returncode)
