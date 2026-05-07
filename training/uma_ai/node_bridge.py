from __future__ import annotations

import os
import subprocess
from pathlib import Path


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
    env = os.environ.copy()
    env.setdefault("TMPDIR", "/tmp")
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
    subprocess.run(command, cwd=repo_root, env=env, check=True)
