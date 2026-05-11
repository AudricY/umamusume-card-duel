"""R12 phase C distillation smoke.

Trains train_bc.py with --data-mode mcts-distill for 2 epochs on the
phase-B smoke selfplay rows. Asserts:
- The script exits 0.
- A checkpoint.pt is written.
- Train loss strictly decreased from epoch 1 to epoch 2.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    selfplay_path = repo_root / "runs" / "R12-selfplay-smoke" / "selfplay.jsonl"
    if not selfplay_path.exists():
        print(f"[distill-smoke] missing {selfplay_path}; run r12_selfplay_smoke.py first", file=sys.stderr)
        sys.exit(2)

    out_dir = repo_root / "runs" / "R12-distill-smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "events.jsonl"
    events_path.unlink(missing_ok=True)

    result = subprocess.run(
        [
            str(repo_root / "training" / ".venv" / "bin" / "python"),
            str(repo_root / "training" / "train_bc.py"),
            "--data", str(selfplay_path),
            "--out-dir", str(out_dir),
            "--epochs", "2",
            "--batch-size", "8",
            "--hidden-dim", "64",
            "--depth", "2",
            "--lr", "3e-4",
            "--value-weight", "1.0",
            "--policy-weight", "1.0",
            "--data-mode", "mcts-distill",
            "--split-by", "seed",
            "--device", "cpu",
            "--events-out", str(events_path),
            "--verbose",
        ],
        cwd=repo_root, check=False, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[distill-smoke] train_bc failed: {result.returncode}", file=sys.stderr)
        sys.stderr.write(result.stdout + "\n" + result.stderr + "\n")
        sys.exit(1)

    ckpt = out_dir / "checkpoint.pt"
    if not ckpt.exists():
        print(f"[distill-smoke] missing checkpoint at {ckpt}", file=sys.stderr)
        sys.exit(1)

    epoch_lines: list[dict] = []
    if events_path.exists():
        with events_path.open("r", encoding="utf8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    if obj.get("event_type") == "epoch":
                        epoch_lines.append(obj)
    if len(epoch_lines) < 2:
        print(f"[distill-smoke] expected ≥2 epoch events, got {len(epoch_lines)}", file=sys.stderr)
        sys.exit(1)
    losses = [float((line.get("data") or {}).get("train_loss") or 0) for line in epoch_lines]
    if losses[1] >= losses[0]:
        print(f"[distill-smoke] FAIL — loss did not decrease: {losses}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps({
        "status": "PASS",
        "epochs": len(epoch_lines),
        "loss_epoch1": losses[0],
        "loss_epoch2": losses[1],
        "checkpoint": str(ckpt),
    }, indent=2))


if __name__ == "__main__":
    main()
