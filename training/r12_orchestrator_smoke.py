"""R12 phase E orchestrator smoke.

Runs a single mini iteration of r12_orchestrator.py end-to-end:
  - 4 self-play games × 8 MCTS simulations → selfplay.jsonl
  - 2 distill epochs (cpu, hidden_dim=64, depth=2)
  - 8 gate games × 8 simulations → gate.manifest.json
  - promotion decision recorded

Asserts all expected artifacts exist and the expected event_types appear
in events.jsonl. Strength is not enforced (a single iteration with 8 sims
gives noisy WR by design).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    init_ckpt = next((
        c for c in [
            repo_root / "runs" / "R4-value-head-retrain" / "checkpoint.pt",
            repo_root / "runs" / "R3-entropy-bc-b005" / "checkpoint.pt",
        ] if c.exists()
    ), None)
    if init_ckpt is None:
        print("[r12-orch-smoke] no init checkpoint found", file=sys.stderr)
        sys.exit(2)

    out_dir = repo_root / "runs" / "R12-orch-smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale events to make the assertions deterministic.
    events_path = out_dir / "events.jsonl"
    if events_path.exists():
        events_path.unlink()
    state_path = out_dir / "orchestrator-state.json"
    if state_path.exists():
        state_path.unlink()

    cmd = [
        str(repo_root / "training" / ".venv" / "bin" / "python"),
        str(repo_root / "training" / "r12_orchestrator.py"),
        "--out-dir", str(out_dir),
        "--init-checkpoint", str(init_ckpt),
        "--iterations", "1",
        "--selfplay-games", "4",
        "--selfplay-seed-start", "70000",
        "--eval-games", "4",
        "--eval-seed-start", "70500",
        "--eval-min-ci-lower", "0.0",
        "--mcts-simulations", "8",
        "--mcts-c-puct", "1.5",
        "--mcts-collapse-max-steps", "32",
        "--mcts-max-nodes", "256",
        "--dirichlet-alpha", "0.3",
        "--dirichlet-epsilon", "0.25",
        "--temperature-moves", "2",
        "--temperature-value", "1.0",
        "--epochs", "2",
        "--batch-size", "8",
        "--hidden-dim", "64",
        "--depth", "2",
        "--lr", "3e-4",
        "--value-weight", "1.0",
        "--device", "cpu",
    ]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"[r12-orch-smoke] orchestrator exited {result.returncode}", file=sys.stderr)
        sys.stderr.write(result.stdout + "\n" + result.stderr + "\n")
        sys.exit(1)

    iter_dir = out_dir / "iter-0"
    expected = [
        iter_dir / "selfplay.jsonl",
        iter_dir / "selfplay.manifest.json",
        iter_dir / "checkpoint.pt",
        iter_dir / "policy.onnx",
        iter_dir / "gate.manifest.json",
        state_path,
        events_path,
    ]
    for path in expected:
        if not path.exists():
            print(f"[r12-orch-smoke] missing {path}", file=sys.stderr)
            sys.exit(1)

    seen: set[tuple[str, str]] = set()
    with events_path.open("r", encoding="utf8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            seen.add((str(obj.get("stage")), str(obj.get("event_type"))))
    required = {
        ("r12-orchestrator", "run_started"),
        ("r12-orchestrator", "iteration_started"),
        ("selfplay", "started"),
        ("selfplay", "completed"),
        ("distill", "started"),
        ("distill", "completed"),
        ("mcts-gate", "started"),
        ("mcts-gate", "completed"),
        ("r12-orchestrator", "iteration_completed"),
        ("r12-orchestrator", "run_completed"),
    }
    missing = required - seen
    if missing:
        print(f"[r12-orch-smoke] missing event types: {missing}", file=sys.stderr)
        sys.exit(1)

    state_payload = json.loads(state_path.read_text(encoding="utf8"))
    if not state_payload.get("iterations"):
        print("[r12-orch-smoke] state has no iterations recorded", file=sys.stderr)
        sys.exit(1)

    print(json.dumps({
        "status": "PASS",
        "iterations": len(state_payload["iterations"]),
        "promoted_wilson_lower": state_payload.get("promoted_wilson_lower"),
        "state": str(state_path),
    }, indent=2))


if __name__ == "__main__":
    main()
