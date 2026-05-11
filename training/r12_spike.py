"""R12 day-1 spike runner.

Stands up serve_onnx for a given checkpoint, runs the eval gate with
``--selection mcts``, and emits stage="mcts-spike" events so dashboard /
TB consumers can identify the spike. The spike is treated as PASS iff
``summary.wilson95.lower >= --min-ci-lower`` and the gate completes with
zero heuristic fallbacks. The plan's GO criterion (>= 0.40) is the caller's
default; pass ``--min-ci-lower 0.30`` etc. to probe weaker thresholds when
debugging.

Usage:
    training/.venv/bin/python training/r12_spike.py \\
        --checkpoint runs/R4-value-head-retrain/checkpoint.pt \\
        --out-dir runs/R12-spike \\
        --games 100 \\
        --mcts-simulations 100 \\
        --min-ci-lower 0.40
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def export_onnx_if_needed(repo_root: Path, ckpt: Path) -> Path:
    onnx = ckpt.with_name("policy.onnx")
    if not onnx.exists() or onnx.stat().st_mtime < ckpt.stat().st_mtime:
        subprocess.run(
            [sys.executable, str(repo_root / "training" / "export_onnx.py"),
             "--checkpoint", str(ckpt), "--out", str(onnx)],
            cwd=repo_root, check=True,
        )
    return onnx


def wait_for_health(port: int, attempts: int = 60) -> bool:
    import urllib.request
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.25)
    return False


def emit_event(events_path: Path, payload: dict[str, Any]) -> None:
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf8") as fh:
        fh.write(json.dumps(payload) + "\n")


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root or Path(__file__).resolve().parents[1])
    ckpt = Path(args.checkpoint).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "events.jsonl"

    onnx_path = export_onnx_if_needed(repo_root, ckpt)
    port = free_port()
    emit_event(events_path, {
        "stage": "mcts-spike", "event_type": "started",
        "checkpoint": str(ckpt), "onnx": str(onnx_path),
        "games": args.games, "mcts_simulations": args.mcts_simulations,
        "mcts_c_puct": args.mcts_c_puct, "min_ci_lower": args.min_ci_lower,
        "ts": time.time(),
    })

    serve_proc = subprocess.Popen(
        [
            sys.executable, str(repo_root / "training" / "serve_onnx.py"),
            "--model", str(onnx_path), "--port", str(port),
            "--default-sampling", "greedy",
        ],
        cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        if not wait_for_health(port):
            raise RuntimeError(f"serve_onnx never reported healthy on :{port}")

        manifest_path = out_dir / "gate.manifest.json"
        progress_path = out_dir / "progress.jsonl"
        gate_log = out_dir / "gate.log"
        gate_started = time.time()
        with gate_log.open("w") as logf:
            result = subprocess.run(
                [
                    "npm", "--workspace", "backend", "run", "sim:eval-gate", "--",
                    "--selection", "mcts",
                    "--games", str(args.games),
                    "--max-steps", "500",
                    "--seed-start", str(args.seed_start),
                    "--model-side", args.model_side,
                    "--model-url", f"http://127.0.0.1:{port}",
                    "--manifest-out", str(manifest_path),
                    "--progress-out", str(progress_path),
                    "--mcts-simulations", str(args.mcts_simulations),
                    "--mcts-c-puct", str(args.mcts_c_puct),
                    "--mcts-leaf", args.mcts_leaf,
                    "--mcts-prior", args.mcts_prior,
                    "--mcts-collapse-max-steps", str(args.mcts_collapse_max_steps),
                    "--mcts-max-nodes", str(args.mcts_max_nodes),
                    "--min-ci-lower", str(args.min_ci_lower),
                    "--min-games", str(args.games),
                ],
                cwd=repo_root,
                stdout=logf, stderr=subprocess.STDOUT,
                check=False,
            )
        gate_elapsed = time.time() - gate_started
    finally:
        serve_proc.terminate()
        try:
            serve_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            serve_proc.kill()

    if not manifest_path.exists():
        emit_event(events_path, {
            "stage": "mcts-spike", "event_type": "failed",
            "reason": "no_manifest", "ts": time.time(),
        })
        print(json.dumps({"status": "NO_GO", "reason": "no_manifest"}, indent=2))
        sys.exit(2)

    payload = json.loads(manifest_path.read_text(encoding="utf8"))
    summary = payload.get("summary", {})
    wilson = summary.get("wilson95", {})
    wilson_lower = float(wilson.get("lower", 0))
    win_rate = float(summary.get("modelWinRate", 0))
    fallbacks = int(summary.get("heuristicFallbacks", 0))

    decision = (
        wilson_lower >= args.min_ci_lower
        and win_rate >= args.min_win_rate
        and fallbacks == 0
    )
    status = "GO" if decision else "NO_GO"

    emit_event(events_path, {
        "stage": "mcts-spike", "event_type": "completed",
        "status": status,
        "wilson_lower": wilson_lower, "win_rate": win_rate, "fallbacks": fallbacks,
        "min_ci_lower": args.min_ci_lower, "min_win_rate": args.min_win_rate,
        "gate_elapsed_sec": gate_elapsed,
        "ts": time.time(),
    })
    print(json.dumps({
        "status": status,
        "wilson_lower": wilson_lower,
        "win_rate": win_rate,
        "fallbacks": fallbacks,
        "elapsed_sec": gate_elapsed,
        "manifest": str(manifest_path),
    }, indent=2))
    if status != "GO":
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--repo-root", default=None)
    p.add_argument("--games", type=int, default=100,
                   help="Side-balanced games per side (total games = 2 × games when --model-side both).")
    p.add_argument("--mcts-simulations", type=int, default=100)
    p.add_argument("--mcts-c-puct", type=float, default=1.5)
    p.add_argument("--mcts-leaf", default="value-head")
    p.add_argument("--mcts-prior", default="uniform", choices=["uniform", "policy"])
    p.add_argument("--mcts-collapse-max-steps", type=int, default=64)
    p.add_argument("--mcts-max-nodes", type=int, default=5000)
    p.add_argument("--model-side", default="both", choices=["both", "player", "opponent"])
    p.add_argument("--seed-start", type=int, default=9000)
    p.add_argument("--min-ci-lower", type=float, default=0.40)
    p.add_argument("--min-win-rate", type=float, default=0.43)
    return p.parse_args()


if __name__ == "__main__":
    main()
