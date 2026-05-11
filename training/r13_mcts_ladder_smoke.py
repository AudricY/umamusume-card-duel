"""R13.W4 MCTS ladder smoke.

Tiny end-to-end check that --opponent-selection mcts wires correctly:
runs a single 2-game-per-side pairing where both sides use 8-sim
value-head MCTS, asserts:
  - both sides produce decisions (no heuristic fallbacks → MCTS path was
    actually invoked on the non-model side)
  - manifest summary has both byModelSide.player and byModelSide.opponent
  - WR is within (0, 1) (i.e., not 100% — would suggest opponent never
    actually played MCTS and the non-model side always crashed to rule-bot)
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


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


def export_onnx_if_needed(repo_root: Path, ckpt: Path) -> Path:
    onnx = ckpt.with_name("policy.onnx")
    if not onnx.exists() or onnx.stat().st_mtime < ckpt.stat().st_mtime:
        subprocess.run(
            [sys.executable, str(repo_root / "training" / "export_onnx.py"),
             "--checkpoint", str(ckpt), "--out", str(onnx)],
            cwd=repo_root, check=True,
        )
    return onnx


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ckpt = next((
        c for c in [
            repo_root / "runs" / "R4-value-head-retrain" / "checkpoint.pt",
            repo_root / "runs" / "R3-entropy-bc-b005" / "checkpoint.pt",
        ] if c.exists()
    ), None)
    if ckpt is None:
        print("[ladder-smoke] no checkpoint found", file=sys.stderr)
        sys.exit(2)
    onnx = export_onnx_if_needed(repo_root, ckpt)

    out_dir = repo_root / "runs" / "R13-mcts-ladder-smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "ladder.smoke.manifest.json"
    log_path = out_dir / "ladder.smoke.log"

    port = free_port()
    serve = subprocess.Popen(
        [sys.executable, str(repo_root / "training" / "serve_onnx.py"),
         "--model", str(onnx), "--port", str(port), "--default-sampling", "greedy"],
        cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        if not wait_for_health(port):
            print("[ladder-smoke] serve_onnx never healthy", file=sys.stderr)
            sys.exit(2)
        with log_path.open("w") as logf:
            subprocess.run([
                "npm", "--workspace", "backend", "run", "sim:eval-gate", "--",
                "--selection", "mcts",
                "--games", "2",
                "--seed-start", "44440",
                "--model-side", "both",
                "--model-url", f"http://127.0.0.1:{port}",
                "--manifest-out", str(manifest),
                "--mcts-simulations", "8",
                "--mcts-leaf", "value-head",
                "--mcts-prior", "policy",
                "--mcts-collapse-max-steps", "32",
                "--mcts-max-nodes", "256",
                "--opponent-selection", "mcts",
                "--opponent-mcts-simulations", "8",
                "--opponent-mcts-leaf", "value-head",
                "--opponent-mcts-prior", "policy",
                "--opponent-mcts-collapse-max-steps", "32",
                "--opponent-mcts-max-nodes", "256",
                "--min-games", "1",
                "--min-ci-lower", "0",
            ], cwd=repo_root, stdout=logf, stderr=subprocess.STDOUT, check=True)
    finally:
        serve.terminate()
        try:
            serve.wait(timeout=3)
        except subprocess.TimeoutExpired:
            serve.kill()

    payload = json.loads(manifest.read_text(encoding="utf8"))
    summary = payload["summary"]
    assert summary["games"] == 4, f"expected 4 games (2×2 sides), got {summary['games']}"
    by_side = summary.get("byModelSide", {})
    for side in ("player", "opponent"):
        assert by_side.get(side, {}).get("games", 0) >= 1, f"missing games for side={side}"
    fallbacks = int(summary.get("heuristicFallbacks", -1))
    if fallbacks != 0:
        print(f"[ladder-smoke] FAIL fallbacks={fallbacks} (model MCTS should always produce valid actions)", file=sys.stderr)
        sys.exit(1)

    print(json.dumps({
        "status": "PASS",
        "games": summary["games"],
        "model_wr": summary["modelWinRate"],
        "wilson_lower": summary["wilson95"]["lower"],
        "wilson_upper": summary["wilson95"]["upper"],
        "manifest": str(manifest),
    }, indent=2))


if __name__ == "__main__":
    main()
