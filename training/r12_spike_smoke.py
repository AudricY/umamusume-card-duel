"""R12 day-1 spike smoke.

Runs a tiny end-to-end MCTS eval (4 games × 8 simulations × value-head leaf)
to verify the plumbing: TS dispatch, /predict transport, manifest write, no
crashes. Does NOT enforce a win-rate threshold — strength is the gate's job.

Asserts:
- The serve_onnx process becomes healthy.
- The eval-gate produces a manifest with the expected schema fields.
- Every recorded game has at least one model decision.
- The summary reports 0 heuristic fallbacks (MCTS should always produce a
  valid action; falling back means a wiring bug).
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
    ckpt_candidates = [
        repo_root / "runs" / "R4-value-head-retrain" / "checkpoint.pt",
        repo_root / "runs" / "R3-entropy-bc-b005" / "checkpoint.pt",
    ]
    ckpt = next((c for c in ckpt_candidates if c.exists()), None)
    if ckpt is None:
        print("[spike-smoke] no checkpoint found in expected locations", file=sys.stderr)
        sys.exit(2)

    onnx = export_onnx_if_needed(repo_root, ckpt)
    out_dir = repo_root / "runs" / "R12-spike-smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    port = free_port()
    serve_proc = subprocess.Popen(
        [sys.executable, str(repo_root / "training" / "serve_onnx.py"),
         "--model", str(onnx), "--port", str(port), "--default-sampling", "greedy"],
        cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        if not wait_for_health(port):
            print("[spike-smoke] serve_onnx never healthy", file=sys.stderr)
            sys.exit(2)
        manifest = out_dir / "gate.manifest.json"
        log_path = out_dir / "gate.log"
        with log_path.open("w") as logf:
            result = subprocess.run(
                [
                    "npm", "--workspace", "backend", "run", "sim:eval-gate", "--",
                    "--selection", "mcts",
                    "--games", "4",
                    "--max-steps", "500",
                    "--seed-start", "9999",
                    "--model-side", "both",
                    "--model-url", f"http://127.0.0.1:{port}",
                    "--manifest-out", str(manifest),
                    "--mcts-simulations", "8",
                    "--mcts-c-puct", "1.5",
                    "--mcts-leaf", "value-head",
                    "--mcts-collapse-max-steps", "64",
                    "--mcts-max-nodes", "512",
                    "--min-games", "1",
                    "--min-ci-lower", "0",
                ],
                cwd=repo_root, stdout=logf, stderr=subprocess.STDOUT,
                check=False,
            )
    finally:
        serve_proc.terminate()
        try:
            serve_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            serve_proc.kill()

    if not manifest.exists():
        print(f"[spike-smoke] manifest missing — see {log_path}", file=sys.stderr)
        sys.exit(2)
    payload = json.loads(manifest.read_text(encoding="utf8"))
    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        print("[spike-smoke] manifest summary missing", file=sys.stderr)
        sys.exit(2)
    assert "games" in summary and "modelWinRate" in summary and "wilson95" in summary, summary
    assert summary["games"] == 8, f"expected 8 games (4 × 2 sides), got {summary['games']}"
    fallbacks = int(summary.get("heuristicFallbacks", -1))
    if fallbacks != 0:
        print(f"[spike-smoke] FAIL — fallbacks={fallbacks} (MCTS must produce valid actions)", file=sys.stderr)
        sys.exit(1)
    by_side = summary.get("byModelSide", {})
    for side in ("player", "opponent"):
        side_summary = by_side.get(side, {})
        assert side_summary.get("games", 0) >= 1, f"missing games for side={side}"
    print(json.dumps({
        "status": "PASS",
        "games": summary["games"],
        "win_rate": summary["modelWinRate"],
        "wilson_lower": summary["wilson95"]["lower"],
        "wilson_upper": summary["wilson95"]["upper"],
        "heuristic_fallbacks": fallbacks,
        "manifest": str(manifest),
    }, indent=2))


if __name__ == "__main__":
    main()
