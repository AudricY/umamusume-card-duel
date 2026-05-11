"""R13.W2 adaptive halt smoke.

Runs the gate twice with --mcts-simulations 200, --mcts-leaf value-head
(cheap), once with adaptive disabled and once with --mcts-adaptive-ratio 3.0.
Asserts:
  - both runs complete successfully
  - the adaptive run's wall-clock is strictly less than the non-adaptive run
    (i.e., the halt actually fired on at least one decision)

We don't enforce a WR floor — strength is the gate's job. The halt should
not collapse strength on this scale, but to keep the smoke fast we only
assert plumbing correctness.
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


def run_gate(repo_root: Path, port: int, out_dir: Path, label: str, adaptive_ratio: float) -> float:
    manifest = out_dir / f"{label}.manifest.json"
    log_path = out_dir / f"{label}.log"
    started = time.perf_counter()
    with log_path.open("w") as logf:
        subprocess.run([
            "npm", "--workspace", "backend", "run", "sim:eval-gate", "--",
            "--selection", "mcts",
            "--games", "4",
            "--seed-start", "55550",
            "--model-side", "both",
            "--model-url", f"http://127.0.0.1:{port}",
            "--manifest-out", str(manifest),
            "--mcts-simulations", "200",
            "--mcts-leaf", "value-head",
            "--mcts-prior", "policy",
            "--mcts-collapse-max-steps", "64",
            "--mcts-max-nodes", "1024",
            "--mcts-adaptive-ratio", str(adaptive_ratio),
            "--mcts-adaptive-min-sims", "10",
            "--min-games", "1",
            "--min-ci-lower", "0",
        ], cwd=repo_root, stdout=logf, stderr=subprocess.STDOUT, check=True)
    return time.perf_counter() - started


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ckpt = next((
        c for c in [
            repo_root / "runs" / "R4-value-head-retrain" / "checkpoint.pt",
            repo_root / "runs" / "R3-entropy-bc-b005" / "checkpoint.pt",
        ] if c.exists()
    ), None)
    if ckpt is None:
        print("[adaptive-smoke] no checkpoint found", file=sys.stderr)
        sys.exit(2)
    onnx = export_onnx_if_needed(repo_root, ckpt)

    out_dir = repo_root / "runs" / "R13-adaptive-smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    port = free_port()
    serve = subprocess.Popen(
        [sys.executable, str(repo_root / "training" / "serve_onnx.py"),
         "--model", str(onnx), "--port", str(port), "--default-sampling", "greedy"],
        cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        if not wait_for_health(port):
            print("[adaptive-smoke] serve_onnx never healthy", file=sys.stderr)
            sys.exit(2)
        t_off = run_gate(repo_root, port, out_dir, label="off", adaptive_ratio=0)
        t_on = run_gate(repo_root, port, out_dir, label="on", adaptive_ratio=2.0)
    finally:
        serve.terminate()
        try:
            serve.wait(timeout=3)
        except subprocess.TimeoutExpired:
            serve.kill()

    if t_on >= t_off:
        print(f"[adaptive-smoke] FAIL — adaptive halt did not speed up ({t_on:.1f}s vs baseline {t_off:.1f}s)", file=sys.stderr)
        sys.exit(1)
    print(json.dumps({
        "status": "PASS",
        "t_off_sec": round(t_off, 2),
        "t_on_sec": round(t_on, 2),
        "speedup": round(t_off / max(0.001, t_on), 2),
    }, indent=2))


if __name__ == "__main__":
    main()
