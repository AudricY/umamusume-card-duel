"""R13.W3 value-retrain experiment — full e2e driver.

Three stages with explicit go/no-go decision rules at the end:
  1. Rollout-leaf selfplay (N games, --workers 4) → corpus with rootValue logged.
  2. Value-head retrain on the corpus (frozen trunk + policy head).
  3. Eval gate at --mcts-leaf value-head with the retrained checkpoint.

Verdict (Wilson lower 95% on the gate WR):
  >= 0.40 → GO: value head is fixable. Phase D (W6) becomes viable.
  0.30 to 0.40 → PARTIAL: full Phase D iter still worth trying with this head as warm-start.
  < 0.30 → NO_GO: value-head distillation doesn't work. Search-at-inference is the
           permanent production path.

Note: stages 1 and 3 stand up their own serve_onnx — stage 1 against the init
checkpoint (for the rollout-leaf MCTS to use the same prior R4 had), stage 3
against the retrained checkpoint.
"""

from __future__ import annotations

import argparse
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


def export_onnx(repo_root: Path, ckpt: Path) -> Path:
    onnx = ckpt.with_name("policy.onnx")
    subprocess.run(
        [sys.executable, str(repo_root / "training" / "export_onnx.py"),
         "--checkpoint", str(ckpt), "--out", str(onnx)],
        cwd=repo_root, check=True,
    )
    return onnx


def export_onnx_if_needed(repo_root: Path, ckpt: Path) -> Path:
    onnx = ckpt.with_name("policy.onnx")
    if not onnx.exists() or onnx.stat().st_mtime < ckpt.stat().st_mtime:
        return export_onnx(repo_root, ckpt)
    return onnx


def spawn_serve(repo_root: Path, onnx: Path, port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, str(repo_root / "training" / "serve_onnx.py"),
         "--model", str(onnx), "--port", str(port), "--default-sampling", "greedy"],
        cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if not wait_for_health(port):
        proc.terminate()
        proc.wait(timeout=5)
        raise RuntimeError(f"serve_onnx never healthy on port {port}")
    return proc


def stop_serve(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()


def run_selfplay(repo_root: Path, port: int, out_path: Path, games: int, workers: int, seed_start: int, log_path: Path) -> None:
    if out_path.exists():
        out_path.unlink()
    with log_path.open("w") as logf:
        subprocess.run([
            "npm", "--workspace", "backend", "run", "sim:mcts-selfplay", "--",
            "--model-url", f"http://127.0.0.1:{port}",
            "--games", str(games),
            "--seed-start", str(seed_start),
            "--workers", str(workers),
            "--mcts-simulations", "100",
            "--mcts-c-puct", "1.5",
            "--mcts-leaf", "rollout",
            "--mcts-rollout-crn-samples", "3",
            "--mcts-rollout-steps", "200",
            "--mcts-prior", "policy",
            "--mcts-collapse-max-steps", "64",
            "--mcts-max-nodes", "5000",
            "--temperature-moves", "6",
            "--temperature-value", "1.0",
            "--out", str(out_path),
        ], cwd=repo_root, stdout=logf, stderr=subprocess.STDOUT, check=True)


def run_retrain(repo_root: Path, data: Path, init_ckpt: Path, retrain_out: Path, events_path: Path, epochs: int, device: str) -> None:
    subprocess.run([
        str(repo_root / "training" / ".venv" / "bin" / "python"),
        str(repo_root / "training" / "r13_value_retrain.py"),
        "--data", str(data),
        "--init-checkpoint", str(init_ckpt),
        "--out-dir", str(retrain_out),
        "--epochs", str(epochs),
        "--batch-size", "128",
        "--lr", "3e-4",
        "--val-fraction", "0.1",
        "--device", device,
        "--events-out", str(events_path),
    ], cwd=repo_root, check=True)


def run_gate(repo_root: Path, port: int, manifest: Path, games_per_side: int, workers: int, seed_start: int, log_path: Path) -> dict:
    with log_path.open("w") as logf:
        subprocess.run([
            "npm", "--workspace", "backend", "run", "sim:eval-gate", "--",
            "--selection", "mcts",
            "--games", str(games_per_side),
            "--seed-start", str(seed_start),
            "--model-side", "both",
            "--model-url", f"http://127.0.0.1:{port}",
            "--manifest-out", str(manifest),
            "--workers", str(workers),
            "--mcts-simulations", "100",
            "--mcts-c-puct", "1.5",
            "--mcts-leaf", "value-head",
            "--mcts-prior", "policy",
            "--mcts-collapse-max-steps", "64",
            "--mcts-max-nodes", "5000",
            "--min-games", "1",
            "--min-ci-lower", "0",
        ], cwd=repo_root, stdout=logf, stderr=subprocess.STDOUT, check=True)
    return json.loads(manifest.read_text(encoding="utf8"))["summary"]


def verdict(wilson_lower: float) -> str:
    if wilson_lower >= 0.40:
        return "GO"
    if wilson_lower >= 0.30:
        return "PARTIAL"
    return "NO_GO"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selfplay-games", type=int, default=100)
    parser.add_argument("--selfplay-seed-start", type=int, default=400000)
    parser.add_argument("--gate-games-per-side", type=int, default=100)
    parser.add_argument("--gate-seed-start", type=int, default=410000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    init_ckpt = next((
        c for c in [
            repo_root / "runs" / "R4-value-head-retrain" / "checkpoint.pt",
            repo_root / "runs" / "R3-entropy-bc-b005" / "checkpoint.pt",
        ] if c.exists()
    ), None)
    if init_ckpt is None:
        print("[w3-experiment] no init checkpoint found", file=sys.stderr)
        sys.exit(2)
    out_dir = Path(args.out_dir) if args.out_dir else (repo_root / "runs" / "R13-value-retrain-experiment")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    selfplay_path = out_dir / "rollout-leaf-selfplay.jsonl"
    selfplay_log = out_dir / "selfplay.log"
    retrain_dir = out_dir / "retrain"
    events_path = out_dir / "events.jsonl"
    if events_path.exists():
        events_path.unlink()
    gate_manifest = out_dir / "gate.manifest.json"
    gate_log = out_dir / "gate.log"

    init_onnx = export_onnx_if_needed(repo_root, init_ckpt)
    sp_port = free_port()
    sp_started = time.perf_counter()
    serve = spawn_serve(repo_root, init_onnx, sp_port)
    try:
        run_selfplay(repo_root, sp_port, selfplay_path, args.selfplay_games, args.workers, args.selfplay_seed_start, selfplay_log)
    finally:
        stop_serve(serve)
    sp_elapsed = time.perf_counter() - sp_started

    rt_started = time.perf_counter()
    run_retrain(repo_root, selfplay_path, init_ckpt, retrain_dir, events_path, args.epochs, device)
    rt_elapsed = time.perf_counter() - rt_started

    retrained_ckpt = retrain_dir / "checkpoint.pt"
    retrained_onnx = export_onnx(repo_root, retrained_ckpt)
    gate_started = time.perf_counter()
    gate_port = free_port()
    serve = spawn_serve(repo_root, retrained_onnx, gate_port)
    try:
        summary = run_gate(repo_root, gate_port, gate_manifest, args.gate_games_per_side, args.workers, args.gate_seed_start, gate_log)
    finally:
        stop_serve(serve)
    gate_elapsed = time.perf_counter() - gate_started

    wl = summary["wilson95"]["lower"]
    v = verdict(wl)
    output = {
        "status": "complete",
        "verdict": v,
        "wilson_lower": wl,
        "wilson_upper": summary["wilson95"]["upper"],
        "win_rate": summary["modelWinRate"],
        "games": summary["games"],
        "by_side": summary.get("byModelSide", {}),
        "init_checkpoint": str(init_ckpt),
        "retrained_checkpoint": str(retrained_ckpt),
        "elapsed": {
            "selfplay_sec": round(sp_elapsed, 1),
            "retrain_sec": round(rt_elapsed, 1),
            "gate_sec": round(gate_elapsed, 1),
        },
        "args": vars(args),
    }
    (out_dir / "verdict.json").write_text(json.dumps(output, indent=2), encoding="utf8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
