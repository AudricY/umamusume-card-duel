"""R14.B engine determinism smoke (post-AsyncLocalStorage fix).

Validates that withRng's RNG context now survives async/await boundaries so
serial-vs-parallel gate runs produce bit-exact per-game results on identical
seeds. Pre-fix (R13.W1) this was *not* required; the engine's withRng wrapper
lost its module-level activeRng across awaits and silently fell back to
Math.random(). After R14.B's AsyncLocalStorage installation, all four `randomFloat`/
`randomInt`/`shuffle` call sites that fire after an `await` (combat,
trainers, setup, ai/core) should read from the per-async-context store.

Test:
  1. Run sim:eval-gate with --workers 1 over a fixed seed range; record
     per-(seed, side) winner + turnNumber.
  2. Re-run with --workers 4 over the same seed range; record same.
  3. Assert per-(seed, side) winner AND turnNumber match EXACTLY between
     the two runs.

If this passes, R-WILD's W1 serial-vs-parallel divergence is closed.

Tiny config so the smoke runs in ~30s.
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


def run_gate(
    repo_root: Path,
    port: int,
    out_dir: Path,
    workers: int,
    label: str,
) -> tuple[Path, float]:
    manifest = out_dir / f"{label}.manifest.json"
    progress = out_dir / f"{label}.progress.jsonl"
    log_path = out_dir / f"{label}.log"
    started = time.perf_counter()
    with log_path.open("w") as logf:
        subprocess.run(
            [
                "npm", "--workspace", "backend", "run", "sim:eval-gate", "--",
                "--selection", "mcts",
                "--games", "6",
                "--max-steps", "500",
                "--seed-start", "141414",
                "--model-side", "both",
                "--model-url", f"http://127.0.0.1:{port}",
                "--manifest-out", str(manifest),
                "--progress-out", str(progress),
                "--workers", str(workers),
                "--mcts-simulations", "8",
                "--mcts-c-puct", "1.5",
                "--mcts-leaf", "value-head",
                "--mcts-collapse-max-steps", "64",
                "--mcts-max-nodes", "512",
                "--min-games", "1",
                "--min-ci-lower", "0",
            ],
            cwd=repo_root, stdout=logf, stderr=subprocess.STDOUT, check=True,
        )
    return progress, time.perf_counter() - started


def load_per_game(progress_path: Path) -> dict[tuple[str, str], dict]:
    by_key: dict[tuple[str, str], dict] = {}
    for line in progress_path.read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        key = (str(obj["seed"]), str(obj["modelSide"]))
        by_key[key] = {
            "winner": obj["winner"],
            "modelWon": obj["modelWon"],
            "turnNumber": obj["turnNumber"],
        }
    return by_key


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ckpt_candidates = [
        repo_root / "runs" / "R4-value-head-retrain" / "checkpoint.pt",
        repo_root / "runs" / "R3-entropy-bc-b005" / "checkpoint.pt",
    ]
    ckpt = next((c for c in ckpt_candidates if c.exists()), None)
    if ckpt is None:
        print("[r14-determinism] no checkpoint found in expected locations", file=sys.stderr)
        sys.exit(2)

    onnx = export_onnx_if_needed(repo_root, ckpt)
    out_dir = repo_root / "runs" / "R14-determinism-smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    port = free_port()
    serve_proc = subprocess.Popen(
        [sys.executable, str(repo_root / "training" / "serve_onnx.py"),
         "--model", str(onnx), "--port", str(port), "--default-sampling", "greedy"],
        cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        if not wait_for_health(port):
            print("[r14-determinism] serve_onnx never healthy", file=sys.stderr)
            sys.exit(2)
        prog_serial, t_serial = run_gate(repo_root, port, out_dir, workers=1, label="w1")
        prog_parallel, t_parallel = run_gate(repo_root, port, out_dir, workers=4, label="w4")
    finally:
        serve_proc.terminate()
        try:
            serve_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            serve_proc.kill()

    serial = load_per_game(prog_serial)
    parallel = load_per_game(prog_parallel)

    expected_count = 12  # 6 games * 2 sides
    if len(serial) != expected_count or len(parallel) != expected_count:
        print(f"[r14-determinism] expected {expected_count} (seed, side) tuples; got serial={len(serial)} parallel={len(parallel)}", file=sys.stderr)
        sys.exit(1)

    if set(serial.keys()) != set(parallel.keys()):
        print(f"[r14-determinism] seed-side keys differ: serial={sorted(serial)} parallel={sorted(parallel)}", file=sys.stderr)
        sys.exit(1)

    mismatches: list[dict] = []
    for key in sorted(serial.keys()):
        s = serial[key]
        p = parallel[key]
        if s["winner"] != p["winner"] or s["turnNumber"] != p["turnNumber"]:
            mismatches.append({
                "key": list(key),
                "serial": s,
                "parallel": p,
            })

    result = {
        "status": "FAIL" if mismatches else "PASS",
        "games": len(serial),
        "t_serial_sec": round(t_serial, 2),
        "t_parallel_sec": round(t_parallel, 2),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "note": "post-R14.B AsyncLocalStorage fix; serial-vs-parallel must be bit-exact",
    }
    print(json.dumps(result, indent=2))
    if mismatches:
        sys.exit(1)


if __name__ == "__main__":
    main()
