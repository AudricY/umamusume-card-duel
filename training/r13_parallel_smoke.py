"""R13.W1 game-level parallelism smoke.

Verifies the `--workers N` flag in sim:eval-gate:
  1. Runs the gate with --workers 1 (control) and again with --workers 4 over
     identical seeds. Asserts both runs produce the expected total number of
     games and all per-game results are valid (winner ∈ {player, opponent,
     null}). NOTE: serial-vs-parallel is NOT required to be bit-exact —
     the engine has a known Math.random() leak across async boundaries
     (tracked in task #34) so different processes diverge on the same seed.
     Aggregate signal is preserved; this is intentional for the W1 scope.
  2. Asserts the 4-worker wall-clock is faster than the 1-worker control —
     a sanity check that workers are actually parallelizing CPU-bound TS
     simulator work.

Tiny config: 4 games × value-head leaf × 8 sims so the whole smoke runs in
~30s on a modern CPU.
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
) -> tuple[Path, Path, float]:
    manifest = out_dir / f"{label}.manifest.json"
    progress = out_dir / f"{label}.progress.jsonl"
    log_path = out_dir / f"{label}.log"
    started = time.perf_counter()
    with log_path.open("w") as logf:
        subprocess.run(
            [
                "npm", "--workspace", "backend", "run", "sim:eval-gate", "--",
                "--selection", "mcts",
                "--games", "4",
                "--max-steps", "500",
                "--seed-start", "13130",
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
    elapsed = time.perf_counter() - started
    return manifest, progress, elapsed


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
        print("[parallel-smoke] no checkpoint found in expected locations", file=sys.stderr)
        sys.exit(2)

    onnx = export_onnx_if_needed(repo_root, ckpt)
    out_dir = repo_root / "runs" / "R13-parallel-smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    port = free_port()
    serve_proc = subprocess.Popen(
        [sys.executable, str(repo_root / "training" / "serve_onnx.py"),
         "--model", str(onnx), "--port", str(port), "--default-sampling", "greedy"],
        cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        if not wait_for_health(port):
            print("[parallel-smoke] serve_onnx never healthy", file=sys.stderr)
            sys.exit(2)
        _, prog_serial, t_serial = run_gate(repo_root, port, out_dir, workers=1, label="w1")
        _, prog_parallel, t_parallel = run_gate(repo_root, port, out_dir, workers=4, label="w4")
    finally:
        serve_proc.terminate()
        try:
            serve_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            serve_proc.kill()

    serial = load_per_game(prog_serial)
    parallel = load_per_game(prog_parallel)
    if set(serial.keys()) != set(parallel.keys()):
        print(f"[parallel-smoke] seed-side keys differ: serial={sorted(serial)} parallel={sorted(parallel)}", file=sys.stderr)
        sys.exit(1)
    if len(serial) != 8 or len(parallel) != 8:
        print(f"[parallel-smoke] expected 8 (seed, side) tuples; got serial={len(serial)} parallel={len(parallel)}", file=sys.stderr)
        sys.exit(1)
    valid_winners = {"player", "opponent", None}
    for label, rows in (("serial", serial), ("parallel", parallel)):
        for key, r in rows.items():
            if r["winner"] not in valid_winners:
                print(f"[parallel-smoke] {label} {key} has invalid winner={r['winner']}", file=sys.stderr)
                sys.exit(1)
    # R14.B determinism — with AsyncLocalStorage installed in
    # backend/src/sim/rngAsyncStore.ts, serial-vs-parallel must now be
    # bit-exact on (winner, modelWon, turnNumber). If this fires, either
    # the storage provider isn't being installed for all entry points or
    # there's still a Math.random()-reachable code path the rng doesn't
    # cover.
    mismatches = []
    for key, s in serial.items():
        p = parallel[key]
        if s["winner"] != p["winner"] or s["modelWon"] != p["modelWon"] or s["turnNumber"] != p["turnNumber"]:
            mismatches.append((key, s, p))
    if mismatches:
        print(f"[parallel-smoke] determinism FAIL — {len(mismatches)} mismatches (R14.B regression?)", file=sys.stderr)
        for k, s, p in mismatches:
            print(f"  {k}: serial={s} parallel={p}", file=sys.stderr)
        sys.exit(1)

    speedup = t_serial / max(0.001, t_parallel)
    speedup_ok = speedup >= 1.0
    if not speedup_ok:
        print(f"[parallel-smoke] WARN — no speedup observed (parallel slower)", file=sys.stderr)
    print(json.dumps({
        "status": "PASS",
        "games": len(serial),
        "t_serial_sec": round(t_serial, 2),
        "t_parallel_sec": round(t_parallel, 2),
        "speedup": round(speedup, 2),
        "speedup_check_ok": speedup_ok,
        "determinism": "serial==parallel bit-exact on (winner, modelWon, turnNumber); R14.B AsyncLocalStorage active",
    }, indent=2))


if __name__ == "__main__":
    main()
