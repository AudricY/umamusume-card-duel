"""R13.W1 selfplay parallelism smoke.

Mirrors r13_parallel_smoke.py for the selfplay generator. Runs the same set
of seeds via --workers 1 (control) and --workers 4, asserts:
  - both runs complete successfully
  - row schemas validate (kind, valueTarget, visitDistribution sums to 1)
  - the union of seeds in the parallel run matches the seeds we requested
  - per-seed result tuples (winner, points) match exactly between serial
    and parallel runs (R14.B AsyncLocalStorage fix now in)

Tiny config: 4 games × 8 sims so the smoke completes in well under a minute.
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


def run_selfplay(
    repo_root: Path,
    port: int,
    out_dir: Path,
    workers: int,
    label: str,
) -> tuple[Path, float]:
    out_path = out_dir / f"{label}.selfplay.jsonl"
    if out_path.exists():
        out_path.unlink()
    log_path = out_dir / f"{label}.log"
    started = time.perf_counter()
    with log_path.open("w") as logf:
        subprocess.run(
            [
                "npm", "--workspace", "backend", "run", "sim:mcts-selfplay", "--",
                "--model-url", f"http://127.0.0.1:{port}",
                "--games", "4",
                "--seed-start", "131301",
                "--mcts-simulations", "8",
                "--mcts-c-puct", "1.5",
                "--mcts-prior", "policy",
                "--mcts-collapse-max-steps", "32",
                "--mcts-max-nodes", "256",
                "--temperature-moves", "2",
                "--temperature-value", "1.0",
                "--workers", str(workers),
                "--out", str(out_path),
            ],
            cwd=repo_root, stdout=logf, stderr=subprocess.STDOUT, check=True,
        )
    return out_path, time.perf_counter() - started


def group_by_seed(out_path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for line in out_path.read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        grouped.setdefault(str(row["seed"]), []).append(row)
    return grouped


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ckpt = next((
        c for c in [
            repo_root / "runs" / "R4-value-head-retrain" / "checkpoint.pt",
            repo_root / "runs" / "R3-entropy-bc-b005" / "checkpoint.pt",
        ] if c.exists()
    ), None)
    if ckpt is None:
        print("[r13-selfplay-smoke] no checkpoint found", file=sys.stderr)
        sys.exit(2)
    onnx = export_onnx_if_needed(repo_root, ckpt)

    out_dir = repo_root / "runs" / "R13-selfplay-parallel-smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    port = free_port()
    serve = subprocess.Popen(
        [sys.executable, str(repo_root / "training" / "serve_onnx.py"),
         "--model", str(onnx), "--port", str(port), "--default-sampling", "greedy"],
        cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        if not wait_for_health(port):
            print("[r13-selfplay-smoke] serve_onnx never healthy", file=sys.stderr)
            sys.exit(2)
        serial_path, t_serial = run_selfplay(repo_root, port, out_dir, workers=1, label="w1")
        parallel_path, t_parallel = run_selfplay(repo_root, port, out_dir, workers=4, label="w4")
    finally:
        serve.terminate()
        try:
            serve.wait(timeout=3)
        except subprocess.TimeoutExpired:
            serve.kill()

    serial = group_by_seed(serial_path)
    parallel = group_by_seed(parallel_path)
    expected_seeds = {str(131301 + i) for i in range(4)}
    if set(serial) != expected_seeds:
        print(f"[r13-selfplay-smoke] serial seeds {set(serial)} != expected {expected_seeds}", file=sys.stderr)
        sys.exit(1)
    if set(parallel) != expected_seeds:
        print(f"[r13-selfplay-smoke] parallel seeds {set(parallel)} != expected {expected_seeds}", file=sys.stderr)
        sys.exit(1)

    for rows in parallel.values():
        for row in rows:
            if row["kind"] != "mcts-selfplay":
                print(f"[r13-selfplay-smoke] bad kind={row['kind']!r}", file=sys.stderr)
                sys.exit(1)
            if row["valueTarget"] not in (-1, 0, 1):
                print(f"[r13-selfplay-smoke] valueTarget={row['valueTarget']} ∉ {{-1,0,1}}", file=sys.stderr)
                sys.exit(1)
            dist = row["visitDistribution"]
            if not (0.999 <= sum(dist) <= 1.001):
                print(f"[r13-selfplay-smoke] visit dist sum={sum(dist)}", file=sys.stderr)
                sys.exit(1)

    # R14.B determinism — bit-exact serial==parallel on per-seed first-row
    # result + row count. Should hold now that AsyncLocalStorage propagates
    # rng across async awaits.
    mismatches = []
    for seed in expected_seeds:
        s_first = serial[seed][0]
        p_first = parallel[seed][0]
        if s_first["result"] != p_first["result"]:
            mismatches.append((seed, "result", s_first["result"], p_first["result"]))
        if len(serial[seed]) != len(parallel[seed]):
            mismatches.append((seed, "row_count", len(serial[seed]), len(parallel[seed])))
    if mismatches:
        print(f"[r13-selfplay-smoke] determinism FAIL — {len(mismatches)} divergences (R14.B regression?)", file=sys.stderr)
        for m in mismatches:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)

    speedup = t_serial / max(0.001, t_parallel)
    total_rows_parallel = sum(len(rs) for rs in parallel.values())
    print(json.dumps({
        "status": "PASS",
        "games": len(expected_seeds),
        "rows_parallel": total_rows_parallel,
        "t_serial_sec": round(t_serial, 2),
        "t_parallel_sec": round(t_parallel, 2),
        "speedup": round(speedup, 2),
    }, indent=2))


if __name__ == "__main__":
    main()
