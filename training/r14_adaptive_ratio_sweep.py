"""R14.F adaptive-ratio Pareto sweep at value-head leaf.

R13.W2 smoke showed --mcts-adaptive-ratio=2.0 gave 3.86x speedup at
value-head leaf on a 4-game smoke. Production needs a defended Pareto
point: a ratio that keeps Wilson lower >= 0.42 while cutting wall-clock
>=30% vs ratio=0. The chosen point feeds W5 UI's default config.

Sweeps ratio in {0, 1.5, 2.0, 3.0, 5.0} at 100 sims, value-head leaf,
n=100, seeds 820000+ against the W6 iter-1 checkpoint (the strongest
value-head-leaf-deployable model).

Each sweep is a single sim:eval-gate invocation with --workers 4. The
script writes per-ratio manifests under runs/R14-adaptive-sweep and a
summary JSON with (ratio, wilson_lower, wilson_upper, win_rate,
elapsed_sec, halts_early_count, speedup_vs_baseline). The "chosen"
ratio is the highest ratio satisfying the exit criterion.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path


RATIOS_DEFAULT = [0.0, 1.5, 2.0, 3.0, 5.0]


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


def run_gate(repo_root: Path, port: int, out_dir: Path, ratio: float, args: argparse.Namespace) -> tuple[Path, float]:
    label = f"ratio-{ratio}"
    manifest = out_dir / f"{label}.manifest.json"
    progress = out_dir / f"{label}.progress.jsonl"
    log_path = out_dir / f"{label}.log"
    started = time.perf_counter()
    with log_path.open("w") as logf:
        subprocess.run(
            [
                "npm", "--workspace", "backend", "run", "sim:eval-gate", "--",
                "--selection", "mcts",
                "--games", str(args.games),
                "--max-steps", "500",
                "--seed-start", str(args.seed_start),
                "--model-side", "both",
                "--model-url", f"http://127.0.0.1:{port}",
                "--manifest-out", str(manifest),
                "--progress-out", str(progress),
                "--workers", str(args.workers),
                "--mcts-simulations", str(args.simulations),
                "--mcts-c-puct", "1.5",
                "--mcts-leaf", "value-head",
                "--mcts-prior", "policy",
                "--mcts-collapse-max-steps", "64",
                "--mcts-max-nodes", "5000",
                "--mcts-adaptive-ratio", str(ratio),
                "--mcts-adaptive-min-sims", str(args.adaptive_min_sims),
                "--min-games", "1",
                "--min-ci-lower", "0",
            ],
            cwd=repo_root, stdout=logf, stderr=subprocess.STDOUT, check=True,
        )
    return manifest, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None,
                        help="Override the default W6 iter-1 checkpoint.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--ratios", default=None,
                        help="Comma-separated list of ratios (default 0,1.5,2,3,5).")
    parser.add_argument("--games", type=int, default=50,
                        help="Games per side (so n=2*games total). Default 50 → n=100.")
    parser.add_argument("--seed-start", type=int, default=820000)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--adaptive-min-sims", type=int, default=20)
    parser.add_argument("--target-wilson-lower", type=float, default=0.42)
    parser.add_argument("--target-wallclock-cut", type=float, default=0.30,
                        help="Required wall-clock reduction vs ratio=0 (e.g. 0.30 = 30%%).")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    ckpt = Path(args.checkpoint) if args.checkpoint else (
        repo_root / "runs" / "R13-W6-phase-d" / "iter-1" / "checkpoint.pt"
    )
    if not ckpt.exists():
        print(f"[r14-adaptive-sweep] checkpoint not found: {ckpt}", file=sys.stderr)
        sys.exit(2)

    ratios = [float(x) for x in args.ratios.split(",")] if args.ratios else RATIOS_DEFAULT
    out_dir = Path(args.out_dir) if args.out_dir else (repo_root / "runs" / "R14-adaptive-sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx = export_onnx_if_needed(repo_root, ckpt)
    port = free_port()
    serve_proc = subprocess.Popen(
        [sys.executable, str(repo_root / "training" / "serve_onnx.py"),
         "--model", str(onnx), "--port", str(port), "--default-sampling", "greedy"],
        cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        if not wait_for_health(port):
            print("[r14-adaptive-sweep] serve_onnx never healthy", file=sys.stderr)
            sys.exit(2)
        per_ratio: list[dict] = []
        for ratio in ratios:
            manifest_path, elapsed = run_gate(repo_root, port, out_dir, ratio, args)
            manifest = json.loads(manifest_path.read_text(encoding="utf8"))
            summary = manifest.get("summary", {})
            per_ratio.append({
                "ratio": ratio,
                "wilson_lower": float(summary.get("wilson95", {}).get("lower", 0)),
                "wilson_upper": float(summary.get("wilson95", {}).get("upper", 0)),
                "win_rate": float(summary.get("modelWinRate", 0)),
                "games": int(summary.get("games", 0)),
                "elapsed_sec": round(elapsed, 2),
                "heuristic_fallbacks": int(summary.get("heuristicFallbacks", 0)),
            })
    finally:
        serve_proc.terminate()
        try:
            serve_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            serve_proc.kill()

    baseline_entry = next((row for row in per_ratio if row["ratio"] == 0.0), None)
    baseline_elapsed = baseline_entry["elapsed_sec"] if baseline_entry else None
    for row in per_ratio:
        if baseline_elapsed and baseline_elapsed > 0:
            row["speedup"] = round(baseline_elapsed / row["elapsed_sec"], 2)
            row["wallclock_cut"] = round(1.0 - (row["elapsed_sec"] / baseline_elapsed), 3)
        else:
            row["speedup"] = None
            row["wallclock_cut"] = None

    pareto: list[dict] = [r for r in per_ratio if r["wilson_lower"] >= args.target_wilson_lower]
    if baseline_elapsed and baseline_elapsed > 0:
        pareto = [r for r in pareto if r["ratio"] != 0.0 and r["wallclock_cut"] is not None and r["wallclock_cut"] >= args.target_wallclock_cut]
    chosen = max(pareto, key=lambda r: r["ratio"]) if pareto else None

    summary = {
        "status": "PASS" if chosen else "FAIL",
        "checkpoint": str(ckpt),
        "target_wilson_lower": args.target_wilson_lower,
        "target_wallclock_cut": args.target_wallclock_cut,
        "baseline_elapsed_sec": baseline_elapsed,
        "per_ratio": per_ratio,
        "chosen_ratio": chosen["ratio"] if chosen else None,
        "chosen_wilson_lower": chosen["wilson_lower"] if chosen else None,
        "chosen_speedup": chosen["speedup"] if chosen else None,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
