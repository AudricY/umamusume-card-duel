"""R15 single-worker CRN determinism replay (production rollout-leaf config).

Audit gap closed by this script
-------------------------------
R14.B's AsyncLocalStorage fix (R-WILD #34) closed the *parallel-worker*
serial-vs-parallel divergence, validated by training/r14_determinism_smoke.py
(workers=1 vs workers=4 on tiny value-head config). It did NOT audit whether a
single worker is *self-consistent*: replaying the SAME seed end-to-end at
--workers 1 must yield the bit-identical full-game outcome every time. Silent
single-worker non-determinism would invalidate every per-game advantage / Wilson
estimate behind the production claim (search-wrapped rollout-leaf MCTS @ W6
iter-2, Wilson lower 0.6479).

What this does
--------------
1. Serves the FROZEN production checkpoint runs/R13-W6-phase-d/iter-2/policy.onnx
   (the exact artifact behind the 0.6479 gate manifest).
2. Replays the same contiguous seed range R times, each as an independent
   --workers 1 sim:eval-gate run, at the EXACT production rollout-leaf MCTS
   config parsed from runs/R13-W6-phase-d/iter-2/gate.manifest.json.args.
3. Fingerprints each (seed, modelSide) game by (winner, modelWon, turnNumber,
   modelActions) and reports the divergence rate: fraction of (seed, side)
   keys whose fingerprint is NOT identical across all R replays.

This is a thin wrapper over the existing serve_onnx + sim:eval-gate path. It is
NOT an orchestration framework and does NOT train or generate self-play.

Usage
-----
  python training/r15_single_worker_determinism_replay.py \
      --seeds 50 --repeats 3 [--dry-run-seeds 3]

--dry-run-seeds N first runs a quick 2-of-N-seed x 2-repeat probe to estimate
per-seed wall-clock before the full replay (the audit brief caps a single full
replay at ~15 min; if the dry run implies overage, the script prints the
estimate and exits 3 so the caller can pick a smaller N).
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ITER2 = REPO_ROOT / "runs" / "R13-W6-phase-d" / "iter-2"
SEED_START = 9000  # matches the production gate manifest seedStart


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_for_health(port: int, attempts: int = 80) -> bool:
    import urllib.request

    for _ in range(attempts):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.25)
    return False


def load_prod_mcts_args() -> dict:
    manifest = json.load((ITER2 / "gate.manifest.json").open())
    return manifest["args"]


def run_gate(port: int, out_dir: Path, games: int, label: str, prod: dict) -> tuple[Path, float]:
    """One independent --workers 1 eval-gate at the production rollout-leaf config."""
    progress = out_dir / f"{label}.progress.jsonl"
    manifest = out_dir / f"{label}.manifest.json"
    log_path = out_dir / f"{label}.log"
    started = time.perf_counter()
    with log_path.open("w") as logf:
        subprocess.run(
            [
                "npm", "--workspace", "backend", "run", "sim:eval-gate", "--",
                "--selection", "mcts",
                "--games", str(games),
                "--max-steps", str(prod["maxSteps"]),
                "--seed-start", str(SEED_START),
                "--model-side", "both",
                "--model-url", f"http://127.0.0.1:{port}",
                "--manifest-out", str(manifest),
                "--progress-out", str(progress),
                "--workers", "1",
                "--mcts-simulations", str(prod["mctsSimulations"]),
                "--mcts-c-puct", str(prod["mctsCPuct"]),
                "--mcts-leaf", str(prod["mctsLeaf"]),
                "--mcts-rollout-crn-samples", str(prod["mctsRolloutCrnSamples"]),
                "--mcts-rollout-steps", str(prod["mctsRolloutSteps"]),
                "--mcts-collapse-max-steps", str(prod["mctsCollapseMaxSteps"]),
                "--mcts-max-nodes", str(prod["mctsMaxNodes"]),
                "--mcts-prior", str(prod["mctsPrior"]),
                "--opponent-selection", str(prod["opponentSelection"]),
                "--min-games", "1",
                "--min-ci-lower", "0",
            ],
            cwd=REPO_ROOT, stdout=logf, stderr=subprocess.STDOUT, check=True,
        )
    return progress, time.perf_counter() - started


def load_fingerprints(progress_path: Path) -> dict[tuple[str, str], tuple]:
    by_key: dict[tuple[str, str], tuple] = {}
    for line in progress_path.read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("event") != "game_completed":
            continue
        key = (str(obj["seed"]), str(obj["modelSide"]))
        by_key[key] = (
            obj["winner"],
            bool(obj["modelWon"]),
            int(obj["turnNumber"]),
            int(obj["modelActions"]),
        )
    return by_key


def divergence(runs: list[dict[tuple[str, str], tuple]]) -> tuple[float, list[dict]]:
    keys = set(runs[0].keys())
    for r in runs[1:]:
        keys &= set(r.keys())
    diverging: list[dict] = []
    for key in sorted(keys):
        fps = {tuple(r[key]) for r in runs}
        if len(fps) != 1:
            diverging.append({"key": list(key), "fingerprints": [list(fp) for fp in sorted(fps)]})
    rate = len(diverging) / len(keys) if keys else 1.0
    return rate, diverging


def serve_and_replay(out_dir: Path, games: int, repeats: int, prod: dict, tag: str) -> dict:
    onnx = ITER2 / "policy.onnx"
    port = free_port()
    serve_proc = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "training" / "serve_onnx.py"),
         "--model", str(onnx), "--port", str(port), "--default-sampling", "greedy"],
        cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    runs: list[dict[tuple[str, str], tuple]] = []
    times: list[float] = []
    try:
        if not wait_for_health(port):
            print("[r15-determinism] serve_onnx never healthy", file=sys.stderr)
            sys.exit(2)
        for rep in range(repeats):
            prog, dt = run_gate(port, out_dir, games, f"{tag}-rep{rep}", prod)
            runs.append(load_fingerprints(prog))
            times.append(dt)
    finally:
        serve_proc.terminate()
        try:
            serve_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            serve_proc.kill()
    rate, diverging = divergence(runs)
    return {
        "games_per_replay": games,
        "repeats": repeats,
        "seed_side_keys_compared": len(set(runs[0]) & set(runs[-1])),
        "divergence_rate": rate,
        "diverging_keys": diverging,
        "replay_times_sec": [round(t, 1) for t in times],
        "per_seed_sec": round(sum(times) / (games * repeats), 2) if games else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50, help="distinct seeds (=games) per replay")
    ap.add_argument("--repeats", type=int, default=3, help="independent single-worker replays")
    ap.add_argument("--dry-run-seeds", type=int, default=2,
                    help="quick probe seeds to estimate per-seed wall-clock first; 0 to skip")
    ap.add_argument("--max-wall-min", type=float, default=15.0,
                    help="cap for one full replay; exceed => print estimate and exit 3")
    args = ap.parse_args()

    onnx = ITER2 / "policy.onnx"
    ckpt = ITER2 / "checkpoint.pt"
    if not onnx.exists() or not ckpt.exists():
        print(f"[r15-determinism] ENVIRONMENT GAP: missing {onnx} or {ckpt}", file=sys.stderr)
        sys.exit(2)

    out_dir = REPO_ROOT / "runs" / "R15-single-worker-determinism"
    out_dir.mkdir(parents=True, exist_ok=True)
    prod = load_prod_mcts_args()
    print(f"[r15-determinism] production rollout-leaf config: "
          f"leaf={prod['mctsLeaf']} sims={prod['mctsSimulations']} "
          f"crn={prod['mctsRolloutCrnSamples']} rolloutSteps={prod['mctsRolloutSteps']} "
          f"prior={prod['mctsPrior']} opp={prod['opponentSelection']}", flush=True)

    if args.dry_run_seeds and args.dry_run_seeds > 0:
        print(f"[r15-determinism] dry run: {args.dry_run_seeds} seeds x 2 replays ...", flush=True)
        dry = serve_and_replay(out_dir, args.dry_run_seeds, 2, prod, "dry")
        per_seed = dry["per_seed_sec"] or 0.0
        est_full_min = per_seed * args.seeds * 2 / 60.0  # 2 sides per seed
        print(json.dumps({"dry_run": dry, "est_full_replay_min": round(est_full_min, 1)}, indent=2),
              flush=True)
        if est_full_min > args.max_wall_min:
            print(f"[r15-determinism] STOP: est full replay {est_full_min:.1f} min > "
                  f"{args.max_wall_min} min cap. Re-run with smaller --seeds.", file=sys.stderr)
            sys.exit(3)

    print(f"[r15-determinism] full replay: {args.seeds} seeds x {args.repeats} "
          f"single-worker replays ...", flush=True)
    full = serve_and_replay(out_dir, args.seeds, args.repeats, prod, "full")
    result = {
        "status": "PASS" if full["divergence_rate"] == 0.0 else "FAIL",
        "checkpoint": str(onnx.relative_to(REPO_ROOT)),
        "config": {
            "mctsLeaf": prod["mctsLeaf"],
            "mctsSimulations": prod["mctsSimulations"],
            "mctsRolloutCrnSamples": prod["mctsRolloutCrnSamples"],
            "mctsRolloutSteps": prod["mctsRolloutSteps"],
            "mctsPrior": prod["mctsPrior"],
            "opponentSelection": prod["opponentSelection"],
            "seedStart": SEED_START,
            "workers": 1,
        },
        **full,
        "note": "single-worker self-consistency at production rollout-leaf config; "
                "0.0 divergence => per-game advantage/Wilson estimates behind 0.6479 are reproducible",
    }
    out_path = out_dir / "result.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    sys.exit(0 if full["divergence_rate"] == 0.0 else 1)


if __name__ == "__main__":
    main()
