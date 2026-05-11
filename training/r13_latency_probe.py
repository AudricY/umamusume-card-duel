"""R13.W2 latency probe.

Runs the eval gate at three latency-dial settings and reports (WR, Wilson95,
p95 decision latency, mean sims actually run). Helps pick a Pareto setting
for live UI play (W5) where p95 decision latency < 3 s with Wilson lower
above some floor.

Settings probed:
  A. Baseline: rollout-leaf K=3, 100 sims, no adaptive halt
  B. K=1 cheap rollouts: rollout-leaf K=1, 100 sims, no adaptive halt
  C. K=1 + adaptive halt: rollout-leaf K=1, 100 sims, adaptive ratio 3.0

Decision latency is measured externally per game via wall-clock / modelActions.
That's an OK proxy for per-decision latency on the model side. For more
accurate p95 we'd need per-decision instrumentation in TS; here we use
per-game elapsed / actions as an average estimate and report the per-game
distribution for percentile reading.
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
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


def run_setting(
    repo_root: Path,
    port: int,
    out_dir: Path,
    label: str,
    games_per_side: int,
    workers: int,
    seed_start: int,
    sims: int,
    rollout_crn: int,
    adaptive_ratio: float,
) -> dict:
    manifest = out_dir / f"{label}.manifest.json"
    progress = out_dir / f"{label}.progress.jsonl"
    log_path = out_dir / f"{label}.log"
    cmd = [
        "npm", "--workspace", "backend", "run", "sim:eval-gate", "--",
        "--selection", "mcts",
        "--games", str(games_per_side),
        "--max-steps", "500",
        "--seed-start", str(seed_start),
        "--model-side", "both",
        "--model-url", f"http://127.0.0.1:{port}",
        "--manifest-out", str(manifest),
        "--progress-out", str(progress),
        "--workers", str(workers),
        "--mcts-simulations", str(sims),
        "--mcts-c-puct", "1.5",
        "--mcts-leaf", "rollout",
        "--mcts-prior", "policy",
        "--mcts-rollout-crn-samples", str(rollout_crn),
        "--mcts-rollout-steps", "200",
        "--mcts-collapse-max-steps", "64",
        "--mcts-max-nodes", "5000",
        "--mcts-adaptive-ratio", str(adaptive_ratio),
        "--mcts-adaptive-min-sims", "20",
        "--min-games", "1",
        "--min-ci-lower", "0",
    ]
    started = time.perf_counter()
    with log_path.open("w") as logf:
        subprocess.run(cmd, cwd=repo_root, stdout=logf, stderr=subprocess.STDOUT, check=True)
    elapsed = time.perf_counter() - started
    payload = json.loads(manifest.read_text(encoding="utf8"))
    summary = payload["summary"]
    # Per-game wall-clock from progress for percentile reading.
    per_game_secs: list[float] = []
    per_decision_secs: list[float] = []
    if progress.exists():
        for line in progress.read_text(encoding="utf8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            g_sec = float(row.get("gameElapsedSec", 0))
            m_actions = int(row.get("modelActions", 0))
            per_game_secs.append(g_sec)
            if m_actions > 0:
                per_decision_secs.append(g_sec / m_actions)
    def pctile(xs: list[float], p: float) -> float | None:
        if not xs:
            return None
        s = sorted(xs)
        k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
        return round(s[k], 3)
    return {
        "label": label,
        "elapsed_sec": round(elapsed, 1),
        "games": summary["games"],
        "wr": summary["modelWinRate"],
        "wilson_lower": summary["wilson95"]["lower"],
        "wilson_upper": summary["wilson95"]["upper"],
        "decision_sec_mean": round(statistics.mean(per_decision_secs), 3) if per_decision_secs else None,
        "decision_sec_p50": pctile(per_decision_secs, 0.50),
        "decision_sec_p95": pctile(per_decision_secs, 0.95),
        "game_sec_mean": round(statistics.mean(per_game_secs), 1) if per_game_secs else None,
        "game_sec_p95": pctile(per_game_secs, 0.95),
        "sims": sims,
        "rollout_crn": rollout_crn,
        "adaptive_ratio": adaptive_ratio,
        "manifest": str(manifest),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-per-side", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=300000)
    parser.add_argument("--out-dir", default=None)
    cli_args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    ckpt = next((
        c for c in [
            repo_root / "runs" / "R4-value-head-retrain" / "checkpoint.pt",
            repo_root / "runs" / "R3-entropy-bc-b005" / "checkpoint.pt",
        ] if c.exists()
    ), None)
    if ckpt is None:
        print("[latency-probe] no checkpoint found", file=sys.stderr)
        sys.exit(2)
    onnx = export_onnx_if_needed(repo_root, ckpt)

    out_dir = Path(cli_args.out_dir) if cli_args.out_dir else (repo_root / "runs" / "R13-latency-probe")
    out_dir.mkdir(parents=True, exist_ok=True)

    port = free_port()
    serve = subprocess.Popen(
        [sys.executable, str(repo_root / "training" / "serve_onnx.py"),
         "--model", str(onnx), "--port", str(port), "--default-sampling", "greedy"],
        cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    runs: list[dict] = []
    try:
        if not wait_for_health(port):
            print("[latency-probe] serve_onnx never healthy", file=sys.stderr)
            sys.exit(2)
        runs.append(run_setting(
            repo_root, port, out_dir,
            label="A-baseline-K3-100sims",
            games_per_side=cli_args.games_per_side, workers=cli_args.workers,
            seed_start=cli_args.seed_start,
            sims=100, rollout_crn=3, adaptive_ratio=0,
        ))
        runs.append(run_setting(
            repo_root, port, out_dir,
            label="B-K1-100sims",
            games_per_side=cli_args.games_per_side, workers=cli_args.workers,
            seed_start=cli_args.seed_start + 10_000,
            sims=100, rollout_crn=1, adaptive_ratio=0,
        ))
        runs.append(run_setting(
            repo_root, port, out_dir,
            label="C-K1-100sims-adaptive3.0",
            games_per_side=cli_args.games_per_side, workers=cli_args.workers,
            seed_start=cli_args.seed_start + 20_000,
            sims=100, rollout_crn=1, adaptive_ratio=3.0,
        ))
    finally:
        serve.terminate()
        try:
            serve.wait(timeout=3)
        except subprocess.TimeoutExpired:
            serve.kill()

    output = {"checkpoint": str(ckpt), "runs": runs}
    (out_dir / "latency.json").write_text(json.dumps(output, indent=2), encoding="utf8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
