"""R13.W4 MCTS-vs-MCTS strength ladder.

Runs head-to-head pairings between MCTS configurations to provide a yardstick
that doesn't saturate against the rule-bot. Three pairings:
  1. R4@200sims rollout-leaf vs R4@50sims rollout-leaf (more compute should win)
  2. R4@100sims rollout-leaf vs R4@100sims value-head-leaf (rollout leaf strictly
     stronger per R12's decomposition; should be a clean signal)
  3. R4 vs R4 at fixed sims, value-head only (sanity: ~50/50)

Each pairing runs N games × 2 sides (player and opponent), gates the WR with
Wilson 95% CI. Writes `ladder.json` with the matrix.

Sanity assertion: pairing 1 should show the higher-sims side winning ≥60% of
the time (Wilson lower > 0.45). If ~50/50, search is noise-limited at these
sim counts and W3 (value-head retrain) becomes more urgent, not less.
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


def export_onnx_if_needed(repo_root: Path, ckpt: Path) -> Path:
    onnx = ckpt.with_name("policy.onnx")
    if not onnx.exists() or onnx.stat().st_mtime < ckpt.stat().st_mtime:
        subprocess.run(
            [sys.executable, str(repo_root / "training" / "export_onnx.py"),
             "--checkpoint", str(ckpt), "--out", str(onnx)],
            cwd=repo_root, check=True,
        )
    return onnx


def run_pairing(
    repo_root: Path,
    port: int,
    out_dir: Path,
    name: str,
    games_per_side: int,
    workers: int,
    seed_start: int,
    model_cfg: dict,
    opponent_cfg: dict,
) -> dict:
    manifest = out_dir / f"{name}.manifest.json"
    progress = out_dir / f"{name}.progress.jsonl"
    log_path = out_dir / f"{name}.log"
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
        "--mcts-simulations", str(model_cfg["sims"]),
        "--mcts-c-puct", "1.5",
        "--mcts-leaf", model_cfg["leaf"],
        "--mcts-prior", model_cfg.get("prior", "policy"),
        "--mcts-rollout-crn-samples", str(model_cfg.get("rollout_crn", 3)),
        "--mcts-rollout-steps", str(model_cfg.get("rollout_steps", 200)),
        "--mcts-collapse-max-steps", "64",
        "--mcts-max-nodes", "5000",
        "--opponent-selection", "mcts",
        "--opponent-mcts-simulations", str(opponent_cfg["sims"]),
        "--opponent-mcts-c-puct", "1.5",
        "--opponent-mcts-leaf", opponent_cfg["leaf"],
        "--opponent-mcts-prior", opponent_cfg.get("prior", "policy"),
        "--opponent-mcts-rollout-crn-samples", str(opponent_cfg.get("rollout_crn", 3)),
        "--opponent-mcts-rollout-steps", str(opponent_cfg.get("rollout_steps", 200)),
        "--opponent-mcts-collapse-max-steps", "64",
        "--opponent-mcts-max-nodes", "5000",
        "--min-games", "1",
        "--min-ci-lower", "0",
    ]
    started = time.perf_counter()
    with log_path.open("w") as logf:
        subprocess.run(cmd, cwd=repo_root, stdout=logf, stderr=subprocess.STDOUT, check=True)
    elapsed = time.perf_counter() - started
    payload = json.loads(manifest.read_text(encoding="utf8"))
    summary = payload["summary"]
    return {
        "name": name,
        "elapsed_sec": round(elapsed, 1),
        "model_cfg": model_cfg,
        "opponent_cfg": opponent_cfg,
        "games": summary["games"],
        "model_win_rate": summary["modelWinRate"],
        "wilson_lower": summary["wilson95"]["lower"],
        "wilson_upper": summary["wilson95"]["upper"],
        "by_side": summary.get("byModelSide", {}),
        "manifest": str(manifest),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-per-side", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=200000)
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
        print("[mcts-ladder] no checkpoint found", file=sys.stderr)
        sys.exit(2)
    onnx = export_onnx_if_needed(repo_root, ckpt)

    out_dir = Path(cli_args.out_dir) if cli_args.out_dir else (repo_root / "runs" / "R13-mcts-ladder")
    out_dir.mkdir(parents=True, exist_ok=True)

    port = free_port()
    serve = subprocess.Popen(
        [sys.executable, str(repo_root / "training" / "serve_onnx.py"),
         "--model", str(onnx), "--port", str(port), "--default-sampling", "greedy"],
        cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    pairings: list[dict] = []
    try:
        if not wait_for_health(port):
            print("[mcts-ladder] serve_onnx never healthy", file=sys.stderr)
            sys.exit(2)
        pairings.append(run_pairing(
            repo_root, port, out_dir,
            name="sim-200-vs-sim-50-rollout",
            games_per_side=cli_args.games_per_side,
            workers=cli_args.workers,
            seed_start=cli_args.seed_start,
            model_cfg={"sims": 200, "leaf": "rollout", "prior": "policy", "rollout_crn": 3},
            opponent_cfg={"sims": 50, "leaf": "rollout", "prior": "policy", "rollout_crn": 3},
        ))
        pairings.append(run_pairing(
            repo_root, port, out_dir,
            name="rollout-leaf-vs-value-head-leaf",
            games_per_side=cli_args.games_per_side,
            workers=cli_args.workers,
            seed_start=cli_args.seed_start + 10_000,
            model_cfg={"sims": 100, "leaf": "rollout", "prior": "policy", "rollout_crn": 3},
            opponent_cfg={"sims": 100, "leaf": "value-head", "prior": "policy"},
        ))
        pairings.append(run_pairing(
            repo_root, port, out_dir,
            name="self-mirror-value-head-sanity",
            games_per_side=cli_args.games_per_side,
            workers=cli_args.workers,
            seed_start=cli_args.seed_start + 20_000,
            model_cfg={"sims": 100, "leaf": "value-head", "prior": "policy"},
            opponent_cfg={"sims": 100, "leaf": "value-head", "prior": "policy"},
        ))
    finally:
        serve.terminate()
        try:
            serve.wait(timeout=3)
        except subprocess.TimeoutExpired:
            serve.kill()

    sanity = next((p for p in pairings if p["name"] == "sim-200-vs-sim-50-rollout"), None)
    sanity_ok = bool(sanity and sanity["wilson_lower"] >= 0.45)

    output = {
        "checkpoint": str(ckpt),
        "games_per_side": cli_args.games_per_side,
        "workers": cli_args.workers,
        "pairings": pairings,
        "sanity_check": {
            "name": "sim-200-vs-sim-50-rollout should win Wilson lower >= 0.45",
            "passed": sanity_ok,
            "wilson_lower": sanity["wilson_lower"] if sanity else None,
        },
    }
    ladder_path = out_dir / "ladder.json"
    ladder_path.write_text(json.dumps(output, indent=2), encoding="utf8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
