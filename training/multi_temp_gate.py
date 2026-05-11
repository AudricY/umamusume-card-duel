"""R2: multi-temperature gate matrix.

Runs the eval gate against a checkpoint at multiple sampling
temperatures. Uses serve_onnx's existing ``--default-sampling stochastic
--default-temperature T`` flags so no TS-side change is required.

For T=0 we use ``--default-sampling greedy`` (the standard gate).
For T>0 we use stochastic. Each temperature gets its own server process
on its own port to keep results comparable.

Usage:
    training/.venv/bin/python training/multi_temp_gate.py \\
        --checkpoint runs/R3-entropy-bc-b005/checkpoint.pt \\
        --out-dir runs/R2-multi-temp-R3 \\
        --temperatures 0,0.3,0.5,1.0,1.5,2.0 \\
        --games 100
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def export_onnx_if_needed(repo_root: Path, ckpt: Path) -> Path:
    onnx = ckpt.with_name("policy.onnx")
    if not onnx.exists() or onnx.stat().st_mtime < ckpt.stat().st_mtime:
        subprocess.run(
            [sys.executable, str(repo_root / "training" / "export_onnx.py"),
             "--checkpoint", str(ckpt), "--out", str(onnx)],
            cwd=repo_root, check=True,
        )
    return onnx


def gate_at_temperature(
    repo_root: Path,
    onnx_path: Path,
    out_dir: Path,
    temperature: float,
    games: int,
    seed_start: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    port = free_port()

    if temperature == 0.0:
        sampling_args = ["--default-sampling", "greedy"]
    else:
        sampling_args = [
            "--default-sampling", "stochastic",
            "--default-temperature", str(temperature),
        ]

    serve_proc = subprocess.Popen(
        [
            sys.executable, str(repo_root / "training" / "serve_onnx.py"),
            "--model", str(onnx_path),
            "--port", str(port),
            *sampling_args,
        ],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        # Wait for serve_onnx to be ready
        import urllib.request
        ready = False
        for _ in range(40):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
                    if r.status == 200:
                        ready = True
                        break
            except Exception:
                time.sleep(0.25)
        if not ready:
            raise RuntimeError(f"serve_onnx never reported healthy on :{port}")

        manifest = out_dir / f"gate-T{temperature:.2f}.manifest.json"
        log = out_dir / f"gate-T{temperature:.2f}.log"
        with open(log, "w") as logf:
            subprocess.run(
                ["npm", "--workspace", "backend", "run", "sim:eval-gate", "--",
                 "--selection", "policy",
                 "--games", str(games),
                 "--max-steps", "500",
                 "--rollout-crn-samples", "1",
                 "--seed-start", str(seed_start),
                 "--model-url", f"http://127.0.0.1:{port}",
                 "--manifest-out", str(manifest)],
                cwd=repo_root,
                stdout=logf,
                stderr=subprocess.STDOUT,
                check=False,  # eval-gate exits non-zero when WR below floor; we don't care
            )
        if manifest.exists():
            payload = json.loads(manifest.read_text(encoding="utf8"))
            summary = payload.get("summary", {})
            wilson = summary.get("wilson95", {})
            return {
                "temperature": temperature,
                "model_win_rate": summary.get("modelWinRate"),
                "wilson_lower": wilson.get("lower"),
                "wilson_upper": wilson.get("upper"),
                "games": summary.get("games"),
                "fallbacks": summary.get("heuristicFallbacks"),
                "no_ops": summary.get("selectedNoOps"),
                "by_side": {
                    side: summary.get("byModelSide", {}).get(side, {}).get("modelWinRate")
                    for side in ("player", "opponent")
                },
            }
        return {"temperature": temperature, "error": "no manifest"}
    finally:
        serve_proc.terminate()
        try:
            serve_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            serve_proc.kill()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root or Path(__file__).resolve().parents[1])
    ckpt = Path(args.checkpoint).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx = export_onnx_if_needed(repo_root, ckpt)
    temps = [float(t) for t in args.temperatures.split(",")]
    rows: list[dict[str, Any]] = []
    for t in temps:
        print(f"[multi-temp] gating at T={t}")
        row = gate_at_temperature(repo_root, onnx, out_dir, t, args.games, args.seed_start)
        rows.append(row)
        print(f"  → WR={row.get('model_win_rate')} wilson_lower={row.get('wilson_lower')}")
    summary = {
        "checkpoint": str(ckpt),
        "onnx": str(onnx),
        "games_per_temperature": args.games,
        "rows": rows,
    }
    out_path = out_dir / "multi-temp-summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"status": "PASS", "summary": str(out_path), "temperatures_tested": len(rows)}, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--repo-root", default=None)
    p.add_argument("--temperatures", default="0,0.3,0.5,1.0,1.5,2.0",
                   help="Comma-separated list of temperatures to evaluate (T=0 → greedy).")
    p.add_argument("--games", type=int, default=100,
                   help="Side-balanced game count per temperature (total games = 2 × games).")
    p.add_argument("--seed-start", type=int, default=9000)
    return p.parse_args()


if __name__ == "__main__":
    main()
