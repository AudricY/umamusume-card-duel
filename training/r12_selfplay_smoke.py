"""R12 phase B: MCTS self-play smoke.

Stands up serve_onnx for an R4-style checkpoint, runs sim:mcts-selfplay for
2 games × 8 sims, and asserts the produced selfplay.jsonl has the expected
schema with backfilled value targets in {-1, 0, +1}.
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


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ckpt = next((
        c for c in [
            repo_root / "runs" / "R4-value-head-retrain" / "checkpoint.pt",
            repo_root / "runs" / "R3-entropy-bc-b005" / "checkpoint.pt",
        ] if c.exists()
    ), None)
    if ckpt is None:
        print("[selfplay-smoke] no checkpoint found", file=sys.stderr)
        sys.exit(2)
    onnx = export_onnx_if_needed(repo_root, ckpt)

    out_dir = repo_root / "runs" / "R12-selfplay-smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    selfplay_path = out_dir / "selfplay.jsonl"
    if selfplay_path.exists():
        selfplay_path.unlink()

    port = free_port()
    serve = subprocess.Popen(
        [sys.executable, str(repo_root / "training" / "serve_onnx.py"),
         "--model", str(onnx), "--port", str(port), "--default-sampling", "greedy"],
        cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        if not wait_for_health(port):
            print("[selfplay-smoke] serve_onnx never healthy", file=sys.stderr)
            sys.exit(2)
        log_path = out_dir / "selfplay.log"
        with log_path.open("w") as logf:
            result = subprocess.run(
                [
                    "npm", "--workspace", "backend", "run", "sim:mcts-selfplay", "--",
                    "--model-url", f"http://127.0.0.1:{port}",
                    "--games", "2",
                    "--seed-start", "55555",
                    "--mcts-simulations", "8",
                    "--mcts-c-puct", "1.5",
                    "--mcts-prior", "policy",
                    "--mcts-collapse-max-steps", "32",
                    "--mcts-max-nodes", "256",
                    "--temperature-moves", "2",
                    "--temperature-value", "1.0",
                    "--out", str(selfplay_path),
                    "--manifest-out", str(out_dir / "selfplay.manifest.json"),
                ],
                cwd=repo_root, stdout=logf, stderr=subprocess.STDOUT,
                check=False,
            )
    finally:
        serve.terminate()
        try:
            serve.wait(timeout=3)
        except subprocess.TimeoutExpired:
            serve.kill()

    if result.returncode != 0:
        print(f"[selfplay-smoke] sim:mcts-selfplay failed, returncode={result.returncode}; see {log_path}", file=sys.stderr)
        sys.exit(1)

    if not selfplay_path.exists():
        print(f"[selfplay-smoke] no selfplay.jsonl at {selfplay_path}", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []
    with selfplay_path.open("r", encoding="utf8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print("[selfplay-smoke] selfplay.jsonl produced 0 rows", file=sys.stderr)
        sys.exit(1)

    required = {
        "schemaVersion", "kind", "seed", "sideId", "step", "turnNumber",
        "observation", "legalActions", "selectedActionIndex", "visitDistribution",
        "rootPriors", "rootMeanQ", "rootValue", "valueTarget", "result",
    }
    for i, row in enumerate(rows):
        missing = required - set(row.keys())
        if missing:
            print(f"[selfplay-smoke] row {i} missing keys: {missing}", file=sys.stderr)
            sys.exit(1)
        if row["kind"] != "mcts-selfplay":
            print(f"[selfplay-smoke] row {i} kind={row['kind']!r} != 'mcts-selfplay'", file=sys.stderr)
            sys.exit(1)
        if row["valueTarget"] not in (-1, 0, 1):
            print(f"[selfplay-smoke] row {i} valueTarget={row['valueTarget']} ∉ {{-1,0,1}}", file=sys.stderr)
            sys.exit(1)
        dist = row["visitDistribution"]
        if not (0.999 <= sum(dist) <= 1.001):
            print(f"[selfplay-smoke] row {i} visitDistribution sum={sum(dist):.4f} not 1", file=sys.stderr)
            sys.exit(1)
        if len(row["rootMeanQ"]) != len(row["legalActions"]):
            print(
                f"[selfplay-smoke] row {i} rootMeanQ length={len(row['rootMeanQ'])} "
                f"!= legalActions length={len(row['legalActions'])}",
                file=sys.stderr,
            )
            sys.exit(1)
        if len(dist) != len(row["legalActions"]):
            print(f"[selfplay-smoke] row {i} dist length {len(dist)} != legal {len(row['legalActions'])}", file=sys.stderr)
            sys.exit(1)
    print(json.dumps({
        "status": "PASS",
        "games": 2,
        "rows": len(rows),
        "selfplay": str(selfplay_path),
    }, indent=2))


if __name__ == "__main__":
    main()
