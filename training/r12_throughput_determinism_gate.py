"""r12 throughput #2/#4 determinism gate (ship/revert decision).

Proves the throughput #2 (rollout hash-carry) and #4 (keep-alive transport)
changes in backend/src/sim/mcts.ts produce BIT-IDENTICAL self-play / gate
trajectories vs. the pre-change path, at the R110 production rollout-leaf
config (sims 100, CRN 3, rollout leaf, rollout-steps 200, collapse 64).

Method
------
Serve the frozen production checkpoint runs/R13-W6-phase-d/iter-2/policy.onnx
once. Run `npm run sim:eval-gate` --workers 1 over the SAME contiguous seed
range TWICE:
  * FLAG-OFF: UMA_MCTS_HASH_CARRY=0 UMA_MCTS_KEEPALIVE=0  (pre-change path)
  * FLAG-ON : defaults                                    (post-change path)
Fingerprint every (seed, modelSide) game by
(winner, modelWon, turnNumber, modelActions, heuristicFallbacks,
 terminalReason) and assert the FLAG-ON and FLAG-OFF fingerprint maps are
identical key-for-key. Identical => bit-identical trajectories + terminal
outcomes + gate win/loss => ship. Any mismatch => the changed item must be
reverted.

This is a thin wrapper over the existing serve_onnx + sim:eval-gate path;
not an orchestrator, does not train.

Usage
-----
  python training/r12_throughput_determinism_gate.py --games 12
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ITER2 = REPO_ROOT / "runs" / "R13-W6-phase-d" / "iter-2"
SEED_START = 9000


def serve_python() -> str:
    """Prefer the repo training venv (numpy/onnxruntime live there) over
    whatever interpreter launched this wrapper; fall back to sys.executable."""
    venv_py = REPO_ROOT / "training" / ".venv" / "bin" / "python"
    return str(venv_py) if venv_py.exists() else sys.executable


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


def load_prod() -> dict:
    return json.load((ITER2 / "gate.manifest.json").open())["args"]


def run_gate(port: int, out_dir: Path, games: int, label: str, prod: dict, env_extra: dict) -> Path:
    progress = out_dir / f"{label}.progress.jsonl"
    manifest = out_dir / f"{label}.manifest.json"
    log_path = out_dir / f"{label}.log"
    env = dict(os.environ)
    env.update(env_extra)
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
            cwd=REPO_ROOT, stdout=logf, stderr=subprocess.STDOUT, check=True, env=env,
        )
    return progress


def fingerprints(progress_path: Path) -> dict[tuple[str, str], tuple]:
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
            int(obj["heuristicFallbacks"]),
            obj["terminalReason"],
        )
    return by_key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=12)
    args = ap.parse_args()

    onnx = ITER2 / "policy.onnx"
    if not onnx.exists():
        print(f"[r12-gate] ENVIRONMENT GAP: missing {onnx}", file=sys.stderr)
        sys.exit(2)

    out_dir = REPO_ROOT / "runs" / "R12-throughput-determinism-gate"
    out_dir.mkdir(parents=True, exist_ok=True)
    prod = load_prod()
    print(f"[r12-gate] config leaf={prod['mctsLeaf']} sims={prod['mctsSimulations']} "
          f"crn={prod['mctsRolloutCrnSamples']} rolloutSteps={prod['mctsRolloutSteps']} "
          f"collapse={prod['mctsCollapseMaxSteps']} prior={prod['mctsPrior']} "
          f"games={args.games} seedStart={SEED_START} workers=1", flush=True)

    port = free_port()
    serve = subprocess.Popen(
        [serve_python(), str(REPO_ROOT / "training" / "serve_onnx.py"),
         "--model", str(onnx), "--port", str(port), "--default-sampling", "greedy"],
        cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        if not wait_for_health(port):
            print("[r12-gate] serve_onnx never healthy", file=sys.stderr)
            sys.exit(2)
        off = fingerprints(run_gate(
            port, out_dir, args.games, "flag-off", prod,
            {"UMA_MCTS_HASH_CARRY": "0", "UMA_MCTS_KEEPALIVE": "0"}))
        on = fingerprints(run_gate(
            port, out_dir, args.games, "flag-on", prod, {}))
    finally:
        serve.terminate()
        try:
            serve.wait(timeout=3)
        except subprocess.TimeoutExpired:
            serve.kill()

    keys = sorted(set(off) & set(on))
    only_off = sorted(set(off) - set(on))
    only_on = sorted(set(on) - set(off))
    mismatches = [
        {"key": list(k), "flag_off": list(off[k]), "flag_on": list(on[k])}
        for k in keys if off[k] != on[k]
    ]
    identical = not mismatches and not only_off and not only_on and bool(keys)
    result = {
        "status": "PASS" if identical else "FAIL",
        "games": args.games,
        "seed_side_keys": len(keys),
        "key_set_symmetric": not only_off and not only_on,
        "only_flag_off_keys": [list(k) for k in only_off],
        "only_flag_on_keys": [list(k) for k in only_on],
        "mismatches": mismatches,
        "note": "FLAG-ON (throughput #2 hash-carry + #4 keep-alive, defaults) vs "
                "FLAG-OFF (UMA_MCTS_HASH_CARRY=0 UMA_MCTS_KEEPALIVE=0) at R110 "
                "rollout-leaf config; identical => bit-identical trajectories + "
                "gate outcomes => ship.",
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    sys.exit(0 if identical else 1)


if __name__ == "__main__":
    main()
