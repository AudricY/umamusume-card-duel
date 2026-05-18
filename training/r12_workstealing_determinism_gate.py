"""r12 work-stealing dispatch determinism gate (ship/revert decision).

Proves the throughput work-stealing dispatch in backend/src/sim/mctsSelfPlay.ts
and backend/src/sim/evalGate.ts produces BIT-IDENTICAL self-play rows AND
identical gate outcomes vs. the pre-change static contiguous chunking, at a
small game count with workers > 1 (the only regime where the dispatch path
differs).

Method
------
Serve the frozen production checkpoint runs/R13-W6-phase-d/iter-2/policy.onnx
once. For BOTH selfplay and gate, run the same seed range TWICE with
workers > 1:
  * FLAG-OFF: UMA_MCTS_WORK_STEALING=0  (static contiguous chunking)
  * FLAG-ON : defaults                  (work-stealing dispatch)
Selfplay: the canonical concatenated selfplay.jsonl must be byte-identical
(both paths emit seed-ordered output). Gate: fingerprint every
(seed, modelSide) game by (winner, modelWon, turnNumber, modelActions,
heuristicFallbacks, terminalReason) and assert FLAG-ON == FLAG-OFF
key-for-key. Identical => trajectory-neutral => ship ON. Any mismatch =>
the dispatch is order-dependent and must be reverted.

Thin wrapper over serve_onnx + the sim runners; not an orchestrator, no train.

Usage
-----
  python training/r12_workstealing_determinism_gate.py --games 10 --workers 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ITER2 = REPO_ROOT / "runs" / "R13-W6-phase-d" / "iter-2"
SELFPLAY_SEED_START = 12000
GATE_SEED_START = 9000


def serve_python() -> str:
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


def run_selfplay(port: int, out_dir: Path, games: int, workers: int,
                  label: str, prod: dict, env_extra: dict) -> Path:
    out_path = out_dir / f"selfplay-{label}.jsonl"
    log_path = out_dir / f"selfplay-{label}.log"
    env = dict(os.environ)
    env.update(env_extra)
    with log_path.open("w") as logf:
        subprocess.run(
            [
                "npm", "--workspace", "backend", "run", "sim:mcts-selfplay", "--",
                "--games", str(games),
                "--seed-start", str(SELFPLAY_SEED_START),
                "--max-steps", str(prod["maxSteps"]),
                "--model-url", f"http://127.0.0.1:{port}",
                "--out", str(out_path),
                "--workers", str(workers),
                "--mcts-simulations", str(prod["mctsSimulations"]),
                "--mcts-c-puct", str(prod["mctsCPuct"]),
                "--mcts-leaf", str(prod["mctsLeaf"]),
                "--mcts-rollout-crn-samples", str(prod["mctsRolloutCrnSamples"]),
                "--mcts-rollout-steps", str(prod["mctsRolloutSteps"]),
                "--mcts-collapse-max-steps", str(prod["mctsCollapseMaxSteps"]),
                "--mcts-max-nodes", str(prod["mctsMaxNodes"]),
                "--mcts-prior", str(prod["mctsPrior"]),
            ],
            cwd=REPO_ROOT, stdout=logf, stderr=subprocess.STDOUT, check=True, env=env,
        )
    return out_path


def run_gate(port: int, out_dir: Path, games: int, workers: int,
             label: str, prod: dict, env_extra: dict) -> Path:
    progress = out_dir / f"gate-{label}.progress.jsonl"
    log_path = out_dir / f"gate-{label}.log"
    env = dict(os.environ)
    env.update(env_extra)
    with log_path.open("w") as logf:
        subprocess.run(
            [
                "npm", "--workspace", "backend", "run", "sim:eval-gate", "--",
                "--selection", "mcts",
                "--games", str(games),
                "--max-steps", str(prod["maxSteps"]),
                "--seed-start", str(GATE_SEED_START),
                "--model-side", "both",
                "--model-url", f"http://127.0.0.1:{port}",
                "--progress-out", str(progress),
                "--workers", str(workers),
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


def selfplay_digest(path: Path) -> tuple[str, int]:
    """Canonical selfplay output is seed-ordered on both paths, so the raw
    file bytes must match. Hash the bytes; also return the row count."""
    data = path.read_bytes()
    rows = sum(1 for ln in data.decode("utf8").splitlines() if ln.strip())
    return hashlib.sha256(data).hexdigest(), rows


def gate_fingerprints(progress_path: Path) -> dict[tuple[str, str], tuple]:
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
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    onnx = ITER2 / "policy.onnx"
    if not onnx.exists():
        print(f"[ws-gate] ENVIRONMENT GAP: missing {onnx}", file=sys.stderr)
        sys.exit(2)

    out_dir = REPO_ROOT / "runs" / "R12-workstealing-determinism-gate"
    out_dir.mkdir(parents=True, exist_ok=True)
    prod = load_prod()
    print(f"[ws-gate] leaf={prod['mctsLeaf']} sims={prod['mctsSimulations']} "
          f"games={args.games} workers={args.workers} "
          f"(selfplay seedStart={SELFPLAY_SEED_START}, gate seedStart={GATE_SEED_START})",
          flush=True)

    port = free_port()
    serve = subprocess.Popen(
        [serve_python(), str(REPO_ROOT / "training" / "serve_onnx.py"),
         "--model", str(onnx), "--port", str(port), "--default-sampling", "greedy"],
        cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        if not wait_for_health(port):
            print("[ws-gate] serve_onnx never healthy", file=sys.stderr)
            sys.exit(2)
        sp_off = selfplay_digest(run_selfplay(
            port, out_dir, args.games, args.workers, "flag-off", prod,
            {"UMA_MCTS_WORK_STEALING": "0"}))
        sp_on = selfplay_digest(run_selfplay(
            port, out_dir, args.games, args.workers, "flag-on", prod, {}))
        g_off = gate_fingerprints(run_gate(
            port, out_dir, args.games, args.workers, "flag-off", prod,
            {"UMA_MCTS_WORK_STEALING": "0"}))
        g_on = gate_fingerprints(run_gate(
            port, out_dir, args.games, args.workers, "flag-on", prod, {}))
    finally:
        serve.terminate()
        try:
            serve.wait(timeout=3)
        except subprocess.TimeoutExpired:
            serve.kill()

    selfplay_identical = sp_off == sp_on and sp_off[1] > 0
    g_keys = sorted(set(g_off) & set(g_on))
    g_only_off = sorted(set(g_off) - set(g_on))
    g_only_on = sorted(set(g_on) - set(g_off))
    g_mismatches = [
        {"key": list(k), "flag_off": list(g_off[k]), "flag_on": list(g_on[k])}
        for k in g_keys if g_off[k] != g_on[k]
    ]
    gate_identical = not g_mismatches and not g_only_off and not g_only_on and bool(g_keys)
    identical = selfplay_identical and gate_identical

    result = {
        "status": "PASS" if identical else "FAIL",
        "games": args.games,
        "workers": args.workers,
        "selfplay": {
            "identical": selfplay_identical,
            "flag_off_sha256": sp_off[0],
            "flag_on_sha256": sp_on[0],
            "flag_off_rows": sp_off[1],
            "flag_on_rows": sp_on[1],
        },
        "gate": {
            "identical": gate_identical,
            "seed_side_keys": len(g_keys),
            "key_set_symmetric": not g_only_off and not g_only_on,
            "only_flag_off_keys": [list(k) for k in g_only_off],
            "only_flag_on_keys": [list(k) for k in g_only_on],
            "mismatches": g_mismatches,
        },
        "note": "FLAG-ON (work-stealing, default) vs FLAG-OFF "
                "(UMA_MCTS_WORK_STEALING=0, static chunking) at workers>1; "
                "identical selfplay bytes + gate fingerprints => "
                "trajectory-neutral => ship ON.",
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    sys.exit(0 if identical else 1)


if __name__ == "__main__":
    main()
