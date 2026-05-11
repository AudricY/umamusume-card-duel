"""R14.A out-of-distribution gate for W6 iter-1 + value-head leaf.

W6 iter-1's reported Wilson lower 0.452 at value-head leaf (R13.W6) is
suspiciously close to its rollout-leaf number on the SAME selfplay-
generated distribution. The value head saw the same leaf scores it'll
be graded on at inference, raising the OOD-overfit suspicion: is the
strength claim distributionally honest, or rule-bot-specific?

Two gates, both at 100 sims, value-head leaf, --workers 4:
  Gate 1: fresh seeds 800000+ vs rule-bot (n=100). Tests strength on
          seeds not in the W6 selfplay corpus's seed range.
  Gate 2: same iter-1 ckpt, value-head-leaf MCTS, vs R4 checkpoint
          running MCTS at the same sim count (R4 as MCTS opponent).
          MCTS-vs-MCTS removes the rule-bot evaluator.

Exit: Wilson lower >= 0.40 on BOTH gates. If Gate 1 holds but Gate 2
fails by >5pp, iter-1 is rule-bot-specific and the deployment claim
should be narrowed.
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


def serve(repo_root: Path, onnx: Path, port: int) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, str(repo_root / "training" / "serve_onnx.py"),
         "--model", str(onnx), "--port", str(port), "--default-sampling", "greedy"],
        cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def run_gate_rule_bot(repo_root: Path, port: int, out_dir: Path, args: argparse.Namespace) -> dict:
    label = "gate1-fresh-seeds"
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
                "--seed-start", str(args.seed_start_fresh),
                "--model-side", "both",
                "--model-url", f"http://127.0.0.1:{port}",
                "--manifest-out", str(manifest),
                "--progress-out", str(progress),
                "--workers", str(args.workers),
                "--mcts-simulations", "100",
                "--mcts-c-puct", "1.5",
                "--mcts-leaf", "value-head",
                "--mcts-prior", "policy",
                "--mcts-collapse-max-steps", "64",
                "--mcts-max-nodes", "5000",
                "--min-games", "1",
                "--min-ci-lower", "0",
            ],
            cwd=repo_root, stdout=logf, stderr=subprocess.STDOUT, check=True,
        )
    elapsed = time.perf_counter() - started
    summary = json.loads(manifest.read_text(encoding="utf8")).get("summary", {})
    return {
        "label": label,
        "elapsed_sec": round(elapsed, 2),
        "wilson_lower": float(summary.get("wilson95", {}).get("lower", 0)),
        "wilson_upper": float(summary.get("wilson95", {}).get("upper", 0)),
        "win_rate": float(summary.get("modelWinRate", 0)),
        "games": int(summary.get("games", 0)),
        "by_side": summary.get("byModelSide", {}),
    }


def run_gate_mcts_vs_mcts(
    repo_root: Path,
    iter1_port: int,
    r4_port: int,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict:
    label = "gate2-mcts-vs-r4"
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
                "--seed-start", str(args.seed_start_ood),
                "--model-side", "both",
                "--model-url", f"http://127.0.0.1:{iter1_port}",
                "--manifest-out", str(manifest),
                "--progress-out", str(progress),
                "--workers", str(args.workers),
                "--mcts-simulations", "100",
                "--mcts-c-puct", "1.5",
                "--mcts-leaf", "value-head",
                "--mcts-prior", "policy",
                "--mcts-collapse-max-steps", "64",
                "--mcts-max-nodes", "5000",
                "--opponent-selection", "mcts",
                "--opponent-model-url", f"http://127.0.0.1:{r4_port}",
                "--opponent-mcts-simulations", "100",
                "--opponent-mcts-leaf", "value-head",
                "--opponent-mcts-prior", "policy",
                "--min-games", "1",
                "--min-ci-lower", "0",
            ],
            cwd=repo_root, stdout=logf, stderr=subprocess.STDOUT, check=True,
        )
    elapsed = time.perf_counter() - started
    summary = json.loads(manifest.read_text(encoding="utf8")).get("summary", {})
    return {
        "label": label,
        "elapsed_sec": round(elapsed, 2),
        "wilson_lower": float(summary.get("wilson95", {}).get("lower", 0)),
        "wilson_upper": float(summary.get("wilson95", {}).get("upper", 0)),
        "win_rate": float(summary.get("modelWinRate", 0)),
        "games": int(summary.get("games", 0)),
        "by_side": summary.get("byModelSide", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iter1-ckpt", default=None)
    parser.add_argument("--r4-ckpt", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--games", type=int, default=50, help="Games per side; n=2*games.")
    parser.add_argument("--seed-start-fresh", type=int, default=800000)
    parser.add_argument("--seed-start-ood", type=int, default=850000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--target-wilson-lower", type=float, default=0.40)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    iter1 = Path(args.iter1_ckpt) if args.iter1_ckpt else (repo_root / "runs" / "R13-W6-phase-d" / "iter-1" / "checkpoint.pt")
    r4 = Path(args.r4_ckpt) if args.r4_ckpt else (repo_root / "runs" / "R4-value-head-retrain" / "checkpoint.pt")
    for label, ckpt in [("iter1", iter1), ("r4", r4)]:
        if not ckpt.exists():
            print(f"[r14-ood-gate] {label} checkpoint not found: {ckpt}", file=sys.stderr)
            sys.exit(2)

    out_dir = Path(args.out_dir) if args.out_dir else (repo_root / "runs" / "R14-ood-gate")
    out_dir.mkdir(parents=True, exist_ok=True)

    iter1_onnx = export_onnx_if_needed(repo_root, iter1)
    r4_onnx = export_onnx_if_needed(repo_root, r4)

    iter1_port = free_port()
    r4_port = free_port()
    iter1_proc = serve(repo_root, iter1_onnx, iter1_port)
    r4_proc = serve(repo_root, r4_onnx, r4_port)
    try:
        if not wait_for_health(iter1_port):
            print("[r14-ood-gate] iter1 serve_onnx never healthy", file=sys.stderr)
            sys.exit(2)
        if not wait_for_health(r4_port):
            print("[r14-ood-gate] r4 serve_onnx never healthy", file=sys.stderr)
            sys.exit(2)
        gate1 = run_gate_rule_bot(repo_root, iter1_port, out_dir, args)
        gate2 = run_gate_mcts_vs_mcts(repo_root, iter1_port, r4_port, out_dir, args)
    finally:
        for proc in (iter1_proc, r4_proc):
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

    gate1_pass = gate1["wilson_lower"] >= args.target_wilson_lower
    gate2_pass = gate2["wilson_lower"] >= args.target_wilson_lower
    gap = gate1["wilson_lower"] - gate2["wilson_lower"]

    interpretation = "both gates pass — strength claim is OOD-robust"
    if gate1_pass and not gate2_pass:
        if gap > 0.05:
            interpretation = "gate 1 passes but gate 2 fails by >5pp — iter-1 is rule-bot-specific"
        else:
            interpretation = "gate 1 passes, gate 2 fails marginally — close to boundary"
    elif not gate1_pass and gate2_pass:
        interpretation = "fresh-seed gate fails — strength claim may be seed-specific"
    elif not gate1_pass and not gate2_pass:
        interpretation = "both gates fail — strength claim does not generalize"

    summary = {
        "status": "PASS" if (gate1_pass and gate2_pass) else "FAIL",
        "interpretation": interpretation,
        "target_wilson_lower": args.target_wilson_lower,
        "gap_gate1_minus_gate2": round(gap, 4),
        "gate1": gate1,
        "gate2": gate2,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
