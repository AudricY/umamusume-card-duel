"""R14.G ORT-threads throughput + determinism smoke.

Validates that --ort-threads=auto (unpinned) restores meaningful
/predict throughput WITHOUT losing the engine-level serial-vs-parallel
bit-exactness that R14.B installed.

Methodology:
  1. With R14.B's AsyncLocalStorage installed, engine-level RNG state
     determines the per-game seed expansion. ORT's per-call FP
     reductions are downstream of the engine RNG: the model receives
     the same observation, must produce the same logits, and a
     different logit ordering would only matter on visit-count
     tiebreaks (rare at >a-few sims).
  2. Run a 60-game value-head-leaf MCTS gate twice over the same
     seeds, once with --ort-threads 1 and once with --ort-threads auto.
     Each run uses --workers 4 so both modes are exercised under load.
  3. Compare wall-clock: assert auto >= 1.5x faster than pinned. The
     sprint plan target is >=2x; we accept >=1.5x as the smoke bar to
     avoid flakiness from background CPU noise.
  4. Assert per-(seed, side) winner+turnNumber matches between the
     two modes. If FP drift is visible at this sim count, the smoke
     will surface it cleanly.

Tiny config: 12 games (6 seeds × 2 sides) × value-head leaf × 32 sims.
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


def run_gate(repo_root: Path, port: int, out_dir: Path, label: str) -> tuple[Path, float]:
    progress = out_dir / f"{label}.progress.jsonl"
    log_path = out_dir / f"{label}.log"
    manifest = out_dir / f"{label}.manifest.json"
    started = time.perf_counter()
    with log_path.open("w") as logf:
        subprocess.run(
            [
                "npm", "--workspace", "backend", "run", "sim:eval-gate", "--",
                "--selection", "mcts",
                "--games", "6",
                "--max-steps", "500",
                "--seed-start", "151515",
                "--model-side", "both",
                "--model-url", f"http://127.0.0.1:{port}",
                "--manifest-out", str(manifest),
                "--progress-out", str(progress),
                "--workers", "4",
                "--mcts-simulations", "32",
                "--mcts-c-puct", "1.5",
                "--mcts-leaf", "value-head",
                "--mcts-collapse-max-steps", "64",
                "--mcts-max-nodes", "512",
                "--min-games", "1",
                "--min-ci-lower", "0",
            ],
            cwd=repo_root, stdout=logf, stderr=subprocess.STDOUT, check=True,
        )
    return progress, time.perf_counter() - started


def load_per_game(progress_path: Path) -> dict[tuple[str, str], dict]:
    by_key: dict[tuple[str, str], dict] = {}
    for line in progress_path.read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        key = (str(obj["seed"]), str(obj["modelSide"]))
        by_key[key] = {"winner": obj["winner"], "turnNumber": obj["turnNumber"]}
    return by_key


def serve(repo_root: Path, onnx: Path, port: int, ort_threads: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, str(repo_root / "training" / "serve_onnx.py"),
         "--model", str(onnx), "--port", str(port),
         "--default-sampling", "greedy", "--ort-threads", ort_threads],
        cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ckpt_candidates = [
        repo_root / "runs" / "R13-W6-phase-d" / "iter-1" / "checkpoint.pt",
        repo_root / "runs" / "R4-value-head-retrain" / "checkpoint.pt",
    ]
    ckpt = next((c for c in ckpt_candidates if c.exists()), None)
    if ckpt is None:
        print("[r14-ort-throughput] no checkpoint found", file=sys.stderr)
        sys.exit(2)

    onnx = export_onnx_if_needed(repo_root, ckpt)
    out_dir = repo_root / "runs" / "R14-ort-throughput-smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for label, ort_threads in [("pinned", "1"), ("auto", "auto")]:
        port = free_port()
        proc = serve(repo_root, onnx, port, ort_threads)
        try:
            if not wait_for_health(port):
                print(f"[r14-ort-throughput] serve_onnx ({label}) never healthy", file=sys.stderr)
                sys.exit(2)
            progress, elapsed = run_gate(repo_root, port, out_dir, label)
            results[label] = {"elapsed": elapsed, "per_game": load_per_game(progress)}
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

    pinned = results["pinned"]
    auto = results["auto"]
    speedup = pinned["elapsed"] / max(0.001, auto["elapsed"])

    expected_count = 12
    if len(pinned["per_game"]) != expected_count or len(auto["per_game"]) != expected_count:
        print(f"[r14-ort-throughput] expected {expected_count} games per run; got pinned={len(pinned['per_game'])} auto={len(auto['per_game'])}", file=sys.stderr)
        sys.exit(1)

    mismatches: list[dict] = []
    for key in sorted(pinned["per_game"]):
        p = pinned["per_game"].get(key)
        a = auto["per_game"].get(key)
        if not p or not a:
            mismatches.append({"key": list(key), "pinned": p, "auto": a, "reason": "missing"})
            continue
        if p["winner"] != a["winner"] or p["turnNumber"] != a["turnNumber"]:
            mismatches.append({"key": list(key), "pinned": p, "auto": a})

    speedup_ok = speedup >= 1.5
    determinism_ok = len(mismatches) == 0
    status = "PASS" if (speedup_ok and determinism_ok) else "FAIL"

    summary = {
        "status": status,
        "elapsed_pinned_sec": round(pinned["elapsed"], 2),
        "elapsed_auto_sec": round(auto["elapsed"], 2),
        "speedup": round(speedup, 2),
        "speedup_target": 1.5,
        "speedup_ok": speedup_ok,
        "determinism_ok": determinism_ok,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:6],
        "games": expected_count,
        "note": "Validates G (unpin ORT) keeps engine-level determinism and yields >=1.5x throughput.",
    }
    print(json.dumps(summary, indent=2))
    if status != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
