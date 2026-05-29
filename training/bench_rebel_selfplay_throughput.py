#!/usr/bin/env python3
"""Measured throughput sweep for sim-rebel-selfplay (ReBeL self-play).

This is a measurement runner, NOT an orchestration framework. It invokes the
prebuilt release binary across a config grid, samples GPU utilisation live, and
reports games/s + mean batch fill + mean/max GPU util per config so the
hidden=256/depth=4 throughput re-baseline can be done empirically.

Why it exists: prior CUDA dispatch verdicts in
docs/ai-research/scoping/rebel-throughput-optimization-slice.md were measured at
the OLD model size on a single-worker smoke (386 rows). They do not transfer to
the production 24-worker hidden=256/depth=4 shape. This re-baselines them.

SAFETY: by default it refuses to launch `--device cuda` configs while another
sim-rebel-selfplay / sim-eval-gate is running, so it never perturbs a live
training run sharing the GPU. Pass --force to override (CPU configs are always
allowed; they only use idle cores).

Example (run AFTER the live run frees the GPU):
  python training/bench_rebel_selfplay_throughput.py \
      --onnx-path runs/R18-rebel-kl-relaxed-20260528/loop/pool/iter-000/policy.onnx \
      --seeds 24 --preset rebaseline --out /tmp/rebel-thru-sweep.csv
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BIN = REPO / "engine-rs/target/release/sim-rebel-selfplay"
FILL_RE = re.compile(r"batched_inference batches=(\d+) requests=(\d+) mean_fill=([\d.]+)")


def _gpu_busy_with_sim() -> bool:
    """True if a sim-rebel-selfplay or sim-eval-gate (other than us) is running."""
    try:
        out = subprocess.run(
            ["pgrep", "-af", "sim-rebel-selfplay|sim-eval-gate"],
            capture_output=True, text=True,
        ).stdout
    except FileNotFoundError:
        return False
    mypid = str(os.getpid())
    lines = [l for l in out.splitlines()
             if l.strip() and "bench_rebel_selfplay" not in l and not l.startswith(mypid + " ")]
    return len(lines) > 0


class GpuSampler(threading.Thread):
    """Polls nvidia-smi util/power while a config runs."""

    def __init__(self, period_s: float = 0.2):
        super().__init__(daemon=True)
        self.period_s = period_s
        self._stop = threading.Event()
        self.utils: list[float] = []
        self.powers: list[float] = []

    def run(self) -> None:
        if not shutil.which("nvidia-smi"):
            return
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2,
                ).stdout.strip()
                u, p = (x.strip() for x in out.splitlines()[0].split(","))
                self.utils.append(float(u))
                self.powers.append(float(p))
            except Exception:
                pass
            self._stop.wait(self.period_s)

    def stop(self) -> dict:
        self._stop.set()
        self.join(timeout=2)
        def stat(xs):
            return (round(sum(xs) / len(xs), 1), round(max(xs), 1)) if xs else (None, None)
        umean, umax = stat(self.utils)
        pmean, pmax = stat(self.powers)
        return {"gpu_util_mean": umean, "gpu_util_max": umax,
                "gpu_power_mean": pmean, "gpu_power_max": pmax}


def run_config(cfg: dict, onnx: str | None, seeds: int, seed_base: int,
               max_steps: int, model_side: str, deck_sampling: str,
               timeout_s: int) -> dict:
    out = Path(f"/tmp/rebel-thru-{cfg['tag']}.jsonl")
    man = Path(f"/tmp/rebel-thru-{cfg['tag']}.manifest.json")
    cmd = [
        str(BIN),
        "--seeds", str(seeds), "--seed-base", str(seed_base),
        "--particles", str(cfg["particles"]),
        "--iterations", str(cfg["iterations"]),
        "--rollout-steps", str(cfg["rollout_steps"]),
        "--max-steps", str(max_steps),
        "--model-side", model_side,
        "--deck-sampling", deck_sampling,
        "--workers", str(cfg["workers"]),
        "--device", cfg["device"],
        "--neural-leaf-weight", str(cfg["neural_leaf_weight"]),
        "--neural-policy-weight", str(cfg["neural_policy_weight"]),
        "--neural-value-weight", str(cfg["neural_value_weight"]),
        "--inference-batch-size", str(cfg["inference_batch_size"]),
        "--inference-max-wait-us", str(cfg["inference_max_wait_us"]),
        "--out", str(out), "--manifest-out", str(man),
    ]
    if onnx:
        cmd += ["--onnx-path", onnx, "--cuda-device-id", "0"]
    # Regime A configs blend in the determinized rollout leaf (neural_leaf_weight
    # < 1.0). After the correctness fix the binary refuses that without an
    # explicit ablation opt-in, so this benchmark — which intentionally MEASURES
    # the rollout ablation's throughput cost — must pass the flag.
    if float(cfg["neural_leaf_weight"]) < 1.0:
        cmd.append("--allow-rollout-leaf")

    sampler = GpuSampler()
    sampler.start()
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    wall = time.monotonic() - t0
    gpu = sampler.stop()

    fill = None
    m = FILL_RE.search(proc.stdout + proc.stderr)
    if m:
        fill = float(m.group(3))
    summary = {}
    if man.exists():
        summary = json.loads(man.read_text()).get("summary", {})
    return {
        **{k: cfg[k] for k in ("tag", "device", "rollout_steps", "neural_leaf_weight",
                               "inference_batch_size", "inference_max_wait_us", "workers")},
        "games": summary.get("games"),
        "games_per_s": round(summary.get("gamesPerSec", 0.0), 3) if summary else None,
        "elapsed_s": round(summary.get("elapsedSecs", wall), 1),
        "mean_fill": fill,
        **gpu,
        "rc": proc.returncode,
    }


def preset_grid(name: str, base: dict) -> list[dict]:
    """One-factor-at-a-time sweeps from the live baseline."""
    grids: list[dict] = []

    def add(tag, **over):
        c = dict(base); c.update(over); c["tag"] = tag; grids.append(c)

    if name == "rebaseline":
        # Regime A: rollout-blended (leaf<1) -> CPU-bound, win = fewer rollout steps + more workers
        for rs in (96, 48, 32, 16):
            add(f"A-rs{rs}", rollout_steps=rs, neural_leaf_weight=0.8)
        for w in (24, 30):
            add(f"A-w{w}", workers=w, neural_leaf_weight=0.8, rollout_steps=96)
        add("A-wait5k", neural_leaf_weight=0.8, rollout_steps=96, inference_max_wait_us=5000)
        # Regime B: neural-only leaf (leaf=1.0, rollouts OFF) -> GPU-bound, win = dispatch tuning
        for bs in (1, 128, 512):
            add(f"B-bs{bs}", neural_leaf_weight=1.0, rollout_steps=0,
                inference_batch_size=bs, inference_max_wait_us=2000)
        add("B-bs512-wait40k", neural_leaf_weight=1.0, rollout_steps=0,
            inference_batch_size=512, inference_max_wait_us=40000)
    elif name == "dispatch":
        # Production regime after the correctness fix: leaf=1.0 (rollouts OFF,
        # GPU-bound). The remaining throughput levers are the GPU dispatch
        # (inference-batch-size, max-wait) and worker count. All configs here are
        # the SOUND production leaf — no --allow-rollout-leaf needed.
        for bs in (1, 64, 256, 512):
            add(f"bs{bs}", neural_leaf_weight=1.0, rollout_steps=0,
                inference_batch_size=bs, inference_max_wait_us=10000)
        add("bs256-wait2k", neural_leaf_weight=1.0, rollout_steps=0,
            inference_batch_size=256, inference_max_wait_us=2000)
        add("bs512-wait40k", neural_leaf_weight=1.0, rollout_steps=0,
            inference_batch_size=512, inference_max_wait_us=40000)
        for w in (24, 30):
            add(f"w{w}", neural_leaf_weight=1.0, rollout_steps=0, workers=w,
                inference_batch_size=256, inference_max_wait_us=10000)
    elif name == "rollout-only":
        for rs in (96, 48, 32, 16, 8):
            add(f"rs{rs}", rollout_steps=rs, neural_leaf_weight=0.8)
    elif name == "live":
        add("live-baseline")
    else:
        raise SystemExit(f"unknown preset {name}")
    return grids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-path", default=None)
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--seed-base", type=int, default=950000)
    ap.add_argument("--max-steps", type=int, default=220)
    ap.add_argument("--model-side", default="both")
    ap.add_argument("--deck-sampling", default="uniform")
    ap.add_argument("--preset", default="rebaseline",
                    choices=["rebaseline", "dispatch", "rollout-only", "live"])
    ap.add_argument("--timeout-s", type=int, default=1800)
    ap.add_argument("--out", default="/tmp/rebel-thru-sweep.csv")
    ap.add_argument("--force", action="store_true",
                    help="run CUDA configs even if another sim process is active")
    args = ap.parse_args()

    if not BIN.exists():
        raise SystemExit(f"missing binary {BIN} (build engine-rs --release first)")

    base = {
        "particles": 32, "iterations": 32, "rollout_steps": 96, "workers": 24,
        "device": "cuda", "neural_leaf_weight": 0.8,
        "neural_policy_weight": 0.35, "neural_value_weight": 0.35,
        "inference_batch_size": 512, "inference_max_wait_us": 40000,
        "tag": "base",
    }
    grid = preset_grid(args.preset, base)

    needs_gpu = any(c["device"] == "cuda" for c in grid)
    if needs_gpu and not args.force and _gpu_busy_with_sim():
        raise SystemExit(
            "REFUSING: another sim-rebel-selfplay/sim-eval-gate is running and this "
            "sweep has CUDA configs that would perturb it. Re-run when the live run "
            "frees the GPU, or pass --force.")

    rows: list[dict] = []
    cols = ["tag", "device", "rollout_steps", "neural_leaf_weight", "inference_batch_size",
            "inference_max_wait_us", "workers", "games", "games_per_s", "elapsed_s",
            "mean_fill", "gpu_util_mean", "gpu_util_max", "gpu_power_mean", "rc"]
    print(",".join(cols))
    for cfg in grid:
        try:
            r = run_config(cfg, args.onnx_path, args.seeds, args.seed_base,
                           args.max_steps, args.model_side, args.deck_sampling, args.timeout_s)
        except subprocess.TimeoutExpired:
            r = {"tag": cfg["tag"], "rc": "TIMEOUT"}
        rows.append(r)
        print(",".join(str(r.get(c, "")) for c in cols), flush=True)

    Path(args.out).write_text(
        ",".join(cols) + "\n" + "\n".join(
            ",".join(str(r.get(c, "")) for c in cols) for r in rows) + "\n")
    print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
