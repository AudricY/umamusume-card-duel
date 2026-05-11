"""R12 phase D: mini-AlphaZero iteration orchestrator.

Each iteration:
  1. Stand up `serve_onnx` for the currently promoted checkpoint.
  2. Run `npm run sim:mcts-selfplay` → selfplay.jsonl rows (visit-count
     soft targets + outcome z-values).
  3. Run `train_bc --data-mode mcts-distill --init-from <promoted>`,
     producing a new checkpoint distilled from the MCTS visit distribution.
  4. Export the new checkpoint to ONNX, stand up a serve_onnx for it.
  5. Run `sim:eval-gate --selection mcts --mcts-simulations N --games G`.
  6. Promote with the Wilson-band tolerance rule (mirrors
     `dagger_orchestrator.decide_promotion`).

Halt-after-2 consecutive promotion failures, identical to DAgger.

Output layout under --out-dir:
  iter-{i}/selfplay.jsonl
  iter-{i}/selfplay.manifest.json
  iter-{i}/checkpoint.pt
  iter-{i}/policy.onnx (+ .meta.json)
  iter-{i}/gate.manifest.json
  orchestrator-state.json
  events.jsonl

Reuses DAgger's `serve_onnx_context`, `decide_promotion`,
`wilson_lower_bound`, and `export_checkpoint_to_onnx` so the surface area
that needs maintenance is small.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from dagger_orchestrator import (
    decide_promotion as dagger_decide_promotion,
    export_checkpoint_to_onnx,
    serve_onnx_context,
    wilson_band_tolerance,
    wilson_lower_bound,
)


@dataclass
class R12State:
    promoted_checkpoint: Path | None = None
    promoted_wilson_lower: float | None = None
    iterations: list[dict[str, Any]] = field(default_factory=list)
    consecutive_failures: int = 0
    halted: bool = False
    halt_reason: str | None = None


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root or Path(__file__).resolve().parents[1])
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "events.jsonl"

    state = R12State()
    if args.resume_state:
        state = load_state(Path(args.resume_state))
    if state.promoted_checkpoint is None and args.init_checkpoint:
        state.promoted_checkpoint = Path(args.init_checkpoint).resolve()

    emit_event(events_path, {
        "stage": "r12-orchestrator", "event_type": "run_started",
        "iterations": args.iterations,
        "selfplay_games": args.selfplay_games,
        "mcts_simulations": args.mcts_simulations,
        "eval_games": args.eval_games,
        "init_checkpoint": str(state.promoted_checkpoint) if state.promoted_checkpoint else None,
        "ts": time.time(),
    })

    if state.promoted_checkpoint is None:
        raise SystemExit("must pass --init-checkpoint or --resume-state with a promoted_checkpoint set")

    last_iter = max((entry["iteration"] for entry in state.iterations), default=-1)
    for iteration in range(last_iter + 1, args.iterations):
        if state.halted:
            emit_event(events_path, {"stage": "r12-orchestrator", "event_type": "halted_before_iteration",
                                     "iteration": iteration, "halt_reason": state.halt_reason, "ts": time.time()})
            break

        try:
            record = run_iteration(repo_root, out_dir, iteration, args, state, events_path)
        except Exception as exc:
            emit_event(events_path, {"stage": "r12-orchestrator", "event_type": "iteration_error",
                                     "iteration": iteration, "error": str(exc), "ts": time.time()})
            raise

        state.iterations.append(record)
        save_state(out_dir / "orchestrator-state.json", state)
        if record["promote"]:
            state.promoted_checkpoint = Path(record["checkpoint"]).resolve()
            state.promoted_wilson_lower = float(record["wilson_lower"])
            state.consecutive_failures = 0
        else:
            state.consecutive_failures += 1
            if state.consecutive_failures >= 2:
                state.halted = True
                state.halt_reason = "halt-after-2 consecutive promotion failures"
                emit_event(events_path, {"stage": "r12-orchestrator", "event_type": "halted",
                                         "iteration": iteration, "halt_reason": state.halt_reason,
                                         "ts": time.time()})
        save_state(out_dir / "orchestrator-state.json", state)

    emit_event(events_path, {"stage": "r12-orchestrator", "event_type": "run_completed",
                             "halted": state.halted, "halt_reason": state.halt_reason,
                             "promoted_wilson_lower": state.promoted_wilson_lower,
                             "ts": time.time()})
    print(json.dumps({
        "status": "HALTED" if state.halted else "COMPLETED",
        "promoted_checkpoint": str(state.promoted_checkpoint) if state.promoted_checkpoint else None,
        "promoted_wilson_lower": state.promoted_wilson_lower,
        "iterations_run": len(state.iterations),
    }, indent=2))


def run_iteration(
    repo_root: Path,
    out_dir: Path,
    iteration: int,
    args: argparse.Namespace,
    state: R12State,
    events_path: Path,
) -> dict[str, Any]:
    iter_dir = out_dir / f"iter-{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    selfplay_path = iter_dir / "selfplay.jsonl"
    selfplay_manifest = iter_dir / "selfplay.manifest.json"
    ckpt_dir = iter_dir
    new_ckpt = iter_dir / "checkpoint.pt"
    new_onnx = iter_dir / "policy.onnx"
    gate_manifest = iter_dir / "gate.manifest.json"

    emit_event(events_path, {"stage": "r12-orchestrator", "event_type": "iteration_started",
                             "iteration": iteration,
                             "promoted_checkpoint": str(state.promoted_checkpoint),
                             "ts": time.time()})

    # --- step 1: self-play under the currently promoted checkpoint ---
    promoted_onnx = ensure_onnx(repo_root, state.promoted_checkpoint)
    emit_event(events_path, {"stage": "selfplay", "event_type": "started",
                             "iteration": iteration, "games": args.selfplay_games,
                             "mcts_simulations": args.mcts_simulations, "ts": time.time()})
    t0 = time.time()
    with serve_onnx_context(repo_root, promoted_onnx, args) as model_url:
        run_selfplay(repo_root, iter_dir, model_url, args, selfplay_path, selfplay_manifest, iteration)
    selfplay_elapsed = time.time() - t0
    n_rows = count_lines(selfplay_path)
    emit_event(events_path, {"stage": "selfplay", "event_type": "completed",
                             "iteration": iteration, "rows": n_rows,
                             "elapsed_sec": selfplay_elapsed, "ts": time.time()})

    # --- step 2: distillation training, warm-started from promoted ckpt ---
    emit_event(events_path, {"stage": "distill", "event_type": "started",
                             "iteration": iteration, "epochs": args.epochs, "ts": time.time()})
    t0 = time.time()
    run_distill(repo_root, ckpt_dir, selfplay_path, state.promoted_checkpoint, iteration, args, events_path)
    distill_elapsed = time.time() - t0
    if not new_ckpt.exists():
        raise RuntimeError(f"distill did not produce checkpoint at {new_ckpt}")
    emit_event(events_path, {"stage": "distill", "event_type": "completed",
                             "iteration": iteration, "elapsed_sec": distill_elapsed,
                             "checkpoint": str(new_ckpt), "ts": time.time()})

    # --- step 3: gate the new checkpoint with MCTS at eval ---
    export_checkpoint_to_onnx(repo_root, new_ckpt, new_onnx)
    emit_event(events_path, {"stage": "mcts-gate", "event_type": "started",
                             "iteration": iteration, "games": args.eval_games,
                             "mcts_simulations": args.mcts_simulations, "ts": time.time()})
    t0 = time.time()
    with serve_onnx_context(repo_root, new_onnx, args) as model_url:
        gate_rc = run_gate(repo_root, iter_dir, model_url, args, gate_manifest, iteration)
    gate_elapsed = time.time() - t0
    summary = load_summary(gate_manifest)
    wilson_lower = float(summary.get("wilson95", {}).get("lower", 0))
    win_rate = float(summary.get("modelWinRate", 0))
    games = int(summary.get("games", 0))
    fallbacks = int(summary.get("heuristicFallbacks", 0))
    emit_event(events_path, {"stage": "mcts-gate", "event_type": "completed",
                             "iteration": iteration,
                             "wilson_lower": wilson_lower, "win_rate": win_rate, "games": games,
                             "heuristic_fallbacks": fallbacks, "elapsed_sec": gate_elapsed,
                             "ts": time.time()})

    decision = dagger_decide_promotion(
        _adapt_for_dagger_promote(state),
        gate_returncode=gate_rc,
        wilson_lower=wilson_lower,
        args=_promotion_args(args, state),
        eval_n=games,
        matchup_violations=None,
    )
    record = {
        "iteration": iteration,
        "checkpoint": str(new_ckpt),
        "wilson_lower": wilson_lower,
        "win_rate": win_rate,
        "games": games,
        "heuristic_fallbacks": fallbacks,
        "selfplay_rows": n_rows,
        "promote": bool(decision.get("promote")),
        "reason": str(decision.get("reason")),
        "selfplay_elapsed_sec": selfplay_elapsed,
        "distill_elapsed_sec": distill_elapsed,
        "gate_elapsed_sec": gate_elapsed,
    }
    emit_event(events_path, {"stage": "r12-orchestrator", "event_type": "iteration_completed",
                             **record, "ts": time.time()})
    return record


def run_selfplay(
    repo_root: Path,
    iter_dir: Path,
    model_url: str,
    args: argparse.Namespace,
    out_path: Path,
    manifest_out: Path,
    iteration: int,
) -> None:
    log_path = iter_dir / "selfplay.log"
    with log_path.open("w") as logf:
        subprocess.run(
            [
                "npm", "--workspace", "backend", "run", "sim:mcts-selfplay", "--",
                "--model-url", model_url,
                "--games", str(args.selfplay_games),
                "--seed-start", str(args.selfplay_seed_start + iteration * args.selfplay_games),
                "--mcts-simulations", str(args.mcts_simulations),
                "--mcts-c-puct", str(args.mcts_c_puct),
                "--mcts-prior", "policy",
                "--mcts-collapse-max-steps", str(args.mcts_collapse_max_steps),
                "--mcts-max-nodes", str(args.mcts_max_nodes),
                "--mcts-dirichlet-alpha", str(args.dirichlet_alpha),
                "--mcts-dirichlet-epsilon", str(args.dirichlet_epsilon),
                "--temperature-moves", str(args.temperature_moves),
                "--temperature-value", str(args.temperature_value),
                "--out", str(out_path),
                "--manifest-out", str(manifest_out),
            ],
            cwd=repo_root, stdout=logf, stderr=subprocess.STDOUT, check=True,
        )


def run_distill(
    repo_root: Path,
    ckpt_dir: Path,
    data_path: Path,
    init_checkpoint: Path,
    iteration: int,
    args: argparse.Namespace,
    events_path: Path,
) -> None:
    log_path = ckpt_dir / "distill.log"
    cmd = [
        str(repo_root / "training" / ".venv" / "bin" / "python"),
        str(repo_root / "training" / "train_bc.py"),
        "--data", str(data_path),
        "--out-dir", str(ckpt_dir),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--hidden-dim", str(args.hidden_dim),
        "--depth", str(args.depth),
        "--lr", str(args.lr),
        "--value-weight", str(args.value_weight),
        "--policy-weight", "1.0",
        "--data-mode", "mcts-distill",
        "--split-by", "seed",
        "--init-from-checkpoint", str(init_checkpoint),
        "--events-out", str(events_path),
        "--events-iteration", str(iteration),
        "--verbose",
    ]
    if args.kl_anchor_weight > 0:
        cmd.extend(["--kl-anchor-checkpoint", str(init_checkpoint),
                    "--kl-anchor-weight", str(args.kl_anchor_weight)])
    with log_path.open("w") as logf:
        subprocess.run(cmd, cwd=repo_root, stdout=logf, stderr=subprocess.STDOUT, check=True)


def run_gate(
    repo_root: Path,
    iter_dir: Path,
    model_url: str,
    args: argparse.Namespace,
    manifest_out: Path,
    iteration: int,
) -> int:
    log_path = iter_dir / "gate.log"
    with log_path.open("w") as logf:
        result = subprocess.run(
            [
                "npm", "--workspace", "backend", "run", "sim:eval-gate", "--",
                "--selection", "mcts",
                "--games", str(args.eval_games),
                "--max-steps", "500",
                "--seed-start", str(args.eval_seed_start),
                "--model-side", "both",
                "--model-url", model_url,
                "--manifest-out", str(manifest_out),
                "--mcts-simulations", str(args.mcts_simulations),
                "--mcts-c-puct", str(args.mcts_c_puct),
                "--mcts-leaf", "value-head",
                "--mcts-prior", "policy",
                "--mcts-collapse-max-steps", str(args.mcts_collapse_max_steps),
                "--mcts-max-nodes", str(args.mcts_max_nodes),
                "--min-ci-lower", str(args.eval_min_ci_lower),
                "--min-games", str(args.eval_games),
            ],
            cwd=repo_root, stdout=logf, stderr=subprocess.STDOUT, check=False,
        )
    return int(result.returncode)


def ensure_onnx(repo_root: Path, ckpt: Path) -> Path:
    onnx = ckpt.with_name("policy.onnx")
    if not onnx.exists() or onnx.stat().st_mtime < ckpt.stat().st_mtime:
        export_checkpoint_to_onnx(repo_root, ckpt, onnx)
    return onnx


def load_summary(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf8"))
    return payload.get("summary", {}) if isinstance(payload, dict) else {}


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf8") as fh:
        return sum(1 for line in fh if line.strip())


def emit_event(events_path: Path, payload: dict[str, Any]) -> None:
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf8") as fh:
        fh.write(json.dumps(payload) + "\n")


def _adapt_for_dagger_promote(state: R12State):
    # decide_promotion only reads .promoted_wilson_lower from the state-like
    # object, so a structural duck-type is enough.
    class _Adapter:
        promoted_wilson_lower = state.promoted_wilson_lower
    return _Adapter()


def _promotion_args(args: argparse.Namespace, state: R12State) -> argparse.Namespace:
    # decide_promotion reads .eval_min_ci_lower and .per_matchup_drop_tolerance.
    # We never set matchup violations so the latter is unused but must exist.
    ns = argparse.Namespace()
    ns.eval_min_ci_lower = args.eval_min_ci_lower
    ns.per_matchup_drop_tolerance = 1.0  # disable matchup floor
    return ns


def save_state(path: Path, state: R12State) -> None:
    payload = {
        "promoted_checkpoint": str(state.promoted_checkpoint) if state.promoted_checkpoint else None,
        "promoted_wilson_lower": state.promoted_wilson_lower,
        "iterations": state.iterations,
        "consecutive_failures": state.consecutive_failures,
        "halted": state.halted,
        "halt_reason": state.halt_reason,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf8")


def load_state(path: Path) -> R12State:
    if not path.exists():
        return R12State()
    payload = json.loads(path.read_text(encoding="utf8"))
    state = R12State()
    state.promoted_checkpoint = Path(payload["promoted_checkpoint"]) if payload.get("promoted_checkpoint") else None
    state.promoted_wilson_lower = payload.get("promoted_wilson_lower")
    state.iterations = list(payload.get("iterations") or [])
    state.consecutive_failures = int(payload.get("consecutive_failures") or 0)
    state.halted = bool(payload.get("halted") or False)
    state.halt_reason = payload.get("halt_reason")
    return state


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--init-checkpoint", default=None,
                   help="Starting checkpoint for iteration 0 (e.g., R4 ckpt). Required unless --resume-state has one.")
    p.add_argument("--resume-state", default=None)
    p.add_argument("--repo-root", default=None)
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--selfplay-games", type=int, default=200)
    p.add_argument("--selfplay-seed-start", type=int, default=30000)
    p.add_argument("--eval-games", type=int, default=200,
                   help="Games per side for the post-iteration gate.")
    p.add_argument("--eval-seed-start", type=int, default=9000)
    p.add_argument("--eval-min-ci-lower", type=float, default=0.30,
                   help="Promotion floor. Distinct from the final-gate target — iterations promote on RELATIVE improvement above the current promoted floor.")
    p.add_argument("--mcts-simulations", type=int, default=100)
    p.add_argument("--mcts-c-puct", type=float, default=1.5)
    p.add_argument("--mcts-collapse-max-steps", type=int, default=64)
    p.add_argument("--mcts-max-nodes", type=int, default=5000)
    p.add_argument("--dirichlet-alpha", type=float, default=0.3)
    p.add_argument("--dirichlet-epsilon", type=float, default=0.25)
    p.add_argument("--temperature-moves", type=int, default=6)
    p.add_argument("--temperature-value", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--value-weight", type=float, default=1.0)
    p.add_argument("--kl-anchor-weight", type=float, default=0.0,
                   help="Optional anti-forgetting anchor weight (init checkpoint as anchor).")
    # serve_onnx_context reads .device/.amp/etc indirectly; keep these
    # minimal Namespace fields so DAgger's helper doesn't crash on .get.
    p.add_argument("--device", default="cpu")
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


# Re-export so callers/inspectors don't have to import dagger_orchestrator.
__all__ = ["main", "R12State", "run_iteration"]


if __name__ == "__main__":
    main()
