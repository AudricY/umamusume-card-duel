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
import os
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

# --- W6 recipe-fix (r110.md §4a) -------------------------------------------
# The W6 loop's iter-2-peak-then-rot was diagnosed as MONOTONE policy-prior
# representation drift from iter-0, driven by two confirmed mechanisms:
#   (1) per-iter-only distill on monotonically softening targets, and
#   (2) catastrophic forgetting from a KL anchor that MOVED to the previous
#       iter's checkpoint (zero cumulative anti-drift).
# These two named flags default ON (the intended new recipe) but are
# trivially flippable for an A/B loop run. See r110.md §3 and §4a.
W6_FIX_CROSS_ITER_REPLAY = True   # change 1: bounded cross-iter replay mixture
W6_FIX_FIXED_KL_ANCHOR = True     # change 2: KL anchor pinned to iter-0/SL ckpt
# Bounded-buffer policy for change 1: how many of the most-recent prior
# iterations' selfplay corpora to mix in alongside the current iter's, and
# the fraction of the mixture drawn from those older vintages. Conservative
# defaults: a 3-iter window, ~40% older vintage, so the current iter still
# dominates while older distributions damp the monotone target-softening.
W6_REPLAY_WINDOW = 3
W6_REPLAY_OLD_FRACTION = 0.40
# ---------------------------------------------------------------------------


@dataclass
class R12State:
    promoted_checkpoint: Path | None = None
    promoted_wilson_lower: float | None = None
    iterations: list[dict[str, Any]] = field(default_factory=list)
    consecutive_failures: int = 0
    halted: bool = False
    halt_reason: str | None = None
    # W6 recipe-fix change 2 (r110.md §4a): the iter-0 / SL warm-start
    # checkpoint, captured ONCE and reused as the KL-anchor reference for
    # every iteration. Pinning it here (instead of the moving promoted
    # checkpoint) is what gives cumulative anti-drift. Persisted across
    # resume so a restarted run keeps anchoring to the same origin ckpt.
    kl_anchor_checkpoint: Path | None = None


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root or Path(__file__).resolve().parents[1])
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Write the orchestrator's own real PID so the harness reads the true
    # process regardless of launch wrapper. `nohup npm run ... &` captures the
    # npm wrapper PID (python is a grandchild); pid.txt is the source of truth.
    (out_dir / "pid.txt").write_text(f"{os.getpid()}\n")
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

    # W6 recipe-fix change 2 (r110.md §4a): capture the fixed KL anchor ONCE,
    # before any iteration runs, as the iter-0 / SL warm-start checkpoint.
    # This is never reassigned in the iteration loop — that fixedness is the
    # whole point (the old recipe re-anchored to the previous iter's ckpt,
    # giving zero cumulative anti-drift). A resumed run keeps whatever anchor
    # was persisted; only a brand-new run sets it from the init checkpoint.
    if state.kl_anchor_checkpoint is None:
        state.kl_anchor_checkpoint = state.promoted_checkpoint
    emit_event(events_path, {
        "stage": "r12-orchestrator", "event_type": "w6_recipe_fix_config",
        "cross_iter_replay": bool(args.w6_fix_cross_iter_replay),
        "fixed_kl_anchor": bool(args.w6_fix_fixed_kl_anchor),
        "kl_anchor_checkpoint": str(state.kl_anchor_checkpoint),
        "replay_window": int(args.w6_replay_window),
        "replay_old_fraction": float(args.w6_replay_old_fraction),
        "ts": time.time(),
    })

    # R14: auto-infer hidden_dim/depth from the init checkpoint so an
    # operator passing the orchestrator defaults (128/3) against a 64/2
    # ckpt does not crash at the distill step with a state-dict shape
    # mismatch — the silent-failure mode we hit on the first W6 extension
    # attempt. CLI-supplied dims still win when they match the checkpoint;
    # we only override when they would cause a load_state_dict failure.
    args = _maybe_override_dims_from_checkpoint(args, state.promoted_checkpoint, events_path)

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
    # W6 recipe-fix change 1 (r110.md §4a): the distill set is a bounded
    # cross-iter replay MIXTURE, not iter-N self-play only. Self-play
    # generation above is untouched; only what distill *consumes* changes.
    # _build_distill_dataset returns selfplay_path unchanged at iter-0/1 or
    # when the fix is flipped off, so pre-fix behavior is preserved.
    distill_data = _build_distill_dataset(
        out_dir, iter_dir, iteration, selfplay_path, args, events_path
    )
    emit_event(events_path, {"stage": "distill", "event_type": "started",
                             "iteration": iteration, "epochs": args.epochs,
                             "distill_data": str(distill_data), "ts": time.time()})
    t0 = time.time()
    # W6 recipe-fix change 2 (r110.md §4a): pass the FIXED iter-0/SL KL
    # anchor (state.kl_anchor_checkpoint), distinct from the moving
    # warm-start (state.promoted_checkpoint used for --init-from-checkpoint).
    run_distill(repo_root, ckpt_dir, distill_data, state.promoted_checkpoint,
                state.kl_anchor_checkpoint, iteration, args, events_path)
    distill_elapsed = time.time() - t0
    if not new_ckpt.exists():
        raise RuntimeError(f"distill did not produce checkpoint at {new_ckpt}")
    emit_event(events_path, {"stage": "distill", "event_type": "completed",
                             "iteration": iteration, "elapsed_sec": distill_elapsed,
                             "checkpoint": str(new_ckpt), "ts": time.time()})

    # --- step 2.5: crossover probe (held-out value-head fidelity check) ---
    # The probe measures whether the freshly-distilled value head's predictions
    # match rollout-CRN K=3 means on a corpus the new ckpt did NOT train on.
    # Held-out = the previous iter's selfplay (this iter trained on this iter's
    # selfplay, generated by the previous promoted ckpt). When two consecutive
    # `crossed=true` events fire, R14.I.3 retries a W8-style cheap-selfplay loop
    # to falsify whether the crossover unlocks compounding without rollouts.
    crossover = run_crossover_probe(repo_root, out_dir, iteration, new_ckpt, iter_dir, args, events_path)

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
        "crossover": crossover,
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
                # selfplay generates the training distribution; the leaf
                # evaluator should match the gate's so the targets the
                # network learns to imitate are scored the same way the
                # eval gate scores them.
                "--mcts-leaf", args.mcts_leaf,
                "--mcts-rollout-crn-samples", str(args.mcts_rollout_crn_samples),
                "--mcts-rollout-steps", str(args.mcts_rollout_steps),
                "--workers", str(args.workers),
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
    kl_anchor_checkpoint: Path,
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
        # R16-P1: select the frozen feature builder for distill. Default 110
        # (v3.0) keeps the loop byte-identical; --state-dim 164 trains v3.1.
        # The resulting checkpoint records this in model_config/feature_schema
        # so the downstream export_onnx + serve_onnx (both checkpoint-/graph-
        # driven) carry it end-to-end with no further orchestrator wiring.
        "--state-dim", str(args.state_dim),
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
        # W6 recipe-fix change 2 (r110.md §4a): the KL anchor is the FIXED
        # iter-0/SL checkpoint (kl_anchor_checkpoint), not the moving
        # warm-start (init_checkpoint == previous iter's promoted ckpt).
        # The old recipe used init_checkpoint here, re-anchoring every
        # iteration and giving zero cumulative anti-drift; v3 drifted ~2.4x
        # harder in KL than 96-d. When the fix is flipped OFF for an A/B
        # run, fall back to the original moving-anchor behavior.
        anchor = kl_anchor_checkpoint if args.w6_fix_fixed_kl_anchor else init_checkpoint
        cmd.extend(["--kl-anchor-checkpoint", str(anchor),
                    "--kl-anchor-weight", str(args.kl_anchor_weight)])
    # R16-P2 C6: opt the distill trainer into the per-Uma slot-token branch.
    # Conditional append (mirroring the kl-anchor passthrough above) so unset
    # is byte-identical to pre-C6 — no flag added to the cmd, no model_config
    # field flip, the produced checkpoint stays v3.0/v3.1. When set, every
    # distill invocation in the loop trains a v3.2 ckpt and the
    # checkpoint-driven export auto-gates the 7-input ONNX graph.
    if args.uma_slot_tokens:
        cmd.append("--uma-slot-tokens")
    with log_path.open("w") as logf:
        subprocess.run(cmd, cwd=repo_root, stdout=logf, stderr=subprocess.STDOUT, check=True)


def run_crossover_probe(
    repo_root: Path,
    out_dir: Path,
    iteration: int,
    new_ckpt: Path,
    iter_dir: Path,
    args: argparse.Namespace,
    events_path: Path,
) -> dict[str, Any] | None:
    """Run training/r14_value_crossover_probe.py and return its result dict.

    Probes the freshly-distilled checkpoint against the previous iter's
    selfplay corpus (held-out: this iter trained on this iter's selfplay).
    Returns None if no prior selfplay corpus exists (e.g., very first iter
    of a fresh run with no resume-state).
    """
    if iteration <= 0:
        prev_data: Path | None = None
    else:
        prev_data = out_dir / f"iter-{iteration - 1}" / "selfplay.jsonl"
        if not prev_data.exists():
            prev_data = None
    if prev_data is None:
        emit_event(events_path, {
            "stage": "r14-crossover", "event_type": "skipped",
            "iteration": iteration, "reason": "no_previous_selfplay_corpus",
            "ts": time.time(),
        })
        return None

    probe_out = iter_dir / "crossover.json"
    log_path = iter_dir / "crossover.log"
    emit_event(events_path, {
        "stage": "r14-crossover", "event_type": "started",
        "iteration": iteration, "data": str(prev_data),
        "checkpoint": str(new_ckpt), "ts": time.time(),
    })
    cmd = [
        str(repo_root / "training" / ".venv" / "bin" / "python"),
        str(repo_root / "training" / "r14_value_crossover_probe.py"),
        "--checkpoint", str(new_ckpt),
        "--data", str(prev_data),
        "--out", str(probe_out),
        "--ratio-cap", str(args.crossover_ratio_cap),
        "--pearson-min", str(args.crossover_pearson_min),
    ]
    if args.crossover_mse_floor is not None:
        cmd.extend(["--mse-floor", str(args.crossover_mse_floor)])
    if args.crossover_mse_floor_manifest:
        cmd.extend(["--mse-floor-manifest", args.crossover_mse_floor_manifest])
    with log_path.open("w") as logf:
        rc = subprocess.run(cmd, cwd=repo_root, stdout=logf, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not probe_out.exists():
        emit_event(events_path, {
            "stage": "r14-crossover", "event_type": "failed",
            "iteration": iteration, "returncode": rc, "ts": time.time(),
        })
        return None
    try:
        result = json.loads(probe_out.read_text(encoding="utf8"))
    except Exception as exc:
        emit_event(events_path, {
            "stage": "r14-crossover", "event_type": "parse_error",
            "iteration": iteration, "error": str(exc), "ts": time.time(),
        })
        return None
    emit_event(events_path, {
        "stage": "r14-crossover", "event_type": "completed",
        "iteration": iteration,
        "val_mse": result.get("val_mse"),
        "rmse": result.get("rmse"),
        "pearson_r": result.get("pearson_r"),
        "ratio": result.get("ratio"),
        "crossed": result.get("crossed"),
        "ts": time.time(),
    })
    return result


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
                "--mcts-leaf", args.mcts_leaf,
                "--mcts-rollout-crn-samples", str(args.mcts_rollout_crn_samples),
                "--mcts-rollout-steps", str(args.mcts_rollout_steps),
                "--mcts-prior", "policy",
                "--mcts-collapse-max-steps", str(args.mcts_collapse_max_steps),
                "--mcts-max-nodes", str(args.mcts_max_nodes),
                "--min-ci-lower", str(args.eval_min_ci_lower),
                "--min-games", str(args.eval_games),
                "--progress-out", str(iter_dir / "gate-progress.jsonl"),
                "--workers", str(args.workers),
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


def _build_distill_dataset(
    out_dir: Path,
    iter_dir: Path,
    iteration: int,
    current_selfplay: Path,
    args: argparse.Namespace,
    events_path: Path,
) -> Path:
    """W6 recipe-fix change 1 (r110.md §4a): build the distill training set
    from a BOUNDED cross-iter replay mixture instead of iter-N self-play only.

    Mechanism being countered: the old recipe distilled iter-N purely on
    iter-N self-play, on monotonically softening targets, producing monotone
    entropy inflation / representation drift from iter-0. Mixing a bounded
    window of older self-play vintages back in damps that monotone softening.

    Buffer policy (conservative, explicit):
      - window = args.w6_replay_window most-recent PRIOR iterations.
      - old_fraction = args.w6_replay_old_fraction of the materialized
        mixture is drawn (round-robin, deterministic, no shuffle/RNG) from
        those older vintages; the rest is the current iter verbatim.
      - The current iter's full corpus is ALWAYS included in full so we
        never train on less signal than the old recipe did.

    Fallback: when the fix is OFF, or at iter-0/1 where no usable prior
    vintage exists, this returns the current iter's selfplay path unchanged
    — byte-identical to the pre-fix behavior.
    """
    if not args.w6_fix_cross_iter_replay:
        return current_selfplay

    window = max(0, int(args.w6_replay_window))
    prior_paths: list[Path] = []
    for i in range(iteration - 1, max(-1, iteration - 1 - window), -1):
        p = out_dir / f"iter-{i}" / "selfplay.jsonl"
        if p.exists() and count_lines(p) > 0:
            prior_paths.append(p)

    # iter-0/1 (or no prior corpus on disk): preserve existing behavior.
    if not prior_paths:
        emit_event(events_path, {
            "stage": "distill-replay", "event_type": "passthrough",
            "iteration": iteration, "reason": "no_prior_vintage",
            "data": str(current_selfplay), "ts": time.time(),
        })
        return current_selfplay

    current_lines = [
        ln for ln in current_selfplay.read_text(encoding="utf8").splitlines() if ln.strip()
    ]
    n_current = len(current_lines)
    if n_current == 0:
        return current_selfplay

    # Older-vintage rows pooled newest-first, sampled deterministically by
    # an evenly-spaced stride so each vintage contributes proportionally
    # without an RNG (reproducible across resume).
    old_pool: list[str] = []
    for p in prior_paths:
        old_pool.extend(
            ln for ln in p.read_text(encoding="utf8").splitlines() if ln.strip()
        )
    old_frac = min(0.95, max(0.0, float(args.w6_replay_old_fraction)))
    # Solve n_old / (n_current + n_old) ~= old_frac for n_old, capped by pool.
    n_old_target = int(round(n_current * old_frac / max(1e-9, 1.0 - old_frac)))
    n_old = min(len(old_pool), n_old_target)
    if n_old <= 0:
        return current_selfplay
    stride = max(1, len(old_pool) // n_old)
    sampled_old = old_pool[::stride][:n_old]

    mixed_path = iter_dir / "distill-mixed.jsonl"
    with mixed_path.open("w", encoding="utf8") as fh:
        for ln in current_lines:
            fh.write(ln + "\n")
        for ln in sampled_old:
            fh.write(ln + "\n")

    emit_event(events_path, {
        "stage": "distill-replay", "event_type": "materialized",
        "iteration": iteration,
        "window": window,
        "old_fraction_target": old_frac,
        "n_current": n_current,
        "n_old_pool": len(old_pool),
        "n_old_mixed": len(sampled_old),
        "n_total": n_current + len(sampled_old),
        "vintages": [str(p) for p in prior_paths],
        "data": str(mixed_path),
        "ts": time.time(),
    })
    return mixed_path


def emit_event(events_path: Path, payload: dict[str, Any]) -> None:
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf8") as fh:
        fh.write(json.dumps(payload) + "\n")


def _maybe_override_dims_from_checkpoint(
    args: argparse.Namespace,
    init_checkpoint: Path,
    events_path: Path,
) -> argparse.Namespace:
    """If the init checkpoint stores a model_config, override hidden_dim/
    depth to match it. Prevents the W6-extension footgun where the
    orchestrator defaults (128/3) were passed against a 64/2 checkpoint.
    """
    try:
        import torch  # local import keeps the orchestrator usable without torch
        payload = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:
        emit_event(events_path, {
            "stage": "r12-orchestrator", "event_type": "dim_inference_skipped",
            "reason": f"could not load checkpoint: {exc}", "ts": time.time(),
        })
        return args
    cfg = payload.get("model_config") or (payload.get("metadata", {}) or {}).get("model_config") or {}
    ckpt_hidden = cfg.get("hidden_dim")
    ckpt_depth = cfg.get("depth")
    overrides: dict[str, Any] = {}
    if isinstance(ckpt_hidden, int) and ckpt_hidden > 0 and ckpt_hidden != args.hidden_dim:
        overrides["hidden_dim"] = (args.hidden_dim, ckpt_hidden)
        args.hidden_dim = ckpt_hidden
    if isinstance(ckpt_depth, int) and ckpt_depth > 0 and ckpt_depth != args.depth:
        overrides["depth"] = (args.depth, ckpt_depth)
        args.depth = ckpt_depth
    if overrides:
        emit_event(events_path, {
            "stage": "r12-orchestrator", "event_type": "dim_inferred_from_checkpoint",
            "checkpoint": str(init_checkpoint),
            "overrides": {k: {"cli": v[0], "checkpoint": v[1]} for k, v in overrides.items()},
            "ts": time.time(),
        })
    return args


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
        # W6 recipe-fix change 2 (r110.md §4a): persist the fixed KL anchor so
        # a resumed run keeps anchoring to the same origin checkpoint.
        "kl_anchor_checkpoint": str(state.kl_anchor_checkpoint) if state.kl_anchor_checkpoint else None,
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
    # W6 recipe-fix change 2 (r110.md §4a): restore the fixed KL anchor.
    state.kl_anchor_checkpoint = (
        Path(payload["kl_anchor_checkpoint"]) if payload.get("kl_anchor_checkpoint") else None
    )
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
    p.add_argument("--mcts-leaf", default="value-head", choices=["value-head", "rollout"])
    p.add_argument("--mcts-rollout-crn-samples", type=int, default=3)
    p.add_argument("--mcts-rollout-steps", type=int, default=200)
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
    p.add_argument("--state-dim", type=int, default=110,
                   help="R16-P1: frozen feature builder selector for the "
                        "distill trainer (96=v2, 110=v3.0, 164=v3.1). Default "
                        "110 (v3.0) — unset is byte-identical to pre-R16-P1 "
                        "loops. Passed straight to train_bc.py --state-dim; the "
                        "per-iter export + serve path is checkpoint-driven "
                        "(export_onnx reads model_config.state_dim) so a 164-d "
                        "distilled ckpt yields a 164-d ONNX graph and serve_onnx "
                        "resolves v3.1 with no extra wiring. The 164-d init "
                        "checkpoint must match (--init-checkpoint).")
    p.add_argument("--uma-slot-tokens", action="store_true",
                   help="R16-P2 C6: opt the distill trainer into the per-Uma "
                        "slot-token branch end-to-end (train_bc.py "
                        "--uma-slot-tokens). Default OFF — unset is "
                        "byte-identical to pre-C6 loops. When ON, the "
                        "distilled checkpoint's model_config.uses_uma_slot_"
                        "tokens=True is the source of truth read by "
                        "export_onnx (auto-gates the 7-input v3.2 ONNX "
                        "graph) and serve_onnx (schema dispatch). The init "
                        "checkpoint must be v3.2-compatible (either a "
                        "v3.2-trained ckpt or one scaffolded via C7's "
                        "make_v32_slot_token_init.py from a v3.0 source). "
                        "No exporter-side CLI flag — the pivot is the "
                        "checkpoint config, per C5.")
    p.add_argument("--kl-anchor-weight", type=float, default=0.0,
                   help="Optional anti-forgetting anchor weight (init checkpoint as anchor).")
    # W6 recipe-fix (r110.md §4a). Default ON via the module constants above;
    # these flags exist so an A/B loop run can flip either fix off without a
    # code edit. --w6-fix-* take precedence over the constants when passed.
    p.add_argument("--w6-fix-cross-iter-replay", dest="w6_fix_cross_iter_replay",
                   default=None, action="store_true",
                   help="W6 recipe-fix change 1: cross-iter replay mixture for distill (default ON).")
    p.add_argument("--no-w6-fix-cross-iter-replay", dest="w6_fix_cross_iter_replay",
                   action="store_false",
                   help="Disable W6 recipe-fix change 1 (per-iter-only distill, the old recipe).")
    p.add_argument("--w6-fix-fixed-kl-anchor", dest="w6_fix_fixed_kl_anchor",
                   default=None, action="store_true",
                   help="W6 recipe-fix change 2: pin KL anchor to iter-0/SL ckpt (default ON).")
    p.add_argument("--no-w6-fix-fixed-kl-anchor", dest="w6_fix_fixed_kl_anchor",
                   action="store_false",
                   help="Disable W6 recipe-fix change 2 (moving previous-iter anchor, the old recipe).")
    p.add_argument("--w6-replay-window", type=int, default=W6_REPLAY_WINDOW,
                   help="W6 recipe-fix: # of most-recent prior iters mixed into the distill set.")
    p.add_argument("--w6-replay-old-fraction", type=float, default=W6_REPLAY_OLD_FRACTION,
                   help="W6 recipe-fix: fraction of the distill mixture drawn from older vintages.")
    # serve_onnx_context reads .device/.amp/etc indirectly; keep these
    # minimal Namespace fields so DAgger's helper doesn't crash on .get.
    p.add_argument("--device", default="cpu")
    p.add_argument("--amp", action="store_true")
    # R13.W1 parallelism — the orchestrator's selfplay + gate stages
    # accept a --workers count that is forwarded to the underlying
    # `sim:mcts-selfplay` and `sim:eval-gate` runners.
    #
    # Throughput diagnosis (docs/ai-research/progress/r110.md "Throughput"):
    # selfplay (~48%) + mcts-gate (~52%) are ~99% of wall time and are pure
    # single-threaded CPU MCTS sim; the R110-W6-repro run used --workers 4 on
    # a 32-core box, leaving ~28 cores idle. The worker count is a PURE
    # distribution knob: both stages build a fixed seed-indexed task set
    # (mctsSelfPlay.ts:120 seed=seedStart+index; evalGate.ts:46) and only
    # partition it across processes (partitionSeeds/partitionTasks); per-game
    # RNG is seeded solely from the game seed (mctsSelfPlay.ts:255,
    # evaluateModelVsHeuristic.ts:269) with no worker/PID/order input. Raising
    # the default is therefore trajectory-neutral and stays bit-comparable to
    # the 4-worker R110-W6-repro baseline. Default 24 leaves headroom for the
    # serve_onnx process + OS on a 32-core box; override for other hardware.
    p.add_argument("--workers", type=int, default=24,
                   help="Parallel worker processes for selfplay and gate stages. "
                        "Pure distribution knob (trajectory-neutral): worker count only "
                        "partitions a fixed seed-indexed game set across processes. "
                        "Default 24 targets a 32-core box; override for other hardware.")
    # R14.I.1 crossover probe wiring — runs after distill against the previous
    # iter's selfplay corpus (held-out). Two consecutive crossed=true events
    # gate R14.I.3.
    p.add_argument("--crossover-ratio-cap", type=float, default=1.10)
    p.add_argument("--crossover-pearson-min", type=float, default=0.7)
    p.add_argument("--crossover-mse-floor", type=float, default=None,
                   help="Override the noise floor explicitly; otherwise read from --crossover-mse-floor-manifest.")
    p.add_argument("--crossover-mse-floor-manifest",
                   default="runs/R13-value-retrain-experiment/retrain/value-retrain.manifest.json",
                   help="W3 value-retrain manifest path; reads .val.loss as the irreducible-noise floor.")
    args = p.parse_args()
    # Resolve the W6 recipe-fix flags: unspecified -> module-constant default
    # (ON); an explicit --w6-fix-*/--no-w6-fix-* on the CLI wins. (r110.md §4a)
    if args.w6_fix_cross_iter_replay is None:
        args.w6_fix_cross_iter_replay = W6_FIX_CROSS_ITER_REPLAY
    if args.w6_fix_fixed_kl_anchor is None:
        args.w6_fix_fixed_kl_anchor = W6_FIX_FIXED_KL_ANCHOR
    return args


# Re-export so callers/inspectors don't have to import dagger_orchestrator.
__all__ = ["main", "R12State", "run_iteration"]


if __name__ == "__main__":
    main()
