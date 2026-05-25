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
import math
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

# Throughput-spike Slice 2 (2026-05-21): default location of the bundled
# ORT 1.22 dynamic library, used by the Rust sim-cli's in-process ORT
# session (see `engine-rs/crates/engine/src/inference/mod.rs`). The
# orchestrator passes this as the `ORT_DYLIB_PATH` env var to the spawned
# Rust binaries. Overridable via the `ORT_DYLIB_PATH` env var on the
# orchestrator's own process (we honor whatever is already exported);
# this constant is the fallback that matches the Python venv libonnxruntime.
DEFAULT_ORT_DYLIB_PATH = (
    Path(__file__).resolve().parent
    / ".venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.22.0"
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
    # deck-pair-sampling Slice 2 (2026-05-22): self-play deck-sampling mode
    # in effect for this run. Captured once at run start and persisted into
    # orchestrator-state.json so a later reader can tell whether a
    # checkpoint's training corpus saw deck variety. Eval gate is always
    # 'fixed' (see run_gate); only the self-play mode is configurable.
    selfplay_deck_sampling: str = "uniform"


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
    # deck-pair-sampling Slice 2 (2026-05-22): CLI override pins the
    # self-play sampling mode for this run. Always reflect the CLI value
    # into state — a resumed run still respects the operator's current
    # --deck-sampling choice instead of silently inheriting the old one.
    state.selfplay_deck_sampling = args.deck_sampling

    emit_event(events_path, {
        "stage": "r12-orchestrator", "event_type": "run_started",
        "iterations": args.iterations,
        "selfplay_games": args.selfplay_games,
        "mcts_simulations": args.mcts_simulations,
        "eval_games": args.eval_games,
        "uma_slot_tokens": bool(args.uma_slot_tokens),
        "model_variant": str(args.model_variant),
        # deck-pair-sampling Slice 2: deck-sampling mode applied to self-play
        # this iter. Eval gate is always 'fixed' (see run_gate); recording the
        # self-play mode lets a later reader tell whether a checkpoint's
        # training corpus saw deck variety. Mirrored into orchestrator-state
        # via state.selfplay_deck_sampling below.
        "selfplay_deck_sampling": args.deck_sampling,
        "eval_deck_sampling": "fixed",
        # per-game-pfsp-league-retry Option B: surface the pool config so a
        # reader of events.jsonl can tell at a glance whether the run used
        # cross-iter opponent diversity. Pool source + per-iter composition
        # land later in the selfplay-pool events.
        "rollout_vs_pool": bool(getattr(args, "rollout_vs_pool", False)),
        "pool_size": int(getattr(args, "pool_size", 5)),
        "pfsp_floor": float(getattr(args, "pfsp_floor", 0.05)),
        "rollout_pool_state_file": getattr(args, "rollout_pool_state_file", None),
        "init_checkpoint": str(state.promoted_checkpoint) if state.promoted_checkpoint else None,
        "ts": time.time(),
    })

    if state.promoted_checkpoint is None:
        raise SystemExit("must pass --init-checkpoint or --resume-state with a promoted_checkpoint set")

    # deck-pair-sampling Slice 2: persist the selected sampling mode up front
    # so orchestrator-state.json carries the choice even if iter-0 crashes
    # mid-selfplay. A later reader can tell whether a checkpoint's training
    # corpus saw deck variety just from orchestrator-state.json.
    save_state(out_dir / "orchestrator-state.json", state)

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

    # R14/R7.b.3: auto-infer model-shape fields from the init checkpoint so an
    # operator passing the orchestrator defaults (128/3) against a 64/2
    # ckpt, or omitting the attention/slot-token flags against a set-attention
    # ckpt, does not crash at the distill step with a state-dict mismatch.
    args = _maybe_override_dims_from_checkpoint(args, state.promoted_checkpoint, events_path)
    if args.model_variant == "set_attention" and not args.uma_slot_tokens:
        raise SystemExit(
            "--model-variant set_attention requires --uma-slot-tokens "
            "(or an init checkpoint whose model_config enables slot tokens)."
        )

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
            if (
                args.halt_after_consecutive_failures > 0
                and state.consecutive_failures >= args.halt_after_consecutive_failures
            ):
                state.halted = True
                state.halt_reason = (
                    f"halt-after-{args.halt_after_consecutive_failures} "
                    "consecutive promotion failures"
                )
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

    # --- step 1: self-play ---
    # Branch on the per-game-pfsp-league-retry opponent-pool plumbing (Option B
    # from docs/ai-research/scoping/cross-iter-opponent-pool-selfplay.md): when
    # --rollout-vs-pool is OFF we run a single sim-mcts-selfplay invocation
    # against the current promoted checkpoint (legacy behavior, byte-identical
    # to pre-Option-B); when ON we run N invocations (one per pool opponent)
    # with per-opponent games allocated by PFSP weight, then concat into
    # iter-N/selfplay.jsonl annotated with opponentCheckpointIter rows.
    emit_event(events_path, {"stage": "selfplay", "event_type": "started",
                             "iteration": iteration, "games": args.selfplay_games,
                             "mcts_simulations": args.mcts_simulations,
                             "rollout_vs_pool": bool(getattr(args, "rollout_vs_pool", False)),
                             "ts": time.time()})
    t0 = time.time()
    pool_record: dict[str, Any] = {"rollout_vs_pool": False}
    if getattr(args, "rollout_vs_pool", False):
        pool_record = _run_pool_selfplay(
            repo_root, iter_dir, iteration, args, state,
            selfplay_path, selfplay_manifest, events_path,
        )
    if not pool_record.get("rollout_vs_pool", False):
        # Legacy path (single opponent = currently promoted ckpt).
        promoted_onnx = ensure_onnx(repo_root, state.promoted_checkpoint)
        with inference_context(repo_root, promoted_onnx, args) as model_url:
            run_selfplay(repo_root, iter_dir, model_url, args,
                         selfplay_path, selfplay_manifest, iteration,
                         games=args.selfplay_games,
                         seed_start=args.selfplay_seed_start + iteration * args.selfplay_games)
    selfplay_elapsed = time.time() - t0
    n_rows = count_lines(selfplay_path)
    emit_event(events_path, {"stage": "selfplay", "event_type": "completed",
                             "iteration": iteration, "rows": n_rows,
                             "elapsed_sec": selfplay_elapsed,
                             "rollout_vs_pool": bool(pool_record.get("rollout_vs_pool", False)),
                             "ts": time.time()})

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
    # Throughput-spike Slice 2: see selfplay call site above for the
    # inference_context contract.
    with inference_context(repo_root, new_onnx, args) as model_url:
        gate_rc = run_gate(repo_root, iter_dir, model_url, args, gate_manifest, iteration)
    gate_elapsed = time.time() - t0
    summary = load_summary(gate_manifest)
    wilson_lower = float(summary.get("wilson95", {}).get("lower", 0))
    win_rate = float(summary.get("modelWinRate", 0))
    games = int(summary.get("games", 0))
    fallbacks = int(summary.get("heuristicFallbacks", 0))
    # Bug 1 of `rust-port-orchestrator-wiring` Slice 2 follow-ups: when a
    # gate.manifest.json reports passed=true but the parser came back with
    # wilson_lower=0/games=0, the manifest shape and the parser have
    # diverged — silently writing zeros to orchestrator-state.json corrupts
    # downstream promotion decisions and iteration tracking. Fail loud
    # here so the next divergence is caught at gate-completion time, not
    # at a later promotion-decision read.
    if gate_manifest.exists():
        raw_manifest = json.loads(gate_manifest.read_text(encoding="utf8"))
        manifest_passed = bool(raw_manifest.get("passed", False)) if isinstance(raw_manifest, dict) else False
        if manifest_passed and (games <= 0 or not math.isfinite(wilson_lower)):
            raise RuntimeError(
                f"gate.manifest.json reports passed=true at {gate_manifest} but parser "
                f"extracted wilson_lower={wilson_lower!r} / games={games} — manifest shape "
                f"and load_summary have diverged. Check the Rust sim-eval-gate emitter "
                f"vs backend/src/sim/evalGate.ts::summarize."
            )
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
        # per-game-pfsp-league-retry Option B: capture the pool composition
        # used to drive this iter's selfplay (mirrors Slice-2's
        # selfplay_deck_sampling shape — operator can stratify corpora by
        # opponent without re-parsing the per-iter selfplay manifests).
        # When --rollout-vs-pool is OFF, this is `{"rollout_vs_pool": False}`.
        "rollout_pool": pool_record,
    }
    emit_event(events_path, {"stage": "r12-orchestrator", "event_type": "iteration_completed",
                             **record, "ts": time.time()})
    return record


def resolve_engine_command(engine: str, sim_name: str, repo_root: Path) -> list[str]:
    """Return the argv prefix that invokes the requested sim CLI under the
    requested engine, BEFORE any flag arguments are appended.

    `sim_name` is the kebab-case sim identifier (e.g. ``mcts-selfplay``,
    ``eval-gate``) — the suffix shared between the TS npm script
    (``sim:<name>``) and the Rust release binary (``sim-<name>``).

    For ``engine="rust"`` we return the absolute path to the prebuilt
    release binary at ``engine-rs/target/release/sim-<name>``. The binary
    is NOT built lazily — if it is missing we raise ``RuntimeError`` with
    a one-line build remediation so an operator can not silently fall
    back to the TS path. Build with::

        cd engine-rs && cargo build --release -p sim-cli

    For ``engine="ts"`` we return the legacy ``npm --workspace backend
    run sim:<name> --`` prefix unchanged. This is the opt-out escape
    hatch — the engine port handoff doc (``docs/ai-research/scoping/
    rust-engine-port-handoff.md``) covers when to reach for it.

    Rust accepts every TS flag name via clap aliases (orchestrator-compat
    surface, handoff doc Phase 1g) so the caller appends flags
    unconditionally regardless of which engine answered.
    """
    if engine == "rust":
        bin_path = repo_root / "engine-rs" / "target" / "release" / f"sim-{sim_name}"
        if not bin_path.exists():
            raise RuntimeError(
                f"Rust sim CLI binary missing: {bin_path}. "
                f"Build with: (cd engine-rs && cargo build --release -p sim-cli) "
                f"or rerun the orchestrator with --engine ts to opt out."
            )
        return [str(bin_path)]
    if engine == "ts":
        return ["npm", "--workspace", "backend", "run", f"sim:{sim_name}", "--"]
    raise ValueError(f"unknown engine: {engine!r} (expected 'rust' or 'ts')")


@contextlib.contextmanager
def inference_context(
    repo_root: Path,
    onnx_path: Path,
    args: argparse.Namespace,
) -> Iterator[str]:
    """Throughput-spike Slice 2 (2026-05-21): unified inference setup.

    For ``args.engine == "rust"`` the Rust sim-cli loads ONNX in-process
    via the bundled ``ort`` crate (Slice 1, commit ``c1b6b0a``), so no
    serve_onnx subprocess is spun up at all — we yield an empty
    ``model_url`` and the run_selfplay / run_gate sites read
    ``onnx_path`` instead. For ``args.engine == "ts"`` we keep the legacy
    serve_onnx HTTP path so the TS sim CLIs (which don't grok
    ``--onnx-path``) continue to work as the opt-out escape hatch.

    Keeping a single context-manager lets the call sites stay shaped the
    same regardless of engine — the run_iteration step body doesn't need
    a per-engine branch around the ``with`` block.
    """
    if args.engine == "rust":
        # Yield the onnx path as the "model url" payload so the
        # downstream subprocess builder doesn't need to know how
        # the inference layer was wired. The sim-cli command builder
        # below branches on args.engine to put it on --onnx-path.
        yield str(onnx_path)
        return
    with serve_onnx_context(repo_root, onnx_path, args) as model_url:
        yield model_url


def rust_subprocess_env(args: argparse.Namespace) -> dict[str, str]:
    """Throughput-spike Slice 2 (2026-05-21): subprocess env carrying
    ``ORT_DYLIB_PATH`` so the spawned sim-cli binary's ``ort`` session
    (load-dynamic, api-21) can resolve ``libonnxruntime.so.1.22.0`` at
    runtime. The Rust binary aborts at session-load time without it.

    Resolution order (first non-empty wins):
      1. Caller's exported ``ORT_DYLIB_PATH`` (lets an operator pin a
         custom ORT build without editing the orchestrator).
      2. ``DEFAULT_ORT_DYLIB_PATH`` constant (training venv's bundled
         libonnxruntime.so.1.22.0 — matches the Python serve_onnx side
         used for parity smokes).

    We splice into ``os.environ`` rather than constructing a fresh dict
    so the child inherits PATH, HOME, locale, NCCL config, etc.
    """
    env = os.environ.copy()
    if not env.get("ORT_DYLIB_PATH"):
        env["ORT_DYLIB_PATH"] = str(DEFAULT_ORT_DYLIB_PATH)
    return env


def run_selfplay(
    repo_root: Path,
    iter_dir: Path,
    model_url: str,
    args: argparse.Namespace,
    out_path: Path,
    manifest_out: Path,
    iteration: int,
    *,
    games: int | None = None,
    seed_start: int | None = None,
    log_name: str = "selfplay.log",
) -> None:
    log_path = iter_dir / log_name
    # per-game-pfsp-league-retry Option B: callers can override games/seeds
    # so the pool wrapper can fire N invocations with disjoint seed windows
    # into the same iter dir. When unset (legacy path) we fall back to
    # args.selfplay_games and the existing iter-stride seed formula —
    # byte-identical to pre-Option-B behavior.
    if games is None:
        games = int(args.selfplay_games)
    if seed_start is None:
        seed_start = int(args.selfplay_seed_start) + int(iteration) * int(args.selfplay_games)
    # Throughput-spike Slice 2: on the Rust path, inference_context yields
    # the onnx path as `model_url` (no serve_onnx subprocess); on the TS
    # path it stays a real http:// URL. The sim-cli accepts --onnx-path on
    # the Rust binary (Slice 1) and --model-url on the TS npm script.
    cmd = resolve_engine_command(args.engine, "mcts-selfplay", repo_root) + [
        "--games", str(games),
        "--seed-start", str(seed_start),
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
        "--wave-size", str(args.mcts_wave_size),
        "--virtual-loss", str(args.mcts_virtual_loss),
        "--out", str(out_path),
        "--manifest-out", str(manifest_out),
        # deck-pair-sampling Slice 2 (2026-05-22): self-play widens its deck
        # distribution by default. Default is `uniform` over the 13 player×AI
        # pairs (cf. `docs/ai-research/scoping/deck-pair-sampling.md`); the
        # Step-3 smoke verdict (aggregate Wilson-lower -4.8pp vs fixed-matchup
        # ceiling, within ±5pp envelope) classified this as no-regret. Eval
        # gate stays --deck-sampling=fixed (run_gate below) so tight-gate
        # history remains apples-to-apples. Override with --deck-sampling on
        # the orchestrator for legacy reproducibility runs.
        "--deck-sampling", args.deck_sampling,
    ]
    # The Rust sim-mcts-selfplay binary makes per-decision row recording
    # opt-in via --record-rows (TS mctsSelfPlay.ts always records). The
    # orchestrator consumes selfplay.jsonl as the distill input, so rows
    # are non-optional here — append the flag on the Rust path only.
    if args.engine == "rust":
        cmd.append("--record-rows")
    # Throughput-spike Slice 2: dispatch the inference handle.
    #   - Rust: in-process ORT, read ONNX directly (model_url here IS the
    #     onnx path from inference_context). Skip --model-url entirely
    #     (it's the legacy/ignored flag post-Slice-1; passing nothing is
    #     cleaner than passing an empty string).
    #   - TS: serve_onnx HTTP path (model_url is the http://... URL).
    env: dict[str, str] | None = None
    if args.engine == "rust":
        cmd.extend(["--onnx-path", model_url])
        env = rust_subprocess_env(args)
    else:
        cmd.extend(["--model-url", model_url])
    with log_path.open("w") as logf:
        subprocess.run(
            cmd, cwd=repo_root, stdout=logf, stderr=subprocess.STDOUT,
            check=True, env=env,
        )


def _load_rollout_pool_state(state_file: Path) -> list[dict[str, Any]]:
    """Read the prior-run orchestrator-state.json and return the list of
    promoted iterations with the fields we need (checkpoint path + Wilson
    lower bound at promotion time).

    Per the scoping doc (cross-iter-opponent-pool-selfplay.md), the pool
    draws from the last N PROMOTED iter ckpts of a prior loop's run.
    Missing/malformed entries are dropped silently — empty pool falls
    back to the warm-start path in `_resolve_pool_opponents`.
    """
    if not state_file.exists():
        return []
    try:
        payload = json.loads(state_file.read_text(encoding="utf8"))
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for entry in (payload.get("iterations") or []):
        if not isinstance(entry, dict):
            continue
        if not entry.get("promote"):
            continue
        ckpt = entry.get("checkpoint")
        if not ckpt:
            continue
        out.append({
            "iteration": int(entry.get("iteration", -1)),
            "checkpoint": str(ckpt),
            "wilson_lower": float(entry.get("wilson_lower", 0.0) or 0.0),
        })
    return out


def _resolve_pool_opponents(
    state: "R12State",
    args: argparse.Namespace,
    events_path: Path,
    iteration: int,
) -> list[dict[str, Any]]:
    """Build the opponent pool for this iter's selfplay.

    Sources (in priority order):
      1. --rollout-pool-state-file: read promoted iters from that file
         (the "history" surface — apples-to-apples with the v3.2 retrain
         currently feeding this experiment).
      2. The current run's own promoted iters (state.iterations) — kicks in
         once the loop has produced one promoted iter on its own.
    Either source may also contribute, so we union and dedupe by
    checkpoint path, then keep the newest --pool-size entries.

    Iter-0 fallback: when no promoted iter is available anywhere, seed the
    pool with the warm-start ckpt (state.kl_anchor_checkpoint, the fixed
    iter-0 origin). Falls back to the legacy single-opponent path if even
    that is unavailable.
    """
    pool_entries: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    # From an external history state file (the "the v3.2 uniform retrain
    # finished, point the pool at its loop dir" surface).
    state_file_str = getattr(args, "rollout_pool_state_file", None)
    if state_file_str:
        state_file = Path(state_file_str)
        external = _load_rollout_pool_state(state_file)
        for entry in external:
            ckpt_path = str(Path(entry["checkpoint"]).resolve())
            if ckpt_path in seen_paths:
                continue
            seen_paths.add(ckpt_path)
            pool_entries.append({
                "iteration": entry["iteration"],
                "checkpoint": ckpt_path,
                "wilson_lower": entry["wilson_lower"],
                "source": "state_file",
            })

    # From the current run's own promoted iters (builds up naturally as the
    # loop progresses — first non-empty when iter >= 1 in a fresh run).
    for entry in (state.iterations or []):
        if not entry.get("promote"):
            continue
        ckpt = entry.get("checkpoint")
        if not ckpt:
            continue
        ckpt_path = str(Path(ckpt).resolve())
        if ckpt_path in seen_paths:
            continue
        seen_paths.add(ckpt_path)
        pool_entries.append({
            "iteration": int(entry.get("iteration", -1)),
            "checkpoint": ckpt_path,
            "wilson_lower": float(entry.get("wilson_lower", 0.0) or 0.0),
            "source": "current_run",
        })

    # Sort newest-first by iteration so the --pool-size truncation keeps
    # the most-recent promoted ckpts (mirrors OpponentPool.retain's
    # recent_promoted policy).
    pool_entries.sort(key=lambda e: e["iteration"], reverse=True)
    pool_size = max(1, int(getattr(args, "pool_size", 5)))
    pool_entries = pool_entries[:pool_size]

    if not pool_entries:
        # Iter-0 fallback: seed the pool with the warm-start origin.
        warm = state.kl_anchor_checkpoint or state.promoted_checkpoint
        if warm is None:
            emit_event(events_path, {
                "stage": "selfplay-pool", "event_type": "empty_pool_no_fallback",
                "iteration": iteration, "ts": time.time(),
            })
            return []
        pool_entries.append({
            "iteration": -1,  # synthetic — pre-iter-0 origin
            "checkpoint": str(Path(warm).resolve()),
            # Use the run's current promoted Wilson lower if known so the
            # PFSP weight is at least informed; else 0.5 (uniform-ish prior).
            "wilson_lower": float(state.promoted_wilson_lower or 0.5),
            "source": "warm_start",
        })

    return pool_entries


def _pfsp_weights_from_pool(
    pool: list[dict[str, Any]], floor: float,
) -> list[float]:
    """PFSP raw weights normalized to sum 1.

    Mirrors training/opponent_pool.py:pfsp_weights — `max(floor, 1 - p_i)`
    where p_i is the opponent's Wilson lower bound vs the rule-bot eval
    (proxy for "how hard is this opponent right now"). Higher Wilson
    means a stronger opponent, which gets a SMALLER weight — wait, that's
    backwards. The intent in the scoping doc is "weight opponents the
    current model loses TO" — but at iter-0 we don't have per-opponent
    win-rate data. The closest available proxy is the opponent's
    rule-bot strength: stronger opponents are harder, so we weight by
    `max(floor, opponent_strength)` instead of `1 - opponent_strength`.
    (training/opponent_pool.py uses 1-p when p is the *current model's*
    win rate vs that opponent; here p is the opponent's win rate vs the
    rule-bot, which is the inverse proxy.)

    Falls back to uniform if all weights collapse to the floor.
    """
    eps = 1e-9
    if not pool:
        return []
    raw: list[float] = []
    for entry in pool:
        # Opponent strength proxy = its rule-bot Wilson lower at promotion.
        # Stronger opponents (higher Wilson lower) are harder for the
        # current model, so they get a larger weight.
        strength = float(entry.get("wilson_lower", 0.5) or 0.5)
        raw.append(max(floor, strength))
    if all(w <= floor + eps for w in raw):
        n = float(len(pool))
        return [1.0 / n] * len(pool)
    total = sum(raw)
    if total <= 0:
        n = float(len(pool))
        return [1.0 / n] * len(pool)
    return [w / total for w in raw]


def _allocate_games_per_opponent(total: int, weights: list[float]) -> list[int]:
    """Distribute `total` games across opponents proportionally to weights.

    Round down to integers, hand the leftover games to the highest-weight
    opponent (deterministic largest-remainder isn't strictly necessary at
    these small N; the highest-weight bucket already absorbs the slack
    naturally). When `total < len(weights)` we fall back to 1 game per
    opponent on the top-`total` opponents (mirrors the scoping doc's
    "min(selfplay-games, pool-size) invocations with 1 game each").
    """
    n = len(weights)
    if n == 0 or total <= 0:
        return []
    if total < n:
        # Sort opponents by weight desc, take top `total` and give each 1 game.
        ranked = sorted(range(n), key=lambda i: weights[i], reverse=True)
        out = [0] * n
        for i in ranked[:total]:
            out[i] = 1
        return out
    raw = [w * total for w in weights]
    floored = [int(x) for x in raw]
    used = sum(floored)
    leftover = total - used
    if leftover > 0:
        # Largest-remainder assignment.
        remainders = sorted(
            range(n),
            key=lambda i: (raw[i] - floored[i], weights[i]),
            reverse=True,
        )
        for i in remainders[:leftover]:
            floored[i] += 1
    return floored


def _augment_jsonl_with_opponent(
    src: Path, dst_handle, opponent_iter: int, opponent_checkpoint: str,
) -> int:
    """Append every non-empty row from src into dst_handle, augmented with
    an `opponentCheckpointIter` + `opponentCheckpointPath` field.

    Returns the count of rows written. Rows that fail to parse as JSON are
    written as-is (defense-in-depth — never drop selfplay data on a parse
    error; downstream distill is tolerant of unknown extra fields).
    """
    n = 0
    with src.open("r", encoding="utf8") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
                if isinstance(row, dict):
                    row["opponentCheckpointIter"] = int(opponent_iter)
                    row["opponentCheckpointPath"] = str(opponent_checkpoint)
                dst_handle.write(json.dumps(row) + "\n")
            except Exception:
                # Preserve the original line on parse failure — the row is
                # still valid distill input even without the annotation.
                dst_handle.write(raw if raw.endswith("\n") else raw + "\n")
            n += 1
    return n


def _run_pool_selfplay(
    repo_root: Path,
    iter_dir: Path,
    iteration: int,
    args: argparse.Namespace,
    state: "R12State",
    out_path: Path,
    manifest_out: Path,
    events_path: Path,
) -> dict[str, Any]:
    """per-game-pfsp-league-retry Option B (cross-iter opponent pool selfplay).

    Replace the single sim-mcts-selfplay invocation with N invocations (one
    per opponent in the pool) and concatenate the per-opponent JSONLs into
    `out_path` annotated with `opponentCheckpointIter` rows. Per the
    scoping doc (docs/ai-research/scoping/cross-iter-opponent-pool-selfplay.md),
    per-INVOCATION opponent fixing is the correct grain at workers=24 and
    ~5-25 games/opponent/iter — each game's opponent is fixed BEFORE the
    game starts, so we never hit the Phase J failure mode (per-RUN
    self-promoted-at-mode collapse).

    Returns a dict to be stored in the iteration record (`rollout_pool`
    field). If the pool resolves empty AND there's no warm-start fallback,
    returns `{"rollout_vs_pool": False}` so the caller falls back to the
    legacy single-opponent path.
    """
    pool = _resolve_pool_opponents(state, args, events_path, iteration)
    if not pool:
        emit_event(events_path, {
            "stage": "selfplay-pool", "event_type": "fallback_to_single",
            "iteration": iteration, "reason": "empty_pool", "ts": time.time(),
        })
        return {"rollout_vs_pool": False}

    floor = float(getattr(args, "pfsp_floor", 0.05) or 0.05)
    weights = _pfsp_weights_from_pool(pool, floor)
    games_alloc = _allocate_games_per_opponent(int(args.selfplay_games), weights)
    # Drop opponents that got 0 games (can only happen when total < pool size).
    nonzero = [(p, w, g) for p, w, g in zip(pool, weights, games_alloc) if g > 0]
    if not nonzero:
        emit_event(events_path, {
            "stage": "selfplay-pool", "event_type": "fallback_to_single",
            "iteration": iteration, "reason": "zero_games_alloc", "ts": time.time(),
        })
        return {"rollout_vs_pool": False}

    emit_event(events_path, {
        "stage": "selfplay-pool", "event_type": "pool_resolved",
        "iteration": iteration,
        "pool": [
            {"opponent_iter": p["iteration"], "checkpoint": p["checkpoint"],
             "wilson_lower": p["wilson_lower"], "source": p["source"],
             "weight": w, "games": g}
            for p, w, g in nonzero
        ],
        "ts": time.time(),
    })

    # Each invocation gets a disjoint seed window inside the iter's overall
    # seed slab so games never collide on seeds (mirrors the iter-stride
    # in the legacy formula).
    iter_seed_base = int(args.selfplay_seed_start) + int(iteration) * int(args.selfplay_games)
    seed_cursor = iter_seed_base

    per_opp_paths: list[Path] = []
    per_opp_manifests: list[Path] = []
    total_rows = 0
    # Concatenate as we go to bound peak disk usage (selfplay JSONL is
    # already largish per opponent on a full-budget run; the smoke case is
    # tiny so this is just future-proofing).
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf8") as concat_fh:
        for opp, weight, games in nonzero:
            opp_iter = opp["iteration"]
            opp_ckpt = Path(opp["checkpoint"])
            opp_label = f"iter{opp_iter}" if opp_iter >= 0 else "warm"
            opp_out = iter_dir / f"selfplay-vs-{opp_label}.jsonl"
            opp_manifest = iter_dir / f"selfplay-vs-{opp_label}.manifest.json"
            opp_log = f"selfplay-vs-{opp_label}.log"
            per_opp_paths.append(opp_out)
            per_opp_manifests.append(opp_manifest)

            opp_onnx = ensure_onnx(repo_root, opp_ckpt)
            emit_event(events_path, {
                "stage": "selfplay-pool", "event_type": "invocation_started",
                "iteration": iteration, "opponent_iter": opp_iter,
                "opponent_checkpoint": str(opp_ckpt), "weight": weight,
                "games": games, "seed_start": seed_cursor, "ts": time.time(),
            })
            with inference_context(repo_root, opp_onnx, args) as model_url:
                run_selfplay(
                    repo_root, iter_dir, model_url, args,
                    opp_out, opp_manifest, iteration,
                    games=games, seed_start=seed_cursor,
                    log_name=opp_log,
                )
            seed_cursor += games

            rows_written = _augment_jsonl_with_opponent(
                opp_out, concat_fh, opp_iter, str(opp_ckpt),
            )
            total_rows += rows_written
            emit_event(events_path, {
                "stage": "selfplay-pool", "event_type": "invocation_completed",
                "iteration": iteration, "opponent_iter": opp_iter,
                "rows": rows_written, "ts": time.time(),
            })

    # Synthesize a top-level selfplay.manifest.json so downstream tooling
    # that reads it (none load-bearing today, but the slice-2 schema
    # established the field) doesn't break. Sum games + reference the
    # per-opponent manifests.
    manifest_payload = {
        "args": {
            "rollout_vs_pool": True,
            "pool_size": int(getattr(args, "pool_size", 5)),
            "pfsp_floor": floor,
            "deckSampling": args.deck_sampling,
            "out": str(out_path),
            "manifestOut": str(manifest_out),
            "perOpponent": [
                {
                    "opponent_iter": p["iteration"],
                    "checkpoint": p["checkpoint"],
                    "weight": w,
                    "games": g,
                    "manifest": str(iter_dir / f"selfplay-vs-{'iter' + str(p['iteration']) if p['iteration'] >= 0 else 'warm'}.manifest.json"),
                }
                for (p, w, g) in nonzero
            ],
        },
        "summary": {
            "games": int(sum(g for _, _, g in nonzero)),
            "rows": total_rows,
            "perOpponentInvocations": len(nonzero),
        },
    }
    manifest_out.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf8")

    return {
        "rollout_vs_pool": True,
        "pool_size": int(getattr(args, "pool_size", 5)),
        "pfsp_floor": floor,
        "pool": [
            {"opponent_iter": p["iteration"], "checkpoint": p["checkpoint"],
             "wilson_lower": p["wilson_lower"], "source": p["source"]}
            for (p, _w, _g) in nonzero
        ],
        "pool_pfsp_weights": [w for (_p, w, _g) in nonzero],
        "games_per_opponent": [g for (_p, _w, g) in nonzero],
    }


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
    if args.model_variant != "mlp":
        cmd.extend(["--model-variant", str(args.model_variant)])
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
        "--state-dim", str(args.state_dim),
    ]
    if args.uma_slot_tokens:
        cmd.append("--uma-slot-tokens")
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
    cmd = resolve_engine_command(args.engine, "eval-gate", repo_root) + [
        "--selection", "mcts",
        "--games", str(args.eval_games),
        "--max-steps", "500",
        "--seed-start", str(args.eval_seed_start),
        "--model-side", "both",
        "--manifest-out", str(manifest_out),
        "--mcts-simulations", str(args.mcts_simulations),
        "--mcts-c-puct", str(args.mcts_c_puct),
        "--mcts-leaf", args.mcts_leaf,
        "--mcts-rollout-crn-samples", str(args.mcts_rollout_crn_samples),
        "--mcts-rollout-steps", str(args.mcts_rollout_steps),
        "--mcts-prior", "policy",
        "--mcts-collapse-max-steps", str(args.mcts_collapse_max_steps),
        "--mcts-max-nodes", str(args.mcts_max_nodes),
        "--wave-size", str(args.mcts_wave_size),
        "--virtual-loss", str(args.mcts_virtual_loss),
        "--min-ci-lower", str(args.eval_min_ci_lower),
        "--min-games", str(args.eval_games),
        "--progress-out", str(iter_dir / "gate-progress.jsonl"),
        "--workers", str(args.workers),
        # deck-pair-sampling Slice 2 (2026-05-22): eval-gate ALWAYS pins to
        # --deck-sampling=fixed regardless of the orchestrator's selfplay
        # sampling. The tight-gate history (0.5811 v3.0 ceiling and every
        # §4d/§4e re-verdict number) is built on fixed-matchup eval; flipping
        # the gate would break apples-to-apples comparability. Per-iter gate
        # at n=2×eval_games is also too noisy (Wilson half-width >±20pp) to
        # absorb matchup variance; diverse-matchup eval is a Slice 3 concern
        # at higher n. (See scoping doc § "Open questions".)
        "--deck-sampling", "fixed",
    ]
    # Throughput-spike Slice 2 (mirror run_selfplay): on the Rust path,
    # `model_url` is actually the onnx path (yielded by inference_context)
    # and we use --onnx-path + ORT_DYLIB_PATH env; on the TS path keep the
    # legacy --model-url/serve_onnx HTTP wiring.
    env: dict[str, str] | None = None
    if args.engine == "rust":
        cmd.extend(["--onnx-path", model_url])
        env = rust_subprocess_env(args)
    else:
        cmd.extend(["--model-url", model_url])
    with log_path.open("w") as logf:
        result = subprocess.run(
            cmd, cwd=repo_root, stdout=logf, stderr=subprocess.STDOUT,
            check=False, env=env,
        )
    return int(result.returncode)


def ensure_onnx(repo_root: Path, ckpt: Path) -> Path:
    onnx = ckpt.with_name("policy.onnx")
    if not onnx.exists() or onnx.stat().st_mtime < ckpt.stat().st_mtime:
        export_checkpoint_to_onnx(repo_root, ckpt, onnx)
    return onnx


def load_summary(manifest_path: Path) -> dict[str, Any]:
    """Return the canonical TS `summary` block from a gate manifest.

    TS shape (`backend/src/sim/evalGate.ts:156`): top-level
    `{ status, failures, ..., summary: { games, modelWinRate,
    wilson95: { lower, upper }, heuristicFallbacks, ... } }`. The Rust
    `sim-eval-gate` binary mirrors this shape (Bug fix 1 of
    `rust-port-orchestrator-wiring` Slice 2 follow-ups).

    Defense-in-depth: if a manifest lacks a `summary` wrapper but has a
    `wilsonLower` / `winRate` / `games` flat shape (legacy Rust manifest
    pre-fix at e.g. `runs/R110-rust-parity-rust/iter-0/gate.manifest.json`),
    re-project it into the TS shape so historical artifacts re-parse
    correctly. New runs must hit the wrapped shape; the legacy fallback
    is for archive readability only and will be removed once no
    pre-fix manifest is in use.
    """
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf8"))
    if not isinstance(payload, dict):
        return {}
    summary = payload.get("summary")
    if isinstance(summary, dict) and summary:
        return summary
    # Legacy Rust flat shape (pre-Bug-1 fix). Re-project into the TS
    # contract so the orchestrator's consumers see non-zero values.
    if "wilsonLower" in payload or "overall" in payload:
        overall = payload.get("overall") if isinstance(payload.get("overall"), dict) else {}
        wilson_lower = overall.get("wilsonLower", payload.get("wilsonLower", 0.0))
        wilson_upper = overall.get("wilsonUpper", payload.get("wilsonUpper", 0.0))
        win_rate = overall.get("winRate", payload.get("winRate", 0.0))
        games = overall.get("games", payload.get("games", 0))
        return {
            "games": games,
            "modelWinRate": win_rate,
            "wilson95": {"lower": wilson_lower, "upper": wilson_upper},
            "heuristicFallbacks": payload.get("heuristicFallbacks", 0),
        }
    return {}


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
    """If the init checkpoint stores a model_config, override model-shape
    fields to match it. Prevents the W6-extension footgun where the
    orchestrator defaults (128/3 or MLP) were passed against a narrower or
    set-attention checkpoint.
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
    ckpt_uses_uma_slot_tokens = bool(cfg.get("uses_uma_slot_tokens", False))
    ckpt_model_variant = cfg.get("model_variant")
    overrides: dict[str, Any] = {}
    if isinstance(ckpt_hidden, int) and ckpt_hidden > 0 and ckpt_hidden != args.hidden_dim:
        overrides["hidden_dim"] = (args.hidden_dim, ckpt_hidden)
        args.hidden_dim = ckpt_hidden
    if isinstance(ckpt_depth, int) and ckpt_depth > 0 and ckpt_depth != args.depth:
        overrides["depth"] = (args.depth, ckpt_depth)
        args.depth = ckpt_depth
    if ckpt_uses_uma_slot_tokens and not args.uma_slot_tokens:
        overrides["uma_slot_tokens"] = (False, True)
        args.uma_slot_tokens = True
    if (
        isinstance(ckpt_model_variant, str)
        and ckpt_model_variant
        and ckpt_model_variant != args.model_variant
    ):
        overrides["model_variant"] = (args.model_variant, ckpt_model_variant)
        args.model_variant = ckpt_model_variant
    if overrides:
        emit_event(events_path, {
            "stage": "r12-orchestrator", "event_type": "model_config_inferred_from_checkpoint",
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
        # deck-pair-sampling Slice 2 (2026-05-22): self-play deck-sampling
        # mode in effect for this run. Reader can stratify corpora by deck
        # variety without re-parsing the per-iter selfplay manifests.
        "selfplay_deck_sampling": state.selfplay_deck_sampling,
        "eval_deck_sampling": "fixed",
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
    # deck-pair-sampling Slice 2 (2026-05-22): older pre-Slice-2 state files
    # do not have this key — default to "uniform" (the new default) since
    # older runs were all "fixed" but the only legitimate resume target is a
    # fresh post-Slice-2 run; CLI override re-sets it on next save_state.
    state.selfplay_deck_sampling = str(payload.get("selfplay_deck_sampling") or "uniform")
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
    p.add_argument("--halt-after-consecutive-failures", type=int, default=2,
                   help="Stop after this many consecutive non-promoted "
                        "iterations. Set to 0 to run all requested iterations.")
    p.add_argument("--mcts-simulations", type=int, default=100)
    p.add_argument("--mcts-c-puct", type=float, default=1.5)
    p.add_argument("--mcts-leaf", default="value-head", choices=["value-head", "rollout"])
    p.add_argument("--mcts-rollout-crn-samples", type=int, default=3)
    p.add_argument("--mcts-rollout-steps", type=int, default=200)
    p.add_argument("--mcts-collapse-max-steps", type=int, default=64)
    p.add_argument("--mcts-max-nodes", type=int, default=5000)
    p.add_argument("--mcts-wave-size", type=int, default=1)
    p.add_argument("--mcts-virtual-loss", type=float, default=1.0)
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
    p.add_argument("--model-variant", choices=["mlp", "set_attention"], default="mlp",
                   help="R7.b.3 set-attention probe: forwarded to "
                        "train_bc.py --model-variant for distill. Default "
                        "mlp preserves legacy v3.0/v3.1/v3.2 loops. "
                        "set_attention requires --uma-slot-tokens and "
                        "hidden_dim=64; when the init checkpoint records a "
                        "non-default model_config.model_variant, the "
                        "orchestrator auto-infers it to avoid state_dict "
                        "mismatches on attention-loop resumes.")
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
    # per-game-pfsp-league-retry Option B (2026-05-22): cross-iter
    # opponent-pool selfplay. When ON, the iter's selfplay budget is split
    # across N PFSP-weighted opponents drawn from the last --pool-size
    # promoted iter ckpts (current run + optional --rollout-pool-state-file
    # for warm-starting from a prior run's history). Each opponent gets its
    # own sim-mcts-selfplay invocation; per-opponent JSONLs are concatenated
    # into the iter's selfplay.jsonl (with `opponentCheckpointIter` row
    # annotation) for the distill stage. Default OFF preserves the legacy
    # single-opponent (= current promoted ckpt) self-play. See scoping at
    # docs/ai-research/scoping/cross-iter-opponent-pool-selfplay.md.
    p.add_argument("--rollout-vs-pool", action="store_true",
                   help="per-game-pfsp-league-retry Option B: split the iter's "
                        "selfplay budget across PFSP-weighted prior-iter "
                        "opponents (default OFF — legacy single-opponent path).")
    p.add_argument("--pool-size", type=int, default=5,
                   help="Last N promoted iter ckpts considered for the pool "
                        "(default 5). Mirrors OpponentPool.retain recent_promoted.")
    p.add_argument("--pfsp-floor", type=float, default=0.05,
                   help="Minimum PFSP weight floor before normalization "
                        "(mirrors training/opponent_pool.py). Default 0.05.")
    p.add_argument("--rollout-pool-state-file", default=None,
                   help="Optional path to a prior run's orchestrator-state.json. "
                        "When set, its promoted iters seed the pool (in addition "
                        "to the current run's own promoted iters). The cleaner "
                        "of the two surfaces in the scoping doc.")
    # deck-pair-sampling Slice 2 (2026-05-22): default-on uniform deck
    # sampling in self-play. Eval gate stays --deck-sampling=fixed
    # unconditionally (see run_gate); this flag controls run_selfplay only.
    # Accepts the same surface as the Rust sim-cli sampler: fixed | uniform
    # | pair=<player>:<opponent>. Override to `fixed` for legacy
    # reproducibility runs that need the single-matchup distribution.
    p.add_argument("--deck-sampling", default="uniform",
                   help="Self-play deck-pair sampling mode (default 'uniform' "
                        "post-Slice-2). Accepts: fixed | uniform | "
                        "pair=<player>:<opponent>. Eval gate ALWAYS uses fixed "
                        "regardless of this flag — see run_gate.")
    # Engine dispatch for the per-iter sim CLIs (selfplay + gate). Rust
    # is the default since the engine port merged (handoff doc Phase 1g —
    # ~140-220x MCTS throughput, flag-compatible via clap aliases). TS
    # remains as an opt-out escape hatch for parity comparisons or if
    # the Rust binaries are unavailable. The dispatch happens inside
    # run_selfplay / run_gate via resolve_engine_command(); call sites
    # otherwise stay byte-identical.
    p.add_argument("--engine", choices=["rust", "ts"], default="rust",
                   help="Sim CLI engine for selfplay + gate. 'rust' (default) "
                        "calls engine-rs/target/release/sim-{mcts-selfplay,"
                        "eval-gate}; 'ts' calls the legacy npm sim:* scripts.")
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
