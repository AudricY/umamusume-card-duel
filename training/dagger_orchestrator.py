"""DAgger iteration orchestrator.

Chains the existing TS exporters/evaluator and the train_bc.py loop into
a single command-line driver that can run N iterations of:

    1. Generate model-visited trajectories with a teacher-relabel pass.
    2. Convert decision traces into trainer-loadable rows.
    3. Mix the relabeled rows with rule-bot rows under a configurable
       weighted policy and a bounded replay buffer.
    4. Retrain via train_bc.py, resuming from the previous iteration's
       promoted checkpoint.
    5. Run the corrected eval gate; promote when the gate's Wilson lower
       bound does not regress the previous promoted iteration's lower
       bound, and otherwise roll back.
    6. Record a per-iteration manifest with parent provenance, source
       counts, mix ratios, eval result, and decision.

This is the minimum viable orchestrator behind item 11 in the AI
performance backlog. Replay buffer staleness eviction, KL-anchor anti-
forgetting, opponent-pool sampling (item 12), and best-response
exploitation tracking (item 13's tracked time-series) are intentionally
left as follow-up wiring.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from events import EventWriter
from opponent_pool import OpponentPool, PoolEntry
from uma_ai.node_bridge import (
    export_training_examples,
    mix_sources,
    relabel_decision_trace,
    run_eval_gate,
    run_evaluator,
)


@dataclass
class IterationConfig:
    iteration: int
    games: int
    teacher: str
    # R7: optional comma-separated multi-teacher override for the TS-side
    # `--trace-teacher` flag. Empty string => fall back to `teacher`. Does
    # NOT change `selection` (the playing agent in iter-0); only changes
    # which teachers get applied to each visited state when emitting the
    # trace rows. Validated downstream by parseTraceTeacherList.
    trace_teachers: str
    rule_bot_weight: float
    relabeled_weight: float
    rule_bot_replay_weight: float
    rollout_steps: int
    rollout_crn_samples: int
    eval_min_games: int
    eval_min_ci_lower: float
    epochs: int
    batch_size: int
    hidden_dim: int
    depth: int
    seed_offset: int = 0
    eval_seed_start: int = 9000


@dataclass
class OrchestratorState:
    promoted_checkpoint: Path | None = None
    promoted_wilson_lower: float | None = None
    iterations: list[dict[str, Any]] = field(default_factory=list)
    # Item 13 residual: halt the orchestrator after 2 consecutive
    # rejections to surface the regression to the operator instead of
    # grinding through more iterations against a broken target. Cleared
    # on any successful promotion.
    consecutive_failures: int = 0
    halted: bool = False
    halt_reason: str | None = None


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root or Path(__file__).resolve().parents[1])
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Write the orchestrator's own real PID so the harness reads the true
    # process regardless of launch wrapper. `nohup npm run ... &` captures the
    # npm wrapper PID (python is a grandchild); pid.txt is the source of truth.
    (out_dir / "pid.txt").write_text(f"{os.getpid()}\n")
    state = OrchestratorState()

    if args.resume_state:
        state = load_state(Path(args.resume_state))

    events = EventWriter(out_dir)
    events.emit_run(
        stage="orchestrator",
        event_type="run_started",
        iterations=args.iterations,
        games=args.games,
        teacher=args.teacher,
        epochs=args.epochs,
        resumed_from_iter=max((e["iteration"] for e in state.iterations), default=-1),
    )

    base_config = IterationConfig(
        iteration=0,
        games=args.games,
        teacher=args.teacher,
        trace_teachers=args.trace_teachers,
        rule_bot_weight=args.rule_bot_weight,
        relabeled_weight=args.relabeled_weight,
        rule_bot_replay_weight=args.replay_weight,
        rollout_steps=args.rollout_steps,
        rollout_crn_samples=args.rollout_crn_samples,
        eval_min_games=args.eval_games,
        eval_min_ci_lower=args.eval_min_ci_lower,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    )

    rule_bot_replay = ensure_rule_bot_replay(repo_root, out_dir, args, base_config, events)
    pool_path = out_dir / "opponent-pool.json"
    pool = OpponentPool.from_json(pool_path)

    last_iter = max((entry["iteration"] for entry in state.iterations), default=-1)
    for iteration in range(last_iter + 1, args.iterations):
        if state.halted:
            print(f"[orchestrator] halted before iteration {iteration}: {state.halt_reason}")
            events.emit_run(stage="orchestrator", event_type="halted_before_iteration", iteration_skipped=iteration, halt_reason=state.halt_reason)
            break
        cfg = IterationConfig(**{**base_config.__dict__, "iteration": iteration})
        record = run_iteration(repo_root, out_dir, cfg, state, rule_bot_replay, pool, args, events)
        state.iterations.append(record)
        save_state(out_dir / "orchestrator-state.json", state)
        pool.to_json(pool_path)

    events.emit_run(
        stage="orchestrator",
        event_type="run_completed",
        promoted_iterations=[i["iteration"] for i in state.iterations if i.get("promoted")],
        halted=state.halted,
        halt_reason=state.halt_reason,
    )

    summary = {
        "iterations": state.iterations,
        "promoted_checkpoint": str(state.promoted_checkpoint) if state.promoted_checkpoint else None,
        "promoted_wilson_lower": state.promoted_wilson_lower,
        "halted": state.halted,
        "halt_reason": state.halt_reason,
    }
    print(json.dumps(summary, indent=2))


def ensure_rule_bot_replay(repo_root: Path, out_dir: Path, args: argparse.Namespace, cfg: IterationConfig, events: EventWriter | None = None) -> Path:
    """Always-on rule-bot rehearsal slice.

    The rule-bot replay buffer grounds the loop against the original
    teacher distribution so the model can't drift to a state distribution
    the rule bot never visits.
    """

    replay = out_dir / "rule-bot-replay.jsonl"
    if replay.exists() and replay.stat().st_size > 0 and not args.refresh_rule_bot:
        if events is not None:
            events.emit_run(stage="rule-bot-replay", event_type="reused", path=str(replay), bytes=replay.stat().st_size)
        return replay
    print(f"[orchestrator] generating rule-bot replay ({args.replay_games} games)")
    if events is not None:
        events.emit_run(stage="rule-bot-replay", event_type="started", games=args.replay_games, seed_start=args.replay_seed_start)
    export_training_examples(
        repo_root,
        replay,
        seed_start=args.replay_seed_start,
        games=args.replay_games,
        max_steps=args.max_steps,
    )
    if events is not None:
        events.emit_run(stage="rule-bot-replay", event_type="completed", path=str(replay), rows=count_lines(replay))
    return replay


def run_iteration(
    repo_root: Path,
    root_dir: Path,
    cfg: IterationConfig,
    state: OrchestratorState,
    rule_bot_replay: Path,
    pool: OpponentPool,
    args: argparse.Namespace,
    events: EventWriter | None = None,
) -> dict[str, Any]:
    iter_dir = root_dir / f"iter-{cfg.iteration:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    print(f"[orchestrator] === iteration {cfg.iteration} ===")
    if events is not None:
        events.emit(
            iteration=cfg.iteration,
            stage="iteration",
            event_type="started",
            games=cfg.games,
            epochs=cfg.epochs,
            rollout_steps=cfg.rollout_steps,
            rollout_crn_samples=cfg.rollout_crn_samples,
            parent_promoted=str(state.promoted_checkpoint) if state.promoted_checkpoint else None,
        )

    trace_path = iter_dir / "trace.jsonl"
    relabeled_path = iter_dir / "relabeled.jsonl"
    mixed_path = iter_dir / "mixed.jsonl"
    trace_manifest = iter_dir / "trace.manifest.json"
    relabel_manifest = iter_dir / "relabel.manifest.json"
    mix_manifest = iter_dir / "mix.manifest.json"

    # Iteration 0 has no trained checkpoint yet — we exercise the data
    # spine with a teacher-driven trajectory so the first checkpoint sees
    # both rule-bot and teacher-on-rule-bot states.
    selection = cfg.teacher if state.promoted_checkpoint is None else "policy"

    seed_start = args.trace_seed_start + cfg.iteration * cfg.games
    eval_seed_start = cfg.eval_seed_start + cfg.iteration * cfg.eval_min_games
    if events is not None:
        events.emit(iteration=cfg.iteration, stage="trace-gen", event_type="started", selection=selection, seed_start=seed_start)
    if state.promoted_checkpoint is None:
        run_evaluator(
            repo_root,
            selection=cfg.teacher,
            games=cfg.games,
            seed_start=seed_start,
            model_side="both",
            max_steps=args.max_steps,
            rollout_steps=cfg.rollout_steps,
            decision_trace_out=trace_path,
            trace_teacher=cfg.trace_teachers or cfg.teacher,
            rollout_crn_samples=cfg.rollout_crn_samples,
            manifest_out=trace_manifest,
        )
    else:
        run_evaluator_with_model(
            repo_root,
            iter_dir,
            state.promoted_checkpoint,
            cfg=cfg,
            seed_start=seed_start,
            trace_path=trace_path,
            manifest_out=trace_manifest,
            args=args,
        )
    if events is not None:
        events.emit(iteration=cfg.iteration, stage="trace-gen", event_type="completed", rows=count_lines(trace_path))

    if events is not None:
        events.emit(iteration=cfg.iteration, stage="relabel", event_type="started")
    relabel_decision_trace(
        repo_root,
        trace_path,
        relabeled_path,
        source=f"model-visited-{cfg.teacher}-relabeled",
        label_source=f"{cfg.teacher}-teacher",
        manifest_out=relabel_manifest,
    )
    if events is not None:
        events.emit(iteration=cfg.iteration, stage="relabel", event_type="completed", rows=count_lines(relabeled_path))

    mix_components: list[tuple[str, float, str]] = [
        (str(relabeled_path), cfg.relabeled_weight, f"iter-{cfg.iteration}-relabeled"),
        (str(rule_bot_replay), cfg.rule_bot_replay_weight, "rule-bot-replay"),
    ]
    if events is not None:
        events.emit(iteration=cfg.iteration, stage="mix", event_type="started", components=[(c[2], c[1]) for c in mix_components])
    mix_sources(
        repo_root,
        out=mixed_path,
        seed=f"iter-{cfg.iteration}-mix",
        components=mix_components,
        manifest_out=mix_manifest,
    )
    if events is not None:
        events.emit(iteration=cfg.iteration, stage="mix", event_type="completed", rows=count_lines(mixed_path))

    train_dir = iter_dir / "model"
    train_args = [
        sys.executable,
        str(repo_root / "training" / "train_bc.py"),
        "--data",
        str(mixed_path),
        "--out-dir",
        str(train_dir),
        "--epochs",
        str(cfg.epochs),
        "--batch-size",
        str(cfg.batch_size),
        "--hidden-dim",
        str(cfg.hidden_dim),
        "--depth",
        str(cfg.depth),
        "--lr-schedule",
        "cosine",
        "--lr-warmup-steps",
        "16",
    ]
    if state.promoted_checkpoint is not None:
        # DAgger semantics: each iteration is a fresh training run with
        # warm-started model weights, not a continuation of the prior
        # iteration's training state. ``--init-from-checkpoint`` loads
        # ``model_state`` only and keeps a fresh optimizer/scheduler/epoch
        # counter so ``--epochs`` is the per-iteration epoch budget.
        train_args += ["--init-from-checkpoint", str(state.promoted_checkpoint)]
    # Item 17: KL-anchor anti-forgetting against the prior promoted checkpoint.
    # Per-iteration weight ablation: --kl-anchor-weights iter0,iter1,iter2,...
    # is consumed positionally; missing positions default to --kl-anchor-weight.
    if state.promoted_checkpoint is not None and args.kl_anchor_weight > 0.0:
        kl_weight = _kl_weight_for_iteration(args, cfg.iteration)
        if kl_weight > 0.0:
            train_args += [
                "--kl-anchor-checkpoint",
                str(state.promoted_checkpoint),
                "--kl-anchor-weight",
                str(kl_weight),
            ]
    if events is not None:
        events.emit(iteration=cfg.iteration, stage="train", event_type="started", epochs=cfg.epochs, batch_size=cfg.batch_size, hidden_dim=cfg.hidden_dim, depth=cfg.depth)
        # Pass events.jsonl + iteration id to train_bc so per-epoch loss
        # events land on the same stream as the orchestrator's stage events.
        train_args += ["--events-out", str(events.path), "--events-iteration", str(cfg.iteration)]
        # Per-iteration TensorBoard subtree under runs/<run>/tb/iter-NNN.
        # `tensorboard --logdir runs/<run>/tb` overlays all iterations.
        tb_dir = events.run_dir / "tb" / f"iter-{cfg.iteration:03d}"
        train_args += ["--tb-log-dir", str(tb_dir)]
    subprocess.run(train_args, cwd=repo_root, check=True)
    train_manifest = json.loads((train_dir / "manifest.json").read_text(encoding="utf8"))
    if events is not None:
        final_train = train_manifest.get("metrics", {}).get("train", {})
        final_val = train_manifest.get("metrics", {}).get("val", {})
        events.emit(
            iteration=cfg.iteration,
            stage="train",
            event_type="completed",
            final_train_loss=final_train.get("loss"),
            final_train_policy_loss=final_train.get("policy_loss"),
            final_train_value_loss=final_train.get("value_loss"),
            final_train_accuracy=final_train.get("accuracy"),
            final_val_loss=final_val.get("loss"),
            final_val_accuracy=final_val.get("accuracy"),
        )

    # The orchestrator needs an ONNX-served model to evaluate strength
    # closed-loop. The full server bring-up is heavy; for the smoke we
    # gate-check by running the eval gate in selection=baseline mode if
    # --skip-policy-gate is set, which lets the loop exercise the
    # promote/reject path without standing up serve_onnx.
    eval_manifest_path = iter_dir / "gate.manifest.json"
    if events is not None:
        events.emit(iteration=cfg.iteration, stage="gate", event_type="started", mode="baseline" if args.skip_policy_gate else "policy", min_games=2 * cfg.eval_min_games)
    if args.skip_policy_gate:
        gate_returncode = run_eval_gate(
            repo_root,
            selection="baseline",
            games=cfg.eval_min_games,
            seed_start=eval_seed_start,
            min_games=2 * cfg.eval_min_games,
            min_ci_lower=cfg.eval_min_ci_lower,
            manifest_out=eval_manifest_path,
            require_zero_no_ops=True,
            rollout_crn_samples=cfg.rollout_crn_samples,
        )
    else:
        gate_returncode = run_policy_gate_with_serve(
            repo_root,
            iter_dir,
            train_dir / "checkpoint.pt",
            cfg=cfg,
            eval_seed_start=eval_seed_start,
            manifest_out=eval_manifest_path,
            args=args,
        )

    gate_payload = json.loads(eval_manifest_path.read_text(encoding="utf8"))
    summary = gate_payload.get("summary", {})
    wilson_lower = float(summary.get("wilson95", {}).get("lower", 0.0))

    eval_n = int(summary.get("games", cfg.eval_min_games * 2))
    if events is not None:
        events.emit(
            iteration=cfg.iteration,
            stage="gate",
            event_type="completed",
            returncode=gate_returncode,
            wilson_lower=wilson_lower,
            win_rate=summary.get("modelWinRate"),
            games=eval_n,
        )

    # Item 12: pool matchup eval. Runs *before* the promotion decision
    # so item 13's per-matchup floor violations can join the gate's failure
    # set. Promotion is rejected if any opponent's Wilson lower bound drops
    # by more than --per-matchup-drop-tolerance vs the prior iteration on
    # that opponent. Aggregate gate alone is preserved when the pool is
    # empty or pool eval is skipped.
    pool_eval_results: list[dict[str, Any]] = []
    pool_aggregate_wilson_lower: float | None = None
    cycling_alarm_iterations: list[int] = []
    if (
        not args.skip_policy_gate
        and args.pool_eval_games > 0
        and len(pool.entries) > 0
        and (train_dir / "checkpoint.pt").exists()
    ):
        if events is not None:
            events.emit(iteration=cfg.iteration, stage="pool-eval", event_type="started", n_opponents=len(pool.entries), games_each=args.pool_eval_games)
        for opponent_entry in list(pool.entries):
            try:
                result = run_pool_matchup_eval(
                    repo_root,
                    iter_dir,
                    train_dir / "checkpoint.pt",
                    opponent_entry,
                    cfg=cfg,
                    args=args,
                )
                pool_eval_results.append(result)
                if events is not None:
                    events.emit(
                        iteration=cfg.iteration,
                        stage="pool-eval",
                        event_type="matchup",
                        opponent_iteration=result.get("opponent_iteration"),
                        wilson_lower=result.get("wilson_lower"),
                        win_rate=result.get("win_rate"),
                        n_games=result.get("n_games"),
                    )
            except Exception as exc:  # pragma: no cover - logged for telemetry only
                print(f"[orchestrator] pool matchup vs iter-{opponent_entry.iteration} failed: {exc}")
                pool_eval_results.append({
                    "opponent_iteration": opponent_entry.iteration,
                    "wilson_lower": 0.0,
                    "win_rate": 0.0,
                    "n_games": 0,
                    "error": str(exc),
                })
                if events is not None:
                    events.emit(iteration=cfg.iteration, stage="pool-eval", event_type="matchup_error", opponent_iteration=opponent_entry.iteration, error=str(exc))
        if pool_eval_results:
            valid = [row for row in pool_eval_results if row.get("n_games", 0) > 0]
            total_n = sum(row["n_games"] for row in valid)
            total_wins = sum(row["win_rate"] * row["n_games"] for row in valid)
            if total_n > 0:
                pool_aggregate_wilson_lower = wilson_lower_bound(int(round(total_wins)), total_n)
        # Update per-opponent win-rate history persisted on the pool's
        # per-orchestrator state. We track history inline in iteration
        # records; cycling_alarm consumes the rebuilt history each call.
        history = build_per_opponent_history(state, pool_eval_results, current_iteration=cfg.iteration)
        cycling_alarm_iterations = OpponentPool.cycling_alarm(history)

    # Item 13 residual: per-matchup floor violations. Compute *before*
    # the promotion decision so they participate in promote/reject. A
    # matchup violates the floor if its current Wilson lower drops by more
    # than --per-matchup-drop-tolerance vs the most recent prior recorded
    # Wilson lower against the same opponent. First-time matchups (no
    # prior history) cannot trigger a drop.
    matchup_violations = compute_matchup_floor_violations(
        state,
        pool_eval_results,
        tolerance=args.per_matchup_drop_tolerance,
    )

    decision = decide_promotion(
        state,
        gate_returncode,
        wilson_lower,
        args,
        eval_n=eval_n,
        matchup_violations=matchup_violations,
    )
    if decision["promote"]:
        state.promoted_checkpoint = train_dir / "checkpoint.pt"
        state.promoted_wilson_lower = wilson_lower
        state.consecutive_failures = 0
    else:
        state.consecutive_failures += 1
        if state.consecutive_failures >= 2:
            state.halted = True
            state.halt_reason = (
                f"halt-after-2: 2 consecutive rejections; last reason "
                f"'{decision['reason']}'"
            )
    if events is not None:
        events.emit(
            iteration=cfg.iteration,
            stage="decision",
            event_type="promoted" if decision["promote"] else "rejected",
            wilson_lower=wilson_lower,
            reason=decision["reason"],
            consecutive_failures=state.consecutive_failures,
            halted=state.halted,
            matchup_violations=matchup_violations,
            cycling_alarm=cycling_alarm_iterations,
        )

    record = {
        "iteration": cfg.iteration,
        "selection": selection,
        "teacher": cfg.teacher,
        "trace_rows": count_lines(trace_path),
        "relabeled_rows": count_lines(relabeled_path),
        "mixed_rows": count_lines(mixed_path),
        "train_manifest": str(train_dir / "manifest.json"),
        "eval_manifest": str(eval_manifest_path),
        "wilson_lower": wilson_lower,
        "promoted": decision["promote"],
        "decision_reason": decision["reason"],
        "gate_returncode": gate_returncode,
        "rule_bot_replay_rows": count_lines(rule_bot_replay),
        "pool_evals": pool_eval_results,
        "pool_aggregate_wilson_lower": pool_aggregate_wilson_lower,
        "cycling_alarm": cycling_alarm_iterations,
        "matchup_floor_violations": matchup_violations,
        "consecutive_failures": state.consecutive_failures,
        "halted": state.halted,
        "halt_reason": state.halt_reason,
    }
    # Snapshot the promoted checkpoint into the pool, then apply
    # retention. Snapshotting *after* the iteration's pool eval avoids
    # self-play contamination of the per-opponent metrics.
    if decision["promote"]:
        snapshot_path = root_dir / "pool" / f"iter-{cfg.iteration:03d}" / "checkpoint.pt"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(train_dir / "checkpoint.pt", snapshot_path)
        meta_src = (train_dir / "checkpoint.pt").with_suffix(".pt.meta.json")
        if meta_src.exists():
            shutil.copy2(meta_src, snapshot_path.with_suffix(".pt.meta.json"))
        pool.add(PoolEntry(
            iteration=cfg.iteration,
            checkpoint_path=str(snapshot_path),
            wilson_lower_at_promotion=wilson_lower,
            value_mean_drift_from_prior=0.0,
        ))
        pool.retain()

    (iter_dir / "iteration-manifest.json").write_text(json.dumps({
        "iteration": cfg.iteration,
        "config": cfg.__dict__,
        "parent_checkpoint": str(state.promoted_checkpoint) if state.promoted_checkpoint else None,
        "previous_promoted_wilson_lower": state.promoted_wilson_lower,
        "result": record,
        "train_manifest": train_manifest,
    }, indent=2) + "\n", encoding="utf8")
    print(f"[orchestrator] iteration {cfg.iteration} {'promoted' if decision['promote'] else 'rejected'} ({decision['reason']})")
    if events is not None:
        events.emit(
            iteration=cfg.iteration,
            stage="iteration",
            event_type="completed",
            promoted=decision["promote"],
            wilson_lower=wilson_lower,
        )
    return record


def build_per_opponent_history(
    state: OrchestratorState,
    current_pool_eval: list[dict[str, Any]],
    *,
    current_iteration: int,
) -> dict[int, list[float]]:
    """Reconstruct each opponent's win-rate history across iterations.

    Pulls prior iteration records' ``pool_evals`` blocks plus the current
    iteration's freshly-computed list and returns a per-opponent ordered
    history suitable for ``OpponentPool.cycling_alarm``.
    """

    history: dict[int, list[float]] = {}
    for record in state.iterations:
        for row in record.get("pool_evals", []) or []:
            opp = int(row.get("opponent_iteration", -1))
            if opp < 0:
                continue
            history.setdefault(opp, []).append(float(row.get("win_rate", 0.0)))
    for row in current_pool_eval:
        opp = int(row.get("opponent_iteration", -1))
        if opp < 0:
            continue
        history.setdefault(opp, []).append(float(row.get("win_rate", 0.0)))
    return history


def wilson_lower_bound(successes: int, total: int, *, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    phat = successes / total
    denom = 1.0 + (z * z) / total
    center = (phat + (z * z) / (2 * total)) / denom
    half = z * math.sqrt((phat * (1 - phat) + (z * z) / (4 * total)) / total) / denom
    return max(0.0, center - half)


def run_pool_matchup_eval(
    repo_root: Path,
    iter_dir: Path,
    current_checkpoint: Path,
    opponent_entry: PoolEntry,
    *,
    cfg: IterationConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Stand up two serve_onnx instances and run a small policy-vs-policy eval.

    Two distinct ports are required because each ``serve_onnx_context``
    binds its own ephemeral port from the OS pool. The current iteration's
    checkpoint serves on the model URL; the pool member serves on the
    opponent URL. Decision is greedy on each side via ``selectedIndex[0]``.
    """

    matchup_dir = iter_dir / "pool" / f"vs-iter-{opponent_entry.iteration:03d}"
    matchup_dir.mkdir(parents=True, exist_ok=True)
    current_onnx = matchup_dir / "current.onnx"
    opponent_onnx = matchup_dir / "opponent.onnx"
    export_checkpoint_to_onnx(repo_root, current_checkpoint, current_onnx)
    export_checkpoint_to_onnx(repo_root, Path(opponent_entry.checkpoint_path), opponent_onnx)
    manifest_out = matchup_dir / "matchup.manifest.json"

    games = max(1, int(args.pool_eval_games))
    seed_start = (args.pool_eval_seed_start
                  + cfg.iteration * 10_000
                  + opponent_entry.iteration)

    with serve_onnx_context(repo_root, current_onnx, args) as current_url:
        with serve_onnx_context(repo_root, opponent_onnx, args) as opponent_url:
            run_evaluator(
                repo_root,
                selection="policy",
                games=games,
                seed_start=seed_start,
                model_side="both",
                max_steps=args.max_steps,
                rollout_steps=cfg.rollout_steps,
                rollout_crn_samples=cfg.rollout_crn_samples,
                model_url=current_url,
                opponent_model_url=opponent_url,
                manifest_out=manifest_out,
            )

    payload = json.loads(manifest_out.read_text(encoding="utf8"))
    summary = payload.get("summary", {})
    n_games = int(summary.get("games", 0))
    win_rate = float(summary.get("modelWinRate", 0.0))
    wins = int(round(win_rate * n_games)) if n_games > 0 else 0
    return {
        "opponent_iteration": opponent_entry.iteration,
        "wilson_lower": wilson_lower_bound(wins, n_games),
        "win_rate": win_rate,
        "n_games": n_games,
    }


def decide_promotion(
    state: OrchestratorState,
    gate_returncode: int,
    wilson_lower: float,
    args: argparse.Namespace,
    *,
    eval_n: int,
    matchup_violations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Promote/reject with a confidence-band tolerance instead of arithmetic ε.

    Backlog item 13 (v4.1): "≥0pp at n≥150, or if the gate runs at n<150,
    by ≥half the Wilson half-width at the configured n." The 1e-9 tolerance
    used in v4 chronically rejected within-noise iterations at small n —
    Wilson half-width at n=50 is ≈12pp, which dwarfs any genuine iter-on-
    iter improvement at this scale.

    Item 13 residual adds the per-matchup floor: any opponent whose Wilson
    lower bound dropped by more than --per-matchup-drop-tolerance fails the
    gate as a *separate* failure mode. The aggregate Wilson lower can pass
    while a single matchup is collapsing — that's exactly the regression
    pattern the per-matchup floor exists to catch.
    """

    if gate_returncode != 0:
        return {"promote": False, "reason": f"gate exited {gate_returncode}"}
    floor = state.promoted_wilson_lower if state.promoted_wilson_lower is not None else args.eval_min_ci_lower
    tolerance = wilson_band_tolerance(eval_n) if eval_n < 150 else 0.0
    if wilson_lower + tolerance < floor:
        return {
            "promote": False,
            "reason": f"wilson_lower {wilson_lower:.4f} + tolerance {tolerance:.4f} < floor {floor:.4f} (n={eval_n})",
        }
    if matchup_violations:
        worst = max(matchup_violations, key=lambda v: v["drop"])
        return {
            "promote": False,
            "reason": (
                f"per-matchup floor: opponent iter-{worst['opponent_iteration']} "
                f"dropped {worst['drop']:.4f} (>{args.per_matchup_drop_tolerance:.4f}) "
                f"vs prior wilson_lower"
            ),
        }
    return {
        "promote": True,
        "reason": f"wilson_lower {wilson_lower:.4f} + tolerance {tolerance:.4f} >= floor {floor:.4f} (n={eval_n})",
    }


def compute_matchup_floor_violations(
    state: OrchestratorState,
    current_pool_eval: list[dict[str, Any]],
    *,
    tolerance: float,
) -> list[dict[str, Any]]:
    """Return per-matchup floor violations for the current iteration.

    A violation is recorded when the opponent's current Wilson lower bound
    falls below the most-recent prior Wilson lower against the same
    opponent by more than `tolerance` (default 0.05 = 5pp). First-time
    matchups never trigger because there is no prior to compare against.
    """

    prior_by_opp: dict[int, float] = {}
    for record in state.iterations:
        for row in record.get("pool_evals", []) or []:
            opp = int(row.get("opponent_iteration", -1))
            if opp < 0 or row.get("n_games", 0) <= 0:
                continue
            prior_by_opp[opp] = float(row.get("wilson_lower", 0.0))
    violations: list[dict[str, Any]] = []
    for row in current_pool_eval:
        opp = int(row.get("opponent_iteration", -1))
        if opp < 0 or row.get("n_games", 0) <= 0:
            continue
        prior = prior_by_opp.get(opp)
        if prior is None:
            continue
        current = float(row.get("wilson_lower", 0.0))
        drop = prior - current
        if drop > tolerance:
            violations.append({
                "opponent_iteration": opp,
                "prior_wilson_lower": prior,
                "current_wilson_lower": current,
                "drop": drop,
                "tolerance": tolerance,
            })
    return violations


def _kl_weight_for_iteration(args: argparse.Namespace, iteration: int) -> float:
    """Resolve the KL-anchor weight for a given iteration.

    Item 17 mandates ablating off/low/high on at least one iteration.
    --kl-anchor-weights "0.0,0.1,0.5" sets per-iteration weights positionally;
    missing positions fall back to --kl-anchor-weight. Empty string disables
    the per-iteration override.
    """

    overrides = (args.kl_anchor_weights or "").strip()
    if overrides:
        parts = [p.strip() for p in overrides.split(",") if p.strip()]
        if 0 <= iteration < len(parts):
            try:
                return float(parts[iteration])
            except ValueError:
                pass
    return float(args.kl_anchor_weight)


def wilson_band_tolerance(n: int, *, p: float = 0.5, z: float = 1.96) -> float:
    """Half of the Wilson half-width at p=0.5 for a sample size n.

    p=0.5 produces the worst-case interval width, so the tolerance does not
    grow narrower mid-iteration if the model's actual win rate drifts.
    """

    if n <= 0:
        return 0.5
    denom = 1.0 + (z * z) / n
    half_width = z * math.sqrt((p * (1 - p) + (z * z) / (4 * n)) / n) / denom
    return half_width / 2.0


def run_evaluator_with_model(
    repo_root: Path,
    iter_dir: Path,
    checkpoint: Path,
    *,
    cfg: IterationConfig,
    seed_start: int,
    trace_path: Path,
    manifest_out: Path,
    args: argparse.Namespace,
) -> None:
    onnx_path = iter_dir / "policy.onnx"
    export_checkpoint_to_onnx(repo_root, checkpoint, onnx_path)
    with serve_onnx_context(repo_root, onnx_path, args) as model_url:
        run_evaluator(
            repo_root,
            selection="policy",
            games=cfg.games,
            seed_start=seed_start,
            model_side="both",
            max_steps=args.max_steps,
            rollout_steps=cfg.rollout_steps,
            decision_trace_out=trace_path,
            trace_teacher=cfg.trace_teachers or cfg.teacher,
            rollout_crn_samples=cfg.rollout_crn_samples,
            model_url=model_url,
            manifest_out=manifest_out,
        )


def run_policy_gate_with_serve(
    repo_root: Path,
    iter_dir: Path,
    checkpoint: Path,
    *,
    cfg: IterationConfig,
    eval_seed_start: int,
    manifest_out: Path,
    args: argparse.Namespace,
) -> int:
    onnx_path = iter_dir / "policy.gate.onnx"
    export_checkpoint_to_onnx(repo_root, checkpoint, onnx_path)
    with serve_onnx_context(repo_root, onnx_path, args) as model_url:
        return run_eval_gate(
            repo_root,
            selection="policy",
            games=cfg.eval_min_games,
            seed_start=eval_seed_start,
            min_games=2 * cfg.eval_min_games,
            min_ci_lower=cfg.eval_min_ci_lower,
            manifest_out=manifest_out,
            require_zero_no_ops=True,
            rollout_crn_samples=cfg.rollout_crn_samples,
            extra=["--model-url", model_url],
        )


def export_checkpoint_to_onnx(repo_root: Path, checkpoint: Path, onnx_path: Path) -> None:
    """Wrap export_onnx with a CalledProcessError-aware error message.

    Reviewer 2 #8: a vocab-hash divergence currently surfaces as a bare
    CalledProcessError. Catch it here and print a helpful guide so an
    operator who regenerated the vocab mid-sweep doesn't have to pattern-
    match on a stack trace.
    """

    try:
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "training" / "export_onnx.py"),
                "--checkpoint",
                str(checkpoint),
                "--out",
                str(onnx_path),
            ],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or ""
        if "Card vocab hash mismatch" in stderr:
            raise RuntimeError(
                f"Card vocab hash diverged between checkpoint {checkpoint} and the "
                f"runtime cardVocab.json. Either retrain from a fresh checkpoint "
                f"under the current vocab, or restore the vocab to the hash recorded "
                f"in the checkpoint's feature_schema.card_vocab.\n\n"
                f"Underlying export_onnx stderr:\n{stderr}"
            ) from exc
        raise


@contextlib.contextmanager
def serve_onnx_context(repo_root: Path, onnx_path: Path, args: argparse.Namespace) -> Iterator[str]:
    """Spin up serve_onnx, yield the model URL, always tear it down.

    Notes:
    - stdout/stderr → DEVNULL. v4 review caught a real bug: PIPE without a
      drainer eventually fills the kernel pipe buffer (~64KB on Linux) and
      blocks the server's writes. At item 17 scale that hangs the eval gate.
    - Outer try/finally guarantees the process is reaped even if the health
      poll, the subsequent run_evaluator, or a Ctrl-C interrupts. v4 had
      try/finally only around the inner run_evaluator call, leaving a
      window where a Popen could leak a port.
    """

    import socket
    import time
    import urllib.request

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    proc = subprocess.Popen(
        [
            sys.executable,
            str(repo_root / "training" / "serve_onnx.py"),
            "--model",
            str(onnx_path),
            "--port",
            str(port),
            "--provider",
            "cpu",
        ],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 30
        health_url = f"http://127.0.0.1:{port}/health"
        while time.time() < deadline:
            try:
                urllib.request.urlopen(health_url, timeout=2).read()
                yield f"http://127.0.0.1:{port}"
                return
            except Exception:
                time.sleep(0.3)
        raise TimeoutError(f"serve_onnx did not become healthy on port {port}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass


def start_serve_onnx(repo_root: Path, onnx_path: Path, args: argparse.Namespace):
    """Legacy entry point retained for backwards compat; prefer serve_onnx_context."""

    import socket
    import time
    import urllib.request

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    proc = subprocess.Popen(
        [
            sys.executable,
            str(repo_root / "training" / "serve_onnx.py"),
            "--model",
            str(onnx_path),
            "--port",
            str(port),
            "--provider",
            "cpu",
        ],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 30
    health_url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        try:
            urllib.request.urlopen(health_url, timeout=2).read()
            return proc, f"http://127.0.0.1:{port}"
        except Exception:
            time.sleep(0.3)
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    raise TimeoutError(f"serve_onnx did not become healthy on port {port}")


def stop_serve_onnx(proc) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.open("r", encoding="utf8"))


def save_state(path: Path, state: OrchestratorState) -> None:
    path.write_text(
        json.dumps(
            {
                "promoted_checkpoint": str(state.promoted_checkpoint) if state.promoted_checkpoint else None,
                "promoted_wilson_lower": state.promoted_wilson_lower,
                "iterations": state.iterations,
                "consecutive_failures": state.consecutive_failures,
                "halted": state.halted,
                "halt_reason": state.halt_reason,
            },
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )


def load_state(path: Path) -> OrchestratorState:
    payload = json.loads(path.read_text(encoding="utf8"))
    return OrchestratorState(
        promoted_checkpoint=Path(payload["promoted_checkpoint"]) if payload.get("promoted_checkpoint") else None,
        promoted_wilson_lower=payload.get("promoted_wilson_lower"),
        iterations=payload.get("iterations", []),
        consecutive_failures=int(payload.get("consecutive_failures", 0)),
        halted=bool(payload.get("halted", False)),
        halt_reason=payload.get("halt_reason"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--teacher", choices=["rollout", "search", "planner"], default="rollout")
    # R7: comma-separated trace-teacher override (e.g. "rollout,search,planner").
    # When non-empty, takes precedence over `--teacher` for the TS-side
    # `--trace-teacher` flag only — `--selection` (the playing agent in iter-0)
    # still comes from `--teacher`. TS-side validation in parseTraceTeacherList.
    parser.add_argument("--trace-teachers", default="",
                        help="Comma-separated trace teachers for multi-teacher BC (R7). Empty => single-teacher via --teacher.")
    parser.add_argument("--rollout-steps", type=int, default=200)
    parser.add_argument("--rollout-crn-samples", type=int, default=3)
    parser.add_argument("--rule-bot-weight", type=float, default=0.0)
    parser.add_argument("--relabeled-weight", type=float, default=0.6)
    parser.add_argument("--replay-weight", type=float, default=0.4)
    parser.add_argument("--replay-games", type=int, default=20)
    parser.add_argument("--replay-seed-start", type=int, default=12000)
    parser.add_argument("--refresh-rule-bot", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--eval-games", type=int, default=10)
    parser.add_argument("--eval-min-ci-lower", type=float, default=0.0)
    parser.add_argument("--trace-seed-start", type=int, default=14000)
    parser.add_argument("--skip-policy-gate", action="store_true",
                        help="Use selection=baseline for the gate to avoid the ONNX server bring-up. Useful for smoke runs.")
    parser.add_argument("--pool-eval-games", type=int, default=20,
                        help="Games per pool member during the per-opponent eval phase. Set to 0 to skip pool matchups (escape hatch for smokes).")
    parser.add_argument("--kl-anchor-weight", type=float, default=0.0,
                        help="Per-iteration KL-anchor weight against the prior promoted checkpoint (item 17).")
    parser.add_argument("--kl-anchor-weights", default="",
                        help="Comma-separated per-iteration overrides, e.g. '0.0,0.1,0.5' for off/low/high ablation.")
    parser.add_argument("--per-matchup-drop-tolerance", type=float, default=0.05,
                        help="Max allowed Wilson lower drop per opponent vs prior recorded eval (item 13).")
    parser.add_argument("--pool-eval-seed-start", type=int, default=20000,
                        help="Seed offset for pool matchup evals; per-opponent seeds are derived deterministically from iteration and opponent index.")
    parser.add_argument("--resume-state", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
