"""PPO iteration orchestrator (F1).

Drives the on-policy PPO loop on top of the DAgger-produced warm-start
checkpoint:

    1. Stand up serve_onnx in stochastic mode against the parent checkpoint.
    2. Run the TS evaluator with --selection policy to collect K games of
       model-vs-rule-bot traces (trace.jsonl). The server's
       --default-sampling=stochastic flag injects stochastic sampling into
       the evaluator's default greedy POST body so no TS-side changes are
       required.
    3. Parse the trace into PPO-format trajectories (state/action features,
       behavior log-prob, reward = α·Δpoints per step + β·win/lose at
       absorbing state, episode boundaries).
    4. Shell out to train_ppo.py for GAE + PPO update; produces an
       updated checkpoint.
    5. Export the updated checkpoint to ONNX, run the corrected eval gate,
       run optional pool matchup eval, and promote/reject under the same
       Wilson-lower + per-matchup-floor + halt-after-2 semantics as
       dagger_orchestrator.

Design choice on stochastic propagation
---------------------------------------
The TS evaluator's chooseModelAction sends ``{observation, legalActions}``
to /predict and does not forward a ``sampling`` field. Rather than
extending the TS evaluator (out of scope; would also require argv
plumbing through evaluateModelVsHeuristic + a refresh of the
DecisionTraceRow contract), we add ``--default-sampling`` and
``--default-temperature`` to serve_onnx. The server then injects those
values when the request body omits them. The gate-eval path overrides
with an explicit ``sampling: greedy`` body and stays deterministic — but
note: this is only invoked when the gate runs through chooseModelAction
which never sets ``sampling``. Since the gate's strength signal comes
from the policy under its deployed (greedy) behavior, we set the gate's
serve_onnx server to ``--default-sampling greedy`` and the collector's
to ``stochastic``. They are two separate serve_onnx processes anyway.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from events import EventWriter
from opponent_pool import OpponentPool, PoolEntry

# Reuse a handful of helpers from the DAgger orchestrator to avoid
# duplicating tested code. The functions used here are pure (no DAgger
# state baked in) and operate on the orchestrator state / config dataclass
# fields they receive as args.
from dagger_orchestrator import (
    compute_matchup_floor_violations,
    decide_promotion,
    export_checkpoint_to_onnx,
    serve_onnx_context as _greedy_serve_onnx_context,
    wilson_lower_bound,
    run_pool_matchup_eval as _dagger_pool_matchup_eval,
    build_per_opponent_history,
    count_lines,
)
from uma_ai.node_bridge import run_eval_gate, run_evaluator


@dataclass
class PPOConfig:
    iteration: int
    games_per_update: int
    max_steps: int
    temperature: float
    lr: float
    clip_epsilon: float
    entropy_coef: float
    value_coef: float
    gae_lambda: float
    gae_gamma: float
    grad_clip: float
    minibatches: int
    ppo_epochs: int
    eval_min_games: int
    eval_min_ci_lower: float


@dataclass
class OrchestratorState:
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
    # Write the orchestrator's own real PID so the harness reads the true
    # process regardless of launch wrapper. `nohup npm run ... &` captures the
    # npm wrapper PID (python is a grandchild); pid.txt is the source of truth.
    (out_dir / "pid.txt").write_text(f"{os.getpid()}\n")
    state = OrchestratorState()

    # Pre-promote the warm-start checkpoint so iteration 0 has a parent to
    # collect against. ``--init-from`` is mandatory: PPO cannot bootstrap
    # without a behavior policy good enough to produce non-degenerate
    # rollouts.
    init_from = Path(args.init_from).resolve()
    if not init_from.exists():
        raise SystemExit(f"--init-from checkpoint {init_from} does not exist")
    # Stage the warm-start as iter--1 / parent so the iter-0 update has a
    # well-defined parent path on disk.
    parent_dir = out_dir / "iter-parent" / "model"
    parent_dir.mkdir(parents=True, exist_ok=True)
    parent_checkpoint = parent_dir / "checkpoint.pt"
    if not parent_checkpoint.exists():
        shutil.copy2(init_from, parent_checkpoint)
    state.promoted_checkpoint = parent_checkpoint
    state.promoted_wilson_lower = args.eval_min_ci_lower

    events_path = Path(args.events_out) if args.events_out else (out_dir / "events.jsonl")
    events = EventWriter(events_path)
    events.emit_run(
        stage="orchestrator",
        event_type="run_started",
        iterations=args.iterations,
        games_per_update=args.games_per_update,
        init_from=str(init_from),
        temperature=args.temperature,
    )

    base_config = PPOConfig(
        iteration=0,
        games_per_update=args.games_per_update,
        max_steps=args.max_steps,
        temperature=args.temperature,
        lr=args.lr,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        gae_lambda=args.gae_lambda,
        gae_gamma=args.gae_gamma,
        grad_clip=args.grad_clip,
        minibatches=args.minibatches,
        ppo_epochs=args.ppo_epochs,
        eval_min_games=args.eval_games,
        eval_min_ci_lower=args.eval_min_ci_lower,
    )

    pool_path = out_dir / "opponent-pool.json"
    pool = OpponentPool.from_json(pool_path)

    for iteration in range(args.iterations):
        if state.halted:
            events.emit_run(
                stage="orchestrator",
                event_type="halted_before_iteration",
                iteration_skipped=iteration,
                halt_reason=state.halt_reason,
            )
            break
        cfg = PPOConfig(**{**base_config.__dict__, "iteration": iteration})
        record = run_iteration(repo_root, out_dir, cfg, state, pool, args, events)
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


def run_iteration(
    repo_root: Path,
    root_dir: Path,
    cfg: PPOConfig,
    state: OrchestratorState,
    pool: OpponentPool,
    args: argparse.Namespace,
    events: EventWriter,
) -> dict[str, Any]:
    iter_dir = root_dir / f"iter-{cfg.iteration:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ppo-orchestrator] === iteration {cfg.iteration} ===")
    events.emit(
        iteration=cfg.iteration,
        stage="iteration",
        event_type="started",
        games_per_update=cfg.games_per_update,
        parent_promoted=str(state.promoted_checkpoint) if state.promoted_checkpoint else None,
    )

    # 1. Rollout: stand up serve_onnx in stochastic mode, run the TS
    #    evaluator with --selection policy, capture trace.jsonl.
    trace_path = iter_dir / "trace.jsonl"
    trace_manifest = iter_dir / "trace.manifest.json"
    seed_start = args.trace_seed_start + cfg.iteration * cfg.games_per_update
    events.emit(
        iteration=cfg.iteration,
        stage="rollout",
        event_type="started",
        seed_start=seed_start,
        games=cfg.games_per_update,
        temperature=cfg.temperature,
    )
    # R5: sample opponent from pool if --rollout-vs-pool is set.
    opponent_checkpoint = None
    if args.rollout_vs_pool:
        opponent_checkpoint = _select_pool_opponent(args, state)
        if opponent_checkpoint is not None:
            events.emit(
                iteration=cfg.iteration,
                stage="rollout",
                event_type="opponent_selected",
                opponent_path=str(opponent_checkpoint),
            )
    rollout_with_stochastic_serve(
        repo_root,
        iter_dir,
        state.promoted_checkpoint,  # type: ignore[arg-type]
        cfg=cfg,
        seed_start=seed_start,
        trace_path=trace_path,
        manifest_out=trace_manifest,
        args=args,
        opponent_checkpoint=opponent_checkpoint,
    )
    events.emit(
        iteration=cfg.iteration,
        stage="rollout",
        event_type="completed",
        rows=count_lines(trace_path),
    )

    # 2. Parse trace -> PPO format.
    trajectories_path = iter_dir / "trajectories.jsonl"
    events.emit(iteration=cfg.iteration, stage="trajectory-parse", event_type="started")
    # R15.S3: linear-decay schedule for the five new shaped reward signals.
    # iter-0 -> reward_shape_start (default 0.0 = off), iter-last -> reward_shape_end
    # (default 0.0). Default-off keeps prior phase H/J/etc. numerics identical.
    if args.iterations <= 1:
        shape_scale = args.reward_shape_start
    else:
        frac = cfg.iteration / max(1, args.iterations - 1)
        shape_scale = args.reward_shape_start + frac * (args.reward_shape_end - args.reward_shape_start)
    shaping_coefs = {
        "active_energy": args.reward_active_energy_coef * shape_scale,
        "bench_energy": args.reward_bench_energy_coef * shape_scale,
        "retreat": args.reward_retreat_coef * shape_scale,
        "throughput": args.reward_throughput_coef * shape_scale,
        "hp_diff": args.reward_hp_diff_coef * shape_scale,
        "value_head": args.reward_value_head_coef * shape_scale,
    }
    n_episodes, n_transitions, shape_attribution = parse_trace_to_trajectories(
        trace_path,
        trajectories_path,
        alpha=args.reward_alpha,
        beta=args.reward_beta,
        shape_coefs=shaping_coefs,
    )
    events.emit(
        iteration=cfg.iteration,
        stage="trajectory-parse",
        event_type="completed",
        n_episodes=n_episodes,
        n_transitions=n_transitions,
        shape_scale=shape_scale,
        shape_attribution=shape_attribution,
    )

    # 3. PPO update.
    train_dir = iter_dir / "model"
    events.emit(
        iteration=cfg.iteration,
        stage="ppo-update",
        event_type="invocation",
        trajectories=str(trajectories_path),
        out_dir=str(train_dir),
    )
    train_args = [
        sys.executable,
        str(repo_root / "training" / "train_ppo.py"),
        "--init-from-checkpoint",
        str(state.promoted_checkpoint),
        "--trajectories",
        str(trajectories_path),
        "--out-dir",
        str(train_dir),
        "--lr",
        str(cfg.lr),
        "--clip-epsilon",
        str(cfg.clip_epsilon),
        "--entropy-coef",
        str(cfg.entropy_coef),
        "--value-coef",
        str(cfg.value_coef),
        "--grad-clip",
        str(cfg.grad_clip),
        "--gae-lambda",
        str(cfg.gae_lambda),
        "--gae-gamma",
        str(cfg.gae_gamma),
        "--ppo-epochs",
        str(cfg.ppo_epochs),
        "--minibatches",
        str(cfg.minibatches),
        "--events-out",
        str(events.path),
        "--events-iteration",
        str(cfg.iteration),
    ]
    subprocess.run(train_args, cwd=repo_root, check=True)
    train_manifest = json.loads((train_dir / "manifest.json").read_text(encoding="utf8"))

    # 4. Gate: export ONNX, stand up greedy serve_onnx, run eval gate.
    eval_seed_start = args.eval_seed_start + cfg.iteration * cfg.eval_min_games
    eval_manifest_path = iter_dir / "gate.manifest.json"
    events.emit(
        iteration=cfg.iteration,
        stage="gate",
        event_type="started",
        mode="baseline" if args.skip_policy_gate else "policy",
        min_games=2 * cfg.eval_min_games,
    )
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
            rollout_crn_samples=1,
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
    events.emit(
        iteration=cfg.iteration,
        stage="gate",
        event_type="completed",
        returncode=gate_returncode,
        wilson_lower=wilson_lower,
        win_rate=summary.get("modelWinRate"),
        games=eval_n,
    )

    # 5. Pool matchup eval.
    pool_eval_results: list[dict[str, Any]] = []
    pool_aggregate_wilson_lower: float | None = None
    cycling_alarm_iterations: list[int] = []
    if (
        not args.skip_policy_gate
        and args.pool_eval_games > 0
        and len(pool.entries) > 0
        and (train_dir / "checkpoint.pt").exists()
    ):
        events.emit(
            iteration=cfg.iteration,
            stage="pool-eval",
            event_type="started",
            n_opponents=len(pool.entries),
            games_each=args.pool_eval_games,
        )
        # The DAgger helper's PPOConfig-shaped expectations are minimal:
        # ``cfg.rollout_steps`` and ``cfg.rollout_crn_samples``. We supply
        # a lightweight shim so we don't have to re-implement the helper.
        class _PoolCfg:
            iteration = cfg.iteration
            rollout_steps = args.rollout_steps
            rollout_crn_samples = 1

        for opponent_entry in list(pool.entries):
            try:
                result = _dagger_pool_matchup_eval(
                    repo_root,
                    iter_dir,
                    train_dir / "checkpoint.pt",
                    opponent_entry,
                    cfg=_PoolCfg(),
                    args=args,
                )
                pool_eval_results.append(result)
                events.emit(
                    iteration=cfg.iteration,
                    stage="pool-eval",
                    event_type="matchup",
                    opponent_iteration=result.get("opponent_iteration"),
                    wilson_lower=result.get("wilson_lower"),
                    win_rate=result.get("win_rate"),
                    n_games=result.get("n_games"),
                )
            except Exception as exc:  # pragma: no cover - logged only
                pool_eval_results.append({
                    "opponent_iteration": opponent_entry.iteration,
                    "wilson_lower": 0.0,
                    "win_rate": 0.0,
                    "n_games": 0,
                    "error": str(exc),
                })
                events.emit(
                    iteration=cfg.iteration,
                    stage="pool-eval",
                    event_type="matchup_error",
                    opponent_iteration=opponent_entry.iteration,
                    error=str(exc),
                )
        if pool_eval_results:
            valid = [row for row in pool_eval_results if row.get("n_games", 0) > 0]
            total_n = sum(row["n_games"] for row in valid)
            total_wins = sum(row["win_rate"] * row["n_games"] for row in valid)
            if total_n > 0:
                pool_aggregate_wilson_lower = wilson_lower_bound(int(round(total_wins)), total_n)
        history = build_per_opponent_history(state, pool_eval_results, current_iteration=cfg.iteration)
        cycling_alarm_iterations = OpponentPool.cycling_alarm(history)

    # 6. Per-matchup floor + promote/reject + halt-after-2.
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
        "trace_rows": count_lines(trace_path),
        "trajectory_rows": count_lines(trajectories_path),
        "train_manifest": str(train_dir / "manifest.json"),
        "eval_manifest": str(eval_manifest_path),
        "wilson_lower": wilson_lower,
        "promoted": decision["promote"],
        "decision_reason": decision["reason"],
        "gate_returncode": gate_returncode,
        "pool_evals": pool_eval_results,
        "pool_aggregate_wilson_lower": pool_aggregate_wilson_lower,
        "cycling_alarm": cycling_alarm_iterations,
        "matchup_floor_violations": matchup_violations,
        "consecutive_failures": state.consecutive_failures,
        "halted": state.halted,
        "halt_reason": state.halt_reason,
    }
    if decision["promote"]:
        snapshot_path = root_dir / "pool" / f"iter-{cfg.iteration:03d}" / "checkpoint.pt"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(train_dir / "checkpoint.pt", snapshot_path)
        pool.add(PoolEntry(
            iteration=cfg.iteration,
            checkpoint_path=str(snapshot_path),
            wilson_lower_at_promotion=wilson_lower,
            value_mean_drift_from_prior=0.0,
        ))
        pool.retain()

    (iter_dir / "iteration-manifest.json").write_text(
        json.dumps({
            "iteration": cfg.iteration,
            "config": cfg.__dict__,
            "parent_checkpoint": str(state.promoted_checkpoint) if state.promoted_checkpoint else None,
            "result": record,
            "train_manifest": train_manifest,
        }, indent=2) + "\n",
        encoding="utf8",
    )
    events.emit(
        iteration=cfg.iteration,
        stage="iteration",
        event_type="completed",
        promoted=decision["promote"],
        wilson_lower=wilson_lower,
    )
    return record


def rollout_with_stochastic_serve(
    repo_root: Path,
    iter_dir: Path,
    checkpoint: Path,
    *,
    cfg: PPOConfig,
    seed_start: int,
    trace_path: Path,
    manifest_out: Path,
    args: argparse.Namespace,
    opponent_checkpoint: Path | None = None,
) -> None:
    """Stand up serve_onnx with stochastic-by-default, collect a trace.

    R5: when ``opponent_checkpoint`` is set, a second serve_onnx
    instance is brought up in greedy mode for the opponent side. The
    TS evaluator's ``--opponent-model-url`` reaches that second server
    so PPO rollouts are model-vs-model instead of model-vs-rule-bot.
    """

    onnx_path = iter_dir / "rollout.onnx"
    export_checkpoint_to_onnx(repo_root, checkpoint, onnx_path)
    if opponent_checkpoint is not None:
        opp_onnx = iter_dir / "opponent.onnx"
        export_checkpoint_to_onnx(repo_root, opponent_checkpoint, opp_onnx)
    with stochastic_serve_onnx_context(repo_root, onnx_path, cfg.temperature, args) as model_url:
        if opponent_checkpoint is not None:
            with _greedy_serve_onnx_context(repo_root, opp_onnx, args) as opp_url:
                run_evaluator(
                    repo_root,
                    selection="policy",
                    games=cfg.games_per_update,
                    seed_start=seed_start,
                    model_side="both",
                    max_steps=cfg.max_steps,
                    rollout_steps=args.rollout_steps,
                    decision_trace_out=trace_path,
                    trace_teacher="none",
                    rollout_crn_samples=1,
                    model_url=model_url,
                    opponent_model_url=opp_url,
                    manifest_out=manifest_out,
                )
        else:
            run_evaluator(
                repo_root,
                selection="policy",
                games=cfg.games_per_update,
                seed_start=seed_start,
                model_side="both",
                max_steps=cfg.max_steps,
                rollout_steps=args.rollout_steps,
                decision_trace_out=trace_path,
                trace_teacher="none",
                rollout_crn_samples=1,
                model_url=model_url,
                manifest_out=manifest_out,
            )


def run_policy_gate_with_serve(
    repo_root: Path,
    iter_dir: Path,
    checkpoint: Path,
    *,
    cfg: PPOConfig,
    eval_seed_start: int,
    manifest_out: Path,
    args: argparse.Namespace,
) -> int:
    onnx_path = iter_dir / "policy.gate.onnx"
    export_checkpoint_to_onnx(repo_root, checkpoint, onnx_path)
    with _greedy_serve_onnx_context(repo_root, onnx_path, args) as model_url:
        return run_eval_gate(
            repo_root,
            selection="policy",
            games=cfg.eval_min_games,
            seed_start=eval_seed_start,
            min_games=2 * cfg.eval_min_games,
            min_ci_lower=cfg.eval_min_ci_lower,
            manifest_out=manifest_out,
            require_zero_no_ops=True,
            rollout_crn_samples=1,
            extra=["--model-url", model_url],
        )


@contextlib.contextmanager
def stochastic_serve_onnx_context(
    repo_root: Path,
    onnx_path: Path,
    temperature: float,
    args: argparse.Namespace,
) -> Iterator[str]:
    """Variant of dagger_orchestrator.serve_onnx_context with stochastic-by-default.

    The TS evaluator's chooseModelAction POSTs ``{observation, legalActions}``
    without a ``sampling`` field. By starting serve_onnx with
    ``--default-sampling stochastic --default-temperature T`` we ensure
    every collection-side call gets stochastic Gumbel-max sampling and a
    finite behavior_logp on every row.
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
            "--default-sampling",
            "stochastic",
            "--default-temperature",
            str(temperature),
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


# ---------------------------------------------------------------------------
# trace -> trajectory parser
# ---------------------------------------------------------------------------


def _bench_energy_total(own: dict[str, Any]) -> float:
    """Sum of ``energyTotal`` across non-empty bench slots."""
    total = 0.0
    for slot in own.get("bench") or []:
        if slot is None:
            continue
        total += float(slot.get("energyTotal", 0) or 0)
    return total


def _hp_ratio(active: dict[str, Any] | None) -> float:
    if not active:
        return 0.0
    max_hp = float(active.get("maxHp", 0) or 0)
    if max_hp <= 0:
        return 0.0
    return float(active.get("hp", 0) or 0) / max_hp


def parse_trace_to_trajectories(
    trace_path: Path,
    out_path: Path,
    *,
    alpha: float,
    beta: float,
    shape_coefs: dict[str, float] | None = None,
) -> tuple[int, int, dict[str, float]]:
    """Convert TS evaluator trace.jsonl rows into PPO trajectory rows.

    Episode = one game = (seed, modelSide). Per-step reward = α·Δpoints
    between successive model-side decisions for the model's side (from the
    ``observation.own.points`` and ``observation.opponent.points`` already
    persisted on each row). Terminal reward = β·win_indicator added to
    the last decision row of the episode (``done=True``). Rows whose
    ``selection`` is not ``policy`` are skipped — they have no behavior
    log-prob to use as the importance-sampling denominator.

    R15.S3 reward shaping. When ``shape_coefs`` is provided, the per-step
    reward additionally receives shaped signals derived from already-
    traced ``PublicObservation`` fields (no sim-side change) plus, for
    Phase O, the policy's own value-head estimate persisted on
    ``behaviorPolicy.valueEstimate``:

    - ``active_energy``: Δ ``own.active.energyTotal``
    - ``bench_energy``: Δ Σ bench[i].energyTotal (non-empty slots)
    - ``retreat``: +1 on the row that flips ``own.usedRetreatThisTurn`` to true
    - ``throughput``: Δ (handCount + len(discard))
    - ``hp_diff``: Δ (own.active.hp/maxHp − opp.active.hp/maxHp)
    - ``value_head``: Δ ``behaviorPolicy.valueEstimate`` (Phase O); 0 on
      the terminal row (no fictional s_{t+1}); 0 on any row where the
      field is missing (backward compat with pre-Phase-O traces).

    Each ``shape_coefs[key]`` is multiplied by its delta and added to the
    step reward. Defaults of 0.0 (omitted dict) reproduce the prior
    α·Δpoints + β·win reward exactly.

    Returns (n_episodes, n_transitions, shape_attribution) where
    ``shape_attribution`` is the total signed contribution of each
    component summed across all written rows (orchestrator emits it as a
    diagnostic for the R15.S3 falsification split).
    """

    coefs = shape_coefs or {}
    attribution = {k: 0.0 for k in ("active_energy", "bench_energy", "retreat", "throughput", "hp_diff", "value_head")}

    # Group rows by episode (seed + modelSide) and order by step.
    by_episode: dict[str, list[dict[str, Any]]] = {}
    with trace_path.open("r", encoding="utf8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("selection") != "policy":
                continue
            # Only the model side's own decisions are PPO time steps.
            if row.get("sideId") != row.get("modelSide"):
                continue
            episode_id = f"{row.get('seed', '?')}::{row.get('modelSide', '?')}"
            by_episode.setdefault(episode_id, []).append(row)

    n_transitions = 0
    n_episodes = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf8") as out:
        for episode_id, rows in by_episode.items():
            rows.sort(key=lambda r: int(r.get("step", 0)))
            if not rows:
                continue
            n_episodes += 1
            prev_own = 0.0
            prev_opp = 0.0
            # R15.S3 shaping state — per-episode previous values.
            prev_active_energy = 0.0
            prev_bench_energy = 0.0
            prev_retreat = False
            prev_throughput = 0.0
            prev_hp_diff = 0.0
            for step_idx, row in enumerate(rows):
                observation = row.get("observation", {})
                own = observation.get("own", {}) or {}
                opp = observation.get("opponent", {}) or {}
                own_points = float(own.get("points", 0))
                opp_points = float(opp.get("points", 0))
                # Δpoints in own's frame between successive model-side
                # decisions. step_idx=0 starts from (0,0) — the model's
                # first decision sees points 0/0 in a fresh game.
                delta = (own_points - opp_points) - (prev_own - prev_opp)
                reward = alpha * delta
                prev_own, prev_opp = own_points, opp_points

                # R15.S3 shaped components. Each delta is computed first;
                # the coefficient (which may be 0) decides whether it
                # contributes to ``reward``. Attribution sums the signed
                # *contribution* (coef·delta), so an all-zero coef dict
                # yields all-zero attribution.
                own_active = own.get("active") or {}
                opp_active = opp.get("active") or {}
                active_energy = float(own_active.get("energyTotal", 0) or 0)
                bench_energy = _bench_energy_total(own)
                retreat_now = bool(own.get("usedRetreatThisTurn", False))
                throughput = float(own.get("handCount", 0) or 0) + float(len(own.get("discard") or []))
                hp_diff = _hp_ratio(own_active) - _hp_ratio(opp_active)
                if step_idx > 0:
                    d_active = active_energy - prev_active_energy
                    d_bench = bench_energy - prev_bench_energy
                    retreat_event = 1.0 if (retreat_now and not prev_retreat) else 0.0
                    d_throughput = throughput - prev_throughput
                    d_hp = hp_diff - prev_hp_diff
                    components = {
                        "active_energy": coefs.get("active_energy", 0.0) * d_active,
                        "bench_energy": coefs.get("bench_energy", 0.0) * d_bench,
                        "retreat": coefs.get("retreat", 0.0) * retreat_event,
                        "throughput": coefs.get("throughput", 0.0) * d_throughput,
                        "hp_diff": coefs.get("hp_diff", 0.0) * d_hp,
                    }
                    for k, v in components.items():
                        reward += v
                        attribution[k] += v
                prev_active_energy = active_energy
                prev_bench_energy = bench_energy
                prev_retreat = retreat_now
                prev_throughput = throughput
                prev_hp_diff = hp_diff

                # R15.S3 Phase O: value-head tempo signal — shape(t) =
                # coef · (v_{t+1} − v_t) on non-terminal rows; 0 on the
                # terminal row (no fictional s_{t+1}) and 0 on any row
                # where either v_t or v_{t+1} is missing from the
                # behaviorPolicy snapshot (backward compat with pre-Phase-O
                # traces and with single-action rows). Forward delta is
                # specified by the Phase O scoping rather than the
                # observation-delta family's backward-delta convention.
                value_head_coef = coefs.get("value_head", 0.0)
                if value_head_coef != 0.0 and step_idx < len(rows) - 1:
                    cur_behavior = row.get("behaviorPolicy") or {}
                    next_behavior = rows[step_idx + 1].get("behaviorPolicy") or {}
                    v_t = cur_behavior.get("valueEstimate")
                    v_next = next_behavior.get("valueEstimate")
                    if isinstance(v_t, (int, float)) and isinstance(v_next, (int, float)):
                        d_value = float(v_next) - float(v_t)
                        contribution = value_head_coef * d_value
                        reward += contribution
                        attribution["value_head"] += contribution

                done = step_idx == len(rows) - 1
                if done:
                    result = row.get("result") or {}
                    winner = result.get("winner")
                    model_side = row.get("modelSide")
                    if winner == model_side:
                        win_indicator = 1.0
                    elif winner is None:
                        win_indicator = 0.0
                    else:
                        win_indicator = -1.0
                    reward = reward + beta * win_indicator

                behavior = row.get("behaviorPolicy") or {}
                # selectedLogProb is the canonical "behavior_logp" the
                # importance ratio's denominator wants. Defensive default to
                # 0.0 (= log 1, ratio 1 at update time, neutral) when the
                # field is missing on a single-action row.
                behavior_logp = float(behavior.get("selectedLogProb") or 0.0)

                trajectory_row = {
                    "observation": observation,
                    "legalActions": row.get("legalActions", []),
                    "selected_idx": int(row.get("selectedActionIndex", 0)),
                    "behavior_logp": behavior_logp,
                    "reward": float(reward),
                    "done": bool(done),
                    "value_pred": 0.0,
                    "episode_id": episode_id,
                    "episode_step": step_idx,
                }
                out.write(json.dumps(trajectory_row) + "\n")
                n_transitions += 1
    return n_episodes, n_transitions, attribution


# ---------------------------------------------------------------------------
# state persistence
# ---------------------------------------------------------------------------


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--init-from", required=True,
                        help="Path to a DAgger-produced checkpoint.pt to warm-start F1 from.")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--games-per-update", type=int, default=10,
                        help="Games per PPO update. Smoke: 10. Prod: 800 (docs/archive/ai-research/f1-design.md).")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--rollout-steps", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature for stochastic rollouts.")
    # PPO hyperparams; defaults are the docs/archive/ai-research/f1-design.md priors.
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--gae-gamma", type=float, default=0.99)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--minibatches", type=int, default=4)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    # Reward shaping coefficients.
    parser.add_argument("--reward-alpha", type=float, default=1.0 / 3.0,
                        help="Per-step Δpoints shaping coefficient.")
    parser.add_argument("--reward-beta", type=float, default=1.0,
                        help="Terminal win-indicator coefficient.")
    # R15.S3: five additional shaped reward signals (orchestrator-only,
    # sourced from already-traced PublicObservation fields). All default
    # to 0.0 → identical numerics to phase H/J without these args.
    # The schedule scaler is iter-linear from --reward-shape-start (iter-0)
    # to --reward-shape-end (final iter). Both default 0.0 → shaping
    # is fully off unless the caller opts in.
    parser.add_argument("--reward-active-energy-coef", type=float, default=0.0,
                        help="R15.S3: Δ own.active.energyTotal per-step coef (recommended 0.02).")
    parser.add_argument("--reward-bench-energy-coef", type=float, default=0.0,
                        help="R15.S3: Δ Σ bench energyTotal per-step coef (recommended 0.02).")
    parser.add_argument("--reward-retreat-coef", type=float, default=0.0,
                        help="R15.S3: retreat-event indicator coef (recommended 0.03).")
    parser.add_argument("--reward-throughput-coef", type=float, default=0.0,
                        help="R15.S3: Δ (handCount + len(discard)) coef (recommended 0.02).")
    parser.add_argument("--reward-hp-diff-coef", type=float, default=0.0,
                        help="R15.S3: Δ (own.active.hp/maxHp − opp.active.hp/maxHp) coef (recommended 0.05).")
    parser.add_argument("--reward-value-head-coef", type=float, default=0.0,
                        help="R15.S3 Phase O: Δ behaviorPolicy.valueEstimate per-step coef "
                             "(value(s_{t+1}) − value(s_t); zero on terminal). Recommended 0.05.")
    parser.add_argument("--reward-shape-start", type=float, default=0.0,
                        help="R15.S3: schedule scale at iter-0 (1.0 = full coefs, 0.0 = off).")
    parser.add_argument("--reward-shape-end", type=float, default=0.0,
                        help="R15.S3: schedule scale at final iter (linear decay between start/end).")
    # Eval / gate / pool.
    parser.add_argument("--eval-games", type=int, default=10)
    parser.add_argument("--eval-min-ci-lower", type=float, default=0.0)
    parser.add_argument("--trace-seed-start", type=int, default=14000)
    parser.add_argument("--eval-seed-start", type=int, default=9000)
    parser.add_argument("--skip-policy-gate", action="store_true",
                        help="Use selection=baseline for the gate to avoid ONNX server bring-up.")
    parser.add_argument("--pool-eval-games", type=int, default=0,
                        help="Per-pool-member games for matchup eval. 0 disables pool eval (smoke default).")
    parser.add_argument("--per-matchup-drop-tolerance", type=float, default=0.05)
    parser.add_argument("--pool-eval-seed-start", type=int, default=20000)
    parser.add_argument("--events-out", default=None,
                        help="Override default events.jsonl path (otherwise out_dir/events.jsonl).")
    # R5: self-play via PFSP-sampled opponent pool.
    parser.add_argument("--rollout-vs-pool", action="store_true",
                        help="R5: rollout against a PFSP-sampled opponent from --pool-path instead of rule-bot.")
    parser.add_argument("--pool-path", default=None,
                        help="Path to opponent-pool.json (default: out_dir/opponent-pool.json).")
    return parser.parse_args()


def _select_pool_opponent(args: argparse.Namespace, state: OrchestratorState) -> Path | None:
    """R5: select an opponent checkpoint from the pool for this iteration.

    Sampling rule: PFSP weights ``max(0.05, 1 - p_i)`` against per-opponent
    history if available; uniform fallback when the pool has only one
    entry or no history yet. Returns ``None`` if the pool is empty or
    cannot be loaded (caller falls back to rule-bot rollout).
    """

    pool_path = Path(args.pool_path) if args.pool_path else (Path(args.out_dir) / "opponent-pool.json")
    if not pool_path.exists():
        print(f"[ppo-orchestrator] --rollout-vs-pool requested but no pool at {pool_path}; falling back to rule-bot")
        return None
    try:
        from opponent_pool import OpponentPool
        pool = OpponentPool.from_json(pool_path)
    except Exception as exc:
        print(f"[ppo-orchestrator] failed to load opponent pool: {exc}; falling back to rule-bot")
        return None
    if not pool.entries:
        return None
    # Without history we sample uniformly from the pool (PFSP weights
    # need per-opponent win-rate signal we don't have at iter 0).
    import random
    seed = args.trace_seed_start + 9973  # stable per-run, doesn't depend on iter
    rng = random.Random(seed)
    entry = rng.choice(list(pool.entries))
    return Path(entry.checkpoint_path)


if __name__ == "__main__":
    main()
