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
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root or Path(__file__).resolve().parents[1])
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    state = OrchestratorState()

    if args.resume_state:
        state = load_state(Path(args.resume_state))

    base_config = IterationConfig(
        iteration=0,
        games=args.games,
        teacher=args.teacher,
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

    rule_bot_replay = ensure_rule_bot_replay(repo_root, out_dir, args, base_config)

    last_iter = max((entry["iteration"] for entry in state.iterations), default=-1)
    for iteration in range(last_iter + 1, args.iterations):
        cfg = IterationConfig(**{**base_config.__dict__, "iteration": iteration})
        record = run_iteration(repo_root, out_dir, cfg, state, rule_bot_replay, args)
        state.iterations.append(record)
        save_state(out_dir / "orchestrator-state.json", state)

    summary = {
        "iterations": state.iterations,
        "promoted_checkpoint": str(state.promoted_checkpoint) if state.promoted_checkpoint else None,
        "promoted_wilson_lower": state.promoted_wilson_lower,
    }
    print(json.dumps(summary, indent=2))


def ensure_rule_bot_replay(repo_root: Path, out_dir: Path, args: argparse.Namespace, cfg: IterationConfig) -> Path:
    """Always-on rule-bot rehearsal slice.

    The rule-bot replay buffer grounds the loop against the original
    teacher distribution so the model can't drift to a state distribution
    the rule bot never visits.
    """

    replay = out_dir / "rule-bot-replay.jsonl"
    if replay.exists() and replay.stat().st_size > 0 and not args.refresh_rule_bot:
        return replay
    print(f"[orchestrator] generating rule-bot replay ({args.replay_games} games)")
    export_training_examples(
        repo_root,
        replay,
        seed_start=args.replay_seed_start,
        games=args.replay_games,
        max_steps=args.max_steps,
    )
    return replay


def run_iteration(
    repo_root: Path,
    root_dir: Path,
    cfg: IterationConfig,
    state: OrchestratorState,
    rule_bot_replay: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    iter_dir = root_dir / f"iter-{cfg.iteration:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    print(f"[orchestrator] === iteration {cfg.iteration} ===")

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
            trace_teacher=cfg.teacher,
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

    relabel_decision_trace(
        repo_root,
        trace_path,
        relabeled_path,
        source=f"model-visited-{cfg.teacher}-relabeled",
        label_source=f"{cfg.teacher}-teacher",
        manifest_out=relabel_manifest,
    )

    mix_components: list[tuple[str, float, str]] = [
        (str(relabeled_path), cfg.relabeled_weight, f"iter-{cfg.iteration}-relabeled"),
        (str(rule_bot_replay), cfg.rule_bot_replay_weight, "rule-bot-replay"),
    ]
    mix_sources(
        repo_root,
        out=mixed_path,
        seed=f"iter-{cfg.iteration}-mix",
        components=mix_components,
        manifest_out=mix_manifest,
    )

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
        train_args += ["--resume", str(state.promoted_checkpoint)]
    subprocess.run(train_args, cwd=repo_root, check=True)
    train_manifest = json.loads((train_dir / "manifest.json").read_text(encoding="utf8"))

    # The orchestrator needs an ONNX-served model to evaluate strength
    # closed-loop. The full server bring-up is heavy; for the smoke we
    # gate-check by running the eval gate in selection=baseline mode if
    # --skip-policy-gate is set, which lets the loop exercise the
    # promote/reject path without standing up serve_onnx.
    eval_manifest_path = iter_dir / "gate.manifest.json"
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

    decision = decide_promotion(state, gate_returncode, wilson_lower, args)
    if decision["promote"]:
        state.promoted_checkpoint = train_dir / "checkpoint.pt"
        state.promoted_wilson_lower = wilson_lower
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
    }
    (iter_dir / "iteration-manifest.json").write_text(json.dumps({
        "iteration": cfg.iteration,
        "config": cfg.__dict__,
        "parent_checkpoint": str(state.promoted_checkpoint) if state.promoted_checkpoint else None,
        "previous_promoted_wilson_lower": state.promoted_wilson_lower,
        "result": record,
        "train_manifest": train_manifest,
    }, indent=2) + "\n", encoding="utf8")
    print(f"[orchestrator] iteration {cfg.iteration} {'promoted' if decision['promote'] else 'rejected'} ({decision['reason']})")
    return record


def decide_promotion(state: OrchestratorState, gate_returncode: int, wilson_lower: float, args: argparse.Namespace) -> dict[str, Any]:
    if gate_returncode != 0:
        return {"promote": False, "reason": f"gate exited {gate_returncode}"}
    floor = state.promoted_wilson_lower if state.promoted_wilson_lower is not None else args.eval_min_ci_lower
    if wilson_lower + 1e-9 < floor:
        return {"promote": False, "reason": f"wilson_lower {wilson_lower:.4f} < floor {floor:.4f}"}
    return {"promote": True, "reason": f"wilson_lower {wilson_lower:.4f} >= floor {floor:.4f}"}


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
    subprocess.run([
        sys.executable,
        str(repo_root / "training" / "export_onnx.py"),
        "--checkpoint",
        str(checkpoint),
        "--out",
        str(onnx_path),
    ], cwd=repo_root, check=True)
    server_proc, model_url = start_serve_onnx(repo_root, onnx_path, args)
    try:
        run_evaluator(
            repo_root,
            selection="policy",
            games=cfg.games,
            seed_start=seed_start,
            model_side="both",
            max_steps=args.max_steps,
            rollout_steps=cfg.rollout_steps,
            decision_trace_out=trace_path,
            trace_teacher=cfg.teacher,
            rollout_crn_samples=cfg.rollout_crn_samples,
            model_url=model_url,
            manifest_out=manifest_out,
        )
    finally:
        stop_serve_onnx(server_proc)


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
    subprocess.run([
        sys.executable,
        str(repo_root / "training" / "export_onnx.py"),
        "--checkpoint",
        str(checkpoint),
        "--out",
        str(onnx_path),
    ], cwd=repo_root, check=True)
    server_proc, model_url = start_serve_onnx(repo_root, onnx_path, args)
    try:
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
    finally:
        stop_serve_onnx(server_proc)


def start_serve_onnx(repo_root: Path, onnx_path: Path, args: argparse.Namespace):
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
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
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
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--teacher", choices=["rollout", "search", "planner"], default="rollout")
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
    parser.add_argument("--resume-state", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
