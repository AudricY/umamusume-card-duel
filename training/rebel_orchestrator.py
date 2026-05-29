from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from events import EventWriter


@dataclass
class RebelLoopState:
    promoted_checkpoint: str | None = None
    promoted_onnx: str | None = None
    promoted_wilson_lower: float | None = None
    kl_anchor_checkpoint: str | None = None
    iterations: list[dict[str, Any]] = field(default_factory=list)
    consecutive_failures: int = 0
    halted: bool = False
    halt_reason: str | None = None


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    validate_leaf_args(args)
    resolve_runtime_devices(args, repo)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.use_release_binary and args.iterations <= 1:
        preflight_release_binaries(repo)
    if args.iterations > 1:
        run_loop(args, repo, out_dir)
        return
    data_path = out_dir / "rebel-selfplay.jsonl"
    selfplay_manifest = out_dir / "selfplay.manifest.json"
    train_dir = out_dir / "train"
    export_path = out_dir / "policy.onnx"

    run(build_selfplay_cmd(args, repo, data_path, selfplay_manifest), cwd=repo / "engine-rs", env=ort_env(repo))
    row_summary = validate_rebel_rows(data_path)

    train_cmd = build_train_cmd(args, repo, data_path, train_dir)
    run(train_cmd, cwd=repo)

    export_status: dict[str, Any] = {"status": "skipped", "reason": "--skip-export"}
    if not args.skip_export:
        run(
            [
                training_python(repo),
                str(repo / "training" / "export_onnx.py"),
                "--checkpoint",
                str(train_dir / "checkpoint.pt"),
                "--out",
                str(export_path),
            ],
            cwd=repo,
        )
        export_status = {"status": "ok", "onnx": str(export_path)}

    gates = run_gates(args, repo, out_dir, export_path) if not args.skip_gates and not args.skip_export else {
        "fixed": {"status": "not-run", "reason": "--skip-gates or --skip-export"},
        "uniform_deck_diverse": {"status": "not-run", "reason": "--skip-gates or --skip-export"},
        "side_split": {"status": "recorded-in-row-fields", "field": "sideId"},
    }

    training_settings = effective_training_settings(args)
    manifest = {
        "name": "R17-rebel-e2e",
        "status": "complete",
        "data_mode": "rebel",
        "engine": "rust",
        "selfplay_binary": "sim-rebel-selfplay",
        "search": "public-belief-cfr-v1",
        "deck_sampling": args.deck_sampling,
        "model_side": args.model_side,
        "artifacts": {
            "selfplay": str(data_path),
            "selfplay_manifest": str(selfplay_manifest),
            "train_dir": str(train_dir),
            "export": export_status,
        },
        "row_summary": row_summary,
        "knobs": {
            "games": args.games,
            "seed_start": args.seed_start,
            "particles": args.particles,
            "search_iterations": args.search_iterations,
            "rollout_steps": args.rollout_steps,
            "max_steps": args.max_steps,
            "workers": args.workers,
            "epochs": args.epochs,
            **training_settings,
            "state_dim": args.state_dim,
            "q_value_head": args.q_value_head,
            "q_value_weight": args.q_value_weight,
            "kl_anchor_weight": args.kl_anchor_weight,
            "entropy_bonus": args.entropy_bonus,
            "use_release_binary": args.use_release_binary,
            "selfplay_onnx_path": args.selfplay_onnx_path,
            "selfplay_device": effective_selfplay_device(args),
            "neural_policy_weight": args.neural_policy_weight,
            "neural_value_weight": args.neural_value_weight,
            "neural_leaf_weight": args.neural_leaf_weight,
            "allow_rollout_leaf": args.allow_rollout_leaf,
            "selfplay_inference_batch_size": args.selfplay_inference_batch_size,
            "selfplay_inference_max_wait_us": args.selfplay_inference_max_wait_us,
        },
        "gates": gates,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    print(json.dumps(manifest, indent=2))


def run_loop(args: argparse.Namespace, repo: Path, out_dir: Path) -> None:
    loop_dir = out_dir / "loop"
    loop_dir.mkdir(parents=True, exist_ok=True)
    events = EventWriter(loop_dir)
    if args.use_release_binary:
        preflight_release_binaries(repo, events)
    state = load_loop_state(loop_dir / "orchestrator-state.json")
    if state.promoted_checkpoint is None and args.init_from_checkpoint:
        state.promoted_checkpoint = str(Path(args.init_from_checkpoint).resolve())
    if state.promoted_onnx is None and args.selfplay_onnx_path:
        state.promoted_onnx = str(Path(args.selfplay_onnx_path).resolve())
    if state.kl_anchor_checkpoint is None:
        if args.kl_anchor_checkpoint:
            state.kl_anchor_checkpoint = str(Path(args.kl_anchor_checkpoint).resolve())
        elif args.init_from_checkpoint:
            state.kl_anchor_checkpoint = str(Path(args.init_from_checkpoint).resolve())
    save_loop_state(loop_dir / "orchestrator-state.json", state)
    events.emit_run(
        stage="rebel-orchestrator",
        event_type="run_started",
        iterations=args.iterations,
        games=args.games,
        deck_sampling=args.deck_sampling,
        model_side=args.model_side,
        cross_iter_replay=args.cross_iter_replay,
        replay_window=args.replay_window,
        replay_old_fraction=args.replay_old_fraction,
        fixed_kl_anchor=args.fixed_kl_anchor,
        device=args.device,
        selfplay_device=effective_selfplay_device(args),
    )

    start_iter = max((int(r["iteration"]) for r in state.iterations), default=-1) + 1
    for iteration in range(start_iter, args.iterations):
        if state.halted:
            events.emit(
                iteration=iteration,
                stage="rebel-orchestrator",
                event_type="halted_before_iteration",
                halt_reason=state.halt_reason,
            )
            break
        record = run_loop_iteration(args, repo, loop_dir, iteration, state, events)
        state.iterations.append(record)
        if record["promote"]:
            record["pool_snapshot"] = snapshot_promoted_artifacts(loop_dir, record)
            (Path(record["dir"]) / "manifest.json").write_text(
                json.dumps(record, indent=2) + "\n",
                encoding="utf8",
            )
            state.promoted_checkpoint = record["checkpoint"]
            state.promoted_onnx = record["onnx"]
            state.promoted_wilson_lower = record["wilson_lower"]
            if state.kl_anchor_checkpoint is None:
                state.kl_anchor_checkpoint = record["checkpoint"]
            state.consecutive_failures = 0
        else:
            state.consecutive_failures += 1
            if args.halt_after_consecutive_failures > 0 and state.consecutive_failures >= args.halt_after_consecutive_failures:
                state.halted = True
                state.halt_reason = f"halt-after-{args.halt_after_consecutive_failures} consecutive promotion failures"
                events.emit(
                    iteration=iteration,
                    stage="rebel-orchestrator",
                    event_type="halted",
                    halt_reason=state.halt_reason,
                )
        save_loop_state(loop_dir / "orchestrator-state.json", state)

    events.emit_run(
        stage="rebel-orchestrator",
        event_type="run_completed",
        halted=state.halted,
        halt_reason=state.halt_reason,
        promoted_checkpoint=state.promoted_checkpoint,
        promoted_onnx=state.promoted_onnx,
        promoted_wilson_lower=state.promoted_wilson_lower,
        kl_anchor_checkpoint=state.kl_anchor_checkpoint,
    )
    final = {
        "name": "R17-rebel-loop",
        "status": "HALTED" if state.halted else "COMPLETED",
        "loop_dir": str(loop_dir),
        "events": str(loop_dir / "events.jsonl"),
        "state": str(loop_dir / "orchestrator-state.json"),
        "promoted_checkpoint": state.promoted_checkpoint,
        "promoted_onnx": state.promoted_onnx,
        "promoted_wilson_lower": state.promoted_wilson_lower,
        "kl_anchor_checkpoint": state.kl_anchor_checkpoint,
        "iterations_run": len(state.iterations),
    }
    (out_dir / "manifest.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf8")
    print(json.dumps(final, indent=2))


def run_loop_iteration(
    args: argparse.Namespace,
    repo: Path,
    loop_dir: Path,
    iteration: int,
    state: RebelLoopState,
    events: EventWriter,
) -> dict[str, Any]:
    iter_dir = loop_dir / f"iter-{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    data_path = iter_dir / "rebel-selfplay.jsonl"
    selfplay_manifest = iter_dir / "selfplay.manifest.json"
    train_dir = iter_dir / "train"
    export_path = iter_dir / "policy.onnx"
    seed_start = args.seed_start + iteration * args.games
    selfplay_onnx = state.promoted_onnx or args.selfplay_onnx_path
    init_checkpoint = state.promoted_checkpoint or args.init_from_checkpoint
    iter_args = argparse.Namespace(**vars(args))
    iter_args.seed_start = seed_start
    iter_args.gate_seed_start = args.gate_seed_start + iteration * args.gate_games * 2
    iter_args.selfplay_onnx_path = selfplay_onnx
    iter_args.init_from_checkpoint = init_checkpoint
    iter_args.kl_anchor_checkpoint = resolve_kl_anchor(args, state, init_checkpoint)
    iter_args.events_out = str(events.path)
    iter_args.events_iteration = iteration

    events.emit(
        iteration=iteration,
        stage="iteration",
        event_type="started",
        selfplay_onnx_path=selfplay_onnx,
        init_checkpoint=init_checkpoint,
        kl_anchor_checkpoint=iter_args.kl_anchor_checkpoint,
    )
    t0 = time.time()
    pool_record = {"selfplay_vs_pool": False}
    events.emit(
        iteration=iteration,
        stage="selfplay",
        event_type="started",
        seed_start=seed_start,
        selfplay_vs_pool=bool(args.selfplay_vs_pool),
    )
    if args.selfplay_vs_pool:
        pool_record = run_pool_selfplay(
            args=iter_args,
            repo=repo,
            loop_dir=loop_dir,
            iter_dir=iter_dir,
            state=state,
            data_path=data_path,
            selfplay_manifest=selfplay_manifest,
            events=events,
        )
    if not pool_record.get("selfplay_vs_pool"):
        run(build_selfplay_cmd(iter_args, repo, data_path, selfplay_manifest), cwd=repo / "engine-rs", env=ort_env(repo))
    row_summary = validate_rebel_rows(data_path)
    selfplay_elapsed = time.time() - t0
    events.emit(
        iteration=iteration,
        stage="selfplay",
        event_type="completed",
        rows=row_summary["rows"],
        elapsed_sec=selfplay_elapsed,
    )

    train_data_path, replay_summary = materialize_replay_mix(
        args=args,
        loop_dir=loop_dir,
        iter_dir=iter_dir,
        iteration=iteration,
        current_selfplay=data_path,
        events=events,
    )
    train_row_summary = validate_rebel_rows(train_data_path)

    t0 = time.time()
    events.emit(
        iteration=iteration,
        stage="train",
        event_type="started",
        init_checkpoint=init_checkpoint,
        kl_anchor_checkpoint=iter_args.kl_anchor_checkpoint,
        data_path=str(train_data_path),
    )
    run(build_train_cmd(iter_args, repo, train_data_path, train_dir), cwd=repo)
    train_elapsed = time.time() - t0
    checkpoint = train_dir / "checkpoint.pt"
    events.emit(
        iteration=iteration,
        stage="train",
        event_type="completed",
        checkpoint=str(checkpoint),
        elapsed_sec=train_elapsed,
    )

    t0 = time.time()
    run(
        [
            training_python(repo),
            str(repo / "training" / "export_onnx.py"),
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(export_path),
        ],
        cwd=repo,
    )
    export_elapsed = time.time() - t0
    events.emit(iteration=iteration, stage="export", event_type="completed", onnx=str(export_path), elapsed_sec=export_elapsed)

    t0 = time.time()
    if args.skip_gates:
        gates = {
            "fixed": {"status": "not-run", "reason": "--skip-gates"},
            "uniform_deck_diverse": {"status": "not-run", "reason": "--skip-gates"},
            "side_split": {"status": "recorded-in-row-fields", "field": "sideId"},
        }
    else:
        gates = run_gates(iter_args, repo, iter_dir, export_path, champion_onnx=state.promoted_onnx)
    gate_elapsed = time.time() - t0
    fixed = gates.get("fixed") or {}
    previous = state.promoted_wilson_lower
    head_to_head = bool(fixed.get("head_to_head"))
    # `wilson_lower` is now a vs-champion number when the fixed gate ran
    # head-to-head (and a vs-heuristic number for the iter-0 bootstrap).
    # Stored on the record and, on promotion, into promoted_wilson_lower.
    wilson_lower = float(fixed.get("wilson_lower") or previous or 0.0)
    # The monotone `wilson_lower >= previous` ratchet is GONE: the fixed
    # gate's `passed` already encodes the right thing — (iter-0) beat the
    # heuristic at --gate-min-ci-lower, OR (later) beat the champion
    # head-to-head at 0.5 + margin.
    gate_passed = bool(fixed.get("passed"))
    promote = args.skip_gates or gate_passed
    if args.skip_gates:
        reason = "skip-gates"
    elif head_to_head:
        reason = "h2h-beat-champion" if promote else "h2h-below-champion"
    else:
        reason = "bootstrap-gate-passed" if promote else "bootstrap-gate-failed"
    record = {
        "iteration": iteration,
        "dir": str(iter_dir),
        "checkpoint": str(checkpoint),
        "onnx": str(export_path),
        "selfplay": str(data_path),
        "selfplay_manifest": str(selfplay_manifest),
        "selfplay_pool": pool_record,
        "train_data": str(train_data_path),
        "row_summary": row_summary,
        "train_row_summary": train_row_summary,
        "replay": replay_summary,
        "neural_leaf_weight": args.neural_leaf_weight,
        "allow_rollout_leaf": args.allow_rollout_leaf,
        "gates": gates,
        "wilson_lower": wilson_lower,
        "previous_wilson_lower": previous,
        "promote": promote,
        "head_to_head": head_to_head,
        "reason": reason,
        "selfplay_elapsed_sec": selfplay_elapsed,
        "train_elapsed_sec": train_elapsed,
        "export_elapsed_sec": export_elapsed,
        "gate_elapsed_sec": gate_elapsed,
    }
    (iter_dir / "manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf8")
    events.emit(iteration=iteration, stage="gate", event_type="completed", **fixed, elapsed_sec=gate_elapsed)
    events.emit(iteration=iteration, stage="decision", event_type="completed", promote=promote, reason=record["reason"])
    return record


def load_loop_state(path: Path) -> RebelLoopState:
    if not path.exists():
        return RebelLoopState()
    payload = json.loads(path.read_text(encoding="utf8"))
    return RebelLoopState(
        promoted_checkpoint=payload.get("promoted_checkpoint"),
        promoted_onnx=payload.get("promoted_onnx"),
        promoted_wilson_lower=payload.get("promoted_wilson_lower"),
        kl_anchor_checkpoint=payload.get("kl_anchor_checkpoint"),
        iterations=list(payload.get("iterations") or []),
        consecutive_failures=int(payload.get("consecutive_failures") or 0),
        halted=bool(payload.get("halted") or False),
        halt_reason=payload.get("halt_reason"),
    )


def save_loop_state(path: Path, state: RebelLoopState) -> None:
    path.write_text(json.dumps(asdict(state), indent=2) + "\n", encoding="utf8")


def preflight_release_binaries(repo: Path, events: EventWriter | None = None) -> None:
    release_dir = repo / "engine-rs" / "target" / "release"
    binaries = [
        release_dir / "sim-rebel-selfplay",
        release_dir / "sim-eval-gate",
    ]
    missing = [path for path in binaries if not path.exists()]
    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        if events is not None:
            events.emit_run(
                stage="rebel-orchestrator",
                event_type="release_binary_preflight_failed",
                reason="missing release binary",
                missing=missing_str,
            )
        raise SystemExit(
            f"Rust release binary missing: {missing_str}. "
            "Build with: (cd engine-rs && cargo build --release -p sim-cli "
            "--bin sim-rebel-selfplay --bin sim-eval-gate)."
        )

    source_paths = [
        repo / "engine-rs" / "crates" / "engine" / "src" / "inference" / "mod.rs",
        repo / "engine-rs" / "crates" / "engine" / "src" / "rebel" / "mod.rs",
        repo / "engine-rs" / "crates" / "sim-cli" / "src" / "bin" / "rebel_selfplay.rs",
        repo / "engine-rs" / "crates" / "sim-cli" / "src" / "bin" / "eval_gate.rs",
    ]
    newest_source = max((path.stat().st_mtime for path in source_paths if path.exists()), default=0.0)
    oldest_binary = min(path.stat().st_mtime for path in binaries)
    if newest_source > oldest_binary:
        if events is not None:
            events.emit_run(
                stage="rebel-orchestrator",
                event_type="release_binary_preflight_failed",
                reason="release binary older than ReBeL/eval Rust source",
                newest_source_mtime=newest_source,
                oldest_binary_mtime=oldest_binary,
            )
        raise SystemExit(
            "Rust release binaries are older than ReBeL/eval Rust sources. "
            "Rebuild with: (cd engine-rs && cargo build --release -p sim-cli "
            "--bin sim-rebel-selfplay --bin sim-eval-gate)."
        )


def snapshot_promoted_artifacts(loop_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    iteration = int(record["iteration"])
    snapshot_dir = loop_dir / "pool" / f"iter-{iteration:03d}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_src = Path(record["checkpoint"])
    onnx_src = Path(record["onnx"])
    checkpoint_dst = snapshot_dir / "checkpoint.pt"
    onnx_dst = snapshot_dir / "policy.onnx"
    shutil.copy2(checkpoint_src, checkpoint_dst)
    shutil.copy2(onnx_src, onnx_dst)

    meta_src = onnx_src.with_suffix(onnx_src.suffix + ".meta.json")
    meta_dst = None
    if meta_src.exists():
        meta_dst = snapshot_dir / meta_src.name
        shutil.copy2(meta_src, meta_dst)

    # ONNX external-data sidecar — required by ONNX Runtime when the export
    # uses external data format. Missing → SIGSEGV in sim-rebel-selfplay.
    data_src = onnx_src.with_suffix(onnx_src.suffix + ".data")
    if data_src.exists():
        shutil.copy2(data_src, snapshot_dir / data_src.name)

    return {
        "iteration": iteration,
        "dir": str(snapshot_dir),
        "checkpoint": str(checkpoint_dst),
        "onnx": str(onnx_dst),
        "onnx_meta": str(meta_dst) if meta_dst else None,
        "wilson_lower": record.get("wilson_lower"),
    }


def load_pool_entries_from_state(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf8"))
    except Exception:
        return []
    entries: list[dict[str, Any]] = []
    for record in payload.get("iterations") or []:
        entry = pool_entry_from_iteration(record, source="state_file")
        if entry is not None:
            entries.append(entry)
    return entries


def pool_entry_from_iteration(record: dict[str, Any], *, source: str) -> dict[str, Any] | None:
    if not record.get("promote"):
        return None
    snapshot = record.get("pool_snapshot") or {}
    onnx = snapshot.get("onnx") or record.get("onnx")
    checkpoint = snapshot.get("checkpoint") or record.get("checkpoint")
    if not onnx or not checkpoint:
        return None
    return {
        "iteration": int(record.get("iteration", -1)),
        "checkpoint": str(Path(checkpoint).resolve()),
        "onnx": str(Path(onnx).resolve()),
        "wilson_lower": float(record.get("wilson_lower") or snapshot.get("wilson_lower") or 0.0),
        "source": source,
    }


def resolve_selfplay_pool(args: argparse.Namespace, state: RebelLoopState) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    if args.pool_state_file:
        for entry in load_pool_entries_from_state(Path(args.pool_state_file)):
            key = entry["onnx"]
            if key not in seen:
                seen.add(key)
                entries.append(entry)
    for record in state.iterations:
        entry = pool_entry_from_iteration(record, source="current_run")
        if entry is None:
            continue
        key = entry["onnx"]
        if key not in seen:
            seen.add(key)
            entries.append(entry)
    entries.sort(key=lambda item: int(item["iteration"]), reverse=True)
    return entries[: max(1, int(args.pool_size))]


def pfsp_weights_from_pool(pool: list[dict[str, Any]], floor: float) -> list[float]:
    if not pool:
        return []
    raw = [max(floor, float(entry.get("wilson_lower") or 0.0)) for entry in pool]
    if sum(raw) <= 0.0:
        return [1.0 / len(pool)] * len(pool)
    if all(weight <= floor + 1.0e-9 for weight in raw):
        return [1.0 / len(pool)] * len(pool)
    total = sum(raw)
    return [weight / total for weight in raw]


def allocate_games(total: int, weights: list[float]) -> list[int]:
    if total <= 0 or not weights:
        return []
    if total < len(weights):
        out = [0] * len(weights)
        for index in sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)[:total]:
            out[index] = 1
        return out
    raw = [weight * total for weight in weights]
    out = [int(value) for value in raw]
    leftover = total - sum(out)
    if leftover > 0:
        order = sorted(range(len(weights)), key=lambda i: (raw[i] - out[i], weights[i]), reverse=True)
        for index in order[:leftover]:
            out[index] += 1
    return out


def run_pool_selfplay(
    *,
    args: argparse.Namespace,
    repo: Path,
    loop_dir: Path,
    iter_dir: Path,
    state: RebelLoopState,
    data_path: Path,
    selfplay_manifest: Path,
    events: EventWriter,
) -> dict[str, Any]:
    pool = resolve_selfplay_pool(args, state)
    if not pool:
        events.emit(
            iteration=args.events_iteration,
            stage="selfplay-pool",
            event_type="fallback_to_single",
            reason="empty_pool",
        )
        return {"selfplay_vs_pool": False, "reason": "empty_pool"}

    weights = pfsp_weights_from_pool(pool, float(args.pfsp_floor))
    games_alloc = allocate_games(int(args.games), weights)
    invocations = [(entry, weight, games) for entry, weight, games in zip(pool, weights, games_alloc) if games > 0]
    if not invocations:
        return {"selfplay_vs_pool": False, "reason": "zero_games_alloc"}

    events.emit(
        iteration=args.events_iteration,
        stage="selfplay-pool",
        event_type="pool_resolved",
        pool=[
            {
                "pool_iter": entry["iteration"],
                "onnx": entry["onnx"],
                "checkpoint": entry["checkpoint"],
                "weight": weight,
                "games": games,
                "source": entry["source"],
            }
            for entry, weight, games in invocations
        ],
    )

    seed_cursor = int(args.seed_start)
    total_rows = 0
    per_invocation: list[dict[str, Any]] = []
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("w", encoding="utf8") as out_fh:
        for entry, weight, games in invocations:
            label = f"iter-{int(entry['iteration']):03d}" if int(entry["iteration"]) >= 0 else "warm"
            part_path = iter_dir / f"rebel-selfplay-vs-{label}.jsonl"
            part_manifest = iter_dir / f"selfplay-vs-{label}.manifest.json"
            part_args = argparse.Namespace(**vars(args))
            part_args.games = games
            part_args.seed_start = seed_cursor
            part_args.selfplay_onnx_path = entry["onnx"]
            events.emit(
                iteration=args.events_iteration,
                stage="selfplay-pool",
                event_type="invocation_started",
                pool_iter=entry["iteration"],
                onnx=entry["onnx"],
                games=games,
                seed_start=seed_cursor,
            )
            run(build_selfplay_cmd(part_args, repo, part_path, part_manifest), cwd=repo / "engine-rs", env=ort_env(repo))
            rows = append_jsonl_with_pool_annotation(part_path, out_fh, entry)
            total_rows += rows
            per_invocation.append(
                {
                    "pool_iter": entry["iteration"],
                    "checkpoint": entry["checkpoint"],
                    "onnx": entry["onnx"],
                    "weight": weight,
                    "games": games,
                    "seed_start": seed_cursor,
                    "rows": rows,
                    "manifest": str(part_manifest),
                }
            )
            events.emit(
                iteration=args.events_iteration,
                stage="selfplay-pool",
                event_type="invocation_completed",
                pool_iter=entry["iteration"],
                rows=rows,
            )
            seed_cursor += games

    manifest = {
        "args": {
            "selfplayVsPool": True,
            "poolSize": args.pool_size,
            "pfspFloor": args.pfsp_floor,
            "seedStart": args.seed_start,
            "games": args.games,
            "out": str(data_path),
            "manifestOut": str(selfplay_manifest),
        },
        "summary": {
            "games": sum(item["games"] for item in per_invocation),
            "rows": total_rows,
            "perPoolInvocations": len(per_invocation),
        },
        "pool": per_invocation,
    }
    selfplay_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    return {
        "selfplay_vs_pool": True,
        "pool_size": args.pool_size,
        "pfsp_floor": args.pfsp_floor,
        "pool": per_invocation,
        "rows": total_rows,
    }


def append_jsonl_with_pool_annotation(src: Path, out_fh, entry: dict[str, Any]) -> int:
    rows = 0
    with src.open("r", encoding="utf8") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            row = json.loads(raw)
            row["poolPolicyIter"] = int(entry["iteration"])
            row["poolPolicyOnnx"] = entry["onnx"]
            row["poolPolicyCheckpoint"] = entry["checkpoint"]
            out_fh.write(json.dumps(row) + "\n")
            rows += 1
    return rows


def resolve_kl_anchor(
    args: argparse.Namespace,
    state: RebelLoopState,
    init_checkpoint: str | None,
) -> str | None:
    if args.kl_anchor_checkpoint:
        return str(Path(args.kl_anchor_checkpoint).resolve())
    if not args.fixed_kl_anchor:
        return init_checkpoint
    return state.kl_anchor_checkpoint or init_checkpoint


def materialize_replay_mix(
    *,
    args: argparse.Namespace,
    loop_dir: Path,
    iter_dir: Path,
    iteration: int,
    current_selfplay: Path,
    events: EventWriter,
) -> tuple[Path, dict[str, Any]]:
    if not args.cross_iter_replay:
        current_rows = count_lines(current_selfplay)
        summary = {
            "status": "disabled",
            "data": str(current_selfplay),
            "current_rows": current_rows,
            "old_rows": 0,
            "total_rows": current_rows,
        }
        events.emit(iteration=iteration, stage="replay", event_type="passthrough", **summary)
        return current_selfplay, summary

    prior_paths: list[Path] = []
    window = max(0, int(args.replay_window))
    for prior in range(iteration - 1, max(-1, iteration - 1 - window), -1):
        candidate = loop_dir / f"iter-{prior}" / "rebel-selfplay.jsonl"
        if candidate.exists() and count_lines(candidate) > 0:
            prior_paths.append(candidate)

    current_lines = read_jsonl_lines(current_selfplay)
    if not prior_paths or not current_lines:
        summary = {
            "status": "passthrough",
            "reason": "no_prior_vintage" if not prior_paths else "empty_current_selfplay",
            "data": str(current_selfplay),
            "current_rows": len(current_lines),
            "old_rows": 0,
            "total_rows": len(current_lines),
            "vintages": [str(p) for p in prior_paths],
        }
        events.emit(iteration=iteration, stage="replay", event_type="passthrough", **summary)
        return current_selfplay, summary

    # Replay-vintage guard: only mix in OLD-iteration rows whose decision used a
    # pure neural leaf. Rows where `rolloutLeafUsed` is truthy are rollout-
    # contaminated; rows missing the field entirely are old-vintage (pre-fix)
    # and treated as contaminated. This stops contaminated policy/Q targets from
    # leaking into the corrected training mix. Current-iteration rows are always
    # kept as-is (handled below).
    old_pool: list[str] = []
    dropped_contaminated = 0
    for path in prior_paths:
        for line in read_jsonl_lines(path):
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                dropped_contaminated += 1
                continue
            if not isinstance(row, dict) or row.get("rolloutLeafUsed"):
                dropped_contaminated += 1
                continue
            if "rolloutLeafUsed" not in row:
                dropped_contaminated += 1
                continue
            old_pool.append(line)
    print(
        "[rebel-orchestrator] replay-vintage guard: kept "
        f"{len(old_pool)} old rows, dropped {dropped_contaminated} "
        "rollout-contaminated/old-vintage rows",
        flush=True,
    )
    old_fraction = min(0.95, max(0.0, float(args.replay_old_fraction)))
    old_target = int(round(len(current_lines) * old_fraction / max(1.0e-9, 1.0 - old_fraction)))
    old_count = min(len(old_pool), old_target)
    if old_count <= 0:
        summary = {
            "status": "passthrough",
            "reason": "zero_old_target",
            "data": str(current_selfplay),
            "current_rows": len(current_lines),
            "old_rows": 0,
            "total_rows": len(current_lines),
            "vintages": [str(p) for p in prior_paths],
        }
        events.emit(iteration=iteration, stage="replay", event_type="passthrough", **summary)
        return current_selfplay, summary

    stride = max(1, len(old_pool) // old_count)
    old_lines = old_pool[::stride][:old_count]
    mixed_path = iter_dir / "rebel-train-mixed.jsonl"
    with mixed_path.open("w", encoding="utf8") as fh:
        for line in current_lines:
            fh.write(line + "\n")
        for line in old_lines:
            fh.write(line + "\n")

    summary = {
        "status": "materialized",
        "data": str(mixed_path),
        "current_rows": len(current_lines),
        "old_rows": len(old_lines),
        "old_pool_rows": len(old_pool),
        "old_dropped_contaminated": dropped_contaminated,
        "total_rows": len(current_lines) + len(old_lines),
        "old_fraction_target": old_fraction,
        "old_fraction_actual": len(old_lines) / max(1, len(current_lines) + len(old_lines)),
        "window": window,
        "vintages": [str(p) for p in prior_paths],
    }
    events.emit(iteration=iteration, stage="replay", event_type="materialized", **summary)
    return mixed_path, summary


def read_jsonl_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf8") as fh:
        return sum(1 for line in fh if line.strip())


def build_selfplay_cmd(
    args: argparse.Namespace,
    repo: Path,
    data_path: Path,
    selfplay_manifest: Path,
) -> list[str]:
    if args.use_release_binary:
        cmd = [str(repo / "engine-rs" / "target" / "release" / "sim-rebel-selfplay")]
    else:
        cmd = ["cargo", "run", "-p", "sim-cli", "--bin", "sim-rebel-selfplay", "--"]
    cmd.extend(
        [
            "--seeds",
            str(args.games),
            "--seed-start",
            str(args.seed_start),
            "--particles",
            str(args.particles),
            "--search-iterations",
            str(args.search_iterations),
            "--rollout-steps",
            str(args.rollout_steps),
            "--max-steps",
            str(args.max_steps),
            "--model-side",
            args.model_side,
            "--deck-sampling",
            args.deck_sampling,
            "--workers",
            str(args.workers),
            "--out",
            str(data_path),
            "--manifest-out",
            str(selfplay_manifest),
        ]
    )
    if args.selfplay_onnx_path:
        selfplay_device = effective_selfplay_device(args)
        cmd.extend(
            [
                "--onnx-path",
                str(args.selfplay_onnx_path),
                "--device",
                selfplay_device,
                "--cuda-device-id",
                str(args.selfplay_cuda_device_id),
                "--neural-policy-weight",
                str(args.neural_policy_weight),
                "--neural-value-weight",
                str(args.neural_value_weight),
                "--neural-leaf-weight",
                str(args.neural_leaf_weight),
                "--inference-batch-size",
                str(args.selfplay_inference_batch_size),
                "--inference-max-wait-us",
                str(args.selfplay_inference_max_wait_us),
            ]
        )
    if getattr(args, "allow_rollout_leaf", False):
        cmd.append("--allow-rollout-leaf")
    return cmd


def validate_leaf_args(args: argparse.Namespace) -> None:
    """Guard against shipping the info-incorrect determinized rollout leaf.

    The determinized rollout leaf is information-incorrect; the loop must train
    the value head on the grounded MC outcome with a pure neural leaf
    (--neural-leaf-weight 1.0). Any blend toward the rollout leaf requires an
    explicit opt-in via --allow-rollout-leaf (ablation only).
    """
    if float(args.neural_leaf_weight) < 1.0 and not args.allow_rollout_leaf:
        raise SystemExit(
            "Refusing to run: --neural-leaf-weight "
            f"{args.neural_leaf_weight} < 1.0 blends in the information-incorrect "
            "determinized rollout leaf. Pass --allow-rollout-leaf to opt into "
            "this ablation, or set --neural-leaf-weight 1.0 for the corrected loop."
        )


def resolve_runtime_devices(args: argparse.Namespace, repo: Path) -> None:
    if args.device == "auto":
        args.device = "cuda" if torch_cuda_available(repo) else "cpu"
    if args.selfplay_device == "auto":
        args.selfplay_device = "cuda" if args.device == "cuda" else "cpu"


def effective_selfplay_device(args: argparse.Namespace) -> str:
    if args.selfplay_device == "auto":
        return "cuda" if args.device == "cuda" else "cpu"
    return args.selfplay_device


def torch_cuda_available(repo: Path) -> bool:
    try:
        result = subprocess.run(
            [
                training_python(repo),
                "-c",
                "import torch; print('1' if torch.cuda.is_available() else '0')",
            ],
            cwd=str(repo),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return result.returncode == 0 and result.stdout.strip().splitlines()[-1:] == ["1"]


def effective_training_settings(args: argparse.Namespace) -> dict[str, Any]:
    batch_size = args.batch_size if args.batch_size is not None else (16 if args.smoke else 256)
    lr = args.lr if args.lr is not None else (6e-4 if batch_size >= 256 else 3e-4)
    dataloader_workers = (
        args.dataloader_workers
        if args.dataloader_workers is not None
        else (0 if args.smoke or args.device != "cuda" else 4)
    )
    amp = args.amp if args.amp is not None else (False if args.smoke else args.device == "cuda")
    return {
        "batch_size": batch_size,
        "lr": lr,
        "amp": amp,
        "dataloader_workers": dataloader_workers,
        "compile": bool(args.compile),
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "dropout": args.dropout,
        "grad_accum": args.grad_accum,
        "init_from_checkpoint": args.init_from_checkpoint,
    }


def build_train_cmd(
    args: argparse.Namespace,
    repo: Path,
    data_path: Path,
    train_dir: Path,
) -> list[str]:
    settings = effective_training_settings(args)
    train_cmd = [
        training_python(repo),
        str(repo / "training" / "train_bc.py"),
        "--data",
        str(data_path),
        "--out-dir",
        str(train_dir),
        "--data-mode",
        "rebel",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(settings["batch_size"]),
        "--state-dim",
        str(args.state_dim),
        "--hidden-dim",
        str(settings["hidden_dim"]),
        "--depth",
        str(settings["depth"]),
        "--dropout",
        str(settings["dropout"]),
        "--lr",
        str(settings["lr"]),
        "--split-by",
        "row",
        "--device",
        args.device,
        "--value-weight",
        str(args.value_weight),
        "--policy-weight",
        str(args.policy_weight),
        "--kl-anchor-weight",
        str(args.kl_anchor_weight),
        "--entropy-bonus",
        str(args.entropy_bonus),
        "--grad-accum",
        str(settings["grad_accum"]),
    ]
    if settings["amp"]:
        train_cmd.append("--amp")
    if settings["compile"]:
        train_cmd.append("--compile")
    if settings["dataloader_workers"] > 0:
        train_cmd.extend(["--dataloader-workers", str(settings["dataloader_workers"])])
    if settings["init_from_checkpoint"]:
        train_cmd.extend(["--init-from-checkpoint", str(settings["init_from_checkpoint"])])
    if args.q_value_head:
        train_cmd.extend(["--q-value-head", "--q-value-weight", str(args.q_value_weight)])
    if args.uma_slot_tokens:
        train_cmd.append("--uma-slot-tokens")
    if getattr(args, "model_variant", "mlp") != "mlp":
        train_cmd.extend(["--model-variant", str(args.model_variant)])
    if args.kl_anchor_checkpoint:
        train_cmd.extend(["--kl-anchor-checkpoint", args.kl_anchor_checkpoint])
    if getattr(args, "events_out", None):
        train_cmd.extend(["--events-out", str(args.events_out)])
        train_cmd.extend(["--events-iteration", str(getattr(args, "events_iteration", -1))])
    return train_cmd


def training_python(repo: Path) -> str:
    venv_python = repo / "training" / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def run_gates(
    args: argparse.Namespace,
    repo: Path,
    out_dir: Path,
    onnx_path: Path,
    champion_onnx: str | None = None,
) -> dict[str, Any]:
    fixed_manifest = out_dir / "gate-fixed.manifest.json"
    uniform_manifest = out_dir / "gate-uniform.manifest.json"
    # FIXED gate is the promotion driver. When a champion exists, run it
    # head-to-head (challenger-vs-champion) with the pass threshold set to
    # 0.5 + margin so `passed` means "challenger beat champion". With no
    # champion (iter-0 bootstrap) keep the absolute vs-heuristic gate at
    # --gate-min-ci-lower.
    if champion_onnx:
        fixed_min_ci_lower: float | None = 0.5 + args.gate_promote_margin
    else:
        fixed_min_ci_lower = None
    fixed = run_gate(
        args,
        repo,
        onnx_path,
        fixed_manifest,
        deck_sampling="fixed",
        seed_start=args.gate_seed_start,
        opponent_onnx=champion_onnx,
        min_ci_lower_override=fixed_min_ci_lower,
    )
    # UNIFORM gate stays exactly as today: absolute vs-heuristic, no
    # champion opponent. It is a regression sanity signal, not the
    # promotion driver.
    uniform = run_gate(
        args,
        repo,
        onnx_path,
        uniform_manifest,
        deck_sampling="uniform",
        seed_start=args.gate_seed_start + args.gate_games,
    )
    return {
        "fixed": fixed,
        "uniform_deck_diverse": uniform,
        "side_split": {
            "status": "recorded-in-gate-manifests",
            "fields": ["summary.playerSide", "summary.opponentSide"],
        },
    }


def run_gate(
    args: argparse.Namespace,
    repo: Path,
    onnx_path: Path,
    manifest_path: Path,
    *,
    deck_sampling: str,
    seed_start: int,
    opponent_onnx: str | None = None,
    min_ci_lower_override: float | None = None,
) -> dict[str, Any]:
    cmd = build_gate_cmd(
        args,
        repo,
        onnx_path,
        manifest_path,
        deck_sampling=deck_sampling,
        seed_start=seed_start,
        opponent_onnx=opponent_onnx,
        min_ci_lower_override=min_ci_lower_override,
    )
    env = ort_env(repo)
    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=str(repo / "engine-rs"), check=False, env=env)
    if result.returncode != 0 and not manifest_path.exists():
        result.check_returncode()
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    summary = manifest.get("summary") or {}
    return {
        "status": manifest.get("status", "unknown"),
        "passed": bool(manifest.get("passed", False)),
        "failures": manifest.get("failures", []),
        "returncode": result.returncode,
        "manifest": str(manifest_path),
        "deck_sampling": deck_sampling,
        "seed_start": seed_start,
        "sims": args.gate_sims,
        "games": summary.get("overall", {}).get("games", summary.get("games")),
        "win_rate": summary.get("overall", {}).get("winRate", summary.get("modelWinRate")),
        "wilson_lower": (summary.get("wilson95") or {}).get("lower"),
        "player_side": summary.get("playerSide"),
        "opponent_side": summary.get("opponentSide"),
        # Head-to-head bookkeeping: when set, this gate measured
        # challenger-vs-champion and `passed`/`wilson_lower` are vs-champion
        # numbers; otherwise it is the absolute vs-heuristic gate.
        "head_to_head": bool(opponent_onnx),
        "opponent_onnx": opponent_onnx,
        "min_ci_lower": (
            min_ci_lower_override if min_ci_lower_override is not None else args.gate_min_ci_lower
        ),
    }


def build_gate_cmd(
    args: argparse.Namespace,
    repo: Path,
    onnx_path: Path,
    manifest_path: Path,
    *,
    deck_sampling: str,
    seed_start: int,
    opponent_onnx: str | None = None,
    min_ci_lower_override: float | None = None,
) -> list[str]:
    if args.use_release_binary:
        cmd = [str(repo / "engine-rs" / "target" / "release" / "sim-eval-gate")]
    else:
        cmd = ["cargo", "run", "-p", "sim-cli", "--bin", "sim-eval-gate", "--"]
    cmd.extend(
        [
            "--games",
            str(args.gate_games),
            "--seed-start",
            str(seed_start),
            "--sims",
            str(args.gate_sims),
            "--mcts-prior",
            "policy",
            "--mcts-leaf",
            args.gate_leaf,
            "--onnx-path",
            str(onnx_path),
            "--model-side",
            "both",
            "--deck-sampling",
            deck_sampling,
            "--max-steps",
            str(args.gate_max_steps),
            "--workers",
            str(args.gate_workers),
            "--batch-size",
            str(args.gate_batch_size),
            "--manifest-out",
            str(manifest_path),
        ]
    )
    if args.gate_min_games > 0:
        cmd.extend(["--min-games", str(args.gate_min_games)])
    # Head-to-head: when a champion is supplied the opponent side runs the
    # champion model (same MCTS config) and the pass threshold becomes
    # 0.5 + margin, so `passed` means "challenger beat champion". The
    # explicit override takes precedence over the absolute vs-heuristic
    # --gate-min-ci-lower (used only for the iter-0 bootstrap gate).
    if opponent_onnx:
        cmd.extend(["--opponent-onnx", opponent_onnx])
    effective_min_ci_lower = (
        min_ci_lower_override if min_ci_lower_override is not None else args.gate_min_ci_lower
    )
    if effective_min_ci_lower > 0.0:
        cmd.extend(["--min-ci-lower", str(effective_min_ci_lower)])
    if args.gate_min_win_rate > 0.0:
        cmd.extend(["--min-win-rate", str(args.gate_min_win_rate)])
    return cmd


def ort_env(repo: Path) -> dict[str, str]:
    import os

    env = os.environ.copy()
    lib_dirs: list[str] = []
    nvidia_root = repo / "training" / ".venv" / "lib" / "python3.12" / "site-packages" / "nvidia"
    if nvidia_root.is_dir():
        lib_dirs.extend(str(p) for p in nvidia_root.glob("*/lib") if p.is_dir())
    capi = repo / "training" / ".venv" / "lib" / "python3.12" / "site-packages" / "onnxruntime" / "capi"
    if capi.is_dir():
        lib_dirs.append(str(capi))
    if lib_dirs:
        existing = env.get("LD_LIBRARY_PATH", "")
        existing_parts = existing.split(":") if existing else []
        prepend = [p for p in lib_dirs if p not in existing_parts]
        env["LD_LIBRARY_PATH"] = ":".join(prepend + existing_parts)
    if "ORT_DYLIB_PATH" not in env:
        candidate = capi / "libonnxruntime.so.1.22.0"
        if candidate.exists():
            env["ORT_DYLIB_PATH"] = str(candidate)
    return env


def validate_rebel_rows(path: Path) -> dict[str, Any]:
    required = {
        "kind",
        "schemaVersion",
        "beliefSchemaVersion",
        "observation",
        "beliefFeatures",
        "publicHistoryDigest",
        "legalActions",
        "selectedActionIndex",
        "searchPolicy",
        "searchActionValues",
        "beliefValue",
        "privateStateValues",
        "valueTarget",
        "particleCount",
        "searchIterations",
        "searchAlgorithm",
        "beliefSampler",
        "playerDeckId",
        "opponentDeckId",
    }
    rows = 0
    policy_entropy_sum = 0.0
    legal_hist: dict[str, int] = {}
    pool_policy_hist: dict[str, int] = {}
    with path.open("r", encoding="utf8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = sorted(required - set(row))
            if missing:
                raise SystemExit(f"{path}:{line_number}: missing keys {missing}")
            if row["kind"] != "rebel-selfplay":
                raise SystemExit(f"{path}:{line_number}: bad kind {row['kind']!r}")
            if int(row["schemaVersion"]) != 1 or int(row["beliefSchemaVersion"]) != 1:
                raise SystemExit(f"{path}:{line_number}: bad schema versions")
            actions = row["legalActions"]
            policy = row["searchPolicy"]
            q_values = row["searchActionValues"]
            if len(actions) < 2 or len(policy) != len(actions) or len(q_values) != len(actions):
                raise SystemExit(f"{path}:{line_number}: action/search target length mismatch")
            selected = int(row["selectedActionIndex"])
            if selected < 0 or selected >= len(actions):
                raise SystemExit(f"{path}:{line_number}: selectedActionIndex {selected} out of range")
            total = sum(float(v) for v in policy)
            if abs(total - 1.0) > 1e-4:
                raise SystemExit(f"{path}:{line_number}: searchPolicy sums to {total}")
            if not all_finite(policy) or not all_finite(q_values):
                raise SystemExit(f"{path}:{line_number}: non-finite search targets")
            belief_value = float(row["beliefValue"])
            if not is_finite(belief_value):
                raise SystemExit(f"{path}:{line_number}: non-finite beliefValue")
            private_values = row["privateStateValues"]
            particle_count = int(row["particleCount"])
            if len(private_values) != particle_count:
                raise SystemExit(
                    f"{path}:{line_number}: privateStateValues length {len(private_values)} != particleCount {particle_count}"
                )
            if not all_finite(private_values):
                raise SystemExit(f"{path}:{line_number}: non-finite privateStateValues")
            belief_vector = (row.get("beliefFeatures") or {}).get("vector") or []
            if len(belief_vector) != 16:
                raise SystemExit(f"{path}:{line_number}: belief feature dim {len(belief_vector)} != 16")
            if not all_finite(belief_vector):
                raise SystemExit(f"{path}:{line_number}: non-finite belief feature")
            diagnostics = row.get("searchDiagnostics") or {}
            for key in ("particleCount", "legalActionCount", "searchIterations", "policyEntropy"):
                if key not in diagnostics:
                    raise SystemExit(f"{path}:{line_number}: missing searchDiagnostics.{key}")
            if int(diagnostics["particleCount"]) != particle_count:
                raise SystemExit(f"{path}:{line_number}: diagnostics particleCount mismatch")
            if int(diagnostics["legalActionCount"]) != len(actions):
                raise SystemExit(f"{path}:{line_number}: diagnostics legalActionCount mismatch")
            pool_fields = ["poolPolicyIter", "poolPolicyOnnx", "poolPolicyCheckpoint"]
            present_pool_fields = [field for field in pool_fields if field in row]
            if present_pool_fields and len(present_pool_fields) != len(pool_fields):
                raise SystemExit(f"{path}:{line_number}: incomplete pool annotation {present_pool_fields}")
            if present_pool_fields:
                pool_key = str(row["poolPolicyIter"])
                pool_policy_hist[pool_key] = pool_policy_hist.get(pool_key, 0) + 1
            policy_entropy_sum += -sum(float(p) * safe_log(float(p)) for p in policy if float(p) > 0.0)
            legal_hist[str(len(actions))] = legal_hist.get(str(len(actions)), 0) + 1
            rows += 1
    if rows == 0:
        raise SystemExit(f"{path}: no rebel-selfplay rows")
    return {
        "rows": rows,
        "legal_action_count_histogram": legal_hist,
        "mean_search_policy_entropy": policy_entropy_sum / rows,
        "pool_policy_histogram": pool_policy_hist,
    }


def is_finite(value: float) -> bool:
    import math

    return math.isfinite(float(value))


def all_finite(values: list[Any]) -> bool:
    return all(is_finite(float(value)) for value in values)


def safe_log(value: float) -> float:
    import math

    return math.log(max(value, 1e-12))


def run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one R17 ReBeL E2E iteration.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--halt-after-consecutive-failures", type=int, default=2)
    parser.add_argument("--cross-iter-replay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--replay-window", type=int, default=3)
    parser.add_argument("--replay-old-fraction", type=float, default=0.4)
    parser.add_argument("--fixed-kl-anchor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--selfplay-vs-pool", action="store_true")
    parser.add_argument("--pool-size", type=int, default=5)
    parser.add_argument("--pfsp-floor", type=float, default=0.05)
    parser.add_argument("--pool-state-file", default=None)
    parser.add_argument("--smoke", action="store_true", help="Use tiny training defaults for CPU smoke runs.")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--particles", type=int, default=16)
    parser.add_argument("--search-iterations", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--model-side", choices=["player", "opponent", "both"], default="both")
    parser.add_argument("--deck-sampling", default="fixed")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--use-release-binary", action="store_true")
    parser.add_argument("--selfplay-onnx-path", default=None)
    parser.add_argument("--selfplay-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--selfplay-cuda-device-id", type=int, default=0)
    parser.add_argument("--neural-policy-weight", type=float, default=0.25)
    parser.add_argument("--neural-value-weight", type=float, default=0.25)
    parser.add_argument("--neural-leaf-weight", type=float, default=1.0)
    parser.add_argument(
        "--allow-rollout-leaf",
        action="store_true",
        default=False,
        help=(
            "Opt into the information-incorrect determinized rollout leaf in "
            "ReBeL search (ablation only). Without this flag a "
            "--neural-leaf-weight < 1.0 is rejected at preflight."
        ),
    )
    parser.add_argument("--selfplay-inference-batch-size", type=int, default=1)
    parser.add_argument("--selfplay-inference-max-wait-us", type=int, default=2_000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--state-dim", type=int, default=110)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dataloader-workers", type=int, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--init-from-checkpoint", default=None)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=0.1)
    parser.add_argument("--q-value-head", action="store_true")
    parser.add_argument("--q-value-weight", type=float, default=0.0)
    parser.add_argument("--uma-slot-tokens", action="store_true")
    parser.add_argument("--model-variant", choices=["mlp", "set_attention", "relational"],
                        default="mlp",
                        help="Trunk architecture forwarded to train_bc. 'relational' "
                             "is the v6 attention-native trunk (ReBeL-belief-token "
                             "aware); requires --uma-slot-tokens. Rides the existing "
                             "belief (8-input) ONNX dispatch with no graph-signature "
                             "change.")
    parser.add_argument("--kl-anchor-checkpoint", default=None)
    parser.add_argument("--kl-anchor-weight", type=float, default=0.0)
    parser.add_argument("--entropy-bonus", type=float, default=0.0)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-gates", action="store_true")
    parser.add_argument("--gate-games", type=int, default=20)
    parser.add_argument("--gate-seed-start", type=int, default=90000)
    parser.add_argument("--gate-sims", type=int, default=100)
    parser.add_argument("--gate-leaf", choices=["rollout", "value-head"], default="rollout")
    parser.add_argument("--gate-max-steps", type=int, default=500)
    parser.add_argument("--gate-workers", type=int, default=1)
    parser.add_argument("--gate-batch-size", type=int, default=1)
    parser.add_argument("--gate-min-games", type=int, default=0)
    parser.add_argument("--gate-min-ci-lower", type=float, default=0.0)
    parser.add_argument("--gate-min-win-rate", type=float, default=0.0)
    parser.add_argument(
        "--gate-promote-margin",
        type=float,
        default=0.0,
        help=(
            "Head-to-head promotion margin. When a champion exists, the FIXED "
            "gate runs challenger-vs-champion (--opponent-onnx <champion>) and "
            "its --min-ci-lower is set to 0.5 + this margin, so the Rust gate's "
            "`passed` directly encodes 'challenger beat the champion at the 95%% "
            "Wilson lower bound + margin'. Iter-0 bootstrap (no champion) keeps "
            "the absolute vs-heuristic gate at --gate-min-ci-lower."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
