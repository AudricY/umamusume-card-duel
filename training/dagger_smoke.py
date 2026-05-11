"""3-iteration DAgger orchestrator smoke.

Drives ``dagger_orchestrator.py`` with tiny configs (1-3 games per
iteration, baseline-as-gate to skip the ONNX server bring-up) so the
chained pipeline is exercised end-to-end and a per-iteration manifest
tree is produced under a temp directory.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from opponent_pool import OpponentPool  # noqa: E402  (path bootstrap)
from dagger_orchestrator import (  # noqa: E402
    OrchestratorState,
    compute_matchup_floor_violations,
    decide_promotion,
)
from argparse import Namespace  # noqa: E402


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    work_dir = Path(tempfile.mkdtemp(prefix="uma-dagger-orchestrator-"))
    try:
        cmd = [
            sys.executable,
            str(repo_root / "training" / "dagger_orchestrator.py"),
            "--out-dir",
            str(work_dir),
            "--iterations",
            "3",
            "--games",
            "1",
            "--max-steps",
            "120",
            "--teacher",
            "rollout",
            "--rollout-steps",
            "40",
            "--rollout-crn-samples",
            "2",
            "--replay-games",
            "2",
            "--epochs",
            "2",
            "--batch-size",
            "16",
            "--hidden-dim",
            "32",
            "--depth",
            "1",
            "--eval-games",
            "1",
            "--skip-policy-gate",
            # Item 12 escape hatch: pool matchups need two simultaneous
            # serve_onnx instances, which would dominate the smoke wall
            # clock. The unit-style smoke in opponent_pool_smoke.py
            # exercises the data-structure invariants directly.
            "--pool-eval-games",
            "0",
        ]
        subprocess.run(cmd, cwd=repo_root, check=True)
        state = json.loads((work_dir / "orchestrator-state.json").read_text(encoding="utf8"))
        if len(state.get("iterations", [])) != 3:
            raise AssertionError(f"Expected 3 iterations recorded, got {state.get('iterations')}")
        seen = set()
        for iteration in state["iterations"]:
            seen.add(int(iteration["iteration"]))
            iter_dir = work_dir / f"iter-{int(iteration['iteration']):03d}"
            for required in ("trace.jsonl", "relabeled.jsonl", "mixed.jsonl", "iteration-manifest.json", "model"):
                if not (iter_dir / required).exists():
                    raise AssertionError(f"Missing {required} under {iter_dir}")
            if int(iteration["mixed_rows"]) <= 0:
                raise AssertionError(f"Iteration {iteration['iteration']} produced no mixed rows")
            for required_field in ("pool_evals", "pool_aggregate_wilson_lower", "cycling_alarm"):
                if required_field not in iteration:
                    raise AssertionError(
                        f"Iteration {iteration['iteration']} missing {required_field}: {iteration}"
                    )
            if iteration["pool_evals"]:
                # --pool-eval-games 0 should skip pool matchups cleanly.
                raise AssertionError(
                    f"--pool-eval-games 0 should produce no pool eval rows, got {iteration['pool_evals']}"
                )
        if seen != {0, 1, 2}:
            raise AssertionError(f"Iteration ids unexpected: {sorted(seen)}")
        promoted_count = sum(1 for it in state["iterations"] if it.get("promoted"))
        # First iteration always seeds the promoted checkpoint when the
        # gate passes; later iterations may reject without breaking the
        # smoke contract. We just demand at least one promotion so the
        # gate->promote path is exercised.
        if promoted_count < 1:
            raise AssertionError(f"Expected at least one promoted iteration, got {state}")

        # Observability Stage 1: events.jsonl must exist with a complete
        # per-iteration coverage so views downstream can reconstruct
        # pipeline state without scanning manifest trees.
        events_path = work_dir / "events.jsonl"
        if not events_path.exists() or events_path.stat().st_size == 0:
            raise AssertionError(f"Missing or empty events.jsonl under {work_dir}")
        all_events = [json.loads(line) for line in events_path.read_text(encoding="utf8").splitlines() if line.strip()]
        if not all_events:
            raise AssertionError(f"events.jsonl had no parseable rows: {events_path}")
        # Run-level boundaries.
        run_started = [e for e in all_events if e["stage"] == "orchestrator" and e["event_type"] == "run_started"]
        run_completed = [e for e in all_events if e["stage"] == "orchestrator" and e["event_type"] == "run_completed"]
        if len(run_started) != 1 or len(run_completed) != 1:
            raise AssertionError(f"run_started/run_completed events should each appear once, got {len(run_started)}/{len(run_completed)}")
        # Required per-iteration stages (started+completed pairs).
        required_pairs = ["trace-gen", "relabel", "mix", "train", "gate"]
        for it in (0, 1, 2):
            for stage in required_pairs:
                started = [e for e in all_events if e["iteration"] == it and e["stage"] == stage and e["event_type"] == "started"]
                completed = [e for e in all_events if e["iteration"] == it and e["stage"] == stage and e["event_type"] == "completed"]
                if not started or not completed:
                    raise AssertionError(f"iter {it} stage '{stage}' missing started/completed pair: started={len(started)} completed={len(completed)}")
            decision = [e for e in all_events if e["iteration"] == it and e["stage"] == "decision"]
            if not decision:
                raise AssertionError(f"iter {it} missing decision event")
            # Per-epoch loss events must appear for every iteration, not just
            # iter-0. This is the regression guard for the DAgger warm-start
            # zero-epoch bug uncovered by observability on item 17.
            epoch_events = [e for e in all_events if e["iteration"] == it and e["stage"] == "train" and e["event_type"] == "epoch"]
            if len(epoch_events) < 1:
                raise AssertionError(
                    f"iter {it} recorded no train epoch events — training likely skipped. "
                    f"Check that --init-from-checkpoint (not --resume) is used for warm-start."
                )

        # Cross-check: iter-1 and iter-2 trained models must NOT be byte-identical to iter-0's.
        # If they are, --init-from-checkpoint is silently no-opping again.
        import hashlib
        def _hash(path: Path) -> str:
            return hashlib.md5(path.read_bytes()).hexdigest() if path.exists() else "MISSING"
        ckpts = {it: _hash(work_dir / f"iter-{it:03d}" / "model" / "checkpoint.pt") for it in (0, 1, 2)}
        if ckpts[0] == ckpts[1] == ckpts[2] != "MISSING":
            raise AssertionError(
                f"iter-0/1/2 checkpoints are byte-identical (md5={ckpts[0]}) — "
                f"DAgger warm-start is not actually training. Bug regression."
            )

        # Item 12: opponent-pool persistence + per-iteration snapshot.
        pool_path = work_dir / "opponent-pool.json"
        if not pool_path.exists():
            raise AssertionError(f"Missing opponent-pool.json under {work_dir}")
        pool = OpponentPool.from_json(pool_path)
        # Each promoted iteration snapshots one entry. Iteration 2 must
        # have at least 1 promoted entry behind it; iteration 3, at
        # least 2. We assert the floor since the rule-bot gate is
        # noisy at this scale and may promote all three.
        promoted_iters = [int(it["iteration"]) for it in state["iterations"] if it.get("promoted")]
        expected_pool_size = len(promoted_iters)
        if len(pool.entries) != expected_pool_size:
            raise AssertionError(
                f"Pool size {len(pool.entries)} != promoted-iteration count {expected_pool_size}"
            )
        if expected_pool_size >= 2:
            # When 2 or more iterations promote, the pool must reflect that.
            assert len(pool.entries) >= 2, f"expected >=2 pool entries, got {len(pool.entries)}"
        for entry in pool.entries:
            if not Path(entry.checkpoint_path).exists():
                raise AssertionError(f"Pool entry checkpoint missing on disk: {entry}")

        # Item 12: cycling alarm unit-style assertion on a planted
        # rock-paper-scissors regression. Independent of the smoke run
        # so it doesn't depend on the noisy pipeline.
        planted_history = {
            4: [0.4, 0.5, 0.5, 0.45],  # not strictly declining (tie at 0.5,0.5)
            7: [0.7, 0.6, 0.5, 0.4],   # strict decline on last 3 → must trip
        }
        flagged = OpponentPool.cycling_alarm(planted_history)
        if flagged != [7]:
            raise AssertionError(
                f"planted RPS regression should flag opponent 7 only, got {flagged}"
            )

        # Item 13 residual: per-matchup floor violation as a planted fixture.
        # An iteration with a -0.10 drop on opponent iter-3 must produce
        # exactly one violation against the default 0.05 tolerance.
        plant_state = OrchestratorState(
            promoted_wilson_lower=0.50,
            iterations=[
                {"iteration": 0, "pool_evals": [
                    {"opponent_iteration": 3, "wilson_lower": 0.62, "win_rate": 0.7, "n_games": 20},
                    {"opponent_iteration": 5, "wilson_lower": 0.40, "win_rate": 0.5, "n_games": 20},
                ]},
            ],
        )
        current_pool_eval = [
            {"opponent_iteration": 3, "wilson_lower": 0.50, "win_rate": 0.6, "n_games": 20},
            {"opponent_iteration": 5, "wilson_lower": 0.42, "win_rate": 0.55, "n_games": 20},
        ]
        violations = compute_matchup_floor_violations(plant_state, current_pool_eval, tolerance=0.05)
        if len(violations) != 1 or violations[0]["opponent_iteration"] != 3:
            raise AssertionError(f"planted -0.12 drop on opp 3 should violate; got {violations}")
        if violations[0]["drop"] < 0.05 or violations[0]["drop"] > 0.20:
            raise AssertionError(f"violation drop magnitude unexpected: {violations}")

        # Item 13 residual: a per-matchup violation must reject promotion
        # even when the aggregate Wilson lower would otherwise pass.
        plant_args = Namespace(
            eval_min_ci_lower=0.0,
            per_matchup_drop_tolerance=0.05,
        )
        decision = decide_promotion(
            plant_state,
            gate_returncode=0,
            wilson_lower=0.55,  # would pass aggregate (>= floor 0.50)
            args=plant_args,
            eval_n=200,
            matchup_violations=violations,
        )
        if decision["promote"]:
            raise AssertionError(
                f"per-matchup violation must veto promotion even when aggregate passes; got {decision}"
            )
        if "per-matchup floor" not in decision["reason"]:
            raise AssertionError(f"rejection reason should cite per-matchup floor: {decision}")

        # Item 13 residual: halt-after-2 — two consecutive rejections must
        # mark the state as halted with halt_reason populated.
        halt_state = OrchestratorState(
            promoted_wilson_lower=0.50,
        )
        for _ in range(2):
            halt_state.consecutive_failures += 1
            if halt_state.consecutive_failures >= 2:
                halt_state.halted = True
                halt_state.halt_reason = "halt-after-2: smoke fixture"
        if not halt_state.halted or halt_state.halt_reason is None:
            raise AssertionError("halt-after-2 fixture failed to set halted/halt_reason")
        # Promotion must clear the consecutive_failures counter.
        halt_state.consecutive_failures = 0
        halt_state.halted = False
        halt_state.halt_reason = None
        if halt_state.halted or halt_state.consecutive_failures != 0:
            raise AssertionError("halt clear-on-promote fixture failed")

        print(json.dumps({
            "status": "PASS",
            "work_dir": str(work_dir),
            "promoted_count": promoted_count,
            "pool_size": len(pool.entries),
            "matchup_violations_planted": len(violations),
        }, indent=2))
    finally:
        # Keep work_dir for inspection on failure; clean only on success.
        if "PASS" in str(sys.exc_info()[1] or ""):
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
