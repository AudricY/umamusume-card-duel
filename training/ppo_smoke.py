"""End-to-end PPO orchestrator smoke (F1).

Runs 2 PPO iterations × 3 games/update × max-steps 80 × eval-games 5
against the warm-start checkpoint at
``runs/item17-2026-05-11/iter-002/model/checkpoint.pt``. Verifies:

- ppo_orchestrator runs to completion;
- events.jsonl contains the PPO-specific stages
  (rollout, gae, ppo-update including per-minibatch rows);
- the promoted checkpoint differs in weights from the warm-start;
- per-minibatch KL stays in [0, 0.5] (relaxed smoke tolerance — the
  production f1-design.md gate is 0.05);
- entropy never collapses to zero (no NaN / no full-greedy degenerate);
- importance ratios are finite (no NaN, no inf);
- the per-update manifest records all five f1-design.md stability
  controls: lr, clip_epsilon, entropy_coef, value_coef, grad_clip.

Smoke target: under 3 minutes wall clock with --skip-policy-gate to avoid
the ONNX server bring-up for the gate. The collection-side serve_onnx is
mandatory and stochastic.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    warm_start = repo_root / "runs" / "item17-2026-05-11" / "iter-002" / "model" / "checkpoint.pt"
    if not warm_start.exists():
        raise SystemExit(f"Warm-start checkpoint missing: {warm_start}")
    work_dir = Path(tempfile.mkdtemp(prefix="uma-ppo-orchestrator-"))
    cleanup_on_success = True
    try:
        cmd = [
            sys.executable,
            str(repo_root / "training" / "ppo_orchestrator.py"),
            "--out-dir",
            str(work_dir),
            "--init-from",
            str(warm_start),
            "--iterations",
            "2",
            "--games-per-update",
            "3",
            "--max-steps",
            "80",
            "--rollout-steps",
            "40",
            "--temperature",
            "1.0",
            "--lr",
            "3e-5",
            "--clip-epsilon",
            "0.2",
            "--entropy-coef",
            "0.005",
            "--value-coef",
            "0.5",
            "--gae-lambda",
            "0.95",
            "--gae-gamma",
            "0.99",
            "--grad-clip",
            "0.5",
            "--minibatches",
            "2",
            "--ppo-epochs",
            "1",
            "--eval-games",
            "5",
            "--skip-policy-gate",
            "--pool-eval-games",
            "0",
        ]
        subprocess.run(cmd, cwd=repo_root, check=True, timeout=240)

        # 1. State file written, both iterations recorded.
        state_path = work_dir / "orchestrator-state.json"
        if not state_path.exists():
            raise AssertionError(f"orchestrator-state.json missing under {work_dir}")
        state = json.loads(state_path.read_text(encoding="utf8"))
        if len(state.get("iterations", [])) != 2:
            raise AssertionError(f"Expected 2 iterations recorded, got {len(state.get('iterations', []))}")

        # 2. Events stream contains the PPO-specific stages.
        events_path = work_dir / "events.jsonl"
        if not events_path.exists() or events_path.stat().st_size == 0:
            raise AssertionError(f"Missing or empty events.jsonl under {work_dir}")
        all_events = [json.loads(line) for line in events_path.read_text(encoding="utf8").splitlines() if line.strip()]
        if not all_events:
            raise AssertionError(f"events.jsonl had no parseable rows under {events_path}")

        for it in (0, 1):
            # Rollout (started + completed).
            for stage in ("rollout", "ppo-update"):
                started = [e for e in all_events if e["iteration"] == it and e["stage"] == stage and e["event_type"] == "started"]
                completed = [e for e in all_events if e["iteration"] == it and e["stage"] == stage and e["event_type"] == "completed"]
                if not started or not completed:
                    raise AssertionError(f"iter {it} stage {stage} missing started/completed pair")
            # gae/completed must carry mean_advantage + mean_return.
            gae = [e for e in all_events if e["iteration"] == it and e["stage"] == "gae" and e["event_type"] == "completed"]
            if not gae:
                raise AssertionError(f"iter {it} missing gae/completed event")
            gae_data = gae[0]["data"]
            for required in ("mean_advantage", "mean_return"):
                if required not in gae_data:
                    raise AssertionError(f"iter {it} gae/completed missing {required}: {gae_data}")

            # ppo-update/minibatch must appear at least once per iteration.
            mb_events = [
                e for e in all_events
                if e["iteration"] == it and e["stage"] == "ppo-update" and e["event_type"] == "minibatch"
            ]
            if not mb_events:
                raise AssertionError(f"iter {it} missing ppo-update/minibatch events")
            for mb in mb_events:
                data = mb["data"]
                for required in ("kl", "entropy", "ratio_mean", "policy_loss", "value_loss"):
                    if required not in data:
                        raise AssertionError(f"iter {it} ppo-update/minibatch missing {required}: {data}")
                # approx-KL = mean(behavior_logp - new_logp) is a sample
                # estimator of KL(behavior || new). It is non-negative *in
                # expectation* but a single minibatch's sample mean can be
                # slightly negative due to noise. Tolerate small negatives;
                # f1-design.md's production gate is +0.05 magnitude, so
                # smoke tolerates [-0.05, 0.5] (relaxed).
                kl_value = float(data["kl"])
                if not (-0.05 <= kl_value <= 0.5):
                    raise AssertionError(
                        f"iter {it} KL out of smoke band [-0.05, 0.5]: {kl_value}"
                    )
                # Entropy is non-negative; zero would indicate full
                # collapse. Tiny but >0 is fine on a small smoke batch.
                if not (float(data["entropy"]) > 0.0):
                    raise AssertionError(
                        f"iter {it} entropy collapsed: {data['entropy']}"
                    )
                if not _is_finite(float(data["ratio_mean"])):
                    raise AssertionError(
                        f"iter {it} ratio_mean non-finite: {data['ratio_mean']}"
                    )
                if not _is_finite(float(data["policy_loss"])):
                    raise AssertionError(
                        f"iter {it} policy_loss non-finite: {data['policy_loss']}"
                    )
                if not _is_finite(float(data["value_loss"])):
                    raise AssertionError(
                        f"iter {it} value_loss non-finite: {data['value_loss']}"
                    )

            # Per-update manifest must record all five stability controls.
            train_manifest_path = work_dir / f"iter-{it:03d}" / "model" / "manifest.json"
            if not train_manifest_path.exists():
                raise AssertionError(f"iter {it} train manifest missing: {train_manifest_path}")
            train_manifest = json.loads(train_manifest_path.read_text(encoding="utf8"))
            hp = train_manifest.get("hyperparams", {})
            for key in ("lr", "clip_epsilon", "entropy_coef", "value_coef", "grad_clip"):
                if key not in hp:
                    raise AssertionError(
                        f"iter {it} train manifest missing stability control '{key}': {hp}"
                    )

        # 3. Promoted checkpoint differs in weights from --init-from.
        # The PPO update with a non-trivial reward signal *must* perturb
        # weights or the trainer is silently no-opping.
        warm_hash = hashlib.md5(warm_start.read_bytes()).hexdigest()
        # Pick the most recently-promoted iteration's checkpoint.
        promoted_iters = [it for it in state["iterations"] if it.get("promoted")]
        # Even if neither iteration is promoted (small smoke is noisy)
        # we can still check the iter-0 PPO update produced a different
        # checkpoint from the warm-start. Promotion is orthogonal to
        # weight-change.
        any_diff = False
        for it_record in state["iterations"]:
            it_idx = it_record["iteration"]
            ckpt = work_dir / f"iter-{it_idx:03d}" / "model" / "checkpoint.pt"
            if ckpt.exists():
                ckpt_hash = hashlib.md5(ckpt.read_bytes()).hexdigest()
                if ckpt_hash != warm_hash:
                    any_diff = True
                    break
        if not any_diff:
            raise AssertionError(
                "PPO update produced byte-identical weights to --init-from across all "
                "iterations; trainer is silently no-opping."
            )

        # 4. The 5 acceptance gates from f1-design.md "Smoke acceptance",
        # verified at smoke tolerances. We've already asserted:
        #   - KL in [0, 0.5] (relaxed from 0.05) — per minibatch above
        #   - entropy > 0 — per minibatch above
        #   - no NaN/inf — per minibatch above
        #   - all 5 stability controls in manifest — above
        # The last gate (pool-aggregate Wilson lower at update N >=
        # update 0 - 5pp) is meaningless on a 2-iter smoke and is the
        # production-only assertion; we document it here and skip.

        print(json.dumps({
            "status": "PASS",
            "work_dir": str(work_dir),
            "iterations": len(state["iterations"]),
            "promoted_count": len(promoted_iters),
            "events_count": len(all_events),
        }, indent=2))
    except BaseException:
        cleanup_on_success = False
        raise
    finally:
        if cleanup_on_success:
            shutil.rmtree(work_dir, ignore_errors=True)


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


if __name__ == "__main__":
    main()
