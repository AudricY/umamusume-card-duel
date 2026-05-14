"""R8 step 2b — gate-evaluate the trained DPO checkpoint.

Mechanical wrapper around three existing primitives:

  * ``export_checkpoint_to_onnx`` (training/dagger_orchestrator.py)
  * ``serve_onnx_context``        (training/dagger_orchestrator.py)
  * ``run_eval_gate``             (training/uma_ai/node_bridge.py)

The dagger orchestrator already chains all three for its policy-gate step;
R8 step 2b only needs that chain on a *trained DPO checkpoint that lives
outside any orchestrator iteration*. Rather than fork the orchestrator
loop, this script imports the helpers and runs them once.

CLI mirrors what the dagger orchestrator passes internally so the gate
output is byte-comparable to e.g. ``runs/R7-multi-teacher-warmstart/iter-000/
gate.manifest.json``.

Usage:

    python training/r8_gate_eval.py \
        --checkpoint runs/R8-dpo/iter-000/model/checkpoint.pt \
        --out-dir runs/R8-dpo/iter-000 \
        --games 500 \
        --seed-start 9000 \
        --min-ci-lower 0.40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# training/ is on sys.path when invoked as `python training/r8_gate_eval.py`
# from the repo root; ensure it's available either way.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from dagger_orchestrator import (  # noqa: E402  (sys.path tweak above)
    export_checkpoint_to_onnx,
    serve_onnx_context,
)
from uma_ai.node_bridge import run_eval_gate  # noqa: E402


REPO_ROOT = _THIS_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained DPO checkpoint.pt to evaluate.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help=(
            "Directory to write policy.gate.onnx and gate.manifest.json. "
            "Created if missing."
        ),
    )
    parser.add_argument(
        "--games",
        type=int,
        default=500,
        help="Total games (side-balanced — half player, half opponent). Default 500.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=9000,
        help=(
            "Seed-start for the gate run. R7 step 4 used 9000 (the dagger "
            "orchestrator default); pass the same value here for honest comparison."
        ),
    )
    parser.add_argument(
        "--min-ci-lower",
        type=float,
        default=0.40,
        help="Pre-registered Wilson 95%% lower bound the gate must meet to PASS.",
    )
    parser.add_argument(
        "--rollout-crn-samples",
        type=int,
        default=1,
        help=(
            "rolloutCrnSamples for the eval-gate run. Matches dagger orchestrator "
            "default of 1 for the policy-gate step (planner CRN is 3)."
        ),
    )
    parser.add_argument(
        "--planner-crn-samples",
        type=int,
        default=3,
        help="plannerCrnSamples for the eval-gate run (default 3 — matches R7 step 4).",
    )
    parser.add_argument(
        "--selection",
        default="policy",
        help=(
            "evaluator selection mode (default 'policy' — same as the dagger "
            "orchestrator's policy-gate. The TS-side parser falls back to "
            "'policy' for any unrecognised value)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    checkpoint: Path = args.checkpoint.resolve()
    out_dir: Path = args.out_dir.resolve()
    if not checkpoint.is_file():
        print(f"ERROR: checkpoint not found at {checkpoint}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = out_dir / "policy.gate.onnx"
    manifest_out = out_dir / "gate.manifest.json"

    print(f"[r8-gate] checkpoint:    {checkpoint}", flush=True)
    print(f"[r8-gate] out-dir:       {out_dir}", flush=True)
    print(f"[r8-gate] games:         {args.games} (side-balanced)", flush=True)
    print(f"[r8-gate] seed-start:    {args.seed_start}", flush=True)
    print(f"[r8-gate] min-ci-lower:  {args.min_ci_lower}", flush=True)
    print(f"[r8-gate] selection:     {args.selection}", flush=True)

    print(f"[r8-gate] exporting checkpoint -> {onnx_path}", flush=True)
    export_checkpoint_to_onnx(REPO_ROOT, checkpoint, onnx_path)

    # serve_onnx_context only consumes the args namespace as an opaque
    # carrier today (port is allocated locally, provider hardcoded to
    # cpu). Pass an empty Namespace so the contract is honoured without
    # the caller having to know the orchestrator's full arg surface.
    serve_args = argparse.Namespace()

    print("[r8-gate] starting serve_onnx", flush=True)
    with serve_onnx_context(REPO_ROOT, onnx_path, serve_args) as model_url:
        print(f"[r8-gate] serve_onnx healthy at {model_url}", flush=True)
        rc = run_eval_gate(
            REPO_ROOT,
            selection=args.selection,
            games=args.games,
            seed_start=args.seed_start,
            min_games=args.games,
            min_ci_lower=args.min_ci_lower,
            manifest_out=manifest_out,
            require_zero_no_ops=True,
            rollout_crn_samples=args.rollout_crn_samples,
            planner_crn_samples=args.planner_crn_samples,
            extra=["--model-url", model_url],
        )

    if rc != 0:
        # Note: run_eval_gate returns the underlying npm process's exit
        # code. A non-zero rc may mean the gate ran cleanly but failed
        # the pre-registered ci-lower — the manifest at manifest_out
        # is still authoritative. Surface the rc + manifest path so the
        # caller can decide whether this is a wiring bug or a research
        # FAIL.
        print(
            f"[r8-gate] eval-gate returned non-zero rc={rc}; "
            f"manifest at {manifest_out} (may still be a clean FAIL).",
            file=sys.stderr,
            flush=True,
        )
        return rc

    print(f"[r8-gate] PASS — manifest at {manifest_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
