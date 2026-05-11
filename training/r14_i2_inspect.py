"""R14.I.2 inspector: read orchestrator-state.json and print sprint verdict.

Reads runs/R13-W6-phase-d/orchestrator-state.json and produces a clean
summary of:
  - Per-iter Wilson lower trajectory + delta vs prev
  - Per-iter crossover probe result (val_mse, pearson_r, ratio, crossed)
  - Sprint exit-criterion evaluation:
      (a) strongest config Wilson lower >= 0.60  → "i_success"
      (b) per-iter Wilson delta < +0.01           → "i_plateau"
      (c) two consecutive crossed=true            → "i3_unlocked"
      otherwise                                    → "continue"
  - Recommended next action.

Pure read-only. No side effects.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state",
        default="runs/R13-W6-phase-d/orchestrator-state.json",
        help="Path to orchestrator-state.json (default: R13-W6-phase-d).",
    )
    parser.add_argument("--target-wilson-lower", type=float, default=0.60)
    parser.add_argument("--plateau-delta", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    state_path = Path(args.state)
    if not state_path.is_absolute():
        state_path = repo_root / state_path
    if not state_path.exists():
        print(f"[r14-i2-inspect] state not found at {state_path}", file=sys.stderr)
        sys.exit(2)

    payload = json.loads(state_path.read_text(encoding="utf8"))
    iterations = payload.get("iterations", []) or []

    if not iterations:
        print(json.dumps({"status": "EMPTY", "iterations": []}, indent=2))
        return

    table: list[dict] = []
    prev_wilson: float | None = None
    crossed_streak = 0
    max_crossed_streak = 0
    success_iter: int | None = None
    plateau_iter: int | None = None
    i3_unlock_iter: int | None = None

    for entry in iterations:
        wilson = float(entry.get("wilson_lower") or 0)
        crossover = entry.get("crossover") or {}
        crossed = bool(crossover.get("crossed"))
        delta = (wilson - prev_wilson) if prev_wilson is not None else None
        row = {
            "iter": int(entry.get("iteration", -1)),
            "wilson_lower": round(wilson, 4),
            "win_rate": round(float(entry.get("win_rate") or 0), 4),
            "delta_vs_prev": round(delta, 4) if delta is not None else None,
            "selfplay_rows": int(entry.get("selfplay_rows") or 0),
            "promoted": bool(entry.get("promote", False)),
            "crossover": {
                "val_mse": crossover.get("val_mse"),
                "pearson_r": crossover.get("pearson_r"),
                "ratio": crossover.get("ratio"),
                "crossed": crossed,
            } if crossover else None,
        }
        table.append(row)

        if wilson >= args.target_wilson_lower and success_iter is None:
            success_iter = row["iter"]
        if delta is not None and delta < args.plateau_delta and plateau_iter is None:
            plateau_iter = row["iter"]
        if crossed:
            crossed_streak += 1
            max_crossed_streak = max(max_crossed_streak, crossed_streak)
            if crossed_streak >= 2 and i3_unlock_iter is None:
                i3_unlock_iter = row["iter"]
        else:
            crossed_streak = 0

        prev_wilson = wilson

    verdict_reasons: list[str] = []
    next_action: str
    if success_iter is not None:
        verdict_reasons.append(f"i_success at iter {success_iter} (wilson_lower >= {args.target_wilson_lower})")
    if plateau_iter is not None:
        verdict_reasons.append(f"i_plateau at iter {plateau_iter} (delta < {args.plateau_delta})")
    if i3_unlock_iter is not None:
        verdict_reasons.append(f"i3_unlocked at iter {i3_unlock_iter} (2 consecutive crossed=true)")
    if payload.get("halted"):
        verdict_reasons.append(f"halted (reason: {payload.get('halt_reason')})")

    if success_iter is not None:
        next_action = "STOP_SUCCESS — final headline gate at the strongest config, then ship."
    elif i3_unlock_iter is not None:
        next_action = "RUN_I3 — kick a W8-style cheap-selfplay iter to falsify the crossover."
    elif plateau_iter is not None:
        next_action = "PLATEAU — consider D (PPO with V-trace) as the fallback per R14 plan."
    elif payload.get("halted"):
        next_action = "HALTED — investigate the halt_reason; do NOT continue I blindly."
    else:
        next_action = "CONTINUE — run more I iters; gate not yet hit."

    summary = {
        "state_path": str(state_path),
        "promoted_checkpoint": payload.get("promoted_checkpoint"),
        "promoted_wilson_lower": payload.get("promoted_wilson_lower"),
        "iterations": table,
        "max_crossed_streak": max_crossed_streak,
        "exit_signals": verdict_reasons,
        "next_action": next_action,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
