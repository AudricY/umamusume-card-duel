"""Calibration metrics for the trained value head vs. a point-margin baseline.

Item 7 (v4.1 backlog): ECE / Brier / reliability-by-turn for the value head and
a point-margin baseline (logistic on `own.points - opponent.points`), plus a
Wilson-style bootstrap CI on the per-row Brier lift driving a tiebreaker-grade
PASS/FAIL gate. CPU-only by default; no numpy/scipy.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from train_bc import resolve_device
from uma_ai.dataset import JsonlPolicyDataset, collate_policy_batch
from uma_ai.model import CandidatePolicyNet, ModelConfig

TURN_BUCKETS: list[tuple[str, int, int]] = [
    ("1-3", 1, 3), ("4-6", 4, 6), ("7-9", 7, 9), ("10+", 10, 10**9),
]


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dataset = JsonlPolicyDataset(args.data, min_actions=2, strict_schema_version=not args.allow_loose_schema)

    rows = collect_rows(dataset)
    if not rows:
        raise SystemExit(f"No labeled rows (winner != null) in {args.data}; cannot calibrate")

    value_preds_all = run_value_head(args.checkpoint, dataset, rows, device=device)

    # Hold out half of labeled rows from the logistic fit so reported metrics
    # are not measured on the same data the point-margin k was tuned on.
    fit_idx, eval_idx = stable_split(len(rows), args.seed, args.fit_fraction)
    margins = [r["margin"] for r in rows]
    k = fit_logistic_k([margins[i] for i in fit_idx], [rows[i]["outcome"] for i in fit_idx])
    pm_preds_all = [sigmoid(k * m) for m in margins]

    eval_rows = [rows[i] for i in eval_idx]
    eval_value = [value_preds_all[i] for i in eval_idx]
    eval_pm = [pm_preds_all[i] for i in eval_idx]

    value_metrics = compute_metrics(eval_rows, eval_value)
    pm_metrics = compute_metrics(eval_rows, eval_pm)

    # Lift = improvement of value head over baseline on Brier (positive = value
    # head's squared error is lower). Gate: bootstrap CI lower bound > 0.
    lift_per_row = [
        brier_pair(r["outcome"], p) - brier_pair(r["outcome"], v)
        for r, v, p in zip(eval_rows, eval_value, eval_pm)
    ]
    lift_summary = bootstrap_ci(lift_per_row, args.bootstrap, args.seed)

    drift = (drift_envelope(args.checkpoints, dataset, rows, device)
             if args.checkpoints else {"status": "skipped", "reason": "no --checkpoints provided"})
    variance = {
        "value_variance": variance_of(eval_value),
        "point_margin_variance": variance_of(eval_pm),
    }
    variance["value_unbounded_flag"] = bool(
        variance["value_variance"] > variance["point_margin_variance"] * 5.0
    )

    gate_pass = lift_summary["lower_2_5"] > 0
    output = {
        "data": str(args.data),
        "checkpoint": str(args.checkpoint),
        "n_rows": len(eval_rows),
        "n_fit_rows": len(fit_idx),
        "point_margin_k": k,
        "value_head": value_metrics,
        "point_margin": pm_metrics,
        "brier_lift": {
            "definition": "point_margin_brier - value_brier (positive = value head better)",
            "mean": lift_summary["mean"],
            "lower_2_5": lift_summary["lower_2_5"],
            "upper_97_5": lift_summary["upper_97_5"],
            "wilson_significant_lift": gate_pass,
            "bootstrap_resamples": args.bootstrap,
        },
        "monotone_calibration": monotone_calibration(eval_rows, eval_value),
        "drift": drift,
        "variance": variance,
        "status": "PASS" if gate_pass else "FAIL",
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf8")
    print(json.dumps({
        "status": output["status"], "out": str(out_path),
        "value_brier": value_metrics["brier"], "point_margin_brier": pm_metrics["brier"],
        "lift_mean": lift_summary["mean"],
        "lift_ci_2_5": lift_summary["lower_2_5"], "lift_ci_97_5": lift_summary["upper_97_5"],
        "value_ece": value_metrics["ece"], "point_margin_ece": pm_metrics["ece"],
        "value_max_bucket_ece": value_metrics["max_bucket_ece"],
    }))


def collect_rows(dataset: JsonlPolicyDataset) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(dataset.samples):
        example = sample.example
        observation = example.get("observation", {}) or {}
        side_id = example.get("sideId")
        winner = (example.get("result") or {}).get("winner")
        if winner not in ("player", "opponent") or side_id not in ("player", "opponent"):
            continue
        own = (observation.get("own") or {}).get("points")
        opp = (observation.get("opponent") or {}).get("points")
        if own is None or opp is None:
            continue
        rows.append({
            "dataset_index": index,
            "outcome": 1.0 if winner == side_id else 0.0,
            "margin": float(own) - float(opp),
            "turn": int(observation.get("turnNumber") or 1),
        })
    return rows


def run_value_head(
    checkpoint_path: str, dataset: JsonlPolicyDataset, rows: list[dict[str, Any]],
    *, device: torch.device, batch_size: int = 64,
) -> list[float]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ModelConfig.from_dict(payload.get("model_config"))
    model = CandidatePolicyNet(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    indices = [r["dataset_index"] for r in rows]
    preds: list[float] = [0.0] * len(indices)
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start:start + batch_size]
            batch = collate_policy_batch([dataset.samples[i] for i in chunk])
            batch = {k: v.to(device) for k, v in batch.items()}
            _, values = model(batch["state_features"], batch["action_features"], batch["action_mask"])
            for offset, value in enumerate(values.detach().cpu().tolist()):
                # Tanh-bounded value in [-1, 1] -> P(win) in [0, 1].
                preds[start + offset] = max(0.0, min(1.0, (value + 1.0) / 2.0))
    return preds


def compute_metrics(rows: list[dict[str, Any]], preds: list[float]) -> dict[str, Any]:
    outcomes = [r["outcome"] for r in rows]
    ece = expected_calibration_error(preds, outcomes, bins=15)
    brier = sum(brier_pair(o, p) for o, p in zip(outcomes, preds)) / max(1, len(outcomes))
    bucket_eces: dict[str, dict[str, float]] = {}
    bucket_errors: list[float] = []
    for label, lo, hi in TURN_BUCKETS:
        bp = [p for p, r in zip(preds, rows) if lo <= r["turn"] <= hi]
        bo = [r["outcome"] for r in rows if lo <= r["turn"] <= hi]
        if not bp:
            bucket_eces[label] = {"ece": 0.0, "n": 0}
            continue
        e = expected_calibration_error(bp, bo, bins=10)
        bucket_eces[label] = {"ece": e, "n": len(bp)}
        bucket_errors.append(e)
    return {
        "ece": ece, "brier": brier,
        "reliability_by_turn": bucket_eces,
        "max_bucket_ece": max(bucket_errors) if bucket_errors else 0.0,
    }


def expected_calibration_error(preds: list[float], outcomes: list[float], *, bins: int) -> float:
    if not preds:
        return 0.0
    counts = [0] * bins
    sum_pred = [0.0] * bins
    sum_outcome = [0.0] * bins
    for p, o in zip(preds, outcomes):
        b = min(bins - 1, max(0, int(p * bins)))
        counts[b] += 1
        sum_pred[b] += p
        sum_outcome[b] += o
    n = len(preds)
    return sum((c / n) * abs(sp / c - so / c) for c, sp, so in zip(counts, sum_pred, sum_outcome) if c > 0)


def brier_pair(outcome: float, pred: float) -> float:
    return (pred - outcome) * (pred - outcome)


def fit_logistic_k(margins: list[float], outcomes: list[float], *, max_iter: int = 400) -> float:
    """Fit P(win) = sigmoid(k * margin) by L-BFGS on log loss."""
    if not margins:
        return 0.0
    x = torch.tensor(margins, dtype=torch.float64)
    y = torch.tensor(outcomes, dtype=torch.float64)
    k = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([k], lr=0.1, max_iter=max_iter, tolerance_grad=1e-7,
                            line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(k * x, y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(k.detach().item())


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def stable_split(n: int, seed: int, fit_fraction: float) -> tuple[list[int], list[int]]:
    """Deterministic shuffle then split; eval slice is always non-empty."""
    permuted = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    fit_count = max(1, min(n - 1, int(round(n * fit_fraction)))) if n >= 2 else 1
    fit_idx = sorted(permuted[:fit_count])
    eval_idx = sorted(permuted[fit_count:]) or list(range(n))
    return fit_idx, eval_idx


def bootstrap_ci(values: list[float], resamples: int, seed: int) -> dict[str, float]:
    """Percentile bootstrap CI on the mean. Wilson-style by spec but a closed-
    form Wilson interval doesn't apply to Brier deltas, so we use the standard
    percentile bootstrap as the closest valid analogue.
    """
    if not values:
        return {"mean": 0.0, "lower_2_5": 0.0, "upper_97_5": 0.0}
    base = torch.tensor(values, dtype=torch.float64)
    mean = float(base.mean().item())
    if resamples <= 0 or len(values) == 1:
        return {"mean": mean, "lower_2_5": mean, "upper_97_5": mean}
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(values), (resamples, len(values)), generator=gen)
    sorted_means, _ = torch.sort(base[idx].mean(dim=1))
    return {
        "mean": mean,
        "lower_2_5": float(sorted_means[int(0.025 * resamples)].item()),
        "upper_97_5": float(sorted_means[min(resamples - 1, int(0.975 * resamples))].item()),
    }


def monotone_calibration(rows: list[dict[str, Any]], preds: list[float], *, bins: int = 10) -> dict[str, Any]:
    """Bin predictions, compute observed P(win) per bin, score monotonicity.

    Strict 0/1 monotone-non-decreasing flag plus Pearson correlation between
    bin index and observed P(win). `passing` requires both.
    """
    counts = [0] * bins
    sum_outcome = [0.0] * bins
    for p, r in zip(preds, rows):
        b = min(bins - 1, max(0, int(p * bins)))
        counts[b] += 1
        sum_outcome[b] += r["outcome"]
    populated = [(i, sum_outcome[i] / counts[i]) for i in range(bins) if counts[i] > 0]
    if len(populated) < 2:
        return {"populated_bins": len(populated), "monotone": True, "correlation": 0.0, "passing": False}
    xs = [float(i) for i, _ in populated]
    ys = [v for _, v in populated]
    xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - xm) ** 2 for x in xs))
    deny = math.sqrt(sum((y - ym) ** 2 for y in ys))
    correlation = (num / (denx * deny)) if denx > 0 and deny > 0 else 0.0
    monotone = all(b >= a for a, b in zip(ys, ys[1:]))
    return {
        "populated_bins": len(populated),
        "monotone": bool(monotone),
        "correlation": correlation,
        "passing": bool(monotone and correlation > 0),
    }


def drift_envelope(
    checkpoints: list[str], dataset: JsonlPolicyDataset,
    rows: list[dict[str, Any]], device: torch.device,
) -> dict[str, Any]:
    means: list[float] = []
    variances: list[float] = []
    for path in checkpoints:
        preds = run_value_head(path, dataset, rows, device=device)
        means.append(sum(preds) / max(1, len(preds)))
        variances.append(variance_of(preds))
    if len(means) < 2:
        return {"status": "skipped", "reason": "fewer than 2 checkpoints"}
    mean_drift = max(means) - min(means)
    sigma = math.sqrt(max(variances)) if max(variances) > 0 else 0.0
    return {
        "status": "computed",
        "checkpoints": list(checkpoints),
        "value_means": means,
        "value_variances": variances,
        "mean_drift": mean_drift,
        "variance_drift": max(variances) - min(variances),
        "envelope_sigma": sigma,
        "passing": bool(mean_drift <= sigma if sigma > 0 else mean_drift == 0.0),
    }


def variance_of(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate value head vs point-margin baseline.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoints", nargs="*", default=None, help="Optional checkpoints for drift envelope.")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--fit-fraction", type=float, default=0.5, help="Fraction of labeled rows used to fit point-margin k.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--allow-loose-schema", action="store_true", help="Disable strict schemaVersion enforcement.")
    return p.parse_args()


if __name__ == "__main__":
    main()
