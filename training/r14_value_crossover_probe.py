"""R14.I.1 value-head crossover probe.

Measures how closely a checkpoint's value head tracks the rollout-CRN
estimator that produced its training corpus's `rootValue`. The W8
postmortem said the value-head leaf is too noisy to drive selfplay
distillation; the value head only becomes a viable cheap-selfplay
leaf evaluator once its predictions are statistically interchangeable
with rollout-CRN K=3 means.

We define a binary "crossed" criterion gating R14.I.3:

    crossed = (val_mse <= ratio_cap * mse_floor) AND (pearson_r >= 0.7)

Where:
  - val_mse is the per-row MSE between the value head's prediction
    (greedy forward pass over the row's state + legalActions) and the
    row's `rootValue` (rollout-CRN K=3 mean at the same root).
  - mse_floor is the W3 retrain's final val_loss (the rollout-leaf
    selfplay corpus's irreducible noise floor at the same evaluator).
  - ratio_cap defaults to 1.10 — the value head must come within 10%
    of the W3 noise floor to count as "crossed", a fairly conservative
    bar that resists single-iteration flukes.
  - pearson_r >= 0.7 catches the failure where MSE is low because the
    head predicts ~mean for every row but disagrees on ranking. Without
    correlation the leaf evaluator can't disambiguate sibling actions.

Reuses ValueTargetDataset + collate_mcts_selfplay_batch from the W3
plumbing. Outputs a single JSON blob with n, val_mse, rmse, pearson_r,
mse_floor, ratio, crossed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from uma_ai.model import CandidatePolicyNet, ModelConfig
from uma_ai.selfplay_dataset import collate_mcts_selfplay_batch
from uma_ai.value_target_dataset import ValueTargetDataset


W3_DEFAULT_MANIFEST = "runs/R13-value-retrain-experiment/retrain/value-retrain.manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to probe (PyTorch .pt).")
    parser.add_argument("--data", required=True, help="Rollout-leaf MCTS selfplay jsonl (rows with rootValue).")
    parser.add_argument(
        "--mse-floor",
        type=float,
        default=None,
        help="MSE floor (irreducible per-row variance, e.g. W3 retrain final val_loss). "
        "If omitted, falls back to --mse-floor-manifest's `val.loss`.",
    )
    parser.add_argument(
        "--mse-floor-manifest",
        default=W3_DEFAULT_MANIFEST,
        help="Path to a value-retrain manifest with `val.loss` (used when --mse-floor is omitted).",
    )
    parser.add_argument("--ratio-cap", type=float, default=1.10, help="Crossed iff val_mse <= ratio_cap * mse_floor.")
    parser.add_argument("--pearson-min", type=float, default=0.7, help="Crossed iff pearson_r >= pearson_min.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None, help="If set, write the result JSON to this path in addition to stdout.")
    parser.add_argument("--state-dim", type=int, default=110,
                        help="Feature width for the value-target corpus. Default 110 (v3.0).")
    parser.add_argument("--uma-slot-tokens", action="store_true",
                        help="Emit and forward v3.2 per-Uma slot tensors during the probe.")
    return parser.parse_args()


def resolve_mse_floor(args: argparse.Namespace, repo_root: Path) -> float:
    if args.mse_floor is not None:
        return float(args.mse_floor)
    manifest_path = Path(args.mse_floor_manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    if not manifest_path.exists():
        raise SystemExit(f"--mse-floor not given and manifest not found at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    val = manifest.get("val") or {}
    floor = val.get("loss")
    if floor is None:
        raise SystemExit(f"Manifest {manifest_path} has no val.loss")
    return float(floor)


def load_model(checkpoint_path: str, device: torch.device) -> CandidatePolicyNet:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_cfg = payload.get("model_config") or payload.get("metadata", {}).get("model_config")
    config = ModelConfig.from_dict(raw_cfg or {})
    model = CandidatePolicyNet(config)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    device = torch.device(args.device)

    mse_floor = resolve_mse_floor(args, repo_root)
    dataset = ValueTargetDataset(
        args.data,
        state_dim=args.state_dim,
        uses_uma_slot_tokens=bool(args.uma_slot_tokens),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_mcts_selfplay_batch,
        drop_last=False,
    )

    model = load_model(args.checkpoint, device)

    preds: list[float] = []
    targets: list[float] = []
    with torch.no_grad():
        for batch in loader:
            state = batch["state_features"].to(device)
            actions = batch["action_features"].to(device)
            mask = batch["action_mask"].to(device)
            value_target = batch["value_targets"].to(device)
            # R16-P0: forward the optional v3 card-embedding tensors so
            # the probe scores through the same trunk the model was
            # trained with. Absent on legacy/compat batches → None.
            card_ids_by_zone = batch.get("card_ids_by_zone")
            action_card_idx = batch.get("action_card_idx")
            if card_ids_by_zone is not None:
                card_ids_by_zone = card_ids_by_zone.to(device)
            if action_card_idx is not None:
                action_card_idx = action_card_idx.to(device)
            uma_slot_card_ids = batch.get("uma_slot_card_ids")
            uma_slot_features = batch.get("uma_slot_features")
            if uma_slot_card_ids is not None:
                uma_slot_card_ids = uma_slot_card_ids.to(device)
            if uma_slot_features is not None:
                uma_slot_features = uma_slot_features.to(device)
            _logits, value_pred = model(
                state,
                actions,
                mask,
                card_ids_by_zone=card_ids_by_zone,
                action_card_idx=action_card_idx,
                uma_slot_card_ids=uma_slot_card_ids,
                uma_slot_features=uma_slot_features,
            )
            preds.extend(value_pred.detach().cpu().tolist())
            targets.extend(value_target.detach().cpu().tolist())

    n = len(preds)
    if n < 8:
        raise SystemExit(f"Too few samples for a stable probe (n={n})")

    diffs = [p - t for p, t in zip(preds, targets)]
    sq = [d * d for d in diffs]
    val_mse = sum(sq) / n
    rmse = math.sqrt(val_mse)

    mean_p = sum(preds) / n
    mean_t = sum(targets) / n
    cov = sum((p - mean_p) * (t - mean_t) for p, t in zip(preds, targets)) / n
    var_p = sum((p - mean_p) ** 2 for p in preds) / n
    var_t = sum((t - mean_t) ** 2 for t in targets) / n
    denom = math.sqrt(var_p * var_t)
    pearson_r = cov / denom if denom > 0 else 0.0

    ratio = val_mse / mse_floor if mse_floor > 0 else float("inf")
    crossed = (val_mse <= args.ratio_cap * mse_floor) and (pearson_r >= args.pearson_min)

    result = {
        "n": n,
        "val_mse": val_mse,
        "rmse": rmse,
        "pearson_r": pearson_r,
        "mse_floor": mse_floor,
        "ratio": ratio,
        "ratio_cap": args.ratio_cap,
        "pearson_min": args.pearson_min,
        "crossed": crossed,
        "checkpoint": args.checkpoint,
        "data": args.data,
        "state_dim": args.state_dim,
        "uma_slot_tokens": bool(args.uma_slot_tokens),
        "mean_pred": mean_p,
        "mean_target": mean_t,
        "std_pred": math.sqrt(var_p),
        "std_target": math.sqrt(var_t),
    }

    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf8")


if __name__ == "__main__":
    main()
