from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from uma_ai.dataset import JsonlPolicyDataset, collate_policy_batch
from uma_ai.features import ACTION_DIM, ACTION_FEATURE_SCHEMA_VERSION, STATE_DIM, STATE_FEATURE_SCHEMA_VERSION
from uma_ai.model import CandidatePolicyNet, ModelConfig


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = JsonlPolicyDataset(args.data, min_actions=2)
    train_indices, val_indices, split_metadata = split_dataset(dataset, args.seed, args.split_by)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_policy_batch,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_policy_batch,
    ) if val_indices else None

    config = ModelConfig(hidden_dim=args.hidden_dim, depth=args.depth, dropout=args.dropout)
    model = CandidatePolicyNet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, value_weight=args.value_weight)
        val_metrics = evaluate(model, val_loader, value_weight=args.value_weight) if val_loader else {}
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        if args.verbose:
            print(json.dumps(record))

    final_train = evaluate(model, train_loader, value_weight=args.value_weight)
    final_val = evaluate(model, val_loader, value_weight=args.value_weight) if val_loader else {}
    checkpoint = {
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_config": config.to_dict(),
        "feature_schema": feature_schema_metadata(),
        "training": {
            "data": str(args.data),
            "samples": len(dataset),
            "train_samples": len(train_indices),
            "val_samples": len(val_indices),
            "split": split_metadata,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "value_weight": args.value_weight,
            "device": str(device),
            "history": history,
            "final_train": final_train,
            "final_val": final_val,
        },
    }
    torch.save(checkpoint, out_dir / "checkpoint.pt")
    manifest = {
        "checkpoint": "checkpoint.pt",
        "model_config": config.to_dict(),
        "feature_schema": feature_schema_metadata(),
        "device": str(device),
        "data": str(args.data),
        "samples": len(dataset),
        "split": split_metadata,
        "metrics": {"train": final_train, "val": final_val},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"status": "PASS", "out_dir": str(out_dir), **manifest}, indent=2))


def run_epoch(
    model: CandidatePolicyNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    value_weight: float,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "accuracy": 0.0, "count": 0.0}
    for batch in loader:
        batch = move_batch(batch, model)
        optimizer.zero_grad(set_to_none=True)
        logits, values = model(batch["state_features"], batch["action_features"], batch["action_mask"])
        weights = normalized_weights(batch["sample_weights"])
        policy_loss = weighted_mean(nn.functional.cross_entropy(logits, batch["targets"], reduction="none"), weights)
        value_loss = weighted_mean(nn.functional.mse_loss(values, batch["value_targets"], reduction="none"), weights)
        loss = policy_loss + value_loss * value_weight
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        accumulate(totals, loss, policy_loss, value_loss, logits, batch["targets"])
    return finish_metrics(totals)


@torch.no_grad()
def evaluate(
    model: CandidatePolicyNet,
    loader: DataLoader | None,
    *,
    value_weight: float,
) -> dict[str, float]:
    if loader is None:
        return {}
    model.eval()
    totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "accuracy": 0.0, "count": 0.0}
    for batch in loader:
        batch = move_batch(batch, model)
        logits, values = model(batch["state_features"], batch["action_features"], batch["action_mask"])
        weights = normalized_weights(batch["sample_weights"])
        policy_loss = weighted_mean(nn.functional.cross_entropy(logits, batch["targets"], reduction="none"), weights)
        value_loss = weighted_mean(nn.functional.mse_loss(values, batch["value_targets"], reduction="none"), weights)
        loss = policy_loss + value_loss * value_weight
        accumulate(totals, loss, policy_loss, value_loss, logits, batch["targets"])
    return finish_metrics(totals)


def accumulate(
    totals: dict[str, float],
    loss: torch.Tensor,
    policy_loss: torch.Tensor,
    value_loss: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    count = float(targets.shape[0])
    totals["loss"] += float(loss.item()) * count
    totals["policy_loss"] += float(policy_loss.item()) * count
    totals["value_loss"] += float(value_loss.item()) * count
    totals["accuracy"] += float((logits.argmax(dim=1) == targets).float().sum().item())
    totals["count"] += count


def finish_metrics(totals: dict[str, float]) -> dict[str, float]:
    count = max(1.0, totals["count"])
    return {
        "loss": totals["loss"] / count,
        "policy_loss": totals["policy_loss"] / count,
        "value_loss": totals["value_loss"] / count,
        "accuracy": totals["accuracy"] / count,
        "samples": totals["count"],
    }


def normalized_weights(weights: torch.Tensor) -> torch.Tensor:
    mean = weights.mean().clamp_min(1.0e-6)
    return weights / mean


def weighted_mean(losses: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (losses * weights).sum() / weights.sum().clamp_min(1.0e-6)


def feature_schema_metadata() -> dict[str, int]:
    return {
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "state_feature_schema_version": STATE_FEATURE_SCHEMA_VERSION,
        "action_feature_schema_version": ACTION_FEATURE_SCHEMA_VERSION,
    }


def move_batch(batch: dict[str, torch.Tensor], model: CandidatePolicyNet) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    return {key: value.to(device) for key, value in batch.items()}


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return resolved


def split_indices(size: int, seed: int) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(size, generator=generator).tolist()
    val_size = max(1, int(size * 0.2)) if size >= 5 else 0
    return indices[val_size:], indices[:val_size]


def split_dataset(dataset: JsonlPolicyDataset, seed: int, split_by: str) -> tuple[list[int], list[int], dict]:
    if split_by == "row":
        train_indices, val_indices = split_indices(len(dataset), seed)
        return train_indices, val_indices, {
            "split_by": "row",
            "seed": seed,
            "train_rows": len(train_indices),
            "val_rows": len(val_indices),
        }

    groups: dict[str, list[int]] = {}
    for index, sample in enumerate(dataset.samples):
        key = group_key(sample.example, split_by)
        groups.setdefault(key, []).append(index)

    generator = torch.Generator().manual_seed(seed)
    group_keys = sorted(groups)
    if group_keys:
        order = torch.randperm(len(group_keys), generator=generator).tolist()
        group_keys = [group_keys[index] for index in order]
    val_group_count = max(1, int(len(group_keys) * 0.2)) if len(group_keys) >= 5 else 0
    val_keys = set(group_keys[:val_group_count])
    train_indices = [index for key in group_keys if key not in val_keys for index in groups[key]]
    val_indices = [index for key in group_keys if key in val_keys for index in groups[key]]
    return train_indices, val_indices, {
        "split_by": split_by,
        "seed": seed,
        "train_rows": len(train_indices),
        "val_rows": len(val_indices),
        "train_groups": sorted(key for key in group_keys if key not in val_keys),
        "val_groups": sorted(val_keys),
    }


def group_key(example: dict, split_by: str) -> str:
    raw = example.get(split_by)
    if raw is None and split_by == "episode":
        raw = example.get("episodeId")
    if raw is None:
        raise ValueError(f"Cannot split by {split_by}: example is missing that field")
    return str(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train candidate-conditioned behavior cloning policy.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split-by", choices=["row", "episode", "seed"], default="episode")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
