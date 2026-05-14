"""R8 DPO trainer.

Bradley-Terry pairwise loss against a frozen reference policy:

    loss = -log_sigmoid( beta * ( (logp_w - logp_l) - (refp_w - refp_l) ) )

Implementation per `docs/ai-research/scoping/r8-dpo.md`:
    § 3.3 loss + beta=0.1 default
    § 3.4 dataset shape (one model forward + index gather per pair side)
    § 3.5 reference policy = frozen item17 warm-start

CLI mirrors `train_bc.py` for the shared knobs (--epochs / --batch-size /
--hidden-dim / --depth / --device / --out-dir / --init-from-checkpoint)
and adds --beta and --reference-checkpoint.

Value head is *not* optimised (scoping § 3.3 last bullet) — DPO is a
policy-only objective. We preserve the warm-start's value head verbatim
in the saved checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

# Reuse shared helpers from train_bc.py — the masked log-softmax, the
# warm-start loader, the ONNX roundtrip smoke and the device resolver are
# all DPO-trainer dependencies and copying them would be drift.
from train_bc import (
    feature_schema_metadata,
    load_init_from_checkpoint,
    masked_log_softmax_logits,
    normalized_weights,
    resolve_device,
    run_onnx_roundtrip_smoke,
    weighted_mean,
)
from pair_corpus import (
    PreferencePairDataset,
    collate_preference_batch,
)
from uma_ai.model import CandidatePolicyNet, ModelConfig


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = PreferencePairDataset(
        args.data,
        tau=args.tau if args.tau >= 0 else None,
    )
    train_indices, val_indices = split_pair_indices(len(dataset), args.seed)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_preference_batch,
    )
    val_loader = (
        DataLoader(
            Subset(dataset, val_indices),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_preference_batch,
        )
        if val_indices
        else None
    )

    config = ModelConfig(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
    )
    model = CandidatePolicyNet(config).to(device)
    if args.init_from_checkpoint:
        load_init_from_checkpoint(Path(args.init_from_checkpoint), model)

    reference = load_reference_policy(args.reference_checkpoint, config, device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            reference,
            train_loader,
            optimizer,
            beta=args.beta,
            device=device,
        )
        val_metrics = (
            evaluate(model, reference, val_loader, beta=args.beta, device=device)
            if val_loader
            else {}
        )
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        if args.verbose:
            print(json.dumps(record))

    final_train = evaluate(
        model, reference, train_loader, beta=args.beta, device=device
    )
    final_val = (
        evaluate(model, reference, val_loader, beta=args.beta, device=device)
        if val_loader
        else {}
    )

    rng_state = {
        "torch": torch.get_rng_state().tolist(),
        "cuda": [t.tolist() for t in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }
    checkpoint = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": config.to_dict(),
        "feature_schema": feature_schema_metadata(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": None,
        "scaler_state": None,
        "rng_state": rng_state,
        "next_epoch": args.epochs + 1,
        "history": history,
        "training": {
            "objective": "dpo",
            "data": str(args.data),
            "samples": len(dataset),
            "train_samples": len(train_indices),
            "val_samples": len(val_indices),
            "beta": float(args.beta),
            "tau": float(dataset.tau),
            "tau_diagnostics": dataset.tau_diagnostics,
            "reference_checkpoint": str(args.reference_checkpoint),
            "init_from_checkpoint": str(args.init_from_checkpoint)
            if args.init_from_checkpoint
            else None,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "device": str(device),
            "history": history,
            "final_train": final_train,
            "final_val": final_val,
        },
    }
    torch.save(checkpoint, out_dir / "checkpoint.pt")
    onnx_smoke = run_onnx_roundtrip_smoke(model, config, out_dir, device)
    manifest = {
        "checkpoint": "checkpoint.pt",
        "model_config": config.to_dict(),
        "feature_schema": feature_schema_metadata(),
        "device": str(device),
        "data": str(args.data),
        "samples": len(dataset),
        "objective": "dpo",
        "training_kwargs": {
            "beta": float(args.beta),
            "tau": float(dataset.tau),
            "tau_diagnostics": dataset.tau_diagnostics,
            "reference_checkpoint": str(args.reference_checkpoint),
            "init_from_checkpoint": str(args.init_from_checkpoint)
            if args.init_from_checkpoint
            else None,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
        },
        "onnx_roundtrip_smoke": onnx_smoke,
        "metrics": {"train": final_train, "val": final_val},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf8"
    )
    print(json.dumps({"status": "PASS", "out_dir": str(out_dir), **manifest}, indent=2))


def load_reference_policy(
    path: str,
    fallback_config: ModelConfig,
    device: torch.device,
) -> CandidatePolicyNet:
    """Load the reference (frozen) policy from a checkpoint.

    Mirrors `train_bc.load_kl_anchor` semantics: prefer the saved
    `model_config` so a reference trained at a different (hidden_dim,
    depth) still loads cleanly. Sets eval mode + requires_grad_(False) on
    every parameter so the gradient assertion in the smoke test holds.
    """

    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = ModelConfig.from_dict(payload.get("model_config")) or fallback_config
    reference = CandidatePolicyNet(config).to(device)
    reference.load_state_dict(payload["model_state"])
    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)
    return reference


def dpo_loss_components(
    model: CandidatePolicyNet,
    reference: CandidatePolicyNet,
    batch: dict[str, torch.Tensor],
    *,
    beta: float,
) -> dict[str, torch.Tensor]:
    """Compute DPO loss + diagnostics for one batch.

    The loss is exposed via a helper so the smoke test can assert
    decreasing loss without re-implementing the math.
    """

    state = batch["state_features"]
    actions = batch["action_features"]
    mask = batch["action_mask"]
    y_w = batch["y_w_index"]
    y_l = batch["y_l_index"]
    weights = normalized_weights(batch["sample_weights"])

    logits, _ = model(state, actions, mask)
    log_probs = masked_log_softmax_logits(logits, mask)
    logp_w = log_probs.gather(1, y_w.unsqueeze(1)).squeeze(1)
    logp_l = log_probs.gather(1, y_l.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        ref_logits, _ = reference(state, actions, mask)
        ref_log_probs = masked_log_softmax_logits(ref_logits, mask)
        refp_w = ref_log_probs.gather(1, y_w.unsqueeze(1)).squeeze(1)
        refp_l = ref_log_probs.gather(1, y_l.unsqueeze(1)).squeeze(1)

    logits_diff = (logp_w - logp_l) - (refp_w - refp_l)
    per_row_loss = -F.logsigmoid(beta * logits_diff)
    loss = weighted_mean(per_row_loss, weights)
    # Accuracy: fraction of pairs where the policy already prefers y_w
    # over y_l (a cheap progress signal — at init, this is the reference
    # policy's preference rate).
    accuracy = (logp_w > logp_l).float().mean()
    margin_signed = (logp_w - logp_l).mean()
    return {
        "loss": loss,
        "accuracy": accuracy,
        "policy_margin": margin_signed,
        "logits_diff_mean": logits_diff.mean(),
    }


def run_epoch(
    model: CandidatePolicyNet,
    reference: CandidatePolicyNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    beta: float,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "accuracy": 0.0,
        "policy_margin": 0.0,
        "logits_diff_mean": 0.0,
        "count": 0.0,
    }
    for batch in loader:
        batch = move_batch(batch, device)
        components = dpo_loss_components(model, reference, batch, beta=beta)
        loss = components["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        accumulate_metrics(totals, components, batch_size=batch["y_w_index"].shape[0])
    return finalize_metrics(totals)


@torch.no_grad()
def evaluate(
    model: CandidatePolicyNet,
    reference: CandidatePolicyNet,
    loader: DataLoader | None,
    *,
    beta: float,
    device: torch.device,
) -> dict[str, float]:
    if loader is None:
        return {}
    model.eval()
    totals = {
        "loss": 0.0,
        "accuracy": 0.0,
        "policy_margin": 0.0,
        "logits_diff_mean": 0.0,
        "count": 0.0,
    }
    for batch in loader:
        batch = move_batch(batch, device)
        components = dpo_loss_components(model, reference, batch, beta=beta)
        accumulate_metrics(totals, components, batch_size=batch["y_w_index"].shape[0])
    return finalize_metrics(totals)


def accumulate_metrics(
    totals: dict[str, float],
    components: dict[str, torch.Tensor],
    *,
    batch_size: int,
) -> None:
    count = float(batch_size)
    totals["count"] += count
    for key in ("loss", "accuracy", "policy_margin", "logits_diff_mean"):
        totals[key] += float(components[key].item()) * count


def finalize_metrics(totals: dict[str, float]) -> dict[str, float]:
    count = max(1.0, totals["count"])
    return {
        "loss": totals["loss"] / count,
        "accuracy": totals["accuracy"] / count,
        "policy_margin": totals["policy_margin"] / count,
        "logits_diff_mean": totals["logits_diff_mean"] / count,
        "samples": totals["count"],
    }


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def split_pair_indices(size: int, seed: int) -> tuple[list[int], list[int]]:
    """Row-level train/val split.

    DPO has no episode-grouped split mode in v1: the corpus comes from a
    single outcome-export run with seed-based episodes, and the trainer's
    only consumer for v1 is the gate-eval which runs against the rule-bot
    on a different seed range entirely. A row split is fine.
    """

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(size, generator=generator).tolist()
    val_size = max(1, int(size * 0.2)) if size >= 5 else 0
    return indices[val_size:], indices[:val_size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DPO policy from rollout-CRN preference pairs."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--reference-checkpoint",
        required=True,
        help="Path to the frozen reference policy checkpoint (DPO anchor).",
    )
    parser.add_argument(
        "--init-from-checkpoint",
        default=None,
        help="Path to checkpoint.pt to warm-start the trainable policy. For "
        "v1 this is the same path as --reference-checkpoint (log-ratio "
        "term starts at 0).",
    )
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument(
        "--tau",
        type=float,
        default=-1.0,
        help="Override the noise-floor on `selectedVsRunnerUpMargin`. "
        "Negative (default) = compute via 100-row probe per scoping § 5(a).",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
