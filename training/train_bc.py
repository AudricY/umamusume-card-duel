from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from events import EventWriter
from uma_ai.dataset import JsonlPolicyDataset, collate_policy_batch
from uma_ai.selfplay_dataset import MctsSelfPlayDataset, collate_mcts_selfplay_batch
from uma_ai.features import ACTION_DIM, ACTION_FEATURE_SCHEMA_VERSION, STATE_DIM, STATE_FEATURE_SCHEMA_VERSION, card_vocab_metadata
from uma_ai.model import CandidatePolicyNet, ModelConfig


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ablations = set(args.ablate)
    if args.data_mode == "mcts-distill":
        # R12 phase C: soft policy target from MCTS visit distribution.
        dataset = MctsSelfPlayDataset(args.data, min_actions=2, ablations=ablations)
        collate_fn = collate_mcts_selfplay_batch
    else:
        dataset = JsonlPolicyDataset(args.data, min_actions=2, ablations=ablations)
        collate_fn = collate_policy_batch
    train_indices, val_indices, split_metadata = split_dataset(dataset, args.seed, args.split_by)
    dataset_summary = summarize_dataset(dataset)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    ) if val_indices else None

    config = ModelConfig(hidden_dim=args.hidden_dim, depth=args.depth, dropout=args.dropout)
    model = CandidatePolicyNet(config).to(device)
    # Item 11/17 KL-anchor anti-forgetting: if --kl-anchor-checkpoint is set,
    # load that checkpoint as a frozen anchor distribution and regularize the
    # current policy toward it via a per-batch KL(anchor || target) penalty.
    # The anchor is the prior promoted iteration so the loop can't drift to
    # a state distribution the prior iteration never visited.
    anchor_model = load_kl_anchor(args.kl_anchor_checkpoint, device) if args.kl_anchor_checkpoint else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args, total_steps=max(1, args.epochs * max(1, len(train_loader))))
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    grad_accum = max(1, int(args.grad_accum))
    start_epoch = 1
    history: list[dict[str, Any]] = []
    resume_metadata: dict[str, Any] | None = None
    if args.resume and args.init_from_checkpoint:
        raise SystemExit("--resume and --init-from-checkpoint are mutually exclusive")
    if args.resume:
        resume_path = Path(args.resume)
        resume_metadata = load_resume(resume_path, model, optimizer, scheduler, scaler)
        start_epoch = int(resume_metadata.get("next_epoch", 1))
        history = list(resume_metadata.get("history", []))
    elif args.init_from_checkpoint:
        load_init_from_checkpoint(Path(args.init_from_checkpoint), model)

    if start_epoch > args.epochs:
        raise SystemExit(
            f"--resume requested start_epoch={start_epoch} but --epochs={args.epochs} "
            "leaves nothing to train. Increase --epochs (cumulative semantics) or "
            "switch to --init-from-checkpoint to start a fresh training run from the "
            "prior model weights."
        )

    events: EventWriter | None = EventWriter(args.events_out) if args.events_out else None
    tb_writer = None
    if args.tb_log_dir:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=args.tb_log_dir)
    if events is not None:
        events.emit(
            iteration=args.events_iteration,
            stage="train",
            event_type="train_run_started",
            data_path=str(args.data),
            samples=len(dataset),
            epochs=args.epochs,
            batch_size=args.batch_size,
            kl_anchor_weight=float(args.kl_anchor_weight),
            kl_anchor_checkpoint=str(args.kl_anchor_checkpoint) if args.kl_anchor_checkpoint else None,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            value_weight=args.value_weight,
            scaler=scaler,
            grad_accum=grad_accum,
            scheduler=scheduler,
            anchor_model=anchor_model,
            kl_anchor_weight=args.kl_anchor_weight,
            entropy_bonus=args.entropy_bonus,
            policy_weight=args.policy_weight,
        )
        val_metrics = evaluate(model, val_loader, value_weight=args.value_weight) if val_loader else {}
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        if args.verbose:
            print(json.dumps(record))
        if events is not None:
            events.emit(
                iteration=args.events_iteration,
                stage="train",
                event_type="epoch",
                epoch=epoch,
                train_loss=train_metrics.get("loss"),
                train_policy_loss=train_metrics.get("policy_loss"),
                train_value_loss=train_metrics.get("value_loss"),
                train_kl_loss=train_metrics.get("kl_loss"),
                train_entropy=train_metrics.get("entropy"),
                train_accuracy=train_metrics.get("accuracy"),
                val_loss=val_metrics.get("loss"),
                val_accuracy=val_metrics.get("accuracy"),
            )
        if tb_writer is not None:
            for key, value in train_metrics.items():
                if isinstance(value, (int, float)) and value == value:  # skip NaN
                    tb_writer.add_scalar(f"train/{key}", float(value), epoch)
            for key, value in (val_metrics or {}).items():
                if isinstance(value, (int, float)) and value == value:
                    tb_writer.add_scalar(f"val/{key}", float(value), epoch)
            tb_writer.flush()

    final_train = evaluate(model, train_loader, value_weight=args.value_weight)
    final_val = evaluate(model, val_loader, value_weight=args.value_weight) if val_loader else {}
    if tb_writer is not None:
        for key, value in final_train.items():
            if isinstance(value, (int, float)) and value == value:
                tb_writer.add_scalar(f"final_train/{key}", float(value), 0)
        for key, value in (final_val or {}).items():
            if isinstance(value, (int, float)) and value == value:
                tb_writer.add_scalar(f"final_val/{key}", float(value), 0)
        tb_writer.flush()
        tb_writer.close()
    diagnostics = {
        "train": evaluate_grouped(model, dataset, train_indices, value_weight=args.value_weight, batch_size=args.batch_size),
        "val": evaluate_grouped(model, dataset, val_indices, value_weight=args.value_weight, batch_size=args.batch_size) if val_indices else {},
    }
    rng_state = {
        "torch": torch.get_rng_state().tolist(),
        "cuda": [tensor.tolist() for tensor in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }
    checkpoint = {
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_config": config.to_dict(),
        "feature_schema": feature_schema_metadata(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "rng_state": rng_state,
        "next_epoch": args.epochs + 1,
        "history": history,
        "training": {
            "data": str(args.data),
            "samples": len(dataset),
            "train_samples": len(train_indices),
            "val_samples": len(val_indices),
            "split": split_metadata,
            "dataset_summary": dataset_summary,
            "feature_ablations": sorted(ablations),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "value_weight": args.value_weight,
            "amp": use_amp,
            "grad_accum": grad_accum,
            "lr_schedule": args.lr_schedule,
            "lr_warmup_steps": args.lr_warmup_steps,
            "resume_from": str(args.resume) if args.resume else None,
            "kl_anchor_checkpoint": str(args.kl_anchor_checkpoint) if args.kl_anchor_checkpoint else None,
            "kl_anchor_weight": float(args.kl_anchor_weight),
            "entropy_bonus": float(args.entropy_bonus),
            "device": str(device),
            "history": history,
            "final_train": final_train,
            "final_val": final_val,
            "diagnostics": diagnostics,
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
        "split": split_metadata,
        "dataset_summary": dataset_summary,
        "feature_ablations": sorted(ablations),
        "training_kwargs": {
            "amp": use_amp,
            "grad_accum": grad_accum,
            "lr_schedule": args.lr_schedule,
            "lr_warmup_steps": args.lr_warmup_steps,
            "resume_from": str(args.resume) if args.resume else None,
            "kl_anchor_checkpoint": str(args.kl_anchor_checkpoint) if args.kl_anchor_checkpoint else None,
            "kl_anchor_weight": float(args.kl_anchor_weight),
            "entropy_bonus": float(args.entropy_bonus),
        },
        "onnx_roundtrip_smoke": onnx_smoke,
        "metrics": {"train": final_train, "val": final_val, "diagnostics": diagnostics},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"status": "PASS", "out_dir": str(out_dir), **manifest}, indent=2))


def build_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace, *, total_steps: int):
    if args.lr_schedule == "none":
        return None
    if args.lr_schedule == "cosine":
        warmup = max(0, int(args.lr_warmup_steps))
        cosine_steps = max(1, total_steps - warmup)
        if warmup > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0 / max(1, warmup), total_iters=warmup)
            cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_steps)
            return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup])
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_steps)
    if args.lr_schedule == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, total_steps // 4), gamma=0.5)
    return None


def load_resume(
    path: Path,
    model: CandidatePolicyNet,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: "torch.cuda.amp.GradScaler | None",
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])
    if "optimizer_state" in payload and payload["optimizer_state"] is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and payload.get("scaler_state") is not None:
        scaler.load_state_dict(payload["scaler_state"])
    rng_state = payload.get("rng_state") or {}
    if isinstance(rng_state.get("torch"), list):
        torch.set_rng_state(torch.tensor(rng_state["torch"], dtype=torch.uint8))
    return payload


def load_init_from_checkpoint(path: Path, model: CandidatePolicyNet) -> None:
    """Warm-start model weights only.

    Unlike ``load_resume``, this does NOT restore optimizer/scheduler/
    scaler/RNG/epoch counter. Each call starts a fresh training run
    that happens to begin from these parameter values. This is the
    correct primitive for DAgger-style outer loops where each
    iteration is a new training run on a new dataset, warm-started
    from the prior promoted checkpoint.
    """

    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])


def run_onnx_roundtrip_smoke(model: CandidatePolicyNet, config: ModelConfig, out_dir: Path, device: torch.device) -> dict[str, Any]:
    """Export a tiny ONNX, run both PyTorch and ONNX, ensure logits agree.

    Promoted checkpoints must be deployable; this is the per-training-run
    guarantee that requirement is met.
    """

    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - exercised only if onnxruntime missing
        return {"status": "skipped", "reason": f"onnxruntime not importable: {exc}"}

    model.eval()
    cpu_model = CandidatePolicyNet(config).cpu()
    cpu_model.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
    cpu_model.eval()
    onnx_path = out_dir / "policy.smoke.onnx"
    state = torch.zeros((1, STATE_DIM), dtype=torch.float32)
    actions = torch.zeros((1, 4, ACTION_DIM), dtype=torch.float32)
    mask = torch.ones((1, 4), dtype=torch.bool)
    torch.onnx.export(
        cpu_model,
        (state, actions, mask),
        onnx_path,
        input_names=["state_features", "action_features", "action_mask"],
        output_names=["logits", "value"],
        dynamic_axes={
            "state_features": {0: "batch"},
            "action_features": {0: "batch", 1: "actions"},
            "action_mask": {0: "batch", 1: "actions"},
            "logits": {0: "batch", 1: "actions"},
            "value": {0: "batch"},
        },
        opset_version=17,
    )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits, onnx_value = session.run(None, {
        "state_features": state.numpy(),
        "action_features": actions.numpy(),
        "action_mask": mask.numpy(),
    })
    with torch.no_grad():
        torch_logits, torch_value = cpu_model(state, actions, mask)
    max_logit_diff = float((torch.from_numpy(onnx_logits) - torch_logits).abs().max())
    max_value_diff = float((torch.from_numpy(onnx_value) - torch_value).abs().max())
    onnx_path.unlink(missing_ok=True)
    if max_logit_diff > 1e-3 or max_value_diff > 1e-3:
        raise RuntimeError(
            f"ONNX roundtrip mismatch: logits {max_logit_diff} value {max_value_diff}"
        )
    return {"status": "PASS", "max_logit_diff": max_logit_diff, "max_value_diff": max_value_diff}


def run_epoch(
    model: CandidatePolicyNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    value_weight: float,
    scaler=None,
    grad_accum: int = 1,
    scheduler=None,
    anchor_model: CandidatePolicyNet | None = None,
    kl_anchor_weight: float = 0.0,
    entropy_bonus: float = 0.0,
    policy_weight: float = 1.0,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "accuracy": 0.0, "count": 0.0, "kl_loss": 0.0, "entropy": 0.0}
    optimizer.zero_grad(set_to_none=True)
    use_amp = scaler is not None
    accum_step = 0
    for batch_index, batch in enumerate(loader):
        batch = move_batch(batch, model)
        autocast_ctx = torch.cuda.amp.autocast() if use_amp else _NullContext()
        with autocast_ctx:
            # R7.b.2 Phase 2: forward the new embedding tensors when present
            # (post-Phase-1 datasets carry them; legacy or test paths may
            # omit them and the model.forward defaults to zero-tensors).
            logits, values = model(
                batch["state_features"],
                batch["action_features"],
                batch["action_mask"],
                card_ids_by_zone=batch.get("card_ids_by_zone"),
                action_card_idx=batch.get("action_card_idx"),
            )
            weights = normalized_weights(batch["sample_weights"])
            policy_targets = batch.get("policy_targets")
            if policy_targets is not None:
                # R12 phase C: soft cross-entropy on the masked log-softmax.
                # Masked positions have action_mask=False and logits forced to
                # -1e9 by `masked_log_softmax`, so they contribute zero to the
                # sum even if policy_targets[i] happens to be > 0.
                log_probs = masked_log_softmax_logits(logits, batch["action_mask"])
                per_row = -(policy_targets * log_probs).sum(dim=1)
                policy_loss = weighted_mean(per_row, weights)
            else:
                policy_loss = weighted_mean(nn.functional.cross_entropy(logits, batch["targets"], reduction="none"), weights)
            value_loss = weighted_mean(nn.functional.mse_loss(values, batch["value_targets"], reduction="none"), weights)
            kl_loss = torch.zeros((), device=logits.device)
            if anchor_model is not None and kl_anchor_weight > 0.0:
                with torch.no_grad():
                    # Forward the same embedding tensors through the anchor
                    # so the KL is computed on the same input distribution
                    # (otherwise the anchor would see zero pooled features
                    # and KL would inflate spuriously).
                    anchor_logits, _ = anchor_model(
                        batch["state_features"],
                        batch["action_features"],
                        batch["action_mask"],
                        card_ids_by_zone=batch.get("card_ids_by_zone"),
                        action_card_idx=batch.get("action_card_idx"),
                    )
                kl_loss = masked_kl_divergence(anchor_logits, logits, batch["action_mask"])
            # R3 entropy bonus: subtract β·H(π) so loss minimization
            # pushes the policy toward higher entropy. The unused-tensor
            # path keeps the metric column populated even when β=0.
            policy_entropy = masked_policy_entropy(logits, batch["action_mask"])
            loss = policy_loss * policy_weight + value_loss * value_weight + kl_loss * kl_anchor_weight - entropy_bonus * policy_entropy
        scaled = loss / max(1, grad_accum)
        if use_amp:
            scaler.scale(scaled).backward()
        else:
            scaled.backward()
        accum_step += 1
        is_last = batch_index == len(loader) - 1
        if accum_step >= grad_accum or is_last:
            if use_amp:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            accum_step = 0
        accumulate(totals, loss, policy_loss, value_loss, logits, batch["targets"], kl_loss=kl_loss, entropy=policy_entropy)
    return finish_metrics(totals)


class _NullContext:
    def __enter__(self):  # noqa: D401, ANN001
        return None

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False


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
        # R7.b.2 Phase 2: forward the embedding tensors when present.
        logits, values = model(
            batch["state_features"],
            batch["action_features"],
            batch["action_mask"],
            card_ids_by_zone=batch.get("card_ids_by_zone"),
            action_card_idx=batch.get("action_card_idx"),
        )
        weights = normalized_weights(batch["sample_weights"])
        policy_targets = batch.get("policy_targets")
        if policy_targets is not None:
            log_probs = masked_log_softmax_logits(logits, batch["action_mask"])
            per_row = -(policy_targets * log_probs).sum(dim=1)
            policy_loss = weighted_mean(per_row, weights)
        else:
            policy_loss = weighted_mean(nn.functional.cross_entropy(logits, batch["targets"], reduction="none"), weights)
        value_loss = weighted_mean(nn.functional.mse_loss(values, batch["value_targets"], reduction="none"), weights)
        loss = policy_loss + value_loss * value_weight
        accumulate(totals, loss, policy_loss, value_loss, logits, batch["targets"])
    return finish_metrics(totals)


@torch.no_grad()
def evaluate_grouped(
    model: CandidatePolicyNet,
    dataset: JsonlPolicyDataset,
    indices: list[int],
    *,
    value_weight: float,
    batch_size: int,
) -> dict[str, dict[str, dict[str, float]]]:
    if not indices:
        return {}
    groups: dict[str, dict[str, list[int]]] = {
        "phase": {},
        "action_kind": {},
        "source": {},
        "margin_bucket": {},
    }
    for index in indices:
        sample = dataset.samples[index]
        add_group(groups["phase"], str(sample.example.get("phase", "unknown")), index)
        add_group(groups["action_kind"], selected_action_kind(sample), index)
        add_group(groups["source"], str(sample.example.get("source") or sample.example.get("policy", "unknown")), index)
        add_group(groups["margin_bucket"], margin_bucket(sample), index)

    return {
        category: {
            name: evaluate(
                model,
                DataLoader(Subset(dataset, group_indices), batch_size=batch_size, shuffle=False, collate_fn=collate_policy_batch),
                value_weight=value_weight,
            )
            for name, group_indices in sorted(category_groups.items())
        }
        for category, category_groups in groups.items()
    }


def summarize_dataset(dataset: JsonlPolicyDataset) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {
        "source": {},
        "policy": {},
        "phase": {},
        "selected_action_kind": {},
        "margin_bucket": {},
    }
    for sample in dataset.samples:
        increment(counts["source"], str(sample.example.get("source") or "unknown"))
        increment(counts["policy"], str(sample.example.get("policy", "unknown")))
        increment(counts["phase"], str(sample.example.get("phase", "unknown")))
        increment(counts["selected_action_kind"], selected_action_kind(sample))
        increment(counts["margin_bucket"], margin_bucket(sample))
    return counts


def increment(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def add_group(groups: dict[str, list[int]], name: str, index: int) -> None:
    groups.setdefault(name, []).append(index)


def selected_action_kind(sample) -> str:
    actions = sample.example.get("legalActions", [])
    target = sample.target_index
    if target < 0 or target >= len(actions):
        return "unknown"
    return str(actions[target].get("kind", "unknown"))


def margin_bucket(sample) -> str:
    margin = sample.example.get("oracle", {}).get("selectedVsRunnerUpMargin")
    if margin is None:
        return "unknown"
    value = float(margin)
    if value < 0.02:
        return "<0.02"
    if value < 0.1:
        return "0.02-0.1"
    return ">=0.1"


def accumulate(
    totals: dict[str, float],
    loss: torch.Tensor,
    policy_loss: torch.Tensor,
    value_loss: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
    kl_loss: torch.Tensor | None = None,
    entropy: torch.Tensor | None = None,
) -> None:
    count = float(targets.shape[0])
    totals["loss"] += float(loss.item()) * count
    totals["policy_loss"] += float(policy_loss.item()) * count
    totals["value_loss"] += float(value_loss.item()) * count
    totals["accuracy"] += float((logits.argmax(dim=1) == targets).float().sum().item())
    totals["count"] += count
    if kl_loss is not None and "kl_loss" in totals:
        totals["kl_loss"] += float(kl_loss.item()) * count
    if entropy is not None and "entropy" in totals:
        totals["entropy"] += float(entropy.item()) * count


def finish_metrics(totals: dict[str, float]) -> dict[str, float]:
    count = max(1.0, totals["count"])
    out = {
        "loss": totals["loss"] / count,
        "policy_loss": totals["policy_loss"] / count,
        "value_loss": totals["value_loss"] / count,
        "accuracy": totals["accuracy"] / count,
        "samples": totals["count"],
    }
    if "kl_loss" in totals:
        out["kl_loss"] = totals["kl_loss"] / count
    if "entropy" in totals:
        out["entropy"] = totals["entropy"] / count
    return out


def masked_log_softmax_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Per-action log-softmax with masked positions forced to a large negative.

    R12 phase C reuses this for the soft-cross-entropy distillation loss:
    `-Σ_a π_target(a) · log_softmax(logits, mask)(a)`. Masked positions get
    log p = -1e9 so `π_target * log_p` is ~0 there even if `π_target[a] = 0`
    (which it should be — the dataset adapter zero-pads).
    """

    mask = action_mask.bool()
    sentinel = torch.full_like(logits, -1.0e9)
    masked = torch.where(mask, logits, sentinel)
    return nn.functional.log_softmax(masked, dim=-1)


def masked_policy_entropy(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Mean entropy H(π) over a batch of mask-aware action distributions.

    R3 / entropy-regularized BC: subtracting β · H(π) from the BC loss
    pushes the trained policy toward higher entropy so the resulting
    warm-start is less peaked. Masked positions are forced to -1e9 so
    they contribute zero probability and zero entropy mass.
    """

    mask = action_mask.bool()
    sentinel = torch.full_like(logits, -1.0e9)
    masked = torch.where(mask, logits, sentinel)
    log_p = nn.functional.log_softmax(masked, dim=-1)
    p = torch.where(mask, log_p.exp(), torch.zeros_like(log_p))
    per_action = -p * log_p.clamp(min=-50.0)
    per_action = torch.where(mask, per_action, torch.zeros_like(per_action))
    per_row = per_action.sum(dim=-1)
    return per_row.mean()


def masked_kl_divergence(anchor_logits: torch.Tensor, target_logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Mean KL(anchor || target) over a batch of mask-aware action distributions.

    Item 11/17 KL anchor: penalizes the *current* policy for diverging from the
    *anchor* (prior promoted iteration). Mask sentinel logits would otherwise
    contribute non-finite KL terms, so masked positions are forced to large
    negative finite logits before the softmax. Forward-KL form means low-prob
    anchor actions are penalized only mildly when the target also assigns them
    low probability — desired behavior.
    """

    mask = action_mask.bool()
    sentinel = torch.full_like(anchor_logits, -1.0e9)
    masked_anchor = torch.where(mask, anchor_logits, sentinel)
    masked_target = torch.where(mask, target_logits, sentinel)
    anchor_log_p = nn.functional.log_softmax(masked_anchor, dim=-1)
    target_log_p = nn.functional.log_softmax(masked_target, dim=-1)
    anchor_p = torch.where(mask, anchor_log_p.exp(), torch.zeros_like(anchor_log_p))
    diff = (anchor_log_p - target_log_p).clamp(min=-50.0, max=50.0)
    per_action = anchor_p * diff
    per_action = torch.where(mask, per_action, torch.zeros_like(per_action))
    per_row = per_action.sum(dim=-1)
    return per_row.mean()


def load_kl_anchor(checkpoint_path: str, device: torch.device) -> CandidatePolicyNet:
    """Load a frozen anchor policy from a checkpoint. Eval mode, no_grad use only."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ModelConfig.from_dict(payload.get("model_config"))
    anchor = CandidatePolicyNet(config).to(device)
    anchor.load_state_dict(payload["model_state"])
    anchor.eval()
    for param in anchor.parameters():
        param.requires_grad_(False)
    return anchor


def normalized_weights(weights: torch.Tensor) -> torch.Tensor:
    mean = weights.mean().clamp_min(1.0e-6)
    return weights / mean


def weighted_mean(losses: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (losses * weights).sum() / weights.sum().clamp_min(1.0e-6)


def feature_schema_metadata() -> dict[str, Any]:
    return {
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "state_feature_schema_version": STATE_FEATURE_SCHEMA_VERSION,
        "action_feature_schema_version": ACTION_FEATURE_SCHEMA_VERSION,
        "card_vocab": card_vocab_metadata(),
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
    parser.add_argument("--ablate", action="append", choices=[
        "state_identity_hashes",
        "state_energy_vectors",
        "state_card_awareness",
        "state_semantic",
        "state_hygiene_v21",
        "action_source_target",
        "action_trainer_semantic",
        "action_tactical",
        "action_semantic",
    ], default=[])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision when CUDA is available.")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr-schedule", choices=["none", "cosine", "step"], default="cosine")
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--resume", default=None, help="Path to a checkpoint.pt to resume FULL training state from (optimizer, scheduler, RNG, epoch counter). For mid-run crash recovery; not for DAgger warm-start.")
    parser.add_argument("--init-from-checkpoint", default=None, help="Path to a checkpoint.pt to warm-start MODEL WEIGHTS only. Fresh optimizer/scheduler/epoch counter. This is the DAgger-iteration warm-start primitive.")
    parser.add_argument("--events-out", default=None,
                        help="Append per-epoch loss events to this JSONL stream (events.jsonl).")
    parser.add_argument("--events-iteration", type=int, default=-1,
                        help="Iteration index recorded on emitted events when orchestrated; -1 for standalone runs.")
    parser.add_argument("--tb-log-dir", default=None,
                        help="Write per-epoch TensorBoard scalars under this directory. View with `tensorboard --logdir <parent>`.")
    parser.add_argument("--entropy-bonus", type=float, default=0.0,
                        help="R3: subtract β·H(π) from BC loss to push the warm-start toward higher entropy. 0 disables (default).")
    parser.add_argument("--kl-anchor-checkpoint", default=None,
                        help="Frozen prior-iteration checkpoint to regularize toward (anti-forgetting).")
    parser.add_argument("--kl-anchor-weight", type=float, default=0.0,
                        help="Per-batch weight on KL(anchor || target). 0 disables. Try 0.05-0.5.")
    parser.add_argument("--data-mode", choices=["bc", "mcts-distill"], default="bc",
                        help="bc: read teacher-labeled rows (hard target index, CE loss). "
                             "mcts-distill: read mcts-selfplay rows (soft visit-distribution target, "
                             "soft cross-entropy loss). R12 phase C.")
    parser.add_argument("--policy-weight", type=float, default=1.0,
                        help="Weight on the policy loss (distill mode uses soft CE).")
    return parser.parse_args()


if __name__ == "__main__":
    main()
