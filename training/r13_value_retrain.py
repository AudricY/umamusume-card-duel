"""R13.W3 value head retrain — the dispositive experiment.

Loads an init checkpoint (R4 by default), freezes the trunk + policy head,
and retrains only the value head against `rootValue` (mean of K rule-bot
CRN rollouts at the root state) read from rollout-leaf MCTS selfplay rows.

The hypothesis: regressing the value head to rollout means produces a
lower-variance target than the backfilled game-z, fixing the bottleneck
R12 found at the leaf evaluator. If the retrained head, run at `--mcts-leaf
value-head` in the gate, clears Wilson lower 0.40, then Phase D (W6) is
viable. If not, search-at-inference is the permanent production path.

Loss is plain MSE on (value_pred - value_target). Tanh saturation at ±1 is
left in the model so predictions stay in [-1, 1]; targets are also clipped.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from uma_ai.model import CandidatePolicyNet, ModelConfig
from uma_ai.selfplay_dataset import collate_mcts_selfplay_batch
from uma_ai.value_target_dataset import ValueTargetDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to mcts-selfplay.jsonl with rollout-leaf rows (rootValue must be set).")
    parser.add_argument("--init-checkpoint", required=True, help="Checkpoint to initialize from (typically R4).")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--events-out", default=None, help="If set, writes per-epoch events to this jsonl path.")
    parser.add_argument("--state-dim", type=int, default=110,
                        help="Feature width for the value-target corpus. Default 110 (v3.0).")
    parser.add_argument("--uma-slot-tokens", action="store_true",
                        help="Emit v3.2 per-Uma slot tensors while retraining the value head.")
    parser.add_argument("--train-scope", choices=["value-head", "all"], default="value-head",
                        help="Train only value_head.* (default) or all model parameters for fit diagnostics.")
    parser.add_argument("--policy-anchor-weight", type=float, default=0.0,
                        help="When >0, add KL(anchor_policy || model_policy) on legal actions.")
    parser.add_argument("--policy-anchor-checkpoint", default=None,
                        help="Checkpoint to use as the policy anchor. Defaults to --init-checkpoint.")
    parser.add_argument("--model-variant", choices=["mlp", "set_attention"], default=None,
                        help="Optionally override model_config.model_variant before loading weights.")
    return parser.parse_args()


def freeze_trunk_and_policy(model: CandidatePolicyNet) -> None:
    for name, param in model.named_parameters():
        if name.startswith("value_head."):
            param.requires_grad = True
        else:
            param.requires_grad = False
    # Drop dropout from trunk so frozen forward is deterministic — value head
    # training shouldn't fight stochastic features. Eval mode on the whole
    # model accomplishes both that and disabling LayerNorm running stats
    # updates in the frozen branch; we'll flip value_head back to train mode
    # below.
    model.eval()
    for module in model.value_head.modules():
        module.train()


def configure_train_scope(model: CandidatePolicyNet, train_scope: str) -> None:
    if train_scope == "value-head":
        freeze_trunk_and_policy(model)
        return
    if train_scope == "all":
        for param in model.parameters():
            param.requires_grad = True
        model.train()
        return
    raise ValueError(f"unknown train_scope={train_scope!r}")


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(~mask, -1e9)
    return torch.log_softmax(masked, dim=1)


def masked_policy_kl(anchor_logits: torch.Tensor, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    anchor_log_probs = masked_log_softmax(anchor_logits, mask)
    log_probs = masked_log_softmax(logits, mask)
    anchor_probs = anchor_log_probs.exp() * mask.float()
    return (anchor_probs * (anchor_log_probs - log_probs)).sum(dim=1)


def load_init_state(model: CandidatePolicyNet, state: dict[str, torch.Tensor], *, allow_new_prefixes: tuple[str, ...]) -> None:
    if not allow_new_prefixes:
        model.load_state_dict(state)
        return
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = list(unexpected)
    disallowed_missing = [
        key for key in missing
        if not any(str(key).startswith(prefix) for prefix in allow_new_prefixes)
    ]
    if unexpected or disallowed_missing:
        raise RuntimeError(
            "init checkpoint does not match requested model variant: "
            f"unexpected={unexpected} disallowed_missing={disallowed_missing}"
        )


def emit_event(events_path: Path | None, stage: str, event_type: str, data: dict) -> None:
    if events_path is None:
        return
    obj = {
        "stage": stage,
        "event_type": event_type,
        "ts": time.time(),
        "data": data,
    }
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf8") as fh:
        fh.write(json.dumps(obj) + "\n")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    events_path = Path(args.events_out) if args.events_out else None
    emit_event(events_path, "r13-value-retrain", "run_started", {
        "data": args.data,
        "init_checkpoint": args.init_checkpoint,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "state_dim": args.state_dim,
        "uma_slot_tokens": bool(args.uma_slot_tokens),
        "train_scope": args.train_scope,
        "policy_anchor_weight": args.policy_anchor_weight,
        "policy_anchor_checkpoint": args.policy_anchor_checkpoint or args.init_checkpoint,
        "model_variant": args.model_variant,
    })

    dataset = ValueTargetDataset(
        args.data,
        state_dim=args.state_dim,
        uses_uma_slot_tokens=bool(args.uma_slot_tokens),
    )
    n_total = len(dataset)
    n_val = max(1, int(round(n_total * args.val_fraction)))
    n_train = max(1, n_total - n_val)
    gen = torch.Generator().manual_seed(args.seed)
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val], generator=gen)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_mcts_selfplay_batch, drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_mcts_selfplay_batch, drop_last=False,
    )

    # Init checkpoint and matching ModelConfig. R4-era checkpoints store
    # model_config at the top level; train_bc.py since then nests it under
    # `metadata`. Check both locations so this script works against any
    # historical init checkpoint without manual surgery.
    payload = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    raw_cfg = payload.get("model_config")
    if raw_cfg is None:
        raw_cfg = payload.get("metadata", {}).get("model_config")
    cfg_dict = dict(raw_cfg or {})
    source_model_variant = str(cfg_dict.get("model_variant", "mlp"))
    if args.model_variant is not None:
        cfg_dict["model_variant"] = args.model_variant
    config = ModelConfig.from_dict(cfg_dict)
    model = CandidatePolicyNet(config)
    new_prefixes: tuple[str, ...] = ()
    if args.model_variant is not None and args.model_variant != source_model_variant:
        new_prefixes = ("set_attention_encoder.",)
    load_init_state(model, payload["model_state"], allow_new_prefixes=new_prefixes)
    model.to(device)
    configure_train_scope(model, args.train_scope)

    anchor_model: CandidatePolicyNet | None = None
    if args.policy_anchor_weight > 0.0:
        anchor_payload = torch.load(args.policy_anchor_checkpoint or args.init_checkpoint, map_location="cpu", weights_only=False)
        anchor_raw_cfg = anchor_payload.get("model_config")
        if anchor_raw_cfg is None:
            anchor_raw_cfg = anchor_payload.get("metadata", {}).get("model_config")
        anchor_config = ModelConfig.from_dict(anchor_raw_cfg or {})
        anchor_model = CandidatePolicyNet(anchor_config)
        anchor_model.load_state_dict(anchor_payload["model_state"])
        anchor_model.to(device)
        anchor_model.eval()
        for param in anchor_model.parameters():
            param.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    mse = nn.MSELoss(reduction="none")

    def run_pass(loader: DataLoader, training: bool) -> dict:
        total = 0
        loss_sum = 0.0
        abs_err_sum = 0.0
        squared_err_sum = 0.0
        kl_sum = 0.0
        if training:
            if args.train_scope == "value-head":
                for module in model.value_head.modules():
                    module.train()
            else:
                model.train()
        else:
            model.eval()
        for batch in loader:
            state = batch["state_features"].to(device)
            actions = batch["action_features"].to(device)
            mask = batch["action_mask"].to(device)
            value_target = batch["value_targets"].to(device)
            sample_weights = batch["sample_weights"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            # R16-P0: forward the optional v3 card-embedding tensors when
            # the loader emitted them (v3 corpora) so the value retrain
            # runs through the same trunk as mcts-distill. Absent
            # (legacy/compat batches) → None, embedding branch inert.
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
            with torch.set_grad_enabled(training):
                logits, value_pred = model(
                    state,
                    actions,
                    mask,
                    card_ids_by_zone=card_ids_by_zone,
                    action_card_idx=action_card_idx,
                    uma_slot_card_ids=uma_slot_card_ids,
                    uma_slot_features=uma_slot_features,
                )
                # Plain weighted MSE — rootValue is a regression target,
                # not a class label. tanh saturation in the head guards
                # against overshoot; clipped targets bound the loss.
                per_row = mse(value_pred, value_target) * sample_weights
                value_loss = per_row.mean()
                kl_loss = torch.zeros((), device=device)
                if anchor_model is not None and args.policy_anchor_weight > 0.0:
                    with torch.no_grad():
                        anchor_logits, _ = anchor_model(
                            state,
                            actions,
                            mask,
                            card_ids_by_zone=card_ids_by_zone,
                            action_card_idx=action_card_idx,
                            uma_slot_card_ids=uma_slot_card_ids,
                            uma_slot_features=uma_slot_features,
                        )
                    kl_per_row = masked_policy_kl(anchor_logits, logits, mask) * sample_weights
                    kl_loss = kl_per_row.mean()
                loss = value_loss + float(args.policy_anchor_weight) * kl_loss
                if training:
                    loss.backward()
                    optimizer.step()
            batch_size = state.size(0)
            total += batch_size
            loss_sum += float(loss.detach().cpu()) * batch_size
            kl_sum += float(kl_loss.detach().cpu()) * batch_size
            err = (value_pred.detach() - value_target.detach()).cpu()
            abs_err_sum += float(err.abs().sum())
            squared_err_sum += float((err ** 2).sum())
        return {
            "loss": loss_sum / max(1, total),
            "mae": abs_err_sum / max(1, total),
            "rmse": (squared_err_sum / max(1, total)) ** 0.5,
            "policy_kl": kl_sum / max(1, total),
            "n": total,
        }

    history: list[dict] = []
    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_metrics = run_pass(train_loader, training=True)
        val_metrics = run_pass(val_loader, training=False)
        epoch_secs = time.time() - epoch_start
        history.append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics, "secs": epoch_secs})
        emit_event(events_path, "r13-value-retrain", "epoch_completed", {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "train_policy_kl": train_metrics.get("policy_kl", 0.0),
            "val_policy_kl": val_metrics.get("policy_kl", 0.0),
            "secs": round(epoch_secs, 2),
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save checkpoint in the same shape train_bc.py uses so serve_onnx/export_onnx work.
    # export_onnx.py reads top-level `model_config`; keep it both there and
    # nested in metadata so either consumer works.
    checkpoint = {
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "model_config": config.to_dict(),
        "metadata": {
            "model_config": config.to_dict(),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "data": args.data,
            "init_checkpoint": args.init_checkpoint,
            "device": str(device),
            "frozen": "trunk+policy_head" if args.train_scope == "value-head" else "none",
            "train_scope": args.train_scope,
            "policy_anchor_weight": args.policy_anchor_weight,
            "policy_anchor_checkpoint": args.policy_anchor_checkpoint or args.init_checkpoint,
            "source_model_variant": source_model_variant,
            "model_variant": config.model_variant,
            "state_dim": args.state_dim,
            "uma_slot_tokens": bool(args.uma_slot_tokens),
            "history": history,
        },
    }
    torch.save(checkpoint, out_dir / "checkpoint.pt")

    final = {
        "samples": len(dataset),
        "train": history[-1]["train"] if history else None,
        "val": history[-1]["val"] if history else None,
        "out_dir": str(out_dir),
        "init_checkpoint": args.init_checkpoint,
    }
    (out_dir / "value-retrain.manifest.json").write_text(json.dumps(final, indent=2), encoding="utf8")
    emit_event(events_path, "r13-value-retrain", "run_completed", final)
    print(json.dumps({"status": "PASS", **final}, indent=2))


if __name__ == "__main__":
    main()
