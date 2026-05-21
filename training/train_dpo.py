"""R8 DPO trainer.

Bradley-Terry pairwise loss against a frozen reference policy:

    loss = -log_sigmoid( beta * ( (logp_w - logp_l) - (refp_w - refp_l) ) )

Implementation per `docs/ai-research/scoping/archive/r8-dpo.md`:
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
    # R16-TD 3b chunk 3: wrap `collate_preference_batch` so it also stitches
    # in `pair_meta` for the grouped-metrics consumer downstream. Keeping
    # this in the trainer (not the collator) preserves the regression
    # contract enforced by `pair_corpus_smoke.py` (exact `set(batch.keys())`).
    def collate_with_meta(samples: list[Any]) -> dict[str, Any]:
        batch = collate_preference_batch(samples)
        batch["pair_meta"] = extract_pair_meta(samples)
        return batch

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_with_meta,
    )
    val_loader = (
        DataLoader(
            Subset(dataset, val_indices),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_with_meta,
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
) -> dict[str, Any]:
    """Compute DPO loss + diagnostics for one batch.

    The loss is exposed via a helper so the smoke test can assert
    decreasing loss without re-implementing the math.

    R16-TD 3b chunk 3: forwards the v3 card-embedding tensors
    (`card_ids_by_zone`, `action_card_idx`) into BOTH the trainable policy
    AND the frozen reference model when present on the batch. Without
    these, `CandidatePolicyNet.forward` silently falls back to
    `embed(0) = 0` via `padding_idx=0` and the v3 branch is inert. Legacy
    synthetic batches that lack these keys still work (the `.get()` +
    `None` guard preserves the pre-chunk-3 behaviour).

    Additionally returns a ``grouped`` dict that stratifies per-pair loss
    and per-pair accuracy by ``negative_kind`` / ``source_kind`` /
    ``phase`` / ``action_kind`` / ``margin_bucket`` so downstream analysis
    can locate regressions to a stratum. Reporting only — never used in
    the gradient.
    """

    state = batch["state_features"]
    actions = batch["action_features"]
    mask = batch["action_mask"]
    y_w = batch["y_w_index"]
    y_l = batch["y_l_index"]
    weights = normalized_weights(batch["sample_weights"])

    # R16-TD 3b chunk 3: full-padded `card_ids_by_zone` / `action_card_idx`
    # — these are the action-axis-full versions emitted by
    # `collate_preference_batch`, matching the BC collator's contract. The
    # `_w` / `_l` suffixed variants are pre-gathered for downstream
    # winner/loser stratification and are NOT used here because the policy
    # net consumes the full legal-action set before we gather log-probs.
    card_ids_by_zone = batch.get("card_ids_by_zone")
    action_card_idx = batch.get("action_card_idx")

    model_kwargs: dict[str, torch.Tensor] = {}
    if card_ids_by_zone is not None and action_card_idx is not None:
        model_kwargs["card_ids_by_zone"] = card_ids_by_zone
        model_kwargs["action_card_idx"] = action_card_idx

    logits, _ = model(state, actions, mask, **model_kwargs)
    log_probs = masked_log_softmax_logits(logits, mask)
    logp_w = log_probs.gather(1, y_w.unsqueeze(1)).squeeze(1)
    logp_l = log_probs.gather(1, y_l.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        ref_logits, _ = reference(state, actions, mask, **model_kwargs)
        ref_log_probs = masked_log_softmax_logits(ref_logits, mask)
        refp_w = ref_log_probs.gather(1, y_w.unsqueeze(1)).squeeze(1)
        refp_l = ref_log_probs.gather(1, y_l.unsqueeze(1)).squeeze(1)

    logits_diff = (logp_w - logp_l) - (refp_w - refp_l)
    per_row_loss = -F.logsigmoid(beta * logits_diff)
    loss = weighted_mean(per_row_loss, weights)
    # Accuracy: fraction of pairs where the policy already prefers y_w
    # over y_l (a cheap progress signal — at init, this is the reference
    # policy's preference rate).
    per_row_correct = (logp_w > logp_l).float()
    accuracy = per_row_correct.mean()
    margin_signed = (logp_w - logp_l).mean()

    grouped = compute_dpo_grouped_metrics(batch, per_row_loss, per_row_correct)

    return {
        "loss": loss,
        "accuracy": accuracy,
        "policy_margin": margin_signed,
        "logits_diff_mean": logits_diff.mean(),
        "grouped": grouped,
    }


# R16-TD 3b chunk 3: warn-once flags for missing pair metadata. We don't
# want a noisy log line per batch when a corpus lacks one of the axes —
# emit one warning per axis per process and then default to "unknown".
_GROUPED_METRIC_WARNED: set[str] = set()


def _warn_missing_axis_once(axis: str) -> None:
    if axis in _GROUPED_METRIC_WARNED:
        return
    _GROUPED_METRIC_WARNED.add(axis)
    print(
        f"[train_dpo] grouped-metric axis {axis!r} missing on batch; "
        "stratifying as 'unknown'. Suppressing further warnings for this axis.",
        flush=True,
    )


def _action_kind_for_row(example: dict[str, Any], winner_index: int) -> str:
    """Lift the action-kind helper from `data_coverage_audit._action_kind`.

    Reads the winner action's `kind` field off `legalActions[winner_index]`.
    Returns "unknown" for malformed rows so a stray entry never crashes
    the metrics reporter.
    """
    la = example.get("legalActions") or []
    if 0 <= winner_index < len(la):
        action = la[winner_index]
        if isinstance(action, dict):
            return str(action.get("kind") or "unknown")
    return "unknown"


def extract_pair_meta(pairs: list[Any]) -> list[dict[str, Any]]:
    """Build the per-row metadata list used by `compute_dpo_grouped_metrics`.

    R16-TD 3b chunk 3: this is the bridge between `PreferencePair` (which
    carries the raw `example` dict from the JSONL plus `pair_source_kind`)
    and the grouped-metrics consumer. Callers (smoke + training loop) wire
    it in by setting ``batch["pair_meta"] = extract_pair_meta(pairs)``
    AFTER calling `collate_preference_batch`. Keeping it out of the
    collator preserves the regression smoke contract in
    `pair_corpus_smoke.py` (which asserts an exact `set(batch.keys())`).
    """
    meta: list[dict[str, Any]] = []
    for pair in pairs:
        example = getattr(pair, "example", {}) or {}
        winner = example.get("winner") or {}
        loser = example.get("loser") or {}
        observation = example.get("observation") or {}
        phase = example.get("phase") or observation.get("phase") or "unknown"
        try:
            winner_index = int(winner.get("index"))
        except (TypeError, ValueError):
            winner_index = int(getattr(pair, "y_w_index", -1))
        meta.append(
            {
                "negative_kind": loser.get("negativeKind"),
                "source_kind": getattr(pair, "pair_source_kind", None)
                or example.get("sourceKind"),
                "phase": phase,
                "action_kind": _action_kind_for_row(example, winner_index),
                "margin": float(getattr(pair, "margin", 0.0) or 0.0),
                "margin_bucket": _margin_bucket(
                    float(getattr(pair, "margin", 0.0) or 0.0)
                ),
            }
        )
    return meta


def _margin_bucket(margin: float) -> str:
    """Match the histogram convention from
    `pair_builder._bucket_for_margin` but with an explicit `<0.05` bucket
    for legacy outcome-v2 rows whose tau-gated margins can be below 0.05.
    """
    try:
        m = float(margin)
    except (TypeError, ValueError):
        return "unknown"
    if not (m == m):  # NaN
        return "unknown"
    if m < 0.05:
        return "<0.05"
    if m < 0.10:
        return "0.05-0.10"
    if m < 0.20:
        return "0.10-0.20"
    return ">=0.20"


def compute_dpo_grouped_metrics(
    batch: dict[str, torch.Tensor],
    per_pair_loss: torch.Tensor,
    per_pair_correct: torch.Tensor,
) -> dict[str, dict[str, dict[str, float]]]:
    """Stratify per-pair loss / accuracy by pair metadata axes.

    R16-TD 3b chunk 3 scoping § P2 step 4: reporting-only axes so that
    downstream analysis can locate regressions. The axes are:

      - ``negative_kind`` (from pair-builder's
        ``loser.negativeKind``; defaults to "unknown" / warn-once).
      - ``source_kind`` (from ``PreferencePair.pair_source_kind``; falls
        back to "outcome-v2" for legacy rows pre-Phase-1).
      - ``phase`` (from the pair row's ``phase`` field or
        ``observation.phase``; default "unknown").
      - ``action_kind`` (kind of the winner action via the lifted helper).
      - ``margin_bucket`` (per `_margin_bucket`).

    The pair metadata is carried by the batch under ``"pair_meta"`` (a
    list-of-dicts, one per row, same length as ``per_pair_loss``). When
    that key is absent (legacy synthetic batches), an empty dict is
    returned and the caller's downstream consumer treats it as "no axes
    fired".
    """

    meta_rows: list[dict[str, Any]] | None = batch.get("pair_meta")  # type: ignore[assignment]
    if not meta_rows:
        return {}

    loss_cpu = per_pair_loss.detach().to("cpu", dtype=torch.float32).tolist()
    correct_cpu = per_pair_correct.detach().to("cpu", dtype=torch.float32).tolist()
    n = min(len(meta_rows), len(loss_cpu), len(correct_cpu))

    # Pre-compute the bucket label for each row per axis. `buckets` is
    # `{axis: [label_for_row_0, label_for_row_1, ...]}`.
    axis_names = ("negative_kind", "source_kind", "phase", "action_kind", "margin_bucket")
    buckets: dict[str, list[str]] = {axis: [] for axis in axis_names}
    for row in range(n):
        meta = meta_rows[row] if isinstance(meta_rows[row], dict) else {}
        # negative_kind
        nk = meta.get("negative_kind")
        if nk is None:
            _warn_missing_axis_once("negative_kind")
            nk = "unknown"
        buckets["negative_kind"].append(str(nk))
        # source_kind
        sk = meta.get("source_kind") or "outcome-v2"
        buckets["source_kind"].append(str(sk))
        # phase
        ph = meta.get("phase") or "unknown"
        buckets["phase"].append(str(ph))
        # action_kind
        ak = meta.get("action_kind") or "unknown"
        buckets["action_kind"].append(str(ak))
        # margin_bucket
        mb = meta.get("margin_bucket")
        if mb is None:
            mb = _margin_bucket(meta.get("margin", 0.0))
        buckets["margin_bucket"].append(str(mb))

    grouped: dict[str, dict[str, dict[str, float]]] = {}
    for axis in axis_names:
        per_bucket: dict[str, dict[str, float]] = {}
        for row in range(n):
            label = buckets[axis][row]
            entry = per_bucket.setdefault(
                label, {"count": 0.0, "_loss_sum": 0.0, "_acc_sum": 0.0}
            )
            entry["count"] += 1.0
            entry["_loss_sum"] += float(loss_cpu[row])
            entry["_acc_sum"] += float(correct_cpu[row])
        # Finalize means and drop the intermediate sums.
        for label, entry in per_bucket.items():
            count = max(1.0, entry["count"])
            per_bucket[label] = {
                "count": entry["count"],
                "mean_loss": entry["_loss_sum"] / count,
                "mean_acc": entry["_acc_sum"] / count,
            }
        grouped[axis] = per_bucket
    return grouped


def run_epoch(
    model: CandidatePolicyNet,
    reference: CandidatePolicyNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    beta: float,
    device: torch.device,
) -> dict[str, Any]:
    model.train()
    totals = {
        "loss": 0.0,
        "accuracy": 0.0,
        "policy_margin": 0.0,
        "logits_diff_mean": 0.0,
        "count": 0.0,
    }
    grouped_totals: dict[str, dict[str, dict[str, float]]] = {}
    for batch in loader:
        batch = move_batch(batch, device)
        components = dpo_loss_components(model, reference, batch, beta=beta)
        loss = components["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        accumulate_metrics(totals, components, batch_size=batch["y_w_index"].shape[0])
        _merge_grouped_totals(grouped_totals, components.get("grouped") or {})
    out = finalize_metrics(totals)
    out["grouped"] = _finalize_grouped_totals(grouped_totals)
    return out


@torch.no_grad()
def evaluate(
    model: CandidatePolicyNet,
    reference: CandidatePolicyNet,
    loader: DataLoader | None,
    *,
    beta: float,
    device: torch.device,
) -> dict[str, Any]:
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
    grouped_totals: dict[str, dict[str, dict[str, float]]] = {}
    for batch in loader:
        batch = move_batch(batch, device)
        components = dpo_loss_components(model, reference, batch, beta=beta)
        accumulate_metrics(totals, components, batch_size=batch["y_w_index"].shape[0])
        _merge_grouped_totals(grouped_totals, components.get("grouped") or {})
    out = finalize_metrics(totals)
    out["grouped"] = _finalize_grouped_totals(grouped_totals)
    return out


def _merge_grouped_totals(
    accum: dict[str, dict[str, dict[str, float]]],
    batch_grouped: dict[str, dict[str, dict[str, float]]],
) -> None:
    """In-place merge of per-batch grouped metrics into the per-epoch
    running totals. We re-derive ``_loss_sum`` / ``_acc_sum`` from the
    batch's ``count`` × ``mean_loss`` so the final per-epoch mean is a
    correct sample-size-weighted average across batches.
    """
    for axis, per_bucket in batch_grouped.items():
        accum_axis = accum.setdefault(axis, {})
        for label, entry in per_bucket.items():
            slot = accum_axis.setdefault(
                label, {"count": 0.0, "_loss_sum": 0.0, "_acc_sum": 0.0}
            )
            count = float(entry.get("count", 0.0))
            slot["count"] += count
            slot["_loss_sum"] += float(entry.get("mean_loss", 0.0)) * count
            slot["_acc_sum"] += float(entry.get("mean_acc", 0.0)) * count


def _finalize_grouped_totals(
    accum: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for axis, per_bucket in accum.items():
        finalized: dict[str, dict[str, float]] = {}
        for label, slot in per_bucket.items():
            count = max(1.0, slot.get("count", 0.0))
            finalized[label] = {
                "count": slot.get("count", 0.0),
                "mean_loss": slot.get("_loss_sum", 0.0) / count,
                "mean_acc": slot.get("_acc_sum", 0.0) / count,
            }
        out[axis] = finalized
    return out


def accumulate_metrics(
    totals: dict[str, float],
    components: dict[str, Any],
    *,
    batch_size: int,
) -> None:
    count = float(batch_size)
    totals["count"] += count
    # R16-TD 3b chunk 3: `components["grouped"]` is a dict-of-dicts (reporting
    # only) — skip it here; per-batch grouped metrics are emitted by the
    # caller via the training-step log dict directly off `components`.
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
    batch: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    # R16-TD 3b chunk 3: `pair_meta` is a list of dicts (reporting metadata)
    # carried alongside the tensor fields — pass it through unchanged.
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


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
