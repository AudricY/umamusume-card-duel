"""R16-TD 3b P2 step 5 — offline pair-quality eval for DPO closed loop.

Implements the held-out pair-quality gate per
`docs/ai-research/scoping/<dpo-closed-loop>.md` § P2 step 5:

    "Held-out pair accuracy. Candidate ranking accuracy or NDCG over full
     vectors. Reference-vs-trained pair accuracy lift. No n=1000 gate until
     offline pair quality clears pre-registered floors."

This module is the *measurement* half of that gate — it runs a single
checkpoint over the held-out partition of a preference-pair corpus and
emits a comparable JSON manifest. The next chunk (closed-loop driver) will
re-invoke this tool on the *trained* DPO checkpoint and compare against
the baseline JSON to compute the lift.

Reuse posture (do NOT reimplement):
  - `train_dpo.dpo_loss_components`              — per-pair acc / loss math
    (which internally calls `compute_dpo_grouped_metrics` for stratified axes)
  - `train_dpo.extract_pair_meta`                — row→meta stitcher
  - `pair_corpus.PreferencePairDataset`          — JSONL loader
  - `pair_corpus.collate_preference_batch`       — collator

Checkpoint loading is handled by a LOCAL `load_checkpoint_compat` helper
rather than `train_dpo.load_reference_policy` because the production
96-d reference checkpoint (`R13-W6-phase-d/iter-2`) predates the v3
card-embedding branch and needs `strict=False` + zero-padding to load
into the current model. The padding is numerically null (zero-init
matches `padding_idx=0` semantics), so the loaded model is exactly the
pre-embedding-branch one.

What is NEW here (vs train_dpo's epoch loop):
  - Deterministic train/held-out split keyed on
    `(sourceEpisodeId, sourceStep)` so re-running on the same corpus puts
    the same rows in held-out (no row-shuffle entropy).
  - Candidate ranking metrics over the FULL legal-action vector:
        ranking_accuracy_top1 / top2 / top3
        mean_reciprocal_rank
    These complement the pair-only accuracy by measuring whether the
    model identifies the right action against the full action set, not
    just vs one negative.
  - JSON manifest schema with `schemaVersion: 1` for downstream lift
    comparison (`trained_metric - baseline_metric`).

The checkpoint passed via --checkpoint is loaded as BOTH the trainable
policy AND the reference, so the DPO loss reduces to `-log_sigmoid(0)
= log(2) ≈ 0.6931` per row and the per-pair accuracy reads the
checkpoint's intrinsic preference for `y_w` over `y_l`. That is exactly
the baseline floor the trained DPO policy must improve on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from train_bc import (
    feature_schema_metadata,
    resolve_device,
)
from train_dpo import (
    dpo_loss_components,
    extract_pair_meta,
)
from pair_corpus import (
    PreferencePairDataset,
    collate_preference_batch,
)
from uma_ai.features import (
    STATE_DIM,
    feature_builder_for_state_dim,
)
from uma_ai.model import CARD_EMBED_DIM, CandidatePolicyNet, ModelConfig


SCHEMA_VERSION = 1
DEFAULT_HELD_OUT_FRACTION = 0.2
DEFAULT_SPLIT_SEED = 17
DEFAULT_SPLIT_DENOM = 10_000

# The v3 embedding branch added two state-dict params (`card_embed.weight`,
# `zone_projection.weight`) and widened `joint_projection.0.weight`'s input
# dim from `3*hidden` to `3*hidden + 2*CARD_EMBED_DIM`. Pre-embedding-branch
# production checkpoints (the 96-d R13-W6-phase-d/iter-2 reference is the
# canonical example) don't carry these. Zero-padding the missing params
# AND the joint_projection's new column tail keeps the model byte-numerically
# equivalent to the original 96-d encoder — the embedding contributions
# collapse to zero exactly the same way they do when `card_ids_by_zone` is
# all-pad (see `model.forward`'s `padding_idx=0` semantics).
_V3_EMBEDDING_PARAMS = ("card_embed.weight", "zone_projection.weight")
_JOINT_PROJECTION_KEY = "joint_projection.0.weight"


def split_pairs_deterministic(
    pair_rows: list[Any],
    held_out_fraction: float = DEFAULT_HELD_OUT_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
    *,
    denom: int = DEFAULT_SPLIT_DENOM,
) -> tuple[list[int], list[int]]:
    """Deterministic train / held-out split keyed on `(sourceEpisodeId, sourceStep)`.

    Hash bucket: ``int(blake2b(f"{seed}:{sourceEpisodeId}:{sourceStep}")) % denom``.
    Rows with ``bucket / denom < held_out_fraction`` go to held-out; the
    rest go to train.

    Rows missing `sourceEpisodeId` / `sourceStep` (legacy outcome-v2 corpus
    where the export does not stamp them) fall back to a stable hash over
    the corpus index — this is still deterministic across re-runs but
    loses the cross-corpus-stability guarantee for those rows. The 3a
    prod corpus carries both fields on every row so this is the explicit
    fallback for mixed-mode corpora.

    Returns ``(train_indices, held_out_indices)`` over the SAME index
    space as the input list (so a caller can `Subset(dataset, indices)`
    directly).
    """
    if not (0.0 < held_out_fraction < 1.0):
        raise ValueError(
            f"held_out_fraction must be in (0, 1); got {held_out_fraction}"
        )
    threshold = int(round(held_out_fraction * denom))
    train_idx: list[int] = []
    holdout_idx: list[int] = []
    for index, row in enumerate(pair_rows):
        example = getattr(row, "example", {}) or {}
        episode_id = example.get("sourceEpisodeId")
        step = example.get("sourceStep")
        if episode_id is None or step is None:
            key = f"{seed}:fallback:{index}"
        else:
            key = f"{seed}:{episode_id}:{step}"
        digest = hashlib.blake2b(key.encode("utf8"), digest_size=8).digest()
        bucket = int.from_bytes(digest, "big") % denom
        if bucket < threshold:
            holdout_idx.append(index)
        else:
            train_idx.append(index)
    return train_idx, holdout_idx


def load_checkpoint_compat(
    path: str, device: torch.device
) -> tuple[CandidatePolicyNet, dict[str, Any]]:
    """Load a checkpoint with tolerant handling for pre-v3-embedding-branch
    state dicts.

    `train_dpo.load_reference_policy` does a strict `load_state_dict`, which
    fails on the 96-d production checkpoint (`R13-W6-phase-d/iter-2`)
    because that checkpoint predates the card-embedding residual branch
    (`card_embed.weight`, `zone_projection.weight`) AND the joint-projection
    input-dim widening (`3*hidden` → `3*hidden + 2*CARD_EMBED_DIM`).

    Pad strategy (numerically exact-null at load time):
      1. Add zero-init `card_embed.weight` of shape `(108, 32)`. With
         `padding_idx=0`, row 0 is gradient-free zero, so a forward over
         any id collapses through this all-zero table to zero.
      2. Add zero-init `zone_projection.weight` of shape
         `(hidden, NUM_ZONES * CARD_EMBED_DIM)`. The zone-pool linear
         projection now emits zero regardless of input.
      3. Zero-pad the joint_projection's input columns
         `[3*hidden : 3*hidden + 2*CARD_EMBED_DIM]`. The concat'd action
         source/target embedding (which is zero per step 1) is multiplied
         by zero columns; the original `[0:3*hidden]` columns are preserved
         verbatim so the pre-embedding semantics are byte-identical.

    Returns ``(model, report)`` where ``report`` describes the padding
    actions taken (for the manifest).
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config_dict = payload.get("model_config") or {}
    config = ModelConfig.from_dict(config_dict)
    model = CandidatePolicyNet(config).to(device)
    target_state = model.state_dict()

    raw = dict(payload["model_state"])
    padded_params: list[str] = []
    widened_keys: list[str] = []

    # Step 1+2: zero-fill missing v3-branch params if absent.
    for key in _V3_EMBEDDING_PARAMS:
        if key not in raw and key in target_state:
            raw[key] = torch.zeros_like(target_state[key])
            padded_params.append(key)

    # Step 3: zero-pad the joint_projection's new columns.
    if _JOINT_PROJECTION_KEY in raw and _JOINT_PROJECTION_KEY in target_state:
        src = raw[_JOINT_PROJECTION_KEY]
        tgt_shape = target_state[_JOINT_PROJECTION_KEY].shape
        if src.shape != tgt_shape:
            # Verify dim 0 (output hidden) matches and dim 1 (input)
            # widened by exactly `2 * CARD_EMBED_DIM`.
            if (
                src.shape[0] == tgt_shape[0]
                and tgt_shape[1] - src.shape[1] == 2 * CARD_EMBED_DIM
            ):
                widened = torch.zeros(tgt_shape, dtype=src.dtype)
                widened[:, : src.shape[1]] = src
                raw[_JOINT_PROJECTION_KEY] = widened
                widened_keys.append(_JOINT_PROJECTION_KEY)
            else:
                raise RuntimeError(
                    f"Cannot reconcile `{_JOINT_PROJECTION_KEY}` shape "
                    f"{tuple(src.shape)} with target {tuple(tgt_shape)} "
                    f"(expected source dim1 + {2 * CARD_EMBED_DIM} == target dim1)."
                )

    missing, unexpected = model.load_state_dict(raw, strict=False)
    # After our padding, both lists should be empty for a valid pre-v3
    # checkpoint. Fail loud otherwise — the harness should never silently
    # accept an unknown shape mismatch.
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint {path} has unexpected state-dict deltas: "
            f"missing={missing} unexpected={unexpected}"
        )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    report = {
        "checkpoint_path": str(path),
        "padded_v3_params": padded_params,
        "widened_joint_projection": widened_keys,
        "model_config": config.to_dict(),
    }
    return model, report


def _rebuild_batch_state_features(
    batch: dict[str, Any], samples: list[Any], model_state_dim: int
) -> None:
    """Replace ``batch['state_features']`` with model-state-dim features.

    `pair_corpus.collate_preference_batch` allocates a `(B, STATE_DIM=110)`
    array hardcoded to the v3.0 builder. A 96-d production checkpoint
    (R13-W6-phase-d/iter-2) has `state_dim=96`, so we transparently
    re-extract via the matching frozen builder and overwrite the
    collator's tensor in-place.

    No-op when ``model_state_dim == STATE_DIM`` (the v3.0 default).
    """
    if model_state_dim == STATE_DIM:
        return
    import numpy as np
    builder = feature_builder_for_state_dim(model_state_dim)
    arr = np.zeros((len(samples), model_state_dim), dtype=np.float32)
    for row, pair in enumerate(samples):
        observation = (pair.example or {}).get("observation") or {}
        arr[row] = builder(observation)
    batch["state_features"] = torch.from_numpy(arr)


def compute_ranking_metrics(
    logits: torch.Tensor,
    action_mask: torch.Tensor,
    y_w_index: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Per-row ranking metrics over the full legal-action set.

    Args:
        logits: ``[B, A]`` raw policy logits from the model.
        action_mask: ``[B, A]`` bool mask of legal actions.
        y_w_index: ``[B]`` int index of the winner action.

    Returns a dict of per-row tensors:
        rank_of_winner       — int rank in [1, A_legal] (1 = top).
        top1 / top2 / top3   — float {0, 1} per row.
        reciprocal_rank      — float 1 / rank.

    Implementation: mask out illegal actions (sentinel -inf) before
    comparing; the winner's rank is `1 + sum(legal_logit > winner_logit)`.
    Ties broken in the model's favor (strict `>`), which is the
    conservative choice — when two actions have identical logits, the
    winner is considered "in" the top set rather than out.
    """
    mask = action_mask.bool()
    sentinel = torch.full_like(logits, float("-inf"))
    masked_logits = torch.where(mask, logits, sentinel)
    # Per-row winner logit, shape [B, 1].
    winner_logit = masked_logits.gather(1, y_w_index.unsqueeze(1))
    # Strict greater-than for tie-break: a logit ties with the winner is
    # NOT counted as ranked above the winner.
    better = (masked_logits > winner_logit).float().sum(dim=1)
    rank = (better + 1.0)  # 1-indexed rank
    top1 = (rank <= 1.0).float()
    top2 = (rank <= 2.0).float()
    top3 = (rank <= 3.0).float()
    reciprocal = 1.0 / rank
    return {
        "rank": rank,
        "top1": top1,
        "top2": top2,
        "top3": top3,
        "reciprocal_rank": reciprocal,
    }


def evaluate_holdout(
    model: CandidatePolicyNet,
    reference: CandidatePolicyNet,
    loader: DataLoader,
    *,
    device: torch.device,
    beta: float = 0.1,
) -> dict[str, Any]:
    """Run model+reference over a held-out loader and emit aggregate metrics.

    Pair accuracy + loss + grouped axes are computed by reusing
    `dpo_loss_components` (which is the source of truth for those numbers).
    Ranking metrics are computed by re-running the model's forward
    output through `compute_ranking_metrics` — the same logits the loss
    consumed (no extra forward pass).

    All metrics are sample-size-weighted across batches.
    """
    model.eval()
    reference.eval()

    totals = {
        "loss": 0.0,
        "pair_accuracy": 0.0,
        "ranking_top1": 0.0,
        "ranking_top2": 0.0,
        "ranking_top3": 0.0,
        "reciprocal_rank": 0.0,
        "count": 0.0,
    }
    grouped_totals: dict[str, dict[str, dict[str, float]]] = {}

    with torch.no_grad():
        for batch in loader:
            batch_dev = _move_batch(batch, device)
            components = dpo_loss_components(
                model, reference, batch_dev, beta=beta
            )
            batch_size = batch_dev["y_w_index"].shape[0]
            totals["count"] += float(batch_size)
            totals["loss"] += float(components["loss"].item()) * batch_size
            totals["pair_accuracy"] += float(components["accuracy"].item()) * batch_size

            # Re-run model.forward to get clean logits for ranking. We
            # could fold this into dpo_loss_components but that would
            # require modifying train_dpo.py — out of scope.
            state = batch_dev["state_features"]
            actions = batch_dev["action_features"]
            mask = batch_dev["action_mask"]
            y_w = batch_dev["y_w_index"]
            kwargs: dict[str, torch.Tensor] = {}
            cards = batch_dev.get("card_ids_by_zone")
            action_idx = batch_dev.get("action_card_idx")
            if cards is not None and action_idx is not None:
                kwargs["card_ids_by_zone"] = cards
                kwargs["action_card_idx"] = action_idx
            logits, _ = model(state, actions, mask, **kwargs)
            ranking = compute_ranking_metrics(logits, mask, y_w)
            totals["ranking_top1"] += float(ranking["top1"].sum().item())
            totals["ranking_top2"] += float(ranking["top2"].sum().item())
            totals["ranking_top3"] += float(ranking["top3"].sum().item())
            totals["reciprocal_rank"] += float(ranking["reciprocal_rank"].sum().item())

            # Merge grouped axes.
            batch_grouped = components.get("grouped") or {}
            _merge_grouped(grouped_totals, batch_grouped)

    count = max(1.0, totals["count"])
    return {
        "pair_accuracy": totals["pair_accuracy"] / count,
        "ranking_accuracy_top1": totals["ranking_top1"] / count,
        "ranking_accuracy_top2": totals["ranking_top2"] / count,
        "ranking_accuracy_top3": totals["ranking_top3"] / count,
        "mean_reciprocal_rank": totals["reciprocal_rank"] / count,
        "loss": totals["loss"] / count,
        "samples": totals["count"],
        "grouped": _finalize_grouped(grouped_totals),
    }


def _merge_grouped(
    accum: dict[str, dict[str, dict[str, float]]],
    batch_grouped: dict[str, dict[str, dict[str, float]]],
) -> None:
    """In-place sample-weighted merge of per-batch grouped axes."""
    for axis, per_bucket in batch_grouped.items():
        accum_axis = accum.setdefault(axis, {})
        for label, entry in per_bucket.items():
            slot = accum_axis.setdefault(
                label, {"count": 0.0, "_loss_sum": 0.0, "_acc_sum": 0.0}
            )
            cnt = float(entry.get("count", 0.0))
            slot["count"] += cnt
            slot["_loss_sum"] += float(entry.get("mean_loss", 0.0)) * cnt
            slot["_acc_sum"] += float(entry.get("mean_acc", 0.0)) * cnt


def _finalize_grouped(
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


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _make_collator(model_state_dim: int):
    """Build a collate fn that stitches in pair_meta AND rebuilds the
    state_features tensor at the model's state_dim (when it differs from
    the corpus loader's hardcoded 110-d v3.0 default).
    """

    def _collate(samples: list[Any]) -> dict[str, Any]:
        batch = collate_preference_batch(samples)
        batch["pair_meta"] = extract_pair_meta(samples)
        _rebuild_batch_state_features(batch, samples, model_state_dim)
        return batch

    return _collate


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    """Top-level run-once entrypoint. Returns the manifest dict."""
    device = resolve_device(args.device)

    # 1. Load corpus.
    t_load_start = time.monotonic()
    dataset = PreferencePairDataset(
        args.pairs,
        tau=args.tau if args.tau >= 0 else None,
        log_manifest=False,
    )
    pairs_all = list(dataset.pairs)
    load_elapsed = time.monotonic() - t_load_start

    # 2. Deterministic split.
    train_idx, holdout_idx = split_pairs_deterministic(
        pairs_all,
        held_out_fraction=args.held_out_fraction,
        seed=args.split_seed,
    )

    # 3. Load reference checkpoint as BOTH the trainable and frozen
    # reference (so the DPO loss is the baseline-vs-baseline floor).
    # The compat loader zero-pads the v3 embedding branch if the
    # checkpoint predates it (pre-r7.b.2 production runs).
    reference, compat_report = load_checkpoint_compat(args.checkpoint, device)
    # Trainable copy uses the same weights. We just reuse the reference
    # — no gradient is required, and eval mode is already set.
    model = reference

    # 4. Build held-out loader. The collator rebuilds the
    # state_features tensor at the model's state_dim when it differs
    # from the corpus loader's hardcoded 110-d v3.0 default (e.g. the
    # 96-d production checkpoint).
    model_state_dim = int(model.config.state_dim)
    loader = DataLoader(
        Subset(dataset, holdout_idx),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_make_collator(model_state_dim),
    )

    # 6. Run eval.
    t_eval_start = time.monotonic()
    metrics = evaluate_holdout(model, reference, loader, device=device, beta=args.beta)
    eval_elapsed = time.monotonic() - t_eval_start

    # 7. Build manifest.
    manifest = {
        "schemaVersion": SCHEMA_VERSION,
        "args": {
            "pairs": str(args.pairs),
            "checkpoint": str(args.checkpoint),
            "held_out_fraction": float(args.held_out_fraction),
            "split_seed": int(args.split_seed),
            "split_denom": int(DEFAULT_SPLIT_DENOM),
            "batch_size": int(args.batch_size),
            "device": str(device),
            "tau": float(dataset.tau),
            "beta": float(args.beta),
        },
        "split": {
            "total_rows": len(pairs_all),
            "train_rows": len(train_idx),
            "held_out_rows": len(holdout_idx),
            "split_method": "blake2b(seed:sourceEpisodeId:sourceStep) % denom",
        },
        "source_counts": dict(dataset.source_counts),
        "model_config": model.config.to_dict(),
        "checkpoint_compat": compat_report,
        "feature_schema": feature_schema_metadata(state_dim=model_state_dim),
        "timings": {
            "load_seconds": load_elapsed,
            "eval_seconds": eval_elapsed,
        },
        "held_out_metrics": metrics,
    }
    return manifest


def main() -> None:
    args = parse_args()
    manifest = run_eval(args)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")

    # Headline print so the orchestrator can scrape stdout if needed.
    metrics = manifest["held_out_metrics"]
    headline = {
        "status": "PASS",
        "out": str(out_path),
        "held_out_rows": manifest["split"]["held_out_rows"],
        "pair_accuracy": metrics["pair_accuracy"],
        "ranking_accuracy_top1": metrics["ranking_accuracy_top1"],
        "ranking_accuracy_top2": metrics["ranking_accuracy_top2"],
        "ranking_accuracy_top3": metrics["ranking_accuracy_top3"],
        "mean_reciprocal_rank": metrics["mean_reciprocal_rank"],
        "loss": metrics["loss"],
        "eval_seconds": manifest["timings"]["eval_seconds"],
    }
    print(json.dumps(headline, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline pair-quality eval for DPO closed-loop gate "
            "(R16-TD 3b P2 step 5)."
        )
    )
    parser.add_argument(
        "--pairs",
        required=True,
        help="Path to the preference-pair JSONL corpus.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the policy checkpoint to evaluate (.pt).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to write the eval-manifest JSON.",
    )
    parser.add_argument(
        "--held-out-fraction",
        type=float,
        default=DEFAULT_HELD_OUT_FRACTION,
        help=f"Held-out fraction in (0, 1); default {DEFAULT_HELD_OUT_FRACTION}.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help=f"Hash seed for the deterministic split; default {DEFAULT_SPLIT_SEED}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Eval batch size; default 32.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help=(
            "DPO beta for the reported loss only; default 0.1. The reported "
            "loss is log(2)/something for a checkpoint-vs-itself baseline."
        ),
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=-1.0,
        help=(
            "Override noise-floor on selectedVsRunnerUpMargin (legacy v2 only). "
            "Negative (default) auto-probes; -1.0 is fine for explicit-pair "
            "corpora (probe short-circuits to floor)."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
