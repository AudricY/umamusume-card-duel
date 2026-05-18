"""R16-P0 MCTS self-play v3 embedding-support smoke.

Verifies the data-path-correctness fix that unblocks honest 110-d
mcts-distill experiments (the bug that confounded R110-W6):

  1. Loader test    — current v3 mcts-selfplay rows load with non-None
                       `card_ids_by_zone` and `action_card_idx`, 100% of
                       usable samples.
  2. Collator test  — `collate_mcts_selfplay_batch()` emits
                       `card_ids_by_zone` int64 `[B, 8, 30]` and
                       `action_card_idx` int64 `[B, A, 2]` for v3 batches.
  3. Gradient smoke  — one MCTS batch fwd/bwd produces a nonzero gradient
                       on `card_embed.weight` AND `zone_projection.weight`
                       (proves the embedding branch is live under
                       mcts-distill, not inert).
  4. Legacy smoke   — a row with `cardIdsByZone` stripped fails loud by
                       default, and succeeds (fields stay None, embedding
                       tensors omitted) only under the explicit
                       `allow_missing_card_ids` compat flag.

Depends on the v3 mcts-selfplay corpus at
runs/R12-selfplay-smoke/selfplay.jsonl (regenerate via
training/r12_selfplay_smoke.py if missing or stale/pre-v3).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uma_ai.features import CARD_ID_SHAPES, ZONE_ORDER  # noqa: E402
from uma_ai.model import CandidatePolicyNet, ModelConfig  # noqa: E402
from uma_ai.selfplay_dataset import (  # noqa: E402
    MctsSelfPlayDataset,
    collate_mcts_selfplay_batch,
    load_mcts_selfplay_samples,
)


def fail(msg: str) -> None:
    print(f"[r16-embed-smoke] FAIL — {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    corpus = repo_root / "runs" / "R12-selfplay-smoke" / "selfplay.jsonl"
    if not corpus.exists():
        print(
            f"[r16-embed-smoke] missing {corpus}; run r12_selfplay_smoke.py first",
            file=sys.stderr,
        )
        sys.exit(2)

    first = json.loads(corpus.read_text(encoding="utf8").splitlines()[0])
    if "cardIdsByZone" not in first.get("observation", {}):
        print(
            "[r16-embed-smoke] corpus predates v3 (no cardIdsByZone); "
            "regenerate via r12_selfplay_smoke.py",
            file=sys.stderr,
        )
        sys.exit(2)

    # 1. Loader: 100% of usable samples carry both embedding fields.
    samples = list(load_mcts_selfplay_samples(corpus, min_actions=2))
    if not samples:
        fail("loader produced 0 samples")
    missing = sum(
        1
        for s in samples
        if s.card_ids_by_zone is None or s.action_card_idx is None
    )
    if missing:
        fail(f"{missing}/{len(samples)} v3 samples missing embedding fields")
    sample0 = samples[0]
    for zone in ZONE_ORDER:
        if zone not in sample0.card_ids_by_zone:
            fail(f"card_ids_by_zone missing zone {zone!r}")
        if sample0.card_ids_by_zone[zone].shape != (CARD_ID_SHAPES[zone],):
            fail(
                f"zone {zone!r} shape {sample0.card_ids_by_zone[zone].shape} "
                f"!= ({CARD_ID_SHAPES[zone]},)"
            )
    if sample0.action_card_idx.shape[1] != 2:
        fail(f"action_card_idx col dim {sample0.action_card_idx.shape} != (A, 2)")

    # 2. Collator: v3 tensors with the expected dtype/shape.
    batch = collate_mcts_selfplay_batch(samples[: min(8, len(samples))])
    if "card_ids_by_zone" not in batch or "action_card_idx" not in batch:
        fail("collator did not emit embedding tensors for a v3 batch")
    czi = batch["card_ids_by_zone"]
    aci = batch["action_card_idx"]
    max_cards = max(CARD_ID_SHAPES.values())
    bsz = min(8, len(samples))
    if czi.dtype != torch.int64 or aci.dtype != torch.int64:
        fail(f"embedding tensors must be int64; got {czi.dtype}, {aci.dtype}")
    if tuple(czi.shape) != (bsz, len(ZONE_ORDER), max_cards):
        fail(f"card_ids_by_zone shape {tuple(czi.shape)} != ({bsz}, {len(ZONE_ORDER)}, {max_cards})")
    if czi.shape[2] != 30:
        fail(f"card_ids_by_zone max-cards dim {czi.shape[2]} != 30")
    if aci.dim() != 3 or aci.shape[0] != bsz or aci.shape[2] != 2:
        fail(f"action_card_idx shape {tuple(aci.shape)} != ({bsz}, A, 2)")
    if int(czi.abs().sum()) == 0:
        fail("card_ids_by_zone is all-zero — no real card ids packed")

    # 3. Gradient smoke: embedding branch is live under mcts-distill.
    torch.manual_seed(0)
    model = CandidatePolicyNet(ModelConfig(hidden_dim=64, depth=2, dropout=0.0))
    model.train()
    logits, values = model(
        batch["state_features"],
        batch["action_features"],
        batch["action_mask"],
        card_ids_by_zone=batch["card_ids_by_zone"],
        action_card_idx=batch["action_card_idx"],
    )
    log_probs = torch.log_softmax(
        logits.masked_fill(~batch["action_mask"], float("-inf")), dim=1
    )
    policy_loss = -(batch["policy_targets"] * log_probs).nan_to_num().sum(dim=1).mean()
    value_loss = torch.nn.functional.mse_loss(values, batch["value_targets"])
    (policy_loss + value_loss).backward()

    ce_grad = model.card_embed.weight.grad
    zp_grad = model.zone_projection.weight.grad
    if ce_grad is None:
        fail("card_embed.weight.grad is None — embedding branch inert")
    if zp_grad is None:
        fail("zone_projection.weight.grad is None — zone path inert")
    ce_norm = float(ce_grad.detach().abs().sum())
    zp_norm = float(zp_grad.detach().abs().sum())
    if not (ce_norm > 0.0):
        fail(f"card_embed.weight grad is zero (sum|grad|={ce_norm}) — branch inert")
    if not (zp_norm > 0.0):
        fail(f"zone_projection.weight grad is zero (sum|grad|={zp_norm}) — branch inert")

    # 4a. Legacy fail-loud by default: strip cardIdsByZone from one row.
    with tempfile.TemporaryDirectory() as td:
        stripped_path = Path(td) / "legacy.jsonl"
        row = dict(first)
        obs = dict(row.get("observation", {}))
        obs.pop("cardIdsByZone", None)
        row["observation"] = obs
        stripped_path.write_text(json.dumps(row) + "\n", encoding="utf8")

        failed_loud = False
        try:
            list(load_mcts_selfplay_samples(stripped_path, min_actions=2))
        except ValueError as exc:
            failed_loud = "cardIdsByZone" in str(exc)
        if not failed_loud:
            fail("stripped row did not fail loud by default")

        # 4b. Named compat path: loads, fields stay None, no embed tensors.
        try:
            compat = MctsSelfPlayDataset(
                stripped_path, min_actions=2, allow_missing_card_ids=True
            )
        except Exception as exc:  # noqa: BLE001
            fail(f"allow_missing_card_ids path raised: {exc!r}")
        if compat.samples[0].card_ids_by_zone is not None:
            fail("compat path should leave card_ids_by_zone None")
        compat_batch = collate_mcts_selfplay_batch(compat.samples)
        if "card_ids_by_zone" in compat_batch:
            fail("compat (None) batch must omit card_ids_by_zone tensor")

    print(
        json.dumps(
            {
                "status": "PASS",
                "samples": len(samples),
                "samples_missing_embed": missing,
                "card_ids_by_zone_shape": list(czi.shape),
                "action_card_idx_shape": list(aci.shape),
                "card_embed_grad_abs_sum": round(ce_norm, 6),
                "zone_projection_grad_abs_sum": round(zp_norm, 6),
                "legacy_fail_loud": True,
                "compat_flag_omits_embed_tensors": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
