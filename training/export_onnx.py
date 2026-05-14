from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from uma_ai.features import ACTION_DIM, CARD_ID_SHAPES, STATE_DIM, card_vocab_metadata
from uma_ai.model import CARD_VOCAB_TABLE_SIZE, NUM_ZONES, CandidatePolicyNet, ModelConfig

# R7.b.2 Phase 3: per-zone max-cards width is FIXED in the ONNX graph for
# ORT shape stability — Phase 4's re-extractor pads every row to this same
# width, so dynamic-axis is unnecessary and adds shape-inference cost. The
# batch dimension and the action count stay dynamic since both vary per
# request. `MAX_CARDS_PER_ZONE` is `max(CARD_ID_SHAPES.values())` = 30
# (ownDiscard/oppDiscard cap), matching the collator in
# `training/uma_ai/dataset.py`.
MAX_CARDS_PER_ZONE = max(CARD_ID_SHAPES.values())


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig.from_dict(checkpoint.get("model_config"))
    if config.state_dim != STATE_DIM or config.action_dim != ACTION_DIM:
        raise ValueError(
            f"Checkpoint feature dimensions {config.state_dim}/{config.action_dim} "
            f"do not match current {STATE_DIM}/{ACTION_DIM}"
        )
    expected_vocab = card_vocab_metadata()
    checkpoint_schema = checkpoint.get("feature_schema") or {}
    checkpoint_vocab = checkpoint_schema.get("card_vocab")
    if checkpoint_vocab is not None and expected_vocab.get("hash") != "missing":
        if checkpoint_vocab.get("hash") != expected_vocab.get("hash"):
            raise ValueError(
                f"Card vocab hash mismatch: checkpoint={checkpoint_vocab.get('hash')} "
                f"runtime={expected_vocab.get('hash')}; rebuild vocab or retrain."
            )

    model = CandidatePolicyNet(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # R7.b.2 Phase 3: fail loud if the embedding table outgrew the vocab —
    # `nn.Embedding(108, ...)` with `padding_idx=0` is one row over the
    # vocab to reserve index 0 for unknown/pad. If a future vocab rebuild
    # bumps vocabSize above 107 we want the export to refuse rather than
    # silently produce an under-sized graph that ORT would later index OOB.
    expected_vocab_size = int(expected_vocab.get("vocabSize", 0))
    if expected_vocab_size > 0 and model.card_embed.num_embeddings != expected_vocab_size + 1:
        raise ValueError(
            f"card_embed.num_embeddings ({model.card_embed.num_embeddings}) must equal "
            f"vocabSize+1 ({expected_vocab_size + 1}); rebuild vocab or update "
            f"CARD_VOCAB_TABLE_SIZE in training/uma_ai/model.py."
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    state = torch.zeros((1, STATE_DIM), dtype=torch.float32)
    actions = torch.zeros((1, args.max_actions, ACTION_DIM), dtype=torch.float32)
    mask = torch.ones((1, args.max_actions), dtype=torch.bool)
    # R7.b.2 Phase 3: two new int64 tensors flow into the embedding pass.
    # All zeros are a valid no-op input (index 0 = pad row, kept zero by
    # `padding_idx=0`), matching the additive-residual null behaviour
    # validated in Phase 2's smoke contract (4).
    card_ids_by_zone = torch.zeros((1, NUM_ZONES, MAX_CARDS_PER_ZONE), dtype=torch.int64)
    action_card_idx = torch.zeros((1, args.max_actions, 2), dtype=torch.int64)
    torch.onnx.export(
        model,
        # Positional args mirror the model's `forward` signature; the two
        # new int inputs must be passed as positional tensors (not kwargs)
        # to participate in the traced graph. Phase 2's optional-kwarg
        # design keeps every other caller (PPO/DPO/value-retrain) working
        # unchanged because they still pass only the three originals.
        (state, actions, mask, card_ids_by_zone, action_card_idx),
        out,
        input_names=[
            "state_features",
            "action_features",
            "action_mask",
            "card_ids_by_zone",
            "action_card_idx",
        ],
        output_names=["logits", "value"],
        dynamic_axes={
            "state_features": {0: "batch"},
            "action_features": {0: "batch", 1: "actions"},
            "action_mask": {0: "batch", 1: "actions"},
            # Per-zone max-cards (axis 2) is intentionally FIXED — Phase 4
            # re-extractor pads every row to MAX_CARDS_PER_ZONE so the
            # graph never sees a variable per-zone width.
            "card_ids_by_zone": {0: "batch"},
            "action_card_idx": {0: "batch", 1: "actions"},
            "logits": {0: "batch", 1: "actions"},
            "value": {0: "batch"},
        },
        opset_version=args.opset,
    )
    sidecar = out.with_suffix(out.suffix + ".meta.json")
    sidecar.write_text(
        json.dumps(
            {
                "state_dim": STATE_DIM,
                "action_dim": ACTION_DIM,
                "card_vocab": expected_vocab,
                "checkpoint_vocab": checkpoint_vocab,
                # R7.b.2 Phase 3: surface the embedding graph shape so
                # downstream consumers (serve_onnx) can shape-validate.
                "num_zones": NUM_ZONES,
                "max_cards_per_zone": MAX_CARDS_PER_ZONE,
                "card_vocab_table_size": CARD_VOCAB_TABLE_SIZE,
            },
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    print(f"Exported {out} (vocab hash={expected_vocab.get('hash')})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export candidate policy checkpoint to ONNX.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-actions", type=int, default=16)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


if __name__ == "__main__":
    main()
