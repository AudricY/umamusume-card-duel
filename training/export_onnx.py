from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from uma_ai.features import ACTION_DIM, STATE_DIM, card_vocab_metadata
from uma_ai.model import CandidatePolicyNet, ModelConfig


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

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    state = torch.zeros((1, STATE_DIM), dtype=torch.float32)
    actions = torch.zeros((1, args.max_actions, ACTION_DIM), dtype=torch.float32)
    mask = torch.ones((1, args.max_actions), dtype=torch.bool)
    torch.onnx.export(
        model,
        (state, actions, mask),
        out,
        input_names=["state_features", "action_features", "action_mask"],
        output_names=["logits", "value"],
        dynamic_axes={
            "state_features": {0: "batch"},
            "action_features": {0: "batch", 1: "actions"},
            "action_mask": {0: "batch", 1: "actions"},
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
