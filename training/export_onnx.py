from __future__ import annotations

import argparse
from pathlib import Path

import torch

from uma_ai.features import ACTION_DIM, STATE_DIM
from uma_ai.model import CandidatePolicyNet, ModelConfig


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig.from_dict(checkpoint.get("model_config"))
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
    print(f"Exported {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export candidate policy checkpoint to ONNX.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-actions", type=int, default=16)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


if __name__ == "__main__":
    main()
