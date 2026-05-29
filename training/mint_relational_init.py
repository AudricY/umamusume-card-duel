"""Mint a fresh (random-weight) v6 relational init checkpoint for cold-start.

The ReBeL loop has no heuristic-only self-play path (the info-incorrect rollout
leaf was removed in b07c2d3), so iteration 0 needs a bootstrap ONNX. There is
no `make_*_init` expander for the relational trunk (it trains from scratch),
so this mints a config-correct random checkpoint that `export_onnx.py` turns
into the v3.2/belief-signature ONNX the Rust dispatch already serves.

Config mirrors what `v6_relational_smoke.py` export-validated:
state_dim=110 (frozen v3.0 head; the relational trunk rides the v3.2 7-input /
belief 8-input signature), hidden_dim=128, depth=3, slot tokens + belief on,
model_variant=relational. Run from repo root with the venv python.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from uma_ai.features import ACTION_FEATURE_SCHEMA_VERSION, card_vocab_metadata
from uma_ai.model import CandidatePolicyNet, ModelConfig

OUT = Path("runs/rebel-v6-smoke/init/checkpoint.pt")
CONFIG = dict(
    state_dim=110,
    hidden_dim=128,
    depth=3,
    dropout=0.05,
    uses_uma_slot_tokens=True,
    model_variant="relational",
    uses_belief_features=True,
)


def main() -> None:
    cfg = ModelConfig(**CONFIG)
    model = CandidatePolicyNet(cfg).eval()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": CONFIG,
            "feature_schema": {
                "card_vocab": card_vocab_metadata(),
                "action_feature_schema_version": ACTION_FEATURE_SCHEMA_VERSION,
            },
        },
        OUT,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"minted relational init checkpoint -> {OUT} ({n_params:,} params)")


if __name__ == "__main__":
    main()
