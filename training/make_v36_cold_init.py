"""v36-cold-init: build a FRESH random-init v3.6 checkpoint (no warm-start).

Use this when the orchestrator requires --init-checkpoint but you want
a true cold-start training run (e.g., to test high-MCTS-sims trajectory
without inheriting any v3.5 / v3.3 prior).

Default model shape mirrors v3.6 baseline (state_dim=246, hidden=128,
depth=2). Override via CLI flags.

Output is structurally identical to other v3.x init ckpts (same keys,
same feature_schema, same action_feature_schema_version). The
orchestrator's --init-checkpoint loader accepts it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uma_ai.features import (  # noqa: E402
    STATE_DIM_V3_6,
    STATE_FEATURE_SCHEMA_VERSION_V3_6,
)
from uma_ai.model import CandidatePolicyNet, ModelConfig  # noqa: E402

ACTION_SCHEMA_VERSION = 3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--state-dim", type=int, default=STATE_DIM_V3_6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.state_dim != STATE_DIM_V3_6:
        raise SystemExit(
            f"--state-dim={args.state_dim} != {STATE_DIM_V3_6} "
            f"(this util builds v3.6 cold-init only)"
        )

    torch.manual_seed(args.seed)

    cfg = ModelConfig(
        state_dim=args.state_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    )
    model = CandidatePolicyNet(cfg)
    model.eval()

    ckpt = {
        "model_state": model.state_dict(),
        "model_config": {
            f.name: getattr(cfg, f.name)
            for f in ModelConfig.__dataclass_fields__.values()
        },
        "feature_schema": {
            "state_dim": args.state_dim,
            "action_dim": cfg.action_dim,
            "state_feature_schema_version": STATE_FEATURE_SCHEMA_VERSION_V3_6,
            "action_feature_schema_version": ACTION_SCHEMA_VERSION,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.output)

    print(
        f"v3.6 cold-init built: state_dim={args.state_dim} "
        f"hidden={args.hidden_dim} depth={args.depth} seed={args.seed} "
        f"→ {args.output}"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "method": "v3.6 cold (random) init — no warm-start",
                "dest": str(args.output),
                "model_config": ckpt["model_config"],
                "feature_schema": ckpt["feature_schema"],
                "seed": args.seed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
