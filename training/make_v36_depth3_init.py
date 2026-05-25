"""v36-depth3: extend v3.6 ckpt from depth=2 to depth=3 via identity-init.

Tests the capacity-axis depth lever on top of v3.6's accepted-as-baseline
schema (state_dim=246, hidden_dim=128). v3.6 cap128 tight-gated at
wl=0.5880 — soft-ship band, essentially flat with v3.5-cap128 (0.5912)
which informed the verdict that the schema axis is exhausted at depth=2.

Methodology: ResidualBlock outputs `x + net(x)`. To append a third block
that is initially the identity, set the block's final Linear weight+bias
to zero so `net(x) = 0` and `block(x) = x`. The depth=3 trunk is then
function-identical to depth=2 at iter-0; training adapts.

For all other parameters (state_encoder, action_encoder, joint_projection,
joint_blocks[0..1], policy_head, value_head, q_value_head, slot encoder,
embeddings): copy verbatim. The depth-only diff is joint_blocks[2].

Refuses if:
  - source state_dim != 246 (not a v3.6 ckpt).
  - source depth != 2 (only depth=2 → depth=3 supported in this slice).
  - source action_schema_version != 3.

Iter-0 contract: Δlogits ≤ 1e-5 vs source on a fixed observation batch
(true identity-init, not approximate like the v3.5-cap128 LayerNorm pad).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uma_ai.features import (  # noqa: E402
    STATE_DIM_V3_6,
    STATE_FEATURE_SCHEMA_VERSION_V3_6,
)
from uma_ai.model import CandidatePolicyNet, ModelConfig  # noqa: E402

SOURCE_DEPTH = 2
TARGET_DEPTH = 3
TOLERANCE = 1e-5


def _build_target_model(source_config: ModelConfig) -> CandidatePolicyNet:
    """Build a depth=3 model with the same other dims as source."""
    target_config_kwargs = {
        f.name: getattr(source_config, f.name)
        for f in ModelConfig.__dataclass_fields__.values()
    }
    target_config_kwargs["depth"] = TARGET_DEPTH
    target_config = ModelConfig(**target_config_kwargs)
    return CandidatePolicyNet(target_config), target_config


def _refuse(msg: str) -> None:
    raise SystemExit(f"REFUSED: {msg}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--source-checkpoint", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    src_path: Path = args.source_checkpoint
    dst_path: Path = args.output

    if not src_path.exists():
        _refuse(f"source checkpoint {src_path} not found")

    print(f"loading source: {src_path}")
    ck = torch.load(src_path, map_location="cpu", weights_only=False)

    src_model_cfg = ModelConfig.from_dict(ck.get("model_config", {}))
    feature_schema = ck.get("feature_schema", {})

    if src_model_cfg.state_dim != STATE_DIM_V3_6:
        _refuse(
            f"source state_dim={src_model_cfg.state_dim} != {STATE_DIM_V3_6} "
            f"(expected v3.6 source for depth=2→3 expansion)"
        )
    if src_model_cfg.depth != SOURCE_DEPTH:
        _refuse(
            f"source depth={src_model_cfg.depth} != {SOURCE_DEPTH} "
            f"(only depth=2 → depth=3 supported in this slice)"
        )
    if feature_schema.get("action_feature_schema_version") != 3:
        _refuse(
            f"source action_schema_version="
            f"{feature_schema.get('action_feature_schema_version')} != 3"
        )

    print(
        f"source ok: state_dim={src_model_cfg.state_dim} "
        f"hidden_dim={src_model_cfg.hidden_dim} depth={src_model_cfg.depth}"
    )

    # Build source model and load weights (for parity verification).
    src_model = CandidatePolicyNet(src_model_cfg)
    src_model.load_state_dict(ck["model_state"], strict=True)
    src_model.eval()

    # Build target model.
    tgt_model, tgt_cfg = _build_target_model(src_model_cfg)
    tgt_model.eval()

    print(
        f"target: state_dim={tgt_cfg.state_dim} "
        f"hidden_dim={tgt_cfg.hidden_dim} depth={tgt_cfg.depth}"
    )

    # Copy all source weights into target. State dict keys differ only at
    # the new joint_blocks.2.* entries; everything else has the same shape.
    src_sd = src_model.state_dict()
    tgt_sd = tgt_model.state_dict()
    copied = 0
    new_keys = []
    for k, v in tgt_sd.items():
        if k in src_sd:
            assert src_sd[k].shape == v.shape, (
                f"shape mismatch at {k}: src={src_sd[k].shape} tgt={v.shape}"
            )
            tgt_sd[k] = src_sd[k].clone()
            copied += 1
        else:
            new_keys.append(k)

    # Identity-init the new joint_blocks.2 ResidualBlock: zero the final Linear
    # so block(x) = x + 0 = x.
    final_linear_w = "joint_blocks.2.net.4.weight"
    final_linear_b = "joint_blocks.2.net.4.bias"
    if final_linear_w not in new_keys or final_linear_b not in new_keys:
        _refuse(
            f"expected new keys joint_blocks.2.net.4.{{weight,bias}} not in "
            f"target sd; new keys = {new_keys[:5]}..."
        )
    # Zero the final Linear of the new block.
    tgt_sd[final_linear_w] = torch.zeros_like(tgt_sd[final_linear_w])
    tgt_sd[final_linear_b] = torch.zeros_like(tgt_sd[final_linear_b])

    # Other new params (LayerNorm + first Linear of the new block):
    # keep their fresh-init values. They don't affect the iter-0 output
    # because the final zero-Linear gates the whole residual to 0.

    tgt_model.load_state_dict(tgt_sd, strict=True)

    # Parity check: build a tiny fixed observation batch and compare logits.
    torch.manual_seed(0)
    batch = 4
    state_features = torch.randn(batch, STATE_DIM_V3_6)
    action_features = torch.randn(batch, 16, tgt_cfg.action_dim)
    action_mask = torch.ones(batch, 16, dtype=torch.bool)

    def _forward(model):
        with torch.no_grad():
            return model(
                state_features=state_features,
                action_features=action_features,
                action_mask=action_mask,
            )

    src_out = _forward(src_model)
    tgt_out = _forward(tgt_model)

    def _logits(out):
        # CandidatePolicyNet returns a dataclass / dict; grab policy logits.
        if hasattr(out, "policy_logits"):
            return out.policy_logits
        if isinstance(out, dict):
            return out.get("policy_logits") or out.get("logits")
        return out[0]  # tuple

    src_logits = _logits(src_out)
    tgt_logits = _logits(tgt_out)
    delta = (src_logits - tgt_logits).abs().max().item()
    print(f"max |Δlogits| = {delta:.3e} (tolerance {TOLERANCE:.0e})")
    if delta > TOLERANCE:
        _refuse(
            f"identity-init failed: Δlogits={delta:.3e} > {TOLERANCE:.0e}; "
            f"this is a bug in the depth expander, not a v3.6 issue"
        )

    # Bump model_config.depth in the output checkpoint.
    output = copy.deepcopy(ck)
    output["model_state"] = tgt_sd
    mc = output.get("model_config", {})
    mc["depth"] = TARGET_DEPTH
    output["model_config"] = mc
    # Drop optimizer/scheduler/scaler/rng state (warm-start convention, matches
    # other v3.x expanders).
    for k in ("optimizer_state", "scheduler_state", "scaler_state", "rng_state"):
        output.pop(k, None)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, dst_path)

    manifest = {
        "status": "PASS",
        "dest": str(dst_path),
        "method": "v3.6 depth-2→3 identity-init expansion",
        "source_checkpoint": str(src_path),
        "source_depth": SOURCE_DEPTH,
        "dest_depth": TARGET_DEPTH,
        "hidden_dim": src_model_cfg.hidden_dim,
        "state_dim": src_model_cfg.state_dim,
        "identity_init_block": "joint_blocks.2",
        "zero_init_params": [final_linear_w, final_linear_b],
        "all_other_params": "copied verbatim",
        "delta_logits_at_init": delta,
        "delta_logits_tolerance": TOLERANCE,
        "copied_params": copied,
        "new_params_total": len(new_keys),
        "builder": "training/make_v36_depth3_init.py",
    }
    print(
        f"v3.6 depth=3 init built from {src_path}, output={dst_path}, "
        f"delta_logits={delta:.3e} (< {TOLERANCE:.0e} tolerance)"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
