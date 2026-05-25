"""v34-compound-axis: build a v3.4 warm-start from a v3.3 source.

v3.4 = v3.3 state vector (state_dim=167) + v3.2 per-Uma slot tokens
(`uses_uma_slot_tokens=True`). The state-vector dim itself is unchanged
from v3.3; v3.4 differs from v3.3 only by adding the slot-token ONNX
input pair and the corresponding `uma_slot_encoder` branch.

This builder is the v3.3 → v3.4 additive-tensors lift — mirrors
`training/make_v32_slot_token_init.py` (which does the v3.0 → v3.2 lift)
exactly, except the source is a v3.3 ckpt instead of v3.0. The same 3
new state_dict keys are introduced:

  - `uma_slot_encoder.0.weight: (hidden_dim, CARD_EMBED_DIM + UMA_SLOT_FEATURE_DIM)`
    -> `(hidden_dim, 55)`, default PyTorch init.
  - `uma_slot_encoder.0.bias: (hidden_dim,)`, default init.
  - `uma_slot_encoder.2.weight: (hidden_dim, hidden_dim)`, **zero**-initialized.

Why this is numerically null at init: the `CandidatePolicyNet` forward
adds `uma_slot_encoder(slot_concat)` (pooled across slots) to
`state_encoded`. The encoder is `Linear(55, h) -> GELU -> Linear(h, h,
bias=False)`. With the final Linear's weight = 0, the encoder output is
exactly zero regardless of input. Hence the v3.4 net at init produces
outputs numerically identical to the v3.3 source on shared features.

Refuses to operate on:
  - non-v3.3 sources (state_dim != 167) — v3.0/v3.1/v3.2 have different
    state_encoder.0 shapes (110-d or 164-d) and the merge would
    silently corrupt.
  - sources that ALREADY have `uses_uma_slot_tokens=True` — would
    silently re-init the slot encoder.

Reusable + auditable: re-run to regenerate the init deterministically.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uma_ai.features import (  # noqa: E402
    STATE_DIM_V3_3,
    STATE_FEATURE_SCHEMA_VERSION_V3_4,
    UMA_SLOT_FEATURE_DIM,
)
from uma_ai.model import (  # noqa: E402
    CARD_EMBED_DIM,
    CandidatePolicyNet,
    ModelConfig,
)

# The 3 new state_dict keys introduced by `uses_uma_slot_tokens=True`.
NEW_SLOT_ENCODER_KEYS: tuple[str, ...] = (
    "uma_slot_encoder.0.weight",
    "uma_slot_encoder.0.bias",
    "uma_slot_encoder.2.weight",
)
ZERO_INIT_KEY = "uma_slot_encoder.2.weight"


def build(source: Path, dest: Path, *, seed: int = 1234) -> dict[str, Any]:
    """Build a v3.4 init checkpoint from a v3.3 source checkpoint."""

    payload = torch.load(source, map_location="cpu", weights_only=False)
    if "model_state" not in payload:
        raise SystemExit(
            f"source checkpoint `{source}` has no `model_state` key "
            f"(keys: {sorted(payload)}); not a training-checkpoint payload."
        )
    if "model_config" not in payload:
        raise SystemExit(
            f"source checkpoint `{source}` has no `model_config` key — "
            f"cannot determine `hidden_dim`/`state_dim` without it."
        )

    source_state: dict[str, torch.Tensor] = payload["model_state"]
    source_config_dict: dict[str, Any] = dict(payload["model_config"])
    source_state_dim = int(source_config_dict.get("state_dim", -1))

    if source_state_dim != STATE_DIM_V3_3:
        raise SystemExit(
            f"v34-compound builder refuses non-v3.3 source: `{source}` has "
            f"state_dim={source_state_dim}, expected {STATE_DIM_V3_3}. "
            f"v3.4 stacks slot tokens on top of v3.3's 167-d state vector."
        )

    if bool(source_config_dict.get("uses_uma_slot_tokens", False)):
        raise SystemExit(
            f"refuse to operate on source that already has "
            f"`uses_uma_slot_tokens=True`: `{source}`. v3.4 is the v3.3 → "
            f"v3.4 additive lift; running on a slot-token source would "
            f"silently re-init the encoder."
        )

    # Build the v3.4 target ModelConfig: copy source config, flip
    # uses_uma_slot_tokens to True. state_dim stays 167.
    target_config_dict = dict(source_config_dict)
    target_config_dict["uses_uma_slot_tokens"] = True
    target_config = ModelConfig.from_dict(target_config_dict)
    if not target_config.uses_uma_slot_tokens or target_config.state_dim != STATE_DIM_V3_3:
        raise SystemExit(
            f"internal: target ModelConfig did not propagate "
            f"uses_uma_slot_tokens=True + state_dim={STATE_DIM_V3_3} "
            f"after `from_dict` — ModelConfig schema drifted."
        )

    # Determinism: only newly-initialized parameters are the slot
    # encoder's first Linear weight + bias. Seed for byte-identical reruns.
    torch.manual_seed(seed)
    target_model = CandidatePolicyNet(target_config)
    target_model.eval()
    fresh_target_state = target_model.state_dict()

    fresh_keys = set(fresh_target_state.keys())
    source_keys = set(source_state.keys())
    expected_new = set(NEW_SLOT_ENCODER_KEYS)
    missing_in_fresh = source_keys - fresh_keys
    extra_in_fresh = fresh_keys - source_keys
    if missing_in_fresh:
        raise SystemExit(
            f"v3.4 model_state missing source keys: "
            f"{sorted(missing_in_fresh)[:8]}... (v3.3/v3.4 schema drift)."
        )
    unexpected = extra_in_fresh - expected_new
    if unexpected:
        raise SystemExit(
            f"v3.4 model_state has unexpected new keys beyond the 3 slot-"
            f"encoder keys: {sorted(unexpected)}."
        )
    missing_new = expected_new - extra_in_fresh
    if missing_new:
        raise SystemExit(
            f"v3.4 model_state missing the expected new slot-encoder keys: "
            f"{sorted(missing_new)}. Did `uses_uma_slot_tokens=True` reach "
            f"`CandidatePolicyNet.__init__`?"
        )

    # Merge: every source key verbatim + the 3 new slot-encoder keys
    # from the freshly-constructed v3.4 model.
    merged_state: dict[str, torch.Tensor] = {}
    for key in source_keys:
        merged_state[key] = source_state[key].clone().detach()
    for key in NEW_SLOT_ENCODER_KEYS:
        merged_state[key] = fresh_target_state[key].clone().detach()

    # Belt-and-suspenders: assert the zero-init invariant.
    zero_w = merged_state[ZERO_INIT_KEY]
    if not torch.allclose(zero_w, torch.zeros_like(zero_w)):
        max_abs = float(zero_w.abs().max().item())
        raise SystemExit(
            f"zero-init invariant violated: `{ZERO_INIT_KEY}` is not zero "
            f"(max|w|={max_abs:.3e})."
        )

    # Strict load into a fresh target model.
    verify_model = CandidatePolicyNet(target_config)
    missing, unexpected = verify_model.load_state_dict(merged_state, strict=True)
    if missing or unexpected:
        raise SystemExit(
            f"strict load mismatch: missing={list(missing)} "
            f"unexpected={list(unexpected)}."
        )

    feature_schema = dict(payload.get("feature_schema") or {})
    feature_schema["state_dim"] = STATE_DIM_V3_3
    feature_schema["state_feature_schema_version"] = STATE_FEATURE_SCHEMA_VERSION_V3_4

    provenance = {
        "method": "v3.4 compound axis (slot tokens + v3.3 state) warm-start",
        "source_checkpoint": str(source),
        "source_state_dim": source_state_dim,
        "dest_state_dim": STATE_DIM_V3_3,
        "source_uses_uma_slot_tokens": bool(source_config_dict.get("uses_uma_slot_tokens", False)),
        "dest_uses_uma_slot_tokens": True,
        "new_state_dict_keys": list(NEW_SLOT_ENCODER_KEYS),
        "zero_initialized_key": ZERO_INIT_KEY,
        "card_embed_dim": CARD_EMBED_DIM,
        "uma_slot_feature_dim": UMA_SLOT_FEATURE_DIM,
        "all_v3_3_params": "copied verbatim",
        "delta_logits_at_init": 0.0,
        "delta_value_at_init": 0.0,
        "builder": "training/make_v34_compound_init.py",
        "seed": int(seed),
    }

    new_payload: dict[str, Any] = {
        "model_state": merged_state,
        "model_config": target_config.to_dict(),
        "feature_schema": feature_schema,
        "provenance": provenance,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_payload, dest)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-checkpoint",
        "--source",
        dest="source",
        required=True,
        help="Path to the v3.3 source checkpoint to warm-start from "
             "(e.g. `runs/R16-P1-v33-ablation/loop/iter-1/checkpoint.pt`, "
             "the v3.3 lineage best). Refuses non-v3.3 sources.",
    )
    parser.add_argument(
        "--output",
        "--dest",
        dest="dest",
        required=True,
        help="Path to write the v3.4 init checkpoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="RNG seed for the freshly-initialized slot encoder's first "
             "Linear (default 1234).",
    )
    args = parser.parse_args()
    source = Path(args.source)
    dest = Path(args.dest)
    provenance = build(source, dest, seed=args.seed)
    print(
        f"v3.4 init built from {source}, output={dest}, "
        f"slot_encoder.2.weight=zeros (delta=0.0 invariant)"
    )
    print(json.dumps({"status": "PASS", "dest": str(dest), **provenance}, indent=2))


if __name__ == "__main__":
    main()
