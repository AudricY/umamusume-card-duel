"""R16-P2 C7: build a v3.2 per-Uma-slot-token additive-init warm-start.

Methodology (mirror of `training/make_v31_additive_init.py`): take a v3.0
source checkpoint (e.g. the production 0.6042 baseline at
`runs/R13-W6-phase-d/iter-2/checkpoint.pt`, or any pre-C1 v3.0 ckpt the user
passes) and produce a v3.2 init checkpoint that loads byte-identical
parameters for every v3.0 key PLUS exactly 3 NEW keys for the
`uma_slot_encoder` branch:

  - `uma_slot_encoder.0.weight: (hidden_dim, CARD_EMBED_DIM + UMA_SLOT_FEATURE_DIM)`
    -> `(hidden_dim, 55)`, default PyTorch init (Kaiming for nn.Linear).
  - `uma_slot_encoder.0.bias: (hidden_dim,)`, default init.
  - `uma_slot_encoder.2.weight: (hidden_dim, hidden_dim)`, **zero**-initialized
    (no `.bias` because the second Linear is `bias=False` per C2).

Why this is exactly numerically null at init: the v3.2 `CandidatePolicyNet`
forward adds `uma_slot_encoder(slot_concat)` (pooled across slots) to
`state_encoded`. The encoder is `Linear(55, h) -> GELU -> Linear(h, h,
bias=False)`. With the final Linear's weight = 0, the encoder output is
exactly zero regardless of input. The treatment forward also masks the 4
board-zone lanes (ownActive/oppActive/ownBench/oppBench) of
`card_ids_by_zone` to 0 before `zone_projection` (C2's double-counting fix);
this masking is the only non-residual difference vs the v3.0 forward and is
absorbed by the smoke's matched-input comparison. Hence the v3.2 net at init
produces outputs numerically identical to the v3.0 source on shared features
(modulo the masked board lanes, which are then directly comparable on a
board-masked v3.0 input — see `training/r16_p2_slot_init_smoke.py` smoke (ii)
and the C2 precedent at `training/r16_p2_slot_encoder_smoke.py` smoke (ii)).

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
    STATE_DIM_V3,
    STATE_FEATURE_SCHEMA_VERSION_V3_2,
    UMA_SLOT_FEATURE_DIM,
)
from uma_ai.model import (  # noqa: E402
    CARD_EMBED_DIM,
    CandidatePolicyNet,
    ModelConfig,
)

# The 3 new state_dict keys introduced by `uses_uma_slot_tokens=True`. Keep
# this enumeration as a module-level constant so the smoke can reference the
# same source of truth (no string drift between builder and verifier).
NEW_SLOT_ENCODER_KEYS: tuple[str, ...] = (
    "uma_slot_encoder.0.weight",
    "uma_slot_encoder.0.bias",
    "uma_slot_encoder.2.weight",
)
ZERO_INIT_KEY = "uma_slot_encoder.2.weight"


def build(source: Path, dest: Path, *, seed: int = 1234) -> dict[str, Any]:
    """Build a v3.2 init checkpoint from a v3.0 source checkpoint.

    Returns the `provenance` dict written into the destination payload.
    """

    payload = torch.load(source, map_location="cpu", weights_only=False)
    if "model_state" not in payload:
        raise SystemExit(
            f"source checkpoint `{source}` has no `model_state` key "
            f"(keys: {sorted(payload)}); not a training-checkpoint payload."
        )
    if "model_config" not in payload:
        raise SystemExit(
            f"source checkpoint `{source}` has no `model_config` key — "
            f"cannot determine `hidden_dim`/`state_dim` without it. Pass a "
            f"v3.0 training checkpoint emitted by `train_bc.py`."
        )

    source_state: dict[str, torch.Tensor] = payload["model_state"]
    source_config_dict: dict[str, Any] = dict(payload["model_config"])

    # Refuse to operate on a v3.2 source. Absent flag is treated as False
    # (older v3.0 checkpoints don't have the field at all).
    if bool(source_config_dict.get("uses_uma_slot_tokens", False)):
        raise SystemExit(
            f"refuse to operate on v3.2 source: `{source}` already has "
            f"`model_config.uses_uma_slot_tokens=True`. This builder is the "
            f"v3.0 -> v3.2 additive lift; running it on a v3.2 source would "
            f"silently re-init the slot encoder. Pass a v3.0 checkpoint."
        )

    state_dim = int(source_config_dict.get("state_dim", STATE_DIM_V3))
    # v3.2 keeps STATE_DIM=110; this builder is the *additive-tensors*
    # widening, NOT a state-vector widening. If you want a 164-d v3.1 init
    # AND slot tokens, run `make_v31_additive_init.py` first then this
    # builder on its output (still v3.0->v3.2 modulo state_dim, since v3.1
    # source has `uses_uma_slot_tokens` absent). For now keep the check
    # informative but not blocking on state_dim.
    if state_dim not in (STATE_DIM_V3, 164):
        raise SystemExit(
            f"source checkpoint state_dim={state_dim}, expected 110 (v3.0) "
            f"or 164 (v3.1). v3.2 is the slot-token additive lift and is "
            f"state-dim-agnostic, but unknown dims indicate a different "
            f"architecture."
        )

    # Build the v3.2 target ModelConfig from the source's config, flipping
    # `uses_uma_slot_tokens` to True. `ModelConfig.from_dict` filters
    # unknown keys, so an older config dict without the flag still loads
    # cleanly.
    target_config_dict = dict(source_config_dict)
    target_config_dict["uses_uma_slot_tokens"] = True
    target_config = ModelConfig.from_dict(target_config_dict)
    if not target_config.uses_uma_slot_tokens:
        raise SystemExit(
            "internal: target ModelConfig.uses_uma_slot_tokens did not "
            "propagate True after `from_dict` — ModelConfig schema drifted."
        )

    # Determinism: the only newly-initialized parameters are the slot
    # encoder's first Linear weight + bias (default PyTorch init). Seed
    # so reruns produce byte-identical inits.
    torch.manual_seed(seed)
    target_model = CandidatePolicyNet(target_config)
    target_model.eval()
    fresh_target_state = target_model.state_dict()

    # Sanity: the freshly-constructed v3.2 state_dict must contain exactly
    # the source's keys PLUS the 3 new slot-encoder keys.
    fresh_keys = set(fresh_target_state.keys())
    source_keys = set(source_state.keys())
    expected_new = set(NEW_SLOT_ENCODER_KEYS)
    missing_in_fresh = source_keys - fresh_keys
    extra_in_fresh = fresh_keys - source_keys
    if missing_in_fresh:
        raise SystemExit(
            f"v3.2 model_state missing source keys: "
            f"{sorted(missing_in_fresh)[:8]}... (v3.0/v3.2 schema drift; the "
            f"v3.2 model architecture is no longer a superset of v3.0)."
        )
    unexpected = extra_in_fresh - expected_new
    if unexpected:
        raise SystemExit(
            f"v3.2 model_state has unexpected new keys beyond the 3 slot-"
            f"encoder keys: {sorted(unexpected)}. C7's additive contract "
            f"requires exactly {sorted(expected_new)} as the v3.0->v3.2 "
            f"delta — a new architecture change must update this builder."
        )
    missing_new = expected_new - extra_in_fresh
    if missing_new:
        raise SystemExit(
            f"v3.2 model_state missing the expected new slot-encoder keys: "
            f"{sorted(missing_new)}. Did `uses_uma_slot_tokens=True` reach "
            f"`CandidatePolicyNet.__init__`?"
        )

    # Build the merged state_dict: copy every source key verbatim, take the
    # 3 new slot-encoder keys from the freshly-constructed v3.2 model
    # (which already zero-inited `uma_slot_encoder.2.weight` per C2).
    merged_state: dict[str, torch.Tensor] = {}
    for key in source_keys:
        merged_state[key] = source_state[key].clone().detach()
    for key in NEW_SLOT_ENCODER_KEYS:
        merged_state[key] = fresh_target_state[key].clone().detach()

    # Belt-and-suspenders: assert the C2 zero-init invariant directly on
    # the merged state. A future ModelConfig change that flips the init
    # would silently break C7's delta=0.0 contract; fail loudly here.
    zero_w = merged_state[ZERO_INIT_KEY]
    if not torch.allclose(zero_w, torch.zeros_like(zero_w)):
        max_abs = float(zero_w.abs().max().item())
        raise SystemExit(
            f"C2 invariant violated: `{ZERO_INIT_KEY}` is not zero "
            f"(max|w|={max_abs:.3e}). C7's delta=0.0 contract requires "
            f"this Linear's weight be zero-initialized at construction; "
            f"check `CandidatePolicyNet.__init__` for an `nn.init.zeros_` "
            f"regression."
        )

    # Strict load into a fresh target model: every key in merged_state
    # must match a target param; no missing, no unexpected.
    verify_model = CandidatePolicyNet(target_config)
    missing, unexpected = verify_model.load_state_dict(merged_state, strict=True)
    if missing or unexpected:
        raise SystemExit(
            f"strict load mismatch: missing={list(missing)} "
            f"unexpected={list(unexpected)} — merged state_dict does not "
            f"match the v3.2 model schema exactly."
        )

    # Update the feature_schema metadata so downstream consumers
    # (export_onnx, serve_onnx, training resume) tag this as v3.2.
    feature_schema = dict(payload.get("feature_schema") or {})
    feature_schema["state_dim"] = state_dim
    feature_schema["state_feature_schema_version"] = STATE_FEATURE_SCHEMA_VERSION_V3_2

    provenance = {
        "method": "v3.2 per-Uma slot-token additive warm-start",
        "source_checkpoint": str(source),
        "source_state_dim": state_dim,
        "dest_state_dim": state_dim,
        "source_uses_uma_slot_tokens": bool(source_config_dict.get("uses_uma_slot_tokens", False)),
        "dest_uses_uma_slot_tokens": True,
        "new_state_dict_keys": list(NEW_SLOT_ENCODER_KEYS),
        "zero_initialized_key": ZERO_INIT_KEY,
        "card_embed_dim": CARD_EMBED_DIM,
        "uma_slot_feature_dim": UMA_SLOT_FEATURE_DIM,
        "first_linear_input_dim": CARD_EMBED_DIM + UMA_SLOT_FEATURE_DIM,
        "all_v3_0_params": "copied verbatim",
        "delta_logits_at_init": 0.0,
        "delta_value_at_init": 0.0,
        "builder": "training/make_v32_slot_token_init.py",
        "seed": int(seed),
    }

    # Warm-start init only (matches `make_v31_additive_init.py`): drop
    # optimizer/scheduler/scaler/rng/epoch so the ablation loop treats this
    # as fresh weights. `load_init_from_checkpoint` in `train_bc.py` reads
    # ONLY `model_state` but tolerates a richer payload, so we also keep
    # `model_config` + `feature_schema` for export_onnx/serve_onnx.
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
        help="Path to the v3.0 source checkpoint to warm-start from "
             "(e.g. `runs/R13-W6-phase-d/iter-2/checkpoint.pt` for the "
             "production 0.6042 baseline; refuses v3.2 sources).",
    )
    parser.add_argument(
        "--output",
        "--dest",
        dest="dest",
        required=True,
        help="Path to write the v3.2 init checkpoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="RNG seed for the freshly-initialized slot encoder's first "
             "Linear (default 1234; rerun with the same seed for "
             "byte-identical reproductions).",
    )
    args = parser.parse_args()
    source = Path(args.source)
    dest = Path(args.dest)
    provenance = build(source, dest, seed=args.seed)
    print(
        f"v3.2 init built from {source}, output={dest}, "
        f"slot_encoder.2.weight=zeros (delta-0.0 invariant)"
    )
    print(json.dumps({"status": "PASS", "dest": str(dest), **provenance}, indent=2))


if __name__ == "__main__":
    main()
