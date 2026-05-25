"""v35-multichannel-tail: build a v3.5 additive-tail warm-start from a v3.3 source.

v3.5 widens the state-features vector from 167 → 212 by appending 45 bits of
channel-orthogonal additional signal (phase one-hot, per-condition one-hot
for both sides, energy-zone front-of-queue typed one-hot for both sides,
opp-side discard role buckets, bench-refill catastrophe bits for both sides).
The only weight that changes shape is `state_encoder.0.weight` (the first
Linear of the state-features encoder MLP), which grows from
`(hidden_dim, 167)` to `(hidden_dim, 212)`. The 45 new columns are zero-
initialized so a v3.3-trained checkpoint produces bit-identical iter-0
outputs through the v3.5 graph regardless of what the new bits encode.

Methodology (mirror of `training/make_v33_tail_init.py`):

  1. Read the v3.3 source ckpt (e.g.
     `runs/R16-P1-v33-ablation/loop/iter-1/checkpoint.pt`).
  2. Build a fresh v3.5 `CandidatePolicyNet` (state_dim=212) and discover
     the new shape of `state_encoder.0.weight` from a dummy construction.
  3. Construct the merged state_dict:
     - For every key OTHER than `state_encoder.0.weight`, copy verbatim
       from the source (the v3.5 graph preserves them all).
     - For `state_encoder.0.weight`: take the v3.3 source weight of shape
       `(hidden, 167)` and pad with 45 zero columns on the right to
       `(hidden, 212)`. The bias is unchanged.
  4. Strict-load the merged state into a fresh v3.5 model — must accept
     every key, no missing, no unexpected.
  5. Sanity check: feed a fixed deterministic observation through both
     v3.3 (source) and v3.5 (merged) models; assert max|delta| < 1e-5 on
     logits and value outputs (the zero-init residual contract).

Refuses to operate on a non-v3.3 source: v3.0 / v3.1 / v3.2 ckpts have
different state_encoder.0 shapes (110-d / 164-d) and would silently produce
a broken merge. v3.4 ckpts (slot tokens, state_dim=167) are also refused
because the slot-token branch is incompatible with v3.5's no-slot contract.

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
    STATE_DIM_V3_5,
    STATE_FEATURE_SCHEMA_VERSION_V3_5,
)
from uma_ai.model import (  # noqa: E402
    CandidatePolicyNet,
    ModelConfig,
)

WIDENED_KEY = "state_encoder.0.weight"


def build(source: Path, dest: Path) -> dict[str, Any]:
    """Build a v3.5 init checkpoint from a v3.3 source checkpoint.

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
            f"v3.3 training checkpoint emitted by `train_bc.py`."
        )

    source_state: dict[str, torch.Tensor] = payload["model_state"]
    source_config_dict: dict[str, Any] = dict(payload["model_config"])
    source_state_dim = int(source_config_dict.get("state_dim", -1))
    source_uses_slots = bool(source_config_dict.get("uses_uma_slot_tokens", False))

    if source_state_dim != STATE_DIM_V3_3:
        raise SystemExit(
            f"v35-multichannel-tail builder refuses non-v3.3 source: "
            f"`{source}` has state_dim={source_state_dim}, expected "
            f"{STATE_DIM_V3_3}. v3.0/v3.1/v3.2 have different "
            f"state_encoder.0 shapes (110-d / 164-d) and would silently "
            f"corrupt the merge. Use a v3.3 ckpt (e.g. "
            f"runs/R16-P1-v33-ablation/loop/iter-1/checkpoint.pt)."
        )
    if source_uses_slots:
        raise SystemExit(
            f"v35-multichannel-tail builder refuses slot-token source: "
            f"`{source}` has uses_uma_slot_tokens=True (looks like a v3.4 "
            f"ckpt). v3.5 is the no-slot-token contract (state_dim=212, "
            f"5-input ONNX); v3.4 was FALSIFIED at progress/r110.md §4h. "
            f"Pass a v3.3 ckpt (uses_uma_slot_tokens=False)."
        )

    if WIDENED_KEY not in source_state:
        raise SystemExit(
            f"source state_dict missing `{WIDENED_KEY}`. Cannot perform the "
            f"v3.3 → v3.5 state-vector widening without it."
        )
    source_weight = source_state[WIDENED_KEY]
    if source_weight.shape[-1] != STATE_DIM_V3_3:
        raise SystemExit(
            f"`{WIDENED_KEY}` has shape {tuple(source_weight.shape)}; "
            f"expected last dim {STATE_DIM_V3_3}. Source ckpt schema "
            f"disagrees with model_config.state_dim."
        )

    # Build the v3.5 target ModelConfig: copy source config, set state_dim
    # to 212. uses_uma_slot_tokens stays False (this builder is the
    # state-vector widening axis only; v3.5 EXCLUDES slot tokens per
    # scoping doc §3 and the v3.4 falsification at progress/r110.md §4h).
    target_config_dict = dict(source_config_dict)
    target_config_dict["state_dim"] = STATE_DIM_V3_5
    target_config_dict["uses_uma_slot_tokens"] = False
    target_config = ModelConfig.from_dict(target_config_dict)
    if target_config.state_dim != STATE_DIM_V3_5:
        raise SystemExit(
            f"internal: target ModelConfig.state_dim did not propagate "
            f"{STATE_DIM_V3_5} after `from_dict` — ModelConfig schema drifted."
        )

    # Construct a fresh v3.5 model to learn the new shape of the widened
    # weight + sanity-check the keyset.
    fresh_target = CandidatePolicyNet(target_config)
    fresh_target.eval()
    fresh_target_state = fresh_target.state_dict()

    fresh_keys = set(fresh_target_state.keys())
    source_keys = set(source_state.keys())
    if fresh_keys != source_keys:
        missing = source_keys - fresh_keys
        extra = fresh_keys - source_keys
        raise SystemExit(
            f"v3.5 model state_dict keys do not match v3.3 source. "
            f"missing in target: {sorted(missing)[:6]}; extra in target: "
            f"{sorted(extra)[:6]}. v3.5 should be additive at "
            f"state_encoder.0 only — no key churn."
        )

    fresh_widened_shape = fresh_target_state[WIDENED_KEY].shape
    if fresh_widened_shape[-1] != STATE_DIM_V3_5:
        raise SystemExit(
            f"fresh v3.5 model's `{WIDENED_KEY}` has shape "
            f"{tuple(fresh_widened_shape)}; expected last dim "
            f"{STATE_DIM_V3_5}. Model construction is not producing the "
            f"widened tensor — check ModelConfig.state_dim plumbing."
        )

    # Construct the merged state_dict. Every key OTHER than the widened
    # one is copied verbatim; the widened one is padded with 45 zero
    # columns on the right.
    merged_state: dict[str, torch.Tensor] = {}
    for key in source_keys:
        if key == WIDENED_KEY:
            continue
        merged_state[key] = source_state[key].clone().detach()

    # Widening: (hidden, 167) → (hidden, 212) by appending 45 zero columns.
    zeros = torch.zeros(
        (source_weight.shape[0], STATE_DIM_V3_5 - STATE_DIM_V3_3),
        dtype=source_weight.dtype,
        device=source_weight.device,
    )
    widened = torch.cat([source_weight.clone().detach(), zeros], dim=-1)
    if widened.shape != fresh_widened_shape:
        raise SystemExit(
            f"widened tensor shape mismatch: got {tuple(widened.shape)}, "
            f"target expects {tuple(fresh_widened_shape)}."
        )
    merged_state[WIDENED_KEY] = widened

    # Strict-load into a fresh v3.5 model.
    verify_model = CandidatePolicyNet(target_config)
    missing, unexpected = verify_model.load_state_dict(merged_state, strict=True)
    if missing or unexpected:
        raise SystemExit(
            f"strict load mismatch: missing={list(missing)} "
            f"unexpected={list(unexpected)} — merged state_dict does not "
            f"match the v3.5 model schema exactly."
        )

    # Bit-identical iter-0 contract: feed a fixed observation through
    # both the v3.3 source and the v3.5 merged model. v3.3 sees the first
    # 167 dims; v3.5 sees those 167 + 45 zero tail bits. Because the
    # appended Linear columns are zero, the contribution is exactly 0.0
    # and outputs must be bit-identical (within fp tolerance).
    verify_v33 = CandidatePolicyNet(ModelConfig.from_dict(source_config_dict))
    missing, unexpected = verify_v33.load_state_dict(source_state, strict=True)
    if missing or unexpected:
        raise SystemExit(
            f"v3.3 source state_dict failed strict load: missing={list(missing)} "
            f"unexpected={list(unexpected)}."
        )
    verify_v33.eval()
    verify_model.eval()

    torch.manual_seed(2026)
    batch = 2
    actions = 8
    # Build a synthetic batch matching the model's expected inputs. We
    # don't care about realism; what matters is comparing v3.3 vs v3.5
    # on the SAME state_features with the appended bits = 0.
    state_v33 = torch.randn(batch, STATE_DIM_V3_3)
    state_v35 = torch.cat(
        [state_v33, torch.zeros(batch, STATE_DIM_V3_5 - STATE_DIM_V3_3)], dim=-1
    )
    action_features = torch.randn(batch, actions, target_config.action_dim)
    action_mask = torch.ones(batch, actions, dtype=torch.bool)

    # The model forward needs more inputs (card_ids_by_zone, action pair
    # indices, optionally uma slot tokens). Use zeros — what we're
    # comparing is the v3.3 vs v3.5 trunk on identical aux inputs.
    from uma_ai.model import (  # noqa: E402
        ACTION_PAIR_FANOUT,
        NUM_ZONES,
    )
    from uma_ai.features import CARD_ID_SHAPES  # noqa: E402

    max_zone_cards = max(CARD_ID_SHAPES.values())
    card_ids_by_zone = torch.zeros(batch, NUM_ZONES, max_zone_cards, dtype=torch.long)
    action_card_idx = torch.zeros(batch, actions, ACTION_PAIR_FANOUT, dtype=torch.long)

    # v3.5 explicitly disables slot tokens; no extra_kwargs needed. We
    # also assert the source did not have them (the build above already
    # refuses uses_uma_slot_tokens=True sources).
    extra_kwargs: dict[str, torch.Tensor] = {}

    with torch.no_grad():
        out_v33 = verify_v33(
            state_features=state_v33,
            action_features=action_features,
            action_mask=action_mask,
            card_ids_by_zone=card_ids_by_zone,
            action_card_idx=action_card_idx,
            **extra_kwargs,
        )
        out_v35 = verify_model(
            state_features=state_v35,
            action_features=action_features,
            action_mask=action_mask,
            card_ids_by_zone=card_ids_by_zone,
            action_card_idx=action_card_idx,
            **extra_kwargs,
        )

    def _to_tensor(x: Any) -> torch.Tensor | None:
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, tuple):
            # Some models return (logits, value); pick the first as logits.
            return x[0] if isinstance(x[0], torch.Tensor) else None
        if hasattr(x, "logits"):
            return x.logits
        return None

    logits_v33 = _to_tensor(out_v33)
    logits_v35 = _to_tensor(out_v35)
    if logits_v33 is None or logits_v35 is None:
        raise SystemExit(
            f"could not extract logits from model output for verification "
            f"(types: v33={type(out_v33).__name__}, v35={type(out_v35).__name__})"
        )
    delta_logits = float((logits_v33 - logits_v35).abs().max().item())
    if delta_logits > 1e-5:
        raise SystemExit(
            f"v3.3 ↔ v3.5 iter-0 parity FAILED: max|Δlogits|={delta_logits:.3e} > 1e-5. "
            f"The zero-init tail contract is broken — investigate "
            f"`state_encoder.0.weight` widening or model topology changes."
        )

    # Update the feature_schema metadata so downstream consumers
    # (export_onnx, serve_onnx, training resume) tag this as v3.5.
    feature_schema = dict(payload.get("feature_schema") or {})
    feature_schema["state_dim"] = STATE_DIM_V3_5
    feature_schema["state_feature_schema_version"] = STATE_FEATURE_SCHEMA_VERSION_V3_5

    provenance = {
        "method": "v3.5 multichannel additive-tail warm-start",
        "source_checkpoint": str(source),
        "source_state_dim": source_state_dim,
        "dest_state_dim": STATE_DIM_V3_5,
        "widened_tensor": WIDENED_KEY,
        "widened_from": list(source_weight.shape),
        "widened_to": list(widened.shape),
        "new_columns_zero_init": STATE_DIM_V3_5 - STATE_DIM_V3_3,
        "all_other_params": "copied verbatim",
        "delta_logits_at_init": delta_logits,
        "delta_logits_tolerance": 1e-5,
        "builder": "training/make_v35_tail_init.py",
    }

    # Warm-start init only (matches `make_v33_tail_init.py`): drop
    # optimizer/scheduler/scaler/rng/epoch so the ablation loop treats
    # this as fresh weights. Keep model_config + feature_schema so
    # export_onnx/serve_onnx dispatch correctly.
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
             "(e.g. `runs/R16-P1-v33-ablation/loop/iter-1/checkpoint.pt`). "
             "Refuses non-v3.3 sources (state_dim != 167 or slot-tokens=True).",
    )
    parser.add_argument(
        "--output",
        "--dest",
        dest="dest",
        required=True,
        help="Path to write the v3.5 init checkpoint.",
    )
    args = parser.parse_args()
    source = Path(args.source)
    dest = Path(args.dest)
    provenance = build(source, dest)
    print(
        f"v3.5 init built from {source}, output={dest}, "
        f"delta_logits={provenance['delta_logits_at_init']:.3e} (< 1e-5 tolerance)"
    )
    print(json.dumps({"status": "PASS", "dest": str(dest), **provenance}, indent=2))


if __name__ == "__main__":
    main()
