"""v5-action-disambiguation: build a v5 init from a v3.8 source checkpoint.

v5 extends v3.8's 52-d action-feature input projection to 57-d by
APPENDING 5 zero-init columns at indices [52:57]. State-feature input
is UNCHANGED (state schema stays at v3.8, state_dim=304). Trunk dims
unchanged (hidden_dim=128, depth=2). The append is pure-zero-init
(no column-drop), so the iter-0 parity contract is strict Δlogits ≤ 1e-5.

Per `docs/ai-research/scoping/v5-action-disambiguation-scoping.md` §3.3:
  1. Reads a v3.8 source ckpt (state_dim=304, action_count=52,
     action_schema=4, state_schema=3.8).
  2. Refuses non-v3.8 sources, non-default trunk (hidden_dim != 128 or
     depth != 2 — defense against accidentally trunk-bumping in this
     slice; v3.8 §13.3 contraindication still applies), and slot-token
     ckpts (uses_uma_slot_tokens=True).
  3. Builds a fresh v5 `CandidatePolicyNet` (state_dim=304,
     action_feature_count=57).
  4. Constructs the merged state_dict:
     - For every key OTHER than the widened action tensor, copy verbatim.
       (state_encoder.0.weight is UNCHANGED — state schema stays v3.8.)
     - action input projection (whichever Linear has last_dim ==
       ACTION_FEATURE_COUNT == 52 in source): widened (out, 52) → (out,
       57). Old columns [0:52] copied verbatim; new columns [52:57]
       zero-init.
  5. Strict-loads the merged state into a fresh v5 model — no missing,
     no unexpected keys.
  6. Sanity check: feed a fixed observation batch through both the v3.8
     source and v5 merged model. Drive action[:, :, 52:57]=0 in the
     synthetic v5 inputs so the only contributions come from the byte-
     identical [0:52] action columns. Residual = bit-identical modulo
     f32 rounding. Tolerance: 1e-5 (strict).
  7. Bumps `model_config.action_feature_count → 57`, `feature_schema
     .action_feature_schema_version → 5`. state_dim and
     state_feature_schema_version UNCHANGED.

Template: training/make_v38_slim_init.py (state+action tail zero-init).
v5 drops the state-tail expansion since state schema is unchanged.
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
    ACTION_FEATURE_SCHEMA_VERSION,
    STATE_DIM_V3_8,
    STATE_FEATURE_SCHEMA_VERSION_V3_8,
)
from uma_ai.model import (  # noqa: E402
    CandidatePolicyNet,
    ModelConfig,
)

# Strict parity tolerance — v5 is a pure-zero-init append on action
# [52:57]. With the new tail driven to 0 in the synthetic batch, iter-0
# logits should be bit-identical modulo f32 rounding. Matches v3.8's
# 1e-5 contract.
PARITY_TOLERANCE = 1e-5

# Source contract: v3.8 ckpts at state_dim=304, action_feature_count=52,
# action_feature_schema_version=4, state_feature_schema_version=3.8,
# hidden_dim=128, depth=2.
_EXPECTED_SOURCE_STATE_DIM = STATE_DIM_V3_8
_EXPECTED_SOURCE_ACTION_DIM = 52
_EXPECTED_SOURCE_ACTION_SCHEMA_VERSION = 4
_EXPECTED_SOURCE_STATE_SCHEMA_VERSION = STATE_FEATURE_SCHEMA_VERSION_V3_8
_EXPECTED_HIDDEN_DIM = 128
_EXPECTED_DEPTH = 2

# Target contract: v5 (action-axis only).
_TARGET_STATE_DIM = STATE_DIM_V3_8  # unchanged from source
_TARGET_ACTION_DIM = 57
_TARGET_ACTION_SCHEMA_VERSION = ACTION_FEATURE_SCHEMA_VERSION  # == 5


def _find_action_input_projection_key(state_dict: dict[str, torch.Tensor]) -> str:
    """Discover the parameter key whose last dimension equals the source
    action_feature_count (52). Mirrors make_v38_slim_init's helper. The
    action input projection's exact name depends on ModelConfig (e.g.
    `action_encoder.0.weight`, `action_proj.weight`, etc.); we resolve
    by shape match rather than hardcoding a name."""

    candidates: list[str] = []
    for key, tensor in state_dict.items():
        if tensor.dim() != 2:
            continue
        if tensor.shape[-1] == _EXPECTED_SOURCE_ACTION_DIM:
            candidates.append(key)
    if not candidates:
        raise SystemExit(
            f"v5 init: could not find any 2-D parameter with last dim "
            f"{_EXPECTED_SOURCE_ACTION_DIM} in the source state_dict. "
            f"Available 2-D shapes: "
            f"{[(k, list(t.shape)) for k, t in state_dict.items() if t.dim() == 2]}"
        )
    action_keyed = [k for k in candidates if "action" in k.lower()]
    if action_keyed:
        candidates = action_keyed
    return candidates[0]


def build(source: Path, dest: Path) -> dict[str, Any]:
    """Build a v5 init checkpoint from a v3.8 source checkpoint.

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
            f"cannot determine `hidden_dim`/`state_dim` without it. Pass "
            f"a v3.8 training checkpoint emitted by `train_bc.py`."
        )

    source_state: dict[str, torch.Tensor] = payload["model_state"]
    source_config_dict: dict[str, Any] = dict(payload["model_config"])
    source_state_dim = int(source_config_dict.get("state_dim", -1))
    source_action_dim = int(source_config_dict.get("action_dim", -1))
    source_hidden = int(source_config_dict.get("hidden_dim", -1))
    source_depth = int(source_config_dict.get("depth", -1))
    source_uses_slots = bool(source_config_dict.get("uses_uma_slot_tokens", False))

    if source_state_dim != _EXPECTED_SOURCE_STATE_DIM:
        raise SystemExit(
            f"v5 action init refuses non-v3.8 source: `{source}` has "
            f"state_dim={source_state_dim}, expected "
            f"{_EXPECTED_SOURCE_STATE_DIM}. v3.7 and earlier ckpts have "
            f"different state_encoder.0 shapes; v5 only widens the ACTION "
            f"input projection (state schema unchanged). Use a v3.8-lineage "
            f"promoted ckpt (per v3.8 A1 verdict)."
        )
    if source_action_dim != _EXPECTED_SOURCE_ACTION_DIM:
        raise SystemExit(
            f"v5 action init refuses source with action_dim={source_action_dim}; "
            f"expected {_EXPECTED_SOURCE_ACTION_DIM}. v5 widens the action "
            f"input from 52 → 57 — any other source width corrupts the merge."
        )
    if source_uses_slots:
        raise SystemExit(
            f"v5 action init refuses slot-token source: `{source}` has "
            f"uses_uma_slot_tokens=True. v5 keeps the no-slot-token contract."
        )
    if source_hidden != _EXPECTED_HIDDEN_DIM:
        raise SystemExit(
            f"v5 action init refuses non-cap128 source: `{source}` has "
            f"hidden_dim={source_hidden}, expected {_EXPECTED_HIDDEN_DIM}. "
            f"v3.8 §13.3 trunk-contraindication still applies to v5."
        )
    if source_depth != _EXPECTED_DEPTH:
        raise SystemExit(
            f"v5 action init refuses non-depth-2 source: `{source}` has "
            f"depth={source_depth}, expected {_EXPECTED_DEPTH}. v3.8 §13.3 "
            f"trunk-contraindication still applies to v5."
        )

    source_feature_schema = dict(payload.get("feature_schema") or {})
    src_action_schema = source_feature_schema.get(
        "action_feature_schema_version", _EXPECTED_SOURCE_ACTION_SCHEMA_VERSION
    )
    if int(src_action_schema) != _EXPECTED_SOURCE_ACTION_SCHEMA_VERSION:
        raise SystemExit(
            f"v5 action init refuses source with "
            f"action_feature_schema_version={src_action_schema}; v3.8 "
            f"expects action schema {_EXPECTED_SOURCE_ACTION_SCHEMA_VERSION}. "
            f"Mismatch signals a wrong-lineage source."
        )

    src_state_schema = source_feature_schema.get("state_feature_schema_version")
    if src_state_schema is not None:
        try:
            ssv = float(src_state_schema)
        except (TypeError, ValueError):
            ssv = None
        if ssv is not None and abs(ssv - _EXPECTED_SOURCE_STATE_SCHEMA_VERSION) > 1e-6:
            raise SystemExit(
                f"v5 action init refuses non-v3.8 lineage: `{source}` has "
                f"state_feature_schema_version={src_state_schema}, expected "
                f"{_EXPECTED_SOURCE_STATE_SCHEMA_VERSION}. Use a v3.8 ckpt."
            )

    action_widened_key = _find_action_input_projection_key(source_state)
    source_action_weight = source_state[action_widened_key]
    if source_action_weight.shape[-1] != _EXPECTED_SOURCE_ACTION_DIM:
        raise SystemExit(
            f"resolved action projection key `{action_widened_key}` has "
            f"shape {tuple(source_action_weight.shape)}; expected last dim "
            f"{_EXPECTED_SOURCE_ACTION_DIM}."
        )

    # Build the v5 target ModelConfig: copy source config, set action_dim
    # to 57. state_dim unchanged at 304. uses_uma_slot_tokens stays False.
    target_config_dict = dict(source_config_dict)
    target_config_dict["state_dim"] = _TARGET_STATE_DIM  # unchanged
    target_config_dict["action_dim"] = _TARGET_ACTION_DIM
    target_config_dict["uses_uma_slot_tokens"] = False
    target_config = ModelConfig.from_dict(target_config_dict)
    if target_config.state_dim != _TARGET_STATE_DIM:
        raise SystemExit(
            f"internal: target ModelConfig.state_dim did not propagate "
            f"{_TARGET_STATE_DIM} after `from_dict`."
        )
    if target_config.action_dim != _TARGET_ACTION_DIM:
        raise SystemExit(
            f"internal: target ModelConfig.action_dim did not propagate "
            f"{_TARGET_ACTION_DIM} after `from_dict`."
        )

    # Construct a fresh v5 model.
    fresh_target = CandidatePolicyNet(target_config)
    fresh_target.eval()
    fresh_target_state = fresh_target.state_dict()

    fresh_keys = set(fresh_target_state.keys())
    source_keys = set(source_state.keys())
    if fresh_keys != source_keys:
        missing = source_keys - fresh_keys
        extra = fresh_keys - source_keys
        raise SystemExit(
            f"v5 model state_dict keys do not match v3.8 source. "
            f"missing in target: {sorted(missing)[:6]}; extra in target: "
            f"{sorted(extra)[:6]}. v5 should be additive at the action "
            f"input projection only — no key churn."
        )

    fresh_action_shape = fresh_target_state[action_widened_key].shape
    if fresh_action_shape[-1] != _TARGET_ACTION_DIM:
        raise SystemExit(
            f"fresh v5 model's action projection `{action_widened_key}` "
            f"has shape {tuple(fresh_action_shape)}; expected last dim "
            f"{_TARGET_ACTION_DIM}."
        )

    # Construct the merged state_dict.
    merged_state: dict[str, torch.Tensor] = {}
    for key in source_keys:
        if key == action_widened_key:
            continue
        merged_state[key] = source_state[key].clone().detach()

    # Action weight: [out_dim, 52] → [out_dim, 57]. Old [0:52] verbatim,
    # new [52:57] zero-init.
    out_dim = source_action_weight.shape[0]
    widened_action = torch.zeros(
        (out_dim, _TARGET_ACTION_DIM),
        dtype=source_action_weight.dtype,
        device=source_action_weight.device,
    )
    widened_action[:, 0:_EXPECTED_SOURCE_ACTION_DIM] = source_action_weight.clone().detach()
    if widened_action.shape != fresh_action_shape:
        raise SystemExit(
            f"widened action tensor shape mismatch: got "
            f"{tuple(widened_action.shape)}, target expects "
            f"{tuple(fresh_action_shape)}."
        )
    merged_state[action_widened_key] = widened_action

    # Strict-load.
    verify_model = CandidatePolicyNet(target_config)
    missing, unexpected = verify_model.load_state_dict(merged_state, strict=True)
    if missing or unexpected:
        raise SystemExit(
            f"strict load mismatch: missing={list(missing)} "
            f"unexpected={list(unexpected)} — merged state_dict does not "
            f"match the v5 model schema exactly."
        )

    # Iter-0 parity contract. v5's zero-init append columns mean the only
    # Linear contributions from the new tail are 0 regardless of input.
    # Drive action_v5[:, :, 52:57] = 0 in the synthetic batch so the only
    # signal comes from bit-identical [0:52] action columns. Residual is
    # bit-identical modulo f32 rounding.
    verify_v38 = CandidatePolicyNet(ModelConfig.from_dict(source_config_dict))
    missing, unexpected = verify_v38.load_state_dict(source_state, strict=True)
    if missing or unexpected:
        raise SystemExit(
            f"v3.8 source state_dict failed strict load: "
            f"missing={list(missing)} unexpected={list(unexpected)}."
        )
    verify_v38.eval()
    verify_model.eval()

    torch.manual_seed(2026)
    batch = 2
    actions = 8
    state_features = torch.randn(batch, _TARGET_STATE_DIM)
    action_features_v38 = torch.randn(batch, actions, _EXPECTED_SOURCE_ACTION_DIM)
    action_features_v5 = torch.zeros(batch, actions, _TARGET_ACTION_DIM)
    action_features_v5[:, :, 0:_EXPECTED_SOURCE_ACTION_DIM] = action_features_v38
    action_mask = torch.ones(batch, actions, dtype=torch.bool)

    from uma_ai.model import (  # noqa: E402
        ACTION_PAIR_FANOUT,
        NUM_ZONES,
    )
    from uma_ai.features import CARD_ID_SHAPES  # noqa: E402

    max_zone_cards = max(CARD_ID_SHAPES.values())
    card_ids_by_zone = torch.zeros(batch, NUM_ZONES, max_zone_cards, dtype=torch.long)
    action_card_idx = torch.zeros(batch, actions, ACTION_PAIR_FANOUT, dtype=torch.long)

    with torch.no_grad():
        out_v38 = verify_v38(
            state_features=state_features,
            action_features=action_features_v38,
            action_mask=action_mask,
            card_ids_by_zone=card_ids_by_zone,
            action_card_idx=action_card_idx,
        )
        out_v5 = verify_model(
            state_features=state_features,
            action_features=action_features_v5,
            action_mask=action_mask,
            card_ids_by_zone=card_ids_by_zone,
            action_card_idx=action_card_idx,
        )

    def _to_tensor(x: Any) -> torch.Tensor | None:
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, tuple):
            return x[0] if isinstance(x[0], torch.Tensor) else None
        if hasattr(x, "logits"):
            return x.logits
        return None

    logits_v38 = _to_tensor(out_v38)
    logits_v5 = _to_tensor(out_v5)
    if logits_v38 is None or logits_v5 is None:
        raise SystemExit(
            f"could not extract logits from model output (types: "
            f"v38={type(out_v38).__name__}, v5={type(out_v5).__name__})"
        )
    delta_logits = float((logits_v38 - logits_v5).abs().max().item())
    if delta_logits > PARITY_TOLERANCE:
        raise SystemExit(
            f"v5 init parity exceeded tolerance; max|Δlogits|="
            f"{delta_logits:.3e} > {PARITY_TOLERANCE:.0e}. v5 is a "
            f"pure-zero-init append on action[52:57] — non-trivial drift "
            f"should be impossible. Likely cause: column-copy mistake OR "
            f"ModelConfig propagation drifted between v3.8 and v5 graphs."
        )

    # Update the feature_schema metadata. state_* fields UNCHANGED.
    feature_schema = dict(source_feature_schema)
    feature_schema["state_dim"] = _TARGET_STATE_DIM  # unchanged
    feature_schema["state_feature_schema_version"] = _EXPECTED_SOURCE_STATE_SCHEMA_VERSION
    feature_schema["action_feature_schema_version"] = _TARGET_ACTION_SCHEMA_VERSION
    feature_schema["action_dim"] = _TARGET_ACTION_DIM

    provenance = {
        "method": "v5 action-disambiguation action-tail zero-init append warm-start",
        "source_checkpoint": str(source),
        "source_state_dim": source_state_dim,
        "dest_state_dim": _TARGET_STATE_DIM,
        "source_action_dim": source_action_dim,
        "dest_action_dim": _TARGET_ACTION_DIM,
        "action_widened_tensor": action_widened_key,
        "action_widened_from": list(source_action_weight.shape),
        "action_widened_to": list(widened_action.shape),
        "state_encoder_widened": False,
        "dropped_columns": None,
        "appended_zero_init_action_columns": _TARGET_ACTION_DIM - _EXPECTED_SOURCE_ACTION_DIM,
        "trunk_widened": False,
        "hidden_dim": _EXPECTED_HIDDEN_DIM,
        "depth": _EXPECTED_DEPTH,
        "all_other_params": "copied verbatim",
        "delta_logits_at_init": delta_logits,
        "delta_logits_tolerance": PARITY_TOLERANCE,
        "builder": "training/make_v5_action_init.py",
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
        "--source-ckpt",
        "--source",
        dest="source",
        required=True,
        help="Path to the v3.8 source checkpoint to warm-start from "
             "(e.g. the v3.8 A1 promoted ckpt). Refuses non-v3.8 sources "
             "(state_dim != 304 OR action_schema != 4), non-default trunk "
             "(hidden != 128 or depth != 2), or slot-token ckpts.",
    )
    parser.add_argument(
        "--output",
        "--dest",
        dest="dest",
        required=True,
        help="Path to write the v5 init checkpoint.",
    )
    args = parser.parse_args()
    source = Path(args.source)
    dest = Path(args.dest)
    provenance = build(source, dest)
    print(
        f"v5 action init built from {source}, output={dest}, "
        f"delta_logits={provenance['delta_logits_at_init']:.3e} "
        f"(< {PARITY_TOLERANCE:.0e} tolerance)"
    )
    print(json.dumps({"status": "PASS", "dest": str(dest), **provenance}, indent=2))


if __name__ == "__main__":
    main()
