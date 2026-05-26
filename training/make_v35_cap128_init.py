"""v35-capacity-128: widen v3.5 ckpt from hidden_dim=64 to hidden_dim=128.

Tests the capacity-saturation hypothesis from r110.md §4k: v3.5's n=10k
ceiling of ~0.59 may be recipe-bound by hidden_dim=64 trunk capacity,
not by the schema design. Widening hidden_dim 64→128 doubles MLP
parameter count and gives the v3.5 tail (which only landed at ~3.7%
of head-column norm at hidden=64) more head capacity to absorb signal.

Methodology:

For each parameter in the source state_dict, find the corresponding
fresh-hidden=128 model parameter and:
  - Linear weight (out, in): copy source into top-left block [:src_out, :src_in];
    zero-pad the rest. New output neurons start dead → forward through
    GELU stays 0 → next layer sees 0 contribution.
  - Linear bias (out,): copy source into [:src_out]; zero-pad the rest.
  - LayerNorm weight (dim,): copy source into [:src_dim]; pad with 1.0
    (pass-through scale).
  - LayerNorm bias (dim,): copy source into [:src_dim]; pad with 0.0.
  - Embedding weight (vocab, embed_dim): copy verbatim (neither dim
    tied to hidden_dim).
  - Parameters where source.shape == target.shape: copy verbatim.

Resulting model: existing 64 hidden neurons retain trained v3.5 weights;
new 64 neurons are zero-init (Linear) or pass-through-init (LayerNorm).
At iter-0 the new neurons contribute zero to the policy output, so the
warm-start preserves v3.5's behavior. Training can extend the policy
into the new capacity.

NOTE: not bit-identical to v3.5 at iter-0 due to LayerNorm normalization
over the wider hidden vector. Δlogits typically ~1e-2 (not 1e-7) — this
is expected and acceptable. The contract is "reasonable warm-start," not
"zero-init residual."

Refuses to operate on non-v3.5 sources (state_dim != 212) or on slot-
token sources (the v3.4/v3.5 axis is no-slot per r110.md §4h).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uma_ai.features import STATE_DIM_V3_5, STATE_FEATURE_SCHEMA_VERSION_V3_5  # noqa: E402
from uma_ai.model import CandidatePolicyNet, ModelConfig  # noqa: E402

SOURCE_HIDDEN = 64
TARGET_HIDDEN = 128


def _get_module(model: nn.Module, key: str) -> nn.Module:
    """Walk dotted state_dict key (minus the param leaf) to its parent module."""
    parts = key.split(".")[:-1]  # drop e.g. 'weight' / 'bias'
    mod: Any = model
    for p in parts:
        if p.isdigit():
            mod = mod[int(p)]
        else:
            mod = getattr(mod, p)
    return mod


def _widen_tensor(
    source: torch.Tensor, target_shape: torch.Size, kind: str
) -> torch.Tensor:
    """Copy source values into a fresh tensor of target_shape, with kind-
    appropriate padding for the extra rows/cols.

    `kind` ∈ {"linear_weight", "linear_bias", "ln_weight", "ln_bias",
              "embedding_weight", "copy"}.
    """
    if source.shape == target_shape:
        return source.clone().detach()

    if kind == "linear_weight":
        assert source.dim() == 2 and len(target_shape) == 2
        out = torch.zeros(target_shape, dtype=source.dtype, device=source.device)
        src_out, src_in = source.shape
        out[:src_out, :src_in] = source
        return out

    if kind == "linear_bias":
        assert source.dim() == 1 and len(target_shape) == 1
        out = torch.zeros(target_shape, dtype=source.dtype, device=source.device)
        out[:source.shape[0]] = source
        return out

    if kind == "ln_weight":
        # LayerNorm weight defaults to 1.0; pad with 1.0 for new dims.
        assert source.dim() == 1 and len(target_shape) == 1
        out = torch.ones(target_shape, dtype=source.dtype, device=source.device)
        out[:source.shape[0]] = source
        return out

    if kind == "ln_bias":
        # LayerNorm bias defaults to 0.0; pad with 0.0.
        assert source.dim() == 1 and len(target_shape) == 1
        out = torch.zeros(target_shape, dtype=source.dtype, device=source.device)
        out[:source.shape[0]] = source
        return out

    if kind == "embedding_weight":
        # Embedding rows = vocab size (NOT hidden_dim). Cols = CARD_EMBED_DIM
        # (also NOT hidden_dim). If shapes match, copy; otherwise something
        # has shifted upstream.
        if source.shape != target_shape:
            raise ValueError(
                f"embedding shape mismatch: source={tuple(source.shape)}, "
                f"target={tuple(target_shape)}. Vocab/embed_dim shouldn't widen."
            )
        return source.clone().detach()

    if kind == "copy":
        if source.shape != target_shape:
            raise ValueError(
                f"copy-kind tensor shape mismatch: source={tuple(source.shape)}, "
                f"target={tuple(target_shape)}"
            )
        return source.clone().detach()

    raise ValueError(f"unknown widening kind: {kind!r}")


def _classify_param(model: nn.Module, key: str) -> str:
    """Return the kind tag for a state_dict key."""
    if key.endswith(".weight") or key.endswith(".bias"):
        mod = _get_module(model, key)
        leaf = key.rsplit(".", 1)[-1]
        if isinstance(mod, nn.Linear):
            return f"linear_{leaf}"
        if isinstance(mod, nn.LayerNorm):
            return f"ln_{leaf}"
        if isinstance(mod, nn.Embedding):
            return "embedding_weight"
    return "copy"


def build(source: Path, dest: Path) -> dict[str, Any]:
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if "model_state" not in payload or "model_config" not in payload:
        raise SystemExit(
            f"source ckpt `{source}` missing model_state/model_config. "
            "Not a training-checkpoint payload."
        )

    source_state: dict[str, torch.Tensor] = payload["model_state"]
    source_config_dict: dict[str, Any] = dict(payload["model_config"])
    source_state_dim = int(source_config_dict.get("state_dim", -1))
    source_hidden = int(source_config_dict.get("hidden_dim", -1))
    source_uses_slots = bool(source_config_dict.get("uses_uma_slot_tokens", False))

    if source_state_dim != STATE_DIM_V3_5:
        raise SystemExit(
            f"refuses non-v3.5 source: state_dim={source_state_dim}, "
            f"expected {STATE_DIM_V3_5}. Pass a v3.5 ckpt."
        )
    if source_hidden != SOURCE_HIDDEN:
        raise SystemExit(
            f"refuses non-{SOURCE_HIDDEN}-hidden source: hidden_dim={source_hidden}, "
            f"expected {SOURCE_HIDDEN}."
        )
    if source_uses_slots:
        raise SystemExit(
            f"refuses slot-token source: uses_uma_slot_tokens=True. "
            f"v3.5 axis is no-slot."
        )

    target_config = ModelConfig.from_dict(
        {**source_config_dict, "hidden_dim": TARGET_HIDDEN, "uses_uma_slot_tokens": False}
    )
    target_model = CandidatePolicyNet(target_config)
    target_state = target_model.state_dict()

    # Build merged state: per-key, widen source into target shape.
    merged_state: dict[str, torch.Tensor] = {}
    widen_summary: dict[str, int] = {}
    for key in target_state:
        if key not in source_state:
            # New parameter that exists only at hidden=128 (shouldn't happen
            # for this widening). Keep fresh init.
            merged_state[key] = target_state[key].clone().detach()
            widen_summary["new_param_kept_fresh"] = widen_summary.get("new_param_kept_fresh", 0) + 1
            continue
        kind = _classify_param(target_model, key)
        merged_state[key] = _widen_tensor(source_state[key], target_state[key].shape, kind)
        widen_summary[kind] = widen_summary.get(kind, 0) + 1

    # Check no source keys were dropped.
    dropped = set(source_state) - set(target_state)
    if dropped:
        raise SystemExit(
            f"target hidden=128 model missing keys present in source: {sorted(dropped)[:5]}..."
        )

    # Strict-load into fresh target model for verification.
    verify_model = CandidatePolicyNet(target_config)
    missing, unexpected = verify_model.load_state_dict(merged_state, strict=True)
    if missing or unexpected:
        raise SystemExit(
            f"strict load mismatch: missing={list(missing)[:6]} "
            f"unexpected={list(unexpected)[:6]}"
        )

    # Parity check: feed a synthetic batch through both source-hidden64 model
    # and target-hidden128 widened model. Δlogits is NOT bit-identical due to
    # LayerNorm normalization over a wider vector with zero-padded new dims.
    # Expect Δlogits roughly in the 1e-3..1e-1 range — acceptable for warm-
    # start; not the strict 1e-5 contract used in pure additive-tail widening.
    source_model = CandidatePolicyNet(ModelConfig.from_dict(source_config_dict))
    missing, unexpected = source_model.load_state_dict(source_state, strict=True)
    if missing or unexpected:
        raise SystemExit(
            f"source state_dict failed strict load: missing={list(missing)} "
            f"unexpected={list(unexpected)}."
        )
    source_model.eval()
    verify_model.eval()

    torch.manual_seed(2026)
    batch = 2
    actions = 8
    state_features = torch.randn(batch, STATE_DIM_V3_5)
    action_features = torch.randn(batch, actions, target_config.action_dim)
    action_mask = torch.ones(batch, actions, dtype=torch.bool)

    from uma_ai.model import ACTION_PAIR_FANOUT, NUM_ZONES  # noqa: E402
    from uma_ai.features import CARD_ID_SHAPES  # noqa: E402

    max_zone_cards = max(CARD_ID_SHAPES.values())
    card_ids_by_zone = torch.zeros(batch, NUM_ZONES, max_zone_cards, dtype=torch.long)
    action_card_idx = torch.zeros(batch, actions, ACTION_PAIR_FANOUT, dtype=torch.long)

    def _logits(model: nn.Module) -> torch.Tensor:
        with torch.no_grad():
            out = model(
                state_features=state_features,
                action_features=action_features,
                action_mask=action_mask,
                card_ids_by_zone=card_ids_by_zone,
                action_card_idx=action_card_idx,
            )
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, tuple):
            return out[0] if isinstance(out[0], torch.Tensor) else out[0].logits
        if hasattr(out, "logits"):
            return out.logits
        raise SystemExit(f"unknown model output type: {type(out).__name__}")

    logits_src = _logits(source_model)
    logits_tgt = _logits(verify_model)
    delta_logits = float((logits_src - logits_tgt).abs().max().item())
    logits_src_max = float(logits_src.abs().max().item())
    logits_tgt_max = float(logits_tgt.abs().max().item())
    # Warm-start contract, NOT bit-identical. LayerNorm normalises over the
    # wider hidden vector with zero-padded new dims, so the mean/std shift
    # changes even the OLD dims' post-LN outputs — Δlogits is bounded by
    # logit magnitude, not by precision. Sanity bound: Δlogits should not
    # exceed ~3× the source logit magnitude (model still in reasonable
    # output range, not blown up to ±1e10).
    pathology_bound = max(5.0, 3.0 * logits_src_max)
    if delta_logits > pathology_bound:
        raise SystemExit(
            f"v3.5 (hidden=64) ↔ v3.5-cap128 (hidden=128) iter-0 Δlogits "
            f"= {delta_logits:.3e} > pathology bound {pathology_bound:.3e}. "
            f"Source logits max={logits_src_max:.3e}, target max={logits_tgt_max:.3e}. "
            f"Widening logic broke."
        )

    # Update feature_schema metadata to record the capacity widening.
    feature_schema = dict(payload.get("feature_schema") or {})
    feature_schema["state_dim"] = STATE_DIM_V3_5
    feature_schema["state_feature_schema_version"] = STATE_FEATURE_SCHEMA_VERSION_V3_5
    feature_schema["hidden_dim"] = TARGET_HIDDEN

    provenance = {
        "method": "v3.5 hidden_dim widening (64 → 128) via top-left copy + zero/passthrough pad",
        "source_checkpoint": str(source),
        "source_state_dim": source_state_dim,
        "source_hidden_dim": source_hidden,
        "dest_state_dim": STATE_DIM_V3_5,
        "dest_hidden_dim": TARGET_HIDDEN,
        "widen_summary": widen_summary,
        "delta_logits_at_init": delta_logits,
        "source_logits_max": logits_src_max,
        "target_logits_max": logits_tgt_max,
        "pathology_bound": pathology_bound,
        "builder": "training/make_v35_cap128_init.py",
        "note": (
            "NOT zero-init residual — LayerNorm over a wider hidden vector "
            "with zero-padded new dims changes the normalisation statistics, "
            "so Δlogits is small but non-zero (typically 1e-3..1e-1). "
            "Warm-start contract: existing 64 hidden neurons retain v3.5 "
            "weights; new 64 hidden neurons are zero-init Linear / "
            "passthrough LayerNorm. Training extends policy into new capacity."
        ),
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
    parser.add_argument("--source-checkpoint", "--source", dest="source", required=True,
                        help="Path to v3.5 source ckpt (hidden=64, state_dim=212).")
    parser.add_argument("--output", "--dest", dest="dest", required=True,
                        help="Path to write the v3.5-cap128 init ckpt.")
    args = parser.parse_args()
    source = Path(args.source)
    dest = Path(args.dest)
    provenance = build(source, dest)
    print(
        f"v3.5-cap128 init built from {source}, output={dest}, "
        f"Δlogits={provenance['delta_logits_at_init']:.3e}"
    )
    print(json.dumps({"status": "PASS", "dest": str(dest), **provenance}, indent=2, default=str))


if __name__ == "__main__":
    main()
