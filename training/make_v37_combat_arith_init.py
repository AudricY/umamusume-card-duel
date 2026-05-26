"""v37-combat-arith-and-catalog: build a v3.7 init from a v3.6 source checkpoint.

v3.7 extends v3.6's 246-d state-features vector to 296-d by APPENDING 50
zero-init columns at indices [246:296] — no column-drop. The new tail
encodes combat-arithmetic + catalog-lookup channels (weakness-adjusted
lethal, coin-flip expected damage, conditional-damage-bonus presence,
energy-ETA + paralysis-window, tool/ability effect-kind one-hot) per the
LOCKED layout in
`docs/ai-research/scoping/v37-combat-arith-and-catalog-scoping.md` §4
step 3.

See scoping doc §3.2 "Existing checkpoint compatibility" for the iter-0
parity contract (Δlogits ≤ 1e-5 — strict, mirrors v3.5's band) and §4
step 6 for the builder contract.

Methodology (mirror of `training/make_v36_priors_init.py` but with
column-drop removed — pure append at tail):

  1. Read a v3.6 source ckpt (state_dim=246).
  2. Refuse if state_dim != 246, if uses_uma_slot_tokens is True (slot
     tokens excluded across v3.5/v3.6/v3.7 doctrine), or if
     action_schema_version != 3.
  3. Build a fresh v3.7 `CandidatePolicyNet` (state_dim=296) and
     discover the new shape of `state_encoder.0.weight` from
     construction.
  4. Construct the merged state_dict:
     - For every key OTHER than `state_encoder.0.weight`, copy verbatim
       from the source (v3.7 graph preserves them all).
     - For `state_encoder.0.weight` of shape `(hidden, 246)`:
         * Copy old columns [0:246] → new [0:246]
         * Zero  new columns [246:296] (truly-new v3.7 tail)
     - The bias is unchanged.
  5. Strict-load the merged state into a fresh v3.7 model — must
     accept every key, no missing, no unexpected.
  6. Sanity check: feed a fixed deterministic observation through both
     the v3.6 source and the v3.7 merged model. Since the new columns
     are exactly zero-init AND the new state vector's [246:296] band
     is whatever the synthetic batch sets (we drive it to ZERO so the
     v3.6 head's contribution is the only signal), the residual is
     bit-identical modulo f32 rounding. Tolerance: 1e-5 (strict,
     matching v3.5's band — there's no column-drop here unlike v3.6
     which used 1e-3).

Refuses non-v3.6 sources: v3.0/v3.1/v3.3/v3.5/v3.4 ckpts have different
state_encoder.0 shapes (110/164/167/212-d) and would silently produce a
broken merge. v3.4 slot-token ckpts are also refused (doctrine: slot
tokens stay retired).

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
    STATE_DIM_V3_6,
    STATE_DIM_V3_7,
    STATE_FEATURE_SCHEMA_VERSION_V3_7,
)
from uma_ai.model import (  # noqa: E402
    CandidatePolicyNet,
    ModelConfig,
)

WIDENED_KEY = "state_encoder.0.weight"

# v3.7 is a pure append at the tail — no column-drop. The strict
# tolerance band matches v3.5's contract (1e-5); v3.6 was looser (1e-3)
# ONLY because of its column-drop semantics on the dead opp.energy_zone
# band. v3.7 reuses every v3.6 column verbatim, so iter-0 logits are
# bit-identical modulo f32 rounding when the new tail is driven to 0.
PARITY_TOLERANCE = 1e-5

# Action schema is shared across v3.3/v3.5/v3.6/v3.7 (unchanged at 3).
_EXPECTED_ACTION_SCHEMA_VERSION = 3


def build(source: Path, dest: Path) -> dict[str, Any]:
    """Build a v3.7 init checkpoint from a v3.6 source checkpoint.

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
            f"a v3.6 training checkpoint emitted by `train_bc.py`."
        )

    source_state: dict[str, torch.Tensor] = payload["model_state"]
    source_config_dict: dict[str, Any] = dict(payload["model_config"])
    source_state_dim = int(source_config_dict.get("state_dim", -1))
    source_uses_slots = bool(source_config_dict.get("uses_uma_slot_tokens", False))

    if source_state_dim != STATE_DIM_V3_6:
        raise SystemExit(
            f"v37-combat-arith builder refuses non-v3.6 source: "
            f"`{source}` has state_dim={source_state_dim}, expected "
            f"{STATE_DIM_V3_6}. v3.0/v3.1/v3.3/v3.5 have different "
            f"state_encoder.0 shapes (110/164/167/212-d) and would "
            f"silently corrupt the merge. Use a v3.6 ckpt (e.g. "
            f"runs/R16-P1-v36-cap128-A1-continue/loop/iter-2/checkpoint.pt)."
        )
    if source_uses_slots:
        raise SystemExit(
            f"v37-combat-arith builder refuses slot-token source: "
            f"`{source}` has uses_uma_slot_tokens=True. v3.7 retains "
            f"the no-slot-token contract (state_dim=296, 5-input ONNX); "
            f"slot tokens FALSIFIED at progress/r110.md §4h. Pass a v3.6 "
            f"ckpt (uses_uma_slot_tokens=False)."
        )

    source_feature_schema = dict(payload.get("feature_schema") or {})
    src_action_schema = source_feature_schema.get(
        "action_feature_schema_version", _EXPECTED_ACTION_SCHEMA_VERSION
    )
    if int(src_action_schema) != _EXPECTED_ACTION_SCHEMA_VERSION:
        raise SystemExit(
            f"v37-combat-arith builder refuses source with "
            f"action_feature_schema_version={src_action_schema}; v3.7 "
            f"expects action schema {_EXPECTED_ACTION_SCHEMA_VERSION}. "
            f"Action vector is untouched across v3.6 → v3.7; mismatched "
            f"action schemas signal a wrong-lineage source."
        )

    # Extra v3.7 guard: refuse non-v3.6 lineage flagged via state feature
    # schema version (catches v3.5 ckpts that happen to share state_dim
    # — they don't today, but the guard is cheap and surfaces lineage
    # surprises loudly).
    src_state_schema = source_feature_schema.get("state_feature_schema_version")
    if src_state_schema is not None:
        try:
            ssv = float(src_state_schema)
        except (TypeError, ValueError):
            ssv = None
        if ssv is not None and abs(ssv - 3.6) > 1e-6:
            raise SystemExit(
                f"v37-combat-arith builder refuses non-v3.6 lineage: "
                f"`{source}` has state_feature_schema_version={src_state_schema}, "
                f"expected 3.6. Use a v3.6 ckpt (cont-iter2 best at "
                f"runs/R16-P1-v36-cap128-A1-continue/loop/iter-2/checkpoint.pt)."
            )

    if WIDENED_KEY not in source_state:
        raise SystemExit(
            f"source state_dict missing `{WIDENED_KEY}`. Cannot perform "
            f"the v3.6 → v3.7 state-vector append without it."
        )
    source_weight = source_state[WIDENED_KEY]
    if source_weight.shape[-1] != STATE_DIM_V3_6:
        raise SystemExit(
            f"`{WIDENED_KEY}` has shape {tuple(source_weight.shape)}; "
            f"expected last dim {STATE_DIM_V3_6}. Source ckpt schema "
            f"disagrees with model_config.state_dim."
        )

    # Build the v3.7 target ModelConfig: copy source config, set state_dim
    # to 296. uses_uma_slot_tokens stays False (v3.7 EXCLUDES slot tokens
    # per scoping doc §3 / §13).
    target_config_dict = dict(source_config_dict)
    target_config_dict["state_dim"] = STATE_DIM_V3_7
    target_config_dict["uses_uma_slot_tokens"] = False
    target_config = ModelConfig.from_dict(target_config_dict)
    if target_config.state_dim != STATE_DIM_V3_7:
        raise SystemExit(
            f"internal: target ModelConfig.state_dim did not propagate "
            f"{STATE_DIM_V3_7} after `from_dict` — ModelConfig schema "
            f"drifted."
        )

    # Construct a fresh v3.7 model to learn the new shape of the widened
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
            f"v3.7 model state_dict keys do not match v3.6 source. "
            f"missing in target: {sorted(missing)[:6]}; extra in target: "
            f"{sorted(extra)[:6]}. v3.7 should be additive at "
            f"state_encoder.0 only — no key churn."
        )

    fresh_widened_shape = fresh_target_state[WIDENED_KEY].shape
    if fresh_widened_shape[-1] != STATE_DIM_V3_7:
        raise SystemExit(
            f"fresh v3.7 model's `{WIDENED_KEY}` has shape "
            f"{tuple(fresh_widened_shape)}; expected last dim "
            f"{STATE_DIM_V3_7}. Model construction is not producing the "
            f"widened tensor — check ModelConfig.state_dim plumbing."
        )

    # Construct the merged state_dict. Every key OTHER than the widened
    # one is copied verbatim. The widened key is rebuilt as
    # [source | zero-padding] at the tail.
    merged_state: dict[str, torch.Tensor] = {}
    for key in source_keys:
        if key == WIDENED_KEY:
            continue
        merged_state[key] = source_state[key].clone().detach()

    # Build the [hidden, 296] tensor:
    #   [0:246]   ← source [0:246]  (copy verbatim — v3.6 head frozen)
    #   [246:296] ← zeros           (truly-new v3.7 tail)
    hidden = source_weight.shape[0]
    widened = torch.zeros(
        (hidden, STATE_DIM_V3_7),
        dtype=source_weight.dtype,
        device=source_weight.device,
    )
    widened[:, 0:STATE_DIM_V3_6] = source_weight.clone().detach()
    # [STATE_DIM_V3_6:STATE_DIM_V3_7] stays zero (appended tail zero-init).
    if widened.shape != fresh_widened_shape:
        raise SystemExit(
            f"widened tensor shape mismatch: got {tuple(widened.shape)}, "
            f"target expects {tuple(fresh_widened_shape)}."
        )
    merged_state[WIDENED_KEY] = widened

    # Strict-load into a fresh v3.7 model.
    verify_model = CandidatePolicyNet(target_config)
    missing, unexpected = verify_model.load_state_dict(merged_state, strict=True)
    if missing or unexpected:
        raise SystemExit(
            f"strict load mismatch: missing={list(missing)} "
            f"unexpected={list(unexpected)} — merged state_dict does not "
            f"match the v3.7 model schema exactly."
        )

    # Iter-0 parity contract: feed a fixed observation through both the
    # v3.6 source and the v3.7 merged model. Because v3.7's Linear has
    # zero-init columns at [246:296], the only Linear contribution from
    # the new tail is 0 regardless of input. We drive state_v37[246:296]
    # to 0 in the synthetic batch so the only signal is from columns
    # [0:246], which were copied byte-identically from the source. The
    # residual is therefore bit-identical modulo f32 rounding (1e-5
    # tolerance — strict).
    verify_v36 = CandidatePolicyNet(ModelConfig.from_dict(source_config_dict))
    missing, unexpected = verify_v36.load_state_dict(source_state, strict=True)
    if missing or unexpected:
        raise SystemExit(
            f"v3.6 source state_dict failed strict load: missing={list(missing)} "
            f"unexpected={list(unexpected)}."
        )
    verify_v36.eval()
    verify_model.eval()

    torch.manual_seed(2026)
    batch = 2
    actions = 8
    # Synthetic batch matching the model's expected inputs.
    state_v36 = torch.randn(batch, STATE_DIM_V3_6)
    # v3.7 sees [0:246] = v3.6, [246:296] = 0 (zero-init append).
    state_v37 = torch.zeros(batch, STATE_DIM_V3_7)
    state_v37[:, 0:STATE_DIM_V3_6] = state_v36
    action_features = torch.randn(batch, actions, target_config.action_dim)
    action_mask = torch.ones(batch, actions, dtype=torch.bool)

    # Auxiliary inputs (zeros). What we're comparing is the v3.6 vs v3.7
    # trunk on identical aux inputs.
    from uma_ai.model import (  # noqa: E402
        ACTION_PAIR_FANOUT,
        NUM_ZONES,
    )
    from uma_ai.features import CARD_ID_SHAPES  # noqa: E402

    max_zone_cards = max(CARD_ID_SHAPES.values())
    card_ids_by_zone = torch.zeros(batch, NUM_ZONES, max_zone_cards, dtype=torch.long)
    action_card_idx = torch.zeros(batch, actions, ACTION_PAIR_FANOUT, dtype=torch.long)

    extra_kwargs: dict[str, torch.Tensor] = {}

    with torch.no_grad():
        out_v36 = verify_v36(
            state_features=state_v36,
            action_features=action_features,
            action_mask=action_mask,
            card_ids_by_zone=card_ids_by_zone,
            action_card_idx=action_card_idx,
            **extra_kwargs,
        )
        out_v37 = verify_model(
            state_features=state_v37,
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
            return x[0] if isinstance(x[0], torch.Tensor) else None
        if hasattr(x, "logits"):
            return x.logits
        return None

    logits_v36 = _to_tensor(out_v36)
    logits_v37 = _to_tensor(out_v37)
    if logits_v36 is None or logits_v37 is None:
        raise SystemExit(
            f"could not extract logits from model output for verification "
            f"(types: v36={type(out_v36).__name__}, v37={type(out_v37).__name__})"
        )
    delta_logits = float((logits_v36 - logits_v37).abs().max().item())
    if delta_logits > PARITY_TOLERANCE:
        raise SystemExit(
            f"v3.7 init parity exceeded tolerance; max|Δlogits|="
            f"{delta_logits:.3e} > {PARITY_TOLERANCE:.0e}. "
            f"v3.7 is a pure-zero-init append — non-trivial drift should "
            f"be impossible. Likely cause: the v3.6 → v3.7 column copy "
            f"is wrong, OR ModelConfig propagation drifted between v3.6 "
            f"and v3.7 graphs."
        )

    # Update the feature_schema metadata so downstream consumers tag this
    # as v3.7.
    feature_schema = dict(source_feature_schema)
    feature_schema["state_dim"] = STATE_DIM_V3_7
    feature_schema["state_feature_schema_version"] = STATE_FEATURE_SCHEMA_VERSION_V3_7

    provenance = {
        "method": "v3.7 combat-arith-and-catalog tail zero-init append warm-start",
        "source_checkpoint": str(source),
        "source_state_dim": source_state_dim,
        "dest_state_dim": STATE_DIM_V3_7,
        "widened_tensor": WIDENED_KEY,
        "widened_from": list(source_weight.shape),
        "widened_to": list(widened.shape),
        "dropped_columns": None,
        "appended_zero_init_columns": STATE_DIM_V3_7 - STATE_DIM_V3_6,
        "all_other_params": "copied verbatim",
        "delta_logits_at_init": delta_logits,
        "delta_logits_tolerance": PARITY_TOLERANCE,
        "builder": "training/make_v37_combat_arith_init.py",
    }

    # Warm-start init only (matches `make_v36_priors_init.py`): drop
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
        "--source-ckpt",
        "--source",
        dest="source",
        required=True,
        help="Path to the v3.6 source checkpoint to warm-start from "
             "(e.g. `runs/R16-P1-v36-cap128-A1-continue/loop/iter-2/checkpoint.pt`). "
             "Refuses non-v3.6 sources (state_dim != 246 or slot-tokens=True).",
    )
    parser.add_argument(
        "--output",
        "--dest",
        dest="dest",
        required=True,
        help="Path to write the v3.7 init checkpoint.",
    )
    args = parser.parse_args()
    source = Path(args.source)
    dest = Path(args.dest)
    provenance = build(source, dest)
    print(
        f"v3.7 init built from {source}, output={dest}, "
        f"delta_logits={provenance['delta_logits_at_init']:.3e} "
        f"(< {PARITY_TOLERANCE:.0e} tolerance)"
    )
    print(json.dumps({"status": "PASS", "dest": str(dest), **provenance}, indent=2))


if __name__ == "__main__":
    main()
