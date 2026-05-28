from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from uma_ai.features import (
    ACTION_DIM,
    ACTION_FEATURE_SCHEMA_VERSION,
    CARD_ID_SHAPES,
    STATE_DIM_V3_3,
    STATE_FEATURE_SCHEMA_VERSION_V3_2,
    STATE_FEATURE_SCHEMA_VERSION_V3_4,
    UMA_SLOT_COUNT,
    UMA_SLOT_FEATURE_DIM,
    card_vocab_metadata,
    feature_builder_for_state_dim,
    schema_version_for_state_dim,
)
from uma_ai.model import CARD_VOCAB_TABLE_SIZE, NUM_ZONES, CandidatePolicyNet, ModelConfig

# R7.b.2 Phase 3: per-zone max-cards width is FIXED in the ONNX graph for
# ORT shape stability — Phase 4's re-extractor pads every row to this same
# width, so dynamic-axis is unnecessary and adds shape-inference cost. The
# batch dimension and the action count stay dynamic since both vary per
# request. `MAX_CARDS_PER_ZONE` is `max(CARD_ID_SHAPES.values())` = 30
# (ownDiscard/oppDiscard cap), matching the collator in
# `training/uma_ai/dataset.py`.
MAX_CARDS_PER_ZONE = max(CARD_ID_SHAPES.values())


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig.from_dict(checkpoint.get("model_config"))
    if args.q_value_scalar is not None:
        if not config.uses_q_value_head:
            raise ValueError("--q-value-scalar override requires a checkpoint with uses_q_value_head=True")
        config = replace(config, q_value_scalar=args.q_value_scalar)
    if args.q_value_scalar_scale != 1.0 or args.q_value_scalar_bias != 0.0:
        if not config.uses_q_value_head:
            raise ValueError("Q scalar calibration overrides require a checkpoint with uses_q_value_head=True")
        config = replace(
            config,
            q_value_scalar_scale=args.q_value_scalar_scale,
            q_value_scalar_bias=args.q_value_scalar_bias,
        )
    # R16-P1: the graph state dim is driven by the CHECKPOINT's
    # config.state_dim (96 / 110 / 164), not the module default STATE_DIM
    # (which stays 110 = v3.0). Validate it is a known builder dim so a
    # 164-d v3.1 checkpoint exports a 164-d graph while 96/110 graphs are
    # byte-unchanged. Unknown dims fail loud (feature_builder_for_state_dim
    # raises) — never silently produce a mis-sized graph.
    if config.action_dim != ACTION_DIM:
        raise ValueError(
            f"Checkpoint action dim {config.action_dim} does not match "
            f"current {ACTION_DIM}"
        )
    graph_state_dim = config.state_dim
    feature_builder_for_state_dim(graph_state_dim)  # fail loud on unknown dim
    expected_vocab = card_vocab_metadata()
    checkpoint_schema = checkpoint.get("feature_schema") or {}
    checkpoint_vocab = checkpoint_schema.get("card_vocab")
    if checkpoint_vocab is not None and expected_vocab.get("hash") != "missing":
        if checkpoint_vocab.get("hash") != expected_vocab.get("hash"):
            raise ValueError(
                f"Card vocab hash mismatch: checkpoint={checkpoint_vocab.get('hash')} "
                f"runtime={expected_vocab.get('hash')}; rebuild vocab or retrain."
            )

    model = CandidatePolicyNet(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # R7.b.2 Phase 3: fail loud if the embedding table outgrew the vocab —
    # `nn.Embedding(108, ...)` with `padding_idx=0` is one row over the
    # vocab to reserve index 0 for unknown/pad. If a future vocab rebuild
    # bumps vocabSize above 107 we want the export to refuse rather than
    # silently produce an under-sized graph that ORT would later index OOB.
    expected_vocab_size = int(expected_vocab.get("vocabSize", 0))
    if expected_vocab_size > 0 and model.card_embed.num_embeddings != expected_vocab_size + 1:
        raise ValueError(
            f"card_embed.num_embeddings ({model.card_embed.num_embeddings}) must equal "
            f"vocabSize+1 ({expected_vocab_size + 1}); rebuild vocab or update "
            f"CARD_VOCAB_TABLE_SIZE in training/uma_ai/model.py."
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    state = torch.zeros((1, graph_state_dim), dtype=torch.float32)
    actions = torch.zeros((1, args.max_actions, ACTION_DIM), dtype=torch.float32)
    mask = torch.ones((1, args.max_actions), dtype=torch.bool)
    # R7.b.2 Phase 3: two new int64 tensors flow into the embedding pass.
    # All zeros are a valid no-op input (index 0 = pad row, kept zero by
    # `padding_idx=0`), matching the additive-residual null behaviour
    # validated in Phase 2's smoke contract (4).
    card_ids_by_zone = torch.zeros((1, NUM_ZONES, MAX_CARDS_PER_ZONE), dtype=torch.int64)
    action_card_idx = torch.zeros((1, args.max_actions, 2), dtype=torch.int64)

    # R16-P2 C5: per-Uma slot-token branch gates the 7-input ONNX graph.
    # When `model_config.uses_uma_slot_tokens` is False (v3.0/v3.1), the
    # export is byte-identical to pre-C5 (5 inputs, 5 input_names, same
    # dynamic_axes, same sidecar fields). When True, two NEW inputs are
    # added positionally AFTER `action_card_idx` to mirror the model's
    # `forward` signature ordering, and the sidecar gains
    # `uses_uma_slot_tokens` + `uma_slot_feature_dim`. The two slot tensors
    # are zero-initialized — at init the `uma_slot_encoder` final Linear is
    # zero-weighted (see C2), so the export trace through the slot branch
    # contributes a structural zero residual; the graph still exercises the
    # Gather/Concat/MatMul ops so ORT shape-inference matches export-time.
    if config.uses_uma_slot_tokens:
        uma_slot_card_ids = torch.zeros((1, UMA_SLOT_COUNT), dtype=torch.int64)
        uma_slot_features = torch.zeros(
            (1, UMA_SLOT_COUNT, UMA_SLOT_FEATURE_DIM), dtype=torch.float32
        )
        positional_inputs = (
            state,
            actions,
            mask,
            card_ids_by_zone,
            action_card_idx,
            uma_slot_card_ids,
            uma_slot_features,
        )
        input_names = [
            "state_features",
            "action_features",
            "action_mask",
            "card_ids_by_zone",
            "action_card_idx",
            "uma_slot_card_ids",
            "uma_slot_features",
        ]
        # Only the batch dim is dynamic on the new inputs. UMA_SLOT_COUNT
        # (=10) and UMA_SLOT_FEATURE_DIM (=23) are FROZEN this chunk; making
        # them dynamic would (a) silently absorb a future layout drift and
        # (b) cost ORT shape-inference time for no real flexibility.
        dynamic_axes = {
            "state_features": {0: "batch"},
            "action_features": {0: "batch", 1: "actions"},
            "action_mask": {0: "batch", 1: "actions"},
            "card_ids_by_zone": {0: "batch"},
            "action_card_idx": {0: "batch", 1: "actions"},
            "uma_slot_card_ids": {0: "batch"},
            "uma_slot_features": {0: "batch"},
            "logits": {0: "batch", 1: "actions"},
            "value": {0: "batch"},
        }
    else:
        positional_inputs = (state, actions, mask, card_ids_by_zone, action_card_idx)
        input_names = [
            "state_features",
            "action_features",
            "action_mask",
            "card_ids_by_zone",
            "action_card_idx",
        ]
        dynamic_axes = {
            "state_features": {0: "batch"},
            "action_features": {0: "batch", 1: "actions"},
            "action_mask": {0: "batch", 1: "actions"},
            # Per-zone max-cards (axis 2) is intentionally FIXED — Phase 4
            # re-extractor pads every row to MAX_CARDS_PER_ZONE so the
            # graph never sees a variable per-zone width.
            "card_ids_by_zone": {0: "batch"},
            "action_card_idx": {0: "batch", 1: "actions"},
            "logits": {0: "batch", 1: "actions"},
            "value": {0: "batch"},
        }
    export_model: torch.nn.Module = model
    if config.uses_belief_features:
        class _BeliefExportWrapper(torch.nn.Module):
            def __init__(self, wrapped: CandidatePolicyNet) -> None:
                super().__init__()
                self.wrapped = wrapped

            def forward(self, state_x, action_x, mask_x, zone_x, action_idx_x, belief_x):  # noqa: ANN001
                return self.wrapped(
                    state_x,
                    action_x,
                    mask_x,
                    card_ids_by_zone=zone_x,
                    action_card_idx=action_idx_x,
                    belief_features=belief_x,
                )

        class _SlotBeliefExportWrapper(torch.nn.Module):
            def __init__(self, wrapped: CandidatePolicyNet) -> None:
                super().__init__()
                self.wrapped = wrapped

            def forward(self, state_x, action_x, mask_x, zone_x, action_idx_x, slot_ids_x, slot_feat_x, belief_x):  # noqa: ANN001
                return self.wrapped(
                    state_x,
                    action_x,
                    mask_x,
                    card_ids_by_zone=zone_x,
                    action_card_idx=action_idx_x,
                    uma_slot_card_ids=slot_ids_x,
                    uma_slot_features=slot_feat_x,
                    belief_features=belief_x,
                )

        export_model = (
            _SlotBeliefExportWrapper(model)
            if config.uses_uma_slot_tokens
            else _BeliefExportWrapper(model)
        )
        export_model.eval()
        belief_features = torch.zeros((1, config.belief_feature_dim), dtype=torch.float32)
        positional_inputs = positional_inputs + (belief_features,)
        input_names.append("belief_features")
        dynamic_axes["belief_features"] = {0: "batch"}
    torch.onnx.export(
        export_model,
        # Positional args mirror the model's `forward` signature; the two
        # new int inputs must be passed as positional tensors (not kwargs)
        # to participate in the traced graph. Phase 2's optional-kwarg
        # design keeps every other caller (PPO/DPO/value-retrain) working
        # unchanged because they still pass only the three originals.
        positional_inputs,
        out,
        input_names=input_names,
        output_names=["logits", "value"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
    )
    sidecar_payload: dict[str, object] = {
        "state_dim": graph_state_dim,
        # R16-P2 C5: v3.2 graphs reuse state_dim=110 but stamp schema 3.2
        # so downstream consumers can distinguish v3.0 from v3.2 by
        # sidecar (in addition to the ONNX input-set). The
        # `schema_version_for_state_dim(110)` table returns 3.0 by default
        # — override to 3.2 only when the slot-token branch is active.
        # v3.4-compound: state_dim=167 + slot tokens both → tag 3.4.
        # v3.2: state_dim=110 + slot tokens → tag 3.2 (preserves legacy).
        # Otherwise: schema_version_for_state_dim handles 96/110/164/167.
        "state_feature_schema_version": (
            STATE_FEATURE_SCHEMA_VERSION_V3_4
            if config.uses_uma_slot_tokens and graph_state_dim == STATE_DIM_V3_3
            else STATE_FEATURE_SCHEMA_VERSION_V3_2
            if config.uses_uma_slot_tokens
            else schema_version_for_state_dim(graph_state_dim)
        ),
        "action_dim": ACTION_DIM,
        # v33-correctness-fix Fix 2-4: stamp the action-feature schema
        # version so the Rust inference loader (and serve_onnx) can
        # fail-fast on train/inference layout mismatch. Bumped 2 → 3
        # when slot 10/26/28 semantics changed.
        "action_feature_schema_version": ACTION_FEATURE_SCHEMA_VERSION,
        "card_vocab": expected_vocab,
        "checkpoint_vocab": checkpoint_vocab,
        # R7.b.2 Phase 3: surface the embedding graph shape so
        # downstream consumers (serve_onnx) can shape-validate.
        "num_zones": NUM_ZONES,
        "max_cards_per_zone": MAX_CARDS_PER_ZONE,
        "card_vocab_table_size": CARD_VOCAB_TABLE_SIZE,
    }
    if config.uses_uma_slot_tokens:
        # R16-P2 C5: only emit these fields under v3.2 so v3.0/v3.1
        # sidecars stay byte-identical to pre-C5 (no spurious key churn
        # in existing checkpoints' meta.json under re-export).
        sidecar_payload["uses_uma_slot_tokens"] = True
        sidecar_payload["uma_slot_feature_dim"] = UMA_SLOT_FEATURE_DIM
        sidecar_payload["uma_slot_count"] = UMA_SLOT_COUNT
    # R7.b.3 set-attention probe: surface the model_variant in the sidecar
    # so downstream consumers (e.g. analysis tooling) can distinguish a
    # 7-input v3.2 graph trained with the MLP sum-pool trunk from one
    # trained with the attention trunk. ONNX-graph-wise the two are
    # identical (same 7-input set, same shapes), so the meta.json is the
    # only place this distinction lives outside the .pt checkpoint.
    # Only emit the key for non-default values so existing v3.0/v3.1/v3.2
    # sidecars stay byte-identical (no spurious key churn under re-export).
    if config.model_variant != "mlp":
        sidecar_payload["model_variant"] = config.model_variant
    if config.value_adapter != "none":
        sidecar_payload["value_adapter"] = config.value_adapter
    if config.uses_q_value_head:
        sidecar_payload["uses_q_value_head"] = True
        sidecar_payload["q_value_scalar"] = config.q_value_scalar
        sidecar_payload["q_value_scalar_scale"] = config.q_value_scalar_scale
        sidecar_payload["q_value_scalar_bias"] = config.q_value_scalar_bias
    if config.uses_belief_features:
        sidecar_payload["uses_belief_features"] = True
        sidecar_payload["belief_feature_dim"] = config.belief_feature_dim
        sidecar_payload["belief_schema_version"] = 1
        sidecar_payload["selfplay_kind"] = "rebel-selfplay"
    sidecar = out.with_suffix(out.suffix + ".meta.json")
    sidecar.write_text(
        json.dumps(sidecar_payload, indent=2) + "\n",
        encoding="utf8",
    )
    print(f"Exported {out} (vocab hash={expected_vocab.get('hash')})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export candidate policy checkpoint to ONNX.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-actions", type=int, default=16)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--q-value-scalar", choices=["max", "mean", "top2_mean", "top3_mean", "policy_mean"], default=None,
                        help="Override scalarization for Q-head checkpoints at export time.")
    parser.add_argument("--q-value-scalar-scale", type=float, default=1.0,
                        help="Affine scale for Q-head scalar value, clamped to [-1, 1].")
    parser.add_argument("--q-value-scalar-bias", type=float, default=0.0,
                        help="Affine bias for Q-head scalar value, clamped to [-1, 1].")
    return parser.parse_args()


if __name__ == "__main__":
    main()
