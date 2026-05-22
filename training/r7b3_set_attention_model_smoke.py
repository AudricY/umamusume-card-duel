"""R7.b.3 set-attention model-branch smoke.

Self-contained model invariant check for the set-attention probe. This is not
a strength test; it verifies the branch's load-bearing schema and warm-start
contracts before Slice 2 training spends wall time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uma_ai.features import (  # noqa: E402
    ACTION_DIM,
    CARD_ID_SHAPES,
    STATE_DIM,
    UMA_SLOT_COUNT,
    UMA_SLOT_FEATURE_DIM,
    ZONE_ORDER,
)
from uma_ai.model import (  # noqa: E402
    CARD_VOCAB_TABLE_SIZE,
    CandidatePolicyNet,
    ModelConfig,
    SET_ATTN_D_MODEL,
)


BATCH = 2
NUM_ACTIONS = 5
DEPTH = 2
MAX_CARDS_PER_ZONE = max(CARD_ID_SHAPES.values())


def fail(message: str) -> None:
    print(f"[r7b3-set-attention-model-smoke] FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def make_inputs(seed: int = 0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    action_mask = torch.ones(BATCH, NUM_ACTIONS, dtype=torch.bool)
    action_mask[:, -1] = False
    return {
        "state_features": torch.randn(BATCH, STATE_DIM, generator=gen),
        "action_features": torch.randn(BATCH, NUM_ACTIONS, ACTION_DIM, generator=gen),
        "action_mask": action_mask,
        "card_ids_by_zone": torch.randint(
            0,
            CARD_VOCAB_TABLE_SIZE,
            (BATCH, len(ZONE_ORDER), MAX_CARDS_PER_ZONE),
            dtype=torch.int64,
            generator=gen,
        ),
        "action_card_idx": torch.randint(
            0,
            CARD_VOCAB_TABLE_SIZE,
            (BATCH, NUM_ACTIONS, 2),
            dtype=torch.int64,
            generator=gen,
        ),
        "uma_slot_card_ids": torch.randint(
            1,
            CARD_VOCAB_TABLE_SIZE,
            (BATCH, UMA_SLOT_COUNT),
            dtype=torch.int64,
            generator=gen,
        ),
        "uma_slot_features": torch.randn(
            BATCH, UMA_SLOT_COUNT, UMA_SLOT_FEATURE_DIM, generator=gen,
        ),
    }


def forward(model: CandidatePolicyNet, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        return model(**inputs)


def build_base_and_treatment(seed: int = 1234) -> tuple[CandidatePolicyNet, CandidatePolicyNet]:
    base_cfg = ModelConfig(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=SET_ATTN_D_MODEL,
        depth=DEPTH,
        dropout=0.0,
        uses_uma_slot_tokens=True,
        model_variant="mlp",
    )
    treatment_cfg = ModelConfig(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=SET_ATTN_D_MODEL,
        depth=DEPTH,
        dropout=0.0,
        uses_uma_slot_tokens=True,
        model_variant="set_attention",
    )

    torch.manual_seed(seed)
    base = CandidatePolicyNet(base_cfg).eval()
    torch.manual_seed(seed)
    treatment = CandidatePolicyNet(treatment_cfg).eval()

    base_state = base.state_dict()
    treatment_state = treatment.state_dict()
    for key, value in base_state.items():
        if key not in treatment_state:
            fail(f"base key {key!r} missing from treatment state_dict")
        treatment_state[key] = value
    treatment.load_state_dict(treatment_state, strict=True)
    return base, treatment


def smoke_requires_slot_tokens() -> None:
    try:
        CandidatePolicyNet(
            ModelConfig(
                hidden_dim=SET_ATTN_D_MODEL,
                uses_uma_slot_tokens=False,
                model_variant="set_attention",
            )
        )
    except ValueError as exc:
        if "uses_uma_slot_tokens=True" not in str(exc):
            fail(f"unexpected set_attention/slot-token error: {exc}")
        print("  PASS  smoke (i): set_attention requires v3.2 slot-token tensors")
        return
    fail("set_attention model constructed without uses_uma_slot_tokens=True")


def smoke_zero_init_parity() -> None:
    base, treatment = build_base_and_treatment()
    inputs = make_inputs()
    base_logits, base_value = forward(base, inputs)
    treatment_logits, treatment_value = forward(treatment, inputs)
    logits_diff = float((base_logits - treatment_logits).abs().max().item())
    value_diff = float((base_value - treatment_value).abs().max().item())
    if logits_diff > 1e-6 or value_diff > 1e-6:
        fail(
            f"zero-init parity violated: logits_diff={logits_diff:.3e} "
            f"value_diff={value_diff:.3e}"
        )
    print(
        "  PASS  smoke (ii): zero-init attention residual preserves "
        "v3.2 MLP outputs"
    )


def smoke_attention_path_can_move_outputs() -> None:
    base, treatment = build_base_and_treatment()
    inputs = make_inputs()
    with torch.no_grad():
        treatment.set_attention_encoder.output_proj.weight.fill_(0.01)  # type: ignore[union-attr]
    base_logits, base_value = forward(base, inputs)
    treatment_logits, treatment_value = forward(treatment, inputs)
    logits_diff = float((base_logits - treatment_logits).abs().max().item())
    value_diff = float((base_value - treatment_value).abs().max().item())
    if logits_diff <= 1e-6 and value_diff <= 1e-6:
        fail("non-zero attention output projection did not move logits or value")
    print(
        "  PASS  smoke (iii): non-zero attention projection can affect outputs "
        f"(logits_diff={logits_diff:.3e}, value_diff={value_diff:.3e})"
    )


def main() -> int:
    print("[r7b3-set-attention-model-smoke] starting")
    smoke_requires_slot_tokens()
    smoke_zero_init_parity()
    smoke_attention_path_can_move_outputs()
    print("[r7b3-set-attention-model-smoke] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
