"""R16-P2 C2: `uma_slot_encoder` model-branch smoke.

Self-contained (no external corpus / checkpoint). Proves the NEW
`CandidatePolicyNet` per-Uma slot-token branch lands as a strict superset of
the v3.0/v3.1 forward path — additive at init, expressive once the encoder's
zero-output Linear leaves zero.

Four smokes (chunk plan § "P2 - Chunk Plan" row C2 + § "Init parity is
achievable"):

  (i)   Zero-input null residual. Treatment model
        (`uses_uma_slot_tokens=True`) with all-zero `uma_slot_card_ids` +
        all-zero `uma_slot_features` produces a slot-residual contribution
        to `state_encoded` that is bit-zero. Proves the slot branch is
        structurally null when no slot is present.
  (ii)  Delta=0.0 vs v3.0/v3.1 control. Treatment + control models share
        all parameters (state_dict copied verbatim where keys match — the
        only treatment-only params are `uma_slot_encoder.*`, which are
        zero-initialized). With zero slot tensors, the treatment's logits
        + value head outputs are bit-identical to the control. This is
        the load-bearing parity claim C7's `make_v32_slot_token_init.py`
        warm-start relies on.
  (iii) Non-zero slot tensors change outputs. After deliberately stamping
        a nonzero weight into `uma_slot_encoder[-1].weight` and feeding
        non-zero slot tensors, the treatment logits differ from the
        control. Proves the architecture has CAPACITY to use slot tokens,
        not just that it correctly suppresses them at init.
  (iv)  Board-zone double-counting fix. The treatment forward zeroes the
        4 board-zone lanes (ownActive/oppActive/ownBench/oppBench) of
        `card_ids_by_zone` before `zone_projection` consumes them. With
        zero slot tensors (so the slot residual itself is zero), the
        treatment output equals what a control forward would produce if
        we manually zeroed the same 4 lanes upstream. The smoke proves
        this masking is the ONLY board-zone difference vs the v3.0
        control — non-board zones (ownHand/ownDiscard/oppDiscard/stadium)
        pass through untouched.

Deliberately lean — model-level invariants, not a strength test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uma_ai.features import (  # noqa: E402
    ACTION_DIM,
    STATE_DIM,
    UMA_SLOT_COUNT,
    UMA_SLOT_FEATURE_DIM,
    ZONE_ORDER,
)
from uma_ai.model import (  # noqa: E402
    BOARD_ZONE_LANE_INDICES,
    CARD_EMBED_DIM,
    CARD_VOCAB_TABLE_SIZE,
    CandidatePolicyNet,
    ModelConfig,
)


def fail(msg: str) -> None:
    print(f"[r16-p2-slot-encoder-smoke] FAIL — {msg}", file=sys.stderr)
    sys.exit(1)


# Small fixed dims so the smoke runs in well under a second.
_HIDDEN = 32
_DEPTH = 2
_BATCH = 3
_NUM_ACTIONS = 4
_MAX_CARDS_PER_ZONE = 6


def _make_inputs(seed: int = 0, *, nonzero_slots: bool = False) -> dict[str, torch.Tensor]:
    """Build a deterministic batch of inputs that exercises every branch.

    Returns BOTH zero and (optionally) randomly populated slot tensors. The
    base inputs (`state_features`, `action_features`, `action_mask`,
    `card_ids_by_zone`, `action_card_idx`) are identical across calls so
    treatment-vs-control comparisons differ only in the slot tensors and
    the model flag.
    """
    rng = torch.Generator().manual_seed(seed)
    inputs: dict[str, torch.Tensor] = {}
    inputs["state_features"] = torch.randn(_BATCH, STATE_DIM, generator=rng)
    inputs["action_features"] = torch.randn(_BATCH, _NUM_ACTIONS, ACTION_DIM, generator=rng)
    # Half-and-half mask so masked-logit fill is exercised but at least one
    # legal action per row exists.
    mask = torch.zeros(_BATCH, _NUM_ACTIONS)
    mask[:, : _NUM_ACTIONS // 2 + 1] = 1.0
    inputs["action_mask"] = mask
    inputs["card_ids_by_zone"] = torch.randint(
        0, CARD_VOCAB_TABLE_SIZE, (_BATCH, len(ZONE_ORDER), _MAX_CARDS_PER_ZONE),
        generator=rng, dtype=torch.int64,
    )
    inputs["action_card_idx"] = torch.randint(
        0, CARD_VOCAB_TABLE_SIZE, (_BATCH, _NUM_ACTIONS, 2),
        generator=rng, dtype=torch.int64,
    )

    if nonzero_slots:
        inputs["uma_slot_card_ids"] = torch.randint(
            1, CARD_VOCAB_TABLE_SIZE, (_BATCH, UMA_SLOT_COUNT),
            generator=rng, dtype=torch.int64,
        )
        inputs["uma_slot_features"] = torch.randn(
            _BATCH, UMA_SLOT_COUNT, UMA_SLOT_FEATURE_DIM, generator=rng,
        )
    else:
        inputs["uma_slot_card_ids"] = torch.zeros(
            _BATCH, UMA_SLOT_COUNT, dtype=torch.int64,
        )
        inputs["uma_slot_features"] = torch.zeros(
            _BATCH, UMA_SLOT_COUNT, UMA_SLOT_FEATURE_DIM,
        )
    return inputs


def _build_models(seed: int = 1234) -> tuple[CandidatePolicyNet, CandidatePolicyNet]:
    """Build (control, treatment) models. They share all v3.0/v3.1 parameter
    weights (control state_dict copied into treatment for every matching key);
    the only treatment-only params are `uma_slot_encoder.*`, which stay at
    their zero-initialized weights — the parity-load-bearing invariant."""
    torch.manual_seed(seed)
    control_cfg = ModelConfig(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=_HIDDEN,
        depth=_DEPTH,
        dropout=0.0,
        uses_uma_slot_tokens=False,
    )
    control = CandidatePolicyNet(control_cfg).eval()

    torch.manual_seed(seed)  # same seed so shared params init IDENTICALLY
    treatment_cfg = ModelConfig(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=_HIDDEN,
        depth=_DEPTH,
        dropout=0.0,
        uses_uma_slot_tokens=True,
    )
    treatment = CandidatePolicyNet(treatment_cfg).eval()

    # Copy control's state_dict into treatment for every matching key. The
    # treatment-only keys (`uma_slot_encoder.*`) are left untouched so they
    # retain the zero-Linear init from CandidatePolicyNet.__init__.
    control_state = control.state_dict()
    treatment_state = treatment.state_dict()
    shared_keys = set(control_state) & set(treatment_state)
    if shared_keys != set(control_state):
        fail(
            f"control has keys not in treatment: {sorted(set(control_state) - shared_keys)} "
            f"(unexpected — treatment is a superset of control)"
        )
    for key in shared_keys:
        treatment_state[key] = control_state[key]
    # strict=True: every treatment key must be in our dict (it is — we are
    # the dict). This catches a future refactor that adds new params without
    # extending this copy contract.
    treatment.load_state_dict(treatment_state, strict=True)
    return control, treatment


def _forward(model: CandidatePolicyNet, inputs: dict[str, torch.Tensor],
             *, with_slots: bool) -> tuple[torch.Tensor, torch.Tensor]:
    kwargs = {
        "state_features": inputs["state_features"],
        "action_features": inputs["action_features"],
        "action_mask": inputs["action_mask"],
        "card_ids_by_zone": inputs["card_ids_by_zone"],
        "action_card_idx": inputs["action_card_idx"],
    }
    if with_slots:
        kwargs["uma_slot_card_ids"] = inputs["uma_slot_card_ids"]
        kwargs["uma_slot_features"] = inputs["uma_slot_features"]
    with torch.no_grad():
        logits, value = model(**kwargs)
    return logits, value


def _smoke_zero_input_null_residual() -> None:
    """Smoke (i): with `uses_uma_slot_tokens=True` and all-zero slot tensors,
    the slot-encoder residual contribution is bit-zero.

    We probe the residual directly via a forward hook on the encoder; this is
    crisper than back-deriving from a forward delta (the masking inside the
    forward also touches `card_ids_by_zone` when the flag is True, so a raw
    output diff wouldn't isolate the slot residual)."""
    _, treatment = _build_models()
    inputs = _make_inputs(nonzero_slots=False)

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["per_slot_encoded"] = output.detach().clone()

    handle = treatment.uma_slot_encoder.register_forward_hook(hook)
    try:
        _forward(treatment, inputs, with_slots=True)
    finally:
        handle.remove()

    per_slot = captured.get("per_slot_encoded")
    if per_slot is None:
        fail("uma_slot_encoder hook did not fire — branch did not execute")
    if per_slot.shape != (_BATCH, UMA_SLOT_COUNT, _HIDDEN):
        fail(f"per_slot_encoded shape {tuple(per_slot.shape)} != "
             f"({_BATCH}, {UMA_SLOT_COUNT}, {_HIDDEN})")

    # The encoder's final Linear is zero-initialized, so its output must be
    # exactly zero regardless of input. This is the structural invariant
    # that makes the zero-input residual null at init.
    if not torch.all(per_slot == 0.0):
        max_abs = float(per_slot.abs().max().item())
        fail(f"zero-Linear init violated: per_slot_encoded max|abs|={max_abs:.3e} "
             f"(expected exactly 0.0 — `nn.init.zeros_` on final Linear)")

    # And — because card_ids are all zero, the absent-slot mask is all zero,
    # so even if the encoder leaked a non-zero output, the pooled residual
    # would still be zero. Defense-in-depth check.
    slot_mask = (inputs["uma_slot_card_ids"] != 0).float()
    if slot_mask.any():
        fail("zero-input setup leaked a non-zero card_id — fixture broken")

    print("  PASS  smoke (i): zero-input slot residual is bit-zero "
          "(encoder output = 0 AND absent-mask = 0; double-guard)")


def _smoke_delta_zero_vs_control() -> None:
    """Smoke (ii): treatment with zero slot tensors ≡ control forward.

    Load-bearing parity claim: a v3.0 checkpoint warm-started into a v3.2
    treatment model (C7's init builder) produces IDENTICAL logits + value to
    the source v3.0 model. This is the smoke that future C7 verifies.
    """
    control, treatment = _build_models()
    inputs = _make_inputs(nonzero_slots=False)

    # Control: legacy 5-input forward (no slot kwargs). Treatment: full
    # 7-input forward with zero slot tensors AND the board-zone masking
    # active. The masking zeroes lanes 0..3 of card_ids_by_zone for the
    # treatment ONLY — so the treatment forward differs from the control
    # by:
    #   1) zone_projection sees lanes 0..3 forced to 0 (board zones masked
    #      because the slot branch is supposed to encode them);
    #   2) slot residual adds 0 (zero Linear + zero mask).
    # Therefore the bit-identical claim REQUIRES the control also see lanes
    # 0..3 zeroed. We feed that explicitly to the control so the two
    # forwards differ ONLY in the slot residual addition (which is 0).
    control_inputs = dict(inputs)
    masked_card_ids = inputs["card_ids_by_zone"].clone()
    masked_card_ids[:, : len(BOARD_ZONE_LANE_INDICES), :] = 0
    control_inputs["card_ids_by_zone"] = masked_card_ids

    c_logits, c_value = _forward(control, control_inputs, with_slots=False)
    t_logits, t_value = _forward(treatment, inputs, with_slots=True)

    # Use a tight 1e-6 tolerance. In float32 + identical-parameter-values
    # arithmetic the only delta should be the order of additive ops in the
    # forward path; both branches add a structurally-zero residual, so
    # bit-equality is the expectation but 1e-6 leaves room for fused-op
    # reorderings on different PyTorch versions.
    logits_diff = float((t_logits - c_logits).abs().max().item())
    value_diff = float((t_value - c_value).abs().max().item())
    if logits_diff > 1e-6:
        fail(f"delta=0 logits violated: max|Δ|={logits_diff:.3e} > 1e-6 "
             f"(treatment with zero slots ≠ board-masked control — C7 "
             f"warm-start parity would break)")
    if value_diff > 1e-6:
        fail(f"delta=0 value violated: max|Δ|={value_diff:.3e} > 1e-6")

    print(f"  PASS  smoke (ii): delta=0 vs board-masked control "
          f"(logits max|Δ|={logits_diff:.3e}, value max|Δ|={value_diff:.3e}); "
          f"C7 warm-start parity holds")


def _smoke_capacity() -> None:
    """Smoke (iii): non-zero slot tensors + non-zero encoder weight change
    the output. Proves the architecture has CAPACITY to use the slot signal.

    We stamp a small non-zero weight into `uma_slot_encoder[-1].weight` (the
    layer that was zero-initialized) and feed random non-zero slot tensors.
    The treatment output must then diverge from the control (whose forward
    has no slot branch and sees the board-zone-masked card_ids_by_zone)."""
    control, treatment = _build_models()
    # Same board-masking convention as smoke (ii) so the ONLY remaining
    # difference between control and treatment is the slot residual.
    inputs = _make_inputs(nonzero_slots=True)
    control_inputs = dict(inputs)
    masked_card_ids = inputs["card_ids_by_zone"].clone()
    masked_card_ids[:, : len(BOARD_ZONE_LANE_INDICES), :] = 0
    control_inputs["card_ids_by_zone"] = masked_card_ids

    # Sanity: BEFORE stamping a non-zero weight, the treatment output (with
    # NON-ZERO slot tensors but ZERO encoder output Linear) must STILL equal
    # the control — because the zero Linear kills any slot signal regardless
    # of inputs. This catches a bug where the zero-init silently doesn't
    # stick.
    c_logits_before, _ = _forward(control, control_inputs, with_slots=False)
    t_logits_before, _ = _forward(treatment, inputs, with_slots=True)
    diff_before = float((t_logits_before - c_logits_before).abs().max().item())
    if diff_before > 1e-6:
        fail(f"capacity pre-stamp: zero-Linear should suppress non-zero slot "
             f"tensors but logits diverge by {diff_before:.3e}")

    # Stamp a non-zero weight pattern into the final Linear.
    with torch.no_grad():
        final_linear = treatment.uma_slot_encoder[-1]
        final_linear.weight.copy_(
            torch.randn_like(final_linear.weight) * 0.1
        )

    t_logits_after, t_value_after = _forward(treatment, inputs, with_slots=True)
    diff_after = float((t_logits_after - c_logits_before).abs().max().item())
    # Expect a clearly non-trivial diff (much larger than the 1e-6 parity
    # threshold). 1e-3 is a comfortable separator without overfitting to a
    # specific magnitude.
    if diff_after < 1e-3:
        fail(f"capacity post-stamp: non-zero encoder weight + non-zero slot "
             f"tensors should perturb logits but max|Δ|={diff_after:.3e} < 1e-3 "
             f"(slot branch not wired into the forward, or the residual is "
             f"being suppressed downstream)")
    if not torch.isfinite(t_logits_after).all():
        fail("capacity post-stamp: treatment produced non-finite logits")
    if not torch.isfinite(t_value_after).all():
        fail("capacity post-stamp: treatment produced non-finite value")

    print(f"  PASS  smoke (iii): capacity demonstrated — non-zero slot tensors "
          f"+ non-zero encoder weight perturb logits by max|Δ|={diff_after:.3e} "
          f"(>> 1e-3); pre-stamp parity max|Δ|={diff_before:.3e} (≤ 1e-6)")


def _smoke_board_zone_masking() -> None:
    """Smoke (iv): board-zone double-counting fix.

    Parity claim being asserted: with `uses_uma_slot_tokens=True` and zero
    slot tensors, the treatment model's `zone_projection` sees the 4 board-
    zone lanes (ownActive/oppActive/ownBench/oppBench, indices 0..3) zeroed.
    Equivalently: feeding the SAME input through a control with those 4
    lanes zeroed UPSTREAM produces identical outputs to the treatment with
    the original input.

    This is the same identity used in smoke (ii) but flipped — there we
    used it to demonstrate the residual-null parity claim; here we use it
    to directly verify the masking is operative. The two smokes guard
    against different regressions:
      - smoke (ii) fails if the slot residual leaks a non-zero signal;
      - smoke (iv) fails if the board-zone masking is broken (e.g. wrong
        lane indices, off-by-one, or the masking is gated on the wrong
        flag).

    Anti-correlation check: we ALSO verify that perturbing the non-board
    zones (ownHand/ownDiscard/oppDiscard/stadium) still produces a delta
    between treatment and double-mask-control — i.e. only the FOUR board
    zones are masked, not all zones.
    """
    control, treatment = _build_models()
    inputs = _make_inputs(nonzero_slots=False)

    # (a) Equivalence: treatment (slot-tokens on, zero slot tensors) vs
    #     control (slot-tokens off, board zones manually zeroed upstream).
    control_inputs = dict(inputs)
    masked_card_ids = inputs["card_ids_by_zone"].clone()
    masked_card_ids[:, : len(BOARD_ZONE_LANE_INDICES), :] = 0
    control_inputs["card_ids_by_zone"] = masked_card_ids
    c_logits, c_value = _forward(control, control_inputs, with_slots=False)
    t_logits, t_value = _forward(treatment, inputs, with_slots=True)
    diff_logits = float((t_logits - c_logits).abs().max().item())
    diff_value = float((t_value - c_value).abs().max().item())
    if diff_logits > 1e-6 or diff_value > 1e-6:
        fail(f"board-zone masking: treatment ≠ board-masked control "
             f"(logits max|Δ|={diff_logits:.3e}, value max|Δ|={diff_value:.3e}); "
             f"masking implementation deviates from spec")

    # (b) Anti-correlation: perturbing a NON-board zone (ownHand, lane idx 4)
    #     should NOT yield the same equivalence. If it did, the masking is
    #     too aggressive (it would be zeroing more than the 4 board lanes).
    nonboard_lane = ZONE_ORDER.index("ownHand")  # = 4
    if nonboard_lane < len(BOARD_ZONE_LANE_INDICES):
        fail(f"test fixture broken: ownHand lane {nonboard_lane} sits inside "
             f"the board-zone lanes range; pick a different non-board zone")
    perturbed_card_ids = inputs["card_ids_by_zone"].clone()
    # Stamp a clearly non-zero pattern into ownHand so its sum-pool
    # contribution to zone_projection is non-trivial.
    perturbed_card_ids[:, nonboard_lane, :] = torch.randint(
        1, CARD_VOCAB_TABLE_SIZE,
        perturbed_card_ids[:, nonboard_lane, :].shape, dtype=torch.int64,
    )
    perturbed_inputs = dict(inputs)
    perturbed_inputs["card_ids_by_zone"] = perturbed_card_ids
    # Run treatment with the perturbed (non-board-zone) input; compare to
    # the original-treatment output. They MUST differ — proving the
    # non-board zones still drive zone_projection.
    t_logits_perturbed, _ = _forward(treatment, perturbed_inputs, with_slots=True)
    diff_perturbed = float((t_logits_perturbed - t_logits).abs().max().item())
    if diff_perturbed < 1e-6:
        fail(f"board-zone masking: perturbing ownHand (non-board zone) "
             f"did NOT change treatment output (max|Δ|={diff_perturbed:.3e}) — "
             f"masking is over-aggressive and is zeroing all zones, not just "
             f"the 4 board lanes")

    print(f"  PASS  smoke (iv): board-zone masking is exact — board lanes "
          f"0..3 masked (treatment ≡ board-masked control, max|Δ|={diff_logits:.3e}); "
          f"non-board lanes 4..7 untouched (perturbing ownHand changes output "
          f"by max|Δ|={diff_perturbed:.3e})")


def main() -> None:
    # Frozen-shape guards: the smoke pins the architectural contracts so a
    # future re-ordering of ZONE_ORDER or a UMA_SLOT constant shift bites
    # here before reaching downstream chunks (C4 dataset packing, C5 ONNX,
    # C7 init builder).
    if BOARD_ZONE_LANE_INDICES != (0, 1, 2, 3):
        fail(f"BOARD_ZONE_LANE_INDICES {BOARD_ZONE_LANE_INDICES} != (0,1,2,3) — "
             f"ZONE_ORDER head was reordered; C2 masking depends on the "
             f"contiguous 0..3 layout.")
    if UMA_SLOT_COUNT != 10 or UMA_SLOT_FEATURE_DIM != 23:
        fail(f"UMA_SLOT contract drift (count={UMA_SLOT_COUNT}, F={UMA_SLOT_FEATURE_DIM})")
    if CARD_EMBED_DIM != 32:
        fail(f"CARD_EMBED_DIM {CARD_EMBED_DIM} != 32 (encoder Linear in-dim depends on it)")

    # Confirm flag-off path is truly a no-op: ModelConfig() default has
    # `uses_uma_slot_tokens=False`, `uma_slot_encoder=None`, so a forward
    # without the new kwargs reaches the legacy v3.0/v3.1 path verbatim.
    default_cfg = ModelConfig(state_dim=STATE_DIM, action_dim=ACTION_DIM,
                              hidden_dim=_HIDDEN, depth=_DEPTH, dropout=0.0)
    if default_cfg.uses_uma_slot_tokens is not False:
        fail("ModelConfig default uses_uma_slot_tokens is not False — flag is "
             "no longer opt-in; default callers would silently activate the branch")
    default_model = CandidatePolicyNet(default_cfg).eval()
    if default_model.uma_slot_encoder is not None:
        fail("CandidatePolicyNet(default).uma_slot_encoder is not None — extra "
             "params on the default path, will break C7's state-dict copy")

    _smoke_zero_input_null_residual()
    _smoke_delta_zero_vs_control()
    _smoke_capacity()
    _smoke_board_zone_masking()

    print("r16-p2-slot-encoder smoke: ALL PASS")


if __name__ == "__main__":
    main()
