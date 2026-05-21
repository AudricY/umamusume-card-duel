"""R16-P2 C7: parity smoke for `make_v32_slot_token_init.py`.

Self-contained (no external corpus / checkpoint). Synthesizes a tiny v3.0
source checkpoint in a temp dir, runs the C7 builder against it, then
verifies the produced v3.2 init satisfies the load-bearing contracts.

Four smokes (chunk plan § "P2 - Chunk Plan" row C7 + § "Init parity is
achievable"):

  (i)   Shape regression. The produced checkpoint has
        `model_config.uses_uma_slot_tokens=True`, `feature_schema`
        re-stamped to "3.2", state_dict equals source's keys plus exactly
        the 3 new slot-encoder keys, `uma_slot_encoder.2.weight` is all
        zeros, and every other parameter matches the source byte-for-byte.

  (ii)  Delta=0.0 vs source. Load the source as a v3.0 model and the
        produced ckpt as a v3.2 model. Forward the SAME synthetic
        observation through both (with zero slot tensors fed to v3.2 and
        the 4 board-zone lanes pre-zeroed in the shared input so the
        treatment's masking is a no-op). Assert
        `max|t_logits - s_logits| <= 1e-6` and same for value. This is
        the parity contract the chunk plan calls out — at init, v3.2
        produces bit-identical outputs to its v3.0 source.

  (iii) Capacity. Stamp a non-zero pattern into the produced model's
        `uma_slot_encoder[-1].weight` and feed non-zero `uma_slot_*`
        tensors. Assert the logits diverge from the source by >> 1e-6
        (>= 1e-3). Proves the slot branch can carry signal once trained.

  (iv)  Refuse v3.2 source. Synthesize a checkpoint whose `model_config`
        already has `uses_uma_slot_tokens=True` and confirm the builder
        raises a `SystemExit` with the "refuse to operate on v3.2 source"
        message. Guards against accidentally re-initing a trained
        slot-encoder by re-running the builder on its own output.

Deliberately lean — file-level invariants and a single forward pass per
smoke, not a strength test.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uma_ai.features import (  # noqa: E402
    ACTION_DIM,
    STATE_DIM,
    STATE_DIM_V3,
    STATE_FEATURE_SCHEMA_VERSION_V3_2,
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

# Re-export the builder's own contract constants so any drift between
# builder and smoke is a build error, not a runtime divergence.
from make_v32_slot_token_init import (  # noqa: E402
    NEW_SLOT_ENCODER_KEYS,
    ZERO_INIT_KEY,
    build as build_v32_init,
)

# Small fixed dims so the smoke runs in well under a second. Real-shape
# enough to exercise the broadcast/contiguous-stride behavior in the
# `uma_slot_encoder` forward, while still cheap.
_HIDDEN = 64
_DEPTH = 2
_BATCH = 3
_NUM_ACTIONS = 4
_MAX_CARDS_PER_ZONE = 6


def fail(msg: str) -> None:
    print(f"[r16-p2-slot-init-smoke] FAIL — {msg}", file=sys.stderr)
    sys.exit(1)


def _make_v30_payload(*, seed: int = 7) -> dict:
    """Build a synthetic v3.0 source checkpoint dict.

    Mirrors the payload shape `train_bc.py` writes (model_state +
    model_config + feature_schema) so the builder treats it as a real
    training checkpoint."""
    torch.manual_seed(seed)
    config = ModelConfig(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=_HIDDEN,
        depth=_DEPTH,
        dropout=0.0,
        uses_uma_slot_tokens=False,
    )
    model = CandidatePolicyNet(config).eval()
    return {
        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "model_config": config.to_dict(),
        "feature_schema": {
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "state_feature_schema_version": 3.0,
            "action_feature_schema_version": 0,
            "card_vocab": {},
        },
    }


def _make_v32_payload(*, seed: int = 7) -> dict:
    """Build a synthetic v3.2 source checkpoint dict — used by smoke (iv)."""
    torch.manual_seed(seed)
    config = ModelConfig(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=_HIDDEN,
        depth=_DEPTH,
        dropout=0.0,
        uses_uma_slot_tokens=True,
    )
    model = CandidatePolicyNet(config).eval()
    return {
        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "model_config": config.to_dict(),
        "feature_schema": {
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "state_feature_schema_version": STATE_FEATURE_SCHEMA_VERSION_V3_2,
            "action_feature_schema_version": 0,
            "card_vocab": {},
        },
    }


def _make_inputs(*, seed: int = 11, nonzero_slots: bool = False) -> dict[str, torch.Tensor]:
    """Synthetic batch for forward-parity comparison. The board-zone lanes
    (0..3) of `card_ids_by_zone` are pre-zeroed so the v3.2 treatment's
    board-masking (which zeroes those same lanes upstream of
    `zone_projection`) is a structural no-op — letting the v3.0 source's
    raw forward see the SAME zone-projection input as the v3.2 forward."""
    rng = torch.Generator().manual_seed(seed)
    inputs: dict[str, torch.Tensor] = {}
    inputs["state_features"] = torch.randn(_BATCH, STATE_DIM, generator=rng)
    inputs["action_features"] = torch.randn(_BATCH, _NUM_ACTIONS, ACTION_DIM, generator=rng)
    mask = torch.zeros(_BATCH, _NUM_ACTIONS)
    mask[:, : _NUM_ACTIONS // 2 + 1] = 1.0
    inputs["action_mask"] = mask
    card_ids = torch.randint(
        0, CARD_VOCAB_TABLE_SIZE,
        (_BATCH, len(ZONE_ORDER), _MAX_CARDS_PER_ZONE),
        generator=rng, dtype=torch.int64,
    )
    # Pre-zero the 4 board-zone lanes so the v3.2 forward's masking is a
    # no-op vs the v3.0 forward's unmasked input. This is the
    # "fair parity comparison" technique from the brief — equivalent to
    # zeroing the source's lanes downstream of `_make_inputs`.
    card_ids[:, : len(BOARD_ZONE_LANE_INDICES), :] = 0
    inputs["card_ids_by_zone"] = card_ids
    inputs["action_card_idx"] = torch.randint(
        0, CARD_VOCAB_TABLE_SIZE,
        (_BATCH, _NUM_ACTIONS, 2),
        generator=rng, dtype=torch.int64,
    )

    if nonzero_slots:
        inputs["uma_slot_card_ids"] = torch.randint(
            1, CARD_VOCAB_TABLE_SIZE,
            (_BATCH, UMA_SLOT_COUNT),
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


def _load_target_model(ckpt_path: Path) -> tuple[CandidatePolicyNet, dict]:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ModelConfig.from_dict(payload["model_config"])
    model = CandidatePolicyNet(config).eval()
    model.load_state_dict(payload["model_state"], strict=True)
    return model, payload


def _load_source_model(payload: dict) -> CandidatePolicyNet:
    config = ModelConfig.from_dict(payload["model_config"])
    model = CandidatePolicyNet(config).eval()
    model.load_state_dict(payload["model_state"], strict=True)
    return model


def _forward_source(model: CandidatePolicyNet, inputs: dict[str, torch.Tensor]):
    with torch.no_grad():
        return model(
            state_features=inputs["state_features"],
            action_features=inputs["action_features"],
            action_mask=inputs["action_mask"],
            card_ids_by_zone=inputs["card_ids_by_zone"],
            action_card_idx=inputs["action_card_idx"],
        )


def _forward_target(model: CandidatePolicyNet, inputs: dict[str, torch.Tensor]):
    with torch.no_grad():
        return model(
            state_features=inputs["state_features"],
            action_features=inputs["action_features"],
            action_mask=inputs["action_mask"],
            card_ids_by_zone=inputs["card_ids_by_zone"],
            action_card_idx=inputs["action_card_idx"],
            uma_slot_card_ids=inputs["uma_slot_card_ids"],
            uma_slot_features=inputs["uma_slot_features"],
        )


def _smoke_shape_regression(source_payload: dict, target_ckpt: Path) -> None:
    """Smoke (i): the produced checkpoint has the exact shape the C6
    implementer's report enumerated.

    Asserts:
      (a) `model_config.uses_uma_slot_tokens=True`
      (b) `feature_schema.state_feature_schema_version == 3.2`
      (c) state_dict has exactly source's keys + 3 new slot-encoder keys
      (d) `uma_slot_encoder.2.weight` is all zeros
      (e) every non-new parameter matches the source byte-for-byte
    """
    payload = torch.load(target_ckpt, map_location="cpu", weights_only=False)

    # (a)
    cfg = payload["model_config"]
    if not cfg.get("uses_uma_slot_tokens", False):
        fail(f"target model_config.uses_uma_slot_tokens != True (got {cfg!r})")

    # (b)
    fs = payload.get("feature_schema") or {}
    ver = fs.get("state_feature_schema_version")
    if ver != STATE_FEATURE_SCHEMA_VERSION_V3_2:
        fail(f"feature_schema.state_feature_schema_version={ver!r}, "
             f"expected {STATE_FEATURE_SCHEMA_VERSION_V3_2!r} (3.2)")

    # (c)
    target_state = payload["model_state"]
    source_state = source_payload["model_state"]
    target_keys = set(target_state.keys())
    source_keys = set(source_state.keys())
    expected_new = set(NEW_SLOT_ENCODER_KEYS)
    new_keys = target_keys - source_keys
    missing = source_keys - target_keys
    if missing:
        fail(f"target state_dict missing source keys: {sorted(missing)[:8]}...")
    if new_keys != expected_new:
        fail(f"new keys delta {sorted(new_keys)} != expected {sorted(expected_new)} "
             f"(extra={sorted(new_keys - expected_new)}, "
             f"missing={sorted(expected_new - new_keys)})")

    # (d)
    zero_w = target_state[ZERO_INIT_KEY]
    if not torch.all(zero_w == 0.0):
        max_abs = float(zero_w.abs().max().item())
        fail(f"{ZERO_INIT_KEY} is not all-zero (max|w|={max_abs:.3e})")

    # (e) bit-exact match on every source key.
    for key in source_keys:
        t = target_state[key]
        s = source_state[key]
        if t.shape != s.shape:
            fail(f"shape mismatch on `{key}`: target {tuple(t.shape)} vs "
                 f"source {tuple(s.shape)}")
        if not torch.equal(t, s):
            max_abs = float((t - s).abs().max().item())
            fail(f"non-bit-exact copy of `{key}`: max|t-s|={max_abs:.3e} "
                 f"(C7 contract requires verbatim copy of every v3.0 param)")

    # Bonus: dump the 3 new key shapes for the report.
    shapes = {k: tuple(target_state[k].shape) for k in NEW_SLOT_ENCODER_KEYS}
    print(f"  PASS  smoke (i): shape regression — "
          f"uses_uma_slot_tokens=True, schema=v3.2, "
          f"+{len(expected_new)} new keys, {ZERO_INIT_KEY}=zeros, "
          f"{len(source_keys)} v3.0 params copied bit-exact. "
          f"New key shapes (hidden={_HIDDEN}): {shapes}")


def _smoke_delta_zero_vs_source(source_payload: dict, target_ckpt: Path) -> None:
    """Smoke (ii): bit-exact parity between v3.0 source and v3.2 init."""
    source_model = _load_source_model(source_payload)
    target_model, _ = _load_target_model(target_ckpt)

    inputs = _make_inputs(nonzero_slots=False)
    s_logits, s_value = _forward_source(source_model, inputs)
    t_logits, t_value = _forward_target(target_model, inputs)

    logits_diff = float((t_logits - s_logits).abs().max().item())
    value_diff = float((t_value - s_value).abs().max().item())
    # Per C2's precedent (`r16_p2_slot_encoder_smoke.py:273`), the expected
    # delta is exactly 0.0; we keep a 1e-6 ceiling to absorb any fused-op
    # reordering across PyTorch versions.
    if logits_diff > 1e-6:
        fail(f"delta=0 logits violated: max|Δ|={logits_diff:.3e} > 1e-6 "
             f"(v3.2 target ≠ v3.0 source on zero-slot input; C7 warm-start "
             f"parity broken)")
    if value_diff > 1e-6:
        fail(f"delta=0 value violated: max|Δ|={value_diff:.3e} > 1e-6")

    print(f"  PASS  smoke (ii): delta=0 parity vs v3.0 source "
          f"(logits max|Δ|={logits_diff:.3e}, value max|Δ|={value_diff:.3e}); "
          f"C7 warm-start parity holds")


def _smoke_capacity(source_payload: dict, target_ckpt: Path) -> None:
    """Smoke (iii): non-zero slot tensors + non-zero encoder weight perturb
    the v3.2 forward — proves the architecture has capacity to use the
    slot branch once trained."""
    source_model = _load_source_model(source_payload)
    target_model, _ = _load_target_model(target_ckpt)

    # Inputs share base tensors with smoke (ii) but carry non-zero slots.
    inputs = _make_inputs(nonzero_slots=True, seed=23)

    # Pre-stamp sanity: with zero encoder weight, even non-zero slot
    # tensors must produce a forward IDENTICAL to the v3.0 source (the
    # zero Linear kills any slot signal). Mirrors the
    # `r16_p2_slot_encoder_smoke.py` smoke (iii) pre-stamp check.
    s_logits_pre, _ = _forward_source(source_model, inputs)
    t_logits_pre, _ = _forward_target(target_model, inputs)
    diff_pre = float((t_logits_pre - s_logits_pre).abs().max().item())
    if diff_pre > 1e-6:
        fail(f"capacity pre-stamp: zero-Linear should suppress non-zero "
             f"slot tensors but max|Δ|={diff_pre:.3e}")

    # Stamp a non-zero pattern into the final Linear's weight.
    with torch.no_grad():
        final_linear = target_model.uma_slot_encoder[-1]
        final_linear.weight.copy_(torch.randn_like(final_linear.weight) * 0.1)

    t_logits_post, t_value_post = _forward_target(target_model, inputs)
    diff_post = float((t_logits_post - s_logits_pre).abs().max().item())
    if diff_post < 1e-3:
        fail(f"capacity post-stamp: non-zero encoder weight + non-zero "
             f"slot tensors should perturb logits but max|Δ|={diff_post:.3e} "
             f"< 1e-3 (slot branch not wired or signal suppressed)")
    if not torch.isfinite(t_logits_post).all():
        fail("capacity post-stamp: produced non-finite logits")
    if not torch.isfinite(t_value_post).all():
        fail("capacity post-stamp: produced non-finite value")

    print(f"  PASS  smoke (iii): capacity demonstrated — pre-stamp parity "
          f"max|Δ|={diff_pre:.3e} (≤ 1e-6); post-stamp logits perturb by "
          f"max|Δ|={diff_post:.3e} (>> 1e-3)")


def _smoke_refuse_v32_source(tmp_dir: Path) -> None:
    """Smoke (iv): the builder refuses a v3.2 source checkpoint.

    Guards against the user accidentally re-running the builder on its
    own output and silently re-initing the slot encoder."""
    v32_src = tmp_dir / "v32_source.pt"
    torch.save(_make_v32_payload(seed=31), v32_src)
    v32_dst = tmp_dir / "v32_from_v32.pt"

    raised = None
    try:
        build_v32_init(v32_src, v32_dst, seed=1234)
    except SystemExit as exc:
        raised = exc
    except BaseException as exc:  # noqa: BLE001 — capture wide for diagnostic
        fail(f"builder raised unexpected exception type {type(exc).__name__}: {exc}")
    if raised is None:
        fail("builder did NOT refuse v3.2 source — output silently produced")

    msg = str(raised)
    if "refuse to operate on v3.2 source" not in msg:
        fail(f"builder raised SystemExit but message does not mention "
             f"'refuse to operate on v3.2 source': {msg!r}")
    if v32_dst.exists():
        fail(f"builder refused v3.2 source but still wrote {v32_dst} — "
             f"a partial write is a bug (downstream could load it)")
    print(f"  PASS  smoke (iv): builder refuses v3.2 source with clear "
          f"message ({msg!r})")


def main() -> None:
    # Frozen-shape guards. C7's contract depends on the EXACT slot-encoder
    # key set (3 keys, one zero-inited) and the shapes the C6 report
    # enumerated. A future C2 refactor that adds a bias to the final
    # Linear (or changes the encoder depth) silently breaks C7 unless we
    # land an updated builder; fail loud here.
    if BOARD_ZONE_LANE_INDICES != (0, 1, 2, 3):
        fail(f"BOARD_ZONE_LANE_INDICES {BOARD_ZONE_LANE_INDICES} != (0,1,2,3) — "
             f"v3.2 masking depends on the contiguous 0..3 layout.")
    if UMA_SLOT_COUNT != 10 or UMA_SLOT_FEATURE_DIM != 23:
        fail(f"UMA_SLOT contract drift (count={UMA_SLOT_COUNT}, "
             f"F={UMA_SLOT_FEATURE_DIM})")
    if CARD_EMBED_DIM != 32:
        fail(f"CARD_EMBED_DIM {CARD_EMBED_DIM} != 32")
    if STATE_DIM != STATE_DIM_V3:
        fail(f"STATE_DIM ({STATE_DIM}) != STATE_DIM_V3 ({STATE_DIM_V3}) — "
             f"v3.2 keeps the 110-d state vector")

    with tempfile.TemporaryDirectory(prefix="r16-p2-slot-init-smoke-") as tmp:
        tmp_dir = Path(tmp)
        # Setup: synthesize a v3.0 source checkpoint and run the builder.
        source_payload = _make_v30_payload(seed=7)
        source_path = tmp_dir / "v30_source.pt"
        torch.save(source_payload, source_path)
        target_path = tmp_dir / "v32_init.pt"
        provenance = build_v32_init(source_path, target_path, seed=1234)
        if not target_path.exists():
            fail(f"builder did not produce output at {target_path}")
        # Confirm provenance carries the expected delta-0.0 claim — this
        # is just metadata, but tests catch a future drift where the
        # builder stops advertising parity (and an operator might be
        # surprised when their run regresses).
        if provenance.get("delta_logits_at_init") != 0.0:
            fail(f"provenance.delta_logits_at_init={provenance.get('delta_logits_at_init')!r}, "
                 f"expected 0.0")

        _smoke_shape_regression(source_payload, target_path)
        _smoke_delta_zero_vs_source(source_payload, target_path)
        _smoke_capacity(source_payload, target_path)
        _smoke_refuse_v32_source(tmp_dir)

    print("r16-p2-slot-init smoke: ALL PASS")


if __name__ == "__main__":
    main()
