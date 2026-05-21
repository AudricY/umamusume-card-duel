"""Focused smoke for the serve_onnx 96/110/164/v3.2 serving-schema guard.

r16 P1 prerequisite + r16 P2 C5 extension. Proves
`serve_onnx._resolve_feature_schema` resolves the feature builder STRICTLY
from the loaded ONNX graph's `(state_features last dim, has_uma_slot_inputs)`
pair:

  - a real 96-d v2 production graph  -> schema "v2"   (no embedding feeds)
  - a real 110-d v3.0 graph          -> schema "v3"   (embedding feeds)
  - a synthetic 164-d v3.1 graph     -> schema "v3.1" (embedding feeds)
                                        [R16-P1: now a REAL builder, no
                                        longer a fail-fast placeholder]
  - a synthetic 110-d v3.2 graph     -> schema "v3.2" (embedding + slot feeds)
                                        [R16-P2 C5: NEW; state_dim shared
                                        with v3.0, distinguished by
                                        input-set presence]
  - a synthetic 999-d graph          -> fail fast (unknown schema; SystemExit)

R16-P2 C5 cross-input fail-fast cases (the LANDMINE — silent mis-routing
between v3.0 and v3.2 would cause runtime crashes deep in ORT):

  - 96-d graph carrying `uma_slot_*` inputs  -> fail fast (v3.2 only at 110)
  - 164-d graph carrying `uma_slot_*` inputs -> fail fast (v3.2 only at 110)
  - 110-d graph declaring `uma_slot_card_ids` only (no features) -> fail fast
  - 110-d graph declaring `uma_slot_features` only (no card_ids) -> fail fast
  - explicit `--feature-schema v3.2` against a 110-d v3.0 graph -> fail fast
  - explicit `--feature-schema v3` against a 110-d v3.2 graph -> fail fast

R16-P2 C5 ONNX roundtrip: build a tiny C2 model with
`uses_uma_slot_tokens=True`, export via Part A's 7-input path, load via
serve_onnx, forward a populated batch, assert PyTorch vs ORT logits/value
agree within 1e-3 (mirrors the existing v3.0/v3.1 roundtrip discipline).

It also runs the production-transparency check: the resolved schema for the
pinned 96-d model is unchanged ("v2"), and `request_to_arrays` still emits
exactly the three frozen v2 feeds (no embedding inputs) for a v2 pin.

Deliberately lean: this is a shape-resolution + roundtrip check, not a
strength/matrix test. ORT is required (it is a serve_onnx dependency).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper

from serve_onnx import _resolve_feature_schema, request_to_arrays
from uma_ai.features import (
    STATE_DIM_V2,
    STATE_DIM_V3,
    STATE_DIM_V3_1,
    UMA_SLOT_COUNT,
    UMA_SLOT_FEATURE_DIM,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
V2_MODEL = REPO_ROOT / "runs" / "R13-W6-phase-d" / "iter-2" / "policy.onnx"
V3_MODEL = REPO_ROOT / "runs" / "R110-W6-repro" / "iter-0" / "policy.onnx"


def _session(path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


def _synthetic_session(
    state_dim: int,
    *,
    with_embedding: bool,
    with_uma_slot_ids: bool = False,
    with_uma_slot_features: bool = False,
) -> ort.InferenceSession:
    """Build a minimal valid ONNX graph with a state_features input of the
    given last dim (and optional card_ids_by_zone / uma_slot_* inputs) so the
    guard can be exercised on input-set combinations for which no real model
    exists.

    R16-P2 C5: `with_uma_slot_ids` and `with_uma_slot_features` are
    independent so the smoke can synthesize a PARTIAL v3.2 graph (only one
    of the two slot tensors) and verify the resolver fails fast on the
    contractual-pair violation."""
    state_in = helper.make_tensor_value_info(
        "state_features", TensorProto.FLOAT, [None, state_dim]
    )
    inputs = [state_in]
    flat = helper.make_node("Identity", ["state_features"], ["out"])
    nodes = [flat]
    if with_embedding:
        emb_in = helper.make_tensor_value_info(
            "card_ids_by_zone", TensorProto.INT64, [None, 8, 30]
        )
        inputs.append(emb_in)
    if with_uma_slot_ids:
        slot_ids_in = helper.make_tensor_value_info(
            "uma_slot_card_ids", TensorProto.INT64, [None, UMA_SLOT_COUNT]
        )
        inputs.append(slot_ids_in)
    if with_uma_slot_features:
        slot_feats_in = helper.make_tensor_value_info(
            "uma_slot_features",
            TensorProto.FLOAT,
            [None, UMA_SLOT_COUNT, UMA_SLOT_FEATURE_DIM],
        )
        inputs.append(slot_feats_in)
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [None, state_dim])
    graph = helper.make_graph(nodes, "g", inputs, [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    # Pin IR version to what the installed ORT supports (newer onnx defaults
    # to an IR version this ORT build rejects). This is purely a synthetic-
    # fixture concern; real exported graphs are produced by export_onnx.py.
    model.ir_version = min(model.ir_version, 10)
    onnx.checker.check_model(model)
    return ort.InferenceSession(
        model.SerializeToString(),
        sess_options=ort.SessionOptions(),
        providers=["CPUExecutionProvider"],
    )


def _expect_systemexit(label: str, fn) -> None:
    try:
        fn()
    except SystemExit as exc:
        assert exc.code == 2, f"{label}: expected SystemExit(2), got {exc.code!r}"
        print(f"  PASS  {label}: failed fast with SystemExit(2)")
        return
    raise AssertionError(f"{label}: expected SystemExit, but resolution succeeded")


def main() -> None:
    for p in (V2_MODEL, V3_MODEL):
        if not p.exists():
            print(
                f"SKIP serve-schema-guard smoke: missing checkpoint {p} "
                f"(environment gap, not a pass).",
                file=sys.stderr,
            )
            raise SystemExit(0)

    # Real 96-d v2 production graph -> "v2".
    v2_sess = _session(str(V2_MODEL))
    schema = _resolve_feature_schema("auto", v2_sess)
    assert schema == "v2", f"96-d graph resolved to {schema!r}, expected 'v2'"
    print(f"  PASS  real 96-d graph -> schema 'v2' (STATE_DIM_V2={STATE_DIM_V2})")

    # Real 110-d v3.0 graph -> "v3".
    v3_sess = _session(str(V3_MODEL))
    schema = _resolve_feature_schema("auto", v3_sess)
    assert schema == "v3", f"110-d graph resolved to {schema!r}, expected 'v3'"
    print(f"  PASS  real 110-d v3.0 graph -> schema 'v3' (STATE_DIM_V3={STATE_DIM_V3})")

    # Explicit pin consistent with graph: OK. Inconsistent: fail fast.
    assert _resolve_feature_schema("v2", v2_sess) == "v2"
    assert _resolve_feature_schema("v3", v3_sess) == "v3"
    print("  PASS  explicit consistent --feature-schema pins accepted")
    _expect_systemexit(
        "explicit v3 against 96-d graph",
        lambda: _resolve_feature_schema("v3", v2_sess),
    )
    _expect_systemexit(
        "explicit v2 against 110-d graph",
        lambda: _resolve_feature_schema("v2", v3_sess),
    )

    # 164-d -> R16-P1 landed: resolves to the REAL v3.1 builder (embedding
    # feeds, v3.0 head). It must NOT fall back to v3 and must NOT fail fast
    # anymore (that was the pre-P1 placeholder behaviour).
    s164 = _synthetic_session(STATE_DIM_V3_1, with_embedding=True)
    schema = _resolve_feature_schema("auto", s164)
    assert schema == "v3.1", f"164-d graph resolved to {schema!r}, expected 'v3.1'"
    assert _resolve_feature_schema("v3.1", s164) == "v3.1"
    print(f"  PASS  synthetic 164-d graph -> schema 'v3.1' (STATE_DIM_V3_1={STATE_DIM_V3_1})")
    # 164-d graph WITHOUT the embedding input is internally inconsistent
    # (v3.1 head is the v3.0 embedding encoding) -> fail fast.
    s164_noemb = _synthetic_session(STATE_DIM_V3_1, with_embedding=False)
    _expect_systemexit(
        "164-d graph missing card_ids_by_zone",
        lambda: _resolve_feature_schema("auto", s164_noemb),
    )
    # Explicit-pin mismatch still fails fast.
    _expect_systemexit(
        "explicit v3 against 164-d graph",
        lambda: _resolve_feature_schema("v3", s164),
    )

    # Wholly unknown dim -> fail fast.
    s999 = _synthetic_session(999, with_embedding=True)
    _expect_systemexit(
        "synthetic 999-d unknown schema",
        lambda: _resolve_feature_schema("auto", s999),
    )

    # Internally-inconsistent graph (110-d but no embedding input) -> fail fast.
    s110_noemb = _synthetic_session(STATE_DIM_V3, with_embedding=False)
    _expect_systemexit(
        "110-d graph missing card_ids_by_zone",
        lambda: _resolve_feature_schema("auto", s110_noemb),
    )

    # --- R16-P2 C5: v3.2 schema dispatch + cross-input fail-fast cases ----

    # Synthetic 110-d v3.2 graph (embedding + slot inputs) -> "v3.2".
    # This is the new schema entry; resolver must distinguish from the
    # v3.0 entry by input-set, NOT by state_dim (both share 110).
    s110_v32 = _synthetic_session(
        STATE_DIM_V3,
        with_embedding=True,
        with_uma_slot_ids=True,
        with_uma_slot_features=True,
    )
    schema = _resolve_feature_schema("auto", s110_v32)
    assert schema == "v3.2", f"110-d v3.2 graph resolved to {schema!r}, expected 'v3.2'"
    assert _resolve_feature_schema("v3.2", s110_v32) == "v3.2"
    print(
        f"  PASS  synthetic 110-d v3.2 graph -> schema 'v3.2' "
        f"(STATE_DIM_V3={STATE_DIM_V3}, has_uma_slot_inputs=True)"
    )

    # Cross-input fail-fast (1): a 110-d graph with `uma_slot_*` inputs
    # must NOT silently resolve to v3.0. Explicit `v3` pin against the
    # v3.2 graph fails fast. This is the load-bearing landmine case —
    # silent fallback here would mis-pack inputs and crash deep in ORT.
    _expect_systemexit(
        "explicit v3 against 110-d v3.2 graph",
        lambda: _resolve_feature_schema("v3", s110_v32),
    )

    # Cross-input fail-fast (2): a 110-d graph WITHOUT `uma_slot_*` inputs
    # must resolve to v3.0 (not v3.2). An explicit v3.2 pin against the
    # v3.0 graph fails fast. Symmetric to case (1); covers the other
    # silent-mis-routing direction.
    s110_v3 = _synthetic_session(
        STATE_DIM_V3,
        with_embedding=True,
        with_uma_slot_ids=False,
        with_uma_slot_features=False,
    )
    schema = _resolve_feature_schema("auto", s110_v3)
    assert schema == "v3", (
        f"110-d v3.0 graph (no slot inputs) resolved to {schema!r}, "
        f"expected 'v3' (must NOT silently promote to v3.2)"
    )
    print("  PASS  synthetic 110-d v3.0 graph (no slot inputs) -> schema 'v3' (no v3.2 promotion)")
    _expect_systemexit(
        "explicit v3.2 against 110-d v3.0 graph",
        lambda: _resolve_feature_schema("v3.2", s110_v3),
    )

    # Cross-input fail-fast (3): v3.2 only exists at state_dim=110.
    # A 96-d graph carrying slot inputs is illegal — refuse to serve.
    # This catches a broken export that wired slot inputs into a v2
    # graph by accident.
    s96_with_slots = _synthetic_session(
        STATE_DIM_V2,
        with_embedding=False,
        with_uma_slot_ids=True,
        with_uma_slot_features=True,
    )
    _expect_systemexit(
        "96-d graph carrying uma_slot_* inputs (v3.2 only at 110)",
        lambda: _resolve_feature_schema("auto", s96_with_slots),
    )

    # Cross-input fail-fast (4): symmetric to (3) on the 164-d v3.1
    # head. A 164-d graph with slot inputs is an unsupported combination
    # (v3.2 lives at state_dim=110); refuse to serve.
    s164_with_slots = _synthetic_session(
        STATE_DIM_V3_1,
        with_embedding=True,
        with_uma_slot_ids=True,
        with_uma_slot_features=True,
    )
    _expect_systemexit(
        "164-d graph carrying uma_slot_* inputs (v3.2 only at 110)",
        lambda: _resolve_feature_schema("auto", s164_with_slots),
    )

    # Cross-input fail-fast (5 + 6): partial-pair violations. The v3.2
    # contract is the PAIR `(uma_slot_card_ids, uma_slot_features)`; a
    # graph declaring exactly one is broken by construction. Both halves
    # are tested independently so an export-side regression that drops
    # one input is caught regardless of which one.
    s110_partial_ids_only = _synthetic_session(
        STATE_DIM_V3,
        with_embedding=True,
        with_uma_slot_ids=True,
        with_uma_slot_features=False,
    )
    _expect_systemexit(
        "110-d graph with only uma_slot_card_ids (no uma_slot_features)",
        lambda: _resolve_feature_schema("auto", s110_partial_ids_only),
    )
    s110_partial_feats_only = _synthetic_session(
        STATE_DIM_V3,
        with_embedding=True,
        with_uma_slot_ids=False,
        with_uma_slot_features=True,
    )
    _expect_systemexit(
        "110-d graph with only uma_slot_features (no uma_slot_card_ids)",
        lambda: _resolve_feature_schema("auto", s110_partial_feats_only),
    )

    # Production transparency: v2 pin still emits ONLY the three frozen feeds
    # (no embedding inputs) so existing 96-d serving is bit-identical. Use
    # the raw-arrays path so the check is independent of the action encoder.
    from uma_ai.features import ACTION_DIM

    payload = {
        "state_features": np.zeros((1, STATE_DIM_V2), dtype=np.float32).tolist(),
        "action_features": np.zeros((1, 1, ACTION_DIM), dtype=np.float32).tolist(),
        "action_mask": np.ones((1, 1), dtype=bool).tolist(),
    }
    feed, _ = request_to_arrays(payload, "v2")
    assert set(feed) == {"state_features", "action_features", "action_mask"}, (
        f"v2 feed keys changed: {sorted(feed)} (must stay the 3 frozen feeds)"
    )
    assert feed["state_features"].shape == (1, STATE_DIM_V2), (
        f"v2 state_features shape {feed['state_features'].shape}, "
        f"expected (1, {STATE_DIM_V2})"
    )
    # And the resolved 96-d model actually runs a prediction unchanged.
    feed3 = dict(feed)
    out = v2_sess.run(None, feed3)
    assert out and np.isfinite(np.asarray(out[0])).all(), "v2 /predict produced non-finite logits"
    print("  PASS  v2 pin transparent: 3 frozen feeds, 96-d, prediction finite")

    # --- R16-P2 C5: end-to-end v3.2 ONNX roundtrip ---------------------
    # Build a small C2 model with `uses_uma_slot_tokens=True`, export via
    # Part A's 7-input path, load via serve_onnx (which resolves to v3.2
    # by input-set), forward a populated batch, and assert PyTorch vs ORT
    # logits/value match within 1e-3. Mirrors the existing v3.0 roundtrip
    # discipline in `train_bc.run_onnx_roundtrip_smoke` and the v3.1
    # roundtrip in `r16_temporal_v31_smoke.py`.
    _v32_onnx_roundtrip()

    print("serve-schema-guard smoke: ALL PASS")


def _v32_onnx_roundtrip() -> None:
    """Build C2 model with `uses_uma_slot_tokens=True`, export via Part A,
    load via serve_onnx, assert PyTorch vs ORT agree within 1e-3 on a
    populated batch.

    R16-P2 C5 acceptance criterion: the full export -> serve roundtrip
    works end-to-end on a v3.2-shaped model. Touches all three parts of
    C5 (export_onnx 7-input graph, serve_onnx v3.2 dispatch, slot-tensor
    packing path in `request_to_arrays`).
    """
    from uma_ai.features import ACTION_DIM, CARD_ID_SHAPES, STATE_DIM
    from uma_ai.model import NUM_ZONES, CandidatePolicyNet, ModelConfig

    # Tiny model: hidden_dim=16, depth=1 keeps the export fast (<1 sec on
    # CPU) while still exercising the slot-encoder branch and the joint
    # projection that fans out the new tensors. State dim is STATE_DIM
    # (110) — v3.2 reuses the v3.0 width.
    config = ModelConfig(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=16,
        depth=1,
        dropout=0.0,
        uses_uma_slot_tokens=True,
    )
    model = CandidatePolicyNet(config).eval()
    # Stamp a small non-zero weight into the encoder's FINAL Linear so the
    # slot branch contributes signal (default zero-init would make the
    # roundtrip trivially pass with all-zero output). This mirrors C2's
    # smoke convention. Use a deterministic rng so failures are reproducible.
    with torch.no_grad():
        gen = torch.Generator().manual_seed(7)
        model.uma_slot_encoder[-1].weight.copy_(
            torch.randn(
                model.uma_slot_encoder[-1].weight.shape, generator=gen
            ) * 0.05
        )

    max_cards = max(CARD_ID_SHAPES.values())
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ckpt_path = tmp_path / "ckpt.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_config": config.to_dict(),
            },
            ckpt_path,
        )
        onnx_path = tmp_path / "policy.v32.onnx"
        # Drive the exporter end-to-end via its CLI entrypoint so we
        # exercise the same code path serve_onnx will encounter in prod.
        import export_onnx

        argv_backup = sys.argv
        sys.argv = [
            "export_onnx.py",
            "--checkpoint",
            str(ckpt_path),
            "--out",
            str(onnx_path),
            "--max-actions",
            "4",
        ]
        try:
            export_onnx.main()
        finally:
            sys.argv = argv_backup

        # Load via serve_onnx's resolver — this confirms the dispatch
        # correctly identifies the graph as v3.2 from the input-set.
        session = ort.InferenceSession(
            str(onnx_path),
            sess_options=ort.SessionOptions(),
            providers=["CPUExecutionProvider"],
        )
        resolved = _resolve_feature_schema("auto", session)
        assert resolved == "v3.2", (
            f"v3.2 export resolved to {resolved!r}, expected 'v3.2'"
        )

        # Populated batch: non-zero state, action features, embedding ids,
        # slot ids + slot features. This exercises the slot Gather +
        # Concat + MatMul + Sum path inside ORT (vs the all-zero
        # structural-null short-circuit).
        gen = torch.Generator().manual_seed(13)
        state = torch.randn((1, STATE_DIM), generator=gen)
        actions = torch.randn((1, 4, ACTION_DIM), generator=gen)
        mask = torch.ones((1, 4), dtype=torch.bool)
        czi = torch.randint(
            low=1,
            high=model.card_embed.num_embeddings,
            size=(1, NUM_ZONES, max_cards),
            dtype=torch.int64,
            generator=gen,
        )
        aci = torch.randint(
            low=1,
            high=model.card_embed.num_embeddings,
            size=(1, 4, 2),
            dtype=torch.int64,
            generator=gen,
        )
        slot_ids = torch.randint(
            low=1,
            high=model.card_embed.num_embeddings,
            size=(1, UMA_SLOT_COUNT),
            dtype=torch.int64,
            generator=gen,
        )
        slot_feats = torch.randn((1, UMA_SLOT_COUNT, UMA_SLOT_FEATURE_DIM), generator=gen)

        ort_logits, ort_value = session.run(
            None,
            {
                "state_features": state.numpy(),
                "action_features": actions.numpy(),
                "action_mask": mask.numpy(),
                "card_ids_by_zone": czi.numpy(),
                "action_card_idx": aci.numpy(),
                "uma_slot_card_ids": slot_ids.numpy(),
                "uma_slot_features": slot_feats.numpy(),
            },
        )
        with torch.no_grad():
            torch_logits, torch_value = model(
                state,
                actions,
                mask,
                card_ids_by_zone=czi,
                action_card_idx=aci,
                uma_slot_card_ids=slot_ids,
                uma_slot_features=slot_feats,
            )
        max_logit_diff = float((torch.from_numpy(ort_logits) - torch_logits).abs().max())
        max_value_diff = float((torch.from_numpy(ort_value) - torch_value).abs().max())
        if max_logit_diff > 1e-3 or max_value_diff > 1e-3:
            raise AssertionError(
                f"v3.2 ONNX roundtrip mismatch: logits {max_logit_diff} "
                f"value {max_value_diff} (tolerance 1e-3)"
            )

        # And the request_to_arrays raw-arrays path produces feeds that
        # ORT accepts (1:1 with the manual feed dict above; this catches a
        # regression in the v3.2 packing branch).
        payload = {
            "state_features": state.numpy().tolist(),
            "action_features": actions.numpy().tolist(),
            "action_mask": mask.numpy().tolist(),
            "card_ids_by_zone": czi.numpy().tolist(),
            "action_card_idx": aci.numpy().tolist(),
            "uma_slot_card_ids": slot_ids.numpy().tolist(),
            "uma_slot_features": slot_feats.numpy().tolist(),
        }
        feed, _ = request_to_arrays(payload, "v3.2")
        assert "uma_slot_card_ids" in feed and "uma_slot_features" in feed, (
            "v3.2 request_to_arrays did not emit slot feeds"
        )
        assert feed["uma_slot_card_ids"].shape == (1, UMA_SLOT_COUNT), (
            f"v3.2 uma_slot_card_ids shape {feed['uma_slot_card_ids'].shape}, "
            f"expected (1, {UMA_SLOT_COUNT})"
        )
        assert feed["uma_slot_features"].shape == (
            1,
            UMA_SLOT_COUNT,
            UMA_SLOT_FEATURE_DIM,
        ), (
            f"v3.2 uma_slot_features shape {feed['uma_slot_features'].shape}, "
            f"expected (1, {UMA_SLOT_COUNT}, {UMA_SLOT_FEATURE_DIM})"
        )
        ort_logits_pkt, _ = session.run(None, feed)
        assert np.isfinite(ort_logits_pkt).all(), (
            "v3.2 /predict packing produced non-finite logits"
        )

        # And confirm the v3.0 path still rejects slot inputs (ORT would
        # reject unknown feed keys; we proactively check request_to_arrays
        # under a v3 pin does NOT emit them, so the v3 graph stays safe).
        feed_v3, _ = request_to_arrays(payload, "v3")
        assert "uma_slot_card_ids" not in feed_v3, (
            "v3 pin leaked uma_slot_card_ids into the feed"
        )
        assert "uma_slot_features" not in feed_v3, (
            "v3 pin leaked uma_slot_features into the feed"
        )

    print(
        f"  PASS  v3.2 ONNX roundtrip: logits diff {max_logit_diff:.2e}, "
        f"value diff {max_value_diff:.2e} (tolerance 1e-3)"
    )
    print("  PASS  v3.2 request_to_arrays packs slot feeds; v3 pin omits them")


if __name__ == "__main__":
    main()
