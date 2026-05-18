"""Focused smoke for the serve_onnx 96/110/164 serving-schema guard.

r16 P1 prerequisite. Proves `serve_onnx._resolve_feature_schema` resolves
the feature builder STRICTLY from the loaded ONNX graph's `state_features`
last dim:

  - a real 96-d v2 production graph  -> schema "v2" (no embedding feeds)
  - a real 110-d v3.0 graph          -> schema "v3" (embedding feeds)
  - a synthetic 164-d v3.1 graph     -> schema "v3.1" (embedding feeds)
                                        [R16-P1: now a REAL builder, no
                                        longer a fail-fast placeholder]
  - a synthetic 999-d graph          -> fail fast (unknown schema; SystemExit)

It also runs the production-transparency check: the resolved schema for the
pinned 96-d model is unchanged ("v2"), and `request_to_arrays` still emits
exactly the three frozen v2 feeds (no embedding inputs) for a v2 pin.

Deliberately lean: this is a shape-resolution check, not a strength/matrix
test. ORT is required (it is a serve_onnx dependency).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from serve_onnx import _resolve_feature_schema, request_to_arrays
from uma_ai.features import STATE_DIM_V2, STATE_DIM_V3, STATE_DIM_V3_1

REPO_ROOT = Path(__file__).resolve().parents[1]
V2_MODEL = REPO_ROOT / "runs" / "R13-W6-phase-d" / "iter-2" / "policy.onnx"
V3_MODEL = REPO_ROOT / "runs" / "R110-W6-repro" / "iter-0" / "policy.onnx"


def _session(path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


def _synthetic_session(state_dim: int, *, with_embedding: bool) -> ort.InferenceSession:
    """Build a minimal valid ONNX graph with a state_features input of the
    given last dim (and optionally a card_ids_by_zone embedding input) so the
    guard can be exercised on dims for which no real model exists."""
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

    print("serve-schema-guard smoke: ALL PASS")


if __name__ == "__main__":
    main()
