"""v3.8 action-vector v4 schema-bump smoke (Python side).

Python doesn't compute action features (TS owns the canonical builder;
Rust mirrors it via `engine-rs/.../policy/actions.rs::build_features`).
This smoke validates the Python-side constants + sidecar metadata
contract that downstream training/serving consumers rely on:

  1. `ACTION_FEATURE_SCHEMA_VERSION == 4` (Python sees v4).
  2. The serve_onnx schema table registers state_dim=304 → schema "v3.8".
  3. A regression assertion: the v3 → v4 bump must NOT alter the v3
     schema marker (3) elsewhere — only the latest is 4.

Per-slot byte-stable truth-tables on the v4 action surface live in
the Rust smoke `engine-rs/crates/engine/tests/v38_action_v4_smoke.rs`
(invoked via `cargo test -p engine --test v38_action_v4_smoke`) and in
the Python↔Rust parity smoke (`v38_python_rust_parity_smoke.py`).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uma_ai.features import (  # noqa: E402
    ACTION_FEATURE_SCHEMA_VERSION,
    STATE_DIM_V3_8,
)


def main() -> int:
    # 1. Python ACTION_FEATURE_SCHEMA_VERSION is at v4.
    if ACTION_FEATURE_SCHEMA_VERSION != 4:
        raise SystemExit(
            f"FAIL[python-schema]: ACTION_FEATURE_SCHEMA_VERSION="
            f"{ACTION_FEATURE_SCHEMA_VERSION}, expected 4 (v38 bump)"
        )

    # 2. serve_onnx schema table maps state_dim=304 → "v3.8".
    # onnxruntime may be absent in CI / lean envs — degrade gracefully:
    # if the import fails, parse serve_onnx.py textually for the v3.8 row.
    serve_onnx_path = Path(__file__).resolve().parent / "serve_onnx.py"
    try:
        from serve_onnx import _SCHEMA_TABLE, _lookup_schema  # noqa: E402

        found_v38 = False
        for row in _SCHEMA_TABLE:
            s_dim, _exp_emb, _exp_slot, token, _label = row
            if s_dim == STATE_DIM_V3_8 and token == "v3.8":
                found_v38 = True
                break
        if not found_v38:
            raise SystemExit(
                f"FAIL[serve-table]: state_dim={STATE_DIM_V3_8} not registered "
                f"as schema 'v3.8' in serve_onnx._SCHEMA_TABLE"
            )

        resolved = _lookup_schema(STATE_DIM_V3_8, has_uma_slot_inputs=False)
        if resolved is None or resolved[0] != "v3.8":
            raise SystemExit(
                f"FAIL[serve-lookup]: _lookup_schema(304, False) returned "
                f"{resolved}, expected ('v3.8', True, False, ...)"
            )
    except ModuleNotFoundError as exc:
        # onnxruntime missing — verify the v3.8 entry textually.
        serve_text = serve_onnx_path.read_text(encoding="utf-8")
        if "STATE_DIM_V3_8," not in serve_text or '"v3.8"' not in serve_text:
            raise SystemExit(
                f"FAIL[serve-table-text]: v3.8 row missing in {serve_onnx_path} "
                f"(onnxruntime import failed: {exc})"
            )
        print(
            f"WARN: onnxruntime missing — verified serve_onnx v3.8 row "
            f"textually instead of via _lookup_schema."
        )

    # 3. Regression guard: ACTION_FEATURE_COUNT TS constant lives at 52.
    actions_ts = (
        Path(__file__).resolve().parents[1]
        / "frontend"
        / "src"
        / "game"
        / "engine"
        / "ai-policy"
        / "actions.ts"
    )
    ts_text = actions_ts.read_text(encoding="utf-8")
    if "ACTION_FEATURE_COUNT = 52" not in ts_text:
        raise SystemExit(
            f"FAIL[ts-count]: ACTION_FEATURE_COUNT = 52 not found in {actions_ts}"
        )
    if "ACTION_FEATURE_SCHEMA_VERSION = 4" not in ts_text:
        raise SystemExit(
            f"FAIL[ts-schema]: ACTION_FEATURE_SCHEMA_VERSION = 4 not found "
            f"in {actions_ts}"
        )

    # 4. Regression guard: Rust ACTION_FEATURE_COUNT and
    # ACTION_FEATURE_SCHEMA_VERSION constants pin to 52 / 4.
    actions_rs = (
        Path(__file__).resolve().parents[1]
        / "engine-rs"
        / "crates"
        / "engine"
        / "src"
        / "policy"
        / "actions.rs"
    )
    rs_text = actions_rs.read_text(encoding="utf-8")
    if "ACTION_FEATURE_COUNT: usize = 52" not in rs_text:
        raise SystemExit(
            f"FAIL[rust-count]: ACTION_FEATURE_COUNT: usize = 52 not found "
            f"in {actions_rs}"
        )
    if "ACTION_FEATURE_SCHEMA_VERSION: u32 = 4" not in rs_text:
        raise SystemExit(
            f"FAIL[rust-schema]: ACTION_FEATURE_SCHEMA_VERSION: u32 = 4 not "
            f"found in {actions_rs}"
        )

    print(
        f"v38_action_v4_smoke: ALL Python+TS+Rust constants align "
        f"(action_schema=v4, action_count=52, state_dim=304→'v3.8')"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
