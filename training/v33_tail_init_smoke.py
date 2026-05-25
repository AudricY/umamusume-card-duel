"""Smoke `training/make_v33_tail_init.py` end-to-end.

Builds a v3.3 init from a v3.1 source ckpt and asserts:
  1. The widened `state_encoder.0.weight` has shape `(hidden, 167)`.
  2. The 3 appended columns are exactly zero.
  3. All other parameters are byte-identical to the source.
  4. `delta_logits` from the builder's internal parity check is exactly 0
     (zero-init residual contract).
  5. The dest ckpt's `feature_schema` is tagged state_dim=167,
     state_feature_schema_version=3.3.

The default source ckpt is the R16-P1 v3.1 lineage best
(`runs/R16-P1-v31-ablation/loop/iter-3/checkpoint.pt`). Tolerates a custom
path for ad-hoc smoking.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_v33_tail_init import WIDENED_KEY, build  # noqa: E402
from uma_ai.features import STATE_DIM_V3_1, STATE_DIM_V3_3, STATE_FEATURE_SCHEMA_VERSION_V3_3  # noqa: E402

DEFAULT_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "runs/R16-P1-v31-ablation/loop/iter-3/checkpoint.pt"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"SKIP v33-tail-init smoke: missing source {source}")

    with tempfile.TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir) / "v33_init.pt"
        provenance = build(source, dest)

        # Provenance contract.
        if provenance["dest_state_dim"] != STATE_DIM_V3_3:
            raise SystemExit(
                f"FAIL: dest_state_dim={provenance['dest_state_dim']}, expected {STATE_DIM_V3_3}"
            )
        if provenance["delta_logits_at_init"] != 0.0:
            raise SystemExit(
                f"FAIL: delta_logits_at_init={provenance['delta_logits_at_init']} (expected 0.0)"
            )

        # Re-load dest and inspect tensors directly.
        payload = torch.load(dest, map_location="cpu", weights_only=False)
        state = payload["model_state"]
        widened = state[WIDENED_KEY]
        if widened.shape[-1] != STATE_DIM_V3_3:
            raise SystemExit(f"FAIL: widened last dim={widened.shape[-1]}, expected {STATE_DIM_V3_3}")

        appended = widened[:, STATE_DIM_V3_1:]
        if not torch.equal(appended, torch.zeros_like(appended)):
            raise SystemExit(f"FAIL: appended {appended.shape[-1]} columns are not all zero")

        source_payload = torch.load(source, map_location="cpu", weights_only=False)
        source_state = source_payload["model_state"]
        source_widened = source_state[WIDENED_KEY]
        if not torch.equal(widened[:, :STATE_DIM_V3_1], source_widened):
            raise SystemExit("FAIL: widened[:, :164] differs from source weight")
        for key, value in source_state.items():
            if key == WIDENED_KEY:
                continue
            if not torch.equal(state[key], value):
                raise SystemExit(f"FAIL: param `{key}` differs from source")

        # Schema metadata.
        fs = payload.get("feature_schema") or {}
        if fs.get("state_dim") != STATE_DIM_V3_3:
            raise SystemExit(f"FAIL: feature_schema.state_dim={fs.get('state_dim')}")
        if fs.get("state_feature_schema_version") != STATE_FEATURE_SCHEMA_VERSION_V3_3:
            raise SystemExit(
                f"FAIL: feature_schema.state_feature_schema_version="
                f"{fs.get('state_feature_schema_version')}"
            )

    print("v33_tail_init_smoke: 5/5 cases PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
