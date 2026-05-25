"""Smoke `training/make_v35_tail_init.py` end-to-end.

Builds a v3.5 init from a v3.3 source ckpt and asserts:
  1. The widened `state_encoder.0.weight` has shape `(hidden, 212)`.
  2. The 45 appended columns are exactly zero.
  3. All other parameters are byte-identical to the source.
  4. `delta_logits` from the builder's internal parity check is below
     the zero-init-residual tolerance (1e-5). Structurally identical via
     `state_v33 + 45 zeros` × widened-with-zero-columns; any non-zero
     value is fp-accumulation jitter, not a contract break.
  5. The dest ckpt's `feature_schema` is tagged state_dim=212,
     state_feature_schema_version=3.5.

The default source ckpt is the R16-P1 v3.3 lineage best
(`runs/R16-P1-v33-ablation/loop/iter-1/checkpoint.pt`). Tolerates a custom
path for ad-hoc smoking.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_v35_tail_init import WIDENED_KEY, build  # noqa: E402
from uma_ai.features import (  # noqa: E402
    STATE_DIM_V3_3,
    STATE_DIM_V3_5,
    STATE_FEATURE_SCHEMA_VERSION_V3_5,
)

DEFAULT_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "runs/R16-P1-v33-ablation/loop/iter-1/checkpoint.pt"
)
DELTA_LOGITS_TOLERANCE = 1e-5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"SKIP v35-tail-init smoke: missing source {source}")

    with tempfile.TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir) / "v35_init.pt"
        provenance = build(source, dest)

        # Provenance contract.
        if provenance["dest_state_dim"] != STATE_DIM_V3_5:
            raise SystemExit(
                f"FAIL: dest_state_dim={provenance['dest_state_dim']}, "
                f"expected {STATE_DIM_V3_5}"
            )
        delta = provenance["delta_logits_at_init"]
        if not (delta < DELTA_LOGITS_TOLERANCE):
            raise SystemExit(
                f"FAIL: delta_logits_at_init={delta:.3e} "
                f">= {DELTA_LOGITS_TOLERANCE:.0e} tolerance"
            )

        # Re-load dest and inspect tensors directly.
        payload = torch.load(dest, map_location="cpu", weights_only=False)
        state = payload["model_state"]
        widened = state[WIDENED_KEY]
        if widened.shape[-1] != STATE_DIM_V3_5:
            raise SystemExit(
                f"FAIL: widened last dim={widened.shape[-1]}, expected {STATE_DIM_V3_5}"
            )

        appended = widened[:, STATE_DIM_V3_3:]
        if not torch.equal(appended, torch.zeros_like(appended)):
            raise SystemExit(
                f"FAIL: appended {appended.shape[-1]} columns are not all zero"
            )

        source_payload = torch.load(source, map_location="cpu", weights_only=False)
        source_state = source_payload["model_state"]
        source_widened = source_state[WIDENED_KEY]
        if not torch.equal(widened[:, :STATE_DIM_V3_3], source_widened):
            raise SystemExit("FAIL: widened[:, :167] differs from source weight")
        for key, value in source_state.items():
            if key == WIDENED_KEY:
                continue
            if not torch.equal(state[key], value):
                raise SystemExit(f"FAIL: param `{key}` differs from source")

        # Schema metadata.
        fs = payload.get("feature_schema") or {}
        if fs.get("state_dim") != STATE_DIM_V3_5:
            raise SystemExit(
                f"FAIL: feature_schema.state_dim={fs.get('state_dim')}"
            )
        if fs.get("state_feature_schema_version") != STATE_FEATURE_SCHEMA_VERSION_V3_5:
            raise SystemExit(
                f"FAIL: feature_schema.state_feature_schema_version="
                f"{fs.get('state_feature_schema_version')}"
            )

        # ModelConfig sanity: dest must be uses_uma_slot_tokens=False
        # (v3.5 explicitly excludes slot tokens, per scoping doc §3).
        target_cfg = payload.get("model_config") or {}
        if target_cfg.get("state_dim") != STATE_DIM_V3_5:
            raise SystemExit(
                f"FAIL: model_config.state_dim={target_cfg.get('state_dim')}"
            )
        if target_cfg.get("uses_uma_slot_tokens", True):
            raise SystemExit(
                "FAIL: v3.5 ckpt must have uses_uma_slot_tokens=False"
            )

    print("v35_tail_init_smoke: 6/6 cases PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
