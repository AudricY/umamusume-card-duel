"""Smoke `training/make_v37_combat_arith_init.py` end-to-end.

Builds a v3.7 init from a v3.6 source ckpt and asserts:
  1. The widened `state_encoder.0.weight` has shape `(hidden, 296)`.
  2. The appended tail columns at [246:296] are exactly zero (truly-new
     v3.7 combat-arith + catalog channels, zero-init).
  3. The preserved columns at [0:246] are byte-identical to the v3.6
     source weight columns at the same indices.
  4. All other parameters are byte-identical to the source.
  5. `delta_logits` from the builder's internal parity check is below
     the v3.7 iter-0 tolerance (1e-5, strict — matching v3.5's band,
     because v3.7 is a pure-zero-init append with no column-drop
     unlike v3.6's 1e-3).
  6. The dest ckpt's `feature_schema` is tagged state_dim=296,
     state_feature_schema_version=3.7.
  7. The dest ckpt's `model_config` records state_dim=296 and
     uses_uma_slot_tokens=False (v3.7 contract — slot tokens excluded).

Source preference order:
  (a) --source CLI argument if provided.
  (b) The v3.6 cont-iter2 lineage best
      (`runs/R16-P1-v36-cap128-A1-continue/loop/iter-2/checkpoint.pt`)
      — the §4l-anchored cont-iter2 ckpt the v3.7 init parity contract
      is measured against.
  (c) Synthesise a minimal random-weight v3.6-shaped ckpt in tempdir.
      CLAUDE.md treats missing `runs/` ckpts as environment gaps, not
      product passes — so the smoke degrades gracefully rather than
      failing on a missing artifact.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_v37_combat_arith_init import (  # noqa: E402
    PARITY_TOLERANCE,
    WIDENED_KEY,
    build,
)
from uma_ai.features import (  # noqa: E402
    STATE_DIM_V3_6,
    STATE_DIM_V3_7,
    STATE_FEATURE_SCHEMA_VERSION_V3_7,
)
from uma_ai.model import CandidatePolicyNet, ModelConfig  # noqa: E402

DEFAULT_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "runs/R16-P1-v36-cap128-A1-continue/loop/iter-2/checkpoint.pt"
)


def _fabricate_v36_ckpt(dest: Path) -> Path:
    """Create a minimal random-weight v3.6-shaped checkpoint at `dest`.

    Matches the payload shape `make_v37_combat_arith_init.build` expects:
    `model_state`, `model_config`, `feature_schema`. Uses the same
    compact `hidden_dim=64, depth=2` config that the smoke fall-back
    path uses across v3.5/v3.6 fabrication; this keeps the smoke
    self-contained when the real ckpt is missing."""

    torch.manual_seed(20260526)
    config = ModelConfig(
        state_dim=STATE_DIM_V3_6,
        action_dim=48,
        hidden_dim=64,
        depth=2,
        dropout=0.05,
        uses_uma_slot_tokens=False,
        model_variant="mlp",
    )
    model = CandidatePolicyNet(config)
    state = model.state_dict()
    feature_schema = {
        "state_dim": STATE_DIM_V3_6,
        "action_dim": 48,
        "state_feature_schema_version": 3.6,
        "action_feature_schema_version": 3,
    }
    payload = {
        "model_state": state,
        "model_config": config.to_dict(),
        "feature_schema": feature_schema,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, dest)
    return dest


def _resolve_source(cli_source: str | None, tmpdir: Path) -> tuple[Path, str]:
    """Choose a v3.6 source ckpt path. Returns (path, kind)."""

    if cli_source:
        p = Path(cli_source)
        if not p.exists():
            raise SystemExit(f"FAIL: --source {p} does not exist")
        return p, "cli"
    if DEFAULT_SOURCE.exists():
        return DEFAULT_SOURCE, "real-v36-cap128-cont-iter-2"
    fabricated = tmpdir / "fabricated_v36.pt"
    _fabricate_v36_ckpt(fabricated)
    return fabricated, "fabricated-random-v36"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp_str:
        tmpdir = Path(tmp_str)
        source, source_kind = _resolve_source(args.source, tmpdir)
        dest = tmpdir / "v37_init.pt"
        provenance = build(source, dest)

        # 1. Shape provenance contract.
        if provenance["dest_state_dim"] != STATE_DIM_V3_7:
            raise SystemExit(
                f"FAIL[1]: dest_state_dim={provenance['dest_state_dim']}, "
                f"expected {STATE_DIM_V3_7}"
            )

        # 5. Δlogits tolerance.
        delta = provenance["delta_logits_at_init"]
        if not (delta < PARITY_TOLERANCE):
            raise SystemExit(
                f"FAIL[5]: delta_logits_at_init={delta:.3e} "
                f">= {PARITY_TOLERANCE:.0e} tolerance"
            )

        # Re-load dest and inspect tensors directly.
        payload = torch.load(dest, map_location="cpu", weights_only=False)
        state = payload["model_state"]
        widened = state[WIDENED_KEY]
        if widened.shape[-1] != STATE_DIM_V3_7:
            raise SystemExit(
                f"FAIL[1b]: widened last dim={widened.shape[-1]}, "
                f"expected {STATE_DIM_V3_7}"
            )

        # 2. Appended tail [246:296] is zero.
        appended = widened[:, STATE_DIM_V3_6:STATE_DIM_V3_7]
        if not torch.equal(appended, torch.zeros_like(appended)):
            raise SystemExit(
                f"FAIL[2]: appended tail [{STATE_DIM_V3_6}:{STATE_DIM_V3_7}] "
                f"({appended.shape[-1]} columns) not all zero"
            )

        # 3. Preserved columns [0:246] byte-identical to source.
        source_payload = torch.load(source, map_location="cpu", weights_only=False)
        source_state = source_payload["model_state"]
        source_widened = source_state[WIDENED_KEY]
        if not torch.equal(widened[:, 0:STATE_DIM_V3_6], source_widened):
            raise SystemExit(
                f"FAIL[3]: widened[:, 0:{STATE_DIM_V3_6}] differs from "
                f"source v3.6 weight"
            )

        # 4. All other parameters byte-identical to source.
        for key, value in source_state.items():
            if key == WIDENED_KEY:
                continue
            if not torch.equal(state[key], value):
                raise SystemExit(f"FAIL[4]: param `{key}` differs from source")

        # 6. Schema metadata.
        fs = payload.get("feature_schema") or {}
        if fs.get("state_dim") != STATE_DIM_V3_7:
            raise SystemExit(
                f"FAIL[6a]: feature_schema.state_dim={fs.get('state_dim')}"
            )
        if fs.get("state_feature_schema_version") != STATE_FEATURE_SCHEMA_VERSION_V3_7:
            raise SystemExit(
                f"FAIL[6b]: feature_schema.state_feature_schema_version="
                f"{fs.get('state_feature_schema_version')}"
            )

        # 7. ModelConfig sanity.
        target_cfg = payload.get("model_config") or {}
        if target_cfg.get("state_dim") != STATE_DIM_V3_7:
            raise SystemExit(
                f"FAIL[7a]: model_config.state_dim={target_cfg.get('state_dim')}"
            )
        if target_cfg.get("uses_uma_slot_tokens", True):
            raise SystemExit(
                "FAIL[7b]: v3.7 ckpt must have uses_uma_slot_tokens=False"
            )

    print(
        f"v37_tail_init_smoke: 7/7 cases PASS "
        f"(source_kind={source_kind}, delta_logits={delta:.3e}, "
        f"tolerance={PARITY_TOLERANCE:.0e})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
