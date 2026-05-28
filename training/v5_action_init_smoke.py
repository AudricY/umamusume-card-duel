"""Smoke `training/make_v5_action_init.py` end-to-end.

Builds a v5 init from a v3.8 source ckpt and asserts:
  1. Widened action input projection has shape (out_dim, 57).
  2. Appended action tail columns at [52:57] are exactly zero.
  3. Preserved action columns at [0:52] are byte-identical to the v3.8
     source weight at the same indices.
  4. state_encoder.0.weight is BYTE-IDENTICAL to the source (v5 does not
     touch the state-axis).
  5. All OTHER parameters are byte-identical to the source.
  6. delta_logits from the builder's internal parity check is below the
     v5 iter-0 tolerance (1e-5, strict).
  7. Dest ckpt feature_schema tagged state_dim=304 (unchanged),
     state_feature_schema_version=3.8 (unchanged),
     action_feature_schema_version=5, action_dim=57.
  8. Dest ckpt model_config records state_dim=304 (unchanged),
     action_dim=57, uses_uma_slot_tokens=False, hidden_dim=128, depth=2
     (NO trunk widening — v3.8 §13.3 contraindication still applies).

Source preference order:
  (a) --source CLI argument if provided.
  (b) The v3.8 A1-promoted lineage best at
      `runs/R16-P1-v38-slim-A1/loop/iter-1/checkpoint.pt` (TBD path
      pending v3.8 verdict; smoke falls back to fabricated if missing).
  (c) Synthesise a minimal random-weight v3.8-shaped (hidden=128,
      depth=2) ckpt in tempdir.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_v5_action_init import (  # noqa: E402
    PARITY_TOLERANCE,
    _find_action_input_projection_key,
    build,
)
from uma_ai.features import (  # noqa: E402
    ACTION_DIM,
    ACTION_FEATURE_SCHEMA_VERSION,
    STATE_DIM_V3_8,
    STATE_FEATURE_SCHEMA_VERSION_V3_8,
)
from uma_ai.model import CandidatePolicyNet, ModelConfig  # noqa: E402

STATE_WIDENED_KEY = "state_encoder.0.weight"

DEFAULT_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "runs/R16-P1-v38-slim-A1/loop/iter-1/checkpoint.pt"
)

_EXPECTED_SOURCE_ACTION_DIM = 52
_TARGET_ACTION_DIM = 57


def _fabricate_v38_ckpt(dest: Path) -> Path:
    """Create a minimal random-weight v3.8-shaped (cap128) checkpoint at
    `dest`. Matches the payload shape `make_v5_action_init.build`
    expects: `model_state`, `model_config`, `feature_schema`.
    """

    torch.manual_seed(20260526)
    config = ModelConfig(
        state_dim=STATE_DIM_V3_8,
        action_dim=_EXPECTED_SOURCE_ACTION_DIM,
        hidden_dim=128,
        depth=2,
        dropout=0.05,
        uses_uma_slot_tokens=False,
        model_variant="mlp",
    )
    model = CandidatePolicyNet(config)
    state = model.state_dict()
    feature_schema = {
        "state_dim": STATE_DIM_V3_8,
        "action_dim": _EXPECTED_SOURCE_ACTION_DIM,
        "state_feature_schema_version": STATE_FEATURE_SCHEMA_VERSION_V3_8,
        "action_feature_schema_version": 4,
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
    if cli_source:
        p = Path(cli_source)
        if not p.exists():
            raise SystemExit(f"FAIL: --source {p} does not exist")
        return p, "cli"
    if DEFAULT_SOURCE.exists():
        return DEFAULT_SOURCE, "real-v38-A1-iter1"
    fabricated = tmpdir / "fabricated_v38.pt"
    _fabricate_v38_ckpt(fabricated)
    return fabricated, "fabricated-random-v38-cap128"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp_str:
        tmpdir = Path(tmp_str)
        source, source_kind = _resolve_source(args.source, tmpdir)
        dest = tmpdir / "v5_init.pt"
        provenance = build(source, dest)

        # 1. Shape provenance contract — state UNCHANGED, action widened.
        if provenance["dest_state_dim"] != STATE_DIM_V3_8:
            raise SystemExit(
                f"FAIL[1a]: dest_state_dim={provenance['dest_state_dim']}, "
                f"expected {STATE_DIM_V3_8} (v5 does not widen state)"
            )
        if provenance["dest_action_dim"] != _TARGET_ACTION_DIM:
            raise SystemExit(
                f"FAIL[1b]: dest_action_dim={provenance['dest_action_dim']}, "
                f"expected {_TARGET_ACTION_DIM}"
            )
        if provenance["state_encoder_widened"]:
            raise SystemExit(
                "FAIL[1c]: v5 expander reported state_encoder_widened=True; "
                "v5 is action-axis only."
            )

        # 6. Δlogits tolerance.
        delta = provenance["delta_logits_at_init"]
        if not (delta < PARITY_TOLERANCE):
            raise SystemExit(
                f"FAIL[6]: delta_logits_at_init={delta:.3e} "
                f">= {PARITY_TOLERANCE:.0e} tolerance"
            )

        # Re-load dest and inspect tensors directly.
        payload = torch.load(dest, map_location="cpu", weights_only=False)
        state = payload["model_state"]
        source_payload = torch.load(source, map_location="cpu", weights_only=False)
        source_state = source_payload["model_state"]

        action_widened_key = _find_action_input_projection_key(source_state)
        widened_action = state[action_widened_key]
        # 1d. Widened action shape.
        if widened_action.shape[-1] != _TARGET_ACTION_DIM:
            raise SystemExit(
                f"FAIL[1d]: widened action `{action_widened_key}` last dim="
                f"{widened_action.shape[-1]}, expected {_TARGET_ACTION_DIM}"
            )
        # 2. Appended action tail [52:57] is zero.
        appended_action = widened_action[:, _EXPECTED_SOURCE_ACTION_DIM:_TARGET_ACTION_DIM]
        if not torch.equal(appended_action, torch.zeros_like(appended_action)):
            raise SystemExit(
                f"FAIL[2]: appended action tail "
                f"[{_EXPECTED_SOURCE_ACTION_DIM}:{_TARGET_ACTION_DIM}] "
                f"({appended_action.shape[-1]} columns) not all zero"
            )
        # 3. Preserved action columns [0:52] byte-identical to source.
        source_action_weight = source_state[action_widened_key]
        if not torch.equal(
            widened_action[:, 0:_EXPECTED_SOURCE_ACTION_DIM], source_action_weight
        ):
            raise SystemExit(
                f"FAIL[3]: widened_action[:, 0:{_EXPECTED_SOURCE_ACTION_DIM}] "
                f"differs from source v3.8 weight"
            )

        # 4. state_encoder.0.weight UNCHANGED.
        if STATE_WIDENED_KEY in source_state:
            if not torch.equal(state[STATE_WIDENED_KEY], source_state[STATE_WIDENED_KEY]):
                raise SystemExit(
                    f"FAIL[4]: state_encoder.0.weight differs from source — "
                    f"v5 should leave the state-axis untouched."
                )
            # Last-dim sanity: still 304.
            if state[STATE_WIDENED_KEY].shape[-1] != STATE_DIM_V3_8:
                raise SystemExit(
                    f"FAIL[4b]: state_encoder.0.weight last dim="
                    f"{state[STATE_WIDENED_KEY].shape[-1]}, expected "
                    f"{STATE_DIM_V3_8} (state unchanged)"
                )

        # 5. All other parameters byte-identical to source.
        for key, value in source_state.items():
            if key == action_widened_key:
                continue
            if not torch.equal(state[key], value):
                raise SystemExit(f"FAIL[5]: param `{key}` differs from source")

        # 7. Schema metadata.
        fs = payload.get("feature_schema") or {}
        if fs.get("state_dim") != STATE_DIM_V3_8:
            raise SystemExit(
                f"FAIL[7a]: feature_schema.state_dim={fs.get('state_dim')}, "
                f"expected {STATE_DIM_V3_8}"
            )
        if fs.get("state_feature_schema_version") != STATE_FEATURE_SCHEMA_VERSION_V3_8:
            raise SystemExit(
                f"FAIL[7b]: feature_schema.state_feature_schema_version="
                f"{fs.get('state_feature_schema_version')}, expected "
                f"{STATE_FEATURE_SCHEMA_VERSION_V3_8}"
            )
        if fs.get("action_feature_schema_version") != ACTION_FEATURE_SCHEMA_VERSION:
            raise SystemExit(
                f"FAIL[7c]: feature_schema.action_feature_schema_version="
                f"{fs.get('action_feature_schema_version')}, expected "
                f"{ACTION_FEATURE_SCHEMA_VERSION}"
            )
        if fs.get("action_dim") != _TARGET_ACTION_DIM:
            raise SystemExit(
                f"FAIL[7d]: feature_schema.action_dim={fs.get('action_dim')}, "
                f"expected {_TARGET_ACTION_DIM}"
            )
        if fs.get("action_dim") != ACTION_DIM:
            raise SystemExit(
                f"FAIL[7e]: feature_schema.action_dim={fs.get('action_dim')} "
                f"!= runtime ACTION_DIM={ACTION_DIM}"
            )

        # 8. ModelConfig sanity — including the NO-trunk-widening guard.
        target_cfg = payload.get("model_config") or {}
        if target_cfg.get("state_dim") != STATE_DIM_V3_8:
            raise SystemExit(
                f"FAIL[8a]: model_config.state_dim={target_cfg.get('state_dim')}"
            )
        if target_cfg.get("action_dim") != _TARGET_ACTION_DIM:
            raise SystemExit(
                f"FAIL[8b]: model_config.action_dim={target_cfg.get('action_dim')}"
            )
        if target_cfg.get("uses_uma_slot_tokens", True):
            raise SystemExit(
                "FAIL[8c]: v5 ckpt must have uses_uma_slot_tokens=False"
            )
        if target_cfg.get("hidden_dim") != 128:
            raise SystemExit(
                f"FAIL[8d-trunk-guard]: hidden_dim="
                f"{target_cfg.get('hidden_dim')}, expected 128. v3.8 §13.3 "
                f"contraindication still applies to v5."
            )
        if target_cfg.get("depth") != 2:
            raise SystemExit(
                f"FAIL[8e-trunk-guard]: depth={target_cfg.get('depth')}, "
                f"expected 2"
            )

    print(
        f"v5_action_init_smoke: 8/8 cases PASS "
        f"(source_kind={source_kind}, delta_logits={delta:.3e}, "
        f"tolerance={PARITY_TOLERANCE:.0e})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
