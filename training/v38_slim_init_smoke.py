"""Smoke `training/make_v38_slim_init.py` end-to-end.

Builds a v3.8 init from a v3.7 source ckpt and asserts:
  1. Widened state_encoder.0.weight has shape (hidden, 304).
  2. Appended state tail columns at [296:304] are exactly zero.
  3. Preserved state columns at [0:296] are byte-identical to the v3.7
     source weight columns at the same indices.
  4. Widened action input projection has shape (out_dim, 52).
  5. Appended action tail columns at [48:52] are exactly zero.
  6. Preserved action columns at [0:48] are byte-identical to the v3.7
     source weight.
  7. All other parameters are byte-identical to the source.
  8. delta_logits from the builder's internal parity check is below the
     v3.8 iter-0 tolerance (1e-5, strict — same band as v3.7 since v3.8
     is a pure-zero-init append on BOTH state and action with no
     column-drop).
  9. Dest ckpt feature_schema tagged state_dim=304,
     state_feature_schema_version=3.8, action_feature_schema_version=4,
     action_dim=52.
  10. Dest ckpt model_config records state_dim=304, action_dim=52,
      uses_uma_slot_tokens=False, hidden_dim=128, depth=2 (NO trunk
      widening — v3.8 §13.3 audit contraindicates it).

Source preference order:
  (a) --source CLI argument if provided.
  (b) The v3.7 cap128 iter-6 lineage best
      (`runs/R16-P1-v37-cap128-A1/loop/iter-6/checkpoint.pt`) — the
      §4n-anchored ckpt the v3.8 init parity contract is measured
      against.
  (c) Synthesise a minimal random-weight v3.7-shaped (hidden=128,
      depth=2) ckpt in tempdir. CLAUDE.md treats missing `runs/`
      ckpts as environment gaps, not product passes — so the smoke
      degrades gracefully rather than failing on a missing artifact.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_v38_slim_init import (  # noqa: E402
    PARITY_TOLERANCE,
    STATE_WIDENED_KEY,
    _find_action_input_projection_key,
    build,
)
from uma_ai.features import (  # noqa: E402
    ACTION_FEATURE_SCHEMA_VERSION,
    STATE_DIM_V3_7,
    STATE_DIM_V3_8,
    STATE_FEATURE_SCHEMA_VERSION_V3_8,
)
from uma_ai.model import CandidatePolicyNet, ModelConfig  # noqa: E402

DEFAULT_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "runs/R16-P1-v37-cap128-A1/loop/iter-6/checkpoint.pt"
)

_EXPECTED_SOURCE_ACTION_DIM = 48
_TARGET_ACTION_DIM = 52


def _fabricate_v37_ckpt(dest: Path) -> Path:
    """Create a minimal random-weight v3.7-shaped (cap128) checkpoint
    at `dest`. Matches the payload shape `make_v38_slim_init.build`
    expects: `model_state`, `model_config`, `feature_schema`. Uses the
    cap128/depth=2 trunk that v3.8 doctrine REQUIRES — fabrication
    cannot use a non-default trunk or the init builder will refuse.
    """

    torch.manual_seed(20260526)
    config = ModelConfig(
        state_dim=STATE_DIM_V3_7,
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
        "state_dim": STATE_DIM_V3_7,
        "action_dim": _EXPECTED_SOURCE_ACTION_DIM,
        "state_feature_schema_version": 3.7,
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
    if cli_source:
        p = Path(cli_source)
        if not p.exists():
            raise SystemExit(f"FAIL: --source {p} does not exist")
        return p, "cli"
    if DEFAULT_SOURCE.exists():
        return DEFAULT_SOURCE, "real-v37-cap128-iter6"
    fabricated = tmpdir / "fabricated_v37.pt"
    _fabricate_v37_ckpt(fabricated)
    return fabricated, "fabricated-random-v37-cap128"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp_str:
        tmpdir = Path(tmp_str)
        source, source_kind = _resolve_source(args.source, tmpdir)
        dest = tmpdir / "v38_init.pt"
        provenance = build(source, dest)

        # 1. Shape provenance contract.
        if provenance["dest_state_dim"] != STATE_DIM_V3_8:
            raise SystemExit(
                f"FAIL[1]: dest_state_dim={provenance['dest_state_dim']}, "
                f"expected {STATE_DIM_V3_8}"
            )
        if provenance["dest_action_dim"] != _TARGET_ACTION_DIM:
            raise SystemExit(
                f"FAIL[1b]: dest_action_dim={provenance['dest_action_dim']}, "
                f"expected {_TARGET_ACTION_DIM}"
            )

        # 8. Δlogits tolerance.
        delta = provenance["delta_logits_at_init"]
        if not (delta < PARITY_TOLERANCE):
            raise SystemExit(
                f"FAIL[8]: delta_logits_at_init={delta:.3e} "
                f">= {PARITY_TOLERANCE:.0e} tolerance"
            )

        # Re-load dest and inspect tensors directly.
        payload = torch.load(dest, map_location="cpu", weights_only=False)
        state = payload["model_state"]
        widened_state = state[STATE_WIDENED_KEY]
        if widened_state.shape[-1] != STATE_DIM_V3_8:
            raise SystemExit(
                f"FAIL[1c]: widened state last dim="
                f"{widened_state.shape[-1]}, expected {STATE_DIM_V3_8}"
            )

        # 2. Appended state tail [296:304] is zero.
        appended_state = widened_state[:, STATE_DIM_V3_7:STATE_DIM_V3_8]
        if not torch.equal(appended_state, torch.zeros_like(appended_state)):
            raise SystemExit(
                f"FAIL[2]: appended state tail "
                f"[{STATE_DIM_V3_7}:{STATE_DIM_V3_8}] "
                f"({appended_state.shape[-1]} columns) not all zero"
            )

        # 3. Preserved state columns [0:296] byte-identical to source.
        source_payload = torch.load(source, map_location="cpu", weights_only=False)
        source_state = source_payload["model_state"]
        source_state_weight = source_state[STATE_WIDENED_KEY]
        if not torch.equal(widened_state[:, 0:STATE_DIM_V3_7], source_state_weight):
            raise SystemExit(
                f"FAIL[3]: widened_state[:, 0:{STATE_DIM_V3_7}] differs "
                f"from source v3.7 weight"
            )

        # 4-6. Action widening checks.
        action_widened_key = _find_action_input_projection_key(source_state)
        widened_action = state[action_widened_key]
        if widened_action.shape[-1] != _TARGET_ACTION_DIM:
            raise SystemExit(
                f"FAIL[4]: widened action `{action_widened_key}` last dim="
                f"{widened_action.shape[-1]}, expected {_TARGET_ACTION_DIM}"
            )
        appended_action = widened_action[:, _EXPECTED_SOURCE_ACTION_DIM:_TARGET_ACTION_DIM]
        if not torch.equal(appended_action, torch.zeros_like(appended_action)):
            raise SystemExit(
                f"FAIL[5]: appended action tail "
                f"[{_EXPECTED_SOURCE_ACTION_DIM}:{_TARGET_ACTION_DIM}] "
                f"({appended_action.shape[-1]} columns) not all zero"
            )
        source_action_weight = source_state[action_widened_key]
        if not torch.equal(
            widened_action[:, 0:_EXPECTED_SOURCE_ACTION_DIM], source_action_weight
        ):
            raise SystemExit(
                f"FAIL[6]: widened_action[:, 0:{_EXPECTED_SOURCE_ACTION_DIM}] "
                f"differs from source v3.7 weight"
            )

        # 7. All other parameters byte-identical to source.
        for key, value in source_state.items():
            if key in (STATE_WIDENED_KEY, action_widened_key):
                continue
            if not torch.equal(state[key], value):
                raise SystemExit(f"FAIL[7]: param `{key}` differs from source")

        # 9. Schema metadata.
        fs = payload.get("feature_schema") or {}
        if fs.get("state_dim") != STATE_DIM_V3_8:
            raise SystemExit(
                f"FAIL[9a]: feature_schema.state_dim={fs.get('state_dim')}"
            )
        if fs.get("state_feature_schema_version") != STATE_FEATURE_SCHEMA_VERSION_V3_8:
            raise SystemExit(
                f"FAIL[9b]: feature_schema.state_feature_schema_version="
                f"{fs.get('state_feature_schema_version')}"
            )
        if fs.get("action_feature_schema_version") != ACTION_FEATURE_SCHEMA_VERSION:
            raise SystemExit(
                f"FAIL[9c]: feature_schema.action_feature_schema_version="
                f"{fs.get('action_feature_schema_version')}, expected "
                f"{ACTION_FEATURE_SCHEMA_VERSION}"
            )
        if fs.get("action_dim") != _TARGET_ACTION_DIM:
            raise SystemExit(
                f"FAIL[9d]: feature_schema.action_dim={fs.get('action_dim')}"
            )

        # 10. ModelConfig sanity — including the NO-trunk-widening guard.
        target_cfg = payload.get("model_config") or {}
        if target_cfg.get("state_dim") != STATE_DIM_V3_8:
            raise SystemExit(
                f"FAIL[10a]: model_config.state_dim={target_cfg.get('state_dim')}"
            )
        if target_cfg.get("action_dim") != _TARGET_ACTION_DIM:
            raise SystemExit(
                f"FAIL[10b]: model_config.action_dim={target_cfg.get('action_dim')}"
            )
        if target_cfg.get("uses_uma_slot_tokens", True):
            raise SystemExit(
                "FAIL[10c]: v3.8 ckpt must have uses_uma_slot_tokens=False"
            )
        if target_cfg.get("hidden_dim") != 128:
            raise SystemExit(
                f"FAIL[10d-trunk-guard]: hidden_dim="
                f"{target_cfg.get('hidden_dim')}, expected 128. v3.8 §13.3 "
                f"audit FORBIDS trunk widening — overfitting evidence "
                f"contraindicates h128/d2 → h256/d4."
            )
        if target_cfg.get("depth") != 2:
            raise SystemExit(
                f"FAIL[10e-trunk-guard]: depth={target_cfg.get('depth')}, "
                f"expected 2"
            )

    print(
        f"v38_slim_init_smoke: 10/10 cases PASS "
        f"(source_kind={source_kind}, delta_logits={delta:.3e}, "
        f"tolerance={PARITY_TOLERANCE:.0e})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
