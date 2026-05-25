"""Smoke `training/make_v36_priors_init.py` end-to-end.

Builds a v3.6 init from a v3.5 source ckpt and asserts:
  1. The widened `state_encoder.0.weight` has shape `(hidden, 246)`.
  2. The dropped+repurposed band at columns [197:207] is exactly zero
     (own.energy_pool typed multihot, zero-init).
  3. The appended tail columns at [212:246] are exactly zero (truly-new
     v3.6 channels, zero-init).
  4. The preserved columns at [0:197] and [207:212] are byte-identical to
     the v3.5 source weight columns at the same indices.
  5. All other parameters are byte-identical to the source.
  6. `delta_logits` from the builder's internal parity check is below the
     v3.6 iter-0 tolerance (1e-3, looser than v3.5's 1e-5; see scoping doc
     §3.2 for the justification).
  7. The dest ckpt's `feature_schema` is tagged state_dim=246,
     state_feature_schema_version=3.6.
  8. The dest ckpt's `model_config` records state_dim=246 and
     uses_uma_slot_tokens=False (v3.6 contract — slot tokens excluded).

Source preference order:
  (a) --source CLI argument if provided.
  (b) The R16-P1 v3.5-extended lineage best
      (`runs/R16-P1-v35-extended-ablation/loop/iter-0/checkpoint.pt`) —
      uses iter-0 for stability across rerolls; any iter-<N> works.
  (c) Synthesise a minimal random-weight v3.5-shaped ckpt in tempdir.
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

from make_v36_priors_init import (  # noqa: E402
    PARITY_TOLERANCE,
    WIDENED_KEY,
    _DROP_BAND_END,
    _DROP_BAND_START,
    _V35_TAIL_END,
    build,
)
from uma_ai.features import (  # noqa: E402
    STATE_DIM_V3_5,
    STATE_DIM_V3_6,
    STATE_FEATURE_SCHEMA_VERSION_V3_6,
)
from uma_ai.model import CandidatePolicyNet, ModelConfig  # noqa: E402

DEFAULT_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "runs/R16-P1-v35-extended-ablation/loop/iter-0/checkpoint.pt"
)


def _fabricate_v35_ckpt(dest: Path) -> Path:
    """Create a minimal random-weight v3.5-shaped checkpoint at `dest`.

    Matches the payload shape `make_v36_priors_init.build` expects:
    `model_state`, `model_config`, `feature_schema`. Uses the same compact
    `hidden_dim=64, depth=2` config that the R16-P1 v3.5 extended ablation
    rides, so the fabricated ckpt mirrors a real-world v3.5 training
    artifact in keyset and tensor shapes."""

    torch.manual_seed(20260525)
    config = ModelConfig(
        state_dim=STATE_DIM_V3_5,
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
        "state_dim": STATE_DIM_V3_5,
        "action_dim": 48,
        "state_feature_schema_version": 3.5,
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
    """Choose a v3.5 source ckpt path. Returns (path, kind)."""

    if cli_source:
        p = Path(cli_source)
        if not p.exists():
            raise SystemExit(f"FAIL: --source {p} does not exist")
        return p, "cli"
    if DEFAULT_SOURCE.exists():
        return DEFAULT_SOURCE, "real-v35-extended-iter-0"
    fabricated = tmpdir / "fabricated_v35.pt"
    _fabricate_v35_ckpt(fabricated)
    return fabricated, "fabricated-random-v35"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp_str:
        tmpdir = Path(tmp_str)
        source, source_kind = _resolve_source(args.source, tmpdir)
        dest = tmpdir / "v36_init.pt"
        provenance = build(source, dest)

        # 1. Shape provenance contract.
        if provenance["dest_state_dim"] != STATE_DIM_V3_6:
            raise SystemExit(
                f"FAIL[1]: dest_state_dim={provenance['dest_state_dim']}, "
                f"expected {STATE_DIM_V3_6}"
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
        widened = state[WIDENED_KEY]
        if widened.shape[-1] != STATE_DIM_V3_6:
            raise SystemExit(
                f"FAIL[1b]: widened last dim={widened.shape[-1]}, "
                f"expected {STATE_DIM_V3_6}"
            )

        # 2. Dropped+repurposed band [197:207] is zero.
        repurposed = widened[:, _DROP_BAND_START:_DROP_BAND_END]
        if not torch.equal(repurposed, torch.zeros_like(repurposed)):
            raise SystemExit(
                f"FAIL[2]: repurposed band [{_DROP_BAND_START}:{_DROP_BAND_END}] "
                f"({repurposed.shape[-1]} columns) not all zero"
            )

        # 3. Appended tail [212:246] is zero.
        appended = widened[:, _V35_TAIL_END:STATE_DIM_V3_6]
        if not torch.equal(appended, torch.zeros_like(appended)):
            raise SystemExit(
                f"FAIL[3]: appended tail [{_V35_TAIL_END}:{STATE_DIM_V3_6}] "
                f"({appended.shape[-1]} columns) not all zero"
            )

        # 4. Preserved columns [0:197] + [207:212] byte-identical to source.
        source_payload = torch.load(source, map_location="cpu", weights_only=False)
        source_state = source_payload["model_state"]
        source_widened = source_state[WIDENED_KEY]
        if not torch.equal(
            widened[:, 0:_DROP_BAND_START], source_widened[:, 0:_DROP_BAND_START]
        ):
            raise SystemExit(
                f"FAIL[4a]: widened[:, 0:{_DROP_BAND_START}] differs from source"
            )
        if not torch.equal(
            widened[:, _DROP_BAND_END:_V35_TAIL_END],
            source_widened[:, _DROP_BAND_END:_V35_TAIL_END],
        ):
            raise SystemExit(
                f"FAIL[4b]: widened[:, {_DROP_BAND_END}:{_V35_TAIL_END}] "
                f"differs from source"
            )

        # 5. All other parameters byte-identical to source.
        for key, value in source_state.items():
            if key == WIDENED_KEY:
                continue
            if not torch.equal(state[key], value):
                raise SystemExit(f"FAIL[5]: param `{key}` differs from source")

        # 7. Schema metadata.
        fs = payload.get("feature_schema") or {}
        if fs.get("state_dim") != STATE_DIM_V3_6:
            raise SystemExit(
                f"FAIL[7a]: feature_schema.state_dim={fs.get('state_dim')}"
            )
        if fs.get("state_feature_schema_version") != STATE_FEATURE_SCHEMA_VERSION_V3_6:
            raise SystemExit(
                f"FAIL[7b]: feature_schema.state_feature_schema_version="
                f"{fs.get('state_feature_schema_version')}"
            )

        # 8. ModelConfig sanity.
        target_cfg = payload.get("model_config") or {}
        if target_cfg.get("state_dim") != STATE_DIM_V3_6:
            raise SystemExit(
                f"FAIL[8a]: model_config.state_dim={target_cfg.get('state_dim')}"
            )
        if target_cfg.get("uses_uma_slot_tokens", True):
            raise SystemExit(
                "FAIL[8b]: v3.6 ckpt must have uses_uma_slot_tokens=False"
            )

    print(
        f"v36_tail_init_smoke: 8/8 cases PASS "
        f"(source_kind={source_kind}, delta_logits={delta:.3e}, "
        f"tolerance={PARITY_TOLERANCE:.0e})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
