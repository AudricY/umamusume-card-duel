"""R16-P1: build a 164-d v3.1 additive-tail warm-start init checkpoint.

Methodology (do NOT substitute random init): take the 110-d v3.0 init the
baseline 0.6042 warm-started from and widen ONLY the scalar state-feature
input projection (`state_encoder.0.weight`, shape `[hidden, state_dim]`):

  - columns [0:110]  -> copied VERBATIM from the source weight,
  - columns [110:164] -> zero-initialized.

Every other parameter (including `state_encoder.0.bias`, the entire
card-embedding residual branch `card_embed` / `zone_projection`, the action
encoder, joint trunk, heads) is copied VERBATIM. `model_config.state_dim`
is set to 164 and the recorded `feature_schema` is re-stamped to v3.1.

Why this is exactly numerically null at init: the 164-d v3.1 feature vector
is `[ v3.0_110d | 54 temporal/turn-state ]` (frozen v3.0 head, by
construction in `observation_to_features_v3_1`). The first layer computes
`W @ x + b`. With the new columns W[:,110:164] == 0, the contribution of the
54 new features is `0 @ x[110:164] == 0`, so `W_164 @ x_164 + b ==
W_110 @ x_110 + b` exactly. The card-embedding residual is added AFTER the
state encoder from a separate, byte-identical branch, so it is unchanged.
Hence the 164-d net at init produces outputs numerically identical to the
110-d source on shared features — isolating the schema's learned lift.

Reusable + auditable: re-run to regenerate the init deterministically.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uma_ai.features import (  # noqa: E402
    STATE_DIM_V3,
    STATE_DIM_V3_1,
    schema_version_for_state_dim,
)

FIRST_LAYER_WEIGHT = "state_encoder.0.weight"


def build(source: Path, dest: Path) -> dict:
    payload = torch.load(source, map_location="cpu", weights_only=False)
    model_state = payload["model_state"]
    config = dict(payload.get("model_config") or {})
    src_dim = int(config.get("state_dim", STATE_DIM_V3))
    if src_dim != STATE_DIM_V3:
        raise SystemExit(
            f"source checkpoint state_dim={src_dim}, expected the frozen "
            f"v3.0 dim {STATE_DIM_V3}; this builder only widens a v3.0 init."
        )
    if FIRST_LAYER_WEIGHT not in model_state:
        raise SystemExit(
            f"source checkpoint has no `{FIRST_LAYER_WEIGHT}` param "
            f"(keys: {sorted(model_state)[:8]}...); model architecture "
            f"changed — re-derive the first-layer name before proceeding."
        )

    w = model_state[FIRST_LAYER_WEIGHT]
    hidden, in_dim = w.shape
    if in_dim != STATE_DIM_V3:
        raise SystemExit(
            f"`{FIRST_LAYER_WEIGHT}` input dim {in_dim} != frozen v3.0 dim "
            f"{STATE_DIM_V3}; refusing to guess the widening."
        )

    # Additive-tail-zero: copy [0:110] verbatim, zero [110:164].
    wide = torch.zeros((hidden, STATE_DIM_V3_1), dtype=w.dtype)
    wide[:, :STATE_DIM_V3] = w
    new_state = dict(model_state)
    new_state[FIRST_LAYER_WEIGHT] = wide

    config["state_dim"] = STATE_DIM_V3_1
    # Warm-start init only: drop optimizer/scheduler/scaler/rng/epoch so the
    # ablation loop treats this as fresh weights (matches the DAgger
    # --init-from-checkpoint primitive in train_bc.load_init_from_checkpoint,
    # which loads ONLY model_state but tolerates a richer payload).
    feature_schema = dict(payload.get("feature_schema") or {})
    feature_schema["state_dim"] = STATE_DIM_V3_1
    feature_schema["state_feature_schema_version"] = schema_version_for_state_dim(
        STATE_DIM_V3_1
    )

    new_payload = {
        "model_state": new_state,
        "model_config": config,
        "feature_schema": feature_schema,
        "provenance": {
            "method": "v3.1 additive-tail-zero warm-start",
            "source_checkpoint": str(source),
            "source_state_dim": STATE_DIM_V3,
            "dest_state_dim": STATE_DIM_V3_1,
            "first_layer_param": FIRST_LAYER_WEIGHT,
            "new_columns": [STATE_DIM_V3, STATE_DIM_V3_1],
            "new_columns_init": "zero",
            "all_other_params": "copied verbatim",
            "builder": "training/make_v31_additive_init.py",
        },
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_payload, dest)
    return new_payload["provenance"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="runs/R7b2-card-embed/iter-000/model/checkpoint.pt",
        help="110-d v3.0 init the v3.0 baseline 0.6042 warm-started from.",
    )
    parser.add_argument(
        "--dest",
        default="runs/R16-P1-v31-ablation/init-164/checkpoint.pt",
        help="Output 164-d v3.1 additive-tail-zero init checkpoint.",
    )
    args = parser.parse_args()
    provenance = build(Path(args.source), Path(args.dest))
    readme = Path(args.dest).parent / "README.md"
    readme.write_text(
        "# R16-P1 v3.1 ablation — 164-d additive-tail init\n\n"
        "`checkpoint.pt` is the 164-d v3.1 warm-start for the "
        "v3.1-vs-v3.0 strength ablation. It is the 110-d v3.0 init the "
        "v3.0 baseline (0.6042) warm-started from, widened by the "
        "additive-tail-zero method: `state_encoder.0.weight` columns "
        "[0:110] copied verbatim, columns [110:164] zeroed, every other "
        "parameter copied verbatim. At init it is numerically identical "
        "to the 110-d source on shared features (new tail contributes "
        "exactly zero), isolating the schema's learned contribution.\n\n"
        "Reproduce: `training/.venv/bin/python "
        "training/make_v31_additive_init.py`\n\n"
        "Provenance:\n```json\n"
        + json.dumps(provenance, indent=2)
        + "\n```\n",
        encoding="utf8",
    )
    print(json.dumps({"status": "PASS", "dest": args.dest, **provenance}, indent=2))


if __name__ == "__main__":
    main()
