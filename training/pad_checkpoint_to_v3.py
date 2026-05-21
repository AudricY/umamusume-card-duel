"""R16-TD 3b chunk 5a — one-shot helper to pad a 96-d production checkpoint
to a v3-compatible (`state_dim=110`) checkpoint usable by `train_dpo.py`.

Why this exists:
  `train_dpo.load_reference_policy` does a strict `load_state_dict`, so the
  96-d R13-W6-phase-d/iter-2 reference (which predates the v3 embedding
  branch AND the v3.0 state-feature widening) cannot be loaded directly.

  `dpo_quality_eval.load_checkpoint_compat` (introduced in commit f6288a1)
  solves this for the eval path by zero-padding the missing v3 params at
  load time AND by rebuilding the batch's `state_features` tensor at the
  source `state_dim=96` via the v2 builder. The trainer has no such hook —
  it consumes whatever `collate_preference_batch` emits (state_dim=110, v3
  builder).

  This helper resolves the gap by pre-padding the source checkpoint so
  that BOTH (a) the v3 embedding params exist (zero-init, `padding_idx=0`
  semantics keep the embed residual structurally null) AND (b)
  `state_encoder.0.weight` is widened from `(hidden, 96)` to
  `(hidden, 110)` by zero-padding columns 96..110. The widening is
  numerically null because slots 0..95 of the v3.0 builder are
  byte-identical to the v2 builder (see `uma_ai/features.py`:122-143 vs
  199-220 — the v3.0 builder reuses the same helpers for slots 0–95 and
  appends 14 new slots [96:110]; the new slots are multiplied by zero
  columns and contribute nothing).

  After padding, the model is byte-numerically equivalent to the original
  96-d encoder when fed v3.0 features, so the saved checkpoint preserves
  the baseline floor measured by `dpo_quality_eval.py`.

CLI:
    python training/pad_checkpoint_to_v3.py \
        --in runs/R13-W6-phase-d/iter-2/checkpoint.pt \
        --out runs/R16-TD-3a-prod-corpus/r13-w6-phasd-iter2-padded.pt

The output file is a drop-in replacement for `--reference-checkpoint` and
`--init-from-checkpoint` in `train_dpo.py` at `--hidden-dim 64 --depth 2`
(matching the source `model_config`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Make the training package importable when invoked as a script.
_REPO_TRAINING = Path(__file__).resolve().parent
if str(_REPO_TRAINING) not in sys.path:
    sys.path.insert(0, str(_REPO_TRAINING))

from uma_ai.model import CARD_EMBED_DIM, CandidatePolicyNet, ModelConfig
from uma_ai.features import STATE_DIM_V3


_V3_EMBEDDING_PARAMS = ("card_embed.weight", "zone_projection.weight")
_JOINT_PROJECTION_KEY = "joint_projection.0.weight"
_STATE_ENCODER_KEY = "state_encoder.0.weight"


def pad_state_dict(
    source_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
    *,
    source_state_dim: int,
    target_state_dim: int,
) -> tuple[dict[str, torch.Tensor], dict[str, list[str]]]:
    """Zero-pad a pre-v3 state_dict into a v3-compatible shape.

    Steps:
      1. Add zero-init `card_embed.weight` / `zone_projection.weight` when
         absent. `padding_idx=0` semantics make row 0 a gradient-free zero;
         the zone-pool projection emits zero regardless of input.
      2. Zero-pad `joint_projection.0.weight` input columns [3*hidden :
         3*hidden + 2*CARD_EMBED_DIM] (the new action-pair-embed tail).
      3. Zero-pad `state_encoder.0.weight` input columns [96:110]. v3.0
         slots [96:110] are the v2.1 hygiene additions; zero columns make
         the model ignore them, preserving v2 semantics under the v3
         builder (slots 0..95 of v3.0 are byte-identical to v2).
    """
    raw = dict(source_state)
    report: dict[str, list[str]] = {
        "padded_v3_params": [],
        "widened_joint_projection": [],
        "widened_state_encoder": [],
    }

    # Step 1.
    for key in _V3_EMBEDDING_PARAMS:
        if key not in raw and key in target_state:
            raw[key] = torch.zeros_like(target_state[key])
            report["padded_v3_params"].append(key)

    # Step 2.
    if _JOINT_PROJECTION_KEY in raw and _JOINT_PROJECTION_KEY in target_state:
        src = raw[_JOINT_PROJECTION_KEY]
        tgt_shape = target_state[_JOINT_PROJECTION_KEY].shape
        if src.shape != tgt_shape:
            if (
                src.shape[0] == tgt_shape[0]
                and tgt_shape[1] - src.shape[1] == 2 * CARD_EMBED_DIM
            ):
                widened = torch.zeros(tgt_shape, dtype=src.dtype)
                widened[:, : src.shape[1]] = src
                raw[_JOINT_PROJECTION_KEY] = widened
                report["widened_joint_projection"].append(_JOINT_PROJECTION_KEY)
            else:
                raise RuntimeError(
                    f"Cannot reconcile `{_JOINT_PROJECTION_KEY}` shape "
                    f"{tuple(src.shape)} with target {tuple(tgt_shape)}."
                )

    # Step 3.
    if _STATE_ENCODER_KEY in raw and _STATE_ENCODER_KEY in target_state:
        src = raw[_STATE_ENCODER_KEY]
        tgt_shape = target_state[_STATE_ENCODER_KEY].shape
        if src.shape != tgt_shape:
            if (
                src.shape[0] == tgt_shape[0]
                and src.shape[1] == source_state_dim
                and tgt_shape[1] == target_state_dim
                and target_state_dim >= source_state_dim
            ):
                widened = torch.zeros(tgt_shape, dtype=src.dtype)
                widened[:, : src.shape[1]] = src
                raw[_STATE_ENCODER_KEY] = widened
                report["widened_state_encoder"].append(_STATE_ENCODER_KEY)
            else:
                raise RuntimeError(
                    f"Cannot reconcile `{_STATE_ENCODER_KEY}` shape "
                    f"{tuple(src.shape)} with target {tuple(tgt_shape)} "
                    f"(source_state_dim={source_state_dim}, "
                    f"target_state_dim={target_state_dim})."
                )

    return raw, report


def pad_checkpoint(in_path: Path, out_path: Path, *, target_state_dim: int) -> dict:
    payload = torch.load(in_path, map_location="cpu", weights_only=False)
    source_config_dict = dict(payload.get("model_config") or {})
    if not source_config_dict:
        raise RuntimeError(f"checkpoint {in_path} has no model_config; cannot infer dims.")
    source_state_dim = int(source_config_dict.get("state_dim", target_state_dim))

    # Build the target model at the widened state_dim, keeping every other
    # dim from the source config so a hidden_dim=64/depth=2 checkpoint
    # produces a hidden_dim=64/depth=2 padded checkpoint.
    target_config_dict = dict(source_config_dict)
    target_config_dict["state_dim"] = target_state_dim
    target_config = ModelConfig.from_dict(target_config_dict)
    target_model = CandidatePolicyNet(target_config)
    target_state = target_model.state_dict()

    padded_state, report = pad_state_dict(
        payload["model_state"],
        target_state,
        source_state_dim=source_state_dim,
        target_state_dim=target_state_dim,
    )

    # Sanity check: a non-strict load should leave both lists empty.
    missing, unexpected = target_model.load_state_dict(padded_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"padded state still has state-dict deltas: "
            f"missing={missing} unexpected={unexpected}"
        )

    out_payload = dict(payload)
    out_payload["model_state"] = padded_state
    out_payload["model_config"] = target_config.to_dict()
    out_payload["pad_report"] = {
        "source_state_dim": source_state_dim,
        "target_state_dim": target_state_dim,
        "source_path": str(in_path),
        **report,
    }
    # Drop optimizer/rng state — they are state-dim-coupled in some
    # layouts (Adam moment buffers shaped to the old encoder) and we are
    # writing a fresh "init checkpoint" not a resume point.
    out_payload.pop("optimizer_state", None)
    out_payload.pop("scheduler_state", None)
    out_payload.pop("scaler_state", None)
    out_payload.pop("rng_state", None)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_payload, out_path)
    return out_payload["pad_report"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pad a pre-v3 96-d checkpoint to a v3-compatible (state_dim=110) checkpoint."
    )
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    parser.add_argument(
        "--target-state-dim",
        type=int,
        default=STATE_DIM_V3,
        help=f"Target state_dim for the padded model; default {STATE_DIM_V3} (v3.0).",
    )
    args = parser.parse_args()

    report = pad_checkpoint(
        Path(args.in_path), Path(args.out_path), target_state_dim=args.target_state_dim
    )
    print("padded:")
    for key, value in report.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
