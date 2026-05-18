from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import (
    ACTION_DIM,
    CARD_ID_SHAPES,
    STATE_DIM,
    ZONE_ORDER,
    action_card_idx_pair,
    feature_builder_for_state_dim,
    legal_actions_to_features,
    observation_to_card_ids,
)

# R7.b.2 Phase 2 NOTE on schema versions:
# There are TWO schema-version axes in this codebase. They are independent.
#   - `ROW_SCHEMA_VERSION` (this constant): the TS row writer's
#     `TrainingExample.schemaVersion` (and `relabelDecisionTrace.ts`'s
#     relabel-row `schemaVersion`). Set by the TS pipeline at row-write
#     time.
#   - `STATE_FEATURE_SCHEMA_VERSION` (in `features.py`): the Python
#     state-encoder version. Bumped 2.1 → 3.0 by Phase 2 to mark the
#     observation_to_card_ids contract going live.
#
# Phase 2 chooses NOT to bump `ROW_SCHEMA_VERSION` past 1 even though
# scoping § 11 Phase 2 anticipated a bump to 3. Rationale: the TS-side
# writers still emit `schemaVersion: 1`, and Phase 2's brief constrains
# us to "no backend/frontend TS changes" — bumping the Python constant
# alone would reject every existing TS-produced corpus (incl. the python
# smoke), with no upside. The fail-loud requirement (reject pre-Phase-1
# corpora that lack `cardIdsByZone`) is enforced inside
# `observation_to_card_ids` itself, which is the substantive guard.
# Phase 4's re-extractor (separate slot) will land the row-level bump
# alongside the TS row-writer bump.
ROW_SCHEMA_VERSION = 1


class RowSchemaError(ValueError):
    """Raised when a JSONL row's schemaVersion is missing or incompatible."""


@dataclass(frozen=True)
class PolicySample:
    state_features: np.ndarray
    action_features: np.ndarray
    target_index: int
    value_target: float
    sample_weight: float
    example: dict[str, Any]
    # R7 step 3: per-state mixture target from `relabelDecisionTrace.ts`. When
    # present, shape is (num_actions,) with mass on each teacher's chosen
    # action; otherwise None and the trainer takes the hard-CE branch on
    # `target_index`. Mirrors the same field on `MctsSelfPlaySample` so
    # collation can emit a padded `policy_targets` tensor uniformly.
    policy_target: np.ndarray | None = None
    # R7.b.2 Phase 2: per-zone packed card-vocab indices (8 zones, fixed
    # `CARD_ID_SHAPES`) and per-action source/target idx (shape `(A, 2)`).
    # Reuses the `policy_target | None` pattern: legacy / pre-Phase-1
    # corpora that lack `cardIdsByZone` raise in `observation_to_card_ids`
    # at load time, so any loaded sample is guaranteed to carry both
    # fields. `None` is kept as the type-level nullable to mirror legacy
    # PolicySamples that may be constructed by tests with the embedding
    # branch disabled.
    card_ids_by_zone: dict[str, np.ndarray] | None = None
    action_card_idx: np.ndarray | None = None


class JsonlPolicyDataset(Dataset[PolicySample]):
    def __init__(
        self,
        path: str | Path,
        *,
        min_actions: int = 2,
        ablations: set[str] | None = None,
        strict_schema_version: bool = True,
        state_dim: int = STATE_DIM,
    ) -> None:
        # `state_dim` selects the frozen builder (96=v2, 110=v3.0, 164=v3.1)
        # via the same dim-keyed mechanism serve_onnx uses. Default is the
        # module STATE_DIM (110 = v3.0) so existing callers are byte-stable;
        # v3.1 training opts in by passing state_dim=STATE_DIM_V3_1.
        self.path = Path(path)
        self.ablations = ablations or set()
        self.samples = list(
            load_policy_samples(
                self.path,
                min_actions=min_actions,
                ablations=self.ablations,
                strict_schema_version=strict_schema_version,
                state_dim=state_dim,
            )
        )
        if not self.samples:
            raise ValueError(f"No usable policy samples found in {self.path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PolicySample:
        return self.samples[index]


def load_policy_samples(
    path: str | Path,
    *,
    min_actions: int = 2,
    ablations: set[str] | None = None,
    strict_schema_version: bool = True,
    state_dim: int = STATE_DIM,
) -> Iterable[PolicySample]:
    encode_state = feature_builder_for_state_dim(state_dim)
    with Path(path).open("r", encoding="utf8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            example = json.loads(line)
            if strict_schema_version:
                version = example.get("schemaVersion")
                if version is None:
                    raise RowSchemaError(
                        f"Missing schemaVersion at {path}:{line_number}; expected {ROW_SCHEMA_VERSION}"
                    )
                if int(version) != ROW_SCHEMA_VERSION:
                    raise RowSchemaError(
                        f"Incompatible schemaVersion at {path}:{line_number}: got {version}, expected {ROW_SCHEMA_VERSION}"
                    )
            actions = example.get("legalActions", [])
            target_index = int(example.get("selectedActionIndex", -1))
            if len(actions) < min_actions or target_index < 0 or target_index >= len(actions):
                continue
            observation = example.get("observation", {})
            state_features = encode_state(observation, ablations=ablations)
            action_features = legal_actions_to_features(actions, ablations=ablations)
            if state_features.shape != (state_dim,):
                raise ValueError(f"Bad state feature shape at line {line_number}: {state_features.shape}")
            if action_features.shape[1:] != (ACTION_DIM,):
                raise ValueError(f"Bad action feature shape at line {line_number}: {action_features.shape}")
            policy_target = _policy_target(example, num_actions=len(actions), line_number=line_number)
            # R7.b.2 Phase 2: emit packed per-zone card-id arrays and
            # per-action source/target idx. `observation_to_card_ids`
            # raises on missing `cardIdsByZone` so pre-Phase-1 corpora
            # surface at load time rather than as silent zero-tensors.
            try:
                card_ids_by_zone = observation_to_card_ids(observation)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            action_card_idx = np.stack([action_card_idx_pair(action) for action in actions], axis=0)
            yield PolicySample(
                state_features=state_features,
                action_features=action_features,
                target_index=target_index,
                value_target=_value_target(example),
                sample_weight=_sample_weight(example),
                example=example,
                policy_target=policy_target,
                card_ids_by_zone=card_ids_by_zone,
                action_card_idx=action_card_idx,
            )


def collate_policy_batch(samples: list[PolicySample]) -> dict[str, torch.Tensor]:
    batch_size = len(samples)
    max_actions = max(sample.action_features.shape[0] for sample in samples)
    # R7.b.2 Phase 2: pack per-zone card-id arrays into a single 3-D
    # `LongTensor[B, NUM_ZONES, max_cards_per_zone]` for `nn.Embedding`
    # gather. `max_cards_per_zone` is the global max across zones (30, the
    # discard cap) so the smaller-cap zones (active=1, bench=4, hand=10)
    # see padding beyond their cap; the embedding's `padding_idx=0`
    # zeroes those positions in the gather, and the `(ids != 0)` mask in
    # `model.forward` zeroes them again in the pool — defense in depth.
    max_cards_per_zone = max(CARD_ID_SHAPES.values())
    num_zones = len(ZONE_ORDER)

    # R16-P1: infer the state dim from the samples (96/110/164) rather than
    # the module STATE_DIM constant so a v3.1 (164-d) dataset collates
    # correctly without a new mechanism. All samples in a batch share a
    # builder (one loader, one schema); assert that to fail loud on a mixed
    # batch instead of silently truncating.
    state_dim = samples[0].state_features.shape[0]
    if any(s.state_features.shape[0] != state_dim for s in samples):
        raise ValueError(
            "collate_policy_batch: mixed state-feature dims in one batch "
            f"(first={state_dim}); all rows must share one feature schema."
        )
    state_features = np.zeros((batch_size, state_dim), dtype=np.float32)
    action_features = np.zeros((batch_size, max_actions, ACTION_DIM), dtype=np.float32)
    action_mask = np.zeros((batch_size, max_actions), dtype=np.bool_)
    targets = np.zeros((batch_size,), dtype=np.int64)
    value_targets = np.zeros((batch_size,), dtype=np.float32)
    sample_weights = np.ones((batch_size,), dtype=np.float32)

    # R7.b.2 Phase 2: only emit the embedding tensors when every sample in
    # the batch has them. Same gating pattern as `policy_targets` below:
    # avoids silently zero-tensoring mixed batches (one v2 row + one v3
    # row), and the model.forward's `is None` branch fires uniformly.
    # In practice all rows in a batch come from the same loader and the
    # same schema; this guard is a defense-in-depth signal.
    # MctsSelfPlaySample does not carry these fields at all (mcts-distill data
    # path predates R7.b.2 schema bump). `getattr(..., None)` keeps the
    # cross-data-path collator non-crashing — the model's optional kwargs
    # default to zero-tensor (padding_idx=0, zone_projection bias=False), so
    # the embedding pathway is exactly inert for that path. Re-extracting
    # mcts-selfplay rows to v3 schema would re-enable it.
    all_have_card_ids = all(
        getattr(sample, "card_ids_by_zone", None) is not None
        and getattr(sample, "action_card_idx", None) is not None
        for sample in samples
    )
    card_ids_buffer = (
        np.zeros((batch_size, num_zones, max_cards_per_zone), dtype=np.int64)
        if all_have_card_ids
        else None
    )
    action_card_idx_buffer = (
        np.zeros((batch_size, max_actions, 2), dtype=np.int64)
        if all_have_card_ids
        else None
    )

    # R7 step 3: emit a padded `policy_targets` tensor only when *every*
    # sample in the batch carries a soft target. Mixed batches (some rows
    # from a single-teacher / legacy trace, some from a multi-teacher trace
    # via `mix-sources`) deliberately fall back to the hard-CE branch in
    # `train_bc.py:361-369`: the trainer's branch is per-batch
    # (`policy_targets is not None`), and synthesising a one-hot for legacy
    # rows would change the loss surface — the brief forbids that. The
    # parent corpus mixer can keep batches homogeneous by source-tagging.
    all_soft = all(sample.policy_target is not None for sample in samples)
    policy_targets = np.zeros((batch_size, max_actions), dtype=np.float32) if all_soft else None

    for row, sample in enumerate(samples):
        count = sample.action_features.shape[0]
        state_features[row] = sample.state_features
        action_features[row, :count] = sample.action_features
        action_mask[row, :count] = True
        targets[row] = sample.target_index
        value_targets[row] = sample.value_target
        sample_weights[row] = sample.sample_weight
        if policy_targets is not None:
            # Pad parallel to action_features (above) — same `:count` slice.
            policy_targets[row, :count] = sample.policy_target  # type: ignore[index]
        if card_ids_buffer is not None and action_card_idx_buffer is not None:
            # Per-zone fill: walk ZONE_ORDER so the packed axis matches the
            # zone enumeration the model.forward expects (NUM_ZONES dim).
            # Each zone's per-sample array has shape `(CARD_ID_SHAPES[zone],)`;
            # the global packed `max_cards_per_zone` is >= that, so we
            # write into the `:width` prefix and leave the rest as 0 (pad).
            for zone_index, zone in enumerate(ZONE_ORDER):
                zone_arr = sample.card_ids_by_zone[zone]  # type: ignore[index]
                width = zone_arr.shape[0]
                card_ids_buffer[row, zone_index, :width] = zone_arr
            # Action source/target idx: pad parallel to action_features.
            pair = sample.action_card_idx  # type: ignore[assignment]
            action_card_idx_buffer[row, :count, :] = pair

    batch: dict[str, torch.Tensor] = {
        "state_features": torch.from_numpy(state_features),
        "action_features": torch.from_numpy(action_features),
        "action_mask": torch.from_numpy(action_mask),
        "targets": torch.from_numpy(targets),
        "value_targets": torch.from_numpy(value_targets),
        "sample_weights": torch.from_numpy(sample_weights),
    }
    if policy_targets is not None:
        batch["policy_targets"] = torch.from_numpy(policy_targets)
    if card_ids_buffer is not None and action_card_idx_buffer is not None:
        batch["card_ids_by_zone"] = torch.from_numpy(card_ids_buffer)
        batch["action_card_idx"] = torch.from_numpy(action_card_idx_buffer)
    return batch


def _policy_target(example: dict[str, Any], *, num_actions: int, line_number: int) -> np.ndarray | None:
    """Parse the R7 multi-teacher mixture target (TS `policyTargets` →
    Python `policy_target`).

    Returns None for legacy rows that do not carry the field (the trainer
    will take the hard-CE path on `selectedActionIndex` instead — see
    `train_bc.py:361-369`). Raises on a present-but-malformed field so
    silent corpus corruption surfaces immediately rather than as a
    mysterious training-loss bug.
    """

    raw = example.get("policyTargets")
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError(f"policyTargets at line {line_number} must be a list, got {type(raw).__name__}")
    if len(raw) != num_actions:
        raise ValueError(
            f"policyTargets at line {line_number} has length {len(raw)} but legalActions has {num_actions}"
        )
    arr = np.asarray(raw, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"policyTargets at line {line_number} contains non-finite values: {raw}")
    if np.any(arr < 0):
        raise ValueError(f"policyTargets at line {line_number} contains negative values: {raw}")
    total = float(arr.sum())
    # `relabelDecisionTrace.ts` builds these as `1/K` per teacher with K
    # teachers, so the sum is exact for K in {1,2,3} under float64 but may
    # drift by ~1e-7 once we cast to float32. Tolerate that; reject anything
    # that signals a real bug upstream.
    if not np.isfinite(total) or abs(total - 1.0) > 1e-5:
        raise ValueError(
            f"policyTargets at line {line_number} must sum to 1.0 ± 1e-5; got {total}"
        )
    return arr


def _value_target(example: dict[str, Any]) -> float:
    winner = example.get("result", {}).get("winner")
    side_id = example.get("sideId")
    if winner is None or side_id not in ("player", "opponent"):
        target = example.get("valueTarget")
        return float(target) if target is not None else 0.0
    return 1.0 if winner == side_id else -1.0


def _sample_weight(example: dict[str, Any]) -> float:
    raw = example.get("sampleWeight", 1.0)
    try:
        return max(0.05, float(raw))
    except (TypeError, ValueError):
        return 1.0
