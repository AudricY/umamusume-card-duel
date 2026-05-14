from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import ACTION_DIM, STATE_DIM, legal_actions_to_features, observation_to_features

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


class JsonlPolicyDataset(Dataset[PolicySample]):
    def __init__(self, path: str | Path, *, min_actions: int = 2, ablations: set[str] | None = None, strict_schema_version: bool = True) -> None:
        self.path = Path(path)
        self.ablations = ablations or set()
        self.samples = list(load_policy_samples(self.path, min_actions=min_actions, ablations=self.ablations, strict_schema_version=strict_schema_version))
        if not self.samples:
            raise ValueError(f"No usable policy samples found in {self.path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PolicySample:
        return self.samples[index]


def load_policy_samples(path: str | Path, *, min_actions: int = 2, ablations: set[str] | None = None, strict_schema_version: bool = True) -> Iterable[PolicySample]:
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
            state_features = observation_to_features(example.get("observation", {}), ablations=ablations)
            action_features = legal_actions_to_features(actions, ablations=ablations)
            if state_features.shape != (STATE_DIM,):
                raise ValueError(f"Bad state feature shape at line {line_number}: {state_features.shape}")
            if action_features.shape[1:] != (ACTION_DIM,):
                raise ValueError(f"Bad action feature shape at line {line_number}: {action_features.shape}")
            policy_target = _policy_target(example, num_actions=len(actions), line_number=line_number)
            yield PolicySample(
                state_features=state_features,
                action_features=action_features,
                target_index=target_index,
                value_target=_value_target(example),
                sample_weight=_sample_weight(example),
                example=example,
                policy_target=policy_target,
            )


def collate_policy_batch(samples: list[PolicySample]) -> dict[str, torch.Tensor]:
    batch_size = len(samples)
    max_actions = max(sample.action_features.shape[0] for sample in samples)

    state_features = np.zeros((batch_size, STATE_DIM), dtype=np.float32)
    action_features = np.zeros((batch_size, max_actions, ACTION_DIM), dtype=np.float32)
    action_mask = np.zeros((batch_size, max_actions), dtype=np.bool_)
    targets = np.zeros((batch_size,), dtype=np.int64)
    value_targets = np.zeros((batch_size,), dtype=np.float32)
    sample_weights = np.ones((batch_size,), dtype=np.float32)

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
