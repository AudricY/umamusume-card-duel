"""MCTS self-play dataset for R12 distillation.

Rows are produced by `backend/src/sim/mctsSelfPlay.ts`. Schema is the
shared schemaVersion=1 plus `kind == "mcts-selfplay"`, and the distillation
target is `visitDistribution` (a soft distribution over `legalActions`).

The adapter mirrors `JsonlPolicyDataset` so collation is uniform with the
existing pipeline: same `state_features`, `action_features`, `action_mask`,
`value_targets`, `sample_weights`. The new field is `policy_targets`, a
(B, max_actions) tensor with row-sums equal to 1.

The `targets` column is the argmax of `visitDistribution` so the existing
`accuracy` accumulator (and any debugger that still uses hard labels) stays
defined; the *training* loss when `--data-mode mcts-distill` is the soft
cross-entropy against `policy_targets`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import ROW_SCHEMA_VERSION, RowSchemaError
from .features import ACTION_DIM, STATE_DIM, legal_actions_to_features, observation_to_features


@dataclass(frozen=True)
class MctsSelfPlaySample:
    state_features: np.ndarray
    action_features: np.ndarray
    target_index: int
    value_target: float
    sample_weight: float
    policy_target: np.ndarray  # shape (num_actions,), sums to 1
    example: dict[str, Any]


class MctsSelfPlayDataset(Dataset[MctsSelfPlaySample]):
    def __init__(
        self,
        path: str | Path,
        *,
        min_actions: int = 2,
        ablations: set[str] | None = None,
    ) -> None:
        self.path = Path(path)
        self.ablations = ablations or set()
        self.samples = list(load_mcts_selfplay_samples(self.path, min_actions=min_actions, ablations=self.ablations))
        if not self.samples:
            raise ValueError(f"No usable mcts-selfplay samples in {self.path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> MctsSelfPlaySample:
        return self.samples[index]


def load_mcts_selfplay_samples(
    path: str | Path,
    *,
    min_actions: int = 2,
    ablations: set[str] | None = None,
) -> Iterable[MctsSelfPlaySample]:
    with Path(path).open("r", encoding="utf8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            example = json.loads(line)
            kind = example.get("kind")
            if kind != "mcts-selfplay":
                raise RowSchemaError(
                    f"Expected kind='mcts-selfplay' at {path}:{line_number}, got {kind!r}"
                )
            version = example.get("schemaVersion")
            if version is None or int(version) != ROW_SCHEMA_VERSION:
                raise RowSchemaError(
                    f"Bad schemaVersion at {path}:{line_number}: got {version}, expected {ROW_SCHEMA_VERSION}"
                )
            actions = example.get("legalActions", [])
            visits = example.get("visitDistribution", [])
            if len(actions) < min_actions or len(visits) != len(actions):
                continue
            arr = np.asarray(visits, dtype=np.float32)
            total = float(arr.sum())
            if total <= 0 or not np.isfinite(total):
                continue
            policy_target = arr / total
            target_index = int(np.argmax(policy_target))
            state_features = observation_to_features(example.get("observation", {}), ablations=ablations)
            action_features = legal_actions_to_features(actions, ablations=ablations)
            if state_features.shape != (STATE_DIM,):
                raise ValueError(f"Bad state feature shape at line {line_number}: {state_features.shape}")
            if action_features.shape[1:] != (ACTION_DIM,):
                raise ValueError(f"Bad action feature shape at line {line_number}: {action_features.shape}")
            value_target = float(example.get("valueTarget", 0) or 0)
            weight = float(example.get("sampleWeight", 1.0) or 1.0)
            yield MctsSelfPlaySample(
                state_features=state_features,
                action_features=action_features,
                target_index=target_index,
                value_target=value_target,
                sample_weight=max(0.05, weight),
                policy_target=policy_target,
                example=example,
            )


def collate_mcts_selfplay_batch(samples: list[MctsSelfPlaySample]) -> dict[str, torch.Tensor]:
    batch_size = len(samples)
    max_actions = max(sample.action_features.shape[0] for sample in samples)

    state_features = np.zeros((batch_size, STATE_DIM), dtype=np.float32)
    action_features = np.zeros((batch_size, max_actions, ACTION_DIM), dtype=np.float32)
    action_mask = np.zeros((batch_size, max_actions), dtype=np.bool_)
    targets = np.zeros((batch_size,), dtype=np.int64)
    value_targets = np.zeros((batch_size,), dtype=np.float32)
    sample_weights = np.ones((batch_size,), dtype=np.float32)
    policy_targets = np.zeros((batch_size, max_actions), dtype=np.float32)

    for row, sample in enumerate(samples):
        count = sample.action_features.shape[0]
        state_features[row] = sample.state_features
        action_features[row, :count] = sample.action_features
        action_mask[row, :count] = True
        targets[row] = sample.target_index
        value_targets[row] = sample.value_target
        sample_weights[row] = sample.sample_weight
        policy_targets[row, :count] = sample.policy_target

    return {
        "state_features": torch.from_numpy(state_features),
        "action_features": torch.from_numpy(action_features),
        "action_mask": torch.from_numpy(action_mask),
        "targets": torch.from_numpy(targets),
        "value_targets": torch.from_numpy(value_targets),
        "sample_weights": torch.from_numpy(sample_weights),
        "policy_targets": torch.from_numpy(policy_targets),
    }
