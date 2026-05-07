from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import ACTION_DIM, STATE_DIM, legal_actions_to_features, observation_to_features


@dataclass(frozen=True)
class PolicySample:
    state_features: np.ndarray
    action_features: np.ndarray
    target_index: int
    value_target: float
    example: dict[str, Any]


class JsonlPolicyDataset(Dataset[PolicySample]):
    def __init__(self, path: str | Path, *, min_actions: int = 2) -> None:
        self.path = Path(path)
        self.samples = list(load_policy_samples(self.path, min_actions=min_actions))
        if not self.samples:
            raise ValueError(f"No usable policy samples found in {self.path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PolicySample:
        return self.samples[index]


def load_policy_samples(path: str | Path, *, min_actions: int = 2) -> Iterable[PolicySample]:
    with Path(path).open("r", encoding="utf8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            example = json.loads(line)
            actions = example.get("legalActions", [])
            target_index = int(example.get("selectedActionIndex", -1))
            if len(actions) < min_actions or target_index < 0 or target_index >= len(actions):
                continue
            state_features = observation_to_features(example.get("observation", {}))
            action_features = legal_actions_to_features(actions)
            if state_features.shape != (STATE_DIM,):
                raise ValueError(f"Bad state feature shape at line {line_number}: {state_features.shape}")
            if action_features.shape[1:] != (ACTION_DIM,):
                raise ValueError(f"Bad action feature shape at line {line_number}: {action_features.shape}")
            yield PolicySample(
                state_features=state_features,
                action_features=action_features,
                target_index=target_index,
                value_target=_value_target(example),
                example=example,
            )


def collate_policy_batch(samples: list[PolicySample]) -> dict[str, torch.Tensor]:
    batch_size = len(samples)
    max_actions = max(sample.action_features.shape[0] for sample in samples)

    state_features = np.zeros((batch_size, STATE_DIM), dtype=np.float32)
    action_features = np.zeros((batch_size, max_actions, ACTION_DIM), dtype=np.float32)
    action_mask = np.zeros((batch_size, max_actions), dtype=np.bool_)
    targets = np.zeros((batch_size,), dtype=np.int64)
    value_targets = np.zeros((batch_size,), dtype=np.float32)

    for row, sample in enumerate(samples):
        count = sample.action_features.shape[0]
        state_features[row] = sample.state_features
        action_features[row, :count] = sample.action_features
        action_mask[row, :count] = True
        targets[row] = sample.target_index
        value_targets[row] = sample.value_target

    return {
        "state_features": torch.from_numpy(state_features),
        "action_features": torch.from_numpy(action_features),
        "action_mask": torch.from_numpy(action_mask),
        "targets": torch.from_numpy(targets),
        "value_targets": torch.from_numpy(value_targets),
    }


def _value_target(example: dict[str, Any]) -> float:
    winner = example.get("result", {}).get("winner")
    side_id = example.get("sideId")
    if winner is None or side_id not in ("player", "opponent"):
        return 0.0
    return 1.0 if winner == side_id else -1.0
