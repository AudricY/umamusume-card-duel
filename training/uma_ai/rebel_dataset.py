"""Dataset adapter for `kind == "rebel-selfplay"` public-belief rows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import ROW_SCHEMA_VERSION, RowSchemaError
from .features import (
    ACTION_DIM,
    CARD_ID_SHAPES,
    STATE_DIM,
    UMA_SLOT_COUNT,
    UMA_SLOT_FEATURE_DIM,
    ZONE_ORDER,
    action_card_idx_pair,
    feature_builder_for_state_dim,
    legal_actions_to_features,
    observation_to_card_ids,
    observation_to_uma_slots,
)
from .model import BELIEF_FEATURE_DIM


@dataclass(frozen=True)
class RebelSelfPlaySample:
    state_features: np.ndarray
    action_features: np.ndarray
    target_index: int
    value_target: float
    sample_weight: float
    policy_target: np.ndarray
    q_target: np.ndarray | None
    belief_features: np.ndarray
    example: dict[str, Any]
    card_ids_by_zone: dict[str, np.ndarray] | None = None
    action_card_idx: np.ndarray | None = None
    uma_slot_card_ids: np.ndarray | None = None
    uma_slot_features: np.ndarray | None = None


class RebelSelfPlayDataset(Dataset[RebelSelfPlaySample]):
    def __init__(
        self,
        path: str | Path,
        *,
        min_actions: int = 2,
        ablations: set[str] | None = None,
        state_dim: int = STATE_DIM,
        uses_uma_slot_tokens: bool = False,
    ) -> None:
        self.path = Path(path)
        self.samples = list(
            load_rebel_selfplay_samples(
                self.path,
                min_actions=min_actions,
                ablations=ablations,
                state_dim=state_dim,
                uses_uma_slot_tokens=uses_uma_slot_tokens,
            )
        )
        if not self.samples:
            raise ValueError(f"No usable rebel-selfplay samples in {self.path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> RebelSelfPlaySample:
        return self.samples[index]


def load_rebel_selfplay_samples(
    path: str | Path,
    *,
    min_actions: int = 2,
    ablations: set[str] | None = None,
    state_dim: int = STATE_DIM,
    uses_uma_slot_tokens: bool = False,
) -> Iterable[RebelSelfPlaySample]:
    encode_state = feature_builder_for_state_dim(state_dim)
    with Path(path).open("r", encoding="utf8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            example = json.loads(line)
            kind = example.get("kind")
            if kind != "rebel-selfplay":
                raise RowSchemaError(
                    f"Expected kind='rebel-selfplay' at {path}:{line_number}, got {kind!r}"
                )
            version = example.get("schemaVersion")
            if version is None or int(version) != ROW_SCHEMA_VERSION:
                raise RowSchemaError(
                    f"Bad schemaVersion at {path}:{line_number}: got {version}, expected {ROW_SCHEMA_VERSION}"
                )
            belief_version = example.get("beliefSchemaVersion")
            if belief_version is None or int(belief_version) != 1:
                raise RowSchemaError(
                    f"Bad beliefSchemaVersion at {path}:{line_number}: got {belief_version}, expected 1"
                )
            actions = example.get("legalActions", [])
            policy = example.get("searchPolicy", [])
            if len(actions) < min_actions or len(policy) != len(actions):
                continue
            policy_target = np.asarray(policy, dtype=np.float32)
            total = float(policy_target.sum())
            if total <= 0 or not np.isfinite(total):
                continue
            policy_target = policy_target / total
            target_index = int(example.get("selectedActionIndex", int(np.argmax(policy_target))))
            target_index = max(0, min(target_index, len(actions) - 1))

            raw_q = example.get("searchActionValues")
            q_target: np.ndarray | None = None
            if raw_q is not None:
                q_arr = np.asarray(raw_q, dtype=np.float32)
                if q_arr.shape != (len(actions),):
                    raise RowSchemaError(
                        f"Bad searchActionValues at {path}:{line_number}: got shape {q_arr.shape}, expected ({len(actions)},)"
                    )
                if not np.isfinite(q_arr).all():
                    raise RowSchemaError(f"Bad searchActionValues at {path}:{line_number}: contains non-finite values")
                q_target = np.clip(q_arr, -1.0, 1.0).astype(np.float32)

            belief_vector = np.asarray(
                (example.get("beliefFeatures") or {}).get("vector", []),
                dtype=np.float32,
            )
            if belief_vector.shape != (BELIEF_FEATURE_DIM,):
                raise RowSchemaError(
                    f"Bad beliefFeatures.vector at {path}:{line_number}: got shape {belief_vector.shape}, "
                    f"expected ({BELIEF_FEATURE_DIM},)"
                )

            observation = example.get("observation", {})
            state_features = encode_state(observation, ablations=ablations)
            action_features = legal_actions_to_features(actions, ablations=ablations)
            if state_features.shape != (state_dim,):
                raise ValueError(f"Bad state feature shape at line {line_number}: {state_features.shape}")
            if action_features.shape[1:] != (ACTION_DIM,):
                raise ValueError(f"Bad action feature shape at line {line_number}: {action_features.shape}")
            try:
                card_ids_by_zone = observation_to_card_ids(observation)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            action_card_idx = np.stack([action_card_idx_pair(action) for action in actions], axis=0)
            if uses_uma_slot_tokens:
                try:
                    uma_slot_card_ids, uma_slot_features = observation_to_uma_slots(observation)
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
            else:
                uma_slot_card_ids = None
                uma_slot_features = None

            # Value head trains on the grounded Monte-Carlo outcome (+1/-1/0 by
            # actual game winner, observer perspective), NOT the search's own
            # bootstrapped/rollout-contaminated `beliefValue`.
            vt = example.get("valueTarget")
            if vt is None:
                raise RowSchemaError(f"Missing valueTarget at {path}:{line_number}")
            value_target = float(vt)
            yield RebelSelfPlaySample(
                state_features=state_features,
                action_features=action_features,
                target_index=target_index,
                value_target=float(np.clip(value_target, -1.0, 1.0)),
                sample_weight=1.0,
                policy_target=policy_target,
                q_target=q_target,
                belief_features=belief_vector,
                example=example,
                card_ids_by_zone=card_ids_by_zone,
                action_card_idx=action_card_idx,
                uma_slot_card_ids=uma_slot_card_ids,
                uma_slot_features=uma_slot_features,
            )


def collate_rebel_selfplay_batch(samples: list[RebelSelfPlaySample]) -> dict[str, torch.Tensor]:
    batch_size = len(samples)
    max_actions = max(sample.action_features.shape[0] for sample in samples)
    state_dim = samples[0].state_features.shape[0]

    state_features = np.zeros((batch_size, state_dim), dtype=np.float32)
    action_features = np.zeros((batch_size, max_actions, ACTION_DIM), dtype=np.float32)
    action_mask = np.zeros((batch_size, max_actions), dtype=np.bool_)
    targets = np.zeros((batch_size,), dtype=np.int64)
    value_targets = np.zeros((batch_size,), dtype=np.float32)
    sample_weights = np.ones((batch_size,), dtype=np.float32)
    policy_targets = np.zeros((batch_size, max_actions), dtype=np.float32)
    q_targets = np.zeros((batch_size, max_actions), dtype=np.float32)
    q_target_mask = np.zeros((batch_size, max_actions), dtype=np.bool_)
    belief_features = np.zeros((batch_size, BELIEF_FEATURE_DIM), dtype=np.float32)

    max_cards_per_zone = max(CARD_ID_SHAPES.values())
    card_ids_buffer = np.zeros((batch_size, len(ZONE_ORDER), max_cards_per_zone), dtype=np.int64)
    action_card_idx_buffer = np.zeros((batch_size, max_actions, 2), dtype=np.int64)
    all_have_uma_slots = all(
        sample.uma_slot_card_ids is not None and sample.uma_slot_features is not None
        for sample in samples
    )
    uma_slot_card_ids_buffer = (
        np.zeros((batch_size, UMA_SLOT_COUNT), dtype=np.int64) if all_have_uma_slots else None
    )
    uma_slot_features_buffer = (
        np.zeros((batch_size, UMA_SLOT_COUNT, UMA_SLOT_FEATURE_DIM), dtype=np.float32)
        if all_have_uma_slots
        else None
    )

    for row, sample in enumerate(samples):
        count = sample.action_features.shape[0]
        state_features[row] = sample.state_features
        action_features[row, :count] = sample.action_features
        action_mask[row, :count] = True
        targets[row] = sample.target_index
        value_targets[row] = sample.value_target
        sample_weights[row] = sample.sample_weight
        policy_targets[row, :count] = sample.policy_target
        belief_features[row] = sample.belief_features
        if sample.q_target is not None:
            q_targets[row, :count] = sample.q_target
            q_target_mask[row, :count] = True
        for zone_index, zone in enumerate(ZONE_ORDER):
            zone_arr = sample.card_ids_by_zone[zone]  # type: ignore[index]
            card_ids_buffer[row, zone_index, : zone_arr.shape[0]] = zone_arr
        action_card_idx_buffer[row, :count] = sample.action_card_idx  # type: ignore[index]
        if uma_slot_card_ids_buffer is not None and uma_slot_features_buffer is not None:
            uma_slot_card_ids_buffer[row] = sample.uma_slot_card_ids  # type: ignore[assignment]
            uma_slot_features_buffer[row] = sample.uma_slot_features  # type: ignore[assignment]

    batch: dict[str, torch.Tensor] = {
        "state_features": torch.from_numpy(state_features),
        "action_features": torch.from_numpy(action_features),
        "action_mask": torch.from_numpy(action_mask),
        "targets": torch.from_numpy(targets),
        "value_targets": torch.from_numpy(value_targets),
        "sample_weights": torch.from_numpy(sample_weights),
        "policy_targets": torch.from_numpy(policy_targets),
        "q_targets": torch.from_numpy(q_targets),
        "q_target_mask": torch.from_numpy(q_target_mask),
        "belief_features": torch.from_numpy(belief_features),
        "card_ids_by_zone": torch.from_numpy(card_ids_buffer),
        "action_card_idx": torch.from_numpy(action_card_idx_buffer),
    }
    if uma_slot_card_ids_buffer is not None and uma_slot_features_buffer is not None:
        batch["uma_slot_card_ids"] = torch.from_numpy(uma_slot_card_ids_buffer)
        batch["uma_slot_features"] = torch.from_numpy(uma_slot_features_buffer)
    return batch
