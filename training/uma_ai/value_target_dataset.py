"""R13.W3 value-target dataset adapter.

Same row format as MctsSelfPlayDataset, but the per-row `value_target` is
overridden with the row's `rootValue`. In rollout-leaf MCTS selfplay (W3
corpus), `rootValue` is the mean of K rule-bot CRN rollouts at the root
state in modelSide frame, lifted out of `runMcts.diagnostics`. That target
is by construction much lower-variance than the backfilled game-z, which
is the W3 hypothesis: regressing the value head to rollout-mean is the
upstream fix for the bottleneck R12 found.

When `rootValue` is missing or non-finite (older selfplay corpora), the
sample is dropped — we never silently fall back to game-z because that
would dilute the experiment's target distribution and make the W3 verdict
unreadable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .dataset import ROW_SCHEMA_VERSION, RowSchemaError
from .features import (
    ACTION_DIM,
    STATE_DIM,
    action_card_idx_pair,
    legal_actions_to_features,
    observation_to_card_ids,
    observation_to_features,
)
from .selfplay_dataset import MctsSelfPlayDataset, MctsSelfPlaySample


def _row_to_sample(
    example: dict,
    *,
    min_actions: int,
    ablations: set[str] | None,
    allow_missing_card_ids: bool = False,
) -> MctsSelfPlaySample | None:
    actions = example.get("legalActions", [])
    visits = example.get("visitDistribution", [])
    if len(actions) < min_actions or len(visits) != len(actions):
        return None
    arr = np.asarray(visits, dtype=np.float32)
    total = float(arr.sum())
    if total <= 0 or not np.isfinite(total):
        return None
    policy_target = arr / total
    target_index = int(np.argmax(policy_target))
    observation = example.get("observation", {})
    state_features = observation_to_features(observation, ablations=ablations)
    action_features = legal_actions_to_features(actions, ablations=ablations)
    if state_features.shape != (STATE_DIM,):
        return None
    if action_features.shape[1:] != (ACTION_DIM,):
        return None
    # R16-P0: forward the v3 card-embedding fields so value-target
    # retrains exercise the same embedding branch as mcts-distill. Same
    # strict-by-default / named-compat contract as
    # `load_mcts_selfplay_samples`.
    card_ids_by_zone: dict[str, np.ndarray] | None
    action_card_idx: np.ndarray | None
    if allow_missing_card_ids and "cardIdsByZone" not in observation:
        card_ids_by_zone = None
        action_card_idx = None
    else:
        card_ids_by_zone = observation_to_card_ids(observation)
        action_card_idx = np.stack(
            [action_card_idx_pair(action) for action in actions], axis=0
        )
    raw_root_value = example.get("rootValue")
    if raw_root_value is None:
        return None
    try:
        rv = float(raw_root_value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(rv):
        return None
    # Rollout values are already in [-1, 1] (mctsTerminalValue contract).
    # Clip defensively to guard against malformed rows.
    value_target = float(max(-1.0, min(1.0, rv)))
    weight = float(example.get("sampleWeight", 1.0) or 1.0)
    return MctsSelfPlaySample(
        state_features=state_features,
        action_features=action_features,
        target_index=target_index,
        value_target=value_target,
        sample_weight=max(0.05, weight),
        policy_target=policy_target,
        example=example,
        card_ids_by_zone=card_ids_by_zone,
        action_card_idx=action_card_idx,
    )


def _load_value_target_samples(
    path: str | Path,
    *,
    min_actions: int = 2,
    ablations: set[str] | None = None,
    allow_missing_card_ids: bool = False,
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
            sample = _row_to_sample(
                example,
                min_actions=min_actions,
                ablations=ablations,
                allow_missing_card_ids=allow_missing_card_ids,
            )
            if sample is not None:
                yield sample


class ValueTargetDataset(MctsSelfPlayDataset):
    """Same shape as MctsSelfPlayDataset; replaces value_target with rootValue."""

    def __init__(
        self,
        path: str | Path,
        *,
        min_actions: int = 2,
        ablations: set[str] | None = None,
        allow_missing_card_ids: bool = False,
    ) -> None:
        # Intentionally skip super().__init__ — load with the alternative reader
        # so we can drop rows missing rootValue without spurious "no samples"
        # errors against the standard reader's stricter contract.
        self.path = Path(path)
        self.ablations = ablations or set()
        self.allow_missing_card_ids = allow_missing_card_ids
        self.samples = list(
            _load_value_target_samples(
                self.path,
                min_actions=min_actions,
                ablations=self.ablations,
                allow_missing_card_ids=allow_missing_card_ids,
            )
        )
        if not self.samples:
            raise ValueError(f"No usable value-target samples in {self.path} (rootValue missing on all rows)")
