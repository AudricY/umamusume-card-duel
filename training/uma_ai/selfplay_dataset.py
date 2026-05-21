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


@dataclass(frozen=True)
class MctsSelfPlaySample:
    state_features: np.ndarray
    action_features: np.ndarray
    target_index: int
    value_target: float
    sample_weight: float
    policy_target: np.ndarray  # shape (num_actions,), sums to 1
    example: dict[str, Any]
    # R16-P0: per-zone packed card-vocab indices (8 zones, fixed
    # `CARD_ID_SHAPES`) and per-action source/target idx (shape `(A, 2)`).
    # Mirrors `PolicySample`'s embedding fields so `train_bc.py
    # --data-mode mcts-distill` exercises the same v3 card-embedding
    # branch the BC path does. By default `load_mcts_selfplay_samples()`
    # fails loud on rows missing `cardIdsByZone` (via
    # `observation_to_card_ids`), so any loaded sample carries both
    # fields. `None` stays the type-level nullable to mirror legacy
    # rows produced under the explicit `allow_missing_card_ids` compat
    # path, and so test fixtures can construct samples with the
    # embedding branch disabled.
    card_ids_by_zone: dict[str, np.ndarray] | None = None
    action_card_idx: np.ndarray | None = None
    # R16-P2 C4: per-Uma slot-token tensors. Mirrors the same optional fields
    # on `PolicySample` (BC path). Populated only when the dataset is loaded
    # with `uses_uma_slot_tokens=True`; otherwise None and the collator omits
    # both batch keys (all-or-nothing pattern). The MCTS self-play observation
    # is the SAME `PublicObservation` shape the BC path consumes (both go
    # through `relabelDecisionTrace.ts` / `mctsSelfPlay.ts` in TS, which write
    # the v3 nested `own`/`opponent` schema), so `observation_to_uma_slots()`
    # works without any payload-shape divergence.
    uma_slot_card_ids: np.ndarray | None = None
    uma_slot_features: np.ndarray | None = None


class MctsSelfPlayDataset(Dataset[MctsSelfPlaySample]):
    def __init__(
        self,
        path: str | Path,
        *,
        min_actions: int = 2,
        ablations: set[str] | None = None,
        allow_missing_card_ids: bool = False,
        state_dim: int = STATE_DIM,
        uses_uma_slot_tokens: bool = False,
    ) -> None:
        # `state_dim` selects the frozen builder (96=v2, 110=v3.0, 164=v3.1)
        # via the same dim-keyed mechanism `JsonlPolicyDataset` (BC path) and
        # serve_onnx use. Default is the module STATE_DIM (110 = v3.0) so the
        # mcts-distill path is byte-stable for existing callers; v3.1 distill
        # opts in by passing state_dim=STATE_DIM_V3_1.
        # R16-P2 C4: `uses_uma_slot_tokens` mirrors `JsonlPolicyDataset` —
        # default False keeps v3.0/v3.1 distill batches byte-identical to
        # pre-C4; v3.2 distill opts in by passing the flag.
        self.path = Path(path)
        self.ablations = ablations or set()
        self.allow_missing_card_ids = allow_missing_card_ids
        self.state_dim = state_dim
        self.samples = list(
            load_mcts_selfplay_samples(
                self.path,
                min_actions=min_actions,
                ablations=self.ablations,
                allow_missing_card_ids=allow_missing_card_ids,
                state_dim=state_dim,
                uses_uma_slot_tokens=uses_uma_slot_tokens,
            )
        )
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
    allow_missing_card_ids: bool = False,
    state_dim: int = STATE_DIM,
    uses_uma_slot_tokens: bool = False,
) -> Iterable[MctsSelfPlaySample]:
    """Load mcts-selfplay rows as `MctsSelfPlaySample`s.

    R16-P0: by default this fails loud on rows that lack
    `observation.cardIdsByZone` (the v3 card-embedding inputs), reusing
    the existing `observation_to_card_ids()` error path. This rejects
    pre-R7.b.2-Phase-1 / pre-v3 self-play corpora at load time rather
    than silently training the 110-d model with the embedding branch
    inert (the exact bug that confounded R110-W6).

    `allow_missing_card_ids=True` is a deliberate, named compatibility
    escape hatch for old corpora: rows missing `cardIdsByZone` are still
    loaded but their `card_ids_by_zone` / `action_card_idx` stay `None`,
    so the collator omits the embedding tensors for those batches. It
    never silently zeroes malformed-but-present v3 rows — a present
    `cardIdsByZone` that fails to parse still raises.
    """
    encode_state = feature_builder_for_state_dim(state_dim)
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
            state_features = encode_state(example.get("observation", {}), ablations=ablations)
            action_features = legal_actions_to_features(actions, ablations=ablations)
            if state_features.shape != (state_dim,):
                raise ValueError(f"Bad state feature shape at line {line_number}: {state_features.shape}")
            if action_features.shape[1:] != (ACTION_DIM,):
                raise ValueError(f"Bad action feature shape at line {line_number}: {action_features.shape}")
            value_target = float(example.get("valueTarget", 0) or 0)
            weight = float(example.get("sampleWeight", 1.0) or 1.0)
            # R16-P0: emit packed per-zone card-id arrays and per-action
            # source/target idx so mcts-distill exercises the v3
            # card-embedding branch. `observation_to_card_ids` raises on
            # a missing `cardIdsByZone`; under the explicit
            # `allow_missing_card_ids` compat path we leave both fields
            # `None` for that row only (a present-but-malformed
            # `cardIdsByZone` still raises — never silently zeroed).
            observation = example.get("observation", {})
            card_ids_by_zone: dict[str, np.ndarray] | None
            action_card_idx: np.ndarray | None
            if allow_missing_card_ids and "cardIdsByZone" not in observation:
                card_ids_by_zone = None
                action_card_idx = None
            else:
                try:
                    card_ids_by_zone = observation_to_card_ids(observation)
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
                action_card_idx = np.stack(
                    [action_card_idx_pair(action) for action in actions], axis=0
                )
            # R16-P2 C4: per-Uma slot-token packing — mirrors the BC dataset
            # path (`dataset.py`). The MCTS self-play observation is the same
            # `PublicObservation` shape (own/opponent nested sides with
            # `active` + `bench` Umas carrying `cardId`/`hp`/`energies`/…),
            # so `observation_to_uma_slots()` works against it unchanged.
            # Fail-loud: a row that lacks the v3-era nested side dicts raises
            # in the C1 builder; we wrap with the `{path}:{line}` prefix the
            # `observation_to_card_ids` branch above uses so error sources
            # are uniformly locatable.
            uma_slot_card_ids: np.ndarray | None
            uma_slot_features: np.ndarray | None
            if uses_uma_slot_tokens:
                try:
                    uma_slot_card_ids, uma_slot_features = observation_to_uma_slots(observation)
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
            else:
                uma_slot_card_ids = None
                uma_slot_features = None
            yield MctsSelfPlaySample(
                state_features=state_features,
                action_features=action_features,
                target_index=target_index,
                value_target=value_target,
                sample_weight=max(0.05, weight),
                policy_target=policy_target,
                example=example,
                card_ids_by_zone=card_ids_by_zone,
                action_card_idx=action_card_idx,
                uma_slot_card_ids=uma_slot_card_ids,
                uma_slot_features=uma_slot_features,
            )


def collate_mcts_selfplay_batch(samples: list[MctsSelfPlaySample]) -> dict[str, torch.Tensor]:
    batch_size = len(samples)
    max_actions = max(sample.action_features.shape[0] for sample in samples)
    # State width is derived from the samples (same pattern as `max_actions`)
    # rather than the module STATE_DIM constant so a 164-d v3.1 distill batch
    # collates to `[B, 164]`. All samples in a batch share one builder (the
    # dataset's `state_dim`), so taking the first row's width is exact.
    state_dim = samples[0].state_features.shape[0]

    state_features = np.zeros((batch_size, state_dim), dtype=np.float32)
    action_features = np.zeros((batch_size, max_actions, ACTION_DIM), dtype=np.float32)
    action_mask = np.zeros((batch_size, max_actions), dtype=np.bool_)
    targets = np.zeros((batch_size,), dtype=np.int64)
    value_targets = np.zeros((batch_size,), dtype=np.float32)
    sample_weights = np.ones((batch_size,), dtype=np.float32)
    policy_targets = np.zeros((batch_size, max_actions), dtype=np.float32)

    # R16-P0: pack the v3 card-embedding tensors, mirroring
    # `collate_policy_batch`. `max_cards_per_zone` is the global cap (30,
    # the discard cap) so smaller-cap zones see padding beyond their cap;
    # the embedding's `padding_idx=0` zeroes those positions in the
    # gather. Only emit them when EVERY sample in the batch carries both
    # fields — a mixed batch (one v3 row + one legacy/compat row) falls
    # back to the embedding-inert path uniformly rather than silently
    # zero-tensoring real v3 rows.
    max_cards_per_zone = max(CARD_ID_SHAPES.values())
    num_zones = len(ZONE_ORDER)
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

    # R16-P2 C4: per-Uma slot-token packing — same ALL-OR-NOTHING emit
    # pattern as `card_ids_by_zone` directly above. If even one sample in
    # the batch lacks both slot fields the dict omits both keys; never
    # partial-emit. v3.0/v3.1 distill batches (default
    # `uses_uma_slot_tokens=False`) see None on every sample, so both keys
    # stay absent from the emitted dict — byte-identical to pre-C4.
    all_have_uma_slots = all(
        getattr(sample, "uma_slot_card_ids", None) is not None
        and getattr(sample, "uma_slot_features", None) is not None
        for sample in samples
    )
    uma_slot_card_ids_buffer = (
        np.zeros((batch_size, UMA_SLOT_COUNT), dtype=np.int64)
        if all_have_uma_slots
        else None
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
        if card_ids_buffer is not None and action_card_idx_buffer is not None:
            for zone_index, zone in enumerate(ZONE_ORDER):
                zone_arr = sample.card_ids_by_zone[zone]  # type: ignore[index]
                width = zone_arr.shape[0]
                card_ids_buffer[row, zone_index, :width] = zone_arr
            action_card_idx_buffer[row, :count, :] = sample.action_card_idx  # type: ignore[index]
        if (
            uma_slot_card_ids_buffer is not None
            and uma_slot_features_buffer is not None
        ):
            # R16-P2 C4: slot tensors are fixed-shape per sample (10,) and
            # (10, 23) per the C1 builder contract — write the row directly.
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
    }
    if card_ids_buffer is not None and action_card_idx_buffer is not None:
        batch["card_ids_by_zone"] = torch.from_numpy(card_ids_buffer)
        batch["action_card_idx"] = torch.from_numpy(action_card_idx_buffer)
    if (
        uma_slot_card_ids_buffer is not None
        and uma_slot_features_buffer is not None
    ):
        batch["uma_slot_card_ids"] = torch.from_numpy(uma_slot_card_ids_buffer)
        batch["uma_slot_features"] = torch.from_numpy(uma_slot_features_buffer)
    return batch
