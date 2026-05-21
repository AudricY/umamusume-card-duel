from __future__ import annotations

import json
import random
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
    UMA_SLOT_COUNT,
    UMA_SLOT_FEATURE_DIM,
    ZONE_ORDER,
    action_card_idx_pair,
    feature_builder_for_state_dim,
    legal_actions_to_features,
    observation_to_card_ids,
    observation_to_uma_slots,
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
    # R16-P2 C4: per-Uma slot-token tensors. Populated only when the dataset
    # is loaded with `uses_uma_slot_tokens=True`. When None the collator
    # omits both keys from the batch dict (all-or-nothing pattern, mirrors
    # `card_ids_by_zone` / `action_card_idx`), so v3.0/v3.1 callers see the
    # exact pre-C4 batch shape. Shapes when populated:
    #   uma_slot_card_ids: int64[UMA_SLOT_COUNT]                 (= [10])
    #   uma_slot_features: float32[UMA_SLOT_COUNT, UMA_SLOT_FEATURE_DIM]
    #                                                            (= [10, 23])
    # Absent slots have card_id=0 and an all-zero feature row, matching the
    # C1 builder's "absence is zero" contract; the model's mask source is
    # `(uma_slot_card_ids != 0)`.
    uma_slot_card_ids: np.ndarray | None = None
    uma_slot_features: np.ndarray | None = None


class JsonlPolicyDataset(Dataset[PolicySample]):
    def __init__(
        self,
        path: str | Path,
        *,
        min_actions: int = 2,
        ablations: set[str] | None = None,
        strict_schema_version: bool = True,
        state_dim: int = STATE_DIM,
        contested_loss_weight: float = 1.0,
        contested_min_legal: int = 4,
        contested_resample_fraction: float | None = None,
        contested_resample_seed: int = 0,
        uses_uma_slot_tokens: bool = False,
    ) -> None:
        # `state_dim` selects the frozen builder (96=v2, 110=v3.0, 164=v3.1)
        # via the same dim-keyed mechanism serve_onnx uses. Default is the
        # module STATE_DIM (110 = v3.0) so existing callers are byte-stable;
        # v3.1 training opts in by passing state_dim=STATE_DIM_V3_1.
        # R16-P2 C4: `uses_uma_slot_tokens` opts into per-Uma slot-token
        # packing (additive `uma_slot_card_ids` / `uma_slot_features` fields
        # on every loaded sample). Default False so v3.0/v3.1 callers AND
        # pre-C4 BC corpora are byte-identical; v3.2 training opts in by
        # passing the flag through. Mirrors `state_dim` kwarg-plumbing style.
        self.path = Path(path)
        self.ablations = ablations or set()
        self.samples = list(
            load_policy_samples(
                self.path,
                min_actions=min_actions,
                ablations=self.ablations,
                strict_schema_version=strict_schema_version,
                state_dim=state_dim,
                contested_loss_weight=contested_loss_weight,
                contested_min_legal=contested_min_legal,
                contested_resample_fraction=contested_resample_fraction,
                contested_resample_seed=contested_resample_seed,
                uses_uma_slot_tokens=uses_uma_slot_tokens,
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
    contested_loss_weight: float = 1.0,
    contested_min_legal: int = 4,
    contested_resample_fraction: float | None = None,
    contested_resample_seed: int = 0,
    uses_uma_slot_tokens: bool = False,
) -> Iterable[PolicySample]:
    """Load retained (>=`min_actions`-legal) policy rows.

    R16 Fork A contested-coverage pilot — two independently-selectable knobs,
    both holding the total retained-row count fixed and both defaulting OFF
    (bit-identical to pre-pilot behavior when unset). Design + verdict home:
    `docs/ai-research/scoping/r16-training-data-backlog-refinement.md`
    § "Fork A — Contested-State Data Coverage"; metric definition in
    `docs/ai-research/analysis/training-data-coverage-audit.md`
    (`legal_action_count`).

    - Option 3 (`contested_loss_weight` != 1.0): multiply `sample_weight` by
      `contested_loss_weight` for rows with `>= contested_min_legal` legal
      actions. Reuses the existing `_sample_weight` /
      `collate_policy_batch` / `normalized_weights` seam — no corpus change,
      no row-count change. `1.0` is a no-op.
    - Option 1 (`contested_resample_fraction` is not None): upsample
      `>= contested_min_legal`-legal rows and downsample 2-legal rows so the
      contested fraction of the *retained* stream hits the target, while
      holding the total retained-row count exactly fixed (sample WITH
      replacement within each group to the computed group target). `None`
      is a no-op (rows yielded once, in file order).

    Both knobs use `contested_min_legal` (default 4) as the contested
    threshold so they match the audit's `legal_action_count` metric
    (fraction of retained rows with >= 4 legal actions).
    """

    weight_contested = contested_loss_weight != 1.0
    resample = contested_resample_fraction is not None
    encode_state = feature_builder_for_state_dim(state_dim)
    if not resample:
        yield from _stream_policy_samples(
            path,
            min_actions=min_actions,
            ablations=ablations,
            strict_schema_version=strict_schema_version,
            state_dim=state_dim,
            encode_state=encode_state,
            weight_contested=weight_contested,
            contested_loss_weight=contested_loss_weight,
            contested_min_legal=contested_min_legal,
            uses_uma_slot_tokens=uses_uma_slot_tokens,
        )
        return

    # Option 1 — contested resampling. Buffer the retained stream so we can
    # split it into a contested (>=`contested_min_legal`-legal) group and a
    # non-contested remainder, then draw WITH replacement from each group to
    # hit `contested_resample_fraction` while keeping the total retained
    # count identical to the no-op load. Deterministic given the seed.
    buffered = list(
        _stream_policy_samples(
            path,
            min_actions=min_actions,
            ablations=ablations,
            strict_schema_version=strict_schema_version,
            state_dim=state_dim,
            encode_state=encode_state,
            weight_contested=weight_contested,
            contested_loss_weight=contested_loss_weight,
            contested_min_legal=contested_min_legal,
            uses_uma_slot_tokens=uses_uma_slot_tokens,
        )
    )
    total = len(buffered)
    if total == 0:
        return
    frac = float(contested_resample_fraction)
    if not 0.0 <= frac <= 1.0:
        raise ValueError(
            f"contested_resample_fraction must be in [0, 1]; got {frac}"
        )
    contested = [
        s for s in buffered if s.action_features.shape[0] >= contested_min_legal
    ]
    others = [
        s for s in buffered if s.action_features.shape[0] < contested_min_legal
    ]
    if not contested or not others:
        # One group is empty — the target fraction is unreachable without
        # fabricating rows. Yield the unmodified buffered stream (count
        # fixed) rather than silently distorting the corpus.
        yield from buffered
        return
    target_contested = int(round(frac * total))
    target_contested = max(0, min(total, target_contested))
    target_others = total - target_contested
    rng = random.Random(contested_resample_seed)
    picked: list[PolicySample] = []
    if target_contested > 0:
        picked.extend(rng.choices(contested, k=target_contested))
    if target_others > 0:
        picked.extend(rng.choices(others, k=target_others))
    rng.shuffle(picked)
    yield from picked


def _stream_policy_samples(
    path: str | Path,
    *,
    min_actions: int,
    ablations: set[str] | None,
    strict_schema_version: bool,
    state_dim: int,
    encode_state: Any,
    weight_contested: bool,
    contested_loss_weight: float,
    contested_min_legal: int,
    uses_uma_slot_tokens: bool = False,
) -> Iterable[PolicySample]:
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
            # R16-P2 C4: per-Uma slot-token packing. Default OFF (fields stay
            # None → collator omits the keys, batch dict is byte-identical to
            # pre-C4). When `uses_uma_slot_tokens=True` we call the C1 builder
            # directly; it raises `ValueError` on rows that lack the required
            # PublicObservation `own`/`opponent` side dicts — fail-loud guard
            # so a corpus-schema mismatch surfaces at load time rather than as
            # silent zero tensors. Mirrors the existing
            # `observation_to_card_ids` fail-loud pattern above.
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
            sample_weight = _sample_weight(example)
            # Option 3 — legal-action-count loss weighting. Scale the policy
            # sample weight for contested (>= contested_min_legal-legal)
            # rows. No-op when contested_loss_weight == 1.0 (weight_contested
            # is False), so the unset path is bit-identical.
            if weight_contested and len(actions) >= contested_min_legal:
                sample_weight *= contested_loss_weight
            yield PolicySample(
                state_features=state_features,
                action_features=action_features,
                target_index=target_index,
                value_target=_value_target(example),
                sample_weight=sample_weight,
                example=example,
                policy_target=policy_target,
                card_ids_by_zone=card_ids_by_zone,
                action_card_idx=action_card_idx,
                uma_slot_card_ids=uma_slot_card_ids,
                uma_slot_features=uma_slot_features,
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

    # R16-P2 C4: per-Uma slot-token packing. Same ALL-OR-NOTHING emit pattern
    # as `card_ids_by_zone` above (the template lives ~10 lines up; if even
    # one sample in the batch lacks both fields the batch dict silently
    # omits both keys, never partial-emit). v3.0/v3.1 batches that never
    # set `uses_uma_slot_tokens=True` see `None` on every sample, so both
    # keys are absent from the emitted dict — byte-identical to pre-C4.
    # Cross-data-path note: MctsSelfPlaySample also carries these as
    # `getattr(..., None)`-default optional fields, so the same all-or-
    # nothing branch fires uniformly across BC and mcts-distill collators.
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
        if (
            uma_slot_card_ids_buffer is not None
            and uma_slot_features_buffer is not None
        ):
            # R16-P2 C4: slot tensors are fixed-shape (UMA_SLOT_COUNT=10)
            # per sample by the C1 builder contract, so we write the full
            # row directly with no per-zone walk / padding logic.
            uma_slot_card_ids_buffer[row] = sample.uma_slot_card_ids  # type: ignore[assignment]
            uma_slot_features_buffer[row] = sample.uma_slot_features  # type: ignore[assignment]

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
    if (
        uma_slot_card_ids_buffer is not None
        and uma_slot_features_buffer is not None
    ):
        batch["uma_slot_card_ids"] = torch.from_numpy(uma_slot_card_ids_buffer)
        batch["uma_slot_features"] = torch.from_numpy(uma_slot_features_buffer)
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
