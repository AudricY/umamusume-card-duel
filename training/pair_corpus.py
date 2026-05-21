"""R8 DPO preference-pair corpus loader.

Reads outcome-export JSONL rows (`policy: rollout-outcome-v2`) and yields
one preference pair per usable state: `y_w` is the highest-`rewardMean`
candidate, `y_l` the runner-up. Drops low-margin rows per the noise filter
in `docs/ai-research/scoping/archive/r8-dpo.md` § 5 risk (a):

    tau = max(0.02, sqrt(mean(rewardVariance) / sampleCount))

The 0.02 floor matches `exportOutcomeTrainingExamples.ts:234`'s default
`--tie-threshold`. Mean variance is computed from a 100-row probe of the
corpus on first load (see `compute_tau_from_probe`).

R16-TD 3b chunk 2: also reads the *explicit* preference-pair JSONL schema
emitted by `training/pair_builder.py` (rows tagged ``kind ==
"preference-pair"``). For explicit-pair rows the winner/loser indices,
margin and weight come straight off the row; tau gating is bypassed
because `pair_builder.py` already applied a margin floor.

This is *not* the BC dataset (`uma_ai/dataset.py`) — DPO consumes pairs of
action *indices* into the same `legal_action_features` matrix, not a
single `target_index` plus a soft `policy_target`. The reuse is at the
feature-builder level only: `observation_to_features` for state and
`legal_actions_to_features` for the action matrix; the v3 embedding
branch additionally reuses `observation_to_card_ids` and
`action_card_idx_pair`.
"""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

import numpy as np
import torch
from torch.utils.data import Dataset

from uma_ai.features import (
    ACTION_DIM,
    CARD_ID_SHAPES,
    STATE_DIM,
    ZONE_ORDER,
    action_card_idx_pair,
    legal_actions_to_features,
    observation_to_card_ids,
    observation_to_features,
)

# Mirrors `dataset.py` ROW_SCHEMA_VERSION so a corpus regen that bumps the
# row schema fails loudly at load time rather than silently mis-shaping
# tensors.
ROW_SCHEMA_VERSION = 1
DEFAULT_TAU_FLOOR = 0.02


@dataclass(frozen=True)
class PreferencePair:
    state_features: np.ndarray
    action_features: np.ndarray
    action_mask: np.ndarray
    y_w_index: int
    y_l_index: int
    margin: float
    sample_weight: float
    example: dict[str, Any]
    # R16-TD 3b chunk 2: additive per-zone packed card-vocab indices and
    # per-action source/target idx, mirroring `PolicySample` in
    # `uma_ai/dataset.py`. Optional so legacy outcome-v2 rows that lack
    # `cardIdsByZone` (or constructed-by-test pairs) still load — the
    # collator zero-tensors those positions and `model.forward` falls back
    # to `embed(0) = 0` via `padding_idx=0`.
    card_ids_by_zone: dict[str, np.ndarray] | None = None
    action_card_idx: np.ndarray | None = None
    # R16-TD 3b chunk 2: source tag for visibility in downstream training
    # manifests. Set to "rollout-outcome-v2" for legacy oracle rows and to
    # `sourceKind` (e.g. "mcts-relabel") for explicit-pair rows.
    pair_source_kind: str = "rollout-outcome-v2"


def compute_tau_from_probe(
    path: str | Path,
    *,
    probe_rows: int = 100,
    floor: float = DEFAULT_TAU_FLOOR,
    seed: int = 0,
) -> tuple[float, dict[str, float]]:
    """Probe the first ``probe_rows`` rows for mean per-candidate variance.

    Returns ``(tau, diagnostics)`` where ``tau = max(floor, sqrt(mean_var /
    sample_count))``. The sample-count divisor is read from the probe's
    median ``oracle.sampleCount`` (rollout-CRN K, typically 3 per scoping
    § 5(a)). Diagnostics carries the raw inputs so the caller can log them.
    """

    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    saw_explicit_pair = False
    saw_any_row = False
    with Path(path).open("r", encoding="utf8") as handle:
        for line in handle:
            if not line.strip():
                continue
            saw_any_row = True
            payload = json.loads(line)
            if payload.get("kind") == "preference-pair":
                saw_explicit_pair = True
                if len(sampled) >= probe_rows * 4:
                    break
                continue
            if "oracle" not in payload:
                continue
            sampled.append(payload)
            if len(sampled) >= probe_rows * 4:  # over-sample then random pick
                break
    if not sampled:
        # R16-TD 3b chunk 2: an explicit-pair-only corpus has no oracle
        # field to probe — `pair_builder.py` already applied a margin
        # floor. Short-circuit with the canonical tau floor so the
        # explicit-pair path bypasses tau gating (legacy/mixed corpora
        # still take the probe path).
        if saw_explicit_pair:
            return float(floor), {
                "tau": float(floor),
                "tau_floor": float(floor),
                "explicit_pair_only": 1.0,
            }
        if not saw_any_row:
            raise ValueError(
                f"Corpus {path} is empty — DPO needs at least one row."
            )
        raise ValueError(
            f"Corpus {path} has no rows with `oracle` field — DPO needs the "
            "v2 outcome-export schema (`policy: rollout-outcome-v2`) or the "
            "explicit `kind: preference-pair` schema."
        )
    probe = rng.sample(sampled, min(probe_rows, len(sampled)))
    variances: list[float] = []
    sample_counts: list[int] = []
    for row in probe:
        oracle = row["oracle"]
        sample_counts.append(int(oracle.get("sampleCount") or 1))
        for cand in oracle.get("candidates", []):
            v = cand.get("rewardVariance")
            if v is not None and math.isfinite(float(v)):
                variances.append(float(v))
    if not variances:
        raise ValueError(
            f"Corpus {path} probe found zero finite per-candidate "
            "rewardVariance values; cannot compute tau."
        )
    mean_var = sum(variances) / len(variances)
    # Median sample_count is robust to a stray legacy row that lacks the
    # field (defaults to 1 above). Rollout-CRN K is the same for every
    # row in a single export run.
    sample_counts.sort()
    median_k = sample_counts[len(sample_counts) // 2] if sample_counts else 1
    if median_k < 1:
        median_k = 1
    sqrt_term = math.sqrt(mean_var / median_k)
    tau = max(floor, sqrt_term)
    return tau, {
        "probe_rows_seen": float(len(probe)),
        "candidate_variance_samples": float(len(variances)),
        "mean_reward_variance": float(mean_var),
        "median_sample_count": float(median_k),
        "sqrt_mean_var_over_k": float(sqrt_term),
        "tau_floor": float(floor),
        "tau": float(tau),
    }


def _build_embedding_arrays(
    observation: dict[str, Any],
    actions: list[dict[str, Any]],
    *,
    path: Path,
    line_number: int,
) -> tuple[dict[str, np.ndarray] | None, np.ndarray | None]:
    """Return ``(card_ids_by_zone, action_card_idx)`` or ``(None, None)``.

    Mirrors the v3 embedding extraction in `uma_ai/dataset.py:266-274`.
    Returns ``(None, None)`` when the observation lacks `cardIdsByZone`
    (legacy outcome-v2 rows). The collator zero-tensors those rows; the
    model's `padding_idx=0` falls back to `embed(0) = 0` so the embedding
    branch is exactly inert for legacy data, matching the
    `MctsSelfPlaySample` path described in `dataset.py:332-337`.
    """

    if not isinstance(observation, dict) or "cardIdsByZone" not in observation:
        return None, None
    try:
        card_ids_by_zone = observation_to_card_ids(observation)
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number}: {exc}") from exc
    action_card_idx = np.stack(
        [action_card_idx_pair(action) for action in actions], axis=0
    )
    return card_ids_by_zone, action_card_idx


def _parse_legacy_outcome_pair(
    example: dict[str, Any],
    *,
    tau: float,
    path: Path,
    line_number: int,
) -> PreferencePair | None:
    """Parse a `policy: rollout-outcome-v2` row into a PreferencePair.

    Returns ``None`` when the row fails any of the documented drop
    predicates (missing oracle, <2 candidates, margin below tau, etc.).
    """
    oracle = example.get("oracle")
    if not oracle:
        return None
    candidates = oracle.get("candidates") or []
    if len(candidates) < 2:
        return None
    margin = float(oracle.get("selectedVsRunnerUpMargin") or 0.0)
    if not math.isfinite(margin) or margin < tau:
        return None
    actions = example.get("legalActions") or []
    if len(actions) < 2:
        return None
    y_w_index = int(candidates[0]["index"])
    y_l_index = int(candidates[1]["index"])
    if (
        y_w_index < 0
        or y_l_index < 0
        or y_w_index >= len(actions)
        or y_l_index >= len(actions)
        or y_w_index == y_l_index
    ):
        return None
    observation = example.get("observation", {})
    state_features = observation_to_features(observation)
    action_features = legal_actions_to_features(actions)
    if state_features.shape != (STATE_DIM,):
        raise ValueError(
            f"{path}:{line_number} bad state feature shape "
            f"{state_features.shape}"
        )
    if action_features.shape[1:] != (ACTION_DIM,):
        raise ValueError(
            f"{path}:{line_number} bad action feature shape "
            f"{action_features.shape}"
        )
    mask = np.ones((action_features.shape[0],), dtype=np.bool_)
    sample_weight = float(example.get("sampleWeight") or 1.0)
    if not math.isfinite(sample_weight) or sample_weight <= 0.0:
        sample_weight = 1.0
    card_ids_by_zone, action_card_idx = _build_embedding_arrays(
        observation, actions, path=path, line_number=line_number
    )
    return PreferencePair(
        state_features=state_features,
        action_features=action_features,
        action_mask=mask,
        y_w_index=y_w_index,
        y_l_index=y_l_index,
        margin=margin,
        sample_weight=sample_weight,
        example=example,
        card_ids_by_zone=card_ids_by_zone,
        action_card_idx=action_card_idx,
        pair_source_kind="rollout-outcome-v2",
    )


def _parse_explicit_pair(
    example: dict[str, Any],
    *,
    path: Path,
    line_number: int,
) -> PreferencePair | None:
    """Parse a `kind == "preference-pair"` row (R16-TD 3b chunk 1 schema).

    The pair-builder already applies a `--min-margin` floor, so this path
    bypasses tau gating. Drops rows whose winner/loser indices fall
    outside `legalActions` or whose `observation` is missing — those
    indicate a malformed pair (the 3a pair-builder smoke covers the happy
    path) and the dataset should surface them rather than silently skip.
    """
    actions = example.get("legalActions") or []
    if len(actions) < 2:
        return None
    winner = example.get("winner") or {}
    loser = example.get("loser") or {}
    try:
        y_w_index = int(winner.get("index"))
        y_l_index = int(loser.get("index"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}:{line_number}: winner.index/loser.index must be int-"
            f"coercible (got winner={winner.get('index')!r}, "
            f"loser={loser.get('index')!r})"
        ) from exc
    if (
        y_w_index < 0
        or y_l_index < 0
        or y_w_index >= len(actions)
        or y_l_index >= len(actions)
        or y_w_index == y_l_index
    ):
        raise ValueError(
            f"{path}:{line_number}: explicit pair has out-of-range or "
            f"degenerate indices (w={y_w_index}, l={y_l_index}, "
            f"n_legal={len(actions)})"
        )
    margin = float(example.get("margin") or 0.0)
    if not math.isfinite(margin):
        raise ValueError(
            f"{path}:{line_number}: explicit pair has non-finite margin"
        )
    observation = example.get("observation") or {}
    state_features = observation_to_features(observation)
    action_features = legal_actions_to_features(actions)
    if state_features.shape != (STATE_DIM,):
        raise ValueError(
            f"{path}:{line_number} bad state feature shape "
            f"{state_features.shape}"
        )
    if action_features.shape[1:] != (ACTION_DIM,):
        raise ValueError(
            f"{path}:{line_number} bad action feature shape "
            f"{action_features.shape}"
        )
    mask = np.ones((action_features.shape[0],), dtype=np.bool_)
    sample_weight = float(example.get("sampleWeight") or 1.0)
    if not math.isfinite(sample_weight) or sample_weight <= 0.0:
        sample_weight = 1.0
    card_ids_by_zone, action_card_idx = _build_embedding_arrays(
        observation, actions, path=path, line_number=line_number
    )
    source_kind = str(example.get("sourceKind") or "explicit-pair")
    return PreferencePair(
        state_features=state_features,
        action_features=action_features,
        action_mask=mask,
        y_w_index=y_w_index,
        y_l_index=y_l_index,
        margin=margin,
        sample_weight=sample_weight,
        example=example,
        card_ids_by_zone=card_ids_by_zone,
        action_card_idx=action_card_idx,
        pair_source_kind=source_kind,
    )


def load_preference_pairs(
    path: str | Path,
    *,
    tau: float,
    strict_schema_version: bool = True,
    log_manifest: bool = False,
    log_stream: TextIO | None = None,
) -> Iterator[PreferencePair]:
    """Stream preference pairs from a JSONL file.

    Dispatches per-row on the ``kind`` field:
      - ``kind == "preference-pair"`` → the explicit-pair schema emitted
        by `training/pair_builder.py` (R16-TD 3b chunk 1).
      - otherwise → the legacy `policy: rollout-outcome-v2` schema where
        winner/loser are picked from `oracle.candidates[0]/[1]`.

    Drops rows where (legacy path):
      - ``schemaVersion`` is missing or not ``ROW_SCHEMA_VERSION`` (when
        ``strict_schema_version`` is on)
      - ``oracle`` field is absent (legacy `rollout-outcome-v1` rows)
      - ``oracle.candidates`` has fewer than 2 entries
      - ``selectedVsRunnerUpMargin < tau``

    Per scoping § 3.4: `y_w` is the index (into the *original*
    `legalActions` array, i.e. matching `legal_actions_to_features`) of
    the best candidate; `y_l` is the index of the runner-up. The
    `oracle.candidates[*].index` field carries that mapping (see
    `exportOutcomeTrainingExamples.ts:178`).

    When ``log_manifest`` is true, emits a single ``[pair_corpus]``
    summary line on EOF reporting the explicit-pair / legacy-outcome
    counts.
    """

    path = Path(path)
    explicit_count = 0
    legacy_count = 0
    with path.open("r", encoding="utf8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            example = json.loads(line)
            if strict_schema_version:
                version = example.get("schemaVersion")
                if version is None or int(version) != ROW_SCHEMA_VERSION:
                    raise ValueError(
                        f"{path}:{line_number} schemaVersion {version!r} != "
                        f"{ROW_SCHEMA_VERSION}"
                    )
            if example.get("kind") == "preference-pair":
                pair = _parse_explicit_pair(
                    example, path=path, line_number=line_number
                )
                if pair is None:
                    continue
                explicit_count += 1
                yield pair
            else:
                pair = _parse_legacy_outcome_pair(
                    example, tau=tau, path=path, line_number=line_number
                )
                if pair is None:
                    continue
                legacy_count += 1
                yield pair

    if log_manifest:
        stream = log_stream if log_stream is not None else sys.stderr
        print(
            f"[pair_corpus] loaded {explicit_count} explicit-pair rows + "
            f"{legacy_count} legacy outcome-v2 rows from {path}",
            file=stream,
        )


class PreferencePairDataset(Dataset[PreferencePair]):
    """In-memory preference-pair dataset.

    Loads the full JSONL on construction (the R15.S1 corpus is ~10k rows
    × ~17 KB ≈ 170 MB of features at most — within budget). Accepts
    either a precomputed ``tau`` or computes one from the probe.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        tau: float | None = None,
        strict_schema_version: bool = True,
        log_manifest: bool = True,
    ) -> None:
        self.path = Path(path)
        if tau is None:
            tau, self.tau_diagnostics = compute_tau_from_probe(self.path)
        else:
            self.tau_diagnostics = {"tau": float(tau), "tau_floor": DEFAULT_TAU_FLOOR}
        self.tau = float(tau)
        self.pairs: list[PreferencePair] = list(
            load_preference_pairs(
                self.path,
                tau=self.tau,
                strict_schema_version=strict_schema_version,
                log_manifest=log_manifest,
            )
        )
        if not self.pairs:
            raise ValueError(
                f"No usable preference pairs in {self.path} at tau={self.tau:.4f}"
            )
        # R16-TD 3b chunk 2: expose the loaded mix for downstream training
        # manifests. Same counters that `load_preference_pairs` already
        # logged when ``log_manifest`` is on.
        self.source_counts: dict[str, int] = {}
        for pair in self.pairs:
            kind = pair.pair_source_kind
            self.source_counts[kind] = self.source_counts.get(kind, 0) + 1

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> PreferencePair:
        return self.pairs[index]


# R16-TD 3b chunk 2: counter for embedding-missing rows, reset on each
# collator call. Exposed as a module attribute so smoke tests can spot
# the fallback fire-count without parsing log output.
_EMBEDDING_FALLBACK_WARNINGS = 0


def collate_preference_batch(
    samples: Iterable[PreferencePair],
) -> dict[str, torch.Tensor]:
    """Pad to max-actions-in-batch like ``collate_policy_batch`` does.

    Emits the indices used by ``train_dpo.py`` to gather log-probs for
    the chosen and runner-up actions per row, plus the v3 embedding
    tensors that the chunk-3 trainer will route into
    ``CandidatePolicyNet.forward``.

    R16-TD 3b chunk 2 additive contract:
      - ``card_ids_by_zone_w`` / ``card_ids_by_zone_l``: per-row packed
        per-zone card-vocab indices. Both are identical (winner and
        loser share the same observation/state); the ``_w`` / ``_l``
        suffix discipline mirrors ``y_w`` / ``y_l`` so DPO consumers can
        request the winner/loser embedding tensor explicitly without
        guessing which side a single ``card_ids_by_zone`` belongs to.
        Shape ``[B, NUM_ZONES, max_cards_per_zone]`` (int64).
      - ``action_card_idx_w`` / ``action_card_idx_l``: pre-gathered
        winner/loser action source+target card-vocab pairs, shape
        ``[B, 2]`` each (int64).
      - ``action_card_idx``: the full padded ``[B, max_actions, 2]``
        per-action source+target pair, matching the shape
        ``collate_policy_batch`` emits. The chunk-3 trainer needs this
        when calling ``model.forward`` over the full action set before
        gathering winner/loser log-probs.

    All existing fields (``state_features``, ``action_features``,
    ``action_mask``, ``y_w_index``, ``y_l_index``, ``margins``,
    ``sample_weights``) are preserved. When any row in the batch lacks
    ``card_ids_by_zone`` / ``action_card_idx`` (legacy outcome-v2 rows,
    or a malformed row that slipped past the loader), the affected row
    positions are zero-tensored and the module-level
    ``_EMBEDDING_FALLBACK_WARNINGS`` counter is incremented — the
    model's ``padding_idx=0`` makes this a true no-op for those rows.
    """

    global _EMBEDDING_FALLBACK_WARNINGS
    items = list(samples)
    batch_size = len(items)
    max_actions = max(item.action_features.shape[0] for item in items)
    num_zones = len(ZONE_ORDER)
    max_cards_per_zone = max(CARD_ID_SHAPES.values())

    state_features = np.zeros((batch_size, STATE_DIM), dtype=np.float32)
    action_features = np.zeros((batch_size, max_actions, ACTION_DIM), dtype=np.float32)
    action_mask = np.zeros((batch_size, max_actions), dtype=np.bool_)
    y_w = np.zeros((batch_size,), dtype=np.int64)
    y_l = np.zeros((batch_size,), dtype=np.int64)
    margins = np.zeros((batch_size,), dtype=np.float32)
    sample_weights = np.ones((batch_size,), dtype=np.float32)

    # R16-TD 3b chunk 2: v3 embedding tensors. We always emit them (zero-
    # tensored for rows that lack the field) so downstream training code
    # has a fixed contract — `model.forward` collapses zero-idx slots to
    # the padding embedding (`embed(0) = 0`) so the branch is exactly
    # inert on those rows.
    card_ids_by_zone = np.zeros(
        (batch_size, num_zones, max_cards_per_zone), dtype=np.int64
    )
    full_action_card_idx = np.zeros(
        (batch_size, max_actions, 2), dtype=np.int64
    )
    action_card_idx_w = np.zeros((batch_size, 2), dtype=np.int64)
    action_card_idx_l = np.zeros((batch_size, 2), dtype=np.int64)

    for row, item in enumerate(items):
        count = item.action_features.shape[0]
        state_features[row] = item.state_features
        action_features[row, :count] = item.action_features
        action_mask[row, :count] = True
        y_w[row] = item.y_w_index
        y_l[row] = item.y_l_index
        margins[row] = item.margin
        sample_weights[row] = item.sample_weight

        zones = item.card_ids_by_zone
        action_idx = item.action_card_idx
        if zones is None or action_idx is None:
            _EMBEDDING_FALLBACK_WARNINGS += 1
            continue
        for zone_index, zone in enumerate(ZONE_ORDER):
            zone_arr = zones[zone]
            width = zone_arr.shape[0]
            card_ids_by_zone[row, zone_index, :width] = zone_arr
        full_action_card_idx[row, :count, :] = action_idx
        action_card_idx_w[row, :] = action_idx[item.y_w_index]
        action_card_idx_l[row, :] = action_idx[item.y_l_index]

    return {
        "state_features": torch.from_numpy(state_features),
        "action_features": torch.from_numpy(action_features),
        "action_mask": torch.from_numpy(action_mask),
        "y_w_index": torch.from_numpy(y_w),
        "y_l_index": torch.from_numpy(y_l),
        "margins": torch.from_numpy(margins),
        "sample_weights": torch.from_numpy(sample_weights),
        # R16-TD 3b chunk 2: v3 embedding tensors. The `_w` / `_l`
        # suffix on `card_ids_by_zone_*` is for downstream-contract
        # symmetry only — both tensors are the same per-row state.
        "card_ids_by_zone": torch.from_numpy(card_ids_by_zone),
        "card_ids_by_zone_w": torch.from_numpy(card_ids_by_zone),
        "card_ids_by_zone_l": torch.from_numpy(card_ids_by_zone),
        "action_card_idx": torch.from_numpy(full_action_card_idx),
        "action_card_idx_w": torch.from_numpy(action_card_idx_w),
        "action_card_idx_l": torch.from_numpy(action_card_idx_l),
    }
