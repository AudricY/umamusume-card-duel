"""R8 DPO preference-pair corpus loader.

Reads outcome-export JSONL rows (`policy: rollout-outcome-v2`) and yields
one preference pair per usable state: `y_w` is the highest-`rewardMean`
candidate, `y_l` the runner-up. Drops low-margin rows per the noise filter
in `docs/ai-research/scoping/r8-dpo.md` § 5 risk (a):

    tau = max(0.02, sqrt(mean(rewardVariance) / sampleCount))

The 0.02 floor matches `exportOutcomeTrainingExamples.ts:234`'s default
`--tie-threshold`. Mean variance is computed from a 100-row probe of the
corpus on first load (see `compute_tau_from_probe`).

This is *not* the BC dataset (`uma_ai/dataset.py`) — DPO consumes pairs of
action *indices* into the same `legal_action_features` matrix, not a
single `target_index` plus a soft `policy_target`. The reuse is at the
feature-builder level only: `observation_to_features` for state and
`legal_actions_to_features` for the action matrix.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

from uma_ai.features import (
    ACTION_DIM,
    STATE_DIM,
    legal_actions_to_features,
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
    with Path(path).open("r", encoding="utf8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if "oracle" not in payload:
                continue
            sampled.append(payload)
            if len(sampled) >= probe_rows * 4:  # over-sample then random pick
                break
    if not sampled:
        raise ValueError(
            f"Corpus {path} has no rows with `oracle` field — DPO needs the "
            "v2 outcome-export schema (`policy: rollout-outcome-v2`)."
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


def load_preference_pairs(
    path: str | Path,
    *,
    tau: float,
    strict_schema_version: bool = True,
) -> Iterator[PreferencePair]:
    """Stream preference pairs from outcome-export JSONL.

    Drops rows where:
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
    """

    path = Path(path)
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
            oracle = example.get("oracle")
            if not oracle:
                continue
            candidates = oracle.get("candidates") or []
            if len(candidates) < 2:
                continue
            margin = float(oracle.get("selectedVsRunnerUpMargin") or 0.0)
            if not math.isfinite(margin) or margin < tau:
                continue
            actions = example.get("legalActions") or []
            if len(actions) < 2:
                continue
            y_w_index = int(candidates[0]["index"])
            y_l_index = int(candidates[1]["index"])
            if (
                y_w_index < 0
                or y_l_index < 0
                or y_w_index >= len(actions)
                or y_l_index >= len(actions)
                or y_w_index == y_l_index
            ):
                continue
            state_features = observation_to_features(
                example.get("observation", {})
            )
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
            yield PreferencePair(
                state_features=state_features,
                action_features=action_features,
                action_mask=mask,
                y_w_index=y_w_index,
                y_l_index=y_l_index,
                margin=margin,
                sample_weight=sample_weight,
                example=example,
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
            )
        )
        if not self.pairs:
            raise ValueError(
                f"No usable preference pairs in {self.path} at tau={self.tau:.4f}"
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> PreferencePair:
        return self.pairs[index]


def collate_preference_batch(
    samples: Iterable[PreferencePair],
) -> dict[str, torch.Tensor]:
    """Pad to max-actions-in-batch like ``collate_policy_batch`` does.

    Emits the indices used by ``train_dpo.py`` to gather log-probs for
    the chosen and runner-up actions per row.
    """

    items = list(samples)
    batch_size = len(items)
    max_actions = max(item.action_features.shape[0] for item in items)

    state_features = np.zeros((batch_size, STATE_DIM), dtype=np.float32)
    action_features = np.zeros((batch_size, max_actions, ACTION_DIM), dtype=np.float32)
    action_mask = np.zeros((batch_size, max_actions), dtype=np.bool_)
    y_w = np.zeros((batch_size,), dtype=np.int64)
    y_l = np.zeros((batch_size,), dtype=np.int64)
    margins = np.zeros((batch_size,), dtype=np.float32)
    sample_weights = np.ones((batch_size,), dtype=np.float32)

    for row, item in enumerate(items):
        count = item.action_features.shape[0]
        state_features[row] = item.state_features
        action_features[row, :count] = item.action_features
        action_mask[row, :count] = True
        y_w[row] = item.y_w_index
        y_l[row] = item.y_l_index
        margins[row] = item.margin
        sample_weights[row] = item.sample_weight

    return {
        "state_features": torch.from_numpy(state_features),
        "action_features": torch.from_numpy(action_features),
        "action_mask": torch.from_numpy(action_mask),
        "y_w_index": torch.from_numpy(y_w),
        "y_l_index": torch.from_numpy(y_l),
        "margins": torch.from_numpy(margins),
        "sample_weights": torch.from_numpy(sample_weights),
    }
