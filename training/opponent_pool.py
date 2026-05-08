"""Opponent snapshot pool with PFSP sampling for backlog item 12.

Maintains a versioned pool of promoted DAgger checkpoints, samples them
via prioritized fictitious self-play (loss-rate-weighted), applies a
hybrid retention policy (last 8 promoted + every 4th historical, hard
cap 24), and exposes a strict-monotone cycling alarm consistent with the
loop promotion gate wording.

The orchestrator owns persistence (JSON next to ``orchestrator-state.json``)
and per-iteration eval results; this module is intentionally pure data.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PoolEntry:
    iteration: int
    checkpoint_path: str
    wilson_lower_at_promotion: float
    value_mean_drift_from_prior: float = 0.0
    created_at_seconds: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PoolEntry":
        return cls(
            iteration=int(payload["iteration"]),
            checkpoint_path=str(payload["checkpoint_path"]),
            wilson_lower_at_promotion=float(payload.get("wilson_lower_at_promotion", 0.0)),
            value_mean_drift_from_prior=float(payload.get("value_mean_drift_from_prior", 0.0)),
            created_at_seconds=float(payload.get("created_at_seconds", time.time())),
        )


@dataclass
class OpponentPool:
    entries: list[PoolEntry] = field(default_factory=list)

    # ------------------------------------------------------------------
    # core mutators

    def add(self, entry: PoolEntry) -> None:
        """Append a new pool entry. Caller is expected to apply retention."""
        self.entries.append(entry)

    def retain(self, *, recent_promoted: int = 8, historical_stride: int = 4, max_total: int = 24) -> list[PoolEntry]:
        """Apply hybrid retention.

        Keep:
        - the last ``recent_promoted`` entries unconditionally (sorted by
          iteration ascending, take from the tail);
        - every ``historical_stride``-th entry from the older slice
          (indices 0, stride, 2*stride, ... after sorting ascending);
        - hard cap at ``max_total`` after the union.

        Pruned entries whose ``checkpoint_path`` lives under a directory
        named ``pool/`` have their parent directory ``shutil.rmtree``'d
        from disk so the snapshot bytes do not leak.

        Returns the list of pruned ``PoolEntry`` objects (post-removal),
        for caller-side logging.
        """

        if not self.entries:
            return []

        sorted_entries = sorted(self.entries, key=lambda e: e.iteration)
        total = len(sorted_entries)
        keep_recent = sorted_entries[-recent_promoted:] if total > recent_promoted else list(sorted_entries)
        recent_iters = {entry.iteration for entry in keep_recent}

        older_slice = sorted_entries[: max(0, total - recent_promoted)]
        keep_historical = [
            entry
            for index, entry in enumerate(older_slice)
            if index % max(1, historical_stride) == 0
        ]
        # Union, deduped by iteration, preserving sorted order.
        kept_by_iter: dict[int, PoolEntry] = {}
        for entry in keep_recent + keep_historical:
            kept_by_iter[entry.iteration] = entry
        # Hard cap: keep newest ``max_total``.
        if len(kept_by_iter) > max_total:
            ordered = sorted(kept_by_iter.values(), key=lambda e: e.iteration)
            ordered = ordered[-max_total:]
            kept_by_iter = {entry.iteration: entry for entry in ordered}

        pruned = [entry for entry in sorted_entries if entry.iteration not in kept_by_iter]
        for entry in pruned:
            self._maybe_rmtree(entry.checkpoint_path)
        self.entries = sorted(kept_by_iter.values(), key=lambda e: e.iteration)
        return pruned

    @staticmethod
    def _maybe_rmtree(checkpoint_path_str: str) -> None:
        try:
            checkpoint_path = Path(checkpoint_path_str)
            parent = checkpoint_path.parent
            # Only delete if any ancestor directory is named "pool"; this
            # protects against accidentally deleting the user's training
            # output if a hand-built pool entry points outside the pool/
            # subtree.
            if any(part == "pool" for part in parent.parts):
                shutil.rmtree(parent, ignore_errors=True)
        except Exception:
            # Disk-side cleanup is best-effort; never raise out of retain().
            return

    # ------------------------------------------------------------------
    # PFSP sampling

    def pfsp_weights(self, win_rates_by_iteration: dict[int, float]) -> dict[int, float]:
        """Compute prioritized fictitious self-play weights.

        ``w_i = max(0.05, 1 - p_i)`` where ``p_i`` is the *current model's*
        win rate against pool member ``i``. Falls back to uniform weights
        over current pool members when:
        - no entries are present;
        - no win rates are supplied for any pool member;
        - all computed weights collapse to the floor (0.05 + epsilon).
        """

        if not self.entries:
            return {}
        eps = 1e-9
        floor = 0.05
        observed = {entry.iteration: win_rates_by_iteration.get(entry.iteration) for entry in self.entries}
        if all(value is None for value in observed.values()):
            uniform = 1.0 / len(self.entries)
            return {entry.iteration: uniform for entry in self.entries}

        raw_weights: dict[int, float] = {}
        for iteration, win_rate in observed.items():
            if win_rate is None:
                # Unobserved opponents start at full weight (we have no
                # evidence the current model already beats them).
                raw_weights[iteration] = 1.0
            else:
                raw_weights[iteration] = max(floor, 1.0 - float(win_rate))

        if all(weight <= floor + eps for weight in raw_weights.values()):
            uniform = 1.0 / len(self.entries)
            return {entry.iteration: uniform for entry in self.entries}

        total = sum(raw_weights.values())
        if total <= 0:
            uniform = 1.0 / len(self.entries)
            return {entry.iteration: uniform for entry in self.entries}
        return {iteration: weight / total for iteration, weight in raw_weights.items()}

    # ------------------------------------------------------------------
    # cycling alarm

    @staticmethod
    def cycling_alarm(per_opponent_win_rate_history: dict[int, list[float]]) -> list[int]:
        """Return opponent iterations whose last 3 win-rate entries strictly decline.

        Strict definition matches backlog item 12 / the loop promotion gate:
        each of the last 3 win-rate entries strictly worse than the prior
        one. Empty list ⇒ no cycling.
        """

        flagged: list[int] = []
        for iteration, history in per_opponent_win_rate_history.items():
            if not history or len(history) < 3:
                continue
            tail = history[-3:]
            if tail[0] > tail[1] > tail[2]:
                flagged.append(int(iteration))
        return sorted(flagged)

    # ------------------------------------------------------------------
    # persistence

    def to_json(self, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {"entries": [entry.to_dict() for entry in self.entries]},
                indent=2,
            )
            + "\n",
            encoding="utf8",
        )

    @classmethod
    def from_json(cls, path: Path) -> "OpponentPool":
        if not path.exists():
            return cls()
        payload = json.loads(path.read_text(encoding="utf8"))
        entries = [PoolEntry.from_dict(item) for item in payload.get("entries", [])]
        return cls(entries=sorted(entries, key=lambda e: e.iteration))
