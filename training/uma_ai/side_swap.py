"""T2.6 side-swap symmetry augmentation.

`swap_observation` returns a deep-copied, perspective-swapped `PublicObservation`:
the "own" and "opponent" views are exchanged so a state seen from the player's
seat is re-expressed from the opponent's seat. The game is zero-sum, so the
ground-truth value satisfies `value(swap(s)) = -value(s)` — that identity is the
only thing the augmentation relies on (the negated value target is applied at the
dataset layer, not here).

Design constraints / what this module does NOT try to solve
-----------------------------------------------------------
* TRUE INVOLUTION. `swap_observation(swap_observation(x))` must deep-equal `x`.
  Every field this function touches is swapped with an exact symmetric partner,
  so a second application restores the original. Fields that have no symmetric
  partner (see below) are left in place — leaving them fixed is also an
  involution (identity on those slots).

* HIDDEN-INFO ASYMMETRY (the action-permutation problem). The observation is
  intentionally asymmetric:
    - `own.handCardIds` / `cardIdsByZone.ownHand` exist; there is NO opponent
      hand (it is hidden information). After a perspective swap the new "own"
      side would need the opponent's concealed hand, which the observation never
      carried. We therefore leave `cardIdsByZone.ownHand` in place (it now holds
      stale data relative to the swapped seats).
    - `legalActions` are NOT a function this module can permute correctly,
      because the swapped seat's legal action set / action ordering is not
      recoverable from a public observation alone.
  Both of these are exactly why the dataset layer trains the swapped copy
  VALUE-ONLY (policy sample-weight forced to 0). The swap is sound for the value
  head (zero-sum identity) and deliberately ignored for the policy head.

Feature-slot coverage (verified against `training/uma_ai/features.py`)
----------------------------------------------------------------------
The frozen state builders read these own/opp-polarized inputs; swapping the
sub-dicts / scalars below makes `encode_state(swap(obs))` the correct
opponent-seat encoding:
  * `own` <-> `opponent` sub-dicts (drives every `_side_*` / `_identity_*` /
    `_energy_*` / per-Uma / turnState / used* slot, the v3.x tails, and the
    per-Uma slot tokens via `observation_to_uma_slots`).
  * `sideToAct` "player" <-> "opponent" (state feature slot 1).
  * `firstPlayer` "player" <-> "opponent" — drives the slot-96
    `_first_player_polarity(observation, side)` sign (own=+1 / opp=-1).
  * `shared.currentSide` "player" <-> "opponent" (kept consistent with
    `sideToAct`; some downstream consumers read it).
  * `temporal.ownTurnsTaken` <-> `temporal.opponentTurnsTaken` and
    `temporal.ownIsFirstTurn` <-> `temporal.opponentIsFirstTurn` (state slots
    110-113 in v3.1+).
  * `cardIdsByZone`: the symmetric zone pairs `ownActive<->oppActive`,
    `ownBench<->oppBench`, `ownDiscard<->oppDiscard`. `stadium` is neutral
    (unchanged); `ownHand` has no opponent partner (unchanged — see hidden-info
    note above).

Top-level `phase`, `turnNumber`, `pendingChoiceKind`, `schemaVersion`, and
`shared.stadiumCardId` / `shared.gameOver` are seat-neutral and untouched.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import numpy as np
from torch.utils.data import Dataset


def _feature_helpers():
    """Lazily import the frozen featurizers.

    Deferred so this module imports cleanly when run as a bare script
    (`python training/uma_ai/side_swap.py` for the involution self-test),
    where the package-relative `.features` import is unavailable. The dataset
    wrapper only needs these at sample-build time, by which point the package
    is importable. Falls back to the absolute path for script-mode callers.
    """

    try:
        from .features import (
            feature_builder_for_state_dim,
            observation_to_card_ids,
            observation_to_uma_slots,
        )
    except ImportError:  # pragma: no cover - script-mode fallback
        from uma_ai.features import (  # type: ignore[no-redef]
            feature_builder_for_state_dim,
            observation_to_card_ids,
            observation_to_uma_slots,
        )
    return (
        feature_builder_for_state_dim,
        observation_to_card_ids,
        observation_to_uma_slots,
    )

# "player" <-> "opponent" string polarity flip. SideId is exactly this 2-arm
# union in `frontend/src/game/engine/ai-policy/types.ts` (re-exported from
# `shared/src/types`); any unknown string is passed through unchanged so a
# malformed row never crashes the augmentation (it just stays un-flipped, which
# is still an involution on that field).
_SIDE_FLIP = {"player": "opponent", "opponent": "player"}

# `cardIdsByZone` symmetric zone pairs. `ownHand` and `stadium` are deliberately
# absent — `ownHand` has no opponent partner (hidden info) and `stadium` is
# seat-neutral; both stay in place so the double-swap restores them.
_ZONE_SWAP_PAIRS = (
    ("ownActive", "oppActive"),
    ("ownBench", "oppBench"),
    ("ownDiscard", "oppDiscard"),
)


def _flip_side(value: Any) -> Any:
    """Flip a SideId-like string; pass through anything else untouched."""

    if isinstance(value, str):
        return _SIDE_FLIP.get(value, value)
    return value


def _swap_pair(container: dict[str, Any], key_a: str, key_b: str) -> None:
    """Exchange `container[key_a]` and `container[key_b]` in place.

    Only swaps when BOTH keys are present so a partial observation stays an
    involution (a one-sided swap would not be its own inverse). Missing both
    keys is a no-op; missing exactly one leaves the dict unchanged.
    """

    if key_a in container and key_b in container:
        container[key_a], container[key_b] = container[key_b], container[key_a]


def swap_observation(obs: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-copied, perspective-swapped `PublicObservation`.

    A true involution: `swap_observation(swap_observation(x))` deep-equals `x`.
    See the module docstring for the full field coverage + hidden-info caveats.
    """

    swapped = copy.deepcopy(obs)

    # Top-level seat scalars.
    if "sideToAct" in swapped:
        swapped["sideToAct"] = _flip_side(swapped["sideToAct"])
    if "firstPlayer" in swapped:
        swapped["firstPlayer"] = _flip_side(swapped["firstPlayer"])

    # `own` <-> `opponent` side sub-dicts (the bulk of the state encoding).
    _swap_pair(swapped, "own", "opponent")

    # `temporal` own/opp-polarized fields.
    temporal = swapped.get("temporal")
    if isinstance(temporal, dict):
        _swap_pair(temporal, "ownTurnsTaken", "opponentTurnsTaken")
        _swap_pair(temporal, "ownIsFirstTurn", "opponentIsFirstTurn")

    # `shared.currentSide` tracks whose turn it is — flip to stay consistent
    # with `sideToAct`. `stadiumCardId` / `gameOver` are seat-neutral.
    shared = swapped.get("shared")
    if isinstance(shared, dict) and "currentSide" in shared:
        shared["currentSide"] = _flip_side(shared["currentSide"])

    # `cardIdsByZone` symmetric zone pairs. `ownHand` / `stadium` untouched.
    zones = swapped.get("cardIdsByZone")
    if isinstance(zones, dict):
        for key_a, key_b in _ZONE_SWAP_PAIRS:
            _swap_pair(zones, key_a, key_b)

    return swapped


def make_side_swapped_sample(
    sample: Any,
    *,
    state_dim: int,
    uses_uma_slot_tokens: bool,
    ablations: set[str] | None = None,
) -> Any:
    """Return a value-only, perspective-swapped copy of a dataset sample.

    Works for any of the frozen sample dataclasses (`PolicySample`,
    `MctsSelfPlaySample`, `RebelSelfPlaySample`) — they all carry `example`
    (with the raw `observation`) plus the re-derivable feature fields. The
    copy:
      * re-featurizes `state_features` from `swap_observation(obs)` via the
        SAME frozen builder the base dataset used (`state_dim`-keyed);
      * re-derives `card_ids_by_zone` / `uma_slot_*` from the swapped obs when
        the base sample carried them (preserves the all-or-nothing collator
        contract — a base sample with `None` stays `None`);
      * NEGATES `value_target` (zero-sum: value(swap(s)) = -value(s));
      * forces `policy_loss_scale = 0.0` so the swapped copy trains value-only
        (the action set / ordering is not recoverable under a public swap);
      * leaves `action_features` / `target_index` / `policy_target` /
        `q_target` untouched — they are simply ignored by the zeroed policy
        loss, and re-using them avoids fabricating an action permutation.

    The returned object is the SAME dataclass type as `sample`, so the base
    dataset's collator handles it unchanged.
    """

    (
        feature_builder_for_state_dim,
        observation_to_card_ids,
        observation_to_uma_slots,
    ) = _feature_helpers()
    obs = sample.example.get("observation", {}) or {}
    swapped_obs = swap_observation(obs)
    encode_state = feature_builder_for_state_dim(state_dim)
    state_features = encode_state(swapped_obs, ablations=ablations)

    updates: dict[str, Any] = {
        "state_features": np.ascontiguousarray(state_features, dtype=np.float32),
        "value_target": -float(sample.value_target),
        "policy_loss_scale": 0.0,
    }

    # Re-derive the optional embedding / slot tensors only when the base
    # sample carried them, so the collator's all-or-nothing emit is preserved.
    if getattr(sample, "card_ids_by_zone", None) is not None:
        updates["card_ids_by_zone"] = observation_to_card_ids(swapped_obs)
    if (
        getattr(sample, "uma_slot_card_ids", None) is not None
        and uses_uma_slot_tokens
    ):
        slot_ids, slot_feats = observation_to_uma_slots(swapped_obs)
        updates["uma_slot_card_ids"] = slot_ids
        updates["uma_slot_features"] = slot_feats

    return dataclasses.replace(sample, **updates)


class SideSwapAugmentedDataset(Dataset):
    """Opt-in side-swap augmentation wrapper (T2.6).

    Length is 2x the base dataset. Even global indices map to the original
    sample; odd indices map to its value-only, perspective-swapped copy. The
    swapped copies are materialized lazily and cached so the involution work
    happens at most once per row.

    The wrapper is collator-agnostic: it yields the same dataclass type the
    base dataset yields, so the existing `collate_*` functions pack the new
    `policy_loss_scale` field via their `getattr(..., 1.0)` fallback.
    """

    def __init__(
        self,
        base: Dataset,
        *,
        state_dim: int,
        uses_uma_slot_tokens: bool,
        ablations: set[str] | None = None,
    ) -> None:
        self.base = base
        self.state_dim = state_dim
        self.uses_uma_slot_tokens = uses_uma_slot_tokens
        self.ablations = ablations or set()
        self._swap_cache: dict[int, Any] = {}
        # Expose `.samples` so split_dataset / summarize_dataset / diagnostics
        # (which read `dataset.samples[i].example`) keep working. The swapped
        # copies share the original `example` dict, so grouping/metrics are
        # keyed identically to the base row — intentional (the swap is a
        # value-only mirror of the same episode/state).
        self.samples = getattr(base, "samples", None)

    def __len__(self) -> int:
        return len(self.base) * 2  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> Any:
        base_index, is_swapped = divmod(index, 2)
        sample = self.base[base_index]
        if not is_swapped:
            return sample
        cached = self._swap_cache.get(base_index)
        if cached is None:
            cached = make_side_swapped_sample(
                sample,
                state_dim=self.state_dim,
                uses_uma_slot_tokens=self.uses_uma_slot_tokens,
                ablations=self.ablations,
            )
            self._swap_cache[base_index] = cached
        return cached


def _sample_observation() -> dict[str, Any]:
    """A representative `PublicObservation` exercising every swapped field."""

    return {
        "schemaVersion": 3,
        "sideToAct": "player",
        "phase": "combat",
        "turnNumber": 4,
        "firstPlayer": "player",
        "pendingChoiceKind": None,
        "temporal": {
            "ownTurnsTaken": 2,
            "opponentTurnsTaken": 1,
            "ownIsFirstTurn": False,
            "opponentIsFirstTurn": True,
        },
        "own": {
            "id": "player",
            "points": 1,
            "handCount": 3,
            "deckCount": 40,
            "discard": ["card-a"],
            "active": {"cardId": "uma-own-active", "hp": 90, "energies": {"fire": 1}},
            "bench": [{"cardId": "uma-own-bench", "hp": 70, "energies": {}}, None, None],
            "energyZone": ["fire", "water"],
            "usedSupporterThisTurn": True,
            "usedRetreatThisTurn": False,
            "usedStadiumThisTurn": False,
            "turnState": {"energyAttachmentsThisTurn": 1},
        },
        "opponent": {
            "id": "opponent",
            "points": 2,
            "handCount": 5,
            "deckCount": 38,
            "discard": ["card-b", "card-c"],
            "active": {"cardId": "uma-opp-active", "hp": 60, "energies": {"water": 2}},
            "bench": [{"cardId": "uma-opp-bench", "hp": 80, "energies": {}}, None, None],
            "energyZone": ["psychic"],
            "usedSupporterThisTurn": False,
            "usedRetreatThisTurn": True,
            "usedStadiumThisTurn": False,
            "turnState": {"energyAttachmentsThisTurn": 0},
        },
        "shared": {
            "stadiumCardId": "stadium-x",
            "currentSide": "player",
            "gameOver": False,
        },
        "cardIdsByZone": {
            "ownActive": [11],
            "oppActive": [21],
            "ownBench": [12, 0, 0, 0],
            "oppBench": [22, 0, 0, 0],
            "ownHand": [31, 32, 33, 0, 0, 0, 0, 0, 0, 0],
            "ownDiscard": [41],
            "oppDiscard": [51, 52],
            "stadium": [61],
        },
    }


def _self_test() -> None:
    """Assert the involution + that the single swap actually changes polarity."""

    original = _sample_observation()
    once = swap_observation(original)

    # 1) A single swap must NOT mutate the input (deep copy contract).
    assert original == _sample_observation(), "swap_observation mutated its input"

    # 2) A single swap must actually flip seats (not a silent no-op).
    assert once != original, "swap_observation produced an identical observation"
    assert once["sideToAct"] == "opponent"
    assert once["firstPlayer"] == "opponent"
    assert once["shared"]["currentSide"] == "opponent"
    assert once["own"]["active"]["cardId"] == "uma-opp-active"
    assert once["opponent"]["active"]["cardId"] == "uma-own-active"
    assert once["temporal"]["ownTurnsTaken"] == 1
    assert once["temporal"]["opponentTurnsTaken"] == 2
    assert once["temporal"]["ownIsFirstTurn"] is True
    assert once["temporal"]["opponentIsFirstTurn"] is False
    assert once["cardIdsByZone"]["ownActive"] == [21]
    assert once["cardIdsByZone"]["oppActive"] == [11]
    # Hidden-info / neutral zones stay fixed.
    assert once["cardIdsByZone"]["ownHand"] == [31, 32, 33, 0, 0, 0, 0, 0, 0, 0]
    assert once["cardIdsByZone"]["stadium"] == [61]
    # Seat-neutral scalars untouched.
    assert once["phase"] == "combat"
    assert once["turnNumber"] == 4

    # 3) THE INVOLUTION: swap(swap(x)) deep-equals x.
    twice = swap_observation(once)
    assert twice == original, "swap_observation is not an involution"

    print("side_swap self-test OK: involution holds and polarity flips.")


if __name__ == "__main__":
    _self_test()
