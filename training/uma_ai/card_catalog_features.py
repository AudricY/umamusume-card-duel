"""Catalog-grounded card feature table for the v6 relational scheme (T1.1).

The card-embedding table (`CandidatePolicyNet.card_embed`) is identity-only and
randomly initialized — it must learn each card's mechanics (attack damage/cost,
HP, weakness, type, stage, effect-kind) purely from RL gradient, which is weak
for the long tail of rarely-seen cards. The relational trunk leans on that
table for EVERY token (card / slot / action), so a structured starting geometry
compounds across the whole model.

This module builds a deterministic, vocab-indexed `[CARD_VOCAB_TABLE_SIZE, K]`
matrix of static catalog mechanics read from `shared/src/data/cards.json` (the
same source the featurizer + engine use). The model fuses a learned projection
of this matrix into its card tokens — so "what is card X" carries mechanics, not
just identity. The matrix is a CONSTANT (registered buffer): it bakes into the
ONNX graph as an initializer, needs NO new graph input, and requires NO Rust /
serving change.

Row 0 is the pad/unknown slot (all zeros), matching `card_embed`'s
`padding_idx=0`. Rows 1..vocabSize are indexed by `card_vocab_index`.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .effect_kinds import (
    AbilityEffectKind,
    ToolEffectKind,
    classify_active_ability,
    classify_tool_effect,
)
from .features import _card_vocab, _get_card

# 10-energy / 10-type order, matching the featurizer's `_UMA_SLOT_ENERGY_TYPES`
# and `_energy_vector` ordering. Uma `type` / weakness `type` are TitleCase in
# the catalog; energy cost keys are lowercase — both normalize to this set.
_TYPE_ORDER: tuple[str, ...] = (
    "grass",
    "fire",
    "water",
    "lightning",
    "psychic",
    "fighting",
    "darkness",
    "steel",
    "colorless",
    "dragon",
)
_TYPE_INDEX = {name: i for i, name in enumerate(_TYPE_ORDER)}
_TRAINER_TYPES: tuple[str, ...] = ("supporter", "item", "tool", "stadium")
_TRAINER_TYPE_INDEX = {name: i for i, name in enumerate(_TRAINER_TYPES)}

# Normalization caps (loose; the table is a prior, not a calibrated input).
_HP_CAP = 200.0
_DAMAGE_CAP = 200.0
_COST_CAP = 6.0
_ATTACK_COUNT_CAP = 3.0
_STAGE_CAP = 2.0
_WEAKNESS_CAP = 60.0
_RETREAT_CAP = 3.0


def _type_one_hot(type_name: str | None) -> np.ndarray:
    out = np.zeros(len(_TYPE_ORDER), dtype=np.float32)
    if not type_name:
        return out
    idx = _TYPE_INDEX.get(str(type_name).strip().lower())
    if idx is not None:
        out[idx] = 1.0
    return out


def _parse_retreat(value) -> float:
    """'Empty'->0, 'Colorless'->1, 'Colorless x2'->2, 'Colorless x3'->3."""
    if value is None:
        return 0.0
    text = str(value).strip().lower()
    if not text or text == "empty":
        return 0.0
    if "x" in text:
        try:
            return float(text.split("x")[-1].strip())
        except ValueError:
            return 1.0
    return 1.0


def _attack_stats(card: dict) -> tuple[float, float, float]:
    """(max base damage, min total energy cost, attack count)."""
    attacks = card.get("attacks") or []
    if not attacks:
        return 0.0, 0.0, 0.0
    max_dmg = 0.0
    min_cost = None
    for atk in attacks:
        if not isinstance(atk, dict):
            continue
        try:
            max_dmg = max(max_dmg, float(atk.get("damage", 0) or 0))
        except (TypeError, ValueError):
            pass
        cost = atk.get("cost") or {}
        total = float(sum(float(v or 0) for v in cost.values()))
        min_cost = total if min_cost is None else min(min_cost, total)
    return max_dmg, (min_cost or 0.0), float(len(attacks))


def _card_feature_row(card: dict | None) -> np.ndarray:
    """Build one catalog-feature row for a resolved base card (or zeros)."""
    type_oh = np.zeros(len(_TYPE_ORDER), dtype=np.float32)
    weak_oh = np.zeros(len(_TYPE_ORDER), dtype=np.float32)
    scalars: list[float] = [0.0] * 6  # stage, hp, max_dmg, min_cost, n_atk, weak_amt
    retreat = 0.0
    is_uma = 0.0
    is_trainer = 0.0
    trainer_oh = np.zeros(len(_TRAINER_TYPES), dtype=np.float32)
    tool_oh = np.zeros(len(ToolEffectKind), dtype=np.float32)
    ability_oh = np.zeros(len(AbilityEffectKind), dtype=np.float32)

    if card is not None:
        kind = str(card.get("kind", "") or "")
        if kind == "umamusume":
            is_uma = 1.0
            type_oh = _type_one_hot(card.get("type"))
            max_dmg, min_cost, n_atk = _attack_stats(card)
            scalars = [
                min(float(card.get("stage", 0) or 0), _STAGE_CAP) / _STAGE_CAP,
                min(float(card.get("hp", 0) or 0), _HP_CAP) / _HP_CAP,
                min(max_dmg, _DAMAGE_CAP) / _DAMAGE_CAP,
                min(min_cost, _COST_CAP) / _COST_CAP,
                min(n_atk, _ATTACK_COUNT_CAP) / _ATTACK_COUNT_CAP,
                0.0,
            ]
            weakness = card.get("weakness") or {}
            if isinstance(weakness, dict):
                weak_oh = _type_one_hot(weakness.get("type"))
                scalars[5] = min(float(weakness.get("amount", 0) or 0), _WEAKNESS_CAP) / _WEAKNESS_CAP
            retreat = _parse_retreat(card.get("retreat")) / _RETREAT_CAP
            ability = card.get("ability")
            if isinstance(ability, dict):
                ability_oh[int(classify_active_ability(ability))] = 1.0
        elif kind == "trainer":
            is_trainer = 1.0
            tt = str(card.get("trainerType", "") or "").lower()
            if tt in _TRAINER_TYPE_INDEX:
                trainer_oh[_TRAINER_TYPE_INDEX[tt]] = 1.0
            effect = card.get("effect")
            if isinstance(effect, dict):
                tool_oh[int(classify_tool_effect(effect))] = 1.0

    return np.concatenate(
        [
            type_oh,
            np.asarray(scalars, dtype=np.float32),
            np.asarray([retreat, is_uma, is_trainer], dtype=np.float32),
            weak_oh,
            trainer_oh,
            tool_oh,
            ability_oh,
        ]
    ).astype(np.float32)


# Feature dim = 10 type + 6 scalars + 3 (retreat/is_uma/is_trainer) + 10 weakness
# + 4 trainer-type + 4 tool-kind + 8 ability-kind = 45.
CATALOG_FEATURE_DIM = (
    len(_TYPE_ORDER) + 6 + 3 + len(_TYPE_ORDER) + len(_TRAINER_TYPES)
    + len(ToolEffectKind) + len(AbilityEffectKind)
)


@lru_cache(maxsize=1)
def build_catalog_feature_table(vocab_table_size: int) -> np.ndarray:
    """Return a deterministic `[vocab_table_size, CATALOG_FEATURE_DIM]` matrix.

    Row 0 = pad (zeros). Row `i` (1..vocabSize) = catalog mechanics for the
    card whose `card_vocab_index` is `i`, resolving full-art / variant ids to
    their base card via `_get_card`. Cards absent from the catalog get a zero
    row (degrades gracefully, same as an unknown id)."""

    table = np.zeros((vocab_table_size, CATALOG_FEATURE_DIM), dtype=np.float32)
    vocab = _card_vocab()
    index_by_id = vocab.get("indexById", {}) or {}
    for card_id, idx in index_by_id.items():
        idx = int(idx)
        if idx <= 0 or idx >= vocab_table_size:
            continue
        table[idx] = _card_feature_row(_get_card(str(card_id)))
    return table


if __name__ == "__main__":
    # Self-check: build the table and report coverage.
    from .model import CARD_VOCAB_TABLE_SIZE  # local import to avoid cycle at module load

    tbl = build_catalog_feature_table(CARD_VOCAB_TABLE_SIZE)
    nonzero_rows = int((np.abs(tbl).sum(axis=1) > 0).sum())
    print(f"catalog table shape={tbl.shape} CATALOG_FEATURE_DIM={CATALOG_FEATURE_DIM}")
    print(f"non-zero (covered) rows={nonzero_rows} / {CARD_VOCAB_TABLE_SIZE - 1} cards")
    assert tbl.shape == (CARD_VOCAB_TABLE_SIZE, CATALOG_FEATURE_DIM)
    assert np.abs(tbl[0]).sum() == 0.0, "row 0 (pad) must be zero"
    assert nonzero_rows > 50, "expected most of the 107-card vocab to resolve"
    print("OK")
