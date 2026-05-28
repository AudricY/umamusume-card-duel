"""Frozen effect-kind enums + pure classifiers for v3.7 Channels 5 + 6.

Python mirror of `engine-rs/crates/engine/src/core/effect_kinds.rs`. The
integer discriminants and classifier precedence are locked at v3.7 scope
time; cross-language byte parity is asserted by
`training/v37_catalog_coverage_smoke.py` and
`engine-rs/crates/engine/tests/catalog_effect_kinds_parity.rs`.

`ToolEffectKind` (4 variants) and `AbilityEffectKind` (8 variants) are
read from JSON camelCase field names (the catalog at
`shared/src/data/cards.json` is the source of truth for both languages).

See `docs/ai-research/scoping/v37-combat-arith-and-catalog-scoping.md`
§3.4.1 + §4.5 Channels 5/6 + §13.5/§13.7 for the locked vocab
definitions and recon corrections.
"""

from __future__ import annotations

import json
from enum import IntEnum
from functools import lru_cache
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Enums (byte-identical to Rust `#[repr(u8)]` discriminants)
# ---------------------------------------------------------------------------


class ToolEffectKind(IntEnum):
    """Tool effect-kind one-hot vocabulary (v3.7 Channel 5)."""

    HEAL_AT_TURN_END = 0
    DAMAGE_REDUCTION = 1
    COUNTER_DAMAGE = 2
    OTHER = 3


class AbilityEffectKind(IntEnum):
    """Active ability effect-kind one-hot vocabulary (v3.7 Channel 6)."""

    HEAL_ON_TURN_START = 0
    DAMAGE_REDUCTION = 1
    HP_BONUS = 2
    ENERGY_ACCELERATION = 3
    CONDITIONAL_ATTACK_BONUS = 4
    DIRECT_DAMAGE = 5
    RETREAT_MODIFIER = 6
    OTHER = 7


# ---------------------------------------------------------------------------
# Classifiers (pure functions over the catalog dict)
# ---------------------------------------------------------------------------


def classify_tool_effect(effect: dict[str, Any]) -> ToolEffectKind:
    """Map a `TrainerEffect` JSON dict to a `ToolEffectKind`.

    Precedence (highest first; first match wins):
        HealAtTurnEnd > DamageReduction > CounterDamage > Other.

    Today's catalog has no multi-flag tools; precedence is preventative
    against silent class drift on future cards.
    """
    if effect is None:
        return ToolEffectKind.OTHER
    if effect.get("toolEndTurnHealActive") is not None:
        return ToolEffectKind.HEAL_AT_TURN_END
    if effect.get("toolDamageReduction") is not None:
        return ToolEffectKind.DAMAGE_REDUCTION
    if effect.get("toolCounterDamage") is not None:
        return ToolEffectKind.COUNTER_DAMAGE
    return ToolEffectKind.OTHER


def classify_active_ability(ability: dict[str, Any]) -> AbilityEffectKind:
    """Map an `Ability` JSON dict to an `AbilityEffectKind`.

    Precedence (highest first; first match wins) mirrors the variant
    declaration order:
        HealOnTurnStart > DamageReduction > HpBonus > EnergyAcceleration
        > ConditionalAttackBonus > DirectDamage > RetreatModifier > Other.
    """
    if ability is None:
        return AbilityEffectKind.OTHER
    if ability.get("heal") is not None:
        return AbilityEffectKind.HEAL_ON_TURN_START
    if ability.get("damageReduction") is not None:
        return AbilityEffectKind.DAMAGE_REDUCTION
    if ability.get("activeHpBonus") is not None:
        return AbilityEffectKind.HP_BONUS
    if ability.get("moveBenchedEnergyToActive") is not None:
        return AbilityEffectKind.ENERGY_ACCELERATION
    if (
        ability.get("attackDamageBonusIfAttachedEnergy") is not None
        or ability.get("attackDamageBonusIfEvolvedLastTurn") is not None
    ):
        return AbilityEffectKind.CONDITIONAL_ATTACK_BONUS
    if ability.get("damageOpponent") is not None:
        return AbilityEffectKind.DIRECT_DAMAGE
    if (
        ability.get("retreatCostZeroIfTookDamageLastTurn") is not None
        or ability.get("retreatCostZeroIfHasEnergy") is not None
    ):
        return AbilityEffectKind.RETREAT_MODIFIER
    return AbilityEffectKind.OTHER


# ---------------------------------------------------------------------------
# Catalog loader (helper for the smoke + v3.7 featurizer)
# ---------------------------------------------------------------------------


_DEFAULT_CATALOG_PATH = (
    Path(__file__).resolve().parents[2] / "shared" / "src" / "data" / "cards.json"
)


@lru_cache(maxsize=4)
def _load_catalog_cached(path_str: str) -> dict[str, dict[str, Any]]:
    raw = json.loads(Path(path_str).read_text(encoding="utf8"))
    base = raw.get("baseCards", {})
    return dict(base)


def load_catalog(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Read `shared/src/data/cards.json` and return `{card_id: card_data}`.

    Returns the **base-card** dict (variant expansion mirrors Rust
    `Card::variant == Base`). Variant clones share the same `ability` /
    `effect` payload as their base; counting variant expansion would
    double-count abilities.
    """
    target = Path(path) if path is not None else _DEFAULT_CATALOG_PATH
    return _load_catalog_cached(str(target))
