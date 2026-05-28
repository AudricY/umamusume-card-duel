"""v3.7 Phase A catalog-coverage smoke.

Reads `shared/src/data/cards.json` directly, runs the Python classifiers
on every tool / ability card, and asserts the per-kind catalog counts
match the v3.7 scope vocab exactly. Cross-language byte parity is
asserted indirectly: the Rust integration test
`engine-rs/crates/engine/tests/catalog_effect_kinds_parity.rs` runs the
same per-kind count check from the Rust side. If both pass, both
classifiers agree on the catalog count breakdown.

Asserts:
1. No card triggers an exception in the classifier.
2. Tool counts: HealAtTurnEnd=1, DamageReduction=1, CounterDamage=1.
3. Ability counts: HealOnTurnStart=1, DamageReduction=4, HpBonus=2,
   EnergyAcceleration=1, ConditionalAttackBonus=2, DirectDamage=1,
   RetreatModifier=3, Other=3.
4. Tool catalog total = 3 (BASE cards).
5. Ability catalog total = 17 (BASE cards).

Failures print the actual breakdown so future card additions surface as
loud diff vs scope expectations rather than silent re-bucketing.

See `docs/ai-research/scoping/v37-combat-arith-and-catalog-scoping.md`
§3.4.1 + §4.5 Channels 5/6 + §13.5.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from uma_ai.effect_kinds import (
    AbilityEffectKind,
    ToolEffectKind,
    classify_active_ability,
    classify_tool_effect,
    load_catalog,
)


EXPECTED_TOOL_COUNTS: dict[ToolEffectKind, int] = {
    ToolEffectKind.HEAL_AT_TURN_END: 1,
    ToolEffectKind.DAMAGE_REDUCTION: 1,
    ToolEffectKind.COUNTER_DAMAGE: 1,
}
EXPECTED_ABILITY_COUNTS: dict[AbilityEffectKind, int] = {
    AbilityEffectKind.HEAL_ON_TURN_START: 1,
    AbilityEffectKind.DAMAGE_REDUCTION: 4,
    AbilityEffectKind.HP_BONUS: 2,
    AbilityEffectKind.ENERGY_ACCELERATION: 1,
    AbilityEffectKind.CONDITIONAL_ATTACK_BONUS: 2,
    AbilityEffectKind.DIRECT_DAMAGE: 1,
    AbilityEffectKind.RETREAT_MODIFIER: 3,
    AbilityEffectKind.OTHER: 3,
}


def classify_catalog(
    catalog: dict[str, dict[str, Any]],
) -> tuple[
    dict[str, ToolEffectKind],
    dict[str, AbilityEffectKind],
    list[str],
]:
    """Classify every tool and ability in the (base) catalog.

    Returns `(tool_kinds, ability_kinds, errors)` where `errors` is a
    list of card_ids whose classification raised. Errors should always
    be empty on a healthy catalog.
    """
    tool_kinds: dict[str, ToolEffectKind] = {}
    ability_kinds: dict[str, AbilityEffectKind] = {}
    errors: list[str] = []

    for card_id, card in catalog.items():
        kind = card.get("kind")
        if kind == "trainer":
            if card.get("trainerType") != "tool":
                continue
            effect = card.get("effect") or {}
            try:
                tool_kinds[card_id] = classify_tool_effect(effect)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{card_id}: tool classifier raised {exc!r}")
        elif kind == "umamusume":
            ability = card.get("ability")
            if ability is None:
                continue
            try:
                ability_kinds[card_id] = classify_active_ability(ability)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{card_id}: ability classifier raised {exc!r}")

    return tool_kinds, ability_kinds, errors


def _format_counts(counts: Counter, enum_cls) -> str:
    parts = []
    for variant in enum_cls:
        parts.append(f"  {variant.name} ({variant.value}) = {counts.get(variant, 0)}")
    return "\n".join(parts)


def main(catalog_path: Path | None = None) -> int:
    catalog = load_catalog(catalog_path)
    tool_kinds, ability_kinds, errors = classify_catalog(catalog)

    if errors:
        print("FAIL: classifier(s) raised on:", file=sys.stderr)
        for line in errors:
            print(f"  {line}", file=sys.stderr)
        return 2

    tool_counts: Counter[ToolEffectKind] = Counter(tool_kinds.values())
    ability_counts: Counter[AbilityEffectKind] = Counter(ability_kinds.values())

    print(f"Tool catalog count: {len(tool_kinds)}")
    print(_format_counts(tool_counts, ToolEffectKind))
    print()
    print(f"Ability catalog count: {len(ability_kinds)}")
    print(_format_counts(ability_counts, AbilityEffectKind))
    print()

    failures: list[str] = []

    # Tool counts
    for kind, expected in EXPECTED_TOOL_COUNTS.items():
        actual = tool_counts.get(kind, 0)
        if actual != expected:
            failures.append(
                f"tool {kind.name}: expected {expected}, got {actual}"
            )
    # Tool unexpected variants
    for kind, actual in tool_counts.items():
        if kind not in EXPECTED_TOOL_COUNTS and actual != 0:
            failures.append(
                f"tool {kind.name}: expected 0 (unmapped variant), got {actual}"
            )

    # Ability counts
    for kind, expected in EXPECTED_ABILITY_COUNTS.items():
        actual = ability_counts.get(kind, 0)
        if actual != expected:
            failures.append(
                f"ability {kind.name}: expected {expected}, got {actual}"
            )
    for kind, actual in ability_counts.items():
        if kind not in EXPECTED_ABILITY_COUNTS and actual != 0:
            failures.append(
                f"ability {kind.name}: expected 0 (unmapped variant), got {actual}"
            )

    # Totals
    if len(tool_kinds) != 3:
        failures.append(
            f"tool catalog total: expected 3, got {len(tool_kinds)}"
        )
    if len(ability_kinds) != 17:
        failures.append(
            f"ability catalog total: expected 17, got {len(ability_kinds)}"
        )

    if failures:
        print("FAIL: catalog count breakdown drifted from v3.7 scope vocab.")
        print("      Update the scope doc (v3.7.1) — do NOT silently re-bucket.")
        for line in failures:
            print(f"  - {line}")
        # Print full per-card listing to aid scope-doc updates.
        print("\nPer-card tool classifications:")
        for cid in sorted(tool_kinds):
            print(f"  {cid}\t{tool_kinds[cid].name} ({tool_kinds[cid].value})")
        print("\nPer-card ability classifications:")
        for cid in sorted(ability_kinds):
            print(f"  {cid}\t{ability_kinds[cid].name} ({ability_kinds[cid].value})")
        return 1

    print("PASS: v3.7 catalog coverage matches scope vocab on all variants.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
