"""Smoke the weakness-bonus correction in `features._can_ko`.

The featurizer previously ignored `weakness_bonus`, causing can_ko to under-
report KOs on ~30% of matchups (every type-disadvantaged defender). This
smoke exercises the corrected branch with the canonical test pair:

  - Attacker `manhattanCafeStage1` — Darkness type, 40 printed damage,
    attack cost {darkness: 1, colorless: 1}.
  - Defender `matikanetannhauserBasic` — Psychic type, 60 hp,
    weakness {Darkness, +20}.

Without weakness: 40 < 60 → can_ko returns 0.
With weakness:    40 + 20 = 60 >= 60 → can_ko returns 1.

The Rust mirror in `engine-rs/.../policy/featurize.rs` has equivalent unit
tests (`can_ko_*` in the `tests` module). Together they form the parity
contract for v33-correctness-fix Fix 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uma_ai.features import _can_ko  # noqa: E402


def make_entry(card_id: str, hp: int, energies: dict[str, int] | None = None) -> dict:
    e = energies or {}
    return {
        "cardId": card_id,
        "hp": hp,
        "energies": e,
        "energyTotal": sum(e.values()),
    }


def main() -> int:
    # Case 1: weakness lethal at threshold.
    attacker = make_entry(
        "manhattanCafeStage1",
        hp=90,
        energies={"darkness": 1, "colorless": 1},
    )
    defender = make_entry("matikanetannhauserBasic", hp=60)
    result = _can_ko(attacker, defender)
    if result != 1.0:
        raise SystemExit(
            f"FAIL: weakness threshold KO. Expected 1.0 (40+20 weakness == 60 hp); got {result}"
        )

    # Case 2: weakness does NOT apply when types don't match. Psychic attacker
    # vs Psychic defender (weakness Darkness). 20 dmg < 90 hp → no KO.
    attacker = make_entry("matikanetannhauserBasic", hp=60, energies={"psychic": 1})
    defender = make_entry("haruUraraBasic", hp=90)
    result = _can_ko(attacker, defender)
    if result != 0.0:
        raise SystemExit(
            f"FAIL: type-mismatch should not apply weakness. Expected 0.0; got {result}"
        )

    # Case 3: zero damage (attacker not attack-ready) bypasses the weakness
    # branch — mirror of simulator rule that weakness only applies when
    # damage > 0.
    attacker = make_entry("manhattanCafeStage1", hp=90, energies={})
    defender = make_entry("matikanetannhauserBasic", hp=60)
    result = _can_ko(attacker, defender)
    if result != 0.0:
        raise SystemExit(
            f"FAIL: zero-energy attacker should not KO. Expected 0.0; got {result}"
        )

    # Case 4: pre-fix baseline check — confirm that WITHOUT the weakness
    # correction, the same matchup would have returned 0. We can't easily
    # toggle the fix off, so instead we exercise a case where weakness
    # makes NO difference (40 dmg vs 30 hp). Result should be 1 either way.
    attacker = make_entry(
        "manhattanCafeStage1",
        hp=90,
        energies={"darkness": 1, "colorless": 1},
    )
    defender = make_entry("matikanetannhauserBasic", hp=30)
    result = _can_ko(attacker, defender)
    if result != 1.0:
        raise SystemExit(
            f"FAIL: trivial overkill. Expected 1.0 (40 > 30); got {result}"
        )

    print("v33_can_ko_weakness_smoke: 4/4 cases PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
