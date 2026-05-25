"""Smoke the v3.6 priors-and-arithmetic builder (`observation_to_features_v3_6`).

v3.6 = v3.5 head (212-d) with [197:207] band repurposed in-place (own.energy_pool
typed multihot replacing the dead opp.energy_zone.front) PLUS 34 new bits at
[212:246]:

  [197:207] own.energy_pool typed multihot (10) — REPURPOSED IN-BAND from the
            dead v3.5 opp.energy_zone.front band.
  [212:222] opp.energy_pool typed multihot (10) — same channel.
  [222:226] own prize one-hot ×4 over remaining-prize {3,2,1,0}.
  [226:230] opp prize one-hot ×4 over remaining-prize {3,2,1,0}.
  [230:240] opp bench typed energy aggregate (10) — multihot.
  [240]     own_lethal_next_turn (face-value).
  [241]     opp_lethal_next_turn (face-value).
  [242]     own_secondary_attack_usable.
  [243]     own_secondary_attack_would_KO.
  [244]     opp_secondary_attack_usable.
  [245]     opp_secondary_attack_would_KO.

See `docs/ai-research/scoping/v36-priors-and-arithmetic-scoping.md` §4 step 2
and §4.5 for the LOCKED layout and synthesis-bit definitions. Own bench typed
aggregate dropped at impl reconciliation (54-bit channel breakdown vs locked
+34 net target; own bench partly redundant with v3.5 head energies).

Asserts:
1. Output shape is (246,).
2. Slots [0:197] are byte-identical to `observation_to_features_v3_5` on the
   same observation (v3.5 byte-stability invariant).
3. Tail / repurposed-band slots respond correctly to mutated observation
   fields (energy_pool, prize one-hot, bench typed, lethal, secondary).
4. Schema-dispatch returns `observation_to_features_v3_6` for state_dim=246
   and `schema_version_for_state_dim(246) == STATE_FEATURE_SCHEMA_VERSION_V3_6`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uma_ai.features import (  # noqa: E402
    STATE_DIM_V3_5,
    STATE_DIM_V3_6,
    STATE_FEATURE_SCHEMA_VERSION_V3_6,
    _UMA_SLOT_ENERGY_TYPES,
    feature_builder_for_state_dim,
    observation_to_features_v3_5,
    observation_to_features_v3_6,
    schema_version_for_state_dim,
)


def make_obs(
    *,
    own_points: int = 0,
    opp_points: int = 0,
    own_energy_pool: list[str] | None = None,
    opp_energy_pool: list[str] | None = None,
    own_active_card: str = "haruUraraBasic",
    opp_active_card: str = "haruUraraBasic",
    own_active_hp: int = 90,
    opp_active_hp: int = 90,
    own_active_energies: dict | None = None,
    opp_active_energies: dict | None = None,
    own_bench: list[dict] | None = None,
    opp_bench: list[dict] | None = None,
) -> dict:
    """Minimal observation suitable for the v3.6 feature builder.

    Adds `energyPool` (Phase 1 obs-contract extension) on top of the v3.5
    smoke's observation shape."""

    def _bench_default() -> list[dict]:
        return [{"cardId": "haruUraraBasic", "hp": 90, "maxHp": 90,
                 "energies": {}, "energyTotal": 0, "specialConditions": [],
                 "stage": 0}]

    return {
        "phase": "stadiumOrEnd",
        "sideToAct": "player",
        "turnNumber": 1,
        "own": {
            "points": own_points,
            "handCount": 5,
            "deckCount": 40,
            "active": {
                "cardId": own_active_card,
                "hp": own_active_hp,
                "maxHp": 90,
                "energies": own_active_energies or {},
                "energyTotal": sum((own_active_energies or {}).values()),
                "specialConditions": [],
            },
            "bench": own_bench if own_bench is not None else [],
            "discard": [],
            "energyZone": [],
            "energyPool": own_energy_pool or [],
            "usedSupporterThisTurn": False,
            "usedRetreatThisTurn": False,
            "usedStadiumThisTurn": False,
            "handCardIds": [],
        },
        "opponent": {
            "points": opp_points,
            "handCount": 5,
            "deckCount": 40,
            "active": {
                "cardId": opp_active_card,
                "hp": opp_active_hp,
                "maxHp": 90,
                "energies": opp_active_energies or {},
                "energyTotal": sum((opp_active_energies or {}).values()),
                "specialConditions": [],
            },
            "bench": opp_bench if opp_bench is not None else [],
            "discard": [],
            "energyZone": [],
            "energyPool": opp_energy_pool or [],
            "usedSupporterThisTurn": False,
            "usedRetreatThisTurn": False,
            "usedStadiumThisTurn": False,
            "handCardIds": [],
        },
        "shared": {"stadiumCardId": ""},
        "temporal": {},
    }


def _energy_idx(name: str) -> int:
    return _UMA_SLOT_ENERGY_TYPES.index(name)


def main() -> int:
    # 1. Shape.
    obs = make_obs()
    v = observation_to_features_v3_6(obs)
    if v.shape != (STATE_DIM_V3_6,):
        raise SystemExit(f"FAIL[1]: expected shape ({STATE_DIM_V3_6},); got {v.shape}")

    # 2. Head byte-identity for [0:197] vs v3.5 (the v3.5 byte-stable head).
    v35 = observation_to_features_v3_5(obs)
    if not np.array_equal(v[:197], v35[:197]):
        raise SystemExit(
            "FAIL[2]: v3.6 head [0:197] not byte-identical to v3.5 builder"
        )

    # 3a. own.energy_pool multihot at [197:207]; opp at [212:222].
    obs_pool = make_obs(own_energy_pool=["fire", "water", "fire"],
                        opp_energy_pool=["psychic"])
    v_pool = observation_to_features_v3_6(obs_pool)
    expected_own = np.zeros(10, dtype=np.float32)
    expected_own[_energy_idx("fire")] = 1.0
    expected_own[_energy_idx("water")] = 1.0
    if not np.array_equal(v_pool[197:207], expected_own):
        raise SystemExit(
            f"FAIL[3a-own]: own pool [fire,water,fire] → {expected_own.tolist()}; "
            f"got {v_pool[197:207].tolist()}"
        )
    expected_opp = np.zeros(10, dtype=np.float32)
    expected_opp[_energy_idx("psychic")] = 1.0
    if not np.array_equal(v_pool[212:222], expected_opp):
        raise SystemExit(
            f"FAIL[3a-opp]: opp pool [psychic] → {expected_opp.tolist()}; "
            f"got {v_pool[212:222].tolist()}"
        )
    # Empty pools → all zeros in both bands (the [197:207] band MUST be
    # zero-overwritten before own pool writes back, so an empty own pool
    # produces all zeros even though v3.5 wrote opp energy_zone bits there).
    obs_empty = make_obs(opp_energy_pool=None, own_energy_pool=None)
    # Also give opp a non-empty energy_zone to ensure v3.5 would have set
    # bits in [197:207] — we want to confirm v3.6 zero-overwrites them.
    obs_empty["opponent"]["energyZone"] = ["fire"]
    v_empty = observation_to_features_v3_6(obs_empty)
    if v_empty[197:207].sum() != 0.0:
        raise SystemExit(
            f"FAIL[3a-overwrite]: empty own pool with non-empty opp.energyZone → "
            f"[197:207] must be zero (zero-overwrite contract); got "
            f"{v_empty[197:207].tolist()}"
        )

    # 3b. Prize one-hot ×4. own [222:226], opp [226:230].
    for own_pts, expected_bit in [(0, 0), (1, 1), (2, 2), (3, 3)]:
        obs_p = make_obs(own_points=own_pts)
        v_p = observation_to_features_v3_6(obs_p)
        own_slice = v_p[222:226]
        if own_slice.sum() != 1.0 or own_slice[expected_bit] != 1.0:
            raise SystemExit(
                f"FAIL[3b-own]: own points={own_pts} → bit {expected_bit} only; "
                f"got {own_slice.tolist()}"
            )
    for opp_pts, expected_bit in [(0, 0), (1, 1), (2, 2), (3, 3)]:
        obs_p = make_obs(opp_points=opp_pts)
        v_p = observation_to_features_v3_6(obs_p)
        opp_slice = v_p[226:230]
        if opp_slice.sum() != 1.0 or opp_slice[expected_bit] != 1.0:
            raise SystemExit(
                f"FAIL[3b-opp]: opp points={opp_pts} → bit {expected_bit} only; "
                f"got {opp_slice.tolist()}"
            )

    # 3c. Opp bench typed aggregate at [230:240].
    # Two bench Umas: one with fire+psychic, one with steel. Bits set: fire,
    # psychic, steel.
    obs_bench = make_obs(opp_bench=[
        {"cardId": "haruUraraBasic", "hp": 90, "maxHp": 90,
         "energies": {"fire": 1, "psychic": 2}, "energyTotal": 3,
         "specialConditions": [], "stage": 0},
        {"cardId": "haruUraraBasic", "hp": 90, "maxHp": 90,
         "energies": {"steel": 1}, "energyTotal": 1,
         "specialConditions": [], "stage": 0},
    ])
    v_bench = observation_to_features_v3_6(obs_bench)
    expected_bench = np.zeros(10, dtype=np.float32)
    for et in ("fire", "psychic", "steel"):
        expected_bench[_energy_idx(et)] = 1.0
    if not np.array_equal(v_bench[230:240], expected_bench):
        raise SystemExit(
            f"FAIL[3c]: opp bench typed aggregate {expected_bench.tolist()}; "
            f"got {v_bench[230:240].tolist()}"
        )
    # Empty bench → all zeros.
    v_no_bench = observation_to_features_v3_6(make_obs())
    if v_no_bench[230:240].sum() != 0.0:
        raise SystemExit(
            f"FAIL[3c-empty]: empty opp bench → [230:240] all zero; got "
            f"{v_no_bench[230:240].tolist()}"
        )

    # 3d. Lethal-next-turn face-value. own_lethal at [240] means opp.active
    # can KO own.active. haruUraraBasic has 1 attack at 10 dmg. Set own HP
    # to 5 → opp can KO (10 ≥ 5); leave opp HP at 90 → own cannot KO
    # (10 < 90). Result: own_lethal=1, opp_lethal=0.
    obs_lethal = make_obs(own_active_hp=5, opp_active_hp=90)
    v_l = observation_to_features_v3_6(obs_lethal)
    if v_l[240] != 1.0 or v_l[241] != 0.0:
        raise SystemExit(
            f"FAIL[3d-lethal]: own_hp=5 vs opp.atk=10 → own_lethal=1, "
            f"opp_lethal=0; got own={v_l[240]}, opp={v_l[241]}"
        )
    # Lethal-false case: both at full HP. own_lethal=0, opp_lethal=0
    # (10 < 90 both ways).
    obs_no_lethal = make_obs(own_active_hp=90, opp_active_hp=90)
    v_nl = observation_to_features_v3_6(obs_no_lethal)
    if v_nl[240] != 0.0 or v_nl[241] != 0.0:
        raise SystemExit(
            f"FAIL[3d-no-lethal]: both at full HP → both lethal bits 0; "
            f"got own={v_nl[240]}, opp={v_nl[241]}"
        )
    # Null defender (null own.active during a pending-promote state) → 0.
    obs_null = make_obs()
    obs_null["own"]["active"] = None
    v_null = observation_to_features_v3_6(obs_null)
    if v_null[240] != 0.0:
        raise SystemExit(
            f"FAIL[3d-null]: null own.active → own_lethal=0; got {v_null[240]}"
        )

    # 3e. Secondary attack: matikanefukukitaruStage1 has 2 attacks. attacks[1]
    # = Divination: Great Fortune, cost {psychic:1, colorless:1}, damage 0.
    # Give own.active 2 psychic energies → usable=1, would_KO=0 (dmg=0 <
    # any HP).
    obs_sec = make_obs(
        own_active_card="matikanefukukitaruStage1",
        own_active_hp=100,
        own_active_energies={"psychic": 2},
    )
    v_sec = observation_to_features_v3_6(obs_sec)
    if v_sec[242] != 1.0 or v_sec[243] != 0.0:
        raise SystemExit(
            f"FAIL[3e-usable]: matikanefukukitaruStage1 + 2x psychic → "
            f"usable=1, would_KO=0 (secondary dmg=0); got "
            f"usable={v_sec[242]}, would_KO={v_sec[243]}"
        )
    # No secondary attack (haruUraraBasic has only 1) → both bits 0.
    obs_no_sec = make_obs()
    v_no_sec = observation_to_features_v3_6(obs_no_sec)
    if v_no_sec[242] != 0.0 or v_no_sec[243] != 0.0:
        raise SystemExit(
            f"FAIL[3e-no-secondary]: haruUraraBasic (1 attack) → both bits 0; "
            f"got usable={v_no_sec[242]}, would_KO={v_no_sec[243]}"
        )
    # Secondary attack present but cost not covered (psychic=0) → both 0.
    obs_sec_short = make_obs(
        own_active_card="matikanefukukitaruStage1",
        own_active_energies={},
    )
    v_short = observation_to_features_v3_6(obs_sec_short)
    if v_short[242] != 0.0 or v_short[243] != 0.0:
        raise SystemExit(
            f"FAIL[3e-no-energy]: matikanefukukitaruStage1 + no energy → "
            f"both bits 0; got usable={v_short[242]}, would_KO={v_short[243]}"
        )

    # 4. Schema dispatch.
    if feature_builder_for_state_dim(STATE_DIM_V3_6) is not observation_to_features_v3_6:
        raise SystemExit(
            "FAIL[4]: feature_builder_for_state_dim(246) did not return v3.6 builder"
        )
    if schema_version_for_state_dim(STATE_DIM_V3_6) != STATE_FEATURE_SCHEMA_VERSION_V3_6:
        raise SystemExit(
            f"FAIL[4]: schema_version_for_state_dim(246) != "
            f"STATE_FEATURE_SCHEMA_VERSION_V3_6 "
            f"({STATE_FEATURE_SCHEMA_VERSION_V3_6})"
        )

    print("v36_priors_arithmetic_smoke: PASS "
          "(shape + v3.5 head-byte-identity + pool/prize/bench/lethal/secondary + dispatch)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
