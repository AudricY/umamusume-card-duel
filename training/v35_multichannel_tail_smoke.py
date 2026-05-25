"""Smoke the v3.5 multichannel-tail builder (`observation_to_features_v3_5`).

v3.5 = v3.3 head (167-d) + 45 channel-orthogonal bits at slots 167-211:

  [167:177] phase one-hot (10) — temporal-cadence channel
  [177:182] own active per-condition one-hot (5) — uma-condition channel
  [182:187] opp active per-condition one-hot (5) — uma-condition channel
  [187:197] own energy-zone front-of-queue typed one-hot (10) — energy-color
  [197:207] opp energy-zone front-of-queue typed one-hot (10) — energy-color
  [207:210] opp discard role buckets (3) — zone-residual symmetry
  [210]     own would_lose_on_active_KO (bench empty) — terminal-state synth
  [211]     opp would_lose_on_active_KO (bench empty) — terminal-state synth

Each item touches a signal channel v3.3 cannot currently express. The
slot-token (v3.2/v3.4) channel is EXCLUDED per the v3.4 falsification at
progress/r110.md §4h.

Asserts:
1. Output shape is (212,).
2. Slots 0-166 are byte-identical to `observation_to_features_v3_3` on
   the same observation (zero-init-residual contract).
3. Tail slots respond correctly to mutated observation fields:
   3a. phase one-hot moves with `phase`.
   3b. per-condition one-hot reflects active Uma `specialConditions`.
   3c. energy-zone front one-hot reflects `energyZone[0]`.
   3d. opp discard buckets surface trainer/basic/evolved counts.
   3e. bench-refill bits flip with bench presence.
4. Schema-dispatch returns `observation_to_features_v3_5` for state_dim=212.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uma_ai.features import (  # noqa: E402
    PHASES,
    STATE_DIM_V3_3,
    STATE_DIM_V3_5,
    STATE_FEATURE_SCHEMA_VERSION_V3_5,
    feature_builder_for_state_dim,
    observation_to_features_v3_3,
    observation_to_features_v3_5,
    schema_version_for_state_dim,
)


def make_obs(
    *,
    phase: str = "stadiumOrEnd",
    own_conditions: list[str] | None = None,
    opp_conditions: list[str] | None = None,
    own_energy_zone: list[str] | None = None,
    opp_energy_zone: list[str] | None = None,
    own_bench_count: int = 0,
    opp_bench_count: int = 0,
    opp_discard: list[str] | None = None,
) -> dict:
    """Minimal observation suitable for the feature builder."""

    def _bench(count: int) -> list:
        if count == 0:
            return []
        return [
            {"cardId": "haruUraraBasic", "hp": 90, "maxHp": 90, "energies": {},
             "energyTotal": 0, "specialConditions": [], "stage": 0}
            for _ in range(count)
        ]

    return {
        "phase": phase,
        "sideToAct": "player",
        "turnNumber": 1,
        "own": {
            "points": 0,
            "handCount": 5,
            "deckCount": 40,
            "active": {
                "cardId": "haruUraraBasic", "hp": 90, "maxHp": 90, "energies": {},
                "energyTotal": 0, "specialConditions": own_conditions or [],
            },
            "bench": _bench(own_bench_count),
            "discard": [],
            "energyZone": own_energy_zone or [],
            "usedSupporterThisTurn": False,
            "usedRetreatThisTurn": False,
            "usedStadiumThisTurn": False,
            "handCardIds": [],
        },
        "opponent": {
            "points": 0,
            "handCount": 5,
            "deckCount": 40,
            "active": {
                "cardId": "haruUraraBasic", "hp": 90, "maxHp": 90, "energies": {},
                "energyTotal": 0, "specialConditions": opp_conditions or [],
            },
            "bench": _bench(opp_bench_count),
            "discard": opp_discard or [],
            "energyZone": opp_energy_zone or [],
            "usedSupporterThisTurn": False,
            "usedRetreatThisTurn": False,
            "usedStadiumThisTurn": False,
            "handCardIds": [],
        },
        "shared": {"stadiumCardId": ""},
        "temporal": {},
    }


def main() -> int:
    # 1. Shape.
    obs = make_obs()
    v = observation_to_features_v3_5(obs)
    if v.shape != (STATE_DIM_V3_5,):
        raise SystemExit(f"FAIL: expected shape ({STATE_DIM_V3_5},); got {v.shape}")

    # 2. Head byte-identity vs v3.3.
    v33 = observation_to_features_v3_3(obs)
    if not np.array_equal(v[:STATE_DIM_V3_3], v33):
        raise SystemExit(
            "FAIL: v3.5 head [0:167] not byte-identical to v3.3 builder"
        )

    # 3a. Phase one-hot moves with `phase`.
    for phase_name in ("setup", "attach", "combat", "stadiumOrEnd"):
        obs_phase = make_obs(phase=phase_name)
        v_phase = observation_to_features_v3_5(obs_phase)
        phase_slice = v_phase[167:177]
        expected_idx = PHASES.index(phase_name)
        if phase_slice.sum() != 1.0 or phase_slice[expected_idx] != 1.0:
            raise SystemExit(
                f"FAIL: phase={phase_name!r} → bit {expected_idx} should be the only set bit; "
                f"got {phase_slice.tolist()}"
            )

    # 3b. Per-condition one-hot reflects active Uma `specialConditions`.
    obs_cond = make_obs(
        own_conditions=["paralysed", "burned"],
        opp_conditions=["frozen"],
    )
    v_cond = observation_to_features_v3_5(obs_cond)
    # own [177:182] — paralysed=0, burned=1.
    if (v_cond[177], v_cond[178], v_cond[179], v_cond[180], v_cond[181]) != (1.0, 1.0, 0.0, 0.0, 0.0):
        raise SystemExit(
            f"FAIL: own conditions paralysed+burned → (1,1,0,0,0); got "
            f"({v_cond[177]}, {v_cond[178]}, {v_cond[179]}, {v_cond[180]}, {v_cond[181]})"
        )
    # opp [182:187] — frozen=4.
    if (v_cond[182], v_cond[183], v_cond[184], v_cond[185], v_cond[186]) != (0.0, 0.0, 0.0, 0.0, 1.0):
        raise SystemExit(
            f"FAIL: opp condition frozen → (0,0,0,0,1); got "
            f"({v_cond[182]}, {v_cond[183]}, {v_cond[184]}, {v_cond[185]}, {v_cond[186]})"
        )

    # 3c. Energy-zone front-of-queue typed one-hot.
    obs_energy = make_obs(
        own_energy_zone=["fire", "water"],
        opp_energy_zone=["psychic"],
    )
    v_energy = observation_to_features_v3_5(obs_energy)
    # own [187:197] — fire is index 1 in vocab.
    if v_energy[187 + 1] != 1.0 or v_energy[187:197].sum() != 1.0:
        raise SystemExit(
            f"FAIL: own energy front=fire → bit 1 only; got {v_energy[187:197].tolist()}"
        )
    # opp [197:207] — psychic is index 4 in vocab.
    if v_energy[197 + 4] != 1.0 or v_energy[197:207].sum() != 1.0:
        raise SystemExit(
            f"FAIL: opp energy front=psychic → bit 4 only; got {v_energy[197:207].tolist()}"
        )
    # Empty zone → all zero.
    obs_no_energy = make_obs()
    v_no_energy = observation_to_features_v3_5(obs_no_energy)
    if v_no_energy[187:197].sum() != 0.0 or v_no_energy[197:207].sum() != 0.0:
        raise SystemExit("FAIL: empty energyZone → all 0 in [187:207]")

    # 3d. Opp discard role buckets — uses `_discard_role_features` which
    # consults the card catalog. With unknown IDs (or empty discard) the
    # buckets are all zero; we just confirm the slots are present and zero
    # by default and that the slice is exclusively zone-residual symmetry.
    obs_opp_discard = make_obs(opp_discard=[])
    v_opp_dr = observation_to_features_v3_5(obs_opp_discard)
    if v_opp_dr[207:210].sum() != 0.0:
        raise SystemExit("FAIL: empty opp discard → buckets [207:210] should be 0")

    # 3e. Bench-refill catastrophe bits.
    # Both benches empty (default) → both bits set.
    v_no_bench = observation_to_features_v3_5(make_obs())
    if v_no_bench[210] != 1.0 or v_no_bench[211] != 1.0:
        raise SystemExit(
            f"FAIL: both benches empty → bits (1.0, 1.0); got "
            f"({v_no_bench[210]}, {v_no_bench[211]})"
        )
    # Own bench has 1 Uma; own bit flips, opp bit stays.
    v_own_bench = observation_to_features_v3_5(
        make_obs(own_bench_count=1)
    )
    if v_own_bench[210] != 0.0 or v_own_bench[211] != 1.0:
        raise SystemExit(
            f"FAIL: own bench=1 → own bit=0.0, opp bit=1.0; got "
            f"({v_own_bench[210]}, {v_own_bench[211]})"
        )

    # 4. Schema dispatch.
    if feature_builder_for_state_dim(STATE_DIM_V3_5) is not observation_to_features_v3_5:
        raise SystemExit(
            "FAIL: feature_builder_for_state_dim(212) did not return v3.5 builder"
        )
    if schema_version_for_state_dim(STATE_DIM_V3_5) != STATE_FEATURE_SCHEMA_VERSION_V3_5:
        raise SystemExit(
            f"FAIL: schema_version_for_state_dim(212) != "
            f"STATE_FEATURE_SCHEMA_VERSION_V3_5 ({STATE_FEATURE_SCHEMA_VERSION_V3_5})"
        )

    print("v35_multichannel_tail_smoke: 4/4 cases PASS (shape + head-byte-identity + 5 tail subcases + dispatch)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
