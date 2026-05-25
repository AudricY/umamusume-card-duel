"""Smoke the v3.3 additive-tail builder (`observation_to_features_v3_3`).

v3.3 = v3.1 head (164-d) + 3 opp-side flag bits at slots 164-166. The
own-side equivalents are at slots 29/30/31 (frozen v2 head, carried into
v3.0/v3.1). v3.3 is the first surfacing of the opponent's used*
booleans — they were genuinely missing from v3.0/v3.1/v3.2.

Asserts:
1. Output shape is (167,).
2. Slots 0-163 are byte-identical to `observation_to_features_v3_1` on
   the same observation (zero-init-residual contract).
3. Tail slots 164-166 reflect `opp.usedSupporterThisTurn`,
   `opp.usedRetreatThisTurn`, `opp.usedStadiumThisTurn`.
4. Schema-dispatch returns `observation_to_features_v3_3` for state_dim=167.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uma_ai.features import (  # noqa: E402
    STATE_DIM_V3_1,
    STATE_DIM_V3_3,
    STATE_FEATURE_SCHEMA_VERSION_V3_3,
    feature_builder_for_state_dim,
    observation_to_features_v3_1,
    observation_to_features_v3_3,
    schema_version_for_state_dim,
)


def make_obs(opp_flags: dict[str, bool] | None = None) -> dict:
    """Minimal observation suitable for the feature builder. The temporal
    block is empty; the v3.1 builder handles missing keys gracefully."""

    flags = opp_flags or {}
    return {
        "phase": "stadiumOrEnd",
        "sideToAct": "player",
        "turnNumber": 1,
        "own": {
            "points": 0,
            "handCount": 5,
            "deckCount": 40,
            "active": {"cardId": "haruUraraBasic", "hp": 90, "energies": {}, "energyTotal": 0, "specialConditions": []},
            "bench": [],
            "discard": [],
            "energyZone": [],
            "usedSupporterThisTurn": False,
            "usedRetreatThisTurn": False,
            "usedStadiumThisTurn": False,
        },
        "opponent": {
            "points": 0,
            "handCount": 5,
            "deckCount": 40,
            "active": {"cardId": "haruUraraBasic", "hp": 90, "energies": {}, "energyTotal": 0, "specialConditions": []},
            "bench": [],
            "discard": [],
            "energyZone": [],
            "usedSupporterThisTurn": flags.get("supporter", False),
            "usedRetreatThisTurn": flags.get("retreat", False),
            "usedStadiumThisTurn": flags.get("stadium", False),
        },
        "shared": {"stadiumCardId": ""},
        "temporal": {},
    }


def main() -> int:
    # 1. Shape.
    obs = make_obs()
    v = observation_to_features_v3_3(obs)
    if v.shape != (STATE_DIM_V3_3,):
        raise SystemExit(f"FAIL: expected shape ({STATE_DIM_V3_3},); got {v.shape}")

    # 2. Head byte-identity vs v3.1.
    v31 = observation_to_features_v3_1(obs)
    if not np.array_equal(v[:STATE_DIM_V3_1], v31):
        raise SystemExit("FAIL: v3.3 head [0:164] not byte-identical to v3.1 builder")

    # 3. Tail slots default to 0 with no opp flags.
    if (v[164], v[165], v[166]) != (0.0, 0.0, 0.0):
        raise SystemExit(
            f"FAIL: default opp flags should be (0,0,0); got ({v[164]}, {v[165]}, {v[166]})"
        )

    # 3b. Tail slots flip with opp flags set.
    obs_flagged = make_obs(opp_flags={"supporter": True, "stadium": True})
    v_flag = observation_to_features_v3_3(obs_flagged)
    if (v_flag[164], v_flag[165], v_flag[166]) != (1.0, 0.0, 1.0):
        raise SystemExit(
            f"FAIL: opp flags supporter=stadium=True should yield (1,0,1); got "
            f"({v_flag[164]}, {v_flag[165]}, {v_flag[166]})"
        )

    # 4. Schema dispatch.
    if feature_builder_for_state_dim(STATE_DIM_V3_3) is not observation_to_features_v3_3:
        raise SystemExit("FAIL: feature_builder_for_state_dim(167) did not return v3.3 builder")
    if schema_version_for_state_dim(STATE_DIM_V3_3) != STATE_FEATURE_SCHEMA_VERSION_V3_3:
        raise SystemExit(
            f"FAIL: schema_version_for_state_dim(167) != "
            f"STATE_FEATURE_SCHEMA_VERSION_V3_3 ({STATE_FEATURE_SCHEMA_VERSION_V3_3})"
        )

    print("v33_additive_tail_smoke: 4/4 cases PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
