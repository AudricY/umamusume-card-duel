"""Smoke the v3.8 slim-feature-add state builder
(`observation_to_features_v3_8`).

v3.8 = v3.7 head (296-d) byte-stable + 8-bit slim tail at [296:304] per
the LOCKED layout in
`docs/ai-research/scoping/v38-slim-feature-add-scoping.md` §4.5.

Channel-isolation discipline: each fixture asserts the EXPECTED bits in
[296:304] AND asserts all other v3.8 tail bits are 0. The v3.7 head
[0:296] is verified byte-identical to `observation_to_features_v3_7` for
every fixture — the v3.7 head MUST stay frozen.

Truth-table coverage:
  1. Empty fixture        — sanity: all 8 v3.8 tail bits == 0 except
                             those forced by the catalog defaults.
  2. Own bench ETA fires  — bench[0]'s primary attack cost is fully
                             coverable by attached + 1 next-turn attach
                             from energyPool. Only own_bench0_eta fires.
  3. Opp bench ETA fires  — symmetric on opp.bench[1].
  4. Gust-swing catastrophe — own.points=2 (1 prize left) + own.bench
                             with low HP + opp.active that can KO it +
                             opp.discard with a gust trainer +
                             opp.usedSupporterThisTurn=False +
                             opp.handCount>0 → bit [302]=1.
  5. Gust-win race        — symmetric: opp.points=2 + opp.bench KO-able
                             by own.active + own.handCardIds contains
                             gust trainer → bit [303]=1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uma_ai.features import (  # noqa: E402
    STATE_DIM_V3_7,
    STATE_DIM_V3_8,
    STATE_FEATURE_SCHEMA_VERSION_V3_8,
    _V38_OPP_BENCH_ETA_BASE,
    _V38_OWN_BENCH_ETA_BASE,
    _V38_OWN_GUST_WIN_RACE,
    _V38_OWN_LOSE_IF_OPP_GUSTS,
    _V38_TAIL_START,
    feature_builder_for_state_dim,
    observation_to_features_v3_7,
    observation_to_features_v3_8,
    schema_version_for_state_dim,
)


def _make_uma(
    *,
    card_id: str = "matikanetannhauserBasic",
    hp: int = 60,
    max_hp: int = 60,
    energies: dict | None = None,
    tool_card_id: str | None = None,
    used_ability_this_turn: bool = False,
    paralysis_recovery_pending: bool = False,
) -> dict:
    return {
        "cardId": card_id,
        "hp": hp,
        "maxHp": max_hp,
        "energies": energies or {},
        "energyTotal": sum((energies or {}).values()),
        "specialConditions": [],
        "stage": 0,
        "toolCardId": tool_card_id,
        "usedAbilityThisTurn": used_ability_this_turn,
        "turnState": {
            "paralysisRecoveryPending": paralysis_recovery_pending,
        },
    }


def _make_obs(
    *,
    own_active: dict | None = None,
    opp_active: dict | None = None,
    own_bench: list | None = None,
    opp_bench: list | None = None,
    own_energy_pool: list[str] | None = None,
    opp_energy_pool: list[str] | None = None,
    own_points: int = 0,
    opp_points: int = 0,
    own_discard: list[str] | None = None,
    opp_discard: list[str] | None = None,
    own_hand_card_ids: list[str] | None = None,
    opp_used_supporter_this_turn: bool = False,
    opp_hand_count: int = 5,
) -> dict:
    return {
        "phase": "stadiumOrEnd",
        "sideToAct": "player",
        "turnNumber": 1,
        "own": {
            "points": own_points,
            "handCount": len(own_hand_card_ids or []) if own_hand_card_ids else 5,
            "deckCount": 40,
            "active": own_active or _make_uma(),
            "bench": own_bench or [],
            "discard": own_discard or [],
            "energyZone": [],
            "energyPool": own_energy_pool or [],
            "usedSupporterThisTurn": False,
            "usedRetreatThisTurn": False,
            "usedStadiumThisTurn": False,
            "handCardIds": own_hand_card_ids or [],
        },
        "opponent": {
            "points": opp_points,
            "handCount": opp_hand_count,
            "deckCount": 40,
            "active": opp_active or _make_uma(),
            "bench": opp_bench or [],
            "discard": opp_discard or [],
            "energyZone": [],
            "energyPool": opp_energy_pool or [],
            "usedSupporterThisTurn": opp_used_supporter_this_turn,
            "usedRetreatThisTurn": False,
            "usedStadiumThisTurn": False,
        },
        "shared": {"stadiumCardId": ""},
        "temporal": {},
    }


def _assert_v37_head_byte_identical(label: str, obs: dict, v38: np.ndarray) -> None:
    v37 = observation_to_features_v3_7(obs)
    if not np.array_equal(v38[:STATE_DIM_V3_7], v37):
        raise SystemExit(
            f"FAIL[{label}]: v3.8 head [0:296] not byte-identical to v3.7. "
            f"v3.8 must NEVER mutate the v3.7 head."
        )


def _assert_only_bits_set(
    label: str,
    v38: np.ndarray,
    expected_indices: set[int],
) -> None:
    """Within [296:304], EXACTLY `expected_indices` are 1.0 and all
    other indices are 0.0."""

    for idx in range(_V38_TAIL_START, STATE_DIM_V3_8):
        got = float(v38[idx])
        if idx in expected_indices:
            if got != 1.0:
                raise SystemExit(
                    f"FAIL[{label}]: expected v38[{idx}]=1.0; got {got}"
                )
        else:
            if got != 0.0:
                raise SystemExit(
                    f"FAIL[{label}]: channel-isolation breach — "
                    f"v38[{idx}]={got} but expected 0.0 "
                    f"(expected_set={sorted(expected_indices)})"
                )


def main() -> int:
    # -------------------------------------------------------------------
    # 1. Shape + dispatch.
    # -------------------------------------------------------------------
    obs0 = _make_obs()
    v0 = observation_to_features_v3_8(obs0)
    if v0.shape != (STATE_DIM_V3_8,):
        raise SystemExit(f"FAIL[shape]: got {v0.shape}; expected ({STATE_DIM_V3_8},)")
    _assert_v37_head_byte_identical("baseline", obs0, v0)
    if feature_builder_for_state_dim(STATE_DIM_V3_8) is not observation_to_features_v3_8:
        raise SystemExit(
            "FAIL[dispatch]: feature_builder_for_state_dim(304) wrong builder"
        )
    if schema_version_for_state_dim(STATE_DIM_V3_8) != STATE_FEATURE_SCHEMA_VERSION_V3_8:
        raise SystemExit(
            "FAIL[dispatch]: schema_version_for_state_dim(304) wrong"
        )

    # -------------------------------------------------------------------
    # 2. Empty fixture — no bench, no gust signals → all 8 tail bits 0.
    # -------------------------------------------------------------------
    _assert_only_bits_set("empty-fixture", v0, set())

    # -------------------------------------------------------------------
    # 3. Own bench ETA — bench[0] = matikanetannhauserBasic (cost
    #    {psychic:1}). attached={}, pool=[] → shortfall={psychic:1}, +1
    #    attach removes it → no surviving shortfall → ETA=1 regardless
    #    of pool (matches v3.7 Ch.4 structural-feasibility semantics:
    #    if every shortfall color is covered after the +1 attach, the
    #    pool requirement is satisfied vacuously). Only own_bench0_eta
    #    bit fires; opp bench is empty → opp ETA bits all 0.
    #    NOTE: own active bench-eta predicate is INDEPENDENT of own
    #    active. The active bench[i] here is bench[0] only.
    # -------------------------------------------------------------------
    bench_uma = _make_uma(card_id="matikanetannhauserBasic", hp=60, max_hp=60)
    obs_own_bench = _make_obs(own_bench=[bench_uma])
    v_own_bench = observation_to_features_v3_8(obs_own_bench)
    _assert_v37_head_byte_identical("own-bench-eta", obs_own_bench, v_own_bench)
    _assert_only_bits_set("own-bench-eta", v_own_bench, {_V38_OWN_BENCH_ETA_BASE + 0})

    # -------------------------------------------------------------------
    # 4. Opp bench ETA — symmetric, opp.bench[1] populated. Only
    #    opp_bench1_eta fires.
    # -------------------------------------------------------------------
    obs_opp_bench = _make_obs(
        opp_bench=[None, bench_uma],
    )
    v_opp_bench = observation_to_features_v3_8(obs_opp_bench)
    _assert_v37_head_byte_identical("opp-bench-eta", obs_opp_bench, v_opp_bench)
    _assert_only_bits_set("opp-bench-eta", v_opp_bench, {_V38_OPP_BENCH_ETA_BASE + 1})

    # -------------------------------------------------------------------
    # 5. Gust-swing catastrophe ([302]=1).
    #    Setup:
    #      - own.points=2 → remaining=1 (one more prize loses).
    #      - own.bench=[low-HP haruUraraBasic 10/90] — KO-able by
    #        opp.active manhattanCafeStage1 (Darkness, 40 damage,
    #        weakness Darkness +20 against haruUrara's weakness? Let's
    #        verify: haruUraraBasic weakness is Lightning, no match.
    #        40 damage face-value vs 10 hp → 40 >= 10, KO without
    #        weakness adjustment.)
    #      - opp.discard contains yayoiAkikawa (gust trainer per
    #        cards.json:1589).
    #      - opp.usedSupporterThisTurn=False, opp.handCount=5.
    #    Expected: [302]=1, ALL bench ETAs 0 (own bench haruUrara has
    #    no typed cost without energy budget? haruUraraBasic cost is
    #    {colorless:1} so the bench ETA predicate skips colorless and
    #    returns 1 vacuously. So own_bench0_eta=1 too.).
    # -------------------------------------------------------------------
    low_hp_bench = _make_uma(card_id="haruUraraBasic", hp=10, max_hp=90)
    opp_active_attacker = _make_uma(
        card_id="manhattanCafeStage1",
        hp=90,
        max_hp=90,
        energies={"darkness": 1, "colorless": 1},
    )
    obs_catastrophe = _make_obs(
        own_active=_make_uma(card_id="matikanetannhauserBasic", hp=60, max_hp=60),
        opp_active=opp_active_attacker,
        own_bench=[low_hp_bench],
        own_points=2,
        opp_discard=["yayoiAkikawa"],
        opp_used_supporter_this_turn=False,
        opp_hand_count=5,
    )
    v_catastrophe = observation_to_features_v3_8(obs_catastrophe)
    _assert_v37_head_byte_identical("catastrophe", obs_catastrophe, v_catastrophe)
    # Bench ETA: haruUraraBasic primary cost is {colorless:1} — typed
    # shortfall dict is empty (colorless is excluded), so the +1 attach
    # is never applied AND the surviving-color check trivially passes
    # → own_bench0_eta=1. Then catastrophe bit also fires.
    expected_catastrophe = {_V38_OWN_BENCH_ETA_BASE + 0, _V38_OWN_LOSE_IF_OPP_GUSTS}
    _assert_only_bits_set("catastrophe", v_catastrophe, expected_catastrophe)

    # Sanity: same setup but opp.usedSupporterThisTurn=True → bit clear.
    obs_no_catastrophe = _make_obs(
        own_active=_make_uma(card_id="matikanetannhauserBasic", hp=60, max_hp=60),
        opp_active=opp_active_attacker,
        own_bench=[low_hp_bench],
        own_points=2,
        opp_discard=["yayoiAkikawa"],
        opp_used_supporter_this_turn=True,  # disables gust proxy
        opp_hand_count=5,
    )
    v_no_catastrophe = observation_to_features_v3_8(obs_no_catastrophe)
    if v_no_catastrophe[_V38_OWN_LOSE_IF_OPP_GUSTS] != 0.0:
        raise SystemExit(
            f"FAIL[no-catastrophe]: expected [302]=0 with opp supporter "
            f"used; got {v_no_catastrophe[_V38_OWN_LOSE_IF_OPP_GUSTS]}"
        )

    # Sanity: catastrophe setup but own.points=0 (3 prizes left) → bit clear.
    obs_no_prize = _make_obs(
        own_active=_make_uma(card_id="matikanetannhauserBasic", hp=60, max_hp=60),
        opp_active=opp_active_attacker,
        own_bench=[low_hp_bench],
        own_points=0,
        opp_discard=["yayoiAkikawa"],
        opp_used_supporter_this_turn=False,
        opp_hand_count=5,
    )
    v_no_prize = observation_to_features_v3_8(obs_no_prize)
    if v_no_prize[_V38_OWN_LOSE_IF_OPP_GUSTS] != 0.0:
        raise SystemExit(
            f"FAIL[no-prize-band]: expected [302]=0 with own prizes>1; "
            f"got {v_no_prize[_V38_OWN_LOSE_IF_OPP_GUSTS]}"
        )

    # -------------------------------------------------------------------
    # 6. Gust-win race ([303]=1).
    #    Setup symmetric: opp.points=2, opp.bench has low-HP KO-able by
    #    own.active, own.handCardIds contains gust trainer.
    # -------------------------------------------------------------------
    own_active_attacker = _make_uma(
        card_id="manhattanCafeStage1",
        hp=90,
        max_hp=90,
        energies={"darkness": 1, "colorless": 1},
    )
    obs_gust_race = _make_obs(
        own_active=own_active_attacker,
        opp_active=_make_uma(card_id="matikanetannhauserBasic", hp=60, max_hp=60),
        opp_bench=[low_hp_bench],
        opp_points=2,
        own_hand_card_ids=["yayoiAkikawa"],
    )
    v_gust_race = observation_to_features_v3_8(obs_gust_race)
    _assert_v37_head_byte_identical("gust-race", obs_gust_race, v_gust_race)
    # Expected bits: own_bench (none) → 0; opp_bench[0] ETA: haruUrara
    # cost colorless:1 → vacuously satisfied → 1; gust-win-race [303]=1.
    expected_gust_race = {_V38_OPP_BENCH_ETA_BASE + 0, _V38_OWN_GUST_WIN_RACE}
    _assert_only_bits_set("gust-race", v_gust_race, expected_gust_race)

    print(
        f"v38_state_v38_smoke: ALL cases PASS "
        f"(STATE_DIM_V3_8={STATE_DIM_V3_8}, "
        f"schema={STATE_FEATURE_SCHEMA_VERSION_V3_8})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
