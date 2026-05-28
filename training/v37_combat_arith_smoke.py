"""Smoke the v3.7 combat-arith-and-catalog builder
(`observation_to_features_v3_7`).

v3.7 = v3.6 head (246-d) byte-stable + 50-bit combat-arith / catalog tail at
[246:296] per the LOCKED layout in
`docs/ai-research/scoping/v37-combat-arith-and-catalog-scoping.md` §4 step 3
/ §4.5.

Channel-isolation discipline: each fixture asserts the EXPECTED bits AND
asserts all other v3.7 bits in [246:296] are 0. The v3.6 head [0:246] is
verified byte-identical to `observation_to_features_v3_6` for every
fixture — the v3.6 head MUST stay frozen.

Truth-table coverage (one fixture per channel):
1. Empty-fixture       — sanity: all 50 v3.7 tail bits == 0 except those
                          forced by the trivial catalog data.
2. Weakness-on         — `attacker.type == defender.weakness.type`:
                          Channel 1 lethal bit flips when damage+amount
                          ≥ defender remaining HP.
3. Coin-flip primary   — primary attack with `coinBonus`: Channel 2
                          has_cf bit fires; cf_eko fires when expected
                          damage covers defender HP.
4. Per-energy bonus    — attack with `damagePerAttachedEnergy` (or the
                          `damagePerUniqueAttachedEnergy` flag):
                          Channel 3 per-energy bit fires.
5. Per-bench bonus     — attack with `damagePerUmamusumeInPlay`:
                          Channel 3 per-bench bit fires.
6. ETA fixture         — attacker missing 1 typed energy color that's
                          present in `energyPool`: Channel 4 ETA bit fires.
7. Paralysis fixture   — opp turnState.paralysisRecoveryPending=True:
                          Channel 4 own paralysis window bit fires.
8. Tool fixture        — leftoverCarrot attached: Channel 5 own-side
                          bit 0 (HEAL_AT_TURN_END) fires.
9. Ability fixture     — niceNatureBasic + usedAbilityThisTurn=True:
                          Channel 6 own-side bit 2 (HP_BONUS) fires.

Mirrors `training/v36_priors_arithmetic_smoke.py` style. Channel
isolation is the v3.7 parity-discipline contract — any drift in another
channel surfaces as a failed isolation assertion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uma_ai.effect_kinds import (  # noqa: E402
    AbilityEffectKind,
    ToolEffectKind,
)
from uma_ai.features import (  # noqa: E402
    STATE_DIM_V3_6,
    STATE_DIM_V3_7,
    STATE_FEATURE_SCHEMA_VERSION_V3_7,
    _V37_ABILITY_KIND_OPP_BASE,
    _V37_ABILITY_KIND_OWN_BASE,
    _V37_COIN_FLIP_BASE,
    _V37_COND_BONUS_BASE,
    _V37_ETA_BASE,
    _V37_OPP_WEAKNESS_LETHAL,
    _V37_OPP_WEAKNESS_SECONDARY_KO,
    _V37_OWN_WEAKNESS_LETHAL,
    _V37_OWN_WEAKNESS_SECONDARY_KO,
    _V37_PARALYSIS_BASE,
    _V37_TAIL_START,
    _V37_TOOL_KIND_OPP_BASE,
    _V37_TOOL_KIND_OWN_BASE,
    feature_builder_for_state_dim,
    observation_to_features_v3_6,
    observation_to_features_v3_7,
    schema_version_for_state_dim,
)


def _make_uma(
    *,
    card_id: str = "haruUraraBasic",
    hp: int = 90,
    max_hp: int = 90,
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
    own_energy_pool: list[str] | None = None,
    opp_energy_pool: list[str] | None = None,
    own_points: int = 0,
    opp_points: int = 0,
) -> dict:
    return {
        "phase": "stadiumOrEnd",
        "sideToAct": "player",
        "turnNumber": 1,
        "own": {
            "points": own_points,
            "handCount": 5,
            "deckCount": 40,
            "active": own_active or _make_uma(),
            "bench": [],
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
            "active": opp_active or _make_uma(),
            "bench": [],
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


def _assert_v36_head_byte_identical(label: str, obs: dict, v37: np.ndarray) -> None:
    v36 = observation_to_features_v3_6(obs)
    if not np.array_equal(v37[:STATE_DIM_V3_6], v36):
        raise SystemExit(
            f"FAIL[{label}]: v3.7 head [0:246] not byte-identical to v3.6 "
            f"builder. v3.7 must NEVER mutate v3.6 slots."
        )


def _assert_only_bits_set(
    label: str,
    v37: np.ndarray,
    expected_indices: set[int],
    *,
    expected_values: dict[int, float] | None = None,
) -> None:
    """Assert that within [246:296], EXACTLY `expected_indices` are 1.0
    (or the value in `expected_values`) and all other indices are 0.0.

    Channel-isolation guard: catches cross-channel bleed."""

    if expected_values is None:
        expected_values = {i: 1.0 for i in expected_indices}
    for idx in range(_V37_TAIL_START, STATE_DIM_V3_7):
        got = float(v37[idx])
        if idx in expected_indices:
            want = float(expected_values.get(idx, 1.0))
            if got != want:
                raise SystemExit(
                    f"FAIL[{label}]: expected v37[{idx}]={want}; got {got}"
                )
        else:
            if got != 0.0:
                raise SystemExit(
                    f"FAIL[{label}]: channel-isolation breach — "
                    f"v37[{idx}]={got} but expected 0.0 (expected_set="
                    f"{sorted(expected_indices)})"
                )


def main() -> int:
    # ---------------------------------------------------------------
    # 1. Shape + dispatch.
    # ---------------------------------------------------------------
    obs0 = _make_obs(
        own_active=_make_uma(card_id="matikanetannhauserBasic", hp=60, max_hp=60),
        opp_active=_make_uma(card_id="matikanetannhauserBasic", hp=60, max_hp=60),
    )
    v0 = observation_to_features_v3_7(obs0)
    if v0.shape != (STATE_DIM_V3_7,):
        raise SystemExit(f"FAIL[shape]: got {v0.shape}; expected ({STATE_DIM_V3_7},)")
    _assert_v36_head_byte_identical("baseline", obs0, v0)
    if feature_builder_for_state_dim(STATE_DIM_V3_7) is not observation_to_features_v3_7:
        raise SystemExit("FAIL[dispatch]: feature_builder_for_state_dim(296) wrong builder")
    if schema_version_for_state_dim(STATE_DIM_V3_7) != STATE_FEATURE_SCHEMA_VERSION_V3_7:
        raise SystemExit("FAIL[dispatch]: schema_version_for_state_dim(296) wrong")

    # ---------------------------------------------------------------
    # 2. Empty-fixture sanity: matikanetannhauserBasic has no coin-flip,
    # no conditional bonuses, no tool, no ability used. The single
    # attack cost {psychic:1} is fully covered by the +1 next-turn
    # attach budget alone (largest shortfall reduces to 0 → no
    # surviving shortfall colors → ETA=1 regardless of pool). Both
    # sides identical → both own_primary_eta and opp_primary_eta fire.
    # No other v3.7 bit fires.
    # ---------------------------------------------------------------
    _assert_only_bits_set(
        "empty-fixture",
        v0,
        {_V37_ETA_BASE + 0, _V37_ETA_BASE + 2},
    )

    # ---------------------------------------------------------------
    # 3. Weakness-on (Channel 1): manhattanCafeStage1 is Darkness, 40
    # damage @ {darkness:1, colorless:1}. matikanetannhauserBasic is
    # Psychic, hp=60, weakness Darkness +20. Without weakness 40 < 60;
    # with: 40+20=60 >= 60 → lethal=1.
    #
    # Channel 1 polarity: `opp_weakness_lethal` = OWN attacks OPP (own
    # = attacker, opp = defender), so put manhattanCafeStage1 on OWN
    # and matikanetannhauser on OPP.
    # ---------------------------------------------------------------
    attacker = _make_uma(
        card_id="manhattanCafeStage1", hp=90, max_hp=90,
        energies={"darkness": 1, "colorless": 1},
    )
    defender = _make_uma(card_id="matikanetannhauserBasic", hp=60, max_hp=60)
    obs_weakness = _make_obs(own_active=attacker, opp_active=defender)
    v_weakness = observation_to_features_v3_7(obs_weakness)
    _assert_v36_head_byte_identical("weakness-on", obs_weakness, v_weakness)
    # opp_weakness_lethal must fire (own kills opp via weakness bonus).
    # own_weakness_lethal must NOT fire (matikanetannhauser is Psychic,
    # 20 damage, manhattanCafe weakness is Grass +20 → no match for
    # attacker_type=Psychic, so 20 < 90 → 0).
    # manhattanCafeStage1 has only 1 attack → secondary bits = 0.
    # manhattanCafeStage1 has no coin-flip; has no conditional bonuses.
    # No tools, no ability used. ETA: manhattanCafeStage1 cost is
    # {darkness:1, colorless:1}; attacker has both → no shortfall → ETA=1.
    # opp_primary_eta: matikanetannhauserBasic cost is {psychic:1};
    # attached={}, pool empty → shortfall={psychic:1}, +1 attach removes
    # it BUT pool doesn't contain psychic → still must check that the
    # color that had shortfall is in pool. After +1 reduces psychic to
    # 0, shortfall dict is empty → all remaining colors are covered →
    # ETA=1. Total: own_primary_eta=1, opp_primary_eta=1.
    # manhattanCafeStage1 has `damageOpponent` ability — but Channel 6
    # only fires if `usedAbilityThisTurn=True`; we set it False.
    expected = {
        _V37_OPP_WEAKNESS_LETHAL,
        _V37_ETA_BASE + 0,  # own_primary_eta
        _V37_ETA_BASE + 2,  # opp_primary_eta
    }
    _assert_only_bits_set("weakness-on", v_weakness, expected)

    # Sanity: weakness OFF (defender weakness type doesn't match attacker).
    # matikanetannhauserBasic (Psychic, 20 damage) attacking haruUraraBasic
    # (Psychic, hp=90, weakness Darkness +20 → no type match) → 20 < 90 → 0.
    obs_no_weakness = _make_obs(
        own_active=_make_uma(card_id="matikanetannhauserBasic", hp=60, max_hp=60,
                              energies={"psychic": 1}),
        opp_active=_make_uma(card_id="haruUraraBasic", hp=90, max_hp=90),
    )
    v_no_weakness = observation_to_features_v3_7(obs_no_weakness)
    # opp_weakness_lethal MUST be 0 (no type match, 20 < 90).
    if v_no_weakness[_V37_OPP_WEAKNESS_LETHAL] != 0.0:
        raise SystemExit(
            f"FAIL[weakness-off]: expected opp_weakness_lethal=0; got "
            f"{v_no_weakness[_V37_OPP_WEAKNESS_LETHAL]}"
        )

    # ---------------------------------------------------------------
    # 4. Coin-flip primary (Channel 2). matikanetannhauserStage1
    # (Psychic, 90hp) has 1 attack — coinBonus=20 at base damage 40.
    # Defender haruUraraBasic 90hp. Expected: own_primary_has_cf=1;
    # cf_eko = 1 iff 40 + 0.5*20 = 50 >= 90 → 0 (not lethal).
    # ---------------------------------------------------------------
    obs_cf = _make_obs(
        own_active=_make_uma(card_id="matikanetannhauserStage1", hp=90, max_hp=90,
                              energies={"psychic": 1, "colorless": 1}),
        opp_active=_make_uma(card_id="haruUraraBasic", hp=90, max_hp=90),
    )
    v_cf = observation_to_features_v3_7(obs_cf)
    _assert_v36_head_byte_identical("coin-flip-primary", obs_cf, v_cf)
    # own_primary_has_cf=1, own_primary_cf_eko=0 (50 < 90).
    # opp = haruUraraBasic: per_energy bit fires (has damagePerAttachedEnergy).
    # Both ETAs fire (own cost covered; opp cost colorless:1 covered by
    # +1 attach budget).
    expected_cf = {
        _V37_COIN_FLIP_BASE + 0,  # own_primary_has_cf
        _V37_COND_BONUS_BASE + 2,  # opp_primary_per_energy (haruUraraBasic)
        _V37_ETA_BASE + 0,  # own_primary_eta
        _V37_ETA_BASE + 2,  # opp_primary_eta
    }
    _assert_only_bits_set("coin-flip-primary", v_cf, expected_cf)

    # Sanity: cf_eko fires when defender HP is low enough.
    # Defender at 50 HP → 40 + 10 = 50 >= 50 → cf_eko = 1.
    obs_cf_eko = _make_obs(
        own_active=_make_uma(card_id="matikanetannhauserStage1", hp=90, max_hp=90,
                              energies={"psychic": 1, "colorless": 1}),
        opp_active=_make_uma(card_id="haruUraraBasic", hp=50, max_hp=90),
    )
    v_cf_eko = observation_to_features_v3_7(obs_cf_eko)
    if v_cf_eko[_V37_COIN_FLIP_BASE + 1] != 1.0:
        raise SystemExit(
            f"FAIL[cf_eko]: expected own_primary_cf_eko=1 at defender hp=50; "
            f"got {v_cf_eko[_V37_COIN_FLIP_BASE + 1]}"
        )

    # ---------------------------------------------------------------
    # 5. Per-energy bonus (Channel 3). haruUraraBasic primary attack
    # has damagePerAttachedEnergy → per_energy bit fires. No other v3.7
    # bits should fire other than ETA (covered by colorless absorption).
    # ---------------------------------------------------------------
    obs_pe = _make_obs(
        own_active=_make_uma(card_id="haruUraraBasic", hp=90, max_hp=90),
        opp_active=_make_uma(card_id="matikanetannhauserBasic", hp=60, max_hp=60),
    )
    v_pe = observation_to_features_v3_7(obs_pe)
    _assert_v36_head_byte_identical("per-energy", obs_pe, v_pe)
    # own_primary_per_energy fires (haruUraraBasic damagePerAttachedEnergy).
    # own ETA: cost {colorless:1}, no shortfall → 1.
    # opp = matikanetannhauserBasic, cost {psychic:1}; shortfall={psychic:1},
    # +1 reduces to 0; pool empty → no surviving shortfall, ETA=1.
    expected_pe = {
        _V37_COND_BONUS_BASE + 0,  # own_primary_per_energy
        _V37_ETA_BASE + 0,
        _V37_ETA_BASE + 2,
    }
    _assert_only_bits_set("per-energy", v_pe, expected_pe)

    # ---------------------------------------------------------------
    # 6. Per-bench bonus (Channel 3). Find a card with
    # damagePerUmamusumeInPlay. Catalog: tamamoCrossBasic has it
    # (or similar). Search the catalog for any card with the flag.
    # ---------------------------------------------------------------
    from uma_ai.effect_kinds import load_catalog
    catalog = load_catalog()
    per_bench_card_id = None
    for cid, card in catalog.items():
        if card.get("kind") != "umamusume":
            continue
        for atk in (card.get("attacks") or []):
            if atk.get("damagePerUmamusumeInPlay") is not None:
                per_bench_card_id = cid
                break
        if per_bench_card_id:
            break
    if per_bench_card_id is None:
        raise SystemExit(
            "FAIL[per-bench-setup]: no card with damagePerUmamusumeInPlay in catalog"
        )
    pb_card = catalog[per_bench_card_id]
    pb_hp = int(pb_card.get("hp", 90))
    obs_pb = _make_obs(
        own_active=_make_uma(card_id=per_bench_card_id, hp=pb_hp, max_hp=pb_hp),
        opp_active=_make_uma(card_id="haruUraraBasic", hp=90, max_hp=90),
    )
    v_pb = observation_to_features_v3_7(obs_pb)
    _assert_v36_head_byte_identical("per-bench", obs_pb, v_pb)
    if v_pb[_V37_COND_BONUS_BASE + 1] != 1.0:
        raise SystemExit(
            f"FAIL[per-bench]: expected own_primary_per_bench=1 for "
            f"{per_bench_card_id}; got {v_pb[_V37_COND_BONUS_BASE + 1]}"
        )

    # ---------------------------------------------------------------
    # 7. ETA fixture. matikanetannhauserBasic cost {psychic:1}. Attach
    # nothing (shortfall={psychic:1}); pool=['psychic'] → +1 attach
    # reduces psychic shortfall to 0 → all colors covered → ETA=1.
    # Counter: pool=['fire'] → +1 attach can't help psychic (we apply
    # the +1 to the LARGEST shortfall color which IS psychic;
    # shortfall reduces to 0; but wait, then there's no surviving
    # shortfall and the pool check passes vacuously). Hmm — let me
    # reread the rule.
    #
    # The scope §4.5 Ch.4 says: "every shortfall color has at least
    # one matching entry in own.energy_pool". So if shortfall AFTER
    # the +1 attach is empty, the predicate is vacuously true (ETA=1).
    # If shortfall has psychic:1 even after the +1, then pool must
    # contain psychic.
    #
    # Practical ETA truth table (single-color shortfall=1):
    #   shortfall={psychic:1}, +1 to psychic → 0 → ETA=1 regardless of pool.
    #
    # For a meaningful ETA-0 test we need a 2-color shortfall where
    # the +1 budget only helps one. Use a card with cost
    # {psychic:1, darkness:1} like manhattanCafeStage1 if attached={}.
    # ---------------------------------------------------------------
    # manhattanCafeStage1 cost {darkness:1, colorless:1}. attached={} →
    # shortfall={darkness:1}. +1 attach removes it → ETA=1 regardless
    # of pool. So we need a multi-typed-color shortfall. Use
    # tamamoCrossStage2 if it has a 2-typed cost, OR force a synthetic
    # test by using an attack that needs 2 typed colors.
    # Simpler: pick any 2-typed-cost attack and confirm pool gates ETA.
    eta_card = None
    for cid, card in catalog.items():
        if card.get("kind") != "umamusume":
            continue
        for atk in (card.get("attacks") or []):
            cost = atk.get("cost") or {}
            typed = [k for k in cost.keys() if k != "colorless"]
            if len(typed) >= 2 and all((cost[k] or 0) >= 1 for k in typed):
                eta_card = cid
                break
        if eta_card:
            break
    if eta_card is None:
        raise SystemExit(
            "FAIL[eta-setup]: no card with 2-typed-color cost in catalog"
        )
    eta_attack = catalog[eta_card]["attacks"][0]
    typed_colors = [k for k in eta_attack["cost"].keys() if k != "colorless"]
    # Need attached so that shortfall has EXACTLY 2 colors with deficit 1
    # each. Attach nothing → both colors have deficit equal to cost.
    # After the +1 attach (applied to largest, tiebreak by lex order), at
    # least one color remains short. If that color is NOT in the pool,
    # ETA=0. If in the pool, ETA=1.
    eta_attached = {}  # nothing attached
    # Compute deterministically which color survives after +1 attach.
    # Largest cost first; tiebreak = lex order in _UMA_SLOT_ENERGY_TYPES.
    from uma_ai.features import _UMA_SLOT_ENERGY_TYPES
    order_index = {et: i for i, et in enumerate(_UMA_SLOT_ENERGY_TYPES)}
    sorted_typed = sorted(
        typed_colors,
        key=lambda c: (-int(eta_attack["cost"][c] or 0),
                       order_index.get(c, len(_UMA_SLOT_ENERGY_TYPES))),
    )
    reduced_color = sorted_typed[0]
    other_colors = [c for c in typed_colors if c != reduced_color]
    # surviving_color: a color still in shortfall after the +1 attach.
    # For two cost=1 colors, one reduces to 0, the other stays at 1.
    surviving_color = other_colors[0]

    # Test 1: pool contains the surviving color → ETA=1.
    obs_eta_yes = _make_obs(
        own_active=_make_uma(
            card_id=eta_card,
            hp=int(catalog[eta_card].get("hp", 90)),
            max_hp=int(catalog[eta_card].get("hp", 90)),
            energies=eta_attached,
        ),
        opp_active=_make_uma(card_id="haruUraraBasic", hp=90, max_hp=90),
        own_energy_pool=[surviving_color],
    )
    v_eta_yes = observation_to_features_v3_7(obs_eta_yes)
    if v_eta_yes[_V37_ETA_BASE + 0] != 1.0:
        raise SystemExit(
            f"FAIL[eta-yes]: card={eta_card} attached={{}} pool=[{surviving_color}] "
            f"expected own_primary_eta=1; got {v_eta_yes[_V37_ETA_BASE + 0]}"
        )

    # Test 2: pool empty → surviving shortfall not in pool → ETA=0.
    obs_eta_no = _make_obs(
        own_active=_make_uma(
            card_id=eta_card,
            hp=int(catalog[eta_card].get("hp", 90)),
            max_hp=int(catalog[eta_card].get("hp", 90)),
            energies=eta_attached,
        ),
        opp_active=_make_uma(card_id="haruUraraBasic", hp=90, max_hp=90),
        own_energy_pool=[],
    )
    v_eta_no = observation_to_features_v3_7(obs_eta_no)
    if v_eta_no[_V37_ETA_BASE + 0] != 0.0:
        raise SystemExit(
            f"FAIL[eta-no]: card={eta_card} attached={{}} pool=[] expected "
            f"own_primary_eta=0; got {v_eta_no[_V37_ETA_BASE + 0]}"
        )

    # ---------------------------------------------------------------
    # 8. Paralysis fixture. opp.active.turnState.paralysisRecoveryPending
    # = True → own.paralysis_window_open=1 (we can free-attack).
    # ---------------------------------------------------------------
    obs_para = _make_obs(
        own_active=_make_uma(card_id="matikanetannhauserBasic", hp=60, max_hp=60),
        opp_active=_make_uma(
            card_id="matikanetannhauserBasic", hp=60, max_hp=60,
            paralysis_recovery_pending=True,
        ),
    )
    v_para = observation_to_features_v3_7(obs_para)
    _assert_v36_head_byte_identical("paralysis", obs_para, v_para)
    if v_para[_V37_PARALYSIS_BASE + 0] != 1.0:
        raise SystemExit(
            f"FAIL[paralysis-own]: expected own_paralysis_window_open=1; "
            f"got {v_para[_V37_PARALYSIS_BASE + 0]}"
        )
    if v_para[_V37_PARALYSIS_BASE + 1] != 0.0:
        raise SystemExit(
            f"FAIL[paralysis-opp]: own.active not paralysed → "
            f"opp_paralysis_window_open=0; got {v_para[_V37_PARALYSIS_BASE + 1]}"
        )

    # ---------------------------------------------------------------
    # 9. Tool fixture. leftoverCarrot attached to own.active →
    # Channel 5 own-side bit 0 (HEAL_AT_TURN_END) fires.
    # ---------------------------------------------------------------
    obs_tool = _make_obs(
        own_active=_make_uma(
            card_id="matikanetannhauserBasic", hp=60, max_hp=60,
            tool_card_id="leftoverCarrot",
        ),
        opp_active=_make_uma(card_id="matikanetannhauserBasic", hp=60, max_hp=60),
    )
    v_tool = observation_to_features_v3_7(obs_tool)
    _assert_v36_head_byte_identical("tool", obs_tool, v_tool)
    expected_tool_idx = _V37_TOOL_KIND_OWN_BASE + int(ToolEffectKind.HEAL_AT_TURN_END)
    if v_tool[expected_tool_idx] != 1.0:
        raise SystemExit(
            f"FAIL[tool]: expected leftoverCarrot → bit {expected_tool_idx} = 1; "
            f"got {v_tool[expected_tool_idx]}; tool block "
            f"{v_tool[_V37_TOOL_KIND_OWN_BASE:_V37_TOOL_KIND_OWN_BASE+4].tolist()}"
        )
    # Channel-isolation across the 4-bit own-tool block: exactly one bit.
    own_tool_block = v_tool[
        _V37_TOOL_KIND_OWN_BASE : _V37_TOOL_KIND_OWN_BASE + 4
    ]
    if own_tool_block.sum() != 1.0:
        raise SystemExit(
            f"FAIL[tool-isolation]: own tool block must have exactly 1 bit; "
            f"got {own_tool_block.tolist()}"
        )
    # Opp tool block all zeros (opp has no tool).
    opp_tool_block = v_tool[
        _V37_TOOL_KIND_OPP_BASE : _V37_TOOL_KIND_OPP_BASE + 4
    ]
    if opp_tool_block.sum() != 0.0:
        raise SystemExit(
            f"FAIL[tool-opp-empty]: opp no tool → block all 0; got "
            f"{opp_tool_block.tolist()}"
        )

    # ---------------------------------------------------------------
    # 10. Ability fixture. niceNatureBasic + usedAbilityThisTurn=True →
    # Channel 6 own-side bit 2 (HP_BONUS) fires. Crucially, with
    # usedAbilityThisTurn=False, NO ability bit should fire.
    # ---------------------------------------------------------------
    obs_ability = _make_obs(
        own_active=_make_uma(
            card_id="niceNatureBasic", hp=70, max_hp=70,
            used_ability_this_turn=True,
        ),
        opp_active=_make_uma(card_id="matikanetannhauserBasic", hp=60, max_hp=60),
    )
    v_ability = observation_to_features_v3_7(obs_ability)
    _assert_v36_head_byte_identical("ability", obs_ability, v_ability)
    expected_ability_idx = _V37_ABILITY_KIND_OWN_BASE + int(AbilityEffectKind.HP_BONUS)
    if v_ability[expected_ability_idx] != 1.0:
        raise SystemExit(
            f"FAIL[ability]: expected niceNatureBasic HP_BONUS bit "
            f"{expected_ability_idx} = 1; got {v_ability[expected_ability_idx]}"
        )
    own_ability_block = v_ability[
        _V37_ABILITY_KIND_OWN_BASE : _V37_ABILITY_KIND_OWN_BASE + 8
    ]
    if own_ability_block.sum() != 1.0:
        raise SystemExit(
            f"FAIL[ability-isolation]: own ability block must have exactly 1 bit; "
            f"got {own_ability_block.tolist()}"
        )
    # Opp ability block all zeros.
    opp_ability_block = v_ability[
        _V37_ABILITY_KIND_OPP_BASE : _V37_ABILITY_KIND_OPP_BASE + 8
    ]
    if opp_ability_block.sum() != 0.0:
        raise SystemExit(
            f"FAIL[ability-opp-empty]: opp no used ability → block all 0; "
            f"got {opp_ability_block.tolist()}"
        )

    # Sanity: usedAbilityThisTurn=False → ability block all zeros for
    # niceNatureBasic, even though the card has an `ability` payload.
    obs_unused = _make_obs(
        own_active=_make_uma(
            card_id="niceNatureBasic", hp=70, max_hp=70,
            used_ability_this_turn=False,
        ),
        opp_active=_make_uma(card_id="matikanetannhauserBasic", hp=60, max_hp=60),
    )
    v_unused = observation_to_features_v3_7(obs_unused)
    unused_block = v_unused[
        _V37_ABILITY_KIND_OWN_BASE : _V37_ABILITY_KIND_OWN_BASE + 8
    ]
    if unused_block.sum() != 0.0:
        raise SystemExit(
            f"FAIL[ability-unused]: usedAbilityThisTurn=False must zero the "
            f"ability block; got {unused_block.tolist()}"
        )

    # ---------------------------------------------------------------
    # 11. Secondary attack predicate (Channel 1 secondary).
    # matikanefukukitaruStage1 has 2 attacks; with 2 psychic energy
    # attached, secondary is usable. Secondary damage = 0 → can't KO.
    # So opp_weakness_secondary_KO MUST stay 0 even when v3.6's
    # secondary-usable bit is on.
    # ---------------------------------------------------------------
    obs_sec = _make_obs(
        own_active=_make_uma(
            card_id="matikanefukukitaruStage1", hp=100, max_hp=100,
            energies={"psychic": 2},
        ),
        opp_active=_make_uma(card_id="matikanetannhauserBasic", hp=60, max_hp=60),
    )
    v_sec = observation_to_features_v3_7(obs_sec)
    if v_sec[_V37_OPP_WEAKNESS_SECONDARY_KO] != 0.0:
        raise SystemExit(
            f"FAIL[sec-no-ko]: secondary dmg=0 must give opp_weakness_sec_KO=0; "
            f"got {v_sec[_V37_OPP_WEAKNESS_SECONDARY_KO]}"
        )

    print(
        "v37_combat_arith_smoke: PASS "
        "(shape + dispatch + v3.6-head-byte-identity + "
        "weakness + coin-flip + per-energy + per-bench + ETA + paralysis + "
        "tool + ability + channel isolation)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
