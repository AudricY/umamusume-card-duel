from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

STATE_DIM = 96
ACTION_DIM = 48
STATE_FEATURE_SCHEMA_VERSION = 1
ACTION_FEATURE_SCHEMA_VERSION = 2

PHASES = [
    "setup",
    "pendingChoice",
    "bench",
    "trainerBefore",
    "evolve",
    "attach",
    "trainerAfter",
    "ability",
    "combat",
    "stadiumOrEnd",
]

SIDES = ["player", "opponent"]


FeatureAblation = str


def observation_to_features(observation: dict[str, Any], ablations: set[FeatureAblation] | None = None) -> np.ndarray:
    features = np.zeros(STATE_DIM, dtype=np.float32)
    phase = observation.get("phase", "stadiumOrEnd")
    side = observation.get("sideToAct", "player")
    own = observation.get("own", {})
    opponent = observation.get("opponent", {})
    shared = observation.get("shared", {})

    features[0] = PHASES.index(phase) / max(1, len(PHASES) - 1) if phase in PHASES else 0.0
    features[1] = SIDES.index(side) if side in SIDES else 0.0
    features[2] = float(observation.get("turnNumber", 0)) / 20.0
    features[3] = float(own.get("points", 0)) / 3.0
    features[4] = float(opponent.get("points", 0)) / 3.0
    features[5] = float(own.get("handCount", 0)) / 10.0
    features[6] = float(opponent.get("handCount", 0)) / 10.0
    features[7] = float(own.get("deckCount", 0)) / 50.0
    features[8] = float(opponent.get("deckCount", 0)) / 50.0
    features[9] = 1.0 if shared.get("stadiumCardId") else 0.0
    features[10:18] = _side_board_features(own)
    features[18:26] = _side_board_features(opponent)
    features[26] = float(len(own.get("discard", []))) / 50.0
    features[27] = float(len(opponent.get("discard", []))) / 50.0
    features[28] = float(len(own.get("energyZone", []))) / 4.0
    features[29] = 1.0 if own.get("usedSupporterThisTurn") else 0.0
    features[30] = 1.0 if own.get("usedRetreatThisTurn") else 0.0
    features[31] = 1.0 if own.get("usedStadiumThisTurn") else 0.0
    features[32:48] = _identity_features(own, opponent)
    features[48:58] = _energy_vector((own.get("active") or {}).get("energies", {}))
    features[58:68] = _energy_vector((opponent.get("active") or {}).get("energies", {}))
    features[68:96] = _card_awareness_features(own, opponent, shared)
    apply_state_ablations(features, ablations or set())
    return features


def legal_actions_to_features(actions: list[dict[str, Any]], ablations: set[FeatureAblation] | None = None) -> np.ndarray:
    rows = []
    for action in actions:
        raw = action.get("features", [])
        if len(raw) != ACTION_DIM:
            action_id = action.get("id", "<unknown>")
            raise ValueError(f"Action feature length mismatch for {action_id}: got {len(raw)}, expected {ACTION_DIM}")
        row = np.zeros(ACTION_DIM, dtype=np.float32)
        row[:] = np.asarray(raw, dtype=np.float32)
        apply_action_ablations(row, ablations or set())
        rows.append(row)
    if not rows:
        return np.zeros((0, ACTION_DIM), dtype=np.float32)
    return np.stack(rows, axis=0)


def apply_state_ablations(features: np.ndarray, ablations: set[FeatureAblation]) -> None:
    if "state_identity_hashes" in ablations:
        features[32:48] = 0
    if "state_energy_vectors" in ablations:
        features[48:68] = 0
    if "state_card_awareness" in ablations:
        features[68:96] = 0
    if "state_semantic" in ablations:
        features[10:32] = 0
        features[48:96] = 0


def apply_action_ablations(features: np.ndarray, ablations: set[FeatureAblation]) -> None:
    if "action_source_target" in ablations:
        features[3:10] = 0
        features[13:18] = 0
        features[25:29] = 0
        features[37:48] = 0
    if "action_trainer_semantic" in ablations:
        features[18:25] = 0
        features[32:37] = 0
        features[42:46] = 0
    if "action_tactical" in ablations:
        features[10:12] = 0
        features[29:32] = 0
        features[46:48] = 0
    if "action_semantic" in ablations:
        features[12:48] = 0


def _side_board_features(side: dict[str, Any]) -> np.ndarray:
    active = side.get("active")
    bench = [entry for entry in side.get("bench", []) if entry]
    board = [active] if active else []
    board.extend(bench)
    hp_total = sum(float(entry.get("hp", 0)) for entry in board)
    max_hp_total = max(1.0, sum(float(entry.get("maxHp", 0)) for entry in board))
    energy_total = sum(float(entry.get("energyTotal", 0)) for entry in board)
    stage_total = sum(float(entry.get("stage", 0)) for entry in board)
    damaged = sum(1.0 for entry in board if float(entry.get("hp", 0)) < float(entry.get("maxHp", 0)))
    statuses = sum(float(len(entry.get("specialConditions", []))) for entry in board)
    active_hp_ratio = float(active.get("hp", 0)) / max(1.0, float(active.get("maxHp", 0))) if active else 0.0
    active_energy = float(active.get("energyTotal", 0)) / 6.0 if active else 0.0
    return np.asarray(
        [
            active_hp_ratio,
            active_energy,
            float(len(board)) / 4.0,
            hp_total / max_hp_total,
            energy_total / 12.0,
            stage_total / 8.0,
            damaged / 4.0,
            statuses / 4.0,
        ],
        dtype=np.float32,
    )


def _identity_features(own: dict[str, Any], opponent: dict[str, Any]) -> np.ndarray:
    values = np.zeros(16, dtype=np.float32)
    own_active = own.get("active") or {}
    opponent_active = opponent.get("active") or {}
    values[0] = _hash_to_unit(str(own_active.get("cardId", "")))
    values[1] = _hash_to_unit(str(opponent_active.get("cardId", "")))
    for offset, entry in enumerate((own.get("bench") or [])[:4], start=2):
        values[offset] = _hash_to_unit(str((entry or {}).get("cardId", "")))
    for offset, entry in enumerate((opponent.get("bench") or [])[:4], start=6):
        values[offset] = _hash_to_unit(str((entry or {}).get("cardId", "")))
    values[10] = _hash_average(own.get("handCardIds") or [])
    values[11] = _hash_average(own.get("discard") or [])
    values[12] = _hash_average(opponent.get("discard") or [])
    values[13] = float(len(own.get("handCardIds") or [])) / 10.0
    values[14] = float(len(own.get("bench") or [])) / 4.0
    values[15] = float(len(opponent.get("bench") or [])) / 4.0
    return values


def _energy_vector(energies: dict[str, Any]) -> np.ndarray:
    types = ["grass", "fire", "water", "lightning", "psychic", "fighting", "darkness", "steel", "colorless", "dragon"]
    return np.asarray([float(energies.get(energy_type, 0)) / 4.0 for energy_type in types], dtype=np.float32)


def _card_awareness_features(own: dict[str, Any], opponent: dict[str, Any], shared: dict[str, Any]) -> np.ndarray:
    values = np.zeros(28, dtype=np.float32)
    hand_ids = own.get("handCardIds") or []
    discard_ids = own.get("discard") or []
    own_board = _board_entries(own)
    opponent_board = _board_entries(opponent)
    own_active = own.get("active") or {}
    opponent_active = opponent.get("active") or {}

    values[0:9] = _hand_role_features(hand_ids)
    values[9] = _matching_evolution_count(hand_ids, own_board) / 4.0
    values[10:14] = _uma_readiness_features(own_active)
    values[14] = _ready_attacker_count(own_board) / 4.0
    values[15] = _ability_ready_count(own_board) / 4.0
    values[16:19] = _discard_role_features(discard_ids)
    values[19:23] = _uma_readiness_features(opponent_active)
    values[23] = _can_ko(opponent_active, own_active)
    values[24] = _can_ko(own_active, opponent_active)
    values[25] = _next_energy_matches_active_need(own)
    values[26] = _hash_to_unit(str(shared.get("stadiumCardId", "")))
    values[27] = _ready_attacker_count(opponent_board) / 4.0
    return values


def _hand_role_features(card_ids: list[Any]) -> np.ndarray:
    values = np.zeros(9, dtype=np.float32)
    if not card_ids:
        return values
    attack_damage_total = 0.0
    attack_cards = 0
    for raw_id in card_ids:
        card = _get_card(str(raw_id))
        if not card:
            continue
        if card.get("kind") == "umamusume":
            stage = float(card.get("stage", 0))
            values[0 if stage <= 0 else 1] += 1
            attack = _primary_attack(card)
            attack_damage_total += float(attack.get("damage", 0))
            attack_cards += 1
            if card.get("ability"):
                values[8] += 1
            continue
        if card.get("kind") == "trainer":
            effect = card.get("effect") or {}
            values[2] += 1
            if effect.get("draw") or effect.get("shuffleHandIntoDeckDraw"):
                values[3] += 1
            if effect.get("searchUmamusume") or effect.get("searchEvolutionUmamusume") or effect.get("searchRandomBasicUmamusume"):
                values[4] += 1
            if effect.get("extraEnergyAttach") or effect.get("attachEnergyFromZoneToBench"):
                values[5] += 1
            if effect.get("heal") or effect.get("recoverActiveSpecialConditions"):
                values[6] += 1
            if effect.get("gustOpponent") or effect.get("discardRandomOpponentActiveEnergy") or card.get("trainerType") == "tool":
                values[7] += 1
    values[:8] /= 10.0
    values[8] = (attack_damage_total / max(1, attack_cards)) / 120.0
    return values


def _discard_role_features(card_ids: list[Any]) -> np.ndarray:
    values = np.zeros(3, dtype=np.float32)
    for raw_id in card_ids:
        card = _get_card(str(raw_id))
        if not card:
            continue
        if card.get("kind") == "trainer":
            values[0] += 1
        elif float(card.get("stage", 0)) <= 0:
            values[1] += 1
        else:
            values[2] += 1
    return np.asarray([values[0] / 20.0, values[1] / 10.0, values[2] / 10.0], dtype=np.float32)


def _uma_readiness_features(entry: dict[str, Any]) -> np.ndarray:
    values = np.zeros(4, dtype=np.float32)
    card = _get_card(str(entry.get("cardId", "")))
    if not card or card.get("kind") != "umamusume":
        return values
    attack = _primary_attack(card)
    cost = attack.get("cost") or {}
    energies = entry.get("energies") or {}
    typed_deficit = _typed_energy_deficit(energies, cost)
    total_cost = _total_cost(cost)
    energy_total = float(entry.get("energyTotal", 0))
    values[0] = min(1.5, energy_total / max(1.0, total_cost))
    values[1] = typed_deficit / 4.0
    values[2] = float(attack.get("damage", 0)) / 150.0
    values[3] = 1.0 if typed_deficit <= 0 and energy_total >= total_cost else 0.0
    return values


def _matching_evolution_count(hand_ids: list[Any], board: list[dict[str, Any]]) -> float:
    species_in_play = {str(entry.get("species", "")) for entry in board if entry}
    count = 0.0
    for raw_id in hand_ids:
        card = _get_card(str(raw_id))
        if card and card.get("kind") == "umamusume" and float(card.get("stage", 0)) > 0 and str(card.get("species", "")) in species_in_play:
            count += 1.0
    return count


def _ready_attacker_count(board: list[dict[str, Any]]) -> float:
    return sum(float(_uma_readiness_features(entry)[3] > 0) for entry in board)


def _ability_ready_count(board: list[dict[str, Any]]) -> float:
    count = 0.0
    for entry in board:
        card = _get_card(str(entry.get("cardId", "")))
        if card and card.get("ability") and not entry.get("usedAbilityThisTurn"):
            count += 1.0
    return count


def _can_ko(attacker: dict[str, Any], defender: dict[str, Any]) -> float:
    if not attacker or not defender:
        return 0.0
    readiness = _uma_readiness_features(attacker)
    if readiness[3] <= 0:
        return 0.0
    damage = readiness[2] * 150.0
    return 1.0 if damage >= float(defender.get("hp", 0)) else 0.0


def _next_energy_matches_active_need(side: dict[str, Any]) -> float:
    zone = side.get("energyZone") or []
    active = side.get("active") or {}
    if not zone or not active:
        return 0.0
    card = _get_card(str(active.get("cardId", "")))
    if not card:
        return 0.0
    attack = _primary_attack(card)
    energy_type = str(zone[0])
    required = float((attack.get("cost") or {}).get(energy_type, 0))
    attached = float((active.get("energies") or {}).get(energy_type, 0))
    return 1.0 if required > attached else 0.0


def _board_entries(side: dict[str, Any]) -> list[dict[str, Any]]:
    board = []
    active = side.get("active")
    if active:
        board.append(active)
    board.extend(entry for entry in side.get("bench", []) if entry)
    return board


def _typed_energy_deficit(energies: dict[str, Any], cost: dict[str, Any]) -> float:
    deficit = 0.0
    for energy_type, amount in cost.items():
        if energy_type == "colorless":
            continue
        deficit += max(0.0, float(amount or 0) - float(energies.get(energy_type, 0)))
    return deficit


def _total_cost(cost: dict[str, Any]) -> float:
    return float(sum(float(amount or 0) for amount in cost.values()))


def _primary_attack(card: dict[str, Any]) -> dict[str, Any]:
    attacks = card.get("attacks") or []
    return attacks[0] if attacks else {}


@lru_cache(maxsize=1)
def _card_catalog() -> dict[str, dict[str, Any]]:
    path = Path(__file__).resolve().parents[2] / "shared" / "src" / "data" / "cards.json"
    raw = json.loads(path.read_text(encoding="utf8"))
    return raw.get("baseCards", {})


def _get_card(card_id: str) -> dict[str, Any] | None:
    cards = _card_catalog()
    if card_id in cards:
        return cards[card_id]
    for suffix in ("FullArtGold", "FullArt", "UncommonPlus"):
        if card_id.endswith(suffix):
            base_id = card_id[: -len(suffix)]
            if base_id in cards:
                return cards[base_id]
    return None


def _hash_average(items: list[Any]) -> float:
    if not items:
        return 0.0
    return float(sum(_hash_to_unit(str(item)) for item in items) / len(items))


def _hash_to_unit(text: str) -> float:
    if not text:
        return 0.0
    value = 2166136261
    for char in text:
        value ^= ord(char)
        value = (value * 16777619) & 0xFFFFFFFF
    return float(value) / 4294967295.0
