from __future__ import annotations

from typing import Any

import numpy as np

STATE_DIM = 96
ACTION_DIM = 32

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


def observation_to_features(observation: dict[str, Any]) -> np.ndarray:
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
    return features


def legal_actions_to_features(actions: list[dict[str, Any]]) -> np.ndarray:
    rows = []
    for action in actions:
        raw = action.get("features", [])
        row = np.zeros(ACTION_DIM, dtype=np.float32)
        limit = min(ACTION_DIM, len(raw))
        if limit:
            row[:limit] = np.asarray(raw[:limit], dtype=np.float32)
        rows.append(row)
    if not rows:
        return np.zeros((0, ACTION_DIM), dtype=np.float32)
    return np.stack(rows, axis=0)


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
