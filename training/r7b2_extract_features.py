"""R7.b.2 Phase 4 — offline feature re-extractor for pre-Phase-1 corpora.

R7's trace corpus (`runs/R7-multi-teacher-warmstart/iter-000/mixed.jsonl`)
was generated BEFORE Phase 1's TS-side schema bump added `cardIdsByZone`
to `PublicObservation` and `actionSourceCardIdx`/`actionTargetCardIdx` to
`LegalAiAction`. Phase 2's `observation_to_card_ids` raises if
`cardIdsByZone` is missing (fail-loud, see `training/uma_ai/features.py`
§ docstring around line 144), so before any Phase 5 SL retrain can run,
the pre-Phase-1 rows must be enriched with the new fields offline.

Per scoping § 9 (R7.b.0 spike VERDICT YES), every needed input field is
preserved in each trace row's raw `observation` (full `PublicObservation`
snapshot) and `legalActions[i].payload`. That makes this a
feature-re-extraction pass, NOT a DAgger regeneration: stream-read each
row, compute the two new structures from the existing raw fields, write
the enriched row.

Per-action source/target resolution table (mirrors
`frontend/src/game/engine/ai-policy/actions.ts`):

  kind                  | actionSourceCardIdx                | actionTargetCardIdx
  ----------------------+------------------------------------+-------------------------------------
  setupChooseBoard      | activeHandIndex → handCardIds → vi | null
  resolvePendingChoice  | null                               | uid→cardId → vi  (pending uid is own)
  playBasic             | handCardIds[handIndex] → vi        | null
  playTrainer           | handCardIds[handIndex] → vi        | choices.umamusumeTargetUid uid→ci→vi
  evolve                | handCardIds[handIndex] → vi        | targetUid uid→ci→vi
  attachEnergy          | null                               | targetUid uid→ci→vi
  useAbility            | sourceUid uid→ci→vi                | (energySourceUid|targetUid|sourceUid)
  attack / retreatAttack| retreatTargetUid|own active        | attackTargetUid uid→ci→vi else null *
  useStadium            | null                               | null
  endTurn               | null                               | null
  pass                  | null                               | null

  * `attack`/`retreatAttack` target is set ONLY when the underlying combat
    decision carries `attackTargetUid` (the planner only emits that for
    any-target attacks). For direct-active attacks, the TS code leaves
    `target = undefined` → `actionTargetCardIdx: null`. We mirror this:
    R7's existing corpus has 0 / 1339 attack actions with
    `attackTargetUid`, so all observed attack targets pad to `null` here.

CLI:
    python training/r7b2_extract_features.py \\
        --input runs/R7-multi-teacher-warmstart/iter-000/mixed.jsonl \\
        --output runs/R7-multi-teacher-warmstart/iter-000/mixed-v3.jsonl

No defaults — fail loud if either flag is missing.

Out of scope this slot: running it on the actual R7 corpus. Phase 5's
brief starts with "run the re-extractor on R7's mixed.jsonl, then
launch retrain."
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Reuse the canonical Python card-vocab helper; keeps Python and TS-side
# (`shared/src/cardVocab.ts`) in lockstep on suffix-fallback ordering.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from uma_ai.features import ZONE_ORDER, card_vocab_index  # noqa: E402


def build_uid_to_card_id(observation: dict[str, Any]) -> dict[int, str]:
    """Walk own/opp active+bench entries and emit uid → cardId map.

    R7's payloads carry uid integers; uid is unique within a row across
    own AND opponent boards, so a single flat dict suffices. Empty bench
    slots (`null`) are skipped. Missing/None uid is skipped silently
    (defense-in-depth — every legitimate uma carries one).
    """

    lookup: dict[int, str] = {}
    for side_key in ("own", "opponent"):
        side = observation.get(side_key) or {}
        active = side.get("active")
        if active and "uid" in active and active.get("cardId"):
            lookup[int(active["uid"])] = str(active["cardId"])
        for entry in side.get("bench") or []:
            if entry and "uid" in entry and entry.get("cardId"):
                lookup[int(entry["uid"])] = str(entry["cardId"])
    return lookup


def build_card_ids_by_zone(observation: dict[str, Any]) -> dict[str, list[int]]:
    """Compute the 8-zone variable-length cardVocab idx arrays.

    Mirrors `frontend/src/game/engine/ai-policy/observation.ts:29-44`
    `buildCardIdsByZone`. TS-side emits variable-length lists per zone;
    Phase 2 Python collator does the fixed-width pad at load time.
    Opponent hand is hidden → empty array (TS-side behaviour).
    """

    own = observation.get("own") or {}
    opp = observation.get("opponent") or {}
    shared = observation.get("shared") or {}

    own_active = own.get("active") or {}
    opp_active = opp.get("active") or {}
    stadium_card_id = shared.get("stadiumCardId")

    def bench_idxs(side: dict[str, Any]) -> list[int]:
        # `bench` entries are null-padded TS-side (length === MAX_BENCH)
        # in the saved row; filter the empties so the variable-length
        # output matches what TS-side emits (skips the null slots).
        return [
            card_vocab_index(str(entry["cardId"]))
            for entry in (side.get("bench") or [])
            if entry and entry.get("cardId")
        ]

    return {
        "ownActive": [card_vocab_index(str(own_active["cardId"]))] if own_active.get("cardId") else [],
        "oppActive": [card_vocab_index(str(opp_active["cardId"]))] if opp_active.get("cardId") else [],
        "ownBench": bench_idxs(own),
        "oppBench": bench_idxs(opp),
        "ownHand": [card_vocab_index(str(card_id)) for card_id in (own.get("handCardIds") or [])],
        "ownDiscard": [card_vocab_index(str(card_id)) for card_id in (own.get("discard") or [])],
        "oppDiscard": [card_vocab_index(str(card_id)) for card_id in (opp.get("discard") or [])],
        "stadium": [card_vocab_index(str(stadium_card_id))] if stadium_card_id else [],
    }


def resolve_action_card_idx(
    action: dict[str, Any],
    observation: dict[str, Any],
    uid_to_card_id: dict[int, str],
) -> tuple[int | None, int | None]:
    """Mirror `frontend/src/game/engine/ai-policy/actions.ts` per-kind
    source/target resolution against the saved row's payload + observation.

    Returns `(source_vocab_idx, target_vocab_idx)` where either side is
    `None` if no clear source/target applies. The Phase 2 Python collator
    converts `None` → 0 (the shared `padding_idx=0` of the embedding
    table; also the `unknownIndex` per `shared/src/cardVocab.json`).
    """

    kind = action.get("kind")
    payload = action.get("payload") or {}
    own = observation.get("own") or {}
    hand = own.get("handCardIds") or []
    own_active = own.get("active") or {}

    def vi_of_uid(uid: Any) -> int | None:
        if uid is None:
            return None
        try:
            card_id = uid_to_card_id.get(int(uid))
        except (TypeError, ValueError):
            return None
        return card_vocab_index(card_id) if card_id else None

    def vi_of_hand(hand_index: Any) -> int | None:
        try:
            idx = int(hand_index)
        except (TypeError, ValueError):
            return None
        if idx < 0 or idx >= len(hand):
            return None
        return card_vocab_index(str(hand[idx]))

    if kind == "pass" or kind == "endTurn" or kind == "useStadium":
        return (None, None)

    if kind == "setupChooseBoard":
        return (vi_of_hand(payload.get("activeHandIndex")), None)

    if kind == "resolvePendingChoice":
        return (None, vi_of_uid(payload.get("targetUid")))

    if kind == "playBasic":
        return (vi_of_hand(payload.get("handIndex")), None)

    if kind == "playTrainer":
        choices = payload.get("choices") or {}
        target_uid = choices.get("umamusumeTargetUid")
        return (
            vi_of_hand(payload.get("handIndex")),
            vi_of_uid(target_uid),
        )

    if kind == "evolve":
        return (
            vi_of_hand(payload.get("handIndex")),
            vi_of_uid(payload.get("targetUid")),
        )

    if kind == "attachEnergy":
        return (None, vi_of_uid(payload.get("targetUid")))

    if kind == "useAbility":
        # actions.ts emits useAbility in four shapes:
        #   * move energy        → target = energySourceUid
        #   * damage opponent any → target = targetUid (opp side)
        #   * discard-to-draw     → target = source (own ability uses self)
        #   * default no-target   → target = source
        source_idx = vi_of_uid(payload.get("sourceUid"))
        if "energySourceUid" in payload:
            return (source_idx, vi_of_uid(payload.get("energySourceUid")))
        if "targetUid" in payload:
            return (source_idx, vi_of_uid(payload.get("targetUid")))
        # discardHandIndex variant and bare-call variant both target the
        # source uma in actions.ts (`target: source` in both literals).
        return (source_idx, source_idx)

    if kind == "attack" or kind == "retreatAttack":
        decision = payload.get("decision") or {}
        # Source: if retreatTargetUid is set, that bench uma is the new
        # active doing the attack; otherwise own.active.
        retreat_uid = decision.get("retreatTargetUid")
        if retreat_uid is not None:
            source_idx = vi_of_uid(retreat_uid)
        else:
            source_card_id = own_active.get("cardId") if own_active else None
            source_idx = card_vocab_index(str(source_card_id)) if source_card_id else None
        # Target: actions.ts only sets `target` if decision carries
        # `attackTargetUid` (any-target attacks). For direct-active
        # attacks the target stays undefined → null. Mirror exactly:
        # don't synthesise opp.active here, because TS-side doesn't.
        attack_target_uid = decision.get("attackTargetUid")
        target_idx = vi_of_uid(attack_target_uid) if attack_target_uid is not None else None
        return (source_idx, target_idx)

    # Unknown kind → fail safe with null/null; matches actions.ts pass
    # default and avoids silently labelling a future kind.
    return (None, None)


def process_row(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Enrich a single row in-place semantics; return (row, per-row stats).

    Stats keys:
      zone_nonempty: Counter[zone -> 1 if zone produced >=1 idx]
      action_source_nonnull_by_kind: Counter[kind]
      action_target_nonnull_by_kind: Counter[kind]
      action_count_by_kind: Counter[kind]
    """

    observation = row.get("observation") or {}

    # Inject `cardIdsByZone` onto the observation.
    card_ids_by_zone = build_card_ids_by_zone(observation)
    observation["cardIdsByZone"] = card_ids_by_zone
    # Bump nested observation schemaVersion 1 → 2 to match Phase 1 TS
    # writer. The outer `row.schemaVersion` STAYS at 1 because
    # `training/uma_ai/dataset.py:ROW_SCHEMA_VERSION` is still 1 (Phase 2
    # split decision; the substantive guard is the presence of
    # `cardIdsByZone`, enforced by `observation_to_card_ids`).
    observation["schemaVersion"] = 2
    row["observation"] = observation

    uid_to_card_id = build_uid_to_card_id(observation)

    legal_actions = row.get("legalActions") or []
    zone_nonempty = Counter({zone: 1 for zone, ids in card_ids_by_zone.items() if ids})
    action_source_nonnull_by_kind: Counter[str] = Counter()
    action_target_nonnull_by_kind: Counter[str] = Counter()
    action_count_by_kind: Counter[str] = Counter()

    for action in legal_actions:
        source_idx, target_idx = resolve_action_card_idx(action, observation, uid_to_card_id)
        action["actionSourceCardIdx"] = source_idx
        action["actionTargetCardIdx"] = target_idx
        kind = str(action.get("kind", "<unknown>"))
        action_count_by_kind[kind] += 1
        if source_idx is not None:
            action_source_nonnull_by_kind[kind] += 1
        if target_idx is not None:
            action_target_nonnull_by_kind[kind] += 1

    stats = {
        "zone_nonempty": zone_nonempty,
        "action_source_nonnull_by_kind": action_source_nonnull_by_kind,
        "action_target_nonnull_by_kind": action_target_nonnull_by_kind,
        "action_count_by_kind": action_count_by_kind,
    }
    return row, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", required=True, help="Path to pre-Phase-1 v1-schema mixed.jsonl")
    parser.add_argument("--output", required=True, help="Path to write enriched v3 JSONL")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise SystemExit(f"r7b2_extract_features: input does not exist: {in_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_processed = 0
    zone_totals: Counter[str] = Counter()
    action_source_nonnull_by_kind: Counter[str] = Counter()
    action_target_nonnull_by_kind: Counter[str] = Counter()
    action_count_by_kind: Counter[str] = Counter()

    with in_path.open("r", encoding="utf8") as in_handle, out_path.open("w", encoding="utf8") as out_handle:
        for line_number, line in enumerate(in_handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"r7b2_extract_features: {in_path}:{line_number}: invalid JSON: {exc}") from exc
            row, stats = process_row(row)
            zone_totals.update(stats["zone_nonempty"])
            action_source_nonnull_by_kind.update(stats["action_source_nonnull_by_kind"])
            action_target_nonnull_by_kind.update(stats["action_target_nonnull_by_kind"])
            action_count_by_kind.update(stats["action_count_by_kind"])
            out_handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            rows_processed += 1

    summary = {
        "rows_processed": rows_processed,
        "zone_nonempty_count": {zone: zone_totals.get(zone, 0) for zone in ZONE_ORDER},
        "action_count_by_kind": dict(sorted(action_count_by_kind.items())),
        "action_source_nonnull_by_kind": dict(sorted(action_source_nonnull_by_kind.items())),
        "action_target_nonnull_by_kind": dict(sorted(action_target_nonnull_by_kind.items())),
        "input": str(in_path),
        "output": str(out_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
