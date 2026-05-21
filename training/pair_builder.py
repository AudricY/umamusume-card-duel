"""R16-TD 3b chunk 5d: preference-pair builder for MCTS-relabel rows.

Reads R16-TD 3a relabel JSONL rows and emits explicit preference-pair
JSONL in one or more *negative-selection modes*. Modes:

  * ``runner-up`` (default, chunk-1 baseline): winner = argmax visit
    share, loser = second-highest visit-share index (tie-break: lower
    index). Negative kind ``runner_up``.
  * ``top-k``: winner vs ranks ``2..K`` of visit share. K is set via
    ``--top-k`` (default 3, i.e. ranks 2 and 3). Each loser keeps the
    rank in its negative kind (``top_k_rank_2`` / ``top_k_rank_3`` ...).
  * ``rule-bot``: winner vs ``heuristicSelectedActionIndex`` whenever the
    rule-bot's pick differs from the MCTS winner. Negative kind
    ``rule_bot``.

Modes are selected via comma-separated ``--modes`` (e.g.
``--modes runner-up,top-k,rule-bot``). With no flag, behaviour is
bit-identical to chunk 1 (``runner-up`` only).

Per-state cap (``--max-pairs-per-state``, default 3) caps the number of
pairs emitted from a single relabel row across all enabled modes. When
more candidates than the cap are available, the priority order is:
``runner-up`` → ``top-k`` (ranks ascending) → ``rule-bot``. Pairs that
get truncated by the cap are *not* counted as drops — they are simply
not emitted (capped pairs are reported via ``pairs_capped`` for INFO).

Schema follows ``docs/ai-research/scoping/r16-training-data-backlog-
refinement.md`` § "Proposed Pair Schema". Backward-compat is locked by
the smoke test ``training/pair_builder_smoke.py``.

Input schema is the MCTS row emitted by ``backend/src/sim/dagger/
relabelDecisionTrace.ts``: ``seed``, ``step``, ``sideId``,
``observation``, ``legalActions``, ``oracle.{visitDistribution, ...}``,
plus ``heuristicSelectedActionIndex`` for the rule-bot mode.

Drop predicates (applied in order; each counted in the manifest):
  1. no_oracle              — oracle or visitDistribution missing/empty
  2. too_few_legal_actions  — legalActions.length < 2 (should be empty
                              on 3a corpus; emit a stderr warning if not)
  3. oracle_zero_visits     — sum(visitDistribution) == 0
  4. oracle_shape_mismatch  — visitDistribution.length != legalActions.length
  5. winner_equals_loser    — degenerate candidate (loser == winner or
                              negative loser index)
  6. index_out_of_range     — winner/loser/heuristic index outside
                              legalActions
  7. margin_below_floor     — visitShareWinner - visitShareLoser < min_margin
  8. rule_bot_matches_winner — heuristic pick equals MCTS winner (rule-bot
                              mode only; structural, not a quality issue)

Drops 1-4 are row-level (block all modes for the row). Drops 5-8 are
per-candidate (some modes' candidates may survive when others drop).

Pure stdlib; streaming JSONL processing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator, TextIO

SCHEMA_VERSION = 1
DEFAULT_MIN_MARGIN = 0.05
DEFAULT_MAX_PAIRS_PER_STATE = 3
DEFAULT_TOP_K = 3
SOURCE_KIND = "mcts-relabel"

MODE_RUNNER_UP = "runner-up"
MODE_TOP_K = "top-k"
MODE_RULE_BOT = "rule-bot"
SUPPORTED_MODES = (MODE_RUNNER_UP, MODE_TOP_K, MODE_RULE_BOT)
DEFAULT_MODES = (MODE_RUNNER_UP,)

DROP_REASONS = (
    "no_oracle",
    "too_few_legal_actions",
    "oracle_zero_visits",
    "oracle_shape_mismatch",
    "winner_equals_loser",
    "index_out_of_range",
    "margin_below_floor",
    "rule_bot_matches_winner",
)

HISTOGRAM_BUCKETS = ("0.05-0.10", "0.10-0.20", ">=0.20")


def _new_counters() -> dict[str, int]:
    return {reason: 0 for reason in DROP_REASONS}


def _new_histogram() -> dict[str, int]:
    return {b: 0 for b in HISTOGRAM_BUCKETS}


def _new_negative_kind_counts() -> dict[str, int]:
    # Populated dynamically; declared empty so unused buckets stay at 0
    # rather than appearing as missing keys.
    return {}


def _bucket_for_margin(margin: float) -> str:
    if margin < 0.10:
        return "0.05-0.10"
    if margin < 0.20:
        return "0.10-0.20"
    return ">=0.20"


def _parse_modes(spec: str | None) -> tuple[str, ...]:
    """Parse the comma-separated --modes flag.

    Returns a tuple of modes in their canonical priority order
    (``runner-up`` → ``top-k`` → ``rule-bot``), de-duplicated and
    validated. ``None`` / empty string → defaults to (runner-up,).
    """
    if spec is None or spec.strip() == "":
        return DEFAULT_MODES
    raw = [m.strip() for m in spec.split(",") if m.strip()]
    seen: set[str] = set()
    for m in raw:
        if m not in SUPPORTED_MODES:
            raise ValueError(
                f"unsupported mode: {m!r}; expected one of {SUPPORTED_MODES}"
            )
        seen.add(m)
    # Canonical priority order: runner-up first, top-k next, rule-bot last.
    ordered = tuple(m for m in SUPPORTED_MODES if m in seen)
    return ordered


def _ranked_by_visit(vd: list[float]) -> list[int]:
    """Return indices of ``vd`` sorted by visit share descending.

    Tie-break: lower index first (stable on equal visits). This mirrors
    the deterministic argmax convention in
    ``relabelDecisionTrace.ts``.
    """
    n = len(vd)
    # Use Python's stable sort with (-visit, index) — stable sort + key
    # already returns lower-index-first on ties, so the explicit second
    # key is just defensive.
    return sorted(range(n), key=lambda i: (-vd[i], i))


def _candidate_pairs(
    row: dict[str, Any],
    *,
    modes: tuple[str, ...],
    min_margin: float,
    top_k: int,
    max_pairs_per_state: int,
    drop_counter: dict[str, int],
) -> tuple[list[dict[str, Any]], int]:
    """Generate pair dicts for one row, applying modes + per-state cap.

    Returns ``(pairs, capped_count)``. ``pairs`` is the list of emitted
    pair dicts (already truncated to ``max_pairs_per_state``). Row-level
    drop reasons (no_oracle, too_few_legal_actions, ...) increment
    ``drop_counter`` and yield ``([], 0)``. Per-candidate drop reasons
    also increment ``drop_counter`` per candidate.
    """
    oracle = row.get("oracle")
    if not isinstance(oracle, dict):
        drop_counter["no_oracle"] += 1
        return [], 0
    vd = oracle.get("visitDistribution")
    if not isinstance(vd, list) or len(vd) == 0:
        drop_counter["no_oracle"] += 1
        return [], 0

    legal = row.get("legalActions")
    if not isinstance(legal, list) or len(legal) < 2:
        drop_counter["too_few_legal_actions"] += 1
        return [], 0

    vd_sum = 0.0
    for v in vd:
        try:
            vd_sum += float(v)
        except (TypeError, ValueError):
            drop_counter["oracle_shape_mismatch"] += 1
            return [], 0
    if vd_sum <= 0.0:
        drop_counter["oracle_zero_visits"] += 1
        return [], 0

    if len(vd) != len(legal):
        drop_counter["oracle_shape_mismatch"] += 1
        return [], 0

    vd_f = [float(v) for v in vd]
    ranked = _ranked_by_visit(vd_f)
    winner_idx = ranked[0]
    n_legal = len(legal)
    if not (0 <= winner_idx < n_legal):
        drop_counter["index_out_of_range"] += 1
        return [], 0

    winner_action = legal[winner_idx]
    if not isinstance(winner_action, dict):
        drop_counter["index_out_of_range"] += 1
        return [], 0
    visit_share_winner = vd_f[winner_idx] / vd_sum

    # Build candidate (loser_idx, negative_kind) list in priority order.
    candidates: list[tuple[int, str]] = []

    if MODE_RUNNER_UP in modes:
        if len(ranked) >= 2:
            loser_idx = ranked[1]
            candidates.append((loser_idx, "runner_up"))

    if MODE_TOP_K in modes:
        # Ranks 2..top_k → indices ranked[1], ranked[2], ..., ranked[top_k-1].
        # If runner-up mode already enabled, rank-2 (ranked[1]) is its
        # candidate — we must NOT double-emit. Top-k contributes ranks
        # that aren't already covered. Implementation: skip ranked[1]
        # iff runner-up is enabled, but always include ranked[2..K-1].
        # Rank labels stay ``top_k_rank_<rank>`` so downstream can split.
        start_rank = 3 if MODE_RUNNER_UP in modes else 2
        for rank in range(start_rank, top_k + 1):
            ridx = rank - 1  # 0-based position in `ranked`
            if ridx >= len(ranked):
                break
            loser_idx = ranked[ridx]
            candidates.append((loser_idx, f"top_k_rank_{rank}"))

    if MODE_RULE_BOT in modes:
        heuristic_idx = row.get("heuristicSelectedActionIndex")
        if isinstance(heuristic_idx, bool) or not isinstance(heuristic_idx, int):
            # Treat missing/non-int as out-of-range; count as index_out_of_range
            # only if rule-bot mode is the *only* contributor — otherwise just
            # silently skip the rule-bot candidate (the runner-up/top-k pairs
            # are still valid). Don't bump a counter for "field absent" to
            # avoid polluting drop totals on policy/search recipes that may
            # omit the heuristic field; we instead record the skip implicitly.
            pass
        elif heuristic_idx == winner_idx:
            drop_counter["rule_bot_matches_winner"] += 1
        elif not (0 <= heuristic_idx < n_legal):
            drop_counter["index_out_of_range"] += 1
        else:
            candidates.append((heuristic_idx, "rule_bot"))

    # Per-candidate drop filtering: margin floor, winner-equals-loser,
    # index-out-of-range. We've already verified winner_idx, but each
    # candidate loser_idx may still hit these.
    obs = row.get("observation") if isinstance(row.get("observation"), dict) else {}
    phase = obs.get("phase") if isinstance(obs, dict) else None
    if not isinstance(phase, str) or not phase:
        phase = "unknown"
    seed = row.get("seed")
    side_id = row.get("sideId")
    step = row.get("step")

    pairs: list[dict[str, Any]] = []
    seen_losers: set[int] = set()
    for loser_idx, negative_kind in candidates:
        if loser_idx == winner_idx:
            drop_counter["winner_equals_loser"] += 1
            continue
        if not (0 <= loser_idx < n_legal):
            drop_counter["index_out_of_range"] += 1
            continue
        loser_action = legal[loser_idx]
        if not isinstance(loser_action, dict):
            drop_counter["index_out_of_range"] += 1
            continue
        visit_share_loser = vd_f[loser_idx] / vd_sum
        margin = visit_share_winner - visit_share_loser
        if margin < min_margin:
            drop_counter["margin_below_floor"] += 1
            continue
        # De-dup: if a later mode would emit a pair for the same loser
        # already covered (e.g. rule-bot lands on the same index as
        # runner-up's loser when ranks happen to align), drop silently
        # rather than double-counting.
        if loser_idx in seen_losers:
            continue
        seen_losers.add(loser_idx)

        pair = {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "preference-pair",
            "sourceKind": SOURCE_KIND,
            "sourceEpisodeId": seed,
            "sourceStep": step,
            "seed": seed,
            "sideId": side_id,
            "phase": phase,
            "observation": obs,
            "legalActions": legal,
            "winner": {
                "index": winner_idx,
                "actionId": winner_action.get("id"),
                "score": visit_share_winner,
            },
            "loser": {
                "index": loser_idx,
                "actionId": loser_action.get("id"),
                "score": visit_share_loser,
                "negativeKind": negative_kind,
            },
            "margin": margin,
            "confidence": {
                "visitShareWinner": visit_share_winner,
                "visitShareLoser": visit_share_loser,
                "sampleCount": oracle.get("simulationsRun"),
            },
            "sampleWeight": 1.0,
        }
        pairs.append(pair)

    # Apply per-state cap. Pairs are already in priority order
    # (runner-up → top-k ranks ascending → rule-bot).
    capped = 0
    if len(pairs) > max_pairs_per_state:
        capped = len(pairs) - max_pairs_per_state
        pairs = pairs[:max_pairs_per_state]
    return pairs, capped


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{lineno}: row is not a JSON object")
            yield row


def process_inputs(
    inputs: list[Path],
    out_path: Path,
    *,
    min_margin: float = DEFAULT_MIN_MARGIN,
    modes: tuple[str, ...] = DEFAULT_MODES,
    top_k: int = DEFAULT_TOP_K,
    max_pairs_per_state: int = DEFAULT_MAX_PAIRS_PER_STATE,
    stderr: TextIO = sys.stderr,
    # Legacy single-mode kwarg for backward-compat with chunk-1 callers.
    mode: str | None = None,
) -> dict[str, Any]:
    """Stream-process inputs, write pairs JSONL, return manifest dict."""
    if mode is not None:
        # chunk-1 callers pass a single mode string; honor it.
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"unsupported mode: {mode!r}")
        modes = (mode,)
    for m in modes:
        if m not in SUPPORTED_MODES:
            raise ValueError(f"unsupported mode: {m!r}")
    if top_k < 2:
        raise ValueError(f"--top-k must be >= 2 (got {top_k})")
    if max_pairs_per_state < 1:
        raise ValueError(
            f"--max-pairs-per-state must be >= 1 (got {max_pairs_per_state})"
        )

    per_input: list[dict[str, Any]] = []
    global_drops = _new_counters()
    global_hist = _new_histogram()
    global_rows_in = 0
    global_pairs = 0
    global_capped = 0
    global_negkind: dict[str, int] = {}
    pair_source_counts: dict[str, int] = {}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf8") as out_handle:
        for in_path in inputs:
            in_drops = _new_counters()
            in_hist = _new_histogram()
            in_negkind: dict[str, int] = {}
            rows_in = 0
            pairs_emitted = 0
            pairs_capped = 0
            for row in _iter_jsonl(in_path):
                rows_in += 1
                pairs, capped = _candidate_pairs(
                    row,
                    modes=modes,
                    min_margin=min_margin,
                    top_k=top_k,
                    max_pairs_per_state=max_pairs_per_state,
                    drop_counter=in_drops,
                )
                pairs_capped += capped
                for pair in pairs:
                    out_handle.write(json.dumps(pair, separators=(",", ":")))
                    out_handle.write("\n")
                    pairs_emitted += 1
                    bucket = _bucket_for_margin(pair["margin"])
                    in_hist[bucket] += 1
                    src = pair["sourceKind"]
                    pair_source_counts[src] = pair_source_counts.get(src, 0) + 1
                    nk = pair["loser"]["negativeKind"]
                    in_negkind[nk] = in_negkind.get(nk, 0) + 1

            if in_drops["too_few_legal_actions"] > 0:
                print(
                    f"WARN pair_builder: {in_path} had "
                    f"{in_drops['too_few_legal_actions']} forced-state rows "
                    "(legalActions<2); 3a emit-path should already filter these.",
                    file=stderr,
                )

            # Fold per-input into globals.
            for k, v in in_drops.items():
                global_drops[k] += v
            for k, v in in_hist.items():
                global_hist[k] += v
            for k, v in in_negkind.items():
                global_negkind[k] = global_negkind.get(k, 0) + v
            global_rows_in += rows_in
            global_pairs += pairs_emitted
            global_capped += pairs_capped

            per_input.append(
                {
                    "path": str(in_path),
                    "rows_in": rows_in,
                    "pairs_emitted": pairs_emitted,
                    "pairs_capped": pairs_capped,
                    "drops_by_reason": in_drops,
                    "margin_histogram": in_hist,
                    "negative_kind_counts": in_negkind,
                }
            )

    manifest = {
        "schemaVersion": SCHEMA_VERSION,
        "modes": list(modes),
        # Legacy 'mode' field for chunk-1 backward-compat: only set when
        # exactly one mode is enabled.
        "mode": modes[0] if len(modes) == 1 else None,
        "min_margin": min_margin,
        "top_k": top_k,
        "max_pairs_per_state": max_pairs_per_state,
        "out_path": str(out_path),
        "inputs": per_input,
        "totals": {
            "rows_in": global_rows_in,
            "pairs_emitted": global_pairs,
            "pairs_capped": global_capped,
            "drops_by_reason": global_drops,
            "margin_histogram": global_hist,
            "pair_source_counts": pair_source_counts,
            "negative_kind_counts": global_negkind,
        },
    }
    return manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "R16-TD 3b chunk 5d: emit preference-pair JSONL from MCTS-"
            "relabel traces. Supports runner-up, top-k, and rule-bot "
            "negative-selection modes."
        )
    )
    p.add_argument(
        "--in",
        dest="inputs",
        action="append",
        required=True,
        help="Input relabel-traces JSONL path. Repeat to aggregate.",
    )
    p.add_argument(
        "--out",
        dest="out",
        required=True,
        help="Output pairs JSONL path.",
    )
    p.add_argument(
        "--manifest-out",
        dest="manifest_out",
        required=True,
        help="Output manifest JSON path.",
    )
    p.add_argument(
        "--min-margin",
        type=float,
        default=DEFAULT_MIN_MARGIN,
        help=f"Minimum visit-share margin (default {DEFAULT_MIN_MARGIN}).",
    )
    p.add_argument(
        "--modes",
        default=None,
        help=(
            "Comma-separated negative-selection modes. Choices: "
            f"{','.join(SUPPORTED_MODES)}. Default: {MODE_RUNNER_UP} "
            "(chunk-1 backward-compat)."
        ),
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            f"K for top-k mode (winner vs ranks 2..K). Default "
            f"{DEFAULT_TOP_K}. Ignored unless top-k mode enabled."
        ),
    )
    p.add_argument(
        "--max-pairs-per-state",
        type=int,
        default=DEFAULT_MAX_PAIRS_PER_STATE,
        help=(
            f"Cap on pairs emitted per input row across all enabled "
            f"modes (default {DEFAULT_MAX_PAIRS_PER_STATE})."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    input_paths = [Path(p) for p in args.inputs]
    for ip in input_paths:
        if not ip.is_file():
            print(f"ERROR pair_builder: input not found: {ip}", file=sys.stderr)
            return 2

    out_path = Path(args.out)
    manifest_path = Path(args.manifest_out)

    try:
        modes = _parse_modes(args.modes)
    except ValueError as exc:
        print(f"ERROR pair_builder: {exc}", file=sys.stderr)
        return 2

    manifest = process_inputs(
        input_paths,
        out_path,
        min_margin=args.min_margin,
        modes=modes,
        top_k=args.top_k,
        max_pairs_per_state=args.max_pairs_per_state,
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf8",
    )

    totals = manifest["totals"]
    print(
        f"pair_builder: modes={list(modes)} rows_in={totals['rows_in']} "
        f"pairs_emitted={totals['pairs_emitted']} "
        f"pairs_capped={totals['pairs_capped']} "
        f"negative_kinds={totals['negative_kind_counts']} "
        f"drops={totals['drops_by_reason']} "
        f"histogram={totals['margin_histogram']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
