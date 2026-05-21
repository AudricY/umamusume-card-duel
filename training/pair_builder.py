"""R16-TD 3b chunk 1: preference-pair builder for MCTS-relabel rows.

Reads R16-TD 3a relabel JSONL rows and emits explicit preference-pair
JSONL in *runner-up* mode: winner = argmax(oracle.visitDistribution),
loser = second-highest visit-share index (tie-break: lower index).

Schema follows `docs/ai-research/scoping/r16-training-data-backlog-
refinement.md` § "Proposed Pair Schema" (lines 554-590). This is chunk
1 of 3b — chunk 2 (`pair_corpus.py` integration) and chunk 3
(`train_dpo.py` integration) are intentionally untouched.

Input schema is the MCTS row emitted by
`backend/src/sim/dagger/relabelDecisionTrace.ts:121-146`: `seed`,
`step`, `sideId`, `observation`, `legalActions`, `oracle.{visitDistri-
bution, simulationsRun, ...}`.

Drop predicates (applied in order; each counted in the manifest):
  1. no_oracle              — oracle or visitDistribution missing/empty
  2. too_few_legal_actions  — legalActions.length < 2 (should be empty
                              on 3a corpus; emit a stderr warning if not)
  3. oracle_zero_visits     — sum(visitDistribution) == 0
  4. oracle_shape_mismatch  — visitDistribution.length != legalActions.length
  5. winner_equals_loser    — degenerate runner-up == argmax
  6. index_out_of_range     — winner/loser index outside legalActions
  7. margin_below_floor     — visitShareWinner - visitShareLoser < min_margin

Manifest emits per-input + global counters, margin histogram (buckets
[0.05, 0.10), [0.10, 0.20), [0.20, inf)), and pair-source counts. Pure
stdlib; streaming JSONL processing (manifest stays in memory).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator, TextIO

SCHEMA_VERSION = 1
DEFAULT_MIN_MARGIN = 0.05
DEFAULT_MODE = "runner-up"
SUPPORTED_MODES = ("runner-up",)
SOURCE_KIND = "mcts-relabel"

DROP_REASONS = (
    "no_oracle",
    "too_few_legal_actions",
    "oracle_zero_visits",
    "oracle_shape_mismatch",
    "winner_equals_loser",
    "index_out_of_range",
    "margin_below_floor",
)

HISTOGRAM_BUCKETS = ("0.05-0.10", "0.10-0.20", ">=0.20")


def _new_counters() -> dict[str, int]:
    return {reason: 0 for reason in DROP_REASONS}


def _new_histogram() -> dict[str, int]:
    return {b: 0 for b in HISTOGRAM_BUCKETS}


def _bucket_for_margin(margin: float) -> str:
    if margin < 0.10:
        return "0.05-0.10"
    if margin < 0.20:
        return "0.10-0.20"
    return ">=0.20"


def _argmax_and_runner_up(vd: list[float]) -> tuple[int, int]:
    """Return (winner_index, loser_index) by visit share.

    Ties broken by *lower* index (i.e., earliest in legalActions wins on
    a tie). This mirrors the deterministic argmax convention in the 3a
    emit-path (`relabelDecisionTrace.ts:111-118`).
    """
    n = len(vd)
    if n == 0:
        return -1, -1
    winner = 0
    winner_val = vd[0]
    for i in range(1, n):
        if vd[i] > winner_val:
            winner_val = vd[i]
            winner = i
    if n == 1:
        return winner, -1
    # Second pass for runner-up, skipping the winner index. Tie-break:
    # lower index wins (strict `>` keeps the earlier index on ties).
    loser = -1
    loser_val = -1.0
    for i in range(n):
        if i == winner:
            continue
        if vd[i] > loser_val:
            loser_val = vd[i]
            loser = i
    return winner, loser


def _build_pair(row: dict[str, Any], min_margin: float) -> tuple[dict[str, Any] | None, str | None]:
    """Return (pair_dict, drop_reason). Exactly one is non-None."""
    oracle = row.get("oracle")
    if not isinstance(oracle, dict):
        return None, "no_oracle"
    vd = oracle.get("visitDistribution")
    if not isinstance(vd, list) or len(vd) == 0:
        return None, "no_oracle"

    legal = row.get("legalActions")
    if not isinstance(legal, list) or len(legal) < 2:
        return None, "too_few_legal_actions"

    vd_sum = 0.0
    for v in vd:
        try:
            vd_sum += float(v)
        except (TypeError, ValueError):
            return None, "oracle_shape_mismatch"
    if vd_sum <= 0.0:
        return None, "oracle_zero_visits"

    if len(vd) != len(legal):
        return None, "oracle_shape_mismatch"

    vd_f = [float(v) for v in vd]
    winner_idx, loser_idx = _argmax_and_runner_up(vd_f)

    if winner_idx == loser_idx or loser_idx < 0:
        return None, "winner_equals_loser"

    n_legal = len(legal)
    if not (0 <= winner_idx < n_legal) or not (0 <= loser_idx < n_legal):
        return None, "index_out_of_range"

    visit_share_winner = vd_f[winner_idx] / vd_sum
    visit_share_loser = vd_f[loser_idx] / vd_sum
    margin = visit_share_winner - visit_share_loser
    if margin < min_margin:
        return None, "margin_below_floor"

    winner_action = legal[winner_idx]
    loser_action = legal[loser_idx]
    if not isinstance(winner_action, dict) or not isinstance(loser_action, dict):
        return None, "index_out_of_range"

    obs = row.get("observation") if isinstance(row.get("observation"), dict) else {}
    phase = obs.get("phase") if isinstance(obs, dict) else None
    if not isinstance(phase, str) or not phase:
        phase = "unknown"

    seed = row.get("seed")
    side_id = row.get("sideId")
    step = row.get("step")

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
            "negativeKind": "runner_up",
        },
        "margin": margin,
        "confidence": {
            "visitShareWinner": visit_share_winner,
            "visitShareLoser": visit_share_loser,
            "sampleCount": oracle.get("simulationsRun"),
        },
        "sampleWeight": 1.0,
    }
    return pair, None


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
    min_margin: float,
    mode: str,
    stderr: TextIO = sys.stderr,
) -> dict[str, Any]:
    """Stream-process inputs, write pairs JSONL, return manifest dict."""
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"unsupported mode: {mode!r} (chunk 1 supports {SUPPORTED_MODES})")

    per_input: list[dict[str, Any]] = []
    global_drops = _new_counters()
    global_hist = _new_histogram()
    global_rows_in = 0
    global_pairs = 0
    pair_source_counts: dict[str, int] = {}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf8") as out_handle:
        for in_path in inputs:
            in_drops = _new_counters()
            in_hist = _new_histogram()
            rows_in = 0
            pairs_emitted = 0
            for row in _iter_jsonl(in_path):
                rows_in += 1
                pair, reason = _build_pair(row, min_margin)
                if reason is not None:
                    in_drops[reason] += 1
                    global_drops[reason] += 1
                    continue
                assert pair is not None  # for type checkers
                out_handle.write(json.dumps(pair, separators=(",", ":")))
                out_handle.write("\n")
                pairs_emitted += 1
                bucket = _bucket_for_margin(pair["margin"])
                in_hist[bucket] += 1
                global_hist[bucket] += 1
                src = pair["sourceKind"]
                pair_source_counts[src] = pair_source_counts.get(src, 0) + 1

            if in_drops["too_few_legal_actions"] > 0:
                print(
                    f"WARN pair_builder: {in_path} had "
                    f"{in_drops['too_few_legal_actions']} forced-state rows "
                    "(legalActions<2); 3a emit-path should already filter these.",
                    file=stderr,
                )

            per_input.append(
                {
                    "path": str(in_path),
                    "rows_in": rows_in,
                    "pairs_emitted": pairs_emitted,
                    "drops_by_reason": in_drops,
                    "margin_histogram": in_hist,
                }
            )
            global_rows_in += rows_in
            global_pairs += pairs_emitted

    manifest = {
        "schemaVersion": SCHEMA_VERSION,
        "mode": mode,
        "min_margin": min_margin,
        "out_path": str(out_path),
        "inputs": per_input,
        "totals": {
            "rows_in": global_rows_in,
            "pairs_emitted": global_pairs,
            "drops_by_reason": global_drops,
            "margin_histogram": global_hist,
            "pair_source_counts": pair_source_counts,
        },
    }
    return manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "R16-TD 3b chunk 1: emit preference-pair JSONL from MCTS-"
            "relabel traces (runner-up mode only)."
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
        "--mode",
        default=DEFAULT_MODE,
        choices=SUPPORTED_MODES,
        help=f"Negative-selection mode (default {DEFAULT_MODE!r}).",
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

    manifest = process_inputs(
        input_paths,
        out_path,
        min_margin=args.min_margin,
        mode=args.mode,
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf8",
    )

    totals = manifest["totals"]
    print(
        f"pair_builder: rows_in={totals['rows_in']} "
        f"pairs_emitted={totals['pairs_emitted']} "
        f"drops={totals['drops_by_reason']} "
        f"histogram={totals['margin_histogram']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
