"""Audit-v2 acceptance gate for the R16-TD 3a production relabel corpus.

Validates per-recipe `traces.jsonl` against scoping § P1:
  (P0) policyTargets shape + |sum-1|<=1e-6 + non-negativity.
  (P1) no rows with <=1 legal actions (forced-state suppression).
  (P1) full 11-key oracle block.
  (P1) stateSource matches the recipe directory name.
  (P1) retained contested-row rate >= --retained-floor (default 0.80).

INFO: per-slice coverage on axes phase/action_kind/turn_bucket/side/
legal_action_count_bucket. Helpers `_action_kind`, `_turn_bucket`,
`_count_bucket`, `_seed_prefix` are lifted from
`training/data_coverage_audit.py` (those are module-private; lifting
avoids coupling the diagnostic tool's argparse to the gate).

Read-only, pure stdlib. Exit 0 on PASS, non-zero on any per-recipe FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DEFAULT_RETAINED_FLOOR = 0.80
POLICY_SUM_TOL = 1e-6
ORACLE_REQUIRED_KEYS = (
    "simulationsRun", "rolloutCrnSamples", "rolloutSteps", "rootValue",
    "rootMeanQ", "rootPriors", "visitDistribution", "rootPriorEntropy",
    "expansions", "leafEvaluations", "haltedEarly",
)
EXPECTED_SOURCES = ("rule-bot-mirror", "policy-vs-rule", "search-vs-rule")
SLICE_AXES = ("phase", "action_kind", "turn_bucket", "side", "legal_action_count_bucket")


# Lifted from training/data_coverage_audit.py (identical semantics).
def _turn_bucket(turn: Any) -> str:
    try:
        t = int(turn)
    except (TypeError, ValueError):
        return "unknown"
    if t <= 2:
        return "t1-2"
    if t <= 5:
        return "t3-5"
    if t <= 9:
        return "t6-9"
    return "t10+"


def _count_bucket(n: int, edges: tuple[int, ...]) -> str:
    prev = 0
    for e in edges:
        if n <= e:
            return f"{prev + 1 if prev else 0}-{e}" if prev else f"<={e}"
        prev = e
    return f">{edges[-1]}"


def _seed_prefix(seed: Any) -> str:
    if seed is None:
        return "none"
    return f"{str(seed)[:3]}xxx"


def _action_kind(example: dict) -> str:
    la = example.get("legalActions", [])
    try:
        ti = int(example.get("selectedActionIndex", -1))
    except (TypeError, ValueError):
        return "unknown"
    if 0 <= ti < len(la):
        return la[ti].get("kind", "unknown")
    return "unknown"


def _slice_values(row: dict) -> dict[str, str]:
    obs = row.get("observation", {})
    la = row.get("legalActions", [])
    return {
        "phase": obs.get("phase", "unknown"),
        "action_kind": _action_kind(row),
        "turn_bucket": _turn_bucket(obs.get("turnNumber")),
        "side": str(row.get("sideId") or row.get("modelSide") or "unknown"),
        "legal_action_count_bucket": _count_bucket(len(la), (1, 2, 4, 8, 16)),
    }


def _check_row(row: dict, expected_source: str) -> list[str]:
    """Return list of failure tags (empty == retained)."""
    fails: list[str] = []
    la = row.get("legalActions")
    pt = row.get("policyTargets")
    if not isinstance(la, list) or not isinstance(pt, list):
        return ["missing_legalActions_or_policyTargets"]
    # (P0)
    if len(pt) != len(la):
        fails.append("policy_length_mismatch")
    else:
        if any((not isinstance(p, (int, float))) or p < 0 for p in pt):
            fails.append("policy_has_negative_or_non_numeric")
        try:
            s = float(sum(pt))
        except TypeError:
            fails.append("policy_non_numeric")
            s = 0.0
        if abs(s - 1.0) > POLICY_SUM_TOL:
            fails.append(f"policy_sum_not_1(sum={s:.9f})")
    # (P1) forced-state suppression.
    if len(la) <= 1:
        fails.append("forced_state_leq_1_legal")
    # (P1) oracle completeness.
    oracle = row.get("oracle")
    if not isinstance(oracle, dict):
        fails.append("oracle_missing")
    else:
        missing = [k for k in ORACLE_REQUIRED_KEYS if k not in oracle]
        if missing:
            fails.append(f"oracle_missing_keys({','.join(missing)})")
    # (P1) stateSource.
    if row.get("stateSource") != expected_source:
        fails.append(
            f"stateSource_mismatch(got={row.get('stateSource')!r},expected={expected_source!r})"
        )
    return fails


def _audit_one(trace_path: Path, recipe: str, retained_floor: float) -> dict[str, Any]:
    total = 0
    retained = 0
    failure_counts: Counter[str] = Counter()
    failure_examples: dict[str, dict[str, Any]] = {}
    slice_counts: dict[str, dict[str, int]] = {ax: defaultdict(int) for ax in SLICE_AXES}
    stateSource_seen: Counter[str] = Counter()
    parse_errors = 0
    parse_error_examples: list[dict[str, Any]] = []

    with trace_path.open("r", encoding="utf8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                parse_errors += 1
                if len(parse_error_examples) < 3:
                    parse_error_examples.append({"line_no": line_no, "error": str(exc)})
                failure_counts["json_decode_error"] += 1
                continue
            stateSource_seen[str(row.get("stateSource"))] += 1
            row_fails = _check_row(row, recipe)
            if not row_fails:
                retained += 1
                sv = _slice_values(row)
                for ax in SLICE_AXES:
                    slice_counts[ax][sv[ax]] += 1
            else:
                for tag in row_fails:
                    head = tag.split("(", 1)[0]
                    failure_counts[head] += 1
                    if head not in failure_examples:
                        failure_examples[head] = {
                            "line_no": line_no, "tag": tag,
                            "seed": row.get("seed"), "step": row.get("step"),
                            "sideId": row.get("sideId"),
                            "phase": row.get("observation", {}).get("phase"),
                        }

    retained_fraction = (retained / total) if total else 0.0
    status_reasons: list[str] = []
    if total == 0:
        status_reasons.append("empty_trace")
    if parse_errors > 0:
        status_reasons.append(f"json_decode_errors={parse_errors}")
    if retained_fraction < retained_floor:
        status_reasons.append(
            f"retained_fraction={retained_fraction:.4f} < floor={retained_floor:.4f}"
        )
    status = "PASS" if not status_reasons else "FAIL"
    stateSource_match = (
        stateSource_seen.get(recipe, 0) == total and total > 0 and len(stateSource_seen) == 1
    )
    materialized = {
        ax: {k: v for k, v in sorted(buckets.items())} for ax, buckets in slice_counts.items()
    }

    return {
        "status": status,
        "total_rows": total,
        "retained_rows": retained,
        "retained_fraction": round(retained_fraction, 6),
        "failures": [
            {"tag": tag, "count": cnt, "example": failure_examples.get(tag)}
            for tag, cnt in failure_counts.most_common()
        ],
        "status_reasons": status_reasons,
        "slice_counts": materialized,
        "stateSource_match": stateSource_match,
        "stateSource_seen": dict(stateSource_seen),
        "parse_errors": parse_errors,
        "parse_error_examples": parse_error_examples,
    }


def _summarize_top_slice(slice_counts: dict[str, dict[str, int]]) -> str:
    best_axis = None
    best_share = 0.0
    best_bucket = ""
    best_count = 0
    best_total = 0
    for ax, buckets in slice_counts.items():
        tot = sum(buckets.values())
        if tot == 0:
            continue
        top_bucket, top_count = max(buckets.items(), key=lambda kv: kv[1])
        share = top_count / tot
        if share > best_share:
            best_share, best_axis, best_bucket, best_count, best_total = (
                share, ax, top_bucket, top_count, tot,
            )
    if best_axis is None:
        return "(no slice data)"
    return f"top skew: {best_axis}=`{best_bucket}` {best_count}/{best_total} ({best_share:.1%})"


def _print_recipe_line(name: str, rep: dict[str, Any]) -> None:
    print(
        f"[{rep['status']}] {name}: {rep['retained_rows']}/{rep['total_rows']} retained "
        f"({rep['retained_fraction']:.2%}); stateSource_match={rep['stateSource_match']}; "
        + _summarize_top_slice(rep["slice_counts"])
    )
    for f in rep["failures"][:3]:
        print(f"    - {f['tag']} x{f['count']}  example={f['example']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--recipe-trace", action="append", required=True, metavar="RECIPE=PATH",
        help="Repeatable. RECIPE must be one of: " + ", ".join(EXPECTED_SOURCES),
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--retained-floor", type=float, default=DEFAULT_RETAINED_FLOOR)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    recipe_paths: dict[str, Path] = {}
    for spec in args.recipe_trace:
        if "=" not in spec:
            print(f"ERROR: --recipe-trace must be RECIPE=PATH, got {spec!r}", file=sys.stderr)
            return 2
        recipe, p = spec.split("=", 1)
        recipe = recipe.strip()
        if recipe not in EXPECTED_SOURCES:
            print(f"ERROR: unknown recipe {recipe!r}; expected one of {EXPECTED_SOURCES}", file=sys.stderr)
            return 2
        if recipe in recipe_paths:
            print(f"ERROR: recipe {recipe!r} specified more than once", file=sys.stderr)
            return 2
        path = Path(p)
        if not path.exists():
            print(f"ERROR: trace file not found: {path}", file=sys.stderr)
            return 2
        recipe_paths[recipe] = path

    partial = len(recipe_paths) < len(EXPECTED_SOURCES)
    if partial:
        missing = [s for s in EXPECTED_SOURCES if s not in recipe_paths]
        print(f"[partial audit] only {sorted(recipe_paths)} provided; missing {missing}")

    per_recipe = {r: _audit_one(p, r, args.retained_floor) for r, p in recipe_paths.items()}

    agg_total = sum(r["total_rows"] for r in per_recipe.values())
    agg_retained = sum(r["retained_rows"] for r in per_recipe.values())
    agg_fraction = (agg_retained / agg_total) if agg_total else 0.0
    any_fail = any(r["status"] == "FAIL" for r in per_recipe.values())
    aggregate = {
        "status": "FAIL" if any_fail else "PASS",
        "total_rows": agg_total,
        "retained_rows": agg_retained,
        "retained_fraction": round(agg_fraction, 6),
        "source_mix": {name: r["total_rows"] for name, r in per_recipe.items()},
        "expected_sources_present": [s for s in EXPECTED_SOURCES if s in per_recipe],
        "missing_sources": [s for s in EXPECTED_SOURCES if s not in per_recipe],
        "partial_audit": partial,
    }
    report = {
        "schemaVersion": SCHEMA_VERSION,
        "retainedFloor": args.retained_floor,
        "per_recipe": per_recipe,
        "aggregate": aggregate,
        "input_paths": {name: str(p) for name, p in recipe_paths.items()},
        "args": {
            "recipe_trace": list(args.recipe_trace),
            "out": str(args.out),
            "retained_floor": args.retained_floor,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf8")

    print("=== audit_relabel_corpus (Audit-v2) ===")
    print(f"retained_floor: {args.retained_floor}")
    for recipe in EXPECTED_SOURCES:
        if recipe in per_recipe:
            _print_recipe_line(recipe, per_recipe[recipe])
    print(
        f"[{aggregate['status']}] AGGREGATE: "
        f"{aggregate['retained_rows']}/{aggregate['total_rows']} retained "
        f"({aggregate['retained_fraction']:.2%}); source_mix={aggregate['source_mix']}"
    )
    if aggregate["missing_sources"]:
        print(
            f"  (partial audit; missing {aggregate['missing_sources']} does not FAIL aggregate)"
        )
    print(f"wrote: {args.out}")
    return 0 if aggregate["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
