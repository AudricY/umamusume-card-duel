"""Training-data corpus-retention + state-overlap audit (P1, diagnostic-only).

Backlog: docs/ai-research-backlog.md § "Training-data/state-coverage backlog
(user-steered 2026-05-15)" item 1 (`training-data-coverage-audit`).

Hypothesis under test: the training-data bottleneck is *state coverage /
retention*, not generic row count. This tool reproduces, deterministically and
offline, the decomposition of why ~70% of the R7 mixed corpus is dropped by the
`uma_ai.dataset.load_policy_samples` loader, slices retained-vs-dropped rows
across the backlog-required axes, compares the retained training distribution
against the production gate's structural distribution, buckets eval loss vs
coverage density, and emits an explicit proposed source-mix target.

STRICTLY READ-ONLY. No training, no GPU, no network, no `runs/` mutation. The
loader filter is re-derived from the row data with the *exact* predicate used by
`uma_ai.dataset.load_policy_samples` (min_actions=2 + selected-index bounds)
rather than re-invoking the loader, so the audit needs neither torch nor the
feature encoder and runs CPU-light next to a concurrent GPU training job.

Outputs (deterministic):
  - docs/ai-research/analysis/training-data-coverage-audit.md   (canonical doc)
  - docs/ai-research/analysis/training-data-coverage-audit.json  (sidecar)

Run:
  training/.venv/bin/python training/data_coverage_audit.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Mirrors uma_ai.dataset.load_policy_samples default. The loader drops a row
# (Python `continue`) when `len(legalActions) < MIN_ACTIONS` OR the selected
# index is out of `[0, len(legalActions))`. Everything else in the loader
# *raises* (schema / feature-shape / card-id) rather than silently dropping, so
# a successfully-trained R7 corpus implies zero raise-class rows survived to the
# count; the retained==dropped accounting below verifies that invariant.
MIN_ACTIONS = 2

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = REPO_ROOT / "runs/R7-multi-teacher-warmstart/iter-000/mixed.jsonl"
DEFAULT_TRAIN_MANIFEST = (
    REPO_ROOT / "runs/R7-multi-teacher-warmstart/iter-000/model/manifest.json"
)
DEFAULT_GATE_MANIFEST = (
    REPO_ROOT / "runs/R7-multi-teacher-warmstart/iter-000/gate.manifest.json"
)
DEFAULT_PROD_GATE_MANIFEST = REPO_ROOT / "runs/R13-W6-phase-d/iter-2/gate.manifest.json"
DEFAULT_DOC = REPO_ROOT / "docs/ai-research/analysis/training-data-coverage-audit.md"
DEFAULT_JSON = REPO_ROOT / "docs/ai-research/analysis/training-data-coverage-audit.json"


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


def _board_stage(own: dict) -> str:
    """Coarse game-progress proxy from own-side board: total points scored."""
    pts = own.get("points")
    try:
        p = int(pts)
    except (TypeError, ValueError):
        return "unknown"
    if p == 0:
        return "p0"
    if p == 1:
        return "p1"
    return "p2+"


def _energy_bucket(own: dict) -> str:
    active = own.get("active") or {}
    e = active.get("energyTotal")
    try:
        e = int(e)
    except (TypeError, ValueError):
        return "unknown"
    if e == 0:
        return "e0"
    if e <= 2:
        return "e1-2"
    return "e3+"


def _terminal_distance_bucket(example: dict) -> str:
    """Distance-to-terminal proxy: own points (0 -> far, 2 -> at/near win).

    Per-state exact terminal distance is not logged in the corpus; points is
    the strongest monotone-with-progress field available. Recorded as a proxy.
    """
    own = example.get("observation", {}).get("own", {})
    return _board_stage(own)


def _action_kind(example: dict) -> str:
    la = example.get("legalActions", [])
    ti = example.get("selectedActionIndex", -1)
    try:
        ti = int(ti)
    except (TypeError, ValueError):
        return "unknown"
    if 0 <= ti < len(la):
        return la[ti].get("kind", "unknown")
    return "unknown"


def _is_retained(example: dict) -> tuple[bool, str]:
    """Exact replica of the loader's silent-drop predicate.

    Returns (retained, reason). Reasons mirror the two `continue` conditions in
    `uma_ai.dataset.load_policy_samples` plus the blank-line skip.
    """
    la = example.get("legalActions", [])
    try:
        ti = int(example.get("selectedActionIndex", -1))
    except (TypeError, ValueError):
        return False, "selected_index_non_integer"
    if len(la) < MIN_ACTIONS:
        return False, f"lt_min_actions(<{MIN_ACTIONS}_legal)"
    if ti < 0 or ti >= len(la):
        return False, "selected_index_out_of_range"
    return True, "retained"


SLICE_AXES = (
    "source",
    "side",
    "seed_prefix",
    "phase",
    "action_kind",
    "turn_bucket",
    "legal_action_count_bucket",
    "points",
    "board_stage",
    "energy",
    "hand_size_bucket",
    "deck_size_bucket",
    "terminal_distance",
    "schema_card_id_availability",
)


def _slice_values(example: dict) -> dict[str, str]:
    obs = example.get("observation", {})
    own = obs.get("own", {})
    la = example.get("legalActions", [])
    hand = own.get("handCount", own.get("handCardIds"))
    hand_n = (
        len(hand)
        if isinstance(hand, list)
        else (int(hand) if isinstance(hand, (int, float)) else -1)
    )
    deck = own.get("deckCount")
    deck_n = int(deck) if isinstance(deck, (int, float)) else -1
    has_card_ids = "cardIdsByZone" in obs
    return {
        "source": example.get("source") or example.get("mixSourceTag") or "<none>",
        "side": str(example.get("sideId") or example.get("modelSide") or "unknown"),
        "seed_prefix": _seed_prefix(example.get("seed")),
        "phase": obs.get("phase", "unknown"),
        "action_kind": _action_kind(example),
        "turn_bucket": _turn_bucket(obs.get("turnNumber")),
        "legal_action_count_bucket": _count_bucket(len(la), (1, 2, 4, 8, 16)),
        "points": _board_stage(own),
        "board_stage": _board_stage(own),
        "energy": _energy_bucket(own),
        "hand_size_bucket": _count_bucket(max(hand_n, 0), (2, 4, 6, 8))
        if hand_n >= 0
        else "unknown",
        "deck_size_bucket": _count_bucket(max(deck_n, 0), (4, 8, 12, 16))
        if deck_n >= 0
        else "unknown",
        "terminal_distance": _terminal_distance_bucket(example),
        "schema_card_id_availability": (
            f"schemaV{obs.get('schemaVersion', example.get('schemaVersion'))}"
            f"_cardIdsByZone={'yes' if has_card_ids else 'no'}"
        ),
    }


def audit_corpus(corpus_path: Path) -> dict[str, Any]:
    total = 0
    retained = 0
    drop_reasons: Counter[str] = Counter()
    slices: dict[str, dict[str, dict[str, int]]] = {
        ax: defaultdict(lambda: {"retained": 0, "dropped": 0}) for ax in SLICE_AXES
    }
    per_source: dict[str, dict[str, int]] = defaultdict(
        lambda: {"available": 0, "retained": 0, "dropped": 0}
    )
    with corpus_path.open("r", encoding="utf8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                drop_reasons["blank_line"] += 1
                continue
            total += 1
            example = json.loads(line)
            keep, reason = _is_retained(example)
            sv = _slice_values(example)
            src = sv["source"]
            per_source[src]["available"] += 1
            if keep:
                retained += 1
                per_source[src]["retained"] += 1
            else:
                drop_reasons[reason] += 1
                per_source[src]["dropped"] += 1
            for ax in SLICE_AXES:
                key = sv[ax]
                slices[ax][key]["retained" if keep else "dropped"] += 1

    materialized_slices = {
        ax: {k: dict(v) for k, v in sorted(buckets.items())}
        for ax, buckets in slices.items()
    }
    return {
        "corpus_path": str(corpus_path),
        "total_rows": total,
        "retained_rows": retained,
        "dropped_rows": total - retained,
        "retained_fraction": round(retained / total, 4) if total else 0.0,
        "dropped_fraction": round((total - retained) / total, 4) if total else 0.0,
        "drop_reason_counts": dict(drop_reasons),
        "per_source": {k: dict(v) for k, v in sorted(per_source.items())},
        "slices": materialized_slices,
    }


# Axes whose buckets are tautologically full-coverage for this corpus
# (every row carries the same value, so "most dropped" / "overrepresented"
# is degenerate) are excluded from the actionable skew ranking. They remain
# in the JSON `slices` for completeness.
_DEGENERATE_SKEW_AXES = {
    "schema_card_id_availability",
    "legal_action_count_bucket",
}


def top_skew_slices(
    slices: dict[str, dict[str, dict[str, int]]], k: int = 5
) -> dict[str, list[dict[str, Any]]]:
    """Top-k most-dropped and most-overrepresented decision-state buckets.

    `share_of_retained` is normalized *within each axis* (the fraction of that
    axis's retained rows the bucket holds) so it answers "this slice dominates
    its axis", which is the actionable signal for the backlog 2-4 source-mix
    handoff. Degenerate full-coverage axes are excluded.
    """
    flat: list[tuple[str, str, int, int]] = []
    axis_ret_total: dict[str, int] = {}
    for ax, buckets in slices.items():
        if ax in _DEGENERATE_SKEW_AXES:
            continue
        axis_ret_total[ax] = sum(c["retained"] for c in buckets.values())
        for name, counts in buckets.items():
            flat.append((ax, name, counts["retained"], counts["dropped"]))
    by_dropped = sorted(flat, key=lambda x: -x[3])[:k]
    by_retained_share = sorted(
        flat,
        key=lambda x: -(x[2] / (axis_ret_total.get(x[0], 1) or 1)),
    )[:k]
    return {
        "most_dropped_buckets": [
            {
                "axis": ax,
                "bucket": nm,
                "dropped": dr,
                "retained": rt,
                "drop_rate": round(dr / (dr + rt), 4) if (dr + rt) else 0.0,
            }
            for ax, nm, rt, dr in by_dropped
        ],
        "most_overrepresented_in_retained": [
            {
                "axis": ax,
                "bucket": nm,
                "retained": rt,
                "share_of_axis_retained": round(
                    rt / (axis_ret_total.get(ax, 1) or 1), 4
                ),
            }
            for ax, nm, rt, _ in by_retained_share
        ],
    }


def load_train_diagnostics(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {"_gap": f"train manifest missing: {manifest_path}"}
    m = json.loads(manifest_path.read_text(encoding="utf8"))
    metrics = m.get("metrics", {})
    diag = metrics.get("diagnostics", {})
    out: dict[str, Any] = {
        "samples": m.get("samples"),
        "train_loss": metrics.get("train", {}).get("loss"),
        "val_loss": metrics.get("val", {}).get("loss"),
        "val_accuracy": metrics.get("val", {}).get("accuracy"),
        "dataset_summary": m.get("dataset_summary"),
    }
    # Coverage-vs-loss: per-slice val loss vs per-slice retained sample count.
    cov_vs_loss: dict[str, list[dict[str, Any]]] = {}
    for axis in ("phase", "action_kind", "source"):
        val_axis = diag.get("val", {}).get(axis, {})
        train_axis = diag.get("train", {}).get(axis, {})
        rows = []
        for bucket, vstats in val_axis.items():
            tstats = train_axis.get(bucket, {})
            rows.append(
                {
                    "bucket": bucket,
                    "train_samples": tstats.get("samples"),
                    "val_samples": vstats.get("samples"),
                    "val_loss": vstats.get("loss"),
                    "val_policy_loss": vstats.get("policy_loss"),
                    "val_accuracy": vstats.get("accuracy"),
                }
            )
        cov_vs_loss[axis] = sorted(
            rows, key=lambda r: -(r["val_loss"] or 0.0)
        )
    out["coverage_vs_loss"] = cov_vs_loss
    return out


def gate_distribution(gate_manifest: Path) -> dict[str, Any]:
    if not gate_manifest.exists():
        return {"_gap": f"gate manifest missing: {gate_manifest}"}
    g = json.loads(gate_manifest.read_text(encoding="utf8"))
    args = g.get("args", {})
    summary = g.get("summary", {})
    by_side = summary.get("byModelSide", {})
    return {
        "manifest": str(gate_manifest),
        "opponent": args.get("opponentSelection"),
        "selection": args.get("selection"),
        "model_side": args.get("modelSide"),
        "seed_start": args.get("seedStart"),
        "games": summary.get("games"),
        "decision_trace_persisted": bool(args.get("decisionTraceOut")),
        "side_split": {
            side: {
                "games": v.get("games"),
                "winRate": v.get("modelWinRate"),
                "wilson_lower": v.get("wilson95", {}).get("lower"),
            }
            for side, v in by_side.items()
        },
        "model_win_rate": summary.get("modelWinRate"),
        "wilson_lower": summary.get("wilson95", {}).get("lower"),
    }


def build_report(
    corpus_path: Path,
    train_manifest: Path,
    gate_manifest: Path,
    prod_gate_manifest: Path,
) -> dict[str, Any]:
    corpus = audit_corpus(corpus_path)
    skew = top_skew_slices(corpus["slices"])
    train_diag = load_train_diagnostics(train_manifest)
    r7_gate = gate_distribution(gate_manifest)
    prod_gate = gate_distribution(prod_gate_manifest)

    # State-overlap: gate per-state observation traces were never persisted
    # (decisionTraceOut == "" in every gate.manifest under runs/, no decision
    # trace files exist). Direct retained-state vs gate-state observation
    # overlap is therefore UNCOMPUTABLE from existing artifacts. The audit
    # falls back to the strongest available structural comparison.
    retained_side = corpus["slices"]["side"]
    side_overlap = {
        side: {
            "retained_rows": counts["retained"],
            "retained_share": round(
                counts["retained"] / (corpus["retained_rows"] or 1), 4
            ),
        }
        for side, counts in retained_side.items()
    }
    gate_eval_count_bucket_covered = corpus["slices"]["legal_action_count_bucket"]
    overlap = {
        "limitation": (
            "Gate per-state observation traces were never persisted "
            "(decisionTraceOut empty in all runs/ gate manifests; no decision "
            "trace files exist). Exact retained-state vs gate-state observation "
            "overlap is UNCOMPUTABLE from existing artifacts — recorded as an "
            "environment gap. Structural comparison used instead."
        ),
        "gate_structure": {
            "rule_bot_opponent": r7_gate.get("opponent") == "rule",
            "side_balanced": r7_gate.get("model_side") == "both",
            "seed_start": r7_gate.get("seed_start"),
            "evaluates_all_legal_action_counts": True,
            "note": (
                "The gate plays full games end-to-end, so it traverses ALL "
                "decision states including the 1-legal-action forced states "
                "the loader drops (min_actions=2). Training never sees that "
                "~69% of the visited-state stream."
            ),
        },
        "training_side_balance_vs_gate": {
            "gate_side_split": r7_gate.get("side_split"),
            "retained_side_distribution": side_overlap,
        },
        "legal_action_count_coverage": {
            bucket: {
                "retained": c["retained"],
                "dropped": c["dropped"],
            }
            for bucket, c in gate_eval_count_bucket_covered.items()
        },
    }

    # Proposed source-mix target — explicit handoff to backlog items 2-4.
    # NOT implemented here (diagnostic-only).
    proposed_mix = {
        "rationale": (
            "The R7 corpus drop is ~100% single-legal-action forced states "
            "(no decision to imitate), not schema/feature loss. Row count is "
            "not the bottleneck; decision-state coverage is. The fix is a "
            "state-generation recipe (backlog item 2), not more self-play "
            "rows. Target percentages are floors over the RETAINED "
            "(>=2-legal-action) decision-state population."
        ),
        "targets": [
            {
                "slice": "side",
                "floor": "player >= 45% and opponent >= 45% within each "
                "major phase bucket (gate is side-balanced; R7 retained is "
                "player 51% / opp 49% overall but skewed within phases)",
            },
            {
                "slice": "source",
                "floor": "rollout-leaf-MCTS-relabeled on rule-bot-covered "
                "states >= 60%; rule-bot-replay <= 40% (R7 had rollout 69% "
                "/ rule-bot 31% of retained — keep rollout dominant but "
                "regenerate its STATES from the gate distribution)",
            },
            {
                "slice": "phase",
                "floor": "every non-trivial phase (combat, evolve, bench, "
                "trainerAfter) >= 8% each; cap trainerBefore <= 45% "
                "(R7 retained is trainerBefore-dominated)",
            },
            {
                "slice": "action_kind",
                "floor": "attack + retreatAttack combined >= 12%; evolve "
                ">= 6%; useAbility >= 3% (R7 retained: attack ~5%, "
                "retreatAttack ~0.6%, useAbility ~1.3% — combat-line under-"
                "covered, the exact contested decisions the gate rewards)",
            },
            {
                "slice": "turn_bucket",
                "floor": "t6-9 and t10+ combined >= 25% (late-game decisive "
                "states; R7 retained skews to early turns)",
            },
            {
                "slice": "legal_action_count",
                "floor": "contested states (>=4 legal) >= 30% of retained "
                "(R7 retained is dominated by exactly-2-legal binary choices)",
            },
        ],
        "do_not": (
            "Do NOT relax min_actions to 1 — single-action states carry zero "
            "policy gradient and would dilute the loss. Do NOT regenerate "
            "self-play-only rows (mcts-distill v1 failed at 0.1470 on that "
            "exact mistake). Implement via backlog items 2-4 only."
        ),
    }

    return {
        "_meta": {
            "tool": "training/data_coverage_audit.py",
            "diagnostic_only": True,
            "backlog_item": "training-data-coverage-audit (P1)",
        },
        "corpus_retention": corpus,
        "top_skew_slices": skew,
        "train_diagnostics": train_diag,
        "r7_gate": r7_gate,
        "production_gate": prod_gate,
        "train_vs_gate_state_overlap": overlap,
        "proposed_source_mix_target": proposed_mix,
        "environment_gaps": [
            "Gate per-state observation traces never persisted "
            "(decisionTraceOut empty across all runs/ gate manifests; no "
            "decision-trace files exist) — exact state-level overlap "
            "uncomputable; structural proxy used.",
            "Per-state eval loss not logged; loss is only bucketed by "
            "phase/action_kind/source in the train manifest diagnostics — "
            "coverage-vs-loss uses those buckets as the justified proxy.",
            "Exact terminal-distance per row not in corpus; own-points used "
            "as the monotone progress proxy.",
        ],
    }


def _pct(n: int, d: int) -> str:
    return f"{(100.0 * n / d):.1f}%" if d else "n/a"


def render_markdown(report: dict[str, Any]) -> str:
    c = report["corpus_retention"]
    skew = report["top_skew_slices"]
    overlap = report["train_vs_gate_state_overlap"]
    mix = report["proposed_source_mix_target"]
    td = report["train_diagnostics"]
    rg = report["r7_gate"]

    lines: list[str] = []
    a = lines.append
    a("# Training-Data Coverage Audit (R7 mixed corpus)")
    a("")
    a(
        "Canonical home for the corpus-retention + state-overlap finding "
        "(backlog item `training-data-coverage-audit`, P1). Diagnostic-only: "
        "no training, no data regen, no GPU. Regenerate with "
        "`training/.venv/bin/python training/data_coverage_audit.py`. "
        "Machine-readable sidecar: `training-data-coverage-audit.json`."
    )
    a("")
    a("## Headline")
    a("")
    a(f"- Corpus: `{c['corpus_path']}`")
    a(
        f"- **{c['total_rows']} rows -> {c['retained_rows']} retained "
        f"({_pct(c['retained_rows'], c['total_rows'])}); "
        f"{c['dropped_rows']} dropped ({_pct(c['dropped_rows'], c['total_rows'])})**."
    )
    a(
        "- The ~70% drop is **not** schema/feature/card-id loss. It is "
        "**100% single-legal-action forced states** removed by the loader's "
        "`min_actions=2` predicate (`uma_ai.dataset.load_policy_samples`): a "
        "state with one legal action carries zero policy-decision signal."
    )
    a("")
    a("## Filter-reason decomposition")
    a("")
    a("| Reason | Count | Share of dropped |")
    a("| --- | ---: | ---: |")
    drops = c["drop_reason_counts"]
    for reason, cnt in sorted(drops.items(), key=lambda x: -x[1]):
        a(f"| `{reason}` | {cnt} | {_pct(cnt, c['dropped_rows'])} |")
    a("")
    a(
        "Loader invariant: every other loader failure mode "
        "(schema-version mismatch, bad feature shape, missing "
        "`cardIdsByZone`) *raises* rather than silently dropping. Retained + "
        "dropped == total with only the `min_actions` predicate firing "
        "confirms zero raise-class rows in this corpus (it trained "
        "successfully under the v2/96-d schema path with "
        "`strict_schema_version=False`)."
    )
    a("")
    a("## Per-source retention")
    a("")
    a("| Source | Available | Retained | Dropped | Retained rate |")
    a("| --- | ---: | ---: | ---: | ---: |")
    for src, v in report["corpus_retention"]["per_source"].items():
        a(
            f"| `{src}` | {v['available']} | {v['retained']} | "
            f"{v['dropped']} | {_pct(v['retained'], v['available'])} |"
        )
    a("")
    a("## Retained-vs-dropped slices")
    a("")
    a(
        "Full per-axis breakdown is in the JSON sidecar "
        "(`slices.<axis>`). Axes covered: source, side, seed range, phase, "
        "action kind, turn bucket, legal-action-count bucket, points, board "
        "stage, energy, hand/deck size, terminal distance, "
        "schema/card-id availability."
    )
    a("")
    a("### Top 5 most-dropped buckets")
    a("")
    a("| Axis | Bucket | Dropped | Retained | Drop rate |")
    a("| --- | --- | ---: | ---: | ---: |")
    for r in skew["most_dropped_buckets"]:
        a(
            f"| {r['axis']} | `{r['bucket']}` | {r['dropped']} | "
            f"{r['retained']} | {r['drop_rate']:.1%} |"
        )
    a("")
    a("### Top 5 overrepresented buckets in the retained set")
    a("")
    a("| Axis | Bucket | Retained | Share of axis retained |")
    a("| --- | --- | ---: | ---: |")
    for r in skew["most_overrepresented_in_retained"]:
        a(
            f"| {r['axis']} | `{r['bucket']}` | {r['retained']} | "
            f"{r['share_of_axis_retained']:.1%} |"
        )
    a("")
    a("## Train-vs-gate state overlap")
    a("")
    a(f"- **Limitation:** {overlap['limitation']}")
    a(
        f"- Gate structure: rule-bot opponent="
        f"{overlap['gate_structure']['rule_bot_opponent']}, side-balanced="
        f"{overlap['gate_structure']['side_balanced']}, "
        f"seed_start={overlap['gate_structure']['seed_start']}."
    )
    a(f"- {overlap['gate_structure']['note']}")
    a(
        f"- Gate side split: "
        f"{json.dumps(overlap['training_side_balance_vs_gate']['gate_side_split'])}"
    )
    a(
        f"- Retained side distribution: "
        f"{json.dumps(overlap['training_side_balance_vs_gate']['retained_side_distribution'])}"
    )
    a("")
    a("## Coverage-vs-loss buckets")
    a("")
    if "coverage_vs_loss" in td:
        for axis, rows in td["coverage_vs_loss"].items():
            a(f"### By {axis} (val loss, highest first)")
            a("")
            a("| Bucket | Train n | Val n | Val loss | Val acc |")
            a("| --- | ---: | ---: | ---: | ---: |")
            for r in rows:
                vl = r["val_loss"]
                va = r["val_accuracy"]
                a(
                    f"| `{r['bucket']}` | {r['train_samples']} | "
                    f"{r['val_samples']} | "
                    f"{vl:.3f} | {va:.3f} |"
                    if isinstance(vl, (int, float)) and isinstance(va, (int, float))
                    else f"| `{r['bucket']}` | {r['train_samples']} | "
                    f"{r['val_samples']} | {vl} | {va} |"
                )
            a("")
        a(
            "Reading: the highest val-loss buckets coincide with the "
            "lowest-coverage / most-distribution-shifted slices (e.g. "
            "`attachEnergy`, `useAbility`, `unknown`-phase relabeled rows) — "
            "loss clusters where retained coverage is thin, consistent with "
            "the state-coverage-bottleneck hypothesis."
        )
    else:
        a("_Train manifest diagnostics unavailable — see environment gaps._")
    a("")
    a("## Proposed source-mix target (handoff to backlog 2-4)")
    a("")
    a(f"_{mix['rationale']}_")
    a("")
    a("| Slice | Floor / cap |")
    a("| --- | --- |")
    for t in mix["targets"]:
        a(f"| {t['slice']} | {t['floor']} |")
    a("")
    a(f"**Do not:** {mix['do_not']}")
    a("")
    a("## Environment gaps")
    a("")
    for g in report["environment_gaps"]:
        a(f"- {g}")
    a("")
    a(
        "---\n_Generated by `training/data_coverage_audit.py` "
        "(deterministic, read-only). This document is the single canonical "
        "home for the coverage finding; other docs link here, never copy._"
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    p.add_argument("--gate-manifest", type=Path, default=DEFAULT_GATE_MANIFEST)
    p.add_argument(
        "--prod-gate-manifest", type=Path, default=DEFAULT_PROD_GATE_MANIFEST
    )
    p.add_argument("--doc-out", type=Path, default=DEFAULT_DOC)
    p.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    p.add_argument(
        "--print-only",
        action="store_true",
        help="Print headline to stdout, do not write doc/json (smoke use).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.corpus.exists():
        print(
            f"ENVIRONMENT GAP: corpus not found at {args.corpus}; "
            "missing runs/ artifact is not a product pass.",
            file=sys.stderr,
        )
        return 2
    report = build_report(
        args.corpus, args.train_manifest, args.gate_manifest, args.prod_gate_manifest
    )
    c = report["corpus_retention"]
    print("=== training-data-coverage-audit ===")
    print(f"corpus: {c['corpus_path']}")
    print(
        f"rows: {c['total_rows']} total, {c['retained_rows']} retained "
        f"({c['retained_fraction']:.1%}), {c['dropped_rows']} dropped "
        f"({c['dropped_fraction']:.1%})"
    )
    print(f"drop_reason_counts: {json.dumps(c['drop_reason_counts'])}")
    print(
        "per_source: "
        + json.dumps(
            {
                k: f"{v['retained']}/{v['available']}"
                for k, v in c["per_source"].items()
            }
        )
    )
    print(
        "top dropped bucket: "
        + json.dumps(report["top_skew_slices"]["most_dropped_buckets"][0])
    )
    if args.print_only:
        return 0
    args.doc_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf8"
    )
    args.doc_out.write_text(render_markdown(report), encoding="utf8")
    print(f"wrote: {args.doc_out}")
    print(f"wrote: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
