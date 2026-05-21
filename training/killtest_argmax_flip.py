#!/usr/bin/env python3
"""Candidate-1 kill-test ladder step 2: compare 100-sim vs 400-sim relabel
argmax distributions on the rule-bot-mirror corpus (server-free, baseline
selection → identical visited states at both sim counts).

Subset proxy: rule-bot-mirror rows where the 100-sim relabel argmax kind
is in {pass, playTrainer, retreatAttack} AND 100-sim relabel max-prob >= 0.8.
This is the broadest reasonable proxy for the kill-test's
"confident-disagreement contested" subset (the canonical metrics file only
emits aggregate counts, no per-row id list).

Output:
  - prints a summary table to stdout
  - writes <out_dir>/flip-analysis.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

CONTESTED_KINDS = {"pass", "playTrainer", "retreatAttack"}
CONF_THRESHOLD = 0.8

REPO = Path(__file__).resolve().parents[1]
ORIG = REPO / "runs/R16-TD-3a-prod-corpus/rule-bot-mirror/traces.jsonl"
NEW = REPO / "runs/R16-TD-3a-prod-corpus-400sim-rulebot/rule-bot-mirror/traces.jsonl"
OUT = REPO / "runs/R16-TD-3a-prod-corpus-400sim-rulebot/flip-analysis.json"


def load_traces(path: Path) -> dict[tuple[str, str, int], dict]:
    rows: dict[tuple[str, str, int], dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (str(r["seed"]), str(r["modelSide"]), int(r["step"]))
            if key in rows:
                raise RuntimeError(f"duplicate key {key} in {path}")
            rows[key] = r
    return rows


def kl(p: list[float], q: list[float]) -> float:
    # KL(p || q) in nats with eps smoothing
    eps = 1e-12
    return sum(
        pi * (math.log(pi + eps) - math.log(qi + eps))
        for pi, qi in zip(p, q)
        if pi > 0
    )


def entropy(p: list[float]) -> float:
    eps = 1e-12
    return -sum(pi * math.log(pi + eps) for pi in p if pi > 0)


def main() -> int:
    if not ORIG.exists():
        print(f"FATAL: missing {ORIG}", file=sys.stderr)
        return 2
    if not NEW.exists():
        print(f"FATAL: missing {NEW}", file=sys.stderr)
        return 2

    orig = load_traces(ORIG)
    new = load_traces(NEW)

    only_orig = set(orig) - set(new)
    only_new = set(new) - set(orig)
    shared = set(orig) & set(new)
    recipe_diverged = bool(only_orig or only_new)
    if recipe_diverged:
        print(
            f"WARNING: rule-bot-mirror trajectories diverged — 100-sim n={len(orig)} "
            f"400-sim n={len(new)} shared={len(shared)} only-100={len(only_orig)} "
            f"only-400={len(only_new)}. Pairing is NOT clean across the corpus; "
            f"the analysis below restricts to the shared (seed, modelSide, step) "
            f"intersection.",
            file=sys.stderr,
        )

    # Verify legalActions parity on the intersection — keep only state-identical
    # rows for the argmax-flip comparison.
    valid_pairs: set[tuple[str, str, int]] = set()
    legal_mismatch = 0
    for key in shared:
        ids100 = [a["id"] for a in orig[key]["legalActions"]]
        ids400 = [a["id"] for a in new[key]["legalActions"]]
        if ids100 == ids400:
            valid_pairs.add(key)
        else:
            legal_mismatch += 1
    n_legal_mismatch = legal_mismatch
    n_state_identical = len(valid_pairs)
    if legal_mismatch:
        print(
            f"WARNING: legalActions diverged on {legal_mismatch}/{len(shared)} shared "
            f"keys — those (seed,modelSide,step) triples represent different game states "
            f"at 100 vs 400 sims. Restricting to {n_state_identical} state-identical pairs.",
            file=sys.stderr,
        )
    # Restrict orig/new to the state-identical intersection going forward.
    orig = {k: orig[k] for k in valid_pairs}
    new = {k: new[k] for k in valid_pairs}

    # Build subset.
    subset = []
    for key, r100 in orig.items():
        pt100 = r100["policyTargets"]
        legals = r100["legalActions"]
        if not pt100 or not legals:
            continue
        amax100 = max(range(len(pt100)), key=lambda i: pt100[i])
        kind = legals[amax100]["kind"]
        max_prob = pt100[amax100]
        if kind in CONTESTED_KINDS and max_prob >= CONF_THRESHOLD:
            subset.append(key)

    # Auxiliary: all confident rows (any kind) and all contested rows (any conf).
    # Lightweight context to bound interpretation.
    n_confident_any = sum(
        1
        for r in orig.values()
        if r["policyTargets"] and max(r["policyTargets"]) >= CONF_THRESHOLD
    )
    n_contested_any = sum(
        1
        for r in orig.values()
        if r["policyTargets"]
        and r["legalActions"][
            max(range(len(r["policyTargets"])), key=lambda i: r["policyTargets"][i])
        ]["kind"]
        in CONTESTED_KINDS
    )

    # Flip + aux stats on the subset.
    flips = 0
    kls = []
    ents100 = []
    ents400 = []
    conf_gained = 0
    conf_lost = 0
    conf_flat = 0
    per_kind = {k: {"n": 0, "flips": 0} for k in CONTESTED_KINDS}

    for key in subset:
        p100 = orig[key]["policyTargets"]
        p400 = new[key]["policyTargets"]
        if len(p100) != len(p400):
            raise RuntimeError(f"policyTargets length mismatch at {key}")
        a100 = max(range(len(p100)), key=lambda i: p100[i])
        a400 = max(range(len(p400)), key=lambda i: p400[i])
        kind = orig[key]["legalActions"][a100]["kind"]
        per_kind[kind]["n"] += 1
        if a100 != a400:
            flips += 1
            per_kind[kind]["flips"] += 1
        kls.append(kl(p400, p100))
        ents100.append(entropy(p100))
        ents400.append(entropy(p400))
        d = p400[a400] - p100[a100]
        if d > 0.05:
            conf_gained += 1
        elif d < -0.05:
            conf_lost += 1
        else:
            conf_flat += 1

    n = len(subset)
    flip_rate = flips / n if n else 0.0
    mean_kl = sum(kls) / n if n else 0.0
    mean_e100 = sum(ents100) / n if n else 0.0
    mean_e400 = sum(ents400) / n if n else 0.0

    # Verdict.
    if flip_rate > 0.30:
        verdict = "LIVES"
    elif flip_rate < 0.20:  # i.e. >80% sticky
        verdict = "DIES"
    else:
        verdict = "AMBIGUOUS"

    out = {
        "schemaVersion": 1,
        "subset_definition": {
            "recipe": "rule-bot-mirror",
            "contested_kinds": sorted(CONTESTED_KINDS),
            "conf_threshold_max_prob": CONF_THRESHOLD,
            "note": (
                "broadest reasonable proxy: kill-test's confident-disagreement "
                "contested subset cannot be reconstructed exactly because the "
                "canonical metrics-confidence.json emits only aggregate counts "
                "and no per-row id list. Disagreement-with-ref filter is dropped; "
                "the contested + confident filters are retained."
            ),
        },
        "recipe_invariant_check": {
            "recipe_diverged": recipe_diverged,
            "n_100_total": 4196,
            "n_400_total": 4233,
            "n_key_shared": len(shared),
            "n_only_100": len(only_orig),
            "n_only_400": len(only_new),
            "n_legal_mismatch_on_shared": n_legal_mismatch,
            "n_state_identical_pairs": n_state_identical,
            "shared_frac": len(shared) / (len(shared) + len(only_orig) + len(only_new))
            if (len(shared) + len(only_orig) + len(only_new)) > 0 else 0.0,
            "state_identical_frac_of_100": n_state_identical / 4196,
            "note": (
                "rule-bot-mirror is supposed to visit identical (seed, modelSide, step) "
                "states at 100 vs 400 sims because selection=baseline is deterministic. "
                "Empirically the trajectories DIVERGE — only the state-identical "
                "intersection (same key AND same legalActions id-set) is used for the "
                "row-by-row argmax comparison. The recipe-divergence itself is a finding "
                "for orchestrator synthesis: the relabel MCTS likely consumes the shared "
                "AsyncLocalStorage RNG via the global random() proxy, perturbing the "
                "downstream game state."
            ) if recipe_diverged else "trajectories matched as expected.",
        },
        "corpus": {
            "total_rule_bot_mirror_shared_rows": len(orig),
            "n_confident_any_kind": n_confident_any,
            "n_contested_any_conf": n_contested_any,
            "n_subset": n,
        },
        "verdict": verdict,
        "flip_rate": flip_rate,
        "flips": flips,
        "n": n,
        "mean_kl_400_to_100_nats": mean_kl,
        "mean_entropy_100_nats": mean_e100,
        "mean_entropy_400_nats": mean_e400,
        "entropy_delta_400_minus_100": mean_e400 - mean_e100,
        "confidence_buckets_on_subset": {
            "gained_gt_005": conf_gained,
            "lost_lt_005": conf_lost,
            "flat_within_005": conf_flat,
            "gained_frac": conf_gained / n if n else 0.0,
            "lost_frac": conf_lost / n if n else 0.0,
            "flat_frac": conf_flat / n if n else 0.0,
        },
        "per_contested_kind": per_kind,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(out, f, indent=2)

    # Print sharp summary.
    print("=" * 70)
    print(f"VERDICT: {verdict}")
    print(
        f"SUBSET: rule-bot-mirror, contested-kind argmax + 100-sim max-prob>={CONF_THRESHOLD}, n={n}"
    )
    print(f"FLIP RATE: {flip_rate * 100:.1f}% ({flips} of {n} argmax flips)")
    print(f"KL(400 || 100) mean: {mean_kl:.4f}")
    print(
        f"Entropy 100 / 400: {mean_e100:.4f} / {mean_e400:.4f} (delta {mean_e400 - mean_e100:+.4f})"
    )
    pct_g = (conf_gained / n * 100) if n else 0.0
    pct_l = (conf_lost / n * 100) if n else 0.0
    pct_f = (conf_flat / n * 100) if n else 0.0
    print(
        f"Confidence: gained {pct_g:.1f}% / lost {pct_l:.1f}% / flat {pct_f:.1f}%"
    )
    print("Per contested kind:")
    for k in sorted(CONTESTED_KINDS):
        v = per_kind[k]
        if v["n"]:
            r = v["flips"] / v["n"]
            print(f"  {k:>15}: n={v['n']:>4}  flips={v['flips']:>4}  flip_rate={r * 100:.1f}%")
        else:
            print(f"  {k:>15}: n=   0  (no rows in subset)")
    print(f"ARTIFACTS:")
    print(f"  400-sim corpus: {NEW.parent}/")
    print(f"  flip analysis : {OUT}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
