# Fork B — Imitation-Target / Label-Quality (MCTS-distill search-recipe forward axis)

- **Date:** 2026-05-18
- **Status:** SCOPING — actionable; gated behind a cheap label-quality probe before any loop compute. Tracked in queue under `w6-loop-anti-degradation` (Ceiling Path A) and related label-quality forward lines.
- **Owns:** the loop-recipe forward axis as a *label-quality* problem. The
  `w6-loop-anti-degradation` queue item / `r110.md` §4c own one mechanism
  (anti-degradation regularizer dose). This doc is the umbrella that frames
  *all* label-quality levers as one test matrix and keeps capacity/volume
  formally fixed so the experiment isolates label quality.
- **Sister fork (do not touch from here):** Fork A = raw-policy SL
  representation/feature work (`scoping/r16-*-backlog-refinement.md`,
  R16 vertical). Different file, different axis.

## 1. Question (Q)

> At fixed model capacity and fixed raw self-play volume, does raising the
> *quality of the imitation target at decision-critical states* move the
> search-wrapped strength metric past the current ~0.65 band that caps the
> W6 `mcts-distill` loop?

**Falsifiable proposition.** Holding capacity and per-iter raw self-play
volume at the W6 reference config (`scoping/r110-w6-reproduction.md` §4), at
least one label-quality intervention from §4 produces a best-promoted
checkpoint whose §5 exit-gate Wilson lower exceeds the current band by a
margin larger than the gate Wilson half-width. Null result = label quality
is not the binding lever inside the loop and the structural cap is
elsewhere.

## 2. Why this is the sanctioned line (evidence by pointer, not restated)

- Capacity is closed: R6 2x-capacity overfit and played worse —
  `docs/ai-research-backlog.md:115` and
  `docs/ai-performance-research-progress.md:1645-1651`.
- The named structural cap is imitation-target / label quality: SL/distill
  onto a ~58%-WR teacher whose disagreement rows are noisy at
  decision-critical states —
  `docs/ai-performance-research-progress.md:1681-1683`, `:1651`.
- What broke the ceiling was *search*, not the net: rollout-leaf MCTS;
  the trained value head is unused at leaf in that mode —
  `docs/ai-research/progress/r110.md` (§3 cause 3 / `r110.md:88`).
- The loop-recipe axis is the only sanctioned forward line; representation
  and capacity are explicitly forbidden to reopen —
  `docs/ai-research-backlog.md:141-142`, `r110.md` §4.
- The recipe rot is monotone prior drift from iter-0 (not an iter-2
  representation inflection); the iter-2 *gate* peak is search+rollout
  masking an already-degrading prior — `r110.md` §4a. So "label quality"
  here concretely means: the soft-visit distillation target degrades each
  iter, and the loop has no replay/anti-drift to keep it crisp.

This doc does NOT re-derive any of the above. It converts the cap into a
test matrix.

## 3. Objective

Raise effective teacher/target label quality at decision-critical states,
measured by the **repo-standard search-wrapped strength metric only** —
the §5 exit gate's aggregate Wilson lower (n>=120, side-balanced, rule-bot
opponent, rollout-leaf MCTS both sides, 100 sims, K=3, 200 steps,
seed-start 9000). No new metric is invented. Comparators are the existing
A/B anchors: R110-W6-repro best-promoted iter-2 **0.6042** and the R111
recipe-fix frozen plateau **0.5527** (`r110.md` §1, §4c). "Past the ~0.65
band" references the rollout-leaf strength band in `r110.md`; the formal
bar is `scoping/r110-w6-reproduction.md` §6 (PASS >= 0.6479).

## 4. Candidate mechanisms — TEST MATRIX (options, not a chosen solution)

Each is a *label-quality* lever; capacity and raw volume held fixed (§5).
Evaluate as independent arms; do not pre-bless one.

| # | Lever | Concrete knob (file:line ground truth) | Hypothesised effect on label quality |
|---|---|---|---|
| M1 | Deeper / more-rollout MCTS at **label time only** | self-play `mctsSimulations` / `mctsRolloutSteps` / CRN `K` raised for the *self-play label pass only*, gate/search-wrap config held at the W6 reference (`scoping/r110-w6-reproduction.md` §4) so the metric stays comparable | sharper, lower-variance visit targets at contested states |
| M2 | Higher-quality leaf evaluation at label time | leaf eval upgrade in the self-play pass (still rollout-class; `backend/src/sim/mcts.ts:552-555` is where leaf value is taken) | less noisy backups feeding the visit distribution |
| M3 | Distill from **visit-count targets vs raw teacher argmax** | `train_bc.py --data-mode mcts-distill` soft-visit branch vs an argmax/temperature variant of the same rows; A/B the target *shape*, not the data | soft visits carry uncertainty; argmax inherits the teacher's noisy 1-hot at disagreement rows (`ai-performance-research-progress.md:1683`) |
| M4 | Filter / down-weight high-teacher-disagreement noisy rows | row weighting keyed on prior↔rule-bot top-1 disagreement (the §4a discriminator already computes this signal) before the distill step | removes the exact decision-critical noisy rows the cap statement names |
| M5 | CRN / variance reduction on labels | extend CRN beyond K=3 on the *label* pass and/or paired-seed antithetic sampling on self-play | lower target variance at fixed expected value, fixed volume |
| M6 | Anti-drift on the target (cross-ref, not re-scoped here) | replay-buffer + fixed-KL-anchor dose, owned by `w6-loop-anti-degradation` / `r110.md` §4c | keeps the cumulative target from softening across iters |

M6 is included for completeness and to prevent double-scoping: its
dose-tuning is **already owned** by the `w6-loop-anti-degradation` queue
item and `r110.md` §4c. This doc does not re-plan M6; it places it in the
same matrix so a future sweep compares label-shape (M3/M4) and
label-variance (M1/M2/M5) arms against the anti-drift arm on one axis.

## 5. Cheap-experiment design (capacity AND volume held fixed)

**Both confounding axes are pinned, by construction:**

- **Capacity fixed:** identical model config to W6 (`--hidden-dim 64
  --depth 2`, dropout 0.05), held across all arms and iters
  (`scoping/r110-w6-reproduction.md` §4). No capacity arm exists in the
  matrix — that is a §6 non-goal.
- **Raw volume fixed:** 60 self-play games/iter, same iteration count and
  promotion rule as W6 (`scoping/r110-w6-reproduction.md` §4). M1/M5 raise
  *search effort per state* and M3/M4 change *target shape/weighting* —
  none increases the raw row count. M5's antithetic variant must keep the
  effective sample budget constant (paired seeds, not extra games).

**Step 0 — READ-ONLY label-quality probe (do FIRST; no loop compute).**
Reuse the §4a discriminator harness (`r110.md` §4a; corpus
`runs/R7-multi-teacher-warmstart/iter-000/mixed-v3.jsonl`). On the existing
on-disk `iter-{0..4}/policy.onnx` from `runs/R13-W6-phase-d` and
`runs/R110-W6-repro`, for each candidate lever compute the *offline*
proxy it most directly moves: M3 → KL(soft-visit ‖ argmax target) and
fraction of rows where they disagree; M4 → contested-row count after the
disagreement filter; M1/M5 → target-variance estimate at fixed seed
budget. **Gate:** an arm only earns loop compute if its offline proxy
predicts a non-trivial change in target quality at the rows the cap names.
This keeps the expensive loop run gated behind a near-zero-cost read.

**Step 1 — single-arm gated loop A/B (only for probe-passing arms).**
For each passing arm, run the W6 loop unchanged except that one lever,
`--workers 24` (trajectory-neutral, `r110.md` §4b). Compare best-promoted
§5-gate Wilson lower against the 0.6042 / 0.5527 anchors. **This is a real
compute job (~1-2 h wall/arm at the post-speedup rate) and is
user-gated — do NOT auto-launch.** Cross-ref the throughput plan:
`scoping/r12-selfplay-gate-throughput.md`.

**Success criterion.** At fixed capacity and fixed raw volume, at least
one arm's best-promoted §5 Wilson lower clears the prior anchor by more
than the gate Wilson half-width *and* reaches the
`scoping/r110-w6-reproduction.md` §6 PASS bar (>= 0.6479). Anything in the
0.60-0.6479 MARGINAL band is "label quality helps but is not sufficient
alone" — record, do not promote, do not pivot to a closed axis.

## 6. Non-goals (hard)

- **No capacity tuning.** `--hidden-dim`/`--depth`/dropout are frozen at
  the W6 reference. Closed: `ai-research-backlog.md:115`,
  `r110.md` §4 / `:141-142`.
- **No generic volume increase.** Self-play games/iter and iteration count
  are frozen. M1/M5 buy search/variance quality per state, not more rows.
- **No reopening raw-policy SL representation work.** That is Fork A /
  R16-representation (`scoping/r16-model-feature-backlog-refinement.md`,
  `scoping/r16-training-data-backlog-refinement.md`) — a different file and
  a different agent's scope. Do not edit those from this line.
- **No new metric.** Strength is the §5 search-wrapped Wilson-lower gate
  only; offline proxies in Step 0 are *gating filters*, never the verdict.
- **No re-planning of the M6 anti-drift dose.** Owned by
  `w6-loop-anti-degradation` / `r110.md` §4c; referenced here only to keep
  the matrix non-overlapping.

## 7. Cross-references

- Verdict + recipe mechanism + discriminator: `docs/ai-research/progress/r110.md`.
- Loop/gate recipe + exit-gate definition: `docs/ai-research/scoping/r110-w6-reproduction.md`.
- Cap statements (evidence, not restated): `docs/ai-performance-research-progress.md:1651`, `:1681-1683`.
- Backlog frontier + guardrail: `docs/ai-research-backlog.md` (anchor; M6 dose item 0).
- Throughput / cost of the gated loop run: `docs/ai-research/scoping/r12-selfplay-gate-throughput.md`.
- Live state: `docs/ai-agent-state/queue.json` (`w6-loop-anti-degradation`).
