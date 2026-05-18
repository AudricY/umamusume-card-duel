# AI Research Backlog

Created 2026-05-11. This is the *research-and-exploration* backlog — distinct
from the infrastructure backlog under `docs/ai-performance-research-backlog.md`
which is now drained. The infra (DAgger orchestrator, F1 PPO, observability,
HP sweep, opponent pool, calibrated-value gate) is built and validated; the
remaining work is experimental science.

**Scope discipline.** This file is forward-looking only (target ≤300 lines).
Finished/failed phases and dated result blocks live in
`docs/ai-performance-research-progress.md`; closed entries here shrink to
one-line pointers. Feature-engineering / UX / eval-tooling work lives in
`docs/ai-feature-engineering-backlog.md`. See CLAUDE.md "Documentation
Discipline" for the rule.

## Current state (anchor; refreshed 2026-05-15)

As of 2026-05-15, after R12 / R13 / R14 / R15.S1–S4:

1. **Primary production claim — rollout-leaf MCTS @ W6 iter-2: Wilson lower
   0.6479** (R14.I.2, n=120, vs rule-bot). Strongest single config on record;
   clears the original R12 north star (≥0.40) by +25pp and the R14 stretch
   (≥0.60) by +5pp. Checkpoint: `runs/R13-W6-phase-d/iter-2/checkpoint.pt`.
2. **Cheap-inference fallback — value-head leaf + adaptive-ratio=1.5 @ iter-2:
   Wilson lower 0.39–0.45 across three independent seed ranges** (F-iter-2 at
   820000+ → 0.443; R14.A at 800000+ → 0.404; R14.A.footnote at 900000+ →
   0.394). 1-of-3 below the 0.40 production bar; ~0.5s/decision at 100 sims.
   Honest framing: empirical range, not a point estimate. Routing decision
   open (see `docs/ai-agent-state/escalations.md`).
3. **The F1 PPO line is closed.** Five F1 PPO phases (2/G/H/J/K) plus R15.S3
   (six reward-shape sweeps L/M/N/O'/O/P) and R15.S1/S2 all confirm: no
   combination of HPs / warm-start scale / strong-pool self-play / per-step
   reward shaping breaks the iter-2 Wilson 0.368 ± 0.001 ceiling from the
   item17 warm-start. R15.S3 axis 1 (observation-delta signals) saturated;
   axis 2 (value-head-delta tempo signal) regressed at both magnitudes.
4. **F1 raw-policy SL line CLOSED across all four axes (2026-05-15).** Every
   structural lever for lifting a *search-free* policy network has now failed
   its SL gate: labels (R7, Wilson 0.2921), objective (R8, 0.3318),
   representation (R7.b.2, 0.3045), label-shape (mcts-distill v1, 0.1470).
   Best-ever 0.3318 never approached the 0.40 gate; the cross-axis pattern is
   a structural raw-policy cap ≈0.30 against the rule-bot gate. The product
   strength does not come from a stronger policy net — it comes from
   inference-time search wrapped around the existing net (item 1). Raw-policy
   SL is not a forward line; one revisit hypothesis is parked at P4
   (`mcts-distill-followup-direction`, rule-bot-mixed-corpus variant only)
   pending explicit new evidence. Single canonical detail home for all four
   closures: `docs/ai-research/progress/r15.md` §§ R7 / R8 / R7.b.2 /
   MCTS-distill v1.
5. **Side asymmetry is real, not a determinism bug.** Engine determinism
   (R14.B AsyncLocalStorage) closed the parallel-worker non-determinism, but
   the player-vs-opponent gap (+0.11–0.30pp Wilson) replicates across four
   gates × two seed ranges at value-head-leaf inference. Disclose in any
   deployment claim.

The pre-R12 framing (DAgger 0.31 cap, three load-bearing questions Q1/Q2/Q3,
warm-start peakedness diagnosis) is historical; full archive in
`docs/ai-performance-research-progress.md` § "2026-05-14 Backlog → Progress
Migration".

## Research north star (refreshed)

Pre-R12: "move some honest measurement past Wilson 0.31." That bar is cleared
by 32pp at max strength and at-or-near at cheap inference. The F1 raw-policy
SL line is now closed across all four axes (Current state #4) — there is no
remaining "one more pivot" option there. The post-R14 north star is the
**search-wrapped production path**:

- **Ship and harden the existing strength.** The production claim is
  search-wrapped rollout-leaf MCTS @ W6 iter-2 (Wilson 0.6479). Forward work:
  finish the R14.E manual UI exercise (human-gated, last R14 acceptance step);
  pick a rollout-leaf vs value-head deployment policy against the empirical
  0.39–0.45 cheap-inference envelope; harden MCTS robustness/determinism so
  the deployed search behaves predictably (see "Production-path forward
  work" below).
- **Raw-policy SL is not a forward line.** R7 / R8 / R7.b.2 / mcts-distill
  all closed FAIL; one parked revisit hypothesis at P4 only, not the next
  pick. Do not re-open without explicit new evidence.

## Production-path forward work (post-F1-closure, refreshed 2026-05-15)

With the raw-policy SL line closed, all forward research leverage is on the
working search-wrapped path. Ranked by leverage:

1. **R14.E UI integration — RESOLVED 2026-05-15 (user-verified).** User
   manually exercised it in-browser; webapp opponent defaults to the
   production rollout-leaf config (Wilson 0.6479). Qualitative acceptance
   (no numeric latency capture — see honest scope in r14-sprint-plan § E).
   The 96-dim/110-dim deployment skew is RESOLVED by an explicit serving
   pin (`serve_onnx --feature-schema auto`→v2 for the 96-d production
   model; commit `f658ad9`). Detail: `docs/ai-research/progress/r15.md`
   §§ "R14.E closure", "96-dim serving pin"; `docs/r14-sprint-plan.md` § E.
   Queue: `r14-e-manual-ui-exercise` → done.
2. **MCTS search determinism + side-asymmetry shape — RESOLVED 2026-05-15.**
   (a) single-worker CRN determinism PASS (0.0 divergence); (b) rollout-leaf
   side gap measured: opponent 0.7833 vs player 0.6833 (+10.0pp directional,
   CIs overlap, player floor 0.5577). 0.6479 is a sound *aggregate
   side-balanced reproducible* claim, not a per-side guarantee. Detail:
   `docs/ai-research/progress/r15.md` § "MCTS production-claim audit".
   Queue: `mcts-determinism-and-side-asymmetry-audit` → done.
3. **Deployment-policy decision (rollout-leaf vs value-head) — DECIDED
   2026-05-15.** Production config = **rollout-leaf @ W6 iter-2** (Wilson
   lower 0.6479 aggregate; ~20pp over value-head's best-case 0.4524 and
   well over the honest-envelope-low 0.394). Tradeoff: rollout-leaf is
   ~2.5× slower (value-head + ratio=1.5 cuts 61% wall-clock, ~0.5s/dec);
   acceptable for a turn-based game with no hard move-clock. Value-head +
   adaptive-ratio=1.5 is the documented latency-SLA fallback only, shipped
   with the "*at* the 0.40 bar, 1-of-3 seeds below" caveat — not an
   equivalent. Asymmetry stance: **disclose-as-aggregate with the per-side
   caveat** (gap directional not separated; config choice does not cleanly
   shrink it — rollout +10pp, value-head +6 to +14pp across seeds, both
   player-weaker). Forward (non-blocking) bet: a side-conditioned sim
   budget, gated on a larger-n side-split that separates the CIs — not a
   deployment blocker, no human research-stance call needed. Detail +
   honest claim string: `docs/ai-research/progress/r15.md` § "Deployment-
   policy decision — rollout-leaf vs value-head-leaf". Queue:
   `deployment-policy-rollout-vs-valuehead` → done.

**Program state (2026-05-15): the search-wrapped production path is fully
consolidated** — F1 raw-policy SL closed across all 4 axes; rollout-leaf @
W6 iter-2 decided, characterized (determinism + side-asymmetry), UI-
integrated, user-verified, and deployment-pinned (96-d). No autonomous
forward research job remains. The next-arc frontier ((a) is now scoped +
loop-actionable; (b)–(d) still require a user steer):
- **(a) A 110-d production-grade model** would unlock the `--feature-schema
  v3` promote path. The *SL* route is closed (R7.b.2 110-d card-embed SL
  Wilson 0.3045; mcts-distill v1 single-pass 0.1470 — both single SL passes,
  raw-policy). **Now SCOPED with a concrete non-SL approach (user-
  commissioned):** reproduce the W6 rollout-leaf MCTS self-play + distill
  *loop* at 110-d/v3.0, evaluated **search-wrapped** — the pipeline that
  produced the pinned 0.6479 96-d model; pipeline audited 110-d-ready.
  Exit gate = search-wrapped Wilson lower ≥ 0.6479 → promote v3. Scoping:
  `docs/ai-research/scoping/r110-w6-reproduction.md`. Queue:
  `r110-w6-reproduction` (P2, ready).
- **(b) Side-conditioned sim budget** — the logged non-blocking bet; gated
  on a larger-n rollout-leaf side-split that separates the player/opponent
  CIs (currently overlap at n=60/side). Not a deployment blocker.
- **(c) RL/PPO from the strong search-wrapped checkpoint, or scaling** —
  the natural "next big bet" beyond SL; a strategic call for the user.
- **(d) Training-data/state-coverage program** — user-steered 2026-05-15.
  Treat the current data evidence as a *coverage bottleneck*, not a generic
  "more rows" ask: R7 retained only 3789/12387 mixed rows, mcts-distill v1
  failed at Wilson 0.1470 on self-play-only state coverage, and search-wrapped
  MCTS remains strong. Work below is diagnostic-first and must not silently
  re-open the closed raw-policy SL line.

## Training-data/state-coverage backlog (user-steered 2026-05-15)

1. **P1 — Corpus retention + state-overlap audit
   (`training-data-coverage-audit`).** No training. Build a report that
   compares raw exported rows, loader-retained samples, and gate/eval traces
   by source, side, seed range, phase, action kind, turn bucket, legal-action
   count, points, board stage, energy, hand/deck size, terminal distance, and
   schema/card-id availability. Explain why ~70% of the R7 mixed corpus is
   dropped, whether the retained set covers failing gate states, and where
   eval losses cluster in low-coverage buckets. Acceptance: one reproducible
   command + JSON/markdown report, with filter-reason counts, top 5 missing or
   overrepresented slices, coverage-vs-error buckets, and an explicit proposed
   source-mix target.
2. **P2 — Rule-bot-covered state corpus recipe.** Generate states from the
   deployment-relevant distribution (raw/search policy vs rule-bot, rule-bot
   mirror, side-balanced), then relabel those exact states with rollout-leaf
   MCTS or the strongest feasible oracle. This is the only plausible
   `mcts-distill` revisit shape: it changes the state distribution, not the
   label shape alone. Acceptance: retained-row rate >=80%, slice floors for
   rare phases/actions, and a closed-loop gate reported as diagnostic only
   unless it beats the best raw-policy SL baseline by a pre-registered margin.
3. **P2 — Preference pairs on rule-bot-covered states.** If the coverage audit
   shows enough contested states, create DPO/BT pairs from the same state set
   using MCTS top action vs legal alternatives / rule-bot action, including
   hard-negative top-K alternatives when candidate reward vectors exist. This
   avoids another self-play-only soft-label corpus and extracts more signal
   from contested states. Acceptance: pair manifest with source-state coverage,
   kept-pair count, margin/variance filters, held-out pair/ranking accuracy,
   and an n=1000 side-balanced gate only after pair quality clears a
   pre-registered floor.
4. **P2 — Side-conditioned retained-data balancing.** If the audit confirms
   the measured player-side weakness is also a data issue, oversample or
   separately label player-side decision states, especially setup / first-
   mover branches. Acceptance: retained rows are side-balanced within each
   major phase/action bucket, the gate reports side split, and player-side
   Wilson lower improves without aggregate regression.
5. **P3 — Rule-bot mistake catalog + forced-state suite.** Replace the vague
   "exploit rule-bot weaknesses" idea with ~50 hand-audited states grouped by
   tactical failure mode. Use them as forced-start probes for data generation
   and for explaining why MCTS beats the rule bot. Acceptance: fixtures,
   expected oracle/rule-bot divergence, and coverage mapping back to corpus
   slices.

**Guardrail.** Do not schedule self-play-only data regen, capacity bumps, or
representation work from this section until the coverage audit shows which
state slices are missing. Raw-policy SL remains closed unless new evidence is
strong enough to explicitly re-open it.

The Tier-1/Tier-2/Tier-3 framework below is preserved as historical context
for early-2026-05 reasoning; most entries are resolved and reduced to pointers.

---

## Tier 1 — discriminating experiments (HISTORICAL)

All four resolved by 2026-05-11. One-line pointers:
- **R1** oracle ceiling — DONE / Q1 REFUTED (not a teacher-strength cap).
  **R2** multi-temp gate — REPURPOSED (deployment tuning, R15.S4). **R3**
  entropy-BC — DONE / DIAGNOSIS WRONG (mechanism did not exist). **R4**
  value-head ablation — DONE / Q3 PARTIAL (calibration alone does not move
  the gate). Detail: progress doc § "Tier 1 results (2026-05-11)".

## Tier 2 — directional experiments (HISTORICAL)

- **R5.** PPO self-play via PFSP pool — DONE / FAIL (gate WR 0.3109; W6
  re-attempt R15.S2 also failed). Pointer: progress doc § "Tier 1 results
  (2026-05-11)" R5 block and § "Phase J".
- **R6.** Larger model capacity — DONE / FAIL (best imitator, weakest player;
  cap is imitation-target-quality, not capacity). Pointer: progress doc.
- **R7 / R8 / R7.b / mcts-distill — F1 raw-policy SL line CLOSED, all four
  axes FAIL** (best-ever Wilson 0.3318, none cleared 0.40; see Current state
  #4). Pointer: `docs/ai-research/progress/r15.md` §§ R7 / R8 / R7.b.2 /
  MCTS-distill v1; scoping `docs/ai-research/scoping/{r7-multi-teacher-
  warmstart,r8-dpo,r7b-feature-representation,mcts-distill}.md`. Forward: one
  parked P4 revisit hypothesis only (Production-path section).

## Tier 3 — structural changes (HISTORICAL)

- **R9 / R10.** Q-learning head / larger DAgger sweep — DELETED 2026-05-11
  (obsoleted by R12 GO; R15.S1 confirmed compute-scaling at fixed capacity
  does not break the cap).
- **R11.** Auxiliary self-supervised trunk objectives — PARKED. No evidence
  the cap is representation-quality now that R12/R14.I.2 land at Wilson
  0.6479; only relevant if R7/R8 also close negative.
- **R12.** MCTS-augmented self-play (mini-AlphaZero) — DONE / GO; the
  production line (rollout-leaf → Wilson 0.6479 at R14.I.2 iter-2). Pointer:
  progress doc §§ "Rollout-leaf spike: GO", "R13.W6 result", "R14 progress
  checkpoint".

## R15 sprint — F1 PPO post-mortem follow-ons (2026-05-14)

The F1 PPO post-mortem (`docs/ai-performance-research-progress.md` § "F1 Phase
summary") ranked four follow-on moves after phases 2/G/H closed at Wilson
0.31. All four are now executed or queued:

- **R15.S1.** Better SL warm-start (compute-scaled DAgger) — DONE / FAIL
  (iter-2 Wilson 0.3269, in falsification band). Pointer: progress doc §
  "R15.S1–S4 result blocks" and § "Phase K".
- **R15.S2.** PFSP self-play PPO with the strong W6 pool — DONE / FAIL
  (iter-2 Wilson 0.2188 regressed). Pointer: progress doc § "R15.S1–S4
  result blocks" and § "Phase J".
- **R15.S3.** Richer per-step reward shaping — DONE / EXHAUSTED (both
  axes closed). Pointer: progress doc § "R15.S1–S4 result blocks" and §§
  Phase L / M / N / O' / O / P.
- **R15.S4.** Sampling-temperature gate (forward, diagnostic) — see below.

### R15.S4 — Sampling-temperature gate (F1 next-move #4) (forward)

- **Motivation:** F1 PPO post-mortem ranked this fourth (diagnostic, not
  solution). All five F1 phases were scored on greedy argmax behavior. If PPO
  is shaping the policy distribution but not flipping argmax, an alternative
  gate that samples at e.g. T=0.5 might reveal latent improvement.
  Repurposes the R2 multi-temperature gate matrix idea against the F1
  phase-H promoted checkpoint rather than the original DAgger artifacts.
- **Next action:** Confirm the `sampling=stochastic&temperature=T` body is
  wired through `serve_onnx` and `evaluateModelVsHeuristic`. Run a
  5-temperature × 2-checkpoint matrix (T ∈ {0, 0.3, 0.5, 1.0, 1.5},
  F1 phase-H iter-2 + DAgger iter-2 warm-start) at n=100.
- **Cost:** ~30 min plumbing check + 5 min × 10 cells = ~1.5h compute.
- **Exit / gate:** Either a non-greedy temperature lifts F1 phase-H above the
  DAgger warm-start by ≥3pp Wilson lower (proves PPO was making *some* signal
  we couldn't see), or all temperatures match → confirms PPO produced nothing
  the gate could measure, closes this hypothesis.
- **Status:** Diagnostic only; informational regardless of outcome. Lower
  priority than R7/R8 if any pivot is being commissioned, but cheap enough to
  queue ad-hoc when the next slot opens.

## Open / wild

- **Side imbalance / determinism.** Measured and closed 2026-05-15; see
  `docs/ai-research/progress/r15.md` § "MCTS production-claim audit".
- **Rule-bot mistake catalog.** Refined into the training-data/state-coverage
  backlog above.
- **Side-asymmetry confirmation gate (R14.A footnote).** Closed 2026-05-14;
  cheap-inference production claim contradicted at independent seeds.
- **Compute-architecture: `--workers` lever vs batched leaf-inference
  (deferred / conditional — do NOT action).** Origin 2026-05-18 GPU-idle
  inquiry; read-only evidence from the live R110-W6-repro run. (1) Self-play
  /gate ≈18.7 min/iter, ≈1–1.5 h/arc fully unattended → NOT the research-
  cadence bottleneck (human judgment turnaround between arcs is). (2) In the
  rollout-leaf production config the CPU hot path is the TS engine running
  rule-bot rollouts, NOT NN inference (eval workers ~92% CPU vs serve_onnx
  sidecar ~19.5%; only the PUCT prior is a model call) — batched inference
  barely helps this config. (3) Cheap lever IF cadence ever binds: raise
  `r12_orchestrator.py --workers` 4→16–24 (nproc=32, load ~7) — pure config,
  ~4–6× faster compute phases, no code; validate via one short worker sweep
  before making default. (4) Batched leaf-inference (virtual-loss leaf
  collector in `backend/src/sim/mcts.ts` + additive `/predict-batch` in
  `training/serve_onnx.py`; M-tier, Python already vectorized) is a real but
  CONDITIONAL bet — only justified when ALL hold: (a) self-play cadence is
  the binding research constraint, (b) a CUDA serve path exists, (c) the
  active config is leaf-eval-bound (value-head-leaf), not rollout-bound.
  Risk: virtual loss perturbs search → must re-clear the determinism-
  sensitive eval gates (R14.B / R15 single-worker replay). Premise falsified
  for the production config; informational, not actionable.
