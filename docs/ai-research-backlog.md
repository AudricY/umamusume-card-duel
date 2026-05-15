# AI Research Backlog

Created 2026-05-11. This is the *research-and-exploration* backlog — distinct
from the infrastructure backlog under `docs/ai-performance-research-backlog.md`
which is now drained. The infra (DAgger orchestrator, F1 PPO, observability,
HP sweep, opponent pool, calibrated-value gate) is built and validated; the
remaining work is experimental science.

**Scope discipline.** This file is forward-looking only (target ≤300 lines).
Finished/failed phases and dated result blocks live in
`docs/ai-performance-research-progress.md`; closed entries here shrink to
one-line pointers. See CLAUDE.md "Documentation Discipline" for the rule.

## Current state (anchor; refreshed 2026-05-14)

As of 2026-05-14, after R12 / R13 / R14 / R15.S1–S4:

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

1. **R14.E manual UI exercise — P1, human-gated, last R14 acceptance step.**
   20 full browser games at value-head leaf + adaptive-ratio=1.5; exit:
   fallback <5%, median decisionMs <3s. Cannot be done autonomously (needs
   a human at the browser). Detail: R14 follow-ups below + `docs/r14-sprint-
   plan.md` § E. Queue: `r14-e-manual-ui-exercise` P1.
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

The Tier-1/Tier-2/Tier-3 framework below is preserved as historical context
for early-2026-05 reasoning; most entries are resolved and reduced to pointers.

---

## Tier 1 — discriminating experiments (HISTORICAL)

All four resolved by 2026-05-11; full result blocks in progress doc § "Tier 1
results (2026-05-11)".

- **R1.** Perfect-oracle WR ceiling — **DONE / Q1 REFUTED.** rollout-CRN×3
  itself hits WR 58.5% (Wilson [0.516, 0.651]) vs rule-bot. The cap is not a
  teacher-strength cap.
- **R2.** Multi-temperature gate matrix — **REPURPOSED.** Not a research
  diagnostic; deployment-tuning task only (R15.S4 reuses the harness).
- **R3.** Entropy-regularized BC warm-start — **DONE / DIAGNOSIS WRONG.**
  β=0.05 gave entropy 0.527 nats, gate WR 38.5%; later direct-measurement
  showed warm-start entropy was already 0.532. The "peakedness mechanism"
  R3 attacked did not exist.
- **R4.** Value-head ablation — **DONE / Q3 PARTIAL.** 50ep + value_weight=1.0
  gave value_brier 0.084 (calibrate_value PASS) but gate WR dropped to 34.5%.
  Calibration alone does not move the gate.

## Tier 2 — directional experiments (HISTORICAL)

- **R5.** PPO self-play via PFSP pool — **DONE / FAIL.** 3 iters × 800 games
  against item17 pool; gate WR locked at 0.3109 same as phase H. Re-attempted
  later as R15.S2 against the stronger W6 pool; that re-attempt also failed.
  See progress doc § "Tier 1 results (2026-05-11)" R5 block and § "Phase J".
- **R6.** Larger model capacity — **DONE / FAIL.** hidden=128/depth=3,
  50ep, value_weight=1.0: best imitator (83% argmax-match) AND weakest player
  (gate WR 33.0%). The cap is imitation-target-quality, not capacity.
- **R7 / R8 / R7.b / mcts-distill — F1 raw-policy SL line CLOSED, all four
  axes FAIL.** One pointer for the whole closed line (see Current state #4):
  R7 multi-teacher BC labels (Wilson 0.2921), R8 DPO objective (0.3318),
  R7.b.2 card-embedding representation (0.3045), mcts-distill v1 soft-visit
  label-shape (0.1470, 2026-05-15). None cleared the 0.40 gate; best-ever
  0.3318. R7.b.3/4/5 (attention/history/aux-heads) were conditional on a
  R7.b.2 lift and did not auto-promote. Canonical detail:
  `docs/ai-research/progress/r15.md` §§ R7 / R8 / R7.b.2 / MCTS-distill v1.
  Scoping docs: `docs/ai-research/scoping/{r7-multi-teacher-warmstart,
  r8-dpo,r7b-feature-representation,mcts-distill}.md`. Forward status: not a
  line; one parked P4 revisit hypothesis only (Production-path section).

## Tier 3 — structural changes (HISTORICAL)

- **R9.** Q-learning head as the policy — **DELETED 2026-05-11.** Obsoleted
  by R12 GO (label-quality fixes inherit the value-head noise floor).
- **R10.** Larger DAgger sweep at production scale — **DELETED 2026-05-11.**
  Same reason. R15.S1 later confirmed compute-scaling at fixed capacity does
  not break the cap.
- **R11.** Auxiliary self-supervised objectives in the trunk — **PARKED.** No
  evidence the cap is a representation-quality problem now that R12 GO and
  R14.I.2 land at Wilson 0.6479; would only be relevant if R7/R8 also close
  negative and we need a structurally different SL pipeline.
- **R12.** MCTS-augmented self-play (mini-AlphaZero) — **DONE / GO.** Rollout-
  leaf spike cleared the 0.40 bar by +15.6pp on 2026-05-11; R13 then compounded
  to Wilson 0.570 at iter-1; R14.I.2 extended to Wilson 0.6479 at iter-2. R12
  is the production line. See progress doc §§ "Rollout-leaf spike: GO",
  "R13.W6 result", "R14 progress checkpoint".

## R15 sprint — F1 PPO post-mortem follow-ons (2026-05-14)

The F1 PPO post-mortem (`docs/ai-performance-research-progress.md` § "F1 Phase
summary") ranked four follow-on moves after phases 2/G/H closed at Wilson 0.31.
All four are now executed or queued:

- **R15.S1.** Better SL warm-start (compute-scaled DAgger) — **DONE / FAIL.**
  iter-2 Wilson 0.3269 inside the pre-registered falsification band; plateau-
  then-overfit signature fired in train-time val_accuracy. ~12 min compute.
  See progress doc § "R15.S1–S4 result blocks" and § "Phase K".
- **R15.S2.** PFSP self-play PPO with the strong W6 pool — **DONE / FAIL.**
  iter-2 Wilson 0.2188 (regressed) after iter-1 self-promotion co-adaptation.
  Largest single PPO step on record (+14.5pp at iter-1) but did not survive.
  See progress doc § "R15.S1–S4 result blocks" and § "Phase J".
- **R15.S3.** Richer per-step reward shaping — **DONE / EXHAUSTED (both
  axes closed).** Six phases (L/M/N/O'/O/P) covered axis 1 (observation-delta
  signals: magnitude, schedule, signal-mix; saturated at iter-2 Wilson
  0.368 ± 0.001) and axis 2 (value-head-delta tempo signal at coef 0.05 and
  0.01; both regressed). ~36 min total compute. Surviving F1-line moves are
  R7 / R8 above. See progress doc § "R15.S1–S4 result blocks" and §§ Phase
  L / M / N / O' / O / P.
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

## R14 outstanding follow-ups

- **R14.E.followup — Manual 20-game UI exercise.**
  - **Motivation:** R14.E's plumbing is verified (headless `/ai/decide` smoke
    PASS, extrapolated decisionMs ~0.5s at ratio=1.5, 100 sims) but the
    in-browser fallback-rate and median latency numbers can only come from
    real play. This is the only R14 acceptance criterion still open.
  - **Next action:** Start backend + frontend dev servers, toggle
    MainMenuScreen AI=MCTS, play 20 full games end-to-end at value-head leaf
    + adaptive-ratio=1.5, capture devtools console log for `decisionMs` per
    turn and fallback events.
  - **Cost:** ~30-45 min of actual play.
  - **Exit / gate:** fallback rate < 5%, median decisionMs < 3s. Append
    result to `docs/r14-sprint-plan.md` Progress section as the E closure.

## Open / wild

- **Side-imbalance verification.** *MEASURED 2026-05-15 at rollout-leaf:
  player WR 0.6833 [0.5577,0.7869] vs opponent 0.7833 [0.6638,0.8688],
  +10.0pp directional gap (same direction as value-head-leaf R14.A
  footnote), CIs overlap at n=60/side; player-side Wilson floor 0.5577 <
  0.6479 aggregate. See `docs/ai-research/progress/r15.md` § "MCTS
  production-claim audit".*
- **Simulator determinism audit.** *CLOSED 2026-05-15: single-worker CRN
  self-consistency PASS, 0.0 full-outcome divergence over 16 (seed,side)
  keys × 2–3 replays at the production rollout-leaf config (run vs the
  production-era SHA bc6db85, 96-dim schema). R14.B parallel-worker case
  already closed (R-WILD #34). See `docs/ai-research/progress/r15.md` §
  "MCTS production-claim audit".*
- **Rule-bot mistake catalog.** The 80% aspirational target requires
  exploiting rule-bot weaknesses; we don't have a catalog of those
  weaknesses. Hand-construct ~50 states + a careful audit.
- **Side-asymmetry confirmation gate (R14.A footnote).** *CLOSED 2026-05-14
  — gate1 FAIL @ Wilson 0.394; the R14 cheap-inference production claim is
  contradicted at independent seeds. See progress doc § "R14.A.footnote"
  and `docs/ai-agent-state/escalations.md` for the open research-stance
  decision the human owns.*
