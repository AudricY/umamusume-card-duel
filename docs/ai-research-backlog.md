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
4. **Surviving F1-line candidates (human-rank, not autonomous-launch):**
   R7 (multi-teacher SL warm-start rebuild) and R8 (DPO objective replacement).
   Both require a directional decision before launch; both are >½-day pivots.
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
by 32pp at max strength and at-or-near at cheap inference. The post-R14 north
star is now a directional choice, not a measurement target:

- **Ship the existing strength.** Finish the R14.E manual UI exercise; route
  cheap-inference traffic by the empirical 0.39–0.45 envelope; pick a
  rollout-leaf vs value-head deployment policy.
- **Or commit to one more F1-line pivot.** R7 (multi-teacher SL rebuild) or
  R8 (DPO) — both require human rank before launch. R15.S3 closeout
  filed an escalation; no autonomous start.

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
- **R7.** Multi-teacher BC blend — **DEFERRED.** Marked obsolete on 2026-05-11
  post-R12 (label-quality fixes inherit the same noise floor), then re-listed
  as a surviving F1-line candidate after R15.S3 closeout. Still unfunded; no
  autonomous launch. Forward brief preserved below.
- **R8.** Direct preference optimization (DPO) — **DEFERRED.** Same trajectory
  as R7: obsoleted post-R12, re-listed post-R15.S3. Forward brief preserved
  below.

### R7. Multi-teacher BC blend (forward, human-rank, escalation open)

- **Q:** Does mixing rollout + planner + search teacher labels per state
  produce a warm-start with broader competence and a different gradient
  landscape than the rollout-CRN-only fit that R3/R4/R6/R15.S1 saturated?
- **Hypothesis:** Teacher diversity breaks the rollout-CRN-noise inheritance
  pattern that R6 made dispositive. PPO from a multi-teacher warm-start may
  see different (non-saturating) gradient direction than the six R15.S3
  phases observed.
- **Design:** Extend `relabel_decision_trace` to attach multiple `labelSource`
  rows per trace row (rollout, planner, search). Train on the mixed corpus;
  use label-smoothing or top-K objective so the policy targets a mixture
  rather than an argmax. Reuses item-5 trace-teacher infrastructure.
- **Cost:** ~2 h code (planner relabel + multi-label dataset) + ~30 min
  training + 1 PPO sweep at phase-H scale (5 min). ~3 h total.
- **Pre-register exit:** warm-start Wilson lower ≥ 0.35 *and* one PPO sweep
  Wilson lower ≥ 0.40, or close the branch with the new SL ceiling number.
- **Status:** Surviving R15.S3 closeout candidate. Human rank required.

### R8. DPO (Direct Preference Optimization) (forward, human-rank, escalation open)

- **Q:** Given rollout-CRN gives us (state, top-1-action, runner-up-action)
  tuples with score margins, can DPO push past the F1 cap on pairs alone
  without per-step reward shaping (which R15.S3 saturated)?
- **Hypothesis:** Dense pairwise preference signal is more learnable than
  sparse ±1 terminal reward; doesn't depend on the per-step shape signal
  family that R15.S3 closed.
- **Design:** New trainer `train_dpo.py` reading the outcome-export's
  per-candidate reward records (already include selected vs runner-up
  margin). Bradley-Terry loss against the warm-start as the reference
  policy.
- **Cost:** ~3 h code + ~10 min training.
- **Pre-register exit:** Wilson lower ≥ 0.40 from the item17 warm-start
  on the rule-bot gate, or document the new ceiling and close.
- **Status:** Surviving R15.S3 closeout candidate. Human rank required;
  bigger pivot than R7 (replaces PPO entirely).

### R7.b. Feature representation expansion (forward, family, parked behind R7)

Full design: `docs/ai-research/scoping/r7b-feature-representation.md` (audit + 6-item intervention menu + ranked picks + cross-cutting deps + exit gates).

**One-liner:** test whether the F1 cap at Wilson 0.368 is binding on *information available to the network* (not labels via R7, not objective via R8) by replacing the 16 hashed-float card-identity slots with a learned embedding table.

**Family (ordered):**

- **R7.b.0 — Trace-JSONL re-encodability spike (precondition, autonomous-eligible).** ~30 min: confirm whether existing R12/R13/R14 trace JSONLs can be re-extracted under a new feature schema without resimulating. Gates the cost of every other entry below.
- **R7.b.1 — Hygiene quick-win (autonomous-eligible, independent of R7).** Wire the three unread JSON fields — `firstPlayer`, `pendingChoiceKind`, per-uma `toolCardId`. Schema-additive (v2.1), ~half day, no regression risk.
- **R7.b.2 — Card embedding + action-target embedding (headline pass).** Bundle `nn.Embedding(108, K=32)` per-zone sum-pool replacing slots 32–47 with action-side embedding lookup on source/target ids. ~1.5 days + half day. Schema bump v3. Targets audit categories 1, 2, 5, 8.
- **R7.b.3 — Set-encoder / attention over per-card tokens (conditional follow-up).** Launch only if R7.b.2 lifts but caps below 0.40. ~1 day on top of R7.b.2.
- **R7.b.4 — Recent action history (separate branch, deprioritised).** Engine ring-buffer + per-action embedding. Gated on R7.b.0 — if SL JSONLs can't be regenerated cheaply, this branch is much more expensive than its expected lift justifies.
- **R7.b.5 — Aux heads / capacity bump (revisit-after).** Capacity (hidden 256, depth 4) and aux heads (predict opp action, Δvalue) re-evaluate after R7.b.2 outcome.

**Exit (verbatim from scoping § 7):** (a) SL warm-start does not regress vs current best (within 2pp) AND (b) one PPO sweep Wilson lower ≥ 0.40, OR document the new ceiling at expanded schema and close.

**Status:** Parked behind R7 (R7 step 3 in flight 2026-05-14). R7.b.0 + R7.b.1 are autonomous-launch eligible and can fire independently of R7 outcome (precondition spike + hygiene). R7.b.2+ promote to ready when R7 closes.

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

- **Side-imbalance verification.** Gate manifests record player/opponent
  splits inconsistently across phases. Worth a one-off script to extract
  the side-WR delta and check whether the model is offensively weak or
  defensively weak. R14.A.footnote confirmed the gap is real and not
  seed-clustered at value-head-leaf inference; question is whether it shows
  the same shape at rollout-leaf (where R14.I.2 already passes by 25pp).
- **Simulator determinism audit.** Replay 100 identical seeds end-to-end;
  measure full-state divergence rate. Silent non-determinism in CRN would
  invalidate every advantage estimate. R14.B AsyncLocalStorage closed the
  parallel-worker case (R-WILD #34); single-worker drift remains unaudited.
- **Rule-bot mistake catalog.** The 80% aspirational target requires
  exploiting rule-bot weaknesses; we don't have a catalog of those
  weaknesses. Hand-construct ~50 states + a careful audit.
- **Side-asymmetry confirmation gate (R14.A footnote).** *CLOSED 2026-05-14
  — gate1 FAIL @ Wilson 0.394; the R14 cheap-inference production claim is
  contradicted at independent seeds. See progress doc § "R14.A.footnote"
  and `docs/ai-agent-state/escalations.md` for the open research-stance
  decision the human owns.*
