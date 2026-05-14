# R7. Multi-Teacher BC Blend — Scoping

- **Date:** 2026-05-14
- **Status:** scoping in progress (no code yet)
- **One-liner:** Train the SL warm-start against a per-state mixture of rollout-CRN, search, and planner teacher labels (instead of one teacher) to test whether teacher diversity breaks the rollout-CRN noise floor that R6 + R15.S1 made dispositive.
- **Forward brief:** `docs/ai-research-backlog.md` § "R7. Multi-teacher BC blend" (lines 101–118).
- **Pick rationale:** `docs/ai-agent-state/escalations.md` § Resolved, 2026-05-14 self-directed bullet (R7 over R8).

## 1. Question (Q)

> Does mixing rollout + planner + search teacher labels per state produce a warm-start with broader competence and a different gradient landscape than the rollout-CRN-only fit that R3/R4/R6/R15.S1 saturated?

Falsifiable proposition: a multi-teacher warm-start either (a) clears Wilson lower ≥ 0.35 on the rule-bot gate AND one phase-H-scale PPO sweep clears Wilson lower ≥ 0.40, or (b) closes the F1 SL-line with a new measured ceiling.

## 2. Hypothesis + Motivation

**Hypothesis.** The cap that R3/R4/R6/R15.S1 hit is not "PPO needs more compute" or "SL needs more capacity"; it is **rollout-CRN label noise inherited by the warm-start**. R6 (hidden=128/depth=3) reached 83% argmax-match on the rollout-CRN labels but the *weakest* gate WR (33.0%) — capacity exceeded teacher quality (`docs/ai-research-backlog.md` lines 90–92). R15.S1 ran compute-scaled DAgger from the same teacher and closed FAIL at iter-2 Wilson 0.3269, inside the pre-registered falsification band (`docs/ai-research-backlog.md` line 161–164).

If both axes (capacity and compute) saturate at the same teacher, the teacher itself is the ceiling. The R7 bet is that a per-state mixture target (rollout + search + planner) regularises away each teacher's idiosyncratic mistakes, leaving a warm-start whose policy is broader and whose PPO gradient direction is not pre-curved by rollout-CRN's variance signature.

**Strength of the bet.** R12 GO and R14.I.2 land at Wilson 0.6479 from rollout-leaf MCTS at *eval time* — i.e. the same rollout family at higher search budget works fine for play; it is specifically the SL distillation target that is noisy. So we have prior-art that the play signal is recoverable — just not via single-teacher SL.

**Weakness of the bet.** If `search` and `planner` are themselves weaker than rollout-CRN (likely true at fixed compute given how `chooseSearchAction` / `choosePlannerAction` are wired in `backend/src/sim/evaluateModelVsHeuristic.ts:754,773`), the mixture target may be a worse imitation target than rollout alone. The hypothesis hinges on the three teachers making *different* mistakes, not on any one being stronger.

## 3. Design

### Teachers committed

Three teachers already exist as trace-teacher options and have selector functions co-located in `backend/src/sim/evaluateModelVsHeuristic.ts`:

- **rollout** (`chooseRolloutAction` at `backend/src/sim/evaluateModelVsHeuristic.ts:709`) — rollout-CRN MCTS leaf, value-head rollout. This is the R3/R4/R6/R15.S1 teacher.
- **search** (`chooseSearchAction` at `backend/src/sim/evaluateModelVsHeuristic.ts:754`) — bounded-depth search with `searchTopK` / `searchSamples`.
- **planner** (`choosePlannerAction` at `backend/src/sim/evaluateModelVsHeuristic.ts:773`) — heuristic planner with `plannerTopK` / `plannerMaxSequences` / `plannerMaxDepth`.

All three are dispatched by `parseTraceTeacher` (`backend/src/sim/evalGate.ts:393`) and attached to trace rows via `chooseTraceTeacher` (`backend/src/sim/evaluateModelVsHeuristic.ts:428`). We commit to all three.

### Trace-schema change — pick: per-row mixture distribution

Two options were considered:

- **(A) Multi-row.** Emit three rows per state, each with one `teacher`. Pros: minimal schema change. Cons: triples corpus size, inflates trace-gen wall-clock 3x, and makes per-state "mixture" learning impossible — the loss sees three separate argmax targets, not a soft target.
- **(B) Per-action distribution.** Emit one row whose `teacher` field becomes `teachers: [{selection, selectedActionIndex, weight}, ...]`. The relabel pass then converts the list into a `policyTargets: number[]` over the row's `legalActions`. **Pick.**

Why (B): the Python SL trainer **already supports** a soft target distribution path (`training/train_bc.py:361–369`, the `policy_targets` branch added in R12 phase C). The dataset loader (`training/uma_ai/dataset.py:46–79`) currently only produces a single `target_index`, so we wire the new field through there. No trainer change needed — only loader + relabel.

Concrete schema delta in `backend/src/sim/dagger/relabelDecisionTrace.ts`:

- `DecisionTraceRow.teacher` (line 31–36) becomes `teachers: Array<{ selection: "rollout"|"search"|"planner"; selectedActionId; selectedActionIndex; selectedOriginalRank? }>` with length 1–3. Back-compat: single-teacher runs emit a length-1 array.
- Training-row output (line 83–105) gains `policyTargets: number[]` of length `legalActions.length`, computed by averaging one-hots over the present teachers (uniform weights for v1).
- Drop the singular `teacher`/`teacherSelection` fields, or keep them as the argmax of `policyTargets` for telemetry.

### Training objective — pick: label-smoothing via soft cross-entropy (already implemented)

Two options were considered:

- **Label-smoothing / mixture target** (Pick). Loss = `-(policy_targets · log_softmax(masked_logits)).sum(dim=1)`. With uniform weights 1/3, this is exactly the "mean of three one-hot CEs" objective, but executed in one pass with the existing soft-target code at `training/train_bc.py:367–369`. Already differentiable and masked-correct.
- **Top-K objective.** `loss = mean_k CE(logits, target_k)`. Mathematically equivalent to the above under uniform weights, but requires a new code path. Rejected as redundant.

If we want non-uniform weights later (e.g. up-weight `rollout` to 0.5, split 0.25/0.25 across the others), this is a one-line change in the relabel `policyTargets` construction.

### Mixed-corpus generation recipe

Trace generation in `training/dagger_orchestrator.py:240–263` runs once per iteration and passes a single `--trace-teacher`. To get three teachers per state, the cheapest plumbing is:

- Modify `chooseTraceTeacher` (`backend/src/sim/evaluateModelVsHeuristic.ts:428`) to accept a `traceTeachers: Array<"rollout"|"search"|"planner">` and emit all of them on the row. Add an `--trace-teachers rollout,search,planner` CLI flag in `evalGate.ts:317,393`.
- Run trace-gen once at the same per-iteration row budget as R15.S1 (so direct compute-cost comparison is honest). Relabel then attaches `policyTargets` from the per-row teacher array; no `mix-sources` change needed because the corpus is single-source.
- Seed structure: reuse the per-step seed (`${seed}:${modelSide}:${step}:trace-teacher`) but suffix each teacher's selector seed with its name (`...:rollout`, `...:search`, `...:planner`) so each teacher's stochastic search is independent and reproducible.

Row budget for v1: match R15.S1's iter-1 row count (so we are comparing teachers at equal SL compute, not extra rows). Concretely, take R15.S1's iter-1 trace count from the progress doc and reuse it.

## 4. Pre-Registered Exit Gates

Verbatim from `docs/ai-research-backlog.md` line 116–117:

> warm-start Wilson lower ≥ 0.35 *and* one PPO sweep Wilson lower ≥ 0.40, or close the branch with the new SL ceiling number.

**Gate evaluation procedure.**

- **SL gate:** eval the trained warm-start with `npm run sim:eval-gate` against the rule-bot at the same n / seed range / opponent set used by R15.S1's iter-1 SL gate (so the comparison is honest). Wilson interval at 95%. Pass = lower bound ≥ 0.35.
- **PPO gate:** one phase-H-scale PPO sweep from the new warm-start (5 min wall-clock per the cost estimate). Same gate eval as above. Pass = Wilson lower ≥ 0.40.
- **Close-out condition:** if either gate fails, write the new SL ceiling number into `docs/ai-performance-research-progress.md` and demote R7 in the backlog.

## 5. Cost Estimate + Risk Callouts

Per forward brief (`docs/ai-research-backlog.md` line 114–115): ~3 h total — ~2 h code (multi-teacher trace + relabel + dataset wiring) + ~30 min SL training + 1 PPO sweep at phase-H scale (~5 min) + eval/writeup time.

Risks:

- **(a) Mixed labels noisier than single-teacher.** If `search` and `planner` make worse decisions than `rollout` at the trace-gen budget we use, averaging them into the target may *increase* SL loss noise. Mitigation: report per-teacher argmax-agreement on a held-out slice before launching PPO; if `search` / `planner` agree with `rollout` < 30% of the time AND the model is much smaller than the teacher, we are mixing in junk and should reweight.
- **(b) Diversity hypothesis is empty if teachers are too similar.** If all three teachers converge to similar actions (e.g. all rollout-flavoured under the hood), the mixture target collapses to single-teacher and we re-derive R15.S1. Pre-flight check: compute pairwise teacher agreement on a small probe corpus before committing to a full trace-gen.
- **(c) PPO from non-rollout-CRN warm-start may behave unpredictably.** The six R15.S3 phases (L/M/N/O'/O/P) were all from a rollout-CRN warm-start. A multi-teacher warm-start may have a different value-head calibration that interacts with PPO's advantage estimation. Mitigation: keep PPO hyperparams identical to R15.S3 phase H for the first sweep; treat any divergence as a finding, not a bug.

## 6. Ordered Execution Plan

Concrete steps the next implementer slot will follow:

1. **Trace schema + chooseTraceTeacher.** Extend the trace row (`relabelDecisionTrace.ts:31–36`) to a `teachers: []` list; modify `chooseTraceTeacher` (`evaluateModelVsHeuristic.ts:428`) to call all three selectors when a multi-teacher flag is set. Update `parseTraceTeacher` (`evalGate.ts:393`) to accept comma-separated input.
2. **Relabel → policyTargets.** In `relabelDecisionTrace.ts:62–113`, replace `row.teacher` consumption with `row.teachers`; build `policyTargets: number[]` over `legalActions` (uniform weights v1).
3. **Dataset loader.** In `training/uma_ai/dataset.py:46–79`, read `policyTargets` if present and pipe it through to the batch (loader will need to handle ragged `legalActions` lengths — the existing `collate_policy_batch` at line 82–109 already pads to `max_actions`, so add a parallel padded `policy_targets` field there).
4. **Generate mixed corpus.** Run trace-gen + relabel once with all three teachers, matching R15.S1 iter-1 row budget. Skip `mix-sources` for the v1 (single mixed-teacher source vs rule-bot replay is fine via existing dagger orchestrator path).
5. **SL train.** `train_bc.py` already routes `policy_targets`-bearing batches into the soft-CE branch (line 361–369). No trainer change.
6. **SL gate eval.** Wilson lower ≥ 0.35 on rule-bot gate. If fail → close out with new ceiling number.
7. **One PPO sweep at phase-H scale.** Same hyperparams as R15.S3 phase H. Wilson lower ≥ 0.40 → branch pass; else → close out with PPO ceiling number.
8. **Writeup.** Result block in `docs/ai-performance-research-progress.md`; backlog R7 entry shrinks to one-line pointer per CLAUDE.md documentation discipline.

## 7. Out of Scope (deliberately deferred)

- PPO objective change (advantage normalisation, clip schedule, KL anchor tuning).
- Reward-shape signals — R15.S3 closed across both axes, no re-opening here.
- Multi-iteration DAgger from the new warm-start (R7 is one SL + one PPO sweep; DAgger iteration is a follow-up if R7 passes).
- Non-uniform teacher weights or learned weights — v1 is uniform 1/3.
- DPO objective (R8) — separate branch, separate scope.
- Q-head replacement (R9), capacity scaling (R10), self-supervised aux (R11) — historical, deprecated by R12 GO.

## 8. Result: pre-flight teacher-agreement probe (step 1)

- **Date:** 2026-05-14
- **Probe code:** `backend/src/sim/r7TeacherAgreementProbe.ts` (~340 lines). Read-only; exports added to the three selectors in `evaluateModelVsHeuristic.ts` (`chooseRolloutAction`, `chooseSearchAction`, `choosePlannerAction`), no behavior change.
- **Run:** `runs/R7-pre-flight-teacher-agreement/probe-300-20260514T061913Z.json`. N=300 states sampled from 50 rule-bot-driven AI-vs-AI games (alternating modelSide). Wall-clock 1m 46s. Selector hyperparameters match `evaluateModelVsHeuristic.ts` `parseArgs` defaults (rolloutCrnSamples=1, rolloutSteps=500, searchDepth=2, searchTopK=4, searchSamples=1, plannerCrnSamples=3, plannerTopK=4, plannerMaxSequences=64, plannerMaxDepth=8) — i.e. honest against the trace-gen recipe R7 § 3 commits to.

**Aggregates:**

| Metric | Value |
|---|---|
| pairwise agreement rollout↔search | 0.800 |
| pairwise agreement rollout↔planner | 0.180 |
| pairwise agreement search↔planner | 0.230 |
| all three agree | 0.133 |
| all three disagree | 0.057 |
| mean normalised mixture entropy | 0.618 |
| mean per-state ms: rollout / search / planner | 13.4 / 22.7 / 315.5 |
| mean legal actions per state | (see JSON `aggregates.legal_actions`) |

**Verdict: GO.** Rule that fired: `rollout↔search = 0.800 is in [0.30, 0.85] AND all-three-agree = 0.133 < 0.50` — meaningful per-state disagreement exists, teachers do not collapse to a single label.

**Key observations:**

- **rollout↔search agreement is high (0.80) but not collapsed.** This is unsurprising — both teachers use the same rollout-heuristic leaf evaluator and the same `enumerateLegalAiActions` ranker; they differ mainly in search depth + CRN structure. They still disagree on ~20% of states, which is the slice the mixture target regularises.
- **planner disagrees a lot with both** (rollout↔planner=0.18, search↔planner=0.23). The planner enumerates *turn-bundles* (chained action sequences) and picks the first action whose grouped first-action score is best — a structurally different decision rule from greedy-leaf rollout. This is the diversity source the R7 hypothesis hinges on; the probe shows it is genuinely present, not synthetic noise.
- The NO-GO-reweight rule (rollout↔search AND rollout↔planner both < 0.30) did *not* fire — rollout and search still agree most of the time, so the mixture does not drown rollout in junk.
- **Planner is ~14× slower than rollout per state** (315ms vs 13ms). At the trace-gen budget R15.S1 used, this projects to roughly +5–6 min wall-clock per iteration of trace-gen if all three teachers run on every row. Not blocking, but worth noting for step 4's row-budget plan — keeping plannerMaxSequences=64 is the right cap.

**Next step:** R7 step 2 (schema change — extend `DecisionTraceRow.teacher` → `teachers: []`, build `policyTargets` in relabel pass).

