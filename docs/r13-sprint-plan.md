# R13 Sprint Plan — Make R12 Deployable, Find the Compounding Path

Created 2026-05-11 after R12 cleared the 0.40 Wilson-lower bar by 15.6pp (final: 0.556 at n=200, rollout-leaf MCTS over R4). This sprint is the natural follow-on. See `docs/r12-sprint-plan.md` for the prior sprint and `docs/ai-research-backlog.md` for the R12 result + diagnostic.

## North star

**Make rollout-leaf MCTS deployable in the live game at <3 s per decision, AND determine whether a retrained value head (distilled from rollout-mean outcomes) is the cheapest path to compounding strength.**

Two parallel deliverables: (1) a playable UI opponent; (2) a falsifiable answer to "is the value head fixable, or is rollout-CRN search permanently the production path".

## The reframing that drives sequencing

R12's diagnostic in one sentence: **rollouts are the signal, the network is just a search organizer.** Decomposition (Wilson lower over R4 baseline 0.30):
- +3pp from uniform MCTS over value-head leaf
- +3pp from policy prior on top
- **+19pp from swapping value-head leaf to rollout-CRN K=3**

Phase D as originally written distills *visit counts* into a policy, but visit counts are derived from the value head's leaf scores. A distilled policy that reuses the value head inherits R4's noise floor. The cheaper fix is to first replace the value head's *training target* — regress to mean-rollout-outcome instead of `z`-the-game-outcome. That target is, by construction, lower-variance.

So W3 (retrain value head) is upstream of W6 (Phase D). W3 is a 4-hour falsifiable experiment. Run it before committing 8h to W6.

## Workstreams

### W1 — Game-level parallelism (2 days, no deps)

The eval gate and MCTS selfplay are embarrassingly parallel. Single serve_onnx can handle multiple concurrent /predict callers (single-row requests; GPU is currently 20-40% utilized).

- **MODIFY** `backend/src/sim/evalGate.ts` (~80 LOC): add `--workers N` flag; partition `args.games` across N child processes via `node:cluster` or a simpler `child_process.fork` pool. Each child plays a slice, writes its own partial progress, parent aggregates manifests.
- **MODIFY** `backend/src/sim/mctsSelfPlay.ts` (~60 LOC): same pattern for selfplay rows; appendFileSync per-child to a shared jsonl is safe under POSIX `O_APPEND`.
- **Exit criterion:** 100-game gate at current MCTS settings completes in ≤1/3 single-process wall-clock. Validate determinism — 4 workers replaying the same seeds must produce the same per-game results.
- **Smoke:** `training/r13_parallel_smoke.py` — 4 workers × 4 games, asserts wall-clock < 80% of single-process control.

### W2 — Latency dials (1.5 days, no deps)

The 5–30 s/decision is unacceptable for human play. Three independent dial moves, ordered by leverage:

1. **K=1 rollouts.** Run the gate at `--mcts-rollout-crn-samples 1`. R1 showed rollout-CRN×3 hits WR 58.5%; K=1 is noisier but cheap. Hypothesis: keeps Wilson lower ≥ 0.45 at 1/3 the rollout cost.
2. **Adaptive sim cap.** Halt MCTS at root when `max_visits / second_max_visits > θ` (e.g. θ=3). 62% of decisions are 1-action forced; another large fraction has a runaway top action by sim 30. Cheap implementation in `runMcts` loop.
3. **Batched /predict.** Inside MCTS, collect `B` newly-expanded leaves before issuing one batched /predict (virtual loss on in-flight expansions). Existing PUCT supports it; requires a small refactor in `buildModelDecisionNode`.

- **MODIFY** `backend/src/sim/mcts.ts` (~120 LOC) for adaptive cap + batched expansion.
- **Exit criterion:** a Pareto setting that holds **Wilson lower ≥ 0.45 at n=100** with **p95 decision latency < 3 s**. Document the (sims, K, θ) settings.
- **Smoke:** `training/r13_latency_probe.py` — runs at 3 candidate settings, prints (WR, p95 ms).

### W3 — Distill rollout-mean outcomes into the value head (1 day for the experiment)

**This is the dispositive experiment of the sprint.** Cheap to run, decisive in outcome.

- **MODIFY** `backend/src/sim/mctsSelfPlay.ts` (~40 LOC): for each row, also log `meanRolloutValue` = mean of the K rollout terminal values at the *root* state (already computed during root leaf eval — currently thrown away).
- **NEW** `training/uma_ai/value_target_dataset.py` (~80 LOC): dataset adapter that uses `meanRolloutValue` as the value target instead of game outcome `z`.
- **NEW** `training/r13_value_retrain.py` (~120 LOC): trains only the value head (freeze trunk + policy heads); 25 epochs over the W3 selfplay corpus; init from R4.
- **EXPERIMENT:**
  1. Run 100 selfplay games with current rollout-leaf settings to generate rows with `meanRolloutValue` logged.
  2. Retrain just the value head on those rows.
  3. Gate at `--mcts-leaf value-head` (NOT rollout) with the retrained head, n=100.
- **Decision rules:**
  - **Wilson lower ≥ 0.40** → the value head is fixable. Phase D (W6) becomes viable. Deploy without rollout-CRN is possible. Big win.
  - **Wilson lower 0.30–0.40** → partial signal; full Phase D iter still worth trying with the retrained head as warm-start.
  - **Wilson lower < 0.30** → value-head distillation doesn't work. Search-at-inference is the permanent production path. Skip W6 entirely; double down on W2/W1.
- **Smoke:** `training/r13_value_retrain_smoke.py` — 2 epochs on 16 synthetic rows; loss decreases.

### W4 — MCTS-vs-MCTS strength ladder (1 day, no deps)

Rule-bot WR is saturating. We need a yardstick that won't.

- **MODIFY** `backend/src/sim/evaluateModelVsHeuristic.ts` (~50 LOC): add `--opponent-selection mcts` so the non-model side runs MCTS with its own config. Existing `opponentModelUrl` plumbing already supports this shape.
- **NEW** `training/r13_mcts_ladder.py` (~80 LOC): runs three pairings (R4@50sims vs R4@200sims, R4@100sims rollout vs R4@100sims value-head, R4 vs R4-W3-retrained at fixed sims) and writes a ladder JSON.
- **Sanity test:** R4@200 vs R4@50 should split ≥60/40 with rollout-leaf. If ~50/50, search is noise-limited and only W3 can move us.
- **Exit criterion:** ladder JSON exists with at least three pairings; sanity test passes.

### W5 — Live UI integration (2 days, depends on W2)

Wire MCTS into the in-game opponent. Today the UI runs `advanceOpponentTurnStep` directly — no model in the loop.

- **NEW** `backend/src/server.ts` route `POST /ai/decide` (~80 LOC). Body: `{state, modelSide, mctsConfig}`. Returns `{actionIndex, diagnostics}`. Wraps `chooseMctsAction`.
- **NEW** `frontend/src/game/engine/ai-policy/mctsClient.ts` (~80 LOC). HTTP client with 5 s timeout → fallback to rule-bot.
- **MODIFY** `frontend/src/app/hooks/useAppRuntimeEffects.ts` (~30 LOC). Add `aiBackend` option in settings; when `"mcts"`, await client; on resolve, advance with the returned action; on timeout, fall back.
- **NEW** settings toggle in `MainMenuScreen.tsx`: "AI: rule-bot | MCTS (slow)". Hide behind a dev flag initially.
- **Exit criterion:** play 5 full UI games vs MCTS, zero hangs, latency reported in dev console. Fallback to rule-bot on timeout works.
- **Smoke:** `frontend/src/app/hooks/__tests__/mctsClient.test.ts` — mocks fetch, asserts dispatch.

### W6 — Phase D real run (conditional on W3, 1 day with W1's parallelism)

Only if W3's value-head retrain produced something usable. Run `r12_orchestrator.py` for 2 iterations at production scale (200 selfplay × 100 sims × rollout-leaf, 200 gate × 100 sims) — with W1 parallelism that's ~2 h.

- Use the W3-retrained checkpoint as the iter-0 promoted checkpoint (instead of R4).
- Iteration target: each iter promotes ≥ +2pp Wilson lower over the previous.
- Halt-after-2 already in place.

If W3 was negative, **skip W6 entirely.** Production answer is W2 (search-at-inference, deployed fast).

## Sequencing & wall-clock

Days 1–2 in parallel: W1 (parallelism), W2 (latency dials), W4 (MCTS ladder). Three small tracks.
Day 3: W3 experiment. ~4h compute + ~30 min analysis. **Decision point.**
Days 4–5: W5 (UI integration), uses W2's tuned settings.
Day 6: W6 conditional on W3.
Day 7: documentation + final 400-game headline gate on the strongest config.

Total: 6–7 days of focused work. ~1100 LOC new, ~250 LOC modified.

## Risk register

| Risk | Severity | Mitigation |
| --- | --- | --- |
| Parallel workers break CRN determinism | MEDIUM | Each worker gets its own seed-start slice; replay assertion in W1 smoke. |
| K=1 rollouts collapse strength | MEDIUM | W2's exit criterion bakes in Wilson lower ≥ 0.45 floor; fall back to K=2 or K=3 if K=1 fails. |
| Batched /predict introduces async races in MCTS | MEDIUM | Standard virtual-loss pattern (AlphaGo Zero §3); add an MCTS replay assertion. |
| W3 value-head retrain produces NaN losses or worse-than-R4 | LOW | Frozen trunk/policy heads → only value head trains. Worst case ~5 min of compute wasted; result still informative. |
| UI client times out frequently, ruining UX | HIGH | W2 must hit p95 <3 s first. W5 timeout is 5 s with rule-bot fallback. Add a "fallback rate" metric to dev console. |
| MCTS-vs-MCTS shows R4 is noise-limited (50/50 across compute) | LOW | W4 sanity test catches this on day 1. If true, W3 becomes more urgent, not less. |

## What we explicitly DROP from the R12 plan

- **Phase D as originally specified** (visit-count → policy distillation, no value-head fix). Replaced by W3 + conditional W6.
- **n=400 / n=800 cosmetic gates.** GO criterion is decisively cleared; bigger n is mostly aesthetic. The final 400-game gate at the *strongest* config (post-W3) goes in W7 docs day, not as a standalone sprint phase.
- **Multi-temperature gate matrix (R2).** Demoted to deployment tuning, not research.

## Open questions for the next decision point

1. After W3 lands, do we go W6 (compounding) or W5 (ship)? Both, sequenced — but if compute is constrained, W5 first.
2. After W5 lands, is the latency low enough to be fun, or do we need GPU/batching investments?
3. After W4 ladder, is rule-bot saturated enough that we should stop reporting WR vs rule-bot in headlines?
