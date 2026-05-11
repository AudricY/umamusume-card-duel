# R14 Sprint Plan — Ship the Cheap Config, Honestly Off-Policy RL

Created 2026-05-11 after R13 closed with two deployable configs (iter-1 + rollout-leaf MCTS at Wilson lower 0.573, iter-1 + value-head-leaf MCTS at Wilson lower 0.452) and a clearly-blocked raw-policy PPO probe (Wilson 0.24–0.30 with importance ratios stuck at ~1.0). See `docs/r13-sprint-plan.md` for the prior sprint and `docs/ai-research-backlog.md` for the R13 result section.

## North star

**Make the cheap-inference config user-playable, prove it isn't over-fit to its own selfplay distribution, and run the only PPO variant we never honestly tried (MCTS-trajectory off-policy).**

R13 produced strong gate numbers but nothing ships yet — the UI hook is plumbed but the runtime path is unfinished. R14 is split between deployment (W5 finish + latency tuning) and one last serious RL probe (MCTS-trajectory PPO) before declaring search-at-inference the permanent answer.

## Workstreams

### A — Out-of-distribution gate for iter-1 (compute only, 30 min)

**Why:** iter-1 + value-head leaf at Wilson 0.452 is suspiciously close to the in-distribution rollout-leaf number. Possible over-fit to the W6 selfplay distribution (the value head saw the same leaf scores it'll be graded on at inference). Needs a clean OOD test.

- **Gates:**
  1. Re-gate at value-head leaf on seeds 800000+ (new), n=100
  2. Gate iter-1 + value-head leaf against the R4 checkpoint as opponent (MCTS-vs-MCTS, not rule-bot), n=100
- **Exit criterion:** Wilson lower ≥ 0.40 on BOTH gates. If gate 1 holds but gate 2 fails by >5pp, we know iter-1 is rule-bot-specific and need to dial down the strength claim.
- **Cost:** ~10 min wall-clock per gate with W1 parallelism. Zero code.

### B — Engine determinism fix (the real R-WILD answer, 1 hour)

**Why:** subagent investigation pinpointed the bug. `withRng` wraps an async function in `runModelVsHeuristicGame`/`runSelfPlayGame` but its module-level `activeRng` doesn't survive `await`. Engine sites in `flow/combat.ts`, `flow/setup.ts`, `flow/trainers.ts`, `flow/ai/core.ts` call `randomFloat()`/`randomInt()`/`shuffle()` and silently fall back to `Math.random()` after the first await. This is the W1 parallel-vs-serial divergence and *possibly* the W7 side asymmetry.

- **Fix:** AsyncLocalStorage in `frontend/src/game/engine/core/random.ts` (~10 LOC). `withRng` becomes a wrapper around `rngStorage.run(rng, run)`; `randomFloat()` reads `rngStorage.getStore()?.next()`.
- **Validation:** rerun `training/r13_parallel_smoke.py` — serial-vs-parallel determinism must now hold bit-exact.
- **Cost:** ~30 min code + ~5 min validation.
- **Closes:** task #34 (R-WILD residual).

### C — W8 stop rule + analysis

**Why:** value-head Phase D loop is running. Concave compounding is expected. Explicit stop rule prevents sunk-cost iteration.

- **Inspection:** after W8 iter-2 lands, check Wilson lower delta vs iter-1. If <+2pp, halt the loop (kill bg) and pivot compute to D/E.
- **If continues:** allow iter-3, iter-4, iter-5 to land. Track per-iter Wilson + per-side WR.
- **Cost:** zero, just an inspection gate.

### D — MCTS-trajectory PPO, the honest off-policy run (~3 days code + 1 day compute)

**Why:** every prior PPO sweep was effectively on-policy because collection = target policy → ratios ≈ 1.0 → no signal. The only PPO variant we never ran is: collect under MCTS-augmented behavior (rollout-leaf or value-head-leaf), log MCTS visit-distribution as `behavior_logprobs`, train the raw policy with importance ratios π_raw/π_mcts. This is the AlphaGo Zero pattern except with PPO instead of supervised distillation.

- **Sub-tasks:**
  - **D.1** Plumb collection-side behavior-logprob logging from MCTS visit distribution into the existing trace format (the F1 trace already has a `behaviorPolicy` field; needs an `mcts` kind).
  - **D.2** `ppo_orchestrator.py` adds `--collection-selection mcts` and `--mcts-leaf {value-head, rollout}` to the trace stage. Sampling at collection time = visit-distribution sampling, not policy softmax.
  - **D.3** Train with V-trace clipping (c̄=1.0, ρ̄=1.0) on the off-policy ratios. Add a guard for very small denominators in π_mcts(a|s).
- **Exit criterion:** Wilson lower at raw-policy gate ≥ 0.40 after 3 iters. Falsifies the "PPO plumbing is broken" hypothesis cleanly — if this still doesn't move with a genuinely off-policy signal, the permanent answer is search-at-inference and RL is closed for this codebase.
- **Cost:** ~300 LOC + ~3 days compute for 3 iters at value-head-leaf collection (cheaper than rollout-leaf).

### E — W5 UI integration finish (~1 day)

**Why:** R13's whole point was a deployable opponent. Plumbing without integration is item-17 redux.

- **Add to engine:** thin wrapper `applyChosenAiAction(state, sideId, action, ...)` that mirrors backend `advanceModeledTurnStep` but lives in the frontend engine and reuses the existing combat/play resolution.
- **Hook:** `useAppRuntimeEffects.ts` opponent branch: when settings.aiBackend === "mcts", call `requestMctsDecision`, on success apply via the new wrapper, on timeout/error fall back to `advanceOpponentTurnStep`.
- **Toggle:** dev-flag setting in `MainMenuScreen.tsx` ("AI: rule-bot | MCTS (slow)").
- **Exit criterion:** play 20 full UI games with aiBackend=mcts, fallback rate <5%, median decision time <3 s.
- **Cost:** ~1 day code.

### F — Adaptive-ratio Pareto sweep at value-head scale (optional, 20 min)

**Why:** R13.W2 smoke at value-head leaf showed 3.86× speedup at ratio=2.0 with no strength assertion. Production needs a defended Pareto point for W5.

- **Sweep:** `--mcts-adaptive-ratio` ∈ {1.5, 2.0, 3.0, 5.0} at 100 sims, value-head leaf, n=100, seeds 820000+.
- **Exit criterion:** identify a setting that keeps Wilson lower ≥ 0.42 while cutting wall-clock ≥30% vs ratio=0. Feeds into W5's default config.
- **Cost:** ~20 min wall-clock total.

### G — Unpin ORT for /predict concurrency (gated on B, ~10 min)

**Why:** R13.W1 set `intra_op_num_threads=1`, `inter_op_num_threads=1`, sequential execution mode in `serve_onnx.py` to remove FP non-determinism under concurrent /predict callers. With R14.B's AsyncLocalStorage fix, engine determinism is enforced at the right layer (the engine's RNG state, not the inference reduction order). Giving back ORT concurrency should restore meaningful /predict throughput when many workers call simultaneously — especially relevant to value-head-leaf workloads (W8-style loops) where the simulator finishes fast and workers are starved on inference.

- **Modify:** `training/serve_onnx.py` — remove the SessionOptions pinning OR add a CLI flag `--ort-threads N` (default `auto`).
- **Validation:** rerun `training/r13_parallel_smoke.py` and `training/r13_selfplay_parallel_smoke.py`. With B+G, serial-vs-parallel results should still be bit-exact (engine-level RNG dominates) AND parallel wall-clock should drop materially (probably ≥30% on W8-style runs).
- **Exit criterion:** ≥2× /predict throughput per worker measured via `training/throughputProbe` (or a new small probe), no determinism regression in the parallel smoke.
- **Cost:** ~10 min code + ~10 min validation. Gated on R14.B landing first.

### H — Batched /predict proxy (conditional on G not being enough)

**Why:** even with ORT unpinned, our model is small enough that each /predict is microseconds and GPU sits idle. A Python proxy in front of serve_onnx could buffer concurrent calls (e.g., 5ms window or 32-call batch), issue one batched inference, and demultiplex the results. Real GPU work happens, and the ~per-call HTTP overhead is amortized.

- **Build:** `training/serve_onnx_proxy.py` with `--upstream-port` and `--max-batch N --batch-window-ms M`. Same /predict interface; same response shape. Demultiplexer assembles batched logits/value back into per-caller responses.
- **Risk:** the upside ceiling is bounded by per-call overhead, not GPU compute. If the model stays small, batching may give only 1.5–2× — worth measuring but not a sprint priority.
- **Validation:** parallel smoke + W8 selfplay wall-clock. Tolerate ≤5% strength regression (introduces tiny FP non-determinism via batch-mate reordering).
- **Exit criterion:** ≥2× selfplay wall-clock improvement at 4 workers vs G alone. If <1.3×, ship without it.
- **Cost:** ~250 LOC + ~half day. Only schedule if G yields <2× throughput.

## Sequencing

- **Day 1:** A (OOD gate) + B (determinism fix) in parallel. Both are short. Once B lands, do G (unpin ORT) and rerun parallel smoke.
- **Day 2:** W8 iter-2 inspection (C); start E (UI integration). F (adaptive ratio sweep) opportunistic.
- **Day 3:** D.1 + D.2 (MCTS-trajectory PPO plumbing).
- **Day 4:** D.3 + first 3-iter run; finish E. If G alone hasn't given ≥2× throughput, consider H.
- **Day 5:** D analysis; F sweep if not done; sprint write-up.

Total: 5 days, ~1 day of compute, ~3 days of code (excluding H which is conditional).

## What we explicitly DROP

- **Larger model.** R6 evidence is dispositive — bigger nets fit the labels harder and play worse.
- **More entropy-regularized BC variants.** R3 already settled this.
- **A standalone n=400 headline gate.** Wilson half-width at n=100 is already tight; cosmetic.
- **Temperature-ramp PPO alone** (subagent 2 E2). Cheap to run but R3 history says peakedness-only interventions don't move the gate. If D fails, this is a "before declaring RL dead" follow-up, not a sprint slot.

## Decision points

1. **After A:** is iter-1 + value-head leaf OOD-robust? If no, the production config is rollout-leaf only.
2. **After B:** does AsyncLocalStorage close the W1 parallel-determinism gap? If yes, W7 side asymmetry was probably just seed noise (close R-WILD as DONE).
3. **After D's first iter:** are importance ratios meaningfully ≠ 1.0? If not, the issue isn't off-policy plumbing — it's the value head's policy-vs-mcts return mismatch, and RL is genuinely closed.

## Risk register

| Risk | Severity | Mitigation |
| --- | --- | --- |
| AsyncLocalStorage adds non-zero perf cost | LOW | Measure W1 smoke wall-clock pre/post; expected < 5% slowdown. |
| MCTS-trajectory PPO denominators explode (visit dist near zero on rare actions) | MEDIUM | Add ε=1e-4 floor on behavior logprobs; clip V-trace ratios at ρ̄=1.0. |
| OOD gate fails (iter-1 is rule-bot-specific) | MEDIUM | Demote the production claim to "rule-bot only"; still ship to UI since the opponent IS a rule-bot variant. |
| UI fallback rate is high in practice | HIGH | Latency probe at value-head + ratio=2.0 first (F); set timeout to 5s; log fallback rate prominently in dev console. |
