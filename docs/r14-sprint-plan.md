# R14 Sprint Plan — Ship the Cheap Config, Honestly Off-Policy RL

Created 2026-05-11. **Refined 2026-05-11 after W8 negative result + subagent re-prioritization.** R13 closed with two deployable configs (iter-1 + rollout-leaf MCTS at Wilson lower 0.573, iter-1 + value-head-leaf MCTS at Wilson lower 0.452) and a clearly-blocked raw-policy PPO probe. W8 then showed value-head-leaf Phase D regresses iter-on-iter from W6 iter-1 — *not a permanent failure*, but a premature switch before the value head has caught up to the rollout estimator. See `docs/r13-sprint-plan.md` for the prior sprint and `docs/ai-research-backlog.md` for result sections.

## North star (refined)

**Extend the only loop that demonstrably compounds (W6 rollout-leaf Phase D), instrument it with crossover telemetry so we know when cheap-selfplay becomes viable, and finish UI integration. Demote the PPO probe to a fallback.**

The W6 → W8 → W6-extension thread is now the highest-information compute spend. Each W6 iter is ~20 min, gives +3pp Wilson on the strongest config, and produces a value head closer to rollout-mean — at some crossover the cheap-selfplay loop becomes viable as a free consequence (no separate W3-style retrain needed).

## Workstreams (priority order)

### I — Extend W6 rollout-leaf Phase D + crossover telemetry (TOP PRIORITY, ~4 hours)

**Why:** W6 already showed +3.4pp iter-on-iter Wilson improvement. The compounding curve hasn't plateaued. W8's negative result told us *when* cheap selfplay becomes viable — when the value head's predictions track rollout-CRN-K=3 means closely enough. Extending W6 has two payoffs: (1) directly stronger rollout-leaf gate numbers, and (2) a value head that eventually unlocks cheap-selfplay-on-the-fly.

- **I.1 New script `training/r13_value_crossover_probe.py` (~80 LOC).** Loads a checkpoint + a held-out rollout-leaf selfplay corpus (rows have `rootValue`), runs one forward pass, emits `{n, mse, rmse, pearson_r, mse_floor, ratio, crossed}` JSON. Threshold logic: `crossed = (mse ≤ 1.10 × mse_floor) AND (pearson_r ≥ 0.7)`, where `mse_floor` is the W3 retrain's `val_loss` final. Anchors against the W3 manifest. Uses `ValueTargetDataset` + `collate_mcts_selfplay_batch` from existing code.
- **I.2 Run W6 extension.** `r12_orchestrator.py` from W6 iter-1's checkpoint, 3 more iters (iter-2/3/4), rollout-leaf MCTS, 60 selfplay × 100 sims K=3, 120-game gate. After each iter's distill step, automatically run the crossover probe against the *previous* iter's selfplay corpus and write `iterations[i].crossover` into orchestrator-state.
- **I.3 Retry W8-style cheap-selfplay** ONE iter after two consecutive `crossed=true` events. Falsifies that the crossover is real and that cheap-selfplay can compound past it.
- **Exit criteria:** strongest config Wilson lower ≥ 0.60 OR per-iter Wilson delta <+1pp (plateau). MSE curve recorded for postmortem.
- **Cost:** ~80 LOC + 80 min compute for 4 iters + 30 min for probe code + 20 min retry. Highest information density on the sprint.

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

### D — MCTS-trajectory PPO (DEMOTED to fallback after I plateaus)

**Status change:** W6's compounding loop is the cheaper, lower-risk path forward. PPO with V-trace was R14's headline RL probe but after W6 confirmed compounding and W8 told us when cheap-selfplay becomes viable, the EV ranking flipped. Run D *only if* I plateaus AND the post-crossover retry W8 still regresses. Then we know iteration alone won't unlock cheap inference and a structural RL change is warranted.

- **Same design:** plumb MCTS visit-distribution as `behavior_logprobs`, train raw policy with V-trace clipping (c̄=1.0, ρ̄=1.0).
- **Cost when run:** ~300 LOC + ~1 day compute. Defer the code work until I's exit criterion is hit.
- **If I succeeds and cheap-selfplay works:** D is **redundant** and skipped entirely. The product win (fast inference) is achieved by the W6 → cheap-selfplay ladder, not by PPO.

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

## Sequencing (refined)

- **Day 1 morning:** I.1 (crossover probe ~80 LOC) + B (AsyncLocalStorage fix). Both short, independent.
- **Day 1 afternoon:** kick off I.2 (W6 extension iter-2/3/4) unattended. ~80 min compute. While it runs: A (OOD gate, 30 min), F (adaptive ratio, 20 min), G (unpin ORT after B lands).
- **Day 2 morning:** read I results. Either iterate more (Wilson still climbing) or evaluate crossover (MSE delta <10%). If crossed: I.3 retry W8-style. If not crossed but climbing: continue iterating.
- **Day 2 afternoon onward:** E (UI integration finish).
- **Day 3-4:** D (PPO with V-trace) — only if I plateaus AND I.3 still regresses. Otherwise: closeout + writeup.

Total: 2–3 days if I succeeds, 5 days if D becomes necessary. The hybrid: while I.2 burns compute unattended, write D's plumbing in parallel so it's ready if needed.

## Progress (2026-05-11)

- **B — DONE.** AsyncLocalStorage via `frontend/.../random.ts` installRngStorageProvider + `backend/.../rngAsyncStore.ts`. `training/r14_determinism_smoke.py` shows 12/12 games bit-exact between --workers 1 and --workers 4 (post-fix); previously divergent. Closes R-WILD #34.
- **I.1 — DONE.** `training/r14_value_crossover_probe.py` measures val_mse + pearson_r between checkpoint value head and rootValue (rollout-CRN K=3 mean) on a held-out selfplay corpus. Crossed iff val_mse <= 1.10 * mse_floor AND pearson_r >= 0.7. mse_floor anchored on the W3 retrain manifest. Wired into `r12_orchestrator.run_iteration` between distill and gate (records `iterations[i].crossover`). Baseline sanity at W6 iter-1 vs its own iter-1 selfplay (in-distribution): val_mse 0.659, pearson 0.535, ratio 1.158, crossed=false — consistent with W8 regression (cheap-leaf selfplay distillation didn't compound because the value head wasn't ready).
- **I.2 — IN FLIGHT.** Resuming W6 phase-d in `runs/R13-W6-phase-d/` for iter-2/3/4 (rollout-leaf MCTS, 60 selfplay × 100 sims K=3, 120-game gate, kl-anchor 0.05). Each iter's distill is followed by a crossover probe against the previous iter's selfplay. ETA ~60 min total wall-clock.
- **E — PARTIAL (code complete, e2e smoke pending).** `/ai/decide` now applies the chosen action server-side and returns `nextState`; frontend swaps state on success or falls back to `advanceOpponentTurnStep` on timeout/error/engine-fallback. Toggle via `localStorage.setItem("umamusume-card-duel-ai-backend", "mcts")`. Visible UI toggle in MainMenuScreen still pending. End-to-end UI smoke deferred until W6 extension frees up serve_onnx.
- **G — CODE READY.** `serve_onnx.py` exposes `--ort-threads` (default `1` preserves R13.W1 legacy). `training/r14_ort_throughput_smoke.py` validates `auto` mode >=1.5x faster than pinned at bit-exact engine determinism. Smoke run deferred until W6 extension finishes.
- **F — CODE READY.** `training/r14_adaptive_ratio_sweep.py` sweeps ratio in {0, 1.5, 2, 3, 5} at value-head leaf; picks the highest ratio satisfying Wilson lower >= 0.42 and wall-clock cut >= 30%. Deferred.
- **A — CODE READY.** `training/r14_ood_gate.py` runs Gate 1 (fresh seeds 800000+ vs rule-bot) + Gate 2 (MCTS-vs-MCTS against R4). Deferred.

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
