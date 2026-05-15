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
- **Cost when run:** ~340 LOC + ~1 day compute (refined by R14 D-survey 2026-05-11). Breakdown:
  - 60 LOC to `train_ppo.py`: extend `TrajectoryRow` with `visit_distribution`, swap the PPO ratio formula for V-trace clipping (ρ̄ + c̄ args), keep the existing log-ratio clamp.
  - 120 LOC for a new MCTS trajectory parser (analog of `parse_trace_to_trajectories` in `ppo_orchestrator`) that consumes `mctsSelfPlay.ts` rows: groups by `(seed, sideId)`, computes per-step Δpoints, uses `valueTarget` as terminal bootstrap, emits `behavior_logp = log(visitDistribution[selected])` with ε=1e-4 floor.
  - 80 LOC for a separate `r14_mcts_ppo_orchestrator.py` (don't mutate the policy-PPO loop — D is a fallback, copy-paste minimizes blast radius if it fails).
  - 40 LOC for V-trace bootstrap glue (per-step Δpoints + terminal outcome → V_s trace).
  - 40 LOC smoke (2 iters × 3 games, assert ratio_mean ≠ 1.0, no NaN, weights change).
- **Risk register additions:** (i) visit distribution near-zero on rare actions → ε=1e-4 floor and importance-ratio halt at >10; (ii) value-head staleness biases V_s upward → only run from W6-iter-1+ checkpoints; (iii) ratio_mean still clustering at 1.0 after iter-1 → RL is genuinely closed (same diagnosis loop as F1/R3/R5).
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

## Progress (started 2026-05-11; closed 2026-05-14)

**Outcome:** all code-side workstreams landed. **Primary production pick: rollout-leaf MCTS at iter-2, Wilson lower 0.6479** (I.2) — this is the validated max-strength deployment claim, untouched by the side-asymmetry confirmation gate. **Cheap-inference fallback: value-head leaf + adaptive-ratio=1.5 at iter-2, Wilson lower 0.39–0.45 across three independent seed ranges** (F-iter-2 0.443 at 820000+ / A 0.404 at 800000+ / A.footnote 0.394 at 900000+), with **one-of-three runs failing the 0.40 production bar** at 2.56× speedup vs ratio=0. The 0.452 number originally cited from F-iter-2 was the upper end of that empirical range at the F seed range, not a stable point estimate. **Update 2026-05-14 (A.footnote):** the A.footnote gate established this range and refuted A's "side asymmetry is partly seed-clustered" caveat — the player/opponent gap (+0.11–0.30pp Wilson) replicates across all four gates and two independent seed ranges. **Resolution 2026-05-14 (self-directed):** docs re-targeted to lead with rollout-leaf MCTS as the primary recommendation and present value-head leaf + ratio=1.5 as the cheap-inference fallback with the qualified range; see `docs/ai-agent-state/escalations.md` `## Resolved`. Engine determinism (B) closes R-WILD #34. UI plumbing (E) verified end-to-end via headless smoke (decisionMs 200 at 16 sims → ~0.5s at 100 sims with ratio=1.5); only the 20-game in-browser exercise is left. ORT unpinning (G) showed 1.05× speedup at this model size — kept `--ort-threads 1`. D (PPO V-trace) and H (predict-batching) skipped per the plan's decision points.


- **B — DONE.** AsyncLocalStorage via `frontend/.../random.ts` installRngStorageProvider + `backend/.../rngAsyncStore.ts`. `training/r14_determinism_smoke.py` shows 12/12 games bit-exact between --workers 1 and --workers 4 (post-fix); previously divergent. Closes R-WILD #34.
- **I.1 — DONE.** `training/r14_value_crossover_probe.py` measures val_mse + pearson_r between checkpoint value head and rootValue (rollout-CRN K=3 mean) on a held-out selfplay corpus. Crossed iff val_mse <= 1.10 * mse_floor AND pearson_r >= 0.7. mse_floor anchored on the W3 retrain manifest. Wired into `r12_orchestrator.run_iteration` between distill and gate (records `iterations[i].crossover`). Baseline sanity at W6 iter-1 vs its own iter-1 selfplay (in-distribution): val_mse 0.659, pearson 0.535, ratio 1.158, crossed=false — consistent with W8 regression (cheap-leaf selfplay distillation didn't compound because the value head wasn't ready).
- **I.2 — DONE (halted at iter-4).** W6 phase-d extended to iter-2/3/4 in `runs/R13-W6-phase-d/`. iter-2 promoted at Wilson lower **0.6479** (WR 0.7333, n=120) — clears the sprint exit criterion ≥0.60. iter-3 regressed to 0.5783 and iter-4 to 0.5612, triggering the orchestrator's halt-after-2 rule. Crossover probe never crossed: pearson_r plateaued ~0.50 (need ≥0.7) and ratio stayed ≥1.18 (need ≤1.10) across iter-2/3/4. Promoted checkpoint: `runs/R13-W6-phase-d/iter-2/checkpoint.pt`. Per decision plan: I succeeded → cheap-selfplay retry (I.3) is **skipped** because crossover never fired; D remains scoped (commit `bc6db85`) as a fallback but is **redundant** for the strength claim.
- **E — CLOSED (user-verified working; 2026-05-15).** Plumbing landed earlier: visible MainMenuScreen toggle (`0371892`), `/ai/decide` server-side action application + `nextState` return verified headlessly via `training/r14_ai_decide_e2e_smoke.py` (PASS, `runs/R14-ai-decide-e2e-smoke/`), Vite `/ai` proxy added (`68c1c62`). **Closure basis:** the user manually exercised R14.E in the browser at production SHA `bc6db85` (the 96-dim feature schema matching the 96-dim production model `runs/R13-W6-phase-d/iter-2/policy.onnx`) and confirmed BOTH MCTS configs functional end-to-end in the UI, with the rollout-leaf opponent visibly beating the naive rule bot. The webapp opponent default is now set to the production rollout-leaf config (`useAppRuntimeEffects.ts`, gate args byte-identical to `runs/R13-W6-phase-d/iter-2/gate.manifest.json`, Wilson lower 0.6479, ~73% WR vs rule bot, n=120); value-head + adaptiveRatio=1.5 is retained as the documented latency fallback. **Honest scope:** this is a qualitative user-acceptance closure plus prior headless gate evidence — the formal numeric exit criterion (median decisionMs <3s, fallback <5%) was NOT captured (no HAR / numeric dataset). Rollout-leaf is the deliberate quality-over-latency product pick and is NOT expected to meet a <3s median in deep states (that is the designed reason value-head exists as the latency fallback). **Open deployment caveat:** the verified-working stack is the 96-dim production code state (`bc6db85`) + the 96-dim model; `feat/ai` HEAD's feature extractor is 110-dim (R7.b.2) and cannot serve the 96-dim production model (`Got: 110 Expected: 96`), and there is no 110-dim production-grade search model (the 110-dim F1 line closed ~0.30). Deployment must ship from the 96-dim code state, pin serving to the 96-dim schema, or produce a 110-dim production-grade model (does not exist). See `docs/ai-research/progress/r15.md` § R14.E closure.
- **G — DONE (NEGATIVE; 2026-05-14).** Smoke ran cleanly post-A: pinned 13.4s vs auto 12.8s → **speedup 1.05×, well below the 1.5× target**. Determinism: 12/12 games winner+turnNumber bit-exact, 0 mismatches (B's AsyncLocalStorage holds). Verdict consistent with the H-section caveat: at this model size, /predict per-call overhead dominates, so ORT thread parallelism gives essentially no throughput win. Decision: **keep `--ort-threads 1` as the default** (no regression vs R13.W1 baseline, no benefit from changing) and **skip H** entirely (G already showed the per-call-overhead ceiling; predict-batching's upside is even more bounded by it). Output: `runs/R14-ort-throughput-smoke/`.
- **F — DONE (cheap-inference fallback pick = ratio 1.5 on iter-2; 2026-05-14).** Ran twice. First on iter-1 (the original sprint target): stated FAIL (0.42 Wilson unreachable at fresh seeds), but ratio=2.0 Pareto-dominated baseline at 2.19× speedup. Re-ran on iter-2 (the I.2 production candidate) — **status PASS at the F seed range**, all 5 ratios clear 0.42 Wilson on seeds 820000+. Pareto frontier on iter-2: ratio=1.5 (Wilson 0.452 at the F seed range, **2.56× speedup**) and ratio=5.0 (Wilson 0.472, 1.53× speedup); 2.0 strictly dominated. **Cheap-inference fallback pick: ratio=1.5** for W5/E default — the 0.452 reading here is the R14.A.footnote-resolved upper end of the empirical range; across three independent seed ranges the cheap-inference number is **Wilson lower 0.39–0.45 with 1-of-3 below the 0.40 production bar** (see A.footnote below). Decision time at 100 sims is ~0.5s (vs R13.W6 iter-1's ~1.25s/decision). The primary production claim is rollout-leaf MCTS @ iter-2 (Wilson 0.6479, I.2); this F result is the cheap-inference fallback with the qualified range. Output: `runs/R14-adaptive-sweep/` (iter-1) and `runs/R14-adaptive-sweep-iter2/` (iter-2).
- **A — DONE (PASS, run on iter-2 instead of iter-1).** Re-targeted from iter-1 to iter-2 since iter-2 is the newly-promoted production candidate (also gives iter-2's first value-head-leaf number). `training/r14_ood_gate.py` Gate 1 (seeds 800000+ vs rule-bot, value-head leaf, n=100): WR 0.50, Wilson95 [**0.404**, 0.596] — passes the 0.40 floor marginally. Gate 2 (seeds 850000+, iter-2 vs R4, MCTS-vs-MCTS at value-head leaf, n=100): WR 0.61, Wilson95 [**0.512**, 0.700] — passes by 11pp. Gap (gate1 − gate2) = −0.108: iter-2 is *stronger* against R4 MCTS than against rule-bot. Conclusion: iter-2 + value-head-leaf is OOD-robust and represents a genuine model improvement, not rule-bot artifact. **Caveat:** severe side asymmetry persists (gate 1 player Wilson 0.31 vs opponent 0.42; gate 2 player Wilson 0.33 vs opponent 0.63). Implication: iter-2's value-head-leaf strength is real on average but uneven across sides — the production claim should note the side gap. Also: iter-2's value-head-leaf Wilson 0.404 is statistically indistinguishable from iter-1's reported 0.452, so iter-2's +18pp rollout-leaf gain does not transfer to value-head-leaf inference. Output: `runs/R14-ood-gate-iter2/summary.json`. Fixed a path-resolution bug in `r14_ood_gate.py` along the way (relative `--out-dir` was getting resolved against `backend/` workspace cwd).
- **A.footnote — third-seed-range confirmation gate, gate1 FAIL @ 0.394 (2026-05-14).** Re-ran `training/r14_ood_gate.py` at independent seeds 900000+ / 950000+ (same MCTS config: value-head leaf, 100 sims, c_puct=1.5, 4 workers, n=100/gate; manifest matches the production deployment config). **gate1 FAIL** (vs rule-bot, seeds 900000+): WR 0.49, Wilson95 [**0.394**, 0.587] — lands 0.6pp **below** the 0.40 production bar. **gate2 PASS** (vs R4 MCTS, seeds 950000+): WR 0.58, Wilson95 [**0.482**, 0.672]. gap_gate1_minus_gate2 = −0.088 (gate1 weaker than gate2, *opposite* of A's original gates 1/2 gap). The R14 production headline "Wilson 0.452 at adaptive-ratio=1.5" (line 121 above) is **NOT confirmed at a third independent seed range** — gate1 here at 0.394 lands below 0.40, where A's original gate1 at seeds 800000+ landed at 0.404. The 0.452 number was on the F-iter-2 production seed range; A's 0.404 was a marginal pass; this gate's 0.394 is a marginal fail. Per-side breakdown across both gates and both seed ranges:

| Seed range | Gate | Player WR | Player Wilson lower | Opponent WR | Opponent Wilson lower | Gap (Opp − Plr) |
| --- | --- | --- | --- | --- | --- | --- |
| 800000+ (A original) | 1 (vs rule-bot) | 0.44 | 0.312 | 0.56 | 0.423 | +0.111 |
| 850000+ (A original) | 2 (vs R4 MCTS) | 0.46 | 0.330 | 0.76 | 0.626 | +0.296 |
| 900000+ (A.footnote) | 1 (vs rule-bot) | 0.42 | **0.294** | 0.56 | **0.423** | +0.129 |
| 950000+ (A.footnote) | 2 (vs R4 MCTS) | 0.50 | **0.366** | 0.66 | **0.522** | +0.156 |

  - **Side asymmetry is real, not seed-clustered.** Same shape (player weaker than opponent by 0.11–0.30pp Wilson), four gates, two independent seed ranges, two opponents. The R14.A note "side-asymmetry is partly seed-clustered" (footnote on line 113 above) is **refuted**: at fresh seeds the gap persists and on gate2 it widens.
  - **The R14 production claim should be re-cited.** "Wilson 0.452 at value-head leaf + ratio=1.5" was a single-seed-range datapoint. Across F-iter-2 (seeds 820000+ Wilson 0.443) + A (seeds 800000+ Wilson 0.404) + A.footnote (seeds 900000+ Wilson **0.394**), the empirical range is **Wilson lower 0.39–0.45** with one-of-three failing the 0.40 bar. Honest framing for downstream docs: "Wilson lower 0.39–0.45 across three independent seed ranges; production headline 0.452 was the upper end of that range, not a stable point estimate."
  - **What this does NOT close.** This gate tested *only* the value-head leaf + ratio=1.5 production pick. The R14 strength headline — **rollout-leaf MCTS at iter-2 Wilson 0.6479** (I.2, line 113) — was NOT tested here and remains the unchanged max-strength deployment claim. The FAIL applies to the cheap-inference deployment config only.
  - **gate1 < gate2 at fresh seeds is itself interesting.** A's original run had gate1 > gate2 in the *opposite* direction (gate1−gate2 = −0.108 because gate2 was stronger, but at the seed range gate1 still cleared 0.40); this run shows gate1−gate2 = −0.088 with gate1 below the bar and gate2 strongly above. The model is closer to "matching its own R4 baseline head-to-head" than to "beating rule-bot at the production claim level" on fresh seeds, suggesting rule-bot behavior at seeds 900000+ differs meaningfully from the R14.A / F seed ranges. Not a separate diagnostic this slot.
  - **Open research-stance decision (escalated to harness).** Three options for the human: (a) re-target the production claim to a CI phrasing (Wilson lower 0.39–0.45 across seed ranges); (b) re-run R14.A at larger n (e.g. n=500/gate) to widen empirical confidence on the 0.452 number; (c) accept the cheap-inference FAIL and route deployment attention to the rollout-leaf iter-2 production pick (Wilson 0.6479, untouched by this gate). Output: `runs/R14-A-side-asymmetry-seed900000/summary.json` + per-gate manifests. Wall-clock 9m 38s total (gate1 275.7s, gate2 302.7s).

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
