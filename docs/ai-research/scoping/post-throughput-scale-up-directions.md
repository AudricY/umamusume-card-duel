# Post-Throughput Scale-Up: Research-Direction Relook

- **Date:** 2026-05-21
- **Status:** SCOPING — Slice 2 of `throughput-optimization-spike` passed at 4.18× wall reduction; user-queued relook on what becomes economic. Re-prioritization landed; no new code lines yet. Predecessor: `docs/ai-research/scoping/throughput-optimization-spike.md` Slice 2.

## 1. Trigger / context

Slice 2 acceptance (`runs/throughput-spike-slice2-acceptance/`) of the in-process ORT Rust path completed 10 iters of the R110-W6-repro-faithful recipe (v3.0, W6-fix-OFF, seeded from R110 iter-2 0.6042 ckpt) in **7.99 min wall vs 33.4 min baseline = 4.18× speedup**. Trajectory: 0.4773 / 0.5022 / 0.5273 / 0.4939 / 0.5189 / 0.4856 / 0.5442 / 0.5358 / 0.5189 / **0.5527** (iter-9 promoted). All 10 finite Wilson-lowers, all promote=True, no crashes.

Per-iter mean breakdown (~48s): selfplay ~16s + distill ~4s + gate ~28s. Gate dominates (n=120 side-balanced). Speedup gate clears both the ≥1.65× PASS bar and the ≥3× scale-up trigger.

Canonical evidence: `runs/throughput-spike-slice2-acceptance/loop/orchestrator-state.json`, `launch.log`.

## 2. Existing queue candidates to re-prioritize (newly economic)

These were already pre-staged with status `scoped-blocked-on-rust-orchestrator-wiring`. They become tractable today.

- **`high-sim-mcts-regime-probe` (P2 → P1):** 50,000+ sims/decision now affordable as a standard experiment (~1 day at 4.18× speedup). Question reframed: does the rule-bot-vs-MCTS gap close monotonically at high sims? If yes, deep-program's kill criterion (chunks 5g/5h/5i) is throughput-bound, not fundamental.
- **`tight-gate-reverdict-program` (P2 → P1):** n=10,000 evals collapse noisy MARGINAL verdicts to definitive. Candidate list: R110-W6-repro iter-2 (0.6042 MARGINAL), R16-P1 v3.1 (NO-GO at Δ=−0.0087 within noise), training-data-deep-program 5g/5h/5i argmax-flip kills around the 30% threshold. At Slice-2 gate cost (~28s × n=120), n=10,000 is ~2,300s = ~40 min wall per re-verdict — feasible as a batched program.
- **`per-game-pfsp-league-retry` (P2 unchanged):** Cross-iter opponent pool selfplay (separate scoping doc at `cross-iter-opponent-pool-selfplay.md`) is the natural first instantiation. Thousand-game tournaments now cheap. Already has acceptance criteria.

## 3. NEW candidates to surface

Each with one-line acceptance criteria. Wallclock estimates anchor on the Slice-2 ~48s/iter mean.

- **Multi-knob W6-fix HP sweep:** 27-cell grid sampled on the diagonal. Knobs: `--kl-anchor-weight` ∈ {0.02, 0.05, 0.10}, `--w6-replay-window` ∈ {2, 3, 5}, `--w6-replay-old-fraction` ∈ {0.20, 0.40, 0.60}. 10 cells × 10 iters each = ~80 min wall (vs ~330 min pre-Slice-2). **Acceptance:** any cell beats extended-2x-games iter-3 wl=0.5538 lineage best by ≥+0.022 (one-Wilson width).
- **Larger model architecture sweep:** currently `--hidden-dim 64 --depth 2`. Try `--hidden-dim 128 --depth 3` and `--hidden-dim 256 --depth 4`. Distill currently ~4s/iter; bigger model may push distill to ~30s. Selfplay/gate stay MCTS-bound, not inference-bound (per Slice 1 inference parity: 106µs/call dominated by node expansion). **Acceptance:** any larger arch beats 0.5538 lineage best by ≥+0.022 AND retains crossover behavior (R14 ratio ≤ 1.10, pearson ≥ 0.7) within first 3 iters.
- **Longer-horizon training (50+ iter recipes):** R110/C8 lineage hits the iter-9 0.5527 plateau under 10 iters. Does another 40 iters push above the 0.6042 R110-W6-repro MARGINAL ceiling, plateau flat, or rot back? Previously prohibitive (~165 min); now ~40 min wall. **Acceptance:** iter-50 wl strictly above 0.6042 (push), or trajectory variance dampens below max |wl[i+1]−wl[i]| < 0.03 in last 20 iters (plateau detection).
- **Side-conditioned eval as default:** `r12-selfplay-gate-throughput.md` mentions side-conditioned eval as an open scoping line; r15 documents player/opponent asymmetry. With cheap evals, "n=10,000 side-balanced, separately reported per modelSide" becomes the default protocol, not a luxury. **Acceptance:** plumb `--report-per-side` into `sim-eval-gate` manifest; rerun the lineage-best ckpts (R110 iter-2, 2x-games iter-3, Slice-2 iter-9) at n=10,000/side and report. Falsifies/confirms whether observed wobble is side-asymmetry artifact.
- **High-sim regime gate on production ckpt:** re-evaluate R110-W6-repro iter-2 (0.6042) at 800 / 4,000 / 20,000 / 50,000 MCTS sims. **Acceptance:** report Wilson-lower curve vs sim count; if monotone-up at 50,000 sims, the production gap to a hypothetical ceiling is throughput-bound. Cross-references `high-sim-mcts-regime-probe` but on the production reference, not the kill candidates.

## 4. Re-prioritization recommendation

Three forward lines, ordered by falsification value × cost. Reasoning grounded in: each must either close a long-open question or open a strength upside that's currently invisible.

**RECOMMENDED PRIMARY (next-up after Slice 3):** `tight-gate-reverdict-program` re-verdict R110-W6-repro iter-2 + R16-P1 v3.1 + 5g/5h/5i flips at n=10,000. *Why:* the cheapest and most decisive — collapses three open MARGINAL/within-noise verdicts to definitive at known cost (~2 hours total). Either promotes a currently-pinned 96-d to something better, or closes the door on the three programs cleanly. Has the highest probability of changing the active backlog state per dollar of compute.

**RECOMMENDED SECONDARY:** Multi-knob W6-fix HP sweep on the 10-cell diagonal. *Why:* the wobble + iter-0 strength gap from C8-W6FIX-ON (the iter-0 −6.0pp dampening that traded for the calibration crossing) is the most recent open lever in the active line. ~80 min wall to either find a knob combo that recovers iter-0 strength while preserving the R14 crossover, or to definitively kill the W6-fix-on-default direction.

**RECOMMENDED TERTIARY:** `high-sim-mcts-regime-probe` against the kill candidates (5g per-kind heads). *Why:* it's the only experiment that can reopen a closed line (training-data-deep-program done-negative). High variance bet, but the new throughput regime makes the cost ~1 day of orchestrator time, not 2 weeks.

**DEPRIORITIZED:** Larger-model architecture sweep and longer-horizon 50+ iter recipes. *Why:* both are "spend more compute, hope for lift" with no specific mechanism failing today. Better to gate them on a positive signal from the three above, otherwise we're just enlarging the search space without a hypothesis.

**HELD AS DEFAULT BACKGROUND CHANGE (not a forward line):** side-conditioned eval. Plumb it into `sim-eval-gate` regardless of which experiment runs next — n=10,000 side-balanced becomes the default report shape going forward. Cost: half a day plumbing; recurring benefit on every subsequent verdict.

## 5. What does NOT change

- **`value-head-leaf-recipe-axis` follow-up still gates on Slice 3.** The user-queued vhleaf loop (continue from 2x-games iter-3 ckpt under `--mcts-leaf value-head`) requires v3.2 per-Uma slot tokens in the Rust featurizer. Slice 2's Rust path is v3.0-only. If the user wants vhleaf on the fast path, Slice 3 (v3.2 featurizer port) lands first. Alternative: run vhleaf on the slower HTTP path now (~33 min/10 iters) and ride Slice 3 for any follow-up.
- **`per-game-pfsp-league-retry` scoping unchanged.** Its acceptance + falsification criteria from `cross-iter-opponent-pool-selfplay.md` stand as-is; throughput just makes the wallclock easier.
- **96-d production pin unchanged** until any of the three re-verdicts above flip MARGINAL → GO.
- **`mcts-relabel-rng-bleed` (P3 deferred) stays deferred.** No sim-count A/B in the recommended primary/secondary/tertiary lines triggers it.
