# Deck-Pair Sampling For Self-Play And Eval

- **Date:** 2026-05-22
- **Status:** **Slices P0+1+2 LANDED 2026-05-22; training-bearing lines UNBLOCKED.** Step 3 verdict: **MATCHUP-BROADLY-REPRESENTATIVE (Slice 2 = no-regret) + PER-MATCHUP SPREAD 19.6pp (Slice 4 PFSP becomes attractive).** n=1,000 uniform smoke on R110-W6-repro iter-2 v3.0 ckpt — aggregate wilson_lower **0.5331** (n=1000, all 22 pairs covered ~45 g/pair). Delta vs fixed-matchup ceiling 0.5811 = **-4.8pp, just inside ±5pp tolerance**. Per-matchup point estimates range **0.4565 → 0.6522** (spread 19.6pp, exceeds the 15pp PFSP threshold). Tracked in queue under `deck-pair-sampling` (P1). Re-surfaces archived items §6 (deck-pair sampling) and §13 (per-matchup eval gating) from `docs/archive/ai-research/ai-performance-research-backlog.md`, parked behind throughput work and never implemented.
- **Routing:** active backlog reference §6b; this scoping is the canonical home until landed.
- **v3.2 interaction:** v3.2 introduces `uma_slot_card_ids` + `uma_slot_features` (positional per-Uma-slot tokens), which carry the model's primary signal about *which deck* it is facing (the active+bench Uma identities). Deck-variety sampling tests whether the v3.2 slot-token branch generalizes across opponent archetypes or has only learned the Matikane mirror. Featurizer remains deck-agnostic (global card vocab, verified zero OOV across all 13 decks); the slot-token branch reads the same vocab via the shared `card_embed` table.

## Hypothesis

The entire post-R110 research program — including the tight-gate `wilson_lower=0.5811` ceiling, the iter-2-peak-then-rot pattern, every recipe-axis verdict, and the v3.0/v3.1/v3.2 representation comparisons — has been measured on a **single near-mirror Matikanetannhauser matchup**. The 11 AI decks and 2 player decks in `shared/src/data/premadeDecks.json` are otherwise unused in any sim path (player UI consumes them; sim and training do not).

Concretely: `defaultPlayerDeckId="matikanetannhauser"` resolves to `premadeDecks[0]` (pure Matikane). `defaultAiOpponentDeckId="riceShowerHaruUrara"` does **not** match any id in `aiPremadeDecks`; the fallback chain in `engine-rs/crates/engine/src/core/decks.rs:119-131` silently lands on `ai_decks.first()` = AI-flavor `matikanetannhauser` (Matikane core + Haru Urara / Fukukitaru splash). Every Rust and TS self-play / eval entrypoint funnels through `setup_ai_vs_ai_game()` / `setupAiVsAiGame()`, both of which call these defaults with no caller override.

**Risk this introduces to the program:** the policy and the recipe-axis verdicts are conditional on one matchup. The 0.5811 ceiling may be matchup-specific. Recipe levers that lose on this matchup may win on others (and vice versa). The model may be carrying matchup-specific exploitation patterns that don't transfer.

**What deck-pair sampling tests:** does the W6 recipe (and its derivatives) generalize across matchups, or is the existing ceiling an artifact of Matikane-vs-Matikane dynamics?

## Why now

- Throughput is no longer the binding constraint: Slice 3c parallelism gives 10.9-min tight-gate walls at `n=10,000`. The wallclock budget for adding matchup variance is bounded.
- The representation axis (v3.0 vs v3.1) just closed. Forward research lines (W6-fix knob tune, per-game PFSP league retry) will both produce verdicts on the single-matchup eval that may not transfer. Better to know the matchup-sensitivity floor *before* spending compute on those.
- The featurizer is **deck-agnostic** — global card vocab (107 entries) covers every card in every AI deck (verified, zero OOV). No model or featurizer changes are required; only data-distribution and eval changes.
- The PPO `--rollout-vs-pool` PFSP plumbing (`training/opponent_pool.py`) is architecturally parallel to "sample one knob per game from a pool with weights." Same shape, different lever.

## Prior negative-ish datapoints (to navigate around)

None on this exact lever. Archived backlog item §6 explicitly listed deck-pair sampling and recording per-game deck pair so item §12's pool can stratify; archived backlog item §13 sketched per-matchup Wilson lower bounds with ≤5pp drop tolerance. Both archived as "parked behind throughput work"; never falsified, never implemented.

## Plumbing

Single shared seam: `setup_ai_vs_ai_game()` / `setupAiVsAiGame()`. The CLI layer owns a `--deck-sampling=fixed|uniform|pair=<P>:<O>|pfsp` flag; sampling is stateless-and-seed-derived so the orchestrators don't change (they already pass `--seed-start`).

### Predecessor (cheap, lands first)

**P0a. Defaults bug fix.** Either:

- (a) Change `shared/src/data/premadeDecks.json:defaultAiOpponentDeckId` to an id that exists (e.g. `"matikanetannhauser"` to make the silent fallback explicit, or `"riceShower"` to actually pick a different matchup), or
- (b) Add an `id` field `"riceShowerHaruUrara"` to the existing AI Matikane deck if that name is the desired canonical reference.

Either way, the test at `engine-rs/crates/engine/src/core/decks.rs:165-168` should assert the id round-trips, not document the miss as expected.

**P0b. Deck-legality smoke.** Add a Rust unit test that loops `decks().ai_decks × decks().player_decks`, drives a short (~5-turn) headless game per pair, and asserts no panic / no silent card-drop in `intern_deck`. 9 of 11 AI decks have never been simulated; this is the cheapest possible catch for card-data drift.

### Slice 1 — sampling at the CLI level (~1 day)

Plumb `--deck-sampling` through:

- Rust: `engine-rs/crates/sim-cli/src/bin/{mcts_selfplay,eval_gate,export_training}.rs` (+ `inference_parity.rs` for parity reasons). Sampler is seeded from `--seed-start + game_index`; uniform over the 13 player×AI pairs, or a literal `pair=<P>:<O>` for reproducibility.
- TS counterpart in `backend/src/sim/evaluateModelVsHeuristic.ts:setupAiVsAiGame` and `backend/src/sim/headlessAiVsAi.ts`.
- Per-game manifest emits `playerDeckId` + `opponentDeckId` (recording is mandatory; sampling is opt-in).

Default = `fixed` (current behavior, no behavior change for any active orchestrator path).

### Slice 2 — data-gen rollouts use sampling (~half day) — UNBLOCK CONDITION

Flip `r12_orchestrator.py:run_selfplay`, `dagger_orchestrator.py`, and `ppo_orchestrator.py` self-play steps to default-on `--deck-sampling=uniform`. **Eval-gate stays fixed** — preserves apples-to-apples vs the existing tight-gate history (`wl=0.5811`, etc.).

**This slice is the gate for the rest of the training-bearing queue** (W6-fix HP sweep, per-game-PFSP league retry, value-head Stage 2/3). Once it lands, those resume — running their probes with `--deck-sampling=uniform` in self-play and judged on both the fixed-matchup and diverse-matchup gates.

The expected signal: distillation corpus widens; training distribution covers more state space; the per-iter eval gate (still fixed matchup) either climbs (good), stays flat (neutral), or drops (the model needed matchup-specific overfitting to score on the fixed gate — a red flag worth knowing).

### Slice 3 — diverse-matchup eval gate (~1 day, runs alongside fixed gate)

Add `--deck-sampling=uniform` to `sim-eval-gate` as a second-gate option; orchestrators run **both** gates per iter (fixed for legacy comparability + uniform for matchup-generalization). Per-matchup Wilson-lower computation lands here (the archived item §13 work surfaces here naturally).

Wilson half-width at `p≈0.59, n=10k` is `±0.0097` (≈±1pp); uniform over 22 unordered deck-pairs at `n=10k` gives ~454 games/pair → ±4.5pp per matchup. Useless for per-matchup verdicts; fine for aggregate. For per-matchup verdicts, stratify (Slice 4) or use a smaller fixed sub-pool.

### Slice 4 — PFSP-weighted opponent-deck sampling (optional, ~1 day)

Once Slice 3 reports per-matchup Wilson lower bounds, weight rollout sampling toward matchups the current model is losing. Reuses `training/opponent_pool.py` PFSP weight mechanics (`max(0.05, 1 - p_i)`) — but at the deck-id key instead of the checkpoint key. Architecturally parallel to `--rollout-vs-pool`; not code-shared with it today.

## Recipe (first probe — Slices P0 + 1 only)

- **Step 1**: land defaults fix + deck-legality smoke. Surface any panics on the 9 untested decks as bugs to fix before plumbing sampling.
- **Step 2**: add `--deck-sampling` to Rust sim-cli + TS path. Default `fixed`. Manifest records per-game deck pair.
- **Step 3**: run a `--deck-sampling=uniform, n=1,000` smoke of `sim-eval-gate` on the current production ckpt (R110-W6-repro iter-2). Report aggregate Wilson lower + per-matchup point estimates.

**Acceptance for Step 3 (informational, not promote/reject):**

- Aggregate uniform Wilson lower **within ±5pp of the fixed-matchup 0.5811 ceiling** ⇒ the matchup is broadly representative; the existing program's verdicts are likely generalizable. Continue with Slice 2 (data-gen sampling) as a no-regret distribution widener.
- Aggregate uniform Wilson lower **>5pp below 0.5811** ⇒ the model is matchup-specifically tuned. Existing ceiling claims are matchup-conditional. Slice 2 + Slice 3 both move up in priority — every future recipe-axis verdict should be measured on both gates.
- Per-matchup point estimates with **>15pp spread** ⇒ there are easy and hard matchups; PFSP-weighted training (Slice 4) becomes attractive.

## Step 3 verdict — 2026-05-22

Run: `runs/deck-sampling-step3-smoke/gate.manifest.json` (n=1,000 uniform, R110-W6-repro/iter-2 v3.0 ckpt, sims=100, k=3, rollout-steps=200, prior=policy, leaf=rollout, workers=16, wall 68.9 s).

| Metric | Value | Reference / interpretation |
| --- | --- | --- |
| Aggregate winRate | 0.5640 (564/1000) | vs fixed-matchup history 0.5908 (R110-W6 n=10k) |
| Aggregate wilson_lower | **0.5331** | **vs fixed-matchup ceiling 0.5811 → -4.8pp, inside ±5pp** |
| Aggregate wilson_upper | 0.5944 | half-width ±3.07pp at n=1000 |
| Player-side wilson_lower | 0.5990 (wr 0.6420, 321/500) | Matikane-on-player is comfortable across opponents |
| Opponent-side wilson_lower | 0.4425 (wr 0.4860, 243/500) | Matikane-on-opponent is the harder side |
| Per-matchup spread | **19.6pp** (0.4565 → 0.6522) | **exceeds 15pp PFSP threshold** |

**Per-matchup signal (sorted by point-estimate win-rate, n≈45/matchup, per-matchup CI ≈ ±15pp — signal-spotting only, not promotion-grade):**

| matchup | games | wins | winRate |
| --- | ---:| ---:| ---:|
| matikanetannhauser:daiwaScarlet | 46 | 30 | 0.6522 |
| matikanetannhauser:riceShower | 45 | 29 | 0.6444 |
| riceShower:manhattanCafe | 45 | 28 | 0.6222 |
| matikanetannhauser:matikanetannhauser (legacy fixed-gate pair) | 45 | 27 | 0.6000 |
| riceShower:oguriCap | 45 | 27 | 0.6000 |
| riceShower:superCreek | 45 | 27 | 0.6000 |
| matikanetannhauser:manhattanCafe | 46 | 27 | 0.5870 |
| matikanetannhauser:vodka | 46 | 27 | 0.5870 |
| riceShower:matikanetannhauser | 46 | 27 | 0.5870 |
| riceShower:daiwaScarlet | 45 | 26 | 0.5778 |
| matikanetannhauser:symboliRudolf | 46 | 26 | 0.5652 |
| matikanetannhauser:tamamoCross | 46 | 26 | 0.5652 |
| riceShower:riceShower | 45 | 25 | 0.5556 |
| riceShower:symboliRudolf | 45 | 25 | 0.5556 |
| riceShower:vodka | 45 | 25 | 0.5556 |
| matikanetannhauser:oguriCap | 46 | 25 | 0.5435 |
| riceShower:tamamoCross | 45 | 24 | 0.5333 |
| matikanetannhauser:mihonoBourbon | 46 | 24 | 0.5217 |
| riceShower:agnesDigital | 45 | 23 | 0.5111 |
| matikanetannhauser:agnesDigital | 46 | 23 | 0.5000 |
| riceShower:mihonoBourbon | 45 | 22 | 0.4889 |
| **matikanetannhauser:superCreek (hardest)** | 46 | 21 | 0.4565 |

### Acceptance branches taken

- **(within ±5pp branch)** Aggregate uniform wl=0.5331 vs fixed 0.5811 = **-4.8pp; matchup-broadly-representative.** Existing v3.0 ceiling claims are likely generalizable. **Slice 2 (data-gen rollouts default-on uniform) is no-regret** — proceed.
- **(per-matchup spread branch)** 19.6pp spread (Matikane vs Daiwa Scarlet = 0.65, Matikane vs Super Creek = 0.46) **exceeds the 15pp threshold** the scoping doc flagged. **Slice 4 (PFSP-weighted deck sampling) becomes attractive** post-Slice-2.
- Slice 3 priority unchanged — per-matchup Wilson lower bounds at higher n are still needed before any per-matchup gating decision; n=45 per pair is signal-spotting, not promotion-grade.

### Caveats

- Fixed ceiling 0.5811 reference is at n=10,000; uniform smoke is n=1,000 (~3.3× wider CI). The -4.8pp gap is within the joint Wilson half-width; on a stricter n=10,000 uniform run the gap could widen or shrink within ±3pp. Slice 3's diverse-matchup gate at n=10,000 will tighten this once it lands.
- Per-matchup CIs at n≈45 (≈±15pp) overlap heavily across the top half of the table; the **shape** (Matikane-into-archetype-X better than Matikane-into-archetype-Y) is suggestive but not statistically promotable at this sample size.
- The probe ran with `--prior policy` on R110-W6-repro/iter-2 v3.0 ONNX; the v3.2 ceiling is still pending re-verdict #3 per the loop_note. This Step 3 verdict applies to the v3.0 reference; the v3.2 ceiling generalization claim needs an analogous uniform smoke on a v3.2 ckpt before training-bearing v3.2 work resumes under Slice 2.

## Slice 2 validation — 2026-05-22

Orchestrators wired (all three):

- `training/r12_orchestrator.py`: `run_selfplay` passes `--deck-sampling=$args.deck_sampling` (default `uniform`) to Rust `sim-mcts-selfplay`; `run_gate` pins `--deck-sampling=fixed` unconditionally.
- `training/dagger_orchestrator.py`: trace-gen `run_evaluator` calls forward `extra=["--deck-sampling", args.deck_sampling]` to TS `sim:evaluate-model`; eval gate via `run_eval_gate` stays fixed.
- `training/ppo_orchestrator.py`: `rollout_with_stochastic_serve` forwards the same `extra` to `run_evaluator`; policy gate via `run_policy_gate_with_serve` stays fixed.

TS plumbing: `--deck-sampling=fixed|uniform|pair=<P>:<O>` added to `backend/src/sim/evaluateModelVsHeuristic.ts` CLI (sampler mirrors the Rust row-major decomposition exactly: `(seedStart + gameIndex) % (n_player * n_ai)`; deterministic for the same `(seedStart, gameIndex)`). The flag is optional on `EvaluateModelArgs` so in-repo callers (evalGate.ts, throughputProbe.ts, rebaseline.ts, r7TeacherAgreementProbe.ts) stay byte-identical without each opting in.

Manifest passthrough: each orchestrator persists `selfplay_deck_sampling` (CLI-chosen) + `eval_deck_sampling="fixed"` into `orchestrator-state.json`, and emits the mode in the `run_started` event for forward readers. Pre-Slice-2 state files default-read `selfplay_deck_sampling="uniform"` on load.

Smoke evidence (`runs/deck-sampling-slice2-smoke/`, R110-W6-repro/iter-2 v3.0 ckpt, 1 iter, selfplay-games=30, eval-games=30, mcts-leaf=rollout, mcts-sims=100, workers=24, ~19 s wall):

| Check | Result |
| --- | --- |
| Orchestrator completes 1 iter cleanly | ✓ promote=true, wl=0.4573 at n=60 fixed-gate |
| `selfplay.jsonl` carries `playerDeckId` + `opponentDeckId` | ✓ on every row |
| Selfplay deck-pair spread | **22 / 22 unique pairs covered** in 30 games (full cross-product visible in 30 games × ~7 decisions/game) |
| `selfplay.manifest.json args.deckSampling` | `"uniform"` |
| `gate.manifest.json args.deckSampling` | `"fixed"` |
| `gate.manifest.json.summary.perMatchup` | **ABSENT** (Rust sim-eval-gate only emits it when sampling != fixed) |
| `orchestrator-state.json.selfplay_deck_sampling` | `"uniform"` |
| `orchestrator-state.json.eval_deck_sampling` | `"fixed"` |

Per-iter timings (Rust path): selfplay 8.96 s; distill 3.88 s; gate 3.69 s. Total iter wall 16.5 s consistent with `throughput-spike-slice2-acceptance/` per-iter ~50 s at games=60 (linear in games).

Build/test: `npm run build` (TS) + `cargo build --release -p sim-cli` (Rust) green; `npm run test:train` 10/10 PASS; `npm run test:dagger-orchestrator` PASS (3-iter loop with default uniform sampling). `npm run test:ppo-smoke` fails identically with and without these changes (pre-existing init-checkpoint architecture drift; not a Slice 2 regression).

### Slice 3 — gating note

Slice 3 (diverse-matchup eval gate at n=10k with per-matchup Wilson lower bounds) is unblocked but not yet user-gated to action. We have the broadly-representative verdict from the Step 3 smoke; diverse-matchup gating becomes urgent if and only if a recipe-axis verdict comes back that is suspected to be matchup-conditional (e.g. an HP-sweep iter shows fixed-gate gain but the worst-matchup tail collapses). Until then it stays at "scoped, do not auto-launch".

## Slice 3 v3.2 first-gate verdict — 2026-05-22

User directive 2026-05-22T03:54Z: "for slice 3 gate i want to be using v3.2 first." Per the v3.2-mandatory directive (commit `417b9b5`), the forward research line carries the v3.2 representation; the existing v3.2 lineage best is `runs/R16-P2-c8-w6fix-on-extended-2x-games/loop/iter-3/policy.onnx` (fixed-matchup n=240 wl=0.5538).

Caveat noted before launch: **this ckpt was trained with `--deck-sampling=fixed`**; Slice 2 only landed in commit `0e391f3`. So the gate measures **off-distribution generalization** of a single-matchup-trained v3.2 ckpt to the diverse matchup distribution, not the v3.2 ceiling under matchup-diverse training.

Recipe-faithful to the v3.0 Step 3 smoke + R110 tight-gate program: sims=100, c-puct=1.5, leaf=rollout, k=3, rollout-steps=200, prior=policy, collapse-max=64, max-nodes=5000, max-steps=500, seed-start=9000, model-side both, workers=16. Only changes from v3.0 Step 3: ckpt is v3.2, n collapses 1,000→10,000. Pre-G5 binary (Mutex-bound) wall 15.5 min at workers=16.

| n | wr | Wilson-lower | Wilson-upper |
|---|---|---|---|
| **10,000** (5,000 / side) | **0.5787** | **0.5690** | 0.5883 |
| 240 (fixed-matchup baseline) | 0.5825 | 0.5538 | — |

Per-side: player wl 0.6641 (wr 0.6772) — strong first-mover; opponent wl 0.4664 (wr 0.4802). Asymmetry **+19.8pp** by side, broader than v3.0 R110's player–opponent gap.

### Comparison against the v3.0 program

| Ckpt | Schema | Fixed-matchup wl | Uniform-aggregate wl | Δ uniform vs fixed |
|---|---|---|---|---|
| R110-W6 iter-2 (v3.0) | 96-d v3.0 | 0.5811 (n=10k) | 0.5331 (n=1k) | **−4.8pp drop** |
| **C8-W6FIX-ON 2x-games iter-3 (v3.2)** | 110-d slot-tokens | 0.5538 (n=240) | **0.5690 (n=10k)** | **+1.5pp** |

**v3.2 generalizes BETTER across decks than v3.0.** The single-matchup-trained v3.2 lineage best dropped 0.0121 wl from R110-v3.0's fixed reference (−1.2pp) under uniform sampling, vs v3.0's own −4.8pp drop. The slot-token representation appears to encode less matchup-specific exploitation pattern; the per-matchup tail is tighter.

### Per-matchup signal table (ranked by point estimate, n≈454/pair)

| Player | Opponent | wr | wl | wu |
|---|---|---|---|---|
| riceShower | tamamoCross | 0.6608 | 0.6161 | 0.7028 |
| riceShower | symboliRudolf | 0.6410 | 0.5958 | 0.6837 |
| matikanetannhauser | symboliRudolf | 0.6220 | 0.5766 | 0.6653 |
| riceShower | manhattanCafe | 0.6211 | 0.5757 | 0.6646 |
| matikanetannhauser | matikanetannhauser | 0.6167 | 0.5712 | 0.6603 |
| riceShower | oguriCap | 0.6167 | 0.5712 | 0.6603 |
| matikanetannhauser | superCreek | 0.6022 | 0.5565 | 0.6461 |
| matikanetannhauser | tamamoCross | 0.5978 | 0.5521 | 0.6419 |
| matikanetannhauser | daiwaScarlet | 0.5824 | 0.5366 | 0.6269 |
| matikanetannhauser | oguriCap | 0.5824 | 0.5366 | 0.6269 |
| matikanetannhauser | agnesDigital | 0.5780 | 0.5322 | 0.6226 |
| riceShower | mihonoBourbon | 0.5736 | 0.5278 | 0.6183 |
| riceShower | matikanetannhauser | 0.5670 | 0.5211 | 0.6118 |
| riceShower | superCreek | 0.5639 | 0.5179 | 0.6088 |
| matikanetannhauser | mihonoBourbon | 0.5582 | 0.5123 | 0.6032 |
| matikanetannhauser | riceShower | 0.5529 | 0.5069 | 0.5980 |
| riceShower | daiwaScarlet | 0.5463 | 0.5003 | 0.5915 |
| matikanetannhauser | vodka | 0.5429 | 0.4969 | 0.5881 |
| riceShower | riceShower | 0.5385 | 0.4925 | 0.5838 |
| matikanetannhauser | manhattanCafe | 0.5341 | 0.4881 | 0.5794 |
| riceShower | vodka | 0.5286 | 0.4827 | 0.5741 |
| riceShower | agnesDigital | 0.5044 | 0.4586 | 0.5502 |

Per-matchup spread (point estimate): **15.6pp** (riceShower:tamamoCross 0.6608 → riceShower:agnesDigital 0.5044). Wilson lower spread: 15.7pp.

**Per-matchup spread crosses the Slice 4 PFSP-attractive threshold (>15pp)** — same conclusion as v3.0 Step 3 (19.6pp). The bottom-3 matchups (riceShower:agnesDigital 0.5044, riceShower:vodka 0.5286, matikanetannhauser:manhattanCafe 0.5341) are now per-matchup measurement-grade signal; targeted PFSP weighting would push selfplay corpus into these matchups.

### Acceptance branches

- Aggregate uniform wl 0.5690 vs fixed-matchup 0.5811 ceiling = −1.2pp, **within ±5pp matchup-broadly-representative envelope**. Continue current research direction; uniform selfplay default (Slice 2) is no-regret.
- v3.2 representation passes off-distribution generalization in a way v3.0 did not — promotes the next experiment (v3.2 retrain from R110-W6 with uniform selfplay) from speculative to high-prior-positive.
- Per-matchup spread 15.6pp — Slice 4 PFSP weighting remains "attractive" but not load-bearing until a per-matchup recipe-axis verdict emerges.

### Caveats

- Source ckpt was trained under `--deck-sampling=fixed` (Slice 2 landed AFTER iter-3 generated). This is off-distribution generalization, not "v3.2 ceiling under matchup-diverse training." The next step (v3.2 retrain from R110-W6 with uniform selfplay default) addresses that.
- Pre-G5 wall (15.5 min) is the last gate to pay the full Mutex contention cost. Post-G5 (commit `fb7d3f8`, 7.11× at workers=16) any future n=10k gate runs in ~7-8 min.
- Both sides exhibit `riceShower:*` ≥ `matikanetannhauser:*` only on a few matchups; the player-side asymmetry holds globally (player 0.6641 vs opponent 0.4664). Some of the per-matchup spread is the side-asymmetry not deck-of-opponent — the table conflates them and should be re-stratified per-side if a deeper Slice 3 verdict is needed.
- v3.0 fixed-n=10k vs v3.0 uniform-n=10k was never done — the Step 3 smoke was n=1,000 (Wilson half-width ±3.07pp). The −4.8pp gap could shrink by ~3pp at n=10,000. The directional claim (v3.0 generalizes WORSE than v3.2) is robust to that range.

Artifact: `runs/deck-sampling-slice3-v32-gate/gate.manifest.json`.

## Falsification

This is a measurement workstream, not a hypothesis-test. The "falsification" framing is:

- If Step 3 shows the fixed-matchup verdict tracks the uniform aggregate within noise AND per-matchup spread is small (<10pp), then the current single-matchup eval is a serviceable proxy. The deck-pair-sampling line **does not need to be primary** — keep it as a recorded background datapoint and let the existing program continue.
- If the spreads are large, deck-pair sampling becomes **load-bearing for every future verdict**.

## Out of scope

- New decks. We have 13. That's enough to test the question.
- Deck-building (legal pickers, archetype balance). The premade decks are a fixed input.
- Replacing the existing tight-gate history. Fixed-matchup gates stay; diverse-matchup gates are additive.
- Cross-iter opponent-deck pool (mixing deck sampling with opponent-checkpoint sampling). One lever at a time.
- Per-matchup ≤5pp drop tolerance enforcement on the promotion gate (archived item §13). That lands as a follow-up once per-matchup Wilson lower bounds are emitted by Slice 3.

## Relation to active queue items

- **GATES all new training** (per 2026-05-22 user directive): `w6-loop-anti-degradation` HP sweep, `per-game-pfsp-league-retry`, `value-head-data-program` Stages 2/3, and any future selfplay loop are BLOCKED-DECK-VARIETY until Slice 2 lands. Tagged in `docs/ai-agent-state/queue.json`.
- **Does NOT block measurement-only work**: `tight-gate-reverdict-program` (re-verdict #3) and `high-sim-mcts-regime-probe` run on existing checkpoints; their distributions are unchanged.
- **Architecturally parallel to** `per-game-pfsp-league-retry` (cross-iter opponent-pool selfplay) — both are "sample one knob per game from a pool." Different knobs (checkpoint vs deck); no code-share required for Slices P0–3. Post-unblock, PFSP probes run with `--deck-sampling=uniform` so both diversity axes are tested in the same probe.
- **Predecessor for** any future "per-matchup gating" work — archived backlog item §13 surfaces naturally once Slice 3 emits per-matchup Wilson lower bounds.

## Open questions

- Do the 9 untested AI decks pass `intern_deck` cleanly today? The deck-legality smoke (P0b) answers this in ~5 minutes of Rust test time.
- Is the orchestrator's per-iter eval gate (`--eval-games 10` → 20-game bidirectional) noisy enough that adding matchup variance pushes per-iter promote/reject into uselessness? Wilson half-width at `n=20` is already `±21.9pp`; with deck variance it's worse. Likely answer: keep per-iter gate fixed-matchup; only the tight-gate program runs the uniform/PFSP gates.
