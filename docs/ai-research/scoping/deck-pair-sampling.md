# Deck-Pair Sampling For Self-Play And Eval

- **Date:** 2026-05-22
- **Status:** ACTIVE FORWARD LINE — gates all new training per user directive 2026-05-22 ("no more AI research or training without deck variety"). P0 + Slices 1, 2 are the unblock condition for the rest of the queue's training-bearing items. Tracked in queue under `deck-pair-sampling` (P1). Re-surfaces archived items §6 (deck-pair sampling) and §13 (per-matchup eval gating) from `docs/archive/ai-research/ai-performance-research-backlog.md`, parked behind throughput work and never implemented.
- **Routing:** active backlog reference §6b; this scoping is the canonical home until landed.

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
