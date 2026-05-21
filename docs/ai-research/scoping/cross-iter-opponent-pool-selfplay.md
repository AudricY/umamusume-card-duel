# Cross-Iter Opponent-Pool Selfplay (W6 Wobble Damper)

- **Date:** 2026-05-21
- **Status:** SCOPING — actionable; depends on a one-day plumbing change to `training/r12_orchestrator.py`. Tracked in queue under `per-game-pfsp-league-retry` (priority bumped P3→P2 by this scoping). Predecessor knowledge: `training/opponent_pool.py` (used by `ppo_orchestrator.py` + `dagger_orchestrator.py`).

## Hypothesis

The W6-fix-ON r12 loop produces a model that overfits to its own latest play style on each iter. Symptom: heavy iter-to-iter wobble — extended trajectory `0.5106 / 0.5022 / 0.5022 / 0.4856 / 0.4690 / 0.4526 / 0.5358 / 0.5358 / 0.4773 / 0.5527`; extended-2x-games (in flight) `0.5285 / 0.5118 NO-PROMOTE / 0.5285`. The model is producing data with the SAME blind spots each iter, so distillation reinforces those blind spots, and the rule-bot eval surfaces them as oscillation.

**Mechanism:** model-vs-self selfplay (current `sim:mcts-selfplay` invocation in r12) explores only the regions of state-space that the current model finds interesting. Prior-iter ckpts have DIFFERENT blind spots — playing against them forces the current model to handle a wider distribution of states than self-mirror does.

## Prior negative datapoints (to navigate around)

- **R5 weak-pool league** missed gate. Diagnosis: too-weak opponents, model trivially won everything, no learning signal.
- **PPO Phase J strong-pool** regressed at iter-2 (missed gate by 18pp). Diagnosis (docs/ai-performance-research-progress.md:808-811): v1 sampler used per-RUN opponent choice with self-promoted-at-mode; current model played mostly itself.
- **Open hypothesis (this scoping):** per-GAME re-sampling + strict PFSP weight enforcement (favor opponents the current model loses to) would rescue it. PFSP — Prioritized Fictitious Self-Play — weights opponents by `1 - p(win)`, so the model trains hardest against the opponents most likely to beat it.

## Plumbing

Single new orchestrator surface, two implementer choices:

### Option A — Multi-model Rust sim-cli flag (clean, ~2 days Rust work)

Add `--opponent-pool <path-to-pool-spec.json>` to `engine-rs/crates/sim-cli/src/bin/mcts_selfplay.rs`. Pool spec = list of `(weight, model_url)` entries. Per-game, sample one opponent by PFSP weight (or uniform fallback); model plays both sides under that opponent's policy/value; trajectory recorded as usual.

### Option B — Batch-and-concatenate (cheap, ~1 day Python work, no Rust changes)

In `training/r12_orchestrator.py:run_selfplay`, replace single `sim:mcts-selfplay` invocation with N invocations — one per opponent in the pool, with games-per-opponent allocated by PFSP weight. Concatenate the per-opponent JSONLs into one selfplay.jsonl that the distill step consumes. Spin serve_onnx per-opponent (or run a multi-headed serve_onnx that loads N ckpts; the latter needs a serve_onnx change).

Option B is the smaller MVP — pick it for the first probe.

## Recipe (first probe)

- **Opponent pool:** W6-fix-ON iter-{0..current-1} ckpts. Limit to last 5 to keep wall budget bounded.
- **Sampling:** PFSP weights from eval-gate Wilson-lower vs rule-bot (proxy for "how hard is this opponent right now"); per-GAME re-sampling enforced.
- **Selfplay games budget:** same total (60 or 120 games per iter); split across opponents proportionally to PFSP weight, minimum N games/opponent so PFSP doesn't starve.
- **Recipe otherwise:** identical to extended-2x-games (W6-fix-ON, v3.2 slot-tokens, mcts-leaf rollout, etc.). Source ckpt: extended iter-9 (the 0.5527 best).
- **Acceptance:** wobble flattens (e.g. max |wl[i+1]-wl[i]| over 10 iters < 0.030, vs current ~0.090 between iter-7 and iter-8 of extended) AND best-iter wl ≥ extended iter-9's 0.5527. Either condition stronger lift = strong positive.

## Falsification

If after 10 iters of cross-iter pool selfplay the wobble is statistically indistinguishable from the model-vs-self baseline (extended-2x-games), opponent diversity is NOT the wobble cause — falls back to W6-fix knob tune (lower `--kl-anchor-weight`) as the next forward line.

## Why now

- Rust selfplay 140× speedup makes the larger N opponent pool wallclock-affordable for the first time.
- C8-W6FIX-ON evidence shows the wobble is real and consistent across short (5 iter) and long (10 iter) runs.
- `training/opponent_pool.py` infra has been sitting unused by r12 since R5; the cost of plumbing it in is bounded.
- The vhleaf gate verdict at 0.4281 (vs R14.A 0.452) means strength upside via the calibration axis is also bounded; opponent diversity is an independent lever.

## Out of scope

- Re-running PPO Phase J with new sampler — that's the queue's `per-game-pfsp-league-retry` original framing; this scoping uses the same infrastructure but on the r12 loop, where the wobble symptom is fresh.
- Self-play league against opponents NOT from this lineage (e.g. R110 ckpts) — keeps the experiment apples-to-apples.
- Cycling alarm tuning — keep alarm at default; if it fires, that's a relevant signal but not a load-bearing knob.
