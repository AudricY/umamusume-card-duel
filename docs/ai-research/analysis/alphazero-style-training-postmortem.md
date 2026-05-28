# AlphaZero-Style Training Postmortem

Date: 2026-05-28

## Summary

The previous AlphaZero-style training line did not fail because one knob was too small. It failed across three increasingly favorable variants:

- Warm-start AZ recipe: `runs/R16-P3-v36-az-from-cont-iter2`
- Cold big-trunk/high-sim recipe: `runs/R16-P3-v36-az-bigtrunk-cold` and `runs/R16-P3-v36-az-bigtrunk-cold-cuda`
- Cold big-trunk/high-sim/5k-games/no-buffer recipe: `runs/R16-P3-v36-az-5k-nobuffer-cuda`

The matched AZ recipe was value-head leaf + two-sided MCTS + policy prior + no KL anchor. That recipe was consistently weaker than the legacy rollout-leaf recipe. Scaling capacity, games, sims, and iterations did not produce a delayed self-improvement curve.

## What Happened

The first AZ run started from the v3.6 cap128 checkpoint and changed only the search/training shape toward AlphaZero canon. Its matched-recipe gate peaked at iter-1 and then degraded:

| Run | Best iter | Best Wilson lower | Notes |
|---|---:|---:|---|
| Warm-start AZ, 100 sims | 1 | 0.4578 | Halted after iter-4 after 3 promotion failures |
| Cold big-trunk CPU, 400 sims | 3 | 0.4773 | Killed after 17/40 iters; no trend |
| Cold big-trunk CUDA, 400 sims | 0 | 0.4363 | Killed after 12/20 iters; no trend |
| Cold 5k-games/no-buffer CUDA, 400 sims | 15 | 0.4281 | Completed 20/20; mean Wilson lower 0.3673 |

The warm-start iter-1 checkpoint tied the v3.6 anchor when evaluated under the old rollout/single-sided recipe, but lost badly under the recipe it was trained for:

| Eval recipe | Wilson lower |
|---|---:|
| Legacy rollout/single-sided | 0.5868 |
| Matched AZ value-head/two-sided | 0.4615 |

That isolates the production regression to the AZ search mechanism, not just the model checkpoint.

## Root Causes

### 1. Value-head leaf is an information bottleneck in this game

The 5k-games run shows the value head learns most of the extractable signal after one iteration and then saturates:

- Root value vs outcome correlation jumps from 0.043 at iter-0 to roughly 0.45-0.51 after iter-1.
- Aggregate correlation across the run averages 0.458 and never breaks 0.508.
- MSE stays around 0.86-1.04.

This game has large stochastic outcome variance from shuffles, draws, and coin flips. Rollout leaf integrates that randomness by simulating it. Value-head leaf asks the net to compress it into a single state value. At the observed correlation ceiling, replacing rollouts with the value head removes signal the old recipe was still using.

### 2. The action space is too narrow for AZ-style visit targets to create rich policy improvement

The 5k-games run averaged only 3.42 legal actions per decision over 663,641
self-play rows. The distribution is sharply concentrated at the low end:

| Legal actions | Share |
|---:|---:|
| 2 | 54.03% |
| 3 | 26.86% |
| 4 | 8.72% |
| 5 | 2.47% |
| 6-10 | 3.05% |
| 11-20 | 4.07% |
| 21+ | 0.80% |

Percentiles: median 2, p75 3, p90 5, p95 10, p99 20, max 45. The high-action tail is mostly `trainerBefore` / `playTrainer` rows from expanded trainer-card choices such as `3starMakeDebutScout` discard/deck-card combinations. Core tactical phases are much narrower:

| Phase | Rows | Mean legal actions | Notes |
|---|---:|---:|---|
| trainerBefore | 421,211 | 3.98 | Owns most of the high-action tail |
| trainerAfter | 96,387 | 2.70 | Mostly 2-3 actions |
| evolve | 81,283 | 2.21 | Almost entirely 2-3 actions |
| attach | 29,741 | 2.71 | Entirely 2-3 actions |
| combat | 20,828 | 2.00 | Binary |
| bench | 14,191 | 2.00 | Binary |

The learned priors and visit targets barely separated:

- Mean top-1 prior rose from 0.416 to about 0.48, then plateaued.
- Mean visit top-1 stayed around 0.52.
- Effective number of visited actions stayed near 2.96.

With only three to four legal moves, 400 sims mostly improve estimate precision for an already tiny choice set. The extra search compute did not create a materially sharper policy target.

### 3. Removing the KL anchor made the loop unstable without unlocking useful exploration

The no-KL AZ recipe removed the W6 fixed-anchor stabilizer. In the warm-start run, the trajectory peaked at iter-1 and regressed through iter-4. In the 5k no-buffer run, gate scores bounced in a low band instead of trending.

Training did not obviously underfit:

- Last-epoch validation loss was flat: first-half mean 1.9412, last-half mean 1.9279.
- Validation accuracy improved from 0.412 to 0.533.
- Train value loss stayed small.

The model was fitting the available targets. The problem was that successive target distributions did not encode a stronger policy, so the loop wandered among roughly equivalent fits instead of accumulating strength.

### 4. Scaling addressed compute limits, not the binding limits

The follow-up run deliberately increased the knobs most favorable to AlphaZero:

- hidden 128/depth 2 to hidden 256/depth 4
- 100 sims to 400 sims
- 240 games/iter to 480 and then 5000 games/iter
- 4-8 iters to 20+ attempted iters
- warm start to cold start

None produced a positive curve. The 5k-games run is the most decisive because it removed small-data and replay-buffer explanations; it still averaged only 0.3673 Wilson lower and peaked at 0.4281.

## Non-Causes

- Not a simple capacity failure. The big-trunk runs were not better.
- Not a simple simulation-budget failure. 400 sims did not lift the curve.
- Not a first-few-iterations cold-start issue. The 5k no-buffer run completed 20 iters and stayed weak.
- Not an export/eval wiring issue. Gate manifests show the intended matched recipe: value-head leaf, two-sided MCTS, policy prior, fixed deck sampling, and no heuristic fallbacks.

## What Did Work

The AZ line had two useful secondary signals:

- It reduced player/opponent side asymmetry when evaluated through the legacy rollout recipe.
- Matched AZ eval was much faster than rollout eval due to batched value-head inference.

Those are not enough to justify the recipe as a strength path. They are better treated as separate engineering or auxiliary-loss ideas.

## Recommended Next Steps

1. Stop treating pure AZ canon as the main strength path for this game.
2. Keep rollout-leaf as the production-strength search target because it integrates stochasticity better than the current value head.
3. If revisiting AZ, use a hybrid rather than pure value-head leaf: value-head/rollout blend, uncertainty-aware leaf values, or rollout-backed value targets.
4. Restore a weak KL anchor for any self-play loop that trains over multiple iterations.
5. Investigate the asymmetry reduction as a targeted loss or sampling correction, not as a reason to keep the full AZ recipe.

## Artifact Pointers

- Scoping and initial result chain: `docs/ai-research/scoping/v36-alphazero-recipe-scoping.md`
- Big-trunk/high-sim result chain: `docs/ai-research/scoping/v36-alphazero-bigtrunk-scoping.md`
- Warm-start AZ run: `runs/R16-P3-v36-az-from-cont-iter2`
- 5k-games/no-buffer run: `runs/R16-P3-v36-az-5k-nobuffer-cuda`
- Diagnostic script: `runs/R16-P3-v36-az-5k-nobuffer-cuda/analyze.py`
