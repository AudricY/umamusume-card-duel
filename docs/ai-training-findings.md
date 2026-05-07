# AI Training Findings

## Goal

Train a Python model that can beat the current rule-based bot in closed-loop play, with an aspirational 80% win-rate target against the existing bot.

## Current Status

The end-to-end AI infra works:

- TypeScript simulator exports legal-action training data.
- Python trains candidate-conditioned policy/value models on CUDA.
- Checkpoints export to ONNX.
- ONNX Runtime serves the model with `CUDAExecutionProvider`.
- Headless evaluation runs model-vs-rule-bot games with zero heuristic fallbacks.

The model is not yet strong enough. The best trained policy reached about 42.5% win rate. The strongest diagnostic oracle tested so far, an online one-step rollout selector, reached only 52.5%, which means the current one-step action-label approach is not enough to plausibly reach 80%.

## Experimental Results

| Approach | Dataset / Method | Eval | Result |
| --- | --- | --- | --- |
| Behavior cloning | Imitate rule-bot local decisions | 200 games | ~28.5% WR |
| Outcome oracle v1 | One-step action rollout labels | 200 games | 42.5% WR |
| Larger outcome oracle | More rollout labels, larger model | 240 games | 18.8% WR |
| Hybrid BC + outcome | BC stabilization plus outcome corrections | 240 games | 40.4% WR |
| Richer features + hybrid | Larger state/action features | 240 games | 34.2% WR |
| Common-random outcome labels | Candidate rollouts share RNG per state | 240 games | 39.6% WR overall, 54.2% as opponent |
| Value-head one-step selection | Choose action by predicted next-state value | 120-160 games | ~1-2% WR |
| Online rollout selector | Try legal actions, rollout rest with rule bot | 80 games | 52.5% WR |

Rule bot mirror baseline is roughly balanced by side:

- 200 rule-vs-rule games: player 49.5%, opponent 50.5%.

## Key Findings

### 1. Label Accuracy Is Not Strength

The behavior-cloned model hit very high held-out action-label accuracy, but that only means it predicts the rule bot's local choices. It does not imply stronger play. Perfect imitation should approach parity, not dominate.

### 2. Outcome Labels Help, But Are Noisy

Outcome-oracle labels improved closed-loop play from ~28.5% to 42.5%. That confirms outcome training is the right direction.

However, scaling the same rollout-label method regressed. The labels are noisy because many decisions are close, stochastic, or depend on future multi-step plans that one-step rollouts do not isolate cleanly.

### 3. The Current Value Head Is Not Usable For Planning

Using the value head to pick actions was much worse than policy logits. The value target is too sparse and too weakly calibrated. It predicts final outcome poorly enough that one-step value planning collapses.

### 4. One-Step Search Is A Hard Ceiling

The online rollout selector is stronger than the learned policies because it directly evaluates legal actions with actual simulator rollouts. It still only reached 52.5%.

That is the most important result: if the oracle used to generate labels cannot approach 80%, a model trained from those labels will not either.

### 5. Action Payloads Are Still Under-Specified

The system now executes model-selected legal actions without fallback, but some action kinds still rely on heuristic subchoices inside execution, especially trainer and ability details. This limits both learning and fair attribution.

The model should eventually choose full actions, including:

- trainer target choices
- discard choices
- search choices
- ability targets
- attack ancillary choices
- multi-step tactical bundles

### 6. Feature Quality Improved, But Did Not Solve The Core Issue

Richer features helped the tensor contract but did not produce a win-rate jump. This suggests the bottleneck is not just representation. It is training signal and planning depth.

## Research Direction

### Target Reframe

80% is not a medium-term supervised-learning target. It is a search/RL target.

A realistic path:

- 45-55%: outcome-weighted and DAgger-style supervised learning.
- 55-65%: value-calibrated policy with model-visited-state training.
- 65-80%: search-enhanced policy, self-play, or RL fine-tuning.
- 80%+: likely requires multi-step planning and exploiting weaknesses in the current rule bot.

## Recommended Plan

### Phase 1: Fix The Action Contract

Make legal actions fully executable without hidden heuristic subchoices.

Deliverables:

- Extend `LegalAiAction.payload` for all trainer, ability, search, discard, and attack choices.
- Add executor tests that replay every exported legal action kind.
- Add an eval invariant: `heuristicFallbacks === 0` and no no-op selected action unless the action is explicit pass.

### Phase 2: DAgger From Model-Visited States

Current data mostly comes from rule-bot states. The learned model visits different states and then performs worse.

Deliverables:

- Run model-vs-rule games.
- Log every model-visited decision state.
- Label those states with rule-bot choice and rollout-oracle choice.
- Train on mixed rule states plus model-visited states.

### Phase 3: Multi-Step Search Oracle

Replace one-step rollout labels with shallow tree search.

Recommended search:

- Expand top `K` actions per state.
- Search depth 2-4 decision points.
- Use common random numbers for candidate comparisons.
- Use final outcome plus point margin as reward.
- Distill best searched action into policy labels.

This is the first approach likely to create labels better than the rule bot.

### Phase 4: Calibrate Value

Train value as a separate serious objective before using it for action selection.

Needed changes:

- Train value on model-visited states, not only rule-bot trajectories.
- Use point-margin and terminal outcome targets.
- Track value AUC / calibration by turn bucket.
- Do not use value planning until value beats simple point-margin heuristics.

### Phase 5: Self-Play / RL Fine-Tuning

Once policy execution and value calibration are stable:

- Use self-play to generate trajectories.
- Use advantage-weighted regression or PPO-style updates.
- Keep periodic evaluation against the fixed rule bot.
- Gate promotion on side-balanced held-out seeds.

## Evaluation Standard

A credible 80% claim should require:

- At least 500 held-out games.
- Equal games as player and opponent.
- Fixed seed range not used in training.
- Zero heuristic fallbacks.
- Same deck/config assumptions as the rule bot.
- Report side-split win rates and average points.

## Next Implementation Step

Do not run more large supervised jobs yet. The next highest-value implementation is the full action contract plus DAgger collection. Without those, additional outcome-label training is likely to keep oscillating around 35-45%.

