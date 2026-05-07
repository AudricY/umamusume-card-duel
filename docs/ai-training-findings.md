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

The model is not yet strong enough. The best trained policy reached about 42.5% win rate. Follow-up search experiments found that the current oracle/planning setup caps stronger online planners around parity, which blocks an honest 80% trained-policy target under the current setup.

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
| Depth-2 search, top-4 | Recursive model-side decisions, narrow candidates | 60 games | 48.3% WR |
| Depth-2 search, top-12 | Recursive model-side decisions, wide candidates | 40 games/side | 57.5% as player, 50.0% as opponent |
| Depth-3 search, top-8 | Deeper recursive search | 20 games/side | 60.0% as player, 55.0% as opponent |
| Depth-2 search, top-12, 4 samples | Multi-sample candidate averaging | 40 games | 52.5% WR |
| Explicit action contract smoke | Sample and apply exported legal actions | 24 games / 1,867 actions | PASS, zero sampled no-op actions |
| Depth-3 search, top-16, 2 samples after explicit payloads | Recursive search over expanded trainer/ability payloads | 30 games/side | 46.7% as player, 53.3% as opponent, zero fallbacks |

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

### 4. Current Search Oracle Is A Hard Ceiling

The online rollout selector is stronger than the learned policies because it directly evaluates legal actions with actual simulator rollouts. It still only reached 52.5%.

Depth-2, depth-3, wider candidate sets, and multi-sample search did not get close to 80%. The best small-sample side-specific result was 60% as player and 55% as opponent. That is the most important result: if the oracle used to generate labels cannot approach 80%, a model trained from those labels will not either.

### 5. Action Contract Is No Longer The Main Blocker

The action payload has been expanded for the high-impact hidden-choice cases:

- trainer discard/search/target choices
- rainbow uncap target/evolution choices
- ability damage targets
- ability energy-source and energy-type choices
- ability discard-to-draw choices

`test:action-contract` now samples exported legal actions and verifies that applying them mutates state unless the action is an explicit pass. The latest run checked 1,867 actions and passed.

This removed an important attribution risk, but it did not raise the search ceiling. The remaining issue is not that legal actions cannot execute; it is that the available oracle is still not finding decisively stronger play.

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

Status: mostly complete for the current high-impact action surface.

Completed:

- Extended `LegalAiAction.payload` for trainer discard/search/target choices and major ability target/discard/energy choices.
- Added `test:action-contract`, which samples exported legal actions and verifies that each non-pass action mutates state.

Still useful later:

- Broaden the contract test into targeted fixtures for every card/effect family instead of relying only on sampled headless games.
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

Do not run more large supervised jobs yet. The explicit action contract now passes a sampled executor smoke test, and stronger depth/top-K search still stayed around parity. The next highest-value implementation is a qualitatively stronger planner/training loop: full-turn or turn-bundle MCTS, model-visited-state DAgger from that planner, and then distillation only after the planner itself clears the target.

## Continuation Log

### Search Follow-Up

After the initial findings, deeper search was added to the evaluator:

- `--selection search`
- `--search-depth`
- `--search-top-k`
- `--search-samples`

Results stayed far below 80%. Wider top-K helped more than depth, which suggests the heuristic candidate ordering can hide useful actions, but deeper recursive planning did not create a decisive advantage.

### Explicit Action Contract Follow-Up

Implemented explicit payload enumeration/execution for trainer choices and the major ability choice types. Added `test:action-contract` to sample exported legal actions during headless games and assert that each selected non-pass action changes state.

Latest verification:

- `npm --workspace backend run build`: pass.
- `TMPDIR=/tmp npm --workspace backend run test:action-contract`: pass, 1,867 sampled actions.
- Depth-3/top-16/two-sample search after the action expansion: 46.7% WR as player over 30 games, 53.3% WR as opponent over 30 games, zero fallbacks.

Interpretation: hidden trainer/ability subchoices were a real correctness gap, but not the dominant strength blocker. Once those choices were exposed, the stronger search still performed roughly like the rule bot.

### Hard Blocker

The current setup has a hard blocker for the requested 80% target:

- The best trained models are below 45%.
- The best online search oracle is only around parity to 60% in small side-specific runs.
- After explicit action payloads, a fresh stronger run was 46.7%/53.3% by side, not better.
- Value-guided planning is currently unusable.
- The sampled legal-action executor contract passes, so the remaining blocker is planner/training signal strength, not basic action executability.

Therefore, continuing to train larger supervised models on the current labels is not a credible path to 80%. The next credible work is structural: build a stronger full-turn planner or RL/self-play loop that can itself clear the target before distillation. Until the teacher clears 80%, a distilled model should not be expected to do so.
