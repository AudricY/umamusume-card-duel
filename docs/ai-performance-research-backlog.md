# Unified AI Performance Research Backlog

Last refined: 2026-05-08.

This is the canonical working backlog for AI strength research. It merges the original performance backlog and the v2 capacity track into one ordered list.

The active goal is not to train a larger supervised model on the current labels. The current blocker is still teacher/planner quality: corrected trained policies are around 34-40% win rate, corrected rollout is 42.5%, and corrected depth-2/top-12 search is 37.5%. The next useful work must either improve the teacher, improve model-visited data, or remove a measurement/representation bottleneck that blocks those steps.

Archived and absorbed items are tracked in `docs/ai-performance-research-backlog-archive.md`. Progress notes remain in `docs/ai-performance-research-progress.md`.

## Promotion Standard

Use this standard before calling a research change successful:

- 500 side-balanced held-out games, unless a tiny smoke is explicitly labeled as a smoke.
- Fixed non-training seed range and documented deck or matchup pool.
- Zero heuristic fallbacks and zero selected no-op actions except explicit legal pass.
- Report win rate with Wilson 95% CI, side split, average points, terminal reasons, selected candidate rank, and fallback/no-op counts.
- Promote policy/model changes only when closed-loop evaluation improves. Row-level accuracy alone is not enough.

## P0: Measurement And Teacher Quality

### 1. Run The Corrected Rebaseline Suite

**Why:** Historical results are not comparable after the corrected evaluator, state fingerprint, modeled action export, and card-aware feature changes. The eval gate exists, but the full corrected suite has not been run at the promotion standard.

**Work:**

- Run rule-bot mirror, `ai-policy` baseline, rollout selector, search selector, planner selector, and the best current trained model on one fixed held-out suite.
- Finish eval-gate residuals: explicit selected no-op count, enforced zero-fallback/zero-no-op thresholds, and manifest output for every comparable run.
- Add the missing bad-policy strength regression so a deliberately bad model fails the gate.

**Acceptance signal:**

- One comparable table covers all baseline methods with the same seeds, decks/matchups, side split, terminal reasons, average points, confidence intervals, selected ranks, fallbacks, and no-op counts.
- The report labels older pre-correction results as historical only.

### 2. Make The Full-Turn Planner A Real Teacher

**Why:** The initial `--selection planner` mode exists, but its tiny smoke was 0/2. The teacher must clear a higher bar before distillation or DAgger can credibly improve closed-loop strength.

**Work:**

- Tune bounded turn-bundle expansion across trainer, evolve, attach, ability, retreat, and combat phases.
- Combine planner leaves with common-random rollout/search scoring.
- Use ranker diversity and exhaustive small-state audits to avoid pruning the best candidate before search sees it.
- Track selected original rank, dropped-best rate, bundle length, and tie/low-margin rates.

**Acceptance signal:**

- Planner beats the rule bot by at least 65% over 500 side-balanced held-out games with zero fallbacks/no-ops.
- Top-K sensitivity decreases, and exhaustive small-state audits show a low dropped-best rate.

### 3. Score Trainer And Ability Choices By Marginal Outcome

**Why:** Search, discard, target, and ability choices still use mostly local/static scores. These decisions often define the turn plan and are a likely teacher-quality bottleneck.

**Work:**

- Simulate trainer and ability choice effects, continue through the best remaining same-turn phases, and score marginal outcome.
- Include pass/skip when an ability consumes resources or worsens survival.
- Prioritize search trainers, discard-to-draw, Rainbow Uncap, damage abilities, and energy movement.

**Acceptance signal:**

- Targeted fixtures improve for search/discard/target decisions.
- Rollout/planner labels produce fewer low-margin discard/search choices.
- Ability use correlates with point gain or survival improvement in held-out eval summaries.

### 4. Turn DAgger Traces Into A Training Loop

**Why:** `--decision-trace-out` exists and can attach rollout/search/planner teacher labels, but there is no explicit trace-to-training recipe or mixed DAgger run yet.

**Work:**

- Convert model-visited decision traces into loadable training rows without hidden opponent-hand leakage.
- Define a repeatable mix of rule-bot, `ai-policy-baseline-visited`, model-visited, rollout-labeled, search-labeled, and planner-labeled rows.
- Run one DAgger round only after the teacher in item 2 or item 3 clears a meaningful gate.

**Acceptance signal:**

- Training manifests report the exact source mix and trace teacher.
- A model trained with model-visited rows beats the same architecture trained only on rule-bot/baseline trajectories under the corrected rebaseline suite.

## P1: Representation, Data Scale, And Training Mix

### 5. Ship One Versioned Feature Migration For Card Identity And Context

**Why:** The current state/action vectors still contain lossy card hashes and positional feature semantics. The v2 capacity track is valid, but it should land as one disciplined schema migration instead of scattered feature churn.

**Work:**

- Add a shared TS/Python card vocabulary and per-card embeddings.
- Replace hand/bench/discard aggregate hashes with set/permutation-invariant encoders.
- Add card metadata used by decisions: weakness, retreat cost, attack count, secondary attack costs, ability presence/recharge state, expected weakness-adjusted damage, and target-survives flags.
- Add a fixed recent-action history slice that also travels in DAgger traces.
- Finish row-level feature schema versions, slot fixture tests, semantic state fixtures, hidden-information safety tests, and closed-loop ablation reporting.

**Acceptance signal:**

- Exports, manifests, checkpoints, ONNX export, and serving all fail fast on schema/vocab mismatch.
- Ablations compare hash baseline, embeddings only, set encoder, metadata/history, and combined features.
- Promotion requires corrected closed-loop improvement or a documented targeted failure-mode fix.

### 6. Parallelize Generation And Add Matchup Sampling

**Why:** Larger or richer datasets are not credible while generation is single-threaded and tied to one fixed deck pair.

**Work:**

- Add worker-thread sharding for rule-bot, outcome, planner, and trace exporters.
- Write per-shard manifests and a deterministic concat manifest.
- Add deck-pool or deck-pair sampling for generation and eval.

**Acceptance signal:**

- `--workers N` scales near-linearly up to physical core count on representative exports.
- Same seed and shard config produces stable rows.
- Manifests record seed shards and deck pair per game.
- Eval reports per-matchup breakdowns.

### 7. Use Margin, Phase, And Action-Kind Signals In Training Mixes

**Why:** CRN metadata, margin buckets, and phase/action metrics now exist, but they are not yet used to control training weights or sampling.

**Work:**

- Run low-margin down-weighting and filtering experiments.
- Add adaptive CRN sample expansion for close decisions.
- Add stratified sampling by phase, action kind, game stage, and one-legal-action pass/end-turn rows.

**Acceptance signal:**

- Training manifests record weighting, filtering, sampler config, and adaptive sample settings.
- Margin-bucket and phase/action-kind metrics improve without common-phase regressions.
- Matched wall-clock comparisons beat fixed one-sample CRN or uniform sampling.

### 8. Harden Procedural Scoring And Turn Goals Under The Gate

**Why:** Some hard-AI scoring has been ported into `ai-policy`, and midgame turn goals exist, but both need larger proof. This remains useful because candidate ordering still feeds search and export.

**Work:**

- Port remaining bench, combat, trainer-choice, and survival-aware scorers without creating engine import cycles.
- Use turn-goal telemetry to validate protect-active, backup-attacker, evolution-dig, and point-lead conversion behavior.

**Acceptance signal:**

- `chooseHighestScoredAction(enumerateLegalAiActions(...))` approaches hard-AI strength in a side-balanced held-out gate.
- Average points improve even when win-rate movement is noisy.
- Search top-K can shrink without losing strength.

## P2: Value, Architecture, And Scale Infrastructure

### 9. Calibrate Value And Action-Value Before Using Them In Search

**Why:** The current value head is unusable for planning. Value work should be treated as a gated research item, not an action selector toggle.

**Work:**

- Train value on model-visited states with terminal outcome, point-margin, and candidate rollout/search rewards.
- Evaluate scalar value, action-value, distributional/quantile value, and auxiliary heads as ablations.
- Consider auxiliary heads for public opponent hand size, next-turn KO risk, turn-end value distance, and opponent next action kind.

**Acceptance signal:**

- Calibration metrics by turn bucket and phase beat simple point-margin baselines.
- One-step value/action-value selection beats heuristic and point-margin baselines.
- Value-augmented search improves win rate without increasing fallback/no-op rate.

### 10. Make Training Scale-Ready Before Large Sweeps

**Why:** Bigger encoders and datasets need observability and reliable resume behavior before multi-hour experiments are worth running.

**Work:**

- Add opt-in TensorBoard or wandb logging for per-batch losses, gradient norms, LR, and per-epoch eval.
- Add sharded JSONL.gz / bounded-memory iterable loading while keeping small map-style smoke tests.
- Add AMP, LR warmup/cosine schedule, gradient accumulation, and optimizer/scheduler/RNG resume.

**Acceptance signal:**

- Smoke tests run unchanged with logging disabled.
- Resume-from-checkpoint matches uninterrupted smoke metrics.
- Manifests record loader, optimizer, scheduler, AMP, accumulation, and resume settings.

### 11. Try Entity-Aware Architecture Only After Feature Tokens Exist

**Why:** Cross-attention or transformer work is not useful on the current hash-heavy snapshot. It becomes meaningful after item 5 creates card/entity tokens.

**Work:**

- Start with action-to-board cross-attention over active, bench, hand pool, discard pool, stadium, energy-zone, and history entities.
- Sweep asymmetric tower scaling after logging and eval gates are stable.
- Keep the transformer encoder as a stretch path only after 100K+ useful rows and entity-token baselines exist.

**Acceptance signal:**

- Cross-attention beats concat-fusion on corrected held-out eval and placement-sensitive phase metrics.
- ONNX export and serving smoke pass with fixed token counts and masks.
- Attention checks on fixtures focus on the relevant target slot/action entity.

### 12. Close Narrow Fidelity And Regression Gaps

**Why:** Several old backlog items are no longer broad active projects, but their residual gaps should be tracked so they do not disappear.

**Work:**

- Add a representative multi-attack fixture once card data includes a real multi-attack card, or add a focused test fixture card.
- Add explicit row-level schema-version fixtures for trainer choice-card, ability discard choice-card, attach readiness, combat attacker metadata, attack index, and target side/slot.
- Enforce manifest presence for ad hoc eval commands that claim comparability.

**Acceptance signal:**

- `npm run test:train` and backend AI tests cover these residual contracts.
- A missing schema version, missing manifest, selected no-op, or incompatible checkpoint fails clearly.

## Not Active Until Gated

- Larger supervised jobs on current labels.
- Self-play or RL before a stronger teacher/planner clears a useful gate.
- Multi-GPU/DDP before single-GPU training becomes the measured bottleneck.
- Masked card/action pretraining unless labeled-data scaling stalls.
- Transformer destination architecture before embeddings, set encoders, cross-attention, and 100K+ useful rows exist.
