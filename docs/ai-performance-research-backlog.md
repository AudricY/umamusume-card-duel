# AI Performance Research Backlog

## Context

The current AI stack is functional end to end: TypeScript exports legal-action examples, Python trains candidate-conditioned policy/value models, ONNX serving works, and headless model-vs-heuristic evaluation runs with no heuristic fallback requirement.

The blocker is strength, not plumbing. `docs/ai-training-findings.md` shows the best trained policies below 45% win rate. Historical online teacher/search runs reached parity to 60% in small side-specific runs, but corrected-evaluator/card-aware follow-ups reset that picture lower: corrected trained policies landed around 34-40%, corrected rollout around 42.5%, and corrected shallow search around 37.5%.

A card-aware feature pass has already expanded action and state semantics, and it did not solve closed-loop strength. A credible 80% win-rate target now depends on a stronger teacher, model-visited-state data, feature schema/versioning discipline, ablation diagnostics, and tighter evaluation gates.

Active execution notes are tracked in `docs/ai-performance-research-progress.md`.

This backlog is ordered by expected impact on closed-loop win rate, then by risk reduction.

## P0: Build A Stronger Teacher Before More Distillation

### 1. Full-Turn Or Turn-Bundle Planner

**Problem:** Current search chooses one micro-phase action at a time. Strong turns often require coordinated trainer, evolve, attach, ability, retreat, and attack sequencing.

**Direction:** Add a planner that enumerates bounded action sequences until attack/end-turn, applies each sequence with `advanceModeledTurnStep`, then scores the resulting state via rollout/search. Start with top-K sequence expansion per phase and hard caps on total sequences.

**Expected impact:** Very high. This is the clearest path to labels better than the rule bot.

**Acceptance signal:**

- Planner beats the rule bot by at least 65% over 500 side-balanced held-out games before distillation.
- `heuristicFallbacks === 0`.
- Results report win rate by side, average points, terminal reasons, and confidence interval.

**Evidence:**

- Current phase stepping is explicit in `backend/src/sim/evaluateModelVsHeuristic.ts`.
- Existing docs already recommend full-turn or turn-bundle MCTS as the next credible step.
- Combat has clone-and-score machinery, but non-combat phases mostly do not.

### 2. Controlled Common-Random Rollout/Search Samples

**Problem:** Noisy action labels make near-tie decisions look meaningful. Outcome export has moved toward seeded candidate comparison, but the current teacher still needs multi-sample rewards, margin/confidence tracking, deterministic tie handling, and candidate-order invariance tests. Evaluator rollout/search paths also need the same discipline.

**Direction:** For each decision, generate N sample seeds and score every candidate against the same seed set. Store sample count, mean reward, reward variance, selected-vs-runner-up margin, and tie policy. Drop or down-weight low-margin labels instead of treating every selected action as equally reliable.

**Expected impact:** High. Lower-variance oracle labels should improve both search quality and supervised targets.

**Acceptance signal:**

- Outcome export records reward mean, variance, and margin per selected label.
- Exported examples include `sampleCount`, per-candidate seed IDs, selected-vs-runner-up margin, and deterministic tie-policy metadata.
- Repeating the same export seed produces identical selected labels.
- Candidate order randomization does not change labels except when margins are below a configured tie threshold.
- Training reports margin-bucket performance, including whether low-margin labels hurt closed-loop win rate.

**Evidence:**

- `backend/src/sim/exportOutcomeTrainingExamples.ts` has a rollout-label path.
- `backend/src/sim/evaluateModelVsHeuristic.ts` has rollout/search selectors but needs more controlled candidate comparison.

### 3. Stop Pruning Search By `features[0]` Alone

**Problem:** Outcome export and search candidate pools sort by one heuristic score. The findings doc notes wider top-K helped more than depth, which suggests useful actions are being pruned.

**Direction:** Add candidate ranker modes:

- heuristic score
- model policy prior
- phase-diverse top-K
- epsilon/random exploratory candidates
- always include baseline and pass/end-turn where legal

Baseline inclusion is expected in the current outcome export path, but pass/end-turn inclusion and ranker modes remain open. Log dropped-best analysis when exhaustive scoring is affordable.

**Expected impact:** High. A better candidate frontier can raise the teacher ceiling without changing the model.

**Acceptance signal:**

- Search reports ranker mode, candidate coverage, and selected candidate original rank.
- Exhaustive small-state audits show low dropped-best rate.
- Top-K sensitivity decreases in held-out evaluation.

**Evidence:**

- `exportOutcomeTrainingExamples.ts` and `evaluateModelVsHeuristic.ts` both sort candidates by `action.features[0]`.

## P0: Train On The States The Model Actually Visits

### 4. Policy-Baseline-Visited Outcome Export

**Problem:** Rule-bot trajectories skip some model-facing decision states. The corrected outcome exporter now advances games with explicit modeled legal actions selected by `chooseHighestScoredAction`, which creates an `ai-policy` baseline state distribution. That is useful, but it is not the same as model-visited DAgger.

**Direction:** Treat `ai-policy-baseline-visited` as a first-class dataset source. Keep it separate from rule-bot, trained-model, rollout-labeled, and search-labeled examples in manifests and training mixes.

**Expected impact:** Medium-high. This is an intermediate distribution-fix before full DAgger and should improve attach/ability phase coverage.

**Acceptance signal:**

- Export manifests report source taxonomy: `rule-bot`, `ai-policy-baseline-visited`, `model-visited`, `rollout-labeled`, and `search-labeled`.
- Baseline-modeled exports show phase/action-kind coverage, especially attach and ability decisions.
- The exporter records when it falls back from modeled baseline action to hard AI because a selected action did not mutate state.

**Evidence:**

- `backend/src/sim/exportOutcomeTrainingExamples.ts` now advances by one explicit modeled legal action, falling back to hard AI only on no-op.

### 5. DAgger-Style Model-Visited-State Export

**Problem:** Rule-bot trajectory data does not match the learned model's state distribution. The model drifts into states that were rare or absent during training.

**Direction:** Add `--decision-trace-out` to model evaluation. For every model decision, write JSONL with:

- observation and legal actions
- model selected action
- heuristic selected action
- rollout/search selected action, if enabled
- scores, margins, fallback status, seed, game result

Use this trace as a mixed dataset for DAgger rounds.

**Expected impact:** Very high once a stronger teacher exists.

**Acceptance signal:**

- A DAgger dataset contains model-visited decisions from both sides with zero hidden opponent-hand leakage.
- Training manifests report source mix: rule-bot, `ai-policy-baseline-visited`, model-visited, rollout-labeled, and search-labeled.
- Closed-loop eval improves over training only on rule-bot trajectories.

**Evidence:**

- `evaluateModelVsHeuristic.ts` counts model actions but discards decision states.
- `docs/ai-training-findings.md` explicitly calls for model-visited-state labeling.

### 6. Episode/Seed-Based Train/Validation Split

**Problem:** Random per-sample splitting can put examples from the same game in both train and validation. That inflates validation accuracy and hides trajectory overfit.

**Direction:** Split by `episodeId` or `seed`, not row index. Preserve complete games in train or validation.

**Expected impact:** Medium. This improves model selection quality and prevents false confidence.

**Acceptance signal:**

- `train_bc.py` supports `--split-by row|episode|seed`, defaulting to seed/episode.
- Manifest records train/validation seed sets.
- Validation accuracy may drop, but held-out closed-loop eval correlates better with validation metrics.

**Evidence:**

- `training/train_bc.py` currently splits shuffled row indices.
- `training/uma_ai/dataset.py` keeps the raw example, including seed/episode fields.

## P1: Fix Action And Feature Fidelity

### 7. Enumerate All Legal Attacks And Attack Choices

**Problem:** AI combat planning uses the primary attack, while the combat engine supports `attackIndex` and additional choices.

**Direction:** Extend `AiCombatDecision` and `buildAttackCandidates` to enumerate legal attacks on the active card. Include attack index, discard choices, switch targets, evolve-from-deck choices, and self-shuffle options where relevant. Coordinate this with feature fidelity work so combat action features include attacker/source card, attack index, attack choice metadata, and target side/slot semantics.

**Expected impact:** High for cards whose best line is not the primary attack.

**Acceptance signal:**

- Action contract covers every attack on representative multi-attack cards.
- Legal combat action export includes attack index.
- Search/eval can attribute win-rate changes by attack index usage.

**Evidence:**

- `frontend/src/game/engine/flow/ai/combatPlanner.ts` calls `getPrimaryAttack`.
- `frontend/src/game/engine/flow/combat.ts` supports richer attack resolution parameters.

### 8. Finish Candidate Feature Schema Migration

**Problem:** Candidate action features have been expanded from 32 to 48 dimensions with trainer effect flags, choice-card role features, attack readiness, and typed deficit. The migration is still incomplete: positional semantics are unversioned, old 32-wide JSONL rows can be silently zero-padded by Python, `target.uid === targetSlot` still looks invalid, and combat actions do not yet pass attacker/source metadata for the new source-card slots.

**Direction:** Version the action-feature schema and finish replacing weak slots with explicit tactical signals:

- target is active / benched / own / opponent
- KO available now
- target survives expected response
- can attack after attach
- energy shortfall by type and total
- points remaining
- under immediate KO threat
- action ends turn
- attacker/source card for combat actions
- attack index and attack choice metadata

**Expected impact:** Medium-high for both learned and score-based policies.

**Acceptance signal:**

- Feature schema version increments.
- TS exports, JSONL examples, training manifests, checkpoints, and ONNX serving all record compatible state/action feature schema versions.
- TS and Python dimensions/constants stay in lockstep, and mismatches fail clearly instead of silently padding/truncating.
- Feature smoke tests validate named slot semantics for trainer choice-card, ability discard choice-card, attach target readiness, combat attacker metadata, and target side/slot fields.

**Evidence:**

- `frontend/src/game/engine/ai-policy/actions.ts` builds 48 action features, but the semantics are positional and unversioned.
- `training/uma_ai/features.py` copies action features into model tensors without schema validation.

### 9. Validate And Ablate State Feature Semantics

**Problem:** State features now fill the 96-wide vector tail with hand role, discard role, readiness, KO, energy-zone, stadium, and ready-attacker aggregates. The remaining risk is that these features may add capacity without improving decisions, and Python now depends directly on `shared/src/data/cards.json` path/schema and variant suffix normalization.

**Direction:** Validate the semantic aggregates, reduce residual hashes where useful, and evaluate learned card embeddings. Add fixture coverage for card-catalog access and hidden-information safety.

**Expected impact:** Medium-high. Better generalization and more informed trainer/search/discard choices.

**Acceptance signal:**

- State feature schema/version is recorded in exports, manifests, checkpoints, and ONNX serving.
- Ablation compares hashed-only, semantic state features, semantic action features, and combined features.
- Phase-level metrics improve for trainer, attach, ability, and combat decisions.
- Feature tests cover hand composition, discard composition, ready attacker, KO availability, variant card IDs, catalog path/schema, and no hidden opponent-hand leakage.

**Evidence:**

- `training/uma_ai/features.py` now adds semantic card-awareness features, but still contains residual hash features and direct card-catalog loading.
- `docs/ai-training-findings.md` reports that card-aware features did not improve corrected closed-loop win rate.

### 10. Feature Ablation And Slot-Importance Runs

**Problem:** The card-aware feature pass fixed real representation gaps but did not improve closed-loop strength. Without ablation, it is unclear whether the features are useful, noisy, undertrained, or simply blocked by teacher quality.

**Direction:** Run controlled comparisons across feature sets and report results by phase/action kind:

- hash-only baseline
- semantic state features only
- semantic action features only
- combined semantic state/action features
- top slot groups removed one at a time

**Expected impact:** Medium. This prevents spending more effort on feature work that the current teacher cannot exploit.

**Acceptance signal:**

- Ablation report includes row-level metrics, closed-loop win rate, phase/action-kind accuracy, margin buckets, and selected candidate rank.
- Feature changes are promoted only when they improve corrected held-out evaluation or explain a targeted failure mode.

**Evidence:**

- Card-aware v1-v4 results in `docs/ai-training-findings.md` did not clear the short-run improvement gate.

## P1: Improve Procedural And Hybrid Policy Quality

### 11. Port Strong Procedural Heuristics Into `ai-policy`

**Problem:** `ai-policy` scoring is shallow compared with the hand-written hard AI. This weakens default action ranking, search pruning, and model candidate features.

**Direction:** Share state-aware scorers between `flow/ai` and `ai-policy`, or create a common policy-evaluation module. Prioritize bench, evolution, attach, trainer, and combat candidate scores.

**Expected impact:** High, especially because search currently depends on candidate ordering.

**Acceptance signal:**

- `chooseHighestScoredAction(enumerateLegalAiActions(...))` performs closer to the hard AI in mirror eval.
- Search top-K can shrink without losing strength.
- No new import cycle across engine modules.

**Evidence:**

- `ai-policy/actions.ts` scores many phases with static constants.
- `flow/ai/core.ts`, `trainerUtils.ts`, `attachUtils.ts`, and `combatPlanner.ts` contain more contextual logic.

### 12. Outcome-Based Trainer And Ability Choice Scoring

**Problem:** Search/discard/target/ability choices are mostly generic scores, not tied to the board plan.

**Direction:** For each trainer or ability choice, simulate the effect, run best remaining phases through combat/end-turn, and score marginal value. Include pass/skip when using an ability consumes energy or worsens survival.

**Expected impact:** High for search trainers, discard-to-draw, Rainbow Uncap, damage abilities, and move-energy abilities.

**Acceptance signal:**

- Target/discard choice accuracy improves on targeted fixtures.
- Rollout teacher selects fewer low-margin discard/search labels.
- Ability use rate correlates with point gain or survival improvements.

**Evidence:**

- `ai-policy/actions.ts` expands trainer choices but scores them with tiny target/discard adjustments.
- Procedural ability ordering in `flow/ai/core.ts` is mostly stage/order driven.

### 13. Broaden Turn-Goal Detection

**Problem:** Goals currently cover immediate lethal, narrow two-turn lethal, immediate KO threat, and no-bench recovery. Strategic midgame goals are missing.

**Direction:** Add goals for:

- build backup attacker
- protect loaded active
- deny opponent setup
- dig for evolution
- stabilize low-deck or low-bench states
- convert point lead safely

Feed goals into trainer, attach, bench, ability, and combat scoring.

**Expected impact:** Medium. This improves strategic consistency across phases.

**Acceptance signal:**

- Telemetry shows goal distribution by turn bucket and side.
- Goal-specific fixtures validate expected action preferences.
- Closed-loop eval improves average points even before win-rate gains.

**Evidence:**

- `frontend/src/game/engine/flow/ai/turnPlan.ts` is compact and currently narrow.
- Attachment and trainer selection already consume `AiTurnGoal`.

## P1: Make Value Useful For Planning

### 14. Train A Calibrated Value Or Action-Value Head

**Problem:** The current value head performed poorly when used for planning. It is trained on sparse game outcomes and not calibrated by turn/phase.

**Direction:** Train value as a first-class objective:

- use model-visited states
- include terminal outcome and point-margin targets
- report AUC/calibration by turn bucket and phase
- consider an action-value head trained from candidate rollout/search rewards

Only use value-guided selection after it beats simple point-margin heuristics.

**Expected impact:** Medium-high. Useful value estimates can reduce rollout cost and improve search leaves.

**Acceptance signal:**

- Value calibration plots/metrics are written to manifest.
- One-step value selection beats point-margin and heuristic-score baselines.
- Value-augmented search improves win rate without increasing fallback/no-op rate.

**Evidence:**

- `training/uma_ai/model.py` has a state value head.
- `training/train_bc.py` trains value with a small weighted MSE.
- `evaluateModelVsHeuristic.ts` already supports value-based selection for experiments.

## P2: Evaluation, Reproducibility, And Regression Gates

### 15. Corrected-Evaluator Rebaseline And Comparability

**Problem:** The state-hash and one-action exporter fixes changed the meaning of historical results. Older rollout/search numbers are not directly comparable with corrected-evaluator/card-aware runs.

**Direction:** Establish a corrected-evaluator baseline suite on one fixed held-out seed set. Run rule-bot mirror, `ai-policy` baseline, rollout selector, search selector, and trained model evaluations on the same side-balanced seeds.

**Expected impact:** High for research quality. This separates real strength gains from measurement changes.

**Acceptance signal:**

- A single table reports all baseline methods on the same corrected evaluator, seed range, decks, model side split, and max step settings.
- Historical results are labeled non-comparable unless rerun under the corrected evaluator.
- Reports include side split, average points, fallback/no-op count, terminal reasons, selected candidate rank, and confidence interval.

**Evidence:**

- `docs/ai-training-findings.md` now contains historical results plus corrected card-aware/evaluator results with lower rollout/search numbers.

### 16. Deterministic Progress Fingerprint And No-Op Detection

**Problem:** No-op detection, stall detection, fallback accounting, and search memoization depend on state fingerprints. The active hash expansion catches more real mutations, but duplicated local implementations and count-only hand/deck/discard summaries still leave blind spots.

**Direction:** Move state fingerprinting into a shared helper and define its contract. Include all public and simulator-relevant mutation fields, and explicitly document any intentionally excluded private contents.

**Expected impact:** High for trustworthy measurement and label generation.

**Acceptance signal:**

- `headlessAiVsAi.ts`, `evaluateModelVsHeuristic.ts`, action-contract tests, and search memoization use one shared fingerprint helper.
- Fixture tests prove the fingerprint changes for active/bench HP, energies, special conditions, tools, ability usage, phase, pending choice, points, turn flags, energy zone, and board movement.
- Tests document whether same-count hand/deck/discard content swaps are intentionally ignored or included.

**Evidence:**

- `backend/src/sim/evaluateModelVsHeuristic.ts` and `backend/src/sim/headlessAiVsAi.ts` currently have duplicated compact state hashing logic.

### 17. Evaluation Gate With Confidence Intervals

**Problem:** Evaluation emits summary JSON but no pass/fail gate, confidence interval, seed split discipline, or phase/action breakdowns.

**Direction:** Add an eval gate script around `sim:evaluate-model` that runs side-balanced held-out seeds and fails below configured thresholds.

**Expected impact:** Medium. Prevents chasing noisy improvements.

**Acceptance signal:**

- Reports win rate, Wilson CI, side split, average points, terminal reasons, fallback count, and no-op count.
- Fails if fallback count is nonzero or sample size is below target.
- Standard 500-game held-out run is documented.

**Evidence:**

- `evaluateModelVsHeuristic.ts` summarizes aggregate win rate and points.
- Root `package.json` exposes eval scripts but no gated command.

### 18. Dataset And Run Manifests

**Problem:** Export scripts write JSONL and minimal console metadata. Training manifests do not preserve simulator/search parameters, git SHA, or dataset distributions.

**Direction:** Write `manifest.json` beside every export, training run, and eval result with:

- command args
- git SHA and dirty flag
- seed ranges
- feature schema versions and source taxonomy
- phase/action-kind counts
- terminal reasons
- reward margin distribution
- sample-weight distribution
- fallback/no-op counts

**Expected impact:** Medium. Makes experiments reproducible and comparable.

**Acceptance signal:**

- Every generated JSONL or checkpoint has a sibling manifest.
- Manifest is sufficient to rerun the experiment.
- Training README documents the experiment artifact layout.

**Evidence:**

- `exportTrainingExamples.ts` and `exportOutcomeTrainingExamples.ts` only print metadata.
- `train_bc.py` writes model metrics but not full dataset/export provenance.

### 19. Stronger AI Training Smoke Tests

**Problem:** Current training smoke proves a tiny model can reduce loss on deterministic rule-bot labels. It does not test outcome labels, trace export, evaluator integration, or worst-action detection.

**Direction:** Add smoke tests for:

- outcome-export schema and sample weights
- deterministic oracle labels across repeated seeds
- DAgger trace export
- fake model server that chooses best/pass/worst actions
- eval gate failure on intentionally bad policies
- feature-schema compatibility across TS export, Python training, ONNX export, and serving
- corrected fingerprint/no-op regression fixtures
- diagnostics by phase, action kind, card family, selected candidate rank, regret, and margin bucket

**Expected impact:** Medium. Catches training-signal regressions before expensive jobs.

**Acceptance signal:**

- `npm run test:train` covers heuristic export, outcome export, and fake-model eval.
- Train/export/serve fails clearly when checkpoint dimensions do not match current `STATE_DIM`/`ACTION_DIM`.
- Modeled baseline action either mutates state or records exactly one fallback to hard AI.
- Action contract remains zero no-op for sampled legal actions.

**Evidence:**

- `backend/src/tests/aiTrainingSmoke.ts` checks only a simple rule-bot imitation loop.
- `backend/src/tests/aiActionContractSmoke.ts` checks mutating legal actions but not label quality.

## Suggested Implementation Order

1. Fix measurement first: corrected-evaluator rebaseline, deterministic progress fingerprint, common-random samples, eval gate, and manifests.
2. Build full-turn/turn-bundle planner and ranker diversity until the teacher itself clears a higher win-rate bar.
3. Use policy-baseline-visited exports as an intermediate source, then add DAgger trace export and train on model-visited states labeled by the stronger teacher.
4. Finish feature migration: schema versioning, compatibility checks, feature smoke tests, ablations, and checkpoint/dataset regeneration policy before larger training runs. Old checkpoints and ONNX exports are incompatible after `STATE_DIM`/`ACTION_DIM` changes unless regenerated.
5. Calibrate value/action-value heads and use them as search accelerators only after offline calibration passes.

## Non-Goals For The Next Round

- Do not run larger supervised jobs on the current label setup and expect an 80% result.
- Do not promote policy/value changes based only on row-level accuracy.
- Do not use value-guided action selection until calibration beats simple baselines.
- Do not weaken the zero-fallback/no-op action contract to make experiments pass.
