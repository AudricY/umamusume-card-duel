# Unified AI Performance Research Backlog

Last refined: 2026-05-08 (v4 — evidence-driven pass after the first real DAgger run).

This is the canonical working backlog for AI strength research.

## End Goal: Working RL Loop

The destination is a **PPO/self-play RL loop with a DAgger warm-start** (Options A → B). v3 framed Option A (DAgger) as the endpoint with Options B/C as gated follow-ups; the v4 evidence reverses that framing. The DAgger orchestrator is now a working warm-start engine, but it does not by itself produce a policy that beats the rule bot. Online PPO (F1) is the path through the imitation cap, not a polish layer; AlphaZero-style search-at-training-time (F2) remains a gated follow-up to F1.

**Empirical strength on the corrected suite (2026-05-08):**

- Rollout selector with CRN samples=3: **68.0% / Wilson [58.3, 76.3]** — strongest measured teacher.
- Corrected planner (CRN samples=3, max-depth 8): 51.0% / Wilson [41.3, 60.6]. Below item 2's pre-registered 55% Wilson-lower-bound escalation floor.
- Trained policy after 2 DAgger iterations on rollout-CRN traces: **34.0%** in iteration 0, **28.0%** in iteration 1 (auto-rolled-back). Side-balanced. Zero fallbacks/no-ops. The ~34% supervised number reproduces the archived imitation cap on the corrected suite.

The "imitation ceiling sits in the 55-65% range" claim from v3 is empirically too optimistic at our data scale. Two interpretations remain credible: (a) the cap is closer to 35-45% on corrected one-step labels and 100-200 games per iteration, (b) the cap is higher but only reached at item 5b (set encoders) + 10× current data. v4 declines to assume (b).

Every active item below earns its place by being a **prerequisite for the RL loop** or by being on the loop itself. Items that were SL strength polish but do not unlock the loop have moved to "Behind The Gate." The archive (`docs/ai-performance-research-backlog-archive.md`) captures the rest. Progress notes remain in `docs/ai-performance-research-progress.md`.

## Status Legend (v4)

`✓ done` — landed and recorded in `progress.md`. `~ partial` — minimum viable subset landed, named gaps remain. `· pending` — work has not started. `↑ priority bump from v3`. `↓ priority drop from v3`.

## Promotion Standard

Use this standard before calling a research change successful. RL iterations also use the additions in the second list.

- 500 side-balanced held-out games, unless a tiny smoke is explicitly labeled as a smoke.
- Fixed non-training seed range and documented deck or matchup pool.
- Zero heuristic fallbacks and zero selected no-op actions except explicit legal pass.
- Report win rate with Wilson 95% CI, side split, average points, terminal reasons, selected candidate rank, and fallback/no-op counts. Wilson lower bound, not point estimate, is the comparison number.
- Promote policy/model changes only when closed-loop evaluation improves. Row-level accuracy alone is not enough.

For RL iterations, additionally:

- Evaluate against an opponent pool: rule bot, last 3 promoted self-play snapshots, planner, rollout selector. Report per-opponent win rate, not just aggregate.
- Promote a new RL checkpoint only if its Wilson lower bound on the pool is above the prior promoted checkpoint's Wilson lower bound.
- Track entropy floor and per-feature drift across iterations to flag policy collapse and cycling.
- Run a side-balanced **best-response exploitation check** against depth-2/top-12 search as a tracked time-series, not a one-shot. Any monotone decline of >5pp across 3 iterations halts the loop pending investigation.
- Action-coverage histogram per iteration: every action with <N labels is flagged for forced exploration (item 15).

## P0 Spike: Throughput Probe

### 0. Measure Current Sim And Planner Throughput Before Sizing The Loop ✓ done

**Status:** `npm run sim:throughput-probe` landed; production-default planner config measured at 22.4 dec/s single-core (40.3 dec/game, 44.6 ms/decision). 500-game DAgger projection: 0.25h on 1 core, 0.031h on 8 cores ideal scaling. ≥100× headroom on a single core for item 2's planner cost growth before approaching the 4h gate. Item 14's targets are now grounded; the "establish targets first" sub-task there is closed by this measurement.

**Why:** Item 14's budget targets and item 6's "near-linear scaling" claim are unfalsifiable until we know what a single planner decision costs today. The whole RL loop is throughput-bound; without numbers we cannot tell whether one DAgger iteration is 2 hours or 2 weeks. This is a one-day spike, not a research project.

**Work:**

- Microbenchmark `headlessAiVsAi.ts` step cost: `enumerateLegalAiActions`, `chooseHighestScoredAction`, full `stateFingerprint`, deep `GameState` clone in `advancePlayerAiTurnStep`. The fingerprint covers full hand/deck/discard content and is called twice per step — likely a hot path.
- Measure planner decisions/sec at current settings on a single core, decisions per game, and total wall-clock per 100-game shard.
- Project DAgger iteration cost at 500 games × N decisions/game × planner cost. Compare to a target iteration budget (e.g., ≤4h on the user's box).
- If the projection is >24h per iteration on parallel hardware after item 6, escalate to a sim profile-and-optimize sub-task before items 11-14 are scheduled.

**Acceptance signal:**

- A short markdown report with three numbers: planner decisions/sec/core, decisions/game, and projected iteration wall-clock at 1× / 8× / 32× cores.
- Either a green light to keep item 14's targets implicit, or a flagged escalation with a candidate optimization (fingerprint diff caching, structural-hash, persistent data structures, etc.).

## P0: RL Critical Path Prerequisites

Items 1-5a must finish (or have credible partial landing) before the first DAgger iteration. Items 1, 6, 8, 10 can land in parallel from day one.

### 1. Run The Corrected Rebaseline Suite ✓ done

**Why:** RL needs a stable measurement baseline to detect per-iteration regressions. Historical results pre-correction are not comparable.

**Work:**

- Run rule-bot mirror, `ai-policy` baseline, rollout selector, search selector, planner selector, and the latest manifest-pinned trained checkpoint on one fixed held-out suite. Cite checkpoints by manifest hash, not by run name.
- Finish eval-gate residuals: explicit selected no-op count, enforced zero-fallback/zero-no-op thresholds, manifest output for every comparable run (named output path, not "somewhere in `runs/`").
- Add a planted bad-policy regression: deliberately invert candidate scoring so the worst legal action is selected; the gate must reject it.
- Tighten stall detection in `headlessAiVsAi.ts` from 1-step fingerprint equality to N-step cycle detection (any state in last K steps repeats). Adds an exploit-hardening floor.

**Acceptance signal:**

- One comparable table at a named artifact path covers all baseline methods with the same seeds, decks/matchups, side split, terminal reasons, average points, Wilson CIs, selected ranks, fallbacks, and no-op counts.
- The planted-bad-policy fixture fails the gate.
- N-step cycle stall detection has a smoke fixture.

### 2. Bring The DAgger Teacher Above The Gate ~ partial — pre-registered escalations exhausted

**Status:** Planner CRN landed (Wilson lower 41.3%, below the 55% floor). Rollout-CRN landed as plan B option 2 (Wilson lower 58.3%, above the 55% floor but 6.7pp short of the 65% target). Both pre-registered v3 fallbacks ("rollout-augmented planner with deeper CRN", "planner + value-head tiebreaker") are partly executed: rollout-CRN is the rollout-augmented half, the value-head half is blocked on item 7. Per the pre-registered "do not silently lower the gate" rule, the v4 decision is recorded explicitly below rather than the gate being moved.

**v4 explicit decision rule:** If after items 7 and 9 land the planner remains below 55% Wilson lower and rollout below 65% Wilson lower, the loop ships with rollout-CRN as the DAgger teacher and **F1 (PPO) becomes the explicit path through the cap**. The 65% gate stays as an aspirational ceiling, not a launch blocker. v3's "stop the program" implication of failing item 2 is retracted: F1's gating is rephrased below.

**Why:** The planner is the DAgger relabeling oracle. If it does not beat the rule bot, every downstream RL iteration amplifies its error. This item is the choke point of the entire program.

**Work (remaining levers, in priority order):**

- **Planner + rollout composition.** Use the planner to enumerate ≤K candidates (current top-K=4 with bundle-style expansion), then score each candidate with rollout-CRN samples=3 instead of the planner's current internal heuristic leaf. This is a literal merge of the two strongest measured methods.
- **Deeper rollout-CRN sweep.** Test K ∈ {3, 5, 8, 12} samples and record marginal lift per sample. Diminishing returns are expected past K=5 but are not measured.
- **Per-lever lift table** for the planner: bundle depth, CRN samples, first-action aggregation (max vs mean), ranker mode (heuristic vs phase-diverse vs epsilon).
- **Item 7 value-head tiebreaker** (gated on item 7 landing): when two first-action groups tie within ε on rollout reward, break the tie with the value head's prediction.

**Acceptance signal (kept for the aspirational ceiling):**

- Planner Wilson lower bound ≥65% over 500 side-balanced held-out games with zero fallbacks/no-ops.
- Per-lever lift table recorded.
- Top-K sensitivity decreases; dropped-best rate <2% on the small-state audit.

**Acceptance signal (relaxed launch criterion, v4 addition):**

- DAgger teacher Wilson lower bound ≥55% on the corrected suite. **Met by rollout-CRN at 58.3%.** This is the gate the orchestrator currently uses; it is not a substitute for the 65% ceiling.

### 3. Score The Choices The Planner Enumerates Over

**Why:** Planner leaves still depend on local trainer/ability/discard scoring. Bad scoring at planner leaves limits item 2 directly.

**Scope vs item 9:** This item scores the *choices inside one planner-enumerated action* (which discard target, which trainer choice). Item 9 scores the *candidate ordering used to enumerate planner inputs in the first place*. Both feed planner quality but at different layers.

**Work:**

- Define "marginal outcome" concretely: CRN-paired rollout return delta, fixed seed set per leaf, configurable rollout depth.
- Simulate trainer and ability choice effects, continue through best remaining same-turn phases, score marginal outcome.
- Include pass/skip when an ability consumes resources or worsens survival.
- Provide a standalone smoke beyond item 2's gate: per-leaf top-1 agreement vs CRN ground truth on a fixture set.

**Acceptance signal:**

- Standalone smoke: per-leaf top-1 agreement ≥80% on the fixture set.
- Item 2's planner gate improves under the new leaf scoring without raising fallback/no-op rates.

### 4. Ship The DAgger Trace-To-Training Data Recipe ✓ done

**Why:** This is the data spine of the RL loop. Item 11 owns the rollout/retrain plumbing; this item owns the data shape only.

**Work:**

- Convert model-visited decision traces into loadable training rows. Includes a hidden-info leak fixture: assert opponent hand contents never appear in the row, even after the recent-action history slice from item 5.
- Pin a default source-mix recipe: e.g., 40% planner-labeled-on-model-visited, 30% rule-bot, 20% baseline-modeled-visited, 10% rollout-labeled. Mix is overridable per iteration; this is the starting ratio.
- Smoke one DAgger round end-to-end (single iteration, no orchestrator yet) to prove the recipe, even with a weak teacher. Promotion of the *trained checkpoint* gates on item 2; execution of the *recipe* does not.

**Acceptance signal:**

- Hidden-info leak fixture passes.
- One DAgger round runs end-to-end with the default mix and produces a manifest recording teacher, mix ratios, source-row counts, and fingerprint of the parent checkpoint.
- A model trained with model-visited rows beats the same architecture trained only on rule-bot/baseline trajectories under the corrected rebaseline suite (gated on item 2).

### 5. Versioned Feature Migration For Card Identity And Context ~ partial (5a done, 5b pending)

**Why:** Hash-heavy state caps any RL ceiling, not just SL accuracy. The model the loop iterates on is the same model used at serving time; representation work has to land before the loop, not during. **Bundled to land as one schema migration**, but split into a blocking subset (5a) and an enhancement subset (5b) so the loop is not gated on the full omnibus.

**5a (blocking item 11):**

- Shared TS/Python card vocabulary indexed by canonical cardId (with the existing `FullArtGold`/`FullArt`/`UncommonPlus` normalization).
- Per-card embedding lookup replacing `_identity_features` hashes in state and action features.
- Row-level schema version + ONNX export/serving fail-fast on schema/vocab mismatch.

**5b (parallel, non-blocking):**

- Hand/bench/discard set/permutation-invariant encoders.
- Card metadata in feature vectors: weakness, retreat cost, attack count, secondary attack costs, ability presence/recharge state, expected weakness-adjusted damage, target-survives flags.
- Recent-action history slice (also travels in DAgger traces).
- Slot fixture tests, semantic state fixtures, hidden-information safety tests beyond item 4's leak fixture, full closed-loop ablation matrix (hash-baseline vs each enhancement vs combined).

**Acceptance signal (5a):**

- Exports, manifests, checkpoints, ONNX export, and serving fail fast on schema/vocab mismatch.
- A model trained with embeddings beats the hash baseline on the corrected suite at matched param count.

**Acceptance signal (5b):**

- Ablation matrix recorded; each enhancement gates on documented closed-loop improvement or a targeted failure-mode fix.

## P1: RL Infrastructure (Parallelizable From Day One)

### 6. Parallelize Generation For Self-Play Throughput And Add Matchup Sampling · pending ↓

**v4 priority drop note.** Throughput probe (item 0) recorded ~100× headroom against item 14's 4h iteration target on a single core; worker-thread sharding is no longer urgent for the first DAgger sweep. It re-enters the critical path once the orchestrator's iteration count rises into 10s of iterations or once F1 needs higher rollout throughput.

**Why:** Self-play and DAgger relabeling are throughput-bound. Single-threaded generation makes RL infeasible. Matchup sampling also seeds the eventual opponent pool.

**Work:**

- Add worker-thread sharding for rule-bot, outcome, planner, and trace exporters.
- Per-shard seed derivation via a deterministic scheme (e.g., `SeedSequence.spawn`-equivalent): same `--workers N` config produces identical concatenated rows.
- Per-shard manifests + a deterministic concat manifest.
- Deck-pool or deck-pair sampling for generation and eval; record per-game deck pair so item 12's pool can stratify.

**Acceptance signal:**

- `--workers N` achieves ≥0.8× ideal speedup up to physical core count on representative exports, with bounded per-worker memory.
- Same seed and shard config produces byte-stable concatenated rows.
- Manifests record seed shards and deck pair per game.
- Eval reports per-matchup breakdowns.

### 8. Make Training Scale-Ready Before Large Sweeps ~ partial

**Why:** RL iterations are multi-hour and crash-prone; resume, AMP, sharded loaders, and live logging are non-negotiable for the orchestrator.

**Work:**

- Add opt-in TensorBoard or wandb logging for per-batch loss, gradient norm, LR, per-epoch eval; offline by default for smoke tests.
- Sharded JSONL.gz / bounded-memory iterable loading; keep small map-style smoke tests.
- AMP, LR warmup/cosine schedule, gradient accumulation, optimizer/scheduler/RNG resume.
- ONNX-roundtrip smoke after every training run so promoted checkpoints are guaranteed deployable.

**Acceptance signal:**

- Smoke tests run unchanged with logging disabled.
- Resume-from-checkpoint matches uninterrupted smoke metrics within tolerance (final loss <1e-5, eval win rate identical at fixed seed; bitwise reproduction restricted to CPU-only smoke).
- Recorded samples/sec target ≥10K rows/sec on the user's box at the standard dataset size.
- Manifests record loader, optimizer, scheduler, AMP, accumulation, resume, and ONNX-roundtrip status.

### 10. Close Reproducibility Gaps ~ partial

**Why:** Manifest enforcement and schema-version fixtures protect the RL loop's reproducibility across iterations.

**Work:**

- Row-level schema-version fixtures for trainer choice-card, ability discard choice-card, attach readiness, combat attacker metadata, attack index, target side/slot.
- Enforce manifest presence for any ad hoc eval command that claims comparability — missing manifest exits non-zero with a structured error JSON, not just a print.

**Acceptance signal:**

- `npm run test:train` and backend AI tests cover these residual contracts.
- Missing schema version, missing manifest, selected no-op, or incompatible checkpoint exits with a structured error class.

## P1-Optional: Recommended Enhancers (Not Blocking The First DAgger Round)

### 7. Calibrate Value/Action-Value As The RL Critic ↑ priority bump → de-facto P0

**v4 priority bump note.** v3 marked this P1-Optional. Two pieces of new evidence move it onto the critical path:

1. Item 2's "planner + value-head tiebreaker" plan B option needs a usable value head; without item 7 that fallback is blocked.
2. F1 (PPO) is now the path through the imitation cap rather than a polish layer. Its advantage estimator requires a calibrated value head. F1 cannot start without item 7.

The work and acceptance signals below are unchanged from v3; only the urgency and the gating relationship are restated. Treat item 7 as the next P0 work after the orchestrator's first sweep.

**Why:** Required for Option B's advantage estimator. Useful as a multi-task auxiliary head during DAgger to regularize the shared encoder. Not on the DAgger critical path, so explicitly not blocking item 11.

**Work:**

- Train value on model-visited states with terminal outcome, point-margin, and candidate rollout/search rewards.
- Evaluate scalar value, action-value, distributional/quantile value, and auxiliary heads as ablations.
- Auxiliary heads under consideration: public opponent hand size, next-turn KO risk, turn-end value distance, opponent next action kind.
- Include cross-iteration stability acceptance: no unbounded drift in value mean/variance across iterations; consider target-network or polyak averaging if drift appears.

**Acceptance signal:**

- Calibration metrics named: ECE, Brier score, reliability diagram by turn bucket. Required: Wilson-significant lift over point-margin baseline on at least one of these.
- One-step value/action-value selection beats heuristic and point-margin baselines under the corrected suite.
- Cross-iteration value mean/variance recorded; no drift outside a configured envelope across 3 iterations.

### 9. Harden The Procedural Scorers The Planner Consumes · pending

**Why:** Item 2 prunes candidates before search sees them. Better candidate ordering makes the planner stronger without any architectural change. Scope is intentionally narrow: only what the planner enumerates over (delineated against item 3 above).

**Work:**

- Port remaining bench, combat, trainer-choice, and survival-aware scorers without creating engine import cycles.
- Drop turn-goal broadening as a standalone target — RL learns goals from reward.

**Acceptance signal:**

- Planner gate (item 2) Wilson lower bound improves by ≥3pp under the new scoring, or planner top-K can shrink by ≥3 candidates without losing strength.

## P2: RL Loop Work (Option A — DAgger)

Required prereqs for item 11: items 1, 4, 5a, 8, 10. Recommended: item 2 cleared, item 6 for budget realism, item 12 for pool eval. Items 7 and 9 are recommended enhancers, not blockers.

### 11. Build A DAgger Iteration Orchestrator ~ partial — minimum viable landed

**Why:** Item 4 produces one DAgger round. The loop needs many rounds with stable bookkeeping, replay-buffer policy, and resume semantics. Manual loops drift on seeds, mix ratios, and checkpoint provenance.

**Work:**

- Orchestrator runs N iterations of: model rollout via `evaluateModelVsHeuristic.ts` / `headlessAiVsAi.ts` → planner relabeling on visited states → mix with replay buffer → retrain via `train_bc.py` → eval gate (item 13) → next iteration.
- **Replay buffer policy:** explicit cap (e.g., 200K rows), staleness cutoff (rows older than M iterations evicted), mix schedule (recent-iteration weight decays geometrically). Record ratios in manifest.
- **Anti-forgetting regularizer:** KL anchor to prior promoted checkpoint's policy distribution at a configurable weight; rehearsal subset of prior-iteration trace rows kept indefinitely. Toggleable so its contribution can be ablated.
- **Reproducibility tiering:** byte-identical resume restricted to CPU-only smoke. GPU/AMP runs target metric-equivalence within tolerance (final eval Wilson interval overlaps reference). Manifest records which tier was used.
- Per-iteration manifests record parent checkpoint, seed shards, opponent pool snapshot, mix ratio, planner config, eval result.
- Resumable from any prior promoted checkpoint without rerunning earlier rounds.

**Acceptance signal:**

- 3-iteration smoke completes end-to-end with one command and produces a chained manifest tree.
- Re-running from iteration K under the CPU-only deterministic tier reproduces iteration K+1 byte-identically; under the GPU tier, eval Wilson intervals overlap.
- Anti-forgetting ablation: removing the KL anchor on a planted forgetting-prone matchup measurably degrades that matchup's win rate.

### 12. Maintain An Opponent Snapshot Pool ↑ priority bump

**v4 priority bump note.** Item 11's orchestrator is landed but its eval gate is structurally incomplete without item 12: the loop promotion gate at the bottom of this doc requires per-opponent Wilson floors and cycling detection, which require the snapshot pool. Item 13's pinned thresholds also depend on item 12. Treat item 12 as the next-after-7 critical-path work.

**Why:** Even DAgger cycles if the model only ever plays the rule bot or its own current weights. Cycle-collapse and overfitting to one opponent are realistic failure modes.

**Work:**

- Snapshot every promoted iteration into a versioned pool with metadata.
- **Pool sampling:** default = prioritized fictitious self-play (PFSP) — opponent sampled with weight proportional to current model's loss rate against it. Falls back to uniform if PFSP weights are degenerate.
- **Retention:** last 8 promoted checkpoints + every 4th historical, capped at 24 total.
- **KL/feature drift reference:** distribution of the prior promoted checkpoint at matched seeds; flag if KL exceeds configured envelope.
- Track per-opponent win rate across iterations to detect cycling (any opponent monotonically losing across 3+ consecutive iterations triggers a halt).

**Acceptance signal:**

- Eval reports per-opponent breakdowns and pool-aggregate Wilson lower bound.
- Deliberately narrow pool (self-only) shows measurable regression vs the diversified pool over 3 iterations.
- Cycling detector trips on a planted RPS scenario.

### 13. Per-Iteration Eval Gate With Regression Rollback ~ partial — aggregate gate landed, per-matchup blocks on item 12

**Why:** Iterations must be promotable or rejectable on a fixed standard. Compounding regressions are otherwise invisible until the loop has diverged.

**Work:**

- Reuse the corrected rebaseline suite (item 1) plus the opponent-pool eval (item 12) and the best-response exploitation check from the promotion standard.
- **Promotion rule:** Wilson lower bound on pool aggregate must exceed prior promoted checkpoint's by ≥0pp (i.e., not regressed); no per-matchup Wilson lower bound drops by more than 5pp; fallback/no-op count must remain zero; best-response exploitation check has not declined >5pp across the last 3 iterations.
- Auto-rollback to prior promoted checkpoint on failure; surface failing metrics; record halt-after-2-consecutive-failures (consistent with the loop promotion gate at the bottom of this doc).

**Acceptance signal:**

- Planted bad iteration is rejected and rolled back without manual intervention.
- Manifest history shows monotonic promoted-checkpoint chain even when intermediate iterations fail.
- Halt-after-2 trigger fires on a planted double-failure fixture.

### 14. Compute Budget And Distillation Path ~ partial — targets grounded by item 0

**Why:** Planner relabeling is the cost driver. Without throughput targets and a distillation fallback, the loop is either too slow to iterate or stuck shipping a planner-bound policy at serving time.

**Work:**

- **Establish initial budget targets** as the first sub-task using item 0's measurements: target ≥200 planner-relabeled decisions/sec at full worker count, target iteration wall-clock ≤4h on the user's box. Record actual numbers per iteration.
- Track planner-decision cost and decisions-per-game so budget regressions are visible.
- **Distillation criterion:** if the promoted policy's pool-aggregate Wilson lower bound trails the planner-augmented policy by >3pp, run a distillation pass (model imitates planner-on-model-states) and re-gate.

**Acceptance signal:**

- Each iteration's manifest records throughput numbers and budget consumed.
- A distilled checkpoint reaches within 3pp of planner-augmented strength on the corrected suite at serving-time latency.

### 15. Exploration During Trace Generation And Action Coverage · pending

**Why:** DAgger inherently visits only states the current policy reaches. Without exploration, the state distribution collapses and rare actions (retreat, specific evolves, gust on opponent bench) atrophy.

**Work:**

- Temperature schedule on model action selection during trace generation (e.g., temperature decays from 1.5 → 1.0 across iterations).
- Epsilon-mix with the candidate ranker's `epsilon` mode for off-policy state coverage.
- Optional planner-branch counterfactual labeling: at a fraction of states, label not just the model's chosen action but a planner-explored alternative branch.
- Per-iteration action-coverage histogram: actions with <50 labels in the buffer are flagged; flagged actions trigger planner-driven counterfactual labeling next iteration.

**Acceptance signal:**

- Histogram recorded in every iteration manifest.
- Planted starvation scenario (deliberately seed a policy that never retreats) recovers retreat usage within 2 iterations once exploration is enabled.

### 16. Iteration-Level Debugging Tooling · pending

**Why:** When an iteration regresses, the orchestrator surfaces aggregate metrics but the operator has no per-decision inspection. Without this, root-causing a regression takes hours of manual JSONL grepping.

**Work:**

- Decision-diff CLI: given two checkpoints and a fixed seed range, dump the states where they disagree on action selection plus per-candidate scores.
- Regressed-game replay viewer: render a single-game trace as readable text (turn, phase, side, action, result delta).
- Per-feature attribution dump for a sampled regressed decision (which feature slot most influenced the policy logit difference).

**Acceptance signal:**

- Decision-diff CLI exists, takes two manifest hashes, outputs a deterministic diff file.
- Replay viewer outputs a complete trace for one game in <5s.
- Tools used at least once on a real regression (documented in `progress.md`).

## RL Loop Promotion Gate

"RL is working" means all of the following hold under the promotion standard above:

- The promoted policy's pool-aggregate Wilson lower bound exceeds the prior promoted policy's for 3 consecutive iterations.
- The promoted policy beats the rule bot at Wilson lower bound ≥75% with zero fallbacks/no-ops.
- Per-opponent Wilson lower bound against every snapshot in the pool from item 12 is ≥45%; no opponent shows a strict monotone decline (each of last 3 iterations strictly worse than the one before it on that opponent).
- Best-response exploitation check (depth-2/top-12) has not declined >5pp across the last 3 iterations.
- Throughput numbers from item 14 are within budget; a distilled deployable checkpoint sits within 3pp of the planner-augmented policy.

**Halt conditions:**

- Failing any of the above for 2 consecutive iterations halts the orchestrator and reverts to the last promoted checkpoint pending investigation.
- **Plateau (graceful stop, separate from regression halt):** pool-aggregate Wilson lower bound moves <1pp across 5 consecutive iterations *and* entropy floor stable. Stop the loop; ship the current checkpoint or escalate to F1.

## SL → RL Handoff Gate

**Items 0-14 are above the gate.** They must finish (or have a credible plan and partial landing) before the first DAgger round runs. Items 15 and 16 are also above the gate but can land between the first round and the multi-iteration sweep. Below this gate are SL refinements that RL self-play either subsumes or makes obsolete; only resume them if the RL loop fails to improve over three consecutive iterations or regresses against the opponent pool.

**v4 evidence-based amendment to the handoff:**

- The first real DAgger sweep (`runs/dagger-real-2026-05-08/`) showed a trained policy at 34% (iteration 0) and 28% (iteration 1, rolled back). The orchestrator works; the supervised half plateaus far below the rule bot at our current data scale.
- The handoff is therefore not "RL begins after DAgger reaches a strong policy" but "DAgger is the warm-start for RL." F1 (PPO) is no longer a gated polish step — it is the next critical-path workstream after items 7 and 12 land.
- Concrete v4 sequence after this commit: **(7) value head → (12) opponent pool → larger DAgger sweep with KL anti-forgetting → F1 PPO smoke**.

## Behind The Gate (Defer Unless RL Stalls)

### B1. Margin/Phase/Action-Kind Training Mixes

Low-margin down-weighting, adaptive CRN sample expansion, stratified phase/action-kind sampling. Real signal for SL strength but RL with advantage-weighted updates subsumes margin reweighting and learns phase distribution from reward. Resume only if DAgger plateaus and ablation shows margin signal is the bottleneck.

### B2. Entity-Aware Architecture (Cross-Attention, Asymmetric Towers)

Cross-attention from action token to board entities, asymmetric tower scaling, transformer destination architecture. All meaningful only on top of item 5b. RL can ship on a competent MLP-over-tokens; architecture sweeps are an optimization for a plateaued loop, not a path to one.

### B3. Procedural / Ability Polish Beyond Planner Needs

Per-card ability scoring sweeps, turn-goal broadening, rule-bot parity for `chooseHighestScoredAction`. RL self-play discovers these from reward. Anything not consumed by item 2 or item 9 belongs here.

## Gated RL Follow-Ups

### F1. Online PPO Plumbing (v4: critical path after items 7 + 12)

**v4 reframe.** v3 gated F1 on "DAgger producing a strong policy" — that gate is no longer informative. The DAgger orchestrator runs cleanly but the supervised cap is well below the rule bot. F1 is now the explicit path through the cap; the gating is rephrased as "F1 begins as soon as item 7 (value head) lands, with the orchestrator's promoted DAgger checkpoint as the warm-start". F2 (AlphaZero-style search-at-training-time) remains gated on F1 plateauing.

PPO is only worth the stability tax once DAgger has produced a policy that meaningfully beats the rule bot and the planner. Engineering beyond Option A is non-trivial.

**Work:**

- **Reward shaping spec:** dense per-turn reward (point delta) + terminal outcome with documented weighting.
- **Episode boundary:** one episode = one game; per-decision step is the time unit.
- **Behavior policy logging:** action probabilities and chosen-action log-probs at serving time so importance ratios are recoverable.
- **Legal-action mask handling:** masks recomputed at update time, not stored as immutable from rollout — re-derived from the state to avoid stale-mask bugs.
- **Buffer sizing:** on-policy buffer of N games per update, with N tied to item 14's throughput.
- **Advantage estimator:** GAE consuming the calibrated value from item 7; lambda and gamma recorded.
- **Stability controls:** KL-clip / KL-coef adaptive control, entropy target, gradient clip — all logged.
- **League sampling:** PFSP from item 12's pool layered on top.
- **HP sweep:** small grid over (lr, KL coef, entropy coef, GAE lambda, mix ratio) with seeds-per-config before the first multi-iteration run.
- Run as a separate orchestrator mode; do not retrofit into the DAgger loop.

**Acceptance signal:**

- Smoke run is stable across ≥10 PPO updates without KL blow-up or entropy collapse, with all controls recorded in the manifest.
- A held-out PPO checkpoint beats its DAgger parent on the corrected suite and on the opponent pool at Wilson lower bound.

### F2. AlphaZero-Style Search-At-Training-Time (Gated On F1) · pending

Only worth scoping if PPO + DAgger plateaus. The planner already does search-style work at label time, so the marginal return of MCTS-at-training-time is unproven for this game.

**Work:**

- Defer concrete work until items 11-14 and F1 have a recorded plateau against the rule bot and the snapshot pool.
- Pre-work allowed: a small MCTS prototype against the existing planner backend and a cost comparison vs item 14's distillation path.

**Acceptance signal:**

- A written go/no-go memo citing measured plateau metrics and a cost estimate vs distillation gains.

## Not Active Until Gated

- Larger supervised jobs on current labels.
- Multi-GPU/DDP before single-GPU training becomes the measured bottleneck for the orchestrator.
- Masked card/action pretraining unless labeled-data scaling stalls.
- Transformer destination architecture before embeddings, set encoders, cross-attention, and 100K+ useful rows exist.
