# AI Feature Engineering Backlog

Created 2026-05-15 from a three-angle subagent brainstorm: gameplay AI,
AI infrastructure/evaluation, and player-facing features.

This is the feature-engineering backlog around the current AI stack. It is
deliberately separate from `docs/ai-research-backlog.md`: the research backlog
tracks model-strength experiments, while this file tracks investigations,
tooling, UX, and production-hardening work that makes the search-wrapped AI
usable, explainable, and easier to improve.

## Current Constraints

- Production strength is the search-wrapped rollout-leaf MCTS path, not a
  stronger raw policy. The current production claim is rollout-leaf MCTS @ W6
  iter-2 with Wilson lower 0.6479 against the rule bot.
- Raw-policy SL / BC / DPO is closed as a forward line unless new coverage
  evidence explicitly re-opens it. Do not spend cycles on another isolated
  raw-policy tweak.
- The next data question is coverage and retention, not generic row count:
  prior mixed corpora retained only ~30%, and mcts-distill v1 failed from
  state-distribution mismatch.
- Side asymmetry is real enough to disclose. Existing rollout-leaf evidence is
  aggregate side-balanced strength, not a per-side guarantee.
- Player-facing AI work should reuse existing legal-action enumeration,
  `/ai/decide`, MCTS diagnostics, deck data, and AI telemetry before adding
  new model families.

## P0 - Highest Leverage

### 1. Corpus Retention And State-Coverage Audit

**Why:** This is the binding diagnostic before more training. It should explain
where rows are dropped, which state slices are undercovered, and whether gate
losses cluster in those slices.

**Approach:** Build one reproducible audit command comparing raw exported rows,
loader-retained samples, and eval/gate traces by source, side, seed range,
phase, action kind, legal-action count, turn bucket, points, board stage,
energy, hand/deck size, terminal distance, schema, card-id availability, and
loss/error buckets.

**Dependencies:** Existing exporters, Python dataset loader diagnostics,
`backend/src/sim/evaluateModelVsHeuristic.ts`, gate manifests.

**Acceptance:** JSON + markdown report with filter-reason counts, retained vs
dropped slices, train-vs-gate coverage buckets, top missing/overrepresented
slices, and a proposed source-mix target.

### 2. Canonical Benchmark Registry

**Why:** The project has many one-off gates. Future AI changes need named,
stable protocols so results can be compared without manual archaeology.

**Approach:** Define benchmark IDs such as `rulebot_side_balanced_n120`,
`rulebot_side_balanced_n1000`, `production_rollout_leaf_smoke`,
`cheap_value_head_fallback`, `side_split_probe`, and `forced_state_tactics`.
Each benchmark records seeds, sides, config, min games, thresholds, and
artifact paths.

**Dependencies:** `backend/src/sim/evalGate.ts`, `runs/*/gate.manifest.json`,
package scripts.

**Acceptance:** A checked-in registry consumed by at least one report or smoke
command; benchmarks distinguish regression gates from research metrics.

### 3. Forced-State Tactical Benchmark Suite

**Why:** Win rate is too blunt for visible quality regressions. A fixture suite
turns bad decisions into durable tests and explains where MCTS beats the rule
bot.

**Approach:** Create roughly 50 forced-start states grouped by missed lethal,
bad retreat, wrong energy attach, poor trainer sequencing, bad ability timing,
over-benching, failing to build a backup attacker, avoiding no-op/pass, and
prize-race errors. Record rule-bot action, oracle/MCTS preferred action, and
accepted alternatives.

**Dependencies:** State serialization/replay support, `stateFingerprint`, legal
action enumerator, MCTS endpoint.

**Acceptance:** Fixture manifest, expected action set per state, runnable
benchmark, and coverage mapping back to corpus slices.

### 4. Explainable Opponent Decisions

**Why:** MCTS decisions are opaque to players and to QA. Showing factual
decision context turns delays into understandable play and makes bad decisions
actionable.

**Approach:** Extend `/ai/decide` behind an opt-in flag to return top-N legal
actions, selected action id, visit share or score, simulations, decision time,
and deterministic labels derived from action kind, phase, card ids, KO/tempo
signals, and fallback reason. Surface this in the battle log or AI telemetry.

**Dependencies:** `backend/src/server.ts`,
`frontend/src/game/engine/ai-policy/actions.ts`,
`frontend/src/game/engine/ai-policy/mctsClient.ts`, `AiTelemetryPanel`.

**Acceptance:** One opponent decision can be inspected with chosen action,
top rejected actions, latency, search metadata, and a factual explanation that
does not claim hidden intent.

### 5. Coach My Turn

**Why:** This is the most direct player-facing AI feature. The engine can
already enumerate legal actions and `/ai/decide` already supports a model side.

**Approach:** Add an opt-in hint request for the player side. Return the best
action, one or two alternatives, and short labels such as "takes a point",
"unlocks next turn attack", or "evolves for HP/damage". Validate responses
against a state hash so stale hints cannot be applied after the board changes.

**Dependencies:** `/ai/decide`, legal-action formatter, match UI controls,
state fingerprinting.

**Acceptance:** Hint button returns legal, state-matched recommendations on a
current player turn; no autoplay path is introduced.

## P1 - Production Hardening And High-Value UX

### 6. Production MCTS Latency And Stability Harness

**Why:** Rollout-leaf is selected for strength but is slower. UI acceptance was
qualitative; deployment needs measured p50/p90/p95/p99 by state type.

**Approach:** Run fixed-seed decision probes recording wall time, node count,
legal-action count, phase/action kind, adaptive halt status, fallback status,
selected rank, and chosen action. Keep instrumentation lightweight enough not
to dominate latency.

**Dependencies:** `backend/src/sim/mcts.ts`, eval trace plumbing, production
ONNX server.

**Acceptance:** Report with latency percentiles and fallback/no-op counts by
bucket, plus raw trace artifact.

### 7. Side-Asymmetry Larger-N Audit

**Why:** The current production claim is aggregate side-balanced; player side
is directionally weaker. Larger-N evidence decides whether this is a tuning
target or just disclosed variance.

**Approach:** Run or script a larger side-balanced split for rollout-leaf and
the cheap value-head fallback. Bucket losses by phase, action kind, turn band,
legal-action count, and setup/first-mover features. If the gap separates, test
side-conditioned sim budgets.

**Dependencies:** Benchmark registry, eval gate, coverage audit.

**Acceptance:** Larger-N side report that states whether CIs separate and lists
the top contributing buckets.

### 8. Eval Manifest Comparator

**Why:** Research decisions currently require manually comparing manifests.

**Approach:** Add a report tool that takes two or more `gate.manifest.json`
files and emits aggregate WR/Wilson deltas, side splits, points, terminal
reasons, fallback/no-op counts, latency/progress timing where available, and
exact config differences.

**Dependencies:** Existing gate manifests.

**Acceptance:** Command-line comparator with markdown or JSON output; used in
the next research decision note.

### 9. AI Investigation Workbench

**Why:** Bad decisions should be reproducible and convertible into fixtures.

**Approach:** Promote `AiTelemetryPanel` from raw JSON to grouped decision
audits: phase, legal actions, chosen action, heuristic score, MCTS diagnostics,
fallback reason, latency, and export/import of a single state.

**Dependencies:** `frontend/src/app/AiTelemetryPanel.tsx`, AI telemetry,
`/ai/decide`, backend tests.

**Acceptance:** A suspicious UI state can be exported, replayed, and added to
the forced-state suite without hand-copying opaque JSON.

### 10. Missed Opportunity Review

**Why:** Post-turn coaching is less intrusive than live advice and can start
with high-confidence rule-based checks.

**Approach:** On player end turn, show at most one concrete missed opportunity:
missed KO, unused energy attach, obvious evolution, playable draw/search
trainer, or unused stadium. Start rule-based with legal actions; later compare
against MCTS deltas.

**Dependencies:** `EndTurnWarningModal`, match UI actions, legal-action
enumeration, battle log.

**Acceptance:** Dismissible post-turn notice fires only on high-confidence
misses and never blocks the normal turn flow.

### 11. Adaptive Difficulty Profiles

**Why:** Difficulty should be a transparent policy, not hidden random weakness.

**Approach:** Expose profiles such as Beginner = rule bot, Standard =
value-head/adaptive fallback, Expert = rollout-leaf. Later add visible
adjustment based on recent player results.

**Dependencies:** AI backend config, match mode/main menu, MCTS config wiring.

**Acceptance:** Player-selectable profile maps to explicit backend/search
configuration and persists locally.

## P2 - Measurement, Data Quality, And Deck UX

### 12. Decision Trace Schema Validator

**Why:** Training and eval JSONL crosses TypeScript and Python, and schema drift
already mattered in the 96-d/vs/110-d transition.

**Approach:** Validate training examples, decision traces, self-play rows,
outcome rows, and manifests. Fail fast on missing seed/source/episode fields,
schema-feature mismatch, invalid card vocab ids, malformed legal actions, and
absent result fields. Support `strict-current` and `legacy-inspect` modes.

**Dependencies:** `training/uma_ai/dataset.py`, `training/uma_ai/features.py`,
TS exporters.

**Acceptance:** Validator catches known legacy incompatibilities without
blocking read-only inspection of old artifacts.

### 13. Benchmark Artifact Index

**Why:** Canonical checkpoints and manifests are spread through `runs/`.

**Approach:** Generate a machine-readable index of run dirs, checkpoints, ONNX
files, feature schema, git SHA/dirty flag, gate status, Wilson lower, side
split, training metrics, parent checkpoint, and benchmark ID.

**Dependencies:** `*.manifest.json`, `events.jsonl`, checkpoint metadata.

**Acceptance:** Query by model family or benchmark ID returns the canonical
artifact and latest comparable result.

### 14. OOD And Coverage Drift Gate

**Why:** New cards or rule changes can push served states outside the model's
training/eval coverage.

**Approach:** Bucket observations into coverage signatures during sim/eval/UI
AI calls. Compare against a baseline corpus for unseen card ids, unseen action
kinds, legal-action count outliers, rare phase/action buckets, and feature
min/max violations. Start report-only.

**Dependencies:** Coverage audit, card vocab, observation builder.

**Acceptance:** Drift report flags state slices that are absent or rare in the
baseline corpus.

### 15. Deck Doctor

**Why:** Deck quality is player-facing and can use existing card/deck data
without new model research.

**Approach:** Analyze custom decks for basic/evolution counts, draw/search
density, trainer mix, energy-type mismatch, unsupported attack curves, disabled
cards, and collection ownership. Later add opening-hand and early-turn
simulation.

**Dependencies:** `shared/src/data/cards.json`,
`shared/src/data/premadeDecks.json`, deck browser helpers, collection data.

**Acceptance:** Deck UI shows 3-5 ranked, concrete fixes with affected cards
and severity.

### 16. Tutorial Scenarios With AI Feedback

**Why:** Fixed puzzle states teach the phase system better than generic text.

**Approach:** Create scripted states for attach timing, evolution, search
trainer use, retreat for lethal, bench protection, ability timing, and stadium
usage. Compare player action against expected or MCTS top action.

**Dependencies:** Engine state construction, AI combat scenario fixtures,
frontend modal/prompt patterns.

**Acceptance:** Each tutorial scenario has a smoke test proving its expected
action remains legal.

## P3 - Later Bets

### 17. Search Ablation Matrix

**Why:** Production strength comes from search, but robustness across search
knobs is only partially characterized.

**Approach:** Controlled matrix over leaf type, simulations, prior, CRN
samples, rollout steps, adaptive ratio, and side. Report the strength/latency
frontier, not only the best point.

**Dependencies:** Benchmark registry, latency harness, served model.

**Acceptance:** Ablation report with comparable Wilson and latency metrics for
each cell.

### 18. Rule-Bot-Covered Relabel Corpus Pipeline

**Why:** If the coverage audit confirms mismatch, the next data program should
target deployment-relevant states instead of generic self-play rows.

**Approach:** Generate states from raw/search policy vs rule bot, rule-bot
mirror, and side-balanced starts. Relabel exact states with rollout-leaf MCTS
or the strongest feasible oracle. Enforce retention >=80% and slice floors
before training.

**Dependencies:** Coverage audit, trace validator, MCTS relabel tooling.

**Acceptance:** Corpus manifest records source coverage, retained-row rate,
slice floors, and relabel oracle; no training run starts until these pass.

### 19. Post-Game "Why Did I Lose?" Investigator

**Why:** High-value coaching, but it needs reliable replay and explanation
infrastructure first.

**Approach:** Replay checkpoints at key turns and compare player actions to
coach/MCTS alternatives. Flag one to three swing moments such as skipped
attach, missed KO, wrong target, failure to bench, ignored evolution/search,
or overcommitted energy.

**Dependencies:** Replayable action logs or periodic state snapshots, hint
endpoint, post-game UI.

**Acceptance:** Post-game panel cites concrete turns/actions and avoids generic
or speculative advice.

### 20. Raw-Policy Reopen Gate

**Why:** Prevent accidental re-opening of the closed raw-policy line.

**Approach:** Add an experiment-template guardrail: any raw-policy SL
experiment must cite new coverage evidence, target a changed state
distribution, pre-register a threshold against the best raw baseline, and
include a side-balanced n>=1000 gate before it is active.

**Dependencies:** Documentation discipline and benchmark registry.

**Acceptance:** Future raw-policy proposals include this checklist or are
explicitly marked deferred.
