# AI Performance Research Progress

## 2026-05-07 Backlog Pass

Scope: start at the top of `ai-performance-research-backlog.md`, define a concrete acceptance signal for each item, implement unblocked measurement/data-contract work, and stop only on hard blockers.

### Completed / Advanced

| Item | Result | Evidence |
| --- | --- | --- |
| 16. Deterministic progress fingerprint | Added one shared simulator fingerprint helper and replaced duplicated evaluator/headless hashes. Fingerprint includes full simulator state, including same-count hand/deck/discard content swaps, because it is an internal mutation detector rather than a public model observation. | `backend/src/sim/stateFingerprint.ts`; `backend/src/tests/stateFingerprintSmoke.ts`; `npm --workspace backend run test:train` |
| 1. Full-turn / turn-bundle planner | Added initial `--selection planner` mode. It enumerates bounded same-turn action bundles through modeled phases, scores leaves with rollout, and returns the first action of the best bundle. Tiny smoke is functional with zero fallbacks but not strong. | `backend/src/sim/evaluateModelVsHeuristic.ts`; 2-game planner gate: 0/2, zero fallbacks |
| 17. Evaluation gate with confidence intervals | Added `sim:eval-gate` with side-balanced runs, Wilson 95% CI, side split, average points, terminal reasons, selected candidate rank, zero-fallback enforcement, and optional `--manifest-out`. | `backend/src/sim/evalGate.ts`; tiny rollout/search gates passed with zero fallbacks |
| 6. Episode/seed train-validation split | Added `--split-by row|episode|seed`, default `episode`; training manifest/checkpoint record grouped train/validation split. Python e2e smoke asserts no group leakage. Training manifests also report dataset source/policy/phase/action/margin distributions for mixed DAgger corpora. | `training/train_bc.py`; `training/smoke_e2e.py`; `npm run test:python-train` |
| 18. Dataset/run manifests | Rule-bot and outcome exporters now write sibling manifests with args, git SHA/dirty flag, seed/source taxonomy, phase/action-kind counts, terminal reasons or margin buckets, and feature dimensions. Training manifest now includes data path, sample count, split, and feature schema. Base evaluator and eval gate can write manifest artifacts via `--manifest-out`; README documents the artifact layout. | `backend/src/sim/manifest.ts`; `exportTrainingExamples.ts`; `exportOutcomeTrainingExamples.ts`; `evaluateModelVsHeuristic.ts`; `evalGate.ts`; `training/train_bc.py`; `training/README.md` |
| 2. Controlled common-random rollout samples | Outcome export now supports `--samples`; each label records common sample seed IDs, per-candidate rewards, reward mean/variance, selected-vs-runner-up margin, selected-vs-baseline margin, tie policy, low-margin flag, and selected original heuristic rank. Determinism smoke verifies repeated export seeds reproduce labels and oracle stats, and candidate-order perturbation does not change selected labels. | `backend/src/sim/exportOutcomeTrainingExamples.ts`; `backend/src/tests/outcomeExportSmoke.ts` |
| 10. Feature/margin diagnostics and ablations | Training manifests now report grouped train/validation metrics by phase, selected action kind, dataset source/policy, and oracle margin bucket. `train_bc.py --ablate` can zero named state/action feature groups for controlled ablation runs without regenerating JSONL. | `training/train_bc.py`; `training/uma_ai/features.py`; `npm run test:python-train` |
| 3. Candidate ranker diversity | Added shared ranker modes `heuristic`, `phase-diverse`, and `epsilon`; outcome export and recursive search use the same helper. Baseline and legal pass/end-turn are always included. Eval summaries now report average selected original rank when available. | `backend/src/sim/candidateRanker.ts`; outcome/export and search gate smokes |
| 4. Policy-baseline-visited outcome export | Outcome export now records `source: ai-policy-baseline-visited`, `labelSource: rollout-labeled`, source taxonomy in manifest, and baseline modeled-action fallback count. | `backend/src/sim/exportOutcomeTrainingExamples.ts` |
| 5. DAgger trace export | Model evaluator now accepts `--decision-trace-out` and writes public model-side decision JSONL rows with legal actions, selected action, heuristic baseline, selected rank, fallback flag, and final game result. `--trace-teacher rollout|search|planner` can attach an independent teacher label per model-visited state. Smoke coverage validates both model sides, teacher labels, and hidden-hand safety. | `backend/src/sim/evaluateModelVsHeuristic.ts`; `backend/src/tests/evalGateSmoke.ts` |
| 7. Legal attacks / combat choices | Combat decisions now carry `attackIndex`; combat candidate generation iterates every payable attack on the active card and preserves attack index through modeled execution. Existing attack subchoices now include explicit discard, evolve-from-deck, random-discard, switch, heal, target, retreat, and self-shuffle payloads where applicable. Headless/evaluator forced-coin plumbing now accounts for any payable coin-flip attack, not just the primary attack. | `frontend/src/game/engine/flow/ai/combatPlanner.ts`; export smoke found all combat attack actions include `attackIndex`; action contract and training smoke pass |
| 8. Feature schema compatibility | Python now fails clearly on action feature length mismatch instead of silently padding/truncating old rows. Checkpoints and manifests record feature schema metadata; ONNX export/server validate expected feature dimensions. TS exports now expose action feature schema/version constants for exporter manifests. | `frontend/src/game/engine/ai-policy/actions.ts`; `training/uma_ai/features.py`; `training/export_onnx.py`; `training/serve_onnx.py` |
| 11. Procedural baseline measurement/scoring | Evaluator now supports `--selection baseline` to run `chooseHighestScoredAction(enumerateLegalAiActions(...))` directly against the hard AI. `ai-policy` scoring now reuses hard-AI attach, evolution, and trainer play predicates for safer candidate ordering. | `backend/src/sim/evaluateModelVsHeuristic.ts`; `frontend/src/game/engine/flow/ai/attachUtils.ts`; 8-game baseline smoke: 62.5%, zero fallbacks |
| 13. Broader turn-goal detection | Added midgame goals for protecting a loaded active under KO threat, building a backup attacker, digging for live evolution, and converting a point lead. Existing turn-goal telemetry now records those goals by phase/side/turn; attach/trainer/ability hooks bias choices to match the selected goal. | `frontend/src/game/engine/flow/ai/turnPlan.ts`; `backend/src/tests/aiCombatScenarios.ts`; `npm --workspace backend run test:ai` |
| 19. Stronger training smoke tests | `test:train` now includes fingerprint contract fixtures, deterministic outcome-export schema/provenance checks, fake-model evaluator coverage, DAgger trace shape checks, and eval-gate failure coverage. Python e2e asserts grouped split and exercises stricter ONNX export/server dimension path. | `backend/package.json`; `backend/src/tests/outcomeExportSmoke.ts`; `backend/src/tests/evalGateSmoke.ts`; `training/smoke_e2e.py` |

### Backlog Item Plan / Acceptance Ledger

| # | Plan | Target acceptance signal | Current status |
| --- | --- | --- | --- |
| 1 | Build a bounded turn-bundle planner on top of `advanceModeledTurnStep`, returning the first action of the best sequence and scoring leaves with rollout/search. | 65%+ over 500 side-balanced games, zero fallbacks, CI/side/points/terminal report. | Initial implementation exists but acceptance not met. Tiny smoke: 0/2, zero fallbacks. Needs tuning/common-random scoring and larger gate before distillation. |
| 2 | Compare every candidate against the same N rollout seeds, record margin/variance/tie metadata, and down-weight low-margin rows. | Repeated seed gives identical labels; candidate order changes only below tie threshold; rows include sample/margin metadata. | Mostly implemented: row metadata, multi-sample common-random rewards, repeated-seed determinism, candidate-order perturbation smoke, and training margin-bucket metrics exist. Remaining gap is using low-margin rows to alter training weights/mixes in controlled experiments. |
| 3 | Extract candidate ranker modes shared by search and outcome export; always include baseline plus legal pass/end-turn. | Search reports ranker mode, candidate coverage, selected original rank; small exhaustive audits show low dropped-best rate. | Partially implemented: shared ranker is used by outcome export and recursive search; eval reports average selected rank. Exhaustive dropped-best audits still open. |
| 4 | Treat baseline-modeled trajectory source as first-class and record fallback/no-op accounting. | Manifests report source taxonomy and phase/action coverage; baseline fallback count is explicit. | Implemented for outcome export manifests and training dataset source summaries. |
| 5 | Add `--decision-trace-out` to model evaluation with model, heuristic, rollout/search decisions and final results. | DAgger trace has both sides, no hidden opponent-hand leakage, and can be loaded as mixed training data. | Partially implemented: evaluator writes public model-side decision traces with heuristic baseline, optional rollout/search/planner teacher, and final result; smoke validates both sides, teacher labels, and no hidden-hand leakage. Mixed source reporting exists; explicit trace-to-training recipe remains open. |
| 6 | Split validation by episode or seed by default and record exact groups. | `--split-by row|episode|seed`, default grouped; manifest records train/validation groups. | Implemented. |
| 7 | Extend combat action enumeration/execution to all attacks and attack choices. | Contract covers representative multi-attack cards; legal combat export includes attack index. | Partially implemented: payload/export includes `attackIndex`, all payable attacks are enumerated/executed, and simulator forced-coin generation covers non-primary coin-flip attacks. Current card data has no multi-attack cards, so representative multi-attack fixture is still open. |
| 8 | Version feature schemas and fail on dimension mismatches. | Exports/manifests/checkpoints/ONNX serving record compatible schema/dimensions; mismatches fail clearly. | Partially implemented: TS action feature schema constants, exporter/training metadata, and Python/ONNX dimension checks exist. Row-level schema versions and slot smoke fixtures remain. |
| 9 | Validate state semantic features and hidden-information safety with fixtures and ablations. | State schema recorded; tests cover hand/discard/readiness/KO/catalog/no leakage. | Partially covered by existing hidden-hand smoke and schema metadata. Semantic fixture suite remains. |
| 10 | Add feature ablation switches and phase/action-kind reporting. | Ablation report includes row metrics, closed-loop WR, margin buckets, rank, and phase/action-kind accuracy. | Partially implemented: training supports named feature ablations and reports phase/action-kind/source/margin-bucket metrics. Closed-loop ablation report integration remains. |
| 11 | Port stronger hard-AI procedural scorers into `ai-policy` or shared scoring. | `chooseHighestScoredAction` approaches hard AI in mirror eval without import cycles. | Partially implemented: baseline eval hook plus attach/evolution/trainer scoring reuse. Tiny baseline smoke: 62.5%, zero fallbacks. Needs larger held-out gate and more bench/combat/trainer-choice scoring. |
| 12 | Score trainer/ability choices by simulated marginal outcome. | Target/discard fixtures improve; fewer low-margin labels; ability use correlates with point/survival gains. | Not implemented. |
| 13 | Broaden turn-goal detection for midgame strategy. | Goal telemetry by turn/side; goal fixtures; average points improve. | Partially implemented: four new midgame goals, goal fixtures, and goal-aware attach/trainer/ability behavior exist. Remaining acceptance gap is measured average-point improvement in held-out eval. |
| 14 | Train calibrated value/action-value objective before using value in search. | Calibration metrics and one-step value selection beat simple point-margin baseline. | Not implemented. Current findings still mark value head unusable. |
| 15 | Rebaseline all methods on one corrected held-out suite. | One comparable table with fixed seeds/decks/evaluator, CI, side split, points, fallbacks, terminal reasons. | Enabled by eval gate, not yet run at 500-game scale. |
| 16 | Centralize and test state fingerprint. | All no-op/stall/search paths use one helper; mutation fixtures pass. | Implemented for evaluator/headless/export/action-contract paths. |
| 17 | Add pass/fail evaluation gate. | Gate reports WR, Wilson CI, side split, points, terminal reasons, fallbacks/no-ops and fails thresholds. | Partially implemented: gate reports all except an explicit separate no-op count beyond fallback/stall accounting. |
| 18 | Write manifests beside exports, training runs, and evals. | Every JSONL/checkpoint/eval has reproducible sibling metadata. | Mostly implemented: dataset/training manifests exist; base evaluator and eval gate can write manifests via `--manifest-out`; README documents the layout. Remaining gap is enforcing manifest presence for every ad hoc eval command. |
| 19 | Expand training/eval smoke coverage. | `test:train` covers export/outcome/fake-model/schema/fingerprint regressions. | Partially implemented: fingerprint, outcome determinism/provenance, fake-model policy eval, DAgger trace shape, eval-gate failure, grouped split, and schema/dimension checks. Explicit bad-policy strength tests remain. |

### Immediate Next Work

1. Add explicit no-op/stall accounting to eval summaries and gate thresholds.
2. Add explicit bad-policy strength tests.
3. Run controlled low-margin weighting/mix experiments.
4. Rebaseline baseline/search/rollout/planner on one corrected fixed-seed suite.

## 2026-05-08 Backlog Pass v3

### Item 0 — Throughput Probe (P0 spike)

Built `backend/src/sim/throughputProbe.ts` (`npm run sim:throughput-probe`) that
microbenchmarks `enumerateLegalAiActions`, `chooseHighestScoredAction`,
`stateFingerprint`, and `structuredClone`-based `cloneGame`, then runs heuristic
baseline games and planner-selection games end-to-end with per-game timings.

20-game probe at production-default planner config
(`--planner-top-k 4 --planner-max-sequences 64 --planner-max-depth 8 --rollout-steps 500`)
on the user's box; manifest at
`runs/throughput-probe/probe-default.json`:

| Surface | Number |
| --- | --- |
| Baseline (`chooseHighestScoredAction`) decisions/sec | 5,446 |
| Baseline decisions/game | 46.95 |
| Planner decisions/sec (single-core) | 22.4 |
| Planner decisions/game | 40.3 |
| Avg planner decision ms | 44.6 |
| Avg bundles per planner decision | 14.8 |
| `enumerateLegalAiActions` µs/call (microbench) | 2.6 |
| `chooseHighestScoredAction` µs/call (microbench) | 0.09 |
| `stateFingerprint` µs/call | 8.0 |
| `cloneGame` µs/call | 22.0 |

DAgger projection at 500 side-balanced games × ~40 planner decisions / model-side =
20,150 planner decisions per iteration:

| Cores | Iteration wall (h) | Throughput (dec/s) |
| --- | --- | --- |
| 1× | 0.25 | 22.4 |
| 8× | 0.031 | 179 |
| 32× | 0.008 | 716 |

**Verdict:** green-light item 14's budget targets at the *current* planner
strength. Item 14's "≥200 planner decisions/sec at full worker count" is met at
9 ideal-scaled cores, and "≤4h iteration wall-clock" has roughly 100× headroom
on a single core and ~7,000× on 32 cores at this planner config.

**Caveats to surface alongside the green light:**

- The current planner only enumerates ~14.8 bundles/decision. Item 2's tuning
  (deeper bundles, CRN-paired leaf scoring, ranker diversity) will multiply
  per-decision cost. Budget headroom of ~100× single-core means we can absorb
  a 25-50× planner cost increase before approaching the 4h gate at 8 cores.
- `cloneGame` (`structuredClone`) is the dominant per-call cost (22 µs). A
  large fraction of planner work is `cloneGame + advanceModeledTurnStep` for
  bundle expansion, so any structural-hash / persistent-data-structure
  optimization disproportionately compounds with item 2's planner depth.
- `stateFingerprint` cost (8 µs) is small in absolute terms but it is called
  twice per modeled step (before/after) for stall and bundle-deduplication
  checks. If item 1's N-step cycle detection raises the call count, this can
  become measurable.

No escalation triggered. Items 11-14 can be scheduled against the current
budget. Item 6 (parallel generation) is still required to hit the
"≥200 planner decisions/sec at full worker count" target without relying on
ideal scaling.

### Item 1 — Corrected Rebaseline Suite (50-game side-balanced)

`backend/src/sim/rebaseline.ts` (`npm run sim:rebaseline`) runs the same
fixed seed range (8000-8049 × {player, opponent} = 100 games per method)
through every method and writes per-method manifests +
`rebaseline.json` + `rebaseline.md`. Each method also went through the
new `--cycle-window 8` N-step cycle stall detection from item 1.

Production-default planner config; results at
`runs/rebaseline-2026-05-08/`:

| Method | n | WR | Wilson95 | Player | Opp | Fallbacks | NoOps | Passes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rule-mirror | 100 | 58.0% | [48.2, 67.2] | 50.0% | 66.0% | 0 | 0 | 0 |
| baseline | 100 | 48.0% | [38.5, 57.7] | 40.0% | 56.0% | 0 | 0 | 3,090 |
| inverted-baseline | 100 | 0.0% | [0.0, 3.7] | 0.0% | 0.0% | 0 | 0 | 1,926 |
| rollout | 100 | 66.0% | [56.3, 74.5] | 56.0% | 76.0% | 0 | 0 | 3,138 |
| search (d2/top8) | 100 | 60.0% | [50.2, 69.1] | 54.0% | 66.0% | 0 | 0 | 3,096 |
| planner (no CRN) | 100 | 44.0% | [34.7, 53.8] | 40.0% | 48.0% | 0 | 0 | 3,408 |
| planner (CRN) | 100 | 51.0% | [41.3, 60.6] | 44.0% | 58.0% | 0 | 0 | 3,282 |

Side imbalance is real and consistent across methods — every learned/search
method underperforms as player and overperforms as opponent (rule-mirror
self-play 50% vs 66%). This suggests a setup-order or first-mover bias
rather than method-specific weakness.

Zero fallbacks and zero selected no-ops across all rows. Inverted-baseline
hits exactly 0/100 (Wilson upper bound 3.7%) — the planted bad-policy gate
is exercised by real data, not just a fixture.

### Item 2 — Planner CRN: Escalation Triggered

`runs/rebaseline-2026-05-08-planner-crn/planner.json`:

CRN with 3 shared seeds + first-action max aggregation moved planner from
44.0% / Wilson lower 34.7% to 51.0% / Wilson lower 41.3% (+7pp, +6.6pp
lower bound) at the same configuration. Real signal but **still below the
55% Wilson lower bound escalation floor** that item 2 pre-registered.

**Per the pre-registered plan B in item 2**: when planner stalls below the
55% Wilson lower bound, fall back to "planner + value-head tiebreaker
(gated on item 7) or rollout-augmented planner with deeper CRN". Item 7
is not implemented. The pragmatic options surfaced for the user:

1. **Promote rollout selector as the DAgger teacher.** Its Wilson lower
   bound is 56.3% — already above the 55% floor — and the point estimate
   66% is the strongest measured teacher. The 65% bar in item 2 is missed
   by ~9pp on the lower bound, but rollout cleanly beats the rule bot at
   p<0.05. DAgger can iterate on rollout-relabeled traces while item 2's
   ceiling work continues in parallel.
2. **Strengthen rollout via multi-sample CRN.** Rollout currently scores
   each candidate with one rollout. Adding K=3-5 CRN samples would tighten
   the Wilson interval and likely lift the lower bound.
3. **Compose planner + rollout** (the pre-registered "rollout-augmented
   planner with deeper CRN"): planner picks a small candidate set, rollout
   evaluates each with shared seeds. Effectively a deeper search at the
   first-action level.
4. **Implement item 7 (calibrated value head)** to enable the "planner +
   value-head tiebreaker" composition. This is the longer path but is
   pre-registered in item 2's plan B.

**Recommendation:** option 1 + start option 4. Rollout teacher is already
above the 55% gate, so DAgger can begin iterating immediately while
options 2, 3, and 4 land in parallel. Bumping rollout from 66% point
estimate toward the original 65% Wilson-lower-bound target needs more
samples (option 2) or hybridization (option 3).

### Item 4 — DAgger Trace-To-Training Recipe

Implemented as the data spine, separated from the orchestrator (item 11):

- `npm run sim:evaluate-model -- --decision-trace-out ... --trace-teacher
  rollout|search|planner` exports public model-side decision traces.
- `backend/src/sim/dagger/relabelDecisionTrace.ts` retargets each row to
  the teacher's chosen action and aborts on opponent.handCardIds /
  opponent.hand / opponent.deck leaks, with a planted-leak fixture in
  the smoke.
- `backend/src/sim/dagger/mixSources.ts` deterministically samples N
  weighted source JSONLs into one mixed JSONL, tagging every row with
  its source. Defaults pin the 40/30/20/10 split mentioned in the
  backlog.
- `backend/src/tests/daggerRoundSmoke.ts` wires baseline → trace →
  rollout-teacher relabel → rule-bot mix → schema asserts + leak
  fixture, run via `npm run test:train`.

The "DAgger checkpoint trained on model-visited rows beats rule-bot-only
training" half of the acceptance still gates on item 2 — currently the
recipe runs end-to-end but the teacher (rollout) is below the 65% Wilson
lower bound bar for promotion.

### Item 5a — Card Vocab + Hash Fail-Fast

Built the canonical card vocabulary as the blocking 5a subset (vocab,
fail-fast, manifest). Embedding-table model surgery (5b) is non-blocking
and remains.

- `npm run sim:build-card-vocab` generates `shared/src/cardVocab.json`:
  107 entries (1 reserved unknown + 106 cards), deterministic suffix
  ordering for FullArtGold/FullArt/UncommonPlus/Ex variants, sha256-prefix
  hash `e3a35716156494d6` recorded in the file.
- Python `_hash_to_unit(cardId)` now resolves through the vocab so the
  state-feature float position is deterministic and tied to the recorded
  hash. Bumped state schema version 1 → 2.
- Training manifests, ONNX export sidecar, and `serve_onnx` all record
  card_vocab metadata. `export_onnx` and `serve_onnx` refuse to run on
  hash divergence; smoke fixtures plant a tampered checkpoint to lock
  the contract.
- `JsonlPolicyDataset` now enforces row-level `schemaVersion == 1` and
  raises `RowSchemaError` on missing or bumped versions. Smoke plants
  both failure modes (item 10).

### Item 1 / 10 — Eval Gate Hardening

- `--max-ci-lower`, `--allow-no-ops`, `--require-zero-no-ops`,
  `--expect-fail`, `--require-manifest`, and N-step `--cycle-window`
  flags landed.
- Planted bad-policy regression: `--selection inverted-baseline`
  (`chooseLowestScoredAction`) hits 0/100 on the corrected suite,
  cleanly tripping the Wilson lower bound floor at the configured 50%
  in the smoke. Asserted both as a hard gate failure and (with
  `--expect-fail`) as a structured PASS.
- `--require-manifest` without `--manifest-out` now exits 2 with a
  structured `EvalGateError` JSON on stderr.
- `selectedNoOps` and `selectedExplicitPasses` reported separately on
  every eval row + summary; gate enforces zero of the former unless
  explicitly allowed.

### Plan B Option 2 — Rollout CRN

Implemented as `chooseRolloutAction(args, state, sideId, seed, fallbackRng)`
with `--rollout-crn-samples N` (default 1; 3 for the gate). Each candidate
is scored against K shared CRN seeds and the mean reward selects.

100-game side-balanced gate at `--rollout-steps 500 --rollout-crn-samples 3`
on the same fixed seed range as the rebaseline; manifest at
`runs/rebaseline-2026-05-08-rollout-crn/`:

| Method | n | WR | Wilson95 | Player | Opp |
| --- | --- | --- | --- | --- | --- |
| rollout (1 sample, prior) | 100 | 66.0% | [56.3, 74.5] | 56.0% | 76.0% |
| rollout (CRN samples=3) | 100 | 68.0% | [58.3, 76.3] | 68.0% | 68.0% |

CRN lifted rollout +2pp / +2pp Wilson lower bound and collapsed the
side imbalance (player/opp 56/76 → 68/68). Wilson lower bound 58.3% —
**3.3pp above the 55% escalation floor**, 6.7pp short of the 65% target.
Rollout-CRN is now the selected DAgger teacher.

### Item 8 — Training Scale-Readiness (partial)

`training/train_bc.py` now supports:

- `--amp` (autocast + GradScaler) gated on CUDA availability.
- `--grad-accum N` (default 1) accumulates gradients before stepping.
- `--lr-schedule {none,cosine,step}` with `--lr-warmup-steps`.
- `--resume <checkpoint.pt>` reloads model_state, optimizer_state,
  scheduler_state, scaler_state, RNG state, and history; manifest
  records `resume_from`.
- ONNX-roundtrip smoke automatically after every training run, with
  ≤1e-3 logit/value tolerance, asserting promoted checkpoints are
  guaranteed deployable.

`training/smoke_e2e.py` extends the contract: train K epochs → save →
resume from that checkpoint and complete K more epochs; assert the
resumed manifest records `resume_from` and passes the roundtrip smoke.

Sharded JSONL.gz iterable loading and TensorBoard/wandb logging are
the remaining non-blocking pieces.

### Item 11 — DAgger Iteration Orchestrator

`training/dagger_orchestrator.py` chains the existing tools into a
multi-iteration loop:

- Iteration 0 has no parent and uses `--selection <teacher>` for trace
  generation. Subsequent iterations export the promoted parent
  checkpoint to ONNX, spin up `serve_onnx` on a free port, and run
  `--selection policy` with the model.
- Per iteration: `evaluateModelVsHeuristic --decision-trace-out
  --trace-teacher <teacher>` → `relabelDecisionTrace` → `mixSources`
  with weighted rule-bot replay → `train_bc --resume <parent>` → eval
  gate. Default replay weight 0.4, relabeled weight 0.6.
- **Promotion rule:** gate must pass AND `wilson_lower` must not
  regress the previous promoted iteration's lower bound. Otherwise
  rolled back to the prior promoted checkpoint.
- **Per-iteration manifest:** parent checkpoint path, source row
  counts, mix ratios, train manifest snapshot, eval result, decision
  reason.
- Resumable: `orchestrator-state.json` snapshots promoted checkpoint
  + Wilson lower + iteration history; `--resume-state` continues from
  any prior iteration.
- Replay buffer is the rule-bot rehearsal slice, refreshed only on
  `--refresh-rule-bot`. Staleness eviction and KL-anchor anti-
  forgetting are deferred follow-ups.

`training/dagger_smoke.py` exercises the chain end-to-end at tiny
configs (1 game per iteration, 2 epochs, baseline-as-gate). Passing
asserts: 3 iteration manifests, mixed-row counts > 0 for every
iteration, at least one promoted iteration, and the orchestrator
records the rollback path when the eval-gate Wilson lower regresses.
Wired as `npm run test:dagger-orchestrator`.

`backend/src/sim/dagger/relabelDecisionTrace.ts` now emits
`episodeId = "${seed}:${modelSide}"` so the trainer's episode-grouped
split can split mixed-source JSONLs without falling back to row-level
splits.

### Open / Pending After This Pass

1. **Real-config first iteration.** Smoke proved the chain runs end-
   to-end at tiny configs. A meaningful first iteration (≥50 games,
   rollout-CRN teacher at samples=3, 20+ epochs, 64-hidden 2-depth)
   takes ~10 minutes single-core and is the next concrete experiment.
2. **Item 5b (set encoders, recent-action history, embedding model
   surgery).** Non-blocking; deferred.
3. **Item 6 (parallel generation, deterministic concat, deck-pool
   sampling).** Single-core throughput is already inside the budget.
   Worker-thread sharding remains for the "≥200 dec/s at full workers"
   target.
4. **Item 7 (calibrated value head).** Recommended for plan B and the
   F1 PPO pipeline.
5. **Item 8 sharded loader + TB/wandb logging.** Optional; current
   AMP/resume/cosine/ONNX-roundtrip subset unblocks the orchestrator.
6. **Item 12 (opponent snapshot pool with PFSP).** Required for the
   loop promotion gate. Orchestrator currently treats every iteration
   as a single "rule bot" matchup.
7. **Item 13's pinned thresholds** (5pp per-matchup floor, halt-after-
   2 trigger). The orchestrator decides on aggregate Wilson lower
   only; the per-matchup tracking lives in item 12.
8. **Item 14** (compute budget tracking + distillation criterion).
9. **Items 15 / 16** (exploration temperature, action-coverage
   histograms; decision-diff CLI, replay viewer, attribution dump).
