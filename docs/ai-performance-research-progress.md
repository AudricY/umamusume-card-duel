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

### Real-Config 2-Iteration DAgger Run

`runs/dagger-real-2026-05-08/`. Configs: 25 games trace + 25 games eval
per iteration (50 side-balanced), rollout-CRN samples=3 teacher,
`--rollout-steps 200`, replay buffer 30 rule-bot games, 12 epochs
batch=32 hidden=64 depth=2, `--lr-warmup-steps 16` cosine.

| Iteration | Selection (trace) | Trace rows | Mixed rows | Trained policy WR | Wilson95 | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | rollout (CRN×3) | 2,375 | 3,555 | 34.0% | [22.4, 47.8] | promoted (floor 0%) |
| 1 | policy (iter-0 model) | 2,536 | 3,651 | 28.0% | [17.5, 41.7] | rejected (lower 17.5% < floor 22.4%) |

**Pipeline ran end-to-end:** 2 iterations, ONNX export, `serve_onnx` on
a free port, policy gate against rule bot, promote/reject decision,
auto-rollback on regression. Zero fallbacks and zero selected no-ops.

**SL ceiling reproduced.** Trained policy at 34% / 28% sits well below
the rollout teacher's 68% (Wilson lower 58.3%) and below the rule-bot
mirror baseline 58%. This matches the archived imitation cap from
`docs/archive/ai-research/ai-training-findings.md` (corrected trained policies 34-40%).
The orchestrator's promote/reject machinery works; the underlying
imitation gap is the same one the unified backlog already pre-
registered. RL self-play (item 11+ → F1 PPO) remains the path through
the cap.

**Hypothesis on the iter-1 regression:** with only 12 epochs and
~2.4k trace rows, the model under-fits the relabeled distribution, so
training again on a slightly different model-visited mix produces a
slightly different — and noisier — local optimum. Larger epochs/data,
KL-anchor anti-forgetting (item 11 follow-up), or a value-head
auxiliary (item 7) should reduce this iteration noise.

### Open / Pending After This Pass

1. **Larger DAgger configs** (50+ games, 30+ epochs, 5b embeddings or
   wider model) to see whether trained policy can approach the
   teacher ceiling at all under the corrected suite.
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

## 2026-05-08 v4.1 Sequence — Phases A/B/C/D Land

### Phase A — Items 7 + 18 (parallel)

**Item 7 (calibrated value head).** `training/calibrate_value.py` loads
a checkpoint, runs the value head over a held-out JSONL dataset, and
reports calibration vs. a logistic point-margin baseline. ECE (15-bin),
Brier, reliability-by-turn over `{1-3, 4-6, 7-9, 10+}` buckets, plus
max-bucket-ECE. The point-margin baseline fits `k` in
`P(win) = sigmoid(k * (own − opp))` via L-BFGS on a deterministic 50%
slice of labeled rows so the reported metrics are evaluated on a
separate slice from the fit.

Tiebreaker-grade gate: bootstrap percentile CI (1000 resamples) on the
per-row Brier lift `pm_brier − value_brier`; lower 2.5% > 0 means
Wilson-significant lift. The spec asked for a "Wilson-style" interval
but closed-form Wilson does not apply to Brier deltas, so the bootstrap
is the closest valid analogue and the JSON output documents the
substitution. GAE-grade additions emitted but non-blocking: monotone-
non-decreasing flag + Pearson correlation across populated bins; drift
envelope across N>=2 checkpoints; value/point-margin variance with an
unbounded flag.

`training/calibrate_smoke.py` generates a tiny rule-bot dataset, trains
a 4-epoch model, runs calibration, and asserts plumbing-level shape
(JSON keys, status enum, ECE/Brier in [0,1], CI not inverted).
Wired as `npm run test:calibrate-value`.

**Item 18 (F1 plumbing prep).** `serve_onnx` `/predict` returns
`actionLogProbs`, `actionProbs`, `selectedLogProb`, and
`behaviorPolicy: {kind, temperature}` alongside the existing
`logits`/`value`/`selectedIndex`. The masked log-softmax uses a `-1e9`
sentinel for masked positions so JSON.parse stays safe and
`exp(masked_lp) ≈ 0`. Behavior policy under greedy serving is the
softmax distribution; selection is argmax of that distribution. At PPO
iteration 0 the importance ratio is exactly 1; drift accumulates as
PPO updates the target.

`evaluateModelVsHeuristic.ts` captures behavior on every
`chooseModelAction` call, including a degenerate single-action snapshot
(`selectedLogProb=0`) on the early-return shortcut so PPO rows are never
missing the field. `relabelDecisionTrace.ts` forwards `behaviorPolicy`
into the relabeled training row when present upstream.

`docs/archive/ai-research/f1-design.md` fixes F1 defaults: reward shaping (per-step Δpoints
× 1/3 + terminal win × 1.0), one-game episode boundary, GAE λ=0.95
γ=0.99, KL clip ε=0.2, entropy bonus 0.005, gradient clip 0.5,
on-policy buffer of N games where `N × decisions_per_game ≈ 32K`
(default N=800), HP sweep grid over `(lr, KL coef, entropy coef, GAE λ)`
with 3 seeds per cell. Smoke acceptance: ≥10 PPO updates without KL
blow-up (≤0.05 per update) or entropy collapse (≥30% of warm-start
entropy).

### Phase B — Item 12 (opponent snapshot pool with PFSP)

`training/opponent_pool.py` adds a versioned pool with PoolEntry
metadata (iteration, checkpoint path, Wilson lower at promotion, value
mean drift, timestamp). Retention keeps the last 8 promoted entries
plus every 4th historical, hard-capped at 24. PFSP weights are
`max(0.05, 1 − p_i)` with uniform fallback. Cycling alarm trips on
strict monotone decline of an opponent's last 3 win rates.

`evaluateModelVsHeuristic.ts` accepts `--opponent-model-url`: the
non-model side consults a second served checkpoint via `/predict`
instead of advancing through the rule bot. Single-action shortcut and
any fetch/parse error fall back silently to the rule bot.
`evalGate.ts` and `node_bridge.run_evaluator/run_eval_gate` thread the
flag.

`dagger_orchestrator.py` snapshots promoted checkpoints into
`<out>/pool/iter-NNN/checkpoint.pt`, persists the pool, and runs
`--selection policy --opponent-model-url` against each pool member when
`--pool-eval-games > 0`. Per-iteration manifest gains `pool_evals`,
`pool_aggregate_wilson_lower`, `cycling_alarm`. Pool eval is non-
blocking on the primary promote/reject — item 13's residual consumes
the metrics in the next phase.

### Phase C — Item 13 residual (per-matchup floor + halt-after-2)

`compute_matchup_floor_violations` walks prior iteration records'
`pool_evals`, takes the most-recent prior Wilson lower per opponent,
and flags any current opponent whose Wilson lower dropped by more than
`--per-matchup-drop-tolerance` (default 5pp). First-time matchups have
no prior to compare against and never trigger.

`decide_promotion` now consumes matchup violations: any non-empty list
vetoes promotion citing the worst-dropping opponent. Halt-after-2:
`OrchestratorState` tracks `consecutive_failures` / `halted` /
`halt_reason`. Reaching 2 sets halted=True and the main loop exits
cleanly before the next iteration; successful promotion clears the
counter.

### Phase D — Item 17 (KL-anchor anti-forgetting + pilot)

**KL-anchor anti-forgetting.** `train_bc.py` accepts
`--kl-anchor-checkpoint <path>` and `--kl-anchor-weight <float>`. The
anchor is loaded as a frozen no-grad model; each batch computes a
mask-aware KL(anchor || target) penalty added to `policy_loss +
value_weight * value_loss`. Per-epoch `kl_loss` is recorded in history
and the manifest's `training_kwargs` persist anchor path and weight.

`dagger_orchestrator.py` automatically threads
`--kl-anchor-checkpoint <prior promoted>` and
`--kl-anchor-weight <w>` to each iteration's `train_bc.py` invocation
when `--kl-anchor-weight > 0`. Per-iteration positional override via
`--kl-anchor-weights "0.0,0.1,0.5"` so item 17's mandated off/low/high
ablation runs inside one orchestrator invocation.

**Pilot — recipe-bug fix verified.** Two-iteration pilot at
`runs/item17-pilot/`:

| Iter | Selection (trace) | Trace rows | Mixed rows | Trained policy WR | Wilson95 | KL weight | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | rollout-CRN×3, **rollout-steps 500** | 525 | 749 | 20.0% | [8.1, 42.0] | 0.0 (no parent) | promoted |
| 1 | policy (iter-0) | 525 | 749 | 35.0% | [18.1, 56.7] | 0.1 | promoted |

Pipeline at item-17-shaped config (rollout-steps=500, rollout-CRN
samples=3, KL anchor on with weight ablation) ran end-to-end with zero
fallbacks and zero selected no-ops. Iteration 1's training manifest
records `kl_anchor_checkpoint=iter-000/checkpoint.pt` and
`kl_anchor_weight=0.1` as expected. Trade-off recorded: at the tiny
n=10 eval used here, Wilson half-width is ≈30pp so the iter-on-iter
+5pp uplift is not significant on its own; the pilot validates only
the plumbing.

**Full item-17 sweep — reproducible command.** Wall-clock estimate
~6h on the user's box (single core). Run as:

```bash
training/.venv/bin/python training/dagger_orchestrator.py \
  --out-dir runs/item17-2026-05-08 \
  --iterations 3 \
  --games 250 \
  --max-steps 500 \
  --teacher rollout \
  --rollout-steps 500 \
  --rollout-crn-samples 3 \
  --replay-games 100 \
  --epochs 30 \
  --batch-size 32 \
  --hidden-dim 64 \
  --depth 2 \
  --eval-games 250 \
  --kl-anchor-weight 0.1 \
  --kl-anchor-weights "0.0,0.1,0.5" \
  --pool-eval-games 20
```

`--eval-games 250` × 2 sides = 500 side-balanced eval games per
iteration. KL ablation runs the off/low/high sweep across the three
iterations. Pool eval at 20 games per opponent populates item 13's
per-matchup floor inputs. Pre-registered escalation: trained policy
Wilson lower ≤45% across all 3 iterations *and* iter-on-iter monotone
improvement <2pp graduates the v4 reframe ("F1 is the path through the
cap") from working hypothesis to finding. The progress entry below
this line will be appended once the sweep runs.

### Phase D status

- KL anchor + weight ablation plumbing validated on the pilot.
- Recipe-bug fix (rollout-steps=200 → 500) is now the orchestrator's
  configured default for the documented sweep.

### Phase D outcome — item 17 sweep, take 1 (2026-05-08, INVALID)

Sweep at `runs/item17-2026-05-08-INVALID-warmstart-bug/` recorded three
iterations with Wilson lower 0.30/0.28/0.31. Cross-checking via
observability stage 1 surfaced that **iter-1 and iter-2 trained zero
epochs**: the previous `dagger_orchestrator.py` passed `--resume` to
each train_bc invocation; `--resume` loaded `next_epoch=26` from the
parent's 25-epoch state, so the loop `for epoch in range(26, args.epochs+1)`
with `args.epochs=25` ran nothing. All three iterations' model weights
were bitwise identical (max-diff = 0.0); ONNX exports shared one md5.
The "Wilson lower" differences were pure eval-gate noise on one model.

**Fix landed (eb4f4f5).** `train_bc.py` now accepts
`--init-from-checkpoint` (loads `model_state` only, fresh
optimizer/scheduler/epoch counter), `dagger_orchestrator.py` switched
to that flag, and `dagger_smoke.py` asserts iter-0/1/2 checkpoints are
not byte-identical to catch any future regression. The previous run
dir is preserved as `runs/item17-2026-05-08-INVALID-warmstart-bug/`
for forensics.

**Observability paid for itself.** Without per-epoch events, the
zero-train bug would have survived to F1 (where the warm-start
checkpoint is the bedrock).

### Phase D outcome — item 17 sweep, take 2 (2026-05-11)

Corrected sweep at `runs/item17-2026-05-11/` — same compute-scaled
config (30 games trace, 100-game eval, 25 epochs, KL weights
"0.0,0.1,0.5") but with the fixed `--init-from-checkpoint` warm-start.
Per-epoch events confirm 25 epoch-records per iteration; pairwise
checkpoint weight max-diff is non-zero (~0.05 between iter-0 and
iter-1).

| Iter | Selection | Win rate (n=200) | Wilson95 | KL weight | Train loss | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | rollout-CRN×3 | 34.5% | [0.283, 0.413] | 0.0 (no parent) | 0.4403 | promoted |
| 1 | policy (iter-0) | 29.5% | [0.236, 0.362] | 0.1 (anchor iter-0) | 0.4076 | rejected (-4.65pp Wilson lower) |
| 2 | policy (iter-0) | 37.5% | [0.311, 0.444] | 0.5 (anchor iter-0) | 0.4308 | promoted (+2.83pp over iter-0 floor) |

**Pre-registered escalation re-evaluation.**

- *Wilson lower ≤45% across all 3 iterations*: **TRIPPED** — max
  Wilson lower observed is 0.311; the 0.45 threshold is unreached
  by every iteration. iter-2's Wilson upper (0.444) is in striking
  distance of the threshold but does not cross it.
- *Iter-on-iter monotone improvement <2pp*: **mixed signal.** Strict
  per-step reading: iter-0→iter-1 was −4.65pp (qualifies as <2pp);
  iter-1→iter-2 was +7.48pp (does NOT qualify). The net iter-0→iter-2
  improvement is +2.83pp on Wilson lower (or +3pp on win rate). At
  n=200 the Wilson half-width is ≈5pp, so the iter-0/iter-2 win-rate
  CIs overlap and the improvement is not statistically significant.

**Finding — v4 reframe partially confirmed.** At this codebase's
SL-warm-start configuration:

1. The supervised cap is real and lands at ~30% win rate (Wilson
   lower 0.28–0.31). All three iterations occupy this band; the
   absolute ceiling criterion (<45%) is unambiguously met.
2. KL-anchored DAgger with the right weight schedule (0.5 here) can
   recover from a single bad iteration (iter-1 regressed under
   weight 0.1; iter-2 climbed back +7.48pp under weight 0.5). This
   refutes the strict "DAgger is completely stuck" reading of v4.
3. Net progress over 3 iterations (+2.83pp Wilson lower) is small
   relative to the gap to 45% (∼14pp) and is inside the noise
   envelope at n=200. F1 (PPO) remains the planned path through the
   ~30% cap; on this evidence DAgger is doing useful warm-start work
   but is unlikely to close the gap on its own at reasonable
   iteration counts.
4. **Secondary finding — KL anchor weight matters.** Iter-1 (KL=0.1)
   regressed; iter-2 (KL=0.5) recovered. The previously-defaulted
   weight 0.1 may be too low for this codebase. A future ablation
   should test KL=0.5 from iter-1 onward.

**Caveats.** Compute-scaled config: 30 games trace and 100-game eval
per iteration, not the documented 250+250. n=200 Wilson half-width is
~5pp, so iter-on-iter deltas of ±5pp are not significant. Three
iterations is the minimum sample size for the escalation criterion;
more iterations would clarify whether iter-2's +7.48pp is real
recovery or noise. The pool-eval channel was disabled
(`--pool-eval-games 0`) for wall-clock reasons; per-matchup floors
remain unexercised at scale.

**Artifacts.** `runs/item17-2026-05-11/orchestrator-state.json`;
per-iteration manifests at `iter-{000,001,002}/iteration-manifest.json`;
per-epoch loss events in `events.jsonl`; TensorBoard event files under
`tb/iter-{000,001,002}/`; live dashboard at
`http://127.0.0.1:5000/run/item17-2026-05-11` after starting
`training/observability_app.py`.

### Phase E — F1 PPO smoke

Blocked on item 17's sweep producing a warm-start checkpoint. Defaults
fixed in `docs/archive/ai-research/f1-design.md`. Implementation
(`training/ppo_orchestrator.py`, stochastic serving mode,
`training/f1_hp_sweep.py`) tracked under F1 in the active backlog.

### Phase F — F1 PPO first production run (2026-05-11)

**Target declared:** F1 promoted checkpoint with Wilson lower ≥ 0.40 (10pp daylight past the DAgger 0.31 ceiling).

**Phase 1** (verification): 2 iters × 5 games × 200 max-steps × 20-game eval. Plumbing passed — checkpoint weights changed between iters, all events fired, iter-1 point-WR 42.5% (Wilson [0.285, 0.578] at n=40). Encouraging but uninformative at small n.

**Phase 2** (production scale): 5 iters × 30 games × 500 max-steps × 100-game eval × f1-design defaults (lr=3e-5, entropy=0.005, λ=0.95, clip=0.2, T=1.0). Result: target NOT met.

| Iter | Wilson lower | Win rate (n=200) | Wilson95 | Decision |
| --- | --- | --- | --- | --- |
| 0 | 0.2920 | 35.5% | [0.292, 0.423] | promoted (warm-start) |
| 1 | 0.2361 | 29.5% | [0.236, 0.362] | rejected |
| 2 | 0.3109 | 37.5% | [0.311, 0.444] | promoted |
| 3 | 0.2407 | 30.0% | [0.241, 0.367] | rejected |
| 4 | 0.2315 | 29.0% | [0.232, 0.356] | rejected → halt-after-2 |

**Max Wilson lower 0.3109 — identical to the DAgger ceiling from item 17.** PPO did not break through.

**Diagnosis from observability:**

- **PPO updates were tiny.** Max checkpoint weight diff between iterations: 0.000238. (DAgger sweep had diffs of order 0.05 between iterations.) The policy is barely moving.
- **KL per minibatch: 0.001–0.02** — well below the 0.2 clip ceiling. Not blocked by the clip; the gradient signal is just small.
- **Ratio mean: 0.99–1.00** across all 20 minibatches. The behavior policy ≈ target policy at every update; no learning happens because the importance-weighted advantage is ≈ 0 × advantage.
- **Root cause: warm-start entropy is ~0.18 nats per decision.** Over ~10 legal actions per step that's `exp(0.18) ≈ 1.2` effective actions — the DAgger-trained policy is nearly deterministic. Stochastic Gumbel-max at T=1.0 over a near-degenerate distribution produces near-greedy trajectories. Behavior ≈ target ⇒ ratio ≈ 1 ⇒ surrogate gradient ≈ 0. PPO has nothing to push against.

**Phase 3a (aggressive: T=2, lr=1e-4, entropy=0.05, ppo-epochs=4):** weights moved 10× more (max diff 0.003 per iter vs 0.0002), KL ranged into 0.04 per minibatch, entropy widened from 0.18 to 0.28. But gate WR: 34.5% → 30% → 32.5% — slight regression, halted at iter-2. The policy moves but doesn't find better argmax decisions.

**Phase 3b (extreme: T=2.5, lr=1e-3, entropy=0.1, clip=0.3):** weights moved another 10× (max diff 0.027 per iter). KL up to 0.054 per minibatch. Entropy widened to 0.37. Gate WR: 33.5% → 28.5% → 33% — bigger regression at iter-1, halted at iter-2.

**Pattern across three PPO phases:**

| Phase | Max weight diff/iter | Avg KL/minibatch | Best WR | Δ vs warm-start |
| --- | --- | --- | --- | --- |
| 2 (defaults) | 0.0002 | 0.011 | 37.5% | 0pp |
| 3a (aggressive) | 0.003 | 0.017 | 34.5% | -3pp |
| 3b (extreme) | 0.027 | 0.018 | 33.5% | -4pp |

**Finding — F1 target unreachable at current compute scale.** The PPO infrastructure is sound: weight diffs scale monotonically with HP aggressiveness exactly as expected. But the gradient signal at 30–60 games/update is too noisy to find a *better-than-warm-start* policy. Phase 2 doesn't move the policy enough to learn; phases 3a/3b move it but into worse territory (walks off the SL local optimum without finding higher ground).

The f1-design spec calls for **800 games/update** for production. We ran at 1/27th that. Going to 800 games × 500 max-steps would be ~20 hours wall-clock per PPO iteration on this machine — infeasible in an interactive session.

**Recommended pivots before declaring F1 a wrong-tool conclusion:**

1. **800-game/update overnight run** — the spec was specific about buffer size; honoring it is the cleanest test of "can F1 break this cap". Single iteration takes ~20h; a 3-iter run is a weekend. Practical if scheduled.
2. **Self-play with opponent pool** — switch rollout opponent from rule-bot to a PFSP sample of prior promoted checkpoints. Item 12 infra exists; PPO orchestrator currently uses `--opponent-model-url` unset (rule-bot default). Could give richer reward signal than always-vs-rule-bot.
3. **Improve the warm-start first** — DAgger at 30% WR may simply not be a strong enough starting point. Larger SL run (more games, more epochs) might land at 40% which gives PPO a meaningful gradient.
4. **Reward shaping refinement** — Δpoints × 1/3 + ±1 terminal might miss strategic depth. Adding shaping for energy attachment, retreat decisions, etc. could expose more learnable signal.

**Path of least regret:** option 2 (self-play). The infra is built (item 12), and a single self-play F1 run is comparable in wall-clock to phases 2/3a — it directly tests whether opponent diversity is the missing ingredient before paying for option 1's huge compute.

### Phase G/H — F1 PPO at spec scale (800 games/update)

The 20h-wall-clock estimate above was wrong by 100×. Phase 2's actual wall-clock was 100s, not 50 min — the simulator is much faster than I'd assumed. That made spec-scale runs cheap (~5-6 min each), so the next two phases honored the f1-design spec buffer size.

**Phase G** — 800 games/update × 3 iters × spec defaults (lr=3e-5, entropy=0.005, T=1.0, ppo-epochs=1, clip=0.2). 328s wall-clock.

| Iter | Wilson lower | WR | Decision |
| --- | --- | --- | --- |
| 0 (warm-start eval) | 0.3156 | 38.0% | promoted |
| 1 | 0.2407 | 30.0% | rejected (-7.5pp) |
| 2 | 0.2920 | 35.5% | rejected (-2.4pp from floor) → halt |

KL per minibatch averaged 0.009 — *identical to phase 2's 30-game run* despite 27× more data. Reason: minibatch reduction averages over more transitions, but the lr × gradient magnitude is unchanged. The spec defaults are conservative at spec scale; they're not designed to extract more signal from more data.

**Phase H** — 800 games/update × 3 iters × aggressive HPs (lr=1e-4, entropy=0.05, ppo-epochs=4, T=1.0). 340s wall-clock.

| Iter | Wilson lower | WR | Decision |
| --- | --- | --- | --- |
| 0 (warm-start) | 0.2826 | 34.5% | promoted |
| 1 | 0.2639 | 32.5% | rejected (-1.87pp) |
| 2 | 0.3109 | 37.5% | promoted (recovered) |

iter-2 promoted at Wilson lower **0.3109 — exactly the DAgger iter-2 Wilson lower from item17-2026-05-11**. After iter-1 perturbed the policy downward, PPO's next gradient step pulled it back to the same local optimum. The run did not halt (only one rejection between two promotions).

### F1 Phase summary — five PPO sweeps, one clear finding

| Phase | Buffer | HP profile | KL/mb avg | Weight max-diff | Best WR | Best Wilson lower |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 30 | spec | 0.011 | 0.0002 | 37.5% | 0.3109 |
| 3a | 30 | aggressive | 0.017 | 0.003 | 34.5% | 0.2826 |
| 3b | 60 | extreme | 0.018 | 0.027 | 33.5% | 0.2732 |
| G | 800 | spec | 0.009 | 0.0002 | 38.0% | 0.3156 |
| H | 800 | aggressive | 0.008 | 0.003 | 37.5% | 0.3109 |
| **L** | **800** | **aggressive + 5-signal reward shaping (linear decay)** | **0.010** | n/a | **39.8% (iter-2)** | **0.3560 (iter-2)** |

**Every well-behaved config (2, G, H) lands the trained policy at Wilson lower 0.31. Every aggressive config that actually perturbs the policy (3a, 3b) lands lower. Phase L (reward shaping) is the first F1 PPO config to materially exceed 0.31 — landing at 0.3560, +4.5pp over the prior ceiling without crossing 0.40.** The 0.31 ceiling is identical to the DAgger warm-start's Wilson lower under the unshaped reward. PPO is performing correctly and finding that **the warm-start is the highest-return policy reachable from itself** under stochastic Gumbel-max with the current reward shape.

**Root mechanism (now confirmed across configs):**

1. Warm-start entropy ≈ 0.18 nats per decision → stochastic policy ≈ greedy policy.
2. Behavior log-probs ≈ target log-probs ⇒ importance ratios stay at ~1.00 across every minibatch.
3. PPO surrogate gradient ∝ (ratio - 1) × advantage ≈ 0 × advantage = 0.
4. The entropy bonus widens *probabilities* but doesn't *flip argmax decisions*, which is what the gate measures.
5. With the current ±1 terminal + Δpoints×1/3 reward shape, the local optimum at WR ≈ 35–38% is the highest-return policy in the neighborhood of the warm-start.

**F1 target 0.40 not yet reached from the item17-2026-05-11 warm-start** under the PPO mechanism active at the time of this post-mortem. PPO can match the SL cap (phase H iter-2) but cannot exceed it under the existing reward shape. [Update 2026-05-14: R15.S3 (Phase L) lifted the rule-bot Wilson lower from 0.3109 (phase H) to **0.3560** by adding five per-step shaped signals — +4.5pp absolute over the prior F1 ceiling, missing the 0.40 bar by only 4.4pp and landing above the falsification band. The "NOT REACHABLE" framing was correct under the *unshaped* reward mechanism but is qualified once the reward axis is allowed to move; the branch is alive and the next attempt is a coefficient-scaling follow-up on the same axis. See "Phase L — F1 PPO + reward shaping" below. Phase M follow-up (1.75× coefs) landed iter-2 at **0.3580**, within Wilson noise of Phase L (Δ +0.002) — the pre-registered "coef saturation" outcome fired; next single-axis move is signal-mix or shaping-schedule change. See "Phase M — F1 PPO + reward-shape coef-scaling follow-up" below. Phase N follow-up (constant schedule, 1.0× coefs) landed iter-2 at **0.3677** (WR 41.0%, n=500) — **+1.2pp over Phase L, the new F1 rule-bot ceiling on record**, but still 3.2pp short of 0.40. With three configurations of the same 5-signal mix now landing 0.356 / 0.358 / 0.368 at iter-2, coef magnitude and schedule axes both moved iter-2 by ≤+1pp — **the binding constraint is the signal set itself**, not magnitude or schedule. See "Phase N — F1 PPO + constant reward shaping" below. **Phase O' capstone (2026-05-14, R15.S3 BRANCH CLOSED):** the signal-mix axis was tested by dropping the 2 weakest of the 5 signals (bench-energy and retreat — both effectively dead in Phase N attribution) and scaling the 2 dominant signals (active-energy 0.02→0.03, throughput 0.02→0.03). Iter-2 Wilson **0.3677 — identical to Phase N's 0.3677 to 4 decimal places** (Δ +0.000pp). With **three single-axis moves** (coef magnitude L→M Δ +0.002, schedule L→N Δ +0.012, signal mix N→O' Δ +0.000) now exhausted inside the 5-signal observation-delta family, the R15.S3 reward-shape **branch is closed at iter-2 Wilson 0.368 ± 0.001**. The post-mortem framing has been correspondingly upgraded: the reward-shape axis has been **fully explored and converged**; the remaining gap to 0.40 is now **categorical — needs a different information source, not more tuning of the existing observation-delta signals**. Surviving F1-line candidates require pulling information from a *different source*: value-head tempo signal (blocked on TS-side ONNX/trace instrumentation, queued as `r15-s3-value-head-trace-instrumentation`), R7 (multi-teacher labels — SL warm-start rebuild), or R8 (DPO — PPO replacement). See "Phase O' — F1 PPO + reduced signal-mix (R15.S3 branch closeout)" below. **Phase O (2026-05-14, value-head tempo signal at coef 0.05):** the value-head trace instrumentation unblocked the original Phase O plan; coef 0.05 was tested and iter-2 landed at Wilson **0.3502** — **1.8pp regression vs Phase N's 0.3677**. Iter-1 rejected at 0.2768 < iter-0 0.2845 floor (same over-shape pattern as Phase M); iter-2 promoted from iter-0 parent. Mechanism diagnosis: the value-head Tanh output gives ±2 per-step delta range; at coef 0.05 the per-step shape contribution is ±0.1, yielding a per-game total of ±6 across ~60 steps — **20× the ±0.3/game design budget**. Signal is correctly wired but magnitude is dominantly loud. Phase P launched as `runs/R15-S3-value-head-tempo-low/` with coef 0.01 (5× smaller) as the natural follow-up — if Phase P also regresses, the value-head-delta signal mechanism is genuinely a wrong shape (pivot R7/R8); if neutral at ~0.368 the signal is redundant; if ≥0.40, first F1 success. See "Phase O — F1 PPO + value-head tempo signal, coef 0.05" below. **Phase P (2026-05-14, value-head tempo signal at coef 0.01, R15.S3 BRANCH GENUINELY CLOSED across both axes):** the 5× lower coef was tested and iter-2 landed at Wilson **0.3463** — **2.1pp regression vs Phase N's 0.3677, AND 0.4pp worse than Phase O's 0.3502**. Trajectory was clean (no rejections; iter-0 0.2730 → iter-1 0.2845 → iter-2 0.3463, all promoted; `promoted_iterations: [0, 1, 2]`, `halted=false`). At coef 0.01 the per-game value-head shape contribution drops to ±1.2 (4× the ±0.3/game budget, ~5× closer to budget than Phase O's ±6) and iter-0 `shape_attribution.value_head` drops from 26.01 to **5.03** (~5× smaller, dimensional check on the coef ratio). **Both magnitudes regressed — lowering the coef did not help; if anything it hurt slightly more.** The value-head-delta signal mechanism is **wrong-shape, not wrong-magnitude** — at coef 0.05 the signal is loud and over-shapes the policy; at coef 0.01 the signal is quiet but noisy and contributes random variance without informational gain. **Both R15.S3 axes are now exhausted: observation-delta (4 phases L/M/N/O', capped 0.368 ± 0.001) and value-head tempo (2 phases O/P, regressed 0.350 / 0.346)**. The previous `87e9e77` "BRANCH CLOSED" commit framing was premature — that closeout had only tested axis 1; Phases O + P add axis 2 and produce the genuine cross-axis closeout. Total R15.S3 branch compute cost ~36 min wall-clock across 6 phases (L 5m54s + M 6m22s + N 5m25s + O' 5m28s + O 5m22s + P 5m23s). F1 post-mortem framing updated again: **"The F1 reward-shape mechanism cannot break 0.368 from this warm-start. Per-step shaping from any observation-derived signal saturates at 0.368, and per-step shaping from the policy's own value-head delta actively regresses. The remaining F1 moves must change either the warm-start (R7 multi-teacher labels rebuild) or the optimization objective (R8 DPO replacement). The reward-shape branch is closed."** See "Phase P — F1 PPO + value-head tempo signal, coef 0.01 (R15.S3 GENUINE BRANCH CLOSEOUT)" below.]

**Recommended next moves, in order of plausibility:**

1. **Better warm-start.** The DAgger sweep at this codebase config plateaued at WR 37.5% with low-entropy. A larger SL run (more games, more epochs, possibly with explicit entropy regularization during BC) could give a starting point PPO can actually move. The right SL ceiling for this representation is unknown.
2. **Self-play instead of vs-rule-bot.** Item 12 opponent pool exists. PPO rollouts against PFSP-sampled prior promoted checkpoints would change the reward distribution from a single-opponent shape to a diversity shape, potentially exposing learnable axes the rule-bot alone doesn't.
3. **Richer reward shaping.** The current shape rewards point delta and win/loss only. Adding per-step shaping for attachment / retreat / energy cycles could expose strategic signal PPO can exploit.
4. **Change the gate.** All five phases are scored on greedy argmax behavior. If PPO is shaping the policy distribution but not flipping argmax, an alternative gate that samples (e.g. temperature 0.5) might reveal latent improvement. Diagnostic, not solution.

Five phases of artifacts under `runs/f1-2026-05-11-*` with full event streams, manifests, TB scalars for forensics. Live dashboard: `http://127.0.0.1:5000/run/f1-2026-05-11-phaseH-big-aggressive`.

### Phase J — F1 PPO + strong-pool self-play (post-R14 retry, 2026-05-14)

Re-attempts F1's "self-play instead of vs-rule-bot" next move with a fundamentally stronger
opponent pool than R5. R5 (`docs/ai-research-backlog.md:250-260`) ran PPO from the DAgger iter-2
warm-start against the **item17 DAgger pool** whose strongest member was Wilson 0.3109 — the same
caliber as the warm-start itself. Phase J ran the same shape against the **R13-W6-phase-d pool**
(`runs/R13-W6-phase-d/iter-{0,1,2}/checkpoint.pt`), whose iter-2 sits at Wilson 0.6479 under
rollout-leaf MCTS — a ~30pp stronger pool. Warm-start was also W6/iter-2 rather than item17/iter-2,
so this run is both a different opponent distribution *and* a different starting point. Config:
`--rollout-vs-pool` (the v1 plumbing picks **one opponent per run uniformly** via
`_select_pool_opponent`, not per-game PFSP — `notes.md` "F1 self-play readiness" called this gap
explicitly), 3 iters × 800 games/update × aggressive HPs (lr=3e-4, clip=0.3, entropy=0.01,
ppo-epochs=4). Wall-clock 5m32s end-to-end.

| Iter | Wilson lower | WR | n | Opponent sampled | mean_return | entropy | KL/mb avg | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 (warm-start eval) | 0.1455 | 30.0% | 20 | R13-W6-phase-d/iter-1 | -0.236 | 0.293 | 0.0100 | promoted (baseline marker) |
| 1 | 0.2993 | 50.0% | 20 | R13-W6-phase-d/iter-1 | -0.173 | 0.273 | 0.0092 | promoted (+14.5pp) |
| 2 | 0.2188 | 40.0% | 20 | **own iter-1 from pool** | -0.153 | 0.267 | 0.0116 | promoted (-8.1pp regression) |

**Mechanism — did PPO move the policy?** Yes, materially. mean_return marched -0.236 → -0.173 →
-0.153 (less-negative reward distribution every iter), entropy contracted 0.293 → 0.267 (policy
sharpened), and iter-1 lifted Wilson +14.5pp over the warm-start eval. This is the *largest single
PPO step recorded in any F1 phase* and the first to materially move the gate. The mechanism that
unlocked it: the pool opponent was ~30pp stronger than rule-bot, so warm-start argmax was no longer
near-optimal — importance ratios actually diverged from 1.0 (ratio min/max ranges 0.04-7.6 at iter-0
vs phase H's ~1.00) and the surrogate gradient (ratio - 1) × advantage was non-zero. **iter-2
regressed.** The most plausible mechanism: the pool sampler picked W6/iter-1 (Wilson ~0.45-class)
for the first two iters but rolled `pool/iter-001/checkpoint.pt` — the just-promoted iter-1 from
this very run — at iter-2. That switched the opponent from a stable external prior to a
co-adapting policy one step removed; the iter-1 gradient direction overfit to W6/iter-1's argmax,
and iter-2 found it had nothing left to learn against itself. The iter-2 ratio max also spiked to
**21.3** (vs 7.6 at iter-0), a numerical-instability signal consistent with running PPO against an
opponent the policy was already correlated with. Secondary contributor: the v1 "one opponent per
run" plumbing means there's no per-game opponent diversity — the gradient signal each iter is
opponent-shaped, not pool-shaped.

**Comparison to references.** Phase J iter-1 (Wilson 0.2993) is the *highest* PPO-iter-1 number on
record but iter-2 falls back to 0.2188 — **worse than R5 (0.3109) and worse than phase H iter-2
(0.3109)**. A stronger pool produced a bigger transient lift and a worse final number than the
weak-pool R5 and the no-pool aggressive phase H. The 0.40 R15.S2 gate is missed by **18pp** at
final and 10pp at the best mid-run iter.

**Pre-registered prediction check.** `notes.md` "F1 self-play readiness — scoping" recommended GO
on the basis that the W6 pool is ~30pp stronger than R5's. The prediction was that this materially
different experiment could break the cap; the empirical result is **NO**. The mechanism narrative
(stronger pool → bigger PPO step, but co-adaptation when the sampler hits a recent self-checkpoint
→ regression) is consistent with the predicted direction (PPO does move) but not the predicted
magnitude (move was transient, not sustained).

| Phase | Buffer | HP profile | KL/mb avg | Weight max-diff | Best WR | Best Wilson lower |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 30 | spec | 0.011 | 0.0002 | 37.5% | 0.3109 |
| 3a | 30 | aggressive | 0.017 | 0.003 | 34.5% | 0.2826 |
| 3b | 60 | extreme | 0.018 | 0.027 | 33.5% | 0.2732 |
| G | 800 | spec | 0.009 | 0.0002 | 38.0% | 0.3156 |
| H | 800 | aggressive | 0.008 | 0.003 | 37.5% | 0.3109 |
| **J** | **800** | **aggressive + strong-pool self-play** | **0.010** | **n/a** | **50.0% (iter-1)** | **0.2993 (iter-1) / 0.2188 (final)** |

**What this closes.** The "strong-pool self-play unlocks PPO" hypothesis is closed FAIL. R5 closed
the weak-pool variant; phase J closes the strong-pool variant; both miss the 0.40 gate, and phase
J adds the new finding that the v1 one-opponent-per-run sampler creates a co-adaptation failure
mode at iter-2 once a self-promotion lands in the pool. A future PFSP retry would need (a) per-game
opponent re-sampling and (b) PFSP weight enforcement to avoid sampling the most-recent self at
all; the lift is real, the regression is what needs fixing. **What's still load-bearing among the
F1 next moves:** R15.S1 (better warm-start) and R15.S3 (richer reward shaping) are now the only
unfalsified candidates. R15.S4 (sampling-temperature gate) remains diagnostic-only.

Artifacts: `runs/R14-f1-self-play-sweep/` — `events.jsonl` (92 lines, full minibatch detail),
`orchestrator-state.json`, per-iter `iteration-manifest.json` + `gate.manifest.json`.

### Phase K — F1 DAgger compute-scaled warm-start (R15.S1 closeout, 2026-05-14)

Tests whether the DAgger warm-start that bottoms F1 PPO at Wilson 0.31 is **compute-starved** rather
than capacity-starved. Pre-registered in `docs/ai-agent-state/notes.md:68-172` ("F1 better
warm-start — scoping"): hold `hidden_dim=64 / depth=2` fixed, scale games 3× (90 vs item17 take-2's
30) and epochs 3× (75 vs 25), keep the rollout-CRN×3 teacher, KL anchor schedule 0.0/0.1/0.5, n=500
side-balanced eval per iter. Predicted lift: iter-2 Wilson lower ≥ 0.45 (i.e. **+14pp over item17's
0.311**). Pre-registered falsification: iter-2 Wilson lower in 0.311 ± 2pp (0.29–0.33).
`runs/R15-S1-warmstart-sweep/` ran the canonical command unchanged.

| Iter | Selection | Gate WR | Wilson lower | n | Opponent | Promoted floor | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | rollout | 32.4% | 0.2845 | 500 | rule-bot | 0.0000 | promoted (baseline) |
| 1 | policy | 32.8% | 0.2883 | 500 | rule-bot | 0.2845 | promoted (+0.4pp) |
| 2 | policy | 36.8% | **0.3269** | 500 | rule-bot | 0.2883 | promoted (+3.9pp) |

**Verdict: FAIL — pre-registered falsification confirmed.** Iter-2 Wilson lower **0.3269** lands
inside the falsification band 0.311 ± 2pp (= [0.291, 0.331]). The +1.6pp lift over item17's 0.311
baseline is within Wilson half-width (~4.2pp at n=500) and an order of magnitude below the
predicted +14pp. Per-iter side asymmetry on iter-2: player WR 32.4% (Wilson [0.269, 0.384]),
opponent WR 41.2% (Wilson [0.353, 0.474]) — same player-side gap that's haunted every F1 phase.

**Falsification mechanism — pre-registered secondary check fired.** `notes.md:138-140` flagged
"if the loss plateaus by epoch ~25 (matching take-2), that itself confirms compute is saturated
at the current size." Per-epoch val_accuracy from `events.jsonl` (75 epochs × 3 iters): iter-0
best val_acc **0.7676 @ ep7**, first-99%-of-best @ **ep3**, final ep75 **0.7382**. Iter-1 best
**0.7574 @ ep2**, first-99%-of-best @ **ep2**, final ep75 **0.7212**. Iter-2 best **0.7656 @ ep3**,
first-99%-of-best @ **ep2**, final ep75 **0.7503**. In every iter, val_acc reaches 99% of its peak
by epoch 2-3, then **declines** over the remaining 70+ epochs while train_acc climbs to 0.91-0.97
(textbook overfit). The secondary diagnostic did not just trigger in iter-0 — it replicated
identically across all three DAgger iterations. The plateau-then-overfit signature converts the
gate-level falsification from "an arbitrary number landed in the band" into "the predicted
falsification mechanism fired in train-time diagnostics first, then validated at eval time."

**Comparison to references.** Iter-2 Wilson 0.3269 vs item17 take-2 iter-2 Wilson **0.311**
(`progress.md:582`), R5 PPO iter-2 0.3109, phase G/H iter-2 0.3109, phase J iter-2 0.2188. The
warm-start sweep sits at the **same Wilson cap** as every other F1-era experiment at this codebase
config, regardless of whether the axis moved is PPO HPs (phases 2/G/H), opponent pool (R5, phase
J), or SL compute (this run). Three distinct axes, one ceiling.

**Wall-clock vs scoping estimate.** Scoping forecast **~1.5h** (`notes.md:146-150`,
`backlog.md:638`). Actual end-to-end **11m 51s** (run_started → run_completed delta:
`1778728116.85 → 1778728828.37 = 711.5s`). Per-iter: iter-0 4m 33s, iter-1 3m 35s, iter-2 3m 42s.
That is ~8× faster than the scoping estimate. The dominant cost was the rollout-CRN×3 teacher
loop (~3.5 min/iter at 90 games) — n=500 eval added <30s/iter. Implication for future scoping:
the eval-games budget was NOT the bottleneck at this scale; rollout-CRN was.

| Phase | Axis | Best iter Wilson lower | Final iter Wilson lower | Compute |
| --- | --- | --- | --- | --- |
| 2 | PPO spec HPs (rule-bot) | 0.3109 | 0.3109 | ~10 min |
| H | PPO aggressive HPs (rule-bot) | 0.3109 | 0.3109 | ~10 min |
| J | PPO aggressive + strong-pool self-play | 0.2993 (iter-1) | 0.2188 | 5m 32s |
| **K** | **DAgger 3× compute, fixed 64/2** | **0.3269 (iter-2)** | **0.3269** | **11m 51s** |

**What this closes.** The "compute-starved at fixed capacity" branch of the F1 next-moves
(R15.S1). Combined with R15.S2 (strong-pool self-play, closed FAIL via Phase J this morning), the
two highest-ranked F1 post-mortem moves have both falsified. The cap is **not** compute (Phase K),
**not** PPO HP tuning (phases 2/G/H), **not** weak-pool self-play (R5), and **not** strong-pool
self-play with v1 plumbing (phase J). Surviving F1 candidates: R15.S3 (richer reward shaping) and
R15.S4 (sampling-temperature gate, diagnostic-only), plus the deeper SL-label-quality branches
the falsification opens up: R7 (multi-teacher labels) and R8 (DPO).

**No-action implication.** F1's "Wilson lower ≥ 0.40 from the current pipeline" is now empirically
out of reach without a *different* axis. PPO can match the SL cap but cannot exceed it; the SL cap
does not move with compute at fixed capacity; the SL cap does not move with capacity at fixed
compute (R6); self-play vs the strong pool regresses through co-adaptation. Three of the four F1
post-mortem next moves are now closed. Continued work on F1 should pre-register a different
mechanism axis (reward shape, label quality, or gate distribution) before paying further compute.

Artifacts: `runs/R15-S1-warmstart-sweep/` — `events.jsonl` (276 lines, per-epoch val/train acc for
all 3 iters), `orchestrator-state.json`, per-iter `iteration-manifest.json` + `gate.manifest.json`.
Pool-eval matchups (n=40 each): iter-2 vs iter-0 Wilson 0.2422 (WR 37.5%), iter-2 vs iter-1 Wilson
0.2635 (WR 40%) — confirms iter-2 is not materially stronger than its own predecessors.

### Phase L — F1 PPO + reward shaping (R15.S3 closeout, 2026-05-14)

Tests whether augmenting PPO's per-step reward with five shaped intermediate signals derived from
already-traced `PublicObservation` fields can lift the F1 PPO ceiling above the 0.31 cap that every
prior phase (2, G, H, J, K) hit. Pre-registered in `docs/ai-agent-state/notes.md` ("F1 reward
shaping — scoping"): orchestrator-only diff (~100 LOC) adds Δactive-energy, Δbench-energy,
retreat-event indicator, Δcard-throughput, and Δactive-hp-relative as per-step rewards at coefs
{0.02, 0.02, 0.03, 0.02, 0.05} with linear decay from full at iter-0 to zero at iter-2. Aggregate
per-game shaped sum targeted ≈ ±0.3 (sub-dominant to ±1 terminal). Warm-start
`runs/item17-2026-05-11/iter-002/checkpoint.pt` and opponent (rule-bot, no pool) held fixed vs
phase H. Pre-registered success: any iter Wilson lower ≥ **0.40** at n=500 side-balanced.
Pre-registered falsification: iter-2 Wilson lower in **[0.291, 0.331]**.

| Iter | Selection | Gate WR | Wilson lower | n | ratio_max | entropy_mean | KL/mb avg | Numerical anomalies | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | rollout | 31.2% | 0.2730 | 500 | 19.06 | 0.168 | 0.0142 | 0 | promoted (baseline) |
| 1 | policy | 31.8% | 0.2787 | 500 | 32.59 | 0.154 | 0.0098 | 0 | promoted (+0.6pp) |
| 2 | policy | 39.8% | **0.3560** | 500 | 11.12 | 0.153 | 0.0097 | 0 | promoted (+7.7pp) |

**Verdict: PARTIAL — best F1 rule-bot result on record (0.3560), missed the 0.40 success bar by
4.4pp, landed above the falsification band [0.291, 0.331] by +2.5pp. Neither success nor
falsification.** The trajectory is monotone (0.2730 → 0.2787 → 0.3560, +8.3pp end-to-end) and the
+7.7pp iter-1→iter-2 step is the largest single PPO jump in any F1 phase that *also* sustained at
the final iter. This is a *go-deeper-on-this-axis* result, not a *close-this-branch* result; the
branch is alive.

**Mechanism check — pre-registered diagnostic fired.** Importance ratios moved off ~1.00 in every
iter: ratio_max 19.1 / 32.6 / 11.1 vs phase H's ~1.00 baseline. This is the branch-split signature
the scoping pre-registered for distinguishing "shape didn't move PPO" (branch 2, ratios stay at
1.0, re-rank R8 over R7) from "shape moved PPO but didn't encode winning" (branch 1, ratios move,
re-rank R7 over R8). **Branch 1 fired.** The shaped reward IS moving the surrogate gradient and
the gradient IS moving the policy; the gap to 0.40 is now quantitative (coefficient magnitudes,
signal mix) rather than mechanistic. Secondary stability checks all clean: entropy stable across
iters (0.168 → 0.154 → 0.153, no collapse), `numerical_anomalies = 0` across all 48 minibatches,
ratio_min stayed above 1e-3 (~0.0015 in all three iters), approx_kl_max < 0.02 throughout.

**Comparison to references.** Iter-2 Wilson 0.3560 vs phase H iter-2 (no shape) **0.3109**, phase
K iter-2 (compute, R15.S1) **0.3269**, phase J iter-2 (strong-pool self-play, R15.S2) **0.2188**.
Phase L is **+4.5pp over the prior F1 ceiling** (phase H 0.3109) and **+2.9pp over R15.S1** — the
first F1 PPO configuration that materially exceeds the SL cap rather than matching it. Gate WR
39.8% is also the highest WR any F1 PPO config has produced (vs 38.0% phase G iter-2 best).

**What this implies for the F1 post-mortem.** The phase-H post-mortem's "F1 target 0.40 NOT
REACHABLE from the item17 warm-start under the current PPO mechanism" was a correct conservative
read of the mechanism *that existed at the time* (unshaped Δpoints + ±1 terminal, ratios pinned
at 1.0, surrogate gradient ≈ 0). R15.S3 demonstrably *changed the mechanism* — the reward axis
unlocks non-zero gradient even from the same warm-start. With the mechanism unlocked, 0.40 is now
reachable in principle from this warm-start; the open question is whether scaling shaping coefs
or changing the signal mix gets there. The "NOT REACHABLE" line in the F1 phase summary above
has been qualified (not deleted) to reflect this.

**Pre-registered prediction check.** Scoping predicted iter-2 Wilson lower ≥ 0.40 as the success
bar and 0.311 ± 2pp as the falsification band; the result lands 4.4pp short of success and 2.5pp
above falsification. The branch-1 diagnostic ("ratio range moves off [0.5, 2.0]") fired clearly —
ratio_max 11-33 across iters is well outside [0.5, 2.0] and matches the R15.S2 gradient-active
signature (digest slot 8, ratio range 0.05-4.6 there). Shaped-component attribution per iter was
not separately surfaced in the gate manifest beyond the aggregate signal — a follow-up to drop
small contributors would benefit from re-extracting it from the trace stream.

**Wall-clock vs scoping estimate.** Scoping forecast ~10-15 min total. Actual end-to-end **5m
53.7s** (run_started → run_completed delta: `1778731408.98 → 1778731762.72 = 353.74s`). Per-iter
~118s including rollout + PPO update + n=500 gate. Reward shaping added zero measurable runtime
cost (a few extra float adds per parsed trace row); the dominant cost remained the rollout
phase, identical to phase H. Implementation cost: **+103 LOC orchestrator-only diff** to
`training/ppo_orchestrator.py` (5 new `--reward-*-coef` args + `--reward-shape-start/-end` linear
decay + shape_attribution event), zero sim-side, zero TS-side. `TMPDIR=/tmp npm run test:ppo-smoke`
PASS pre-launch.

| Phase | Axis | Best iter Wilson lower | Final iter Wilson lower | Compute |
| --- | --- | --- | --- | --- |
| 2 | PPO spec HPs (rule-bot) | 0.3109 | 0.3109 | ~10 min |
| H | PPO aggressive HPs (rule-bot) | 0.3109 | 0.3109 | ~10 min |
| J | PPO aggressive + strong-pool self-play | 0.2993 (iter-1) | 0.2188 | 5m 32s |
| K | DAgger 3× compute, fixed 64/2 | 0.3269 (iter-2) | 0.3269 | 11m 51s |
| **L** | **PPO aggressive + 5-signal reward shaping** | **0.3560 (iter-2)** | **0.3560** | **5m 54s** |
| **M** | **Phase L + all 5 coefs scaled 1.75×** | **0.3580 (iter-2)** | **0.3580** | **6m 22s** |
| **N** | **Phase L + constant schedule (no decay)** | **0.3677 (iter-2)** | **0.3677** | **5m 25s** |
| **O'** | **Phase N + reduced 3-signal mix (drop bench-energy + retreat; scale active-energy + throughput)** | **0.3677 (iter-2)** | **0.3677** | **5m 28s** |
| **O** | **Phase N + value-head tempo signal at coef 0.05 (axis 2 — learned-signal, magnitude=design budget × 20)** | **0.3502 (iter-2)** | **0.3502** | **5m 22s** |
| **P** | **Phase N + value-head tempo signal at coef 0.01 (axis 2 — learned-signal, magnitude=design budget × 4)** | **0.3463 (iter-2)** | **0.3463** | **5m 23s** |

**What this opens.** A coefficient-scaling follow-up sweep on the same axis: hold the five signals
fixed, scale all five coefs 1.5–2× (the current ~0.13/step per-game sum is at the low end of the
scoping target ±0.3 budget), rerun the 3-iter sweep at the same warm-start / opponent / HPs.
Expected ~10-15 min compute. If iter-2 crosses 0.40 → first F1 success on record. If iter-2 stalls
at ~0.36 → coef scaling is saturated and the next move is signal-mix change (drop low-attribution
components, add new ones) or more iterations. Queued as `r15-s3-followup-tune-shaping`.

### Phase M — F1 PPO + reward-shape coef-scaling follow-up (R15.S3 1.75×, 2026-05-14)

Sub-result of the R15.S3 question (the parent Phase L closeout above asked "does richer reward
shaping move PPO past 0.31?" — yes, to 0.3560). Phase M asks the queued follow-up: "does scaling
those same five coefs 1.5–2× break past 0.36?" Run `runs/R15-S3-followup-tune/`. Same five signals
as Phase L, all coefs scaled **1.75×** (midpoint of the queued 1.5–2× range): active-energy 0.035,
bench-energy 0.035, retreat 0.0525, throughput 0.035, hp-diff 0.0875. Per-game shape budget now
~0.23 (vs Phase L ~0.13; scoping target ±0.3). Same warm-start
(`runs/item17-2026-05-11/iter-002/model/checkpoint.pt`), same opponent (rule-bot, no pool), same
HPs (`--lr 3e-4 --clip-epsilon 0.3 --entropy-coef 0.01 --ppo-epochs 4`), same code (commit
`8da53bd` shaping diff intact, no new diff). Schedule unchanged at linear decay full→zero.
Pre-registered: success if any iter Wilson lower ≥ **0.40**, falsification if iter-2 in
**[0.291, 0.331]**, **saturation** if iter-2 stalls at ~0.36 → next move is signal-mix change.

| Iter | Shape scale | Selection | Gate WR | Wilson lower | n | ratio_max | entropy_mean | Numerical anomalies | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.0 | rollout | 33.6% | **0.2960** | 500 | 15.86 | 0.171 | 0 | **promoted** (baseline, +2.3pp vs Phase L iter-0) |
| 1 | 0.5 | policy | 32.2% | **0.2825** | 500 | 12.50 | 0.164 | 0 | **rejected** (regressed -1.3pp vs iter-0) |
| 2 | 0.0 | policy | 39.8% | **0.3580** | 500 | 18.82 | 0.164 | 0 | **promoted** (rolled forward from iter-0 parent after iter-1 reject) |

**Verdict: SATURATION as pre-registered.** Iter-2 Wilson **0.3580** lands within Wilson noise floor
of Phase L iter-2 **0.3560** (Δ +0.002). 1.75× coef scaling moved iter-0 by +2.3pp (0.2730 →
0.2960) but did not compound: iter-1 over-shaped and was rejected (first F1
rejection-inside-a-single-sweep event on 2026-05-14), iter-2 retreating to the iter-0 parent
recovered to within noise of the unscaled Phase L final. The (warm-start, opponent, signal set,
linear-decay schedule) tuple under shaping has a true ceiling at ~0.358. Further coef scaling on
this axis will not move iter-2 closer to 0.40.

**Comparison to Phase L (parent sub-result).**

| Iter | Phase L (1.0× coefs) Wilson lower | Phase M (1.75× coefs) Wilson lower | Δ |
| --- | --- | --- | --- |
| 0 | 0.2730 | **0.2960** | **+0.023** |
| 1 | 0.2787 | 0.2825 | +0.004 |
| 2 | **0.3560** | **0.3580** | **+0.002** |

The iter-0 lift confirms that higher coefs do still move the policy (ratio_max 15.86 at iter-0 vs
Phase L iter-0 19.06; both well off the phase-H ~1.00 baseline). The iter-2 collapse-to-noise
confirms the saturation hypothesis: by the time decay zeroes out the shape signal, the policy
converges to roughly the same terminal-only optimum regardless of whether iter-0 had a slightly
stronger push.

**Mechanism check.** Importance ratios still moved decisively off ~1.00 in every iter (ratio_max
15.86 / 12.50 / 18.82 — still strongly gradient-active, comparable to Phase L's 19.06 / 32.59 /
11.12 range). The gradient is *not* frozen — the policy *is* moving — but it converges to the
same ceiling as Phase L. This is not a "PPO stopped working" outcome; it is a "the reachable
optimum under this signal set is what it is" outcome. Entropy stable across iters (0.171 → 0.164
→ 0.164, no collapse). `numerical_anomalies = 0` across all 48 minibatches. approx_kl_max < 0.02
throughout. The iter-1 rejection (`wilson_lower 0.2825 + tolerance 0.0000 < floor 0.2960`) is
notable: it is the first sweep-internal regression of any F1 PPO run on record and a clean
falsifier of "more shaping is always better" — at 1.75× iter-1 coefs (still 0.5 of the iter-0
scale via decay) the policy over-corrected toward shape and lost ground vs the iter-0 parent.

**Wall-clock vs Phase L.** Phase M end-to-end **6m 22.3s** (run_started → run_completed delta:
`1778732302.62 → 1778732684.93 = 382.31s`). Phase L was 5m 53.7s. Comparable wall-clock — the
coef scaling adds zero runtime cost.

**What this closes and opens.** Closes: further coef scaling on the existing 5-signal mix under
linear decay. The pre-registered saturation outcome fired, the queued `r15-s3-followup-tune-shaping`
item is resolved as SATURATED, and no further sweep along the coef-magnitude axis is justified.
Opens: signal-mix or schedule change as the next single-axis move. Three candidates ranked in
`docs/ai-agent-state/notes.md` "F1 reward shaping — scoping" follow-up block: (1) constant-shaping
schedule (`--reward-shape-end 1.0`, smallest single-axis change, defensible v1), (2) drop low-
attribution signals + scale survivors, (3) add a value-head-derived strategic-tempo signal (most
ambitious). Queued as P3 `r15-s3-signal-mix-or-schedule` with constant-shaping as the recommended
v1.

Artifacts: `runs/R15-S3-followup-tune/` — `events.jsonl` (per-iter ppo-update minibatch detail
with ratio/KL/entropy/anomaly counts; per-iter `trajectory-parse/completed.data.shape_attribution`
showing iter-0 totals active 147.6 / bench 8.5 / retreat 0 / throughput 299.6 / hp-diff -14.0,
iter-1 exactly 0.5× those, iter-2 zero), `orchestrator-state.json` (promoted_wilson_lower
0.35797823817783564), per-iter `iteration-manifest.json` + `gate.manifest.json`.

**No-action implication (updated).** F1's "Wilson lower ≥ 0.40" remains unreached but is no longer
empirically out of reach. Three of the four phase-H-era F1 post-mortem moves are closed FAIL
(warm-start R15.S1, self-play R15.S2, the SL-compute / capacity / HP-tuning baselines); the
reward-shape axis is open and producing the strongest F1 numbers on record.

Artifacts: `runs/R15-S3-reward-shaping-sweep/` — `events.jsonl` (per-iter ppo-update minibatch
detail with ratio/KL/entropy/anomaly counts), `orchestrator-state.json` (promoted_wilson_lower
0.35602922648461244), per-iter `iteration-manifest.json` + `gate.manifest.json`.

### Phase N — F1 PPO + constant reward shaping (R15.S3 schedule axis, 2026-05-14)

Third F1 reward-shaping data point and **the new F1 rule-bot ceiling on record**. Phase L closed
the parent R15.S3 question (does shaped reward move PPO past 0.31? yes, to 0.3560). Phase M closed
the coef-magnitude axis at SATURATION (1.75× coefs landed iter-2 at 0.3580, Δ +0.002 vs L).
Phase N closes the **shaping-schedule axis**: same five signals, same Phase L 1.0× coefs (active
0.02 / bench 0.02 / retreat 0.03 / throughput 0.02 / hp-diff 0.05), same warm-start
(`runs/item17-2026-05-11/iter-002/model/checkpoint.pt`), same opponent (rule-bot, no pool), same
HPs (`--lr 3e-4 --clip-epsilon 0.3 --entropy-coef 0.01 --ppo-epochs 4`), **only diff vs Phase L:
`--reward-shape-end 0.0` → `--reward-shape-end 1.0`** (constant full-strength shaping across all
iters, no linear decay). Run `runs/R15-S3-constant-shape/`. Pre-registered: success if any iter
Wilson lower ≥ **0.40** (first F1 success on record), partial if 0.36 < iter-2 < 0.40,
reward-hacking if iter-2 ≥ iter-0 by ≥ +5pp Wilson while WR drops vs heuristic gate (constant
shape gamed but games lost), saturation if iter-2 ≈ 0.36 → pivot to v2 signal-mix change.

| Iter | Shape scale | Selection | Gate WR | Wilson lower | n | ratio_max | entropy_mean | Numerical anomalies | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.0 | rollout | 31.8% | **0.2787** | 500 | 11.82 | 0.173 | 0 | **promoted** (baseline, +0.6pp vs Phase L iter-0 0.2730) |
| 1 | 1.0 | policy | 32.8% | **0.2883** | 500 | 7.97 | 0.160 | 0 | **promoted** (+1.0pp vs iter-0; vs Phase M iter-1 0.2825 rejected, **no rejection under constant shape**) |
| 2 | 1.0 | policy | 41.0% | **0.3677** | 500 | 32.78 | 0.155 | 0 | **promoted** (+7.9pp vs iter-1; **new F1 ceiling**) |

**Verdict: PARTIAL — new F1 ceiling at iter-2 Wilson 0.3677.** Constant shaping cleared Phase L
by **+1.2pp** (0.3677 vs 0.3560) and Phase M by **+1.0pp** (0.3677 vs 0.3580). Trajectory
monotone 0.2787 → 0.2883 → 0.3677 (+8.9pp end-to-end), all three iters promoted, **no
rejections** (in contrast to Phase M's iter-1 reject), `run_completed clean, halted=false`. Still
short of the 0.40 success bar by **3.2pp** — does not clear, but is the smallest gap to date and
the best F1 rule-bot result on record across every sweep. The schedule-axis change is real but
small: +0.012 absolute at iter-2 sits within roughly 1.5σ Wilson noise at n=500, trending
positive but not significant in isolation. Crossed with Phase M (Δ +0.002 from coef scaling), the
joint message is that both single-axis follow-ups on the existing 5-signal mix moved iter-2 by
≤+1pp — the ceiling at 0.356–0.368 is the binding constraint.

**Comparison to Phases L and M (cross-axis table).**

| Phase | Schedule | Coefs | iter-0 Wilson | iter-1 Wilson | iter-2 Wilson | iter-2 Δ vs L | iter-1 rejected? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L (R15.S3 parent) | linear-decay (full→zero) | 1.0× | 0.2730 | 0.2787 | **0.3560** | — | no |
| M (1.75× follow-up) | linear-decay (full→zero) | 1.75× | 0.2960 | 0.2825 (rejected) | **0.3580** | +0.002 | **yes** |
| **N (constant follow-up)** | **constant (full→full)** | **1.0×** | **0.2787** | **0.2883** | **0.3677** | **+0.012** | **no** |

Phase L → Phase N Δ iter-2 = **+0.0117** (round to +1.2pp). Phase L → Phase M Δ iter-2 = +0.0020
(+0.2pp). The schedule axis moved iter-2 by ~5× more than the coef-magnitude axis, but both are
small. The iter-1 picture is more striking: constant shape's iter-1 (0.2883, promoted, +1.0pp
over iter-0) cleanly beats Phase M's iter-1 (0.2825, rejected, -1.3pp under iter-0) — supporting
the pre-scoping hypothesis that decaying shape mid-sweep is itself destabilizing, and that the
policy keeps climbing on shaped signal when shape is held constant instead of being asked to
converge to a deterministic terminal-only optimum mid-run.

**Mechanism check.** Importance ratios moved decisively off ~1.00 in every iter (ratio_max
**11.82 / 7.97 / 32.78** — gradient strongly active in every iter, comparable to Phase L's
19.06 / 32.59 / 11.12 and Phase M's 15.86 / 12.50 / 18.82 ranges; iter-2's 32.78 is the highest
ratio_max of any iter across the three runs but did not cause any numerical anomaly). Entropy
stable across iters (0.173 → 0.160 → 0.155, no collapse). `approx_kl_max < 0.015` every iter
(0.0147 / 0.0116 / 0.0150). `numerical_anomalies = 0` across all 48 minibatches. PPO is doing
exactly the same work it did in Phases L and M — moving the policy hard from the warm-start —
and landing in the same neighborhood. **Reward-hacking check.** Gate WR tracks Wilson lower in
lockstep across iters (31.8% → 32.8% → 41.0%, monotone increasing, matches Wilson trajectory
0.2787 → 0.2883 → 0.3677). No iter where Wilson rises while WR falls; no iter where Wilson
exceeds WR by more than the expected Wilson-from-n=500 lower-bound gap. The +7.9pp Wilson lift
from iter-1 to iter-2 is mirrored by a +8.2pp WR lift. The policy is winning more games under
constant shape, not gaming a shape signal at the expense of winning.

**What this implies for the binding constraint.** Three configurations of the same 5-signal mix
(L 1.0× linear-decay, M 1.75× linear-decay, N 1.0× constant) have now landed iter-2 at
0.3560 / 0.3580 / 0.3677 — a 1.2pp spread. The mechanism (importance ratios off 1.0, gradient
active, KL bounded, entropy stable) has fired correctly in all three; PPO is healthy and
moving the policy in every case. The reward-shape axis (Δactive-energy, Δbench-energy, retreat
indicator, Δcard-throughput, Δactive-hp-relative) encodes strategic information worth ~+5pp
Wilson over the unshaped Phase H baseline (0.3109 → 0.3677), but **scaling its magnitude or
its schedule does not get us the +9pp needed to clear 0.40**. The next move on this branch is
therefore **signal-set change**, not further schedule or magnitude tuning. The highest-leverage
candidate is option (3) from the scoping notes — add a strategic-tempo signal derived from the
trained value head's own per-step delta in own-win-probability, which taps a different
information source (the model's own probabilistic assessment of game state) rather than just
hand-engineered observation deltas. Implementation cost ~20-40 LOC additive to
`parse_trace_to_trajectories` (see `training/ppo_orchestrator.py:715-739` for the existing
5-signal computation pattern); the value head output is already in the trajectory inference
stream.

**Wall-clock.** Phase N end-to-end **5m 25.4s** (run_started → run_completed delta:
`1778733032.80 → 1778733358.24 = 325.44s`). Phase L was 5m 53.7s; Phase M 6m 22.3s. Schedule
change adds zero runtime cost; this run is the *fastest* of the three (~30s under Phase L,
~60s under Phase M, within iter-level rollout variance). Implementation cost: **zero LOC**
(one CLI flag flip from the Phase L baseline, no code diff).

**What this closes and opens.** Closes: the shaping-schedule axis at the existing 5-signal /
1.0× coef tuple. Constant beats linear-decay by ~+1pp at iter-2 — real but small relative to
the 3.2pp remaining gap to 0.40. The queued `r15-s3-signal-mix-or-schedule` item is resolved as
PARTIAL with the schedule axis explored. Opens: signal-set change. The recommended next single-
axis move is **value-head-delta as a 6th additive signal** (coef ~0.05, scale similar to the
existing hp-diff signal which was the strongest existing component; keep the same 5 existing
signals, this is additive; same constant schedule established by Phase N; same Phase H HPs;
same warm-start). Pre-register: success if iter-2 Wilson ≥ 0.40 (first F1 success on record),
partial if 0.37 < iter-2 < 0.40, saturation if iter-2 ≈ 0.37 (then the entire 6-signal mix has
saturated and the next move is either a different architectural axis or accepting that the
warm-start neighborhood does not contain a 0.40 policy under our current eval gate). Queued as
P3 `r15-s3-value-head-tempo-signal`.

Artifacts: `runs/R15-S3-constant-shape/` — `events.jsonl` (per-iter ppo-update minibatch detail
with ratio/KL/entropy/anomaly counts; per-iter `trajectory-parse/completed.data.shape_attribution`
showing all five components active at full scale across all three iters with no decay),
`orchestrator-state.json` (promoted_wilson_lower 0.3677343110002554), per-iter
`iteration-manifest.json` + `gate.manifest.json`, `launch.log` (banner shows
`--reward-shape-start 1.0 --reward-shape-end 1.0` confirmed).

### Phase O' — F1 PPO + reduced signal-mix (R15.S3 branch closeout, 2026-05-14)

Capstone observation that closes the R15.S3 observation-delta reward-shaping branch as a
research direction. Phase O original plan (value-head tempo signal as a 6th additive
component) was audited at launch time and found blocked on TS-side ONNX/trace instrumentation
(rollout emits placeholder `value_pred=0.0` at `ppo_orchestrator.py:858`, PPO recomputes value
at training-time, the value head is not in the decision-trace schema). Filed as P3
`r15-s3-value-head-trace-instrumentation` ready (separate sprint-slot work crossing TS and
Python). Pivoted to the scoping doc's option (2) — drop the weakest of the 5 existing signals
and scale the dominant 2-3 — as the cheapest remaining single-axis move inside the R15.S3
branch. Phase N iter-2 absolute coef-weighted contributions (from
`runs/R15-S3-constant-shape/events.jsonl` `trajectory-parse/completed.data.shape_attribution`):
throughput 171.4, active-energy 90.4, hp-diff **-6.9** (anti-correlated, consistent across all
3 iters), bench-energy 4.2, retreat **0.0** (the agent never retreats — the rule-bot opponent
doesn't pressure retreat decisions and the warm-start policy doesn't develop them). Run
`runs/R15-S3-reduced-mix/` (Phase O') dropped retreat and bench-energy, scaled active-energy
0.02→0.03 and throughput 0.02→0.03, and **held hp-diff 0.05 as control** rather than
amplifying an anti-correlated signal (preserves signal-set parity test integrity). Per-game
shape budget ~0.24, close to Phase M's 0.23. Same warm-start
(`runs/item17-2026-05-11/iter-002/model/checkpoint.pt`), opponent (rule-bot, no pool), HPs
(`--lr 3e-4 --clip-epsilon 0.3 --entropy-coef 0.01 --ppo-epochs 4 --reward-shape-start 1.0
--reward-shape-end 1.0`), code as Phase N. Pre-registered: success if iter-2 Wilson ≥ 0.40
(first F1 success on record — signal-mix unlocks more headroom); partial if 0.368 < iter-2 <
0.40 (mix has incremental headroom worth pursuing); **saturation if iter-2 ≈ 0.37**
(signal-SET is genuinely binding, not the MIX over the 5 signals — branch is closed and
attention pivots to a different information source).

| Iter | Shape scale | Eval gate | WR | Wilson lower | n=games | ratio_max | entropy_mean | approx_kl_max | numerical_anomalies | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.0 | rollout | 32.2% | **0.2825** | 500 | 10.53 | 0.171 | 0.016 | 0 | **promoted** (baseline, +1.0pp vs Phase N iter-0 0.2787) |
| 1 | 1.0 | policy | 32.8% | **0.2883** | 500 | 18.02 | 0.161 | 0.014 | 0 | **promoted** (+0.6pp vs iter-0; **identical to Phase N iter-1 0.2883 to 4dp**) |
| 2 | 1.0 | policy | 41.0% | **0.3677** | 500 | 10.57 | 0.153 | 0.015 | 0 | **promoted** (+7.9pp vs iter-1; **identical to Phase N iter-2 0.3677 to 4dp**) |

**Verdict: SATURATION as pre-registered — R15.S3 BRANCH CLOSED.** Iter-2 Wilson **0.3677
identical to Phase N's 0.3677 to 4 decimal places** (Δ +0.0000pp). Dropping 2 of 5 signals
(both shown by Phase N attribution to be effectively dead) and scaling the 2 dominant signals
moved iter-2 by **0.0000pp**. All 3 iters promoted, monotone, no rejections, `run_completed
clean, halted=false`. Attribution data confirms the drop choices were correct: bench-energy
**0.0** and retreat **0.0** across all 3 Phase O' iters (signals correctly silenced —
`shape_attribution` reads `bench_energy: 0.0, retreat: 0.0` for iter-0/1/2); throughput
dominant (258.4 / 255.7 / 259.3 across iter-0/1/2); active-energy secondary (126.1 / 130.8
/ 132.3); hp-diff consistently anti-correlated (-6.4 / -6.7 / -6.2). The signal-mix change
that this Phase O' tested was the cleanest possible "drop the dead and amplify the live"
move — and it did not move iter-2 at all.

**Comparison to Phase N (parent constant-shape sub-result).**

| Iter | Phase N (5-signal, 1.0× coefs, constant) Wilson lower | Phase O' (3-signal, scaled, constant) Wilson lower | Δ |
| --- | --- | --- | --- |
| 0 | 0.2787 | **0.2825** | +0.004 |
| 1 | 0.2883 | **0.2883** | **+0.000** |
| 2 | **0.3677** | **0.3677** | **+0.000** |

Iter-1 and iter-2 match Phase N to 4 decimal places. Iter-0 shows a small +0.4pp lift that
does not compound — within Wilson noise at n=500.

**Mechanism check (healthy in all 3 iters).** `ratio_max` per iter **10.53 / 18.02 / 10.57**
— gradient strongly active, comparable to Phase L's 19.06 / 32.59 / 11.12, Phase M's
15.86 / 12.50 / 18.82, and Phase N's 11.82 / 7.97 / 32.78 ranges. Entropy stable
(0.171 → 0.161 → 0.153, no collapse). `numerical_anomalies = 0` across all 48 minibatches.
`approx_kl_max < 0.016` each iter (well within stability bound). `ratio_mean ≈ 1.000`
throughout (PPO is taking real gradient steps that integrate to ~zero KL, the canonical
healthy signature). PPO is not stuck; it is converging to the same local optimum the other
three phases found.

**Reward-hacking check (passes).** WR tracks Wilson in lockstep: 32.2% → 32.8% → 41.0% — the
+7.9pp iter-2 Wilson lift is mirrored by +8.2pp WR. No iter where Wilson rises while WR falls.
No iter where Wilson rises while WR stalls. The policy that maximizes the reduced 3-signal
shape is also the policy that wins more games against rule-bot. This is not a signal-gaming
outcome; it is a "the signals are encoding winning correctly, but they have run out of
information content" outcome.

**4-axis cross-phase synthesis (capstone — the binding constraint).**

| Phase | Schedule | Coef scale | Signal mix | iter-0 Wilson | iter-1 Wilson | iter-2 Wilson | Δ iter-2 vs Phase L | iter-1 rejected? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L (R15.S3 parent) | linear-decay (full→zero) | 1.0× | 5 signals | 0.2730 | 0.2787 | **0.3560** | — | no |
| M (1.75× follow-up) | linear-decay (full→zero) | 1.75× | 5 signals | 0.2960 | 0.2825 (rejected) | **0.3580** | +0.002 | **yes** |
| N (constant follow-up) | constant (full→full) | 1.0× | 5 signals | 0.2787 | 0.2883 | **0.3677** | +0.012 | no |
| **O' (signal-mix follow-up)** | **constant (full→full)** | **1.0× (active+throughput scaled 1.5×)** | **3 signals (drop bench-energy + retreat)** | **0.2825** | **0.2883** | **0.3677** | **+0.012** | **no** |
| **O (value-head tempo, coef 0.05)** | **constant (full→full)** | **1.0× obs-delta + value-head 0.05** | **5 obs-delta + value-head** | **0.2845** | **0.2768 (rejected)** | **0.3502** | **-0.006** | **yes** |

Three single-axis moves now tested independently inside the R15.S3 observation-delta family,
each from the Phase L baseline:

| Axis | Configs | Δ iter-2 Wilson |
| --- | --- | --- |
| **Coef magnitude** | Phase L 1.0× → Phase M 1.75× | **+0.002** |
| **Shaping schedule** | Phase L decay → Phase N constant | **+0.012** |
| **Signal mix** | Phase N 5-signal → Phase O' 3-signal | **+0.000** |

PPO is healthy in all four phases — importance ratios decisively off 1.0, KL bounded, entropy
stable, WR tracks Wilson, no anomalies, no reward hacking. The 5-signal hand-engineered
observation-delta family encodes ~+5pp Wilson over the unshaped Phase H baseline (0.3109 →
0.3677), but **scaling magnitude, changing schedule, or dropping inactive components all
leave iter-2 at 0.368 ± 0.001**. The binding constraint is the *information content* of the
signal set, not the signals' weights, schedule, or which subset is active.

**Wall-clock and total branch cost.** Phase O' end-to-end **5m 28.2s** (run_started →
run_completed delta: `1778733964.90 → 1778734293.08 = 328.18s`); fastest of the L/M/N/O'
quartet alongside Phase N. Signal-mix change adds zero runtime cost (the 5 reward-component
computations all run per row regardless; setting a coef to 0.0 zeroes the contribution but
not the compute). **Total R15.S3 branch wall-clock ≈ 23 minutes across 4 phases** (Phase L
5m 54s + Phase M 6m 22s + Phase N 5m 25s + Phase O' 5m 28s) — cheap research, decisive
answer. Implementation cost: **zero LOC diff vs Phase N** (CLI args only).

**What this implies for the F1 post-mortem.** The Phase L closeout qualified the original
"0.40 NOT REACHABLE" claim with "the reward axis demonstrably changed the mechanism; 0.40 is
now reachable in principle." Phase M / N narrowed that to "the gap is quantitative —
coefficient scale + signal mix." **Phase O' closes the qualification cleanly: the
reward-shape axis has been fully explored and converged at 0.368.** The remaining gap to
0.40 is now **categorical** — needs a different *information source*, not more tuning of the
existing observation-delta signals. The post-mortem update line at 736 above is rewritten
accordingly.

**What this opens — three surviving F1-line candidates (research-stance decision, escalated
to human).** The only ways to break the 0.368 cap from this warm-start + opponent combination
require pulling in information from a *different source*:

1. **Value-head tempo signal** — per-step delta of the policy's own value estimate as a
   strategic-tempo reward signal. Highest-leverage of the three because it adds *learned*
   strategic information rather than hand-engineered observation deltas. Blocked on TS-side
   ONNX/trace instrumentation: the value head is not in the current decision-trace schema and
   rollout emits a placeholder. Requires (a) ONNX export audit to confirm `rollout.onnx` /
   `policy.gate.onnx` expose a named value-head output (extend the export step if not),
   (b) TS-side ONNX inference call-site instrumentation in `backend/src/sim/` to capture the
   value scalar per request, (c) decision-trace schema extension (likely
   `behaviorPolicy.valueEstimate: number` or top-level `valuePred`) with
   `relabelDecisionTrace.ts` forward, (d) regenerate a smoke trace and verify the new field
   is present. Then unblock the queued P3 `r15-s3-value-head-tempo-signal`. Multi-file change
   crossing TS / Python; ~separate sprint slot.

2. **R7 multi-teacher labels** — change the SL warm-start by retraining DAgger with multiple
   teachers (e.g. rollout-teacher mixed with rule-bot policy labels, or with value-head leaf
   MCTS labels). PPO inherits a different starting policy. Does not touch the reward axis.
   Larger code surface (SL pipeline rebuild).

3. **R8 DPO** — replace PPO with Direct Preference Optimization, an objective that doesn't
   depend on hand-shaped reward signal. Bigger pivot; orthogonal to the reward-shape work.

A fourth, untested possibility: **non-per-step reward-shaping framework** — e.g. turn-based or
game-phase aggregate reward shaping rather than the per-step delta framework R15.S3 used.
Untested in the R15 line; ranks below the three above without evidence one way or another.

**Recommendation: human picks among (1)/(2)/(3) before further autonomous F1-line work.** The
autonomous loop should NOT pick one of these on its own — each is a meaningfully different
research direction (sim-side instrumentation vs SL rebuild vs PPO replacement). Escalation
filed at `docs/ai-agent-state/escalations.md` `## Open`; queue item
`r15-s3-branch-synthesis-and-next-pick` P2 ready autonomous-launch ineligible. The full
synthesis is at `docs/ai-agent-state/notes.md` `## F1 reward shaping — scoping (2026-05-14)`
Phase O' Closeout + 4-axis synthesis block.

Artifacts: `runs/R15-S3-reduced-mix/` — `events.jsonl` (per-iter ppo-update minibatch detail
with ratio/KL/entropy/anomaly counts; per-iter `trajectory-parse/completed.data.shape_attribution`
showing bench-energy=0.0 + retreat=0.0 across all 3 iters; throughput / active-energy /
hp-diff signals active at full scale across all 3 iters with no decay),
`orchestrator-state.json` (promoted_wilson_lower 0.3677343110002554, identical to Phase N to
4 decimal places), per-iter `iteration-manifest.json` + `gate.manifest.json`, `launch.log`
(banner shows `--reward-active-energy-coef 0.03 --reward-bench-energy-coef 0.0
--reward-retreat-coef 0.0 --reward-throughput-coef 0.03 --reward-hp-diff-coef 0.05
--reward-shape-start 1.0 --reward-shape-end 1.0` confirmed).

### Phase O — F1 PPO + value-head tempo signal, coef 0.05 (2026-05-14)

**Naming note.** The label "Phase O'" was assigned earlier the same day to the reduced-mix
data-driven variant that landed *before* the value-head signal could be implemented (TS-side
instrumentation was blocked at the time). After the queued P3
`r15-s3-value-head-trace-instrumentation` work landed (slot-25 — 7 LOC TS + ~30 LOC orchestrator,
both smokes green) and unblocked the original Phase O plan, this run carries the unprimed name
"Phase O". Chronologically Phase O' precedes Phase O; numerically they are siblings testing two
different ways of changing the signal *content* (drop-and-amplify within the existing
observation-delta family vs adding a new learned-tempo signal).

**Verdict: REGRESSION vs Phase N; coef was too aggressive.** Run `runs/R15-S3-value-head-tempo/`
finished clean (`run_completed`, `halted=false`), all 3 iters completed with iter-2 promoted from
the iter-0 parent after iter-1 was rejected. Trajectory: iter-0 Wilson **0.2845** (WR 32.4%,
promoted) → iter-1 **0.2768** (WR 31.6%, **REJECTED** — `wilson_lower 0.2768 < floor 0.2845`,
same regression pattern as Phase M iter-1) → iter-2 **0.3502** (WR 39.2%, rolled forward from
iter-0 parent after iter-1 reject, promoted). `promoted_iterations: [0, 2]`. Iter-2 **0.3502 is
1.8pp worse than Phase N's 0.3677** — adding the value-head signal at coef 0.05 made things
worse, not better. Wall-clock **5m 22.0s** (run_started ts 1778735254.14 → run_completed
1778735576.01).

**Mechanism diagnosis — value-head signal is correctly wired but magnitude is ~20× the design
budget.** The value head output is Tanh-bounded so the per-step range is [−1, +1] and the
per-step delta range is [−2, +2]. At `--reward-value-head-coef 0.05`, the per-step
contribution to the shaped reward is bounded by ±0.1. At the F1 episode length of ~60 steps
per game, the per-game contribution is therefore bounded by ±6.0 — **20× the ±0.3/game
shape budget** the R15.S3 scoping doc set. The iter-0
`trajectory-parse/completed.data.shape_attribution.value_head = 26.01` confirms the magnitude
empirically dominates: it is third-largest in absolute terms behind throughput (171.4 — but
that signal accumulates positively across all 60 steps so its absolute magnitude is largely
positive bias rather than per-step variance) and active-energy (85.1). The value-head signal
is mean-zero in expectation (Tanh delta) so its absolute attribution of 26.0 represents
genuine per-step *variance* contributed to the surrogate gradient, larger than every
observation-delta signal except the two energy-related ones. iter-1's
`value_head = 9.28` and iter-2's `value_head = 8.25` drop after iter-0's promoted policy
learns to flatten the value-head delta — exactly the over-shaping pattern.

**Cross-phase iter-2 Wilson summary.**

| Iter | Phase N (5-signal constant, 1.0×) Wilson lower | Phase O (5-signal + value-head 0.05, constant) Wilson lower | Δ |
| --- | --- | --- | --- |
| 0 | 0.2787 | 0.2845 (promoted) | +0.006 |
| 1 | 0.2883 (promoted) | 0.2768 (REJECTED) | -0.012 |
| 2 | **0.3677** (promoted) | **0.3502** (promoted, iter-0 parent) | **-0.018** |

**Why iter-1 rejected.** Same mechanism as Phase M's 1.75×-coef regression: an over-shaped
reward at iter-0 promotes a policy that has learned to exploit the shape; iter-1 trains
against more reward from that same shape and overfits to it further, *reducing* greedy
match-vs-rule-bot win rate. The orchestrator's tolerance=0 rejection floor catches this:
iter-1's 0.2768 < iter-0's 0.2845 floor → rejected → iter-2 rolls back to iter-0 as parent.
Iter-2 then *re-trains* against the iter-0 shaped reward distribution and lands at 0.3502 —
this is **not** evidence that the value-head signal helped iter-2; rather it is evidence that
*restarting* from the over-shaped iter-0 with one more pass of PPO can claw back ~6pp from
the iter-1 trough, but cannot recover the ground Phase N covered without the over-shaping.

**Mechanism still healthy in the non-reward dimensions.** Per-iter `ratio_max` 12.7 / 12.3 /
22.4 (gradient strongly active, comparable to Phase L/N), `approx_kl_mean` 0.014 / 0.011 /
0.012 (no anomalies), entropy 0.167 → 0.159 → 0.158 (stable, no collapse),
`numerical_anomalies = 0` across all 48 minibatches. PPO is *fine*; the signal at this
coefficient is wrong.

**What this opens.** The signal isn't *wrong*, it's *too loud*. Phase P proposal: same setup
with `--reward-value-head-coef 0.01` (5× smaller). Per-step contribution becomes ±0.02; at
~60 steps/game, per-game total is ±1.2 — still above the ±0.3 budget but much closer. If
Phase P at 0.01 also regresses, the value-head-delta signal mechanism is genuinely a wrong
shape (not a magnitude issue) — pivot to R7/R8. If Phase P stalls neutrally at ~0.368 (Phase N
ceiling), the value-head signal is redundant with the observation-delta family at low
magnitudes. If Phase P lifts to ≥0.40, first F1 success on record. Launched as
`runs/R15-S3-value-head-tempo-low/` (same warm-start, opponent, HPs, 5 obs-delta coefs at
Phase L 1.0×, constant schedule — only `--reward-value-head-coef 0.01` differs).

**Summary-table addendum.** The 4-phase L/M/N/O' summary is now 5-phase L/M/N/O'/O. Phase O
adds a regression row at iter-2 Wilson 0.3502 alongside the Phase O' 0.3677 sibling.

Artifacts: `runs/R15-S3-value-head-tempo/` — `events.jsonl` (per-iter `ratio_max` /
`approx_kl_mean` / entropy / numerical_anomalies + per-iter
`trajectory-parse/completed.data.shape_attribution.value_head` 26.01 / 9.28 / 8.25),
`orchestrator-state.json` (`promoted_wilson_lower 0.35018663004287004`,
`promoted_iterations: [0, 2]`, `consecutive_failures: 0`, `halted: false`), per-iter
`iteration-manifest.json` + `gate.manifest.json` (model WR 0.324 / 0.316 / 0.392), `launch.log`
(banner shows `--reward-value-head-coef 0.05 --reward-shape-start 1.0 --reward-shape-end 1.0`
confirmed). Wall-clock 322s end-to-end.

### Phase P — F1 PPO + value-head tempo signal, coef 0.01 (R15.S3 GENUINE BRANCH CLOSEOUT, 2026-05-14)

**Verdict: REGRESSED — second value-head data point worse than the first; R15.S3 branch
genuinely closed across both axes.** Run `runs/R15-S3-value-head-tempo-low/` finished clean
(`run_completed`, `halted=false`, 5m 23.2s wall-clock; run_started ts 1778735914.37 →
run_completed 1778736237.61, delta 323.24s). Single change vs Phase O:
`--reward-value-head-coef 0.05` → `0.01` (5× smaller). All else identical: same warm-start
(`runs/item17-2026-05-11/iter-002/model/checkpoint.pt`), same opponent (rule-bot, no pool),
same Phase H HPs (`--lr 3e-4 --clip-epsilon 0.3 --entropy-coef 0.01 --ppo-epochs 4`), same
5 obs-delta coefs at Phase L 1.0× (active 0.02 / bench 0.02 / retreat 0.03 / throughput
0.02 / hp-diff 0.05), same constant schedule (`--reward-shape-start 1.0
--reward-shape-end 1.0`), same `--reward-alpha 0.333 --reward-beta 1.0 --eval-games 250
--iterations 3 --games-per-update 800`.

**Trajectory.** All three iters promoted, monotone increasing, **no rejections** (unlike
Phase O which had iter-1 rejected):

| Iter | Shape | Source | WR | Wilson lower | Games | ratio_max | entropy | approx_kl_mean | numerical_anomalies | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.0 | rollout | 31.2% | **0.2730** | 500 | 12.53 | 0.173 | 0.013 | 0 | **promoted** (baseline; **identical to Phase L iter-0 0.2730 to 4dp** — at coef 0.01 the value-head signal contributes effectively nothing at iter-0) |
| 1 | 1.0 | policy | 32.4% | **0.2845** | 500 | 13.29 | 0.164 | 0.012 | 0 | **promoted** (+1.1pp vs iter-0; **no rejection**, unlike Phase O iter-1 which rejected at 0.2768 < 0.2845 floor) |
| 2 | 1.0 | policy | 38.8% | **0.3463** | 500 | 21.63 | 0.164 | 0.009 | 0 | **promoted** (+6.2pp vs iter-1; **2.1pp regression vs Phase N's 0.3677, AND 0.4pp worse than Phase O's 0.3502**) |

`promoted_iterations: [0, 1, 2]`, `consecutive_failures: 0`, `halted: false`,
`promoted_wilson_lower: 0.34629528411824795`.

**Both magnitudes regressed — the value-head-delta signal mechanism is wrong-shape, not
wrong-magnitude.** Phase O at coef 0.05 landed iter-2 0.3502 (-1.8pp vs Phase N 0.3677);
Phase P at coef 0.01 (5× smaller) lands iter-2 0.3463 (-2.1pp vs Phase N, and -0.4pp
vs Phase O). Lowering the coefficient did not help — if anything it hurt slightly more,
probably because lower coef means less "averaging-out" of the noisy delta signal during
PPO updates so noise dominates more. At coef 0.05 the signal was *loud and over-shaped*
the policy (Phase O iter-1 rejection signature); at coef 0.01 the signal is *quiet and
noisy* and contributes random variance without informational gain (Phase P clean
trajectory but lower ceiling). **Two data points spanning a 5× coef range, both lower
than Phase N's no-value-head baseline — the per-step value-head-delta signal cannot lift
PPO past 0.368 from this warm-start at any tested magnitude.**

**Dimensional check on the coef ratio.** iter-0
`shape_attribution.value_head = 5.027` (Phase P) vs `26.01` (Phase O). Ratio
26.01 / 5.027 = **5.17×**, matching the 5× coef ratio (Phase O coef 0.05 / Phase P coef
0.01) within rounding — the value-head signal is wired and scaled correctly, and iter-0
shape attribution moves linearly with the coef. iter-1 / iter-2 attributions are
1.45 / 1.27 (Phase P) vs 9.28 / 8.25 (Phase O); both show the policy partially flattening
the value-head delta after iter-0, but the absolute level is now small enough that the
flattening contributes neither helpful gradient (Phase O over-shape signature) nor harm
(Phase O iter-1 rejection).

**Mechanism still healthy in non-reward dimensions.** Per-iter `ratio_max` 12.53 / 13.29
/ 21.63 (gradient strongly active, comparable to Phase L/M/N/O ranges); `approx_kl_mean`
0.013 / 0.012 / 0.009 (no anomalies); entropy 0.173 → 0.164 → 0.164 (stable, no
collapse); `numerical_anomalies = 0` across all 48 minibatches. PPO is *fine*; the
information-content of the value-head-delta signal at any tested magnitude does not
contribute usefully to greedy-rule-bot win rate from this warm-start.

**Cross-phase iter-2 Wilson summary — R15.S3 BRANCH CLOSED across both axes.**

| Phase | Signal axis | Config | iter-2 Wilson lower | Δ vs Phase N | Decision |
| --- | --- | --- | --- | --- | --- |
| L | obs-delta | 5 signals, 1.0× coefs, linear-decay | 0.3560 | -0.012 | promoted |
| M | obs-delta | 5 signals, 1.75× coefs, linear-decay | 0.3580 | -0.010 | promoted |
| N | obs-delta | 5 signals, 1.0× coefs, constant | **0.3677** | — (baseline) | promoted (F1 rule-bot ceiling on record) |
| O' | obs-delta | 3 signals (scaled), constant | 0.3677 | +0.000 | promoted |
| O | value-head | + value-head coef 0.05, constant | 0.3502 | -0.018 | promoted (iter-1 rejected; iter-2 from iter-0 parent) |
| P | value-head | + value-head coef 0.01, constant | **0.3463** | **-0.021** | promoted (no rejections; iter-2 from iter-1 parent) |

**R15.S3 final synthesis (both axes exhausted).** Two axes tested:

- **Axis 1 — hand-engineered observation-delta signals.** 4 phases (L decay 1.0×, M decay
  1.75×, N constant 1.0×, O' reduced-mix constant 1.0×). Capped at iter-2 Wilson **0.368
  ± 0.001**. Coef magnitude (L vs M, Δ +0.002), schedule (L vs N, Δ +0.012), signal mix
  (N vs O', Δ +0.000) — all single-axis moves saturated.
- **Axis 2 — per-step value-head-delta tempo signal.** 2 phases (O coef 0.05, P coef
  0.01). Both regressed below Phase N. Ceiling **0.350** at the higher coef; **0.346** at
  the lower coef. Lowering the coef did not help — the signal mechanism is wrong-shape,
  not wrong-magnitude.

**Total R15.S3 branch cost.** ~36 min wall-clock across 6 phases (L 5m54s + M 6m22s +
N 5m25s + O' 5m28s + O 5m22s + P 5m23s). All six runs PPO-healthy in all non-reward
dimensions; no compute-floor or implementation issues found. Cheap research, decisive
answer.

**F1 post-mortem framing — final update.** The prior `87e9e77` framing said: "reward-
shape axis fully explored and converged at 0.368; the gap to 0.40 is categorical — needs
a different information source, not more tuning of the existing observation-delta
signals." Phases O + P then tested *exactly* that — a different information source (the
policy's own value-head delta) — and that information source actively regressed iter-2
at both tested magnitudes. **The F1 reward-shape mechanism cannot break 0.368 from this
warm-start. Per-step shaping from any observation-derived signal saturates at 0.368, and
per-step shaping from the policy's own value-head delta actively regresses. The
remaining F1 moves must change either the warm-start (R7 multi-teacher labels rebuild)
or the optimization objective (R8 DPO replacement). The reward-shape branch is closed.**

**What survives.** Two F1-line candidates remain:
- **R7 multi-teacher labels** — retrain the DAgger SL warm-start with multiple expert
  teachers so the policy that PPO inherits is a different starting point; doesn't touch
  the reward signal, changes the SL pipeline.
- **R8 DPO** — replace PPO with Direct Preference Optimization, which doesn't depend on
  per-step hand-shaped reward signal at all; bigger pivot.

Both are human-rank decisions, not autonomous-launch. Escalation re-opened in
`docs/ai-agent-state/escalations.md` `## Open` with updated framing.

Artifacts: `runs/R15-S3-value-head-tempo-low/` — `events.jsonl` (per-iter `ratio_max`
12.53 / 13.29 / 21.63, `approx_kl_mean` 0.013 / 0.012 / 0.009, entropy 0.173 / 0.164 /
0.164, `numerical_anomalies = 0` × 48 minibatches, per-iter
`trajectory-parse/completed.data.shape_attribution.value_head` **5.03 / 1.45 / 1.27**
— ~5× smaller than Phase O's 26.01 / 9.28 / 8.25, matches the 5× coef ratio),
`orchestrator-state.json` (`promoted_wilson_lower 0.34629528411824795`,
`promoted_iterations: [0, 1, 2]`, `consecutive_failures: 0`, `halted: false`),
per-iter `gate.manifest.json` (model WR 0.312 / 0.324 / 0.388, Wilson 0.2730 / 0.2845 /
0.3463). Wall-clock 323s end-to-end.

## 2026-05-14 Backlog → Progress Migration (archived dated result blocks)

The blocks below were moved verbatim from `docs/ai-research-backlog.md` to bring
the backlog into compliance with the Documentation Discipline rule
("backlog is forward-looking only, target ≤300 lines"). They are preserved here
as the canonical record of the 2026-05-11 and 2026-05-14 R1–R14 / R15.S1–S4
experiments. Original ordering is preserved; section headers retain their
original H3 form. Where the backlog had identical/overlapping write-ups in
existing dated progress-doc sections (Phase J / K / L / M / N / O / O' / P
above), the duplicate is still included for completeness because the backlog
narrative framing sometimes differs from the per-phase write-up.

### Tier 1 results (2026-05-11)

#### R1 — Q1 REFUTED

rollout-CRN×3 vs rule-bot at n=200 (100 games × 2 sides): **WR 58.5%, Wilson95 [0.516, 0.651].**

The teacher comfortably beats the rule-bot. The trained-policy ceiling at 37.5% WR has a **21pp imitation gap** to its own teacher. The cap is not a teacher-strength cap. Whatever the binding constraint is, it lives between the rollout-CRN labels and the trained policy.

This refocuses the research: stop hunting for stronger teachers (search, MCTS) and start asking why the SL fit doesn't capture what the teacher knows. Candidates:

- **Feature representation gap.** State encoding may lose strategic information the rollout uses.
- **Label quality at rare states.** Rollout teacher is strong on common states but bad on rare ones; SL averages.
- **Policy capacity.** 64-hidden × depth-2 may underfit a 58.5%-strength teacher.
- **Mix bias.** Rule-bot replay rows in the mixed corpus may dilute the rollout-labeled rows.

#### R4 — Q3 partially resolved

R4 (50 epochs, value_weight 1.0, fresh from iter-2 mixed data): train accuracy **94.7%** (up from 88%), `calibrate_value` PASS with value_brier 0.084 (down from 0.243), lift_mean +0.134 (Wilson lower +0.119 — significant).

**Value-head miscalibration is fixable** by training longer at higher weight. But the same checkpoint shows gate WR **34.5%** (Wilson [0.283, 0.413]) — slightly below the 37.5% warm-start, with a stark side imbalance (26% as player, 43% as opponent).

Calibration alone doesn't move the gate. The fixed value head matters only if it's used as a critic (PPO advantage signal). Open question: does PPO from R4 produce coherent gradient direction now that the advantages aren't systematically biased?

#### R3 — entropy intervention works, PPO test pending

R3 (β=0.05): final entropy **0.527 nats** per decision (3× the warm-start's ~0.18), train accuracy 82.2% (down ~3pp from warm-start), gate WR **38.5%, Wilson [0.320, 0.454]** — statistically equivalent to the warm-start.

**Peakedness intervention succeeds without hurting gate WR.** PPO from this warm-start is running; the critical question is whether the higher entropy lets PPO's importance ratios stay non-trivial and produce coherent gradient direction.

#### Side imbalance — opportunistic finding

Across DAgger iter-2 warm-start, R3, R4 gate evals: the trained policy consistently performs much better as the *opponent* than as the *player* (DAgger 41% vs 30%; R4 43% vs 26%; R3 not yet broken out). The model is offensively weak. This may be its own research question (does training set under-represent player-side decisions? feature parity issue?) or a manifestation of the same imitation gap.

R-WILD's side-imbalance subitem should be promoted to its own Tier-1-adjacent task.

#### R3+PPO and R4+PPO — interventions don't break the gate cap

PPO from R3 (entropy-regularized warm-start) and PPO from R4 (calibrated-value warm-start) both land at the same ~0.31 Wilson lower ceiling that every prior F1 phase hit:

| Run | Warm-start | Final Wilson lower | Best WR |
| --- | --- | --- | --- |
| F1 phase 2 (defaults at 30g) | DAgger iter-2 | 0.3109 | 37.5% |
| F1 phase G (spec defaults at 800g) | DAgger iter-2 | 0.3156 | 38.0% |
| F1 phase H (aggressive HPs at 800g) | DAgger iter-2 | 0.3109 | 37.5% |
| R3+PPO (high entropy + aggressive HPs) | R3 entropy-reg | 0.3156 | ~38% |
| R4+PPO (calibrated value + aggressive HPs) | R4 retrain | 0.3014 | ~36% |

R4+PPO's GAE `mean_advantage` improved from phase-H's −0.10 to −0.03 (calibration *did* help the value baseline). Per-minibatch KL rose from 0.008 to 0.022 — PPO made slightly bigger updates. **But none of it moved the gate WR.**

#### Diagnostic finding: the peakedness diagnosis was wrong

The "warm-start entropy is 0.18 nats" claim from F1 phase F was based on `train_ppo`'s per-minibatch entropy metric, which apparently reports something different than the policy entropy on the training distribution. Directly measuring `masked_policy_entropy(model_logits, action_mask)` over `iter-002/mixed.jsonl` shows:

| Checkpoint | Policy entropy on mixed.jsonl |
| --- | --- |
| DAgger iter-2 (warm-start) | **0.532** |
| R3-b005 (β=0.05 entropy bonus) | 0.528 |
| R3-b020 (β=0.2 + value_weight=1.0 + 50ep) | 0.386 |
| R3+PPO iter-2 | 0.574 |
| R4 (value re-train) | (not measured but accuracy 94.7% suggests low) |

So the warm-start's actual entropy is comparable to R3's — the "peakedness mechanism" we built R3 around was a misdiagnosis. R3+PPO didn't help because the warm-start wasn't actually peaked.

#### Diagnostic finding: imitation accuracy ≠ play strength

argmax-match rate against rollout-CRN teacher labels on `iter-002/relabeled.jsonl`:

| Checkpoint | argmax-match | Gate WR |
| --- | --- | --- |
| DAgger iter-2 (warm-start) | 74.7% | 37.5% |
| R3-b005 | 75.2% | 38.5% |
| R4 (50ep, value_weight=1.0) | 81.0% | 34.5% |
| R3-b020 (combined) | 80.9% | (gate eval pending) |

**Better imitation does not yield better play.** R4 imitates the teacher 6pp more accurately and plays 3pp worse. This rules out the simple "fit the labels harder" path: the teacher's 19–25% argmax errors must concentrate on a small set of high-leverage states (combat finishers, evolution decisions) where one wrong move blows the game.

#### The 0.31 cap is structural

Combining R1 (teacher at 58.5%) + R3 + R4 + R3-b020 (planned) + 5 F1 PPO phases:

- Teacher strength: 58.5% (Wilson lower 0.516). Not the cap.
- Imitation accuracy: 74–81%. Not the limit — improving it doesn't move gate WR.
- Value head calibration: now PASS. Doesn't move gate WR.
- Policy entropy at warm-start: 0.5 nats. Already non-peaked.
- PPO HPs: tried spec, aggressive, extreme. Always 0.30 ± 0.01.

**The gate WR ceiling at 0.31 is structural** — not from any single mechanism we've tested. The remaining hypotheses:

1. **State-feature gap.** The 25% argmax disagreements between trained policy and teacher concentrate on high-leverage decisions; the policy can't tell those states apart in feature space.
2. **Sample-weight bias.** Low-margin rollout labels (where teacher itself was unsure) get the same weight as high-margin ones; training picks up noise.
3. **Action-coverage gap.** Candidate ranker filters out some actions the teacher would pick; the BC fit can't recover those.
4. **Side asymmetry.** The 26–30% player vs 41–43% opponent gap suggests a systematic blind spot in player-side decisions.

#### Pivot

Tier-1 verdict: peakedness and calibration are not the binding constraints. Pivot to:

- **R5 (PFSP self-play)** — different opponent distribution, different reward landscape. Smallest infra delta.
- **Margin-weighted training** (new Tier-2 idea) — sample weights by `selectedVsRunnerUpMargin` so high-margin rows get more training emphasis.
- **R-WILD side imbalance** — measure where in the game the player-side gap comes from; might suggest a feature gap or a data-skew fix.

R5 is the next concrete experiment. R6/R7/R8/R10 stay on the backlog as larger-cost asymmetric-upside bets.

#### R6 (capacity) — overfit teacher labels, plays worse

Trained `--hidden-dim 128 --depth 3 --epochs 50 --value-weight 1.0` on iter-002 mixed data:

| Metric | Warm-start (64/2) | R4 (64/2, 50ep) | R6 (128/3, 50ep) |
| --- | --- | --- | --- |
| Train accuracy | 0.88 | 0.95 | **0.97** |
| Argmax-match vs teacher | 0.747 | 0.810 | **0.832** |
| value_brier | 0.243 | 0.084 | **0.062** |
| Gate WR | 37.5% | 34.5% | **33.0%** |

R6 is the strongest imitator (83% argmax-match, 97% accuracy, best calibration) AND the weakest player. **The cap isn't a capacity issue — it's an imitation-target-quality issue.** The teacher's 25% argmax errors get inherited by the SL fit; bigger model just locks them in more cleanly.

#### Combined intervention (R3-b020 + PPO) — same 0.30 cap

PPO from R3-b020 (entropy 0.39 + calibrated value + accuracy 89%) at aggressive HPs:

| Iter | Wilson lower | Decision |
| --- | --- | --- |
| 0 (warm-start eval) | 0.3014 | promoted |
| 1 | 0.2639 | rejected |
| 2 | 0.2826 | rejected → halt |

Max 0.3014 — same as every other PPO sweep. **Combining peakedness fix + calibration fix + aggressive HPs doesn't break the cap.**

#### R5 (PFSP self-play) — opponent distribution changed, gate cap held

PPO from DAgger iter-2 warm-start, rollouts vs the item17 opponent pool (DAgger iter-002 sampled uniformly). 3 iters × 800 games × aggressive HPs:

| Iter | Wilson lower | GAE mean_return | GAE mean_advantage |
| --- | --- | --- | --- |
| 0 (warm-start eval) | 0.2873 | -0.11 | -0.07 |
| 1 | 0.2500 | -0.14 | -0.04 |
| 2 | 0.3109 (matched DAgger iter-2 exactly) | -0.14 | -0.04 |

Self-play DID materially change the reward distribution: mean_return moved from -0.20 (vs rule-bot rollouts in phase H) to -0.14 (more even games against a same-strength opponent). Mean_advantage less negative too. **But gate WR (vs rule-bot, the unchanged evaluator) still locks at 0.3109.**

The opponent change perturbs the trained policy's locality but doesn't help against the *eval* distribution. The policy learns to do something different against itself, but that something different doesn't generalize to rule-bot.

#### Imitation-cap statement

After 8 PPO sweeps + 4 BC variants + 1 capacity bump + 1 self-play:

> **No combination of warm-start adjustment + PPO HP tuning + opponent distribution broke the Wilson-lower 0.31 ceiling vs rule-bot.** Better imitation, larger models, calibrated values, higher entropy, longer training, bigger PPO buffers, self-play opponents — every well-behaved variant lands at WR ≈ 33–38%, Wilson lower 0.27–0.32. The 0.31 cap is the imitation cap: SL on a 58.5%-WR teacher whose 41% disagreement rows are noisy at decision-critical states, and PPO can't escape its local optimum without a different training signal.

To break it, we need labels or training signal that isn't just "imitate the teacher harder":

- **R5 (PFSP self-play)** — different opponent, different reward shape, possibly different gradient direction. Running.
- **R7 (multi-teacher BC)** — different label distribution. Reduces single-teacher mode-collapse.
- **R8 (DPO)** — different objective. Trains on preference pairs instead of argmax labels.
- **Margin-weighted training** — weight samples by `selectedVsRunnerUpMargin`; high-margin rows are the teacher's confident decisions and likely the high-leverage ones.
- **Reward shape refinement** — Δpoints + ±1 win/loss may miss strategic value. F1 could find direction with richer reward.

#### Diagnostic: forced-move dominance

Inspection of iter-002 relabeled rows (n=2883):

| Legal action set size | Row count | Fraction |
| --- | --- | --- |
| 1 (forced) | 1776 | **61.6%** |
| 2 (binary) | 544 | 18.9% |
| 3 | 321 | 11.1% |
| 4–10 | 200 | 6.9% |
| 10+ | ~42 | 1.5% |

JsonlPolicyDataset's `min_actions=2` correctly excludes forced rows from training. But the gate-eval games include them — every game has ~62% of its "decisions" forced, meaning the strategic difference between two policies plays out over ~38% of state transitions, mostly binary choices.

This is the **lever density problem**. A 4% WR gap between policies has to be earned on the ~15 meaningful decisions per game (out of ~40 total). Those few decisions are exactly where teacher labels are most likely noisy (because they're high-leverage). The cap is concentrated in the highest-leverage 1/3 of decisions per game.

#### Removing rule-bot replay mix (R-cleanlabels)

Trained BC on JUST `iter-002/relabeled.jsonl` (no rule-bot replay mix-in):

| Variant | Training data | Train accuracy | Gate WR |
| --- | --- | --- | --- |
| DAgger iter-2 (mix=70% rollout-relabeled + 30% rule-bot-replay) | mixed | 0.85 | 37.5% |
| R-cleanlabels (rollout-relabeled only) | clean | 0.92 | **33.0%** |

Removing the rule-bot replay actually *hurts* (-4.5pp). The replay buffer acts as regularization / state coverage; without it the model overfits the smaller relabeled set. **The mix is doing useful work**, not adding noise.

#### Pause — single-axis interventions exhausted

Tier 1 (R1–R4) plus stretch experiments R5 (self-play), R6 (capacity), R-cleanlabels all converge on: the 0.31 gate cap is robust to every single-axis intervention. Remaining promising bets require either multi-axis combinations or fundamentally different approaches:

1. **Multi-teacher BC (R7)** — relabel iter-2 trace with planner + search teachers, train on the union. Tests whether label diversity at high-leverage decisions matters. (Compute estimate: planner relabel is slow, maybe 30 min wall-clock.)
2. **DPO (R8)** — train on (selected, runner-up) preference pairs from outcome export. Different objective entirely; doesn't fit argmax labels. (Code effort: ~3 hours for a new trainer.)
3. **MCTS-augmented self-play (R12)** — skip the warm-start + RL split. Game is small enough. (Code effort: ~1 week.)
4. **State-feature audit** — what's missing in features that would let the model tell apart high-leverage decisions? (Diagnostic, not a fix.)

User-facing summary of where we are: the infra works, the cap is structural, and breaking it requires a directional decision about how much code to invest. None of (1)–(4) is going to take less than half a day; (3) is multi-day.

### Decision (2026-05-11): commit to R12, fallbacks ready

After Tier-1 + stretch experiments + strategist + auditor analysis, the chosen path is **R12 (mini-AlphaZero)**. Reasoning recorded in detail in the chat thread; key points:

1. **R6 is the dispositive evidence.** Better imitation makes play worse. Every variant that fits the rollout-CRN labels harder (R4, R3-b020, R6) drops gate WR below the warm-start. The cap isn't "fit labels harder" — it's "the labels are noisy on high-leverage decisions."
2. **R7 and R8 inherit the noise.** Both train on the same rollout-CRN argmax/preference data. Different averaging / objective; same target distribution.
3. **R12 generates new labels via search.** Visit-count distributions from PUCT MCTS with N=100 sims integrate over the variance that single-rollout-CRN samples once. Label quality scales with compute (search depth) instead of being capped at teacher's single-sample noise floor.
4. **Game structure favors MCTS.** 62% forced moves means search budget concentrates on the ~15 meaningful decisions per game. At our current simulator throughput, 100 sims × 15 real decisions × 200 games = 25–35 minute wall-clock per gate.

**Sprint plan: see `docs/archive/ai-research/sprints/r12-sprint-plan.md`.** Day-1 spike has a hard go/no-go criterion (Wilson lower ≥ 0.40 at n=100). If NO-GO, write a postmortem and pivot to fallbacks (R7, R8, R9, R10) in the order ranked by tier.

R7/R8/R9/R10 are kept on the backlog as fallbacks; their task descriptions are annotated to reflect their fallback status.

#### R12 implementation progress (2026-05-11)

All sprint phases are coded and smoke-validated. Day-1 spike at 100 games × 100 sims is running; pending result.

| Phase | Artifact | Smoke result |
| --- | --- | --- |
| Day-1 | `backend/src/sim/mcts.ts`, `training/r12_spike.py`, `r12_spike_smoke.py` | smoke PASS (8 games, 0 fallbacks). 100-game gate in flight. |
| A | `mcts.ts` policy-prior + Dirichlet noise (`--mcts-prior policy`, `--mcts-root-dirichlet`) | TS build clean; activated by Phase B/D smokes. |
| B | `backend/src/sim/mctsSelfPlay.ts`, `npm sim:mcts-selfplay`, `r12_selfplay_smoke.py` | PASS — 2 games × 8 sims → 65 rows, schema valid. |
| C | `training/uma_ai/selfplay_dataset.py`, `train_bc.py --data-mode mcts-distill`, `r12_distill_smoke.py` | PASS — 2 epochs, loss 1.75 → 1.47, accuracy 61% → 72%. |
| D | `training/r12_orchestrator.py`, `r12_orchestrator_smoke.py` | PASS — 1 iter × 4 games × 8 sims, all 10 expected event_types emitted. |
| E | `observability_app.py` STAGES extension (`selfplay`, `distill`, `mcts-gate`, `mcts-spike`, `r12-orchestrator`) | n/a (dashboard render check, no smoke). |

Decision: Day-1 spike is the gate on whether to launch a Phase D multi-iteration run. If GO, run 4 iterations × (200 games / 100 sims) per the plan; on the trained Phase-D output, run final 400-game gate at `--mcts-simulations 200` for the headline ≥0.40 Wilson-lower target.

#### Day-1 spike: uniform-prior result (NO_GO, but signal-positive)

Spike stopped at n=178 (full schedule was 200; halted early after the trajectory committed to NO_GO).

| Metric | R4 value-head (1-ply) | Day-1 MCTS uniform | Δ |
| --- | --- | --- | --- |
| WR | 0.345 | **0.4045** | +6pp |
| Wilson lower | 0.30 | **0.335** | +3pp |
| Heuristic fallbacks | 0 | 0 | — |
| Player WR | ~0.27 | 0.36 | +9pp |
| Opponent WR | ~0.43 | 0.462 | +3pp |

**Interpretation.** MCTS with uniform prior + R4 value-head leaf is **better than the value-head alone** at the same checkpoint — search adds 6pp WR. But it does not clear the ≥0.40 Wilson-lower bar. The side gap (10pp player vs opponent) persists and is the largest single sink: if both sides hit the opponent-side WR (~46%), we'd land at Wilson lower ~0.39 — almost at the bar from uniform prior alone.

**Day-1 verdict: NO_GO** on the hard criterion. **But** the structural intervention (MCTS over R4) does add positive value, so the next iteration of the spike is warranted instead of skipping straight to a fallback.

#### Phase A spike: policy-prior retry (NO_GO, marginal improvement)

n=200, --mcts-prior policy + R4 value-head leaf, otherwise identical to day-1.

| Metric | Phase A | Day-1 uniform | R4 baseline |
| --- | --- | --- | --- |
| WR | 0.4350 | 0.4045 | 0.345 |
| Wilson lower | 0.368 | 0.335 | 0.30 |
| Player WR | 0.36 | 0.36 | 0.27 |
| Opponent WR | 0.51 | 0.46 | 0.43 |

Policy prior gave +3pp Wilson lower over uniform — meaningful but not enough for GO. The diagnostic finding: **the prior is not the bottleneck**. Search budget reorganization helps modestly, but Wilson lower is still 3pp short of the 0.40 bar.

#### Rollout-leaf spike: GO (2026-05-11)

Same R4 ckpt, same policy prior, same 100 sims — only the leaf evaluator changed from `value-head` to `rollout` (K=3 rule-bot playouts to terminal, side-relative ±1/0 backed up).

| Metric | Rollout-leaf | Phase A | Day-1 uniform | R4 baseline |
| --- | --- | --- | --- | --- |
| WR | **0.625** | 0.435 | 0.405 | 0.345 |
| Wilson lower | **0.556** | 0.368 | 0.335 | 0.30 |
| Wilson upper | 0.689 | 0.504 | 0.478 | — |
| Player WR | 0.58 | 0.36 | 0.36 | 0.27 |
| Opponent WR | 0.67 | 0.51 | 0.46 | 0.43 |
| Heuristic fallbacks | 0 | 0 | 0 | 0 |

**R12 north star cleared by +15.6pp** (target 0.40 vs achieved 0.556) and **stretch criterion (0.50) also cleared**. Side gap inverted: player 58%, opponent 67%, both winning majority.

**The dispositive diagnostic.** Search adds ~3pp over R4 (uniform-prior MCTS). Policy prior adds another ~3pp. Swapping the value-head leaf for rollout-CRN at leaves adds **+19pp**. The value head — even at R4's calibrated 0.084 Brier — was producing leaf evaluations too noisy for 100-sim MCTS to disambiguate the per-action means. Rollout-CRN K=3 averages directly over the same outcome distribution the value head approximates; the per-action means rank correctly.

**Implications.**
- The "trained policy + MCTS at inference" path of the R12 north star is achieved. Deploy is rollout-leaf MCTS over the R4 checkpoint.
- The "distilled policy without search" path is not achieved by definition — rollout-CRN at leaves IS search. A distilled policy reusing the value head would inherit the same noise floor R4 hit.
- Phase D distillation is now optional: it would compound iter-on-iter and produce a stronger prior, but is not required to meet the criterion.

#### Recommended next steps (post-R12 GO) → R13 sprint

See `docs/archive/ai-research/sprints/r13-sprint-plan.md` for the detailed plan. Headline shift:

**Phase D as originally written (visit-count → policy distillation) is no longer the obvious next step.** The R12 diagnostic shows the value head is the bottleneck; a distilled policy would inherit that noise floor. Instead, the next sprint asks a sharper question:

> Is the value head fixable, or is rollout-CRN search permanently the production path?

The cheap falsifiable answer: retrain JUST the value head on rollout-mean outcomes (not game-z), freeze trunk + policy, gate at `--mcts-leaf value-head`. ~4 hours of compute. If Wilson lower ≥ 0.40 with the new value head, distillation is unlocked. Otherwise, search-at-inference is the permanent answer.

In parallel: game-level parallelism (~4× speedup), latency dials (K=1 / adaptive sims / batched /predict) to push p95 decision time under 3 s, UI integration, and an MCTS-vs-MCTS strength ladder so we stop relying solely on a saturating rule-bot.

#### R13.W6 result — Phase D iterations compound (2026-05-11)

2 production iterations of `r12_orchestrator` from the W3-retrained warm-start (60 selfplay × 100 sims × rollout-leaf K=3; 20 epochs of mcts-distill with KL anchor 0.05; 120-game gate per iter).

| Iter | WR | Wilson lower | Δ vs prev |
| --- | --- | --- | --- |
| 0 | 0.625 | 0.536 | +5.3pp vs R12 baseline (0.483 / 0.556 at n=200) |
| 1 | **0.658** | **0.570** | +3.4pp vs iter-0 |

Iter-1 promoted as the new strongest model. Per-side: player 0.667 / opponent 0.65 — **R-WILD side gap is closed** (was historically ~16pp opp-favored, briefly 9pp player-favored in R12, now within 2pp). Zero heuristic fallbacks → MCTS execution is clean. Iter-1 checkpoint: `runs/R13-W6-phase-d/iter-1/checkpoint.pt`.

#### R13.W8 result — value-head-only Phase D does NOT compound (2026-05-11)

Tried 5 iterations of `r12_orchestrator` from W6 iter-1 with `--mcts-leaf value-head` for selfplay (instead of rollout). Killed at iter-3 mid-gate after the trend was clear:

| | Wilson lower | WR |
| --- | --- | --- |
| W6 iter-1 baseline (start) | 0.452 | 0.55 |
| W8 iter-0 | 0.404 | 0.49 |
| W8 iter-1 | 0.340 | 0.43 |
| W8 iter-2 | 0.380 | 0.47 |

Every iteration was below the starting baseline. Conclusion: **cheap-leaf selfplay targets are too noisy for distillation to compound** — the visit-count targets from value-head-leaf selfplay are noisier than rollout-CRN-K=3 targets, and distillation regresses strength rather than improving it. Production iteration *requires* rollout-leaf selfplay even if gate-time inference is value-head leaf.

Practical implication: W6's "use rollout-leaf for selfplay, value-head-leaf for inference" decomposition is load-bearing — both halves matter. Don't try to cheap-out the training loop.

### R14 sprint summary blocks (2026-05-11 → 2026-05-14)

#### R14 sprint — refinement (2026-05-11)

See `docs/archive/ai-research/sprints/r14-sprint-plan.md`. After R13's two production configs landed, the next sprint splits between shipping (W5 UI finish + Pareto-tuned latency) and one final honest RL attempt (MCTS-trajectory off-policy PPO). Also includes the engine determinism fix that closes R-WILD — a subagent investigation pinpointed `withRng` losing `activeRng` across `await` boundaries; AsyncLocalStorage is the ~10-LOC fix.

R14 workstreams:
- **A** OOD gate for iter-1 (compute only, 30 min) — falsifies "iter-1 value-head leaf overfits its own selfplay distribution"
- **B** Engine determinism fix via AsyncLocalStorage (~1 hour) — closes #34 if it works
- **C** W8 stop rule after iter-2 — concave-compounding guard
- **D** MCTS-trajectory PPO with V-trace (~3 days code + 1 day compute) — the only PPO variant we never honestly ran
- **E** W5 UI integration finish (~1 day code) — the deliverable
- **F** Adaptive-ratio Pareto sweep (~20 min) — picks the W5 default config

Dropped: larger model, more entropy-BC variants, n=400 headline gate (cosmetic), temperature-ramp-only PPO (R3 already settled peakedness-alone).


#### R14 progress checkpoint (started 2026-05-11; finished 2026-05-14)

**Sprint outcome:** code-side workstreams all landed. **Primary production claim: rollout-leaf MCTS @ iter-2, Wilson lower 0.6479** (R14.I.2) — max strength, untouched by the side-asymmetry confirmation gate. **Cheap-inference fallback: value-head leaf + adaptive-ratio=1.5 @ iter-2, Wilson lower 0.39–0.45 across three independent seed ranges** (F-iter-2 0.443 at 820000+ / A 0.404 at 800000+ / A.footnote 0.394 at 900000+), 1-of-3 below the 0.40 production bar at 2.56× speedup vs ratio=0. The 0.452 reading originally headlined from F-iter-2 was the upper end of that empirical range at the F seed range, not a stable point estimate (resolved 2026-05-14 after R14.A.footnote — see `docs/ai-agent-state/escalations.md` `## Resolved`). The F-iter-1 sweep had earlier shown the 0.42 target was unreachable on iter-1 at the F seed range; re-running on iter-2 PASSes it across all ratios at the F seed range. The original A "side asymmetry is partly seed-clustered" caveat was refuted by A.footnote — the player/opponent gap (+0.11–0.30pp Wilson) replicates across four gates and two independent seed ranges. Engine determinism (R14.B) and orchestrator inspection (R14.I.2 inspector) shipped. UI plumbing (R14.E) verified end-to-end via headless smoke (decisionMs=200 at 16 sims → ~0.5s at 100 sims with ratio=1.5, comfortably under E's <3s target); only the 20-game in-browser exercise remains unautomatable. ORT unpinning (R14.G) showed 1.05× — kept at `--ort-threads 1`. PPO-with-V-trace (D) and predict-batching (H) skipped per the sprint plan's decision points (I succeeded; G negative). After the sprint re-refinement promoted I to top and demoted D to fallback:

- **B DONE.** AsyncLocalStorage installed via `installRngStorageProvider` + side-effect `backend/.../rngAsyncStore.ts`. The frontend keeps its sync-module fallback so the browser bundle stays clean of `node:async_hooks`. `training/r14_determinism_smoke.py` now passes bit-exact 12/12 between --workers 1 and --workers 4 at value-head MCTS, n=12, seeds 141414+. Pre-fix: divergent. R-WILD #34 closed.
- **I.1 DONE.** `training/r14_value_crossover_probe.py` measures (val_mse, pearson_r) between a checkpoint's value head and the rollout-CRN K=3 means in `rootValue`. crossed = (val_mse <= 1.10 × W3-floor) AND (pearson_r >= 0.7). Wired into `r12_orchestrator.run_iteration` between distill and gate, against the **previous** iter's selfplay (held-out). Baseline at W6 iter-1 vs its own iter-1 selfplay (in-distribution): val_mse 0.659, pearson 0.535, ratio 1.158, crossed=false — explains the W8 regression (cheap-leaf selfplay distillation failed because the value head hadn't caught up).
- **I.2 DONE (halted at iter-4).** W6 phase-d extended through iter-4. iter-2 promoted at Wilson **0.6479** (WR 0.7333, n=120), clearing the sprint's ≥0.60 north star. iter-3 dropped to 0.5783 and iter-4 to 0.5612 — two consecutive promotion failures triggered the orchestrator's auto-halt. Crossover probe never satisfied both gates: pearson_r stayed ~0.50 (target ≥0.7), ratio stayed 1.18–1.29 (target ≤1.10). So I.3 (cheap-selfplay retry after two `crossed=true` events) never fired and is dropped from the sprint. Strongest checkpoint: `runs/R13-W6-phase-d/iter-2/checkpoint.pt`. (Note: the earlier orchestrator dim-mismatch crash was fixed by auto-inferring hidden_dim/depth from the init checkpoint — commit `1c47146`.)
- **E DONE (plumbing); 20-game manual UI exercise pending.** Visible MainMenuScreen toggle in (commit `0371892`). `training/r14_ai_decide_e2e_smoke.py` PASS (2026-05-14): real mid-game state via headlessAiVsAi → /ai/decide → returns valid action + nextState (fingerprint advances). decisionMs=200 at 16 sims → extrapolates to ~1.25s at 100 sims (under E's <3s target). Only the in-browser 20-game fallback-rate exercise remains; not headless-automatable.
- **G DONE (NEGATIVE; 2026-05-14).** Smoke: pinned 13.4s vs auto 12.8s, speedup **1.05×** vs 1.5× target → FAIL on speedup, PASS on determinism (12/12 winner+turnNumber match). At this model size, per-call /predict overhead dominates compute, so ORT thread parallelism doesn't help. Decision: keep `--ort-threads 1` default (R13.W1 legacy preserved), skip H predict-batching (same overhead ceiling).
- **F DONE (Pareto pick = ratio 2.0; 2026-05-14).** Sweep on iter-1 + value-head leaf, n=100 each. Stated FAIL (0.42 Wilson target unreachable at fresh seeds 820000+; ratio=0 baseline only 0.366) but ratio=2.0 Pareto-dominates baseline: +4pp WR (0.50 vs 0.46), 2.19× speedup (118s vs 258s). Higher ratios over-prune. Production pick for W5 default: ratio=2.0. See dedicated R14.F section below.
- **A DONE (PASS on iter-2; 2026-05-14).** Re-targeted to iter-2 (newly-promoted production candidate) — see the dedicated A result section below. Both gates pass at 0.40 floor; iter-2 beats R4 head-to-head MCTS by 11pp Wilson lower; side asymmetry persists; +18pp rollout-leaf gain does not transfer to value-head-leaf inference.

#### R14.F — Adaptive-ratio Pareto sweep on W6 iter-1 (2026-05-14)

Ran `training/r14_adaptive_ratio_sweep.py` on iter-1 + value-head leaf, 100 sims, n=100 per ratio, seeds 820000+, 4 workers, single shared serve_onnx. Sweep complete:

| ratio | WR | Wilson lower | Wilson upper | elapsed (s) | speedup vs baseline | wallclock cut |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 0.46 | 0.366 | 0.557 | 258.0 | 1.00× | 0% |
| 1.5 | 0.46 | 0.366 | 0.557 | 123.0 | 2.10× | 52% |
| **2.0** | **0.50** | **0.404** | **0.596** | **117.6** | **2.19×** | **54%** |
| 3.0 | 0.45 | 0.356 | 0.548 | 155.0 | 1.66× | 40% |
| 5.0 | 0.49 | 0.394 | 0.587 | 181.3 | 1.42× | 30% |

**Stated verdict: FAIL** (no ratio satisfies the original exit criterion `Wilson lower >= 0.42`). But that target was set against the W6 paper baseline of 0.452 at seeds 700000+. On the fresh seeds used here, even ratio=0 baseline is only 0.366 — the 0.42 target is unreachable for any ratio. The exit criterion did not anticipate the seed-distribution variance later confirmed by R14.A.

**Real verdict:** ratio=2.0 **Pareto-dominates baseline** on both axes — +4pp WR (0.50 vs 0.46), 2.19× speedup. Wilson CIs at n=100 overlap heavily ([0.366, 0.557] vs [0.404, 0.596]) so the strength gain is within sample noise; the **safe claim is "no strength regression at 2.19× speedup"**. Above ratio=2.0 the curve is concave: 3.0 over-prunes (-1pp WR, slower because of bookkeeping), 5.0 partial recovery in WR but slower still.

ratio=1.5 is bit-identical strength to baseline (same wins on the same seeds — same Wilson) at 2.10× speedup — confirms adaptive halts only fire when they don't change the chosen action; below ratio=2.0 the halt rule never triggers cases where it could disagree with full search.

**Production pick:** ratio=2.0 for W5 UI / E default config. The 2.19× speedup roughly halves median decision wall-clock — directly relevant to E's "<3s decision time" exit criterion.

**Followup before locking:** re-run the sweep on iter-2 (the new I.2 production candidate). iter-2 may shift the optimum (different value-head profile → different halt-rule firing pattern). Cheap (~25 min) but only do this once iter-2 deployment is closer to landing.

Output: `runs/R14-adaptive-sweep/{ratio-*.{log,manifest.json,progress.jsonl},summary.json}`.

#### R14.F — Adaptive-ratio sweep re-targeted on iter-2 (2026-05-14)

Re-ran the F sweep on the I.2-promoted checkpoint `runs/R13-W6-phase-d/iter-2/checkpoint.pt` (same script, same seeds 820000+, n=100, 100 sims, value-head leaf). On iter-2 all five ratios clear the 0.42 Wilson floor — the F exit criterion is satisfied here in a way it could not be on iter-1.

| ratio | WR | Wilson lower | Wilson upper | elapsed (s) | speedup | wallclock cut |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 0.54 | 0.443 | 0.634 | 269.0 | 1.00× | 0% |
| **1.5** | **0.55** | **0.452** | 0.644 | **105.0** | **2.56×** | **61%** |
| 2.0 | 0.55 | 0.452 | 0.644 | 132.7 | 2.03× | 51% |
| 3.0 | 0.53 | 0.433 | 0.625 | 154.1 | 1.75× | 43% |
| 5.0 | 0.57 | 0.472 | 0.663 | 175.7 | 1.53× | 35% |

**Status: PASS** (script's rule auto-picks ratio=5.0 as highest-passing ratio).

**Pareto frontier on iter-2:**
- **ratio=1.5** (max speed, recommended W5/E cheap-inference fallback pick): Wilson 0.452 at the F seed range + 2.56× speedup. Decision time at 100 sims extrapolates to ~0.5s — well under E's <3s target with margin to spare. The 0.452 reading here is the R14.A canonical-seed-range upper end; the empirical range across three independent seed ranges (F-iter-2 / A / A.footnote) is **Wilson lower 0.39–0.45** with 1-of-3 below the 0.40 bar (R14.A.footnote 2026-05-14). Primary production claim is rollout-leaf MCTS @ iter-2 Wilson 0.6479.
- **ratio=5.0** (max strength): Wilson 0.472 + 1.53× speedup. Worth +2pp Wilson if compute is cheap, but loses 1.7× of the speedup.
- **ratio=2.0** is strictly dominated by 1.5 on iter-2 (same Wilson, slower) — different from the iter-1 sweep where 2.0 was the Pareto pick. iter-2's value head produces a different halt-rule firing pattern.

**Headline (qualified post-A.footnote, 2026-05-14):** value-head-leaf iter-2 at ratio=1.5 reads **Wilson lower 0.452 at the F seed range (820000+)** at ~0.5s/decision vs R13.W6's reported iter-1 at ~1.25s/decision; across three independent seed ranges (F-iter-2 / A / A.footnote) the empirical range is **Wilson lower 0.39–0.45** with 1-of-3 below the 0.40 production bar — the 0.452 number is the upper end, not a stable point estimate. The W6 → I.2 training translates to a ~2.5× speed-equivalent strength gain at the cheap-inference fallback deployment point. Primary production claim is rollout-leaf MCTS @ iter-2 Wilson 0.6479 (R14.I.2).

**Implication for the A side-asymmetry caveat:** F-iter-2's seeds 820000+ show iter-2 at 0.443 Wilson (n=100), but A's seeds 800000+ showed 0.404 Wilson (also n=100). The 4pp swing across overlapping seed ranges suggests the side-asymmetry from A is partly seed-clustered. Worth a confirmation gate at a third seed range before deployment.

Output: `runs/R14-adaptive-sweep-iter2/{ratio-*.{log,manifest.json,progress.jsonl},summary.json}`.

#### R14.A — OOD gate result on W6 iter-2 (2026-05-14)

Ran `training/r14_ood_gate.py` against the I.2-promoted checkpoint `runs/R13-W6-phase-d/iter-2/checkpoint.pt`. Re-targeted from the originally-specified iter-1 because iter-2 is the newly-promoted production candidate (and had no value-head-leaf number yet). Two 100-game gates, value-head leaf, 100 sims, 4 workers, seeds 800000+ / 850000+.

| Gate | n | WR | Wilson lower | Wilson upper | Verdict |
| --- | --- | --- | --- | --- | --- |
| 1: fresh seeds, vs rule-bot | 100 | 0.50 | **0.404** | 0.596 | passes 0.40 marginally |
| 2: iter-2 vs R4, MCTS-vs-MCTS | 100 | 0.61 | **0.512** | 0.700 | passes 0.40 by 11pp |

Per-side breakdown reveals severe asymmetry on both gates:

| Gate | Player WR | Player Wilson lower | Opponent WR | Opponent Wilson lower |
| --- | --- | --- | --- | --- |
| 1 (vs rule-bot) | 0.44 | 0.312 | 0.56 | 0.423 |
| 2 (vs R4 MCTS) | 0.46 | 0.330 | 0.76 | 0.626 |

**Interpretation.**

1. **Both gates pass — iter-2 is OOD-robust.** Strength claim is not seed-distribution-specific.
2. **iter-2 dominates R4 head-to-head MCTS** (gate 2 Wilson lower 0.512). Gate 2 stronger than gate 1 by ~11pp: iter-2 is a genuine model improvement over R4, not a rule-bot artifact.
3. **iter-2's value-head-leaf strength is statistically indistinguishable from iter-1's.** Wilson lower 0.404 (iter-2) vs 0.452 (iter-1's R13.W6 reported number). The +18pp Wilson gain from iter-1 → iter-2 at rollout-leaf MCTS (0.452 → 0.6479) does **not** transfer to value-head-leaf inference. iter-2's added training improved the policy/rollout combination, not the value head's ability to score leaves directly.
4. **Side asymmetry persists post-B.** AsyncLocalStorage closed the parallel-determinism gap, but iter-2 is materially weaker as player (Wilson lower 0.31–0.33 across both gates) than as opponent (0.42 vs rule-bot, 0.63 vs R4). This is not a determinism bug — both sides are bit-exact reproducible — but a real *strategic* asymmetry in the iter-2 policy at value-head-leaf inference. Likely tied to first-move/initiative dynamics: the model handles defending better than initiating. The production claim should disclose the side gap.
5. **Cheap-inference production config:** rollout-leaf iter-2 at Wilson 0.6479 remains the headline. Value-head-leaf iter-2 at Wilson 0.404 is the cheap-inference fallback — defensible but the side asymmetry caveat sticks.

Followups (low priority):
- Side-asymmetry-specific gate (player-only n=200) to tighten the per-side Wilson CI before any deployment claim.
- Repeat at a third seed range to confirm the asymmetry isn't seed-clustered.

Bug fix landed in this run: `r14_ood_gate.py` was passing a relative `--out-dir` to `npm --workspace backend run sim:eval-gate`, which resolves against `backend/` workspace cwd → manifest landed at `backend/runs/...` and the orchestrator failed to read it. Fixed by resolving `out_dir` to absolute at parse time.

#### R14.A.footnote — Side-asymmetry confirmation gate at independent seeds (2026-05-14) — DONE / gate1 FAIL

- **Motivation:** R14.A's caveat noted the player-vs-opponent gap may be partly seed-clustered (F-iter-2 at 0.443 vs A at 0.404 across overlapping seed ranges). One independent seed range at the same MCTS production config (value-head leaf, 100 sims, c_puct=1.5) would either tighten the production claim or contradict it. Cheap diagnostic, ~15 min.
- **Result (2026-05-14):** **gate1 FAIL @ Wilson 0.394; gate2 PASS @ Wilson 0.482; side-asymmetry confirmed real, not seed-clustered.** Re-ran `training/r14_ood_gate.py` against `runs/R13-W6-phase-d/iter-2/checkpoint.pt` at independent seeds 900000+ (gate1, vs rule-bot) / 950000+ (gate2, vs R4 MCTS), n=100/gate, same MCTS config as R14.A.

| Gate | n | WR | Wilson lower | Wilson upper | Verdict |
| --- | --- | --- | --- | --- | --- |
| 1: fresh seeds 900000+, vs rule-bot | 100 | 0.49 | **0.394** | 0.587 | **FAIL** (0.6pp below 0.40 bar) |
| 2: OOD seeds 950000+, iter-2 vs R4 | 100 | 0.58 | **0.482** | 0.672 | PASS by 8pp |

  Per-side breakdown (all four gates × both seed ranges):

| Seed range | Gate | Player WR | Player Wilson lower | Opponent WR | Opponent Wilson lower | Gap |
| --- | --- | --- | --- | --- | --- | --- |
| 800000+ (R14.A) | 1 (vs rule-bot) | 0.44 | 0.312 | 0.56 | 0.423 | +0.111 |
| 850000+ (R14.A) | 2 (vs R4 MCTS) | 0.46 | 0.330 | 0.76 | 0.626 | +0.296 |
| 900000+ (footnote) | 1 (vs rule-bot) | 0.42 | **0.294** | 0.56 | **0.423** | +0.129 |
| 950000+ (footnote) | 2 (vs R4 MCTS) | 0.50 | **0.366** | 0.66 | **0.522** | +0.156 |

  **What this changes about the R14 production claim:** the cheap-inference deployment headline "Wilson 0.452 at value-head leaf + ratio=1.5" (F-iter-2, 2026-05-14, sprint-plan line 121) was **not confirmed at a third independent seed range**. The empirical range across F-iter-2 (0.443 at seeds 820000+), R14.A (0.404 at seeds 800000+), and this footnote (0.394 at seeds 900000+) is **Wilson lower 0.39–0.45**, with one-of-three runs failing the 0.40 bar. Honest framing: report the cheap-inference number as "Wilson lower 0.39–0.45 across three independent seed ranges" rather than as a stable 0.452 point estimate.

  **What this does NOT change:** the R14 max-strength headline — **rollout-leaf MCTS at iter-2 Wilson 0.6479** (I.2) — was NOT tested by this gate and remains the unchanged max-strength deployment claim. The FAIL applies only to the cheap-inference value-head-leaf + ratio=1.5 deployment pick.

  **What this refutes about R14.A:** the original A note "side-asymmetry is partly seed-clustered" (sprint-plan line 113) is contradicted. The player-Wilson < opponent-Wilson gap replicates with the same shape across all four gates and two independent seed ranges; it is a real strategic asymmetry in the iter-2 policy at value-head-leaf inference, not seed-distribution noise.

  **gap_gate1_minus_gate2 = −0.088** (gate2 stronger than gate1). The model is closer to matching its own R4 baseline head-to-head than to beating rule-bot at the production claim level on fresh seeds — suggests rule-bot behavior at seeds 900000+ differs meaningfully from the R14.A / F seed ranges, but not a separate diagnostic this slot.

  Wall-clock 9m 38s total (gate1 275.7s + gate2 302.7s). Output: `runs/R14-A-side-asymmetry-seed900000/{summary.json,gate1-fresh-seeds.manifest.json,gate2-mcts-vs-r4.manifest.json}`.

  **Escalated to harness:** the R14 cheap-inference production claim is materially weakened; the human owns the research-stance decision: (a) re-target to CI phrasing, (b) re-run R14.A at larger n, or (c) accept and route to rollout-leaf MCTS. See `docs/ai-agent-state/escalations.md`.

#### R13 PPO probe on iter-1 — still doesn't move (2026-05-11)

Three iterations of `ppo_orchestrator` from the W6 iter-1 checkpoint (20 games/update, 30-game gate per iter, `--selection policy` for both collection and gate):

| Iter | Wilson lower (policy gate) | mean_advantage |
| --- | --- | --- |
| 0 | 0.242 | -0.150 |
| 1 | 0.301 | +0.049 |
| 2 | 0.256 | -0.139 |

Compare to iter-1 at value-head-leaf MCTS: Wilson 0.452. PPO at raw policy regressed strength. Diagnostics:

- **Importance ratios ~1.0** across all minibatches → policy still close to argmax.
- **Per-minibatch entropy ~0.27 nats** — higher than the original R3 BC (0.18) but the visit-count distillation didn't soften the policy enough for PPO to differentiate trajectories.
- **mean_advantage still flips sign** between iterations — W3's value-head retrain helped MCTS-augmented decisions but doesn't carry to raw-policy GAE returns.

Verdict: iter-1's strength is **coupled to MCTS at decision time**. PPO from the raw policy is still blocked by the same two issues that killed the original F1 attempts (peaked policy + return-vs-value mismatch). To unblock RL: (a) generate trajectories under MCTS instead of raw policy (expensive), or (b) restart from a much softer init (Dirichlet-warmed BC with high temperature). Both are R14+ work.

#### R13 cheap-inference verdict — value-head-leaf at iter-1 clears GO (2026-05-11)

Gate at the W6 iter-1 checkpoint with `--mcts-leaf value-head` (no rollouts), 100 sims, 100 games seeds 700000+:

- **WR 0.55 (55/100), Wilson95 [0.452, 0.644]**
- Player 0.48 (Wilson [0.348, 0.615]); Opponent 0.62 (Wilson [0.482, 0.741])
- Zero fallbacks, 194s wall-clock for 100 games (~2s/game with 4 workers)

Value-head-leaf progression at the same evaluator and 100 sims:

| Config | Wilson lower | Δ vs R4 |
| --- | --- | --- |
| R4 baseline | 0.30 | — |
| W3 retrain alone | 0.347 | +4.7pp |
| W6 iter-1 (W3 + visit-count distill) | **0.452** | **+15.2pp** |

Visit-count distillation on top of W3's variance fix did real work. Iter-1 at cheap inference clears the 0.40 GO bar — **cheap-inference deployment is viable**. Trade-off vs rollout-leaf (Wilson 0.573): -12pp Wilson for ~20× latency reduction (2s/game vs 40s/game). For human-facing UI play, value-head-leaf is the natural production config.

#### R13.W7 headline — n=100 validation (2026-05-11)

Independent gate (seeds 600000+) at the W6 iter-1 checkpoint under the same rollout-leaf MCTS config (100 sims, K=3). Originally launched for n=400, truncated to n=100 since the Wilson half-width was already tight enough that 4× the compute is cosmetic:

- **WR 0.67 (67/100), Wilson95 [0.573, 0.754]**
- Player: 0.551 (27/49), Wilson [0.413, 0.681]
- Opponent: 0.784 (40/51), Wilson [0.654, 0.875]

Side asymmetry returned at this seed range (opponent +23pp over player) — Wilson CIs do overlap so likely seed-distribution noise rather than a regression, but worth a quick repro at a different seed-start before claiming the R-WILD gap is closed unconditionally. Bottom line: the production headline is **Wilson lower ≥ 0.57 vs rule-bot, n≥100, rollout-leaf MCTS 100 sims**, comfortably past the R12 baseline.

#### R13.W3 result — value head retrain is **PARTIAL** (2026-05-11)

50 rollout-leaf selfplay games (100 sims, K=3, R4 prior) → 25-epoch frozen-trunk MSE retrain → 100-game gate at `--mcts-leaf value-head` over the retrained checkpoint.

| Metric | Retrained head (W3) | R4 baseline (R12 Phase A) | Δ |
| --- | --- | --- | --- |
| WR vs rule-bot | 0.44 | 0.345 | +9.5pp |
| Wilson lower (n=100) | **0.347** | 0.30 | +4.7pp |
| Wilson upper | 0.538 | 0.40 | +14pp |
| Player side WR | 0.40 | — | — |
| Opponent side WR | 0.48 | — | — |

The retrained head is materially better than R4's at the same leaf evaluator, but doesn't clear the 0.40 GO threshold by itself. Verdict: **PARTIAL**. Per the sprint plan, full Phase D (W6) is still worth running with this checkpoint as warm-start — the rollout-mean target reduces value-head variance, and visit-count distillation could compound on top of that. Rollout-leaf inference (Wilson lower 0.556 at n=200, R12) remains the strongest single config we have; W3 narrowed but did not close the gap between value-head leaf and rollout-CRN leaf.

#### Tasks deleted as obsolete (2026-05-11 post-R12)

R7 (multi-teacher BC blend), R8 (DPO), R9 (Q-learning head), R10 (full-scale DAgger) were all queued only as fallbacks IF R12 failed. R12 didn't fail. Tasks #29-32 removed from the active backlog. The hypotheses they tested (label-quality fixes for the imitation cap) are also obsolete: R12 proved the cap is downstream of the *value-head leaf noise*, not the *training labels*.

R2 (multi-temperature gate matrix) is repurposed as a deployment-tuning task, not a research diagnostic.

R-WILD's side-imbalance portion is largely resolved by R12 (gap inverted: player 58%, opponent 67%). Simulator determinism + rule-bot mistake catalog remain as low-urgency follow-ups.


### R15.S1–S4 result blocks (2026-05-14)

These four blocks were the backlog's per-experiment write-ups for the F1
post-mortem follow-on phases. The detailed per-phase progress-doc sections
(Phase J / K / L / M / N / O / O' / P) above are the canonical evidence; the
backlog framing is preserved here verbatim because it gives the branch-level
narrative (motivation, exit/gate, branch synthesis) that the per-phase blocks
do not always restate.

#### R15.S1 — Better SL warm-start (F1 next-move #1) — DONE / FAIL

- **Motivation:** F1 PPO post-mortem ranked "better warm-start" first. The DAgger sweep at this
  codebase config plateaued at WR 37.5% with low-entropy. Larger SL run (more games, more epochs,
  possibly explicit entropy regularization during BC) might give PPO an actually-movable starting
  point. Distinct from R3's β=0.05 entropy-bonus probe, which already showed entropy alone is not
  the issue — this is the *SL-scaling* angle.
- **Next action:** Scope a single full-scale DAgger run: ~3× the games (≥200 trace games per iter),
  ≥50 epochs, hidden_dim=64/depth=2 unchanged; capture warm-start Wilson lower at greedy + a stretch
  PPO sweep from that checkpoint. Pre-register: SL warm-start Wilson lower ≥ 0.45 before any PPO
  is run; otherwise PPO has the same gradient-signal problem as F1 phases 2/G/H.
- **Cost:** ~1–2 h compute (item-17 take-2 was ~30 games × 25 epochs in <10 min; 3× scale ≤ 1.5h).
- **Exit / gate:** warm-start Wilson lower ≥ 0.45 *or* document the new SL ceiling and close the
  branch.
- **Result (2026-05-14):** **FAIL — pre-registered falsification confirmed.** Run
  `runs/R15-S1-warmstart-sweep/` — 3 DAgger iters × 90 trace games × 75 epochs at fixed
  hidden=64/depth=2, rollout-CRN×3 teacher, KL anchor 0.0/0.1/0.5, n=500 side-balanced gate per
  iter. Wilson lower per iter: iter-0 **0.2845** (WR 32.4%) → iter-1 **0.2883** (WR 32.8%) →
  iter-2 **0.3269** (WR 36.8%). Iter-2 Wilson 0.3269 lands inside the pre-registered falsification
  band 0.311 ± 2pp (= [0.291, 0.331]) — the +1.6pp lift over item17 take-2's 0.311 is within
  Wilson half-width at n=500 and an order of magnitude below the predicted +14pp. The
  pre-registered **secondary check fired**: per-epoch val_accuracy reached 99% of peak by epoch
  2-3 in every iter (best: iter-0 0.7676 @ ep7, iter-1 0.7574 @ ep2, iter-2 0.7656 @ ep3), then
  *declined* over the remaining 70+ epochs while train_acc climbed to 0.91-0.97. Classic
  plateau-then-overfit at this size — the predicted falsification mechanism fired in train-time
  diagnostics first, then validated at eval time. Wall-clock 11m 51s, ~8× faster than the
  ~1.5h scoping estimate (the rollout-CRN×3 teacher dominated; n=500 eval was not the
  bottleneck). Full writeup + per-iter trajectory + comparison table in
  `docs/ai-performance-research-progress.md` § "Phase K — F1 DAgger compute-scaled warm-start".
  Combined with R15.S2 (closed FAIL this morning), the two highest-ranked F1 post-mortem next
  moves have both falsified. The F1 cap is **not** compute at fixed capacity (this run), **not**
  PPO HPs (phases 2/G/H), **not** weak-pool self-play (R5), and **not** strong-pool self-play
  with v1 plumbing (phase J). Surviving F1 candidates: R15.S3 (richer reward shaping), R15.S4
  (sampling-temperature gate, diagnostic), and the deeper SL-label-quality branches the
  falsification opens up (R7 multi-teacher labels, R8 DPO).

#### R15.S2 — PFSP self-play PPO (F1 next-move #2) — DONE / FAIL

- **Motivation:** F1 PPO post-mortem ranked self-play second. Item-12 opponent pool already exists;
  `ppo_orchestrator` currently uses `--opponent-model-url` unset (rule-bot default). Rollouts against
  PFSP-sampled prior promoted checkpoints would change the reward distribution from
  single-opponent-shape to diversity-shape. R5 (a prior tier-2 attempt against the item17 pool)
  did NOT break the cap, but the W6/I.2 pool is a much stronger opponent set; worth one cheap
  re-attempt with the new pool.
- **Next action:** Confirm `--rollout-vs-pool` plumbing is intact in `ppo_orchestrator.py` (or add it
  if missing). Run one PPO sweep with PFSP-sampled rollout opponents drawn from
  `runs/R13-W6-phase-d/iter-{0,1,2}/checkpoint.pt`, aggressive HPs, 800 games/update, 3 iters.
- **Cost:** ~10–15 min compute + any plumbing patches.
- **Exit / gate:** Wilson lower ≥ 0.40 on the rule-bot eval gate. Otherwise close the branch.
- **Result (2026-05-14):** **FAIL.** Run `runs/R14-f1-self-play-sweep/` — 3 iters × 800 games at
  aggressive HPs from W6/iter-2 warm-start against the W6/iter-{0,1,2} pool. Wilson lower per iter:
  warm-start eval **0.1455** (WR 30%, n=20) → iter-1 **0.2993** (WR 50%, n=20, opponent W6/iter-1)
  → iter-2 **0.2188** (WR 40%, n=20, opponent the *just-promoted iter-1 from this run*). Iter-1's
  +14.5pp lift is the largest single PPO step recorded in any F1 phase — confirming the stronger
  pool does break the "ratios ≈ 1.0" stasis that hobbled phases 2/G/H — but iter-2 regressed when
  the v1 one-opponent-per-run sampler rolled a self-promotion, creating co-adaptation. Final
  promoted Wilson **0.2188** missed the 0.40 gate by 18pp; iter-1 best missed by 10pp. Both
  numbers are *worse* than R5's iter-2 (0.3109) and phase H's iter-2 (0.3109) despite materially
  different mechanism (the policy actually moved). Full writeup +
  per-iter mean_return / entropy / KL trace in
  `docs/ai-performance-research-progress.md` § "Phase J — F1 PPO + strong-pool self-play".
  Closes the strong-pool branch of the self-play hypothesis. R15.S1 (better warm-start) and
  R15.S3 (richer reward shaping) remain the only unfalsified F1 next moves.

#### R15.S3 — Richer reward shaping (F1 next-move #3) — DONE / EXHAUSTED (both axes closed)

**Final branch closeout (2026-05-14, Phase P capstone, both axes exhausted).** The R15.S3
reward-shaping branch is now **fully explored and closed across both signal axes**. The
prior `87e9e77` "BRANCH CLOSED" framing was premature — it covered only axis 1
(observation-delta signals); Phases O and P then tested axis 2 (per-step value-head-delta
tempo signal) and both regressed. Six sweeps total covered all single-axis moves available:

- **Axis 1 — hand-engineered observation-delta signals.** 4 phases (L decay 1.0×, M decay
  1.75×, N constant 1.0×, O' reduced-mix constant 1.0×). Capped at iter-2 Wilson **0.368
  ± 0.001**. Coef magnitude (L vs M, Δ +0.002), schedule (L vs N, Δ +0.012), signal mix
  (N vs O', Δ +0.000) — all single-axis moves saturated.
- **Axis 2 — per-step value-head-delta tempo signal.** 2 phases (O coef 0.05, P coef
  0.01). Phase O iter-2 **0.3502** (-1.8pp vs Phase N); Phase P iter-2 **0.3463**
  (-2.1pp vs Phase N, AND -0.4pp vs Phase O). **Both magnitudes regressed.** Lowering
  the coef did not help — the value-head-delta signal mechanism is **wrong-shape, not
  wrong-magnitude**.

All six phases have PPO healthy in all non-reward dimensions (importance ratios off 1.0,
KL bounded, entropy stable, `numerical_anomalies = 0`, WR tracks Wilson). Total R15.S3
branch compute cost **~36 min wall-clock** across 6 phases (L 5m54s + M 6m22s + N 5m25s
+ O' 5m28s + O 5m22s + P 5m23s) — cheap research, decisive answer on both axes. The F1
post-mortem framing is now: **"The F1 reward-shape mechanism cannot break 0.368 from this
warm-start. Per-step shaping from any observation-derived signal saturates at 0.368, and
per-step shaping from the policy's own value-head delta actively regresses. The remaining
F1 moves must change either the warm-start (R7 multi-teacher labels rebuild) or the
optimization objective (R8 DPO replacement). The reward-shape branch is closed."**
Surviving F1-line candidates (human-rank, not autonomous-launch): (b) **R7 multi-teacher
labels** — retrain DAgger SL warm-start with multiple expert teachers; doesn't touch
reward, changes SL pipeline. (c) **R8 DPO** — replace PPO with a different objective
that doesn't depend on hand-shaped per-step reward signal; bigger pivot. Path (a)
value-head tempo signal has been executed and exhausted at Phases O + P. Escalation
re-opened at `docs/ai-agent-state/escalations.md` `## Open`; queue item
`r15-s3-branch-synthesis-and-next-pick` P2 ready autonomous-launch ineligible. Full
synthesis at `docs/ai-agent-state/notes.md` `## F1 reward shaping — scoping (2026-05-14)`
Phase P Closeout + final R15.S3 branch summary block; per-phase writeups at
`docs/ai-performance-research-progress.md` §§ Phase L / M / N / O' / O / "Phase P — F1
PPO + value-head tempo signal, coef 0.01 (R15.S3 GENUINE BRANCH CLOSEOUT)".

- **Motivation:** F1 PPO post-mortem ranked richer reward shaping third. Current reward is
  Δpoints × 1/3 + terminal ±1. Strategic depth around attachment / retreat / energy cycles is not
  rewarded per-step. PPO might exploit a denser signal even if the existing gradient mechanism
  stays argmax-ratio-bound.
- **Next action:** Catalog 3–5 candidate intermediate rewards from the existing turn-goal /
  candidate-ranker telemetry (e.g. successful attach, retreat survival, KO threat resolution).
  Pre-register one shaping schedule (decay-to-terminal weight), implement in `ppo_orchestrator.py`'s
  reward computation hook, smoke at f1 phase H scale.
- **Cost:** ~3–4 h code + ~20 min compute.
- **Exit / gate:** Wilson lower ≥ 0.40 on the rule-bot eval gate. Diagnostic regardless: if it
  doesn't move WR but does change `mean_advantage` distribution, that itself is publishable.
- **Result (2026-05-14):** **PARTIAL — best F1 rule-bot result on record, neither success nor
  falsification.** Run `runs/R15-S3-reward-shaping-sweep/` — 3 iters × 800 games at phase-H
  aggressive HPs from `runs/item17-2026-05-11/iter-002/checkpoint.pt`, opponent rule-bot, five
  per-step reward signals (Δactive-energy 0.02, Δbench-energy 0.02, retreat indicator 0.03,
  Δthroughput 0.02, Δactive-hp-relative 0.05) with linear decay full→zero across iter-0..2.
  Wilson lower per iter: iter-0 **0.2730** (WR 31.2%) → iter-1 **0.2787** (WR 31.8%) → iter-2
  **0.3560** (WR 39.8%); n=500 side-balanced per iter, all three promoted. Iter-2 missed the 0.40
  success bar by **4.4pp** and landed **+2.5pp above** the pre-registered falsification band
  [0.291, 0.331]. **+4.5pp absolute over the prior F1 ceiling** (phase H iter-2 0.3109; R15.S1
  iter-2 0.3269; R15.S2 iter-2 0.2188). **Pre-registered branch-1 diagnostic fired**: importance
  ratios moved decisively off ~1.00 in every iter (ratio_max 19.06 / 32.59 / 11.12 vs phase H
  ~1.00), confirming the shaped reward unlocked non-zero PPO gradient from the same warm-start;
  the gap to 0.40 is now quantitative (coefficient magnitudes / signal mix) rather than
  mechanistic. Entropy stable across iters (0.168 → 0.154 → 0.153, no collapse);
  `numerical_anomalies = 0` across all 48 minibatches; approx_kl_max < 0.02. Wall-clock **5m 53.7s**
  end-to-end. Implementation cost: **+103 LOC orchestrator-only diff** to `training/ppo_orchestrator.py`
  (5 new `--reward-*-coef` args + `--reward-shape-start/-end` linear decay + `shape_attribution`
  event), zero sim-side, `TMPDIR=/tmp npm run test:ppo-smoke` PASS pre-launch. **Branch is alive,
  not closed.** Full writeup + per-iter trajectory + comparison table in
  `docs/ai-performance-research-progress.md` § "Phase L — F1 PPO + reward shaping". The phase-H
  "F1 target 0.40 NOT REACHABLE" framing has been qualified (not deleted) — it was correct under
  the unshaped reward mechanism but R15.S3 demonstrably changed that mechanism. Next move queued
  as `r15-s3-followup-tune-shaping`: hold the same 5 signals, scale all five coefs 1.5–2× (current
  ~0.13/game shape sum is at the low end of the scoping ±0.3 target), rerun the 3-iter sweep.
  Expected ~10-15 min compute. If iter-2 crosses 0.40 → first F1 success on record; if iter-2
  stalls at ~0.36 → coef scaling is saturated and the next move is signal-mix change.
- **Follow-up Result (R15.S3 1.75× coef-scaling, 2026-05-14):** **DONE / SATURATED.** Run
  `runs/R15-S3-followup-tune/` — same 5 signals, all coefs scaled 1.75× (midpoint of queued
  1.5–2× range), same warm-start / opponent / HPs / code as Phase L. Wilson lower per iter:
  iter-0 **0.2960** (WR 33.6%, +2.3pp vs Phase L iter-0 0.2730, promoted) → iter-1 **0.2825**
  (WR 32.2%, **rejected** — first F1 sweep-internal regression on record, `wilson_lower 0.2825
  < floor 0.2960`) → iter-2 **0.3580** (WR 39.8%, rolled forward from iter-0 parent after iter-1
  reject, promoted). Iter-2 0.3580 lands within Wilson noise of Phase L iter-2 0.3560
  (**Δ +0.002**); the iter-0 lift did not compound. Pre-registered "coef saturation" outcome
  fired cleanly — the (warm-start, opponent, 5-signal set, linear-decay schedule) tuple has a
  true ceiling at ~0.358. Importance ratios still moved decisively off ~1.00 (ratio_max
  15.86 / 12.50 / 18.82 across iters — gradient still active, comparable to Phase L's
  19.06 / 32.59 / 11.12); entropy stable (0.171 → 0.164 → 0.164); `numerical_anomalies = 0`;
  approx_kl_max < 0.02. Wall-clock **6m 22.3s**. Next single-axis move is signal-mix or
  shaping-schedule change, not further coef scaling. Three candidates ranked in
  `docs/ai-agent-state/notes.md` follow-up block: (1) constant-shaping schedule
  (`--reward-shape-end 1.0`, smallest single-axis change, recommended v1), (2) drop
  low-attribution signals + scale survivors, (3) add value-head-derived strategic-tempo signal.
  Queued as P3 `r15-s3-signal-mix-or-schedule`. Full writeup: `docs/ai-performance-research-progress.md`
  § "Phase M — F1 PPO + reward-shape coef-scaling follow-up".
- **Follow-up Result (R15.S3 constant-shape schedule axis, 2026-05-14):** **DONE / PARTIAL —
  new F1 ceiling on record.** Run `runs/R15-S3-constant-shape/` — same five signals at Phase L
  1.0× coefs, same warm-start / opponent / HPs / code; only diff vs Phase L is
  `--reward-shape-end 0.0` → `1.0` (constant full-strength shaping across all iters, no linear
  decay). Wilson lower per iter: iter-0 **0.2787** (WR 31.8%, +0.6pp vs Phase L iter-0 0.2730,
  promoted) → iter-1 **0.2883** (WR 32.8%, +1.0pp vs iter-0, **promoted — no rejection**, in
  contrast to Phase M's iter-1 reject under linear-decay shape at 0.5×) → iter-2 **0.3677**
  (WR 41.0%, +7.9pp vs iter-1, promoted). All three iters promoted, monotone trajectory,
  `run_completed clean, halted=false`. **Iter-2 0.3677 is +1.2pp over Phase L 0.3560 and
  +1.0pp over Phase M 0.3580 — the new F1 rule-bot ceiling on record across every sweep.**
  Still 3.2pp short of the 0.40 success bar. The schedule-axis lift is real but small
  (~1.5σ Wilson noise at n=500); crossed with Phase M's Δ +0.002 from coef scaling, the joint
  message is that both single-axis follow-ups on the existing 5-signal mix moved iter-2 by
  ≤+1pp. **The binding constraint is the signal set itself, not magnitude or schedule.**
  Mechanism check (healthy): ratio_max 11.82 / 7.97 / 32.78 across iters — gradient still
  decisively off ~1.00; entropy stable (0.173 → 0.160 → 0.155, no collapse);
  `numerical_anomalies = 0` across all 48 minibatches; approx_kl_max < 0.015. **Reward-hacking
  check (passes):** WR tracks Wilson in lockstep (31.8% → 32.8% → 41.0%); no iter where Wilson
  rises while WR falls; the +7.9pp Wilson lift at iter-2 is mirrored by +8.2pp WR. Wall-clock
  **5m 25.4s** (fastest of L / M / N); zero LOC diff vs Phase L (one CLI flag flip). Next
  single-axis move is **signal-set change**, not further schedule or magnitude tuning. The
  recommended candidate is option (3) from the prior scoping — add a value-head-derived
  strategic-tempo signal (per-step delta in own-win-probability from the trained value head's
  output) as a 6th additive signal at coef ~0.05; keep the existing 5 signals; same constant
  schedule established by Phase N; same Phase H HPs; same warm-start. Implementation cost
  ~20-40 LOC additive to `training/ppo_orchestrator.py:parse_trace_to_trajectories` (value head
  output is already in the trajectory inference stream). Queued as P3
  `r15-s3-value-head-tempo-signal`. Full writeup:
  `docs/ai-performance-research-progress.md` § "Phase N — F1 PPO + constant reward shaping".
- **Follow-up Result (R15.S3 reduced signal-mix axis — Phase O' BRANCH CAPSTONE, 2026-05-14):**
  **DONE / SATURATED — iter-2 Wilson 0.3677 identical to Phase N's 0.3677 to 4 decimal places;
  R15.S3 BRANCH CLOSED.** Phase O original plan (add value-head tempo signal as 6th additive
  component) audited at launch time and found blocked on TS-side ONNX/trace instrumentation
  (rollout emits placeholder `value_pred=0.0` at `ppo_orchestrator.py:858`; value head is not
  in the decision-trace schema). Pivoted to the scoping doc's option (2) — drop the weakest
  of the existing 5 signals, scale the strongest. Audit: Phase N iter-2 absolute coef-weighted
  contributions were throughput 171.4 / active-energy 90.4 / hp-diff **-6.9** (anti-
  correlated) / bench-energy 4.2 / retreat **0.0** (agent never retreats). Run
  `runs/R15-S3-reduced-mix/` dropped retreat (zero) and bench-energy (smallest non-zero,
  redundant with active-energy); scaled active-energy 0.02→0.03 and throughput 0.02→0.03;
  held hp-diff 0.05 as control (preserves signal-set parity test integrity rather than
  amplifying an anti-correlated signal). Per-game shape budget ~0.24, close to Phase M's
  0.23. Same warm-start (`runs/item17-2026-05-11/iter-002/model/checkpoint.pt`), opponent
  (rule-bot, no pool), HPs (`--lr 3e-4 --clip-epsilon 0.3 --entropy-coef 0.01 --ppo-epochs 4
  --reward-shape-start 1.0 --reward-shape-end 1.0`), code as Phase L/M/N. Wilson lower per
  iter: iter-0 **0.2825** (WR 32.2%, +1.0pp vs Phase N iter-0 0.2787, promoted) → iter-1
  **0.2883** (WR 32.8%, +0.6pp vs iter-0, promoted — identical to Phase N iter-1 0.2883 to
  4dp) → iter-2 **0.3677** (WR 41.0%, +7.9pp vs iter-1, promoted — **identical to Phase N
  iter-2 0.3677 to 4 decimal places**). All 3 iters promoted, monotone trajectory, no
  rejections, `run_completed clean, halted=false`. **Δ iter-2 vs Phase N: +0.0000pp.** The
  signal-mix change (dropping 2 of 5 signals, scaling the dominant 2) did not move iter-2
  at all. Mechanism healthy: ratio_max 10.53 / 18.02 / 10.57 (gradient strongly active);
  entropy stable 0.171 → 0.161 → 0.153 (no collapse); `numerical_anomalies = 0` across all
  48 minibatches; approx_kl_max < 0.016 each iter; WR tracks Wilson in lockstep (no
  reward-hacking signature). Phase O' attribution confirms the drop choices: bench-energy
  0.0 / retreat 0.0 across all 3 iters (signals correctly silenced); throughput dominant
  (~259), active-energy secondary (~130), hp-diff consistently anti-correlated (~-6.5).
  Wall-clock **5m 28.2s** (fastest of L/M/N/O' alongside Phase N); zero LOC diff vs Phase N
  (CLI args only). **4-axis synthesis (capstone) — the binding constraint is the
  information content of the signal set, not weights / schedule / mix.** Three single-axis
  moves now tested: coef magnitude L→M Δ +0.002, schedule L→N Δ +0.012, signal mix N→O' Δ
  +0.000. All four phases have PPO healthy. The 5-signal family encodes ~+5pp Wilson over
  the unshaped Phase H baseline (0.3109 → 0.3677), but **scaling magnitude, changing
  schedule, or dropping inactive components all leave iter-2 at 0.368 ± 0.001**. R15.S3
  observation-delta branch CLOSED / SATURATED. Surviving F1-line candidates: (a) value-head
  tempo signal (blocked on `r15-s3-value-head-trace-instrumentation`); (b) R7 multi-teacher
  labels; (c) R8 DPO. Human research-stance decision filed at
  `docs/ai-agent-state/escalations.md`. Full writeup: `docs/ai-performance-research-progress.md`
  § "Phase O' — F1 PPO + reduced signal-mix (R15.S3 branch closeout)".
- **Follow-up Result (R15.S3 value-head tempo signal, coef 0.05 — Phase O, 2026-05-14):**
  **DONE / REGRESSED — iter-2 Wilson 0.3502, 1.8pp worse than Phase N's 0.3677.** TS-side
  value-head trace instrumentation (queued P3 `r15-s3-value-head-trace-instrumentation`)
  landed first (7 LOC TS + ~30 LOC orchestrator), unblocking the original Phase O plan that
  was deferred at Phase O' time. Run `runs/R15-S3-value-head-tempo/` added a 6th additive
  signal — per-step delta of the policy's own value-head output — at
  `--reward-value-head-coef 0.05`, holding all other Phase L/N parameters fixed (5 obs-delta
  signals at 1.0× coefs, constant schedule, same warm-start, same opponent, same HPs). Per-iter
  Wilson lower: iter-0 **0.2845** (WR 32.4%, promoted) → iter-1 **0.2768** (WR 31.6%,
  **REJECTED** — `wilson_lower 0.2768 < floor 0.2845`, same regression pattern as Phase M
  iter-1) → iter-2 **0.3502** (WR 39.2%, promoted, rolled forward from iter-0 parent).
  `promoted_iterations: [0, 2]`. **Δ iter-2 vs Phase N: -0.018**. Adding the value-head signal
  at coef 0.05 made iter-2 *worse*, not better. Wall-clock **5m 22.0s**. **Mechanism diagnosis:
  signal is correctly wired but magnitude is ~20× the design budget.** Value head Tanh output
  has range [−1, +1] so per-step delta range is [−2, +2]; at coef 0.05 the per-step
  contribution is ±0.1; at ~60 steps/game the per-game contribution is ±6.0, **20× the
  ±0.3/game budget the R15.S3 scoping doc set**. iter-0
  `shape_attribution.value_head = 26.01` confirms empirically: third-largest absolute
  attribution behind throughput's 171.4 (accumulates positively across all steps so its
  magnitude is mostly positive bias) and active-energy's 85.1; for a mean-zero (Tanh delta)
  signal, an absolute attribution of 26 represents genuine per-step *variance* dominating
  every observation-delta signal except the two energy-related ones. iter-1 / iter-2
  attributions drop to 9.28 / 8.25 as iter-0's promoted policy learns to flatten the
  value-head delta — the classic over-shaping signature. Iter-1 rejection mirrors Phase M's
  1.75×-coef regression: over-shaped iter-0 promotes a policy that exploits the shape, iter-1
  overfits further, greedy WR falls below the tolerance=0 floor. PPO healthy in all
  non-reward dimensions: ratio_max 12.7 / 12.3 / 22.4 (gradient active); approx_kl_mean
  0.014 / 0.011 / 0.012; entropy 0.167 → 0.159 → 0.158 (stable); `numerical_anomalies = 0`
  across 48 minibatches. Next move: Phase P launched as `runs/R15-S3-value-head-tempo-low/`
  with `--reward-value-head-coef 0.01` (5× smaller; per-game shape contribution ±1.2, closer
  to the ±0.3 budget). Exit gate for Phase P: success ≥0.40 (first F1 success), partial
  improvement 0.368 < iter-2 < 0.40 (signal contributes additively), neutral ≈0.36-0.37
  (signal redundant with obs-delta family at low magnitude), regression <0.35 (signal
  disrupts even at low magnitude — value-head info quality is the problem, not magnitude;
  pivot to R7/R8). Full writeup: `docs/ai-performance-research-progress.md` § "Phase O — F1
  PPO + value-head tempo signal, coef 0.05".
- **Follow-up Result (R15.S3 value-head tempo signal, coef 0.01 — Phase P, R15.S3 GENUINE
  BRANCH CLOSEOUT, 2026-05-14):** **DONE / REGRESSED — iter-2 Wilson 0.3463, 2.1pp worse
  than Phase N's 0.3677 AND 0.4pp worse than Phase O's 0.3502; lowering the coef did not
  help — the value-head-delta signal mechanism is wrong-shape, not wrong-magnitude.** Run
  `runs/R15-S3-value-head-tempo-low/` — single change vs Phase O:
  `--reward-value-head-coef 0.05` → `0.01` (5× smaller). All else identical (same warm-
  start `runs/item17-2026-05-11/iter-002/model/checkpoint.pt`, same rule-bot opponent,
  same Phase H HPs, same 5 obs-delta coefs at Phase L 1.0×, same constant schedule). Per-
  iter Wilson lower: iter-0 **0.2730** (WR 31.2%, promoted — **identical to Phase L
  iter-0 0.2730 to 4dp**; at coef 0.01 the value-head signal contributes effectively
  nothing at iter-0) → iter-1 **0.2845** (WR 32.4%, promoted, +1.1pp; **no rejection**,
  unlike Phase O which rejected iter-1 at 0.2768 < 0.2845 floor) → iter-2 **0.3463**
  (WR 38.8%, promoted, +6.2pp from iter-1). `promoted_iterations: [0, 1, 2]`,
  `consecutive_failures: 0`, `halted: false`, `promoted_wilson_lower 0.34629528411824795`.
  Wall-clock **5m 23.2s** (run_started ts 1778735914.37 → run_completed 1778736237.61).
  **Δ iter-2 vs Phase N: -0.021; Δ iter-2 vs Phase O: -0.004.** Both value-head magnitudes
  (0.05, 0.01) regressed vs Phase N's no-value-head baseline (0.3677); the 5× lower coef
  did not help and slightly hurt. **The value-head-delta signal mechanism is wrong-shape,
  not wrong-magnitude** — at coef 0.05 the signal is loud and over-shaped the policy
  (Phase O iter-1 rejection signature); at coef 0.01 the signal is quiet but noisy and
  contributes random variance without informational gain (Phase P clean trajectory but
  lower ceiling). Dimensional check on the coef ratio: iter-0
  `shape_attribution.value_head = 5.03` (Phase P) vs `26.01` (Phase O); ratio **5.17×**
  matches the 5× coef ratio within rounding — signal is wired and scaled correctly.
  iter-1 / iter-2 attributions 1.45 / 1.27 (Phase P) vs 9.28 / 8.25 (Phase O); same
  policy-flattens-the-delta pattern at both magnitudes but the absolute level is now
  small enough that flattening contributes neither helpful gradient nor harm. PPO healthy
  in non-reward dimensions: ratio_max 12.53 / 13.29 / 21.63 (gradient active);
  approx_kl_mean 0.013 / 0.012 / 0.009; entropy 0.173 → 0.164 → 0.164 (stable);
  `numerical_anomalies = 0` across all 48 minibatches. **R15.S3 BRANCH CLOSED across both
  axes:** obs-delta saturated at 0.368, value-head regressed at 0.346-0.350. Total
  branch cost ~36 min compute across 6 phases. Surviving F1-line candidates: R7 (multi-
  teacher labels, SL warm-start rebuild) and R8 (DPO, PPO replacement). Human-rank
  decision filed at `docs/ai-agent-state/escalations.md` `## Open` (re-opened). Full
  writeup: `docs/ai-performance-research-progress.md` § "Phase P — F1 PPO + value-head
  tempo signal, coef 0.01 (R15.S3 GENUINE BRANCH CLOSEOUT)".
