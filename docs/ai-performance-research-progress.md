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
`docs/ai-training-findings.md` (corrected trained policies 34-40%).
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

`docs/f1-design.md` fixes F1 defaults: reward shaping (per-step Δpoints
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
fixed in `docs/f1-design.md`. Implementation
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

**F1 target 0.40 not yet reached from the item17-2026-05-11 warm-start** under the PPO mechanism active at the time of this post-mortem. PPO can match the SL cap (phase H iter-2) but cannot exceed it under the existing reward shape. [Update 2026-05-14: R15.S3 (Phase L) lifted the rule-bot Wilson lower from 0.3109 (phase H) to **0.3560** by adding five per-step shaped signals — +4.5pp absolute over the prior F1 ceiling, missing the 0.40 bar by only 4.4pp and landing above the falsification band. The "NOT REACHABLE" framing was correct under the *unshaped* reward mechanism but is qualified once the reward axis is allowed to move; the branch is alive and the next attempt is a coefficient-scaling follow-up on the same axis. See "Phase L — F1 PPO + reward shaping" below. Phase M follow-up (1.75× coefs) landed iter-2 at **0.3580**, within Wilson noise of Phase L (Δ +0.002) — the pre-registered "coef saturation" outcome fired; next single-axis move is signal-mix or shaping-schedule change. See "Phase M — F1 PPO + reward-shape coef-scaling follow-up" below. Phase N follow-up (constant schedule, 1.0× coefs) landed iter-2 at **0.3677** (WR 41.0%, n=500) — **+1.2pp over Phase L, the new F1 rule-bot ceiling on record**, but still 3.2pp short of 0.40. With three configurations of the same 5-signal mix now landing 0.356 / 0.358 / 0.368 at iter-2, coef magnitude and schedule axes both moved iter-2 by ≤+1pp — **the binding constraint is the signal set itself**, not magnitude or schedule. See "Phase N — F1 PPO + constant reward shaping" below. **Phase O' capstone (2026-05-14, R15.S3 BRANCH CLOSED):** the signal-mix axis was tested by dropping the 2 weakest of the 5 signals (bench-energy and retreat — both effectively dead in Phase N attribution) and scaling the 2 dominant signals (active-energy 0.02→0.03, throughput 0.02→0.03). Iter-2 Wilson **0.3677 — identical to Phase N's 0.3677 to 4 decimal places** (Δ +0.000pp). With **three single-axis moves** (coef magnitude L→M Δ +0.002, schedule L→N Δ +0.012, signal mix N→O' Δ +0.000) now exhausted inside the 5-signal observation-delta family, the R15.S3 reward-shape **branch is closed at iter-2 Wilson 0.368 ± 0.001**. The post-mortem framing has been correspondingly upgraded: the reward-shape axis has been **fully explored and converged**; the remaining gap to 0.40 is now **categorical — needs a different information source, not more tuning of the existing observation-delta signals**. Surviving F1-line candidates require pulling information from a *different source*: value-head tempo signal (blocked on TS-side ONNX/trace instrumentation, queued as `r15-s3-value-head-trace-instrumentation`), R7 (multi-teacher labels — SL warm-start rebuild), or R8 (DPO — PPO replacement). See "Phase O' — F1 PPO + reduced signal-mix (R15.S3 branch closeout)" below.]

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
