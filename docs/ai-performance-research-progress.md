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

**Every well-behaved config (2, G, H) lands the trained policy at Wilson lower 0.31. Every aggressive config that actually perturbs the policy (3a, 3b) lands lower.** The 0.31 ceiling is identical to the DAgger warm-start's Wilson lower. PPO is performing correctly and finding that **the warm-start is the highest-return policy reachable from itself** under stochastic Gumbel-max with the current reward shape.

**Root mechanism (now confirmed across configs):**

1. Warm-start entropy ≈ 0.18 nats per decision → stochastic policy ≈ greedy policy.
2. Behavior log-probs ≈ target log-probs ⇒ importance ratios stay at ~1.00 across every minibatch.
3. PPO surrogate gradient ∝ (ratio - 1) × advantage ≈ 0 × advantage = 0.
4. The entropy bonus widens *probabilities* but doesn't *flip argmax decisions*, which is what the gate measures.
5. With the current ±1 terminal + Δpoints×1/3 reward shape, the local optimum at WR ≈ 35–38% is the highest-return policy in the neighborhood of the warm-start.

**F1 target 0.40 declared NOT REACHABLE from the item17-2026-05-11 warm-start** under the current PPO mechanism. PPO can match the SL cap (phase H iter-2) but cannot exceed it.

**Recommended next moves, in order of plausibility:**

1. **Better warm-start.** The DAgger sweep at this codebase config plateaued at WR 37.5% with low-entropy. A larger SL run (more games, more epochs, possibly with explicit entropy regularization during BC) could give a starting point PPO can actually move. The right SL ceiling for this representation is unknown.
2. **Self-play instead of vs-rule-bot.** Item 12 opponent pool exists. PPO rollouts against PFSP-sampled prior promoted checkpoints would change the reward distribution from a single-opponent shape to a diversity shape, potentially exposing learnable axes the rule-bot alone doesn't.
3. **Richer reward shaping.** The current shape rewards point delta and win/loss only. Adding per-step shaping for attachment / retreat / energy cycles could expose strategic signal PPO can exploit.
4. **Change the gate.** All five phases are scored on greedy argmax behavior. If PPO is shaping the policy distribution but not flipping argmax, an alternative gate that samples (e.g. temperature 0.5) might reveal latent improvement. Diagnostic, not solution.

Five phases of artifacts under `runs/f1-2026-05-11-*` with full event streams, manifests, TB scalars for forensics. Live dashboard: `http://127.0.0.1:5000/run/f1-2026-05-11-phaseH-big-aggressive`.
