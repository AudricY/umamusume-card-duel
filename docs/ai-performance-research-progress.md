# AI Performance Research Progress

## 2026-05-07 Backlog Pass

Scope: start at the top of `ai-performance-research-backlog.md`, define a concrete acceptance signal for each item, implement unblocked measurement/data-contract work, and stop only on hard blockers.

### Completed / Advanced

| Item | Result | Evidence |
| --- | --- | --- |
| 16. Deterministic progress fingerprint | Added one shared simulator fingerprint helper and replaced duplicated evaluator/headless hashes. Fingerprint includes full simulator state, including same-count hand/deck/discard content swaps, because it is an internal mutation detector rather than a public model observation. | `backend/src/sim/stateFingerprint.ts`; `backend/src/tests/stateFingerprintSmoke.ts`; `npm --workspace backend run test:train` |
| 1. Full-turn / turn-bundle planner | Added initial `--selection planner` mode. It enumerates bounded same-turn action bundles through modeled phases, scores leaves with rollout, and returns the first action of the best bundle. Tiny smoke is functional with zero fallbacks but not strong. | `backend/src/sim/evaluateModelVsHeuristic.ts`; 2-game planner gate: 0/2, zero fallbacks |
| 17. Evaluation gate with confidence intervals | Added `sim:eval-gate` with side-balanced runs, Wilson 95% CI, side split, average points, terminal reasons, selected candidate rank, zero-fallback enforcement, and optional `--manifest-out`. | `backend/src/sim/evalGate.ts`; tiny rollout/search gates passed with zero fallbacks |
| 6. Episode/seed train-validation split | Added `--split-by row|episode|seed`, default `episode`; training manifest/checkpoint record grouped train/validation split. Python e2e smoke asserts no group leakage. | `training/train_bc.py`; `training/smoke_e2e.py`; `npm run test:python-train` |
| 18. Dataset/run manifests | Rule-bot and outcome exporters now write sibling manifests with args, git SHA/dirty flag, seed/source taxonomy, phase/action-kind counts, terminal reasons or margin buckets, and feature dimensions. Training manifest now includes data path, sample count, split, and feature schema. | `backend/src/sim/manifest.ts`; `exportTrainingExamples.ts`; `exportOutcomeTrainingExamples.ts`; `training/train_bc.py` |
| 2. Controlled common-random rollout samples | Outcome export now supports `--samples`; each label records common sample seed IDs, per-candidate rewards, reward mean/variance, selected-vs-runner-up margin, selected-vs-baseline margin, tie policy, low-margin flag, and selected original heuristic rank. | `backend/src/sim/exportOutcomeTrainingExamples.ts`; `/tmp/uma-outcome-smoke/examples.jsonl` smoke |
| 3. Candidate ranker diversity | Added shared ranker modes `heuristic`, `phase-diverse`, and `epsilon`; outcome export and recursive search use the same helper. Baseline and legal pass/end-turn are always included. Eval summaries now report average selected original rank when available. | `backend/src/sim/candidateRanker.ts`; outcome/export and search gate smokes |
| 4. Policy-baseline-visited outcome export | Outcome export now records `source: ai-policy-baseline-visited`, `labelSource: rollout-labeled`, source taxonomy in manifest, and baseline modeled-action fallback count. | `backend/src/sim/exportOutcomeTrainingExamples.ts` |
| 5. DAgger trace export | Model evaluator now accepts `--decision-trace-out` and writes public model-side decision JSONL rows with legal actions, selected action, heuristic baseline, selected rank, fallback flag, and final game result. | `backend/src/sim/evaluateModelVsHeuristic.ts`; trace smoke |
| 8. Feature schema compatibility | Python now fails clearly on action feature length mismatch instead of silently padding/truncating old rows. Checkpoints and manifests record feature schema metadata; ONNX export/server validate expected feature dimensions. | `training/uma_ai/features.py`; `training/export_onnx.py`; `training/serve_onnx.py` |
| 19. Stronger training smoke tests | `test:train` now includes fingerprint contract fixtures. Python e2e asserts grouped split and exercises stricter ONNX export/server dimension path. | `backend/package.json`; `training/smoke_e2e.py` |

### Backlog Item Plan / Acceptance Ledger

| # | Plan | Target acceptance signal | Current status |
| --- | --- | --- | --- |
| 1 | Build a bounded turn-bundle planner on top of `advanceModeledTurnStep`, returning the first action of the best sequence and scoring leaves with rollout/search. | 65%+ over 500 side-balanced games, zero fallbacks, CI/side/points/terminal report. | Initial implementation exists but acceptance not met. Tiny smoke: 0/2, zero fallbacks. Needs tuning/common-random scoring and larger gate before distillation. |
| 2 | Compare every candidate against the same N rollout seeds, record margin/variance/tie metadata, and down-weight low-margin rows. | Repeated seed gives identical labels; candidate order changes only below tie threshold; rows include sample/margin metadata. | Partially implemented: row metadata and multi-sample common-random rewards exist. Order-invariance test and training margin-bucket metrics remain. |
| 3 | Extract candidate ranker modes shared by search and outcome export; always include baseline plus legal pass/end-turn. | Search reports ranker mode, candidate coverage, selected original rank; small exhaustive audits show low dropped-best rate. | Partially implemented: shared ranker is used by outcome export and recursive search; eval reports average selected rank. Exhaustive dropped-best audits still open. |
| 4 | Treat baseline-modeled trajectory source as first-class and record fallback/no-op accounting. | Manifests report source taxonomy and phase/action coverage; baseline fallback count is explicit. | Implemented for outcome export manifests. Training mix reporting still needs source counts across combined datasets. |
| 5 | Add `--decision-trace-out` to model evaluation with model, heuristic, rollout/search decisions and final results. | DAgger trace has both sides, no hidden opponent-hand leakage, and can be loaded as mixed training data. | Partially implemented: evaluator writes public model-side decision traces with heuristic baseline and final result. Mixed dataset loader/reporting still open. |
| 6 | Split validation by episode or seed by default and record exact groups. | `--split-by row|episode|seed`, default grouped; manifest records train/validation groups. | Implemented. |
| 7 | Extend combat action enumeration/execution to all attacks and attack choices. | Contract covers representative multi-attack cards; legal combat export includes attack index. | Not implemented. Needs combat planner/type migration. |
| 8 | Version feature schemas and fail on dimension mismatches. | Exports/manifests/checkpoints/ONNX serving record compatible schema/dimensions; mismatches fail clearly. | Partially implemented: Python/ONNX dimension checks and training schema metadata. TS row-level schema versions and slot smoke fixtures remain. |
| 9 | Validate state semantic features and hidden-information safety with fixtures and ablations. | State schema recorded; tests cover hand/discard/readiness/KO/catalog/no leakage. | Partially covered by existing hidden-hand smoke and schema metadata. Semantic fixture suite remains. |
| 10 | Add feature ablation switches and phase/action-kind reporting. | Ablation report includes row metrics, closed-loop WR, margin buckets, rank, and phase/action-kind accuracy. | Not implemented. |
| 11 | Port stronger hard-AI procedural scorers into `ai-policy` or shared scoring. | `chooseHighestScoredAction` approaches hard AI in mirror eval without import cycles. | Not implemented. |
| 12 | Score trainer/ability choices by simulated marginal outcome. | Target/discard fixtures improve; fewer low-margin labels; ability use correlates with point/survival gains. | Not implemented. |
| 13 | Broaden turn-goal detection for midgame strategy. | Goal telemetry by turn/side; goal fixtures; average points improve. | Not implemented. |
| 14 | Train calibrated value/action-value objective before using value in search. | Calibration metrics and one-step value selection beat simple point-margin baseline. | Not implemented. Current findings still mark value head unusable. |
| 15 | Rebaseline all methods on one corrected held-out suite. | One comparable table with fixed seeds/decks/evaluator, CI, side split, points, fallbacks, terminal reasons. | Enabled by eval gate, not yet run at 500-game scale. |
| 16 | Centralize and test state fingerprint. | All no-op/stall/search paths use one helper; mutation fixtures pass. | Implemented for evaluator/headless/export/action-contract paths. |
| 17 | Add pass/fail evaluation gate. | Gate reports WR, Wilson CI, side split, points, terminal reasons, fallbacks/no-ops and fails thresholds. | Partially implemented: gate reports all except an explicit separate no-op count beyond fallback/stall accounting. |
| 18 | Write manifests beside exports, training runs, and evals. | Every JSONL/checkpoint/eval has reproducible sibling metadata. | Partially implemented: dataset/training manifests exist; eval gate can write a manifest via `--manifest-out`; base evaluator still console-only. |
| 19 | Expand training/eval smoke coverage. | `test:train` covers export/outcome/fake-model/schema/fingerprint regressions. | Partially implemented: fingerprint, grouped split, and schema/dimension checks. Outcome determinism/fake-model/eval-gate failure tests remain. |

### Immediate Next Work

1. Add outcome label order-invariance and low-margin bucket training tests.
2. Add base evaluator manifest writing or route standard evals through `sim:eval-gate`.
3. Extend trace rows with optional rollout/search teacher alternatives when policy/value selection is used.
4. Implement the turn-bundle planner with trace telemetry enabled.
