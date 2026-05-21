# R16 Archive — Rolled Detail

## Multi-Worker Corpus Harness — full benchmark + design notes

Compact summary lives in `r16.md` § "Multi-Worker Corpus Harness".

### Wall-clock benchmark (full, pilot's exact recipe)

50 games × modelSide=both, baseline selection, 50 sims, 200 rollout steps,
CRN 3, rule-bot-mirror:

| `--workers` | Wall-clock | Speedup vs N=1 |
| ---: | ---: | ---: |
| 1 | 873.73s (14m34s) | 1.0× (matches the 14m23s pilot) |
| 16 | 149.85s (2m30s) | **5.83×** |
| 24 | 154.31s (2m34s) | 5.66× (past the parallel knee) |

### Design notes (full)

- `evalGate.ts` already implemented the full work-stealing harness
  (`MCTS_WORK_STEALING_ENABLED`, `WORK_STEALING_PREFETCH`, deterministic
  slot ordering) using the same `runModelVsHeuristicGame` per-game
  function — but it hardcoded `relabelMcts: false` and never serialized
  worker `decisionTraces`. Closing those two gaps lifted evalGate into
  the corpus generator with no new harness.
- Trace contract: orchestrator owns the trace file, buffers
  `result.decisionTraces` by `taskIndex`, flushes in ascending order at
  run end. Byte-deterministic across `--workers` (verified bit-identical
  at N=1/16/24, sorted multisets byte-equal across all 2088 rows).
- Corpus-mode bypass: when `--relabel-mcts` is set, gate floors
  (`--min-games` / `--min-win-rate` / `--min-ci-lower`) become
  informational (still reported, no non-zero exit). A weak-model corpus
  run does not need to override floors per recipe.
- Smoke `evalGateCorpusGenSmoke.ts` locks: (1) flag pass-through;
  (2) `--workers 4` vs `--workers 1` produce identical sorted multisets
  of trace rows; (3) corpus-mode bypass exits PASS under absurd floors.
  Wired into `npm run test:train`.
- Why sub-linear: per-game wall-time variance (~5-30s/game) × small
  per-worker queue depth (~6 games each) → straggler tail dominates.
  N=24 is slightly worse than N=16, confirming the parallel knee. No
  serve_onnx contention (rule-bot-mirror is server-free); no IPC
  backpressure observed on the per-task IPC trace flush path.
- Projected prod-corpus wall-clock at N=16: ~25 min total (rule-bot-
  mirror ~5 min + each served recipe ~10 min, ~2× slower per recipe due
  to served + 2 MCTS calls per decision for search-vs-rule). Down from
  the naive ~2h sequential single-thread.

## R16-P1 v3.1 trajectory + reasoning (full prose)

Compact summary lives in `r16.md` § "R16-P1 step 3". Full trajectory
table + reasoning retained here.

### Full trajectory — v3.1 6-iter vs v3.0 R110-W6-repro 5-iter

n=120 side-balanced rollout-leaf gate, Wilson lower vs rule bot. All
v3.1 iters: 120-game, 0 heuristic fallbacks, promoted.

| Iter | v3.1 Wilson-lower | v3.1 win-rate | v3.0 Wilson-lower |
|---|---|---|---|
| 0 | 0.5189 | 0.608 | 0.5273 |
| 1 | 0.5869 | 0.675 | 0.5612 |
| 2 | 0.5442 | 0.633 | **0.6042 (best)** |
| 3 | **0.5955 (best)** | 0.683 | 0.5106 |
| 4 | 0.5527 | 0.642 | 0.5612 |
| 5 | 0.5527 | 0.642 | — |

v3.1 best-promoted **0.5955** (iter-3) vs v3.0 best-promoted **0.6042**
(iter-2): −0.0087, within-noise, no Wilson win → **NO-GO**.

### Reasoning (full)

- **No Wilson-lower win.** v3.1 peak 0.5955 is below the v3.0 peak
  0.6042; the difference is within gate noise — at best parity, not
  a gain.
- **Oscillation, not climb.** v3.1 sits in a ~0.52-0.60 band and
  never clears the v3.0 ceiling.
- **Extension confirmed no late climb.** The user requested a +2-iter
  extension (iters 4-5) after the 4-iter run; iter-4/5 are flat at
  0.5527, confirming no delayed gain.
- **Schema-independent W6 pattern.** The iter-peak-then-oscillate
  behavior reproduces under v3.1 as it did under v3.0 and 96-d —
  consistent with the canonical R110 finding that this is an
  intrinsic W6-recipe property, not a representation effect.
  Canonical: `docs/ai-research/progress/r110.md` §3.

## R16 Contested-Coverage Pilot — Methodology + cross-references (full prose)

Compact summary lives in `r16.md` § "R16 Training-Data — Contested-Coverage
Pilot"; full prose retained here.

### Methodology / discipline (full)

Pre-registered protocol followed in order: (1) no-op invariant — loader
output bit-identical with knobs unset (verified, synthetic v3 corpus +
default-arg equality); (2) coverage-moves — both knobs move
`legal_action_count` across `{baseline, ~30%, ~45%}` on the real audited
corpus at fixed retained count (Option 1 hits 0.2022/0.3001/0.4500
exactly; Option 3 holds count fixed and scales contested weight mass);
(3) one cheap point measured first (train ~10s + export ~1s + gate ~90s ≈
~2 min/point ≪ 30-min threshold) → full 3-point sweep run. The n≥1000
expensive closed-loop gate was NOT launched (stop rule: forbidden until
the cheap slope is positive — which it now is, unlocking but not
performing the next tier).

### Cross-references (full)

- Pre-registered design (not restated here):
  `docs/ai-research/scoping/r16-training-data-backlog-refinement.md`
  § "Fork A — Contested-State Data Coverage"
- Contested metric definition + baseline:
  `docs/ai-research/analysis/training-data-coverage-audit.md`
  (`legal_action_count`)
- Environment note: the v2 `mixed.jsonl` is unloadable by the current
  R7.b.2-Phase-2-hardened loader (no `cardIdsByZone`); the v3
  re-extraction `mixed-v3.jsonl` (same 3789 retained rows) is the
  audit-faithful loadable corpus used here.

## 3a Emit-Path — Corrected Cost Finding (full historical framing)

Compact summary lives in `r16.md` § "R16 Training-Data — Online MCTS
Relabel Infra (3a emit-path)" § "Corrected cost finding"; full prose
below.

A read-only sizing pass (2026-05-18) found the prior
"INFRA-BLOCKED / traces don't serialize GameState / offline relabel
impossible / heavy expensive arm" framing materially misrepresented the
cost and was driving priority wrong. Rollout-leaf MCTS already runs on
the live in-game `GameState` at the decision point
(`evaluateModelVsHeuristic.ts:315-316,711`; `runMcts` `mcts.ts:170`); the
rollout-leaf / no-noise / uniform-prior knobs already exist; the full
diagnostics already map field-for-field onto the `oracle` schema; the
forced-state skip and the downstream leak/sum/range audits already exist.
The *only* blocker was that `runMctsForSide` discarded
`MctsResult.visits`/`diagnostics`. Closing that is a bounded ~2-4 eng-day
emit-path wiring, not a new serializer — and the same infra unblocks both
3a and 3b. `training-data-deep-program` 3a was therefore reprioritized
ahead of `r16-p2-per-uma-slot-tokens`; the larger-n contested-coverage
confirmation gate is decoupled (parallel/later, no longer an infra
blocker).

Sibling of `r16.md`. Holds verbose detail that has been rolled out of the
primary progress doc to keep the per-sprint file under its ≤500-line cap.
Read-mostly. Update only when r16.md trims another older block.

---

## R16-P1 v3.1 ablation — Methodology / harness facts / residual gaps (full prose)

Compact summary lives in `r16.md` § R16-P1 step 5; full prose retained
here for future re-litigation if v3.1 (or any v3.x variant) is ever
re-opened.

### Faithful R110-W6-repro mirror

Exact v3.0 R110-W6-repro recipe: rollout leaf, 100 sims, CRN 3, 200
rollout steps, n=120 side-balanced gate at `--eval-seed-start 9000`,
**unregularized** — W6-fix flags OFF, verified in `events.jsonl`.
Differed from v3.0 only in `--state-dim 164` + the additive-tail init.

### Init equivalence

The 164-d additive-tail warm-start is output-identical to the v3.0
init (logit/value delta = 0.0) → init does not confound the A/B.
Builder: `make_v31_additive_init.py` (commit `3ce1404`).

### Faithful resume

Iters 4-5 were a `--resume-state` continuation, not a fresh run; the
self-play seed sequence is verified continuous (iter-4 seed
`30240 = 30000 + 4*60`).

### Two corrected recipe-confounds (process lessons)

- **W6-fix defaulted ON** (`r12_orchestrator.py:61-62`) — would have
  confounded v3.1 vs the unregularized 0.6042 baseline; corrected
  with `--no-w6-fix-*`. **Lesson:** verify effective config in
  `events.jsonl`, never infer flag defaults.
- **`--eval-games 120` was wrong**; the real R110-W6-repro gate is
  `--games 60` (n=120 side-balanced). **Lesson:** verify recipe args
  against the real baseline `gate.log`, not reconstructions.

### Durable harness fact — serve_onnx TCP backlog fix (`e846881`)

Under 24-worker gate fan-in, `serve_onnx.py` `PolicyServer` used the
Python default TCP `request_queue_size=5`; the listen backlog
overflowed (kernel SYN-flood on the serve port) → ECONNREFUSED
cascade → gate crash (bogus `wilson_lower=0.0`), voiding two
launches. Fix: `request_queue_size=128` + per-iter `serve.log`
observability. Empirically proven against the exact `listen(5)`
mechanism; held **0 connection errors** across all subsequent
iterations. Durable robustness fact for any high-fan-in gate,
independent of v3.1.

### Known residual gap — benign, low-priority

`training/r14_value_crossover_probe.py` is not wired for 164-d, so
the r14-crossover diagnostic fails every v3.1 iter. This is a
logged-only diagnostic with **zero effect on the gated/promoted
checkpoint** (confirmed against the orchestrator data-flow and the
v3.0 run, which also gated plain distilled checkpoints). Optional
future one-knob fix: thread `--state-dim` through the probe —
backlog-level only, not a blocker.

## R16-TD 3a — Loader Verification full detail

Compact summary lives in `r16.md` § "Production Corpus — Audit-v2
ALL PASS". Rolled detail:

Audit-v2 PASS is hollow if the Python BC loader silently drops the
new schema. Verified by smoke-loading the sealed phase 1+2 traces
through `training/uma_ai/dataset.py`
`JsonlPolicyDataset.load_policy_samples`:

- `rule-bot-mirror/traces.jsonl`: **4196 / 4196 retained**;
  4196/4196 carry `policy_target` + `card_ids_by_zone` +
  `action_card_idx`.
- `policy-vs-rule/traces.jsonl`: **3664 / 3664 retained**; same
  soft coverage.

Loader retention equals corpus row count exactly. The
`policyTargets`/`policy_target` camelCase-vs-snake-case mismatch
concern is unfounded: `dataset.py:406-440` reads `policyTargets`
verbatim and validates len/finite/non-negative/sum==1±1e-5.

Soft path actively exercised in training: `train_bc.py:490-498`
and `:576-579` branch on `policy_targets is not None` and take the
soft cross-entropy path (`-(policy_targets * log_probs).sum(dim=1)`),
not argmax hard-CE. The 3a upgrade's value is realized end-to-end.

`oracle`, `stateSource`, `selectedOriginalRank` are silently
ignored by the loader (not rejected). Default `state_dim=110`
(v3.0); v3.1 is opt-in via `state_dim=STATE_DIM_V3_1`. **3b
training can proceed directly off this corpus — no loader adapter
needed.**

Latent risk (forward-looking, not a current blocker):
`dataset.py:34-43` notes `ROW_SCHEMA_VERSION` was deliberately held
at 1 across Phase 2 to keep TS-produced corpora loadable. If a
future phase bumps the row schema without coordinating the TS
writer at `backend/src/sim/evaluateModelVsHeuristic.ts:367`, this
corpus will retroactively fail to load.

## R16-TD 3a — Served-Recipe Smokes full detail

Compact summary lives in `r16.md` § "Production Corpus — Audit-v2
ALL PASS". Rolled detail:

Before the prod corpus launch (2026-05-21), each served recipe was
smoked end-to-end against a live `serve_onnx` instance at tiny N
(`--workers 2 --games 1`) to validate the missing wiring (server
startup, `/health`, `--model-url`, `--mcts-prior`):

| Recipe | `--selection` / `--mcts-prior` | Rows | Audit |
| --- | --- | ---: | --- |
| `policy-vs-rule` | `policy` / `uniform` | 16 | PASS (oracle + simplex + stateSource) |
| `search-vs-rule` | `mcts` / `policy` | 28 | PASS (oracle + simplex + stateSource) |

Both ran against a shared `serve_onnx` instance pinned to
`runs/R13-W6-phase-d/iter-2/policy.onnx` (96-d v2 schema,
`--ort-threads auto`, `[CUDAExecutionProvider,
CPUExecutionProvider]`). `/health` returned 200 within ~1s on this
host. Clean SIGTERM shutdown. The launch script
`runs/R16-TD-3a-prod-corpus/launch.sh` runs all three recipes in
sequence (rule-bot-mirror first, server-free; then serve_onnx start
+ policy-vs-rule + search-vs-rule; trap-handled server shutdown).

## R16-TD 3a — Production Corpus per-slice coverage

Compact summary lives in `r16.md` § "Production Corpus — Audit-v2
ALL PASS". Rolled detail:

Audit invocation:

```
python training/audit_relabel_corpus.py \
  --recipe-trace rule-bot-mirror=runs/R16-TD-3a-prod-corpus/rule-bot-mirror/traces.jsonl \
  --recipe-trace policy-vs-rule=runs/R16-TD-3a-prod-corpus/policy-vs-rule/traces.jsonl \
  --recipe-trace search-vs-rule=runs/R16-TD-3a-prod-corpus/search-vs-rule/traces.jsonl \
  --out runs/R16-TD-3a-prod-corpus/audit-v2.json
```

All three `stateSource` tags match the recipe directory; every row
carries the full 11-key `oracle` block; simplex `policyTargets`; no
forced-state row emitted. Per-recipe top skew is benign side-balance
(~52-53% to the weaker side). Coverage is healthy across all slice
axes:

- **Phases (7/7 present per recipe)**: `ability` / `attach` /
  `bench` / `combat` / `evolve` / `trainerAfter` / `trainerBefore`
  — `trainerBefore` dominates (~37-41% of rows per recipe),
  consistent with deck-loading early-game pressure.
- **Action kinds (8/8 present per recipe)**: `attachEnergy` /
  `attack` / `evolve` / `pass` / `playBasic` / `playTrainer` /
  `retreatAttack` / `useAbility`. Rule-bot-mirror skews higher on
  `pass` (1362 vs 831 in policy-vs-rule) and `useAbility` (176 vs
  78) — consistent with rule-bot behavioral biases relative to the
  model.
- **Legal-action buckets** (≥3 legal = contested): rule-bot-mirror
  49.9%, policy-vs-rule 49.7%, search-vs-rule 52.6%. ~50% of rows
  carry the contested signal that 3b preference-pair extraction
  consumes.

Promote criteria (per scoping § P1 acceptance): retained
contested-row rate ≥80% — MET trivially at 100% across all
recipes. Loader retention verified separately (above): the
`JsonlPolicyDataset` accepts 100% of these rows.
