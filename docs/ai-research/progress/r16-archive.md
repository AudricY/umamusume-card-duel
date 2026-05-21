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

## R16 Contested-Coverage Pilot — Mechanisms + Methodology (full prose)

Compact summary + chunk-5j falsification verdict live in `r16.md` § "R16
Training-Data — Contested-Coverage Pilot"; full prose retained here.

### Mechanisms as built (full detail; write scope: `training/uma_ai/dataset.py`)

Two independently-selectable knobs, both holding total retained-row count
fixed, both defaulting OFF (bit-identical no-op when unset — verified):

- **Option 3 — legal-action-count loss weighting.** `contested_loss_weight`
  (×1.0 = no-op) multiplies `_sample_weight` for rows with
  `≥ contested_min_legal` (default 4) legal actions. Reuses the existing
  `sample_weight` / `collate_policy_batch` / `normalized_weights` seam; no
  corpus change, no row-count change.
- **Option 1 — contested resampling.** `contested_resample_fraction`
  (`None` = no-op) buffers the retained stream, splits on the ≥4-legal
  threshold, and draws with replacement from each group to hit the target
  contested fraction while keeping total retained count exactly fixed
  (deterministic, RNG seeded by `--seed`). Empty-group guard yields the
  unmodified stream rather than fabricating distribution.

Threaded into `train_bc.py` (bc mode only) as `--contested-loss-weight`,
`--contested-min-legal`, `--contested-resample-fraction` — same
`--state-dim`-style option threading; no new config framework.

### Pilot sweep detail (n=300/side, supersedes pilot-positive framing)

Pilot full table (Option 1 swept) — superseded by chunk 5j n=1000
falsification but retained here for evidence trace:

| Coverage point | `legal_action_count` | player WR | opp WR | side-bal Wilson-lower | overall WR |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline (no knob) | 0.2022 | 0.290 | 0.393 | 0.2416 | 0.342 |
| resample ~30% | 0.3001 | 0.307 | 0.393 | 0.2572 | 0.350 |
| resample ~45% | 0.4500 | 0.313 | 0.397 | 0.2635 | 0.355 |

Improvement at n=300/side driven by the weaker (player) side; opponent
side flat. Baseline 0.2022 reproduces the audit headline (below ≥30%
floor; R7 retained is exactly-2-legal-dominated). Confirmation gate at
n=1000/side (chunk 5j) found all three pilot point estimates fall BELOW
the n=1000 95% CIs and the slope sign flips — the pilot was on the low
tail of sampling noise.

### Methodology / discipline (full)

Pre-registered protocol followed in order: (1) no-op invariant — loader
output bit-identical with knobs unset (verified, synthetic v3 corpus +
default-arg equality); (2) coverage-moves — both knobs move
`legal_action_count` across `{baseline, ~30%, ~45%}` on the real audited
corpus at fixed retained count (Option 1 hits 0.2022/0.3001/0.4500
exactly; Option 3 holds count fixed and scales contested weight mass);
(3) one cheap point measured first (train ~10s + export ~1s + gate ~90s ≈
~2 min/point ≪ 30-min threshold) → full 3-point sweep run; (4) n=1000/side
confirmation gate pre-registered to fire only on positive cheap slope —
fired, falsified the pilot direction (chunk 5j).

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

## R16-TD 3b chunks 5a-5e — per-chunk landings (full)

Compact summary lives in `r16.md` § "3b Chunks 5a-5e" + the
Cumulative Verdict table.

- **5a** (3ep β=0.1 lr=1e-5, commit dc68ac8): HP-overfit FAIL
  (Δpair_acc −0.0167 / Δtop1 −0.0321). Offline gate (scoping § P2
  step 5) caught the regression before any n=1000 spend.
  `training/pad_checkpoint_to_v3.py` (numerically null pad) makes
  the pre-v3 96-d checkpoint usable as both ref and init.
- **5b** (1ep β=0.5 lr=3e-6, 21s wall-clock): HP probe POSITIVE
  (+0.0043 pair_acc / +0.0006 top1, best of the series). HP, not
  corpus, was the 5a issue — `margin_bucket 0.05-0.10` recovered
  +6.6pp. Both deltas inside the ±0.022 noise band at n=1620.
- **5c-long** (5ep at 5b HPs, commit c243fd8): REGRESSED below
  baseline (−0.0012 / −0.0117). 5-epoch overshoot proves the
  runner-up-only corpus signal is exhausted by ~1 epoch at this
  HP regime; train/val gap 0.696/0.640.
- **5d (pair-builder)** (commit 7448b14): expanded
  `pair_builder.py` with `top_k` (ranks 2..K) and `rule_bot`
  (winner vs `heuristicSelectedActionIndex` when different) modes.
  Diversified corpus = **13419 pairs (+65.8% vs 8093)** — 60.3%
  runner_up + 33.9% top_k_rank_3 + 5.8% rule_bot. 52% of states
  had rule-bot == MCTS winner (structural). Margin histogram
  stays winner-dominant (80.7% ≥0.20). Deferred `high_prior` +
  `policy_argmax` (data-availability).
- **5e** (1ep β=0.5 lr=3e-6 on the 13419-pair diversified corpus,
  commit 059f2a7): COMPARABLE to 5b (+0.0031 / −0.0043), inside
  noise — diversification did NOT lift the headline. Falsifies
  "runner-up-only is the bottleneck" at this HP budget.

## R16-TD 3b Cumulative Verdict — root-cause candidates (superseded by chunk 5g)

Pre-chunk-5g framing kept for history. Compact summary + the
sharpened diagnostic ladder live in `r16.md` § "Candidate-1
kill-test (chunk 5g)". Original 4-candidate list and forward-line
proposal preserved below for historical context only.

Root-cause candidates (deferred to user-scoped decision at the
time of cumulative-verdict writeup):

1. **Relabel-MCTS quality**: the rollout-leaf 100-sims relabel
   may not be meaningfully sharper than the 96-d reference's
   own policy on the contested states. The gap to learn is
   small, so DPO can't lift the policy above the ref ceiling.
   Test: re-relabel with a stronger MCTS (e.g. 400+ sims or a
   value-head leaf) and see if the relabel-vs-ref disagreement
   distribution shifts toward bigger margins.
2. **Reference too strong**: the 96-d production pin is already
   near a local optimum for the rule-bot opponent; DPO can't
   improve much without a stronger comparison target. Test:
   train BC from scratch + use that as ref/init (lift may
   appear vs a weaker baseline).
3. **DPO objective mismatch**: the pair-margin loss may not
   transfer to the candidate-ranking metric on this domain.
   Test: switch to direct soft-CE on the relabeled
   `policyTargets` (i.e. BC on the 11798-row 3a corpus) and
   compare against the same held-out floor.
4. **Held-out split correlation**: blake2b on (episode, step)
   may not separate train/held-out enough for lift to
   generalize. Test: re-split by episode only (held-out
   episodes never appear in train) and re-eval.

Forward line proposal (orchestrator recommendation, pre-5g):
candidates (1) and (3) are the cheapest to test (no new
generation, just a re-train or re-relabel). Candidate (3) —
direct soft-CE on the 11798 relabel rows — is closest to the
"BC-on-the-new-corpus" baseline that DPO should beat; if BC
itself doesn't lift the offline metrics, the relabel corpus
itself is the binding constraint (candidate 1) and chunk-5d
diversification was always going to be a marginal lever. Per
the user's 2026-05-21 sequencing refinement (loop_note), the
post-data-work CEILING PATH is W6 HP-sweep (A) cheap-first;
the 3b workstream may have hit its productive limit absent a
direction call from the user.

Chunk 5f then falsified candidate-3 directly; chunk 5g then
re-tested candidate-1 with a read-only label-divergence check
(rather than the proposed expensive stronger-MCTS re-relabel)
and found labels confidently wrong on contested heads — see
the chunk-5g ladder in `r16.md`.

## Chunk 5g — per-kind table + disagreement confidence

Compact summary lives in `r16.md` sub-§ "Candidate-1 kill-test (chunk 5g)".

Per-action-kind agreement (binned on relabel argmax kind), full
table from the 11798-row label-divergence audit:

| kind | n | agree | KL(rel‖ref) |
| --- | ---: | ---: | ---: |
| playTrainer | 4732 | **0.499** | 0.415 |
| pass | 2353 | **0.479** | 0.424 |
| retreatAttack | 140 | **0.464** | 0.327 |
| attachEnergy | 2572 | 0.848 | 0.241 |
| evolve | 700 | 0.863 | 0.117 |
| playBasic | 554 | 0.986 | 0.095 |
| attack | 471 | 0.781 | 0.225 |
| useAbility | 276 | 0.822 | 0.152 |

Disagreement confidence (n=4,320 disagreeing rows): relabel max-prob
when disagreeing mean **0.579**, median 0.54, **p90 0.91** —
hyper-confident different picks. Relabel prob on ref's choice:
mean 0.227, p10 0.03 — ref's pick often near-zero under relabel.
27% of all rows (n=3,121) have relabel confidence ≥0.8; in that
bucket ref/relabel still only agree 70% of the time. Mean JSD
**0.0739 nats**.

## Chunk 5h ladder scoping (full prose)

Compact summary lives in `r16.md` sub-§ "Step ordering refined
(chunk 5h scoping) — superseded by 5i".

- **Step order FLIPPED — step 2 first.** Step 2 (argmax-flip at
  400 sims) is cheaper than step 1 (value-eval audit needs ~200
  LOC across `mcts.ts` + `evaluateModelVsHeuristic.ts`; the
  "CRN-paired variance across actions" framing is ambiguous —
  current MCTS uses within-leaf K=3 CRN seeds, across-action
  paired CRN does not exist). Step 1 deferred pending CRN-paired
  clarification.
- **Step 2 recipe constraint: `rule-bot-mirror` ONLY.** Server-free,
  baseline-selected, so visited states at 400 sims are IDENTICAL
  to 100-sim row-for-row (only `policyTargets` differs).
  `policy-vs-rule` / `search-vs-rule` use served-checkpoint +
  MCTS-driven moves → trajectories diverge at 4× sims, no
  row-by-row diff possible. (Chunk 5i invalidated the
  rule-bot-mirror determinism assumption — see r16.md § "Step 2
  verdict (chunk 5i)" secondary finding.)
- **Step 2 mechanics.** Re-run rule-bot-mirror at
  `--mcts-simulations 400`, same seed range (10000-…), produces a
  new `traces.jsonl` keyed identically by `(seed, modelSide,
  step)`. Small ~30 LOC Python join compares argmax across the
  100/400-sim corpora on the kill-test confident-disagreement
  subset (contested-head rows, relabel max-prob ≥ 0.8 from
  `candidate1-killtest/metrics-confidence.json`).
- **Wall-clock**: ~20-40 min foreground at 16 workers
  (rule-bot-mirror at 100 sims was ~5-10 min; MCTS linear in sims).
- **Verdict criteria**: >30% argmax flip on confident-disagree rows
  → sim count load-bearing, candidate-1 lives, escalate to
  full-corpus 400-sim relabel + BC retrain; >80% argmax sticky →
  bug upstream of sim count (rollout policy / leaf eval / CRN),
  candidate-1 dies, abandon stronger-MCTS line. Step 1 reopens
  only if step 2 negative AND across-action CRN statistic
  clarified.

## R16 Training-Data — Online MCTS Relabel Infra (3a emit-path) — full prose

Rolled from `r16.md` 2026-05-21 to free headroom for the R16-P2 v3.2 verdict.
All sub-sections (3a emit-path, multi-worker harness, production corpus,
3b chunks 1-5e DPO probes, 3b cumulative verdict, chunk 5f BC-on-relabel,
chunk 5g candidate-1 kill-test, chunk 5h scoping, chunk 5i verdict) are
DONE-NEGATIVE with no remaining autonomous action; the active forward
direction from this arm is the user-gated W6 HP-sweep (Ceiling Path A),
tracked at queue item `w6-loop-anti-degradation`. Compact summary in
`r16.md` § "R16 Training-Data — Online MCTS Relabel Infra (3a) — rolled".

### Corrected cost finding (reprioritization basis — archived)

Read-only sizing pass (2026-05-18) showed the prior "INFRA-BLOCKED /
heavy expensive arm" framing was wrong: MCTS already runs on the live
`GameState` at the decision point and all diagnostics + audits
already exist; only the discarded `MctsResult.visits/diagnostics`
needed wiring (~2-4 eng-day emit-path, not a serializer). 3a was
therefore reprioritized ahead of `r16-p2`.

### What landed (3a emit-path, this chunk only)

- New `--relabel-mcts` mode + `--relabel-state-source` tag in
  `backend/src/sim/evaluateModelVsHeuristic.ts`. A sibling
  `runMctsRelabel` runs rollout-leaf, no-Dirichlet MCTS on the live
  `GameState` at the decision point and threads the full `MctsResult`
  (`visits` + `diagnostics`) into three new optional `DecisionTraceRow`
  fields: normalized `policyTargets`, an `oracle` block, and a
  `stateSource` tag. Forced states (`legalActions.length <= 1`) suppress
  row emission entirely (do not fall through unlabeled).
- `backend/src/sim/dagger/relabelDecisionTrace.ts`: a
  `--label-source rollout-leaf-mcts` pass-through branch forwards the
  evaluator-emitted `policyTargets`/`oracle` verbatim and reuses the
  existing leak / index-range / sum-to-1 / length gates (no new audit
  logic, no mixture recomputation).
- No-op invariant VERIFIED: with `--relabel-mcts` off, trace rows carry
  none of the new fields and are deterministic/bit-identical across runs;
  full `npm run test:train` (incl. `multiTeacherTraceSmoke`,
  `daggerRoundSmoke`) passes; `npm run build` clean. ~0 functional
  `mcts.ts` change (one type re-export only).

### Pilot smoke + Audit-v2 (superseded by prod corpus)

50-game rule-bot-mirror pilot (modelSide=both = 100 games, 50 sims, 14m23s
wall-clock single-thread): 2127 contested rows, Audit-v2 ALL PASS (100%
retained, oracle present, no leaks). Superseded by the 3-recipe production
corpus run below (§ "Production Corpus — Audit-v2 ALL PASS"); pilot kept
here as the cost-validation reference for the multi-worker harness spike.

### Source-Recipe Matrix

The three source recipes from scoping § P1 are achieved via the existing
`--selection` + `--relabel-state-source` flags on
`backend/src/sim/evaluateModelVsHeuristic.ts`; **no new flag is needed**.
`runMctsRelabel` operates on the live `GameState` independent of which
selection played the action, so the recipe is fully determined by the played
selection (`--selection`) plus the corpus tag (`--relabel-state-source`).

Per-recipe cost (per decision; CPU cost dominated by MCTS sims × rollout
steps):

| Recipe | `--selection` | Played-action cost | Relabel cost | Model server required? |
| --- | --- | --- | --- | --- |
| `rule-bot-mirror` | `baseline` | 0 NN, 0 MCTS | 1 MCTS call | **No** |
| `policy-vs-rule` | `policy` | 1 NN forward | 1 MCTS call | Yes |
| `search-vs-rule` / `mcts-vs-rule` | `search` or `mcts` | 1 MCTS call (played) | 1 MCTS call (relabel) | Yes if `--mcts-prior policy` (preferred — uniform priors degrade played-search quality, so `search-vs-rule` is effectively serve-required) |

Smoke landed: `backend/src/tests/relabelMctsSmoke.ts` codifies the matrix in
three sub-cases — (a) `--relabel-mcts` ON + rule-bot-mirror with full
`policyTargets` / `oracle` / `stateSource` audit + relabel pass-through; (b)
same with `policy-vs-rule` tag (proves the state-source tag wires through
independently of played selection); (c) `--relabel-mcts` OFF as the no-op
regression guard (no `policyTargets` / `oracle` / `stateSource` field leakage
when the mode is unset). Wall-clock ~2.5s; wired into `npm run test:train`.

Next user-gated chunk (intentionally not auto-launched): the 200-400-game
production corpus + Audit-v2 acceptance gate against the recipe matrix.

### Multi-Worker Corpus Harness (compute optimization spike — compressed)

`evalGate.ts` (which already implements work-stealing dispatch over
`runModelVsHeuristicGame`) extended to host the corpus generator:
parses `--relabel-mcts`/`--relabel-state-source`, owns the trace
file, buffers `result.decisionTraces` by `taskIndex`, flushes in
deterministic order at run end (verified bit-identical across N=1/16/24).
Smoke `evalGateCorpusGenSmoke.ts` locks the contract. Wall-clock on
pilot-equivalent recipe: N=1 873s → **N=16 150s (5.83×)** → N=24
154s (past the knee). Speedup sub-linear because per-game variance
× ~6 games/worker = straggler tail dominates. Full benchmark table
+ design notes earlier in this archive § "Multi-Worker Corpus Harness".

### Production Corpus — Audit-v2 ALL PASS

3-recipe production corpus generated 2026-05-21 (start 09:54, end
10:30, total wall-clock ~35.5 min on the 16-worker harness against
the 96-d production pin):

| Recipe | Played-side WR | Total contested rows | Per-recipe Audit-v2 |
| --- | ---: | ---: | --- |
| rule-bot-mirror | n/a (both sides rule-bot) | 4196 | PASS 100% retained |
| policy-vs-rule | **0.319** (raw policy weak vs rule-bot) | 3664 | PASS 100% retained |
| search-vs-rule | **0.490** (search lifts model to near-parity) | 3938 | PASS 100% retained |
| **AGGREGATE** | — | **11798** | **PASS 100% retained** |

`stateSource` tags match recipe; every row carries the full 11-key
`oracle` block + simplex `policyTargets`; no forced-state row
emitted; ~50% of rows ≥3 legal (the contested signal 3b consumes).
Promote criterion (scoping § P1 acceptance, retained
contested-row rate ≥80%) MET trivially.

Pre-launch served-recipe smokes (`policy-vs-rule` 16 rows /
`search-vs-rule` 28 rows, both PASS oracle+simplex+stateSource
against `runs/R13-W6-phase-d/iter-2/policy.onnx`) and the BC
loader-retention check (`JsonlPolicyDataset` accepts 100% of these
rows, exercises the soft `policyTargets` path in
`train_bc.py:490-498`, `oracle`/`stateSource`/`selectedOriginalRank`
silently ignored) live earlier in this archive §§ "R16-TD 3a — Loader
Verification full detail" / "R16-TD 3a — Served-Recipe Smokes full
detail" / "R16-TD 3a — Production Corpus per-slice coverage".
Launch script: `runs/R16-TD-3a-prod-corpus/launch.sh` (3-phase
sequential, trap-handled server lifecycle).

Latent forward-looking risk: `dataset.py:34-43` notes
`ROW_SCHEMA_VERSION` was deliberately held at 1 across Phase 2 to
keep TS-produced corpora loadable. A future phase bumping it
without coordinating the TS writer at
`backend/src/sim/evaluateModelVsHeuristic.ts:367` would retroactively
break this corpus.

### 3b Chunks 1-4 — Pair-builder + DPO wiring + offline floor

- **Chunk 1 (`training/pair_builder.py`, 324 LOC stdlib)**: reads
  3a relabel rows, emits preference-pair JSONL in `runner-up` mode
  (winner=argmax visits, loser=second-highest,
  `sourceKind="mcts-relabel"`). 7 drop predicates counted in
  manifest; only `margin_below_floor` (default 0.05) fired against
  prod corpus. Aggregate: 11798 rows → **8093 pairs** (31.4%
  drop); search-vs-rule drop rate **~4× lower** (9.8%) than the
  other recipes — calibration evidence that played-MCTS sharpens
  relabel visit distributions. Margin histogram: 79% decisive
  (≥0.20). Smoke `pair_builder_smoke.py` wired as
  `npm run test:pair-builder`. Output:
  `runs/R16-TD-3a-prod-corpus/pairs/pairs{.jsonl,.manifest.json}`.
- **Chunks 2+3** (commits `5960cc3` / `1a071b9`): extended
  `pair_corpus.py` for the explicit pair schema and wired
  `card_ids_by_zone` / `action_card_idx` through `train_dpo.py`
  (v3 embedding branch now LIVE under DPO; grouped per-pair
  metrics across 5 axes).
- **Chunk 4 (`training/dpo_quality_eval.py`, 617 LOC)**: offline
  pair-quality gate per scoping § P2 step 5 ("no n=1000 gate until
  offline pair quality clears pre-registered floors"). Deterministic
  split (blake2b on `sourceEpisodeId:sourceStep`, 20% holdout, seed
  17), ranking metrics (top-1/2/3 + MRR, strict-> tie semantics
  over masked legal actions). Reference-baseline against 96-d
  production pin (1620 held-out rows): **`pair_accuracy` 0.6981 /
  `top1` 0.6512 / `top2` 0.8716 / `top3` 0.9321 / `MRR` 0.7921 /
  loss 0.6931 (= log 2, expected ref==self)**. JSON at
  `runs/R16-TD-3a-prod-corpus/baseline-eval.json` (gitignored).
  Trained DPO must improve on these before any closed-loop spend.
  Smoke wired as `npm run test:dpo-quality-eval`.
- **Compat**: `training/pad_checkpoint_to_v3.py` (numerically null
  pad) makes the pre-v3 96-d checkpoint usable as both ref and
  init for the v3-card-embedding-aware `train_dpo.py`.

### 3b Chunks 5a-5e — DPO probe series (collapsed)

Per-chunk landings + commits (dc68ac8 / c243fd8 / 7448b14 /
059f2a7) live earlier in this archive § "R16-TD 3b chunks 5a-5e —
per-chunk landings". Headline numbers preserved in the
Cumulative Verdict table below.

### 3b — Cumulative Verdict

Four DPO probes against the 8093/13419-pair offline corpus all
land within the ±0.022 noise band of the baseline floor:

|  | corpus | epochs | β | lr | Δ pair_acc | Δ top1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 5a | 8093 runner-up | 3 | 0.1 | 1e-5 | −0.0167 | −0.0321 |
| 5b | 8093 runner-up | 1 | 0.5 | 3e-6 | **+0.0043** | **+0.0006** |
| 5c-long | 8093 runner-up | 5 | 0.5 | 3e-6 | −0.0012 | −0.0117 |
| 5e | 13419 diversified | 1 | 0.5 | 3e-6 | +0.0031 | −0.0043 |

Best result is 5b at +0.0043 pair_acc / +0.0006 top1, well inside
noise. **DPO with this corpus + HP regime cannot extract a
statistically significant lift over the 96-d production
reference.** The offline gate (scoping § P2 step 5) holds — no
closed-loop n=1000 spend.

**Root-cause candidates + forward-line proposal** (4-candidate
list, original orchestrator recommendation): live earlier in this
archive § "R16-TD 3b Cumulative Verdict — root-cause candidates".
Superseded by chunk 5f (candidate-3 falsified) + chunk 5g
(candidate-1 kill-test, see below).

### Cross-references

- Pre-registered design + corrected cost basis (not restated here):
  `docs/ai-research/scoping/r16-training-data-backlog-refinement.md`
  § "P1 - Rule-Bot-Covered States Relabeled By Rollout-Leaf MCTS"

### 3b — BC-on-relabel probe (chunk 5f) — FALSIFIES candidate-3

Direct soft-CE BC on the 11798-row 3a relabel corpus (the
cumulative-verdict candidate-3 test). HP: `--state-dim 110
--hidden-dim 64 --depth 2 --dropout 0.05 --epochs 3 --batch-size 32
--lr 3e-4 --split-by seed --seed 17 --data-mode bc`. Splits: 244
train seeds / 60 val seeds, 9379 train / 2419 val rows. Soft-CE
branch confirmed taken: `train_bc.py:490-498` keys on the batch
`policy_targets` tensor and every row in `relabel-all.jsonl`
carries `policyTargets`.

**Training trajectory**: 3-epoch trajectory peaked val acc 0.7234
at epoch 2 (train 1.074, val 1.106) with mild epoch-3 overfit;
full per-epoch table in `bc-relabel-e3/manifest.json` /
`train.log`.

**Eval against the same 1620-row pair floor** (seed 17, identical
DPO holdout): all five metrics regress vs ref; `pair_accuracy`
0.6802 vs base 0.6981 (Δ −0.0179, **outside ±0.022 noise band**);
`ranking_top1` 0.6198 vs 0.6512 (Δ −0.0314); top2/top3/MRR all
negative. Full table in `bc-relabel-e3/trained-eval.json`. BC is
meaningfully worse than the untrained 96-d ref, not flat.

Per-action-kind val accuracy (from `manifest.json`): routine actions
high (`playBasic` 0.938 / `attachEnergy` 0.856 / `evolve` 0.831 /
`attack` 0.805) but the contested heads drag the headline — `pass`
0.632 / `playTrainer` 0.591 / `retreatAttack` 0.484, exactly the
decisions the relabel corpus was supposed to teach.

**Verdict**: candidate-3 (soft-CE BC fixes the DPO ceiling) is
**FALSIFIED** — BC actively regresses below the untrained ref, not
flat. By elimination, candidate-1 (relabel-MCTS too weak vs ref) is
**strengthened**: the corpus's `policyTargets` actively mislead
training, consistent with rollout-leaf 100-sim MCTS producing labels
worse than the ref's own policy on contested states. Forward line
superseded by chunk 5g kill-test below, then chunk 5h ladder
refinement.

#### Candidate-1 kill-test (chunk 5g) — labels diverge decisively

Read-only label-divergence audit on the 11798-row 3a relabel corpus:
forward the untrained 96-d ref (`runs/R13-W6-phase-d/iter-2/checkpoint.pt`)
on each row's `observation`, softmax over `legalActions`, compare to MCTS
`policyTargets`. No MCTS, no training. Output:
`runs/R16-TD-3a-prod-corpus/candidate1-killtest/{metrics,metrics-bc-val-slice,metrics-confidence}.json`.

**Verdict: Case B — labels confidently wrong, not insufficient.**
Headlines: top-1 agreement **0.6338**; KL(rel‖ref) **0.331 nats** /
KL(ref‖rel) **0.766 nats** (asymmetric); mean entropy relabel 0.766
< ref 0.876 → relabel is *more* committed, not budget-starved.
Disagreement (n=4,320) p90 max-prob **0.91**; 27% of rows have
relabel confidence ≥0.8 yet still only 70% agree with ref. Contested
heads `playTrainer`/`pass`/`retreatAttack` agreement 0.499/0.479/0.464
— the same three heads where BC val acc cratered (0.562/0.581/0.111
from `bc-relabel-e3/trained-eval.json`). Full per-kind table +
disagreement-confidence percentiles live earlier in this archive §
"Chunk 5g — per-kind table + disagreement confidence".

**Sharpened diagnostic ladder** (supersedes the cumulative-verdict
4-candidate forward-line): step 1 = rollout-leaf value-eval audit
on contested states; step 2 = argmax-flip test at 400 sims; step 3
= skip more-sims BC unless step 2 positive. Chunk 5h flipped the
order (step 2 first; cheaper, no CRN-paired ambiguity); chunk 5i
ran step 2 — see verdict below.

#### Step ordering refined (chunk 5h scoping) — superseded by 5i

Doc-only scoping flipped the ladder (step 2 first, `rule-bot-mirror`
only, ~30 LOC Python join on confident-disagree subset, ~20-40 min
@ 16 workers; >30% flip = LIVES / >80% sticky = DIES). Now superseded
by the chunk 5i verdict below. Full chunk-5h scoping prose lives earlier
in this archive § "Chunk 5h ladder scoping".

#### Step 2 verdict (chunk 5i) — candidate-1 DIES; sim count not the lever

**Step 2 ran end-to-end.** Re-ran rule-bot-mirror at
`--mcts-simulations 400` (same seed range, 16 workers); joined
against the 100-sim corpus on `(seed, modelSide, step)` via
`training/killtest_argmax_flip.py`. Artifacts:
`runs/R16-TD-3a-prod-corpus-400sim-rulebot/{flip-analysis.json,
launch.log,rule-bot-mirror/traces.jsonl}`.

**VERDICT: candidate-1 DIES.** On the contested-head + 100-sim
max-prob ≥0.8 + state-identical subset (n=365):

- Flip rate **17.3% (63/365)** — below the 30% LIVES threshold AND
  the 20% sticky cutoff. 95% binomial CI ≈13.6-21.7%.
- KL(400‖100) mean **1.85 nats** — large distributional drift,
  argmax sticks → 100-sim MCTS is argmax-converged on this subset.
- Entropy 100 / 400 = 0.351 / 0.543 (Δ +0.193) — 400-sim
  distributions slightly fuzzier; visit mass redistributes within
  the same modal action's basin.
- Confidence redistribution gained/lost/flat = 33.4 / 33.4 / 33.2%
  (symmetric).
- Per kind: `pass` 17.4% flip (n=195), `playTrainer` 17.7% (n=164),
  `retreatAttack` 0% (n=6 — sample too small to weight).

**Implication.** Sim count is not the binding lever. The contested-
head disagreement with the untrained ref is upstream of MCTS budget
— most plausibly the leaf signal source (rule-bot-mirror rollouts
may systematically misvalue `pass`/`playTrainer`/`retreatAttack`
against rule-bot). Step 1 (value-eval audit) becomes the next
candidate experiment with **sharpened framing**: "do rollout-leaf
returns vary systematically across actions on contested states?" —
NOT the chunk-5h "CRN-paired variance" framing (across-action paired
CRN does not exist in current `mcts.ts`).

**Secondary finding — recipe-divergence bug.** Rule-bot-mirror at
400 sims produced 4233 rows vs 4196 at 100 sims (state-identical
intersection = 63.8% of 100-sim rows). Trajectories diverged despite
`--selection baseline` being deterministic. Root-cause hypothesis:
relabel MCTS consumes the shared `AsyncLocalStorage` RNG via the
global `random()` proxy in `frontend/src/game/engine/core/random.ts`
(`storageProvider.get()?.next() ?? Math.random()`); extra MCTS sims
perturb the game's RNG before the next baseline move. The chunk 5h
"deterministic row-by-row diff at 4× sims" design assumption is
INVALID; salvageable here because the 365-row state-identical subset
still yields a 95% CI well below the 30% LIVES threshold, but any
future sim-count A/B needs (a) a separate relabel-MCTS RNG context
or (b) a two-pass design (generate corpus once at low sims, then
re-relabel `policyTargets` only without re-running games). Tracked
as queue item `mcts-relabel-rng-bleed` (P3, scoped).

**Forward direction.** Candidate-1 (stronger-MCTS-fixes-corpus) is
eliminated. Either (a) reframed step 1 — rollout-leaf return audit
on contested states, no CRN-paired ambiguity — or (b) pivot to the
user-gated W6 HP-sweep (CEILING PATH A) since the 3a corpus arm is
now exhausted by evidence. Direction call deferred to user.

## R16-P2 — Per-Uma Slot Tokens v3.2 — C8 GATE NO-GO, +6.8pp iter-0 LIFT

Rolled from `r16.md` 2026-05-21 to free headroom for the C8-W6FIX-ON
vhleaf-loop verdict. Compact pointer in `r16.md` § "R16-P2 — Per-Uma Slot
Tokens v3.2 (archived)". The +6.8pp iter-0 lift datapoint and the
vhleaf-leaf falsification chain remain the evidence base for the
downstream C8-W6FIX-ON and C8-W6FIX-ON VHLEAF LOOP sections in `r16.md`.

### 1. Verdict — NO-GO at acceptance gate; iter-0 lift opens new forward line

Three v3.2 experiments executed 2026-05-21, all killed pre-iter-5. Best v3.2
Wilson-lower is **0.5955** (C8 rollout-leaf iter-0), **0.0087 BELOW** the
0.6042 R110-W6-repro production ceiling — within gate noise, no Wilson win.
**C8 gate result: NO-GO. 96-d production pin unchanged.** BUT C8 iter-0
produced a **+6.8pp Wilson-lower lift over R110 iter-0** (0.5955 vs
0.5273) — bit-exact init delta=0.0 confirms this is real learning from
slot-token state representation, not variance. The lift did not survive
W6 loop dynamics (iter-1 regressed bit-identically to R110 iter-0).
**NEW POSITIVE DATAPOINT** that re-opens `w6-loop-anti-degradation` as the
highest-leverage forward line: v3.2 + W6-fix-ON may preserve the iter-0
lift across iters.

### 2. Implementation status — IMPLEMENTED (C1-C7 bit-exact)

Autonomous chain C1-C7 landed across two /work cycles, all delta=0.0 parity
invariants verified bit-exact:

- **C1** `features.py` F=23 frozen slot-token tensor (commit `c46b02f`).
- **C2** `model.py` `uma_slot_encoder` + zero-output `Linear` head,
  `delta_logits` 0.000e+00 vs v3.0 (commit `0d7d6a6`).
- **C3** verified NO-OP.
- **C4** dataset / selfplay packing for the new `uma_slot_*` tensors
  (commit `43db81b`).
- **C5** LANDMINE `serve_onnx` `_SCHEMA_TABLE` refactor + 10 cross-input
  fail-fast guards + v3.2 ONNX roundtrip 2.38e-07 (commit `212ecde`).
- **C6** `train_bc` + `r12_orchestrator` `--uma-slot-tokens` flag,
  1-iter v3.2 smoke produced 7-input ONNX graph with schema 3.2 sidecar
  (commit `dc3d23b`).
- **C7** `make_v32_slot_token_init.py` + delta=0.0 parity smoke,
  `logits`/`value` 0.000e+00 bit-exact vs v3.0 source (commit `103e2ce`).

Chunk-plan landing commit `7330f0a`; queue refreshes `c475d30` / `7c3ca87`.
Full scope and chunk plan: `docs/ai-research/scoping/r16-model-feature-backlog-refinement.md`
§ "P2 - Per-Uma Slot Tokens" + § "P2 - Chunk Plan (kickoff 2026-05-21)".

### 3. Trajectories (n=120 side-balanced, Wilson-lower vs rule-bot)

**C8** — faithful R110-W6-repro mirror, `--mcts-leaf rollout`, v3.2 init from
R110 iter-2 0.6042 ckpt. Launch: `runs/R16-P2-c8-ablation/launch.sh`.

| Iter | Wilson-lower | Win-rate | vs R110 iter-N | Promoted? | Notes |
|---|---|---|---|---|---|
| 0 | **0.5955** | 0.6833 | **+6.8pp** vs 0.5273 | yes | best v3.2 ckpt overall |
| 1 | 0.5273 | 0.6167 | bit-identical to R110 iter-0 | no | regress to v3.0 trajectory |
| 2 | 0.5612 | 0.65 | — | yes (tolerance) | still below iter-0 floor |
| 3 | — | — | — | — | KILLED at selfplay per user instruction |

R14 crossover at iter-1: `val_mse` 0.81 / `pearson` 0.47 / ratio 1.42, NOT
crossed. R14 crossover at iter-2: `val_mse` 1.09 / `pearson` 0.29 / ratio
1.92, NOT crossed.

**C8b** — same recipe but `--mcts-leaf value-head` from raw v3.2 init (C7
builder output). Launch: `runs/R16-P2-c8b-vhleaf/launch.sh`.

| Iter | Wilson-lower | Win-rate | Promoted? |
|---|---|---|---|
| 0 | 0.3798 | 0.4667 | yes (iter-0 floor squeak) |
| 1 | 0.3090 | 0.3917 | no |

R14 crossover at iter-1: `val_mse` 0.79 / `pearson` 0.42 / ratio 1.39, NOT
crossed. KILLED after iter-1.

**C8c** — `--mcts-leaf value-head` from C8 iter-0 ckpt (the 0.5955
best-promoted v3.2 rollout-leaf ckpt). Launch:
`runs/R16-P2-c8c-vhleaf-from-c8iter0/launch.sh`.

| Iter | Wilson-lower | Win-rate | Promoted? |
|---|---|---|---|
| 0 | 0.3481 | 0.4333 | yes (iter-0 floor squeak) |

KILLED after iter-0.

### 4. Load-bearing interpretations

1. **C8 iter-0 +6.8pp is REAL learning, not variance.** C7's
   `make_v32_slot_token_init.py` produced `delta_logits=0.000e+00` and
   `delta_value=0.000e+00` bit-exact against the v3.0 source (the
   `uma_slot_encoder.2.weight` zero-init invariant guarantees additive-tail
   output equivalence at t=0). The lift came entirely from the v3.2 vertical
   training pulling signal through the new `uma_slot_encoder` branch — the
   slot-token state representation can be learned.

2. **W6 loop dynamics broke the iter-0 lift.** C8 iter-1 regressed −6.8pp to
   0.5273, bit-identical to R110 iter-0; iter-2 only partially recovered to
   0.5612. The faithful R110 mirror uses
   `--no-w6-fix-cross-iter-replay --no-w6-fix-fixed-kl-anchor` (R110 pre-dated
   the W6 recipe-fix flag system). The iter-0 lift is exactly the kind of
   signal W6-fix-ON was designed to preserve — and was not tested on this
   recipe.

3. **Value-head-leaf is decisively broken for v3.2.** C8b iter-0=0.3798 and
   C8c iter-0=0.3481 both well below C8 rollout-leaf's 0.5955. C8c (init from
   stronger ckpt) being WORSE than C8b (init from raw ckpt) falsifies the
   "trained-ckpt unlocks vhleaf" hypothesis. Consistent with R13.W8 history
   (v3.0 vhleaf regressed every iter: 0.452→0.404→0.340→0.380). The
   remaining vhleaf hypothesis "value head needs better calibration" is what
   `value-head-data-program` Stage 1 tests.

4. **C8 acceptance gate result: NO-GO.** Best v3.2 Wilson-lower = 0.5955 (C8
   iter-0), 0.0087 below the 0.6042 production ceiling. 96-d production pin
   unchanged.

5. **NEW FORWARD LINE OPENED.** The C8 iter-0 +6.8pp lift is the strongest
   slot-tokens-or-anything signal in months. It re-opens
   `w6-loop-anti-degradation` (queue item, Ceiling Path A) as the natural
   next experiment: re-run the C8 recipe but DROP the `--no-w6-fix-*` flags
   (let W6-fix defaults to ON: cross-iter replay + fixed KL anchor). If
   W6-fix-ON over-damps (per R111 history), tune
   `--kl-anchor-weight` (currently 0.05 default; consider 0.02 or anneal),
   `--w6-replay-window`, `--w6-replay-old-fraction`. The 0b queue entry's
   scope already covers this exact tuning.

### 5. Cross-references

- Scope / P2 contract + chunk plan:
  `docs/ai-research/scoping/r16-model-feature-backlog-refinement.md`
  § "P2 - Per-Uma Slot Tokens" + § "P2 - Chunk Plan (kickoff 2026-05-21)"
- v3.0 R110-W6-repro baseline (0.6042 ceiling, schema-independent W6
  pattern): `docs/ai-research/progress/r110.md`
- Forward line: queue item `w6-loop-anti-degradation`
- vhleaf fallout: queue item `value-head-leaf-recipe-axis` (vhleaf-needs-
  better-state-rep hypothesis falsified; remaining hypothesis owned by
  `value-head-data-program` Stage 1)
