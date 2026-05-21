# Rust Engine Port — Scoping

- **Date:** 2026-05-21
- **Status:** scoping (Phase 0 go/no-go gate is the actionable next step;
  Phase 1 not started)
- **One-liner:** Port the TS game engine + heuristic opponent to Rust to
  unblock further selfplay/gate throughput once the JS-side ceiling
  (worker fan-out + hash-carry + work-stealing, ~2.85× over the
  4-worker baseline) is exhausted. Behaviorally a no-op; strictly a
  performance lever for the simulator hot loop.
- **Non-goal:** rewriting the UI, the backend HTTP surface, the Python
  training stack, or the `serve_onnx` path. The TS engine stays
  authoritative for the frontend day-1; Rust replaces only the
  simulator binaries that the Python orchestrators already shell out
  to via `npm run sim:*`.

## 1. Why now

R110-W6-repro per-iter wall-clock is **~99% pure single-thread CPU
MCTS sim**: selfplay ~603s (~48%) + mcts-gate ~694s (~52%); distill
~8.5s (~0.6%); GPU correctly idle (`docs/ai-research/progress/r110.md`
§4b lines 180-214). The R12 selfplay/gate throughput sprint has now
landed everything cheap on the JS side:

- Worker fan-out (`r12_orchestrator.py` `--workers` default 1→24),
- Rollout/collapse fingerprint hash-carry,
- HTTP keep-alive to `serve_onnx`,
- Work-stealing dispatch.

Measured speedup in a full R111 production loop: **2.11× wall-clock**,
trajectory-neutral; the honest balanced max is ~2.6-2.85×
(`docs/ai-research/scoping/r12-selfplay-gate-throughput.md` § "Ranked
speedup plan" item #5). Item #4 (CRN/value-head leaf) is deferred by
research-recipe decision, not by throughput potential.

Further sim-side wall-clock therefore has to come from **leaving the
TS allocator/JSON hot path**, not from another JS-level refactor.
Throughput-probe microbenchmark numbers from
`runs/throughput-probe/probe-default.json` (single-core,
`legalActionCount=1`):

- `cloneNsPerCall: 22007` — `structuredClone` of `GameState`
  (`frontend/src/game/engine/core/stateClone.ts`).
- `fingerprintNsPerCall: 8034` — `JSON.stringify` of a compact state
  view (`backend/src/sim/stateFingerprint.ts:10-30`).
- `enumerateNsPerCall: 2559` — legal-action enumeration
  (`frontend/src/game/engine/ai-policy/actions.ts`).

Single-core planner throughput in the same probe is **22.4 dec/s
against a 200 dec/s target** (probe-default.json
`summary.notes`). That gap is the same one R12 already drained at the
worker level; remaining single-core gains require a different
language.

## 2. Scope

### In (Phase 1 ports)

`frontend/src/game/engine/` — 41 `.ts` files, 5,692 LOC measured
2026-05-21 (audit via `find ... | xargs wc -l`). Concretely:

- `core/` — RNG (`random.ts`, 91 LOC; port-stable, see §4), state
  clone (`stateClone.ts`, currently `structuredClone`), catalog,
  constants, labels, log, playTypes, umamusume.
- `flow/` — pure rules: `turn.ts` (139), `playRules.ts` (182),
  `setup.ts` (141), `board.ts` (129), `trainers.ts` (231),
  `combat.ts` (532), `eligibility.ts`, `energy.ts`, `evolution.ts`,
  `retreat.ts`, `specialConditions.ts`, `abilityRules.ts`.
- `flow/ai/` — the heuristic opponent (2,691 LOC across 14 files
  incl. `combatPlanner.ts`, `attachUtils.ts`, `abilityUtils.ts`,
  `turnPlan.ts`, `trainerUtils.ts`, `midLevel.ts`,
  `opponentHeuristics.ts`, `deckInference.ts`, `combatUtils.ts`,
  `core.ts`, `publicInfo.ts`, `telemetry.ts`, `energyAwareness.ts`,
  `types.ts`). **Included in Phase 1** — initial framing said defer;
  that was wrong (rationale: §2 "Why heuristic opponent is in scope"
  below).
- `ai-policy/` — `actions.ts` (legal-action enumeration, 666 LOC),
  `observation.ts`, `phase.ts`, `mctsClient.ts`, `types.ts`.
- `backend/src/sim/stateFingerprint.ts` — replaced by a structural
  hash over the packed state buffer (xxhash3 or Blake3), not a JSON
  serialization.

### Out

- The frontend UI continues to import the TS engine directly. Rust
  does not replace `frontend/src/game/engine/` at the import level
  until Phase 2 (optional WASM) is explicitly opened.
- The live `backend/src/server.ts` `/predict`/`/game` path. Phase 1
  ships only the Rust replacements for `sim:eval-gate`,
  `sim:mcts-selfplay`, `sim:export-training`,
  `sim:throughput-probe`. NAPI integration into the live backend is
  Phase 2.
- The Python training stack (`training/`), orchestrators
  (`r12_orchestrator.py`, `dagger_orchestrator.py`,
  `ppo_orchestrator.py`), `serve_onnx`, `export_onnx`. Boundary is
  the CLI surface, not the Python code.

### Why heuristic opponent is in scope

The initial scoping framing deferred `flow/ai/*`. That is wrong for
three reasons:

1. **MCTS rollouts simulate opponent turns.** If the Rust binary
   RPC'd back to TS for every opponent decision in a rollout, the
   language boundary would sit in the tightest hot loop and erase
   the speedup. The whole point is to keep the simulator in one
   process.
2. **Determinism gating wants a single language.** The R12 and R15
   determinism gates byte-diff JSONL (`training/
   r12_throughput_determinism_gate.py:85`,
   `r12_workstealing_determinism_gate.py:85,115`). A split-language
   binary multiplies the float/iteration-order/RNG audit surface.
3. **Rust sum types fit the heuristic better than TS.** The
   ability/attach/combat-planner logic is exactly the "fiddly
   discriminated-union" code Rust handles naturally.

If any of `abilityUtils.ts`, `attachUtils.ts`, or `combatPlanner.ts`
proves intractable during Phase 1, the **last-resort fallback** is TS
RPC for that subsystem only — explicitly accepted as a Phase 1
escape hatch, not a default.

## 3. Phased plan

### Phase 0 — Golden-trace harness (1–2 days, go/no-go gate)

Pure TS work; useful regardless of whether Phase 1 lands.

- Record `(seed, action-sequence, terminal-state-hash)` tuples from
  the current TS sim across ~500 seeds at the R110 production
  config (rollout-leaf MCTS, 100 sims, K=3, rollout-steps 200,
  prior=policy, mctsCollapseMaxSteps=64).
- Serialize to a stable JSONL format under
  `runs/rust-port-golden-traces/`.
- Provide a TS replay tool that re-runs each seed and asserts
  byte-equality of the recorded trace.

**Kill signal.** If the harness reveals that the *TS sim itself*
has nondeterminism (Map iteration order divergence across Node
versions, Set ordering, float drift, RNG re-seeding via worker
state), Phase 1 **pauses** until that's repaired. That is on its
own a useful catch — the R12 and R15 byte-diff gates only cover
their own seed ranges and one Node version.

### Phase 1 — Full engine port (4–6 engineer-weeks)

Realistic estimate, not 2–4. Heuristic opponent inclusion roughly
doubles the surface vs. the original "rules + actions only" framing.

Deliverables:

- Port `core/`, `flow/*` (including `flow/ai/*`), and
  `ai-policy/actions.ts` to Rust.
- Replace `structuredClone` with a packed fixed-size state buffer
  (zero-allocation clone via `Copy` or arena memcpy).
- Replace `JSON.stringify` fingerprinting with structural hashing
  (xxhash3 or Blake3) over the packed buffer.
- Pin RNG sequence to match TS `core/random.ts` exactly (the TS
  RNG is a 91-LOC seeded PRNG; reproduce its state transitions
  bit-for-bit and validate against per-call golden traces).
- Expose Rust CLI binaries matching the four
  `backend/package.json` `sim:*` scripts flag-for-flag:
  `sim:eval-gate`, `sim:mcts-selfplay`, `sim:export-training`,
  `sim:throughput-probe`. Same stdout JSON shapes,
  same `--manifest-out` semantics.
- Drop-in mechanism for orchestrators: the Python side already
  shells out via `subprocess`; flipping in the Rust binary is a
  single path swap.
- Gate the entire port on Phase 0 golden traces — bit-identical
  required at all 500 seeds.

### Phase 2 — Optional, post-Phase 1

- NAPI binding for the live `backend/src/server.ts` `/predict` and
  `/game` paths.
- WASM bundle for the frontend UI **only if rules churn rate
  makes the dual-source-of-truth painful** (see §5 risk #5).

## 4. Conformance strategy

Determinism reproduction across languages is the highest risk
(§5 risk #1). Four axes need explicit pinning:

1. **Golden traces** (Phase 0 above). The single load-bearing gate.
2. **RNG sequence pinning.** `core/random.ts` is a small seeded PRNG
   (`shuffle`, `rollEnergyFromPool` exports;
   `frontend/src/game/engine/index.ts:8`). The Rust port reproduces
   its per-call output stream bit-for-bit; unit-test per-seed first
   1,000 outputs vs. TS reference vectors.
3. **Map/Set iteration order.** JS spec preserves insertion order;
   Rust `HashMap` doesn't. The port must use `BTreeMap` /
   `IndexMap` or explicitly sort keys at every iteration site. Any
   place TS iterates an object or `Map` to build a hash, log line,
   or action list is a candidate divergence point — audit during
   Phase 0 with the golden-trace harness as the catcher.
4. **Float behavior.** Audit rules math for floats during Phase 0
   (damage calc, probability rolls, retreat-cost reductions). If any
   transcendental is used, IEEE 754 conformance + library choice
   (`f64::powf` vs. JS `Math.pow`) is a divergence risk. Prefer
   integer math wherever the spec allows it.

The TS-side R12/R15 byte-diff gates
(`training/r12_throughput_determinism_gate.py:85`,
`r12_workstealing_determinism_gate.py:85,115`) become the same gate
applied across the language boundary — same JSONL byte-diff, same
selfplay corpus, same per-seed gate fingerprints.

## 5. Risk register

1. **Determinism reproduction across languages.** Highest risk.
   Mitigated by Phase 0 (golden traces are the kill signal *and*
   the day-by-day port checker). If Phase 0 reveals TS-side
   nondeterminism, the entire port pauses until it's fixed.
2. **Float behavior divergence.** IEEE 754 base ops are portable;
   transcendentals + intermediate-precision differences (x87 vs.
   SSE2 on edge platforms) are not. Audit float use during
   Phase 0; replace with integer math where the spec permits.
3. **Map/Set iteration order.** Covered in §4 #3; surfaces as
   silent hash divergence, not a loud error. Golden traces catch
   it.
4. **Scope creep into `flow/ai/*`.** Accepted with explicit
   escape hatch: if `abilityUtils.ts`, `attachUtils.ts`, or
   `combatPlanner.ts` resists porting, fall back to TS RPC for
   that subsystem only. Last-resort, not a default; tracked
   explicitly so it can't silently propagate.
5. **Rules churn during the port window.** A 4–6 week port
   against an actively-mutating engine forces parity to be a
   moving target. Mitigation: before committing to Phase 1,
   `git log --oneline frontend/src/game/engine/` for the last 90
   days and characterize churn cadence. If rules edits land
   weekly, scope a freeze window or accept a deliberate rebase
   cadence with the golden-trace harness re-run on each rebase.

## 6. Payoff estimate

- **Current production wall-clock per loop iter** (post-R12
  speedups, 5-iter prod, 32-core box):
  `r110.md` §4c (lines 256-258) reports stage-sum 3,088 s ≈
  51.5 min per iter — equivalently the "51.5 min/run" number for
  the speedup-relevant stages.
- **Single-worker prod gate baseline** for per-seed cost:
  `runs/R15-single-worker-determinism/result.json`
  `per_seed_sec: 55.25`.
- **Realistic per-core speedup on Rust port:** 15–40×.
  Justification: the JS hot path is allocator-bound
  (`structuredClone` per MCTS step is 22 μs single-call;
  `JSON.stringify` fingerprint is 8 μs single-call). A packed
  fixed-size state buffer + structural hash collapses both costs
  to sub-microsecond. The legal-action enumerator (2.6 μs) shrinks
  more modestly. The exact multiplier is benchmark-dependent and
  must be re-measured against Phase 0 corpus.
- **Projected per-iter wall:** ~3–5 min vs. current 51.5 min.
- **Strategic payoff:** at ~3–5 min per iter, sweeps that are
  currently uneconomic (multi-axis HP doses, larger-n confirmation
  gates, broader source-recipe matrices for the 3a corpus) move
  inside the iteration budget. The 15–40× projection is
  optimistic — even a 5–10× landing is worth Phase 0 + Phase 1.

## 7. Open questions

- **RNG portability detail.** Is `core/random.ts` a known PRNG
  family (mulberry32, xorshift, etc.) or a custom mix? Phase 0
  should answer before Phase 1 RNG implementation; the bit-exact
  reproduction strategy depends on this.
- **Card catalog ingestion.** `core/catalog.ts` and the
  `shared/src/types.ts` card data — Rust port re-reads the same
  JSON or compiles to a const table? Lean toward a build-time
  codegen step from the existing JSON to avoid a dual source of
  truth.
- **`flow/ai/*` deck inference.** `deckInference.ts` does
  partial-information opponent modeling; verify during Phase 0
  whether it consumes RNG and whether its iteration order over
  hand candidates is deterministic.
- **Worker model.** The Python orchestrator already fan-outs at
  the process level (24 workers). Does the Rust binary stay
  single-process-per-task to slot into the existing
  work-stealing dispatcher, or expose internal parallelism? The
  former preserves the bit-identical work-stealing determinism
  gate (`r12_workstealing_determinism_gate.py:85,115`); the
  latter requires re-clearing it.

## 8. Cross-references

- **JS-side ceiling and what's already landed:**
  `docs/ai-research/scoping/r12-selfplay-gate-throughput.md`
  § "Ranked speedup plan".
- **Wall-clock breakdown supporting "99% sim":**
  `docs/ai-research/progress/r110.md` § 4b (lines 180-214).
- **Speedup-confirmation numbers (2.11×, 51.5 min/iter):**
  `docs/ai-research/progress/r110.md` § 4c (lines 256-271).
- **Single-worker gate baseline (per_seed_sec 55.25):**
  `runs/R15-single-worker-determinism/result.json`.
- **Microbenchmark evidence for clone/fingerprint/enumerate
  costs:** `runs/throughput-probe/probe-default.json`
  `summary.micro`.
- **Determinism gating templates the Rust binary must satisfy:**
  `training/r12_throughput_determinism_gate.py:85`,
  `training/r12_workstealing_determinism_gate.py:85,115`.
- **Live queue item:** `docs/ai-agent-state/queue.json`
  (`rust-engine-port`).
