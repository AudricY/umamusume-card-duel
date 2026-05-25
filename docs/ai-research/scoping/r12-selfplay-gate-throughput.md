# Scoping: r12 selfplay + gate throughput

Status: DONE — Deliverables 1, 2, 3, 5 all landed ON, all proven
bit-identical by the determinism gates (below). Item #4 (CRN /
value-head leaf) graduated into its own queue item
`value-head-leaf-recipe-axis` (P4, deferred + user-gated, R111 axis)
and its data-side predecessor `value-head-data-program` (P1
autonomous, R14-crossover gate). Item #6 (Rust engine port) LANDED
end-to-end (Phase 0/1/2 merged as commit `4e86ea7`); follow-ups
tracked at queue items `rust-port-orchestrator-wiring` /
`rust-port-v32-schema-verification` / `rust-port-backend-napi-consumer`.
Canonical: `docs/ai-agent-state/queue.json`.

## Context

`runs/R110-W6-repro` (5 iters, 4 workers, 32-core box) per-iter
wall-clock: selfplay ~603s (~48%), mcts-gate ~694s (~52%), distill
~8.5s (~0.6%). Selfplay + gate are ~99% of wall time and are pure
single-threaded CPU MCTS sim. GPU correctly idle. Mechanism +
per-stage numbers + trajectory-neutrality proof live in
`docs/ai-research/progress/r110.md` §4b (one canonical home).

## Ranked speedup plan

1. **DONE — raise `--workers` default 1→24** (`r12_orchestrator.py`).
   Pure distribution knob, trajectory-neutral (proof in r110.md §4b).
   Expected ~4-6× wall-clock reduction on the 32-core box from the
   4-worker baseline (selfplay/gate scale near-linearly with cores
   until the serve_onnx HTTP path or memory bandwidth saturates).

2. **DONE — rollout/collapse fingerprint hot-loop refactor.** The
   leaf-rollout and opponent-collapse loops recomputed the full-state
   JSON fingerprint (`stateHash`) TWICE per step — pre-advance `before`
   + post-advance no-progress check. The post-advance hash of step N
   is, by construction, exactly the pre-advance `before` of step N+1
   (same `GameState` object content). Carry it forward instead of
   recomputing: identical `stateHash` string comparisons for the
   no-progress break, ~half the `JSON.stringify` cost. No per-step
   deep clone was removed (each rollout/collapse already owned ONE
   `cloneGame` at entry — verified, advance fns mutate the owned copy,
   no aliasing escapes to MCTS node state). Gated `MCTS_HASH_CARRY_ENABLED`
   (`mcts.ts`), default ON, `UMA_MCTS_HASH_CARRY=0` reverts for an
   exact A/B (mirrors the W6 recipe-fix flag pattern).
   **GATE SATISFIED.** `training/r12_throughput_determinism_gate.py`
   exists and PASSED at the R110 production config — see determinism
   gate result below.

3. **DONE — HTTP keep-alive between workers and serve_onnx.** Bare
   per-call `fetch()` on the `/predict` path (200k+ calls/run) replaced
   with a shared keep-alive `node:http`/`node:https` Agent
   (`backend/src/sim/keepAliveClient.ts`, `postJsonKeepAlive`). Pure
   transport reuse: byte-identical request body (pre-serialized string
   passed through), identical response parsing/ordering/error
   semantics, only socket reuse differs. Gated `MCTS_KEEPALIVE_ENABLED`,
   default ON, `UMA_MCTS_KEEPALIVE=0` reverts to bare fetch.

5. **DONE — work-stealing dispatch (selfplay + gate).** Ranked
   successor to #1. Static contiguous chunking
   (`sliceSize=ceil(N/workers)` then `.slice(w*size,(w+1)*size)`,
   `mctsSelfPlay.ts` `partitionSeeds`, `evalGate.ts` `partitionTasks`)
   had two defects: (a) selfplay 60 games / 24 workers →
   `ceil(60/24)=3` → only 20 slices, **workers 20-23 dark the entire
   selfplay stage**; (b) per-game length variance is ~100× (gate games
   1.6s-168s) so static assignment clusters long games onto unlucky
   workers → 2-3× per-worker wall spread. **This refutes the
   "make games a multiple of workers" framing**: the gate is already
   exactly 5 games/worker and STILL ~20% imbalanced — equal slice
   counts do not fix variance-driven skew; only dynamic assignment
   does. Fix: shared task-index queue, each worker seeded with 2 tasks
   then refilled one-per-completion; workers spun = `min(N,workers)`
   (kills the dark-core bug). Result aggregation is by task/seed slot
   (selfplay concatenates per-seed-index shards in ascending order;
   gate stores into a slot array) — **never completion order** — so
   output is byte-identical to static. Gated
   `MCTS_WORK_STEALING_ENABLED` (`workStealing.ts`), default ON,
   `UMA_MCTS_WORK_STEALING=0` reverts to static chunking for an exact
   A/B. **GATE SATISFIED** — see work-stealing determinism gate below.
   Expected wall: the static path delivered ~2.11×; balanced dispatch
   lifts the ceiling to ~2.6-2.85×. It is NOT ~6×: ≈half the
   2.11×-vs-idealized gap is the 4-worker-baseline re-anchoring
   artifact + per-game rollout-cost floor (never recoverable), so
   ~2.9× is the honest balanced max, not 6×. serve_onnx-sharding is
   explicitly NOT worth it and out of scope (stdlib threaded server,
   ~7% predict share at batch=1 — see bottleneck note).

4. **DEFERRED to R111 — CRN / value-head leaf change.** Switching the
   leaf evaluator or CRN sample count changes the learning targets and
   is a research-recipe decision, NOT a unilateral throughput change.
   Held OUT of scope so the R111 W6 recipe-fix A/B stays comparable to
   R110. Owned by the research backlog, not this scoping doc.

6. **NEXT-TIER — Rust port of the TS engine + heuristic opponent.**
   Successor lever once #1-#3+#5 are exhausted (the JS-side ceiling is
   ~2.6-2.85× over the 4-worker baseline; further sim wall-clock must
   leave the TS allocator/JSON hot path). Phase 0 golden-trace harness
   is the go/no-go gate and the actionable next step. Full plan, scope
   delineation (engine surface in / out, why `flow/ai/*` is in scope),
   conformance strategy (RNG pinning, map-iteration order, float
   audit), risk register, and payoff numbers (15-40× per-core
   projection): `docs/ai-research/scoping/rust-engine-port-plan.md`.

## Determinism gate result

`training/r12_throughput_determinism_gate.py --games 6` (R110 prod
config: rollout leaf, sims 100, CRN 3, rollout-steps 200, collapse 64,
prior policy; checkpoint `runs/R13-W6-phase-d/iter-2/policy.onnx`,
seed-start 9000, workers 1): **PASS** — 12 seed-side keys, symmetric
key set, zero mismatches. FLAG-ON (#2+#3 defaults) is bit-identical to
FLAG-OFF (`UMA_MCTS_HASH_CARRY=0 UMA_MCTS_KEEPALIVE=0`): identical
trajectories + terminal outcomes + gate win/loss. Both ship ON.
Artifacts: `runs/R12-throughput-determinism-gate/result.json`. `cd
backend && npx tsc --noEmit` clean.

## Work-stealing determinism gate result

`training/r12_workstealing_determinism_gate.py --games 8 --workers 3`
(R110 prod config; checkpoint `runs/R13-W6-phase-d/iter-2/policy.onnx`;
selfplay seedStart 12000, gate seedStart 9000): **PASS** — selfplay
canonical `selfplay.jsonl` byte-identical (sha256
`d63b89c7…`, 314 rows both) AND all 16 gate `(seed,modelSide)`
fingerprints identical, zero mismatches, symmetric key set, FLAG-ON
(work-stealing default) vs FLAG-OFF (`UMA_MCTS_WORK_STEALING=0`,
static chunking) at workers>1 (the only regime where dispatch
differs). NOTE: the first cut prefetched 2 tasks AND ran them
concurrently in one worker → NOT identical (RNG AsyncLocalStorage +
shared serve_onnx client interleave); fixed by buffering prefetched
tasks in an internal FIFO drained STRICTLY SEQUENTIALLY (one game in
flight per worker, matching the static per-worker for-loop) — load
balancing still comes from the orchestrator refilling on completion.
Re-ran: bit-identical. Trajectory-neutral → shipped ON. Artifacts:
`runs/R12-workstealing-determinism-gate/result.json`. `cd backend &&
npx tsc --noEmit` clean. Per-game RNG is seeded solely from the game
seed (`mctsSelfPlay.ts` `:selfplay`, `evaluateModelVsHeuristic.ts`
`:modelSide`) — no worker/order input — so dispatch order cannot
perturb trajectories; the gate confirms result-aggregation is
slot-ordered, not append-ordered.

## serve_onnx bottleneck note

`serve_onnx` is a stdlib `ThreadingHTTPServer` with one shared,
thread-safe ORT `InferenceSession` (`serve_onnx.py:35,60-71`). At 24
workers and ~7% predict share at batch=1 it is not expected to
saturate. A cheap `intra_op_num_threads`/`inter_op_num_threads` knob
exists (`serve_onnx.py:62-63`) if it does — flagged, not changed now
(would not affect trajectories but is out of the trusted-A/B scope).

## Slice 3d — Rust `sim-mcts-selfplay` worker pool (2026-05-25)

The Rust port of `mcts_selfplay.rs` shipped (Phase 1g, Slice 2 schema fix)
but kept the `--workers` flag as a no-op (`let _ = args.workers` at
`engine-rs/crates/sim-cli/src/bin/mcts_selfplay.rs:540`). The 50k-row
corpus was produced by 8 hand-launched processes as a workaround
(`docs/ai-agent-state/digests/2026-05-25.md:5`). Slice 3d ports the
`eval_gate.rs:540-620` work-stealing pool to `mcts_selfplay.rs`: a single
shared `OnceLock<InferenceSession>` reused by all workers (G5 lock-free
`UnsafeCell<Session>` contract, `engine-rs/crates/engine/src/inference/
mod.rs:186-200`), `std::thread::scope` + atomic task cursor, per-task
`GameRecord` slots collected after the scope so JSONL output stays
bit-identical to `--workers 1` regardless of completion order. Default
`--workers 0` ⇒ `available_parallelism()` to match `sim-eval-gate`.

### Parity smoke

`--sims 50 --seeds 6 --model-side both --prior uniform --leaf rollout
--record-rows` at workers ∈ {1, 4, 8, 16, 32}: all five outputs share
md5 `65fc72a1…` (the small-game probe used a coarser config; the large
run below uses the same md5 verification). Per-game RNG is seeded solely
from `(seed, side)` (`mcts_selfplay.rs:322, 353`) — dispatch order
cannot perturb trajectories; the gate confirms result-aggregation is
slot-ordered, not append-ordered. Matches the TS-side guarantee proven
in the work-stealing determinism gate above.

### Throughput sweep (sims=800, K=3, rollout_steps=200, prior=uniform, leaf=rollout, 40 games)

This is the "Critical pre-Tier-3 measurement" from
`mcts-selfplay-throughput-handoff.md`: re-run the baseline manifest with
`--prior uniform --leaf rollout` to isolate the rollout-step share of
wall (no ORT calls). The baseline anchor at `prior=policy` was 0.60 g/s
(`runs/qhead-v32-highsim-label-stability-probe/selfplay-s800.manifest.json`,
single process); the uniform-prior workers=1 number here re-anchors at
**0.636 g/s**, confirming the rollout-leaf path dominates wall.

| workers | wall (s) | g/s   | speedup | CPU%   |
|---------|----------|-------|---------|--------|
| 1       | 62.87    | 0.636 | 1.00×   | 99%    |
| 4       | 18.06    | 2.215 | 3.48×   | 381%   |
| 8       | 11.20    | 3.570 | 5.62×   | 707%   |
| 16      | 11.28    | 3.546 | 5.58×   | 1090%  |
| 32      | 11.16    | 3.585 | 5.64×   | 1524%  |

**Plateau at workers=8** despite CPU% continuing to climb through
workers=32. Plateau is consistent with per-game wall-clock variance
(longest game determines wall when worker count exceeds active games)
and/or allocator/cache contention — workers=16-32 burns ~2× CPU work for
the same wall as workers=8. Resolution belongs to flamegraph (Tier 2 in
the handoff doc). 5.6× is below the gate-path 7.11× ceiling at
workers=16 (G5 measurement, value-head leaf, ORT-dominated). Selfplay's
larger relative rollout share thins the ceiling.

### Slice 3e — drop redundant `score_candidate` clones (2026-05-25, `bfb4213`)

samply flamegraph at sims=800 workers=16 (`perf_event_paranoid=1` needed
for unprivileged sampling) revealed `combat_planner::score_candidate` →
`SideState::clone` as the dominant rollout-heuristic cost (95% of CPU in
`rollout_heuristic`, ~50% under `build_combat_candidates`). Three of the
four clones at the top of `score_candidate` are pure TS-port artifacts:
`base_state` is already `&GameState`, every use of
`before`/`acting_before`/`defending_before` is a read, and the penalty
helpers take `&GameState` borrows. Only `simulated` needs to be owned
(`perform_attack` mutates it in place). workers=16 went 3.546 → 5.357
g/s (+51%); the previous w=8 vs w=16 plateau collapsed.

### Slice 3f — gate AI telemetry payload assembly (2026-05-25, `bfc6514`)

`flow/ai/telemetry.rs` early-exits when `__UMA_AI_TELEMETRY__` is unset
(the default in selfplay), but callers built the full
`Map<String, Value>` payload before discovering the sink discards it:

- `emit_turn_goal_telemetry`: Map + 5 `json!` + `explain_ai_turn_goal`
  Vec per call (attach/trainer/ability per turn)
- combat-candidates: `candidates.clone()` + sort-by-score + top-3 Map
- trainer-bundle-scores: per-iteration Map alloc inside the score loop

Score computation for decision logic stays outside the gate; only the
observer-side payload assembly is skipped. workers=16: 5.357 → 6.141 g/s
(+15%).

### Slice 3g — `state_fingerprint` (u128) replaces hex `state_hash` in MCTS equality (2026-05-25, `adf38ea`)

Three MCTS sites compared two state hashes for equality
(`mcts/driver.rs:470` modeled-step stall, `:494` collapse loop, `:643`
rollout_heuristic loop). The TS-port used `state_hash -> String`
(`format!("{:032x}", …)` over a freshly-allocated `Vec<u8>`), allocating
two transient String + two Vec<u8> per rollout step.

Added `state_fingerprint(state: &GameState) -> u128` that `pack_into`s a
thread-local reusable buffer and returns the raw u128 fingerprint. Use
it at every equality site. `state_hash` is now a thin format wrapper
over `state_fingerprint`, so the one cache-key path
(`mcts/driver.rs:325` `"{hex}:precollapse"`) no longer double-allocates.
workers=1: 0.831 → 0.932 g/s (+12%); workers=16 +1% (near noise).

### Slice 3h — `get_all_umamusume` returns inline ArrayVec (2026-05-25, `00ba1fe`)

`get_all_umamusume` was called from ~25 sites across the heuristic AI,
combat, and policy code, allocating a fresh `Vec<&UmamusumeInstance>`
each time (11.2% inclusive in the post-3g flamegraph). Since
`MAX_UMA_IN_PLAY_PER_SIDE = MAX_BENCH + 1 = 4`, an
`ArrayVec<&UmamusumeInstance, 4>` fits on the stack with no heap
involvement. The call-site interfaces (`.iter()`, `.into_iter()`,
`.len()`, indexing, `.is_empty()`) carry over verbatim; only one caller
needed a type-annotation update. workers=1: 0.932 → 1.033 g/s (+11%);
workers=16: 6.199 → 7.438 g/s (+20%).

### Cumulative throughput vs baseline anchor (sims=800, 40 games, prior=uniform/leaf=rollout)

| Slice                              | w=1 g/s | w=16 g/s | Cumulative vs 0.60 anchor |
|------------------------------------|---------|----------|---------------------------|
| Pre-Tier-1 (8-process workaround)  | n/a     | n/a      | 1.0× (anchor)             |
| 3d worker pool                     | 0.636   | 3.546    | 5.9×                      |
| 3e score_candidate clones          | 0.743   | 5.357    | 8.9×                      |
| 3f telemetry gating                | 0.831   | 6.141    | 10.2×                     |
| 3g state_fingerprint               | 0.932   | 6.199    | 10.3×                     |
| 3h get_all_umamusume ArrayVec      | 1.033   | 7.438    | **12.4×**                 |

JSONL md5 `65fc72a1…` unchanged across every slice — all changes are
provably trajectory-neutral, no parity gate required.

### Follow-ups (not yet landed)

- Rip `training/r12_orchestrator._run_pool_selfplay` (`:812-962`)
  multi-process fanout once the binary `--workers` flag is the canonical
  parallel axis. The 8-process bash hack noted in the 2026-05-25 digest
  retires with it.
- Re-run with `prior=policy` (ORT in the loop) once an ONNX checkpoint
  is staged in this worktree, to confirm the ORT-path speedup matches
  the gate-side 7.11× at workers=16.
- Tier 2 algorithmic: subtree reuse across moves (`mcts/driver.rs`),
  transposition cache on `state_fingerprint`. Estimated 1.5-2×
  sims-equivalent at fixed wall; doesn't help wall directly but lifts
  policy strength per second.
- Tier 4 String → interned IDs (`SideState.title`,
  `UmamusumeInstance.species`, `used_ability_names_*`). Reaches the
  "GameState::clone is close to a memcpy" ceiling but is multi-day and
  invasive; defer until the easy wins are exhausted.

## Cross-references

- Mechanism + numbers + neutrality proof: `docs/ai-research/progress/r110.md` §4b.
- W6 recipe-fix reproduction recipe: `docs/ai-research/scoping/r110-w6-reproduction.md`.
- Live queue item: `docs/ai-agent-state/queue.json` (`r12-throughput`).
