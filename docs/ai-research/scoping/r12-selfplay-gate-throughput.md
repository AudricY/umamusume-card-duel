# Scoping: r12 selfplay + gate throughput

Status: Deliverable 1 landed (worker default 1→24). Items #2
(rollout hash-carry) and #3 (HTTP keep-alive) landed ON, both
proven bit-identical by the determinism gate (see below). Item #4
(CRN/value-head leaf) stays deferred to R111 — it is a research-recipe
decision and changing it would break the trusted R110 A/B comparison.

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

4. **DEFERRED to R111 — CRN / value-head leaf change.** Switching the
   leaf evaluator or CRN sample count changes the learning targets and
   is a research-recipe decision, NOT a unilateral throughput change.
   Held OUT of scope so the R111 W6 recipe-fix A/B stays comparable to
   R110. Owned by the research backlog, not this scoping doc.

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

## serve_onnx bottleneck note

`serve_onnx` is a stdlib `ThreadingHTTPServer` with one shared,
thread-safe ORT `InferenceSession` (`serve_onnx.py:35,60-71`). At 24
workers and ~7% predict share at batch=1 it is not expected to
saturate. A cheap `intra_op_num_threads`/`inter_op_num_threads` knob
exists (`serve_onnx.py:62-63`) if it does — flagged, not changed now
(would not affect trajectories but is out of the trusted-A/B scope).

## Cross-references

- Mechanism + numbers + neutrality proof: `docs/ai-research/progress/r110.md` §4b.
- W6 recipe-fix reproduction recipe: `docs/ai-research/scoping/r110-w6-reproduction.md`.
- Live queue item: `docs/ai-agent-state/queue.json` (`r12-throughput`).
