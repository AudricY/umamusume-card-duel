# Scoping: r12 selfplay + gate throughput

Status: Deliverable 1 landed (worker default 1→24). Remaining items
are ranked follow-ups; none may be applied without the stated gate.

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

2. **Rollout clone/fingerprint hot loop refactor.** The leaf-rollout
   path clones game state and recomputes fingerprints per simulation;
   profiling flagged this as the dominant single-core cost inside the
   ~99% selfplay+gate budget. Estimated material per-core speedup but
   it changes simulation internals.
   **GATE (hard prerequisite): a determinism replay gate must exist
   and pass before this refactor lands.** The refactor risks
   perturbing trajectories; without a byte-level replay check against
   a frozen seed corpus it would silently contaminate the upcoming
   trusted W6 A/B run (R111). Do NOT implement before the gate. This
   is the explicit "determinism-gate-before-rollout-clone-refactor"
   requirement.

3. **HTTP keep-alive between workers and serve_onnx.** Each predict
   currently pays connection setup. Low risk (transport only, no
   trajectory effect) but lower expected payoff than (1)/(2) at
   batch=1 / ~7% predict share. Cheap follow-up after (1) lands and
   the new bottleneck is re-measured at 24 workers.

4. **CRN / value-head leaf change.** Switching the leaf evaluator or
   CRN sample count changes the learning targets and is a
   research-recipe decision, NOT a unilateral throughput change. Owned
   by the research backlog, not this scoping doc.

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
