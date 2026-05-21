# R16 Archive — Rolled Detail

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
