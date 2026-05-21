# Throughput Optimization Spike

> STATUS: **Slice 2 ACCEPTANCE PASSED 4.18× 2026-05-21** — 7.99 min wall vs 33.4 min baseline on 10-iter R110-faithful recipe. Slice 4 scale-up relook landed at `docs/ai-research/scoping/post-throughput-scale-up-directions.md`. Slice 3 (v3.2 featurizer port) now optional — required only for fast-path vhleaf loop.

Routing: `docs/ai-research/README.md`. Source-of-truth analysis:
`docs/ai-research/analysis/r12-loop-throughput.md` (commit `f19cb4c`).
Queue entry: `throughput-optimization-spike` in `docs/ai-agent-state/queue.json`.

## Decision

**Option A** (Rust `ort` crate, in-process ONNX inference) chosen over Option B
(NAPI bridge) because:

- Rust `ort` removes the HTTP-RTT entirely AND keeps the engine canonical
  (per `project-rust-first-class-default`); NAPI would have moved the cost
  but still required serializing the predict body through V8 ABI.
- ort 2.0.0-rc.12 `load-dynamic` reuses the system / venv `libonnxruntime.so`
  (no bundled binary, no compile-time download); pairs with the existing
  Python venv `libonnxruntime.so.1.22.0` already present in CI.
- The Rust path is direct ndarray → ORT C API → ndarray; the NAPI path
  added a JS marshal layer for no functional gain on the sim-cli code path
  (the NAPI bridge is still useful for the backend MCTS-serve path,
  which is the `rust-port-backend-napi-consumer` queue item).

## v3.0 I/O contract (frozen this slice)

Five input tensors fed to the v3.0 graph, matching
`training/serve_onnx.py:request_to_arrays` exactly:

| name                | dtype | shape                        |
|---------------------|-------|------------------------------|
| `state_features`    | f32   | `[1, 110]`                   |
| `action_features`   | f32   | `[1, A, 48]`                 |
| `action_mask`       | bool  | `[1, A]`                     |
| `card_ids_by_zone`  | i64   | `[1, 8, 30]`                 |
| `action_card_idx`   | i64   | `[1, A, 2]`                  |

Outputs: `[logits f32[1, A], value f32[1]]`. The Rust path applies the
serve_onnx greedy masked softmax (`max-shift` + `exp` + renormalize) and
returns `(probs[A], value scalar)`.

v3.2 (per-Uma slot tokens) is REJECTED at session load — the schema
validator surfaces a clean error pointing at v3.2 inputs in the graph.
Follow-up slice covers v3.2 once parity is confirmed end-to-end.

## Parity smoke result

Binary: `sim-inference-parity` (new `engine-rs/crates/sim-cli/src/bin/inference_parity.rs`).

Run command:
```
ORT_DYLIB_PATH=training/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.22.0 \
  engine-rs/target/release/sim-inference-parity \
  --onnx-path runs/R110-W6-repro/iter-2/policy.onnx \
  --serve-onnx-url http://127.0.0.1:8775 \
  --n-states 50 --tol 1e-5
```

Serve_onnx side: `python -m training.serve_onnx --model runs/R110-W6-repro/iter-2/policy.onnx --provider cpu --ort-threads 1 --port 8775` (single-threaded, FP-deterministic).

Result on 50 random `(state, legal_actions)` snapshots:
```
{"status":"PASS","n_states":50,"tol":1e-5,
 "max_prob_diff":2.384e-7,"max_value_diff":2.980e-7,
 "rust_mean_us":106,"http_mean_us":1606,"speedup":15.15,
 "elapsed_secs":0.088}
```

Per-action prob agreement ≤ 2.4e-7 abs (well under 1e-5 tol); scalar
value agreement ≤ 3.0e-7 abs. Mean inference latency: Rust 106µs,
HTTP 1606µs → **15x speedup at the call site**.

## 5-game cross-run determinism

```
ORT_DYLIB_PATH=... engine-rs/target/release/sim-mcts-selfplay \
  --onnx-path runs/R110-W6-repro/iter-2/policy.onnx \
  --seeds 5 --seed-base 1000 --mcts-simulations 20 \
  --mcts-leaf value-head --mcts-prior policy --no-root-dirichlet \
  --record-rows --out /tmp/run-{a,b}.jsonl
```

Both runs produced identical JSONL (`diff /tmp/run-a.jsonl /tmp/run-b.jsonl` empty). 5 games: 4 player wins / 1 opponent win, terminalReason=gameOver on all. Visit distributions + final winners match game-for-game across the two runs.

## Implementation notes

- `engine` crate gains `ort = 2.0.0-rc.12` (load-dynamic, api-21) +
  `ndarray = 0.17` (matches ORT's transitive major); gated behind the
  default-on `inference` feature so leaner consumers (codegen, golden-
  replay) can drop the binding.
- API version pinned to `api-21` (not the ort-rs default `api-24`) so
  the in-process binding stays compatible with ORT 1.22's runtime ABI
  (api-22's `GetEpDevices` discovery aborts on ORT 1.22 with an
  empty-vector C++ assertion).
- `MctsConfig.model_url` kept as ignored-legacy field for one release
  so orchestrator argument plumbing doesn't break in lockstep with the
  binary change.
- `crate::inference::set_global` initializes the session once per sim-cli
  process; MCTS predict paths borrow via `inference::global()`.
- HTTP `/predict` path removed from `mcts/driver.rs`; the `ureq` dep is
  now only pulled by the `sim-inference-parity` smoke binary (kept in
  sim-cli's Cargo.toml not the engine).

## Slice 2 acceptance — PASSED 4.18× (2026-05-21)

10-iter R110-faithful recipe (v3.0, W6-fix-OFF, seeded from R110-W6-repro iter-2 v3.0 ckpt wl=0.6042) ran end-to-end at `runs/throughput-spike-slice2-acceptance/`:

- **Total wall 7.99 min** (13:02:07Z → 13:10:33Z) vs 33.4 min baseline = **4.18× speedup**. Clears both the ≥1.65× PASS gate and the ≥3× scale-up trigger.
- Trajectory: 0.4773 / 0.5022 / 0.5273 / 0.4939 / 0.5189 / 0.4856 / 0.5442 / 0.5358 / 0.5189 / **0.5527** (iter-9 promoted).
- 10 finite Wilson-lower iters, all promote=True, no crashes / no heuristic fallbacks.
- Per-iter mean ~48s = selfplay ~16s + distill ~4s + gate ~28s. Gate dominates.
- R14 value-head crossover crossed at iter-4 (ratio 0.787, pearson 0.727) and iter-9 (ratio 0.730, pearson 0.746) — two crossings without explicit W6-fix-ON, consistent with R110 chain.

Canonical evidence: `runs/throughput-spike-slice2-acceptance/loop/orchestrator-state.json`, `launch.log`.

## Slice 4 — scale-up research-direction relook (LANDED)

Per user direction: if Slice 2 confirms ≥3× wall reduction, do a scale-up relook on what was previously infeasible that now becomes tractable. Condition met (4.18×). Relook landed at `docs/ai-research/scoping/post-throughput-scale-up-directions.md` (SCOPING, 51 lines).

Re-prioritization summary:

- PRIMARY: `tight-gate-reverdict-program` (P2→P1) — n=10,000 re-verdicts of R110-W6-repro iter-2 + R16-P1 v3.1 + 5g/5h/5i flips.
- SECONDARY: multi-knob W6-fix HP sweep (10-cell diagonal, ~80 min wall).
- TERTIARY: `high-sim-mcts-regime-probe` (P2→P1) — strength-vs-sim curve at 800/4k/20k/50k sims.
- DEPRIORITIZED: larger-model arch sweep, 50+ iter long-horizon (gate on positive signal).
- BACKGROUND: side-conditioned eval as default `sim-eval-gate` report shape.

## Slice 3 (deferred, optional)

Port v3.2 per-Uma slot featurizer (`uma_slot_card_ids` + `uma_slot_features`) + ONNX inputs to the Rust path so the v3.2 lineage best (`runs/R16-P2-c8-w6fix-on-extended-2x-games/loop/iter-3/`, wl=0.5538) can ride the in-process path. Required ONLY for the user-queued vhleaf loop on the fast path. Alternative: run vhleaf on the slower HTTP path now (~33 min/10 iters) and ride Slice 3 for any follow-up.
