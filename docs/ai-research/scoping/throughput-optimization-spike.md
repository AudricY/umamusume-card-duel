# Throughput Optimization Spike

> STATUS: **Slice 1 LANDED 2026-05-21** — in-process ORT (Option A) parity gate green. Orchestrator-wiring follow-up gated next.

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

## Next slice

`r12_orchestrator.py` wiring: orchestrator currently spawns serve_onnx
and passes `--model-url <url>` to sim-mcts-selfplay / sim-eval-gate.
Replace with `--onnx-path <path>` + `ORT_DYLIB_PATH` env on the worker
processes. Skip the serve_onnx subprocess for the Rust path entirely.
Acceptance: 10-iter recipe wall < 20 min (vs 33.4 min baseline) +
strength within ±0.022. Tracked as the next concrete step under the
`throughput-optimization-spike` queue entry.
