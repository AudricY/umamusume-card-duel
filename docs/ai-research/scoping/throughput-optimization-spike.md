# Throughput Optimization Spike

> STATUS: **Slice 3 LANDED 2026-05-21** — v3.2 per-Uma slot featurizer ported to Rust; parity smoke 50/50 PASS (max_prob_diff 1.79e-7, max_value_diff 4.47e-7) + 5-game bit-identical cross-run on v3.2 ONNX. Slice 2 acceptance PASSED 4.18× (7.99 min vs 33.4 min baseline). vhleaf loop now unblocked for fast path.

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

## Slice 3 — v3.2 featurizer port LANDED (2026-05-21)

Added the v3.2 per-Uma slot featurizer to the Rust in-process path so v3.2 ONNX graphs (`runs/R16-P2-c8-w6fix-on-extended-2x-games/loop/iter-3/policy.onnx`, sidecar `state_feature_schema_version=3.2`, `uses_uma_slot_tokens=true`) ride the fast path. Implementation:

- `engine-rs/crates/engine/src/policy/featurize.rs` gains `observation_uma_slots(obs) -> (Vec<i64>[10], Vec<f32>[10*23])`, a verbatim port of Python `observation_to_uma_slots` + `_uma_slot_feature_row` (slot order, 23-d column layout, "absence is zero" contract).
- `engine-rs/crates/engine/src/inference/mod.rs` detects v3.0 vs v3.2 at session-load time by ONNX input-set (5 vs 7); v3.0 dispatches the original 5-input feed, v3.2 dispatches a 7-input feed with the two new slot tensors at `[1, 10]` (int64) and `[1, 10, 23]` (float32). Partial-pair guard mirrors `serve_onnx._graph_has_partial_uma_slot_inputs`. v3.1 (164-d) still hard-rejected.
- 6 new unit tests in `policy::featurize` (shape, polarity/role/present flags, absent 4th-bench reserved slot, hp+damage sum-to-1, typed-energy sum, absent-active row). Total engine unit-test count 91/91 green.

### Slice 3 parity smoke

Binary unchanged (`sim-inference-parity` is schema-agnostic — it POSTs `{observation, legalActions, sampling}` and lets serve_onnx build its own tensors; the Rust side builds its own from the same observation). Source ONNX: `runs/R16-P2-c8-w6fix-on-extended-2x-games/loop/iter-3/policy.onnx` (`state_dim=110`, `state_feature_schema_version=3.2`, `uses_uma_slot_tokens=true`, `card_vocab.hash=e3a35716156494d6`).

Run command:
```
ORT_DYLIB_PATH=training/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.22.0 \
  engine-rs/target/release/sim-inference-parity \
  --onnx-path runs/R16-P2-c8-w6fix-on-extended-2x-games/loop/iter-3/policy.onnx \
  --serve-onnx-url http://127.0.0.1:8776 \
  --n-states 50 --tol 1e-5
```

Serve_onnx side: `cd training && .venv/bin/python serve_onnx.py --model ../runs/R16-P2-c8-w6fix-on-extended-2x-games/loop/iter-3/policy.onnx --provider cpu --ort-threads 1 --port 8776 --host 127.0.0.1`.

Result on 50 random `(state, legal_actions)` snapshots:
```
{"status":"PASS","n_states":50,"tol":1e-5,
 "max_prob_diff":1.788e-7,"max_value_diff":4.470e-7,
 "rust_mean_us":150,"http_mean_us":1632,"speedup":10.88,
 "elapsed_secs":0.092}
```

Per-action prob agreement ≤ 1.79e-7 abs (well under 1e-5 tol); scalar value agreement ≤ 4.47e-7 abs. Mean inference latency Rust 150µs / HTTP 1632µs → 10.88× speedup at the call site on the v3.2 graph (similar to Slice 1's 15× on v3.0; small gap is the extra ndarray reshape + 2 ORT input bindings).

### Slice 3 5-game cross-run determinism (rollout leaf, v3.2)

```
ORT_DYLIB_PATH=... engine-rs/target/release/sim-mcts-selfplay \
  --onnx-path runs/R16-P2-c8-w6fix-on-extended-2x-games/loop/iter-3/policy.onnx \
  --seeds 5 --seed-base 2500 --sims 20 \
  --leaf rollout --prior policy --no-root-dirichlet \
  --record-rows --out /tmp/run-{a,b}-v32.jsonl
```

Both runs: `diff /tmp/run-a-v32.jsonl /tmp/run-b-v32.jsonl` empty (37 lines each, byte-identical). 5 games: 2 player wins / 3 opponent wins, terminalReason=gameOver on all. Used `--leaf rollout` (not `--leaf value-head`) per the brief — vhleaf will be exercised by the user-queued loop downstream once the value head is calibrated.

vhleaf loop unblock: with v3.2 on the Rust fast path, the user-queued `runs/R16-P2-c8-w6fix-on-extended-2x-games/loop/iter-3/checkpoint.pt`-seeded vhleaf loop can run at ~12 min/10-iter wall (the Slice 2 4.18× pace), down from ~33 min/10-iter on the HTTP path.
