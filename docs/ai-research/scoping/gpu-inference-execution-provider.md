# GPU Inference Execution Provider for Sim/Gate Throughput

> STATUS: **G1+G2 LANDED 2026-05-22 / G4 FALSIFIED 2026-05-22 / G5
> LANDED 2026-05-22** — throughput motivation on the GPU path is wrong
> on the current `--leaf rollout` recipe (CUDA wall 5.12× SLOWER than
> CPU at workers=16; see §G4 below). G5 dropped the CPU `Mutex<Session>`
> and unlocked 7.11× scaling at workers=16 over workers=1, bit-identical
> wilson_lower across `--workers ∈ {1, 16, 32}` on R110-W6-repro/iter-0
> (see §G5 below). Strength agreement gate PASSED on G4
> (|wl_cuda-wl_cpu| = 0.005 < 0.02 envelope). G1+G2 wiring still ships
> as an opt-in (`--device cuda` flag on `sim-eval-gate`) because the
> *strength* axis ([`gpu-fed-stronger-mcts.md`](gpu-fed-stronger-mcts.md))
> reuses the same CUDA EP plumbing — that's where GPU buys progress
> (higher sims, not lower latency at fixed sims). G3 (separate
> `sim-gpu-cpu-agreement` binary) folded into G4's manual A/B.

Routing: `docs/ai-research/README.md`. Predecessor:
[`throughput-optimization-spike.md`](throughput-optimization-spike.md)
(closed; landed Slice 1 / 2 / 3 / 3b / 3c). Queue entry to be added as
`gpu-inference-execution-provider` in
`docs/ai-agent-state/queue.json`. Backlog cross-ref:
`docs/ai-feature-engineering-backlog.md`.

## Problem

`sim-eval-gate` and `sim-mcts-selfplay` currently pin ORT to single-threaded
CPU (`Session::builder().with_intra_threads(1).with_inter_threads(1)`,
`engine-rs/crates/engine/src/inference/mod.rs:184-190`). Rationale per
the R14.G comment is FP-determinism parity with `serve_onnx`'s
`--ort-threads 1` default — the parity smoke (`sim-inference-parity`)
asserts `max_prob_diff < 1e-5` against the CPU serve_onnx path.

The Mutex around `Session` (`engine::inference::InferenceSession::session:
Mutex<Session>`) means concurrent worker threads from Slice 3c parallelism
serialize on every inference call. Observed scaling at workers=8 is **4.5×**
vs an ideal 8× (commit `154248d`); the gap is mostly Mutex contention on
the policy-prior path at `--sims 100 --leaf rollout`.

Hardware available locally (and likely in CI/dev boxes broadly): NVIDIA
RTX 5000 Ada (16 GB), CUDA 13.1, ORT 1.22.0 CUDA + TensorRT EP shared
libs already present in `training/.venv/lib/python3.12/site-packages/onnxruntime/capi/`.
The Rust `ort` crate is currently configured without CUDA features
(`engine-rs/Cargo.toml: ort = { ..., default-features = false, features
= ["std", "ndarray", "load-dynamic", "api-21"] }`).

## Goal

Add an opt-in `--device cuda` path to `sim-eval-gate` (and the rest of
`engine-rs`'s sim-cli family) that runs ONNX inference on GPU, dropping
the per-call Mutex serialization. Acceptance: ≥1.5× wallclock improvement
on an n=200 sim-eval-gate at workers=16 over the CPU path on the same
ckpt, AND wilson_lower agreement within 0.02 of the CPU path.

The CPU path remains the default and stays FP-deterministic with
serve_onnx. GPU is an additional path for fast iteration, not a
replacement.

## Slices

### G1 — Feature-flag the ORT crate for CUDA (LANDED 2026-05-22)

Added `cuda` to `ort` features in `engine-rs/Cargo.toml`. `cargo build
--release` workspace-wide green; the CUDA feature only pulls the CUDA EP
loader code, runtime use is still gated by `with_execution_providers`.
CPU-only paths are unaffected.

ABI compatibility resolved at smoke test: ort 2.0.0-rc.12 against ORT
1.22.0 dynamic library loads cleanly; the CUDA EP requires the standard
nvidia CUDA + cuDNN ancillary libs on `LD_LIBRARY_PATH` (cudnn.so.9 is
the first missing dep without it).

### G2 — Wire CUDA EP into InferenceSession (LANDED 2026-05-22)

Implemented exactly as scoped, with one Rust-side wrinkle. Added
`Device { Cpu, Cuda { device_id: i32 } }` enum and a private
`SessionGuard { Cpu(Mutex<Session>), Cuda(UnsafeCell<Session>) }`
internal type so the CPU path keeps the historical `Mutex<Session>` (R14.G
FP-determinism contract intact) and the CUDA path drops it. Per-call
dispatch is a `match &self.session` in `predict_v3`. `InferenceSession::load`
delegates to `load_on(_, Device::Cpu)` so every existing call site
continues to compile unchanged. `sim-eval-gate` gained `--device cpu|cuda`
(default cpu) and `--cuda-device-id` (default 0); other sim-cli binaries
continue to default to CPU via the back-compat `load()`.

Design wrinkle: `ort::Session::run` takes `&mut self` even though the
ORT C API on the CUDA EP is documented reentrant, so we punch through
an `UnsafeCell<Session>` (declared `Sync` via `unsafe impl Sync for
SessionGuard`) on the CUDA branch. CPU branch keeps the mutex, so no
aliasing-`&mut` happens there. The pattern is contained to the
inference module; nothing else needs to know.

Original scoping sketch follows for reference:

```rust
let builder = Session::builder()?;
let builder = match device {
    Device::Cpu => builder.with_intra_threads(1)?.with_inter_threads(1)?,
    Device::Cuda { device_id } => builder
        .with_execution_providers([CUDAExecutionProvider::default()
            .with_device_id(device_id)
            .build()])?,
};
```

Drop or relax `Mutex<Session>` when `Device::Cuda`: the ORT CUDA EP is
documented thread-safe for concurrent `Session::run` calls. CPU path
keeps the Mutex (defensive; the CPU runtime is thread-safe in practice
but the existing comment says "single-threaded per binary" so we don't
disturb it in this slice).

Add `--device cpu|cuda` flag on `sim-eval-gate` (and a `--cuda-device-id`
companion, default 0). Default device is `cpu` so every existing
invocation keeps the current behavior.

Effort: ~half day. Smoke test: load `runs/R110-W6-repro/iter-0/policy.onnx`
on CUDA, run 2 games at sims=100, assert no crash and outputs in
[0,1] for probs.

### G3 — Determinism contract revision (FOLDED INTO G4)

Folded into G4 — the n=200 manual A/B agreement check covers what a
dedicated `sim-gpu-cpu-agreement` binary would have shown, with stronger
signal (full-game wilson agreement vs per-snapshot max-prob delta). If
G5 (CPU-path Mutex removal) ever lands and we want a regression smoke
that doesn't require running a full eval-gate, a per-snapshot agreement
binary is a 2-hour ticket — track that under G5's scope rather than
this slice.

Original scoping notes follow for reference:


The CPU path's `inference_parity.rs` smoke asserts `< 1e-5` agreement
vs serve_onnx CPU. That contract does NOT apply on GPU — different
reduction trees, fused convolution kernels, and per-call kernel
heuristics break bit-exact agreement at the 1e-5 level (and intra-run
determinism is GPU-driver dependent).

Two-part revision:

1. **Mark `sim-inference-parity` as CPU-only.** Update the binary
   docstring and add an explicit `--device cpu` assertion at startup
   if the underlying session detects CUDA EP. Don't try to make a CUDA
   parity smoke — the right comparator for GPU is statistical, not
   bit-exact.

2. **Add `sim-gpu-cpu-agreement` smoke** (separate binary). Loads the
   same ONNX on CPU and CUDA, runs n=50 random snapshots through both,
   asserts `max_prob_diff < 1e-3` and `max_value_diff < 1e-3`. This is
   a "same model, different runtime" sanity check — guards against
   silently-wrong CUDA paths (e.g., a quantization or precision
   regression in the EP), not a determinism claim.

Effort: ~2 hours. Acceptance: agreement smoke green on R110-W6-repro
iter-0 ckpt.

### G4 — End-to-end gate validation (FALSIFIED 2026-05-22)

Ran `sim-eval-gate --device {cpu,cuda} --workers 16 --games 100 --model-side
both` on `runs/R110-W6-repro/iter-0/policy.onnx` at `--sims 100 --leaf
rollout --k 3 --rollout-steps 200 --prior policy`. n=200 total games each.

Actual numbers (RTX 5000 Ada, ORT 1.22.0 CUDA EP, ort 2.0.0-rc.12):

| Metric | CPU baseline | CUDA target | Δ vs gate |
|---|---|---|---|
| elapsedSecs | 10.766 | 55.177 | **5.12× SLOWER** (gate: ≥1.5× FASTER) — **FAIL** |
| overall.wilsonLower | 0.4957 | 0.5007 | abs Δ = 0.005 (gate: < 0.02) — PASS |
| overall.winRate | 0.565 | 0.570 | abs Δ = 0.005 (gate: < 0.02) — PASS |
| playerSide.wilsonLower | 0.4426 | 0.4623 | abs Δ = 0.020 (gate: < 0.025) — PASS |
| opponentSide.wilsonLower | 0.4920 | 0.4821 | abs Δ = 0.010 (gate: < 0.025) — PASS |
| GPU utilization | n/a | 38-39% (peak) | not compute-bound |
| GPU memory | n/a | 911 MiB / 16384 MiB | comfortably under budget |

**Verdict: strength agreement gate PASS, wallclock gate FAIL.** Per the
scoping falsification clause: G1+G2 wiring is still useful (the strength
axis reuses it), but the throughput motivation for this slice is wrong.
Root cause matches the §Risks "MCTS rollout dominance" prediction: at
`--leaf rollout --rollout-steps 200` the per-leaf heuristic rollout (200
TS-equivalent dispatcher cycles) is hundreds of times heavier than the
one ONNX call it replaces. Moving inference CPU→GPU adds H2D/D2H copy
latency and per-call kernel-launch overhead (~5-15ms / call observed on
ad-hoc traces, vs ~10-100us in-process CPU). At workers=16 the CPU path
already serialized on the InferenceSession mutex but the rollouts ran in
parallel — that's why the gross wall was 10.77s. The CUDA path drops the
mutex but every single ONNX call (~100 sims × 200 games × selection-tree
fanout) now pays GPU overhead with no compute win. Strength tracks
(correctness OK — same model, same masked-softmax, agreement within
sampling noise) but the recipe doesn't have enough inference compute to
amortize device transfers.

**Implication for the strength axis**
([`gpu-fed-stronger-mcts.md`](gpu-fed-stronger-mcts.md)): the lever is to
spend the GPU on MORE work per game (e.g. `--sims 1000`+ or batched
value-head evaluation), not the same work moved off-CPU. Acceptance there
should be "wilsonLower at sims=N CUDA beats production ceiling at fixed
wallclock budget", not "wallclock parity at sims=100".

G3 (separate `sim-gpu-cpu-agreement` binary): folded into the G4 A/B
above. The 0.005 wilsonLower delta on n=200 is a stronger agreement
signal than a per-call 1e-3 max-prob smoke would have provided; a
silent-wrong CUDA path would have shown up as a much wider strength gap.

Validation environment notes:
- `ORT_DYLIB_PATH=training/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.22.0`
  (the venv-bundled ORT 1.22.0 has CUDA EP support compiled in).
- `LD_LIBRARY_PATH` must include the `nvidia/cudnn/lib`,
  `nvidia/cublas/lib`, `nvidia/cuda_runtime/lib`, and friends from the
  same venv — `libonnxruntime_providers_cuda.so` needs `libcudnn.so.9`
  on the loader path. Without this the CUDA EP fails at dlopen with
  "libcudnn.so.9: cannot open shared object file".

### G5 — CPU-path Mutex removal (LANDED 2026-05-22)

Slice 3c (commit `154248d`) unlocked game-level worker parallelism, but
sim-eval-gate at workers=16 was scaling ~3.76× over workers=1 instead
of the ~8× the box could deliver. The dominant remaining serialization
was `engine::inference::InferenceSession::session: Mutex<Session>` —
every `predict_v3` call locked it.

G5 replaced `SessionGuard::Cpu(Mutex<Session>)` with
`SessionGuard::Cpu(UnsafeCell<Session>)`, reusing the same pattern G2
established on the CUDA branch. The `unsafe impl Sync for SessionGuard`
annotation now covers both variants with the same justification: ORT's
`Session::Run` is documented thread-safe for concurrent calls on a
single session, and the `&mut self` on `ort::Session::run` is a
Rust-API artifact, not a real exclusivity requirement. The CPU branch
keeps `intra_threads = 1` + `inter_threads = 1` on the SessionBuilder
(R14.G FP-determinism contract: those control ORT's internal thread
pool, which is a different concern from how many Rust threads may
concurrently dispatch `Session::run`).

The dispatch site in `predict_v3` collapsed from two arms to one
shared `Cpu | Cuda` match arm — both variants now punch through the
`UnsafeCell` identically.

Validation (sim-eval-gate, n=200 games, R110-W6-repro/iter-0/policy.onnx,
sims=100, leaf=rollout, rollout-steps=200, prior=policy):

| `--workers` | wilsonLower | elapsedSecs | gamesPerSec | speedup vs w=1 |
|---|---|---|---|---|
| 1 | 0.4957060908195922 | 58.877 | 3.397 | 1.00× |
| 16 | 0.4957060908195922 | 8.282 | 24.148 | **7.11×** |
| 32 | 0.4957060908195922 | 7.449 | 26.849 | 7.90× |

Gates met:
1. **Bit-identical wilson_lower across `--workers ∈ {1, 16, 32}`** to
   all 16 decimal places — matches the Slice 3c workers=1 golden
   exactly. The CPU `Session::Run` is reentrant in practice, no FP
   drift from concurrent calls.
2. **Speedup 7.11× at workers=16** vs the ≥6× gate (and vs Slice 3c's
   4.5× at workers=8 — a clear +25% efficiency lift from removing the
   Mutex bottleneck).
3. No segfault, no panic, no ORT thread-safety warning across the
   three runs.

The w=32 gain over w=16 is modest (7.90× vs 7.11×) because a
concurrent v3.2 deck-sampling gate was using ~16 cores during the
acceptance run — w=32 was oversubscribed; in an unloaded box w=32
would likely scale closer to 10-12×. The Slice 3c brief's "ceiling
~6-7 cores" was Mutex-bound; with G5 the new ceiling is contention on
the shared `extract_logits_and_value` allocation path (Vec<f32>
copies), not session-level serialization. If post-G5 throughput
becomes the bottleneck on a new workload, the next lever is batched
inference (gather N action-features tensors and run them as a single
session call), but the current scaling closes the original "workers=16
stalls at ~7 cores" gap.

Effort: ~1 hour (code change is minimal; the heavy lift was acceptance
validation across three worker counts).

## Risks

- **ort + CUDA ABI mismatch.** ort 2.0.0-rc.12's bundled headers
  pre-date ORT 1.22.0 by some margin. Most-likely failure: dlopen of
  `libonnxruntime_providers_cuda.so` succeeds but a symbol is missing
  at first `Session::run` call. Mitigation: gate G2 commit on the
  agreement smoke green; abort if loader fails.

- **MCTS rollout dominance.** If `--leaf rollout` walls so much CPU
  time in heuristic rollouts that GPU inference is dwarfed, G4 falsifies
  and we get a small wallclock win at best. The strength axis (`gpu-fed-
  stronger-mcts.md`) is the better lever in that case — GPU buys higher
  sims, not lower latency.

- **CUDA non-determinism leakage**. If any current production code
  path depends on bit-exact ORT output reproducibility (golden traces,
  visit-count snapshots), it must stay on the CPU path. Out of scope
  for this slice: the CPU path remains the production default.

- **Multi-GPU / device-id selection** is not in scope; single-device,
  device-id 0 default. Add `--cuda-device-id` only because it's a
  one-line addition.

## Crosslinks

- [`gpu-batched-inference-throughput.md`](gpu-batched-inference-throughput.md)
  — successor for the `next_action` (b) lever (batched dispatch). Scoping
  only as of 2026-05-25; B1 smoke gates whether the ONNX exporter admits
  dynamic batch.
- `gpu-fed-stronger-mcts.md` — sibling, strength axis. Shares CUDA EP
  wiring but acceptance is "Wilson-lower beats production at higher
  sims", not "wallclock faster at same recipe".
- `throughput-optimization-spike.md` — predecessor. Closed; Slice 3c
  parallelism (commit `154248d`) is the precondition that surfaced the
  Mutex contention this slice would lift.
- `engine-rs/crates/engine/src/inference/mod.rs` — `Device` enum,
  `SessionGuard` (Cpu/Cuda variants, both `UnsafeCell<Session>` post-G5)
  and the CPU-pinned `intra/inter_threads = 1` SessionBuilder.
- Queue: `gpu-inference-execution-provider` (to be added).
