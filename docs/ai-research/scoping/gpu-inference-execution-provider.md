# GPU Inference Execution Provider for Sim/Gate Throughput

> STATUS: **SCOPING** — no implementation yet. Distinct from
> [`gpu-fed-stronger-mcts.md`](gpu-fed-stronger-mcts.md), which is the
> *strength* axis (use GPU to buy MORE search). This doc is the
> *throughput* axis (use GPU to make the EXISTING recipe wallclock-cheaper).
> They share the underlying CUDA EP wiring but have different acceptance
> gates and may land independently.

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

### G1 — Feature-flag the ORT crate for CUDA (compile-only)

Add `cuda` to `ort` features in `engine-rs/Cargo.toml`. Verify
`cargo build` still green on a CPU-only host (the `cuda` feature should
only pull the CUDA EP loader code; runtime use is gated by
`with_execution_providers`).

**Open question**: ort 2.0.0-rc.12 + CUDA 13.1 + ORT shared lib 1.22.0
ABI compatibility. Quick test before committing: `cargo build` then
load a model with the CUDA EP and verify no `dlopen` failure on
`libonnxruntime_providers_cuda.so`.

Effort: ~30 min. Reversible (toggling the feature is a Cargo.toml
diff).

### G2 — Wire CUDA EP into InferenceSession

Add `device: Device` field on `InferenceSession`. In `load()`:

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

### G3 — Determinism contract revision

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

### G4 — End-to-end gate validation

Run `sim-eval-gate --device cpu --workers 16 --games 100 --model-side
both` and `--device cuda --workers 16 --games 100 --model-side both`
on `runs/R110-W6-repro/iter-0/policy.onnx` (n=200 total games each).
Compare:

| Metric | CPU baseline | CUDA target |
|---|---|---|
| wilson_lower | reference | within ±0.02 of CPU |
| wallclock | reference | ≥1.5× faster than CPU |
| running_win_rate | reference | within ±0.02 |
| Per-side wilson | reference | within ±0.025 each side |

Falsification: if CUDA wallclock is no better than CPU at workers=16,
abort and write a one-page negative result. Means rollouts + MCTS-select
dominate so heavily that inference moving to GPU is dwarfed — the
forward lever becomes parallelizing the rollouts themselves, not the
inference. (Plausible: at `--leaf rollout` with `--rollout-steps 200`,
every selection→rollout cycle is hundreds of TS-equivalent CPU steps
vs one ONNX call.)

Effort: ~1 hour wallclock. Acceptance binary: this scoping doc
populated with the actual numbers.

### G5 (optional, post-G4) — CPU-path Mutex removal

If G4 is decisive, G5 is a separate concern: does the **CPU** path
benefit from dropping the InferenceSession Mutex? The original `--workers
1` rationale assumed a single thread; the Mutex was defensive. ORT
CPU runtime with `intra_threads=1` may be concurrent-safe — worth a
1-day smoke test.

Acceptance: workers=16 CPU scaling improves from 4.5× → ≥6× on the
same `sim-eval-gate` smoke. Falsification: segfault or wilson_lower
divergence from workers=1 → revert.

Effort: ~1 day. Independent of G1-G4.

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

- `gpu-fed-stronger-mcts.md` — sibling, strength axis. Shares CUDA EP
  wiring but acceptance is "Wilson-lower beats production at higher
  sims", not "wallclock faster at same recipe".
- `throughput-optimization-spike.md` — predecessor. Closed; Slice 3c
  parallelism (commit `154248d`) is the precondition that surfaced the
  Mutex contention this slice would lift.
- `engine-rs/crates/engine/src/inference/mod.rs:148-203` — the
  Mutex-wrapped Session and CPU-pinned builder this slice opens up.
- Queue: `gpu-inference-execution-provider` (to be added).
