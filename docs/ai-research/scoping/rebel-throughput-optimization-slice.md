# ReBeL Throughput Optimization Slice

- **Date:** 2026-05-28
- **Status:** IMPLEMENTED-MEASURED
- **Parent scope:** `docs/ai-research/scoping/rebel-e2e-scoping.md`
- **Predecessors:** `docs/ai-research/scoping/r12-selfplay-gate-throughput.md`, `docs/ai-research/scoping/distill-throughput-spike.md`, `docs/ai-research/scoping/gpu-batched-inference-throughput.md`, `docs/ai-research/scoping/action-count-bucketed-wave-batching.md`
- **Intent:** Make the new ReBeL E2E line cheap enough for real iteration without changing its algorithmic target or leaking hidden information.

## Summary

The previous throughput program carries over to ReBeL only at the shared engine and training layers. It does **not** automatically carry over to ReBeL search/self-play.

What carries over:

- Rust simulator, dispatcher, legal-action enumeration, clone/fingerprint work, deck sampling, and headless setup.
- `train_bc.py` training knobs: larger batch, AMP, dataloader workers, optional compile, KL anchor, entropy bonus, Q-head loss.
- ReBeL dataset/model/export path: `RebelSelfPlayDataset`, belief-feature model input, and belief-input ONNX export.

What does not carry over yet:

- `sim-rebel-selfplay --workers` is accepted but currently sequential.
- ReBeL search is sequential over `particles * legal_actions`.
- Existing MCTS wave batching is not used by `engine::rebel`.
- Belief-input ONNX graphs are exportable but not served by the current `serve_onnx.py` request path.
- `rebel_orchestrator.py` does not pass the landed distill-throughput recipe by default.

The right first slice is therefore **outer-loop throughput before inner-search throughput**:

1. Add deterministic game-level worker fan-out to `sim-rebel-selfplay`.
2. Add ReBeL timing diagnostics to identify wall share: belief build, search, row serialization, advance.
3. Remove low-risk duplicate work and allocation in belief/search hot paths.
4. Wire the known training-throughput knobs through `rebel_orchestrator.py`.
5. Only after those measurements, decide whether to parallelize particle/action evaluations or add belief-model batched inference.

## Current Bottleneck Shape

### Search cost

`engine-rs/crates/engine/src/rebel/mod.rs::run_public_belief_search` evaluates every particle/action pair for the configured search budget:

```text
work per decision ~= particle_count * legal_action_count * search_iterations * leaf_cost
```

At defaults, that can mean roughly:

```text
64 particles * O(2-20 legal actions) * 64 iterations * leaf_cost
```

per modeled decision. With neural leaves, those leaf states are batched into one ONNX value-head call per public-belief decision.

### Self-play cost

`engine-rs/crates/sim-cli/src/bin/rebel_selfplay.rs` runs a plain nested side/seed loop. It accepts `--workers` for compatibility, but it warns and runs sequentially. This is the highest-confidence throughput gap because the MCTS self-play worker pool already proved deterministic and effective.

### Belief-build cost

`engine-rs/crates/engine/src/belief/mod.rs::build_public_belief_state` clones a full `GameState` once per particle, then builds particle assignments and probability summaries with several avoidable Vec/String allocations. This is not necessarily dominant versus rollout search, but it is cheap to instrument and has low-risk cleanup opportunities.

### Training cost

`train_bc.py --data-mode rebel` already supports the shared throughput knobs if passed. The dedicated ReBeL orchestrator does not pass them. Its defaults are smoke-oriented (`epochs=1`, `batch_size=16`, no AMP, no dataloader workers, no compile flag, no LR override), not the landed distill recipe from `distill-throughput-spike.md`.

## Non-Goals

- Do not change ReBeL into per-particle perfect-information MCTS.
- Do not make root action selection depend on hidden particle identity.
- Do not introduce a serve/eval deployment path for ReBeL ONNX in this slice unless needed for a smoke; exportability is enough.
- Do not tune strength or promotion thresholds here.
- Do not default to MCTS wave batching inside ReBeL before measuring whether inference is even the bottleneck.

## Slice A: Deterministic ReBeL Worker Pool

### Scope

Port the proven `sim-mcts-selfplay` worker-pool pattern to `sim-rebel-selfplay`:

- Build task list from `(seed, model_side)`.
- Resolve deck pairs per task index.
- Use `Arc<Vec<Task>>`, `AtomicUsize` cursor, and scoped worker threads.
- Have each worker run `drive_one_game`.
- Serialize row lines inside the worker outcome.
- Store outcomes by task index.
- Drain outcomes in task-index order so JSONL output is deterministic.
- Support `--workers 0` as `available_parallelism()/2` or the same policy used by `sim-mcts-selfplay`.

### Acceptance

- `--workers 1`, `--workers 4`, and `--workers 8` produce byte-identical JSONL for a fixed small config:

```bash
cargo run -p sim-cli --bin sim-rebel-selfplay -- \
  --seeds 6 --model-side both --particles 4 --search-iterations 4 \
  --rollout-steps 8 --max-steps 40 --out /tmp/rebel-w1.jsonl --workers 1
```

Repeat for workers 4 and 8, then compare hashes.

- A throughput sweep is recorded for a rollout-heavy anchor:

```bash
--seeds 40 --model-side both --particles 16 --rollout-steps 40 --max-steps 120
```

Report wall, games/sec, rows/sec, CPU%, and speedup for workers `{1, 4, 8, 16}`.

- No row schema changes except optional diagnostics fields.

## Slice B: ReBeL Timing Diagnostics

### Scope

Add env-gated, low-overhead timing counters to `sim-rebel-selfplay` and/or `engine::rebel`:

- belief construction wall;
- public-belief search wall;
- row serialization wall;
- simulator advance wall;
- particle count;
- legal-action count;
- particle/action evaluation count;
- rollout leaf calls.

Use an env flag such as:

```text
UMA_LOG_REBEL_TIMING=1
```

Keep logs capped to avoid massive output, following `UMA_LOG_WAVE_TIMING` and `UMA_LOG_WAVE_NACTIONS`.

### Acceptance

- Tiny smoke prints per-decision timing for the first N decisions when the env flag is set.
- Default path has no material overhead and no extra output.
- ReBeL manifest records aggregate timing totals when the flag is on.

## Slice C: Low-Risk Hot-Path Cleanup

### Scope

Implement cleanup only where output remains deterministic and root public-policy semantics are unchanged:

- Avoid duplicate legal-action enumeration by allowing belief construction to reuse already enumerated legal actions.
- Move `belief.public_observation` into the row instead of rebuilding it in `row_from_search`.
- Use `state_fingerprint` instead of `state_hash` where only equality is needed for stall detection.
- In search, avoid redundant `GameState` clone before `advance_modeled_turn_step`, which already returns an owned next state.
- Combine RNG installation for forced coin generation and modeled advance.
- In belief features, compute means from running sums instead of count Vecs.
- Replace formatted particle-diversity strings with a cheaper structural key or hash.
- Consider making per-particle string assignments lazy/optional if JSONL rows do not need them.

### Acceptance

- Existing ReBeL smoke still passes.
- Worker determinism hashes stay identical before/after cleanup when run with `workers=1`.
- Timing counters show non-negative deltas; no throughput regression above noise on the anchor config.

## Slice D: ReBeL Orchestrator Training Knobs

### Scope

Wire the landed training-throughput knobs into `training/rebel_orchestrator.py`:

- `--lr`, default `6e-4` when `batch_size=256`.
- `--batch-size`, default promoted from smoke value to `256` for non-smoke runs.
- `--amp` / `--no-amp`.
- `--dataloader-workers`, default `4` for CUDA-oriented runs.
- `--compile`, opt-in, default off until measured for ReBeL.
- `--hidden-dim`, `--depth`, `--dropout`.
- `--grad-accum`.
- `--init-from-checkpoint`.
- `--workers` forwarded to `sim-rebel-selfplay`.
- Option to use release binary instead of `cargo run` once the binary is built.

Keep a clear smoke recipe that still uses tiny defaults or explicit smoke flags.

### Acceptance

- CLI plumbing test verifies the generated `train_bc.py` command includes `--batch-size 256 --lr 6e-4 --amp --dataloader-workers 4` when requested.
- One tiny CPU smoke still completes with explicit small settings.
- One CUDA-capable run records the throughput settings in the `train_bc.py` manifest and `rebel_orchestrator.py` manifest.

## Slice E: Measured Inner Search Parallelism

Do this only after Slices A-C show the wall share.

Candidate options:

1. **Parallel particle/action rollout evaluation.**
   - Split the `particles * actions` matrix across scoped workers inside one search.
   - Preserve one aggregated public root policy.
   - Use deterministic seed labels per `(particle_index, action_index)`.
   - Avoid sharing mutable RNG across workers.

2. **Batch leaf value inference for belief models.**
   - Once Rust inference accepts `belief_features`, add a ReBeL-specific batch path.
   - Adapt `predict_v3_batch` packing and action-count bucketing.
   - Do not use old MCTS policy/value graph dispatch for belief-input graphs without schema guards.

3. **Wave/CFR batching.**
   - Later target only.
   - Requires regret/average-strategy tables keyed by public information set.
   - Must pass an information-set invariant test: same public info set returns one root policy, independent of hidden particle identity.

### Acceptance

- Inner-search parallelism must preserve root policy within exact deterministic equality when using fixed per-cell seed labels, or explain and bound any intentional FP-order drift.
- V1 information-set policy audit passes.
- Wall gain must exceed `1.25x` on a measured search-dominated config before keeping extra complexity.

## Measurement Matrix

### Baseline anchors

Use three anchors:

1. **Smoke:** `seeds=1`, `model-side=player`, `particles=2`, `rollout-steps=5`, `max-steps=20`.
2. **Worker anchor:** `seeds=40`, `model-side=both`, `particles=16`, `rollout-steps=40`, `max-steps=120`.
3. **Search anchor:** `seeds=8`, `model-side=both`, `particles=64`, `rollout-steps=120`, `max-steps=120`.

### Metrics

Report:

- games/sec;
- rows/sec;
- modeled decisions/sec;
- mean legal-action count;
- mean particle count;
- particle/action evaluations/sec;
- belief-build wall share;
- search wall share;
- row serialization wall share;
- training wall by stage;
- ONNX export wall;
- output hash for determinism cells.

## Expected Payoff

Likely payoff order:

1. **Worker pool:** highest-confidence, likely `3-6x` on self-play generation until worker plateau.
2. **Training knob wiring:** carries over from distill; likely large on real corpora, low impact on tiny smokes.
3. **Duplicate-work cleanup:** likely `5-20%`, useful because it compounds with workers.
4. **Inner search parallelism:** potentially high, but only justified after timing says search dominates and worker pool has plateaued.
5. **Belief-model batched inference:** future payoff once ReBeL search uses neural belief leaves instead of rollout leaves.

## Risks And Guardrails

- **Strategy fusion risk:** never let worker or inner parallelism return per-particle root policies. Root output remains one public action distribution.
- **Determinism risk:** worker completion order must not affect row order, RNG streams, or terminal value targets.
- **False speedup risk:** `cargo run` build time and tiny smoke overhead can hide real throughput. Use release binary or warmed cargo for measurement.
- **Training deployment gap:** ReBeL ONNX export has belief inputs, but current `serve_onnx.py` does not feed them. Do not claim deployable eval until serving/eval schema is extended.
- **Iterations knob honesty:** `search_iterations` must correspond to actual particle/action leaf samples, not only a recorded diagnostic.

## Implementation Order

1. Add timing diagnostics.
2. Port deterministic worker fan-out.
3. Run worker determinism and throughput sweep.
4. Apply low-risk hot-path cleanup.
5. Re-run sweep and record deltas.
6. Wire orchestrator training knobs.
7. Run one tiny full E2E and one medium self-play-only throughput run.
8. Decide whether inner search parallelism is needed based on timing wall share.

## Files To Touch

- `engine-rs/crates/sim-cli/src/bin/rebel_selfplay.rs`
- `engine-rs/crates/engine/src/rebel/mod.rs`
- `engine-rs/crates/engine/src/belief/mod.rs`
- `training/rebel_orchestrator.py`
- Optional tests/smokes:
  - `training/rebel_throughput_smoke.py`
  - `engine-rs/crates/sim-cli/tests` if a Rust integration-test harness is added later

## Definition Of Done

The slice is complete when:

- `sim-rebel-selfplay --workers N` actually parallelizes games.
- Worker outputs are deterministic across `workers=1/4/8`.
- The scoping measurement matrix has before/after numbers.
- ReBeL orchestrator can launch the known fast training recipe.
- A tiny full ReBeL E2E run still completes and exports a belief-input ONNX.
- The docs clearly state which throughput optimizations carry over today, which were ported, and which remain future work.

## Implementation Notes - 2026-05-28

Ported in `0d82f40`:

- `sim-rebel-selfplay --workers N` now uses deterministic game-level fan-out with task-index ordered JSONL drain.
- `--workers 0` resolves to `available_parallelism()/2`, matching the MCTS self-play policy.
- `UMA_LOG_REBEL_TIMING=1` records capped per-decision stderr timing and aggregate `rebelTiming` manifest totals.
- ReBeL search diagnostics now include `particleActionEvaluations` and `rolloutLeafCalls`.
- Low-risk cleanup landed for duplicate legal-action enumeration, row observation reuse, stall `state_fingerprint`, the redundant search clone before modeled advance, and belief-feature mean vectors.
- `training/rebel_orchestrator.py` now forwards `--workers` and the ReBeL training-throughput recipe knobs: `--batch-size`, `--lr`, `--amp/--no-amp`, `--dataloader-workers`, `--compile`, `--hidden-dim`, `--depth`, `--dropout`, `--grad-accum`, and `--init-from-checkpoint`. It also has `--use-release-binary` for a prebuilt `target/release/sim-rebel-selfplay`.
- The orchestrator uses `training/.venv/bin/python` when available so `train_bc.py` and `export_onnx.py` run with the expected Torch/ONNX dependencies.

Determinism smoke:

```text
config: --seeds 6 --seed-start 17000 --model-side both --particles 4 --search-iterations 4 --rollout-steps 8 --max-steps 40
workers 1/4/8 sha256: 92cc53453e17b9515a0d01e9aa28e59486d68bce82957a616835cfc5e7f966f8
```

Release build:

```text
cargo build --release -p sim-cli --bin sim-rebel-selfplay: PASS
```

Release-binary worker anchor sweep:

```text
config: --seeds 40 --model-side both --particles 16 --search-iterations 16 --rollout-steps 40 --max-steps 120
workers=1  wall=2.91s games/sec=27.474  rows/sec=155.670  cpu=99%   speedup=1.00x
workers=4  wall=0.85s games/sec=93.558  rows/sec=532.941  cpu=388%  speedup=3.41x
workers=8  wall=0.56s games/sec=142.735 rows/sec=808.929  cpu=718%  speedup=5.20x
workers=16 wall=0.42s games/sec=190.661 rows/sec=1078.571 cpu=1196% speedup=6.94x
```

Timing sample:

```text
config: --seeds 8 --model-side both --particles 16 --search-iterations 16 --rollout-steps 40 --max-steps 120 --workers 8
decisions=76 meanLegalActionCount=4.25 particleActionEvaluations=5168
beliefBuildWallShare=0.50% searchWallShare=99.07% rowSerializationWallShare=0.06% advanceWallShare=0.37%
```

E2E smoke:

```text
command: python training/rebel_orchestrator.py --out-dir /tmp/rebel-e2e-export-smoke --smoke --games 1 --particles 2 --search-iterations 4 --rollout-steps 5 --max-steps 20 --model-side player --workers 1 --epochs 1 --device cpu
result: self-play rows=4, train_bc.py PASS, ONNX roundtrip PASS, exported /tmp/rebel-e2e-export-smoke/policy.onnx
```

Neural leaf-evaluation progress:

- `sim-rebel-selfplay --onnx-path ...` can now evaluate ReBeL particle/action leaves with batched belief-input ONNX inference via `--neural-leaf-weight`.
- `--neural-leaf-weight 1.0` disables rollout leaves for nonterminal model-evaluable cells, so the hot `particles * legal_actions` matrix becomes one batched ORT value-head call per modeled decision instead of many CPU rollouts.
- `--inference-batch-size` and `--inference-max-wait-us` route self-play through the same ORT dispatcher shape used by MCTS/eval, allowing cross-worker GPU coalescing.
- Root output is still one public policy over public legal actions. Neural leaf values are aggregated over particles; no per-particle root policy is emitted.

Neural self-play smoke:

```text
command: sim-rebel-selfplay --seeds 1 --model-side player --particles 2 --search-iterations 4 --rollout-steps 5 --max-steps 20 --workers 1 --onnx-path /tmp/rebel-e2e-export-smoke/policy.onnx --device cpu --neural-leaf-weight 1.0 --neural-policy-weight 0.25 --neural-value-weight 0.25 --inference-batch-size 4
first decision: neuralLeafBatchRows=20 neuralLeafCalls=20 rolloutLeafCalls=0
```

CUDA setup and smoke:

```text
training/.venv/bin/python -m pip install -r training/requirements-gpu.txt
torch: 2.7.1+cu126, cuda_available=True, device=NVIDIA RTX 5000 Ada Generation Laptop GPU
onnxruntime: 1.22.0, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
command: sim-rebel-selfplay --seeds 1 --model-side player --particles 2 --search-iterations 4 --rollout-steps 5 --max-steps 20 --workers 1 --onnx-path /tmp/rebel-two-stage-smoke/loop/iter-1/policy.onnx --device cuda --cuda-device-id 0 --neural-leaf-weight 1.0 --neural-policy-weight 0.25 --neural-value-weight 0.25 --inference-batch-size 4
result: loaded device=Cuda { device_id: 0 }; first decision neuralLeafBatchRows=20 neuralLeafCalls=20 rolloutLeafCalls=0.
```

CUDA training smoke:

```text
command: python training/rebel_orchestrator.py --out-dir /tmp/rebel-cuda-train-smoke --smoke --games 1 --particles 2 --search-iterations 4 --rollout-steps 5 --max-steps 20 --model-side player --workers 1 --epochs 1 --device cuda --amp --skip-export --skip-gates
result: train_bc.py PASS, device=cuda, amp=true, ONNX roundtrip PASS.
```

CUDA two-iteration E2E smoke:

```text
command: training/.venv/bin/python training/rebel_orchestrator.py --out-dir /tmp/rebel-cuda-e2e-smoke --iterations 2 --smoke --games 1 --particles 2 --search-iterations 4 --rollout-steps 5 --max-steps 20 --model-side player --workers 1 --epochs 1 --device cuda --amp --skip-gates --selfplay-device cuda --selfplay-inference-batch-size 1 --neural-leaf-weight 1.0
result: COMPLETED; iteration 0 trained/exported on CUDA; iteration 1 loaded iteration-0 ONNX with device=Cuda { device_id: 0 }, generated neural-leaf rows, replay-mixed 7 rows, trained/exported on CUDA.
iteration 1 sampled diagnostics: neuralLeafWeight=1.0, neuralLeafBatchRows=4, neuralLeafCalls=4, rolloutLeafCalls=0, neuralValue=-0.118075259.
```

Neural worker determinism:

```text
config: --seeds 4 --seed-start 19000 --model-side both --particles 2 --search-iterations 4 --rollout-steps 5 --max-steps 30 --onnx-path /tmp/rebel-e2e-export-smoke/policy.onnx --device cpu --neural-leaf-weight 1.0 --inference-batch-size 4
workers 1/4 sha256: 2660b66cb66dee3b0ecface0124312a34884dad8968591be2b428f3843163058
```

Two-stage ReBeL loop smoke:

```text
command: python training/rebel_orchestrator.py --out-dir /tmp/rebel-two-stage-smoke --iterations 2 --smoke --games 1 --particles 2 --search-iterations 4 --rollout-steps 5 --max-steps 20 --model-side player --workers 1 --epochs 1 --device cpu --skip-gates --selfplay-inference-batch-size 4 --neural-leaf-weight 1.0
result: iteration 0 generated rollout-bootstrap rows, trained, exported ONNX; iteration 1 loaded iteration-0 ONNX for self-play, trained from the neural-leaf rows, and exported a new ONNX.
iteration 1 row diagnostics: neuralLeafWeight=1.0, neuralLeafBatchRows=4, neuralLeafCalls=4, rolloutLeafCalls=0 for sampled rows.
```

Cross-iteration replay and fixed KL anchor:

- The multi-iteration orchestrator now materializes a mixed training JSONL from current self-play plus prior loop vintages (`--cross-iter-replay`, `--replay-window`, `--replay-old-fraction`).
- `--fixed-kl-anchor` keeps the KL regularizer anchored to the original warm-start/promoted checkpoint instead of chasing the moving previous checkpoint.
- The training command receives the loop event sink, so train/eval events are part of the same per-iteration trace.
- Promoted checkpoints/ONNX files are snapshotted under `loop/pool/iter-XXX/` so later runs can consume a stable policy pool.
- `--selfplay-vs-pool` can generate ReBeL rows from a bounded promoted policy pool, using Wilson-lower-based PFSP weights with `--pool-size`, `--pfsp-floor`, and optional `--pool-state-file`. Rows are annotated with the pool policy checkpoint/ONNX provenance.

Replay/KL smoke:

```text
command: python training/rebel_orchestrator.py --out-dir /tmp/rebel-replay-kl-smoke --iterations 2 --smoke --games 1 --particles 2 --search-iterations 4 --rollout-steps 5 --max-steps 20 --model-side player --workers 1 --epochs 1 --device cpu --skip-gates --selfplay-inference-batch-size 4 --neural-leaf-weight 1.0
result: iteration 1 trained on rebel-train-mixed.jsonl with 4 current rows + 3 replay rows; kl_anchor_checkpoint remained the iteration-0 checkpoint.
```

CUDA throughput sweep:

```text
workload: sim-rebel-selfplay --seeds 64 --model-side player --particles 128 --search-iterations 16 --rollout-steps 20 --max-steps 80 --workers 8
rollout CPU, no ONNX:     1.54s, 368 rows, 239.0 rows/s, rolloutLeafCalls=163712
ONNX leaf CPU, batch=1:   3.01s, 386 rows, 128.2 rows/s, neuralLeafBatchRows=168232, rolloutLeafCalls=0
ONNX leaf CUDA, batch=1:  1.65s, 386 rows, 233.9 rows/s, neuralLeafBatchRows=168232, rolloutLeafCalls=0, avg/max GPU util=26%/53%
ONNX leaf CUDA, batch=32: 6.83s, 386 rows,  56.5 rows/s, neuralLeafBatchRows=168232, rolloutLeafCalls=0, avg/max GPU util=27.8%/32%
ONNX leaf CUDA, batch=128: 2.39s, 386 rows, 161.5 rows/s, neuralLeafBatchRows=168232, rolloutLeafCalls=0, avg/max GPU util=23.1%/34%
```

Interpretation:

- ReBeL model-valued search now uses CUDA for the hot value-head path when `--onnx-path ... --device cuda --neural-leaf-weight 1.0` is set.
- On this measured search shape, inline CUDA (`--inference-batch-size 1`) is best because each public-belief decision already submits a large particle/action matrix as one batch. Dispatcher batching is still available for cross-worker coalescing, but it is not the right default for this workload yet.
- Rollout-only CPU can still be competitive because the current rollout evaluator is lightweight. The gain from CUDA is that ideal/ReBeL-style neural value search is no longer CPU-bound by per-leaf model calls; it replaces 168k rollout leaves with 168k GPU-backed value predictions at roughly rollout-only wall time on this slice.

Device resolution:

- `rebel_orchestrator.py --device auto` resolves to CUDA when the training Python can import Torch and `torch.cuda.is_available()` is true.
- `--selfplay-device auto` follows the resolved training device, so a CUDA training run also uses CUDA ONNX Runtime for ReBeL self-play once a self-play ONNX is available. Override with `--selfplay-device cpu` when intentionally keeping search inference on CPU.

Production gates:

- `rebel_orchestrator.py` now forwards `--gate-min-games`, `--gate-min-ci-lower`, and `--gate-min-win-rate` to both fixed and uniform/deck-diverse `sim-eval-gate` runs.
- Gate failures are parsed from the written manifest and recorded in the iteration decision instead of crashing the orchestrator before promotion logic can reject the checkpoint.

Search-iteration budget:

- `run_public_belief_search` now uses `RebelSearchConfig.iterations` as repeated leaf samples per particle/action cell.
- Diagnostics now report `particleActionEvaluations = particles * legal_actions * search_iterations`; rollout-only search increments `rolloutLeafCalls` by the same count, while neural-leaf search batches the expanded leaf matrix and reports matching `neuralLeafBatchRows` when all leaves are model-evaluable.
- Determinism check after wiring the budget: `--workers 1` and `--workers 4` produced identical JSONL SHA-256 `d719a66bd77345cadb669a9005f6eadb2a38192aa43c9fb702cc249bc5b1ce3c` for `--seeds 4 --model-side both --particles 2 --search-iterations 4 --rollout-steps 5 --max-steps 30`.

Remaining work:

- Calibrate the production threshold values for the neural-leaf recipe; the orchestrator plumbing now supports enforcing them.

## 2026-05-29 — Re-baseline at hidden=256/depth=4 (R18 live profile)

**Why this section exists:** the CUDA sweep above (lines ~411–417) was measured on
a single-worker smoke (386 rows) at the *old* model size. The model transitioned
to `hidden=256 / depth=4`, which invalidates the "inline `--inference-batch-size 1`
is best / GPU caps ~28%" verdict for the production shape. Do not cite those rows
as the current optimum.

**Live profile (R18, `runs/R18-rebel-kl-relaxed-20260528`, iter-2 self-play, completed):**
config `--particles 32 --search-iterations 32 --rollout-steps 96 --max-steps 220
--workers 24 --neural-leaf-weight 0.8 --inference-batch-size 512
--inference-max-wait-us 40000 --device cuda`.

- Throughput: **2.05 games/s, 52.8k inference rows/s** (768 games / 373.9s).
- Dispatcher: batches fill to 512 **97.5%** of the time; `run_us` avg 4.6ms;
  ~105.7 batches/s ⇒ GPU forward duty ≈ **49% of wall** (CPU-bound, GPU ~half idle).
  `max_run_us` hits ~39ms — the `--inference-max-wait-us 40000` tail: partial
  batches (fill 1–3) stall a full 40ms. Lowering max-wait caps that tail.

**Crux — rollout cost at `neural-leaf-weight 0.8`:** the rollout gate is
`use_rollouts = neural_leaf_weight < 1.0` (`engine-rs/crates/engine/src/rebel/mod.rs:123-125`),
so at 0.8 the full `--rollout-steps 96` random rollout runs for **every** leaf cell
and is then weighted only **20%** in the blend (`rebel/mod.rs:225-231`). We pay
100% of rollout CPU for a 20%-weighted signal.

**Measured rollout-step scaling** (CPU-only, `--neural-leaf-weight 0`, no ONNX, 6
workers, 32 games, production search shape; ran alongside the live run without
perturbing it — selfplay is `--device cpu` so zero GPU contention):

```text
rollout-steps  games/s   elapsed
96             1.698     18.8s   (baseline)
24             3.177     10.1s   (+87%)
```

Linear fit `cost = fixed + k·steps`: rollout is **~62% of CPU self-play time at 96
steps**; fixed search/belief/CFR/clone overhead ~38%. Cutting 96→24 nearly doubles
CPU-side throughput.

**Recommendations for the NEXT run (cannot be applied to the live run — orchestrator
args are fixed at launch):**

1. **Lowest-risk, biggest win:** drop `--rollout-steps 96 → 24-32`. Rollout is only
   20%-weighted at `leaf=0.8`, so quality cost is small; CPU saving is large
   (measured ~+87% CPU-side; expect ~+40-60% end-to-end once neural inference is
   included, since it dilutes the rollout share).
2. **Cap dispatch tail:** `--selfplay-inference-max-wait-us 40000 → ~5000`. Removes
   the 40ms stalls on the ~2.5% of batches that never fill.
3. **Cheap:** `--workers 24 → 28-30` (32-core box; CPU-bound regime has headroom).
4. **Quality-gated option:** `--neural-leaf-weight 1.0` removes rollouts entirely →
   GPU-bound regime → then dispatch tuning (batch-size/max-wait) matters. Validate
   gate win-rate does not regress before adopting.

**Harness:** `training/bench_rebel_selfplay_throughput.py` runs the empirical
re-baseline (`--preset rebaseline`): one-factor sweeps of rollout-steps, workers,
max-wait (Regime A, leaf<1, CPU-bound) and inference-batch-size (Regime B, leaf=1.0,
GPU-bound), sampling live GPU util. It **refuses CUDA configs while a live
sim-rebel-selfplay/sim-eval-gate is running** (`--force` to override), so the heavy
sweep can only run once the live run frees the GPU. Run it then to settle items 1-4
with production-scale numbers.

## 2026-05-29 — Correctness sweep (info-incorrect leaf + mistargeted value head)

A correctness audit of the loop (not throughput) found and fixed the following.
These ship together; a corrected run must start fresh (the live R18 run was
trained on the pre-fix targets and is not retro-fixable).

1. **Value head trained on the wrong field (headline bug).**
   `training/uma_ai/rebel_dataset.py` read `beliefValue` (the search's own
   bootstrapped, rollout-contaminated, partly self-referential value) because
   `dict.get("beliefValue", example.get("valueTarget", 0))` always finds
   `beliefValue` — the grounded MC outcome `valueTarget` (+1/-1/0 by winner,
   observer perspective) never fired. **Fix:** train the value head (and the
   q-head scalar-value consistency term) on `valueTarget`; raise if absent.
   Decision: the value target is the **grounded MC outcome**, not a bootstrap —
   correct for this depth-1, no-recursive-solve setup.

2. **Determinized rollout leaf is information-incorrect (strategy fusion).**
   `rollout_leaf_value_for_state` plays out a fully-instantiated particle world
   (opponent hidden cards revealed) with a fixed heuristic, then averages over
   particles — PIMC strategy fusion, the pathology ReBeL's belief value exists
   to avoid. The neural/observation leaf (opponent hand masked, `observation.rs`)
   is the only sound leaf. **Fix:** rollout is now OFF by default and reachable
   only via an explicit `--allow-rollout-leaf` ablation. `--neural-leaf-weight`
   default stays 1.0; the orchestrator rejects `< 1.0` without the ablation flag
   (`validate_leaf_args`). The previous silent `inference::global().is_none()`
   fallback to rollout, and the silent zero-leaf reset on inference error, now
   **panic** instead of emitting corrupted targets.

3. **Replay-vintage guard.** Cross-iteration replay now drops old rows whose
   `rolloutLeafUsed` is true or absent (pre-fix / contaminated), logging counts,
   so rollout-contaminated policy/Q targets cannot leak into a corrected mix.

4. **Info-set invariant test now exists** (was only a requirement at
   "Wave/CFR batching" / e2e §F): `rebel::tests::info_set_invariant_under_particle_permutation`
   installs a public-only value oracle and asserts `root_policy` is invariant to
   particle permutation with zero rollout calls at weight 1.0. Coverage caveat:
   it checks permutation invariance exactly; full distinct-sample invariance is
   not bit-checkable (post-action public observations can legitimately differ).
   The decision side was already info-set-clean by construction (policy is a
   particle-aggregated public distribution); this guards against regression.

5. **Nits:** played move is now `argmax(root_policy)` (on-policy with the
   distilled target, was `argmax(Q)`); `temperature_value` now actually scales
   the played-move sampling (`p^(1/T)`), recorded target unscaled; the
   `searchIterations` diagnostic reports the effective budget; the CFR block is
   honestly labeled (regret-matching over a fixed value vector ≈ smooth argmax,
   NOT equilibrium CFR — the `public-belief-cfr-v1` wire string is kept for
   schema continuity). New `--allow-rollout-leaf` and `rolloutLeafUsed` /
   `allowRolloutLeaf` / `effectiveNeuralLeafWeight` provenance are stamped.
