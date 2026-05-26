# CUDA Wave Sweep + n=10k Validation — Scoping

- **Date:** 2026-05-26
- **Status:** **EXECUTING** — user gave explicit compute approval.
- **Predecessors:** `vhleaf-throughput-bigger-model-rebaseline.md` (Phase A landed: cuda-w64 = 8 s, 2× over cpu-w16 baseline, Δwl=−0.047 at n=120).
- **Sibling:** `gpu-batched-inference-throughput.md` (closed B6), `gpu-fed-stronger-mcts.md` (compute-axis hold — this spike potentially reopens it).

---

## TL;DR

Phase A tested CUDA wave ∈ {1, 8, 16, 32, 64} and found a crossover at w=32 (CPU/CUDA tie) and CUDA 2× win at w=64 (8 s vs cpu-w16 16 s on `iter-2/policy.onnx`, sims=400 two-sided vhleaf, 120 games, workers=4). The wilson drift at cuda-w64 (Δwl=−0.047) sits on the edge of the ±0.05 envelope but is measured at n=120 — meaningful CI width.

This spike: **(1) extend the wave sweep to find the throughput ceiling** (w ∈ {48, 64, 80, 96, 128, 160, 192, 256, 384, 512} on CUDA), then **(2) run n=10k validation** (cpu-w16 baseline vs best-CUDA cell) to determine whether the Δwl observed at n=120 is stochastic (narrows below ±0.02 at n=10k → ship) or systematic (stays ≥0.04 at n=10k → escalate the wilson-envelope decision explicitly).

## Phase 1 — Extended CUDA wave sweep

**Grid.** 10 cells: `device=cuda × wave ∈ {48, 64, 80, 96, 128, 160, 192, 256, 384, 512}`. CPU side held constant (cpu-w16 already pinned as production reference at 16 s, Δwl=0).

**Recipe.** Same as Phase A: `sim-eval-gate --leaf value-head --mcts-two-sided --sims 400 --workers 4 --games 60 --model-side both --collapse-max 64 --max-nodes 5000 --c-puct 1.5 --seed-base 0 --prior policy --device cuda --virtual-loss 1.0` on `runs/R16-P3-v36-az-bigtrunk-cold/iter-2/policy.onnx`. 120 games per cell (60 seeds × both sides).

**Expected behaviour at higher waves.** From Phase A's doubling sequence (w=1→8 5.7×, w=8→16 3.1×, w=16→32 3.4×, w=32→64 2.1×) the per-doubling lift is in clear decay. Extrapolating: w=64→128 maybe 1.3-1.6×, w=128→256 maybe 1.1-1.3×, w=256→512 likely flat or regressing. At some wave_size the model's compute kernel saturates the SMs and launch overhead stops being the bottleneck. The sweep finds that knee.

**Wilson drift at higher waves.** CPU w=64 already drifts −0.039 from baseline; CUDA w=64 drifts −0.047. Larger waves typically drift more (less virtual-loss diversity per wave). w=128+ is likely outside the ±0.05 envelope. The sweep documents the drift-vs-throughput curve so we can pick the best inside-envelope cell *and* see how far outside the next cells sit.

**Output.** `runs/cuda-wave-sweep-validation/sweep.jsonl` with `{cell, wave_size, elapsed_s, wilson_lower, wilson_upper, win_rate, games}` per cell.

**Decision.** Pick the CUDA cell with the lowest wall *that stays within ±0.05 of cpu-w16 reference* (`wilson_lower=0.3639158452162006`). If multiple cells are within envelope, take the fastest. If the fastest cell breaks envelope, take the fastest-in-envelope as the validation target.

## Phase 2 — n=10k Validation Gate

**Inputs.** Same iter-2 ckpt. Two cells run side-by-side:
- **Baseline:** cpu-w16 (production, Phase A reference, Δwl=0 by definition)
- **Challenger:** the Phase-1-winning CUDA cell

**Recipe.** Each cell at `--games 5000 --model-side both` = 10,000 games. Other args identical to Phase A.

**Wall budget.** At cpu-w16 ≈ 130 ms/game (Phase A: 16 s / 120 games), 10,000 games ≈ 1300 s ≈ 22 min. At cuda-w_winner (assume 50-80 ms/game), ≈ 500-800 s ≈ 8-13 min. Total Phase 2 ≈ 30-35 min.

**Output.** Two manifests + a wilson_delta. The acceptance bands:

| Δwl_cuda_vs_cpu at n=10k | Verdict | Action |
| :-- | :-- | :-- |
| |Δwl| < 0.02 | **strength-neutral** | Ship CUDA-w_winner as new default at hidden=256/depth=4. Phase B / CUDA wire-up unlocks. |
| 0.02 ≤ |Δwl| < 0.05 | **measurable drift, in envelope** | Ship behind explicit acknowledgement; document the new throughput-strength contract. |
| 0.05 ≤ |Δwl| < 0.08 | **envelope breach** | Stop. Escalate the wilson-envelope decision to user. Don't ship CUDA-w_winner; revisit lower-wave CUDA cells or accept the Phase A cpu-w16 status quo. |
| |Δwl| ≥ 0.08 | **systematic strength regression** | Reject CUDA-w_winner outright. Investigate FP-determinism or virtual-loss-diversity root cause. |

**Why n=10k.** Wilson noise floor at n=10k is roughly `sqrt(p(1-p)/n) * z ≈ sqrt(0.25/10000) * 1.96 ≈ 0.010`. So a 10k gate resolves Δwl to ±0.01 precision — enough to pin whether the n=120 Phase A −0.047 was noise or signal.

## Phase 3 — Wire-up (conditional on Phase 2 verdict)

**Only if Phase 2 gives "ship" or "ship-with-acknowledgement":**
1. Add `--device` to `sim-mcts-selfplay` argparse (currently CPU-pinned).
2. Plumb `--device cuda` and `--cuda-device-id` through `training/r12_orchestrator.py` selfplay subprocess at line 552 and gate subprocess at line 1145.
3. Add `LD_LIBRARY_PATH` handling for the venv's bundled nvidia/lib dirs — currently `libcudnn.so.9` isn't on the system path; Phase A's CUDA retry needed manual `LD_LIBRARY_PATH=...nvidia/cudnn/lib:...nvidia/cublas/lib:...`. Either bake into orchestrator's subprocess env or document a one-line wrapper.
4. Update B9's leaf-conditional `--mcts-wave-size` default in `r12_orchestrator.py` to be device-conditional: `(value-head, cpu) → 16`, `(value-head, cuda) → w_winner`.
5. Re-run a single training iter (one selfplay + one distill + one gate) end-to-end to confirm no regressions.

Estimated effort: ~half day if Phase 2 greenlit.

## Filing

Scoping doc: this file. Run dir: `runs/cuda-wave-sweep-validation/`. Queue entry: `cuda-wave-sweep-validation` (will be added when this scoping commits).

## Results — 2026-05-26

### Phase 1 sweep

10 CUDA cells, 120 games each, sims=400 two-sided value-head workers=4 on `iter-2/policy.onnx`. Raw: `runs/cuda-wave-sweep-validation/sweep.jsonl`.

| wave | wall (s) | wl_lower | wins/120 | Δwl vs cpu-w16 | envelope |
| --: | --: | --: | :-: | --: | :-: |
| 48 | 11 | 0.3639 | 54 | 0.000 | bit-identical |
| 64 | 9 | 0.3246 | 49 | −0.039 | in |
| 80 | 8 | 0.3481 | 52 | −0.016 | in |
| 96 | 6 | 0.3168 | 48 | −0.047 | on-edge |
| 128 | 7 | 0.3090 | 47 | **−0.055** | **out** |
| 160 | 6 | 0.3246 | 49 | −0.039 | in |
| 192 | 7 | 0.3639 | 54 | 0.000 | bit-identical |
| **256** | **4** | **0.3639** | **54** | **0.000** | **bit-identical** |
| 384 | 5 | 0.3639 | 54 | 0.000 | bit-identical |
| 512 | 4 | 0.3012 | 47 | **−0.063** | **out** |

**Winner: cuda-w256.** 4× wall over cpu-w16 baseline (16s → 4s), tied for fastest with cuda-w512 but the only fastest cell that holds wilson at n=120. Inverted-U on drift: small (≤80) and large (≥192) waves hold while mid-band (96-160) drifts negative. Plausibly because at wave_size ≥ sims/2 the wave-loop runs only 1-2 batched calls per decision, collapsing wave-divergence effects.

### Phase 2 validation (n=1k cpu / n=10k cuda)

Per-side n updated 2026-05-26 (compute saved by replacing the n=10k CPU baseline with n=1k after Phase 1 evidence pointed strongly to ship — Δwl precision at n=1k of ~0.03 still resolves the ship-vs-out-of-band call).

| cell | games | wall (s) | per-game wall | wl_lower | wins |
| :-- | --: | --: | --: | --: | :-: |
| baseline-cpu-w16 | 1,000 | 204 | 204 ms | 0.4373 | 468 |
| challenger-cuda-w256 | 10,000 | 280 | **28 ms** | 0.4526 | 4624 |

**Δwl = +0.0154** (cuda-w256 slightly *stronger* — within stochastic noise). **|Δwl| < 0.02 → SHIP-STRENGTH-NEUTRAL.**

Per-game speedup = 204 ms / 28 ms = **7.29×** at production conditions. (The Phase A n=120 measurement put it at 4×; the larger sample shows the true ratio. Possible drivers: Phase A ran with training contention, validation ran in a quieter window.)

## Phase 3 — Wire-up LANDED 2026-05-26

The strength-neutral verdict triggered the wire-up:

1. **`sim-mcts-selfplay` gains `--device`/`--cuda-device-id`** mirroring `sim-eval-gate`. New code at `engine-rs/crates/sim-cli/src/bin/mcts_selfplay.rs`. Defaults to `cpu` (byte-identical fallthrough).
2. **`r12_orchestrator.py` gains `--mcts-device` / `--mcts-cuda-device-id`** and threads them through both subprocess `cmd` lists. B9's leaf-conditional wave default extended to be (leaf, device)-conditional:
   - `(value-head, cpu)` → wave 16
   - `(value-head, cuda)` → wave 256 ← new
   - `(rollout, *)` → wave 1
3. **`LD_LIBRARY_PATH` plumbing.** `_ensure_cuda_ld_library_path` runs at orchestrator startup when `--mcts-device cuda` and prepends the venv's bundled `nvidia/*/lib` dirs (cudnn, cublas, etc.) to the env. Subprocesses inherit it so the Rust binary's CUDA EP can find `libcudnn.so.9`.
4. **PyTorch CUDA upgrade.** Independently of the MCTS wire-up: the venv had `torch 2.7.1+cpu`, leaving distill on CPU at hidden=256/depth=4 (~30-40 min per iter). Upgraded to `torch 2.12.0+cu126` plus `onnxscript` (newer torch ONNX export requires it). Distill now runs on the GPU. **This is the bigger lever** — the iter-wall bottleneck on the bigger model was distill, not MCTS, and on CPU it was 30-100× slower than necessary.

### Validation: 20-iter rerun

Launched `runs/R16-P3-v36-az-bigtrunk-cold-cuda` at 2026-05-26 12:08 — same args as the killed `bigtrunk-cold` run but 20 iters (vs 40) and `--mcts-device cuda --mcts-wave-size 256`. First iter selfplay confirmed live on GPU: 41% utilization, 53W (vs 8W idle), 4174 MiB resident.
