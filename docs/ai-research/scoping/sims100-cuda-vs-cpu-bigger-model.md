# sims=100 CUDA-vs-CPU Re-validation (bigger model) — Scoping

- **Date:** 2026-05-26
- **Status:** **CLOSED-RECORD-ONLY-2026-05-26** — point measurement, no wire-up. Documents that the old `gpu-batched-inference-throughput.md` claim "CUDA loses at sims=100" is no longer true on the current model size (hidden=256/depth=4); CUDA at wave≥64 wins.
- **Predecessors:**
  - `gpu-batched-inference-throughput.md` (CLOSED 2026-05-25 on hidden=128: "wave_size=16 on CPU runs vhleaf sims=100 in 1.02s vs serial CPU 7.45s; CUDA at any wave_size still loses to wave-batched CPU on every recipe").
  - `cuda-wave-sweep-validation.md` (LANDED 2026-05-26 on hidden=256/depth=4 sims=400: cuda-w256 wins 7.29× per-game wall, strength-neutral at n=10k).
  - `vhleaf-throughput-bigger-model-rebaseline.md` (the model-size transition that invalidated the prior claim).

---

## Recipe

`sim-eval-gate --leaf value-head --mcts-two-sided --sims 100 --workers 4 --games 60 --model-side both --collapse-max 64 --max-nodes 5000 --c-puct 1.5 --seed-base 0 --prior policy --virtual-loss 1.0` on `runs/R16-P3-v36-az-bigtrunk-cold/iter-2/policy.onnx` (hidden=256, depth=4, state-dim=246).

## Results

| cell | wall | per-game | speedup vs cpu-w16 | wins/120 | wilsonLower | Δwl vs cpu-w16 |
| :-- | --: | --: | --: | --: | --: | --: |
| cpu-w16 | 3.85s | 32 ms | 1.00× | 59 | 0.4039 | — |
| cuda-w16 | 8.31s | 69 ms | 0.46× | 59 | 0.4039 | 0.000 |
| cpu-w64 | 4.69s | 39 ms | 0.82× | 48 | 0.3168 | −0.087 |
| cuda-w64 | 2.57s | 21 ms | 1.50× | 48 | 0.3168 | −0.087 |
| cpu-w128 | 4.52s | 38 ms | 0.85× | 54 | 0.3639 | −0.040 |
| **cuda-w128** | **2.18s** | **18 ms** | **1.76×** | **53** | **0.3560** | **−0.048** |

Raw: `runs/sims100-cpu-vs-cuda/{cpu,cuda}-w{16,64,128}/{manifest.json,log}`.

## Findings

1. **Old claim FALSIFIED on bigger model.** `gpu-batched-inference-throughput.md` established "CUDA loses at sims=100" on hidden=128. On hidden=256/depth=4, CUDA at wave≥64 wins. The compute-vs-launch-overhead ratio shifted in CUDA's favor with the 2.8× larger model.

2. **Strength drift is wave-size effect, NOT device effect.** cuda-w64 and cpu-w64 are wilson-IDENTICAL (48/120, wl=0.3168). cuda-w128 and cpu-w128 are within 1 game (53 vs 54, wl 0.356 vs 0.364). Same envelope behavior `cuda-wave-sweep-validation.md` Phase 2 characterized at n=10k: wave>32 with virtual_loss=1.0 drifts at small n; precedent shows drift narrows at n=10k.

3. **Apples-to-apples cuda-w128 vs cpu-w128: cuda 2.07× wall at iso-strength.** The cleanest answer at sims=100.

4. **cuda-w16 loses badly (0.46×)** — at small wave_size the CUDA launch overhead per call dominates. Use wave≥64 on CUDA or stay on CPU at small wave.

## What this rules out / doesn't

- **Ruled out:** the lingering belief from `gpu-batched-inference-throughput.md` that "CUDA can't beat CPU at sims=100." That was a model-size-specific finding, not a general truth.
- **Not ruled out:** strength envelope at production scale. n=120 wilson noise is ±0.089, so the cuda-w128 Δwl of −0.048 is within ~0.5σ. For an in-flight production-scale run, the `cuda-wave-sweep-validation.md` precedent applies — strength likely narrows.

## Practical implication (no auto-action)

The in-flight `R16-P3-v36-az-15k-shallow-cpu` orchestrator (launched 2026-05-26) uses `--mcts-device cpu --mcts-wave-size 16` at sims=100 / hidden=256-depth=4. Switching to `--mcts-device cuda --mcts-wave-size 128` would give ~1.78× selfplay wall on the iter-2 model with strength delta in the cuda-wave-sweep envelope. Not auto-applied — user's call whether to mid-flight-switch the run or apply on next launch.

## Why "CLOSED-RECORD-ONLY"

This is a point measurement that overturns a stale claim, not a sprint. No wire-up because cuda-mid-flight-switch is a user decision and the orchestrator default-flip from `cuda-wave-sweep-validation.md` Phase 3 already covers sims=400 (the prior recipe). If a future run pins sims=100, it should pass `--mcts-device cuda --mcts-wave-size 128` explicitly.
