# r12 Loop Throughput — Spike Candidates

- **Date:** 2026-05-21
- **Status:** LANDED 2026-05-21 — pointer to `docs/ai-research/analysis/r12-loop-throughput.md`. Headline: GPU util mean 0.26% (98.5% of wall samples = 0%) confirms idle, but the bottleneck is HTTP-RTT between Rust sims and serve_onnx, NOT serve_onnx provider — candidate A (cpu→cuda) refuted, candidate B (NAPI in-process predict) recommended first spike. Body below preserved as historical pre-completion scoping context. Tracked in queue under `throughput-optimization-spike`.

## Why

User-requested. The C8-W6FIX-ON 5-iter run took ~17 min wallclock — fine for one experiment, painful for a multi-knob HP sweep (10 W6-fix knob combinations × 17 min = ~3 hrs). Doubling throughput would change which experiments are tractable.

## Mid-flight observations (T+~6 min, iter-0 of extension complete)

Sourced from `runs/R16-P2-c8-w6fix-on-extended/profile-{gpu.csv,cpu.txt,mem.txt}` (live-writing) and the prior 5-iter run's `orchestrator-state.json`.

### 1. GPU sits idle the entire loop

`nvidia-smi` samples consistently report `0% util / 0 MiB used / 4.4 W idle power` during selfplay + gate. Distill runs on GPU briefly (~4s/iter) but the long-pole stages (selfplay ~70s, gate ~134s = ~95% of wall) are CPU-only because `serve_onnx --provider cpu` is the default.

**Root cause:** `--provider cpu` hard-coded at `training/dagger_orchestrator.py:949` (selfplay-serve) and `:1001` (gate-serve). Both `r12_orchestrator.py` and `ppo_orchestrator.py` import `serve_onnx_context` and inherit the default.

**Spike candidate A — flip provider to CUDA:** One-line edit per call site (or expose as a CLI flag plumbed through `args`). Risk: serve_onnx and distill compete for GPU memory, but they don't run concurrently (distill happens between teardown of selfplay-serve and stand-up of gate-serve). Expected payoff: depends on whether inference compute or HTTP round-trip dominates serve_onnx latency for this model size (state_dim=110, hidden=64, depth=2). For a small model, HTTP RTT may dominate — see candidate B.

### 2. Gate (~134s) is ~2× selfplay (~70s) despite the same MCTS recipe

Both run 60 games × 100 sims × `--mcts-leaf rollout`. Selfplay is single-process via Rust binary (no `--workers`). Gate uses `--workers 24` and the Rust binary too. The 2× delta is unexplained at first glance — needs further inspection.

**Hypothesis 1:** Gate has model-side rotation × heuristic-rule-bot overhead absent from selfplay (selfplay is model-vs-model). The rule-bot decisions are cheap but the side-swap doubles the inference count if both sides use MCTS.

**Hypothesis 2:** Rust `sim-eval-gate --workers 24` isn't actually parallelizing the way expected — single-process under the hood (per Slice 1 finding that Rust `--workers` is ignored). The 24× value isn't being realized.

**Spike candidate B — eliminate HTTP round-trip via NAPI:** The `rust-port-backend-napi-consumer` queue entry (P3 today) already scopes this. NAPI gives in-process predict at 7 μs/step vs HTTP RTT at ~ms. If HTTP RTT dominates, the win is multiplicative across all sims × all games × all iters.

### 3. distill is fast (~4s) but uses GPU exclusively

Not a bottleneck. No spike candidate here.

## Tentative candidate ranking (pre-completion)

| Candidate | Effort | Expected speedup |
|---|---|---|
| A — flip serve_onnx to CUDA provider | trivial (1-2 line edit + flag plumb) | 1.5-3× on inference, if inference compute dominates |
| B — NAPI in-process predict (eliminate HTTP RTT) | moderate (P3 queue item; engine-rs/crates/napi-bridge/ already exposes the surface) | 2-10× on inference, if RTT dominates |
| C — investigate gate=2×-selfplay gap | small (read + maybe a small fix) | 1.5-2× on gate |
| D — Rust `--workers` actually parallelizes | medium (re-architect single-process Rust sim) | 2-5× on selfplay+gate |

Final ranking + numeric utilization data lands in `docs/ai-research/analysis/r12-loop-throughput.md` post-completion of the extended run.

## Out-of-scope for this spike

- Bigger model (changing state_dim/hidden/depth) — different concern; out of "loop throughput" optimization.
- Distill GPU efficiency — distill is ~4s/iter, not load-bearing.
- Rewriting r12_orchestrator.py architecture — a spike is bounded; multi-week rewrites are not.

## When the extended run completes

Process the three `profile-*` files into a single analysis doc:
- Per-iter wall budget breakdown (selfplay / distill / gate as % of total)
- GPU util mean/peak across the run (expected: ~0% during selfplay+gate; spike during distill)
- CPU util mean across the run (expected: high during selfplay+gate)
- Memory pressure (expected: low — small model)
- Concrete numeric estimates for candidates A-D
- Recommendation on which to spike first
