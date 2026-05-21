# r12 Loop Throughput Analysis — C8-W6FIX-ON extended baseline

- **Date:** 2026-05-21
- **Run analyzed:** `runs/R16-P2-c8-w6fix-on-extended/` (10 iters, 33.4 min wallclock, Rust default engine, workers=24, mcts-sims=100, 60 selfplay games / 60+60 side-balanced eval games, state-dim=110 / hidden=64 / depth=2)
- **Source data:** `profile-gpu.csv` (n=396 @ 5s; 54 `[Unknown Error]` util rows dropped, mem/temp preserved), `profile-cpu.txt` (n=401 top samples @ 5s; even-numbered cores 0,2,…,30), `profile-mem.txt` (n=402 free samples @ 5s), `loop/orchestrator-state.json` per-iter `selfplay_elapsed_sec`/`distill_elapsed_sec`/`gate_elapsed_sec`.
- **Scoping framework:** `docs/ai-research/scoping/r12-throughput-spike-candidates.md` (LANDED — pointer to this file).

## Per-iter wall budget

| iter | selfplay s | distill s | gate s | total s | sp% | di% | ga% |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 73.8 | 4.1 | 120.1 | 198.0 | 37.3% | 2.1% | 60.6% |
| 1 | 63.2 | 4.2 | 121.9 | 189.3 | 33.4% | 2.2% | 64.4% |
| 2 | 60.1 | 4.5 | 124.1 | 188.6 | 31.8% | 2.4% | 65.8% |
| 3 | 68.7 | 4.6 | 121.9 | 195.2 | 35.2% | 2.4% | 62.4% |
| 4 | 75.5 | 4.5 | 123.5 | 203.5 | 37.1% | 2.2% | 60.7% |
| 5 | 70.9 | 4.4 | 120.8 | 196.1 | 36.2% | 2.2% | 61.6% |
| 6 | 62.3 | 4.6 | 125.4 | 192.3 | 32.4% | 2.4% | 65.2% |
| 7 | 71.0 | 4.6 | 126.2 | 201.8 | 35.2% | 2.3% | 62.5% |
| 8 | 66.7 | 4.6 | 124.7 | 196.0 | 34.0% | 2.3% | 63.6% |
| 9 | 81.3 | 4.8 | 131.4 | 217.5 | 37.4% | 2.2% | 60.4% |

- **Mean ± stddev:** selfplay 69.3 ± 6.5s · distill 4.49 ± 0.22s · gate 124.0 ± 3.3s · total 197.8 ± 8.4s.
- **Aggregate share (sum across 10 iters = 1978.3s instrumented):** selfplay 35.1% · distill 2.3% · gate 62.7%.
- **Wrapper/orchestrator overhead** (wall − Σ instrumented): 24.4s = 1.2% of wall. Negligible.
- **Gate is 1.79× selfplay** (124.0s vs 69.3s) for 2× the games (120 vs 60). So per-game wall: selfplay 1.16 s/game, gate 1.03 s/game — gate is actually *faster* per game because the rule-bot side is cheap heuristic (no MCTS). The headline "gate is ~2× selfplay" from the scoping doc is explained by game count alone, not by hidden eval-gate overhead.

## GPU utilization

| metric | value |
|---|---|
| samples (util valid) | 342/396 |
| util.gpu mean | **0.26%** |
| util.gpu peak | 26% |
| util=0% samples | 337/342 = **98.5%** |
| util ≥ 50% samples | 0/342 = **0.0%** |
| memory.used mean | 6 MiB |
| memory.used peak | 356 MiB |
| power.draw mean | 15.66 W (≈ idle) |
| power.draw peak | 39.26 W |

**Per-stage GPU util (aggregated across iters):**

| stage | n samples | mean util | peak | % util=0 | mem peak |
|---|---:|---:|---:|---:|---:|
| selfplay | 121 | 0.00% | 0% | 100.0% | 0 MiB |
| distill | 7 | 0.00% | 0% | 100.0% | 0 MiB (sampled too rarely; distill is 4.5s vs 5s sample period) |
| gate | 210 | 0.43% | 26% | 97.6% | 356 MiB |

The five nonzero util samples (3%, 12%, 23%, 26%, 26%) all fall in gate windows of iters 1/3/6/7 — almost certainly the brief `serve_onnx_context` stand-up/teardown ONNX-runtime initialization, not steady-state inference. Peak memory 356 MiB is consistent with the small ONNX graph loading momentarily.

Distill's GPU footprint is invisible because each distill is ~4.5s and the GPU sampler runs at 5s — it slips between samples. Distill is also only 2.3% of wall, so unmeasured GPU activity there is bounded by ~45s/2002s × 100% = 2.2% of overall samples worst-case. The headline GPU=0.26% mean is robust.

## CPU utilization

`top -b -d 5 -c -1` reported even-numbered cores 0,2,…,30 (16 distinct core IDs from a 32-thread host).

| metric | value |
|---|---|
| system mean-busy% across 16 reported cores | **5.4%** (stddev 1.3%) |
| avg # cores ≥80% busy per sample | **0.0 / 16** |
| max # cores ≥80% busy in any sample | 1 |
| heaviest user process during selfplay (sampled) | small native binary (~8 KiB VIRT, R-state) at **26-34 %CPU** = ~0.3 of one core — almost certainly Rust `sim-mcts-selfplay` |
| second-heaviest user process during selfplay | python `serve_onnx` (1.6 GiB VIRT, S-state) at **~10 %CPU** |
| heaviest user process during gate | similar pair (Rust `sim-eval-gate` ~26-34%, serve_onnx ~10%) |
| load average peak | 2.26 (iter-5 gate) — out of 32 logical threads |

**Per-stage CPU (top sample alignment):**

| stage | n samples | mean-busy% | peak-busy% | avg cores ≥80% | max cores ≥80% |
|---|---:|---:|---:|---:|---:|
| selfplay | 138 | 5.4% | 10.0% | 0.0/16 | 0/16 |
| distill | 10 | 5.4% | 7.4% | 0.0/16 | 0/16 |
| gate | 247 | 5.3% | 12.2% | 0.0/16 | 0/16 |

There is no observable CPU saturation at any point during the 33-minute run. Load average never exceeded 2.26. The Rust workers spin one R-state process at ~0.3 of one core; serve_onnx spins one S-state process at ~0.1 of one core. The remaining 31 logical threads are idle. The `--workers 24` flag is not parallelizing — consistent with Slice 2's finding that Rust `--workers` is single-process by design.

## Memory

| metric | value |
|---|---|
| host total | 64 078 MiB |
| used mean | 8 000 MiB |
| used peak | 9 008 MiB |
| free mean | 28 017 MiB |
| free min | 26 992 MiB |
| free/total min ratio | 42.1% |

Memory is comfortable throughout. The loop never approaches pressure. Not a lever.

## Hypothesis check — scoping doc claim

> "GPU sits idle during 95% of wall because serve_onnx defaults to `--provider cpu`."

**Confirmed on the idle observation, refuted on the causal mechanism.**

- Observation: GPU util=0% for 98.5% of wall samples; selfplay+gate together are 97.7% of instrumented wall. The GPU is idle, as predicted.
- Causal mechanism is more nuanced: **serve_onnx on CUDA would not move the needle, because serve_onnx is not the bottleneck either.** During steady-state selfplay/gate, serve_onnx burns ~10 %CPU = ~0.1 of one core, and the GPU is idle. The Rust worker burns ~30 %CPU = ~0.3 of one core. Together: **~0.4 cores active out of 32**, with neither GPU nor CPU saturated.
- Where the wall actually goes: **HTTP round-trip wait between Rust `sim-mcts-selfplay`/`sim-eval-gate` and `serve_onnx`**. With ~620 predicts/sec during selfplay (434 decisions × 100 sims / 70s) and ~2 400 predicts/sec during gate, each predict has ~400-1 600 μs of unaccounted wall. ONNX inference on this 14 KiB model is ~10-50 μs. The rest is localhost TCP + JSON serialize/parse on both sides.
- Implication for **candidate A** (flip provider to cuda): inference compute is already ~10 μs on CPU for this model. Moving it to CUDA might reduce predict compute from 10 μs to 1 μs, but the predict round trip is bound by HTTP RTT (≈ms scale), not the 9 μs delta. **Expected speedup ≤ 5%**, possibly worse if CUDA stand-up adds per-request driver latency. Spike-A is the wrong lever.
- Implication for **candidate B** (NAPI in-process predict): eliminates HTTP RTT entirely. Existing engine-rs/crates/napi-bridge measured 11.2× over subprocess in the rust-port-handoff smoke; the analog here is "eliminate per-predict TCP round trip", which is exactly the bound. **Expected payoff: 3-10× wall reduction on selfplay+gate** — the only candidate that addresses the actual binding constraint.

## Candidate ranking (numeric, updated)

| Candidate | Effort | Pre-doc estimate | Post-doc estimate | Notes |
|---|---|---|---|---|
| **A** — flip `serve_onnx` provider cpu→cuda | trivial (1-line edit ×2 + flag plumb) | 1.5-3× on inference | **≤ 1.05× on wall** | Refuted. Inference isn't the bottleneck; HTTP RTT is. CUDA may even regress for this tiny model due to PCIe + driver overhead per request. |
| **B** — NAPI in-process predict (eliminate HTTP RTT) | moderate (P3 queue item; surface already exists in engine-rs/crates/napi-bridge/) | 2-10× on inference if RTT dominates | **3-10× on selfplay+gate wall** → ~5-15 min/iter wallclock instead of 20 min | The lever. Directly attacks the binding constraint. Requires re-architecting Rust sim-cli to call into NAPI bridge in-process rather than fork serve_onnx subprocess + HTTP. Largest engineering scope of the four. |
| **C** — investigate gate=2× selfplay gap | n/a | 1.5-2× on gate | **0× — gap is explained** | Gate ran 2× the games (120 vs 60) and per-game wall is actually lower than selfplay (1.03 s/game vs 1.16 s/game). No hidden eval-gate overhead to chase. Retire this candidate. |
| **D** — Rust `--workers` actually parallelizes | medium (re-architect Rust sim to multi-process or async-pool game-level concurrency) | 2-5× | **6-20× ceiling, lower realized** | Loop currently uses ~0.4 cores of 32 available. A worker pool of N games concurrently calling serve_onnx (still over HTTP) could approach N× until serve_onnx saturates (currently at 0.1 cores → would saturate at ~10× concurrency). Stackable with B but not as clean. Risk: serve_onnx HTTP server thread is single-process Python (GIL) and would become the new bottleneck at >~5× concurrency. |

## Recommendation — spike **B (NAPI in-process predict) first**

Rationale:
1. **Directly attacks the binding constraint.** The data shows neither CPU nor GPU is saturated; the loop is HTTP-RTT-bound. Only B removes HTTP from the path.
2. **A is a near-no-op on this workload.** Predict compute is ~10 μs; CUDA cannot meaningfully accelerate something already dominated by ~1 ms of network/serialization.
3. **D is upper-bounded by serve_onnx single-process throughput.** Adding game-level concurrency over HTTP saturates serve_onnx at roughly 5-10× before the Python GIL caps it. B + D together can compound (NAPI in-process + multiple parallel game workers) but D alone hits a ceiling.
4. **Engineering surface for B already exists.** `engine-rs/crates/napi-bridge/` exposes `runMctsJson`/`mctsStepJson`/`advanceStepJson`. The sim-cli binaries currently invoke serve_onnx over HTTP; replacing that with an in-process Rust→ONNX path (via the bridge's existing `predict` analog, or direct `ort` crate consumption) is a single integration target.
5. **Acceptance is easy to measure.** Re-run the same 10-iter recipe with the NAPI-predict path and compare wall: B passes if total wall drops below 20 min (vs 33 min baseline). Falsification: if wall is unchanged, the bottleneck is elsewhere (unlikely given the data) and we'd re-instrument.

**Concrete next experiment:** wire `sim-mcts-selfplay` and `sim-eval-gate` to call an in-process ONNX inference path (skip the HTTP `serve_onnx_context` stand-up entirely for the Rust engine path). Smoke against the 10-iter C8-W6FIX-ON-extended recipe. Acceptance: total wallclock < 20 min (≥1.65× speedup); strength trajectory unchanged within ±0.022 noise.

## Surprises worth pinning

- The system is **idle**, not saturated. Going in we expected either GPU underuse OR CPU saturation; we got neither. Load average peaks at 2.26/32. This *only* makes sense if the bottleneck is in waiting on synchronous I/O — exactly what localhost-HTTP-per-predict produces.
- Gate's 2×-selfplay timing is fully accounted for by game count alone. Per-game wall is comparable. No hidden gate-side overhead to chase.
- Distill (4.49s/iter, 2.3% of wall) is bandwidth-invisible to GPU sampling at 5s cadence. It does not appear in GPU util numbers despite being the only stage that uses CUDA. Acceptable — distill is not a lever regardless.
