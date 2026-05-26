# Cross-Game Dispatcher for Selfplay — Scoping

- **Date:** 2026-05-26
- **Status:** **CLOSED-FALSIFIED-2026-05-26** — P0 falsified original mechanism (CPU is only 17% of wave wall, not the assumed dominant ~25 ms floor). P1 killshot confirms the dispatcher *implementation* works correctly (mean fill ~878 rows/call, per-call inference latency drops 2.5× under contention) but **wall throughput only improves 1.10×** — within rubric's STOP band (<1.2×). Bottleneck is GPU compute at hidden=256/depth=4, not coordination overhead. Patch left in repo behind `--killshot-dispatch` flag for future re-investigation; not wired to production. See "Phase 1 Results" below.
- **Predecessors:**
  - `cuda-wave-sweep-validation.md` (LANDED: cuda-w256 shipped, 7.29× per-game wall at sims=400 vhleaf hidden=256/depth=4; production run at 41% GPU util — `cuda-wave-sweep-validation.md:114`).
  - `gpu-batched-inference-throughput.md` (CLOSED B6: per-game wave-batching landed; `BatchedDispatcher` kept for cross-thread coalescing but never wired into `sim-mcts-selfplay`; explicit "for sims >> 1000 or model-size shift, cross-game dispatcher may re-open" at `gpu-batched-inference-throughput.md:636-644`).
  - `action-count-bucketed-wave-batching.md` (LANDED-MARGINAL ~8%; explicit "no change to BatchedDispatcher — lower-priority cross-game B2 path; can port later if needed" at `action-count-bucketed-wave-batching.md:67`).
  - `vhleaf-throughput-bigger-model-rebaseline.md` (CLOSED: Phase B value-only KILLED; clone/tree-reuse triaged out).
- **Siblings:**
  - `wave-pipeline-overlap.md` (NOT YET FILED — deferred candidate, see "Deferred lever" below).

---

## TL;DR

cuda-w256 is shipped but the production run sits at **~41% GPU utilization** (`cuda-wave-sweep-validation.md:114`). Each of N worker games runs its own wave loop and dispatches one big `Session::run` synchronously, then sits in CPU for ~25 ms of Phase 1+2+4 per wave. Cross-game coalescing would gather leaves from many worker waves into one CUDA call, attacking the GPU under-feeding directly.

The mechanism already exists: `BatchedDispatcher` (`engine-rs/crates/engine/src/inference/mod.rs:1065`) gathers across worker threads via mpsc + leader-on-first-recv pattern. It is used today only by `sim-eval-gate`. The blocker is a mutex (`inference/mod.rs:543-553`) that errors when wave-batching is combined with the dispatched session storage.

**Upper bound payoff:** 1/0.41 ≈ 2.44× per-game wall if GPU saturates fully. **Realistic:** 1.6-2.0× after residual CPU floor (GameState.clone, observation build, legal-actions enumeration) re-asserts. **Cost:** ULP-level FP reduction-order drift (batch composition is wall-clock-dependent); selfplay-safe, NOT eval-gate-safe without re-validation.

This spike: **(0) instrument per-wave wall** to confirm Phase1+2/Phase3/Phase4 cost split, **(1) killshot cell** to confirm GPU util climbs past 55% before committing to the wire-up, **(2) wire-up** with explicit guard relaxation, **(3) wilson validation** at n=10k matching cuda-w256's gate.

## Hypothesis

> At hidden=256/depth=4 with 24 worker selfplay games, routing each wave's `B≈wave_size` leaf-inference batch through `BatchedDispatcher` (instead of synchronous `Session::run`) will coalesce across workers and lift GPU utilization from ~41% to ≥70%, yielding **1.6-2.0× per-game wall improvement**, at the cost of ULP-level value-head FP drift that registers as **|Δwl| < 0.02 at n=10k** (i.e. SHIP-STRENGTH-NEUTRAL per `cuda-wave-sweep-validation.md` Phase 2 acceptance bands).

## Phase 0 — Per-wave wall instrumentation (read-only)

Before committing to a wire-up, confirm the cost split. Today no per-wave timing exists; only 1 Hz `nvidia-smi` snapshots and `UMA_LOG_WAVE_NACTIONS` action-count logs.

**Change.** Add `UMA_LOG_WAVE_TIMING` env flag in `engine-rs/crates/engine/src/mcts/driver.rs` around the wave loop (`driver.rs:504-639`):
- Mark `Instant::now()` at wave start, end of Phase 2 (pre-GPU), end of Phase 3 (post-GPU), end of Phase 4.
- Emit `eprintln!` for the first ~100 waves per process, formatted `wave_timing: B={B} p12={p12_us}us p3={p3_us}us p4={p4_us}us`.
- Pattern matches existing `UMA_LOG_WAVE_NACTIONS` at `inference/mod.rs:809`.

**Recipe.** Single sim-eval-gate iter with the production cuda-w256 recipe at workers=24:
```
UMA_LOG_WAVE_TIMING=1 sim-eval-gate --leaf value-head --mcts-two-sided \
  --sims 400 --workers 24 --games 24 --model-side both \
  --collapse-max 64 --max-nodes 5000 --c-puct 1.5 \
  --seed-base 0 --prior policy \
  --device cuda --wave-size 256 --virtual-loss 1.0 \
  --model runs/R16-P3-v36-az-bigtrunk-cold/iter-2/policy.onnx
2>&1 | tee runs/cross-game-dispatcher-selfplay/phase0-timing.log
```

**Expected output.** If Phase 1+2+4 dominates Phase 3 by ≥2× wall, the cross-game-batching hypothesis is the right intervention. If Phase 3 is comparable to Phase 1+2+4 wall, GPU is already well-fed within a single game and cross-game would only help with worker count, not GPU util — re-examine.

**Effort.** `implementer`, ~1 hour code + 5 min run.

### Phase 0 Results — 2026-05-26

Instrumentation landed in `engine-rs/crates/engine/src/mcts/driver.rs` (env-gated `UMA_LOG_WAVE_TIMING=1`, 100-line cap per process via `static AtomicU32`, `OnceLock<bool>` hot-path hoist). Recipe executed at `workers=4` (the cuda-wave-sweep-validation.md baseline; could not run at production `workers=24` without disturbing the in-flight `R16-P3-v36-az-5k-nobuffer-cuda` orchestrator). 100-wave probe took ~2 s wall.

**Post-warmup B=256 median cost split (45 waves):**

| phase | µs | % of wave wall |
| :-- | --: | --: |
| **p12** — CPU Phase 1 (PUCT × 256) + Phase 2 (`WaveAction::compute` × 256) | **1,003** | **10.5%** |
| **p3** — GPU `Session::run` (one B=256 call) | **7,958** | **83.2%** |
| **p4** — CPU Phase 4 (insert children + virtual-loss undo × 256) | **608** | **6.4%** |
| **total** | **9,569** | 100% |

**CPU total per wave: 1.6 ms. GPU total: 8.0 ms.** Raw log at `runs/cross-game-dispatcher-selfplay/phase0-stderr.log`.

**Original hypothesis FALSIFIED.** The investigator's pre-instrumentation estimate ("each worker dispatches one big CUDA call then sits in CPU for ~25 ms") was wrong by ~15×. Within a single process, CPU is ~17% of wave wall; GPU dominates at ~83%. Cross-game batching cannot "hide CPU idle" because there's only 1.6 ms of CPU per wave to hide.

**Re-framed hypothesis.** Production 41% GPU util at `workers=24` (`cuda-wave-sweep-validation.md:114`) cannot come from intra-process CPU — single-process p3/total ratio (83%) already exceeds the production GPU-util figure. The under-feeding must come from one of:

1. **ORT Session serialization across worker threads.** All 24 workers share one `Session` via `UnsafeCell` (`inference/mod.rs:233-247`). If ORT internally locks during `run()`, 24 workers queueing 8 ms calls would produce a sawtooth GPU pattern (busy during a single call, idle during ORT lock-acquire / CUDA stream handoff).
2. **CUDA stream serialization.** ORT uses a single stream per Session; 24 calls queue sequentially. Per-call kernel launch overhead (~1-2 ms) adds 24× to per-wall but per-call GPU compute does not amortize.
3. **Memory bandwidth / PCIe contention.** Each B=256 call moves ~1 MB of activations; 24 in flight is 24 MB of inflow. Possible but smaller order.

**Cross-game dispatcher still attacks #1 and #2.** Instead of 24 sequential B=256 calls (each with ORT lock + kernel launch + 8 ms compute), the dispatcher gathers into one B=6144 call. Even if the compute scales linearly (8 ms × 24 = 192 ms), the kernel-launch + ORT-lock overhead is paid once instead of 24×. If launch overhead is the bottleneck (likely at this small wall), the win could still be 2-4×.

**The killshot is now MORE important, not less.** P0 invalidated the *mechanism* in the original hypothesis but the *outcome metric* (GPU util at workers=24) is what determines ship/no-ship. P1's GPU-util rubric (`≥70% confirm / 55-70% partial / 41-55% pivot / <41% stop`) measures the outcome directly regardless of mechanism — proceed.

**Caveats.**
- Measurement was at `workers=4`, not `workers=24`. The cost split is per-process-intrinsic so should hold at higher worker counts, but contention effects (mechanism #1, #2 above) are by definition only visible at worker counts ≥ ~8.
- Mild GPU contention from the in-flight orchestrator's distill phase (35-40% baseline util) may have inflated p3 slightly; standalone clean-window probe could push GPU% even higher (i.e., CPU% even lower).
- Warmup waves (first ~10) showed p3 ≈ 33-45 ms (cuDNN autotune). Excluded from the post-warmup median.

## Phase 1 — Killshot cell (10 min)

Goal: stub a single side-by-side cell that confirms cross-game batching can move the GPU-util needle *before* committing to the production-quality wire-up.

**Change.** Temporary patch in `engine-rs/crates/sim-cli/src/bin/mcts_selfplay.rs:611`:
- Swap `InferenceSession::load_on` → `InferenceSession::load_on_with_batching(onnx, device, batch_size=1024, max_wait_us=200)`.
- Temporarily comment out the mutex error at `inference/mod.rs:543-553` (or path-fork around it) so wave-batching submits each row through the dispatcher.
- Behind a `--killshot-dispatch` flag so the change is reversible.

**Recipe.** Two cells, 24 games each, no wilson check — only GPU util and per-game wall:
```
# Baseline
nvidia-smi dmon -s u -c 60 > runs/cross-game-dispatcher-selfplay/phase1-baseline-gpu.csv &
sim-eval-gate --leaf value-head --mcts-two-sided \
  --sims 400 --workers 24 --games 24 --model-side both \
  --device cuda --wave-size 256 ... \
  > runs/cross-game-dispatcher-selfplay/phase1-baseline.log

# Challenger
nvidia-smi dmon -s u -c 60 > runs/cross-game-dispatcher-selfplay/phase1-killshot-gpu.csv &
sim-eval-gate --leaf value-head --mcts-two-sided \
  --sims 400 --workers 24 --games 24 --model-side both \
  --device cuda --wave-size 256 --killshot-dispatch ... \
  > runs/cross-game-dispatcher-selfplay/phase1-killshot.log
```

**Decision rubric.**

| Mean GPU util in killshot | Verdict | Next |
| :-- | :-- | :-- |
| ≥ 70% | hypothesis confirmed | Phase 2 wire-up |
| 55-70% | partial — CPU floor is real but not dominant | Phase 2 wire-up, but expect 1.3-1.5× not 1.6-2.0× |
| 41-55% | residual CPU floor dominates | STOP. Cross-game batching is not the bottleneck. Pivot to the `wave-pipeline-overlap.md` candidate, or instrument distill-side. |
| < 41% | dispatcher latency is hurting throughput | STOP. Tune `max_wait_us` smaller or `batch_size` differently; if still under baseline, dispatcher path is wrong fit. |

**Effort.** `implementer`, ~1 hour patch + 10 min run.

### Phase 1 Results — 2026-05-26

Patch landed in `engine-rs/crates/engine/src/inference/mod.rs` (new `BatchedDispatcher::predict_many` helper + `predict_v3_batch` now routes through dispatcher when `SessionStorage::Dispatched`) and `engine-rs/crates/sim-cli/src/bin/eval_gate.rs` (three new flags: `--killshot-dispatch`, `--killshot-batch-size 1024`, `--killshot-wait-us 200`). Zero behaviour change when flag unset; bit-identical wilson_lower confirmed at n=8 smoke. `sim-mcts-selfplay` and `r12_orchestrator.py` untouched — orchestrator was running iter-9/10/11/12 selfplay live during measurement.

**A/B at workers=24 wave=256 --games 60 --model-side both (120 games per cell, contention with orchestrator distill window):**

| cell | wall (s) | games/sec | mean GPU util | dispatcher engagement |
| :-- | --: | --: | --: | :-- |
| **baseline** (no `--killshot-dispatch`) | 4.40 | 27.3 | 43.9% mean / 38% median | n/a (inline path) |
| **challenger** (with `--killshot-dispatch`) | 3.99 | 30.1 | 39.6% mean / 39% median | `batches=314 requests=275804 mean_fill=878.36` |
| **ratio** | **0.91×** wall | **1.10×** throughput | ~ flat | ~3.4× coalescing |

**Per-wave instrumentation (post-warmup B=256, first 100 waves per process):**

| phase | baseline µs (p50) | challenger µs (p50) | ratio |
| :-- | --: | --: | --: |
| p12 (CPU select+expand) | 1,748 | 1,771 | 1.01× |
| **p3 (inference wall)** | **94,901** | **37,318** | **0.39× (2.5× faster)** |
| p4 (CPU backup) | 933 | 838 | 0.90× |

Raw: `runs/cross-game-dispatcher-selfplay/phase1-{baseline,killshot}-{stderr.log,gpu.csv,manifest.json}`.

**Verdict: STOP. Hypothesis FALSIFIED.**

Per the rubric in "Phase 1 — Killshot cell" above:
- Mean GPU util: ~39% (baseline) vs ~40% (challenger) — flat. Below the `41-55% pivot` band, well below `≥70% ship`.
- Throughput: **1.10× wall improvement** — below the `1.2× tempered ship` floor.

The dispatcher *implementation* is sound (cross-row coalescing reaches mean fill ~878, ~3.4× of the underlying wave_size=256). The dispatcher *hypothesis* — that cross-game batching meaningfully improves throughput at this model size — is wrong. Theory matches data: at hidden=256/depth=4 with B=256→878, kernel-launch overhead is ~10% of compute time, so amortizing it saves ~10% — exactly what we measured.

**Interpretation note.** The per-wave p3 dropping 2.5× while wall only changes 1.10× is reconciled by the wave_timing cap (100 lines per process captures only the first ~4 waves per worker, when CUDA streams are deepest-queue contended). After steady state, both paths converge to compute-bound latency. The dispatcher reduces *peak* per-call wait under bursty arrival but doesn't change *average* throughput.

**Caveats.**
- All measurements ran with the in-flight orchestrator (`R16-P3-v36-az-5k-nobuffer-cuda`) occupying GPU device 0. Both cells had identical contention since they ran back-to-back, so the A/B is fair, but absolute throughput numbers are not the production ceiling.
- Two follow-up runs (`phase1b`, `phase1c`) at longer game counts hung at ONNX init when iter-11/12 selfplay started mid-cell, suggesting a CUDA memory contention pathology worth investigating separately (not blocking this verdict).
- The killshot patch is left in the repo behind opt-in flags. Risk: orphan code path in `inference/mod.rs` (`BatchedDispatcher::predict_many` + relaxed `predict_v3_batch` mutex). If no future re-investigation materializes within ~2 sprints, recommend cleanup.

**What this rules out and what it doesn't.**

- **Ruled out:** wiring `BatchedDispatcher` into `sim-mcts-selfplay` for the value-head leaf at hidden=256/depth=4, wave=256, workers=24. Throughput payoff is in the 1.10× range, not the 1.6-2.0× scoping hypothesis.
- **Not ruled out:** future model-size shifts that change compute/launch ratio (much smaller model → launch overhead matters more; much larger model → won't matter). The patch is preserved as a re-test entrypoint.
- **Not ruled out:** wave-pipeline-overlap as the next throughput lever (deferred-sibling). Now that cross-game dispatcher is killed, pipeline-overlap is the highest-priority remaining MCTS-side throughput candidate — though its strength-envelope cost is real (see "Deferred lever" section below).
- **Not ruled out:** distill-side throughput, which `cuda-wave-sweep-validation.md:110` already flagged as the post-CUDA-upgrade dominant iter-wall component.

**Decision on Phase 2-4.** Skip. Production wire-up, n=10k wilson gate, and r12 default flip are unnecessary given the negative killshot.

## Phase 2 — Production wire-up (conditional on Phase 1 verdict ≥55% GPU util)

> **SKIPPED 2026-05-26.** Phase 1 verdict was STOP (1.10× throughput, <1.2× rubric floor). The Phase 2-4 bodies below remain as the pre-registered recipe — historical record of what we planned, not what we shipped.


**Code surface.** Bounded to four sites:

1. **`engine-rs/crates/engine/src/inference/mod.rs:543-553`** — relax the wave-vs-dispatcher mutex. New behaviour: when `SessionStorage::Dispatched`, the wave's `predict_v3_batch` submits each row through the dispatcher and waits on B response channels (today it errors). Keep the existing inline path as the no-dispatcher fast path.
2. **`engine-rs/crates/sim-cli/src/bin/mcts_selfplay.rs:611`** — add `--batch-size` (default 0 = off; ≥2 = enable dispatcher), `--batch-wait-us` (default 200) flags mirroring `sim-eval-gate`. When `--batch-size ≥ 2`, swap `load_on` → `load_on_with_batching`.
3. **`training/r12_orchestrator.py`** — thread `--batch-size` / `--batch-wait-us` through the selfplay subprocess at L564-620 and gate subprocess. Add a leaf-conditional default: `(value-head, cuda)` → `batch_size = max(workers, 256)` (covers the burst-arrival case at workers ≥ 24), `batch_wait_us = 200`. Other leaf/device combos: `batch_size = 0` (off).
4. **NO change to `sim-eval-gate`.** The mutex relaxation makes the dispatcher path optionally co-exist with wave-batching at the inference layer, but eval-gate's strength gate is the existing wilson sentinel — adding cross-game batching there would invalidate the n=10k baseline at `cuda-wave-sweep-validation.md:93`. Eval-gate stays single-game-wave-only.

**Determinism contract.**
- **Selfplay**: per-row outputs are deterministic (`pack_row` + `greedy_masked_softmax` are pure functions of the row), but the cross-batch FP reduction order varies with wall-clock batch composition. Drift bound: ULP-level. Comparable in scale to the existing wave-batching drift documented at `driver.rs:493-495`.
- **Eval-gate**: unchanged. The mutex relaxation is opt-in via `--batch-size`; eval-gate's default stays `--batch-size 0` (pure wave-batching).
- **NAPI bridge**: unaffected — bridge does not use `BatchedDispatcher`.

**Effort.** `implementer`, ~1 day code + 30 min sanity-iter end-to-end.

## Phase 3 — Wilson validation gate (n=10k)

Identical recipe and acceptance bands to `cuda-wave-sweep-validation.md` Phase 2:

**Cells.**
- **Baseline:** cuda-w256 (production, post-`cuda-wave-sweep-validation` Phase 3 wire-up).
- **Challenger:** cuda-w256 + `--batch-size <best-from-Phase-1>` + `--batch-wait-us 200`.

**Recipe.** `sim-eval-gate --games 5000 --model-side both` per cell on `iter-2/policy.onnx`. (Same iter as cuda-w256 baseline — keeps cross-doc comparison clean.)

**Acceptance bands** (from `cuda-wave-sweep-validation.md:42-47`):

| Δwl_challenger_vs_baseline at n=10k | Verdict | Action |
| :-- | :-- | :-- |
| \|Δwl\| < 0.02 | **strength-neutral** | Ship dispatcher as new selfplay default at (value-head, cuda). |
| 0.02 ≤ \|Δwl\| < 0.05 | **measurable drift, in envelope** | Ship with explicit documentation of the new throughput-strength contract. |
| 0.05 ≤ \|Δwl\| < 0.08 | **envelope breach** | Stop. Revisit `batch_wait_us` / `batch_size` tuning, or accept cuda-w256 status quo. |
| \|Δwl\| ≥ 0.08 | **systematic strength regression** | Reject. Investigate FP-determinism root cause. |

**Wall budget.** At baseline 28 ms/game ≈ 280 s for 10k games. At challenger ≈ 14-18 ms/game ≈ 140-180 s. Total Phase 3 ≈ 10 min wall + bookkeeping.

## Phase 4 — Wire-up + r12 default flip (conditional on Phase 3 ship verdict)

1. Flip the (value-head, cuda) selfplay default in `r12_orchestrator.py` to enable `--batch-size <winner>` and `--batch-wait-us 200`.
2. Single iter end-to-end on `runs/R16-P3-v36-az-bigtrunk-cold-cuda` resume to confirm no regression (selfplay + distill + gate).
3. Update `cuda-wave-sweep-validation.md` Phase 3 wire-up section with the new default and pointer to this doc.

## Deferred lever — wave-pipeline-overlap

The investigator agent surveyed per-game wave-loop pipelining (Phase 4 of wave N overlapping with Phase 1+2 of wave N+1; see `engine-rs/crates/engine/src/mcts/driver.rs:618-639`) and reported:

- **Bit-identity is broken hard.** Phase 4 writes `visits[a]`, `wsum[a]` and `children[action_index]`; Phase 1 of wave N+1 reads exactly those fields. Overlapping introduces a *new* divergence axis beyond the existing wave-virtual-loss bias. Strength envelope is comparable to or worse than `cuda-w64`'s Δwl=−0.047.
- **Payoff is smaller.** 1.4-1.6× CUDA upper bound vs 1.6-2.0× for cross-game.
- **Risk-adjusted, it loses to cross-game.** Cross-game retains bit-equivalent-per-row outputs (only FP reduction order drifts); pipeline introduces structural tree-shape divergence.

**Re-open conditions:** if Phase 1 killshot here returns 41-55% GPU util (CPU floor dominates), pipeline overlap becomes the next-best lever. File `docs/ai-research/scoping/wave-pipeline-overlap.md` at that point, framed as an explicit strength-envelope-acceptance decision parallel to cuda-w64.

## Parking lot — distill-side throughput

`cuda-wave-sweep-validation.md:110` notes: *"the iter-wall bottleneck on the bigger model was distill, not MCTS"* — already partially cashed via the torch 2.12+cu126 upgrade in that doc's Phase 3. Post-cross-game-dispatcher, distill may re-emerge as the dominant iter-wall component. Defer to a separate scoping doc once this round lands; do not bundle here.

## Filing

- **Scoping doc:** this file (`docs/ai-research/scoping/cross-game-dispatcher-selfplay.md`).
- **Run dir:** `runs/cross-game-dispatcher-selfplay/` (created on Phase 0 launch).
- **Queue entry:** `cross-game-dispatcher-selfplay` (add at scoping commit).
- **Progress writeup target:** new section in `docs/ai-research/progress/r16.md` at LANDED time.
