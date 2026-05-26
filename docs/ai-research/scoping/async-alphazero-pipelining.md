# Async AlphaZero Pipelining — Scoping

- **Date:** 2026-05-26
- **Status:** **P0-GREEN-2026-05-26** — CUDA coexistence smoke cleared. Concurrent sim-mcts-selfplay (workers=24 wave=256 cuda) + train_bc.py (b=256 amp dl=4) on RTX 5000 Ada: distill epoch wall unchanged (4.84-5.32s during overlap vs ~5.4s baseline = ~0% degradation), selfplay 28ms/game (~31ms solo baseline = ~slight noise speedup). GPU util 47-74% during overlap vs ~40% solo. No OOM. **P1 proceed.**
- **Predecessors:**
  - `cross-game-dispatcher-selfplay.md` (KILLSHOT-FALSIFIED 2026-05-26: intra-process cross-game batching gave only 1.10× at hidden=256/depth=4 because GPU is compute-bound. MCTS-side throughput ceiling reached.)
  - `distill-throughput-spike.md` (LANDED-SHIP 2026-05-26: distill 2.19× via b=256+sqrt-LR+amp+dl. Distill is now ~62 s/iter vs selfplay ~155 s/iter.)
  - `cuda-wave-sweep-validation.md` (LANDED: cuda-w256 + torch CUDA upgrade are the underpinning that makes both phases GPU-resident.)
- **Sibling:** none.

---

## TL;DR

Current iter wall ≈ 220 s (selfplay 155 + distill 62 + gate 3). All phases run sequentially per `training/r12_orchestrator.py:232-266`. **No async / threading / multiprocessing in r12** — fully greenfield.

Async AlphaZero pattern: while iter-N distill+gate execute, **iter-N+1 selfplay starts in parallel using whatever was promoted at iter-N start** (one-iter-stale data). Steady-state wall = max(selfplay, distill+gate) = max(155, 65) = **155 s** ⇒ **1.42× iter-wall**.

Cost: selfplay always trains against a one-iter-stale policy. Literature (KataGo continuous-train, AZ original) characterizes this as ~negligible at scale, but unmeasured on this codebase. Wilson gate at n=10k after K iters of A/B vs sequential baseline is the verdict mechanism.

## Hypothesis

> Restructuring `r12_orchestrator.main()` so iter-N+1 selfplay launches when iter-N selfplay completes (in parallel with iter-N distill+gate), with iter-N+1 selfplay reading `state.promoted_checkpoint` at the moment iter-N starts (one-iter-stale), will improve **iter-wall by 1.42×** at **|Δwl_promoted_trajectory| < 0.02 after K=10 pipelined iters vs sequential baseline at n=10k tight-gate** on the current `R16-P3-v36-az-bigtrunk-cold-cuda` recipe.

## Current pipeline (read-only ground truth from investigator)

Per `training/r12_orchestrator.py`:

- Outer loop: `main()` L232-266 sequentially calls `run_iteration(iteration)`.
- One iter: `run_iteration` at L280. Phases all blocking `subprocess.run(check=True)`:
  - Selfplay: `_run_pool_selfplay` L318 or `run_selfplay` L326, subprocess at L625
  - Distill: `run_distill` L354, subprocess at L1082
  - Crossover probe: `run_crossover_probe` L370, subprocess at L1139 (reads `iter-(N-1)/selfplay.jsonl` — frozen, safe under overlap)
  - Gate: `export_checkpoint_to_onnx` L373 + `run_gate` L381, subprocess at L1226
- Promoted ckpt: `R12State.promoted_checkpoint` (dataclass L88). Set in-memory by `main()` L249 after `run_iteration` returns (gated by `record["promote"]`). Persisted to `orchestrator-state.json` via `save_state` L1457.
- Iter-(N+1) selfplay reads `state.promoted_checkpoint` at L324 — this is the natural injection point for staleness, no extra plumbing needed.
- Gate→promote decision: `dagger_decide_promotion` at `dagger_orchestrator.py:690-736`. Promotes iff `wilson_lower + tolerance >= floor`. Halt-after-N-consecutive-non-promotes at L253-265 (default `--halt-after-consecutive-failures=2`).
- State files: `orchestrator-state.json` (single-writer, main thread today), `events.jsonl` (append-only), per-iter `iter-N/{selfplay,checkpoint,policy.onnx,gate.manifest,gate-progress,distill,crossover}.*`.
- Resume invariant (L232): `last_iter = max(state.iterations)` — iter is committed iff appended to `state.iterations` at L246. **Must preserve under pipelining.**

## Phase 0 — CUDA coexistence smoke (30 min, no code change)

Risk: investigator flagged that "ORT-CUDA + torch-CUDA concurrent steady-state has not been smoke-tested." Before committing to the pipeline refactor, confirm the two CUDA workloads coexist on one device without OOM or pathological contention.

**Recipe.** Launch sim-mcts-selfplay (production recipe: workers=24 wave=256 cuda) AND train_bc.py (new defaults: b=256 amp dl=4) concurrently on the same GPU. Measure:
- VRAM peak (target: <14 GB on 16 GB card)
- Per-phase wall vs solo (target: each phase < 1.5× its solo wall — i.e. <50% wall degradation per phase)

**Decision rubric.**

| Per-phase wall degradation | Verdict |
| :-- | :-- |
| <30% | **Green** — pipelined wall ≈ max(selfplay×1.3, distill×1.3) ≈ 200 s. Still 1.10× over sequential. Worth pursuing. |
| 30-50% | **Yellow** — pipelined wall ≈ max(155×1.5, 65×1.5) = 232 s. **WORSE than sequential 220 s.** Stop unless we can split CUDA devices or use MPS. |
| >50% | **Red** — pipelined is strictly worse. Stop. |
| OOM | **Red** — split devices required. Plumbing more involved. |

**Effort.** `implementer` or main session — ~30 min, no code change beyond launch script.

**Note:** the rig is single-GPU (RTX 5000 Ada, 16 GB). If degradation forces dual-device, we'd need a 2nd GPU or CUDA MPS (multi-process service) — both out of scope for this spike.

### Phase 0 Results — 2026-05-26

Serendipitous: the `R16-P3-v36-az-15k-shallow-cuda` orchestrator was running iter-2 distill (b=256+amp+dl=4, new defaults) when this measurement started. Launched a sim-mcts-selfplay concurrent on the same GPU device 0; measured both phases' walls.

| measurement | solo baseline | concurrent | degradation |
| :-- | --: | --: | --: |
| sim-mcts-selfplay per-game wall (workers=24 wave=256 cuda, sims=400) | ~31 ms/game (production extrapolation) | **28 ms/game** (500 games in 14.0 s) | ~slight speedup — within noise |
| train_bc.py per-epoch wall (b=256 amp dl=4, hidden=256/depth=4, ~150k samples) | ~5.4 s/epoch (orchestrator pre-overlap) | **4.84-5.32 s/epoch** (4 epochs during overlap window) | ~0% |
| GPU util (nvidia-smi dmon) | ~40% (selfplay solo) | **47-74%** during overlap | clean stream overlap |
| VRAM | selfplay ~4 GB resident | ~5 GB during overlap | well under 16 GB cap |

Raw: `runs/async-alphazero-pipelining/p0-concurrent-{gpu.csv,manifest.json,selfplay.jsonl,log}`. Orchestrator events: `runs/R16-P3-v36-az-15k-shallow-cuda/events.jsonl` epochs 19-22 (timestamps 1779786486-1779786501).

**Verdict: GREEN.** Both phases sustain near-100% of their solo throughput under concurrent execution. RTX 5000 Ada handles two concurrent CUDA workloads (ORT-CUDA + torch-CUDA) cleanly. GPU util climbs above the selfplay-solo ~40% baseline, confirming there's real compute headroom that pipelining can exploit. Proceeds to P1.

**Caveats.** Measurement window was short (~14 s overlap = 4 distill epochs sampled). Tail-distribution risk (e.g., periodic VRAM allocation spikes) is unmeasured but unlikely given peak VRAM was ~5/16 GB.

## Phase 1 — Pipeline refactor (~1-2 days)

Restructure `r12_orchestrator.main()` L232-266 from sequential to 2-stage pipeline. Concrete plan:

1. **Selfplay-stage worker (main thread).** Drives selfplay-N for iter N=0,1,2... in order. After selfplay-N completes, atomically (a) freeze `iter-N/selfplay.jsonl`, (b) submit a "distill+gate" task for iter-N to the background queue, (c) start selfplay-(N+1) using current `state.promoted_checkpoint`.

2. **Distill+gate worker (single background thread, via `concurrent.futures.ThreadPoolExecutor(max_workers=1)`).** Pulls (iter-N, frozen selfplay.jsonl, init_ckpt) tuples from the queue. For each: distill → crossover probe → ONNX export → gate → promotion decision. Posts the iter record back to main thread via a `Queue`.

3. **Main thread state coordination.** Drains the post-queue between selfplay launches. Updates `state.iterations.append(record)` and `state.promoted_checkpoint` (gated on `record["promote"]`). Calls `save_state()` after each update. **This preserves the single-writer invariant for `orchestrator-state.json`** (only main thread touches it).

4. **Init checkpoint for distill-N.** Currently `init_from_checkpoint = state.promoted_checkpoint` at the moment distill runs. In pipelined mode: capture the promoted ckpt at iter-N start (the same one selfplay-N used) and pass it explicitly to distill-N. This decouples distill-N from any intervening promotion.

5. **Selfplay-(N+1) ckpt selection.** Reads `state.promoted_checkpoint` at the moment selfplay-(N+1) launches. If iter-N gate hasn't finished, this is iter-(N-1)'s promoted ckpt = one-iter-stale (intended). If iter-N gate finished and promoted, it's iter-N's. The pipeline's "depth" determines staleness; max depth = 2 (selfplay-(N+1) running, distill+gate-N running, promoted ckpt = iter-(N-1)'s).

6. **Halt-on-consecutive-failures.** Current logic counts consecutive non-promotes. In pipelined mode, iter-N's verdict isn't known when iter-(N+1) selfplay starts. Resolution: keep counting on `state.iterations` order, but the halt check only blocks the NEXT selfplay launch. So one-iter overhang on failure is acceptable.

7. **Resume semantics.** On restart: replay `state.iterations` as today (L232). Any iter-N missing from `state.iterations` re-runs from selfplay. The pipelined version may have partially-completed iters (e.g. selfplay-N done, selfplay-(N+1) done, distill-N crashed); resume re-runs distill-N (using the frozen `iter-N/selfplay.jsonl`) and discards `iter-(N+1)/selfplay.jsonl` since iter-N wasn't committed. **Minor data-loss tradeoff for simpler invariants.**

8. **CLI flag.** New `--async-pipeline` flag (default OFF). Default OFF preserves byte-identical sequential behavior; explicit opt-in for validation. Flip default after Phase 2 validates.

**Code surface estimate.** Refactor of `main()` L232-266 (~50 lines), `R12State` to track one-step-stale init ckpts per pending iter (~20 lines), `_run_iteration_phases` split (~150 lines moved into stage workers). Net diff: ~250-400 lines in `training/r12_orchestrator.py`. No changes to `train_bc.py`, sim-mcts-selfplay binary, dagger_orchestrator helpers.

**Effort.** `implementer`, ~1-2 days. Bigger than any prior throughput spike but bounded (single file).

## Phase 2 — Wilson validation gate (3-5 h compute)

K=10 iters end-to-end, **two runs side-by-side**:
- Baseline: current sequential orchestrator on the production recipe (`bigtrunk-cold-cuda`-style: hidden=256 depth=4 sims=400 selfplay=5000)
- Challenger: same recipe + `--async-pipeline`

Compare:
- **Iter-wall trajectory** (target ratio: baseline / challenger ≥ 1.30 in steady state; theoretical max 1.42)
- **`promoted_wilson_lower` trajectory** (target: |Δwl| < 0.02 cumulative across the 10-iter run; per-iter Δwl tracked but expected noisier than n=10k)
- **Final tight-gate** at n=10k on each run's iter-10 promoted ckpt (the gate that matters; matches `cuda-wave-sweep-validation.md` Phase 2 acceptance bands)

**Acceptance bands.**

| Wall ratio | Final tight-gate Δwl | Verdict |
| :-- | :-- | :-- |
| ≥1.30 | \|Δwl\| < 0.02 | **SHIP** — flip default ON. |
| ≥1.30 | 0.02-0.05 | **SHIP-with-acknowledgement** — document the staleness-strength contract. |
| ≥1.30 | 0.05-0.08 | **Envelope breach** — investigate (longer warmup? two-iter-stale? different recipe knobs?) before ship. |
| ≥1.30 | ≥0.08 | **Reject async pipeline.** Strength cost of staleness exceeds throughput win. |
| <1.30 | any | **Stop** — Phase 0 underestimated GPU contention; pipeline doesn't deliver promised wall. |

Compute budget: 10 iters × (155 + 62 + 3) ≈ 2200 s sequential per run × 2 runs = ~75 min for the loop + ~30 min for the final n=10k gate = **~2 h total**. Padding to 3-5 h for setup, contingency.

## Phase 3 — Default flip (conditional on Phase 2 SHIP)

Flip `--async-pipeline` default ON in `r12_orchestrator.py` argparse. Add a one-line warning when the orchestrator detects a multi-iter run with `--async-pipeline=False` (mirroring the `distill-throughput-spike` legacy-recipe warning pattern).

Update the canonical launch.sh template in CLAUDE.md (or scoping README) to note that async pipelining is now default and `--no-async-pipeline` opts out.

## Out of scope (deferred)

- **Multi-step staleness (two-iter, three-iter)** — could overlap more phases but multiplies staleness cost. Only if Phase 2 shows zero strength impact at one-step.
- **Per-phase device split** (selfplay GPU 0, distill GPU 1) — requires second GPU. Hardware question, not code.
- **CUDA MPS / streaming inference server** — out of scope; not a single-file change. If Phase 0 shows CUDA contention is the dominant problem and degradation >30%, file separate scoping for MPS.
- **Pipelining inside a single iter** (selfplay rows streaming into distill mid-iter) — discarded; coordination complexity is much higher and the inter-iter staleness pattern is established literature.
- **Replacing `concurrent.futures.ThreadPoolExecutor` with `multiprocessing`** — overkill since the heavy lifting is in subprocess.run anyway; thread is just a coordination shim.

## Risks ranked

1. **GPU contention (Phase 0 gates this).** If concurrent ORT-CUDA + torch-CUDA degrade each phase by >50%, the pipeline LOSES wall. Single-GPU rig makes this real.
2. **Strength impact of staleness.** Unmeasured on this codebase. Phase 2 Wilson gate is the verdict.
3. **State file coordination.** Single-writer-main-thread invariant should keep this safe, but partial-iter crashes need testing. Phase 1 implementation responsibility.
4. **Resume semantics** edge cases. Crash between (selfplay-N+1 done, distill-N crashed) needs re-runnable. Phase 1 implementation responsibility.
5. **Halt-on-consecutive-failures off-by-one.** Pipelined iters may overrun by 1 before halt fires. Acceptable; documented.

## Why this is the highest-EV remaining throughput lever

| Lever | Status | Realistic wall impact |
| :-- | :-- | :-- |
| MCTS wave-batch tuning | LANDED-SHIP via cuda-wave-sweep-validation | 7.29× per-game wall (done) |
| Distill knobs (amp, dl, batch) | LANDED-SHIP via distill-throughput-spike | 2.19× distill wall = 17-20% iter-wall (done) |
| Cross-game inference dispatcher | KILLSHOT-FALSIFIED | 1.10× (dead) |
| Wave-pipeline overlap (intra-game) | DEFERRED | 1.4-1.6× MCTS wall but breaks bit-identity |
| **Async iter pipelining (this doc)** | **SCOPING** | **1.42× iter-wall** |
| Bigger model | DEFERRED | strength-axis, not throughput |
| Multi-GPU / MPS | OUT OF SCOPE | hardware-gated |

Pipeline overlap is the **biggest remaining throughput lever within the current single-GPU hardware budget**. Worth the 1-2 day implementation cost if Phase 0 clears.

## Filing

- **Scoping doc:** this file.
- **Run dir:** `runs/async-alphazero-pipelining/` (created at Phase 0).
- **Queue entry:** `async-alphazero-pipelining` (add at scoping commit).
- **Progress writeup target:** new section in `docs/ai-research/progress/r16.md` at LANDED time.
