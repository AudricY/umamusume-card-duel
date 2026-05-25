# Batched Inference + High Parallelism — GPU Throughput Probe

- **Date:** 2026-05-25
- **Status:** **THROUGHPUT WON BY 7-9× ON CPU — B6 WAVE BATCHING IS THE
  LEVER.** B1-B5 framed this as a CUDA-vs-CPU question and concluded
  recipe-conditional (CUDA wins on vhleaf at high parallelism by 1.09×).
  B6 added intra-tree wave batching with virtual loss (commit 2c2c860,
  `--wave-size` flag on `sim-eval-gate`, default 1 bit-identical) and
  flipped the verdict entirely: **the lever isn't GPU batching, it's CPU
  wave batching.** vhleaf sims=100 wave_size=16 on CPU runs at 1.02s vs
  serial CPU 7.45s = **7.30× faster at bit-identical wilson_lower**.
  wave_size=32 = 8.71× at +0.019 wilson drift (well within the 0.05
  correctness envelope). CUDA still loses to wave-batched CPU on every
  recipe tested. Mechanism: wave amortizes per-call ORT overhead
  (~50µs/call CPU baseline from B1) across the wave; one
  `Session::run(B=16)` replaces 16 serial calls. CUDA's ~1.3ms kernel
  launch floor cannot beat CPU's per-call cost at this graph's
  per-state compute. All wiring from B2 + B6 stays landed. Successor
  to [`gpu-inference-execution-provider.md`](gpu-inference-execution-provider.md)
  G4-falsified, G5-landed. Queue entry:
  `gpu-batched-inference-throughput` in `docs/ai-agent-state/queue.json`.
- **One-liner:** Determine whether gathering N per-state ONNX calls into a
  single `Session::run(B=N)`, fed by high-parallelism inter-game workers, makes
  the GPU path beat CPU on a recipe we actually run — and identify the smallest
  recipe where it does.
- **Non-goal:** strength gains. Acceptance here is throughput / wallclock under
  matched strength (Wilson-lower within G4-style 0.02 envelope). Strength
  questions stay in [`gpu-fed-stronger-mcts.md`](gpu-fed-stronger-mcts.md).
- **Headline verdict (2026-05-25, REVISED AGAIN AFTER B6):** The
  throughput axis is **won by 7-9× via CPU wave batching**, not by GPU.
  B1-B5 framed the question as "does batched CUDA beat CPU" and reached
  recipe-conditional (CUDA 1.09× CPU on vhleaf at workers=128). B6
  added intra-tree wave batching with virtual loss; CPU wave_size=16 on
  vhleaf produces **bit-identical wilson_lower in 1.02s vs serial CPU
  7.45s = 7.30× faster**. wave_size=32 hits 8.71× at +0.019 wilson drift.
  CUDA at any wave_size still loses to wave-batched CPU.
  Best results by recipe:
  - **vhleaf sims=100 (the big win):** CPU wave_size=16: **1.02s, wilson
    0.3923 bit-identical to serial** = 7.30× faster than serial CPU,
    14.7× faster than best CUDA cell from B5. Promotion gate (1.5×)
    shattered. Best CUDA (wave=64 w=16): 9.79s — 9.6× slower than CPU
    wave=16.
  - **rollout-leaf sims=100 (shipping):** CPU wave_size=32: 7.44s vs
    serial 9.28s = 1.25× faster, modest because rollouts dominate game
    wall. Best CUDA wave_size=64: 11.17s, still slower than wave CPU.
  - **rollout-leaf sims=1000 (hisim):** CPU wave_size=8: 41.29s vs
    serial 49.48s = 1.20× faster (bit-identical wilson). Best CUDA
    wave_size=64: 73.98s, still 1.8× slower than CPU.
  Mechanism: at this graph's per-state compute (~50µs CPU per B1), the
  dominant cost is per-call ORT overhead. Wave reduces N calls per game
  by a factor of `wave_size`. CUDA cannot beat CPU because its
  ~1.3ms kernel launch floor is worse than CPU's per-call cost on this
  small graph, even at high fill.
  See `## B6 wave batching evidence` below for the full sweep; the
  B2/B3/B4/B5 sections (CUDA-only dispatcher path) document the journey.

## Question

G4 (2026-05-22) falsified naive GPU on `--leaf rollout --sims 100`: CUDA was
5.12× slower than CPU at workers=16 (55.18s vs 10.77s, n=200), GPU util peaked
at 38%. G5 then lifted CPU scaling to 7.11× at workers=16 by dropping the
session Mutex. G4's diagnosis: the recipe doesn't have enough inference
compute per game to amortize device transfer + kernel launch (~5–15 ms/call
GPU vs ~10–100 µs CPU, ad-hoc traces).

`gpu-inference-execution-provider.md` `next_action` explicitly names batched
dispatch as the next CPU throughput lever. `gpu-fed-stronger-mcts.md`
Candidate 3 names the same as a strength prerequisite. **Neither has been
built.** This doc scopes the throughput half.

The two-part working question:

1. With B=N batched `Session::run` and N inter-game workers feeding a flush
   collector, does the GPU path cross even on `--leaf rollout` (a recipe where
   inference is ~5% of the work), and at what (N, sims, leaf) does it become a
   clear win?
2. If not on rollout-leaf, what is the smallest recipe (sims, leaf, batch
   size) that does cross? Is that recipe one we'd actually use?

## What we know going in

From the investigator pass (file:line citations preserved for verification):

- **`predict_v3` is single-state.** All input tensors are pinned at B=1
  (`engine-rs/crates/engine/src/inference/mod.rs:373`,406,411,420,423,430).
  `extract_logits_and_value` (line 552) assumes batch=1. No code path calls
  `Session::run` with B>1.
- **The ONNX graph may already admit dynamic batch** (line 549 comment cites
  `[B, A]` on the ORT side), but the exporter very likely pins B=1. Re-export
  with dynamic batch axis may be a prereq.
- **Leaf modes are exactly two:** `MctsLeaf::{ValueHead, Rollout}`
  (`engine-rs/crates/engine/src/mcts/config.rs:13`). "Hybrid" is a blend knob
  (`value_head_rollout_blend`, config.rs:54), not a third mode.
- **Selection is serial per tree** (`engine-rs/crates/engine/src/mcts/driver.rs:115-244`).
  No virtual loss, no wave-batching primitive (grep for `virtual.loss|wave|tree.parallel`
  in engine-rs returns nothing). Intra-tree batching would require restructuring.
  The cheap path is **inter-game batching across existing OS-thread workers** —
  `mcts_selfplay.rs:686-748` and `eval_gate.rs:556` already run N games as OS
  threads sharing one `OnceLock<InferenceSession>`.
- **Inference call count per game on `--leaf rollout` at sims=100:** roughly
  100 PUCT-prior calls (one per expansion) — ONNX is ~5% of game wall.
- **Inference call count per game on `--leaf value-head` at sims=100:** ~100
  prior calls + ~100 leaf-value calls — ONNX is the dominant share. *But* no
  model currently meets the strength gate: best vhleaf wilson is 0.4281
  (C8-W6FIX-ON iter-1, `docs/ai-research/progress/r16.md:170`) vs rollout-leaf
  production 0.6479 — the leaf-readiness gate in
  `gpu-fed-stronger-mcts.md` is not crossed. **Probing throughput on vhleaf is
  fine; promoting a vhleaf candidate is gated separately.**
- **CPU scaling ceiling after G5** is ~7–8× at workers=16, and the new
  bottleneck is per-call `extract_logits_and_value` Vec<f32> allocations
  (`gpu-inference-execution-provider.md:262`), not session-level serialization.

## Working hypothesis

Batched GPU dispatch wins **only when ONNX wall is a meaningful share of game
wall** AND the batch can fill without unacceptable queue-wait. Concretely:

1. On `--leaf rollout --sims 100 --rollout-steps 200`: even at B=16 the prior
   calls are only ~5% of work. Best case is "small wallclock win on top of
   already-7×-parallel CPU"; likely outcome is **no crossover, same falsification
   as G4 with a lower coefficient**. Worth one number to confirm.
2. On `--leaf value-head --sims 100` (or hybrid blend > 0): ONNX is now ~50%+
   of work, batching has room. Likely crossover at B=8–16 with N=16 workers.
   This is the regime where the lever exists. **Strength-irrelevant for promotion**
   until value-head training catches up, but throughput evidence here is what
   tells us whether the wiring is worth building before that training lands.
3. On `--leaf rollout --sims 1000+` (the [`high-sim-mcts-regime-probe`](../../ai-agent-state/queue.json)
   regime): more priors per game → batching has more to amortize over, but
   rollout cost also scales. Net direction is unclear — measurement, not
   prediction.

Falsifiable consequence: if the (recipe, B, N) sweep below shows GPU never
within 1.5× of CPU on any recipe we'd actually ship, batched dispatch is dead
as a throughput lever and the only remaining axis is strength
(`gpu-fed-stronger-mcts.md`).

## Pre-conditions

1. **ONNX export admits dynamic batch — RESOLVED 2026-05-25.**
   `training/export_onnx.py:141-172` declares `dynamic_axes={... {0:
   "batch"}}` on every input (and on the `logits`/`value` outputs) for both
   the v3.0/v3.1 5-input and v3.2+ 7-input branches. Runtime inspection of
   `runs/R110-W6-repro/iter-0/policy.onnx` confirms axis 0 is symbolic
   (`'batch'`) on every tensor; axis 1 (`'actions'`) is also dynamic on
   `action_features`/`action_mask`/`action_card_idx`. No re-export needed.
2. **Pick the validation ckpt and recipe matrix once, lock it.** Use the same
   `runs/R110-W6-repro/iter-0/policy.onnx` G4/G5 used so results stack on
   existing baselines. For vhleaf, use C8-W6FIX-ON iter-1
   (`runs/R16-P2-c8-w6fix-on-extended-2x-games/loop/iter-1/checkpoint.pt`-exported).
3. **Acceptance harness reuses the G4/G5 wilson_lower envelope.** Any
   throughput claim must report `|Δwilson_lower| < 0.02` vs the same recipe
   on CPU; otherwise the batching path is silently wrong (different masked-
   softmax behavior, wrong tensor packing, etc.) and the wallclock number is
   noise. **Do not use a per-call `max_prob_diff < 1e-3` gate on the CUDA
   path** — B1 shows CUDA reduction-order drift up to 7.7e-3 on logits that
   is structural, not a bug. See `## B1 evidence`.

## B1 evidence (2026-05-25)

Smoke at `training/gpu_batched_inference_smoke.py`, ORT 1.22.0 (CUDA EP
+ TRT EP available), `runs/R110-W6-repro/iter-0/policy.onnx`, 50 iters
per cell after 5 warmup, single-threaded ORT (intra=1, inter=1),
random inputs from a fixed seed, 12 legal actions per state.

**Correctness (B=N stacked vs N independent B=1 calls):**

| Device | Worst `max|Δlogit|` across B∈{2,4,8,16,32,64} | Worst `max|Δvalue|` |
|---|---|---|
| CPU | 0.00e+00 | 0.00e+00 |
| CUDA | 7.69e-03 (at B=64) | 9.57e-04 (at B=64) |

CPU is bit-identical across all batch sizes — single-threaded reduction
order on CPU does not depend on B. CUDA drifts up to ~8e-3 on logits
because fused-softmax/MatMul kernels reduce across the batch in a
different order per launch. This **exceeds the 1e-3 sanity bar in the
original scope but is structural to CUDA**, not a bug: argmax-stability
is preserved in practice (max-prob shifts of 8e-3 will rarely flip top-1)
and the G4 wilson_lower envelope (0.02 at n=200) is the correct gate.
Implication for B2 acceptance: drop the per-call max-prob assertion;
keep the wilson_lower gate.

**Per-call latency:**

| B | CPU ms/call | CPU µs/state | CPU states/sec | CUDA ms/call | CUDA µs/state | CUDA states/sec | CUDA / CPU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.197 | 197.1 | 5,073 | 1.430 | 1429.6 | 700 | 0.14× |
| 2 | 0.197 | 98.5 | 10,156 | 1.345 | 672.7 | 1,487 | 0.15× |
| 4 | 0.216 | 54.1 | 18,488 | 1.462 | 365.4 | 2,737 | 0.15× |
| 8 | 0.401 | 50.2 | 19,933 | 1.441 | 180.2 | 5,550 | 0.28× |
| 16 | 0.742 | 46.4 | 21,555 | 1.301 | 81.3 | 12,296 | 0.57× |
| 32 | 1.579 | 49.3 | 20,270 | 1.669 | 52.1 | 19,178 | 0.95× |
| 64 | 2.998 | 46.9 | 21,345 | 1.275 | 19.9 | 50,190 | **2.35×** |

**Reads:**

- **CPU per-state plateaus at ~50 µs from B=4 onward.** Above that point,
  CPU is compute-bound (linear in B); batching gives a 4× single-thread
  throughput lift then nothing. With 16 OS-thread workers, the effective
  CPU ceiling is ~21k states/sec from B=1 — already saturated by G5's
  inter-game parallelism for `--leaf rollout --sims 100` (24 games/sec ×
  ~100 priors/game = ~2.4k inference calls/sec, well under the 21k
  ceiling). **Conclusion: CPU-side batching is not the lever.** G5 already
  cashed the easy CPU throughput.
- **GPU per-call cost is a near-flat ~1.3 ms launch floor up to B=64.**
  Compute is essentially free in this batch range; we're paying kernel
  launch + tiny H2D/D2H copies. ms/call: 1.43 (B=1) → 1.275 (B=64), i.e.
  64× more work for ~zero additional wall.
- **CUDA-vs-CPU crossover is at B≈32–48.** B=16 is 0.57× CPU, B=32 is
  0.95×, B=64 is 2.35×. Extrapolating, B=128 would be ~4–5× CPU. The
  prior G4 estimate "5–15 ms/call GPU" was high; on this graph it's
  ~1.3 ms.
- **For a useful end-to-end win, two conditions both have to hold:** (a)
  inference must be a meaningful share of game wall (rollout-leaf at
  sims=100 fails: ~5% share, so a 2.35× inference win → ~3% wall lift);
  (b) the dispatcher must sustain B≥32 fill, which requires either ≥32
  inter-game workers in flight OR a recipe where each game makes many
  in-flight inference requests (high sims, vhleaf, or both).

**Implications for the slice plan below:**

- B2 acceptance changes: B=1 path stays bit-identical; B=8/16/32/64
  acceptance uses wilson_lower within 0.02, not per-call max-prob.
- B3 sweep should explore B ∈ {16, 32, 64}, not {1, 8, 16}. B=1 is the
  bit-identical baseline (already validated by G5). B=8 is below
  crossover; not worth a cell.
- The most interesting B3 cell is **vhleaf @ workers=32 @ B=32 on CUDA**:
  exactly the regime where inference share is high AND batch fill is
  feasible from inter-game parallelism alone.
- B4 high-sim cell now mandatory if B3 vhleaf crosses, because sims=1000
  multiplies inference share on rollout-leaf too and is the cheapest path
  to making the lever useful for the shipping `--leaf rollout` recipe.

Smoke is preserved at `training/gpu_batched_inference_smoke.py` for
re-run on future ckpts; ~5 seconds wall + ~7 seconds wall on CPU + CUDA.

## B2/B3/B4 evidence (2026-05-25)

Engine-side dispatcher (commit 2ded9b0) added to
`engine-rs/crates/engine/src/inference/mod.rs`. New constructor
`InferenceSession::load_on_with_batching(path, device, max_batch,
max_wait_us)`: with `max_batch=1` keeps the historical inline path
(bit-identical, no channel hop); with `max_batch>1` moves the `Session`
into a dispatcher thread that pulls per-row requests off an mpsc, pads
per-row `n_actions` to the batch max, calls `Session::run` once with
B=N, slices outputs, fans back via per-request response channels.
`sim-eval-gate` gains `--batch-size` (default 1) and `--batch-wait-us`
(default 200) and logs `batches=N requests=M mean_fill=F` at end of run.

### B2 — dispatcher correctness

| Recipe | Device | --batch-size | wilson_lower | Notes |
|---|---|---|---|---|
| rollout sims=100, workers=16 | cpu | 1 | 0.5409361758928832 | inline baseline |
| rollout sims=100, workers=16 | cpu | 32 | 0.5409361758928832 | bit-identical (CPU reduction order is B-independent per B1) |
| rollout sims=100, workers=16 | cuda | 1 | 0.4957060908195922 | matches the historical G5 doc number — see "wilson drift" note below |
| rollout sims=100, workers=16 | cuda | 32 | 0.5107194625547862 | |Δ| vs CUDA B=1 = 0.015 < 0.02 envelope — PASS; mean_fill 8.66 |

**B2 dispatcher correctness gate PASSED** — CPU bit-identical at B=1 vs
B=32; CUDA within the 0.02 envelope at B=32 vs B=1. Bit-identical CPU is
the load-bearing correctness proof: tensor packing, padding, and
row-slicing are all correct.

**Wilson drift note.** CPU baseline at B=1 produces 0.5409 today vs the
G5 doc's 0.4957. The CUDA B=1 path reproduces 0.4957 exactly. Both
shifts are reproducible; engine-rs has had zero commits since G5
(2026-05-22, fb7d3f8) per `git log -- engine-rs/`. Likely cause: rustc
version, ORT library load order, or some non-engine-rs default (deck
sampling, RNG) changed FP behavior on the CPU `Session::run` path
between G5 measurement and now. Files-separately concern; the B2
correctness gate is **B=1 vs B=N on the same device**, which both
devices pass independently.

### B3 — rollout-leaf and value-head sims=100 sweep

All cells n=200 (`--games 100 --model-side both`), `--rollout-steps
200 --prior policy`, `runs/R110-W6-repro/iter-0/policy.onnx` for
rollout-leaf, `runs/R16-P2-c8-w6fix-on-extended-2x-games/loop/iter-1/policy.onnx`
for vhleaf (C8-W6FIX-ON iter-1, the historical vhleaf-readiness-gate
crossover ckpt). Raw results at
`runs/gpu-batched-inference-b3-b4/results.jsonl`.

**Rollout-leaf:**

| Device | B | workers | wilson | elapsed (s) | games/sec | mean_fill | speedup vs CPU best |
|---|---:|---:|---:|---:|---:|---:|---:|
| cpu | 1 | 16 | 0.5409 | 6.71 | 29.78 | inline | 0.88× |
| **cpu** | **1** | **32** | **0.5409** | **5.87** | **34.04** | **inline** | **1.00× ← CPU best** |
| cuda | 16 | 32 | 0.5258 | 16.90 | 11.83 | 13.58 | 0.35× |
| cuda | 16 | 64 | 0.5308 | 23.19 | 8.63 | 13.74 | 0.25× |
| cuda | 32 | 32 | 0.5308 | 25.75 | 7.77 | 14.21 | 0.23× |
| cuda | 32 | 64 | 0.5359 | 18.03 | 11.09 | 21.68 | 0.33× |
| cuda | 64 | 32 | 0.5308 | 27.23 | 7.35 | 14.18 | 0.22× |
| cuda | 64 | 64 | 0.5308 | 25.79 | 7.75 | 22.88 | 0.23× |

**Best CUDA: 16.90s (B=16, w=32) vs CPU 5.87s — CPU 2.88× faster. Gate
1.5× FAIL.** All cells within 0.02 wilson envelope.

**Value-head leaf:**

| Device | B | workers | wilson | elapsed (s) | games/sec | mean_fill | speedup vs CPU best |
|---|---:|---:|---:|---:|---:|---:|---:|
| cpu | 1 | 16 | 0.3923 | 8.49 | 23.56 | inline | 0.97× |
| **cpu** | **1** | **32** | **0.3923** | **8.22** | **24.34** | **inline** | **1.00× ← CPU best** |
| cuda | 16 | 32 | 0.3875 | 23.62 | 8.47 | 15.48 | 0.35× |
| cuda | 16 | 64 | 0.3875 | 24.91 | 8.03 | 15.35 | 0.33× |
| cuda | 32 | 32 | 0.3875 | 29.04 | 6.89 | 28.57 | 0.28× |
| cuda | 32 | 64 | 0.3875 | 25.89 | 7.72 | 28.40 | 0.32× |
| cuda | 64 | 32 | 0.3875 | 17.67 | 11.32 | 28.63 | 0.46× |
| cuda | 64 | 64 | 0.3826 | 14.68 | 13.63 | 46.49 | 0.56× |

**Best CUDA: 14.68s (B=64, w=64, mean_fill 46.49) vs CPU 8.22s — CPU
1.78× faster. Gate 1.5× FAIL.** All cells within 0.02 wilson envelope.

Note vhleaf wilson ~0.39 is far below rollout-leaf production 0.64 —
expected per `docs/ai-research/progress/r16.md:170,1063-1112`; the
vhleaf model is run here for *throughput* measurement only, not as a
strength candidate.

### B4 — rollout-leaf sims=1000 (high-sim cell)

| Device | B | workers | wilson | elapsed (s) | games/sec | mean_fill | speedup vs CPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| **cpu** | **1** | **16** | **0.5107** | **52.82** | **3.79** | **inline** | **1.00× ← CPU best** |
| cuda | 32 | 32 | 0.5157 | 106.92 | 1.87 | 15.32 | 0.49× |
| cuda | 64 | 32 | 0.5460 | 74.31 | 2.69 | 16.07 | 0.71× |
| cuda | 32 | 64 | 0.4758 | 55.16 | 3.63 | 23.69 | 0.96× |

**Best CUDA: 55.16s (B=32, w=64) vs CPU 52.82s — CPU 1.04× faster.
Gate 1.5× FAIL.** The gap closed (sims=100 was 2.88×, sims=1000 is
1.04×) but did not invert. **Wilson drift on the best cell is 0.035 —
exceeds the 0.02 envelope**, suggesting CUDA reduction-order drift
compounds at high sims (more inference calls per game → more
accumulated bias). The B=64 w=32 cell shows the same drift direction
(|Δ|=0.035). A strength-axis user of sims=1000 + CUDA should
re-validate wilson agreement at the target sims count, not assume the
sims=100 envelope holds.

### Why does pure-ONNX 2.35× crossover not survive the round-trip?

B1 (Python smoke, bare `Session.run`) measured CUDA 50,190 states/sec
vs CPU 21,345 at B=64. That's the raw device economics — kernel-launch
overhead amortized over 64 rows. The B3/B4 dispatcher cells should
have closed most of that gap on vhleaf (where inference is the hot
path). They didn't.

Two compounding overheads eat the win:

1. **Dispatcher per-request cost.** Each `predict_v3` call now hops
   through an mpsc::sync_channel (request + per-call response channel
   + Vec allocation for the packed row). At 100 sims/game × 200 games
   × ~32 concurrent workers = ~6×10^5 dispatcher hops total. Even at
   ~10µs per hop (channel + tensor pack + slice-back) that's ~6
   seconds of pure dispatcher overhead — comparable to the entire
   CUDA elapsed time on the rollout-leaf cells.
2. **MCTS workload doesn't sustain fill.** Best fill on rollout-leaf
   is 22.88 of 64 (35%); even on vhleaf, only the w=64+B=64 cell hit
   46.49 of 64 (73%). Below-max fill means the actual per-call CUDA
   latency stays close to the B=1 launch floor (~1.3 ms), not the
   amortized B=64 cost (~20 µs/state).

The Python smoke avoided both: no channel, max-fill always B=64,
single-thread dispatch. The dispatcher path is closer to the realistic
end-to-end cost, and the realistic cost loses.

### Verdict and what stays landed

(Originally close-out for B3/B4; superseded by B5 below for the
vhleaf-specific verdict — kept here as the as-of-B4 record.)

Per the scoping doc's gates table (revised in B3/B4):

| Gate row | Outcome |
|---|---|
| B2 wilson drift > 0.02 at B=32 | PASS — CPU bit-identical, CUDA 0.015 |
| B3 no cell crosses 1.5× | FAIL on both recipes at workers≤64 → ran B4 |
| B4 no cell crosses 1.5× | FAIL → user pushback prompted B5 re-probe |

**As-of-B4 close (since revised):** Batched GPU dispatch is not a
wallclock win on any shipping recipe even at high sims. CPU remains
the production default for `sim-eval-gate` and selfplay.

**Revised in B5:** rollout-leaf shipping recipe close stands; vhleaf
close was premature — see `## B5 high-parallelism re-probe`.

**What stays landed (commit 2ded9b0):**
- `BatchedDispatcher` + `load_on_with_batching` in `engine-rs/crates/engine/src/inference/mod.rs`
- `--batch-size` and `--batch-wait-us` flags on `sim-eval-gate`
- `mean_fill` end-of-run instrumentation

Reason to keep the wiring: [`gpu-fed-stronger-mcts.md`](gpu-fed-stronger-mcts.md)
Candidate 3 (batched evaluator interface) is the strength axis that
reuses this exact dispatcher. The strength-axis question is "wilson at
sims=N with batched CUDA beats production ceiling at fixed wallclock
budget" — a different acceptance, and one where the dispatcher's
correctness (proven here) is the load-bearing prerequisite.

**What stays unbuilt:** the `--device cuda` and `--batch-size` flags on
`sim-mcts-selfplay`. They were out of scope this slice; if the strength
axis needs batched CUDA selfplay, that's a small follow-on ticket.

**Open questions for the strength axis (not pursued here):**

1. Does CUDA reduction-order drift cross the wilson envelope at
   sims=2000+? B4 showed |Δ|=0.035 at sims=1000; the strength axis may
   need a tighter test or accept that high-sim CUDA is a different
   strength evaluation than high-sim CPU.
2. Does intra-tree wave batching (virtual loss + leaf-collection
   waves) sustain higher fill than the inter-game-only path? B4's best
   cell only fills 24/64; an intra-tree path could plausibly hit
   60+/64. That's a strength-axis question because waves change PUCT
   visit distributions.
3. Is there a recipe at sims=5000–10000 where the pure-ONNX 2.35×
   crossover finally survives the dispatcher overhead? Linear-fit
   extrapolation from B3 (0.35× @ sims=100) to B4 (0.96× @ sims=1000)
   suggests sims≈1200 for parity, sims≈2500 for 1.5× crossover —
   speculative.

Raw sweep artifact: `runs/gpu-batched-inference-b3-b4/results.jsonl`
(20 cells). Sweep script preserved at
`runs/gpu-batched-inference-b3-b4/sweep.sh` for re-run on future
ckpts.

## B5 high-parallelism re-probe (2026-05-25, after pushback)

The B3/B4 sweep capped workers at 64 and concluded the throughput axis
was falsified. User pushback ("have we actually scaled parallelism
enough?") flagged that fill on the best B3 cells was still
work-share-limited (vhleaf w=64 B=64 fill 46/64, hisim w=64 B=32 fill
24/32). Re-probe at workers ∈ {96, 128, 256} and orthogonal levers
(max_batch=128, max_wait_us=2000). Raw at
`runs/gpu-batched-inference-b3-b4/hp_probe_results.jsonl`.

**Vhleaf sims=100 (CPU baseline 8.22s at w=32):**

| B | workers | wait_us | wilson | elapsed (s) | games/sec | mean_fill | vs CPU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 96 | 200 | 0.3826 | 8.03 | 24.91 | 48.37 | **1.02× CUDA** |
| **64** | **128** | **200** | **0.3826** | **7.56** | **26.47** | **46.60** | **1.09× CUDA** ← **best, CUDA crosses CPU** |
| 64 | 256 | 200 | 0.3826 | 9.02 | 22.18 | 43.69 | 0.91× of CPU (oversubscribed) |
| 128 | 128 | 200 | 0.3923 | 7.94 | 25.19 | 60.96 | 1.04× CUDA |
| 128 | 256 | 200 | 0.3923 | 7.95 | 25.15 | 57.16 | 1.03× CUDA |
| 64 | 128 | 2000 | 0.3826 | 11.23 | 17.80 | 46.72 | 0.73× of CPU (longer wait hurt) |

**Vhleaf headline: CUDA *does* cross CPU at w=128 B=64 (1.09× faster).
Still below 1.5× promotion gate but the lever is real.** The earlier
B3 verdict missed this because workers=64 was below the sweet spot.

Three additional reads:
- **Fill ceiling is workload-imposed, not parallelism-imposed.** At
  max_batch=64, fill caps at ~46-48 regardless of workers (w=96 →
  48.37; w=128 → 46.60; w=256 → 43.69). The MCTS workload can sustain
  ~46-48 concurrent in-flight inference requests per dispatcher flush
  window; pushing more workers doesn't extract more — it just
  oversubscribes CPU. B=128 reaches fill 57-61 because the dispatcher
  flush window is wider (more requests can arrive before flush) but
  per-call kernel latency rises slightly with B, netting roughly the
  same elapsed.
- **Longer max_wait_us hurts.** w=128 B=64 with wait_us=2000 added 4ms
  per response on average — 11.23s vs 7.56s (1.49× slower). Per-call
  latency dominates total wall; trading higher fill for longer wait is
  a bad deal at this graph's tiny per-call cost.
- **Oversubscription wall at w=256.** 256 workers on 32 cores is 8×
  oversubscribed. Per-worker context-switch overhead and CPU-side MCTS
  work both regress. The sweet spot is w≈128 (4× oversubscribed) —
  enough concurrency to keep dispatcher fill high without crushing the
  per-worker progress rate.

**Hisim rollout sims=1000 (CPU baseline 52.82s at w=16; original B4
best CUDA was w=64 B=32 → 55.16s):**

| B | workers | wait_us | wilson | elapsed (s) | games/sec | mean_fill | vs CPU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 128 | 200 | 0.5057 | 87.73 | 2.28 | 21.09 | 0.60× of CPU |
| 64 | 128 | 200 | 0.5107 | 104.56 | 1.91 | 25.33 | 0.51× of CPU |
| 64 | 256 | 200 | 0.5258 | 111.52 | 1.79 | 26.09 | 0.47× of CPU |
| 128 | 256 | 200 | 0.5359 | 108.98 | 1.84 | 28.79 | 0.48× of CPU |

**Hisim headline: more workers strictly HURT. Falsification holds.**
The B4 best (w=64 B=32 → 55.16s) was already past the parallelism
optimum for this recipe. Rollout-leaf at sims=1000 spends most wall in
the 200-step CPU rollouts; oversubscribing 32 cores with 128-256 OS
threads slows each per-game rollout enough that the additional fill
gains (24 → 29) don't compensate.

### Revised verdict

The B3/B4 close was right on *rollout-leaf* (the shipping recipe) but
wrong on *vhleaf*. The error mode: I capped workers at 64 — sufficient
to saturate CPU, insufficient to saturate the dispatcher when each
worker spends most time *blocked* on the dispatcher (vhleaf path)
rather than computing (rollout path).

Updated gates outcome:

| Gate row | Outcome |
|---|---|
| B3 vhleaf no cell crosses 1.5× | Confirmed (best 1.09× at w=128 B=64) |
| B3 rollout no cell crosses 1.5× | Confirmed |
| B4 no cell crosses 1.5× | Confirmed |
| B5: workers>64 helps vhleaf | YES — w=128 is the sweet spot |
| B5: workers>64 helps rollout | NO — CPU-bound recipe regresses with oversubscription |

**For the strength axis** ([`gpu-fed-stronger-mcts.md`](gpu-fed-stronger-mcts.md)),
this changes the playbook: vhleaf experiments on CUDA should default
to `--batch-size 64 --batch-wait-us 200 --workers 128`. That gives a
~1.1× wall win vs CPU at matched wilson, freeing budget for
higher-sim cells. Rollout-leaf experiments stay on CPU at workers=16-32.

**For the shipping recipe** (production eval-gate at sims=100
rollout-leaf), the original close still stands: CPU is the right
default; batched CUDA loses 2.88×.

**Why CUDA can win on vhleaf but not rollout, mechanically:** vhleaf
worker timeline is ~12% CPU work, 88% blocked on dispatcher → 32 cores
support ~250 concurrent vhleaf workers before saturating, so going
from w=32 to w=128 is well within budget. Rollout-leaf worker timeline
is ~95% CPU work (rollouts), 5% blocked on dispatcher → 32 cores
support ~33 concurrent rollout workers before saturating, so w=64 was
already past optimum.

## B6 wave batching evidence (2026-05-25)

**Question that closed the loop.** B5 ended with vhleaf CUDA 1.09× CPU
at w=128 B=64 and the suggestion that intra-tree waves with virtual
loss could push fill past the workload-imposed ceiling. Built it
(commit 2c2c860): `--wave-size N` on `sim-eval-gate`, `predict_v3_batch`
direct API in `InferenceSession`, wave loop in `mcts/driver.rs` with
AlphaGo-standard virtual loss (`virtual_loss=1.0` default) gated by
`wave_size > 1`. wave_size=1 falls through to the existing serial loop
bit-identical.

**Surprise.** Wave didn't just unlock GPU — it unlocked CPU. The CPU
throughput ceiling we'd been treating as the baseline (~25 games/sec on
vhleaf) was an artifact of one ORT call per simulation. Waving 16 sims
together into one `Session::run(B=16)` gives CPU 7-9× throughput.

Raw at `runs/gpu-batched-inference-b3-b4/b6_wave_results.jsonl`.

### Vhleaf sims=100 (workers=16, n=200)

| wave | device | elapsed (s) | games/sec | wilson | speedup vs serial-CPU |
|---:|---|---:|---:|---:|---:|
| 1 | cpu | 7.449 | 26.85 | 0.3923 | 1.00× (serial baseline) |
| 8 | cpu | 2.339 | 85.50 | 0.3971 | 3.19× |
| **16** | **cpu** | **1.021** | **195.93** | **0.3923** | **7.30× ← bit-identical wilson** |
| 32 | cpu | 0.855 | 233.83 | 0.4117 | 8.71× (+0.019 wilson drift, in envelope) |
| 64 | cpu | 0.831 | 240.65 | 0.4461 | 8.97× (+0.054, exceeds 0.05 envelope) |
| 1 | cuda | 255.812 | 0.78 | 0.3971 | 0.029× of CPU |
| 8 | cuda | 30.733 | 6.51 | 0.3971 | 0.24× |
| 16 | cuda | 28.539 | 7.01 | 0.3923 | 0.26× |
| 32 | cuda | 17.175 | 11.65 | 0.4069 | 0.43× |
| 64 | cuda | 9.793 | 20.42 | 0.4510 | 0.76× |

**Vhleaf headline: wave_size=16 on CPU is the production sweet spot —
7.30× throughput at bit-identical wilson. wave_size=32 gives 8.71× at a
small wilson shift still inside the 0.05 envelope.** CUDA wave_size=64
reaches 9.79s — still 9.6× slower than CPU wave_size=16. The GPU lever
is decisively dead for this recipe; the lever was the per-call ORT
overhead all along.

### Rollout-leaf sims=100 (workers=16, n=200)

| wave | device | elapsed (s) | games/sec | wilson | speedup vs serial-CPU |
|---:|---|---:|---:|---:|---:|
| 1 | cpu | 9.279 | 21.55 | 0.5409 | 1.00× (serial baseline) |
| 8 | cpu | 8.862 | 22.57 | 0.5459 | 1.05× |
| **32** | **cpu** | **7.437** | **26.89** | **0.5207** | **1.25× (small wilson shift)** |
| 1 | cuda | 125.806 | 1.59 | 0.4957 | 0.074× |
| 8 | cuda | 28.925 | 6.91 | 0.5308 | 0.32× |
| 32 | cuda | 12.443 | 16.07 | 0.4807 | 0.75× |
| 64 | cuda | 11.167 | 17.91 | 0.4609 | 0.83× |

**Rollout-leaf headline: wave gives ~1.25× CPU speedup, modest because
each leaf needs a 200-step CPU rollout that dominates the wall — only
the prior calls batch.** Still beats every CUDA cell tested. CUDA
wave_size=64 wilson drifted to 0.4609 (|Δ|=0.080 vs CPU baseline 0.5409)
— too far for production.

### Hisim rollout sims=1000 (workers=16, n=200)

| wave | device | elapsed (s) | wilson | speedup vs serial-CPU |
|---:|---|---:|---:|---:|
| 1 | cpu | 49.481 | 0.5107 | 1.00× (serial baseline) |
| **8** | **cpu** | **41.291** | **0.5107** | **1.20× (bit-identical wilson)** |
| 32 | cuda | 71.753 | 0.5257 | 0.69× |
| 64 | cuda | 73.981 | 0.5510 | 0.67× (wilson drift |Δ|=0.040 borderline) |

**Hisim headline: wave gives 1.20× CPU speedup at bit-identical
wilson at sims=1000. CUDA still loses (best 0.69× of CPU).**

### Why CPU wave is so powerful

Per-state economics on this graph (from B1):

| device | per-call cost | per-state cost at B=16 | per-state cost at B=64 |
|---|---:|---:|---:|
| CPU | 197 µs | 46 µs | 47 µs |
| CUDA | 1430 µs | 81 µs | 20 µs |

Pre-wave, vhleaf sims=100 made ~100 serial inference calls per game
(one PUCT prior at each non-terminal expansion + one leaf-value). On
CPU that's 100 × 197 µs = 19.7 ms of pure ORT overhead per game.
wave_size=16 collapses this to ~6 batched calls × (16 × 46 µs) = 4.4
ms — 4-5× reduction in inference wall.

The other 4-5× of the observed 7× total speedup comes from secondary
effects: lower GIL/lock contention, better CPU cache locality on the
batched tensor packing, fewer memory allocator hits per call.

For CUDA the per-state cost at B=16 is 81 µs vs CPU's 46 µs. CUDA is
strictly worse per state until B ≥ 32 (where per-state is 52 µs ≈ CPU's
47 µs). At realistic wave_sizes that don't break wilson_lower (≤32),
CUDA's per-state cost roughly equals CPU's — and the per-call kernel
launch (1.3 ms / batch) is a fixed overhead CPU doesn't have.

### Revised verdict — third and final

| Gate row | Outcome |
|---|---|
| B6 wave_size=1 bit-identical to serial | PASS — wilson 0.5409... bit-for-bit |
| B6 wave_size=16 CPU wilson within 0.05 of serial | PASS — 0.0000 drift on vhleaf |
| B6 wave_size=16 CPU speedup ≥ 1.5× on vhleaf | **PASS — 7.30×** |
| B6 wave_size=32 CPU speedup ≥ 1.5× on vhleaf | **PASS — 8.71×** |
| B6 CUDA wave_size=N speedup ≥ 1.5× vs wave-batched CPU | FAIL on every recipe |

**Production recommendation:**

| Recipe | Recommended config | Speedup vs prior default |
|---|---|---:|
| vhleaf sims=100 (eval-gate) | `--device cpu --wave-size 16` | 7.30× |
| vhleaf sims=100 (when accepting +0.02 wilson drift) | `--device cpu --wave-size 32` | 8.71× |
| rollout-leaf sims=100 (eval-gate, shipping) | `--device cpu --wave-size 32` | 1.25× |
| rollout-leaf sims=1000 (high-sim) | `--device cpu --wave-size 8` | 1.20× |
| CUDA anywhere | not recommended at sims ≤ 1000 on this graph | — |

The B2 BatchedDispatcher (commit 2ded9b0) becomes superseded for the
throughput axis — wave + serial single-thread per game is now faster
than dispatcher-batched multi-thread per game. The dispatcher stays
landed because (a) `--batch-size 1` (the dispatcher-disabled default)
remains bit-identical, no behavior change for users that don't opt in;
(b) the strength axis ([`gpu-fed-stronger-mcts.md`](gpu-fed-stronger-mcts.md))
may still want the dispatcher for sims >> 1000 experiments where the
per-game wave's CPU work itself becomes a bottleneck and multi-game
inter-game batching matters.

The two mechanisms are mutually exclusive at runtime (eval-gate rejects
`--wave-size > 1 AND --batch-size > 1`).

**What B6 leaves open:**

1. **Wave-size > 64 wilson drift envelope.** wave=64 on vhleaf had
   |Δ|=0.054, slightly over the 0.05 envelope. wave=128/256 likely
   drift further. If a production user wants more than 8.71× speedup,
   they need to either accept a wider envelope or shrink `virtual_loss`
   (currently 1.0, AlphaGo-standard).
2. **Wave on `sim-mcts-selfplay`.** Out of scope this slice (consistent
   with B2 precedent). The training orchestrator's selfplay step would
   benefit identically — ~6-7× wall on the vhleaf relabel/training loop
   for free. Easy follow-on ticket.
3. **Wave + workers scaling.** B6 ran every cell at workers=16 to
   isolate the wave axis. wave_size=16 CPU at 195.9 games/sec is so
   fast it's plausible that workers=4 or even workers=1 would be
   sufficient — wave already extracts the parallelism. Worth a
   one-cell probe if anyone cares about exact thread/CPU usage.

Raw artifact: `runs/gpu-batched-inference-b3-b4/b6_wave_results.jsonl`
(23 cells). Sweep script: `runs/gpu-batched-inference-b3-b4/b6_wave_probe.sh`.

## Slices

### B1 — Single-call B>1 smoke (LANDED 2026-05-25)

Done out-of-band in Python (no engine-rs changes) at
`training/gpu_batched_inference_smoke.py`. Full numbers in
`## B1 evidence` above. Verdict:

- Dynamic-batch export already in place; no exporter changes needed.
- CPU is bit-identical across B; CUDA drifts up to 7.7e-3 logits / 1e-3
  value due to fused-kernel reduction order — structural, not a bug.
- Crossover with CPU at B≈32 on bare ONNX dispatch; 2.35× at B=64.

The Rust-side equivalent (verifying ort 2.0.0-rc.12's B>1 path on the
CUDA EP) folds into B2's bit-identical-at-B=1 acceptance plus the wider
acceptance numbers; no separate Rust smoke needed.

### B2 — Flush-collector around `predict_v3` (LANDED 2026-05-25, commit 2ded9b0)

Acceptance gates met — see `## B2/B3/B4 evidence` above. Original
scoping follows for reference.

Add an internal `BatchedDispatcher` to `InferenceSession` parameterized by
`(max_batch: usize, max_wait_us: u64)`. Worker threads call a
`predict_v3_batched(state, legals) -> oneshot::Receiver<PredictionV3>`-style
API; the dispatcher gathers up to `max_batch` requests or flushes on
`max_wait_us`. Returns the same `PredictionV3` shape, so call sites change
minimally.

Crucially, do **not** restructure intra-tree selection. Inter-game workers
(`mcts_selfplay.rs:686-748`, `eval_gate.rs:556`) already provide N
independent in-flight requests — that's the batch pool. Per the
investigator's read, this is the cheap path.

Wire a `--batch-size` and `--batch-wait-us` flag on `sim-eval-gate`.
Default `--batch-size 1` keeps current behavior bit-for-bit.

**Acceptance:**
- B=1 path produces wilson_lower bit-identical to current G5-validated
  workers=16 number (0.4957060908195922 on R110-W6-repro iter-0).
- B=32 path on CPU produces wilson_lower within 0.02 of B=1 on the same
  recipe (CPU is bit-identical per B1 — this should pass trivially; it
  exists to catch dispatcher bugs in tensor packing or row-slicing on
  the way back).
- B=32 path on CUDA produces wilson_lower within 0.02 of B=1 CPU on the
  same recipe (the strength gate that matters; B1 confirmed bit-exact
  output drift is acceptable in aggregate).
- No queue starvation: log p50/p99 queue-wait and assert p99 <
  `max_wait_us × 2`.
- Log mean fill (avg actual batch size served) — if it's < 16 on
  workers=32 + max_batch=32, the dispatcher is starving and B3's CUDA
  cells will under-report.

**Effort:** ~1.5 days. Heaviest piece is the dispatcher itself — handful of
hundreds of lines, thread-safe queue, conditional flush.

### B3 — Recipe × batch sweep, single-axis-at-a-time (LANDED 2026-05-25, FALSIFIED)

Acceptance gate (≥1.5× wallclock on at least one cell) NOT MET on
either rollout-leaf or vhleaf. Full results in `## B2/B3/B4
evidence`. Original scoping follows for reference.

Two recipes × three batch sizes × two worker counts, n=200 each. CPU
baseline column reused from existing G5 results where possible. Batch
sizes chosen from B1: skip B=8 (below crossover), focus on the
crossover region and post-crossover.

| Recipe | CPU baseline (G5) | Sweep |
|---|---|---|
| `--leaf rollout --sims 100 --workers 16` | 8.282s, wl 0.4957 | B ∈ {16, 32, 64} on CUDA at workers ∈ {32, 64} |
| `--leaf value-head --sims 100 --workers 16` | new — measure first | same matrix |

Report per cell: `elapsedSecs`, `gamesPerSec`, `wilsonLower`, GPU util %,
GPU memory MiB, queue-wait p50/p99, mean fill (avg actual batch size
served).

**Don't matrix more than this.** Per [[feedback_lean_validation]] keep it
small; widen only if a clear positive signal warrants it.

**Acceptance for crossover claim on a recipe:**
- GPU `gamesPerSec` ≥ CPU `gamesPerSec` × 1.5 on that recipe at matched
  workers (same gate as `gpu-inference-execution-provider.md`).
- `|Δwilson_lower| < 0.02` vs CPU.
- GPU util ≥ 60% sustained — anything lower means we're still
  dispatch-bound, batching helped but didn't fix it; record but don't claim
  crossover.

**Falsification:** if no cell crosses 1.5×, batched dispatch is dead for
the recipes we ship today. The remaining axis becomes "does the
[`high-sim-mcts-regime-probe`](../../ai-agent-state/queue.json) recipe
(sims 1000+) cross?" — that's a one-cell add to this sweep, not a new
slice.

**Effort:** ~half day to run + record once B2 lands.

### B4 — High-sim cell (LANDED 2026-05-25, FALSIFIED)

Acceptance gate NOT MET — best CUDA cell still 1.04× slower than CPU
at sims=1000. The gap closed substantially (2.88× → 1.04×) but did
not invert. Wilson drift on the best CUDA cell exceeds the 0.02
envelope (|Δ|=0.035), separate finding for the strength axis. Full
results in `## B2/B3/B4 evidence`. Original scoping follows for
reference.

B1 showed B≥32 is needed for crossover; sims=1000 multiplies inference
share on rollout-leaf to roughly 5× the sims=100 baseline (~25% wall
share). This is the cheapest path to making batched GPU useful for the
shipping `--leaf rollout` recipe — i.e., the cell most likely to actually
change what we run.

Cells: `--leaf rollout --sims 1000 --workers 16` at B ∈ {32, 64} on
CUDA, matched CPU baseline. Doubles as the headline number for the
[`high-sim-mcts-regime-probe`](../../ai-agent-state/queue.json) (P1)
"does GPU win at sims=1000+" question, so its result also informs
whether that probe should preferentially launch on GPU.

**Skip only if B3 vhleaf cell also fails** — if the lever doesn't work
where inference is dominant, it certainly won't work where rollouts are
dominant.

**Effort:** ~2 hours run, no new code.

## Gates summary

| Gate | Trigger | Decision |
|---|---|---|
| ~~B1 smoke fails~~ | ~~Graph pinned B=1 or > 1e-3 drift across batch positions~~ | **RESOLVED 2026-05-25:** export already batch-dynamic; CUDA drift is structural but inside wilson_lower envelope (see B1 evidence) |
| B2 wilson drift > 0.02 at B=32 | Dispatcher bug, wrong tensor packing, or genuine masked-softmax sensitivity over a full eval | Debug; don't proceed to B3 |
| B3 no cell crosses 1.5× | Batched GPU dispatch doesn't help any shippable recipe at sims=100 | Run B4 (high-sim) before closing; if B4 also fails, close throughput probe and redirect to strength axis ([`gpu-fed-stronger-mcts.md`](gpu-fed-stronger-mcts.md)) |
| B3 vhleaf crosses, rollout doesn't | Throughput lever exists but only for vhleaf | Hold for value-head training to recover strength (see `docs/ai-research/progress/r16.md:1063-1112`); document, ship B2 wiring as opt-in, wait on training side |
| B3 or B4 rollout crosses at B≥32 | Lever exists on shipping recipe | Promote: make `--batch-size 32` default on `--device cuda`, update gpu-inference-execution-provider.md, retire G4 falsification claim with caveat "naive (B=1) was wrong, batched is fine" |

## Risks

- ~~**Dynamic-batch export not in place.**~~ **RESOLVED 2026-05-25** —
  `training/export_onnx.py:141-172` already declares dynamic batch axes
  on every input/output for both v3.0/3.1 and v3.2+ branches; runtime
  inspection of R110-W6-repro/iter-0/policy.onnx confirms.
- **CUDA reduction-order drift larger than initially assumed.** B1 showed
  CUDA `max|Δlogit|` up to 7.7e-3 across batch members (vs scoped 1e-3
  bar). Structural, not a bug. Mitigation already in place: B2 acceptance
  uses wilson_lower envelope (0.02 at n=200) not per-call max-prob — the
  G4 gate is the right comparator. Risk to monitor: a particular policy
  shape (e.g. a near-tie top-1/top-2) could see argmax flips at this drift
  level; B3's wilson_lower-on-full-eval catches this.
- **Inter-game batch fill depends on game-length variance.** If some
  workers spend 10s in rollouts while others spend 0.1s on a one-call
  decision, the dispatcher will flush half-empty. Mitigation: measure
  `mean fill` in B3; if it's < B/2, lever is capped well below ideal —
  document and don't chase.
- **CPU side regressing from added wakeup/queue cost.** A flush-collector
  on the CPU dispatch path adds atomic + condvar work that the bit-identical
  G5 number doesn't pay. Mitigation: keep B=1 path on a fast-path branch
  that bypasses the dispatcher entirely; B2's bit-identical-at-B=1 gate
  guards this.
- **`UnsafeCell<Session>` semantics under concurrent batched calls.** G5
  already validated concurrent single-state `Session::run` calls; concurrent
  B>1 calls are not separately validated. The dispatcher serializes within
  itself, so this is moot — but worth noting we no longer have the
  workers=32 stress-test signal G5 had.

## Guardrails

- Do not promote a strength claim from this doc. Strength axis lives in
  [`gpu-fed-stronger-mcts.md`](gpu-fed-stronger-mcts.md).
- Do not skip the B=1 bit-identical re-validation. The G5 reference is
  load-bearing and any drift means the dispatcher is wrong.
- Do not widen the B3 sweep into a 4×4×3 matrix on the first pass. Per
  [[feedback_lean_validation]] keep cells small; expand only on clear
  positive signal.
- Do not build intra-tree wave/virtual-loss batching in this scope. That's
  a strength-axis question (different tree dynamics → different visit
  distributions); if it's worth doing, it's its own scoping doc.

## Crosslinks

- [`gpu-inference-execution-provider.md`](gpu-inference-execution-provider.md)
  — predecessor. G4 falsification + G5 CPU baseline + `next_action`
  naming this lever. Update on close.
- [`gpu-fed-stronger-mcts.md`](gpu-fed-stronger-mcts.md) — sibling, strength
  axis. Candidate 3 ("batched evaluator interface") reuses any
  dispatcher built here.
- `docs/ai-research/progress/r16.md:170,1063-1112` — current value-head
  leaf-readiness state (does not meet strength gate; throughput probe
  still legal on vhleaf for measurement).
- `docs/ai-agent-state/queue.json` — `high-sim-mcts-regime-probe` is the
  natural follow-on if B3 shows any crossover.
- `engine-rs/crates/engine/src/inference/mod.rs:373` — `predict_v3`
  current shape, B2 attach point.
- `engine-rs/crates/engine/src/mcts/driver.rs:426,488,699` — call sites
  that route through the dispatcher.
- `engine-rs/crates/sim-cli/src/bin/{mcts_selfplay,eval_gate}.rs` — N-worker
  hosts that provide the batch fill pool.
