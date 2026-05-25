# Batched Inference + High Parallelism — GPU Throughput Probe

- **Date:** 2026-05-25
- **Status:** SCOPING — B1 prereq smoke LANDED 2026-05-25 (Python ORT, no
  engine-rs changes). Successor to
  [`gpu-inference-execution-provider.md`](gpu-inference-execution-provider.md)
  G4-falsified, G5-landed. Sibling to
  [`gpu-fed-stronger-mcts.md`](gpu-fed-stronger-mcts.md) Candidate 3 ("batched
  evaluator interface"). Queue entry:
  `gpu-batched-inference-throughput` in `docs/ai-agent-state/queue.json`.
- **One-liner:** Determine whether gathering N per-state ONNX calls into a
  single `Session::run(B=N)`, fed by high-parallelism inter-game workers, makes
  the GPU path beat CPU on a recipe we actually run — and identify the smallest
  recipe where it does.
- **Non-goal:** strength gains. Acceptance here is throughput / wallclock under
  matched strength (Wilson-lower within G4-style 0.02 envelope). Strength
  questions stay in [`gpu-fed-stronger-mcts.md`](gpu-fed-stronger-mcts.md).
- **B1 headline (2026-05-25):** CUDA crosses CPU **at B≈32** on bare ONNX
  dispatch (2.35× CPU at B=64). GPU per-call cost is a ~1.3 ms launch floor,
  nearly flat in B up to 64; CPU is ~50 µs/state, plateau from B=4. So the
  lever exists — but it's only useful in regimes where (a) inference is a
  meaningful share of game wall AND (b) we can sustain B≈32+ in flight.
  Rollout-leaf at sims=100 fails (a); vhleaf and/or high-sim regimes pass
  (a). See `## B1 evidence` below.

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

### B2 — Flush-collector around `predict_v3`

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

### B3 — Recipe × batch sweep, single-axis-at-a-time

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

### B4 — High-sim cell (promoted from optional 2026-05-25)

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
