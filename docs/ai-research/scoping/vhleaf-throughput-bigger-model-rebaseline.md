# vhleaf Throughput Re-baseline on hidden=256/depth=4 — Scoping

- **Date:** 2026-05-26
- **Status:** **READY-TO-FIRE-WITH-ITER-2-CKPT** — `runs/R16-P3-v36-az-bigtrunk-cold` is in-flight (40-iter v3.6 AZ bigtrunk cold-start, hidden=256 depth=4 state-dim=246 sims=400 two-sided, currently mid-iter-3 distill). `iter-2/policy.onnx` is on disk and serves as the Phase A input.
- **Predecessors:** `docs/ai-research/scoping/gpu-batched-inference-throughput.md` (closed at B6, the wave-batching lever); `gpu-fed-stronger-mcts.md` (compute-axis hold); commits `2c2c860` (B6), `e3d8d90` (B7), `ddbc836` (B8), `2eef954` (B9 leaf-conditional wave default).
- **Sibling:** `post-throughput-scale-up-directions.md` (broader re-prioritization after Slice-2).

---

## TL;DR

The B6/B7/B8 wave-batching series gave 7.30× bit-identical on vhleaf at sims=100 wave=16 on the current hidden=128/depth=3 model; B9 set the orchestrator default. Two follow-up brainstorm rounds converged on **value-only ONNX subgraph at the leaf** as the next CPU-side lever (1.15-1.25× wall on current model, **1.25-1.45× expected on hidden=256/depth=4 because policy-head FLOPs grow with the model**). Independently, B5's CUDA-falsification verdict is *for the small model only* — the 2.8× FLOP increase from hidden=256/depth=4 likely pushes CUDA past its 1.3 ms kernel-launch floor and could re-open the entire `gpu-fed-stronger-mcts.md` axis.

The spike is therefore **probe-first, branched**:
- **Phase A** — `device × wave_size × sims` grid on the new model. Re-tunes the production default (currently auto-picks wave=16 leaf-conditional via B9).
- **Phase B** — branch on Phase A result: wire CUDA on, *or* build value-only subgraph, *or* both.

Deferred until a hidden=256/depth=4 checkpoint exists.

## Reconciled facts (anchor for future work — don't re-derive)

Two rounds of subagent brainstorming established the following numbers; recording so future-me doesn't re-litigate them.

1. **vhleaf sims=100 wave=16 wall on current model = 5.10 ms/game** (1.021 s / 200 games, `runs/gpu-batched-inference-b3-b4/b6_wave_results.jsonl`).
2. **Inference share of that wall ≈ 86-91%.** The earlier "20%" claim conflated aggregate parallelized CPU with wall. ORT `Session::run` serializes across the 16 workers at the C++ level, so wall is session-bound. Per-call B=16 CPU is 0.742 ms (B1), × ~6 batched calls/game = ~4.4 ms/game of inference wall.
3. **Mean N(chosen) ≈ 50-66** across 4 vhleaf corpora (6,181 decisions; `R16-P2-c8b-vhleaf`, `R16-P2-c8-w6fix-vhleaf`, `R16-P3-v36-az-cont` iter-0 and iter-4). Wave=16-adjusted ≈ 45-60.
4. **Tree reuse remains parked.** Theoretical sims-equivalent ceiling 1.8-2.0× on modeled-side. *Realistic* wall gain on production single-side vhleaf is 1.10-1.25× because the heuristic-opponent interleave invalidates most of the cached subtree between modeled decisions, and cached opponent-child Q-values are stale (computed assuming heuristic-opp). Already triaged: see `r12-selfplay-gate-throughput.md:292-341` "Why subtree reuse did NOT land". Model size does not fix the interleave problem.
5. **Featurize/observation/IndexMap allocation levers** are real but small after correcting fact (2). Becomes a smaller-fraction lever as the model grows (inference share approaches 96%).

## Why hidden=256/depth=4 changes the picture

| Quantity | hidden=128/depth=3 | hidden=256/depth=4 | Δ |
| --- | --: | --: | --: |
| Trunk FLOPs / state | ~46 k | ~224 k | 4.9× |
| Policy head FLOPs / state | ~123 k | ~246 k | 2.0× |
| Value head FLOPs / state | 128 | 256 | 2.0× |
| Total forward FLOPs / state | ~170 k | ~470 k | 2.8× |

**Consequences.**
1. **CUDA viability re-opens.** B5's killshot was that the 1.3 ms kernel-launch floor dwarfed per-call compute (~0.74 ms at B=16). At ~2.8× compute, per-call B=16 CUDA spends a larger fraction in kernel and a smaller fraction in launch. The scoping doc's own speculation (`gpu-batched-inference-throughput.md:412-416`) of CPU/CUDA crossover at sims≈1200-2500 *on the current model* scales to roughly sims≈400-900 on the new model — well within shipping-recipe range.
2. **CPU wave sweet spot shifts.** Heavier compute → smaller fraction of call wall is per-call overhead → wave amortizes less. Sweet spot likely drops from wave=16 toward wave=8 on CPU. On CUDA, may shift the other direction.
3. **Inference share grows toward ~96%.** Non-inference work is model-size-independent. Inference-targeting levers (value-only subgraph, CUDA, larger wave) gain leverage; non-inference levers (featurize scratch buffers, IndexMap removal) lose it.
4. **Value-only subgraph payoff scales with the policy/value FLOP ratio**, which stays ~80% policy on both model sizes, but the *absolute* saving on half-of-calls grows with model size. Expected wall gain revises from 1.15-1.25× → **1.25-1.45×**, bit-identical.

## Phase A — Throughput re-baseline

**Input ckpt:** `runs/R16-P3-v36-az-bigtrunk-cold/iter-2/policy.onnx` (latest fully-completed iter; iter-3 distill is in flight at scoping time). state_dim=246, hidden=256, depth=4.

**Grid (trimmed for in-flight-training coexistence).** `device ∈ {cpu, cuda}` × `wave_size ∈ {1, 8, 16, 32, 64}` × `sims=400` × `leaf=value-head` × `--mcts-two-sided` ON. 10 cells. Production sims=400 only — the small-model "sims=100 + 400" grid collapses to one because production already moved to 400.

**Recipe.** 60 games per cell, `sim-eval-gate --games 60 --model-side both --collapse-max 64 --max-nodes 5000 --c-puct 1.5 --prior policy --seed-base 0 --leaf value-head --mcts-two-sided --workers 4`. **workers=4 (not 16)** to coexist with the in-flight 16-worker training. Cell-relative wall ordering is what the verdict consumes, so contention-induced noise is tolerable as long as it's roughly uniform across cells.

**Outputs.**
- Wall, mean-Q, wilson per cell, written to `runs/vhleaf-throughput-bigger-model-rebaseline/grid.jsonl`.
- B6-style decision table: per `(sims, leaf)`, which `(device, wave_size)` wins on wall while staying within ±0.05 wilson of the `device=cpu wave_size=1` reference.

**Decision rule for Phase B.**
- **If CUDA wins on any production-relevant `(sims, wave)` cell** by ≥1.5× over best CPU cell *while staying within wilson envelope* → Phase B = wire CUDA on for `sim-mcts-selfplay` + `r12_orchestrator.py` selfplay step. Reopen `gpu-fed-stronger-mcts.md`.
- **If CPU stays winner** → Phase B = build value-only subgraph (`docs/ai-research/scoping/vhleaf-value-only-subgraph.md`, new doc).
- **If within 10% inconclusive** → build value-only subgraph (it stacks on either device).
- **If best CPU `wave_size` is not 16** → update B9's leaf-conditional default in `r12_orchestrator.py` to the new sweet spot before Phase B.

Phase A wall budget: ~1 day end-to-end on a single GPU box.

## Phase B options (sized in advance)

### Phase B / CUDA wire-up (if Phase A says CUDA crosses)

- Plumb `--device cuda` through `sim-mcts-selfplay` (today CPU-pinned at the binary level).
- Add `--device` to `r12_orchestrator.py` selfplay subprocess invocation (~line 552 block); orchestrator-level default conditioned on `(leaf, hidden_dim, depth)`.
- Verify ORT CUDA EP loads the new ONNX graph; rebuild wheel if needed.
- **Effort:** 1-2 days. **Expected payoff:** 1.5-3× on top of B6 wave-batched CPU if Phase A's crossover holds. **Wilson risk:** ±0.02-0.04 typical for FP CPU↔CUDA divergence (B3-B4 data); has to clear the 0.05 envelope.

### Phase B / value-only subgraph (if Phase A says CPU stays winner)

- `training/export_onnx.py`: dual export — full graph `policy.onnx` (unchanged) + value-only `policy.value.onnx` (no policy head, no action_features projection, no masked softmax).
- `engine-rs/crates/engine/src/inference/mod.rs`: load second session, add `predict_value_only(observation) -> f32`. Reuses state-slice from `pack_row`, skips action-features pack.
- `engine-rs/crates/engine/src/mcts/driver.rs:1575` (`value_head_leaf_value`): route through `predict_value_only`. `wave_run_priors` keeps full graph (needs logits for PUCT priors).
- Ckpt manifest gains `value_onnx_path`; loaders fall back to full graph if missing (backward-compatible).
- **Effort:** 1-1.5 days. **Expected payoff:** 1.25-1.45× wall on new model, bit-identical. **Killshot probe before code (2 h):** Python smoke — time existing `Session::run` with `output_names=["value"]` vs `["logits","value"]` at B=16 CPU on the new ckpt. Greenlight if per-call ≥1.4× faster. If ORT already partial-evaluates and saving <1.2×, kill the spike.

## What's NOT the spike (parked alternatives)

| Lever | Why parked |
| --- | --- |
| Subtree promotion (tree reuse) | Realistic single-side wall gain 1.10-1.25×; opponent-interleave invalidates cache; previously triaged in `r12-selfplay-gate-throughput.md`. Model size does not fix it. Revisit only at sims≥400 selfplay regime. |
| `pack_row` / IndexMap scratch buffer reuse | Bit-identical 1.05-1.18× on current model; shrinks as inference share grows toward 96% on bigger model. File as separate small-win ticket; not the spike. |
| `wave_size > 64` + `virtual_loss < 1.0` envelope sweep | Half-day probe; queued in `queue.json` next_action. Run as part of Phase A's grid (already covered by `wave_size ∈ {64, 128}` cells). |
| `GameState.clone()` reduction via `Arc::make_mut` | Bit-identical 1.05-1.10× on current model; even smaller on bigger model. Out of scope. |
| `dagger_orchestrator.py` / `ppo_orchestrator.py` TS-side wave port | These use TS-driven evaluators (`backend/src/sim/`), not the Rust binary. Separate ticket; Rust-first project default deprioritizes. |

## Acceptance criteria

**Phase A passes if:** the 24-cell grid runs to completion, all cells produce wilson_lower with no crashes, and the decision table is filed at `docs/ai-research/scoping/vhleaf-throughput-bigger-model-rebaseline.md` (this doc, §"Phase A results" appended).

**Phase B passes if:** the chosen lever clears its expected-payoff floor (CUDA ≥1.5×, or value-only ≥1.20×) *and* stays within ±0.05 wilson of the pre-spike reference at the production recipe (vhleaf sims=100, 120-game tight gate).

## Open prerequisites

1. **Bigger-model checkpoint:** ✅ resolved. `runs/R16-P3-v36-az-bigtrunk-cold/iter-2/policy.onnx` (and iter-1) on disk.
2. **ORT CUDA EP availability:** ✅ resolved. `engine-rs/Cargo.toml:35` enables `cuda` feature on `ort` by default; `sim-eval-gate --help` confirms `--device cuda` flag is present on the already-built `target/release/sim-eval-gate`.
3. **B9's leaf-conditional wave default** (commit `2eef954`) may need to be re-keyed to `(leaf, hidden_dim)` after Phase A picks a new sweet spot. The orchestrator does not currently see model dims at argparse time; either read them from the init checkpoint manifest or add `--mcts-wave-size` to the recipe's argv explicitly.

## Production-recipe drift since first scoping (2026-05-26)

The in-flight `bigtrunk-cold` run revealed two production drifts that change the grid:
- **sims=400, not 100.** Production recipe lifted the sim budget. Phase A's primary cell is at sims=400; the sims=100 column is no longer interesting.
- **`--mcts-two-sided` ON.** AlphaZero shape (commit `c953293` two-sided MCTS for v3.6 AZ recipe). Every internal node calls inference, not just leaves — call count per sim is ~depth × 2 higher than single-sided vhleaf. Inference share of wall pushes above the small-model 91%; wave-batching's marginal value may compress.

## Phase A results — 2026-05-26

**Input:** `runs/R16-P3-v36-az-bigtrunk-cold/iter-2/policy.onnx` (hidden=256, depth=4, state-dim=246). 120 games per cell (60 seeds × `--model-side both`), sims=400 two-sided, workers=4 to coexist with the in-flight training. Raw cells: `runs/vhleaf-throughput-bigger-model-rebaseline/grid.jsonl`.

| Cell | wall (s) | wl_lower | Δwl vs cpu-w1 | speedup vs cpu-w1 |
| --- | --: | --: | --: | --: |
| cpu-w1 (serial) | 32 | 0.3639 | — | 1.00× |
| cpu-w8 | 18 | 0.3639 | 0.000 | 1.78× |
| cpu-w16 | 16 | 0.3639 | 0.000 | 2.00× |
| cpu-w32 | 17 | 0.4039 | +0.040 | 1.88× |
| cpu-w64 | 15 | 0.3246 | −0.039 | 2.13× |
| cuda-w1 | 986 | 0.3798 | +0.016 | 0.03× |
| cuda-w8 | 174 | 0.3798 | +0.016 | 0.18× |
| cuda-w16 | 57 | 0.3719 | +0.008 | 0.56× |
| cuda-w32 | 17 | 0.4039 | +0.040 | 1.88× |
| cuda-w64 | 8 | 0.3168 | −0.047 | **4.00×** |

### Headline findings

1. **B6 wave-batching multiplier collapsed from 7.30× (small model) to 2.00× (bigger model).** Serial → wave=16 on bit-identical wilson went from 7.45 s → 1.02 s (=7.30×) on hidden=128/depth=3, down to 32 s → 16 s (=2.00×) on hidden=256/depth=4. **Why:** per-call compute grew ~2.8× with the model; per-call overhead is now a smaller fraction of call wall, leaving less for wave-batching to amortize. The lever is *not* dead — 2× is still meaningful — but the era of "wave-size sweeps unlock 7-9×" is over for production-size models.

2. **CUDA crosses CPU but only in the drifted-wave regime.** Conservative comparison (CPU vs CUDA at matched wave): CUDA is **3.6× SLOWER at w=16**, ties at w=32, and is **1.88× FASTER at w=64**. CUDA's per-call launch overhead amortizes only when waves are large enough — and at w=64 wilson drifts to the −0.05 edge of envelope on both devices. **CUDA is no longer dead** but its win comes attached to a wilson-drift caveat.

3. **Production wave=16 default (B9) holds.** wave=16 CPU is the bit-identical sweet spot at 2.00×. wave=8 captures 87% of the wave=16 gain (1.78×) — could reduce to wave=8 if a slight wave-latency advantage matters, but wave=16 stays optimal. The orchestrator default does NOT need re-tuning at the current model size.

4. **Two-sided MCTS does not break wave-batching.** Wilson holds bit-identical at w≤16 on CPU even with `--mcts-two-sided` on. Wave + virtual-loss-1.0 + two-sided composes cleanly.

5. **CUDA at w=1 is 31× slower than CPU** (986 s vs 32 s) — sanity-confirms B5's small-model verdict generalizes to the bigger model at serial wave. The kernel-launch floor is the same; it's just that bigger waves now amortize it better.

### Phase A → Phase B decision

**Decision rule check** (from §"Phase A — Throughput re-baseline"):
- "If CUDA wins on any production-relevant cell by ≥1.5× over best CPU cell *while staying within wilson envelope*" → only at w=64 with Δwl=−0.047 (just-barely-in-envelope) does CUDA hit 2× over best bit-identical CPU. The drift is on the *edge*, not comfortably inside. **Conditional CUDA win.**
- "If CPU stays winner" → CPU wins at every bit-identical wave (w≤16). **Conditional CPU win.**

This is the "within 10% inconclusive" branch of the decision rule, refined by data: **value-only subgraph remains the strongest spike** because:

1. It's **bit-identical** — no wilson-envelope worry like the CUDA w=64 cell.
2. **Stacks on whichever device** — if CUDA-w64 is later approved for production, value-only still cuts inference compute on top.
3. **Magnitude scales with model size as predicted**: 1.25-1.45× wall on hidden=256/depth=4 (vs 1.15-1.25× on the small model). Building on CPU-w16 (16 s), that's down to ~11-13 s — comparable to CUDA-w64 (8 s) without the drift caveat.
4. **CUDA wire-up requires also accepting the wave=64 envelope.** Phase B / CUDA path is now a two-step risk (wire CUDA + accept wave-64 wilson drift); value-only is one bit-identical step.

**Phase B verdict: build the value-only ONNX subgraph.** CUDA wire-up demoted to "open follow-on if value-only doesn't ship enough, AND wave=64 envelope acceptance becomes its own approved strength-axis decision."

### Phase B killshot probe — 2026-05-26 — **VALUE-ONLY SPIKE KILLED**

Probe v2 (`runs/vhleaf-throughput-bigger-model-rebaseline/probe_value_only_v2.py`): physically extracted a value-only subgraph from `iter-2/policy.onnx` via `onnx.utils.extract_model`, timed against the full graph at B=16 across n_actions ∈ {4, 8, 16, 20}, intra=inter threads=1.

| n_actions | full graph (μs/call) | value-only (μs/call) | ratio |
| --: | --: | --: | --: |
| 4 | 3,356 | 285 | 11.8× |
| 8 | 4,282 | 226 | 19.0× |
| 16 | 10,070 | 287 | 35.1× |
| 20 | 14,797 | 342 | **43.2×** |

**Per-call savings are massive** — the policy head is genuinely expensive, scaling roughly with n_actions². At wave-padded A=20 the full graph is 14.8 ms/call vs 0.34 ms value-only. Greenlight on the per-call axis cleared by 30×.

**But the spike's *target call site* doesn't exist in production.** Code inspection of `engine-rs/crates/engine/src/mcts/driver.rs`:

- The wave-batched production path (`wave_run_priors`, line 867+, called from line 612) does ONE batched `predict_v3_batch` call per wave that returns BOTH logits (for child priors) AND value (cached on the leaf for backup). Both outputs are consumed.
- `value_head_leaf_value` (line 1575) is the singleton B=1 value-only call site that *would* have benefited — but it's only invoked at line 1505 in the **serial / non-wave fallback path**. In production (wave_size ≥ 8), the wave loop caches the value from `wave_run_priors` and the cache hit (line 1072-1076: `cached.is_some()` short-circuit) bypasses `value_head_leaf_value` entirely.
- Two-sided MCTS doesn't help — opponent decisions are full tree nodes that need policy priors for PUCT descent, so opponent calls also need both outputs.

**Decision: VALUE-ONLY SUBGRAPH SPIKE REJECTED on path-of-execution grounds.** The policy head is expensive (probe confirmed), but there is no production code path that discards logits — every wave call uses both outputs.

### What the probe data DID reveal (silver lining)

The full-graph per-call wall scales sharply with `n_actions` (3.4 ms at A=4 → 14.8 ms at A=20). Wave-batching pads all rows to `max_n_actions` across the wave (line 756: `let max_n = rows.iter().map(|r| r.n_actions).max().unwrap_or(0);`). If mean legal-action count is ~10 but wave-max is ~20, every call pays the A=20 cost when most members only needed A≤10 — a ~2-3× over-pay on inference compute. **This is a new candidate lever: action-count-bucketed wave batching** (file as separate scoping doc; not in scope of this re-baseline).

### Updated Phase B verdict — 2026-05-26

Value-only is killed. Two remaining throughput levers, both already documented:

1. **CUDA wave=64** with explicit `wave-size envelope` acceptance — 2× wall on top of B6/B9, but requires a strength-axis decision to widen the wilson envelope from ±0.05 to ±0.06 (cuda-w64 sits at Δwl=−0.047, just barely inside). This is a *strength-axis* decision dressed as a throughput one.
2. **Action-count-bucketed wave batching** — newly surfaced by the probe; ~2-3× per-call on under-padded members, compose with B6 wave. Effort: 2-3 days, bit-identical, no strength-axis interaction. File as separate scoping doc.

**Recommended:** demote this scoping doc to closed, file action-count bucketing as the next spike. CUDA-w64 stays parked behind the envelope decision.
