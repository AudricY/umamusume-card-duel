# vhleaf Throughput Re-baseline on hidden=256/depth=4 — Scoping

- **Date:** 2026-05-26
- **Status:** **SCOPING-DEFERRED** — fires when a hidden=256/depth=4 checkpoint exists. No code lines yet.
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

## Phase A — Throughput re-baseline (blocking)

When a hidden=256/depth=4 checkpoint exists:

**Grid.** `device ∈ {cpu, cuda}` × `wave_size ∈ {1, 8, 16, 32, 64, 128}` × `sims ∈ {100, 400}` × `leaf = value-head`. 24 cells.

**Recipe.** 60 games per cell, `sim-eval-gate --games 60 --model-side both --collapse-max 64 --max-nodes 5000 --c-puct 1.5 --prior policy --seed-base 0 --workers 16`, identical to B6's `b6_wave_probe.sh` modulo device/wave/sims. Reuse `runs/gpu-batched-inference-b3-b4/b6_wave_probe.sh` as the template.

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

1. **A hidden=256/depth=4 checkpoint must exist.** `training/make_v36_cold_init.py` (commit `b133aaa`) can produce a random-init checkpoint; for the throughput grid, untrained weights are acceptable because wall is FLOP-dominated, not what-the-FLOPs-compute. A trained checkpoint would be needed for Phase B wilson validation.
2. **ORT CUDA EP must be available in the engine-rs build** — already gated behind a feature flag per B3-B4 history. Confirm the build still produces a CUDA-capable binary before Phase A.
3. **B9's leaf-conditional wave default** (commit `2eef954`) may need to be re-keyed to `(leaf, hidden_dim)` after Phase A picks a new sweet spot. The orchestrator does not currently see model dims at argparse time; either read them from the init checkpoint manifest or add `--mcts-wave-size` to the recipe's argv explicitly.

## Recording space (fill on fire)

### Phase A results (TBD)

_Grid table, decision verdict, link to `grid.jsonl`._

### Phase B verdict (TBD)

_Which lever fired, wall delta, wilson delta, commit hash._
