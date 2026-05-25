# GPU-Fed Stronger MCTS — Scoping Seed

- **Date:** 2026-05-18
- **Status:** SCOPING — seed only, no implementation. Compounds with `rust-port-orchestrator-wiring` (Rust 140× MCTS shrinks the cost of high-sim regimes); revisit when a load-bearing question requires the high-sim datapoint. Related queue item: `high-sim-mcts-regime-probe`.
- **One-liner:** Use GPU inference to buy strictly stronger search-wrapped play
  than rollout-leaf MCTS @ W6 iter-2, not merely the same strength at lower
  latency.
- **Non-goal:** raw-policy promotion. All success metrics are MCTS-wrapped.

## Question

Can batched neural inference, learned/hybrid leaves, or GPU-amortized root
search improve the search-wrapped strength/latency frontier beyond the current
production baseline?

Baseline: rollout-leaf MCTS @ W6 iter-2, Wilson lower 0.6479 vs rule bot
(n=120 side-balanced), with player-side lower 0.5577 and opponent-side lower
0.6638.

## Working Hypothesis

Naive batching will not strengthen the current rollout-leaf config because its
hot path is TS simulator rollouts; the model mostly supplies PUCT priors. GPU
inference becomes a strength lever only if the active config is value/head
evaluable or hybrid-leaf evaluable, so extra throughput can buy more evaluated
nodes, stronger leaf estimates, ensembles, or root search variants.

## Required Pre-Work

1. **R16-P0 fixed and proven.** `test:r16-mcts-embedding` passes and current v3
   MCTS rows feed `card_ids_by_zone` / `action_card_idx` through
   mcts-distill/value paths.
2. **R110 W6 reproduction decision.** Either promote a 110-d/v3 search-wrapped
   model or explicitly keep the 96-d production model as the baseline.
3. **Benchmark identity fixed.** Every candidate reports the same seeds/sides,
   fallback/no-op counts, per-side Wilson, and timing buckets.

## Candidate Experiments

1. **Bottleneck probe.** Matrix over rollout-leaf, value-head-leaf, and one
   hybrid learned-leaf candidate; vary workers, simulations, and batching mode.
   Record selection/clone/hash, legal-action enumeration, rollout leaf,
   feature encoding, HTTP, ORT, queue wait, batch size, CPU, and GPU utilization.
2. **Value/hybrid leaf data.** Generate rule-bot-covered contested states and
   relabel with rollout-leaf MCTS. Train value/action-value heads on those
   states, not self-play-only states.
3. **Batched evaluator interface.** Split MCTS tree policy/backup from
   "evaluate N states" so serial, batched, and microbatched evaluators are
   comparable without changing PUCT semantics. Dispatcher LANDED 2026-05-25
   via [`gpu-batched-inference-throughput.md`](gpu-batched-inference-throughput.md)
   B2 (commit 2ded9b0): `InferenceSession::load_on_with_batching(path,
   device, max_batch, max_wait_us)` + `--batch-size`/`--batch-wait-us`
   flags on `sim-eval-gate`. The throughput axis closed (CPU still beats
   batched CUDA on every shipping recipe) but the dispatcher correctness
   gate passed; strength experiments here can reuse it directly. Two
   strength-relevant findings from the throughput close: (a) inter-game
   batching only sustains fill 23-47/64 in MCTS workloads (intra-tree
   waves may be needed for >60/64 fill); (b) CUDA reduction-order drift
   exceeds the 0.02 wilson envelope at sims=1000 (|Δ|=0.035) — assume
   high-sim CUDA is a different strength evaluation than high-sim CPU and
   gate accordingly.
4. **Strength candidates.** Test value-head with larger sim budgets, hybrid
   neural+selective-rollout leaves, root ensembles, and Gumbel/sequential-
   halving-style root search.

## Gates

- **Leaf readiness gate:** crossover Pearson >=0.70 and MSE <=1.10x rollout-
  noise floor on held-out deployment-like states; value-head-leaf MCTS Wilson
  lower >=0.50 before it can be a production-strength candidate.
- **Promotion gate:** candidate beats production by >=+3pp Wilson lower under
  the rule-bot protocol, or wins head-to-head vs production with Wilson lower
  >0.52. Require zero fallbacks/no-ops and no per-side regression >5pp unless
  explicitly accepted.
- **Non-promotion:** latency-only wins, raw-policy wins, or value-head gains
  that do not improve the search-wrapped player.

## Guardrails

- Do not replace rollout-leaf production until a wrapped candidate beats it.
- Do not count raw-policy gates as success.
- Do not implement broad batching infrastructure before the bottleneck probe
  shows a leaf-eval-bound or hybrid-leaf-bound target.
- Batched PUCT must re-clear determinism-sensitive evals; virtual loss,
  wave-ordering, and tie-breaking can change search behavior.
