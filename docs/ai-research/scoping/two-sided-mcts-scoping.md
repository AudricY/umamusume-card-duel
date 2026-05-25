# Two-Sided (AlphaZero-style) MCTS — Scoping

- **Date:** 2026-05-25
- **Status:** **IMPLEMENTED-AWAITING-A1** — flag landed in TS+Rust+sim-cli+NAPI; default OFF preserves byte-identity. A1 probe awaits user-approved fire on the v3.6 cap128 cont-iter2 ckpt.
- **Parent:** `docs/ai-research-backlog.md` MCTS-strength axis. Sibling to `gpu-fed-stronger-mcts.md` (compute axis) and `cross-iter-opponent-pool-selfplay.md` (training-signal axis).

---

## TL;DR

- Existing MCTS is single-side: opponent turns get collapsed via the rule-bot between expansions, every node lives in `modelSide` frame, backup adds the leaf value unmodified.
- Add `MctsConfig.twoSided` (default `false`). When `true`, opponent decision points become real tree nodes, `predict_policy_and_value` is queried for whoever is to move, and backup uses AZ sign-flips per ply.
- Default-off path is byte-identical to today (no touch to `buildModelDecisionNode` / `stepFromModelDecision` / `collapseUntilModelOrTerminal`). All existing goldens (`golden_mcts_full_game`, `golden_mcts_decisions`, `mcts_result_invariants`) pass without changes.
- Root invariant preserved: root is constructed as a `modelSide` decision node so `rootValue` / `rootMeanQ` / `visitDistribution` stay in modelSide frame. `value_target_dataset.py` and `selfplay_dataset.py` consumers untouched.

## Hypothesis

The rule-bot collapse caps search strength at the opponent model's quality. Replacing collapse with policy-net opponent simulation should lift winrate at a fixed sim budget when the policy is stronger than the rule-bot — which §4l of `progress/r110.md` confirms for v3.6 at sims=100.

## Design

### Flag

`MctsConfig.twoSided: bool` (Rust: `two_sided`, serde `twoSided`). Default `false`. Mirrors the W6-fix flag (`UMA_MCTS_HASH_CARRY`) pattern.

### Two-sided semantics (flag ON)

1. `MctsNode` carries `sideToMove: SideId` (always `modelSide` in flag-off mode by construction).
2. `buildDecisionNodeTwoSided(state, sideToMove)` does NOT precollapse opponent turns. Queries `/predict` (in-process inference for Rust) with `sideToMove`; priors are for that side's choices, cached leaf value is in that side's frame.
3. `stepFromDecisionTwoSided(state, sideToMove, action)` applies one ply via `advanceModeledTurnStep(state, sideToMove, ...)` — which is side-agnostic in both TS and Rust today. No collapse loop. The child node's `sideToMove` is whatever `nextState.currentSide` ends up as.
4. Backup with sign-flip: `leafFrame = leaf.sideToMove` (or for terminal/no-store paths, the active side at the leaf state). For each `step.node` in the path: `sign = (step.node.sideToMove === leafFrame) ? +1 : -1; step.node.wsum[i] += sign * leafValueScalar`. This is the AZ formulation.
5. **Root invariant.** The root is still constructed with `sideToMove = modelSide` (caller passes `modelSide`). `diagnostics.rootValue = root.cachedLeafValue` continues to be the modelSide-frame value. Visit distribution, mean-Q, priors at root are all in modelSide frame. Downstream JSONL schema is unchanged.
6. **Terminal value at leaf.** Compute `terminalInLeafFrame = mctsTerminalValue(state, leafFrame)`, which equals `mctsTerminalValue(state, modelSide) * (leafFrame === modelSide ? +1 : -1)`. Sign-flip backup turns this back into the right per-node frame.
7. **PUCT** is mathematically identical — `q + u` always means "good for whoever is to move at this node" in both modes (single-sided: that's always modelSide; two-sided: it varies).

### What does NOT change

- Single-sided code path (flag off): zero touch beyond threading `sideToMove` through node construction with `modelSide` as the constant value.
- On-disk JSONL schema (`rootValue`, `rootMeanQ`, `visitDistribution` stay in modelSide frame).
- Python training code (`value_target_dataset.py`, `selfplay_dataset.py`).
- Golden tests.
- Behavior of `MctsConfig.collapseMaxSteps` (still consulted when flag is off; ignored when on).

### Parallel TS+Rust requirement

Rust-first-class default applies. Both engines change together. Golden parity tests already pin the single-sided path; two-sided default-off must keep them byte-identical, and two-sided default-on is a new behavior that does not need a parity goldenfile yet (the probe is the verdict).

## Probe recipe

- **A0 reference:** `runs/R16-P1-v36-cap128-cont-iter2-tight-gate/gate.manifest.json` — v3.6 ckpt at sims=100, single-sided MCTS. Reuse the existing wl_lower as A0.
- **A1 (this slice):** same ckpt, same recipe (sims=100, `--leaf rollout`, `--prior policy`, `--collapse-max 64`, `--max-nodes 5000`), with `--mcts-two-sided` added.
- **n schedule:** smoke at n=120 (matches the 13-iter cadence); arbiter at n=1k if smoke passes a Wilson noise check (Δ ≈ ±0.045 at n=120 means we need either decisive lift or arbiter follow-up). Full tight-gate (n=10k) only if arbiter clears.
- **Falsification:** `wl_lower(A1) - wl_lower(A0)` must be > 0 with overlap-free Wilson CIs. Null or regression closes the slice; the rule-bot collapse was either neutral or beneficial at v3.6 strength.

### Recipe parameters

```
--onnx-path runs/R16-P1-v36-cap128-A1-continue/loop/iter-2/policy.onnx
--games 60 --model-side both        # n=120 fast smoke
--sims 100 --leaf rollout --prior policy
--mcts-two-sided                     # NEW flag (this slice)
--collapse-max 64 --max-nodes 5000
```

## Decision rules

1. **Lift:** wl_lower(A1) > wl_lower(A0) + 0.013 at n=1k arbiter → flag becomes the new selfplay default for sims≥100 probes; queue a v3.7-axis follow-up to retrain with self-play data generated under two-sided search.
2. **Soft pass:** within Wilson overlap and lift point estimate positive → run n=10k tight gate before adoption; this is a quiet "rule-bot was about as good as policy-at-this-strength" result and probably means the next gain lever is sim budget, not opponent realism.
3. **Null / regression:** rule-bot collapse was either neutral or stronger than the v3.6 policy at sims=100. Pause the two-sided axis; revisit when (a) we have a stronger policy ckpt where the rule-bot is clearly weaker (R7+ candidate retrains), or (b) we lift sim budget where deeper opponent realism would matter more.

## Out of scope (explicit)

- No Python orchestrator flag exposure (`r12_orchestrator.py`). Lower-priority follow-up; flag is reachable via the sim-cli binary directly for the probe.
- No behavior changes beyond the structural switch (no value-clipping, no exploration tweaks, no PUCT changes).
- No on-disk schema bump. Selfplay JSONL stays modelSide-frame.
- No two-sided `mcts_selfplay` / `relabel_leaf_values` plumbing (CLI flag stub only — probe doesn't need them).
- No two-sided value-target relabeling. If A1 lifts we'd separately scope whether two-sided rollout-leaf values should replace the single-sided ones in distillation corpora; that decision waits on A1's verdict.

## File pointers

### Code (changed in this slice)
- `backend/src/sim/mcts.ts` — TS `MctsConfig.twoSided`, `MctsNode.sideToMove`, two-sided build/step helpers, sign-flip backup.
- `engine-rs/crates/engine/src/mcts/{config.rs,node.rs,driver.rs}` — Rust mirror.
- `engine-rs/crates/sim-cli/src/bin/eval_gate.rs` — `--mcts-two-sided` CLI flag.
- `engine-rs/crates/napi-bridge/src/lib.rs` — `mctsTwoSided` JSON config field.
- `backend/src/sim/{evaluateModelVsHeuristic.ts,evalGate.ts,mctsSelfPlay.ts}` — flag plumbing.

### Code (existing, untouched but consulted)
- `backend/src/sim/mcts.ts:374` `buildModelDecisionNode` (precollapse fast path stays for flag-off).
- `engine-rs/crates/engine/src/mcts/driver.rs:317` (Rust mirror).
- `training/uma_ai/value_target_dataset.py` (reads `rootValue`; unchanged).
- `training/uma_ai/selfplay_dataset.py` (reads `visitDistribution`, `rootMeanQ`; unchanged).

### Tests (must pass unchanged after the flag lands)
- `engine-rs/crates/engine/tests/golden_mcts_full_game.rs`
- `engine-rs/crates/engine/tests/golden_mcts_decisions.rs`
- `engine-rs/crates/engine/tests/mcts_result_invariants.rs`

## Status line

**`IMPLEMENTED-AWAITING-A1 2026-05-25`** — TS+Rust+CLI+NAPI plumbing landed; default-off byte-identity verified via `cargo test -p engine --tests`. A1 probe is user-gated.
