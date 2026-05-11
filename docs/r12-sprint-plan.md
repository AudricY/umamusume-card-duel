# R12 Mini-AlphaZero Sprint Plan

Created 2026-05-11 after Tier-1 + stretch research established the SL+RL imitation cap at Wilson lower 0.31. R12 is the chosen structural intervention — see `docs/ai-research-backlog.md` for the full path.

## North star

**Gate Wilson lower ≥ 0.40 vs rule-bot at n=200 side-balanced.** Achieved by trained policy + MCTS at inference, or by a learned policy distilled from MCTS visit counts (deployed at inference without search). Either qualifies.

Stretch: ≥ 0.50.

## Architecture

**MCTS lives in TypeScript** under `backend/src/sim/`. Rationale:

- The TS simulator already exposes the primitives we need: `cloneGame` (`frontend/src/game/engine/core/stateClone.ts`), `advanceModeledTurnStep` and `enumerateLegalAiActions` (both used by `evaluateModelVsHeuristic.ts`'s existing search/planner paths), `createSeededRng` with `.fork()` for deterministic playouts, `stateFingerprint` for hashing.
- A Python-side MCTS would HTTP-ship game state per node expansion. With 100 sims × ~15 real decisions/game × ~5 simulator steps per playout ≈ ~7500 sim-steps per game, that's a lot of round trips. TS-side keeps it in-process.
- Leaf evaluation uses the existing `serve_onnx` HTTP `/predict` — one request per leaf, identical to how `chooseValueAction` already queries the model in `evaluateModelVsHeuristic.ts`. Latency budget: 100 leaf-evals × 10–15 ms × 15 real-decisions × 100 games = 22–34 min wall-clock per 100-game gate, well within budget given the 5–10 min existing PPO sweeps run.

**Candidate ranker is bypassed** for MCTS. The auditor finding (failure rate 20% → 100% as legal-action count goes 2 → 15+) is exactly the optionality problem — MCTS must see all legal actions. `candidateRanker.ts` filtering is replaced with `enumerateLegalAiActions(state, { topK: Infinity })` at the MCTS root and every expansion. The ranker's `selectedOriginalRank` reporting is kept for trace compatibility but plays no role in selection.

**Value-head semantic contract** (addressing the risk agent's concern): MCTS uses the model's `value` output (tanh ∈ [-1, 1], side-relative) as the SOLE leaf evaluator. We do NOT mix it with `rewardForRollout`'s point-margin formula. The contract is: value is "expected outcome from this side's POV, +1 win, -1 loss" — already what `_value_target` in `training/uma_ai/dataset.py:115` produces. On terminal states MCTS uses `terminalValue(state, modelSide) = +1 win, -1 loss, 0 unfinished`. This is the same contract under training and inference, so backups are self-consistent.

**Determinism / CRN**: every MCTS root gets a deterministic seed `${gameSeed}:${modelSide}:${turnNumber}:mcts`. Each simulation index forks a sub-RNG. Replays are bit-exact, same as the rollout/search/planner modes.

## Day 1 — 8-hour spike (go/no-go gate)

### Subject
Prove **uniform-prior MCTS + R4 value-head leaf eval** clears `gate Wilson lower ≥ 0.40 vs rule-bot at n=100`.

### Files

- **NEW** `backend/src/sim/mcts.ts` (~250 LOC) — PUCT search.
  - `MctsNode { state, priors[], visits[], wsum[], children: (MctsNode | null)[], expanded: boolean }`
  - `runMcts(rootState, modelSide, args, valueFn, seed) → bestIndex`
  - Selection: `score(a) = Q(a) + c_puct * P(a) * sqrt(sumN) / (1 + N(a))`. Day-1 prior is uniform `1/|A|`.
  - Backup: side-relative — leaf value is in `modelSide` frame; flip sign when traversing a node where `currentSide ≠ modelSide`.
  - Leaf eval: POST `serve_onnx` `/predict` with leaf observation; read `payload.value[0]`. On terminal: `terminalValue(state, modelSide)`.
  - Default: 100 sims, `c_puct = 1.5`.
  - Memory cap: 5000 nodes per move; LRU evict on overflow.

- **MODIFY** `backend/src/sim/evaluateModelVsHeuristic.ts` (~30 LOC):
  - Add `"mcts"` to selection union + CLI parser + dispatch table.
  - New args: `--mcts-simulations`, `--mcts-c-puct`, `--mcts-leaf` (default `"value-head"`).

- **MODIFY** `backend/src/sim/evalGate.ts` (~5 LOC): pass-through.

- **NEW** `training/r12_spike.py` (~60 LOC) — wrapper that emits `stage="mcts-spike"` events and shells `sim:eval-gate --selection mcts`.

- **NEW** `training/r12_spike_smoke.py` (~50 LOC) — tiny end-to-end: `--games 4 --mcts-simulations 8`. Asserts manifest parses, every game has ≥1 model decision, no fallbacks.

### Go/no-go criterion

```bash
npm --workspace backend run sim:eval-gate -- \
  --selection mcts \
  --mcts-simulations 100 \
  --games 100 \
  --model-side both \
  --model-url http://127.0.0.1:8765 \  # serve R4 checkpoint
  --min-ci-lower 0.40
```

**GO** if `summary.wilson95.lower ≥ 0.40` AND `summary.modelWinRate ≥ 0.43` AND `summary.heuristicFallbacks == 0`.

**NO-GO**: write a postmortem in `docs/ai-research-backlog.md`. Probable causes if no-go: (a) value head isn't a good leaf evaluator → swap to rollout-CRN at leaves and re-spike; (b) 100 sims is too few → spike again with 200; (c) MCTS bug in backup math → audit.

### Day-1 hour budget
- 4 h: implement `mcts.ts` + wire selection
- 1.5 h: spike runner + smoke + dashboard tag
- 1.5 h: run the 100-game gate (likely ~25–35 min on machine)
- 1 h: buffer / debug

## Days 2–7 — full build (gated on spike GO)

### Phase A — PUCT with policy prior (4 h, depends: spike)

Subject: swap uniform prior for the served policy's softmax; cache priors by `stateFingerprint`.

- **MODIFY** `mcts.ts` (~40 LOC): batch-request `/predict` for each newly-expanded node; pull `actionProbs[0]` as priors.
- Dirichlet noise at root (α=0.3, ε=0.25) only during self-play (phase B); off during gate.
- Retune `c_puct = 1.0` once prior is informative.
- Events: `stage="mcts"`, `event_type="prior_diagnostic"` with `mean_prior_entropy`, `root_q_argmax_match_with_policy_argmax`.

### Phase B — MCTS self-play data generation (8 h, depends: A)

Subject: generate `(state_features, action_features, mask, π_target = visit_count/sum_N, z_target = outcome)` rows.

- **NEW** `backend/src/sim/mctsSelfPlay.ts` (~200 LOC): same skeleton as `evaluateModelVsHeuristic` but both sides use MCTS; writes one row per real decision to `selfplay.jsonl`.
- **NEW** `training/uma_ai/selfplay_dataset.py` (~120 LOC): mirrors `JsonlPolicyDataset` shape, but `targets` is the visit-count distribution (soft) and there's a per-row `value_target` (outcome).
- Temp=1.0 for first 6 moves (visit-count proportional sampling), greedy thereafter.
- Default: 200 self-play games per iteration; 100 MCTS sims per decision.
- Events: `stage="selfplay"`, types `started`, `completed` with `games`, `rows`, `mean_game_length`, `mean_visit_entropy`.

### Phase C — Policy/value distillation training (4 h, depends: B)

Subject: distill MCTS visit-count policy + outcome value into the network.

- **MODIFY** `training/train_bc.py` (~80 LOC): add `--target-mode mcts-distill`. Loss branches: `policy_loss = -Σ π_target · log_softmax(logits)` (soft cross-entropy), `value_loss = MSE(value, z_target)`.
- Init from R4 checkpoint (known-strong policy/value baseline).
- HPs: lr=3e-4, 25 epochs, value_weight=1.0, policy_weight=1.0.
- **NEW** `training/r12_distill_smoke.py` (~50 LOC) — trains 2 epochs on a 16-row synthetic dataset; asserts loss decreases.

### Phase D — Iteration loop (8 h, depends: A+B+C)

Subject: orchestrate `selfplay → distill → gate → promote`.

- **NEW** `training/r12_orchestrator.py` (~500 LOC) — adapt `dagger_orchestrator.py` skeleton (`OrchestratorState`, `serve_onnx_context`, `decide_promotion`, `wilson_lower_bound` all reusable as imports). Stages per iteration:
  1. Stand up `serve_onnx` for promoted checkpoint.
  2. Shell `npm run sim:mcts-selfplay -- --games 200 --mcts-simulations 100`.
  3. Shell `train_bc.py --target-mode mcts-distill --init-from <promoted>`.
  4. Shell `sim:eval-gate --selection mcts --mcts-simulations 100 --games 200`.
  5. Promote/halt with the existing rules (same Wilson-lower floor + per-matchup floor + halt-after-2).
- Target: 4–6 iterations.
- Observability: new stages `selfplay`, `distill`, `mcts-gate` — register in `observability_app.py` STAGES.

### Phase E — Smokes + observability + final gate (4 h, depends: D)

- **NEW** `training/r12_orchestrator_smoke.py` — 1 iter × 4 self-play games × 8 sims; asserts manifest, checkpoint, gate manifest, all expected event types appear.
- TB scalars (auto-wired via `train_bc.py`'s SummaryWriter): `mcts/visit_entropy`, `mcts/value_target_mean`, `mcts/policy_distill_kl`.
- Flask dashboard: extend `observability_app.py` STAGES list.
- **Final gate**: `sim:eval-gate --selection mcts --mcts-simulations 200 --games 400 --min-ci-lower 0.40`.

## Risk register

| Risk | Severity | Mitigation |
| --- | --- | --- |
| HTTP serve_onnx latency dominates MCTS | LOW (post-math) | 100 leaves × 10 ms × 15 dec × 100 games = ~25 min — within budget. If slower, swap to batched `/predict` request (multiple leaves per call). |
| Value-head semantic mismatch with rollout-CRN reward | MEDIUM | Plan uses ONLY model.value as leaf evaluator; never mixes with `rewardForRollout`. Contract: tanh win-prob, side-relative. Same under training and inference. |
| Random card draws / coin flips create chance nodes | MEDIUM | MCTS collapses chance via heuristic rollout to the next model decision (same as existing search/planner paths). No explicit chance-node expansion. Documented in `mcts.ts`. |
| Candidate ranker accidentally filters MCTS | LOW | Bypass via `enumerateLegalAiActions(state, { topK: Infinity })`. Test in spike smoke. |
| Hidden information: opponent hand IDs unknown | LOW | Model trains on `PublicObservation` (handCount only, no card IDs). MCTS uses the same observation. Determinism is over PUBLIC state. |
| Memory: search tree explosion | LOW | 5000-node cap per move; LRU evict. With ~10 legal actions × 100 sims = ~1000 nodes typical. |
| Determinism / CRN regressions | LOW | Existing `createSeededRng` pattern reused. Add seed-replay regression test in spike smoke. |
| Phase D iteration loop fragility | MEDIUM | Mirror DAgger orchestrator pattern (proven). Same promotion gates, same halt-after-2. Smoke validates the loop end-to-end. |
| Self-play game length unbounded | LOW | `max-steps 500` cap, same as DAgger. |
| Policy distillation diverges (high KL between distilled and visit-count target) | MEDIUM | Init from R4 (known good); lr=3e-4 conservative; monitor `mcts/policy_distill_kl` via TB; if it spikes, lower lr or warm-start from a non-MCTS checkpoint. |

## Observability integration

- `events.jsonl`: new stages `mcts`, `selfplay`, `distill`, `mcts-gate`. Each emits standard `started/completed` plus stage-specific data (sim count, visit entropy, KL, etc.).
- TensorBoard: per-epoch distill scalars under `mcts-distill/*`. Per-iteration self-play scalars under `selfplay/*`.
- Flask dashboard: STAGES list extension at `training/observability_app.py:21`. Pipeline state strip renders the new stages alongside DAgger/PPO stages.

## File summary

**NEW (target LOC):**
- `backend/src/sim/mcts.ts` (~250)
- `backend/src/sim/mctsSelfPlay.ts` (~200)
- `training/r12_orchestrator.py` (~500)
- `training/r12_spike.py` (~60)
- `training/r12_spike_smoke.py` (~50)
- `training/r12_orchestrator_smoke.py` (~120)
- `training/r12_distill_smoke.py` (~50)
- `training/uma_ai/selfplay_dataset.py` (~120)

**MODIFY (target LOC delta):**
- `backend/src/sim/evaluateModelVsHeuristic.ts` (+30)
- `backend/src/sim/evalGate.ts` (+5)
- `training/train_bc.py` (+80 for `--target-mode mcts-distill`)
- `training/observability_app.py` (+15 for new stage names)
- `package.json` (+2 for new smokes)

**Total: ~1450 new LOC, ~130 modified LOC.**

## Wall-clock budget

- Day 1: 8 h coding + run
- Days 2–7: ~28 h coding + ~10 h wall-clock for runs

If Day 1 NO-GO, total cost is 8 h and the docs writeup. Asymmetric upside.