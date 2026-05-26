# v3.6 AlphaZero-style Recipe — Scoping

- **Date:** 2026-05-26
- **Status:** **IMPLEMENTED-FIRING-P0-P3** — `--mcts-two-sided` plumbed through `r12_orchestrator.py` (run_selfplay + run_gate + run_started event). Phase 1 probes P0-P3 firing on the v3.6 cap128 cont-iter2 ckpt; Phase 2 training loop user-pre-approved.
- **Parent:** `v36-priors-and-arithmetic-scoping.md` (LANDED-SOFT-SHIP-2026-05-25, wl_lower=0.5880) + `two-sided-mcts-scoping.md` (IMPLEMENTED-AWAITING-A1) + `progress/r110.md §4a` (value-head-leaf at n=10k flagged as untested follow-up).
- **Scope:** Convert the v3.6-extended training recipe into the closest practical match to AlphaZero canon, holding state-feature schema (state_dim=246), trunk capacity (hidden=128, depth=2), sim budget (100), and replay-window-3 fixed. Tests whether AZ search shape (two-sided MCTS + value-head leaf + no KL anchor) lifts strength vs the rule-bot-collapse + rollout-leaf + KL-pinned v3.6-extended baseline.

---

## TL;DR

- **Three deltas vs v3.6-extended:** (1) `--mcts-leaf value-head` (no rollouts), (2) `--mcts-two-sided` (opp nodes real, sign-flip backup), (3) `--kl-anchor-weight 0.0` (drop the W6 KL anchor entirely).
- **Everything else inherits v3.6-extended:** state_dim=246 (5-input ONNX), hidden=128 depth=2, sims=100, c-puct=1.5, temperature schedule (T=1 for first 6 plies, argmax after), Dirichlet noise ε=0.25, single-opponent self-play vs current promoted, W6 replay window=3 / old-fraction=0.40, 240 self-play games × 8 iters, 25 epochs, lr=3e-4, batch=64.
- **AZ canon already present:** value/policy joint head, π=visits target, z=outcome target, Dirichlet root noise, temperature schedule, self-play same-net both sides. The three deltas above are the only places v3.6-extended diverges from AZ canon.
- **Init:** v3.6 cap128 cont-iter2 (the SOFT-SHIP ckpt) — not from-scratch. Cleanest A/B against the v3.6 anchor (wl_lower=0.5880); from-scratch ("Zero") is a separate, costlier bet for later.
- **Anchor:** `runs/R16-P1-v36-cap128-cont-iter2-tight-gate/gate.manifest.json` — wl_lower=0.5880, mean implied 0.5976, upper=0.6072, n=10k. Per-side: P=0.5500, O=0.6179 (the capacity-bound asymmetry from `progress/r110.md §4l`).
- **Falsification:** wl_lower(Phase-2-best) ≥ 0.5778 (soft-ship band); wl_lower ≥ 0.6040 (lift band, clears v3.0 upper bound 0.6004); wl_lower < 0.5778 (hard fail → run isolation probes).

---

## 1. Bridge from v3.6 to AZ

| AZ canon ingredient | v3.6-extended state | Action |
|---|---|---|
| Policy + value joint head | Present (value-weight=1.0, policy-weight=1.0) | inherit |
| Self-play same net both sides | Default (single opponent = current promoted) | inherit |
| MCTS PUCT with learned priors | `--mcts-prior policy` | inherit |
| Dirichlet noise at root, ε=0.25 | Present in Rust driver | inherit |
| Temperature schedule (T=1 early, argmax late) | T=1 first 6 plies, argmax after | inherit (AZ used 30 plies; our games are shorter) |
| Value target z (game outcome) | Present | inherit |
| Policy target π (visit distribution) | Present | inherit |
| Value-net leaf (no rollouts) | Code present; v3.6 recipe uses `rollout` | **CHANGE to `--mcts-leaf value-head`** |
| Two-sided MCTS (opp nodes real, sign-flip backup) | Code landed default-off; orchestrator flag landed 2026-05-26 | **CHANGE to `--mcts-two-sided`** |
| No KL anchor | v3.6 W6-fix pins anchor to iter-0 | **CHANGE: drop with `--kl-anchor-weight 0.0`** |
| FIFO replay over recent games | W6 fix: 3-iter window, 40% old | Inherit (approximates AZ FIFO; widening is separate axis) |
| 800 sims/move | We run 100 | Inherit (sim budget axis lives in `gpu-fed-stronger-mcts-scoping.md`) |
| Network-replace every N steps, no gate | v3.6 uses Wilson-band gate w/ 2-failure halt | Inherit (safety net; loosen halt to 3 failures for AZ-noise headroom) |

The three CHANGE rows are the entire delta. Everything else is configured-AZ-compatible already.

## 2. Pre-registered hypotheses

### H1 — AZ search shape lifts at fixed sim budget

- **Falsifiable signal (Phase 1):** wl_lower(P3 at n=1k) > wl_lower(P0 at n=1k) with non-overlapping Wilson CIs. Equivalent statement: AZ-search recipe at fixed v3.6 ckpt + sims=100 measurably out-strengthens the rule-bot-collapse + rollout-leaf recipe.
- **Theory:** at v3.6 strength, the policy network is stronger than the rule-bot (`progress/r110.md §4l`); replacing rule-bot collapse with policy-net opponent simulation should make the search more truthful. Value-head leaf removes the rollout-noise-versus-evaluation-noise tradeoff; whether it helps or hurts is empirical (cheap-inference fallback baseline was 0.39-0.45 wl_lower, but in a different recipe).

### H2 — Two-sided alone vs value-head alone is decomposable

- **Falsifiable signal (Phase 1):** P1 (two-sided, rollout) and P2 (single-sided, value-head) each measurable independently of P3. If P3 lifts but P1 and P2 both regress, the two changes are individually harmful but co-stabilising — interesting but suggests the synergy is recipe-fragile.
- **Theory:** rule-bot collapse only matches the leaf model; switching one without the other can mismatch the search frame (e.g. value-head leaf under single-sided still gets rule-bot-collapse terminal frames, which is what the value head was implicitly trained against in v3.6 self-play).

### H3 — Dropping KL anchor does not regress training stability over 8 iters

- **Falsifiable signal (Phase 2):** wl_lower trajectory across the 8-iter loop monotonically tracks the v3.6-extended trajectory shape (the iter-2 peak then plateau pattern). Catastrophic degradation (any iter wl_lower < 0.50) falsifies.
- **Risk:** the iter-2-peak-then-regress pattern is a characterized v3.6 recipe property (`progress/r110.md §1`). The W6 KL anchor was added to fight degradation. Strict AZ drops it; we accept the risk explicitly and widen `--max-consecutive-failures` to 3 to give the no-KL run room to recover from a noisy iter.

## 3. Phase 1 — probe (measurement, no retrain)

Two-axis A/B on the v3.6 cap128 cont-iter2 ckpt at n=120 smoke, then n=1k arbiter if smoke clears.

| Arm | Leaf | Two-sided | Source |
|---|---|---|---|
| P0 | rollout | off | Existing anchor — `R16-P1-v36-cap128-cont-iter2-tight-gate/gate.manifest.json` (n=10k, wl_lower=0.5880); re-measure at n=120 with same recipe for apples-to-apples Phase 1 |
| P1 | rollout | on | Tests two-sided alone (mirrors the two-sided-mcts-scoping A1 probe) |
| P2 | value-head | off | Tests value-head leaf alone (closes the r110 §4a follow-up) |
| P3 | value-head | on | Full AZ-style search shape |

### Recipe (Phase 1, all arms)

```
--onnx-path .claude/worktrees/feature-improvement/runs/R16-P1-v36-cap128-A1-continue/loop/iter-2/policy.onnx
--games 60 --model-side both
--sims 100 --prior policy
--collapse-max 64 --max-nodes 5000
# arm-specific: --leaf {rollout|value-head}, [--mcts-two-sided]
```

### Decision rules (Phase 1)

1. **All four arms regress or null at n=120 (smoke):** AZ-search recipe is null at v3.6 strength. Document negative result; do not fire Phase 2.
2. **Any of P1/P2/P3 ≥ P0 within Wilson noise at n=120:** escalate the leading arm(s) to n=1k arbiter.
3. **P3 collapses at n=120 (wl_lower < 0.50):** AZ-search recipe is incompatible with the v3.6 ckpt at sims=100. Document and isolate via P1/P2.
4. **Arbiter lift at n=1k:** fire Phase 2.

## 4. Phase 2 — AZ training loop

Only if Phase 1 arbiter shows any AZ arm ≥ baseline at n=1k.

```
python3 training/r12_orchestrator.py \
  --run-dir runs/R16-P2-v36-az-from-cont-iter2 \
  --init-from .claude/worktrees/feature-improvement/runs/R16-P1-v36-cap128-A1-continue/loop/iter-2/checkpoint.pt \
  --max-iters 8 --selfplay-games 240 --epochs 25 --lr 3e-4 --batch-size 64 \
  --mcts-simulations 100 --mcts-prior policy \
  --mcts-leaf value-head \
  --mcts-two-sided \
  --kl-anchor-weight 0.0 \
  --max-consecutive-failures 3
  # all other knobs at v3.6-extended defaults
```

Per-iter eval cadence:
- n=240 smoke after each train.
- n=1k arbiter if any iter beats the prior best at n=240 by > Wilson noise.
- n=10k tight gate on the best-promoted ckpt at end of run.

### Decision rules (Phase 2)

1. **Lift (wl_lower ≥ 0.6040 at n=10k):** AZ recipe is the new training default; v3.6 anchor retires. Open follow-ups: from-scratch ("Zero") run, sim-budget bump, replay-window widening.
2. **Soft-ship (0.5778 ≤ wl_lower < 0.6040):** AZ recipe ties v3.6-extended. Records that AZ search shape doesn't break the capacity ceiling at hidden=128/depth=2; next pivot is recipe-axis (hidden→256, depth→4) per `progress/r110.md §4k` or off-axis (set-attention).
3. **Regression (wl_lower < 0.5778):** isolate via single-axis retrains:
   - Re-fire with just `--mcts-two-sided` (drop value-head). Tests whether value-head-leaf during training is the culprit (would confirm r110 §4a).
   - Re-fire with just `--mcts-leaf value-head` (drop two-sided). Tests whether two-sided destabilises training.
   - If both single-axis runs also regress, the KL anchor was load-bearing — restore weak anchor (`--kl-anchor-weight 0.01`) and re-fire the full bundle.

## 5. Risks (pre-registered)

| Risk | Evidence | Mitigation |
|---|---|---|
| Value-head leaf untested at n=10k under v3.6 | `progress/r110.md §4a` flagged as follow-up | Phase 1 P2 isolates value-head leaf vs rollout on the same ckpt at the same recipe — closes the gap before any training compute is spent |
| Iter-2-peak-then-regress is a characterized recipe property | `progress/r110.md §1`, `docs/ai-research-backlog.md` | `--max-consecutive-failures 3` gives the no-KL run one extra noisy-iter recovery slot; falsification rule 3 step "restore weak anchor" is the compromise position |
| v3.6 trunk capacity-bound at hidden=128/depth=2 | v3.6 SOFT-SHIP verdict — per-side asymmetry (P=0.5500, O=0.6179) matches v3.5-cap128 exactly | Plan acknowledges "soft-ship" as the modal Phase 2 outcome; lift is the optimistic case |
| Two-sided + value-head together is untested | Two-sided-mcts-scoping A1 hasn't fired (rollout-leaf form); value-head leaf hasn't been measured at v3.6 | Phase 1 fires P1, P2, P3 independently to disentangle |
| Cheap-inference value-head baseline was 0.39-0.45 wl_lower | `docs/ai-research-backlog.md` "Cheap inference remains a fallback, not the production claim" | That baseline was a different recipe (adaptive-ratio=1.5, not full MCTS); not directly comparable. P2 measures value-head leaf under the v3.6 full-MCTS recipe — the first apples-to-apples test |

## 6. Out of scope (explicit)

- **From-scratch ("Zero") run.** Continues from v3.6 cap128 cont-iter2 for Phase 2. From-scratch is a follow-up if Phase 2 lifts; not bundled here.
- **Sim-budget axis.** `--mcts-simulations 100` held constant. The `gpu-fed-stronger-mcts-scoping.md` axis covers sim-budget bumps.
- **Replay-window widening.** W6 window=3 / old-fraction=0.40 held constant. The `cross-iter-opponent-pool-selfplay.md` axis covers replay expansion.
- **Opponent diversity / PFSP pool.** Single-opponent (current promoted) held constant. Pool variants are post-AZ literature.
- **Deck variety / deck-pair-sampling.** Inherits whatever deck-sampling is active at fire time (per the 2026-05-22 user directive). Not contested here.
- **Per-Uma slot tokens (v3.2 directive).** v3.6 explicitly excludes slot tokens per its scoping doc; this recipe inherits that exclusion. The slot-token directive predates the v3.4 falsification on the slot-token axis.

## 7. Implementation status

| Component | Status | Evidence |
|---|---|---|
| `--mcts-two-sided` flag in `r12_orchestrator.py` (run_selfplay + run_gate + run_started event) | LANDED 2026-05-26 | argparse exposed; cmd lists append when set; events.jsonl records `mcts_two_sided` |
| Two-sided MCTS in TS engine | LANDED (commit d2b34cd) | `two-sided-mcts-scoping.md §12` |
| Two-sided MCTS in Rust engine + sim-cli + NAPI | LANDED (commit d2b34cd) | sim-eval-gate accepts `--mcts-two-sided`; sim-mcts-selfplay accepts `--mcts-two-sided` |
| v3.6 schema in Rust binaries (state_dim=246) | BUILT 2026-05-26 09:14 | `engine-rs/target/release/sim-eval-gate` newer than v3.6 commits |
| v3.6 cap128 cont-iter2 ckpt + ONNX | PRESENT (worktree) | `.claude/worktrees/feature-improvement/runs/R16-P1-v36-cap128-A1-continue/loop/iter-2/policy.onnx` |
| A0 gate manifest (n=10k anchor) | PRESENT (worktree) | `.claude/worktrees/feature-improvement/runs/R16-P1-v36-cap128-cont-iter2-tight-gate/gate.manifest.json`, wl_lower=0.5880 |

## 8. File pointers

### Code
- `training/r12_orchestrator.py` — orchestrator with `--mcts-two-sided` flag (added 2026-05-26).
- `engine-rs/crates/sim-cli/src/bin/eval_gate.rs:116` — `--mcts-two-sided` in gate binary.
- `engine-rs/crates/sim-cli/src/bin/mcts_selfplay.rs:99` — `--mcts-two-sided` in selfplay binary.
- `engine-rs/crates/engine/src/mcts/driver.rs:93-100` — Dirichlet noise at root.
- `engine-rs/crates/sim-cli/src/bin/mcts_selfplay.rs:411-421` — temperature schedule on visit distribution.
- `training/uma_ai/selfplay_dataset.py:165-192` — value target z + policy target π.
- `training/train_bc.py:117-122,716` — KL anchor mechanism (drop via weight 0.0).
- `training/make_v36_priors_init.py` — v3.5 → v3.6 ckpt expander (template if from-scratch follow-up is needed).
- `training/make_v36_cold_init.py` — v3.6 cold-init (random) for future from-scratch runs.

### Research docs
- `docs/ai-research/scoping/v36-priors-and-arithmetic-scoping.md` — v3.6 schema definition.
- `docs/ai-research/scoping/two-sided-mcts-scoping.md` — two-sided MCTS implementation details.
- `docs/ai-research/progress/r110.md §4a, §4l` — value-head-leaf follow-up and v3.6 capacity verdict.
- `docs/ai-research-backlog.md` — cheap-inference value-head fallback context.

### State files
- `docs/ai-agent-state/queue.json` — Phase 1 probe + Phase 2 training queue entries (to add).
- `docs/ai-agent-state/escalations.md` — register here if Phase 1 surfaces a blocker.

## 9. Status line

**`IMPLEMENTED-FIRING-P0-P3 2026-05-26`** — wiring landed; Phase 1 probes firing on v3.6 cap128 cont-iter2; Phase 2 training loop user-pre-approved; gate decisions follow §3 and §4 falsification bands.
