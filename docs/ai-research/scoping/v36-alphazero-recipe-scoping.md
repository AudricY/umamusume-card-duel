# v3.6 AlphaZero-style Recipe — Scoping

- **Date:** 2026-05-26
- **Status:** **LANDED-AZ-RECIPE-INFERIOR-AT-V36-SCALE** — Phase 1 collapsed across all three AZ deltas (P3 wl=0.3798 at n=120, falsifying rule 3 at smoke); overrode into Phase 2 on calibration-mismatch grounds. Phase 2 halted at iter-4 after the predicted iter-2-peak-then-regress pattern (best iter-1 wl=0.4578 at n=240, in-recipe). Tight-gate at n=10k under the **anchor recipe** (rollout, single-sided): iter-1 wl=0.5868, **statistically tied** with v3.6 anchor 0.5880, with **per-side asymmetry halved** (0.068 → 0.034). Tight-gate at n=10k under the **matched (AZ) recipe**: iter-1 wl=0.4615, a **−0.125 regression** — the AZ recipe at inference is genuinely worse than rule-bot-collapse + rollout, even on the model trained for it. **Net: H1 (overall-strength lift) FALSIFIED. The only positive signals are (a) cross-recipe asymmetry redistribution and (b) ~7× faster eval (67s vs 495s at n=10k with wave-size=16).**
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

**`LANDED-NULL-WITH-ASYMMETRY-REDISTRIBUTION 2026-05-26`** — see §10 for full result chain.

---

## 10. Results

### 10.1 Phase 1 — gate-eval probes at n=120 on v3.6 cap128 cont-iter2

`runs/R16-P3-v36-az-phase1/{P0,P1,P2,P3}/gate.manifest.json` (n=120 each, --games 60 --model-side both, sims=100, prior=policy):

| arm | leaf | two-sided | wl_lower | win_rate | wl_upper | Δ vs P0 |
|---|---|---|---|---|---|---|
| P0 | rollout | off | 0.5442 | 0.6333 | 0.7142 | — (within Wilson noise of n=10k anchor 0.5880) |
| P1 | rollout | on | 0.4526 | 0.5417 | 0.6281 | −0.0916 |
| P2 | value-head | off | 0.3719 | 0.4583 | 0.5474 | −0.1724 |
| P3 | value-head | on | 0.3798 | 0.4667 | 0.5556 | −0.1644 |

**Pre-registered rule 3 fired** (P3 < 0.50). Override into Phase 2 on calibration-mismatch grounds: the v3.6 value head was trained on rule-bot-collapse self-play; using it as a MCTS leaf is OOD. The eval result measures inference compatibility; only retraining can measure adaptation.

### 10.2 Phase 2 — AZ training loop (8-iter cap, halt-after-3)

`runs/R16-P3-v36-az-from-cont-iter2/iter-*/gate.manifest.json` (n=240 each, in-recipe gate: value-head + two-sided, sims=100):

| iter | wl_lower (n=240) | promote | reason |
|---|---|---|---|
| 0 | 0.4372 | ✓ | floor=0.30 (init re-eval) |
| 1 | 0.4578 | ✓ | +0.0206 vs iter-0, BEST |
| 2 | 0.3964 | ✗ | < floor 0.4578 (FAIL #1) |
| 3 | 0.3883 | ✗ | < floor 0.4578 (FAIL #2) |
| 4 | 0.4005 | ✗ | < floor 0.4578 (FAIL #3 → HALT) |

The predicted iter-2-peak-then-regress pattern (scoping risk #2) fired hard with KL anchor dropped. Each iter ~30s wall (5s self-play + 16-27s distill + 5s gate); total run ~4 min wall.

### 10.3 Phase 3 — tight gates at n=10k on iter-1 (best AZ ckpt)

`runs/R16-P3-v36-az-from-cont-iter2/tight-gate-iter-1/gate.manifest.json` (anchor recipe: rollout, single-sided, sims=100):

| | overall wl_lower | player_wl | opp_wl | asymmetry |
|---|---|---|---|---|
| v3.6 cap128 cont-iter2 anchor | **0.5880** | 0.5500 | 0.6179 | 0.0679 |
| AZ iter-1 (anchor-recipe eval) | **0.5868** | 0.5659 | 0.5998 | 0.0339 |
| Δ | −0.0012 (tied) | +0.0159 | −0.0181 | **−0.0340 (halved)** |

`runs/R16-P3-v36-az-from-cont-iter2/tight-gate-az-recipe-iter-1/gate.manifest.json` (matched recipe: value-head + two-sided + wave-size=16, sims=100, n=10k):

| | overall wl_lower | player_wl | opp_wl | asymmetry | elapsed |
|---|---|---|---|---|---|
| iter-1 (AZ-recipe eval) | **0.4615** | 0.4424 | 0.4726 | 0.0302 | **67s** |
| iter-1 (anchor-recipe eval) | 0.5868 | 0.5659 | 0.5998 | 0.0339 | 495s |
| Δ (AZ vs anchor recipe, same ckpt) | **−0.1253** | −0.1235 | −0.1272 | −0.0037 | **7.4× faster** |

Production verdict: under the recipe iter-1 was TRAINED for, it loses 12.5 points of wl_lower vs the legacy rollout recipe. The AZ search mechanism (value-head leaf + two-sided opp nodes) is the regression source; rule-bot-collapse + rollout-to-terminal is the stronger search at v3.6 scale.

### 10.4 Interpretation

1. **Overall-strength bet (H1) FALSIFIED at hidden=128/depth=2.** Under the matched AZ recipe (the production form), iter-1 regresses −0.125 vs the v3.6 anchor. Under the legacy rollout recipe, iter-1 ties the v3.6 anchor (Δ=−0.0012). AZ search shape is the regression source — at v3.6 scale, rule-bot-collapse + rollout-to-terminal is genuinely a stronger search than value-head leaf + two-sided opponent nodes, even on the model trained for the AZ recipe.
2. **Per-side asymmetry redistribution is a real off-axis lift (anchor-recipe only).** The v3.6 anchor's known weak-player-side / strong-opp-side asymmetry (the surface that commit `cdd8ae5` flagged as the next probe target) **shrinks by half** when iter-1 is evaluated under the anchor recipe: player gains +0.0159, opp gives back −0.0181. Under the matched AZ recipe the asymmetry is also reduced (0.034 → 0.030) but at a floor 12 points lower, so the lift is moot for production. The asymmetry-redux finding is recipe-specific.
3. **Iter-2-peak-then-regress (H3) FALSIFIED as recipe-stability claim.** Dropping the KL anchor produces the same pattern as r110 §1 documented — peak at iter-1 then degrade. The W6 fixed-KL-anchor remains load-bearing for stability past iter-1.
4. **Cross-recipe transfer is robust for the underlying model.** Iter-1 was trained under value-head + two-sided MCTS; evaluating it under rollout + single-sided MCTS still ties the v3.6 anchor. The model's learned policy/value is generalizable across MCTS recipes — the inference-time gap is the search-mechanism difference, not a model-quality difference.
5. **Compute-side benefit of AZ recipe: 7.4× faster eval at n=10k.** Wave-size=16 batched inference on the value-head MCTS path completes 10k games in 67s vs 495s for the rollout path. If asymmetry-redistribution or compute were the production levers (rather than overall winrate), the AZ recipe would be the right choice. They aren't, so it isn't.

### 10.5 Follow-up backlog (if pursued)

- **AZ with weak KL anchor (`--kl-anchor-weight 0.01`).** Could the AZ recipe with a weak anti-drift regularizer survive past iter-1 and accumulate more asymmetry-reduction gains? Cheap (~4 min wall) follow-up.
- **Asymmetry-targeted recipe.** If asymmetry reduction is the only consistent AZ lift, an explicit asymmetry-penalty in the distill loss might dominate AZ search shape. Different axis.
- **From-scratch ("Zero") AZ.** Untested. The full AZ promise is bootstrap-from-random; this scoping inherited a v3.6 ckpt. Estimate: ~10× the compute of this Phase 2 (cold init + many more iters needed).
- **Sim-budget bump (sims=400 or 800).** Orthogonal to the search-shape axis tested here; lives in `gpu-fed-stronger-mcts-scoping.md`.

