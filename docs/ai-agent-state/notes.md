# Claude Harness Notes

The Claude Code harness is intentionally lightweight.

- Default command: `/work`
- Default execution path: one general `worker` subagent
- Purpose of the worker: context isolation, not parallel throughput
- No hooks in the initial implementation
- No Python harness in the initial implementation

Keep this file for durable harness notes that do not belong in sprint plans or AI research progress docs.

## F1 self-play readiness — scoping (2026-05-14)

**What R5 actually did.** R5 ran PPO from the DAgger iter-2 warm-start with rollouts against the
**item17 opponent pool, sampled uniformly** (see `docs/ai-research-backlog.md` line 252: "PPO from
DAgger iter-2 warm-start, rollouts vs the item17 opponent pool (DAgger iter-002 sampled uniformly)").
That pool was DAgger-era checkpoints from `runs/item17-2026-05-11/` whose strongest member topped
out at Wilson lower **0.3109** — the same number as the warm-start. R5 ran 3 iters × 800 games and
landed at Wilson 0.3109 (iter-2), matching the gate cap exactly. Mean_return moved -0.20 → -0.14
(reward distribution did shift), gate WR did not (`docs/ai-research-backlog.md` lines 254-260).

**What item-12 added on top.** `training/opponent_pool.py` defines `OpponentPool` with PFSP weights
`w_i = max(0.05, 1 - p_i)` (lines 122-159), strict-monotone cycling alarm (lines 164-180), and
retention "last 8 + every 4th historical, hard cap 24" (lines 56-102). This is the infra that lets
the orchestrator maintain a *versioned* pool of promoted checkpoints across iterations and weight
sampling by per-opponent win-rate signal.

**Current wiring state.** `ppo_orchestrator.py` has `--rollout-vs-pool` (line 829) and `--pool-path`
(line 831). When `--rollout-vs-pool` is set, `_select_pool_opponent` (lines 836-863) loads the pool
and picks **one opponent per run** via `rng.choice(list(pool.entries))` (line 862, seed
`trace_seed_start + 9973` — explicitly "stable per-run, doesn't depend on iter"). The selected
checkpoint is exported to ONNX and served as `--opponent-model-url` for all rollout games this
iteration (lines 538-555). **Two real gaps vs the post-mortem's "PFSP self-play" framing:**
(a) sampling is one-shot uniform per run, not per-game and not PFSP-weighted; (b)
`opponent_pool.pfsp_weights()` exists but is never called from `ppo_orchestrator`.

**The diagnostic question.** Is "PFSP pool of W6/iter-2 + earlier promoted checkpoints" a different
experiment from R5? **Yes, materially.** R5's pool was DAgger iter-002 checkpoints (Wilson ~0.31).
W6/iter-2 is Wilson **0.6479** (R13-W6-phase-d) — a fundamentally stronger opponent set. R5 closed
the "self-play against a weak pool" question; it did NOT close the "self-play against a strong
pool" question.

**Go / no-go: GO, but with plumbing-clarity caveat.** The proposed sweep is a genuinely different
experiment from R5 because the pool members are ~30pp stronger. Cost is cheap (~10-15 min compute
per F1 wall-clock datapoint). However, the operator should know: today's `--rollout-vs-pool` picks
one opponent per run and does NOT exercise PFSP weights. For an honest "PFSP self-play" test the
sweep needs either (i) to accept "one strong opponent sampled once per run" as the actual
experiment (simplest, recommended for first attempt) or (ii) a small plumbing patch to call
`pfsp_weights()` with rolling history and re-sample per game (deeper, defer to iteration 2 if
iteration 1 promotes).

**Proposed sweep config (first attempt, "one strong opponent per run"):**
- Pool composition: hand-built `opponent-pool.json` with three entries — `runs/R13-W6-phase-d/iter-0/checkpoint.pt`,
  `runs/R13-W6-phase-d/iter-1/checkpoint.pt`, `runs/R13-W6-phase-d/iter-2/checkpoint.pt`. Iteration
  numbers 0/1/2 to match retain() ordering.
- Warm-start: `runs/R13-W6-phase-d/iter-2/checkpoint.pt` (same checkpoint also in pool; the
  one-per-run sampler may pick a self-game, that's fine).
- HPs: aggressive (mirror F1 phase H so the comparison axis is opponent only) — lr=3e-4,
  clip=0.3, entropy_coef=0.01, ppo_epochs=4.
- Rollouts: 3 iters × 800 games/update; ratio of self-play vs vs-rule-bot rollouts = 100% self-play
  (no mix; the gate already evaluates vs rule-bot so we don't double-up).
- Wall-clock: F1 datapoint is ~5-6 min per 800-game update + a few minutes for ONNX export + gate.
  3 iterations ≈ 30-45 min total.
- Exit / gate: same R15.S2 criterion — Wilson lower ≥ 0.40 on the rule-bot gate at any iter.
  Stretch: if iter-1 promotes, run an iteration 2 with the PFSP patch in place.

## F1 better warm-start — scoping (2026-05-14)

**Hypothesis (pre-registered).** The DAgger warm-start at `hidden_dim=64 / depth=2` is
*compute-starved, not capacity-starved*: holding model size fixed and scaling games + epochs by
~3× will lift the warm-start's greedy-eval Wilson lower from item17-2026-05-11's **0.311** (iter-2,
n=200, [0.311, 0.444], `docs/ai-performance-research-progress.md:582`) to **≥ 0.45**, giving PPO an
actually-movable starting point. Prediction: ≥ +14pp Wilson-lower lift over item17 iter-2.
Falsification: Wilson lower stays within ±2pp of 0.311 → the cap is upstream of compute (label
noise / representation / capacity), and additional epochs only sharpen the same imitation error.

**What we know (current item17 config + result).**
- Canonical command pre-registered for the take-1 sweep: `dagger_orchestrator.py --iterations 3
  --games 250 --max-steps 500 --teacher rollout --rollout-steps 500 --rollout-crn-samples 3
  --replay-games 100 --epochs 30 --batch-size 32 --hidden-dim 64 --depth 2 --eval-games 250
  --kl-anchor-weight 0.1 --kl-anchor-weights "0.0,0.1,0.5"`
  (`docs/ai-performance-research-progress.md:511-528`).
- **What actually ran at item17-2026-05-11 (take-2, the bedrock warm-start).** Compute-downscaled:
  30 trace games, 100-game eval, 25 epochs (`docs/ai-performance-research-progress.md:571-573,
  618-619`). This is the run whose iter-2 checkpoint sits at
  `runs/item17-2026-05-11/iter-002/checkpoint.pt` and is the warm-start every F1 PPO phase
  loaded from.
- `dagger_orchestrator.py` defaults (`training/dagger_orchestrator.py:1024-1056`): `--iterations 3`,
  `--games 20`, `--rollout-steps 200`, `--rollout-crn-samples 3`, `--epochs 8`, `--batch-size 32`,
  `--hidden-dim 64`, `--depth 2`, `--kl-anchor-weight 0.0`.
- `train_bc.py` defaults fed in (`training/train_bc.py:722-766`): `--lr 3e-4` cosine
  (`--lr-warmup-steps 0`), `--weight-decay 1e-4`, `--dropout 0.05`, `--value-weight 0.1`,
  `--entropy-bonus 0.0`, `--split-by episode`, `--data-mode bc`, `--policy-weight 1.0`.
- Result: iter-2 Wilson lower **0.311** at WR 37.5% (n=200). All F1 PPO phases stuck within ±0.5pp
  of this number (`docs/ai-performance-research-progress.md:725-735`).

**What has already been tried (so this run is genuinely new).**
- **R3** added entropy bonus β ∈ {0.05, 0.2} to the BC loss; β=0.05 hit entropy 0.527 nats but
  gate WR stayed at 38.5% — statistically equivalent to the warm-start
  (`docs/ai-research-backlog.md:147`).
- **R4** retrained value head with `--epochs 50 --value-weight 1.0`; value Brier dropped but gate
  WR fell to 34.5% (`docs/ai-research-backlog.md:233-235`).
- **R6 (capacity)** trained `--hidden-dim 128 --depth 3 --epochs 50 --value-weight 1.0`; train
  accuracy hit 0.97, argmax-match 0.832, gate WR **33.0%** — *worse* than the 64/2 warm-start
  (`docs/ai-research-backlog.md:227-236`). Standing diagnosis: "the cap is
  imitation-target-quality, not capacity."
- **R3-b020 + PPO** combined β=0.2 entropy + value retrain + aggressive PPO, still stuck at
  Wilson lower 0.30 (`docs/ai-research-backlog.md:238-248`).
- **Not yet tried: same size, same data recipe, more games + more epochs (no other axis moved).**
  R6 covaried capacity and epochs; R4 covaried value-weight and epochs. The pure "more compute,
  same shape" axis at the canonical 64/2 size has not been run as a controlled follow-up to
  item17-2026-05-11. The pre-registered 250-game/30-epoch sweep was *documented* but never run
  (take-2 was downscaled to 30/25 for wall-clock).

**Proposed config (refines queue's "3x data, 3x epochs").** Launch
`training/dagger_orchestrator.py` with:

- `--games 90` (3× item17 take-2's 30; the documented full-scale 250 is 8× and out of scope here).
- `--epochs 75` (3× item17 take-2's 25).
- `--iterations 3` (unchanged — single DAgger inner loop on the same teacher).
- `--max-steps 500`, `--rollout-steps 500`, `--rollout-crn-samples 3` (unchanged from take-2).
- `--replay-games 100`, `--relabeled-weight 0.6`, `--replay-weight 0.4` (unchanged).
- `--batch-size 32` (unchanged; bumping to 64 is allowed only if memory headroom — not required).
- `--hidden-dim 64 --depth 2` (UNCHANGED — load-bearing pre-registered constraint; this is what
  isolates compute-starved from capacity-starved).
- `--kl-anchor-weight 0.1 --kl-anchor-weights "0.0,0.1,0.5"` (unchanged off/low/high schedule).
- `--eval-games 250` (the originally-documented eval budget; keeps n=500 side-balanced per iter so
  Wilson half-width is ~4pp — required to resolve a +14pp lift against the noise floor).
- `train_bc.py` flags via orchestrator pass-through stay at default: lr=3e-4 cosine,
  weight-decay 1e-4, dropout 0.05, value-weight 0.1, **entropy-bonus 0.0** (we are NOT redoing R3;
  this run isolates the SL-compute axis). If train loss bottoms out but Wilson lower stalls, a
  follow-up may layer `--entropy-bonus 0.05` as a single-axis ablation.

**Exit criteria.**
- *Primary success:* any iter's greedy-eval Wilson lower **≥ 0.45** at n ≥ 500 side-balanced.
  Promotes to a tier-1 F1 PPO re-attempt from this checkpoint.
- *Secondary:* SL train loss strictly decreasing across the 75 epochs (no plateau before
  epoch 50). If the loss plateaus by epoch ~25 (matching take-2), that itself confirms
  "compute is saturated at the current size" — falsification path is supported.
- *Falsification:* iter-2 Wilson lower in 0.311 ± 2pp (i.e. 0.29–0.33). Closes the
  compute-starved branch; the cap is then *not* compute. Remaining attention routes to R7
  (multi-teacher labels), R8 (DPO), or the R12-line evidence that the cap is downstream of
  value-head leaf noise rather than SL labels.

**Expected cost.** Item17 take-2 (30 games × 25 epochs) ran in <10 min wall-clock
(`docs/ai-research-backlog.md:638`). Scaling games 3× (more rollout-CRN teacher invocations, the
dominant cost) and epochs 3× gives a ~9× wall-clock estimate → **~1.5h compute** for the
3-iteration sweep at the proposed config. Within the "1–2h" envelope at
`docs/ai-research-backlog.md:638`. n=500 eval × 3 iters adds ~20–30 min.

**Risk register.**
- *Overfit to noisy teacher labels.* R6 already showed 128/3 + 50 epochs over-imitates the
  teacher's 25% argmax errors and drops gate WR. The same risk applies at 64/2 + 75 epochs; the
  secondary "loss-still-declining" check is the early warning.
- *Label-distribution drift.* Each DAgger iter relabels with the same rollout-CRN×3 teacher; at 3×
  games per iter the trace distribution shifts (more states visited by the iter-0 policy rather
  than rule-bot baseline). The KL anchor schedule (0.0/0.1/0.5) mitigates but does not eliminate;
  monitor per-iter Wilson rather than only iter-2.
- *Compute under-spend.* If `--games 90 / --epochs 75` is still below saturation, a negative
  result is uninformative about the hypothesis. Mitigation: log train loss per epoch and verify
  the curve is still declining at epoch 75. If it is, the hypothesis is *not* falsified — only
  the specific 3× scale is.

**Decision gate (human-owned, before paying compute).** The /work harness must NOT launch this
sweep autonomously (warn-before-launch rule in `CLAUDE.md`). Before running, the human should
confirm: (a) the diagnosis-quality argument above is acceptable — specifically that R6's capacity
result does not subsume this experiment because R6 covaried capacity with epochs and this run
isolates the compute axis at fixed capacity; (b) the ~1.5h compute budget is available; (c) the
falsification path (Wilson lower stays at 0.31 ± 2pp) is acceptable as a research outcome and
will close the branch rather than spawn further SL ablations. On green light, launch with the
command in the "Proposed config" section; on red light, demote to R15.S3 / R15.S4 instead.

**Closeout (2026-05-14).** **FALSIFIED.** Run `runs/R15-S1-warmstart-sweep/` finished iter-2 at
Wilson lower **0.3269** (WR 36.8%, n=500 side-balanced) — inside the pre-registered band
0.311 ± 2pp = [0.291, 0.331] (line 141). The +1.6pp lift vs item17 take-2's 0.311 is below Wilson
half-width at n=500 and an order of magnitude below the predicted +14pp. The **secondary check
(line 138-140) also fired**: per-epoch val_acc reached 99% of its peak by epoch 2-3 in every iter
(best @ ep7 / ep2 / ep3 for iter-0/1/2) then *declined* over the remaining 70+ epochs while
train_acc climbed to 0.91-0.97 — plateau-then-overfit replicated across all three DAgger iters,
not just iter-0. Wall-clock 11m 51s (~8x faster than the ~1.5h estimate; rollout-CRN×3 teacher
dominated, eval budget was not the bottleneck). The compute-starved branch is closed; attention
routes to R15.S3 (reward shaping), R15.S4 (sampling-temperature gate, diagnostic), R7
(multi-teacher labels), or R8 (DPO). Full writeup:
`docs/ai-performance-research-progress.md` § "Phase K — F1 DAgger compute-scaled warm-start".

## F1 reward shaping — scoping (2026-05-14)

**Hypothesis (pre-registered).** Augmenting PPO's per-step reward with shaped intermediate
signals derived from already-traced `PublicObservation` fields (Δactive-energy-attachment,
Δbench-development, retreat-event indicator, Δcard-throughput, Δactive-hp-relative) at
aggregate per-game shaped magnitudes ≈ ±0.3 (sub-dominant to ±1 terminal, comparable order
to current α·Δpoints) will lift iter-2 greedy-eval Wilson lower from the F1 cap of **0.3109**
(item17 iter-2, `docs/ai-performance-research-progress.md:582`; matched within ±0.5pp by phase
H, phase J/R5, R15.S1) to **≥ 0.40** at n=500 side-balanced vs rule-bot, holding warm-start
(`runs/item17-2026-05-11/iter-002/checkpoint.pt`) and opponent (rule-bot, no pool) fixed.
**Falsification:** iter-2 Wilson lower lands in 0.311 ± 2pp (= [0.291, 0.331]); closes the
reward-shape branch of the F1 post-mortem and re-ranks R7 (multi-teacher labels) and R8 (DPO).

**What we know (current reward shape).** Per-step reward is constructed in
`training/ppo_orchestrator.py:parse_trace_to_trajectories` at lines **715-739**: episode-level
loop over trace rows, computing
`delta = (own_points - opp_points) - (prev_own - prev_opp)` (line 724), then
`reward = alpha * delta` (line 725) with default `alpha = 1/3` (line 811). On the terminal row,
`reward = reward + beta * win_indicator` (line 739) with default `beta = 1.0` (line 813) and
`win_indicator ∈ {+1, 0, -1}` (lines 733-738). This is **pure orchestrator-side trace
post-processing** — no sim-side reward emission. Every shaped reward at training time is
computed from fields already on each `row.observation` snapshot.

Empirical mean_return magnitudes (the per-episode reward sum the surrogate gradient gets):
R14-f1-self-play-sweep `stage=gae` events show `mean_return = -0.236 / -0.173 / -0.153` for
iter-0/1/2 (`runs/R14-f1-self-play-sweep/events.jsonl`); R15-S1's reward shape is identical so
the gradient floor is comparable. **Design constraint** for new shaping: per-game shaped sum
should target ±0.3 — same order as Δpoints sums, sub-dominant to the ±1.0 terminal so the
policy does not chase shaping at expense of winning. Naive coefs (e.g. 0.05/step × 60
decisions = 3.0) overwhelm the terminal and trigger reward-hacking.

**What has already been tried (so this run is genuinely new).**
- **R3 (BC entropy bonus β ∈ {0.05, 0.2}, FAIL).** Gate WR 38.5%
  (`docs/ai-research-backlog.md:147`). SL-loss change; did not touch PPO reward.
- **R4 (value-head retrain, FAIL).** Gate WR 34.5% (`docs/ai-research-backlog.md:233-235`).
  Did not touch PPO reward.
- **R6 (capacity 128/3, FAIL).** Gate WR 33.0% (`docs/ai-research-backlog.md:227-236`). Did
  not touch PPO reward.
- **R15.S1 (compute 3× games + 3× epochs at 64/2, FAIL).** Iter-2 Wilson 0.3269 inside
  falsification band (notes.md "F1 better warm-start — scoping" closeout). Did not touch PPO
  reward.
- **R15.S2 (strong-pool self-play, FAIL).** Iter-2 Wilson 0.2188; regressed via
  co-adaptation (digest slot 8). Did not touch PPO reward. **Load-bearing for this scoping:**
  R15.S2's ratio range exploded from phase H's ~1.00 to 0.05-4.6 (digest slot 8) — proving
  PPO *can* move when the gradient is non-zero; the cap is the reward shape, not PPO itself.
- **R7 (multi-teacher labels)** and **R8 (DPO)** sit upstream of PPO (different SL labels /
  different loss class); neither subsumes the per-step reward-shape axis.

Net: no prior R-line touched `parse_trace_to_trajectories`'s per-step reward. This is a clean
new axis.

**Proposed reward signals (5 candidates, all from `PublicObservation` already on every trace
row — see `frontend/src/game/engine/ai-policy/observation.ts:6-60`).**

1. **Δactive-energy-total** — `own.active.energyTotal - prev_own.active.energyTotal`. Captures
   "develop active uma" strategic axis. Source: `observation.own.active.energyTotal`
   (`observation.ts:46, 54`: `energyTotal = Σ Object.values(umamusume.energies)`). Expected
   per-step magnitude: +1/turn → recommended coef ≈ 0.02 → ~0.02/step shaped contribution.
2. **Δbench-development (sum-of-bench-energyTotal)** — `Σ(b.energyTotal for b in own.bench if
   b) - prev`. Captures battery-bench strategy distinct from face-active. Source:
   `observation.own.bench[i].energyTotal` (`observation.ts:26, 35, 46`). Recommended coef ≈
   0.02; ~0.02/step.
3. **Retreat-event indicator** — `1.0 if (own.usedRetreatThisTurn && !prev_own
   .usedRetreatThisTurn) else 0.0` per row. Source: `observation.own.usedRetreatThisTurn`
   (`observation.ts:38`). Captures defensive recoveries (sparse, ~0-3 per game). Recommended
   coef ≈ 0.03; sub-game total ~0.0-0.1.
4. **Δcard-throughput** — `(own.handCount + len(own.discard)) - prev`. Captures cycling tempo
   (discard grows monotonically; together with hand they index "cards seen"). Source:
   `observation.own.handCount`, `observation.own.discard` (`observation.ts:31, 33`).
   Recommended coef ≈ 0.02; ~0.02/step.
5. **Δactive-hp-relative** — `(own.active.hp/own.active.maxHp) - (opp.active.hp/opp.active
   .maxHp)` delta. Captures chip damage trade efficiency at the active-vs-active interface
   that Δpoints marginalizes over (Δpoints fires only on KO). Source: `observation.own
   .active.hp/maxHp`, `observation.opponent.active.hp/maxHp` (`observation.ts:52-53`).
   Recommended coef ≈ 0.05 (the largest of the five — direct damage proxy); ~0.05/step on
   active turns.

Aggregate target per game (≈60 model decisions × combined per-step magnitude ≈ 0.13 ×
average |delta| 0.4) ≈ ±0.3 — sub-dominant to ±1 terminal, comparable to existing Δpoints
sum. **No sim-side emission required.**

**Proposed shaping schedule (pre-registered).** **Linear decay to zero across the 3 iters:**
iter-0 full shaping (coefs as above), iter-1 0.5× shaping, iter-2 0× shaping. Rationale: the
policy must converge to optimize the *unshaped* terminal+Δpoints signal; shaping only exists
to provide non-zero gradient at the start of training where importance ratios sit at ~1.0
(phase H, phase J-iter-0 ratios near 1.0). Pre-registered backup: if iter-0/1 promote but
iter-2 regresses, follow up with constant-throughout shaping (NOT done in this sweep — that's
a separate experiment).

**Implementation site (orchestrator-only — load-bearing for human work-size).**
`training/ppo_orchestrator.py:parse_trace_to_trajectories` lines **715-740**. Add five new
coefficient args mirroring `--reward-alpha`/`--reward-beta` at lines 811-814 (e.g.
`--reward-active-energy-coef 0.02`, `--reward-bench-energy-coef 0.02`,
`--reward-retreat-coef 0.03`, `--reward-throughput-coef 0.02`, `--reward-hp-diff-coef 0.05`),
plus `--reward-shaping-schedule {linear-decay,constant}` consumed by the iteration loop to
scale all five per iter. Inside the row loop, parse the five new fields from `observation`
and accumulate into `reward` after line 725 and before line 739.

**Estimated diff:** ~50-80 lines added in `training/ppo_orchestrator.py`, zero lines in
`training/train_ppo.py` (the trajectories.jsonl schema carries an opaque `reward` float —
line 753), zero lines in any TS/sim file. **No sim-side change required.** Human work: ~1h
to implement + ~30 min for `TMPDIR=/tmp npm run test:ppo-smoke` verification ≈ 1.5h before
launch. This is NOT a launch-and-watch decision like R15.S1/S2 — code lands first.

**Proposed experiment config (first attempt).**
- Warm-start: `runs/item17-2026-05-11/iter-002/checkpoint.pt` (same as every F1 phase —
  preserves the comparison axis to phase H 0.3109 and R15.S1 0.3269).
- Opponent: rule-bot only (no `--rollout-vs-pool` — R15.S2 closed the strong-pool branch).
- HPs: mirror phase H aggressive (`docs/ai-performance-research-progress.md:725-735`):
  `--lr 3e-4 --clip-epsilon 0.3 --entropy-coef 0.01 --ppo-epochs 4`.
- Sweep shape: `--iterations 3 --games-per-update 800`.
- Reward args (existing preserved): `--reward-alpha 0.333 --reward-beta 1.0`; plus the five
  new coefs at the magnitudes in the signal catalog above; plus
  `--reward-shaping-schedule linear-decay`.
- Eval: in-orchestrator gate at n=500 side-balanced (`--eval-games 250` × 2 sides — matches
  R15.S1).
- Diagnostics required: per-iter `mean_return` (already emitted via `stage=gae`); per-iter
  shaped-component attribution (NEW event: sum of each component across episodes — needed for
  the branch-1 vs branch-2 falsification split below).

**Exit criteria.**
- *Primary success:* any iter's gate Wilson lower **≥ 0.40** at n=500 side-balanced.
- *Falsification (branch 1, "reward-shape branch closed"):* iter-2 Wilson lower in
  [0.291, 0.331] AND importance ratios moved off 1.0 (per-iter ratio range > [0.5, 2.0],
  matching R15.S2's gradient-active signature, digest slot 8). Says "PPO moved the policy
  toward the shaped reward, but the shaped reward did not encode winning" — imitation cap is
  downstream of any orchestrator-side reward.
- *Falsification (branch 2, "even shaping didn't move PPO"):* iter-2 Wilson in same band AND
  importance ratios still ≈1.00 (matching phase H stasis). Says the issue is upstream of
  reward — behavior-policy log-probs are sharp enough that no per-step reward changes the
  surrogate gradient. Re-ranks R8 (DPO) over R7.
- *Secondary diagnostic:* shaped-component attribution per iter. If one component dominates
  by >5× the others, future shaping work drops the small contributors.

**Expected cost.** Phase H 3 iters × 800 games ran in ~340s (~5-6 min) for the PPO loop
itself (notes.md "F1 self-play readiness — scoping" line 63). Reward shaping adds zero
runtime cost (a few extra float adds per trace row parsed from already-loaded JSON).
**Total compute: ~10-15 min for sweep + gates.** Implementation: ~1.5h human work before
launch.

**Risk register.**
- *Reward hacking.* Policy maximizes shaped signal at expense of winning (e.g. piles energy
  on active uma instead of attacking). **Mitigation:** linear-decay schedule guarantees iter-2
  evaluates on the clean terminal+Δpoints reward; shaped-component attribution exposes a
  stuck-on-shaping policy at iter-1 before it locks in. Coefs sized to keep per-game shaped
  sum ≈ ±0.3 (sub-dominant to ±1.0 terminal).
- *Signal redundancy.* The five signals correlate with Δpoints (winning ≈ attaching energy +
  drawing cards + KOing). **Mitigation:** branch-1 vs branch-2 falsification split
  distinguishes "reward moved gradient but didn't help" (redundancy) from "reward did not move
  gradient" (upstream issue). Counter-evidence to full redundancy: R15.S2 showed PPO *can* be
  moved with a richer opponent (ratios 0.05-4.6 vs 1.00 — digest slot 8); the five fields
  above span axes (active-development, bench-development, defensive plays, throughput,
  damage) that the Δpoints aggregate marginalizes over.
- *Sim-side change creeping in.* If a desired sixth signal turns out to require a field not in
  `PublicObservation`, the orchestrator-only work-size assumption breaks. **Mitigation:** the
  five signals above are pre-committed-to and all confirmed present in
  `observation.ts:6-60`. A sixth signal routes back through human decision gate.

**Decision gate (human-owned, before paying code work + compute).** Three items the human must
confirm:
(a) The orchestrator-side diff scope (~50-80 lines in
`training/ppo_orchestrator.py:parse_trace_to_trajectories` + the five new CLI args; zero
sim-side; ~1.5h human work) is acceptable. This is NOT a "compute only" decision.
(b) The falsification outcome (either branch 1 or branch 2) is acceptable as a research
deliverable. Closing the reward-shape branch routes attention to R7/R8 and re-ranks them per
which branch fires.
(c) The pre-registered linear-decay schedule is acceptable as the first attempt vs
constant-throughout. (Constant-throughout is reserved as a follow-up only if linear-decay
produces a non-degenerate iter-1 lift that decays at iter-2 — a partial-success signature
distinct from outright falsification.)

On green light: implement the diff, run `TMPDIR=/tmp npm run test:ppo-smoke` to verify
trajectories.jsonl still parses end-to-end, then launch the sweep. On red light: demote to
R15.S4 (sampling-temperature diagnostic) or route directly to R7/R8.

**Closeout (2026-05-14).** **PARTIAL — fourth quadrant.** Run
`runs/R15-S3-reward-shaping-sweep/` finished iter-2 at Wilson lower **0.3560** (WR 39.8%, n=500
side-balanced) — *missed* the success bar 0.40 by 4.4pp, *landed above* the falsification band
[0.291, 0.331] by +2.5pp. Neither outcome from the pre-registered binary scoping fired cleanly.
Trajectory monotone 0.2730 → 0.2787 → 0.3560 (+8.3pp end-to-end, +7.7pp iter-1→iter-2). All three
iters promoted (decision floor 0.2787, tolerance 0). **Pre-registered branch-1 mechanism check
fired**: importance ratios moved decisively off ~1.00 in every iter (ratio_max 19.06 / 32.59 /
11.12 vs phase H's ~1.00 baseline) — the shaped reward IS moving the surrogate gradient and the
gradient IS moving the policy. Entropy stable (0.168 → 0.154 → 0.153, no collapse);
`numerical_anomalies = 0` across all 48 minibatches; approx_kl_max < 0.02. Wall-clock **5m 53.7s**
end-to-end (~4-9× faster than the 10-15 min scoping forecast). **Headline narrative:** R15.S3 is
the **best F1 rule-bot result on record** at +4.5pp absolute over the prior phase-H/K ceiling of
0.31, and the first F1 PPO configuration to materially exceed (not match) the SL cap. The branch
is **alive, not closed**; the gap to 0.40 is now quantitative (coefficient scale + signal mix)
rather than mechanistic. **Implication for F1 post-mortem:** the "0.40 NOT REACHABLE" framing in
`docs/ai-performance-research-progress.md:735` was correct under the unshaped reward but is now
qualified — once the reward axis is allowed to move, 0.40 is reachable in principle from this
warm-start; the open question is whether scaling coefs gets us there. **Implementation cost:**
+103 LOC orchestrator-only diff (`training/ppo_orchestrator.py`: 5 new `--reward-*-coef` args +
`--reward-shape-start/-end` linear decay + `shape_attribution` event); zero sim-side; zero TS-side;
`TMPDIR=/tmp npm run test:ppo-smoke` PASS pre-launch. **Next move (queued as
`r15-s3-followup-tune-shaping`):** hold the same 5 signals, scale all five coefs 1.5–2× (current
~0.13/game sum sits at the low end of the scoping ±0.3 budget), rerun the 3-iter sweep at the
same warm-start / opponent / HPs. Expected ~10-15 min compute. If iter-2 crosses 0.40 → first F1
success on record. If iter-2 stalls at ~0.36 → coef scaling is saturated; next move is signal-mix
change. Full writeup: `docs/ai-performance-research-progress.md` § "Phase L — F1 PPO + reward
shaping".

**Follow-up Closeout — Phase M (R15.S3 1.75× coef-scaling, 2026-05-14).** **SATURATION as
pre-registered.** Run `runs/R15-S3-followup-tune/` — same 5 signals, all coefs scaled 1.75×
(active 0.035 / bench 0.035 / retreat 0.0525 / throughput 0.035 / hp-diff 0.0875), same
warm-start / opponent / HPs / code as Phase L; per-game shape budget ~0.23 (vs Phase L ~0.13,
scoping target ±0.3). Wilson lower per iter: iter-0 **0.2960** (WR 33.6%, **+2.3pp vs Phase L
iter-0 0.2730**, promoted) → iter-1 **0.2825** (WR 32.2%, **REJECTED**, `wilson_lower 0.2825 <
floor 0.2960` — **first F1 sweep-internal regression on record**) → iter-2 **0.3580** (WR
39.8%, rolled forward from iter-0 parent after iter-1 reject, promoted). iter-2 0.3580 vs
Phase L iter-2 0.3560: **Δ +0.002, within Wilson noise**. The pre-registered "iter-2 stalls
at ~0.36 → coef scaling is saturated" outcome fired cleanly. The 1.75× scale is the saturation
point of the existing 5-signal mix under linear-decay shaping; further coef scaling will not
move iter-2 closer to 0.40 from this warm-start. Mechanism check (still healthy, not a
PPO-stopped-working outcome): ratio_max 15.86 / 12.50 / 18.82 across iters — gradient still
decisively off ~1.00, comparable to Phase L's 19.06 / 32.59 / 11.12 range; entropy stable
(0.171 → 0.164 → 0.164, no collapse); `numerical_anomalies = 0` across all 48 minibatches;
approx_kl_max < 0.02. Wall-clock **6m 22.3s** (run_started → run_completed delta:
`1778732302.62 → 1778732684.93 = 382.31s`); coef scaling added zero runtime cost. Full
writeup: `docs/ai-performance-research-progress.md` § "Phase M — F1 PPO + reward-shape
coef-scaling follow-up".

**Pre-scoping for the next single-axis move (queued as P3 `r15-s3-signal-mix-or-schedule`).**
Phase M closed the coef-magnitude axis at the existing 5-signal / linear-decay tuple. Three
plausible next single-axis moves, ranked least-to-most ambitious:

1. **Constant-shaping schedule (recommended v1).** Hold the 5 signals and original Phase L
   1.0× coefs fixed; change only the schedule by setting `--reward-shape-end 1.0` (instead of
   0.0) to disable the linear decay. Rationale: the iter-1 rejection in Phase M is evidence
   that decaying shape mid-sweep is *itself* destabilizing — iter-1 over-corrects toward shape
   at scale 0.5×, gets rejected, iter-2 retreats. Holding shape constant across iters lets the
   policy keep climbing on the shaped signal instead of being asked to converge to a
   deterministic terminal-only optimum. Risk: reward hacking at iter-2 (policy learns to
   maximize shape at expense of winning). Mitigation: ratio_max / entropy / shape_attribution
   diagnostics already in place; falsifier is iter-2 Wilson < 0.30 with shape_attribution
   dominating a single component. **Smallest possible change** (one CLI flag flip from R15.S3
   baseline), same warm-start / opponent / HPs / code. Expected ~6 min compute. Pre-register:
   success if iter-2 ≥ 0.40; partial if 0.36 < iter-2 < 0.40; saturation if iter-2 ≤ 0.36 (then
   pivot to signal-mix change next).

2. **Signal-mix change (drop weak + scale strong).** Audit per-signal contribution from
   Phase L/M `events.jsonl` `trajectory-parse/completed.data.shape_attribution`. Phase M iter-0
   attribution: active-energy 147.6, bench-energy 8.5, retreat 0.0, throughput 299.6, hp-diff
   -14.0 (sums over 1600 episodes). throughput dominates by 2× over active-energy; bench-energy
   and retreat contribute nearly nothing; hp-diff is net-negative (the policy is taking damage
   on the way to its shaped reward). A v2 mix would drop bench-energy + retreat (sparse / near-
   zero), keep throughput + active-energy at ~0.10-0.15 each, and either drop hp-diff or flip
   its sign (a net-negative contribution suggests the signal is rewarding the wrong direction).
   Requires re-pre-registering coefs but no code change. Expected ~6 min compute. **Larger
   change than v1** — touches the signal axis directly.

3. **Value-head-derived strategic-tempo signal (most ambitious).** Replace one shaped signal
   with a per-turn "win-probability-delta" computed from the model's own value head: shape =
   `value(s_{t+1}) - value(s_t)`. Conceptually closer to "shape towards what the value head
   thinks is winning" than "shape towards in-game observables". Magnitude small (probability
   delta, not raw score) so the per-step rew sum stays sub-dominant. Requires plumbing the
   value-head output through the orchestrator's trace stream (currently the orchestrator only
   sees `PublicObservation` row data, not model outputs at trace time). **Code work ~2-3h**;
   compute identical. Defer until v1 or v2 has been tried.

**Recommendation: try v1 first** (constant-shaping). It is one CLI flag away from the R15.S3
baseline, isolates the schedule axis cleanly from the signal axis, and the falsification outcome
("iter-2 still stalls at ~0.36 with constant shape") is informative — it would point definitively
at the signal mix being the bottleneck rather than the schedule. Queued as P3
`r15-s3-signal-mix-or-schedule` `ready` (autonomous-launch eligible under user's blanket
AI-launch permission).

**Phase N Closeout — R15.S3 constant-shape schedule axis (2026-05-14).** **PARTIAL — new F1
ceiling on record at iter-2 Wilson 0.3677.** Run `runs/R15-S3-constant-shape/` — same 5 signals
at Phase L 1.0× coefs (active 0.02 / bench 0.02 / retreat 0.03 / throughput 0.02 / hp-diff
0.05), same warm-start (`runs/item17-2026-05-11/iter-002/model/checkpoint.pt`), same opponent
(rule-bot, no pool), same HPs (`--lr 3e-4 --clip-epsilon 0.3 --entropy-coef 0.01 --ppo-epochs 4`),
**only diff vs Phase L: `--reward-shape-end 0.0` → `1.0`** (constant full-strength shaping, no
linear decay). Wilson lower per iter: iter-0 **0.2787** (WR 31.8%, +0.6pp vs Phase L iter-0
0.2730, promoted) → iter-1 **0.2883** (WR 32.8%, +1.0pp vs iter-0, **promoted — no rejection**
vs Phase M's iter-1 reject under decaying shape) → iter-2 **0.3677** (WR 41.0%, +7.9pp vs
iter-1, promoted). All three iters promoted, monotone, no halts. Iter-2 0.3677 vs Phase L
0.3560: **Δ +0.012**; vs Phase M 0.3580: **Δ +0.010**. **The new F1 rule-bot ceiling on
record across every sweep.** Still 3.2pp short of 0.40 — schedule-axis lift is real but small
(~1.5σ Wilson noise at n=500). Mechanism healthy: ratio_max 11.82 / 7.97 / 32.78 (gradient
strongly active, comparable to Phase L 19.06 / 32.59 / 11.12 and Phase M 15.86 / 12.50 /
18.82); entropy stable 0.173 → 0.160 → 0.155 (no collapse); `numerical_anomalies = 0` across
48 minibatches; approx_kl_max < 0.015 each iter. Reward-hacking check passes: WR tracks Wilson
in lockstep (31.8% → 32.8% → 41.0%); the +7.9pp Wilson lift at iter-2 is mirrored by +8.2pp
WR. Wall-clock **5m 25.4s** (fastest of the L/M/N trio, zero runtime cost from constant shape);
zero LOC diff vs Phase L. Full writeup: `docs/ai-performance-research-progress.md` § "Phase N
— F1 PPO + constant reward shaping".

**Cross-phase summary (L / M / N) — the binding constraint is the signal set.** Three
configurations of the same 5-signal mix have now landed iter-2 at 0.3560 (L, 1.0× linear-decay)
/ 0.3580 (M, 1.75× linear-decay) / 0.3677 (N, 1.0× constant) — a 1.2pp spread across two
single-axis moves (coef magnitude and schedule). PPO is healthy and moving the policy in all
three (ratio_max well off 1.00, KL bounded, entropy stable, no anomalies); the reward-shape
axis encodes strategic information worth ~+5pp Wilson over the unshaped Phase H baseline
(0.3109 → 0.3677); but **scaling magnitude (M) or schedule (N) on the existing 5-signal mix
does not produce the +9pp needed to clear 0.40**. The next move on this branch is
**signal-set change, not further schedule or magnitude tuning**.

**Pre-scoping for the next single-axis move (queued as P3 `r15-s3-value-head-tempo-signal`).**
The highest-leverage candidate from the original three-option scoping is option (3) — add a
strategic-tempo signal computed from the trained value head's per-step delta in own-win-
probability. Concrete design: append a 6th additive signal to the existing 5, computed as
`value(s_{t+1}) - value(s_t)` from the value head output already present in the trajectory
inference stream. Coef scale **~0.05** (similar magnitude to the existing hp-diff signal, the
strongest existing component). Keep the existing 5 signals at Phase L 1.0× coefs. Same
**constant** schedule established as winning by Phase N (`--reward-shape-start 1.0
--reward-shape-end 1.0`). Same Phase H aggressive HPs. Same warm-start. Implementation cost
~**20-40 LOC additive** to `training/ppo_orchestrator.py:parse_trace_to_trajectories` — read
the existing 5-signal computation pattern at ~lines 715-739 and add a 6th component. Pre-
register: success if any iter Wilson ≥ 0.40 (first F1 success on record); partial if iter-2
in (0.37, 0.40) — moved past Phase N ceiling but not enough; saturation if iter-2 ≈ 0.37 (then
the entire 6-signal mix has saturated and the next move is either a different architectural
axis or accepting that the warm-start neighborhood does not contain a 0.40 policy under our
current eval gate); reward-hacking if iter-2 Wilson rises ≥+5pp over iter-0 but WR drops (the
value-head-delta signal can be self-referentially gamed if the policy learns to inflate its
own value estimates — important to quote WR alongside Wilson). Run as new Phase O. Smoke
(`TMPDIR=/tmp npm run test:ppo-smoke`) before launching. Expected ~6 min wall-clock per L/M/N
precedent.

**Phase O' Closeout — R15.S3 reduced-mix / signal-mix axis + R15.S3 BRANCH CLOSEOUT
(2026-05-14).** **CAPSTONE — branch fully saturated; iter-2 Wilson 0.3677 identical to Phase N
to 4 decimal places.** Phase O original (value-head tempo signal) audited at launch time and
found blocked on TS-side ONNX/trace schema instrumentation (filed as P3
`r15-s3-value-head-trace-instrumentation`, ready). Pivoted to scoping option (2): drop the
weakest of the 5 existing signals, scale the strongest 2-3. Audit pulled per-iter
`shape_attribution` from Phase N `events.jsonl` — iter-2 absolute coef-weighted contributions:
throughput 171.4, active-energy 90.4, hp-diff **-6.9** (anti-correlated), bench-energy 4.2,
retreat **0.0** (agent never retreats). Dropped retreat (zero) and bench-energy (smallest
non-zero, redundant with active-energy); scaled active-energy 0.02→0.03 and throughput
0.02→0.03; held hp-diff at 0.05 as control (preserves signal-set parity test integrity rather
than amplifying an anti-correlated signal). Per-game shape budget ~0.24, close to Phase M's
0.23. Run `runs/R15-S3-reduced-mix/` (same warm-start, opponent, HPs, code as Phase L/M/N;
only diff is the coef vector). Wilson lower per iter: iter-0 **0.2825** (WR 32.2%, +1.0pp vs
Phase N iter-0 0.2787, promoted) → iter-1 **0.2883** (WR 32.8%, +0.6pp vs iter-0, promoted —
identical to Phase N iter-1 0.2883 to 4dp) → iter-2 **0.3677** (WR 41.0%, +7.9pp vs iter-1,
promoted — **identical to Phase N iter-2 0.3677 to 4 decimal places**). All three iters
promoted, monotone, no rejections, `run_completed clean, halted=false`. Δ iter-2 vs Phase N:
**+0.0000pp**. Mechanism check (healthy): ratio_max 10.53 / 18.02 / 10.57; entropy stable
(0.171 → 0.161 → 0.153, no collapse); `numerical_anomalies = 0` across all 48 minibatches;
approx_kl_max < 0.016 each iter; WR tracks Wilson in lockstep (32.2% → 32.8% → 41.0%). Phase
O' attribution confirms the drop choices: bench-energy 0.0 / retreat 0.0 across all three
iters (signals correctly silenced); throughput 258.4 / 255.7 / 259.3 (consistent, dominant);
active-energy 126.1 / 130.8 / 132.3 (consistent, secondary); hp-diff ≈ -6.5 (consistent
anti-correlated). Wall-clock **5m 28.2s** (run_started → run_completed delta:
`1778733964.90 → 1778734293.08 = 328.18s`); fastest of the L/M/N/O' quartet alongside Phase N.

**4-axis synthesis — R15.S3 observation-delta reward-shaping branch CLOSED.** Three single-
axis moves have now been tested independently inside the 5-signal observation-delta family,
each from the Phase L baseline (1.0× coefs, linear-decay schedule, all 5 signals active):

| Axis | Configs | Δ iter-2 Wilson | Notes |
| --- | --- | --- | --- |
| Coef magnitude | Phase L 1.0× → Phase M 1.75× linear-decay | **+0.002** | iter-1 rejected at 1.75×; iter-2 within Wilson noise of Phase L |
| Shaping schedule | Phase L decay → Phase N constant | **+0.012** | constant > decay by ~1pp; new ceiling 0.3677 at Phase N |
| Signal mix | Phase N 5-signal → Phase O' 3-signal (drop bench-energy + retreat) | **+0.000** | iter-2 identical to Phase N to 4dp |

All four phases have PPO healthy in all configs: importance ratios off 1.0, gradient
decisively active, WR tracks Wilson in lockstep, no reward hacking, no numerical anomalies,
entropy stable, approx_kl bounded. The 5-signal hand-engineered observation-delta family
encodes ~+5pp Wilson over the unshaped Phase H baseline (0.3109 → 0.3677), but **all three
axes inside that family are now exhausted** — iter-2 caps at **0.368 ± 0.001** regardless of
coefs, schedule, or which subset of the 5 signals is active. The binding constraint is the
*information content* of the signal set, not the signals' computation, weights, or schedule.
Total R15.S3 branch compute cost ≈ **23 min wall-clock across 4 phases** (Phase L 5m 54s +
Phase M 6m 22s + Phase N 5m 25s + Phase O' 5m 28s) — cheap research, decisive answer.

**F1 post-mortem framing update.** Phase L closeout qualified the original "0.40 NOT
REACHABLE" claim with "the reward axis demonstrably changed the mechanism; 0.40 is now
reachable in principle." Phase M / N narrowed that to "the gap is quantitative — coef + signal
mix." **Phase O' closes the qualification cleanly**: the reward-shape axis has been **fully
explored** and converged at 0.368. The remaining gap to 0.40 is now **categorical** — needs a
different *information source*, not more tuning of the existing observation-delta signals.

**Surviving F1-line candidates (research-stance decision, escalated to human).** The only ways
to break the 0.368 cap from here require pulling in information from a *different source*:

1. **Value-head tempo signal** — per-step delta of the policy's own value estimate as a
   strategic-tempo reward signal. Highest-leverage of the three because it adds *learned*
   strategic information rather than hand-engineered observations. Blocked on TS-side ONNX
   call-site instrumentation: rollout currently emits placeholder `value_pred=0.0`
   (`ppo_orchestrator.py:858`), PPO recomputes value at training time, the value head is not
   a named output of `rollout.onnx`/`policy.gate.onnx`. Implementation requires ONNX export
   audit + TS-side inference call-site instrumentation (search `backend/src/sim/` for the
   policy-server client) + decision-trace schema extension (likely
   `behaviorPolicy.valueEstimate: number` or top-level `valuePred`) +
   `relabelDecisionTrace.ts` forward. Multi-file change crossing TS/Python; ~separate sprint
   slot. Queued as `r15-s3-value-head-trace-instrumentation` ready.

2. **R7 multi-teacher labels** — change the SL warm-start by retraining DAgger with multiple
   teachers (e.g. rollout-teacher mixed with rule-bot policy labels, or with value-head leaf
   MCTS labels). PPO inherits a different starting policy. Does not touch the reward axis.
   Larger code surface (SL pipeline rebuild). Pre-scoped in the R7 backlog entry.

3. **R8 DPO** — replace PPO with Direct Preference Optimization, an objective that doesn't
   depend on hand-shaped reward signal. Bigger pivot. Pre-scoped in the R8 backlog entry.

**Recommendation:** human picks among (1)/(2)/(3) before further autonomous F1-line work. The
autonomous loop should NOT pick one of these on its own — each is a meaningfully different
research direction (sim-side instrumentation vs SL rebuild vs PPO replacement) and the
ordering depends on which axis the researcher believes is most likely to crack the 0.40 bar.
Escalation filed in `docs/ai-agent-state/escalations.md` `## Open`; queue item
`r15-s3-branch-synthesis-and-next-pick` P2 ready (autonomous-launch ineligible). A
non-per-step reward-shaping framework (turn-based or game-phase aggregate shaping) is a
**fourth, untested** possibility — call it out but rank below the three above.

**Phase O Closeout — value-head tempo signal at coef 0.05, REGRESSED (2026-05-14).** After
the queued P3 `r15-s3-value-head-trace-instrumentation` landed (~7 LOC TS + ~30 LOC
orchestrator, both smokes green), the original Phase O plan was unblocked and executed.
Run `runs/R15-S3-value-head-tempo/` added a 6th additive signal — per-step delta of the
policy's own value-head Tanh output — at `--reward-value-head-coef 0.05`, holding all other
parameters at Phase N's setup. Result: iter-0 Wilson **0.2845** (WR 32.4%, promoted) →
iter-1 **0.2768** (WR 31.6%, **REJECTED** at floor 0.2845, same regression pattern as Phase
M iter-1 at 1.75× coefs) → iter-2 **0.3502** (WR 39.2%, promoted, rolled from iter-0 parent).
`promoted_iterations: [0, 2]`. **Δ iter-2 vs Phase N: -0.018**. Adding the value-head signal
at coef 0.05 made iter-2 *worse* than the unshaped Phase N baseline. Wall-clock 5m 22.0s.

**Mechanism diagnosis: signal is correctly wired but magnitude is ~20× the design budget.**
Value head Tanh output range [−1, +1] gives per-step delta range [−2, +2]; at coef 0.05 the
per-step shape contribution is bounded by ±0.1; at the F1 episode length of ~60 steps/game
the per-game contribution is therefore ±6.0 — **20× the ±0.3/game budget the R15.S3 scoping
doc set**. iter-0 `shape_attribution.value_head = 26.01` confirms the magnitude empirically
dominates: it is third-largest in absolute terms behind throughput (171.4 — accumulates
positively across all steps so its absolute magnitude is mostly positive bias) and
active-energy (85.1). For a mean-zero (Tanh delta) signal, an absolute attribution of 26
represents genuine per-step *variance* contributed to the surrogate gradient — larger than
every observation-delta signal except the two energy-related ones. iter-1 / iter-2
attributions drop to 9.28 / 8.25 as iter-0's promoted policy learns to flatten the
value-head delta, the classic over-shaping signature. iter-1 rejection mirrors Phase M:
over-shaped iter-0 promotes a policy that exploits the shape; iter-1 trains against more of
the same shape and overfits to it; greedy match-vs-rule-bot win rate falls below the
tolerance=0 floor; iter-2 rolls back to iter-0 as parent and re-trains one more pass,
clawing back ~6pp from the iter-1 trough but not recovering the ground Phase N covered
without over-shaping.

**PPO healthy in non-reward dimensions.** ratio_max 12.7 / 12.3 / 22.4 (gradient strongly
active, comparable to Phase L/M/N); approx_kl_mean 0.014 / 0.011 / 0.012 (no anomalies);
entropy 0.167 → 0.159 → 0.158 (stable, no collapse); `numerical_anomalies = 0` across all
48 minibatches. PPO is fine; the signal at this coefficient is wrong.

**Phase P scoping — coef 0.01 (5× smaller).** The signal isn't *wrong*, it's *too loud*.
At `--reward-value-head-coef 0.01`, per-step contribution becomes ±0.02; at ~60 steps/game,
per-game total ±1.2 — still 4× above the ±0.3 budget but ~5× closer than Phase O.
**Single change** vs Phase O. All else identical: same warm-start
(`runs/item17-2026-05-11/iter-002/model/checkpoint.pt`), same opponent (rule-bot, no pool),
same Phase H HPs (`--lr 3e-4 --clip-epsilon 0.3 --entropy-coef 0.01 --ppo-epochs 4`), same
5 obs-delta coefs at Phase L 1.0× (active 0.02 / bench 0.02 / retreat 0.03 / throughput
0.02 / hp-diff 0.05), same constant schedule (`--reward-shape-start 1.0
--reward-shape-end 1.0`), same `--reward-alpha 0.333 --reward-beta 1.0 --eval-games 250
--iterations 3 --games-per-update 800`. Output dir: `runs/R15-S3-value-head-tempo-low/`.
**Exit gate.** Success if any iter Wilson lower ≥0.40 (first F1 success on record). Partial
improvement if iter-2 in (0.368, 0.40) — value-head signal contributes additively at low
magnitude. Neutral if iter-2 ≈ 0.36-0.37 — signal is redundant with the observation-delta
family at low magnitude. Regression if iter-2 < 0.35 — value-head info quality (not
magnitude) is genuinely the problem; pivot to R7/R8.

**Phase P Closeout — value-head tempo signal at coef 0.01, REGRESSED (2026-05-14).** Run
`runs/R15-S3-value-head-tempo-low/` finished clean (`run_completed`, `halted=false`,
**5m 23.2s** wall-clock; run_started ts 1778735914.37 → run_completed 1778736237.61, delta
323.24s). Trajectory: iter-0 Wilson **0.2730** (WR 31.2%, promoted — **identical to Phase
L iter-0 0.2730 to 4dp**; at coef 0.01 the value-head signal contributes effectively
nothing at iter-0) → iter-1 **0.2845** (WR 32.4%, promoted, +1.1pp; **no rejection**,
unlike Phase O which rejected iter-1 at 0.2768 < 0.2845 floor) → iter-2 **0.3463** (WR
38.8%, promoted, +6.2pp from iter-1). `promoted_iterations: [0, 1, 2]`, halted=false. **Δ
iter-2 vs Phase N: -0.021; Δ iter-2 vs Phase O: -0.004**. Adding the value-head signal
at coef 0.01 made iter-2 *worse* than Phase N's unshaped-by-value-head baseline AND
slightly worse than Phase O at coef 0.05. **Lowering the coef did not help — the value-
head-delta signal mechanism is wrong-shape, not wrong-magnitude.**

**Dimensional check on the coef ratio.** iter-0 `shape_attribution.value_head = 5.03`
(Phase P) vs `26.01` (Phase O); ratio **5.17×** matches the 5× coef ratio within rounding
— the value-head signal is wired and scaled correctly, and iter-0 shape attribution moves
linearly with the coef. iter-1 / iter-2 attributions are 1.45 / 1.27 (Phase P) vs 9.28 /
8.25 (Phase O); same policy-flattens-the-delta pattern at both magnitudes, but at Phase
P's absolute level the flattening contributes neither helpful gradient (Phase O over-
shape signature) nor harm (Phase O iter-1 rejection). PPO healthy in non-reward
dimensions: ratio_max 12.53 / 13.29 / 21.63 (gradient active, comparable to L/M/N/O');
approx_kl_mean 0.013 / 0.012 / 0.009 (no anomalies); entropy 0.173 → 0.164 → 0.164
(stable); `numerical_anomalies = 0` across all 48 minibatches.

**Why coef 0.01 didn't help and slightly hurt.** At coef 0.05 the value-head signal was
*loud and over-shaped* the policy (Phase O iter-1 rejection signature; iter-0 over-shaped
policy that exploits the shape, iter-1 overfits, greedy WR falls below floor). At coef
0.01 the signal is *quiet but noisy* and contributes random variance to the surrogate
gradient without informational gain (Phase P clean trajectory but lower ceiling than
Phase O). Lower coef means less "averaging-out" of the noisy delta during PPO updates,
so noise dominates more relative to signal. Both magnitudes regressed vs Phase N; the
data point is the *mechanism*, not the coefficient.

**Final R15.S3 branch summary — both axes exhausted (2026-05-14).** This is the
authoritative final synthesis. The R15.S3 reward-shaping branch is now **fully explored
and closed across both signal axes**. The prior `87e9e77` "BRANCH CLOSED" framing was
premature — it tested only axis 1 (observation-delta) and queued axis 2 (value-head
tempo) as the next move. Phases O + P then executed axis 2 and both regressed. Six
sweeps total:

| Phase | Signal axis | Config | iter-2 Wilson | Δ vs Phase N | Decision |
| --- | --- | --- | --- | --- | --- |
| L | obs-delta | 5 signals, 1.0× coefs, linear-decay | 0.3560 | -0.012 | promoted |
| M | obs-delta | 5 signals, 1.75× coefs, linear-decay | 0.3580 | -0.010 | promoted |
| N | obs-delta | 5 signals, 1.0× coefs, constant | **0.3677** | — | promoted (F1 rule-bot ceiling on record) |
| O' | obs-delta | 3 signals (scaled), constant | 0.3677 | +0.000 | promoted |
| O | value-head | + value-head coef 0.05, constant | 0.3502 | -0.018 | promoted (iter-1 rejected) |
| P | value-head | + value-head coef 0.01, constant | **0.3463** | **-0.021** | promoted (no rejections) |

**Axis 1 — hand-engineered observation-delta signals.** 4 phases. Capped at iter-2 Wilson
**0.368 ± 0.001**. Coef magnitude (L vs M, Δ +0.002), schedule (L vs N, Δ +0.012), signal
mix (N vs O', Δ +0.000) — all saturated.

**Axis 2 — per-step value-head-delta tempo signal.** 2 phases spanning 5× coef range.
Both regressed. Phase O at coef 0.05 → 0.3502 (loud, over-shaped, iter-1 rejection).
Phase P at coef 0.01 → 0.3463 (quiet, noisy, lower ceiling). Lowering the coef did not
help — the signal mechanism is wrong-shape, not wrong-magnitude.

**Total cost.** ~36 min wall-clock across 6 phases (L 5m54s + M 6m22s + N 5m25s + O'
5m28s + O 5m22s + P 5m23s). All six PPO-healthy in non-reward dimensions. Cheap research,
decisive answer on both axes.

**F1 post-mortem framing — final update.** Prior framing (`87e9e77`): "reward-shape axis
fully explored and converged at 0.368; gap to 0.40 is categorical — needs a different
information source, not more tuning." Phases O + P tested *exactly* that — a different
information source (value-head delta) — and that information source actively regressed at
both magnitudes. **New framing: "The F1 reward-shape mechanism cannot break 0.368 from
this warm-start. Per-step shaping from any observation-derived signal saturates at 0.368,
and per-step shaping from the policy's own value-head delta actively regresses. The
remaining F1 moves must change either the warm-start (R7 multi-teacher labels rebuild)
or the optimization objective (R8 DPO replacement). The reward-shape branch is closed."**

**What survives.** Two F1-line candidates remain, both human-rank (not autonomous-launch):
(a) **R7 multi-teacher labels** — retrain DAgger SL warm-start with multiple expert
teachers; doesn't touch reward, changes SL pipeline. (b) **R8 DPO** — replace PPO with
Direct Preference Optimization; doesn't depend on per-step hand-shaped reward signal at
all; bigger pivot. The previously-queued "value-head tempo signal" path (a-prime) has
been executed and exhausted at Phases O + P; it is no longer a surviving candidate. A
non-per-step reward-shaping framework (turn-based or game-phase aggregate shaping)
remains a fourth, untested possibility — call it out but rank below R7/R8. Escalation
re-opened at `docs/ai-agent-state/escalations.md` `## Open`; queue item
`r15-s3-branch-synthesis-and-next-pick` P2 ready autonomous-launch ineligible.

## F1 reward shaping value-head tempo — TS instrumentation scoping (2026-05-14)

**Scope at a glance: is the value-head trace instrumentation orchestrator-only? No — but it
is NOT a "TS+ONNX+schema half-day change" either.** It is a **2-file TS-only change** (no
ONNX-export change, no Python-side schema change required). The prior worker's
"deeper than a parsing-time Python diff" framing was correct, but the depth is smaller than
they implied: the ONNX export already names a `value` output, the `/predict` server already
returns it, and the TS client just drops it on the floor.

**ONNX export (already correct).** `training/export_onnx.py:46` passes
`output_names=["logits", "value"]` and `dynamic_axes={..., "value": {0: "batch"}}` (line 52)
to `torch.onnx.export`. The model class `CandidatePolicyNet` returns
`(logits, value)` from `forward()` (`training/uma_ai/model.py:81-95`), with the value head
defined at `model.py:73-79` (LayerNorm → Linear → GELU → Linear(1) → Tanh, range [-1, +1]).
**No ONNX export change required.**

**ONNX server side (already correct).** `training/serve_onnx.py:95` unpacks
`logits, value = self.server.session.run(None, arrays)` and the `/predict` response includes
`"value": value.tolist()` at line 146. **No `serve_onnx.py` change required.**

**TS-side `/predict` client (instrumentation needed here).**
`backend/src/sim/evaluateModelVsHeuristic.ts:504` is the per-decision `/predict` call inside
`chooseModelAction`. The current return-payload type at lines **513-519** destructures
`{selectedIndex, actionLogProbs, actionProbs, selectedLogProb, behaviorPolicy}` and **omits
`value`** — the field returns from Python but is discarded. To capture it: add
`value?: number[]` to the payload type (line 519 +1), and in the snapshot construction at
**lines 527-536** copy `payload.value?.[0]` onto the snapshot (+1 line, with a defensive
typeof-number check). The single-action shortcut at **lines 494-502** does NOT call
`/predict` (it short-circuits before fetch), so on those rows the field is absent — that is
correct behavior, mirroring how `actionLogProbs` etc. are absent there today.

**Schema types (two declarations, must stay in lockstep).** The `BehaviorPolicySnapshot`
type is duplicated:
- `backend/src/sim/evaluateModelVsHeuristic.ts:140-146` (5 fields, +1 for `valueEstimate?: number`)
- `backend/src/sim/dagger/relabelDecisionTrace.ts:5-11` (5 fields, +1 for `valueEstimate?: number`)

`relabelDecisionTrace.ts:109` already forwards the entire `row.behaviorPolicy` object through
to the training row unchanged, so **no relabeler logic change is required** — only the type
declaration needs the new field for type-safety.

**Trace writer (no logic change).** The trace row is constructed at
`backend/src/sim/evaluateModelVsHeuristic.ts:308-329`. `behaviorPolicy` is populated at line
**328** via `trace.behaviorPolicy = behavior` from `decision.behavior`. Because we attach
`valueEstimate` to the `BehaviorPolicySnapshot` object inside `chooseModelAction` (not as a
top-level trace field), this site needs **zero new lines**. Design choice rationale: nesting
under `behaviorPolicy` matches the existing convention that policy-server-derived scalars
live on the snapshot (kind/temperature/actionLogProbs/selectedLogProb), and it inherits the
existing "only present for policy-source rows" semantics so single-action and
rule/heuristic/rollout/planner selections cleanly omit it (matching the test asserts at
`backend/src/tests/evalGateSmoke.ts:182-203` which check presence/absence by selection
source).

**Backward compat — readers.**
- Python parser `training/ppo_orchestrator.py:844` does
  `behavior = row.get("behaviorPolicy") or {}` and only reads `behavior.get("selectedLogProb")`
  (line 849). Unknown keys on the snapshot are silently ignored. **No Python-side change
  required for backward compat;** the value-head tempo signal consumer will be a separate diff
  inside `parse_trace_to_trajectories` that does `behavior.get("valueEstimate")` (the actual
  use of the new field — that's the unblocked R15.S3 Phase P work, NOT part of this
  instrumentation slot).
- TS readers: `relabelDecisionTrace.ts:109` does object-pass-through; no shape check.
  `evalGateSmoke.ts:182-213` does explicit `assert.equal(row.behaviorPolicy, undefined, ...)` /
  `assert.ok(row.behaviorPolicy, ...)` checks but never asserts the snapshot shape is exactly
  5 fields — adding a 6th optional field is safe.
- `daggerRoundSmoke.ts:65-86` plants a `behaviorPolicy` and asserts forwarded-through; the
  planted object has no `valueEstimate` and the test won't break (the new field is optional).

**Estimated per-file diff.**
- `training/export_onnx.py`: **0 LOC** (already exports value).
- `training/serve_onnx.py`: **0 LOC** (already returns value in /predict response).
- `backend/src/sim/evaluateModelVsHeuristic.ts`:
  - `BehaviorPolicySnapshot` type at line 140-146: **+1 LOC** (`valueEstimate?: number;`).
  - `/predict` payload type at line 513-519: **+1 LOC** (`value?: number[];`).
  - `chooseModelAction` snapshot construction at line 527-536: **+1-2 LOC** (assign
    `payload.value?.[0]` with a `typeof === "number"` guard).
- `backend/src/sim/dagger/relabelDecisionTrace.ts`:
  - `BehaviorPolicySnapshot` type at line 5-11: **+1 LOC** (`valueEstimate?: number;`).
- Smoke-test updates: **0 LOC strictly required** but recommended **+1-2 LOC** in
  `backend/src/tests/evalGateSmoke.ts` policy-source block (~line 201-213) to assert
  `typeof row.behaviorPolicy.valueEstimate === "number"` when `kind ∈ {greedy, stochastic}` —
  catches future regressions where the field gets dropped.

**Grand total: ~4-6 LOC across 2 production files (`evaluateModelVsHeuristic.ts`,
`relabelDecisionTrace.ts`), optionally +1-2 LOC in 1 smoke test.** Human time estimate:
**~30-45 min** including a `TMPDIR=/tmp npm run test:train` + `head -1 .../trace.jsonl | jq
.behaviorPolicy` smoke check. **This is closer to the "30-min Python diff equivalent" than
to a "half-day TS+ONNX+schema change"** because the export and server are already correct.

**Validation strategy.**
1. `TMPDIR=/tmp npm run build` — TS type-check confirms both schema declarations stay in
   sync (production code uses both).
2. `TMPDIR=/tmp npm run test:train` — full ts/test suite catches any
   behaviorPolicy assertion regression.
3. Single-game smoke: `node --experimental-strip-types backend/src/sim/evaluateModelVsHeuristic.ts
   --games 1 --selection policy --decision-trace-out /tmp/smoke-trace.jsonl
   --model-url http://localhost:8000 ...` (needs a running serve_onnx); then
   `jq -c 'select(.selection=="policy") | .behaviorPolicy.valueEstimate'
   /tmp/smoke-trace.jsonl | head -5` — expect numbers in [-1, +1] (Tanh output range).
4. Trace size regression check: compare bytes-per-row on a 100-row trace before/after; expect
   <1% inflation (a single float adds ~25 bytes to ~3 KB rows).

**Risks.**
- **(low) value-head output index ordering.** `serve_onnx.py:95` assumes
  `(logits, value)` tuple order matching `output_names=["logits", "value"]`. If a future
  re-export reorders outputs (e.g. someone changes to `["value", "logits"]`), the unpack
  silently swaps. Mitigation: `serve_onnx.py:95` could be tightened with a named-output
  lookup, but that is outside this slot's scope. Today's behavior matches.
- **(low) trace-file bloat.** 232 MB × ~1% = ~2.3 MB additional per R15-S3 trace. Negligible.
- **(medium) gate vs collector value semantics drift.** The same value scalar will be logged
  under both greedy gate-eval and stochastic PPO collector modes. Phase P (the eventual
  consumer) needs to decide whether to use the value Δ as a shaping signal only on collector
  rows or on both; this is a Phase P design decision, not a blocker for the instrumentation
  slot.
- **(low) duplicate type-decl drift.** `BehaviorPolicySnapshot` exists in two TS files; the
  build catches mismatched usages at compile time, but the *type itself* can drift if a future
  edit only touches one. Mitigation deferred: a shared types file in `shared/src/` is the
  right answer but is out of scope here. Note left for a future refactor slot.

**Bottom line.** This is a **2-file, ~5-LOC TS-only schema-extension change** taking
**~30-45 min** including smoke validation. The ONNX export and `/predict` server are already
emitting the value scalar; the TS client just needs to start carrying it through to the
trace row. Comparable cost to a small `parse_trace_to_trajectories` parameter addition.
