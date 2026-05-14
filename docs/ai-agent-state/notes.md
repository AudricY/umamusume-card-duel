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
