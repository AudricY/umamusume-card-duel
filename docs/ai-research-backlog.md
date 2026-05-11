# AI Research Backlog

Created 2026-05-11. This is the *research-and-exploration* backlog — distinct
from the infrastructure backlog under `docs/ai-performance-research-backlog.md`
which is now drained. The infra (DAgger orchestrator, F1 PPO, observability,
HP sweep, opponent pool, calibrated-value gate) is built and validated; the
remaining work is experimental science.

## Current state (anchor for every question below)

1. **DAgger SL warm-start ceiling: Wilson lower 0.31 (WR ~37.5%) vs rule-bot.** Reproducible across iterations, seeds, and HP variants. The trained policy entropy is ~0.18 nats per decision (near-deterministic).
2. **F1 PPO cannot exceed the warm-start.** 5 sweeps spanning 30 to 800 games-per-update, spec defaults to "extreme" HPs. Every well-behaved config matches the DAgger ceiling at 0.31; aggressive configs regress.
3. **Mechanism confirmed: importance ratios stay at ~1.00** because the stochastic policy ≈ greedy policy on the peaked warm-start.
4. **Value head is systematically optimistic.** Verified via PPO events.jsonl: `mean_advantage` is negative across all 5 F1 sweeps (-0.03 to -0.15). The value head predicts returns ~0.05-0.15 higher than actually achieved. `calibrate_value` smoke confirms `calibration_status=FAIL` (lift over point-margin baseline is negative).
5. **Wall-clock is cheap.** 800-game PPO iteration runs in ~110 s; 200-game gate eval in ~60 s. Each experiment is minutes, not hours.

## Research north star

Move *some honest measurement* of model strength past the DAgger 0.31 Wilson-lower ceiling. The exact mechanism doesn't matter — better warm-start, better RL, better reward, larger model — but we don't yet know which one matters because we haven't separated the contributing causes.

The three load-bearing questions:

- **Q1. Is the cap a teacher-strength cap?** If rollout-CRN×3 itself only achieves WR 35–40% vs rule-bot, the trained policy is already near its target's ceiling and we need a stronger teacher (search, MCTS, or improved rule-bot replacement).
- **Q2. Is the cap a peakedness cap?** If a deliberately higher-entropy warm-start enables PPO to move, then the F1 plumbing is fine and we just need the right starting point.
- **Q3. Is the cap a value-head cap?** Confirmed value head is miscalibrated. Does fixing it unblock either PPO advantage signal or one-step Q-selection?

The Tier-1 experiments are designed to discriminate between these three in <2 hours total.

---

## Tier 1 — discriminating experiments (trivial cost, high information)

### R1. Perfect-oracle WR ceiling
- **Q:** What's the win rate of rollout-CRN×3 *itself* (skipping the policy entirely) vs rule-bot at n=200?
- **Hypothesis:** if it's ≤45%, the SL cap is the teacher's cap and Q1 is the binding constraint.
- **Design:** `sim:eval-gate --selection rollout --rollout-crn-samples 3 --games 100`. No code changes.
- **Cost:** ~3 min wall-clock.
- **Decides:** Q1.

### R2. Multi-temperature gate matrix
- **Q:** What's the gate WR of the current promoted F1 checkpoints at temperature T ∈ {0 (greedy), 0.3, 0.5, 1.0, 1.5, 2.0}?
- **Hypothesis:** If PPO improved the *distribution* but not the argmax, non-greedy temperatures will reveal latent improvement that the greedy gate hides.
- **Design:** New CLI flag on the gate path (or pass `sampling=stochastic&temperature=T` through the existing serve_onnx body). Run against the phaseH iter-2 checkpoint and DAgger iter-2 warm-start.
- **Cost:** ~30 min (add the flag) + 5 min × 6 temperatures × 2 checkpoints.
- **Decides:** whether all five F1 phases were measuring the wrong thing.

### R3. Entropy-regularized BC warm-start
- **Q:** If we add `−β · H(π)` to the BC loss with β ∈ {0.01, 0.05, 0.1}, does the resulting warm-start have entropy ≥ 0.5 nats and let PPO actually move?
- **Hypothesis:** if yes, the F1 cap is the peakedness cap (Q2 is binding).
- **Design:** Add `--entropy-bonus β` to `train_bc.py`'s policy loss. Retrain DAgger iter-0 only. Eval gate at the same n=200 to keep comparable; then feed into PPO phase-H config and observe.
- **Cost:** ~15 min code + ~15 min training × 3 β values + PPO sanity.
- **Decides:** Q2.

### R4. Value-head ablation: fresh re-train with longer epochs + value-loss tuning
- **Q:** Can we make the value head non-miscalibrated by training longer / re-weighting value loss? Or is the architecture itself the issue?
- **Hypothesis:** if calibrate-value's lift can be flipped positive by training adjustments alone, then the value head is fixable and Q3 becomes actionable.
- **Design:** Run `train_bc.py --epochs 50 --value-weight 1.0` (currently 25 epochs, weight 0.1) against the existing item17 mixed dataset. Then re-run `calibrate_value.py` on the resulting checkpoint.
- **Cost:** ~10 min training + 1 min calibration.
- **Decides:** Q3 (first-pass).

---

## Tier 2 — directional experiments (small effort, moderate signal)

### R5. PPO self-play via existing PFSP pool
- **Q:** Does opponent diversity change the gradient distribution enough to break the cap?
- **Hypothesis:** Single-opponent (rule-bot) reward shape gives flat advantages because the policy has converged to it; PFSP-sampled prior promoted checkpoints expose different state distributions.
- **Design:** `ppo_orchestrator.py` already supports the pool eval channel. Add `--rollout-vs-pool` flag that passes `--opponent-model-url` to the trace evaluator, sampling per game by PFSP weights. Reuses item-12 infrastructure.
- **Cost:** ~1 h code + ~10 min run.

### R6. Larger model capacity
- **Q:** Is the 0.31 ceiling a representation-capacity cap?
- **Hypothesis:** if hidden_dim=128 (4× current 64) lifts DAgger Wilson lower, then we've been compute-starved.
- **Design:** `dagger_orchestrator.py --hidden-dim 128 --depth 3` (currently 64/2). Same iterations, same data.
- **Cost:** ~40 min wall-clock (DAgger iter wall-clock scales weakly with model size; training is fast).

### R7. Multi-teacher BC blend
- **Q:** Does mixing rollout + planner + search teacher labels per state produce a warm-start with higher entropy and broader competence?
- **Hypothesis:** Teacher diversity prevents single-teacher mode collapse; the resulting policy should be less peaked.
- **Design:** Extend `relabel_decision_trace` to attach multiple `labelSource` rows per trace row (rollout, planner, search). Train on the mixed corpus; use a label-smoothing or top-K objective so the policy targets a mixture rather than an argmax.
- **Cost:** ~2 h code + ~30 min training. Reuses item-5 trace-teacher infrastructure.

### R8. Direct preference optimization (DPO) on top of warm-start
- **Q:** Given rollout-CRN gives us (state, top-1-action, runner-up-action) tuples with score margins, can DPO push the warm-start past the cap on pairs alone?
- **Hypothesis:** Dense pairwise preference signal is more learnable than sparse ±1 terminal reward.
- **Design:** New trainer `train_dpo.py` reading the outcome-export's per-candidate reward records (already include selected vs runner-up margin). Bradley-Terry loss against the warm-start as the reference policy.
- **Cost:** ~3 h code + ~10 min training.

---

## Tier 3 — structural changes (medium-large effort, asymmetric upside)

### R9. Q-learning head as the policy
- **Q:** Bypass the policy entirely: train Q(s, a) on Bellman targets from rollout-CRN values, act argmax-Q at gate time. Does it beat 0.31?
- **Design:** Repurpose the existing per-action features pipeline. Q(s, a) = MLP(state_embed ⊕ action_embed). Targets from rollout-CRN at each (s, a) pair.
- **Cost:** ~1 day implementation. Existing infra (features, ONNX export, serve) reusable.

### R10. Larger DAgger sweep at fixed-recipe
- **Q:** Does the SL ceiling lift if we run DAgger to convergence at production scale (200 games × 50 epochs × 8 iterations)?
- **Hypothesis:** Item-17 was compute-scaled; full-scale might break 0.40 even before considering F1.
- **Cost:** Compute-bound (~few hours wall-clock).

### R11. Auxiliary self-supervised objectives in the trunk
- **Q:** Adding next-state prediction and opponent-action prediction as auxiliary heads to the shared trunk — does it improve representation quality enough to lift WR?
- **Cost:** ~1-2 days.

### R12. MCTS-augmented self-play (mini-AlphaZero)
- **Q:** Pure search + policy refinement, skip the warm-start + RL split entirely. AlphaZero's signature regime; the game is small enough that this is plausible.
- **Cost:** ~1 week to scaffold properly. High asymmetric upside.

---

## Pre-registered next batch (Tier 1 in order)

1. **R1** (perfect-oracle ceiling) — first, because it decides whether everything else is even targeting the right problem. If rollout-CRN×3 only hits 45% WR, we need a stronger teacher before more RL.
2. **R2** (multi-temperature gate) — second, because it tells us whether the existing F1 sweeps already produced value the gate didn't see. Re-interprets phase 2-H without new runs.
3. **R3** (entropy-regularized BC) — third, the highest-leverage *intervention* if R2 still shows greedy-cap. Surgically attacks the diagnosed mechanism.
4. **R4** (value-head re-train) — fourth, in parallel with R3 wall-clock. Resolves whether calibration is fixable in-place.

After Tier 1 completes, the picture should be clear enough to commit to Tier 2 or escalate to Tier 3.

## Tier 1 results (2026-05-11)

### R1 — Q1 REFUTED

rollout-CRN×3 vs rule-bot at n=200 (100 games × 2 sides): **WR 58.5%, Wilson95 [0.516, 0.651].**

The teacher comfortably beats the rule-bot. The trained-policy ceiling at 37.5% WR has a **21pp imitation gap** to its own teacher. The cap is not a teacher-strength cap. Whatever the binding constraint is, it lives between the rollout-CRN labels and the trained policy.

This refocuses the research: stop hunting for stronger teachers (search, MCTS) and start asking why the SL fit doesn't capture what the teacher knows. Candidates:

- **Feature representation gap.** State encoding may lose strategic information the rollout uses.
- **Label quality at rare states.** Rollout teacher is strong on common states but bad on rare ones; SL averages.
- **Policy capacity.** 64-hidden × depth-2 may underfit a 58.5%-strength teacher.
- **Mix bias.** Rule-bot replay rows in the mixed corpus may dilute the rollout-labeled rows.

### R4 — Q3 partially resolved

R4 (50 epochs, value_weight 1.0, fresh from iter-2 mixed data): train accuracy **94.7%** (up from 88%), `calibrate_value` PASS with value_brier 0.084 (down from 0.243), lift_mean +0.134 (Wilson lower +0.119 — significant).

**Value-head miscalibration is fixable** by training longer at higher weight. But the same checkpoint shows gate WR **34.5%** (Wilson [0.283, 0.413]) — slightly below the 37.5% warm-start, with a stark side imbalance (26% as player, 43% as opponent).

Calibration alone doesn't move the gate. The fixed value head matters only if it's used as a critic (PPO advantage signal). Open question: does PPO from R4 produce coherent gradient direction now that the advantages aren't systematically biased?

### R3 — entropy intervention works, PPO test pending

R3 (β=0.05): final entropy **0.527 nats** per decision (3× the warm-start's ~0.18), train accuracy 82.2% (down ~3pp from warm-start), gate WR **38.5%, Wilson [0.320, 0.454]** — statistically equivalent to the warm-start.

**Peakedness intervention succeeds without hurting gate WR.** PPO from this warm-start is running; the critical question is whether the higher entropy lets PPO's importance ratios stay non-trivial and produce coherent gradient direction.

### Side imbalance — opportunistic finding

Across DAgger iter-2 warm-start, R3, R4 gate evals: the trained policy consistently performs much better as the *opponent* than as the *player* (DAgger 41% vs 30%; R4 43% vs 26%; R3 not yet broken out). The model is offensively weak. This may be its own research question (does training set under-represent player-side decisions? feature parity issue?) or a manifestation of the same imitation gap.

R-WILD's side-imbalance subitem should be promoted to its own Tier-1-adjacent task.

## Open / wild

- **Side-imbalance verification.** Gate manifests record player/opponent splits inconsistently across phases. Worth a one-off script to extract the side-WR delta and check whether the model is offensively weak or defensively weak.
- **Simulator determinism audit.** Replay 100 identical seeds end-to-end; measure full-state divergence rate. Silent non-determinism in CRN would invalidate every advantage estimate.
- **Rule-bot mistake catalog.** The 80% aspirational target requires exploiting rule-bot weaknesses; we don't have a catalog of those weaknesses. Hand-construct ~50 states + a careful audit.
