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

### R3+PPO and R4+PPO — interventions don't break the gate cap

PPO from R3 (entropy-regularized warm-start) and PPO from R4 (calibrated-value warm-start) both land at the same ~0.31 Wilson lower ceiling that every prior F1 phase hit:

| Run | Warm-start | Final Wilson lower | Best WR |
| --- | --- | --- | --- |
| F1 phase 2 (defaults at 30g) | DAgger iter-2 | 0.3109 | 37.5% |
| F1 phase G (spec defaults at 800g) | DAgger iter-2 | 0.3156 | 38.0% |
| F1 phase H (aggressive HPs at 800g) | DAgger iter-2 | 0.3109 | 37.5% |
| R3+PPO (high entropy + aggressive HPs) | R3 entropy-reg | 0.3156 | ~38% |
| R4+PPO (calibrated value + aggressive HPs) | R4 retrain | 0.3014 | ~36% |

R4+PPO's GAE `mean_advantage` improved from phase-H's −0.10 to −0.03 (calibration *did* help the value baseline). Per-minibatch KL rose from 0.008 to 0.022 — PPO made slightly bigger updates. **But none of it moved the gate WR.**

### Diagnostic finding: the peakedness diagnosis was wrong

The "warm-start entropy is 0.18 nats" claim from F1 phase F was based on `train_ppo`'s per-minibatch entropy metric, which apparently reports something different than the policy entropy on the training distribution. Directly measuring `masked_policy_entropy(model_logits, action_mask)` over `iter-002/mixed.jsonl` shows:

| Checkpoint | Policy entropy on mixed.jsonl |
| --- | --- |
| DAgger iter-2 (warm-start) | **0.532** |
| R3-b005 (β=0.05 entropy bonus) | 0.528 |
| R3-b020 (β=0.2 + value_weight=1.0 + 50ep) | 0.386 |
| R3+PPO iter-2 | 0.574 |
| R4 (value re-train) | (not measured but accuracy 94.7% suggests low) |

So the warm-start's actual entropy is comparable to R3's — the "peakedness mechanism" we built R3 around was a misdiagnosis. R3+PPO didn't help because the warm-start wasn't actually peaked.

### Diagnostic finding: imitation accuracy ≠ play strength

argmax-match rate against rollout-CRN teacher labels on `iter-002/relabeled.jsonl`:

| Checkpoint | argmax-match | Gate WR |
| --- | --- | --- |
| DAgger iter-2 (warm-start) | 74.7% | 37.5% |
| R3-b005 | 75.2% | 38.5% |
| R4 (50ep, value_weight=1.0) | 81.0% | 34.5% |
| R3-b020 (combined) | 80.9% | (gate eval pending) |

**Better imitation does not yield better play.** R4 imitates the teacher 6pp more accurately and plays 3pp worse. This rules out the simple "fit the labels harder" path: the teacher's 19–25% argmax errors must concentrate on a small set of high-leverage states (combat finishers, evolution decisions) where one wrong move blows the game.

### The 0.31 cap is structural

Combining R1 (teacher at 58.5%) + R3 + R4 + R3-b020 (planned) + 5 F1 PPO phases:

- Teacher strength: 58.5% (Wilson lower 0.516). Not the cap.
- Imitation accuracy: 74–81%. Not the limit — improving it doesn't move gate WR.
- Value head calibration: now PASS. Doesn't move gate WR.
- Policy entropy at warm-start: 0.5 nats. Already non-peaked.
- PPO HPs: tried spec, aggressive, extreme. Always 0.30 ± 0.01.

**The gate WR ceiling at 0.31 is structural** — not from any single mechanism we've tested. The remaining hypotheses:

1. **State-feature gap.** The 25% argmax disagreements between trained policy and teacher concentrate on high-leverage decisions; the policy can't tell those states apart in feature space.
2. **Sample-weight bias.** Low-margin rollout labels (where teacher itself was unsure) get the same weight as high-margin ones; training picks up noise.
3. **Action-coverage gap.** Candidate ranker filters out some actions the teacher would pick; the BC fit can't recover those.
4. **Side asymmetry.** The 26–30% player vs 41–43% opponent gap suggests a systematic blind spot in player-side decisions.

### Pivot

Tier-1 verdict: peakedness and calibration are not the binding constraints. Pivot to:

- **R5 (PFSP self-play)** — different opponent distribution, different reward landscape. Smallest infra delta.
- **Margin-weighted training** (new Tier-2 idea) — sample weights by `selectedVsRunnerUpMargin` so high-margin rows get more training emphasis.
- **R-WILD side imbalance** — measure where in the game the player-side gap comes from; might suggest a feature gap or a data-skew fix.

R5 is the next concrete experiment. R6/R7/R8/R10 stay on the backlog as larger-cost asymmetric-upside bets.

### R6 (capacity) — overfit teacher labels, plays worse

Trained `--hidden-dim 128 --depth 3 --epochs 50 --value-weight 1.0` on iter-002 mixed data:

| Metric | Warm-start (64/2) | R4 (64/2, 50ep) | R6 (128/3, 50ep) |
| --- | --- | --- | --- |
| Train accuracy | 0.88 | 0.95 | **0.97** |
| Argmax-match vs teacher | 0.747 | 0.810 | **0.832** |
| value_brier | 0.243 | 0.084 | **0.062** |
| Gate WR | 37.5% | 34.5% | **33.0%** |

R6 is the strongest imitator (83% argmax-match, 97% accuracy, best calibration) AND the weakest player. **The cap isn't a capacity issue — it's an imitation-target-quality issue.** The teacher's 25% argmax errors get inherited by the SL fit; bigger model just locks them in more cleanly.

### Combined intervention (R3-b020 + PPO) — same 0.30 cap

PPO from R3-b020 (entropy 0.39 + calibrated value + accuracy 89%) at aggressive HPs:

| Iter | Wilson lower | Decision |
| --- | --- | --- |
| 0 (warm-start eval) | 0.3014 | promoted |
| 1 | 0.2639 | rejected |
| 2 | 0.2826 | rejected → halt |

Max 0.3014 — same as every other PPO sweep. **Combining peakedness fix + calibration fix + aggressive HPs doesn't break the cap.**

### R5 (PFSP self-play) — opponent distribution changed, gate cap held

PPO from DAgger iter-2 warm-start, rollouts vs the item17 opponent pool (DAgger iter-002 sampled uniformly). 3 iters × 800 games × aggressive HPs:

| Iter | Wilson lower | GAE mean_return | GAE mean_advantage |
| --- | --- | --- | --- |
| 0 (warm-start eval) | 0.2873 | -0.11 | -0.07 |
| 1 | 0.2500 | -0.14 | -0.04 |
| 2 | 0.3109 (matched DAgger iter-2 exactly) | -0.14 | -0.04 |

Self-play DID materially change the reward distribution: mean_return moved from -0.20 (vs rule-bot rollouts in phase H) to -0.14 (more even games against a same-strength opponent). Mean_advantage less negative too. **But gate WR (vs rule-bot, the unchanged evaluator) still locks at 0.3109.**

The opponent change perturbs the trained policy's locality but doesn't help against the *eval* distribution. The policy learns to do something different against itself, but that something different doesn't generalize to rule-bot.

### Imitation-cap statement

After 8 PPO sweeps + 4 BC variants + 1 capacity bump + 1 self-play:

> **No combination of warm-start adjustment + PPO HP tuning + opponent distribution broke the Wilson-lower 0.31 ceiling vs rule-bot.** Better imitation, larger models, calibrated values, higher entropy, longer training, bigger PPO buffers, self-play opponents — every well-behaved variant lands at WR ≈ 33–38%, Wilson lower 0.27–0.32. The 0.31 cap is the imitation cap: SL on a 58.5%-WR teacher whose 41% disagreement rows are noisy at decision-critical states, and PPO can't escape its local optimum without a different training signal.

To break it, we need labels or training signal that isn't just "imitate the teacher harder":

- **R5 (PFSP self-play)** — different opponent, different reward shape, possibly different gradient direction. Running.
- **R7 (multi-teacher BC)** — different label distribution. Reduces single-teacher mode-collapse.
- **R8 (DPO)** — different objective. Trains on preference pairs instead of argmax labels.
- **Margin-weighted training** — weight samples by `selectedVsRunnerUpMargin`; high-margin rows are the teacher's confident decisions and likely the high-leverage ones.
- **Reward shape refinement** — Δpoints + ±1 win/loss may miss strategic value. F1 could find direction with richer reward.

### Diagnostic: forced-move dominance

Inspection of iter-002 relabeled rows (n=2883):

| Legal action set size | Row count | Fraction |
| --- | --- | --- |
| 1 (forced) | 1776 | **61.6%** |
| 2 (binary) | 544 | 18.9% |
| 3 | 321 | 11.1% |
| 4–10 | 200 | 6.9% |
| 10+ | ~42 | 1.5% |

JsonlPolicyDataset's `min_actions=2` correctly excludes forced rows from training. But the gate-eval games include them — every game has ~62% of its "decisions" forced, meaning the strategic difference between two policies plays out over ~38% of state transitions, mostly binary choices.

This is the **lever density problem**. A 4% WR gap between policies has to be earned on the ~15 meaningful decisions per game (out of ~40 total). Those few decisions are exactly where teacher labels are most likely noisy (because they're high-leverage). The cap is concentrated in the highest-leverage 1/3 of decisions per game.

### Removing rule-bot replay mix (R-cleanlabels)

Trained BC on JUST `iter-002/relabeled.jsonl` (no rule-bot replay mix-in):

| Variant | Training data | Train accuracy | Gate WR |
| --- | --- | --- | --- |
| DAgger iter-2 (mix=70% rollout-relabeled + 30% rule-bot-replay) | mixed | 0.85 | 37.5% |
| R-cleanlabels (rollout-relabeled only) | clean | 0.92 | **33.0%** |

Removing the rule-bot replay actually *hurts* (-4.5pp). The replay buffer acts as regularization / state coverage; without it the model overfits the smaller relabeled set. **The mix is doing useful work**, not adding noise.

### Pause — single-axis interventions exhausted

Tier 1 (R1–R4) plus stretch experiments R5 (self-play), R6 (capacity), R-cleanlabels all converge on: the 0.31 gate cap is robust to every single-axis intervention. Remaining promising bets require either multi-axis combinations or fundamentally different approaches:

1. **Multi-teacher BC (R7)** — relabel iter-2 trace with planner + search teachers, train on the union. Tests whether label diversity at high-leverage decisions matters. (Compute estimate: planner relabel is slow, maybe 30 min wall-clock.)
2. **DPO (R8)** — train on (selected, runner-up) preference pairs from outcome export. Different objective entirely; doesn't fit argmax labels. (Code effort: ~3 hours for a new trainer.)
3. **MCTS-augmented self-play (R12)** — skip the warm-start + RL split. Game is small enough. (Code effort: ~1 week.)
4. **State-feature audit** — what's missing in features that would let the model tell apart high-leverage decisions? (Diagnostic, not a fix.)

User-facing summary of where we are: the infra works, the cap is structural, and breaking it requires a directional decision about how much code to invest. None of (1)–(4) is going to take less than half a day; (3) is multi-day.

## Decision (2026-05-11): commit to R12, fallbacks ready

After Tier-1 + stretch experiments + strategist + auditor analysis, the chosen path is **R12 (mini-AlphaZero)**. Reasoning recorded in detail in the chat thread; key points:

1. **R6 is the dispositive evidence.** Better imitation makes play worse. Every variant that fits the rollout-CRN labels harder (R4, R3-b020, R6) drops gate WR below the warm-start. The cap isn't "fit labels harder" — it's "the labels are noisy on high-leverage decisions."
2. **R7 and R8 inherit the noise.** Both train on the same rollout-CRN argmax/preference data. Different averaging / objective; same target distribution.
3. **R12 generates new labels via search.** Visit-count distributions from PUCT MCTS with N=100 sims integrate over the variance that single-rollout-CRN samples once. Label quality scales with compute (search depth) instead of being capped at teacher's single-sample noise floor.
4. **Game structure favors MCTS.** 62% forced moves means search budget concentrates on the ~15 meaningful decisions per game. At our current simulator throughput, 100 sims × 15 real decisions × 200 games = 25–35 minute wall-clock per gate.

**Sprint plan: see `docs/r12-sprint-plan.md`.** Day-1 spike has a hard go/no-go criterion (Wilson lower ≥ 0.40 at n=100). If NO-GO, write a postmortem and pivot to fallbacks (R7, R8, R9, R10) in the order ranked by tier.

R7/R8/R9/R10 are kept on the backlog as fallbacks; their task descriptions are annotated to reflect their fallback status.

### R12 implementation progress (2026-05-11)

All sprint phases are coded and smoke-validated. Day-1 spike at 100 games × 100 sims is running; pending result.

| Phase | Artifact | Smoke result |
| --- | --- | --- |
| Day-1 | `backend/src/sim/mcts.ts`, `training/r12_spike.py`, `r12_spike_smoke.py` | smoke PASS (8 games, 0 fallbacks). 100-game gate in flight. |
| A | `mcts.ts` policy-prior + Dirichlet noise (`--mcts-prior policy`, `--mcts-root-dirichlet`) | TS build clean; activated by Phase B/D smokes. |
| B | `backend/src/sim/mctsSelfPlay.ts`, `npm sim:mcts-selfplay`, `r12_selfplay_smoke.py` | PASS — 2 games × 8 sims → 65 rows, schema valid. |
| C | `training/uma_ai/selfplay_dataset.py`, `train_bc.py --data-mode mcts-distill`, `r12_distill_smoke.py` | PASS — 2 epochs, loss 1.75 → 1.47, accuracy 61% → 72%. |
| D | `training/r12_orchestrator.py`, `r12_orchestrator_smoke.py` | PASS — 1 iter × 4 games × 8 sims, all 10 expected event_types emitted. |
| E | `observability_app.py` STAGES extension (`selfplay`, `distill`, `mcts-gate`, `mcts-spike`, `r12-orchestrator`) | n/a (dashboard render check, no smoke). |

Decision: Day-1 spike is the gate on whether to launch a Phase D multi-iteration run. If GO, run 4 iterations × (200 games / 100 sims) per the plan; on the trained Phase-D output, run final 400-game gate at `--mcts-simulations 200` for the headline ≥0.40 Wilson-lower target.

### Day-1 spike: uniform-prior result (NO_GO, but signal-positive)

Spike stopped at n=178 (full schedule was 200; halted early after the trajectory committed to NO_GO).

| Metric | R4 value-head (1-ply) | Day-1 MCTS uniform | Δ |
| --- | --- | --- | --- |
| WR | 0.345 | **0.4045** | +6pp |
| Wilson lower | 0.30 | **0.335** | +3pp |
| Heuristic fallbacks | 0 | 0 | — |
| Player WR | ~0.27 | 0.36 | +9pp |
| Opponent WR | ~0.43 | 0.462 | +3pp |

**Interpretation.** MCTS with uniform prior + R4 value-head leaf is **better than the value-head alone** at the same checkpoint — search adds 6pp WR. But it does not clear the ≥0.40 Wilson-lower bar. The side gap (10pp player vs opponent) persists and is the largest single sink: if both sides hit the opponent-side WR (~46%), we'd land at Wilson lower ~0.39 — almost at the bar from uniform prior alone.

**Day-1 verdict: NO_GO** on the hard criterion. **But** the structural intervention (MCTS over R4) does add positive value, so the next iteration of the spike is warranted instead of skipping straight to a fallback.

### Phase A spike: policy-prior retry (NO_GO, marginal improvement)

n=200, --mcts-prior policy + R4 value-head leaf, otherwise identical to day-1.

| Metric | Phase A | Day-1 uniform | R4 baseline |
| --- | --- | --- | --- |
| WR | 0.4350 | 0.4045 | 0.345 |
| Wilson lower | 0.368 | 0.335 | 0.30 |
| Player WR | 0.36 | 0.36 | 0.27 |
| Opponent WR | 0.51 | 0.46 | 0.43 |

Policy prior gave +3pp Wilson lower over uniform — meaningful but not enough for GO. The diagnostic finding: **the prior is not the bottleneck**. Search budget reorganization helps modestly, but Wilson lower is still 3pp short of the 0.40 bar.

### Rollout-leaf spike: GO (2026-05-11)

Same R4 ckpt, same policy prior, same 100 sims — only the leaf evaluator changed from `value-head` to `rollout` (K=3 rule-bot playouts to terminal, side-relative ±1/0 backed up).

| Metric | Rollout-leaf | Phase A | Day-1 uniform | R4 baseline |
| --- | --- | --- | --- | --- |
| WR | **0.625** | 0.435 | 0.405 | 0.345 |
| Wilson lower | **0.556** | 0.368 | 0.335 | 0.30 |
| Wilson upper | 0.689 | 0.504 | 0.478 | — |
| Player WR | 0.58 | 0.36 | 0.36 | 0.27 |
| Opponent WR | 0.67 | 0.51 | 0.46 | 0.43 |
| Heuristic fallbacks | 0 | 0 | 0 | 0 |

**R12 north star cleared by +15.6pp** (target 0.40 vs achieved 0.556) and **stretch criterion (0.50) also cleared**. Side gap inverted: player 58%, opponent 67%, both winning majority.

**The dispositive diagnostic.** Search adds ~3pp over R4 (uniform-prior MCTS). Policy prior adds another ~3pp. Swapping the value-head leaf for rollout-CRN at leaves adds **+19pp**. The value head — even at R4's calibrated 0.084 Brier — was producing leaf evaluations too noisy for 100-sim MCTS to disambiguate the per-action means. Rollout-CRN K=3 averages directly over the same outcome distribution the value head approximates; the per-action means rank correctly.

**Implications.**
- The "trained policy + MCTS at inference" path of the R12 north star is achieved. Deploy is rollout-leaf MCTS over the R4 checkpoint.
- The "distilled policy without search" path is not achieved by definition — rollout-CRN at leaves IS search. A distilled policy reusing the value head would inherit the same noise floor R4 hit.
- Phase D distillation is now optional: it would compound iter-on-iter and produce a stronger prior, but is not required to meet the criterion.

### Recommended next steps (post-R12 GO) → R13 sprint

See `docs/r13-sprint-plan.md` for the detailed plan. Headline shift:

**Phase D as originally written (visit-count → policy distillation) is no longer the obvious next step.** The R12 diagnostic shows the value head is the bottleneck; a distilled policy would inherit that noise floor. Instead, the next sprint asks a sharper question:

> Is the value head fixable, or is rollout-CRN search permanently the production path?

The cheap falsifiable answer: retrain JUST the value head on rollout-mean outcomes (not game-z), freeze trunk + policy, gate at `--mcts-leaf value-head`. ~4 hours of compute. If Wilson lower ≥ 0.40 with the new value head, distillation is unlocked. Otherwise, search-at-inference is the permanent answer.

In parallel: game-level parallelism (~4× speedup), latency dials (K=1 / adaptive sims / batched /predict) to push p95 decision time under 3 s, UI integration, and an MCTS-vs-MCTS strength ladder so we stop relying solely on a saturating rule-bot.

### R13.W6 result — Phase D iterations compound (2026-05-11)

2 production iterations of `r12_orchestrator` from the W3-retrained warm-start (60 selfplay × 100 sims × rollout-leaf K=3; 20 epochs of mcts-distill with KL anchor 0.05; 120-game gate per iter).

| Iter | WR | Wilson lower | Δ vs prev |
| --- | --- | --- | --- |
| 0 | 0.625 | 0.536 | +5.3pp vs R12 baseline (0.483 / 0.556 at n=200) |
| 1 | **0.658** | **0.570** | +3.4pp vs iter-0 |

Iter-1 promoted as the new strongest model. Per-side: player 0.667 / opponent 0.65 — **R-WILD side gap is closed** (was historically ~16pp opp-favored, briefly 9pp player-favored in R12, now within 2pp). Zero heuristic fallbacks → MCTS execution is clean. Iter-1 checkpoint: `runs/R13-W6-phase-d/iter-1/checkpoint.pt`. W7 headline n=400 gate is the final validation step before deployment.

### R13.W3 result — value head retrain is **PARTIAL** (2026-05-11)

50 rollout-leaf selfplay games (100 sims, K=3, R4 prior) → 25-epoch frozen-trunk MSE retrain → 100-game gate at `--mcts-leaf value-head` over the retrained checkpoint.

| Metric | Retrained head (W3) | R4 baseline (R12 Phase A) | Δ |
| --- | --- | --- | --- |
| WR vs rule-bot | 0.44 | 0.345 | +9.5pp |
| Wilson lower (n=100) | **0.347** | 0.30 | +4.7pp |
| Wilson upper | 0.538 | 0.40 | +14pp |
| Player side WR | 0.40 | — | — |
| Opponent side WR | 0.48 | — | — |

The retrained head is materially better than R4's at the same leaf evaluator, but doesn't clear the 0.40 GO threshold by itself. Verdict: **PARTIAL**. Per the sprint plan, full Phase D (W6) is still worth running with this checkpoint as warm-start — the rollout-mean target reduces value-head variance, and visit-count distillation could compound on top of that. Rollout-leaf inference (Wilson lower 0.556 at n=200, R12) remains the strongest single config we have; W3 narrowed but did not close the gap between value-head leaf and rollout-CRN leaf.

### Tasks deleted as obsolete (2026-05-11 post-R12)

R7 (multi-teacher BC blend), R8 (DPO), R9 (Q-learning head), R10 (full-scale DAgger) were all queued only as fallbacks IF R12 failed. R12 didn't fail. Tasks #29-32 removed from the active backlog. The hypotheses they tested (label-quality fixes for the imitation cap) are also obsolete: R12 proved the cap is downstream of the *value-head leaf noise*, not the *training labels*.

R2 (multi-temperature gate matrix) is repurposed as a deployment-tuning task, not a research diagnostic.

R-WILD's side-imbalance portion is largely resolved by R12 (gap inverted: player 58%, opponent 67%). Simulator determinism + rule-bot mistake catalog remain as low-urgency follow-ups.

## Open / wild

- **Side-imbalance verification.** Gate manifests record player/opponent splits inconsistently across phases. Worth a one-off script to extract the side-WR delta and check whether the model is offensively weak or defensively weak.
- **Simulator determinism audit.** Replay 100 identical seeds end-to-end; measure full-state divergence rate. Silent non-determinism in CRN would invalidate every advantage estimate.
- **Rule-bot mistake catalog.** The 80% aspirational target requires exploiting rule-bot weaknesses; we don't have a catalog of those weaknesses. Hand-construct ~50 states + a careful audit.
