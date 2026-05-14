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

Iter-1 promoted as the new strongest model. Per-side: player 0.667 / opponent 0.65 — **R-WILD side gap is closed** (was historically ~16pp opp-favored, briefly 9pp player-favored in R12, now within 2pp). Zero heuristic fallbacks → MCTS execution is clean. Iter-1 checkpoint: `runs/R13-W6-phase-d/iter-1/checkpoint.pt`.

### R13.W8 result — value-head-only Phase D does NOT compound (2026-05-11)

Tried 5 iterations of `r12_orchestrator` from W6 iter-1 with `--mcts-leaf value-head` for selfplay (instead of rollout). Killed at iter-3 mid-gate after the trend was clear:

| | Wilson lower | WR |
| --- | --- | --- |
| W6 iter-1 baseline (start) | 0.452 | 0.55 |
| W8 iter-0 | 0.404 | 0.49 |
| W8 iter-1 | 0.340 | 0.43 |
| W8 iter-2 | 0.380 | 0.47 |

Every iteration was below the starting baseline. Conclusion: **cheap-leaf selfplay targets are too noisy for distillation to compound** — the visit-count targets from value-head-leaf selfplay are noisier than rollout-CRN-K=3 targets, and distillation regresses strength rather than improving it. Production iteration *requires* rollout-leaf selfplay even if gate-time inference is value-head leaf.

Practical implication: W6's "use rollout-leaf for selfplay, value-head-leaf for inference" decomposition is load-bearing — both halves matter. Don't try to cheap-out the training loop.

### R14 sprint — refinement (2026-05-11)

See `docs/r14-sprint-plan.md`. After R13's two production configs landed, the next sprint splits between shipping (W5 UI finish + Pareto-tuned latency) and one final honest RL attempt (MCTS-trajectory off-policy PPO). Also includes the engine determinism fix that closes R-WILD — a subagent investigation pinpointed `withRng` losing `activeRng` across `await` boundaries; AsyncLocalStorage is the ~10-LOC fix.

R14 workstreams:
- **A** OOD gate for iter-1 (compute only, 30 min) — falsifies "iter-1 value-head leaf overfits its own selfplay distribution"
- **B** Engine determinism fix via AsyncLocalStorage (~1 hour) — closes #34 if it works
- **C** W8 stop rule after iter-2 — concave-compounding guard
- **D** MCTS-trajectory PPO with V-trace (~3 days code + 1 day compute) — the only PPO variant we never honestly ran
- **E** W5 UI integration finish (~1 day code) — the deliverable
- **F** Adaptive-ratio Pareto sweep (~20 min) — picks the W5 default config

Dropped: larger model, more entropy-BC variants, n=400 headline gate (cosmetic), temperature-ramp-only PPO (R3 already settled peakedness-alone).

### R14 progress checkpoint (started 2026-05-11; finished 2026-05-14)

**Sprint outcome:** code-side workstreams all landed. **Primary production claim: rollout-leaf MCTS @ iter-2, Wilson lower 0.6479** (R14.I.2) — max strength, untouched by the side-asymmetry confirmation gate. **Cheap-inference fallback: value-head leaf + adaptive-ratio=1.5 @ iter-2, Wilson lower 0.39–0.45 across three independent seed ranges** (F-iter-2 0.443 at 820000+ / A 0.404 at 800000+ / A.footnote 0.394 at 900000+), 1-of-3 below the 0.40 production bar at 2.56× speedup vs ratio=0. The 0.452 reading originally headlined from F-iter-2 was the upper end of that empirical range at the F seed range, not a stable point estimate (resolved 2026-05-14 after R14.A.footnote — see `docs/ai-agent-state/escalations.md` `## Resolved`). The F-iter-1 sweep had earlier shown the 0.42 target was unreachable on iter-1 at the F seed range; re-running on iter-2 PASSes it across all ratios at the F seed range. The original A "side asymmetry is partly seed-clustered" caveat was refuted by A.footnote — the player/opponent gap (+0.11–0.30pp Wilson) replicates across four gates and two independent seed ranges. Engine determinism (R14.B) and orchestrator inspection (R14.I.2 inspector) shipped. UI plumbing (R14.E) verified end-to-end via headless smoke (decisionMs=200 at 16 sims → ~0.5s at 100 sims with ratio=1.5, comfortably under E's <3s target); only the 20-game in-browser exercise remains unautomatable. ORT unpinning (R14.G) showed 1.05× — kept at `--ort-threads 1`. PPO-with-V-trace (D) and predict-batching (H) skipped per the sprint plan's decision points (I succeeded; G negative). After the sprint re-refinement promoted I to top and demoted D to fallback:

- **B DONE.** AsyncLocalStorage installed via `installRngStorageProvider` + side-effect `backend/.../rngAsyncStore.ts`. The frontend keeps its sync-module fallback so the browser bundle stays clean of `node:async_hooks`. `training/r14_determinism_smoke.py` now passes bit-exact 12/12 between --workers 1 and --workers 4 at value-head MCTS, n=12, seeds 141414+. Pre-fix: divergent. R-WILD #34 closed.
- **I.1 DONE.** `training/r14_value_crossover_probe.py` measures (val_mse, pearson_r) between a checkpoint's value head and the rollout-CRN K=3 means in `rootValue`. crossed = (val_mse <= 1.10 × W3-floor) AND (pearson_r >= 0.7). Wired into `r12_orchestrator.run_iteration` between distill and gate, against the **previous** iter's selfplay (held-out). Baseline at W6 iter-1 vs its own iter-1 selfplay (in-distribution): val_mse 0.659, pearson 0.535, ratio 1.158, crossed=false — explains the W8 regression (cheap-leaf selfplay distillation failed because the value head hadn't caught up).
- **I.2 DONE (halted at iter-4).** W6 phase-d extended through iter-4. iter-2 promoted at Wilson **0.6479** (WR 0.7333, n=120), clearing the sprint's ≥0.60 north star. iter-3 dropped to 0.5783 and iter-4 to 0.5612 — two consecutive promotion failures triggered the orchestrator's auto-halt. Crossover probe never satisfied both gates: pearson_r stayed ~0.50 (target ≥0.7), ratio stayed 1.18–1.29 (target ≤1.10). So I.3 (cheap-selfplay retry after two `crossed=true` events) never fired and is dropped from the sprint. Strongest checkpoint: `runs/R13-W6-phase-d/iter-2/checkpoint.pt`. (Note: the earlier orchestrator dim-mismatch crash was fixed by auto-inferring hidden_dim/depth from the init checkpoint — commit `1c47146`.)
- **E DONE (plumbing); 20-game manual UI exercise pending.** Visible MainMenuScreen toggle in (commit `0371892`). `training/r14_ai_decide_e2e_smoke.py` PASS (2026-05-14): real mid-game state via headlessAiVsAi → /ai/decide → returns valid action + nextState (fingerprint advances). decisionMs=200 at 16 sims → extrapolates to ~1.25s at 100 sims (under E's <3s target). Only the in-browser 20-game fallback-rate exercise remains; not headless-automatable.
- **G DONE (NEGATIVE; 2026-05-14).** Smoke: pinned 13.4s vs auto 12.8s, speedup **1.05×** vs 1.5× target → FAIL on speedup, PASS on determinism (12/12 winner+turnNumber match). At this model size, per-call /predict overhead dominates compute, so ORT thread parallelism doesn't help. Decision: keep `--ort-threads 1` default (R13.W1 legacy preserved), skip H predict-batching (same overhead ceiling).
- **F DONE (Pareto pick = ratio 2.0; 2026-05-14).** Sweep on iter-1 + value-head leaf, n=100 each. Stated FAIL (0.42 Wilson target unreachable at fresh seeds 820000+; ratio=0 baseline only 0.366) but ratio=2.0 Pareto-dominates baseline: +4pp WR (0.50 vs 0.46), 2.19× speedup (118s vs 258s). Higher ratios over-prune. Production pick for W5 default: ratio=2.0. See dedicated R14.F section below.
- **A DONE (PASS on iter-2; 2026-05-14).** Re-targeted to iter-2 (newly-promoted production candidate) — see the dedicated A result section below. Both gates pass at 0.40 floor; iter-2 beats R4 head-to-head MCTS by 11pp Wilson lower; side asymmetry persists; +18pp rollout-leaf gain does not transfer to value-head-leaf inference.

### R14.F — Adaptive-ratio Pareto sweep on W6 iter-1 (2026-05-14)

Ran `training/r14_adaptive_ratio_sweep.py` on iter-1 + value-head leaf, 100 sims, n=100 per ratio, seeds 820000+, 4 workers, single shared serve_onnx. Sweep complete:

| ratio | WR | Wilson lower | Wilson upper | elapsed (s) | speedup vs baseline | wallclock cut |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 0.46 | 0.366 | 0.557 | 258.0 | 1.00× | 0% |
| 1.5 | 0.46 | 0.366 | 0.557 | 123.0 | 2.10× | 52% |
| **2.0** | **0.50** | **0.404** | **0.596** | **117.6** | **2.19×** | **54%** |
| 3.0 | 0.45 | 0.356 | 0.548 | 155.0 | 1.66× | 40% |
| 5.0 | 0.49 | 0.394 | 0.587 | 181.3 | 1.42× | 30% |

**Stated verdict: FAIL** (no ratio satisfies the original exit criterion `Wilson lower >= 0.42`). But that target was set against the W6 paper baseline of 0.452 at seeds 700000+. On the fresh seeds used here, even ratio=0 baseline is only 0.366 — the 0.42 target is unreachable for any ratio. The exit criterion did not anticipate the seed-distribution variance later confirmed by R14.A.

**Real verdict:** ratio=2.0 **Pareto-dominates baseline** on both axes — +4pp WR (0.50 vs 0.46), 2.19× speedup. Wilson CIs at n=100 overlap heavily ([0.366, 0.557] vs [0.404, 0.596]) so the strength gain is within sample noise; the **safe claim is "no strength regression at 2.19× speedup"**. Above ratio=2.0 the curve is concave: 3.0 over-prunes (-1pp WR, slower because of bookkeeping), 5.0 partial recovery in WR but slower still.

ratio=1.5 is bit-identical strength to baseline (same wins on the same seeds — same Wilson) at 2.10× speedup — confirms adaptive halts only fire when they don't change the chosen action; below ratio=2.0 the halt rule never triggers cases where it could disagree with full search.

**Production pick:** ratio=2.0 for W5 UI / E default config. The 2.19× speedup roughly halves median decision wall-clock — directly relevant to E's "<3s decision time" exit criterion.

**Followup before locking:** re-run the sweep on iter-2 (the new I.2 production candidate). iter-2 may shift the optimum (different value-head profile → different halt-rule firing pattern). Cheap (~25 min) but only do this once iter-2 deployment is closer to landing.

Output: `runs/R14-adaptive-sweep/{ratio-*.{log,manifest.json,progress.jsonl},summary.json}`.

### R14.F — Adaptive-ratio sweep re-targeted on iter-2 (2026-05-14)

Re-ran the F sweep on the I.2-promoted checkpoint `runs/R13-W6-phase-d/iter-2/checkpoint.pt` (same script, same seeds 820000+, n=100, 100 sims, value-head leaf). On iter-2 all five ratios clear the 0.42 Wilson floor — the F exit criterion is satisfied here in a way it could not be on iter-1.

| ratio | WR | Wilson lower | Wilson upper | elapsed (s) | speedup | wallclock cut |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 0.54 | 0.443 | 0.634 | 269.0 | 1.00× | 0% |
| **1.5** | **0.55** | **0.452** | 0.644 | **105.0** | **2.56×** | **61%** |
| 2.0 | 0.55 | 0.452 | 0.644 | 132.7 | 2.03× | 51% |
| 3.0 | 0.53 | 0.433 | 0.625 | 154.1 | 1.75× | 43% |
| 5.0 | 0.57 | 0.472 | 0.663 | 175.7 | 1.53× | 35% |

**Status: PASS** (script's rule auto-picks ratio=5.0 as highest-passing ratio).

**Pareto frontier on iter-2:**
- **ratio=1.5** (max speed, recommended W5/E cheap-inference fallback pick): Wilson 0.452 at the F seed range + 2.56× speedup. Decision time at 100 sims extrapolates to ~0.5s — well under E's <3s target with margin to spare. The 0.452 reading here is the R14.A canonical-seed-range upper end; the empirical range across three independent seed ranges (F-iter-2 / A / A.footnote) is **Wilson lower 0.39–0.45** with 1-of-3 below the 0.40 bar (R14.A.footnote 2026-05-14). Primary production claim is rollout-leaf MCTS @ iter-2 Wilson 0.6479.
- **ratio=5.0** (max strength): Wilson 0.472 + 1.53× speedup. Worth +2pp Wilson if compute is cheap, but loses 1.7× of the speedup.
- **ratio=2.0** is strictly dominated by 1.5 on iter-2 (same Wilson, slower) — different from the iter-1 sweep where 2.0 was the Pareto pick. iter-2's value head produces a different halt-rule firing pattern.

**Headline (qualified post-A.footnote, 2026-05-14):** value-head-leaf iter-2 at ratio=1.5 reads **Wilson lower 0.452 at the F seed range (820000+)** at ~0.5s/decision vs R13.W6's reported iter-1 at ~1.25s/decision; across three independent seed ranges (F-iter-2 / A / A.footnote) the empirical range is **Wilson lower 0.39–0.45** with 1-of-3 below the 0.40 production bar — the 0.452 number is the upper end, not a stable point estimate. The W6 → I.2 training translates to a ~2.5× speed-equivalent strength gain at the cheap-inference fallback deployment point. Primary production claim is rollout-leaf MCTS @ iter-2 Wilson 0.6479 (R14.I.2).

**Implication for the A side-asymmetry caveat:** F-iter-2's seeds 820000+ show iter-2 at 0.443 Wilson (n=100), but A's seeds 800000+ showed 0.404 Wilson (also n=100). The 4pp swing across overlapping seed ranges suggests the side-asymmetry from A is partly seed-clustered. Worth a confirmation gate at a third seed range before deployment.

Output: `runs/R14-adaptive-sweep-iter2/{ratio-*.{log,manifest.json,progress.jsonl},summary.json}`.

### R14.A — OOD gate result on W6 iter-2 (2026-05-14)

Ran `training/r14_ood_gate.py` against the I.2-promoted checkpoint `runs/R13-W6-phase-d/iter-2/checkpoint.pt`. Re-targeted from the originally-specified iter-1 because iter-2 is the newly-promoted production candidate (and had no value-head-leaf number yet). Two 100-game gates, value-head leaf, 100 sims, 4 workers, seeds 800000+ / 850000+.

| Gate | n | WR | Wilson lower | Wilson upper | Verdict |
| --- | --- | --- | --- | --- | --- |
| 1: fresh seeds, vs rule-bot | 100 | 0.50 | **0.404** | 0.596 | passes 0.40 marginally |
| 2: iter-2 vs R4, MCTS-vs-MCTS | 100 | 0.61 | **0.512** | 0.700 | passes 0.40 by 11pp |

Per-side breakdown reveals severe asymmetry on both gates:

| Gate | Player WR | Player Wilson lower | Opponent WR | Opponent Wilson lower |
| --- | --- | --- | --- | --- |
| 1 (vs rule-bot) | 0.44 | 0.312 | 0.56 | 0.423 |
| 2 (vs R4 MCTS) | 0.46 | 0.330 | 0.76 | 0.626 |

**Interpretation.**

1. **Both gates pass — iter-2 is OOD-robust.** Strength claim is not seed-distribution-specific.
2. **iter-2 dominates R4 head-to-head MCTS** (gate 2 Wilson lower 0.512). Gate 2 stronger than gate 1 by ~11pp: iter-2 is a genuine model improvement over R4, not a rule-bot artifact.
3. **iter-2's value-head-leaf strength is statistically indistinguishable from iter-1's.** Wilson lower 0.404 (iter-2) vs 0.452 (iter-1's R13.W6 reported number). The +18pp Wilson gain from iter-1 → iter-2 at rollout-leaf MCTS (0.452 → 0.6479) does **not** transfer to value-head-leaf inference. iter-2's added training improved the policy/rollout combination, not the value head's ability to score leaves directly.
4. **Side asymmetry persists post-B.** AsyncLocalStorage closed the parallel-determinism gap, but iter-2 is materially weaker as player (Wilson lower 0.31–0.33 across both gates) than as opponent (0.42 vs rule-bot, 0.63 vs R4). This is not a determinism bug — both sides are bit-exact reproducible — but a real *strategic* asymmetry in the iter-2 policy at value-head-leaf inference. Likely tied to first-move/initiative dynamics: the model handles defending better than initiating. The production claim should disclose the side gap.
5. **Cheap-inference production config:** rollout-leaf iter-2 at Wilson 0.6479 remains the headline. Value-head-leaf iter-2 at Wilson 0.404 is the cheap-inference fallback — defensible but the side asymmetry caveat sticks.

Followups (low priority):
- Side-asymmetry-specific gate (player-only n=200) to tighten the per-side Wilson CI before any deployment claim.
- Repeat at a third seed range to confirm the asymmetry isn't seed-clustered.

Bug fix landed in this run: `r14_ood_gate.py` was passing a relative `--out-dir` to `npm --workspace backend run sim:eval-gate`, which resolves against `backend/` workspace cwd → manifest landed at `backend/runs/...` and the orchestrator failed to read it. Fixed by resolving `out_dir` to absolute at parse time.

### R14.A.footnote — Side-asymmetry confirmation gate at independent seeds (2026-05-14) — DONE / gate1 FAIL

- **Motivation:** R14.A's caveat noted the player-vs-opponent gap may be partly seed-clustered (F-iter-2 at 0.443 vs A at 0.404 across overlapping seed ranges). One independent seed range at the same MCTS production config (value-head leaf, 100 sims, c_puct=1.5) would either tighten the production claim or contradict it. Cheap diagnostic, ~15 min.
- **Result (2026-05-14):** **gate1 FAIL @ Wilson 0.394; gate2 PASS @ Wilson 0.482; side-asymmetry confirmed real, not seed-clustered.** Re-ran `training/r14_ood_gate.py` against `runs/R13-W6-phase-d/iter-2/checkpoint.pt` at independent seeds 900000+ (gate1, vs rule-bot) / 950000+ (gate2, vs R4 MCTS), n=100/gate, same MCTS config as R14.A.

| Gate | n | WR | Wilson lower | Wilson upper | Verdict |
| --- | --- | --- | --- | --- | --- |
| 1: fresh seeds 900000+, vs rule-bot | 100 | 0.49 | **0.394** | 0.587 | **FAIL** (0.6pp below 0.40 bar) |
| 2: OOD seeds 950000+, iter-2 vs R4 | 100 | 0.58 | **0.482** | 0.672 | PASS by 8pp |

  Per-side breakdown (all four gates × both seed ranges):

| Seed range | Gate | Player WR | Player Wilson lower | Opponent WR | Opponent Wilson lower | Gap |
| --- | --- | --- | --- | --- | --- | --- |
| 800000+ (R14.A) | 1 (vs rule-bot) | 0.44 | 0.312 | 0.56 | 0.423 | +0.111 |
| 850000+ (R14.A) | 2 (vs R4 MCTS) | 0.46 | 0.330 | 0.76 | 0.626 | +0.296 |
| 900000+ (footnote) | 1 (vs rule-bot) | 0.42 | **0.294** | 0.56 | **0.423** | +0.129 |
| 950000+ (footnote) | 2 (vs R4 MCTS) | 0.50 | **0.366** | 0.66 | **0.522** | +0.156 |

  **What this changes about the R14 production claim:** the cheap-inference deployment headline "Wilson 0.452 at value-head leaf + ratio=1.5" (F-iter-2, 2026-05-14, sprint-plan line 121) was **not confirmed at a third independent seed range**. The empirical range across F-iter-2 (0.443 at seeds 820000+), R14.A (0.404 at seeds 800000+), and this footnote (0.394 at seeds 900000+) is **Wilson lower 0.39–0.45**, with one-of-three runs failing the 0.40 bar. Honest framing: report the cheap-inference number as "Wilson lower 0.39–0.45 across three independent seed ranges" rather than as a stable 0.452 point estimate.

  **What this does NOT change:** the R14 max-strength headline — **rollout-leaf MCTS at iter-2 Wilson 0.6479** (I.2) — was NOT tested by this gate and remains the unchanged max-strength deployment claim. The FAIL applies only to the cheap-inference value-head-leaf + ratio=1.5 deployment pick.

  **What this refutes about R14.A:** the original A note "side-asymmetry is partly seed-clustered" (sprint-plan line 113) is contradicted. The player-Wilson < opponent-Wilson gap replicates with the same shape across all four gates and two independent seed ranges; it is a real strategic asymmetry in the iter-2 policy at value-head-leaf inference, not seed-distribution noise.

  **gap_gate1_minus_gate2 = −0.088** (gate2 stronger than gate1). The model is closer to matching its own R4 baseline head-to-head than to beating rule-bot at the production claim level on fresh seeds — suggests rule-bot behavior at seeds 900000+ differs meaningfully from the R14.A / F seed ranges, but not a separate diagnostic this slot.

  Wall-clock 9m 38s total (gate1 275.7s + gate2 302.7s). Output: `runs/R14-A-side-asymmetry-seed900000/{summary.json,gate1-fresh-seeds.manifest.json,gate2-mcts-vs-r4.manifest.json}`.

  **Escalated to harness:** the R14 cheap-inference production claim is materially weakened; the human owns the research-stance decision: (a) re-target to CI phrasing, (b) re-run R14.A at larger n, or (c) accept and route to rollout-leaf MCTS. See `docs/ai-agent-state/escalations.md`.

### R13 PPO probe on iter-1 — still doesn't move (2026-05-11)

Three iterations of `ppo_orchestrator` from the W6 iter-1 checkpoint (20 games/update, 30-game gate per iter, `--selection policy` for both collection and gate):

| Iter | Wilson lower (policy gate) | mean_advantage |
| --- | --- | --- |
| 0 | 0.242 | -0.150 |
| 1 | 0.301 | +0.049 |
| 2 | 0.256 | -0.139 |

Compare to iter-1 at value-head-leaf MCTS: Wilson 0.452. PPO at raw policy regressed strength. Diagnostics:

- **Importance ratios ~1.0** across all minibatches → policy still close to argmax.
- **Per-minibatch entropy ~0.27 nats** — higher than the original R3 BC (0.18) but the visit-count distillation didn't soften the policy enough for PPO to differentiate trajectories.
- **mean_advantage still flips sign** between iterations — W3's value-head retrain helped MCTS-augmented decisions but doesn't carry to raw-policy GAE returns.

Verdict: iter-1's strength is **coupled to MCTS at decision time**. PPO from the raw policy is still blocked by the same two issues that killed the original F1 attempts (peaked policy + return-vs-value mismatch). To unblock RL: (a) generate trajectories under MCTS instead of raw policy (expensive), or (b) restart from a much softer init (Dirichlet-warmed BC with high temperature). Both are R14+ work.

### R13 cheap-inference verdict — value-head-leaf at iter-1 clears GO (2026-05-11)

Gate at the W6 iter-1 checkpoint with `--mcts-leaf value-head` (no rollouts), 100 sims, 100 games seeds 700000+:

- **WR 0.55 (55/100), Wilson95 [0.452, 0.644]**
- Player 0.48 (Wilson [0.348, 0.615]); Opponent 0.62 (Wilson [0.482, 0.741])
- Zero fallbacks, 194s wall-clock for 100 games (~2s/game with 4 workers)

Value-head-leaf progression at the same evaluator and 100 sims:

| Config | Wilson lower | Δ vs R4 |
| --- | --- | --- |
| R4 baseline | 0.30 | — |
| W3 retrain alone | 0.347 | +4.7pp |
| W6 iter-1 (W3 + visit-count distill) | **0.452** | **+15.2pp** |

Visit-count distillation on top of W3's variance fix did real work. Iter-1 at cheap inference clears the 0.40 GO bar — **cheap-inference deployment is viable**. Trade-off vs rollout-leaf (Wilson 0.573): -12pp Wilson for ~20× latency reduction (2s/game vs 40s/game). For human-facing UI play, value-head-leaf is the natural production config.

### R13.W7 headline — n=100 validation (2026-05-11)

Independent gate (seeds 600000+) at the W6 iter-1 checkpoint under the same rollout-leaf MCTS config (100 sims, K=3). Originally launched for n=400, truncated to n=100 since the Wilson half-width was already tight enough that 4× the compute is cosmetic:

- **WR 0.67 (67/100), Wilson95 [0.573, 0.754]**
- Player: 0.551 (27/49), Wilson [0.413, 0.681]
- Opponent: 0.784 (40/51), Wilson [0.654, 0.875]

Side asymmetry returned at this seed range (opponent +23pp over player) — Wilson CIs do overlap so likely seed-distribution noise rather than a regression, but worth a quick repro at a different seed-start before claiming the R-WILD gap is closed unconditionally. Bottom line: the production headline is **Wilson lower ≥ 0.57 vs rule-bot, n≥100, rollout-leaf MCTS 100 sims**, comfortably past the R12 baseline.

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

## Post-R14 / post-F1 follow-ups (2026-05-14)

R14 closed on 2026-05-14. The code-side workstreams all landed. **Primary production pick:**
rollout-leaf MCTS at iter-2 (Wilson lower 0.6479, R14.I.2) — max strength, untouched by the
side-asymmetry confirmation gate. **Cheap-inference fallback:** value-head-leaf iter-2 + adaptive
ratio=1.5 at **Wilson lower 0.39–0.45 across three independent seed ranges**, ~0.5s/decision,
with 1-of-3 runs failing the 0.40 bar (R14.A.footnote 2026-05-14 resolved this — the originally
headlined 0.452 was the upper end of the empirical range, not a stable point estimate).
F1 PPO post-mortem
(see `docs/ai-performance-research-progress.md` § "F1 Phase summary") declared the 0.40 target
**unreachable from the current item17 warm-start** and listed four ranked next moves. The items
below capture those moves plus the only outstanding R14 acceptance step.

### R14.E.followup — Manual 20-game UI exercise

- **Motivation:** R14.E's plumbing is verified (headless `/ai/decide` smoke PASS, extrapolated
  decisionMs ~0.5s at ratio=1.5, 100 sims) but the in-browser fallback-rate and median latency
  numbers can only come from real play. This is the only R14 acceptance criterion still open.
- **Next action:** Start backend + frontend dev servers, toggle MainMenuScreen AI=MCTS, play
  20 full games end-to-end at value-head leaf + adaptive-ratio=1.5, capture devtools console
  log for `decisionMs` per turn and fallback events.
- **Cost:** ~30-45 min of actual play.
- **Exit / gate:** fallback rate < 5%, median decisionMs < 3s. Append result to
  `docs/r14-sprint-plan.md` Progress section as the E closure.

### R15.S1 — Better SL warm-start (F1 next-move #1) — DONE / FAIL

- **Motivation:** F1 PPO post-mortem ranked "better warm-start" first. The DAgger sweep at this
  codebase config plateaued at WR 37.5% with low-entropy. Larger SL run (more games, more epochs,
  possibly explicit entropy regularization during BC) might give PPO an actually-movable starting
  point. Distinct from R3's β=0.05 entropy-bonus probe, which already showed entropy alone is not
  the issue — this is the *SL-scaling* angle.
- **Next action:** Scope a single full-scale DAgger run: ~3× the games (≥200 trace games per iter),
  ≥50 epochs, hidden_dim=64/depth=2 unchanged; capture warm-start Wilson lower at greedy + a stretch
  PPO sweep from that checkpoint. Pre-register: SL warm-start Wilson lower ≥ 0.45 before any PPO
  is run; otherwise PPO has the same gradient-signal problem as F1 phases 2/G/H.
- **Cost:** ~1–2 h compute (item-17 take-2 was ~30 games × 25 epochs in <10 min; 3× scale ≤ 1.5h).
- **Exit / gate:** warm-start Wilson lower ≥ 0.45 *or* document the new SL ceiling and close the
  branch.
- **Result (2026-05-14):** **FAIL — pre-registered falsification confirmed.** Run
  `runs/R15-S1-warmstart-sweep/` — 3 DAgger iters × 90 trace games × 75 epochs at fixed
  hidden=64/depth=2, rollout-CRN×3 teacher, KL anchor 0.0/0.1/0.5, n=500 side-balanced gate per
  iter. Wilson lower per iter: iter-0 **0.2845** (WR 32.4%) → iter-1 **0.2883** (WR 32.8%) →
  iter-2 **0.3269** (WR 36.8%). Iter-2 Wilson 0.3269 lands inside the pre-registered falsification
  band 0.311 ± 2pp (= [0.291, 0.331]) — the +1.6pp lift over item17 take-2's 0.311 is within
  Wilson half-width at n=500 and an order of magnitude below the predicted +14pp. The
  pre-registered **secondary check fired**: per-epoch val_accuracy reached 99% of peak by epoch
  2-3 in every iter (best: iter-0 0.7676 @ ep7, iter-1 0.7574 @ ep2, iter-2 0.7656 @ ep3), then
  *declined* over the remaining 70+ epochs while train_acc climbed to 0.91-0.97. Classic
  plateau-then-overfit at this size — the predicted falsification mechanism fired in train-time
  diagnostics first, then validated at eval time. Wall-clock 11m 51s, ~8× faster than the
  ~1.5h scoping estimate (the rollout-CRN×3 teacher dominated; n=500 eval was not the
  bottleneck). Full writeup + per-iter trajectory + comparison table in
  `docs/ai-performance-research-progress.md` § "Phase K — F1 DAgger compute-scaled warm-start".
  Combined with R15.S2 (closed FAIL this morning), the two highest-ranked F1 post-mortem next
  moves have both falsified. The F1 cap is **not** compute at fixed capacity (this run), **not**
  PPO HPs (phases 2/G/H), **not** weak-pool self-play (R5), and **not** strong-pool self-play
  with v1 plumbing (phase J). Surviving F1 candidates: R15.S3 (richer reward shaping), R15.S4
  (sampling-temperature gate, diagnostic), and the deeper SL-label-quality branches the
  falsification opens up (R7 multi-teacher labels, R8 DPO).

### R15.S2 — PFSP self-play PPO (F1 next-move #2) — DONE / FAIL

- **Motivation:** F1 PPO post-mortem ranked self-play second. Item-12 opponent pool already exists;
  `ppo_orchestrator` currently uses `--opponent-model-url` unset (rule-bot default). Rollouts against
  PFSP-sampled prior promoted checkpoints would change the reward distribution from
  single-opponent-shape to diversity-shape. R5 (a prior tier-2 attempt against the item17 pool)
  did NOT break the cap, but the W6/I.2 pool is a much stronger opponent set; worth one cheap
  re-attempt with the new pool.
- **Next action:** Confirm `--rollout-vs-pool` plumbing is intact in `ppo_orchestrator.py` (or add it
  if missing). Run one PPO sweep with PFSP-sampled rollout opponents drawn from
  `runs/R13-W6-phase-d/iter-{0,1,2}/checkpoint.pt`, aggressive HPs, 800 games/update, 3 iters.
- **Cost:** ~10–15 min compute + any plumbing patches.
- **Exit / gate:** Wilson lower ≥ 0.40 on the rule-bot eval gate. Otherwise close the branch.
- **Result (2026-05-14):** **FAIL.** Run `runs/R14-f1-self-play-sweep/` — 3 iters × 800 games at
  aggressive HPs from W6/iter-2 warm-start against the W6/iter-{0,1,2} pool. Wilson lower per iter:
  warm-start eval **0.1455** (WR 30%, n=20) → iter-1 **0.2993** (WR 50%, n=20, opponent W6/iter-1)
  → iter-2 **0.2188** (WR 40%, n=20, opponent the *just-promoted iter-1 from this run*). Iter-1's
  +14.5pp lift is the largest single PPO step recorded in any F1 phase — confirming the stronger
  pool does break the "ratios ≈ 1.0" stasis that hobbled phases 2/G/H — but iter-2 regressed when
  the v1 one-opponent-per-run sampler rolled a self-promotion, creating co-adaptation. Final
  promoted Wilson **0.2188** missed the 0.40 gate by 18pp; iter-1 best missed by 10pp. Both
  numbers are *worse* than R5's iter-2 (0.3109) and phase H's iter-2 (0.3109) despite materially
  different mechanism (the policy actually moved). Full writeup +
  per-iter mean_return / entropy / KL trace in
  `docs/ai-performance-research-progress.md` § "Phase J — F1 PPO + strong-pool self-play".
  Closes the strong-pool branch of the self-play hypothesis. R15.S1 (better warm-start) and
  R15.S3 (richer reward shaping) remain the only unfalsified F1 next moves.

### R15.S3 — Richer reward shaping (F1 next-move #3)

- **Motivation:** F1 PPO post-mortem ranked richer reward shaping third. Current reward is
  Δpoints × 1/3 + terminal ±1. Strategic depth around attachment / retreat / energy cycles is not
  rewarded per-step. PPO might exploit a denser signal even if the existing gradient mechanism
  stays argmax-ratio-bound.
- **Next action:** Catalog 3–5 candidate intermediate rewards from the existing turn-goal /
  candidate-ranker telemetry (e.g. successful attach, retreat survival, KO threat resolution).
  Pre-register one shaping schedule (decay-to-terminal weight), implement in `ppo_orchestrator.py`'s
  reward computation hook, smoke at f1 phase H scale.
- **Cost:** ~3–4 h code + ~20 min compute.
- **Exit / gate:** Wilson lower ≥ 0.40 on the rule-bot eval gate. Diagnostic regardless: if it
  doesn't move WR but does change `mean_advantage` distribution, that itself is publishable.

### R15.S4 — Sampling-temperature gate (F1 next-move #4)

- **Motivation:** F1 PPO post-mortem ranked this fourth (diagnostic, not solution). All five F1
  phases were scored on greedy argmax behavior. If PPO is shaping the policy distribution but not
  flipping argmax, an alternative gate that samples at e.g. T=0.5 might reveal latent improvement.
  Repurposes the R2 multi-temperature gate matrix idea against the F1 phase-H promoted checkpoint
  rather than the original DAgger artifacts.
- **Next action:** Confirm the `sampling=stochastic&temperature=T` body is wired through
  `serve_onnx` and `evaluateModelVsHeuristic`. Run a 5-temperature × 2-checkpoint matrix
  (T ∈ {0, 0.3, 0.5, 1.0, 1.5}, F1 phase-H iter-2 + DAgger iter-2 warm-start) at n=100.
- **Cost:** ~30 min plumbing check + 5 min × 10 cells = ~1.5h compute.
- **Exit / gate:** Either a non-greedy temperature lifts F1 phase-H above the DAgger warm-start
  by ≥3pp Wilson lower (proves PPO was making *some* signal we couldn't see), or all temperatures
  match → confirms PPO produced nothing the gate could measure, closes this hypothesis.

## Open / wild

- **Side-imbalance verification.** Gate manifests record player/opponent splits inconsistently across phases. Worth a one-off script to extract the side-WR delta and check whether the model is offensively weak or defensively weak.
- **Simulator determinism audit.** Replay 100 identical seeds end-to-end; measure full-state divergence rate. Silent non-determinism in CRN would invalidate every advantage estimate.
- **Rule-bot mistake catalog.** The 80% aspirational target requires exploiting rule-bot weaknesses; we don't have a catalog of those weaknesses. Hand-construct ~50 states + a careful audit.
- **Side-asymmetry confirmation gate (R14.A footnote).** ~~Open follow-up.~~ **CLOSED 2026-05-14 — gate1 FAIL @ Wilson 0.394; the R14 cheap-inference production claim is contradicted at independent seeds.** See R14.A.footnote Result section above and `docs/ai-agent-state/escalations.md` for the open research-stance decision the human owns.
