# AI Research Backlog

This is the forward-looking model-strength research backlog. Keep it compact:
landed or failed work shrinks to a pointer, and detailed mechanisms/numbers live
in progress, scoping, or analysis docs. See `docs/ai-research/README.md`.

## Current Anchor

As of 2026-05-18:

- **Production stays pinned 96-d.** Rollout-leaf MCTS at W6 iter-2 is the
  current production claim: Wilson lower 0.6479 vs rule bot at the
  side-balanced n=120 gate. Canonical details:
  `docs/ai-research/progress/r15.md` and the R1-R14 historical record in
  `docs/ai-performance-research-progress.md`.
- **The iter-2-peak-then-regress is now a characterized schema-independent
  recipe property.** Reproduced across BOTH 96-d and 110-d v3; 0.6479 is
  itself W6's *transient iter-2 peak*, not a stable optimum. The R110 v3
  reproduction was MARGINAL (best-promoted iter-2 0.6042) and not promoted.
  Canonical: `docs/ai-research/progress/r110.md`.
- **The search-free/raw-policy SL line is closed.** Labels, objective,
  representation, and label-shape all failed to approach the 0.40 gate
  (R7/R8/R7.b.2/mcts-distill v1). Do not spend compute on another isolated
  raw-policy tweak without explicit new evidence.
- **Cheap inference remains a fallback, not the production claim.** Value-head
  leaf plus adaptive-ratio=1.5 sits around the 0.39-0.45 Wilson-lower envelope
  across seed ranges and is only a latency fallback.
- **Side asymmetry is real enough to disclose.** Current rollout-leaf evidence
  is aggregate side-balanced strength, not a per-side guarantee.
- **The active frontier is W6 loop anti-degradation (recipe axis), now at a
  user-gated regularization-dose decision.** The recipe-fix landed and was
  confirmed via a full R111 loop (iter-3 rot eliminated) but over-damps at
  default HP — net ceiling loss vs the 0.6042 baseline. R16-P1
  temporal/turn-state features are now implemented (commit `357f0d6`); its
  strength verdict is also user-gated. Live operational state is in
  `docs/ai-agent-state/queue.json`.

## Active Search-Wrapped Frontier

0. **W6 loop anti-degradation recipe — TOP PRIORITY, at a user-gated dose
   decision.** Discriminator SATISFIED, recipe-fix LANDED (commit `d44de3f`;
   cross-iter replay buffer + fixed iter-0/SL KL anchor in
   `r12_orchestrator.py`), and CONFIRMED via a full R111 loop: the iter-3 rot
   is eliminated (iter-3 holds 0.5527, no crater) but the fix over-damps at
   default HP — policy froze then decayed (0.5527→0.5358), entire trajectory
   below the 0.6042 baseline iter-2 peak. Forward line is now a **user-gated
   regularization-dose sweep** (DO NOT auto-launch): lower/anneal the
   fixed-KL-anchor weight from 0.05 and/or reduce replay old-fraction (0.40)
   / window (3) to damp ONLY the iter-3 rot while keeping the iter-2 climb.
   Canonical: `docs/ai-research/progress/r110.md` §4c; knobs in queue
   `w6-loop-anti-degradation`.

0b. **Fork B — label-quality test matrix (umbrella over item 0's dose
   sweep).** Raises target/teacher quality at *fixed* capacity and volume
   (both axes closed); the M6 anti-drift arm is the item-0 dose sweep, not a
   re-plan. Gated behind a near-zero-cost read-only label-quality probe
   before any user-gated loop compute. Canonical:
   `docs/ai-research/scoping/forkb-label-quality-loop-recipe.md`.

1. **R110 W6 reproduction — DONE.**
   Verdict MARGINAL (best-promoted iter-2 Wilson lower 0.6042, in the
   0.60-0.6479 band); reproduced-but-not-superior, not promoted; pinned 96-d
   stays production. R16-P0 embedding fix confirmed working.
   `docs/ai-research/progress/r110.md`.

2. **R16-P1 temporal/turn-state features — IMPLEMENTED (commit `357f0d6`).**
   v3.1 164-d temporal/turn-state vertical landed end-to-end (TS observation
   schemaVersion 2→3 → new `observation_to_features_v3_1` reusing frozen v3.0
   0–109 byte-identical → serve_onnx 164→v3.1 → 164-d ONNX export); all
   smokes green incl. real 96/110-d checkpoints. Forward line is a
   **user-gated ~0.5-day v3.1 strength ablation/gate** vs the 110-d v3.0
   baseline (DO NOT auto-launch; promote only on a gate Wilson-lower win or
   targeted-phase diagnostic gain). Canonical:
   `docs/ai-research/scoping/r16-model-feature-backlog-refinement.md`
   (P1 = IMPLEMENTED, § P1 "Resolved").

2b. **R16-P2 per-Uma slot tokens — NEXT REPRESENTATION LIFT, gated.**
   Dependency-unblocked (P0 done + P1 done) and fully scoped, but it is a
   3–4-day model/ONNX migration (signature churn, old-checkpoint
   incompatibility) and the dependency graph requires a cheap P1 signal
   baseline before P2 is *evaluated*. Do not start until the user-gated P1
   ablation gives that baseline or the user explicitly calls for it. Scope:
   same r16 doc § "P2 - Per-Uma Slot Tokens".

3. **GPU-fed stronger MCTS — P1 scoping, not implementation.**
   First prove which search configs can use GPU inference for strength:
   batched predict, hybrid learned leaves, larger sim budgets, ensembles, or
   root-search variants. Promotion requires a search-wrapped gain over
   production, not raw-policy WR. Seed:
   `docs/ai-research/scoping/gpu-fed-stronger-mcts.md`.

4. **Value/action-value data program — P1/P2.**
   Train value or action-value heads on rule-bot-covered contested states
   relabeled by rollout-leaf MCTS, not self-play-only states. This is the data
   prerequisite for any learned-leaf strength candidate.

5. **Side-conditioned sim budget — P2.**
   Run only after a larger-n rollout-leaf side split separates CIs or a
   search variant shows side-specific regression.

6. **RL/PPO from a strong search-wrapped checkpoint — P3.**
   Later strategic bet. Do not use it to bypass search-wrapped gates above.

## Training-Data / State-Coverage Backlog

**Fork A — contested-state coverage** frames items 1–2 below as its
sub-tasks. Metric = retained `>=4`-legal fraction, floor `>=30%` (audit's
existing `legal_action_count` target); cheap-experiment design and
closed-axes non-goals (no capacity, no generic volume, no `min_actions`
relaxation) in
`docs/ai-research/scoping/r16-training-data-backlog-refinement.md` (Fork A
section).

1. **Corpus retention + state-overlap audit — DONE.**
   Canonical report:
   `docs/ai-research/analysis/training-data-coverage-audit.md`. Finding:
   retained rows are bottlenecked by contested decision-state coverage, not
   schema/card-id loss or generic row count.

2. **Rule-bot-covered state corpus recipe — P1.**
   Generate deployment-relevant states, side-balanced and rule-bot-covered,
   then relabel those exact contested states with rollout-leaf MCTS or the
   strongest feasible oracle. Acceptance: retained contested-row rate >=80%,
   audit slice floors, and held-out value targets before training starts.

3. **Preference pairs on rule-bot-covered states — P2.**
   Build DPO/BT pairs only after the coverage recipe proves enough contested
   states. Acceptance: pair manifest with source-state coverage, kept-pair
   count, margin/variance filters, held-out pair/ranking accuracy, and a gate
   only after pair quality clears a pre-registered floor.

4. **Side-conditioned retained-data balancing — P2.**
   If side weakness is data-linked, balance retained player/opponent decision
   states within major phase/action buckets and require side-split gate
   reporting.

5. **Rule-bot mistake catalog + forced-state suite — P3.**
   Build hand-audited tactical states that explain where MCTS beats the rule
   bot and feed the fixture/eval tooling backlog.

## Guardrails

- Do not relax `min_actions=1`; forced single-action states carry no policy
  gradient and would dilute training.
- Do not schedule self-play-only data regeneration for raw-policy SL; the
  mcts-distill v1 failure is the canonical negative example.
- Do not re-open raw-policy SL unless a new coverage result explicitly
  falsifies the current diagnosis.
- Do not auto-launch the two pending user-gated verdicts (the W6
  regularization-dose sweep and the R16-P1 v3.1 strength ablation); they
  need explicit go-ahead. The R16-P1 serving-schema guard prerequisite is
  satisfied and P1 implementation has landed (`357f0d6`) — that old stop
  line is retired.
- Do not start R16-P2 per-Uma slot tokens (3–4-day migration) before the
  user-gated P1 signal baseline exists or the user calls for it.
- Do not re-open representation/capacity tuning off the R110 MARGINAL band;
  the forward line is the loop-recipe axis only (`docs/ai-research/progress/r110.md`).

## Historical Pointers

- R1-R14 evidence: `docs/ai-performance-research-progress.md`.
- R15 result blocks: `docs/ai-research/progress/r15.md` and
  `docs/ai-research/progress/r15-archive.md`.
- R110 W6-repro verdict + recipe mechanism: `docs/ai-research/progress/r110.md`.
- Closed scoping docs: `docs/ai-research/scoping/archive/`.
- Closed sprint/design/proposal docs: `docs/archive/ai-research/`.
