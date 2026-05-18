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
- **User reprioritization (2026-05-19): hyperparameter tuning is
  deprioritized.** Land ALL model-feature-upgrade and training-data-upgrade
  items first, then return to the W6 regularization-dose sweep. The active
  frontier is now (1) the cheap contested-state coverage pilot, then (2) the
  R16-P2 per-Uma slot-token model-feature migration, then downstream data
  items; the W6 loop anti-degradation **regularization-dose sweep is the
  deprioritized HP tuning** and runs only after that block lands (still
  user-gated). R16-P1 temporal/turn-state features were implemented and
  **ablated NO-GO** (v3.1 ≈ v3.0, no Wilson-lower win;
  `docs/ai-research/progress/r16.md`). Live operational state and forward
  order are in `docs/ai-agent-state/queue.json`.

## Active Search-Wrapped Frontier

0. **Contested-state coverage pilot — TOP PRIORITY (training-data
   upgrade).** The R16-P1 NO-GO resolved the pre-registered contingency: the
   cheap single-variable coverage pilot is now the highest-leverage forward
   line, and the 2026-05-19 user reprioritization confirms training-data and
   model-feature upgrades land before any HP tuning. No new data — a
   `training/uma_ai/dataset.py` loader/loss change only. Canonical:
   Training-Data / State-Coverage Backlog item 2 below and
   `docs/ai-research/scoping/r16-training-data-backlog-refinement.md`; queue
   `training-data-coverage-pilot`.

0a. **R16-P2 per-Uma slot tokens — model-feature upgrade, user-gate now
   OPEN.** The 2026-05-19 "land all model feature upgrade items first"
   directive is the explicit user call the prior guardrail required. A 3–4-day
   model/ONNX migration; sequence *after* the cheap coverage pilot
   (leverage-per-cost). Scope: `docs/ai-research/scoping/r16-model-feature-backlog-refinement.md`
   § "P2 - Per-Uma Slot Tokens"; queue `r16-p2-per-uma-slot-tokens`.

0b. **W6 loop anti-degradation recipe — DEPRIORITIZED (regularization-dose
   sweep = the deprioritized HP tuning).** Demoted from TOP PRIORITY to P3 by
   the 2026-05-19 user reprioritization: runs only AFTER all three core
   predecessors land (coverage pilot → R16-P2 → deep data program: corpus
   recipe → preference/value-data), and stays user-gated. Gate-depth =
   option B: it is **NOT** gated on the conditional side-balancing item or
   the P3 manual mistake catalog (those fire on their own triggers; gating
   W6 behind the conditional item could block it indefinitely). Operational
   predecessor set is canonical in `docs/ai-agent-state/queue.json`. Mechanism still
   CONFIRMED via R111 (iter-3 rot eliminated; recipe-fix landed commit
   `d44de3f`, cross-iter replay buffer + fixed iter-0/SL KL anchor in
   `r12_orchestrator.py`) but over-damps at default HP — policy froze then
   decayed (0.5527→0.5358), below the 0.6042 baseline iter-2 peak. Reopen
   knobs: lower/anneal the fixed-KL-anchor weight from 0.05 and/or reduce
   replay old-fraction (0.40) / window (3). Canonical:
   `docs/ai-research/progress/r110.md` §4c; queue `w6-loop-anti-degradation`.

0c. **Fork B — label-quality test matrix (umbrella over the W6 dose
   sweep).** Raises target/teacher quality at *fixed* capacity and volume
   (both axes closed); the M6 anti-drift arm is the W6 dose sweep, not a
   re-plan. Deprioritized with the W6 sweep above; gated behind a
   near-zero-cost read-only label-quality probe before any user-gated loop
   compute. Canonical:
   `docs/ai-research/scoping/forkb-label-quality-loop-recipe.md`.

1. **R110 W6 reproduction — DONE.**
   Verdict MARGINAL (best-promoted iter-2 Wilson lower 0.6042, in the
   0.60-0.6479 band); reproduced-but-not-superior, not promoted; pinned 96-d
   stays production. R16-P0 embedding fix confirmed working.
   `docs/ai-research/progress/r110.md`.

2. **R16-P1 temporal/turn-state features — ABLATED NO-GO.**
   v3.1 164-d implemented end-to-end (commits `357f0d6`/`3ce1404`); the
   faithful strength ablation produced **no Wilson-lower win** (v3.1 ≈ v3.0,
   best-promoted 0.5955 < v3.0 0.6042). v3.1 NOT promoted; production stays
   pinned 96-d. Null strength signal. Canonical:
   `docs/ai-research/progress/r16.md`.

2b. **R16-P2 per-Uma slot tokens — user-gated larger lift, NOT auto.**
   Dependency-unblocked (P0/P1 done) and fully scoped, but P1's strength
   signal was **null**, so P2 no longer has a cheap-P1-baseline rationale.
   It remains a 3–4-day model/ONNX migration (signature churn,
   old-checkpoint incompatibility); do not start unless the user explicitly
   calls for it. Scope: r16 scoping doc § "P2 - Per-Uma Slot Tokens".

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
   reporting. **Conditional + NOT a W6 predecessor** (option B): it fires
   only if side weakness proves data-linked; gating the W6 sweep behind it
   could block W6 indefinitely.

5. **Rule-bot mistake catalog + forced-state suite — P3.**
   Build hand-audited tactical states that explain where MCTS beats the rule
   bot and feed the fixture/eval tooling backlog. **P3 manual + NOT a W6
   predecessor** (option B): fires on its own track, not in the HP-gate
   chain.

## Guardrails

- Do not relax `min_actions=1`; forced single-action states carry no policy
  gradient and would dilute training.
- Do not schedule self-play-only data regeneration for raw-policy SL; the
  mcts-distill v1 failure is the canonical negative example.
- Do not re-open raw-policy SL unless a new coverage result explicitly
  falsifies the current diagnosis.
- Do not auto-launch the W6 regularization-dose sweep. Per the 2026-05-19
  user reprioritization it is DEPRIORITIZED (P3): it runs only AFTER all
  three core predecessors land (coverage pilot → R16-P2 → deep data
  program) AND with explicit go-ahead. Gate-depth = option B: do NOT extend
  the W6 gate to the conditional side-balancing item or the P3 manual
  mistake catalog.
  (The R16-P1 v3.1 ablation is now CLOSED = NO-GO; no auto-launch remains on
  that line — `docs/ai-research/progress/r16.md`.)
- R16-P2 per-Uma slot tokens: the prior "do not start unless the user
  explicitly calls for it" gate is now SATISFIED by the 2026-05-19
  reprioritization. Still sequence it AFTER the cheap coverage pilot
  (leverage-per-cost); it remains a 3–4-day model/ONNX migration.
- Do not re-open representation/capacity tuning off the R110 MARGINAL band;
  the forward line is the loop-recipe axis only (`docs/ai-research/progress/r110.md`).

## Historical Pointers

- R1-R14 evidence: `docs/ai-performance-research-progress.md`.
- R15 result blocks: `docs/ai-research/progress/r15.md` and
  `docs/ai-research/progress/r15-archive.md`.
- R110 W6-repro verdict + recipe mechanism: `docs/ai-research/progress/r110.md`.
- R16-P0/P1 results (P1 v3.1 ablation NO-GO): `docs/ai-research/progress/r16.md`.
- Closed scoping docs: `docs/ai-research/scoping/archive/`.
- Closed sprint/design/proposal docs: `docs/archive/ai-research/`.
