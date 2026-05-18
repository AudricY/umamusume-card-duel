# AI Research Backlog

This is the forward-looking model-strength research backlog. Keep it compact:
landed or failed work shrinks to a pointer, and detailed mechanisms/numbers live
in progress, scoping, or analysis docs. See `docs/ai-research/README.md`.

## Current Anchor

As of 2026-05-18:

- **Production strength comes from search-wrapped play.** Rollout-leaf MCTS at
  W6 iter-2 is the current production claim: Wilson lower 0.6479 vs rule bot
  at the side-balanced n=120 gate. Canonical details:
  `docs/ai-research/progress/r15.md` and the R1-R14 historical record in
  `docs/ai-performance-research-progress.md`.
- **The search-free/raw-policy SL line is closed.** Labels, objective,
  representation, and label-shape all failed to approach the 0.40 gate
  (R7/R8/R7.b.2/mcts-distill v1). Do not spend compute on another isolated
  raw-policy tweak without explicit new evidence.
- **Cheap inference remains a fallback, not the production claim.** Value-head
  leaf plus adaptive-ratio=1.5 sits around the 0.39-0.45 Wilson-lower envelope
  across seed ranges and is only a latency fallback.
- **Side asymmetry is real enough to disclose.** Current rollout-leaf evidence
  is aggregate side-balanced strength, not a per-side guarantee.
- **The active frontier is 110-d/v3 search-wrapped reproduction and then
  stronger search.** Live operational state is in
  `docs/ai-agent-state/queue.json`.

## Active Search-Wrapped Frontier

1. **R110 W6 reproduction — ACTIVE.**
   Re-run the W6 rollout-leaf self-play plus mcts-distill loop at 110-d/v3
   after the R16-P0 embedding-path fix. Success reproduces or beats the 96-d
   production search-wrapped model and unlocks the v3 promotion path. Scoping:
   `docs/ai-research/scoping/r110-w6-reproduction.md`.

2. **R16-P1 temporal/turn-state features — BLOCKED.**
   Additive-tail feature schema work is ready, but it must wait for the R110
   verdict and a serving-schema guard so 96/110/164-d models resolve safely.
   Scoping: `docs/ai-research/scoping/r16-model-feature-backlog-refinement.md`.

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
- Do not land R16-P1 schema/feature edits while R110 is running.

## Historical Pointers

- R1-R14 evidence: `docs/ai-performance-research-progress.md`.
- R15 result blocks: `docs/ai-research/progress/r15.md` and
  `docs/ai-research/progress/r15-archive.md`.
- Closed scoping docs: `docs/ai-research/scoping/archive/`.
- Closed sprint/design/proposal docs: `docs/archive/ai-research/`.
