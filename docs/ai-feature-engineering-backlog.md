# AI Feature Engineering Backlog

This backlog tracks AI tooling, UX, eval hardening, and production-hardening
work. Model-strength research lives in `docs/ai-research-backlog.md`.

## Constraints

- Production strength is rollout-leaf search-wrapped MCTS, not a stronger raw
  policy.
- Raw-policy SL/BC/DPO is closed as a forward line unless new coverage evidence
  explicitly re-opens it.
- The next data question is contested state coverage, not generic row count.
- Side asymmetry should be measured and disclosed.
- Player-facing AI features should reuse legal-action enumeration, `/ai/decide`,
  MCTS diagnostics, deck data, and existing AI telemetry before adding new
  model families.

## P0 - Highest Leverage

1. **Canonical benchmark registry.**
   Define named protocols such as `rulebot_side_balanced_n120`,
   `rulebot_side_balanced_n1000`, `production_rollout_leaf_smoke`,
   `cheap_value_head_fallback`, `side_split_probe`, and
   `forced_state_tactics`. Each benchmark records seeds, sides, config, min
   games, thresholds, and artifact expectations. Acceptance: checked-in
   registry consumed by at least one report or smoke command.

2. **Forced-state tactical benchmark suite.**
   Create roughly 50 replayable states grouped by missed lethal, bad retreat,
   energy attach, trainer sequencing, ability timing, over-benching, backup
   attacker setup, no-op/pass avoidance, and prize-race errors. Acceptance:
   fixture manifest, expected action set per state, runnable benchmark, and
   corpus-slice mapping.

3. **Explainable opponent decisions.**
   Extend `/ai/decide` behind an opt-in flag to return top-N legal actions,
   selected action id, visit share or score, simulations, latency, and factual
   labels. Surface it in battle log or AI telemetry without claiming hidden
   intent.

4. **Coach My Turn.**
   Add an opt-in hint request for the player side. Return the best legal action
   plus alternatives and short factual labels. Validate responses against a
   state hash so stale hints cannot be applied.

## P1 - Production Hardening And UX

5. **Production MCTS latency and stability harness.**
   Run fixed-seed decision probes recording wall time, node count, legal-action
   count, phase/action kind, adaptive halt status, fallback status, selected
   rank, and chosen action. Acceptance: latency percentiles and fallback/no-op
   counts by bucket plus raw trace artifact.

6. **Side-asymmetry larger-N audit.**
   Run or script a larger side-balanced split for rollout-leaf and the cheap
   value-head fallback. Bucket losses by phase, action kind, turn band,
   legal-action count, and setup/first-mover features. Acceptance: state
   whether CIs separate and list contributing buckets.

7. **Eval manifest comparator.**
   Add a tool that compares gate manifests and emits WR/Wilson deltas, side
   splits, points, terminal reasons, fallback/no-op counts, latency/progress
   timing where available, and exact config differences.

8. **AI investigation workbench.**
   Promote `AiTelemetryPanel` from raw JSON to grouped decision audits with
   export/import of a single state, so bad decisions can become fixtures.

9. **Adaptive difficulty profiles.**
   Expose transparent profiles such as Beginner = rule bot, Standard =
   value-head/adaptive fallback, Expert = rollout-leaf. Persist the choice
   locally.

## P2 - Measurement, Data Quality, And Deck UX

10. **Decision trace schema validator.**
    Validate training examples, traces, self-play rows, outcomes, and manifests
    for seed/source/episode fields, schema-feature mismatch, card vocab ids,
    malformed legal actions, and invalid selected actions.

11. **Benchmark artifact index.**
    Generate a searchable index of canonical manifests and reports, including
    config, seed ranges, side split, model artifact, and headline result.

12. **OOD and coverage drift gate.**
    Compare new data/eval traces against the coverage audit slices and fail
    fast on drift in contested decision-state coverage.

13. **Deck Doctor.**
    Analyze deck composition and suggest legal, concrete improvements using
    card roles, energy curve, attacker lines, and trainer/supporter balance.

14. **Tutorial scenarios with AI feedback.**
    Build small forced scenarios that teach tactical lessons using the same
    fixture machinery as the forced-state suite.

## P3 - Later Bets

15. **GPU-fed MCTS scaling probe.**
    Tooling support for the research line in `docs/ai-research-backlog.md`;
    promote only if it changes the search-wrapped strength/latency frontier.

16. **Search ablation matrix.**
    Compare rollout leaf, value leaf, hybrid leaf, adaptive halt, simulation
    budgets, and root variants under named benchmarks.

17. **Rule-bot-covered relabel corpus pipeline.**
    Productionize the P1 data recipe after the research acceptance criteria
    prove it is worth keeping.

18. **Post-game why-did-I-lose investigator.**
    Summarize concrete game-swing moments from logs, legal actions, and MCTS
    deltas without overclaiming hidden intent.

19. **Raw-policy reopen gate.**
    A placeholder only: require new coverage evidence and a pre-registered
    reopen criterion before spending compute on raw-policy SL again.

## Closed Pointer

- Corpus retention and state-coverage audit is done. Canonical report:
  `docs/ai-research/analysis/training-data-coverage-audit.md`.
