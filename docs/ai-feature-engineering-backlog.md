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

0. **Schema-contract hardening preflight.**
   Add one guard that checks TS action/state feature constants, Rust policy
   feature dispatch, Python `ACTION_DIM`/state schema constants, card vocab
   hash, checkpoint metadata, and ONNX metadata before training/export/eval.
   Acceptance: a planted mismatch fails clearly before any long run starts.
   Queue: `schema-contract-hardening`.

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

7. **Rust matchup balance dashboard.**
   Use Rust `sim-eval-gate --deck-sampling=uniform` manifests as the source of
   truth for per-matchup balance. Report Wilson interval, side split, average
   points, terminal reason, turn count, fallback/no-op count, and hardest/easiest
   matchup deltas. Acceptance: one generated report from an existing or new
   uniform gate manifest. Queue: `matchup-balance-dashboard`.

8. **Eval manifest comparator.**
   Add a tool that compares gate manifests and emits WR/Wilson deltas, side
   splits, points, terminal reasons, fallback/no-op counts, latency/progress
   timing where available, and exact config differences.

9. **AI investigation workbench.**
   Promote `AiTelemetryPanel` from raw JSON to grouped decision audits with
   export/import of a single state, so bad decisions can become fixtures.

10. **Adaptive difficulty profiles.**
   Expose transparent profiles such as Beginner = rule bot, Standard =
   value-head/adaptive fallback, Expert = rollout-leaf. Persist the choice
   locally.

11. **GPU inference execution provider for sim/gate throughput.**
    Add opt-in `--device cuda` to sim-eval-gate / sim-mcts-selfplay so ONNX
    inference can run on the local NVIDIA RTX 5000 Ada (or any CUDA box).
    Lifts the `Mutex<Session>` serialization that caps Slice 3c parallelism
    at 4.5× on 8 workers. Acceptance: ≥1.5× wallclock improvement at
    workers=16 on n=200 eval-gate, wilson_lower agreement within ±0.02 of
    CPU path. CPU path stays default + FP-deterministic vs serve_onnx.
    Scoping: `docs/ai-research/scoping/gpu-inference-execution-provider.md`.
    Distinct from gpu-fed-stronger-mcts.md (strength axis).

## P2 - Measurement, Data Quality, And Deck UX

12. **Decision trace schema validator.**
    Validate training examples, traces, self-play rows, outcomes, and manifests
    for seed/source/episode fields, schema-feature mismatch, card vocab ids,
    malformed legal actions, and invalid selected actions.

13. **Benchmark artifact index.**
    Generate a searchable index of canonical manifests and reports, including
    config, seed ranges, side split, model artifact, and headline result.

14. **OOD and coverage drift gate.**
    Compare new data/eval traces against the coverage audit slices and fail
    fast on drift in contested decision-state coverage.

15. **Rich data bank at new throughput + deck-variety regime.**
    Contingent P2 data-direction probe (queue
    `rich-data-bank-at-throughput-variety-regime`). Tests whether the R7
    raw-policy SL plateau (wl <= 0.33 across R7/R8/R7.b.2/mcts-distill-v1) is
    a corpus-distribution artifact, not a model-capacity or architecture
    limit. The existing-corpus data axis was DONE-NEGATIVE across chunks
    5a-i + 5j on the 3a relabel corpus (model-policy-driven rollouts, single
    matchup, ~12k rows). Two unlocks landed 2026-05-22 make a fundamentally
    different corpus tractable: G5 + Slice 3 Rust path (~140-220x TS engine,
    7.11x at workers=16) and `deck-pair-sampling` Slice 2 default-on uniform
    selfplay (22/22 matchup coverage). The conjunction rule-bot-driven x
    deck-uniform x high-sim relabel x 50k+ games x fresh raw-policy SL is
    novel under the new regime. Acceptance: `wl >= 0.40` reopens raw-policy
    SL. Falsification (`wl < 0.30`) combined with attention-probe Slice 2
    closes the raw-policy ceiling as fundamental. CONTINGENT on
    `set-attention-architecture-probe` Slice 2 falsifying OR explicit user
    gate; promotes P2 -> P1 on falsification. Architecture axis sibling, not
    a duplicate of the attention probe. Scope:
    `docs/ai-research/scoping/rich-data-bank-at-throughput-variety-regime.md`.

16. **Deck Doctor.**
    Analyze deck composition and suggest legal, concrete improvements using
    card roles, energy curve, attacker lines, and trainer/supporter balance.

17. **Tutorial scenarios with AI feedback.**
    Build small forced scenarios that teach tactical lessons using the same
    fixture machinery as the forced-state suite.

## P3 - Later Bets

18. **GPU-fed MCTS scaling probe.**
    Tooling support for the research line in `docs/ai-research-backlog.md`;
    promote only if it changes the search-wrapped strength/latency frontier.

19. **Search ablation matrix.**
    Compare rollout leaf, value leaf, hybrid leaf, adaptive halt, simulation
    budgets, and root variants under named benchmarks.

20. **Rule-bot-covered relabel corpus pipeline.**
    Productionize the P1 data recipe after the research acceptance criteria
    prove it is worth keeping.

21. **Post-game why-did-I-lose investigator.**
    Summarize concrete game-swing moments from logs, legal actions, and MCTS
    deltas without overclaiming hidden intent.

22. **Raw-policy reopen gate.**
    A placeholder only: require new coverage evidence and a pre-registered
    reopen criterion before spending compute on raw-policy SL again. (Item
    15 is the active execution path for this gate; this item remains as the
    generic placeholder for future reopen criteria.)

## Closed Pointer

- Corpus retention and state-coverage audit is done. Canonical report:
  `docs/ai-research/analysis/training-data-coverage-audit.md`.
