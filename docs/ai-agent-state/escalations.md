# Claude Harness Escalations

## Open

- **2026-05-21: chunk 5i candidate-1 ladder step 2 = DIES; next
  direction call needed.** Step 2 (rule-bot-mirror @ 400 sims) ran
  end-to-end; argmax flip 17.3% (n=365) on contested-head + max-prob
  ≥0.8 subset — below the 30% LIVES threshold AND the 20% sticky
  cutoff. Sim count is not the binding lever. Candidate-1
  (stronger-MCTS-fixes-corpus) eliminated. Forward direction is
  either (a) reframed step 1 (rollout-leaf return audit on contested
  states, no CRN-paired ambiguity) OR (b) pivot to user-gated W6
  HP-sweep (CEILING PATH A) since the 3a corpus arm is exhausted by
  evidence. Secondary finding: relabel-MCTS RNG bleed via global
  `random()` proxy invalidates sim-count A/B determinism (tracked as
  P3 queue item `mcts-relabel-rng-bleed`). Canonical:
  `docs/ai-research/progress/r16.md` sub-§ "Step 2 verdict (chunk 5i)".

## Resolved Pointers

- 2026-05-21: 3b candidate-1 forward-line direction call RESOLVED
  (DIES at step 2). Chunk 5g labels-confidently-wrong + chunk 5h
  ladder scoping + chunk 5i argmax-flip @ 400 sims (17.3%, n=365)
  collectively eliminate candidate-1 (stronger-MCTS fixes corpus).
  Sim count not the lever. Canonical:
  `docs/ai-research/progress/r16.md` sub-§§ "Candidate-1 kill-test
  (chunk 5g)" / "Step 2 verdict (chunk 5i)".
- 2026-05-18: R16 contested-coverage pilot (Fork A cheap tier, W6
  predecessor #1) DONE = POSITIVE slope. Option1 resample sweep at fixed
  retained-count/capacity: legal_action_count 0.20/0.30/0.45 →
  side-balanced gate Wilson-lower 0.2416/0.2572/0.2635 (monotone).
  Pre-registered acceptance MET; contested-coverage hypothesis NOT
  falsified. Margin small (within single-gate noise) → next bounded step
  is a larger-n confirmation gate; expensive n≥1000 closed-loop still
  gated. Mechanisms shipped in `training/uma_ai/dataset.py` (default OFF).
  Canonical: `docs/ai-research/progress/r16.md`.
- 2026-05-19: User reprioritization — HP tuning (W6 regularization-dose
  sweep) DEPRIORITIZED P1→P3. Gate-depth = option B (user-confirmed): W6
  runs only after all three core predecessors land —
  training-data-coverage-pilot → r16-p2-per-uma-slot-tokens →
  training-data-deep-program (corpus recipe → preference/value-data).
  R16-P2 user-gate now OPEN; it is the LAST model-feature item (series =
  P0/P1/P2, no P3+). NOT W6 predecessors by explicit decision: conditional
  side-balancing (TD item 4) and P3 manual mistake catalog (TD item 5) —
  gating W6 behind the conditional item could block it indefinitely.
  Canonical: `docs/ai-agent-state/queue.json`, `docs/ai-research-backlog.md`.
- 2026-05-19: R16-P1 v3.1 strength ablation COMPLETE = NO-GO
  (best-promoted 0.5955 < v3.0 0.6042; v3.1 not promoted, 96-d pin
  unchanged). serve_onnx 24-worker gate-fan-in crash blocker resolved
  `e846881` (TCP backlog 5→128, durable harness fact); trainer-wiring
  gap resolved `3ce1404` (`--state-dim` + delta-0.0 additive init).
  Canonical: `docs/ai-research/progress/r16.md`.
- 2026-05-18: R16-P1 stop line CLEARED — serving-schema 96/110/164 guard
  prerequisite IMPLEMENTED (commits `66adca6`/`692091d`); R110 verdict
  MARGINAL/non-blocking. P1 implementation unblocked. Canonical:
  `docs/ai-research/scoping/r16-model-feature-backlog-refinement.md`
  § "Load-bearing implementation prerequisite".
- 96-dim serving pin: `docs/ai-research/progress/r15.md`.
- Raw-policy SL line closed; search-wrapped path is the forward line:
  `docs/ai-research-backlog.md`.
- 2026-05-18: R110 MARGINAL (not a FAIL → no blocker). Research-direction
  call — W6 loop anti-degradation recipe-fix line, discriminator-gated,
  declared a loop-recipe axis distinct from the closed representation axis
  — made self-directed by the main session with rationale:
  `docs/ai-research/progress/r110.md`.
- 2026-05-18: W6 discriminator gate SATISFIED (recipe-rot = monotone
  representation drift from iter-0, not an iter-2 peak). Forward line is
  the implementer r12_orchestrator.py recipe-fix (cross-iter replay
  buffer + fixed iter-0/SL KL anchor). Canonical:
  `docs/ai-research/progress/r110.md` §4a.
- 2026-05-18: W6 recipe-fix mechanism CONFIRMED via R111 full loop
  (iter-3 rot eliminated) but it over-damps at default HP — net ceiling
  loss vs baseline 0.6042 (frozen 0.5527 then decay 0.5358). Forward
  line is now a user-gated regularization-dose sweep. Canonical:
  `docs/ai-research/progress/r110.md` §4c.
- 2026-05-18: REPRIORITIZATION (evidence-driven, OVERRIDES 2026-05-19
  user-confirmed sequence — pending user veto). Sizing pass falsified the
  "TD-3a = heavy/INFRA-BLOCKED" basis: MCTS already runs on live
  GameState; emit-path is ~2-4d cheap plumbing, a shared 3a+3b
  unblocker. 3a moved AHEAD of `r16-p2-per-uma-slot-tokens`; 3a
  emit-path + 50-game pilot + Audit-v2 LANDED (all PASS); larger-n
  coverage gate decoupled. Canonical: `docs/ai-research/progress/r16.md`.
