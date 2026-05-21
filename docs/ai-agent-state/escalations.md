# Claude Harness Escalations

## Open

- **2026-05-21: 3b BC-on-relabel probe FALSIFIES candidate-3 —
  corpus binding constraint, direction call needed.** Soft-CE BC on
  the 11798 relabel rows (3ep, lr 3e-4, h64 d2 dropout 0.05) lands
  `pair_accuracy` 0.6802 vs 0.6981 baseline = **−0.0179, OUTSIDE
  the ±0.022 noise band**; all 5 ranking metrics regress vs ref.
  Candidate-3 (objective mismatch fixes the DPO ceiling) is
  FALSIFIED — BC is meaningfully worse than the untrained ref, not
  flat. By elimination, candidate-1 (relabel-MCTS too weak vs ref)
  is now the binding hypothesis: the rollout-leaf 100-sim labels
  actively mislead training. Forward line is a user direction call
  between (a) re-relabel with stronger MCTS (400+ sims / value-head
  leaf) — direct candidate-1 test, or (b) W6 HP-sweep (CEILING PATH
  step A, cheap-first). Canonical:
  `docs/ai-research/progress/r16.md` § "3b — BC-on-relabel probe
  (chunk 5f)".

## Resolved Pointers

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
