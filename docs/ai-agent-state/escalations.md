# Claude Harness Escalations

## Open

- **2026-05-21: chunk 5j n=1000 FALSIFIED contested-coverage.**
  Both autonomous P1 data-side lines exhausted: coverage-pilot →
  done-falsified; deep-program → done-negative. n=1000 sign-flipped
  margin (−1.8pp vs pilot +2.2pp). Only autonomous-actionable work
  is diagnostic-only reframed step 1. Canonical: r16.md § "Chunk 5j".

## Resolved Pointers

- 2026-05-21: 3b corpus arm DONE-NEGATIVE direction call RESOLVED;
  both autonomous P1 data-side forward lines now closed end-to-end
  (training-data-deep-program done-negative chunks 5a-i; sibling
  training-data-coverage-pilot done-falsified at chunk 5j n=1000
  confirmation gate). Canonical: r16.md chunks 5a-j.

- 2026-05-21: 3b candidate-1 forward-line direction call RESOLVED
  (DIES at step 2). Chunk 5g labels-confidently-wrong + chunk 5h
  ladder scoping + chunk 5i argmax-flip @ 400 sims (17.3%, n=365)
  collectively eliminate candidate-1 (stronger-MCTS fixes corpus).
  Sim count not the lever. Canonical:
  `docs/ai-research/progress/r16.md` sub-§§ "Candidate-1 kill-test
  (chunk 5g)" / "Step 2 verdict (chunk 5i)".
- 2026-05-18: R16 contested-coverage pilot (Fork A cheap tier) DONE
  positive-slope at n=300 — later FALSIFIED at chunk 5j n=1000 gate (see
  open bullet above). Canonical: r16.md.
- 2026-05-19: User reprioritization — HP tuning (W6 dose sweep)
  DEPRIORITIZED P1→P3 (option B gate-depth); R16-P2 user-gate OPEN as
  last model-feature item. Canonical: queue.json, ai-research-backlog.md.
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
