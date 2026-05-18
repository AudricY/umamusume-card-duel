# Claude Harness Escalations

## Open

- _(none)_

## Resolved Pointers

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
