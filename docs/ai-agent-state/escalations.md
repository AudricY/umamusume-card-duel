# Claude Harness Escalations

## Open

- **2026-05-18 stop line:** Do not land R16-P1 feature/schema edits until the
  serving-schema 96/110/164 guard prerequisite is in place. R110 verdict is
  now reached (MARGINAL, non-blocking) so it no longer gates P1; the schema
  guard still does. Details: `docs/ai-agent-state/queue.json`,
  `docs/ai-research/scoping/r16-model-feature-backlog-refinement.md`.

## Resolved Pointers

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
