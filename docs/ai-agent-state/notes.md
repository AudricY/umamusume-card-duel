# Claude Harness Notes

Keep this file for durable harness conventions that do not belong in sprint
plans, progress docs, scoping docs, or digests. State here should be stable
operating guidance, not research evidence.

## Active Pointers

- R110 W6 reproduction: see `docs/ai-agent-state/queue.json` and
  `docs/ai-research/scoping/r110-w6-reproduction.md`.
- R16 P1 temporal/turn-state features: blocked until R110 verdict; see
  `docs/ai-research/scoping/r16-model-feature-backlog-refinement.md`.

## Canonical Homes

- Research doc routing and caps: `docs/ai-research/README.md`.
- Harness workflow and launch/PID convention: `docs/claude-loop-harness.md`.
- R15/F1 result history: `docs/ai-research/progress/r15.md`,
  `docs/ai-research/progress/r15-archive.md`, and
  `docs/ai-performance-research-progress.md`.
- Current model-strength frontier: `docs/ai-research-backlog.md`.

## Durable Harness Conventions

- Default command: `/work`.
- Default execution path: one role-specific subagent (`investigator` or
  `implementer`) when exploration or implementation detail would bloat the
  main context.
- Subagents provide context isolation, not automatic parallel throughput.
- Do not add hooks, a Python harness, or additional slash commands unless
  repeated concrete failures prove the need.
