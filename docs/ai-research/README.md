# AI Research Docs Index

This directory is the map for current AI research documentation. Use it before
adding or moving research docs.

## Current Sources Of Truth

| Question | Canonical home |
| --- | --- |
| What is the active research frontier? | `docs/ai-research-backlog.md` |
| What AI tooling, UX, eval, or production-hardening work is queued? | `docs/ai-feature-engineering-backlog.md` |
| What happened in a completed R15+ phase? | `docs/ai-research/progress/r<N>.md` |
| What is the pre-registered hypothesis, sweep recipe, or exit gate for a topic? | `docs/ai-research/scoping/<topic>.md` |
| What durable analysis report supports a backlog item? | `docs/ai-research/analysis/<topic>.md` plus any sidecar JSON |
| What is the live agent queue or blocker state? | `docs/ai-agent-state/queue.json` and `docs/ai-agent-state/escalations.md` |
| What are durable harness conventions? | `docs/ai-agent-state/notes.md` |

## Historical Sources

- `docs/ai-performance-research-progress.md` is the R1-R14 historical record
  and should be treated as read-mostly.
- Closed sprint plans, old proposals, F1 design notes, and superseded harness
  proposals are historical context. Do not use them as active planning sources.
- Archive index: `docs/archive/README.md`.
- Closed scoping docs: `docs/ai-research/scoping/archive/`.
- Historical docs may link to old paths. Prefer updating active docs to point
  here instead of rewriting old evidence.

## Implementation Entry Points

- TypeScript simulator, search, and eval: `backend/src/sim/`
- Live AI endpoint: `backend/src/server.ts`
- Frontend AI policy and telemetry: `frontend/src/game/engine/ai-policy/`,
  `frontend/src/app/AiTelemetryPanel.tsx`
- Python training, serving, and orchestration: `training/`
- Model/data/schema internals: `training/uma_ai/`
- Training setup, artifact layout, and observability: `training/README.md`

## Write Hygiene

Before writing a new research fact, search for the topic and update the
canonical home instead of creating a second narrative copy. State files should
link to canonical evidence; they should not restate it.

When a file hits its cap, trim before appending:

- Digest: <=150 lines.
- Backlog: <=300 lines.
- Per-sprint progress file: <=500 lines.
- Per-sprint progress file: roll older result blocks to a sibling archive file.
- Queue: digest completed entries, then remove them.
- Escalations: collapse resolved items to one-line links.
- Notes: keep only durable harness conventions; move scoping/results elsewhere.
