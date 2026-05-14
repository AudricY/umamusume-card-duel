# umamusume-card-duel — Claude Code Entrypoint

Read this at the start of every Claude Code session in this repo.

## Project Shape

This is a TypeScript card-game implementation with a Python AI training stack.

- TypeScript owns game rules, legal actions, simulation, backend endpoints, and UI.
- Python owns model training, long-running AI research orchestration, ONNX export/serving helpers, and analysis scripts.
- Existing training orchestrators are the source of truth for serious loops: `training/dagger_orchestrator.py`, `training/r12_orchestrator.py`, and `training/ppo_orchestrator.py`.

Do not create a new orchestration framework unless the user explicitly asks. Prefer the existing npm and Python scripts.

## Default Workflow

Use `/work` as the default operating command.

The main Claude session should stay small. It owns judgment and synthesis: choose the highest-leverage bounded job, delegate context-heavy work to a role-specific subagent, then persist the useful result.

Two subagents are available:

- `investigator` — read-only. Use for deep code exploration, run analysis, log triage, and validation-as-evidence. Returns findings.
- `implementer` — write-capable. Use for code edits, doc updates (sprint/backlog/progress, queue, escalations, digests), and validation cycles after changes.

Pick by whether the job ends in evidence or in a change. The point is context isolation, not parallelism. Small direct edits and simple state updates can stay in the main session. Do not fan out multiple subagents unless the tasks are independent, low-resource, and have disjoint write scopes.

Backlog refinement and big-picture planning are normal `/work` jobs. The loop should not only debug and implement. Choose planning/refinement when the queue is stale, run evidence changes priorities, escalations block the current path, sprint/backlog/progress docs disagree, or local work is no longer clearly moving the AI objective forward.

Do not use `/loop` or a timed wakeup just to continue ordinary work. Use a timeout/wakeup only when the next useful action is blocked on wall-clock time or an external dependency, such as:

- A training/eval/background process that is still running.
- A scheduled checkpoint or log file that will exist later.
- Human input or approval.
- A deliberate cool-down after a resource-heavy run.

## Claude Harness State

Lightweight Claude operating state lives in:

```text
docs/ai-agent-state/
  queue.json
  escalations.md
  notes.md
  digests/
```

Use these files as steering aids, not as a project-management database.

- `queue.json`: small list of next useful jobs.
- `escalations.md`: blockers, unsafe stop lines, missing local artifacts, or human decisions needed.
- `notes.md`: durable harness notes that do not belong in sprint/progress docs.
- `digests/YYYY-MM-DD.md`: short summaries from meaningful `/work` runs.

Canonical AI research evidence still belongs in the existing AI docs, especially:

- `docs/ai-performance-research-harness.md`
- `docs/ai-performance-research-progress.md`
- `docs/ai-research-backlog.md`
- `docs/r14-sprint-plan.md`

Use `docs/ai-agent-state/queue.json` for immediate Claude operating priorities, and update it when planning/refinement work changes the next best action.

## Safety And Scope

- Never revert user changes unless explicitly requested.
- Check `git status --short` before edits.
- If unrelated dirty files exist, leave them alone.
- Do not mutate historical `runs/` artifacts unless the user explicitly asks or the task is creating a new run output.
- Warn before launching long GPU, training, or large evaluation jobs.
- Missing checkpoints under `runs/` are environment gaps, not product passes.
- Hooks are intentionally not part of this harness unless repeated concrete failures prove they are needed.

## Validation Tiers

Choose the smallest validation set that matches the change.

Fast repo confidence:

```bash
npm run build
TMPDIR=/tmp npm run test:train
```

Python/training smoke:

```bash
TMPDIR=/tmp npm run test:python-train
```

Orchestrator smoke:

```bash
TMPDIR=/tmp npm run test:dagger-orchestrator
```

PPO smoke:

```bash
TMPDIR=/tmp npm run test:ppo-smoke
```

Use deeper targeted smokes only when touching their area: determinism, MCTS, `/ai/decide`, ONNX serving, observability, or parallel simulation.

## Documentation Discipline

Keep docs close to the evidence.

- Update existing sprint/backlog/progress docs instead of creating new planning sprawl.
- Use `docs/ai-agent-state/digests/` for short Claude run summaries.
- Keep queue entries concise and actionable.
- Add escalations when useful work is blocked by missing artifacts, ambiguous direction, or unsafe next steps.
- `docs/ai-research-backlog.md` is forward-looking only (target ≤300 lines). Finished/failed sprint phases and dated result blocks belong in `docs/ai-performance-research-progress.md`; the backlog entry shrinks to a one-line pointer.
- Daily digests target ≤150 lines. If a doc is already past its target, roll older blocks forward before appending new content.
