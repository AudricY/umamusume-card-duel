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

The main Claude session should stay small:

1. Read only enough repo state to choose a bounded next job.
2. Spawn one general `worker` subagent with a precise brief.
3. Wait for the worker.
4. Read the worker's concise summary and any changed files.
5. Persist only useful conclusions in `docs/ai-agent-state/`.
6. Stop unless the user explicitly asks to keep going.

The worker is the default way to handle context-heavy exploration. The point is context isolation, not parallelism. Do not fan out multiple workers unless the tasks are independent, low-resource, and have disjoint write scopes.

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
