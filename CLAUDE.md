# umamusume-card-duel — Claude Code Entrypoint

Read this at the start of every Claude Code session in this repo.

## Project Shape

This is a TypeScript card-game implementation with a Python AI training stack.

- TypeScript owns game rules, legal actions, simulation, backend endpoints, and UI.
- Python owns model training, long-running AI research orchestration, ONNX export/serving helpers, and analysis scripts.
- Existing training orchestrators are the source of truth for serious loops: `training/dagger_orchestrator.py`, `training/r12_orchestrator.py`, and `training/ppo_orchestrator.py`.

Do not create a new orchestration framework unless the user explicitly asks. Prefer the existing npm and Python scripts.

## Default Workflow

Use `/work` as the default operating command. The detailed workflow lives in
`.claude/commands/work.md`; role-specific subagent instructions live in
`.claude/agents/`.

The main session owns judgment and synthesis. Keep bulky exploration, logs, and
bounded implementation detail in the appropriate subagent.

## AI Research Docs

Use `docs/ai-research/README.md` as the routing map for AI research docs:
active sources of truth, historical context, implementation entrypoints, and
write hygiene.

The governing rule is one fact, one canonical home. State files point to
evidence; they do not restate it.

## Safety And Scope

- Never revert user changes unless explicitly requested.
- Check `git status --short` before edits.
- If unrelated dirty files exist, leave them alone.
- Do not mutate historical `runs/` artifacts unless the user explicitly asks or the task is creating a new run output.
- Missing checkpoints under `runs/` are environment gaps, not product passes.
- Hooks are intentionally not part of this harness unless repeated concrete failures prove they are needed.

## Long Compute Tasks

During long jobs (training, large evals, multi-minute scripts), give frequent observable status updates — stream via `Monitor`, tail the log, or surface step/epoch ticks. Silence is a bug; "still running, step N at T+Xm" beats nothing.

## Documentation Discipline

Before writing docs, search for the topic with `rg`, then update the canonical
home from `docs/ai-research/README.md`. Archived or historical docs are
read-only context unless the task is explicitly archive cleanup.

Hard caps and state-file behavior are defined in `docs/ai-research/README.md`
and `.claude/commands/work.md`. Trim before append.
