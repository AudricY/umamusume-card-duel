---
description: Pick one useful AI research job, delegate it to the general worker, then persist the result.
argument-hint: "[optional objective]"
---

You are the thin orchestrator for this repo's Claude Code loop. Keep your own context small. Your default behavior is to spawn exactly one `worker` subagent to do the context-heavy work.

Requested objective: $ARGUMENTS

## Read State

Read only enough to choose and brief one bounded job:

- `git status --short`
- `docs/ai-agent-state/queue.json`
- `docs/ai-agent-state/escalations.md`
- `docs/ai-agent-state/notes.md`
- Tail of `docs/ai-performance-research-progress.md`
- Current sprint/backlog docs when relevant
- Recent run state only when relevant: `runs/*/orchestrator-state.json`, `runs/*/events.jsonl`

Do not deep-dive in the main context. If a file/log/run needs real inspection, put that in the worker brief.

## Pick One Job

If the user supplied an objective, use it unless it is unsafe or impossible.

Otherwise choose the queue head unless current repo state makes another job clearly higher value.

Good jobs are bounded:

- Diagnose one failing smoke.
- Inspect one run directory.
- Make one small code/doc fix.
- Update backlog/progress docs from one accepted result.
- Run or inspect one appropriate validation tier.
- Identify the next concrete step when the queue is stale.

Avoid starting long training/eval work unless the queue or user explicitly calls for it.

## Brief The Worker

Spawn the `worker` subagent with:

- Objective.
- Why this job is next.
- Files, docs, or run directories to inspect.
- Allowed write scope.
- Commands allowed and commands to avoid.
- Validation expectations.
- Required final response shape.

Use one worker by default. Do not fan out multiple workers unless the tasks are independent, low-resource, and have disjoint write scopes.

## After The Worker Returns

Read the worker summary and inspect any changed files. Do not redo the worker's exploratory work unless something is clearly inconsistent.

Persist only useful state:

- Append a short note to `docs/ai-agent-state/digests/YYYY-MM-DD.md` for meaningful results.
- Update `docs/ai-agent-state/queue.json` only when the next action changed.
- Update `docs/ai-agent-state/escalations.md` only for blockers, missing artifacts, unsafe stop lines, or human decisions.
- Update canonical AI docs only when the worker produced evidence that belongs there.

## Wrap

Print a concise summary:

- Job chosen.
- Worker result.
- Files changed.
- Validation run or skipped.
- Next recommended action.

Then stop. Do not schedule another wake in the first version of this harness.
