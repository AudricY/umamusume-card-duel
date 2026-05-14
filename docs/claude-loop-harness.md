# Claude Loop Harness

This repo uses a lightweight Claude Code harness for AI research operations.

The harness is intentionally small:

- One command: `.claude/commands/work.md`
- One general worker agent: `.claude/agents/worker.md`
- One repo entrypoint: `CLAUDE.md`
- Small state files under `docs/ai-agent-state/`

There is no Python harness, no hooks, and no separate command mode per task type.

## Operating Model

The main Claude session is an orchestrator. It reads enough state to choose one bounded job, then delegates the context-heavy part to the general worker subagent.

The worker can handle any task type:

- Investigation.
- Implementation.
- Docs update.
- Run analysis.
- Validation.

The worker returns a concise summary. The orchestrator then persists only the useful result and stops.

This keeps the main context from filling with logs, long file reads, failed exploration paths, and intermediate command output.

## Parallelism

Default to one worker per `/work` run.

The worker exists for context isolation, not parallel throughput. Parallel fan-out is usually the wrong default here because training/eval jobs, run directories, checkpoints, and planning docs are shared resources.

Use multiple workers only when tasks are independent, low-resource, and have disjoint write scopes.

## State Files

```text
docs/ai-agent-state/
  queue.json
  escalations.md
  notes.md
  digests/
```

State files should stay small and readable.

- `queue.json` tracks the next few useful jobs.
- `escalations.md` tracks blocked or unsafe next steps.
- `notes.md` records durable harness notes.
- `digests/YYYY-MM-DD.md` records short summaries from meaningful runs.

Canonical model/training evidence belongs in the existing AI research docs, not only in harness state.

## Non-Goals

- Do not replace `dagger_orchestrator.py`, `r12_orchestrator.py`, or `ppo_orchestrator.py`.
- Do not add hooks unless repeated concrete mistakes show a need.
- Do not add more slash commands until `/work` becomes overloaded in practice.
- Do not add specialized subagents until the general worker repeatedly proves too broad.
