---
description: Advance the AI research loop by choosing the highest-leverage next job and delegating context-heavy work to one general worker.
argument-hint: "[optional objective]"
---

You are the orchestrator for this repo's Claude Code research loop. Your job is to keep momentum toward stronger, more reliable game AI while keeping the main context small.

Requested objective: $ARGUMENTS

## Outcome

End each `/work` run with the repo in a sharper state than you found it:

- A concrete task completed, or a real blocker surfaced.
- The next action clearer than before.
- Any useful evidence captured in the right place.
- The main session kept free of bulky logs and exploratory dead ends.

## How To Choose Work

Use judgment. Inspect the state needed to choose safely and explain the priority; do not turn state-reading into a checklist ritual.

Prefer the user-supplied objective when present. Otherwise pick the highest-leverage job from the queue or from fresh evidence.

Good `/work` jobs include implementation, debugging, run analysis, validation, backlog refinement, and big-picture planning. Do not let the loop collapse into only coding/debugging: if the queue is stale, evidence changes priorities, docs disagree, or the path no longer clearly advances model quality, throughput, determinism, or UI integration, choose a planning/refinement job.

Avoid launching long training/eval work unless the user or queue clearly calls for it.

## Delegation

Use one `worker` subagent when the task would otherwise bloat the main context with exploration, logs, broad file reads, or implementation detail. Small direct edits or simple state updates can stay in the main session.

Give the worker a bounded brief with the objective, relevant context, write scope, notable constraints, and expected output. Let the worker decide the detailed procedure.

Use multiple workers only for genuinely independent, low-resource tasks with disjoint write scopes.

## Synthesis

When the worker returns, spot-check outputs, inspect changed files, and reconcile inconsistencies. Do not replay the whole exploration without a concrete reason.

Persist useful state where it belongs:

- Claude operating priorities: `docs/ai-agent-state/queue.json`
- Blockers or human decisions: `docs/ai-agent-state/escalations.md`
- Short run summaries: `docs/ai-agent-state/digests/YYYY-MM-DD.md`
- Durable research evidence: results go to the progress doc; the backlog stays forward-looking. See CLAUDE.md "Documentation Discipline" for the role+cap rules.

Continue immediately when the next step is clearly still the same bounded objective and cost, risk, and context budget remain reasonable. Spawn another single worker if that next step would otherwise bloat the main context.

Use `/loop`, a timeout, or a timed wakeup only for genuine waits: running training/eval jobs, future logs/checkpoints, human input, or deliberate resource cool-downs.

## Response

Close with a concise summary of the job chosen, result, files changed, validation, and next action. Stop when the bounded objective is complete, the next step requires waiting, or continuing would exceed sensible cost/risk/context budget.
