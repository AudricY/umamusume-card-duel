---
description: Advance the AI research loop by choosing the highest-leverage next job and delegating context-heavy work to a role-specific subagent.
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

Use a subagent when the task would otherwise bloat the main context with exploration, logs, broad file reads, or implementation detail. Pick by whether the work ends in evidence or in a change:

- `investigator` — read-only. Use for code exploration, run/log analysis, and validation-as-evidence.
- `implementer` — write-capable. Use for code edits, doc updates, and validation cycles after changes.

Small direct edits or simple state updates can stay in the main session.

Give the subagent a scoped brief with the objective, relevant context, write scope (for the implementer), notable constraints, and expected output. Let it decide the detailed procedure.

Size the brief to the next external gate (training run, re-extraction, verdict, human input) — not to the next code unit that smokes green. If nothing real separates two phases of work, brief them as one step.

Use multiple subagents only for genuinely independent, low-resource tasks with disjoint write scopes.

## Synthesis

When the subagent returns, spot-check outputs, inspect changed files, and reconcile inconsistencies. Do not replay the whole exploration without a concrete reason.

Persist useful state where it belongs. Follow CLAUDE.md "Documentation Discipline" — one canonical home per fact, hard caps, trim before append.

- `queue.json` — operating priorities; `summary`/`next_action` are pointers.
- `escalations.md` — blockers; ≤500 chars per bullet.
- `digests/YYYY-MM-DD.md` — daily index; one slot ≤8 lines.
- `docs/ai-research/scoping/<topic>.md` — scoping docs (not `notes.md`).
- `docs/ai-research/progress/r<N>.md` — phase result writeups (R15+).
- Backlog stays forward-looking.

Commit any coherent workstream result (code + related docs) before continuing or stopping; split by workstream and follow the repo's commit-message convention. This overrides the global "never commit unless asked" default for `/work` runs.

Continue immediately when the next step is clearly still the same workstream and cost, risk, and context budget remain reasonable. Spawn another single subagent (investigator or implementer) if that next step would otherwise bloat the main context.

Use `/loop`, a timeout, or a timed wakeup only for genuine waits: running training/eval jobs, future logs/checkpoints, human input, or deliberate resource cool-downs.

## Response

Close with a concise summary of the job chosen, result, files changed, validation, commit(s) made, and next action. Do not exit with an uncommitted coherent result. Stop when the workstream is complete, the next step requires waiting, or continuing would exceed sensible cost/risk/context budget.
