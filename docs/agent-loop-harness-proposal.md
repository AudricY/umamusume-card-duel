# Claude Code Loop Harness Proposal

Created 2026-05-14. Revised after review to bias toward first-class Claude Code support and a lightweight harness.

## Goal

Bring the useful parts of the Claude Code looping-agent harness into this repo as a small Claude-native operating layer for AI research work.

This should not become a Codex abstraction, a Python framework, or another training orchestrator. The repo already has serious orchestration in `training/dagger_orchestrator.py`, `training/r12_orchestrator.py`, and `training/ppo_orchestrator.py`. The missing piece is a thin Claude Code control surface that helps choose, run, and document the next useful research task.

## Main Decision

Build this as first-class Claude Code support:

- Use one Claude command: `.claude/commands/work.md`.
- Use a minimal `.claude/agents/*.md` only if one clearly pays for itself.
- Use `CLAUDE.md` as the repo entrypoint.
- Reuse existing npm, Python, and training scripts directly.
- Keep persistent state as plain markdown/JSON files, not a new service.

Do not build a general Codex/Codex-compatible harness unless there is a later reason to support multiple agent runtimes.

## What To Transfer

Transfer these ideas from the existing Claude harness:

- A `/work` command that reads current repo state, picks the next useful job, runs it, summarizes what changed, and continues immediately while the next step is available now.
- One general worker subagent that executes the selected job while the main Claude session stays small.
- Durable state over chat memory: queue, escalations, progress notes, run summaries.
- A concise operating contract so future Claude sessions know how to work in this repo.

Do not transfer:

- Bug-bounty proxy/VPN hooks.
- In-scope host enforcement.
- HTTP budget/rate-limit hooks.
- Report-slot logic.
- Heavy lease machinery designed for many parallel target probers.
- Multiple slash-command modes for work types.
- Multiple specialized subagents up front.
- Automatic rescheduling as a first milestone.

## Subagent Stance

Default to one general subagent per `/work` run.

The main reason to use a subagent here is context control, not parallelism. The main Claude session should remain a thin orchestrator:

1. Read only enough state to pick a job.
2. Spawn one general worker with a bounded brief.
3. Wait for the worker.
4. Read the worker's concise result.
5. Persist the summary and update queue/escalations if needed.

The worker can handle any task type: investigation, implementation, docs update, run analysis, or verification. This avoids bloating the orchestrator context with full logs, long file reads, and exploratory dead ends.

Default to no parallel fan-out. Resource contention is real:

- Training and evaluation jobs can consume CPU/GPU for minutes or hours.
- Run directories and checkpoints are shared mutable resources.
- Docs like sprint plans and backlog files are easy to conflict on.
- Parallel agents can duplicate expensive investigation when a single sequential loop would have been enough.

Use more than one subagent only when all of these are true:

- The tasks are genuinely independent.
- They are read-only or have disjoint write sets.
- They will not start heavyweight training/eval jobs.
- The expected results are concise.

This means the first harness can omit lease files entirely. If contention becomes a real problem, add a single coarse lock later, not a full target-lease system.

## Hooks Stance

Hooks are not core for this repo.

In the source harness, hooks were valuable because they enforced bug-bounty programme rules before dangerous commands could run. This repo does not have the same external compliance boundary.

For this project, hooks would mostly duplicate normal engineering judgment:

- Do not run destructive git commands.
- Do not overwrite unrelated dirty files.
- Do not launch long GPU runs accidentally.
- Do not mutate historical `runs/` artifacts unless explicitly intended.

Those are better captured in `CLAUDE.md` and command instructions first. Add hooks only after a repeated concrete failure shows that documentation is not enough.

## Proposed Lightweight Layout

```text
CLAUDE.md
docs/
  agent-loop-harness-proposal.md
  claude-loop-harness.md

.claude/
  commands/
    work.md
  agents/
    worker.md

docs/ai-agent-state/
  queue.json
  escalations.md
  digests/
    YYYY-MM-DD.md
  notes.md
```

No new Python package is required for the first version.

If a helper script becomes useful later, add one narrow script such as:

```text
scripts/agent-status.sh
```

It should print current git status, latest run summaries, queue head, and open escalations. It should not become the owner of the loop.

## Command Semantics

### `/work`

The only command in the first version.

Read only enough state to choose the next job:

- `git status --short`
- `docs/ai-agent-state/queue.json`
- `docs/ai-agent-state/escalations.md`
- Tail of `docs/ai-performance-research-progress.md`
- Current sprint plan and backlog
- Recent relevant `runs/*/orchestrator-state.json` or `events.jsonl`

Then define a bounded worker brief:

- Objective.
- Relevant files/runs/docs to inspect.
- Allowed write scope.
- Commands allowed or forbidden.
- Validation expectation.
- Expected final response shape.

Spawn `.claude/agents/worker.md` for the actual work. The worker should do the heavy context gathering, run commands if appropriate, edit files if in scope, and return a concise summary.

After the worker returns, `/work` should:

- Read only the worker summary and any files it changed.
- Update `docs/ai-agent-state/digests/YYYY-MM-DD.md` if the result is meaningful.
- Update `queue.json` or `escalations.md` only when the result changes the next action.
- Report what changed and what remains.
- Continue immediately if the worker identified a clear next step inside the same bounded objective.
- Stop when the bounded objective is complete or the next step requires waiting.
- Do not schedule a timed wakeup just to continue ordinary work.

Use a timeout/wakeup only when the next useful action is blocked on wall-clock time or an external dependency: a still-running training/eval process, a future checkpoint/log/manifest, human input, or a deliberate cool-down after resource-heavy work.

The worker's task can be any type:

- Diagnose one failing smoke.
- Inspect one run directory.
- Add one small guard or smoke assertion.
- Test one hypothesis with a tiny run.
- Analyze one training regression.
- Update backlog/progress docs from accepted evidence.
- Run or inspect the right validation subset.

For verification tasks, the worker should know the common tiers:

- Fast repo confidence: `npm run build`, `npm run test:train`
- Python/training smoke: `TMPDIR=/tmp npm run test:python-train`
- Orchestrator smoke: `TMPDIR=/tmp npm run test:dagger-orchestrator`
- PPO smoke: `TMPDIR=/tmp npm run test:ppo-smoke`
- Targeted deeper smokes only when touching determinism, serving, MCTS, or observability

Missing local checkpoints should be reported as environment gaps, not product passes.

## Durable State

Use a small state directory:

```text
docs/ai-agent-state/
  queue.json
  escalations.md
  notes.md
  digests/
    2026-05-14.md
```

Suggested `queue.json`:

```json
{
  "updated": "2026-05-14",
  "items": [
    {
      "id": "r14-ood-gate-followup",
      "priority": "P1",
      "status": "ready",
      "summary": "Inspect the OOD gate result and decide whether to update the R14 plan.",
      "next_action": "Run /work with a worker brief for training/r14_ood_gate.py and current docs diffs."
    }
  ]
}
```

Keep this intentionally small. It is a steering aid, not a project-management database.

## Repo Entry Instructions

Add `CLAUDE.md` with:

- The goal of the project.
- Where AI research state lives.
- The default `/work` workflow.
- The default general-worker subagent workflow.
- Dirty-worktree policy.
- Long-run caution.
- Testing tiers.
- Instruction to prefer existing training orchestrators over new orchestration code.

Do not add `AGENTS.md` for the initial version. This harness targets Claude Code first.

## General Worker Subagent

Use one generic subagent:

```text
.claude/agents/worker.md
```

Purpose:

- Execute one bounded task from `/work`.
- Keep exploratory context out of the main orchestrator session.
- Handle investigation, implementation, docs updates, run analysis, or verification.
- Avoid heavyweight training jobs unless the brief explicitly allows them.
- Do not spawn more agents.
- Return concise findings, changed files, commands run, validation result, and next recommendation.

Do not add multiple specialized agents until repeated usage shows a real need. The single worker keeps the harness simple while still solving the main context-bloat problem.

## Implementation Phases

### Phase 1: Docs-Only Harness

Add:

- `CLAUDE.md`
- `.claude/commands/work.md`
- `.claude/agents/worker.md`
- `docs/ai-agent-state/queue.json`
- `docs/ai-agent-state/escalations.md`
- `docs/ai-agent-state/notes.md`

No Python. No hooks. No extra command modes.

### Phase 2: Tiny Helper Script

If manual state reads become repetitive, add:

```bash
scripts/agent-status.sh
```

The script should only summarize:

- Git status.
- Queue head.
- Open escalations.
- Latest digest.
- Latest run directories.

It should not execute experiments.

### Phase 3: Automation

Only after the manual Claude loop is useful, consider scheduled wakeups.

Scheduling should be independent from the state files. A failed wakeup should not corrupt the harness. Do not use automation to create idle polling; only schedule a wakeup when there is a concrete time-blocked reason to resume later.

## Validation

Because this is mostly a Claude command/docs harness, validation is simple:

- Start a fresh Claude Code session and confirm `CLAUDE.md` gives enough context.
- Run `/work` in dry/manual mode and confirm it picks a sensible next action.
- Confirm `/work` delegates the heavy context gathering to one worker.
- Confirm the worker chooses the right test tier without launching excessive work.
- Confirm queue/digest/escalation updates are small and readable.

When a harness command causes code changes, use the normal repo validation tiers:

```bash
npm run build
TMPDIR=/tmp npm run test:train
TMPDIR=/tmp npm run test:python-train
TMPDIR=/tmp npm run test:dagger-orchestrator
TMPDIR=/tmp npm run test:ppo-smoke
```

Run deeper smokes only when the touched area warrants them.

## Final Shape

The harness should feel like a thin operating discipline for Claude Code:

1. Read current evidence.
2. Pick one useful job.
3. Spawn one general worker to do the context-heavy part.
4. Wait, synthesize, and persist the result.
5. Continue immediately while the next step is available now.
6. Use timed looping only for genuine waits.
7. Update the queue only when evidence changes.

Keep it boring. The existing training code is already complex enough; the Claude layer should reduce operator drift, not add another system to maintain.
