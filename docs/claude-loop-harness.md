# Claude Loop Harness

This repo uses a lightweight Claude Code harness for AI research operations.

The harness is intentionally small:

- One command: `.claude/commands/work.md`
- Two role-specific subagents: `.claude/agents/investigator.md` (read-only) and `.claude/agents/implementer.md` (write-capable)
- One repo entrypoint: `CLAUDE.md`
- Small state files under `docs/ai-agent-state/`
- Per-sprint research evidence under `docs/ai-research/progress/`; scoping docs under `docs/ai-research/scoping/`
- Research doc routing index: `docs/ai-research/README.md`

There is no Python harness, no hooks, and no separate command mode per task type.

## Operating Model

The main Claude session is an orchestrator. It owns judgment, prioritization, and synthesis. It delegates context-heavy work to a role-specific subagent when that keeps the main context cleaner.

Pick the subagent by whether the job ends in evidence or in a change:

- `investigator` — read-only: code exploration, run/log analysis, validation-as-evidence.
- `implementer` — write-capable: code edits, doc updates, validation cycles after changes.

The subagent returns a concise summary. The orchestrator persists the useful result and either continues with the next immediate step or stops when the bounded objective is complete, cost/risk rises, or context budget is better preserved for a fresh subagent.

## Looping And Wakeups

Do not use a timed `/loop` just to continue ordinary work. If the next useful step is available now, continue in the same `/work` run.

Use a timeout or wakeup only when progress is blocked by wall-clock time or an external dependency:

- A training, evaluation, or background process is still running.
- A checkpoint, manifest, or log file is expected later.
- Human input or approval is required.
- A deliberate cool-down is needed after resource-heavy work.

The default loop is therefore synchronous: worker returns, orchestrator synthesizes, and work continues immediately when appropriate.

## Planning And Refinement

The single `/work` command still owns backlog refinement and strategic planning. These are job types inside `/work`, not separate modes.

Use planning/refinement when judgment says the loop is about to drift: the
queue is stale or vague, recent evidence changes priorities, docs disagree, an
escalation blocks the apparent next step, or the work is no longer clearly tied
to model quality, throughput, determinism, or UI integration.

Queue refresh should be small and practical: reconcile enough live evidence,
blockers, and active backlog context to make the next few safe jobs clear. Do
not exhaustively re-rank the whole backlog unless the current state is genuinely
unusable.

Planning output should be small: update the queue, escalations, digest, or existing AI docs. Do not create new planning documents unless the user asks. Before writing, search for the topic and update the canonical home from `docs/ai-research/README.md` instead of creating another copy.

## Parallelism

Use a subagent when it protects the main context from bulky exploration, logs, broad file reads, or implementation detail.

Subagents exist for context isolation, not parallel throughput. Parallel fan-out is usually the wrong default here because training/eval jobs, run directories, checkpoints, and planning docs are shared resources.

Use multiple subagents only when tasks are independent, low-resource, and have disjoint write scopes.

## State Files

```text
docs/ai-agent-state/
  queue.json          # next few jobs; summary/next_action are short pointers
  escalations.md      # ≤500 chars per bullet; longer rationale → scoping doc
  notes.md            # durable harness conventions only (cap ≤200 lines)
  digests/YYYY-MM-DD.md  # daily index; one slot = ≤8 lines (cap ≤150 lines/day)

docs/ai-research/
  progress/r<N>.md    # per-sprint phase writeups (R15+); cap ≤500 lines
  scoping/<topic>.md  # one file per scoping exercise
```

State files are pointers to evidence, not copies of it. See CLAUDE.md "Documentation Discipline" for the full role+cap rules and the trim-before-append trigger.

Canonical model/training evidence belongs in `docs/ai-research/progress/r<N>.md` (R15+) or the historical `docs/ai-performance-research-progress.md` (R1–R14), not in harness state.

Archived and historical docs are read-only context unless the current task is
explicitly archive cleanup. Active prompts should link to canonical docs rather
than paste old sprint-plan or monolith details into state files.

## Orchestrator Launch Convention

Launch long-running Python orchestrators directly, not through `npm run`:

```bash
nohup training/.venv/bin/python -u training/<orchestrator>.py ... > <out-dir>/launch.log 2>&1 &
```

Launching through `nohup npm run <script> &` captures the npm wrapper PID, not
the live Python worker. Use `<out-dir>/pid.txt` as the source of truth for the
orchestrator PID; `dagger_orchestrator.py`, `r12_orchestrator.py`, and
`ppo_orchestrator.py` write it at startup.

## Non-Goals

- Do not replace `dagger_orchestrator.py`, `r12_orchestrator.py`, or `ppo_orchestrator.py`.
- Do not add hooks unless repeated concrete mistakes show a need.
- Do not add more slash commands until `/work` becomes overloaded in practice.
- Do not add more specialized subagents until investigator/implementer repeatedly prove too broad.
