# umamusume-card-duel — Claude Code Entrypoint

Read this at the start of every Claude Code session in this repo.

## Project Shape

This is a TypeScript card-game implementation with a Python AI training stack.

- TypeScript owns game rules, legal actions, simulation, backend endpoints, and UI.
- Python owns model training, long-running AI research orchestration, ONNX export/serving helpers, and analysis scripts.
- Existing training orchestrators are the source of truth for serious loops: `training/dagger_orchestrator.py`, `training/r12_orchestrator.py`, and `training/ppo_orchestrator.py`.

Do not create a new orchestration framework unless the user explicitly asks. Prefer the existing npm and Python scripts.

## Default Workflow

Use `/work` as the default operating command — see `.claude/commands/work.md` for the loop's full prose. Two subagents live in `.claude/agents/`: `investigator` (read-only) and `implementer` (write-capable). The main session stays small; it owns judgment and synthesis.

**Size each step to the next external gate** — a training run, a re-extraction, a verdict eval, human input. Not to the next code unit that smokes green. If nothing real separates two phases of work, they are one step. Phase splits inside a single end-to-end code change (e.g. schema → encoder → ONNX wire, all smoke-validated together) are artificial and waste session overhead.

Use a timeout or `/loop` only when blocked on wall-clock or external dependency. Backlog refinement and big-picture planning are normal `/work` jobs.

## Claude Harness State

Lightweight Claude operating state lives in `docs/ai-agent-state/`:

- `queue.json`: small list of next useful jobs. `summary`/`next_action` are pointers to scoping/progress docs, not inlined plans. Strip `status: done` entries to digests when they pile up.
- `escalations.md`: blockers, unsafe stop lines, missing local artifacts, or human decisions needed.
- `notes.md`: durable harness notes.
- `digests/YYYY-MM-DD.md`: short summaries from meaningful `/work` runs.

Canonical AI research evidence:

- `docs/ai-research/progress/r<N>.md` — phase writeups (R15+).
- `docs/ai-performance-research-progress.md` — R1–R14 history, read-mostly.
- `docs/ai-research/scoping/<topic>.md` — hypothesis + sweep config + exit gate. **Do not pre-decompose implementation into phase counts + LOC budgets here** — that turns scoping into a /work-slot schedule and over-splits the work.
- `docs/ai-research-backlog.md` — forward-looking.
- `docs/ai-performance-research-harness.md`, `docs/r<N>-sprint-plan.md`.

## Safety And Scope

- Never revert user changes unless explicitly requested.
- Check `git status --short` before edits.
- If unrelated dirty files exist, leave them alone.
- Do not mutate historical `runs/` artifacts unless the user explicitly asks or the task is creating a new run output.
- Warn before launching long GPU, training, or large evaluation jobs.
- Missing checkpoints under `runs/` are environment gaps, not product passes.
- Hooks are intentionally not part of this harness unless repeated concrete failures prove they are needed.

## Documentation Discipline

Each fact lives in one file. Everywhere else links to it. State files are pointers, not copies.

Homes:

- Phase result (mechanism + numbers): `docs/ai-research/progress/r<N>.md` (R15+) or the R1–R14 monolith.
- Scoping: `docs/ai-research/scoping/<topic>.md` — hypothesis, sweep config, exit gate. Not implementation phase plans.
- Backlog: forward-looking only. Landed/failed → one-line pointer.
- `notes.md`: durable harness conventions.
- `digests/YYYY-MM-DD.md`: daily index. One slot = ≤8 lines (verdict, number, 1-line mechanism, link). No file changelogs, no "next action" prose.
- `escalations.md`: ≤500 chars per bullet; longer rationale → link a scoping doc.
- `queue.json`: pointers, not plans.

Caps (hard): digest ≤150 lines, backlog ≤300, per-sprint progress ≤500. Trim before append.
