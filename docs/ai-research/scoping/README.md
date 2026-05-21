# Scoping Documents

One file per scoping exercise (pre-registered hypothesis, sweep config, exit gate, go/no-go reasoning).

Filename convention: `<sprint-or-topic>-<axis>.md` — e.g. `r15-s3-axis-2-value-head-tempo.md`, `f1-self-play-readiness.md`.

This directory replaces the "scoping dump" usage of `docs/ai-agent-state/notes.md`. Per CLAUDE.md "Documentation Discipline":

- Scoping content lives here, in its own file.
- `notes.md` is durable harness conventions only.
- The digest, escalations, and queue link to the file here; they do not re-state its contents.
- Closed scoping docs move to `archive/` and are treated as historical context.

## STATUS line convention

Every scoping doc must carry a `**Status:** <state> — <one-line outcome + pointer>` line at the top, kept current as the experiment progresses. Six valid states:

- **`SCOPING`** — pre-registered, not yet launched. Body describes the planned recipe.
- **`IN-FLIGHT`** — experiment running. Pointer to the `runs/<dir>/` and the queue entry tracking it.
- **`LANDED`** — experiment finished, verdict written. Pointer to the result writeup in `docs/ai-research/progress/r<N>.md`.
- **`SUPERSEDED`** — replaced by a different line. Pointer to the successor scoping doc or queue item.
- **`CLOSED`** — deprioritized/abandoned. One-sentence reason.
- **`HISTORICAL`** — kept for context only; the canonical doc is elsewhere. Pointer to the canonical home.

The STATUS line flips at the same commit that writes the result writeup to the progress doc. A `LANDED` STATUS without a progress-doc pointer is a bug.

When the STATUS leaves `SCOPING`/`IN-FLIGHT`, do NOT delete the body — it has historical value (what the recipe was, what the hypothesis was). The STATUS line + one-paragraph pointer is what changes; the rest stays as evidence.
