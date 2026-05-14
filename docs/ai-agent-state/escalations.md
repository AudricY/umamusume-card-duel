# Claude Harness Escalations

## Open

- **2026-05-14 — Whole queue gated on human action or compute green-light.** Updated after 6 `/work` firings of the 60s loop (`CronCreate` id `6f926118`) exhausted autonomous bounded work. Current queue state:
  - **P1 `r14-e-manual-ui-exercise`** — blocked on human. The R14.E acceptance step is to play 20 full UI games in a browser, capture devtools `decisionMs` + fallback events, and append to `docs/r14-sprint-plan.md`. Claude cannot drive an interactive browser session here. Either (a) the human runs the 20 games and pastes the devtools numbers, or (b) a headless puppeteer/playwright harness is built first (separate sprint slot, ~half-day).
  - **P2 `f1-self-play-sweep`** — needs explicit green-light. Real PPO compute (~5–6 min × 3 iters at 800 games/update) per F1 cost evidence; CLAUDE.md says warn before launching long GPU/training/large-eval jobs.
  - **P2 `f1-warmstart-sweep`** — needs explicit green-light. ~1.5h DAgger compute + ~20–30 min eval per scoping doc.
  - **P3 `side-asymmetry-confirmation-gate`** — needs explicit green-light. `r14_ood_gate.py` n=100 evaluation; cheaper than the P2s but still a real eval.
  - Fast validation tier (`npm run build` + `TMPDIR=/tmp npm run test:train`) is clean as of 2026-05-14 fifth slot. Repo is healthy; nothing to fix.
  - **Recommendation:** either unblock one of the queue items, or `CronDelete 6f926118` to stop the loop. Each subsequent firing will produce diminishing-returns no-ops or risk drifting into work the harness shouldn't take autonomously.

Use this file for blockers, unsafe stop lines, missing local artifacts, ambiguous research direction, or human decisions needed before useful autonomous work can continue.
