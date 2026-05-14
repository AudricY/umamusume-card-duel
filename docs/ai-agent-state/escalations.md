# Claude Harness Escalations

## Open

- **2026-05-14 — Queue head `r14-e-manual-ui-exercise` (P1) is blocked on human action.** The R14.E acceptance step is to play 20 full UI games in a browser, capture devtools `decisionMs` + fallback events, and append to `docs/r14-sprint-plan.md`. Claude cannot drive an interactive browser session here. Either (a) the human runs the 20 games and pastes the devtools numbers, or (b) a headless puppeteer/playwright harness is built first (separate sprint slot, ~half-day). Until resolved, future `/work` runs will skip past this item and pick the next bounded P2.

Use this file for blockers, unsafe stop lines, missing local artifacts, ambiguous research direction, or human decisions needed before useful autonomous work can continue.
