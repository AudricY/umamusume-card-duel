# Claude Harness Escalations

## Open

Use this file for blockers, unsafe stop lines, missing local artifacts, or human decisions needed before useful autonomous work can continue. ≤500 chars per bullet — longer rationale belongs in a scoping doc.

_(none open)_

## Resolved

- **2026-05-14 self-directed: R15.S3 closeout → R7 picked over R8.** Reward-shape branch closed across both axes (obs-delta saturated 0.368 ±0.001, value-head tempo regressed at coefs 0.05 + 0.01). User feedback: "be self-directed for these kinds of decisions." Pick: **R7 multi-teacher BC blend.** Reasoning: R15.S1 already closed FAIL (compute-scaled single-teacher), so the open SL question is teacher diversity; R7 reuses PPO infra and has cheaper SL-side falsification than R8's pure-DPO bet. Full detail: `docs/ai-research-backlog.md` § R7. Queue: `r7-multi-teacher-warmstart-scoping` P2 ready.

- **Resolved 2026-05-14 self-directed (premature framing).** Prior R15.S3 closeout had only tested axis 1; axis 2 (value-head tempo, Phases O + P) later regressed at coefs 0.05 + 0.01. Both axes now confirmed exhausted. Full per-phase detail at `docs/ai-performance-research-progress.md` §§ Phase L / M / N / O' / O / P.

- **Resolved 2026-05-14 self-directed: R14 cheap-inference claim stance.** Side-asymmetry gate landed at Wilson lower 0.394 — below the 0.40 bar at independent seeds. Three options scoped (CI re-targeting / larger-n re-run / accept FAIL + route to rollout-leaf). Action: (a) docs re-targeted to "Wilson 0.39–0.45 across seed ranges"; (c) rollout-leaf MCTS Wilson 0.6479 elevated as primary production claim; (b) deferred — re-run at larger n didn't change qualitative picture. Full writeup: `docs/r14-sprint-plan.md` (R14.A.footnote) and `docs/ai-research-backlog.md` (R14.A.footnote Result section).
