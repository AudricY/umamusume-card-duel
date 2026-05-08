# AI Performance Research Harness

End goal (v4.1): graduate the v4 reframe ("DAgger is the warm-start; F1 is the path through the cap") from working hypothesis to finding by running **item 17** — the larger-scale DAgger sweep at n≥500 eval, ≥30 epochs, KL anti-forgetting on, rollout-steps aligned with the rebaseline gate. Item 17 either confirms the SL cap (and unlocks F1 with confidence) or falsifies it (and saves the F1 stability tax).

## Files

- `docs/ai-performance-research-backlog.md` — canonical, ordered, gated. Currently at v4.1.
- `docs/ai-performance-research-progress.md` — append-only evidence log.
- `docs/ai-performance-research-backlog-archive.md` — what was deferred/absorbed and why; v4.1 section near the bottom.
- `docs/ai-training-findings.md` — raw experimental results.

## Sequence (v4.1)

The concrete v4.1 critical-path sequence to the end goal:

1. **Item 7 (calibrated value head) + Item 18 (F1 plumbing prep), in parallel.** Item 7 has a dual exit gate — tiebreaker-grade unblocks item 2's tiebreaker fallback, GAE-grade unblocks F1's GAE; failing only the GAE-grade gate degrades F1 to point-margin advantages instead of blocking it. Item 18 must complete before the warm-start checkpoint generates F1's rollouts so importance-sampling ratios are recoverable.
2. **Item 12 (opponent snapshot pool with PFSP).** Required for item 13's per-matchup Wilson floors and for F1's league sampling.
3. **Item 13 residual work** (per-matchup gating + halt-after-2 trigger) — unblocked by item 12.
4. **Item 17 (this harness's end goal).** A full sweep with the recipe bug from `runs/dagger-real-2026-05-08/` fixed (rollout-steps aligned at 500, ≥30 epochs, KL anchor on). Pre-registered escalation: trained policy Wilson lower ≤45% across 3 iterations *and* monotone improvement <2pp graduates the v4 reframe from working hypothesis to finding.
5. **F1 PPO smoke** — only after items 7 and 18 land, regardless of item 17's outcome.

## Operating Notes

- The backlog is opinionated and load-bearing. Read it, don't skim it. Cross-references and gates exist for reasons explained in the archive.
- Append findings to `progress.md`. Don't create new analysis/planning/decision docs.
- Escalation branches are pre-registered for a reason — surface them to the user instead of grinding through. v4.1 retains: item 2's 55% relaxed launch criterion, item 7's dual exit gates with the three-branch fork on outcome, item 17's "graduate the reframe" escalation, item 0's 24h projection (cleared at 0.25h, recorded for completeness).
- Planted fixtures tripping on real data is a finding to report, not a test to fix.
- v4.1 records confidence carefully: the n=50 trained-policy evidence is overlapping intervals; the v3 "55-65% imitation cap" claim is *not yet falsified* until item 17 lands. Don't write the cap as fact in `progress.md` until then.
- Code-level deferred fixes from v4.1 reviewer 2 are recorded in the archive's v4.1 section. Land them before item 17's sweep runs at scale, otherwise the orchestrator will hit them.
