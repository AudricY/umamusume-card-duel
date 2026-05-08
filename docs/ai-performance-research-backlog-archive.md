# AI Performance Research Backlog Archive

Last refined: 2026-05-08 (v4 — evidence-driven pass after the first real DAgger run).

This file keeps the useful discarded signal from the original `ai-performance-research-backlog.md` and `ai-performance-research-backlog-v2.md` after consolidation into `docs/ai-performance-research-backlog.md`.

## Completed Or Absorbed

- **v1-4 Policy-baseline-visited outcome export:** Implemented as a first-class source in outcome manifests/training summaries. Any remaining use belongs under DAgger source-mix work, not a standalone backlog item.
- **v1-6 Episode/seed train-validation split:** Implemented with grouped defaults and manifest/checkpoint recording.
- **v1-16 Deterministic progress fingerprint:** Implemented via a shared simulator fingerprint helper across evaluator/headless/export/action-contract paths.
- **v2-A5 Oracle metadata persistence:** The important oracle fields now exist: sample seeds, candidate rewards, reward mean/variance, selected margins, tie policy, low-margin flags, and selected original rank. Remaining work is to use those fields in weighting and sampling.
- **High-impact action payload plumbing:** Trainer and ability hidden-choice payloads are mostly exposed and sampled action-contract smoke passed with zero sampled no-op actions.

## Narrowed From Standalone Items

- **v1-2 Controlled CRN samples:** Plumbing and determinism are mostly done. Active work is now low-margin weighting, adaptive sample expansion, and matched-wall-clock training comparisons.
- **v1-3 Candidate ranker diversity:** Shared ranker modes exist. Active work is dropped-best auditing and planner/search use under the teacher-quality item.
- **v1-7 Legal attacks and combat choices:** `attackIndex` and all payable attacks are wired. Active work is only the representative multi-attack fixture gap.
- **v1-8/v1-9/v1-10 Feature schema, state semantics, ablations:** Merged into one versioned feature migration item so embeddings, set encoders, metadata, history, schema versions, fixtures, and closed-loop ablations land together.
- **v1-11/v1-13 Procedural scoring and turn goals:** Merged into one procedural hardening item gated by held-out eval and average points.
- **v1-17/v1-18/v1-19 Eval gate, manifests, smoke tests:** Mostly present. Active work is explicit no-op gate counts, manifest enforcement for comparable evals, and bad-policy strength tests.
- **v2-B4 Distributional value and v2-B3 auxiliary heads:** Merged into calibrated value/action-value work.
- **v2-C2 Adaptive CRN and v2-C4 stratified sampling:** Merged into the margin/phase/action-kind training mix item.
- **v2-C3 trajectory bundling:** Deferred into DAgger/sequence-loader work only if sequence losses become a near-term need.

## Deferred Until Gated

- **Self-play / RL:** Now the explicit end goal. Critical-path and gated follow-up items (Options A/B/C) live in the unified backlog rather than being deferred.
- **Transformer encoder destination architecture:** Not active until card/entity tokens, cross-attention baselines, scale-ready training, and 100K+ useful rows exist. Tracked as B2 "behind the gate" in the unified backlog.
- **Multi-GPU / DDP:** Not active until single-GPU training is a measured bottleneck for the RL orchestrator.
- **Masked card/action pretraining:** Not active unless labeled-data scaling stalls after generation parallelism and matchup sampling.
- **Larger supervised runs on current labels:** Explicit non-goal. SL is bounded by teacher quality (~55-65% imitation cap); RL self-play is the mechanism for clearing it.

## Refactored Into Behind-The-Gate (2026-05-08, RL endpoint pass)

After the RL-endpoint refinement, three former active items moved below the SL→RL handoff gate. They are SL polish that the RL loop either subsumes or makes obsolete.

- **Margin/phase/action-kind training mixes (former item 7):** Subsumed by RL advantage-weighted updates; phase distribution is learned from reward. Resume only if DAgger plateaus and ablation shows margin signal is the bottleneck.
- **Entity-aware architecture sweeps (former item 11):** Cross-attention/asymmetric scaling/transformer. Optimization on top of feature migration; RL ships on competent MLP-over-tokens. Resume after a plateaued loop.
- **Procedural/ability polish beyond planner needs (residual of former items 3 and 8):** RL self-play discovers ability/turn-goal heuristics from reward. Only scorers consumed by the planner remain in the active backlog.

Items renumbered in the same pass: planner is now item 2 ("Bring the DAgger Teacher Above the Gate"), procedural scoring is item 9 (planner-consumer scope), value head is item 7 (framed as the RL critic), reproducibility guards are item 10. Items 11-14 cover the DAgger orchestrator, opponent pool, per-iteration eval gate, and compute/distillation.

## Adversarial Pass (2026-05-08, v3)

A second refinement round added structural fixes flagged by dependency, failure-mode, and per-item adversarial reviews:

- **New item 0:** Throughput probe spike ahead of items 11-14, so item 14's budget targets and item 6's "near-linear" claim are grounded in measurement.
- **Item 5 split:** 5a (vocab + embeddings + schema fail-fast) is the only blocking subset for item 11; 5b (set encoder + metadata + history + ablations) is non-blocking.
- **Item 7 moved to "P1-Optional":** It is required for F1 (PPO advantages) and recommended as an aux head for DAgger, but does not block item 11. Stops the prior implication of single-thread sequencing.
- **Item 4 / item 11 boundary redrawn:** item 4 owns the data recipe only; item 11 owns rollout/retrain. Stale "(item 13)" cross-reference fixed.
- **Item 11 expanded:** explicit replay buffer policy (cap, staleness cutoff, mix schedule), KL-anchor anti-forgetting regularizer, reproducibility tiering (CPU-only byte-identical; GPU statistical-equivalence within tolerance) replacing the earlier unrealistic byte-identical bar.
- **Item 12 expanded:** prioritized fictitious self-play sampling, retention policy (last 8 + every 4th historical, cap 24), KL drift reference pinned.
- **Item 13 thresholds pinned:** ≤5pp per-matchup drop tolerance; cross-references to the loop promotion gate; halt-after-2 restated.
- **Item 14 demands establishing throughput targets first** with concrete defaults (≥200 planner decisions/sec at full worker count, ≤4h iteration wall-clock); distillation margin pinned at 3pp.
- **New items 15 and 16:** exploration/action-coverage during trace generation, and iteration-level debugging tooling (decision diff, replay viewer). Neither was previously in the backlog despite being known RL failure modes.
- **Item 1 extended:** planted bad-policy regression specified; N-step cycle stall detection added.
- **Item 2 escalation criterion:** if planner stalls below 55% Wilson lower bound after a 2-week budget, fall back to teacher composition (planner + value-head tiebreaker, or rollout-augmented planner with deeper CRN). Pre-registered, not improvised.
- **Item 3 vs item 9 delineated:** item 3 scores choices *inside* a planner-enumerated action; item 9 scores the candidate ordering *feeding* the planner.
- **Promotion standard tightened:** Wilson lower bound (not point estimate) is the comparison number; per-opponent floor pinned at 45%; "monotonic regression" precisely defined; best-response exploitation check elevated from one-line bullet to tracked time-series with its own halt threshold; plateau (graceful stop) defined separately from regression halt.
- **F1 (PPO) work list expanded** with reward shaping, episode boundary, behavior-policy logging, mask handling, buffer sizing, GAE/KL/entropy controls, league sampling, HP sweep methodology. Smoke raised to ≥10 updates.

Reviewers' raw reports are preserved in conversation history.

## Evidence-Driven Pass (2026-05-08, v4)

The v4 refinement is anchored to fresh empirical data from the first real DAgger orchestrator run rather than to a fresh adversarial review. Five structural shifts:

- **End-goal framing flipped.** v3 framed Option A (DAgger) as the endpoint with F1/F2 as gated follow-ups. v4 reframes Option A as the warm-start for F1, on the grounds that the supervised cap is well below the rule bot at our data scale.
- **Imitation cap revised down.** v3 stated "55-65% imitation ceiling". The first real DAgger sweep produced 34% (iter 0) and 28% (iter 1, rolled back). v4 records the "55-65%" claim as empirically too optimistic at the current data scale and declines to lower acceptance bars on its strength.
- **Item 2 escalations exhausted.** v3 pre-registered "rollout-augmented planner with deeper CRN" and "planner + value-head tiebreaker" as plan-B fallbacks. The first half landed (rollout-CRN at Wilson lower 58.3%, above 55% but short of 65%). The second half is blocked on item 7. v4 adds an explicit decision rule: if the planner stays below 55% and rollout below 65% after items 7 and 9, the loop ships rollout-CRN as the teacher and F1 becomes the path through the cap, not the planner.
- **Item 7 promoted from P1-Optional to de-facto P0.** Required for both item 2's value-head tiebreaker fallback and F1's advantage estimator. With both downstream lanes now load-bearing, item 7 is the next critical-path work after the orchestrator's first sweep.
- **Item 12 promoted from P2 to next-after-7 critical-path.** The orchestrator's per-iteration gate is structurally incomplete without the snapshot pool: per-opponent Wilson floors, cycling detection, and item 13's pinned thresholds all depend on item 12. The orchestrator currently checks aggregate Wilson lower only; item 12 closes that gap.
- **Item 6 priority dropped.** The v3 throughput probe (item 0) measured ~100× single-core headroom against item 14's 4h iteration target, so worker-thread sharding is no longer urgent for the first DAgger sweep.
- **F1 gating rephrased.** v3 said "PPO is only worth the stability tax once DAgger has produced a policy that meaningfully beats the rule bot." v4 says: F1 begins as soon as item 7 lands, with the orchestrator's promoted DAgger checkpoint as the warm-start. F1's "wait for DAgger to clear the gate" gating is no longer informative because DAgger empirically does not clear the gate at our current scale.

Status fields (`✓ done`, `~ partial`, `· pending`, `↑/↓` priority change vs v3) added to every active item so the next refinement can diff against v4 cleanly.

The named-status convention is durable; the priority-change arrows are diff-against-v3 only and should be removed at v5.

## Historical Results Policy

Older rollout/search/trained-model results before the corrected evaluator, state fingerprint, modeled action export, and card-aware reset are historical context only. They should not be compared directly with future results unless rerun through the corrected rebaseline suite.

The active promotion standard is the one in `docs/ai-performance-research-backlog.md`: 500 side-balanced held-out games, fixed non-training seeds, documented decks/matchups, zero fallbacks/no selected no-ops, CI, side split, average points, terminal reasons, selected rank, and manifests.
