# AI Performance Research Backlog Archive

Last refined: 2026-05-08.

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

## Historical Results Policy

Older rollout/search/trained-model results before the corrected evaluator, state fingerprint, modeled action export, and card-aware reset are historical context only. They should not be compared directly with future results unless rerun through the corrected rebaseline suite.

The active promotion standard is the one in `docs/ai-performance-research-backlog.md`: 500 side-balanced held-out games, fixed non-training seeds, documented decks/matchups, zero fallbacks/no selected no-ops, CI, side split, average points, terminal reasons, selected rank, and manifests.
