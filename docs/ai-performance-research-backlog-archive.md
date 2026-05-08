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

- **Self-play / RL:** Not active until the planner or another teacher reliably beats the rule bot. Self-play on a weak teacher is likely to amplify noise.
- **Transformer encoder destination architecture:** Not active until card/entity tokens, cross-attention baselines, scale-ready training, and 100K+ useful rows exist.
- **Multi-GPU / DDP:** Not active until single-GPU training is a measured bottleneck.
- **Masked card/action pretraining:** Not active unless labeled-data scaling stalls after generation parallelism and matchup sampling.
- **Larger supervised runs on current labels:** Explicit non-goal. Current labels and teachers do not justify expecting an 80% trained policy.

## Historical Results Policy

Older rollout/search/trained-model results before the corrected evaluator, state fingerprint, modeled action export, and card-aware reset are historical context only. They should not be compared directly with future results unless rerun through the corrected rebaseline suite.

The active promotion standard is the one in `docs/ai-performance-research-backlog.md`: 500 side-balanced held-out games, fixed non-training seeds, documented decks/matchups, zero fallbacks/no selected no-ops, CI, side split, average points, terminal reasons, selected rank, and manifests.
