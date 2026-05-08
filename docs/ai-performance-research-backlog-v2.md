# AI Performance Research Backlog (v2 — Capacity Track)

## Context

The first backlog (`ai-performance-research-backlog.md`) targets teacher quality, data distribution, action/feature fidelity, value calibration, and eval reproducibility. That work is necessary but is bounded by a representation ceiling we have not yet challenged.

Concretely:

- The state vector is 96 hand-engineered floats. Card identity collapses to FNV-1a hashes (`_identity_features`, `_hash_average`); 64 unique cards → single-float buckets that alias. The whole hand becomes one averaged hash.
- The action vector is 48 hand-engineered floats with positional semantics; attacker/source/target identities also reduce to hashes.
- The model is ~600K–1M params (`hidden_dim=128–160`, `depth=3`) with elementwise-product fusion between state and action towers. No attention, no per-card embeddings, no temporal context.
- Largest dataset trained on is ~17.9K rows (`card-aware-v2`); outcome export strips its own oracle metadata before write because of an `as TrainingExample` cast in `exportOutcomeTrainingExamples.ts`. CRN runs use `--samples 1` and run single-threaded.
- Training is single-GPU, fixed LR, no AMP, no schedule, no wandb/tb, full-dataset in-RAM. It "feels fast" because both model and dataset are tiny.

The hypothesis behind this v2 backlog: model capacity is matched to lossy features, and dataset scale is matched to single-process generation throughput. Once teacher quality plateaus, the next ceiling is representation × architecture × data scale × training infra. This document enumerates work in those four axes.

This backlog is intended for **later consolidation** with the v1 backlog. Each item flags v1 overlap explicitly.

Active execution notes for v1 work continue in `docs/ai-performance-research-progress.md`.

## P0: Kill The Lossy Compressions In Features

### A1. Per-Card Embedding Table

**Problem:** `_identity_features` (`training/uma_ai/features.py:142-158`) maps every cardId — active, bench, hand-average, discard-average — through `_hash_to_unit` into a single float. With 64 cards in `shared/src/data/cards.json`, hash collisions are routine and the model cannot learn per-card policy ("retreat from X but not Y").

**Direction:** Build a fixed-vocab embedding table indexed by canonical cardId (with the same `FullArtGold`/`FullArt`/`UncommonPlus` normalization `_get_card` already does). Replace the hash slots in state and action features with embedding lookups. Initialize from card metadata where useful (type one-hot, stage, base HP) so the model has a non-random prior on unseen cards.

**Expected impact:** Very high. This is the single largest information-recovery change available; everything downstream (set encoders, attention, auxiliary heads) requires it.

**Acceptance signal:**

- Embedding lookup replaces `_identity_features` and the active/bench identity slots in action features.
- Vocab is shared TS↔Python with a versioned schema; mismatched ids fail fast.
- Held-out eval improves over hash-baseline at matched param count.
- Per-card ablation (zero one card's embedding) changes that card's action ranking and nothing else.

**Overlap with v1:** Touches v1 items 8 and 9 (feature schema versioning, state semantic validation). Embedding vocab version should fold into the v1 schema-version contract.

**Evidence:** `training/uma_ai/features.py:142-158, 354-367`; `frontend/src/game/engine/ai-policy/actions.ts` (action-side hash slots); `shared/src/data/cards.json` (vocab source).

---

### A2. Set/Permutation-Invariant Encoder For Hand, Bench, Discard

**Problem:** Hand identity collapses to one averaged hash (`_hash_average(handCardIds)`); discard collapses to 3 role-count scalars. Bench reduces to per-slot hashes with no learned cross-slot interaction. The model cannot answer "do I have at least one of {evolver, draw, healer} in hand?" without overfitting to aggregates.

**Direction:** Replace hand/bench/discard scalar aggregates with a DeepSets-style encoder (per-card linear projection over A1 embeddings + concatenated metadata, then mean+max pool with attention-weighted aggregation). Pad to fixed max sizes (10 hand, 4 bench, 50 discard) for ONNX export; mask the rest.

**Expected impact:** High. Unlocks composition-aware decisions that aggregates cannot represent.

**Acceptance signal:**

- Hand/bench/discard pooling replaces averaged-hash and role-count scalars without changing observation contract.
- ONNX export uses fixed max sizes with mask inputs; runtime smoke covers variable-population batches.
- Ablation comparing aggregate-only vs. set-encoder shows accuracy lift on phases dominated by hand composition (trainer search, evolve, discard choices).

**Overlap with v1:** Subsumes parts of v1 item 9 ("reduce residual hashes where useful, evaluate learned card embeddings").

**Evidence:** `training/uma_ai/features.py:152-154, 175-180, 226-239`.

---

### A3. Card-Metadata Enrichment In State And Action Vectors

**Problem:** Cards in `cards.json` carry attacks, abilities, weakness, retreat cost, full energy cost breakdown, type, species, stage. State features expose only stage, primary-attack damage/cost via readiness scalars. Action features mention attack damage but not weakness multipliers, retreat cost, secondary attacks, or status conditions imposed.

**Direction:** Pipe per-card metadata fields into both the embedding initialization (A1) and explicit slots: weakness type, retreat cost, attack count, secondary-attack costs, ability presence/recharge state. For action features, include attacker weakness/retreat against target, expected damage after weakness multiplier, and target-survives-this-attack indicator computed at feature time.

**Expected impact:** Medium-high. Cheap to wire, provides labels the embedding cannot recover from data alone.

**Acceptance signal:**

- Action features include weakness-adjusted expected damage and target-survives flag.
- State features include weakness/retreat aggregates per side.
- Phase-level metrics improve on combat and retreat decisions.

**Overlap with v1:** Direct extension of v1 item 8 (finish candidate feature schema migration) — should consolidate into one schema-bump.

**Evidence:** `training/uma_ai/features.py:242-257, 283-290` (readiness already computes a partial form); `shared/src/data/cards.json` for the unused fields.

---

### A4. Action History / Temporal Context Slice

**Problem:** Observation is a pure snapshot — no record of "what just happened." Phase usage flags (`usedSupporterThisTurn` etc.) are the only temporal hint. The model cannot reason "opponent just gusted my bench, so they're setting up KO next turn" or "I drew last turn, so search is less valuable now."

**Direction:** Append a fixed-length recent-action ring buffer to the observation: last K (≈4) decisions per side encoded as (actor, phase, action-kind, primary-card-id-embedding-or-null). Cheap to record in the simulator since exporters already see one decision at a time.

**Expected impact:** Medium-high. Opens turn-sequencing patterns the snapshot model literally cannot represent.

**Acceptance signal:**

- Observation contract extends with `recentActions[K]` per side.
- ONNX export accepts the new fixed-shape slice; serving smoke covers turn-1 (empty buffer) and steady state.
- Ablation comparing `K=0` vs. `K=4` shows lift on attach/trainer/combat phases that follow opponent actions.

**Overlap with v1:** None direct. Compatible with DAgger trace export (v1 item 5) — same buffer can travel in trace rows.

**Evidence:** `training/uma_ai/features.py:42-59` (current snapshot-only state).

---

### A5. Drop The `as TrainingExample` Cast And Persist Oracle Metadata

**Problem:** `backend/src/sim/exportOutcomeTrainingExamples.ts` builds rich oracle metadata (per-candidate rewards, sample seeds, mean/variance, selected-vs-runner-up margin, candidate ranks, low-margin flag) and then strips it via `as TrainingExample` before write. The downstream Python loader has no access to the most expensive signal we generate.

**Direction:** Extend the `TrainingExample` schema with an optional `oracle` field; remove the cast; teach `JsonlPolicyDataset` to load it. This is a one-day change and it 3× the richness of every CRN row already produced.

**Expected impact:** High and free. The cost was already paid at generation time.

**Acceptance signal:**

- New rows carry `oracle` with sample seeds, per-candidate reward arrays, mean/variance, margins, candidate ranks.
- Python loader exposes `oracle` to training; default loss path is unchanged.
- A regenerated outcome dataset deserializes losslessly through TS export → JSONL → Python.

**Overlap with v1:** Strong overlap with v1 items 2 and 4 (CRN samples / source taxonomy). Should fold into the v1 acceptance signal that says "rows include sample/margin metadata" — we already write it, we just throw it away.

**Evidence:** `backend/src/sim/exportOutcomeTrainingExamples.ts:162-185`.

---

## P0: Architecture Upgrades That Need The New Features

### B1. Cross-Attention Between Candidate Action And Board Entities

**Problem:** Current fusion is `concat(state, action, state ⊙ action)` → MLP. Symmetric, low-rank, and forces the network to learn "this action targets that bench slot" purely through positional features.

**Direction:** Treat the action vector as a query and board entities (active, bench[0:4] per side, hand-pool, stadium, energy-zone summary) as keys/values for one or two cross-attention heads. The action gets an entity-aware context vector before the policy head.

**Expected impact:** High once A1+A2 are in place. Without per-entity tokens this is meaningless.

**Acceptance signal:**

- Policy net consumes entity tokens; ablation vs. concat-fusion shows lift on placement-sensitive phases (attach, retreat, gust, combat target).
- ONNX export at fixed entity count works in serving smoke.
- Attention rollouts on a small fixture set focus on the relevant slot for each action kind.

**Overlap with v1:** None direct.

**Evidence:** `training/uma_ai/model.py:81-95`.

---

### B2. Asymmetric Tower Scaling And Bigger State Encoder

**Problem:** State and action encoders are both 128-d × 3 residual blocks. State is information-dense (whole board, hand, discard, history); action is sparse. Equal capacity is probably the wrong allocation.

**Direction:** Independently tune state-tower (depth, width) and action-tower (shallower, narrower). Once A1+A2 land, push state hidden to 256–384 with depth 4–6; keep action tower compact. Treat as a hyperparameter sweep, not a permanent commitment.

**Expected impact:** Medium. Cheap to test once richer features exist; pointless before.

**Acceptance signal:**

- Sweep results recorded with matched data and gates from v1 item 17.
- Final config beats symmetric baseline at matched eval.

**Overlap with v1:** None direct. Depends on D1+D2 (logging, sweeps).

**Evidence:** `training/uma_ai/model.py:48-66`.

---

### B3. Auxiliary Heads For Multi-Task Supervision

**Problem:** With ~17K labels and a single policy/value objective, larger nets will overfit. We have many cheap auxiliary signals available from the simulator.

**Direction:** Add auxiliary heads on the shared state encoder, trained jointly with small loss weights:

- Predict opponent hand size (regression) — already public.
- Predict whether the active will be KO'd next turn (binary, computable from game continuation).
- Predict turn-end value distance from current value (helps the value head reach calibration).
- Predict opponent's next action kind (categorical, available from DAgger trace once v1 item 5 is mature).

**Expected impact:** Medium. Multi-task regularization is standard practice when label budget is small relative to capacity.

**Acceptance signal:**

- Each auxiliary head trains stably with ablatable weight.
- Removing all auxiliary heads with the same architecture hurts policy accuracy.
- ONNX export gates auxiliary heads off for serving (policy + value only).

**Overlap with v1:** Compatible with v1 item 14 (calibrated value head) — distributional value is a sub-case.

**Evidence:** `training/uma_ai/model.py:73-79` (current single value head).

---

### B4. Distributional / Quantile Value Head

**Problem:** Scalar tanh value with weighted MSE assumes near-Gaussian outcome distribution. Game results are bimodal (win/loss) and outcomes given a state are heavy-tailed.

**Direction:** Replace the scalar value head with a 21–51 bucket categorical (C51-style) or 5-quantile regression head over `[-1, 1]`. Expectation gives the scalar back; the spread quantifies uncertainty for search.

**Expected impact:** Medium-high. The v1 backlog already calls out value calibration as a blocker (item 14); this is a concrete shape change that helps.

**Acceptance signal:**

- Calibration metrics (ECE per turn bucket / phase) improve over scalar baseline.
- Search using value spread for leaf selection beats search using only point estimate.

**Overlap with v1:** Direct subset of v1 item 14. Should consolidate.

**Evidence:** `training/uma_ai/model.py:73-79`; `training/train_bc.py` value-MSE path.

---

### B5. Transformer Encoder Over Entity Tokens (Stretch)

**Problem:** Beyond cross-attention (B1), a small transformer over `[CLS, action, own-active, own-bench[0:4], own-hand-pooled, own-discard-pooled, opp-active, opp-bench[0:4], opp-discard-pooled, stadium, history-tokens[K]]` would let multi-card synergies emerge end-to-end.

**Direction:** 2-layer, 2-head, hidden 128–192 transformer. Action token's final hidden produces the policy logit. Keep ONNX-exportable (fixed token count, mask input).

**Expected impact:** High but unproven on this label budget. Treat as the destination architecture once A1, A2, B1, B2, B3 land and dataset clears 100K rows.

**Acceptance signal:**

- Beats cross-attention baseline by ≥2 points held-out win rate at matched compute.
- Training stable without aggressive scheduling tricks.

**Overlap with v1:** None direct.

**Evidence:** N/A — greenfield.

---

## P1: Scale The Data Pipeline

### C1. Worker-Thread Parallelization Of Exporters

**Problem:** `exportTrainingExamples.ts` and `exportOutcomeTrainingExamples.ts` both run a single-threaded `for` loop over seeds. CRN cost is `maxActions × samples × rolloutSteps` per labeled state, single-process. Wall clock scales with cores ignored.

**Direction:** Add a worker pool (`worker_threads` or `piscina`) that shards seeds across cores; each worker writes its own JSONL shard; a final pass concatenates and writes the manifest. Preserve per-shard seeds and per-shard manifests.

**Expected impact:** High. ~`min(cores, games)` speedup with no algorithmic change. Required to make any of the C2/C3/C4 items affordable.

**Acceptance signal:**

- Export accepts `--workers N`; smoke proves shard determinism (same seed → same row).
- Manifests record per-worker seed shards; concat preserves order or seed-stable order.
- Wall-clock scales near-linearly with cores up to physical core count.

**Overlap with v1:** Touches v1 item 18 (manifests) — shard manifests must extend the existing manifest schema.

**Evidence:** `backend/src/sim/exportTrainingExamples.ts:55`; `backend/src/sim/exportOutcomeTrainingExamples.ts:55-60`.

---

### C2. Adaptive CRN Sample Count And Rollout Depth

**Problem:** Defaults are `--samples 1` and `--rollout-steps 500` for every state. Most decisions are not close calls. Spending equal compute on every state both undersamples critical states and wastes rollout on obvious ones.

**Direction:** Heuristic: cheap initial pass with `samples=1`, then expand `samples` to 5–10 only on states where top-2 candidates are within a margin threshold. Optionally adapt `rolloutSteps` by turn number.

**Expected impact:** Medium-high. Same wall-clock can produce noticeably better labels on the states that matter.

**Acceptance signal:**

- Adaptive flag exposes `--initial-samples`, `--expand-samples`, `--margin-threshold`.
- Per-state sample count recorded in the oracle field (A5).
- Held-out training comparing fixed-1 vs. adaptive at matched wall-clock shows lift on margin-bucket metrics already added in v1.

**Overlap with v1:** Direct extension of v1 item 2 (CRN samples).

**Evidence:** `backend/src/sim/exportOutcomeTrainingExamples.ts` (rollout loop and `--samples` arg).

---

### C3. Trajectory Bundling At Dataset Load

**Problem:** Rows are independent. `episodeId` and `step` allow grouping but no loss exploits it. Sequence-aware training (returns over remaining trajectory, opponent-conditional context, multi-step value targets) is impossible without bundling.

**Direction:** In `JsonlPolicyDataset`, group by `episodeId` and expose trajectory slices as an alternate `__getitem__` mode. No exporter change needed — this is purely a Python loader feature.

**Expected impact:** Medium. Enables D-track sequence losses and better value-head training without regenerating data.

**Acceptance signal:**

- Loader supports per-episode iteration and per-state iteration in one dataset class.
- Sequence-mode batches produced with masking; smoke covers ragged-length collation.

**Overlap with v1:** Compatible with v1 items 5, 6 (DAgger / episode-split).

**Evidence:** `training/uma_ai/dataset.py`.

---

### C4. Stratified Sampling By Phase / Action Kind / Game Stage

**Problem:** A 360-step game has many "pass" and "end-turn" decisions and few evolve/trainer/combat critical decisions. Uniform sampling drowns the model in trivial states.

**Direction:** Add `--phase-weights` / `--kind-weights` to the loader (or a `WeightedRandomSampler`). Default to over-sample evolve, attach, ability, combat decisions and under-sample pass/end-turn that have only one legal action.

**Expected impact:** Medium. Especially impactful for rare-but-important phases.

**Acceptance signal:**

- Sampler config recorded in training manifest (extends v1 item 18).
- Per-phase accuracy metrics show no regression on common phases and improvement on rare phases.

**Overlap with v1:** None direct.

**Evidence:** `training/uma_ai/dataset.py`; v1 progress doc already notes phase/action-kind grouped metrics exist.

---

### C5. Randomized Deck Sampling For Generation

**Problem:** `setupAiVsAiGame()` uses one fixed deck pair. State diversity is bounded by what one matchup can reach. Card-aware models will overfit to that matchup.

**Direction:** Define a deck pool (or programmatic deck generation) and sample matchups per game. Record deck IDs in row metadata so we can stratify training and eval by matchup.

**Expected impact:** Medium-high. Required before any embedding/architecture work generalizes beyond one matchup.

**Acceptance signal:**

- Generation accepts `--deck-pool` or `--deck-pair-sampler`.
- Manifests record per-game deck pair.
- Eval gate (v1 item 17) supports per-matchup breakdown.

**Overlap with v1:** Strengthens v1 item 15 (rebaseline) — the rebaseline should be on a sampled-matchup suite, not a single matchup, before it's worth re-running.

**Evidence:** `backend/src/sim/headlessAiVsAi.ts` (single deck pair setup).

---

### C6. Self-Play And Paired Two-Player Generation

**Problem:** All current data is "policy X plays opponent Y." We have no symmetric self-play data, and we don't capture paired (player-perspective, opponent-perspective) rows from the same trajectory.

**Direction:** Add a self-play exporter mode where the trained model plays itself, with optional epsilon noise for exploration. Capture both sides' decisions per game. Useful both as a data source and as a non-stationary training curriculum.

**Expected impact:** High once the teacher (v1 item 1 planner) is strong enough to make self-play interesting. Premature otherwise.

**Acceptance signal:**

- Self-play exporter runs with the same manifest discipline as v1.
- Training mix can include self-play data with source taxonomy (extends v1 item 4).
- Closed-loop eval improves over rule-bot-only training when self-play data is added.

**Overlap with v1:** Builds on v1 items 1, 5 (planner + DAgger).

**Evidence:** `backend/src/sim/headlessAiVsAi.ts`, `evaluateModelVsHeuristic.ts`.

---

## P1: Training Infrastructure For Bigger Runs

### D1. Wandb / TensorBoard Logging

**Problem:** Per-epoch JSON only. No per-batch loss, no gradient norms, no live curves. Diverging large-model runs will not be diagnosable.

**Direction:** Add wandb (or tensorboard) integration with per-batch loss/gradient-norm/lr logging, per-epoch eval metrics, manifest as a wandb artifact. Make it opt-in via env var so smoke tests stay offline.

**Expected impact:** Medium. Necessary tooling, not a strength gain in itself.

**Acceptance signal:**

- Training runs log per-batch and per-epoch metrics when env is configured.
- Smoke tests run unchanged with logging disabled.
- Existing manifest.json continues to be written.

**Overlap with v1:** Extends v1 item 18 (manifests) into live observability.

**Evidence:** `training/train_bc.py:46-53`.

---

### D2. IterableDataset / Sharded JSONL.gz

**Problem:** `JsonlPolicyDataset` loads full JSONL into memory. At 500K rows this hits ~2–5 GB RAM; at 5M it falls over. A1/A2 will also balloon per-row size.

**Direction:** Switch to `torch.utils.data.IterableDataset` reading from sharded gzipped JSONL with per-shard shuffle buffer. Keep map-style loader for small smoke runs.

**Expected impact:** High at scale. Without this, A1+A2+C1 work is bottlenecked at the loader.

**Acceptance signal:**

- Loader handles 1M+ row sharded dataset with bounded memory.
- Per-shard shuffle preserves training stability vs. global shuffle.
- Smoke tests still use map-style loader.

**Overlap with v1:** None direct.

**Evidence:** `training/uma_ai/dataset.py`.

---

### D3. LR Schedule, AMP, Gradient Accumulation, Resume

**Problem:** Flat LR, fp32, no accumulation, no resume from optimizer state. Fine for current runs; blocking once architecture and data scale up.

**Direction:** Add cosine LR with warmup, `torch.autocast` AMP, gradient accumulation, and resume-from-checkpoint that restores optimizer + scheduler + RNG state. Wire into existing manifest.

**Expected impact:** Medium. Each piece is standard; together they are required for multi-hour runs.

**Acceptance signal:**

- All four features present and individually toggleable.
- Resume-from-checkpoint produces bit-identical metrics to uninterrupted run on a smoke fixture.
- Manifest records optimizer state and scheduler config.

**Overlap with v1:** Compatible with v1 items 18, 19.

**Evidence:** `training/train_bc.py:42-53`.

---

### D4. Multi-GPU / DDP Readiness (Stretch)

**Problem:** Single-GPU only. If B5 transformer + 100K+ rows lands, single GPU will be the bottleneck. Adding DDP later is harder than designing for it now.

**Direction:** Wrap model + sampler in DDP-friendly factories even when running single-process. Confirm one config can run `torchrun --nproc_per_node 1` without behavior change.

**Expected impact:** Low immediate, high optionality. Defer until B5 lands.

**Acceptance signal:**

- Single-process and `torchrun --nproc_per_node 2` produce equivalent metrics on a tiny dataset.
- No DDP code paths are taken when world size is 1.

**Overlap with v1:** None.

**Evidence:** Greenfield in `train_bc.py`.

---

## P2: Self-Supervised Pretraining (Exploratory)

### E1. Masked Card / Masked Action Pretraining

**Problem:** Even at 100K labeled rows, the embedding/encoder benefits from a self-supervised warmup. Card co-occurrence patterns ("if hand has stage-1 evolver, board likely has matching basic") are learnable from cheap unlabeled data.

**Direction:** Generate unlabeled state snapshots from headless simulation (no rollout cost). Mask 15–20% of card slots and predict from context. Initialize the policy run from the pretrained encoder.

**Expected impact:** Medium-high if labeled-data scaling stalls. Low if data scale grows fast.

**Acceptance signal:**

- Pretrained encoder beats random init by ≥2 points held-out at matched labeled-data budget.
- Pretraining recipe is reproducible from manifest alone.

**Overlap with v1:** None direct.

**Evidence:** Greenfield.

---

## Suggested Implementation Order (v2 Track)

1. **A5 (oracle persistence)** — One-day cleanup with immediate downstream value.
2. **C1 (worker parallelization)** — Unlocks every later C item by making generation cheap.
3. **A1 (per-card embeddings)** — Foundation for A2/A3/A4 and B-track entirely.
4. **A2/A3/A4 in parallel** — Set encoder, metadata enrichment, history slice. All schema-bumps; bundle into one TS↔Python schema migration aligned with v1 item 8.
5. **D1/D2/D3 (training infra)** — Required before architecture sweeps are credible.
6. **C5 (deck sampling)** — Required before we trust generalization results from richer features.
7. **B1 + B3 (cross-attn + auxiliary heads)** — First architecture upgrade that exploits the new features.
8. **B2/B4 (asymmetric scaling, distributional value)** — Sweeps once D1 logging exists.
9. **C2/C4 (adaptive CRN, stratified sampling)** — Tune label quality.
10. **C6 (self-play)** — Only after teacher quality from v1 item 1 makes self-play data better than rule-bot.
11. **B5 (transformer)** — Destination architecture; gate on previous items.
12. **D4 (DDP)** — Gate on B5.
13. **E1 (pretraining)** — Only if labeled scaling stalls.

## Consolidation Notes For Merging With v1 Backlog

| v2 item | v1 overlap | Suggested merge |
| --- | --- | --- |
| A1, A2, A3 | v1-8, v1-9 | Bump schema once for embeddings + set encoder + metadata; one acceptance signal. |
| A4 | v1-5 | History slice should travel in DAgger trace rows from the start. |
| A5 | v1-2, v1-4, v1-18 | Already-computed oracle metadata is the cheapest fix to v1-2's "rows include sample/margin metadata" line. |
| B3, B4 | v1-14 | Distributional value + auxiliary heads consolidate v1's value-head item. |
| C1, C5 | v1-15, v1-18 | Rebaseline must run on sampled-matchup suite produced by parallel exporter. |
| C2 | v1-2 | Adaptive CRN is the natural next step after CRN sample plumbing. |
| C3 | v1-5, v1-6 | Loader-side feature; no exporter changes needed. |
| C6 | v1-1, v1-5 | Self-play depends on stronger teacher and DAgger trace already on v1. |
| D1, D2, D3, D4 | v1-18, v1-19 | Manifests/smoke discipline extends naturally into live logging and resume. |

## Non-Goals For The Capacity Track

- Do not pursue B-track architecture work before A-track features land — symmetric MLP on hash features is the correct architecture for the current observation.
- Do not scale dataset volume (C-track) before C1 parallelization — single-thread generation will time-out before producing useful volume.
- Do not promote larger models on row-level accuracy alone; gate on the v1 eval gate (item 17).
- Do not let v2 schema bumps invalidate v1's reproducibility discipline — every change here must extend the v1 manifest format, not replace it.
