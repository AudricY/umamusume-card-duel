# R7.b. Feature Representation Expansion — Scoping

- **Date:** 2026-05-14
- **Status:** scoping in progress (no code yet); parked behind R7
- **One-liner:** Expand the policy's input representation — primarily a per-card embedding table replacing the 16 hashed-float identity slots — to test whether the F1 cap at Wilson 0.368 is binding on *information available to the network*, not on labels (R7) or objective (R8).
- **Forward brief:** `docs/ai-research-backlog.md` § "R7.b. Feature representation expansion".
- **Pick rationale:** R15.S3 closeout language ("needs a different information source, not more tuning") + four unfinished pointers in `docs/ai-performance-research-progress.md` (L362 item 5b set encoders / history / embedding model; L1547 feature-representation gap; L1625 state-feature gap; L1727 state-feature audit) — never landed.

## 1. Question (Q)

> Does the current 96-d hand-engineered observation lose information that caps the policy at Wilson 0.368 regardless of label source (R7) or optimization objective (R8)?

Falsifiable proposition: a representation-expanded warm-start either (a) clears Wilson lower ≥ 0.40 on one phase-H-scale PPO sweep from a multi-teacher (R7-style) warm-start, or (b) closes the representation branch with the new measured ceiling at the expanded-feature schema.

## 2. Hypothesis + Motivation

**Hypothesis.** The 96-d encoder collapses several decision-relevant categories — most acutely card identity (16 slots are single hashed floats) and per-bench attached state — into representations that demonstrably alias distinct game states. Expanding the input restores function-class capacity along axes the post-mortem named.

**Why now.** R7's pre-flight teacher-agreement probe (2026-05-14) confirmed teachers genuinely diverge (rollout↔planner 0.18, all-three-agree 0.13). If R7 lifts the SL ceiling but PPO still saturates near 0.368, the residual cap is most plausibly representational, not label-quality. R7.b is the structural lever for that scenario.

**Strength.** R12/R14.I.2 reach Wilson 0.6479 via *search-time* rollout-leaf MCTS — i.e. the *same* network family produces strong play when given more compute at inference. The pure-model gap of 0.6479 → 0.368 is the search→model distillation gap, and a richer representation is one of the shortest paths to closing it.

**Weakness.** A schema change invalidates every prior comparison and forces a full DAgger regeneration unless trace JSONLs preserve enough state to re-encode (see § 5 risk (a) and the precondition spike in § 6 step 0). If they don't, the cost balloons.

## 3. Information-Loss Audit (categories that R7.b targets)

Source: investigator audit 2026-05-14 against `training/uma_ai/features.py` (413 lines) and `frontend/src/game/engine/ai-policy/observation.ts`. Each category lists the lossy slot range, a one-line "indistinguishable pair" demonstrating the loss, and severity.

| # | Category | Slots / source | Indistinguishable pair | Severity |
|---|----------|----------------|------------------------|----------|
| 1 | Card identity collapse | slots 32–47 single hashed floats; hand/discard via mean-hash slots 10–12 | hand `[idx=10, idx=90]` vs `[idx=20, idx=80]` → same mean ≈ 0.472; role-bucket histogram (slots 68–76) ties iff cards share role | **HIGH** |
| 2 | Hand multiset structure | slot 10 mean-hash + slot 13 count + slots 68–76 role buckets | `[X, X]` vs `[X, Y]` where both same-role same-stage indistinguishable | MED-HIGH |
| 3 | Recent action history | fully absent; only own-side per-turn-used flags slots 29/30/31 | board reached via "opp gusted then attacked" vs "opp drew then attacked" → identical | **HIGH** |
| 4 | Opponent hidden info | slots 6/8/12 only (handCount, deckCount, discard-mean-hash) | opp 5-card hand `[KO×4, search×1]` vs `[draw×4, healer×1]` → identical encoding | **HIGH** |
| 5 | Per-card attached state (bench) | slots 14–25 are side aggregates only | bench `[A:60hp+poisoned, B:100hp]` vs `[A:100hp, B:60hp+poisoned]` → identical | MED |
| 6 | Multi-step planning context | turn/points captured (slots 2–4); `enteredTurn`, paralysis timer, attack buffs all stripped at `toPublicUmaObservation` | evolution-sickness window invisible | MED |
| 7 | Bench typed energies | only active uma typed energies encoded (slots 48–67) | bench `{fire:2, water:0}` vs `{fire:0, water:2}` collide on `energyTotal` | **HIGH for promotion play** |
| 8 | Discard composition | discard size (slots 26–27) + mean-hash + own role buckets (84–86) only | own discard `[supA, supA]` vs `[supA, supB]` (same role) → role tie + hash bracket | MED-HIGH |

**Adequately captured (do not re-engineer):** KO-this-turn booleans (slots 91–92), active typed-energy vs attack-cost match (slots 25, 78–81), stadium presence (slots 9, 94), basic phase/turn/points (slots 0–8).

**Free signals sitting unused (cheap hygiene):** `firstPlayer` and `pendingChoiceKind` are present in the JSON but never read by `observation_to_features`; `toolCardId` is present on `PublicUmaObservation` but never read. Wiring these is a half-day patch with no schema invalidation.

## 4. Intervention Menu

Source: investigator menu 2026-05-14 against `training/uma_ai/model.py`, `features.py`, and `backend/src/sim/`. Six interventions, ranked by conviction × leverage. **#1, #2, #3 are the production picks for the v1 launch; #4–#6 are deprioritised for explicit reasons.**

### #1. Card embedding table (per-zone sum-pool) — **headline**

- **Mechanism.** Replace the 16 identity-hash floats (slots 32–47) with `nn.Embedding(108, K=32)` lookups; sum-pool within 8 zones (own active / opp active / own bench / opp bench / own hand / own discard / opp discard / stadium) → 256 features fed via a separate trunk projection `nn.Linear(8*32, hidden)`.
- **TS / Python plumbing.** Observation emits per-zone int id arrays (`buildPublicObservation` extension at `frontend/src/game/engine/ai-policy/observation.ts:6-23`). Python features.py reads them; `CandidatePolicyNet.__init__` adds the embedding module + zone projection; ONNX export gains a new int input tensor (`training/export_onnx.py:38-55` adds it). Vocab-hash guard at `export_onnx.py:22-30` already exists; embedding inherits.
- **Cost.** ~1.5 days code; full SL retrain (~30 min); schema bump v3.
- **Targets.** Audit categories 1, 2, 5, 8.
- **Conviction.** **High.** Single biggest information lever; both investigators ranked #1.

### #2. Action-target embedding (gated on #1) — **bundled with #1**

- **Mechanism.** After #1 lands, the 48-d action vector gathers its source/target card identity from the same embedding table. `CandidatePolicyNet.forward` accepts `action_target_card_idx: LongTensor[B, A]` and `action_source_card_idx`; concat the 32-d embed to the action_encoded path before the state×action interaction.
- **Cost.** ~half day on top of #1; ONNX adds two int inputs; schema bump (action features carry int ids).
- **Targets.** Closes the action-side identity gap (action features today carry target hp/energy/stage but only a hash for identity).
- **Conviction.** **Medium-high conditional on #1.** The candidate-conditioned trunk concatenates `[state, action, state*action]`; identity participating in the interaction term is the natural completion of #1.

### #3. Set-encoder / attention over per-card tokens — **conditional follow-up**

- **Mechanism.** Replace the per-zone sum-pool with a small set encoder. Per-card token = concat(embedding[K=32], one-hot zone[8], scalar features per card[hp_ratio, energy_total, stage, status_count, used_ability]) → 1–2 layer self-attention (4 heads, hidden=64) → mean-pool to hidden.
- **Cost.** +1 day on top of #1 (~3 days total); ONNX MHA at opset 17 is fine, smoke needed.
- **Conviction.** **Medium-high IF #1 lifts and the lift looks bottlenecked by zone-aggregation.** Defer if #1 alone clears 0.40; else attention is the right next compute spend (preferred over capacity bump #5 below).

### #4. Recent action history (last-N self + opponent) — **deprioritised, separate branch**

- **Mechanism.** Engine ring-buffer of last N=4 own + N=4 opp actions; encode each as `embed_card(src) + embed_card(tgt) + onehot(kind, 13) + scalar(turn_delta)`; aggregate to 32-d via small GRU or sum.
- **Cost.** ~2 days. **Critical risk:** existing SL JSONLs have no history field; if the trace JSONLs don't preserve enough state to reconstruct history offline, BC datasets become useless until regenerated. Confirm in § 6 step 0 spike.
- **Conviction.** **Medium.** Targets audit category 3, but most F1-cap decisions are short-horizon where the immediate observation already carries most state. Best as a PPO/self-play lever once #1 lands.

### #5. Capacity bump (hidden 128→256, depth 3→4) — **revisit-after-#1 only**

- **Mechanism.** `ModelConfig(hidden_dim=256, depth=4)` (`training/uma_ai/model.py:15-16`). No schema change.
- **Cost.** ~10 minutes code, retrain only.
- **Conviction.** **Low alone, medium as #1 follow-up.** R15.S3 already showed tuning saturated at the current capacity; only worth retrying once #1 gives the trunk new signal to consume.

### #6. Auxiliary heads (predict opponent next action + Δ value) — **orthogonal regulariser**

- **Mechanism.** Two `nn.Linear(hidden, {13, 1})` aux heads. Loss = policy + value + 0.1·aux_action + 0.05·aux_value_delta. Targets need replay-time labels (opponent action from next event, value-delta from later evaluation).
- **Cost.** ~2 days (label plumbing dominates).
- **Conviction.** **Speculative.** Won't help if cap is genuinely representation-bound; do #1 first.

## 5. Cross-Cutting Dependencies + Risks

**Risks (in priority order).**

- **(a) Trace-JSONL re-encodability — gating.** Whether existing R12/R13/R14 SL datasets can be re-extracted under a new schema without resimulating depends on what the raw event traces preserve. If not, every schema bump = full DAgger regeneration (multi-hour compute). Spike this in step 0 (§ 6) before committing to #1.
- **(b) PPO interaction with embedding tables.** The six R15.S3 PPO phases all ran on the 96-d schema. New embedding-bearing trunk may have different value-head calibration and interact unpredictably with PPO's advantage estimation. Mitigation: hold PPO hyperparams identical to phase H/N for the first sweep; treat divergence as a finding.
- **(c) Vocab is small (107 cards).** Embeddings can overfit on rare cards; need padding mask for empty slots; ONNX `Gather` opset 17 supports it but per-zone pad logic adds correctness surface.
- **(d) R7 dependency.** R7.b assumes R7 surfaces useful gradient signal. If R7's teacher-blend warm-start does not lift, R7.b inherits the same weak supervision and a richer representation has nothing to fit.

**Plumbing notes (reduce cost).**

- `card_vocab_index` already exists in `features.py:371-383` with `unknownIndex` fallback and suffix handling (`FullArtGold`/`FullArt`/`UncommonPlus`/`Ex`). Embeddings call it directly.
- ONNX export vocab-hash guard at `training/export_onnx.py:22-30` is already wired; embedding work inherits validation.
- `frontend/src/game/engine/ai-policy/observation.ts:6-23` is the single chokepoint for adding zone-id arrays (#1) and `recentActions` (#4); both also need `types.ts` updates and a corresponding `r7TeacherAgreementProbe.ts` / dagger pipeline regen.

## 6. Ordered Execution Plan

0. **(Precondition spike) Trace-JSONL re-encodability.** ~30-min spike: read one R14-vintage trace JSONL and confirm whether `observation_to_features` can be re-run under a new schema without resimulating the game. If yes, schema bumps cost only feature re-extraction. If no, every schema-changing intervention also pays a full DAgger regen. **Gating step — promote R7.b only after this answer is known.** Autonomous-launch eligible.
1. **(Hygiene quick-win) Wire unused JSON fields.** Read `firstPlayer`, `pendingChoiceKind` in `observation_to_features`; read `toolCardId` per-uma. Bumps STATE_DIM by ~6–10 floats; schema v2.1 (additive). ~half day. SL gate eval to confirm no regression. Autonomous-launch eligible. Pulls in cheap signal independent of R7.
2. **(Headline pass) #1 Card embedding + #2 Action-target embedding bundled.** Implement together because #2 reuses #1's table. Single SL retrain + one PPO sweep at phase-H scale. Pre-registered exit per § 7.
3. **(Conditional) #3 Set-encoder / attention.** Launch only if step 2 lifts but caps below 0.40. Same SL + PPO eval pattern.
4. **(Closeout writeup.)** Result block in `docs/ai-performance-research-progress.md`; backlog R7.b family shrinks to one-line pointer per CLAUDE.md doc discipline.

Steps 4 (history) and 6 (aux heads) from the menu are explicitly *not* in the v1 plan — re-evaluate after step 2.

## 7. Pre-Registered Exit Gates

Verbatim from `docs/ai-research-backlog.md` § R7.b:

> (a) SL warm-start does not regress vs current best (no harm from added dim) AND (b) one PPO sweep Wilson lower ≥ 0.40, OR document the new ceiling at 96-d and close the representation branch.

**Gate evaluation procedure.**

- **SL non-regression gate (a):** train on the same trace corpus used by R7 (or R15.S1 if R7 has not landed), eval with `npm run sim:eval-gate` against the rule-bot at the same n / seed range. Pass = Wilson lower of the new schema's warm-start ≥ Wilson lower of the prior 96-d warm-start − 0.02 (allow 2pp slack for noise).
- **PPO gate (b):** one phase-H/N-scale PPO sweep from the new warm-start. Pass = Wilson lower ≥ 0.40.
- **Close-out condition:** if (a) fails, the schema bump is hurting — debug or close. If (a) passes but (b) fails, we have a new measured representation-bound ceiling; record it in the progress doc and close R7.b.

## 8. Out of Scope (deliberately deferred)

- DPO / RL objective change (R8).
- Reward-shape signals (R15.S3 closed across both axes).
- Multi-iteration DAgger from the new warm-start (v1 is one SL + one PPO sweep).
- Capacity bump (#5) and aux heads (#6) — gated on #1 outcome.
- Recent-action history (#4) — separate branch, gated on the trace-reencodability spike.
- Q-head replacement (R9), structural self-supervised aux (R11) — historical, deprecated by R12 GO.

## 9. Result: R7.b.0 trace-reencodability spike (2026-05-14)

**Verdict: YES.** Existing R12/R13/R14/R15.S1-vintage trace JSONLs can be re-extracted under a new feature schema without resimulating. Every R7.b schema-bumping intervention costs only feature re-extraction (CPU-bound JSON read + numpy pass over rows), not a full DAgger regeneration.

**Evidence.** Sampled `runs/R14-f1-self-play-sweep/iter-000/trace.jsonl` (row1 = 2051 bytes, 228MB file) and `runs/R15-S1-warmstart-sweep/iter-000/trace.jsonl` (row1 = 2025 bytes, 27MB file). Both rows carry `schemaVersion: 1` and identical top-level shape. Each row stores the **raw `PublicObservation` snapshot** under `observation`, not just the encoded vector. Field-by-field check against `observation_to_features` (`training/uma_ai/features.py:34-65`):

| Encoder input | Trace row field | Present? |
|---------------|-----------------|----------|
| `phase`, `sideToAct`, `turnNumber` | `observation.{phase, sideToAct, turnNumber}` | yes |
| `own/opponent.{points, handCount, deckCount, energyZone, discard, usedSupporterThisTurn, usedRetreatThisTurn, usedStadiumThisTurn}` | identical paths | yes |
| `own/opponent.active.{cardId, species, stage, hp, maxHp, energyTotal, energies{10 types}, specialConditions, toolCardId, uid, usedAbilityThisTurn}` | identical paths | yes (full per-card detail on every bench slot too) |
| `own/opponent.bench[i]` (4 slots) | identical, null-padded | yes |
| `own.handCardIds` (true ids) | present on own only | yes |
| `shared.stadiumCardId` | identical | yes |
| `firstPlayer`, `pendingChoiceKind` (unused today, hygiene-pick targets) | `observation.{firstPlayer, pendingChoiceKind}` | yes (free for R7.b.1 to wire offline) |
| `legalActions[i].{features, id, kind, phase, payload}` | identical | yes (`payload` carries `targetUid` / `handIndex` / `attackIndex` / retreat target — enough to recompute action-target embeddings for #2 over the stored observation) |

**Surprise (mild).** Rows carry *more* than encoder-inputs need: `selectedActionId`, `heuristicSelectedActionId`, `behaviorPolicy`, `teacher` (R15.S1 only, post-R7 step 2), `result`, `seed`, `sideId`, `modelSide`, `source` — i.e. the corpus also retains label provenance for R7-style teacher relabeling. Only opponent-side hidden info absent (`opponent.handCardIds`) — but encoder already doesn't read this (slot 12 uses discard-mean-hash only), so its absence does not constrain R7.b interventions #1, #2, #3, #5 over the public-info policy assumption.

**Cost implication.** Every R7.b schema bump (v2.1 hygiene → v3 embedding → v3.x set-encoder) re-extracts features by a Python script that reads each `trace.jsonl` line, calls a new `observation_to_features_vN(row['observation'])` + recomputes action vectors from `row['legalActions'][i].payload`, and writes a new feature-cache file alongside the trace. Order-of-magnitude wall-clock at ~2KB/row, ~50k rows/sweep: tens of seconds per corpus, not multi-hour. **DAgger regen is NOT required for any schema-change-only intervention in the v1 R7.b plan.** Exception: intervention #4 (recent action history) needs cross-row state that is not in any single trace row — confirm row-stream offline reconstruction works for that one separately when R7.b.4 promotes.

**One-line digest pointer.** R7.b.0 verdict YES: R14/R15.S1 trace rows carry raw `PublicObservation` + legalAction payloads, so schema bumps re-extract from JSONL (~tens of sec) instead of regenerating DAgger (~hours).

## 10. Result: R7.b.1 hygiene wire-unused-fields (2026-05-14)

**Verdict: LANDED.** Schema v2.1 (additive) wires three free JSON fields the encoder ignored: `firstPlayer`, `pendingChoiceKind`, per-uma `toolCardId`. STATE_DIM 96 → **110** (+14 scalar slots); `STATE_FEATURE_SCHEMA_VERSION` 2 → **2.1**.

**Layout (additive, slots 96-109).** `[96]` firstPlayer polarity vs `sideToAct` (+1 own / -1 opp / 0 absent). `[97:100]` `pendingChoiceKind` one-hot in hard-coded order `[none, promoteAfterKnockout, switchAfterGust]` (`PENDING_CHOICE_KINDS` in `features.py` — append-only). `[100:110]` per-uma `toolCardId` hashed-float, 10 slots in `_identity_features` accounting order (own active, own bench 0-3, opp active, opp bench 0-3); null → 0.0; uses the same vocab-indexed `_hash_to_unit` as the existing identity-hash slots.

**TS-side check.** All three fields already emitted at `frontend/src/game/engine/ai-policy/observation.ts:13-14, 57`; no TS edits required.

**Files changed (LOC delta).** `training/uma_ai/features.py` +63 (3 helper fns + ablation hook + slot wiring + header note); `training/train_bc.py` +1 (`state_hygiene_v21` added to `--ablate` choices). All downstream `STATE_DIM` consumers (`dataset.py`, `selfplay_dataset.py`, `value_target_dataset.py`, `pair_corpus.py`, `train_bc.py`, `train_ppo.py`, `export_onnx.py`, `serve_onnx.py`, `smoke_e2e.py`) pick up the new dim via the constant — no edits. Existing 96-d checkpoints fail loud on load via strict `model.load_state_dict` mismatch + `export_onnx.py:17-21` config check.

**Smokes (TMPDIR=/tmp).** `npm run build` PASS; `npm run test:train` PASS (6/6 TS smokes); `npm run test:python-train` PASS — manifest verifies `feature_schema.state_dim=110`, `state_feature_schema_version=2.1`, `model_config.state_dim=110`, ONNX roundtrip PASS, served prediction matches direct ONNX.

**Followup.** SL gate retrain on the R7 corpus to confirm Wilson non-regression (scoping § 7 gate (a)) is **out of scope for this slot** — single-iter retrain + n=500 gate eval (~30 min). Will be reused by R7.b.2 (card-embedding pass) anyway, so deferring is cheap.

## 11. R7.b.2 Execution Plan (concrete, 2026-05-14, post-R7.b.1)

Detailed phase-by-phase plan for § 6 step 2 (the headline pass: #1 card embedding + #2 action-target embedding, bundled). Total ~515 LOC, ~1.5 days code + ~30 min retrain + ~5–10 min gate eval. Phases 1–3 sequential; Phase 4 parallel with Phase 5 SL training; Phase 6 follows.

### Phase 1 — TS-side schema bump (~80 LOC)

- `frontend/src/game/engine/ai-policy/types.ts:23-37`: bump `schemaVersion: 1 → 2`; add `cardIdsByZone: Record<ZoneKey, number[]>` on `PublicObservation`; new `ZoneKey` type with 8 members (`"ownActive" | "oppActive" | "ownBench" | "oppBench" | "ownHand" | "ownDiscard" | "oppDiscard" | "stadium"`).
- `types.ts:15-21` (`LegalAiAction`): add `actionSourceCardIdx: number | null` and `actionTargetCardIdx: number | null` (or `-1` sentinel, matching Python pad convention).
- `frontend/src/game/engine/ai-policy/observation.ts:6-23`: extend `buildPublicObservation` to emit `cardIdsByZone`. Per-zone source: own/opp `active.cardId`, `bench[i].cardId`, `own.hand`, `own.discard`, `opp.discard`, `state.stadium?.cardId`.
- Per-action source/target uid → cardId → vocab idx happens where actions are built (legal-actions enumerator). Grep `backend/src/sim/` for action `features` array construction.
- **Risk:** need TS-side `cardVocabIndex(cardId)` helper producing identical indices to Python. Grep for existing helper in `shared/src/`. If absent, ~20 LOC + a TS-Python parity smoke.

### Phase 2 — Python encoder change (~190 LOC)

- `training/uma_ai/features.py` (~80 LOC): add `observation_to_card_ids(observation) -> dict[str, np.ndarray]` returning fixed-shape int array per zone with `0` padding (own bench/opp bench up to 4, hand up to 10, discards up to N_cap=30, active/stadium single). Wire `card_vocab_index:459`. Schema bump `STATE_FEATURE_SCHEMA_VERSION 2.1 → 3.0`; parallel constant `CARD_ID_SHAPES: dict[str, int]` for collator pad widths.
- `training/uma_ai/model.py:48-66` (~60 LOC): `self.card_embed = nn.Embedding(108, 32, padding_idx=0)` (index 0 = unknownIndex + pad), `self.zone_projection = nn.Linear(8 * 32, hidden)`. `forward` accepts packed `LongTensor[B, 8, max_cards_per_zone]`; per zone: `embedded = self.card_embed(ids); mask = (ids != 0).unsqueeze(-1); pooled = (embedded * mask).sum(dim=-2)` → concat 8 × 32 → `zone_projection` → add to `state_encoded`. For #2: reuse `self.card_embed`; accept `LongTensor[B, A, 2]` (source+target); concat 32-d embed to `action_encoded` before joint projection. Widen `joint_projection` input dim.
- `training/uma_ai/dataset.py:90-134` (~50 LOC): `collate_policy_batch` adds packed `LongTensor[B, 8, max_cards_per_zone]` + `LongTensor[B, max_actions, 2]`. `PolicySample:21-35` gains two new fields. `load_policy_samples:52-87` calls new `features.py` helpers. Schema-version strict-check at line 64 bumps to `3`.
- `training/train_bc.py:355-371` (~5 LOC): widen positional `model(batch["state_features"], batch["action_features"], batch["action_mask"])` → keyword call adding `card_ids_by_zone` + `action_card_idx`. **No other trainer-side changes.** Soft-CE branch at `train_bc.py:362-369` keys on `policy_targets` in batch dict — fires automatically.

### Phase 3 — ONNX export change (~65 LOC)

- `training/export_onnx.py:38-55` (~25 LOC): two new dummy inputs `card_ids_by_zone: LongTensor[1, 8, MAX_CARDS_PER_ZONE]` + `action_card_idx: LongTensor[1, max_actions, 2]`. Add `input_names`, `dynamic_axes`. Vocab guard at lines 22-30 already wired; assert `model.card_embed.num_embeddings == card_vocab_metadata()["vocabSize"] + 1`. **Opset 17 supports `Gather` natively — no bump.**
- `training/serve_onnx.py:206-234` (`request_to_arrays`, ~40 LOC): build `card_ids_by_zone` + `action_card_idx` from observation/legalActions JSON using new `features.py` helpers; shape-validate; add to ORT input dict.
- `training/uma_ai/node_bridge.py`: **no change** (shells out to npm; new tensors flow as JSON fields).
- **TS-side ONNX client: no change.** Confirmed by grep: TS only sends `{ observation, legalActions, sampling }` to Python `serve_onnx`; Python builds all tensors. As long as `buildPublicObservation` emits `cardIdsByZone` and `legalActions[i]` carries source/target idx, Python rebuilds them.

### Phase 4 — Feature re-extractor (~120 LOC, parallelisable)

- New file `training/r7b2_extract_features.py` (~120 LOC, standalone). Reads `runs/R7-multi-teacher-warmstart/iter-000/mixed.jsonl` row-by-row, calls new v3 `observation_to_features` + `observation_to_card_ids` + reconstructs action source/target from `legalActions[i].payload.{targetUid, handIndex, attackIndex, retreatTarget}` mapped against `observation.{own,opponent}.{active,bench}.{uid,cardId}` + `observation.own.handCardIds[handIndex]`. Writes new JSONL with `schemaVersion: 3` plus new fields; preserves `policyTargets` unchanged.
- Wall-clock: tens of seconds (12387 rows × ~2KB/row, single-threaded numpy).
- Re-encodability evidence: § 9 confirms all needed fields preserved.

### Phase 5 — SL retrain (~30 min CPU)

- **Reuse R7's 12387-row mixed corpus** (`runs/R7-multi-teacher-warmstart/iter-000/mixed.jsonl`) over item17's 3870-row corpus despite R7's lower starting Wilson (0.2921 vs 0.311). Reasons: (a) R7 corpus carries 3-teacher mixture targets that exercise the soft-CE pathway — testing representational lift wants the best-available labels; item17 confounds schema change with label change. (b) Pre-reg gate (a) is "no regression vs prior best on this schema"; apples-to-apples requires R7 corpus.
- Soft-CE branch verification: `train_bc.py:362-369` keys on `policy_targets` being in batch dict — not on schema version. Re-extractor preserves `policyTargets`. Fires automatically.
- Train cmd: standard `train_bc.py --data <re-extracted v3 jsonl> --epochs 75 --hidden 64 --depth 2`. Same hyperparams as R7 baseline.

### Phase 6 — Gate eval (~5–10 min)

**`training/r8_gate_eval.py` reuses as-is.** Schema-agnostic (lines 115-176): chains `export_checkpoint_to_onnx` → `serve_onnx_context` → `run_eval_gate`. Checkpoint's `model_config.state_dim` asserted to match (lines 17-21).

```
python training/r8_gate_eval.py \
  --checkpoint runs/R7b2-card-embed/iter-000/model/checkpoint.pt \
  --out-dir runs/R7b2-card-embed/iter-000 \
  --games 500 --seed-start 9000 --min-ci-lower 0.40
```

### Pre-flight cost-cutters (concrete answers)

- **Fits in existing trainer? YES.** `train_bc.py` change is ~5 LOC (positional → keyword model call). No new loop/optim/loss plumbing.
- **New ONNX input tensor cost? LOW.** Two new int inputs; opset 17 `Gather` native; ORT-CPU embedding lookup is microseconds at batch=1. Smoke: `npm run test:python-train` already does ONNX roundtrip — extend assertion to new tensors. Residual risk = `padding_idx=0` semantics under ONNX export (verified for embedding via `Gather`; `(ids != 0).unsqueeze(-1)` mask exports cleanly).
- **Vocab fits exactly.** `cardVocab.json` has `vocabSize=107, unknownIndex=0`. Use `nn.Embedding(108, 32, padding_idx=0)` — index 0 doubles as unknownIndex (matches `card_vocab_index` behaviour) + explicit pad. No vocab bump.
- **DAgger regen avoided** (per § 9): every R7.b schema bump re-extracts from JSONL in tens of seconds. Phase 4 is the single touchpoint.
- **TS-side smokes already cover this.** `npm run test:dagger-orchestrator` exercises observation JSON → relabel pipeline; catches any TS-side `cardIdsByZone` shape break.

### Open questions (smoke-testable only)

- ONNX `Gather` with batched int inputs at opset 17, specifically `nn.Embedding(padding_idx=0)` export. Expected to work; fallback is manual `F.embedding(idx, weight)`. Extend `train_bc.py:283-300` roundtrip smoke.
- TS-side vocab consumer (`shared/src/cardVocab.ts` or equivalent) existence — grep first before adding duplicate.
- Padding cap for hand/discard. Recommend fixed cap **30** for ONNX shape stability; clip with WARNING log if exceeded.
- Action-target uid → cardId resolution at action build time. ~15 LOC TS helper; locate via grep on `payload.targetUid`.

### Validation gates (cumulative, per phase)

- **After Phase 3:** `npm run test:python-train` PASS (extend to assert ONNX roundtrip on new tensors); `npm run test:train` PASS; `npm run test:dagger-orchestrator` PASS.
- **After Phase 4:** re-extractor produces JSONL with `schemaVersion: 3`, expected row count, all card-id arrays non-empty.
- **After Phase 5:** SL training completes ~30 min, val_acc not catastrophically below R7's plateau peak (0.7855 @ ep16).
- **After Phase 6:** Wilson lower at n=500. Exit per § 7: ≥ 0.40 PASS, ≥ 0.37 ∧ < 0.40 → R7.b.3 attention follow-up, < 0.37 → close R7.b family + escalate.

## 12. Result: R7.b.2 Phase 1 — TS-side schema bump LANDED (2026-05-14)

**Verdict: LANDED.** `PublicObservation.schemaVersion 1 → 2`; new `ZoneKey` 8-member union; new `cardIdsByZone: Record<ZoneKey, number[]>` field on the observation; new `actionSourceCardIdx: number | null` and `actionTargetCardIdx: number | null` on every `LegalAiAction`. `null` is the sentinel for "no clear source/target" (endTurn / pass / useStadium / setup / pendingChoice has-no-source / attachEnergy has-no-source); Phase 2 Python collator maps `null` → 0 (the shared `padding_idx=0` of the embedding table, doubling as `unknownIndex`). Variable-length zones TS-side; Phase 2 collator pads to fixed shapes.

**Helper.** Step 4 added a new TS helper `shared/src/cardVocab.ts` (50 LOC) — no prior TS-side `cardVocabIndex` consumer existed (the only TS reference was `backend/src/sim/buildCardVocab.ts` which writes the JSON). The helper imports `cardVocab.json` via `resolveJsonModule` and mirrors Python's `card_vocab_index` (`training/uma_ai/features.py:459-471`) line-for-line: empty → 0; direct hit → integer; suffix fallback in order `FullArtGold > FullArt > UncommonPlus > Ex` (FullArtGold first because FullArt is a strict suffix); unknown → `unknownIndex` (0).

**Files changed (LOC delta).**
- `frontend/src/game/engine/ai-policy/types.ts` +28 / -2 (ZoneKey union, schemaVersion bump, two new action fields, cardIdsByZone field, doc comments).
- `frontend/src/game/engine/ai-policy/observation.ts` +28 / -3 (buildCardIdsByZone helper, opp-hand-hidden empty-array per zone rule, imports).
- `frontend/src/game/engine/ai-policy/actions.ts` +34 / -0 (per-call-site `actionSourceCardIdx` / `actionTargetCardIdx` resolution; comment block replaced unused `actionCardIdxs` helper draft).
- `shared/src/cardVocab.ts` +50 (new — TS-side parity helper).
- `backend/src/tests/cardVocabIndexSmoke.ts` +50 (new — TS-side unit smoke; wired into `test:train`).
- `backend/package.json` +1 / -1 (smoke registration).

Total delta ~93 modified LOC + 100 new LOC. The +13 LOC over scoping's ~80 LOC estimate comes from per-call-site source/target idx (12 LegalAiAction literals) + the parity smoke.

**Smokes (TMPDIR=/tmp).** `npm run build` PASS; `npm run test:train` PASS (7/7 TS smokes including new `cardVocabIndexSmoke`); `npm run test:dagger-orchestrator` PASS — observation → relabel pipeline unaffected by the additive schema bump (Python row-schemaVersion is the relabel row's own version, independent of the nested `observation.schemaVersion`).

**Out of scope (deferred to Phase 2).** Python encoder change (`model.py`, `features.py` per-zone int extraction, `dataset.py` collation); ONNX export; feature re-extractor; SL retrain; gate eval. `STATE_FEATURE_SCHEMA_VERSION` Python-side bump waits for Phase 2 (no Python consumer yet reads `cardIdsByZone`).
