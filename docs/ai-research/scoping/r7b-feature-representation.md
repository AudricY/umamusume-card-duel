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
