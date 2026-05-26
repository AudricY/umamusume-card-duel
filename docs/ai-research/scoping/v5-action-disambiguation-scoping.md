# v5 Action Disambiguation (choice-card stat plumbing for search/discard/rainbow-evo trainer variants) — Scoping

- **Date:** 2026-05-26.
- **Status:** **DRAFT — USER-GATED.** Action-axis only; state schema unchanged. Pre-condition: v3.8 A1 verdict must land first so v5 has a stable predecessor (A0 reuse target). If v3.8 LIFTs or SOFT-SHIPs, v5 lands on top. If v3.8 HARD-FAILs, this doc is paused until the action-axis ceiling is re-anchored.
- **Parent:** `v38-slim-feature-add-scoping.md` (the v3 → v4 action-schema bump that introduced cross-bit slots [48:52]) + empirical action-feature degeneracy audit (this doc §13).
- **Predecessor schema:** v3.8 (state_dim=304, action_dim=52, action_feature_schema_version=4, hidden=128/depth=2). A0 reference TBD at fire time — REUSE whichever lineage-best ckpt the v3.8 A1 verdict promotes.
- **Scope:** Action-axis-only additive tail. **+5 action slots** at `[52:57]` encoding the per-action *choice card*'s stats (HP / primary-attack damage / total-cost / ability presence + a present flag). **No state-axis change.** **No trunk bump.** **No obs-contract bump.** Closes the empirically-measured 99.4%-of-degeneracy slice without re-opening any v3.x feature surface.

---

## TL;DR

- **ACTION_FEATURE_COUNT_V5 = 57** (+5 slots over v4 = 52). v4 head `[0:52]` byte-stable. State vector unchanged at 304-d.
- **ACTION_FEATURE_SCHEMA_VERSION = 5.** Second action-schema bump since the v33 cleanup (commit `5ea1758`, v2→v3) and the v3.8 cross-bit add (v3→v4).
- **Single new feature class:** the choice-card stats (`choice_card_hp_norm`, `choice_card_attack_damage_norm`, `choice_card_attack_cost_total_norm`, `choice_card_has_ability`, `choice_card_present`). All derived from existing `LegalAiAction.payload.choices` plumbing + static catalog. Same `getChoiceCardId` helper already feeds slots 42-45 (cardRole bits) — v5 adds raw stat slots alongside.
- **Single ablation:** A0 = v3.8 lineage best (REUSE post-v3.8-verdict). A1 = v3.8 + v5 action tail. **Wall ~30 min** following v3.8 cadence.
- **The bet:** the action-feature audit (this doc §13) found 13.6% of action slots / 3.2% of selected-action rows in the R16-P3-v36 corpus sit inside feature-identical degenerate pairs that the policy head cannot rank. 99.4% of those pairs are `3starMakeDebutScout` deck-search rows differing only in which deck card is fetched. Raw role bits (slots 42-45) already plumb the searched cardId but compress to 4 type buckets; multiple basic umamusume collide. v5 adds the per-card stat granularity that disambiguates within the role bucket. If v5 LIFTs, choice-card discrimination was bottlenecking the policy head; if FLAT, MCTS visit counts already break the tie behaviourally and the gradient cost of 3.2% smeared rows was inside Wilson noise.

**Why this is the minimal scope:** the empirical audit (§13) is unusually concentrated — one trainer card drives 99.4% of degeneracy in the current deck pool. The v5 design is `getChoiceCardId`-route only, so the same 5 slots simultaneously cover deck-search, hand-discard, and rainbow-evolution trainer variants without per-axis plumbing. The other audit candidates (combat sub-parameter variants — `useShuffleSelfIntoDeck`, `randomDiscardIndex`, `discardHandIndex`, `evolutionDeckCardIndex`, `switchTargetUid` cross-products) showed 0% degeneracy in the corpus after v4's slot 49 expected-damage differentiation; held until a corpus exercises them (deferred to v6 or post-deck-pool-change re-audit).

---

## 1. The v3.8 verdict and what this scope respects

v3.8 introduced the first action-axis thick add since the v33 slot cleanup (v3 → v4, +4 cross-bit slots at [48:52]). v5 inherits the schema-axis-closed verdict §4n hardens to also cover per-bench temporal + action cross-bit predicates (modulo v3.8's measured outcome). v5 does **NOT** reopen the schema-axis bet at h128/d2 — its claim is narrower and structural:

> The v4 action vector is **discriminatively incomplete** at the per-action-row level. Empirically 13.6% of action slots sit in feature-identical pairs where the policy logit is forced to be equal to within learned-bias, and 3.2% of selected actions sit in such a pair. The head cannot prefer one variant over the other from features alone; the gradient is smeared.

This is a different bet than "more bits help the trunk learn at h128/d2." It's "the head currently cannot express a ranking it provably needs." Even if v3.8's §4n verdict generalises to v5 (flat at h128/d2), the *expressivity* fix is independently defensible: the policy head should not be structurally degenerate.

### v3.8 lessons preserved

- **Channel orthogonality (v3.4→v3.8 doctrine).** v5 slots fire ONLY when `choiceCardId` resolves (= `payload.choices.{deckCardIndex | discardHandIndex | rainbowEvolutionHandIndex}` is set). All 5 slots = 0 for actions without a choice card. No overlap with v4's [48:52] cross-bits which fire on attack/retreatAttack/attachEnergy/useAbility-damage kinds.
- **Slot tokens stay retired (v3.4 doctrine).** UMA_SLOT_FEATURE_DIM unchanged at 23.
- **Locked vocabs at scope time** — choice-card stats are pure-function predicates over static catalog. No vocab risk.
- **No n=120 verdict-grade comparisons.** A1 arbitrated at n=10k tight-gate.
- **`feedback_bigger_fixes_at_once`** — all 5 slots land in a single commit (Phase A + B + C), single ckpt expander, single parity smoke.

---

## 2. Pre-registered hypotheses

### H1 — Choice-card stats discriminate within search-trainer variants

- **Falsifiable signal:** wl_lower(A1) ≥ wl_lower(A0) + 0.013 = (A0 + 0.013) → bundle lifts above the v3.8 anchor.
- **Theory:** §13 audit found 4,605 of 4,631 degenerate pairs in the iter-19 corpus are `3starMakeDebutScout` deck-search rows. The trainer fetches an umamusume from the deck; two rows differ only in `choices.deckCardIndex` (which umamusume is searched). `getChoiceCardId` already routes that index through `side.deck[deckCardIndex]`, but the resulting cardId is consumed only by slots 42-45 (4 role bits). Multiple basic umamusume collide on role bits (kind=1 / progression=0 / similar output bucket / ability=0|1). v5 stats (HP, primary damage, total cost, ability bit) refine within the role bucket — two basics with same role but different HP (e.g. 60 vs 80) now produce different feature vectors and the policy head can prefer one.

### H2 — The fix generalises to hand-discard and rainbow-evolution variants

- **Falsifiable signal:** post-A1 re-audit of the new corpus shows degenerate-group prevalence falls from 13.6% of slots to ≤2% of slots, with the residual NOT dominated by `playTrainer` kinds.
- **Theory:** `getChoiceCardId` covers three payload routes — `deckCardIndex` (search trainers), `discardHandIndex` (discard-cost trainers, e.g. for an "discard 1 to do X" effect), and `rainbowEvolutionHandIndex` (rainbow evolution). All three flow through the SAME `choiceCardId` field at action emission. The same 5 v5 slots cover all three routes — no per-route plumbing. If H1 confirms on search-trainers and H2 fails on the other two routes, the fix is route-incomplete and v6 must extend per-route. If H2 confirms, the choice-card axis closes in one slice.

### H3 — Parity surfaces hold

- **Risk:** v5 slots read `getCardById(choiceCardId)` and call `getPrimaryAttack` + read `card.ability` + `card.hp`. All catalog lookups; the parity-surface concern is bit-exact mirror between TS (`actions.ts`) and Rust (`engine-rs/.../policy/actions.rs::build_features`) and the cross-language smoke.
- **Falsifiable signal:** `v5_python_rust_parity_smoke.py` (Gate C STRICT on action[52:57]) fails on any fixture → block A1 fire. Existing v4 Gate B STRICT [48:52] band preserved.

### H4 — Choice-card axis is independent of v3.8's combat-arith axis

- **Falsifiable signal:** A1 lift profile by action `kind` shows the lift concentrated on `playTrainer` rows (not attack/retreatAttack/attachEnergy where v4's slots [48:52] are the load-bearing channel).
- **Theory:** v4's slots [48:52] target combat / attach decisions. v5's slots [52:57] target trainer-driven card-selection decisions. The kinds are disjoint; if A1 lifts, the lift attribution should be measurable per-kind.

---

## 3. Freeze contract

| Surface | v3.8 (current) | v5 (proposed) | Compatibility |
|---|---|---|---|
| `state_features` dim | 304 | **304 (unchanged)** | No state-axis change. |
| `STATE_FEATURE_SCHEMA_VERSION` | 3.8 | **3.8 (unchanged)** | No bump on state side. |
| `ACTION_FEATURE_COUNT` | 52 | **57** | Action-vector additive tail; +5 slots at [52:57]. v4 head [0:52] byte-stable. |
| `ACTION_FEATURE_SCHEMA_VERSION` | 4 | **5** | Second action-schema bump since v3.8 (`abb2d62`). |
| `_SCHEMA_BY_STATE_DIM` dispatch | {…, 304} | **unchanged** | Dispatch is keyed on STATE dim; action-axis bump uses the existing schema-version sidecar field instead. |
| Action-schema-version guard | accepts 4 when state_dim=304 | **accepts 5 when state_dim=304 (AND legacy 4 for v3.8-only graphs)** | Schema-version dispatch in `engine-rs/.../inference/mod.rs` extends `GraphSchema::V3_8` to carry an action_feature_schema_version variant; `pack_row` slices 57-d → 52-d for legacy v4 graphs (mirrors v3.8's 52→48 backward-compat shim). |
| ONNX input set | 5 inputs | **5 inputs unchanged** | `legal_action_features` last dim 52 → 57. `state_features` last dim unchanged 304. |
| `PublicSideObservation` | unchanged | **NO NEW FIELDS** | All v5 features read existing `LegalAiAction.payload.choices` + static catalog. No obs-contract bump. |
| Trunk dims (`hidden_dim`, `depth`) | 128 / 2 | **128 / 2 (unchanged)** | No trunk bump. v3.8 §13.3 overfitting evidence still applies. |
| `action_encoder` input | Linear(52 → hidden) | **Linear(57 → hidden)** | First action Linear weight shape [hidden, 52] → [hidden, 57]. New columns [52:57] zero-init via expander §3.3. |
| Init parity | v3.8 anchor | **v3.8 → v5 zero-init residual** | v3.8 ckpt loaded into v5 graph produces bit-identical iter-0 outputs IFF action-input cols [52:57] zero-init AND v4 cols [0:52] preserved bit-stable. |

### 3.1 Bit-exact mirror requirement

Action-vector v5 builder lands in:
- `frontend/src/game/engine/ai-policy/actions.ts` (TS canonical) — extend the v4 `features()` builder at lines ~496-560 with 5 new slot computations per §4.5.
- `engine-rs/crates/engine/src/policy/actions.rs::build_features` (Rust mirror) — extend the bit-identical port.
- `engine-rs/crates/engine/src/policy/featurize.rs` constant `ACTION_DIM = 52 → 57`.
- `training/uma_ai/features.py:50` Python constants `ACTION_FEATURE_COUNT = 57`, `ACTION_FEATURE_SCHEMA_VERSION = 5`.

Parity smoke `v5_python_rust_parity_smoke.py`:
- **Gate C STRICT** on action[52:57] across one fixture per `getChoiceCardId` route:
  - `playTrainer` with `deckCardIndex` set (search-trainer fetching umamusume).
  - `playTrainer` with `deckCardIndex` set (search-trainer fetching trainer, edge case where choice card is a non-uma).
  - `playTrainer` with `discardHandIndex` set (discard-cost trainer).
  - `playTrainer` with `rainbowEvolutionHandIndex` set (rainbow evolution).
  - Plus one negative fixture per kind WITHOUT a choice card (attack, attachEnergy, useAbility-damage, endTurn) confirming all 5 slots = 0.
- Existing v4 Gate B STRICT band on action[48:52] preserved.
- Existing v3 STRICT band on action[0:48] preserved.

### 3.2 Obs-contract surface — NO NEW FIELDS

All v5 features derive from `LegalAiAction.payload.choices.{deckCardIndex | discardHandIndex | rainbowEvolutionHandIndex}` (already in payload per `frontend/src/game/engine/core/playTypes.ts:1-6`, Rust mirror at `engine-rs/.../core/play_types.rs:5-12`) + static catalog (`getCardById`, `getPrimaryAttack`). No `PublicSideObservation` change. No backend obs change.

### 3.3 Existing checkpoint compatibility

- Pre-v5 ckpts (v3.0–v3.8) load via the existing `_SCHEMA_BY_STATE_DIM` paths; action-schema sidecar resolves to v3 or v4 → backward `pack_row` shim.
- Loading a v3.8 ckpt into a v5 graph requires `training/make_v5_action_init.py`:
  1. Reads v3.8 lineage best (TBD at fire time — A0-promoted ckpt from v3.8 A1 verdict).
  2. Expands `action_encoder[0].weight` by 5 columns at indices [52:57] zero-init. State-feature Linear unchanged (state schema v3.8 preserved).
  3. Bumps `model_config.action_feature_count → 57`, `feature_schema.action_feature_schema_version → 5`. `state_dim` and `state_feature_schema_version` UNCHANGED.
  4. Trunk dims UNCHANGED (`hidden_dim=128, depth=2` preserved).
  5. Saves a v5-shaped ckpt.
  6. Asserts bit-identical iter-0 outputs vs v3.8 source on a fixed observation batch (tolerance `1e-5`).
- Refuses non-v3.8 sources (sidecar `state_feature_schema_version != 3.8` or `action_feature_schema_version != 4`). Refuses ckpts with non-default trunk dims (defense against accidentally trunk-bumping in this slice).

### 3.4 Action-schema-version guard

Action loader guard at `engine-rs/.../inference/mod.rs:380-405` currently accepts {3, 4}. v5 extends to {3, 4, 5}. `action_dim_for_schema(5) = 57`. `pack_row` extends its slicing table: v5 binary running a v4 sidecar truncates 57-d row buffer to 52-d before ORT input; v5 binary running a v5 sidecar passes the full 57-d row. Mirrors v3.8's v3→v4 backward-compat shim verbatim.

---

## 4. Implementation order

**Three phases. Per `feedback_bigger_fixes_at_once`, related fixes bundle into single commits.**

1. **Phase A — Action featurizer (TS canonical + Rust mirror, single commit).**
   - Bump `ACTION_FEATURE_COUNT = 57`, `ACTION_FEATURE_SCHEMA_VERSION = 5` in `frontend/src/game/engine/ai-policy/actions.ts:50-51`, `training/uma_ai/features.py:50`, `engine-rs/crates/engine/src/policy/actions.rs:64-71`, `engine-rs/crates/engine/src/policy/featurize.rs:72`.
   - Extend TS `features()` builder at `actions.ts:~496-560` with 5 new slot computations per §4.5. Helper accepts `choiceCardId?: string` already (line 187 / 199 pattern).
   - Mirror in `engine-rs/.../policy/actions.rs::build_features` (lines ~1104-1746 region).
   - Extend `engine-rs/.../inference/mod.rs` `GraphSchema::V3_8` to also dispatch action_feature_schema_version=5 → `action_dim_for_schema(5) = 57`; `pack_row` slicing path adds v5 → v4 (truncate to 52) and v5 → v3 (truncate to 48) backward shims.
   - Smoke: `v5_action_v5_smoke.py` truth-table per [52:57] slot, fixtures one per `getChoiceCardId` route + one per kind without choice card.

2. **Phase B — Init parity + ckpt expander.**
   - Write `training/make_v5_action_init.py` per §3.3. Template: `make_v38_slim_init.py` (action-tail zero-init pattern; remove the state-tail expansion since state schema is unchanged here).
   - Smoke: `v5_action_init_smoke.py` asserts Δlogits ≤ 1e-5 vs v3.8 source ckpt on fixed observation batch.

3. **Phase C — Cross-language parity smoke + recipe + queue entry.**
   - `v5_python_rust_parity_smoke.py` per §3.1.
   - Rust parity fixtures: `engine-rs/crates/engine/tests/v5_python_parity_fixtures.rs` — 5 fixtures (one per choice-card route + 1 negative without choice card).
   - Recipe: clone v3.8 recipe (KL=0.05, W6-fix ON, 8×240 selfplay, MCTS sims=100, hidden=128, depth=2) — identical except init from `make_v5_action_init.py`. Reads state_features identically to v3.8 (state schema unchanged); action_features feed length 57.
   - Queue entry id: `v5-action-disambiguation`. A0 (v3.8 lineage REUSE) + A1 (v5 bundle) only. NO multi-arm bundle.

### 4.5 Bit / slot definitions (LOCKED here to prevent parity drift)

**Locked at scope time. Any changes after A1 fire require a fresh scoping doc (v5.1).**

#### Action channel — v4→v5 slot definitions (slots [52:57], 5 slots)

All 5 slots fire iff `choiceCardId !== undefined` (resolved via `getChoiceCardId(side, choices)` at `frontend/src/game/engine/ai-policy/actions.ts:704-709`). When `choiceCardId === undefined`, all 5 slots = 0.

| Slot | Field | Definition | Source |
|---|---|---|---|
| 52 | `choice_card_present` | 1 iff `getChoiceCardId` resolves a valid catalog card. **0 otherwise.** | `actions.ts:704-709 getChoiceCardId` |
| 53 | `choice_card_hp_norm` | `getCardById(choiceCardId).hp / 180` if `card.kind === "umamusume"`. **0 if card is non-uma (trainer) or choice card absent.** | `core/catalog.ts getCardById`; `Card.hp` |
| 54 | `choice_card_attack_damage_norm` | `getPrimaryAttack(getUmamusumeCard(choiceCardId)).damage / 150` if `card.kind === "umamusume"`. **0 otherwise.** | `core/catalog.ts getPrimaryAttack` |
| 55 | `choice_card_attack_cost_total_norm` | `min(sum(getPrimaryAttack(getUmamusumeCard(choiceCardId)).cost values), 4) / 4` if `card.kind === "umamusume"`. **0 otherwise.** | `core/effects.ts Attack.cost: EnergyCost` |
| 56 | `choice_card_has_ability` | 1 iff `card.kind === "umamusume" && card.ability !== undefined`. **0 otherwise.** | `Card.ability` |

**Defaults:** all 5 slots emit `0` when no choice card is resolved. For non-umamusume choice cards (rare; only fires when a search-trainer fetches a trainer — currently no such trainer exists in catalog but the guard prevents accidental NaN), slot 52 = 1 and slots 53-56 = 0.

**Why these 5 stats and not more:**
- §13 audit shows the dominant collision is "two basic umamusume share role bits but differ in HP / primary damage." HP + attack damage are the load-bearing discriminators.
- `cost_total_norm` (slot 55) discriminates 1-cost-primary basics from 2-cost basics (e.g., evolution-line entry points vs glass-cannon basics).
- `has_ability` (slot 56) discriminates ability-having basics (rare but high-impact when fetched) from plain basics.
- Adding type / weakness one-hots would add ~10 bits for marginal disambiguation; deferred to v6 if a corpus shows residual degeneracy after v5 lands.

**Locked exclusions (defer to v5.1 or v6):**
- **Choice-card type one-hot.** ~10 bits for marginal disambiguation; defer until v5 post-A1 audit shows residual degeneracy clustered on same-stat-different-type basics.
- **Choice-card weakness one-hot.** Symmetric reason; defer.
- **Combat sub-parameter disambiguators** (`useShuffleSelfIntoDeck`, `randomDiscardIndex`, `evolutionDeckCardIndex`, `combat_discardHandIndex`, `switchTargetUid` flags). §13 audit shows 0% degeneracy in the iter-19 corpus on `attack`/`retreatAttack` after v4's slot 49 expected-damage differentiation. Re-audit on a deck pool that exercises these mechanics before scoping; defer to v6 conditional on observed degeneracy.
- **`actionChoiceCardIdx` vocab embedding head.** Would give per-card identity via the card-vocab embedding (precedent: `actionSourceCardIdx` and `actionTargetCardIdx`), but requires a model.py architecture change (new embedding lookup + projection). v5 keeps the change action-vector-only; if v5 LIFTs and the residual degeneracy concentrates on same-stat-different-id basics, v6 considers the vocab head.
- **Pure `retreat` kind features.** kindIndex 8 remains dead vocabulary (`actions.ts:701`, retreat is always paired with attack). Removing the dead slot is a separate cleanup, not a v5 concern.
- **Trunk capacity bump (h128/d2 → h256/d4).** §13.3 of v3.8 contraindication still applies.

---

## 5. Ablation plan

**Recipe:** v3.8 mirror (hidden_dim=128, depth=2, KL=0.05, W6-fix ON, 8×240 selfplay, MCTS sims=100). Same recipe that produced v3.8 A1.

**Arms:**
- **A0** (baseline): v3.8 lineage best — REUSE the v3.8 A1 wl_lower at n=10k. Manifest path TBD at fire time (the v3.8 A1 promoted ckpt).
- **A1** (v5 bundle): v3.8 ckpt + 5-slot action tail. Built from `make_v5_action_init.py` applied to v3.8 source ckpt. Iter-0 parity contract verified (Δlogits ≤ 1e-5). Then run v3.8 recipe mirror at action_dim=57, state_dim unchanged at 304, 5-input ONNX.

**Sub-arms (CONDITIONAL — fire only if A1 lands in regression band wl_lower < 0.5750):**
- **A2: v5 minus choice_card stats** — A1 with slots [53:57] zero'd, only `choice_card_present` flag retained. Tests whether the stat slots cause parity-fragile drag or the present-bit alone carries signal.
- **A3: v5 minus `has_ability`** — A1 with slot 56 zero'd. Tests whether the ability bit was non-discriminating noise.

**Sample size:** n=10k tight gate for A1 (mirrors v3.7/v3.8 cadence). Smoke at n=240 first as sanity.

**Decision rules (anchored against v3.8 A1 result, TBD at fire time — write the band thresholds as `A0 + offset` rather than absolutes):**
1. **Hard ship (LIFT band):** `wl_lower(A1) ≥ wl_lower(A0) + 0.013` AND `wl_lower(A1) ≥ 0.6040` → v5 promoted as new action-schema baseline; clears v3.0 upper bound 0.6004.
2. **Soft ship (non-regression):** `wl_lower(A0) − 0.013 ≤ wl_lower(A1) < wl_lower(A0) + 0.013` → v5 ships as new baseline; v3.8 retires. The structural-expressivity fix ships because the slim add is cheap and the head is no longer degenerate, regardless of measured lift.
3. **Hard fail:** `wl_lower(A1) < wl_lower(A0) − 0.013` → fire A2-A3 to attribute. If A2 clears, slot 56 (`has_ability`) was drag → ship v5 minus slot 56. If A3 clears, the stat-block was drag and `choice_card_present` alone is the safe minimum → ship trimmed.

**Estimated wall:**
- Per-arm wall per v3.8 ground truth: ~18 min loop + ~9 min n=10k gate = **~27 min wall for A1.**
- A0 reused = 0 min.
- A2-A3 conditional only on A1 regression; +2 × ~27 min = ~55 min IF needed.
- **Total: ~30 min (no regression) or ~85 min (full attribution).**

---

## 6. Out of scope (explicit)

Items deferred past v5:

- **Combat sub-parameter disambiguators** (`useShuffleSelfIntoDeck`, `randomDiscardIndex`, `evolutionDeckCardIndex`, `combat_discardHandIndex`, `switchTargetUid`). §13 corpus shows 0% degeneracy on these in the current deck pool after v4. Deferred to v6 conditional on a re-audit against a deck pool that exercises combat sub-parameters at non-trivial rate.
- **Choice-card type one-hot + weakness one-hot.** ~20 bits for residual disambiguation. Defer to v5.1 if post-A1 corpus audit shows same-stat-different-type degeneracy.
- **`actionChoiceCardIdx` vocab embedding head.** Model architecture change (new embedding + projection). Held; v5 keeps the change to the action vector only.
- **Pure `retreat` kindIndex 8 dead-vocab cleanup.** Separate housekeeping, not v5.
- **State-axis additions** (per-bench secondary-attack ETA, deck-residual searchable-targets, hand-trainer multihot). State schema stays at v3.8.
- **Trunk capacity bump.** §13.3 of v3.8 contraindication holds.
- **Set-attention Slice 3, history features, per-side asymmetry probe.** Different axes; held.

---

## 7. Constraints honored

Per `v38-slim-feature-add-scoping.md` §7:

- **No opponent hand IDs.** v5 reads only own-side payload choices. Choice-card identity is from own-side deck/hand.
- **No raw turn-stamps.** All v5 features are per-action structural lookups.
- **No ability-name strings.** Slot 56 is a bare presence flag (`card.ability !== undefined`).
- **v3.8 state byte freeze** ([0:304]) preserved. v5 does NOT touch state vector.
- **v4 action backward read** (slots [0:52]) preserved. v5 is strictly additive at [52:57].
- **Action-schema-version guard** rebumped to accept {3, 4, 5}.
- **Init parity / zero-init residual** — `make_v5_action_init.py` ensures bit-identical iter-0 outputs vs v3.8 source (Δlogits ≤ 1e-5).
- **No process-level frozen vocabs needed** — choice-card stats are pure-function predicates over static catalog.

---

## 8. Conditional escalation paths

**If A1 lands in the LIFT band (rule 1):**
- v5 becomes a real lift over v3.8. The choice-card axis was a structural bottleneck. Promotion-ready against the production pin pending: (a) v5-on-v3.8-recipe verification at the deck-pair-sampling sweep, (b) Rust observation builder emitting action_dim=57 in self-play binary (already covered by phase A), (c) export-onnx parity smoke pass.
- Open v6 scoping for `actionChoiceCardIdx` vocab head + combat sub-parameter disambiguators bundled.

**If A1 SOFT-SHIPS (rule 2):** ship v5 because the structural-expressivity fix is cheap and the head should not be feature-degenerate even if the lift is invisible inside Wilson noise. v5 becomes the action-schema baseline.

**If A1 HARD-FAILS (rule 3):** fire A2-A3 to attribute. Most likely culprit is `has_ability` slot 56 noisy interaction with role bits 42-45 (which already encode an ability bucket via `cardRoleUtility`). If A2 or A3 clears, ship trimmed. If both regress, the v5 thesis is falsified — choice-card discrimination is NOT what's bottlenecking the head, and §13's degeneracy framing was a false positive at this trunk. In that case the v5 doc is archived and v6 considers either the vocab head (higher-bandwidth fix) or accepting MCTS visit counts as the existing tie-breaker.

**Comparison anchors at decision time:**
- v3.0: wl_lower 0.5811.
- v3.7 cap128 iter-6: wl_lower 0.5870.
- v3.8 A1: TBD (this doc finalizes only after v3.8 fires).
- v5 NON-REGRESSION band: `wl_lower(A0) − 0.013`.
- v5 LIFT band: `wl_lower(A0) + 0.013` AND `≥ 0.6040`.

---

## 9. The bet (explicit framing)

1. **The v5 thesis is structural, not capacity.** v3.8's §4n verdict was about how much *signal* the trunk can absorb at h128/d2. v5 is about whether the *head* can express a ranking the audit shows it cannot currently express. Even if the trunk is signal-saturated, the head being degenerate is independently fixable.

2. **The empirical anchor is unusually concentrated.** 99.4% of measured degeneracy is one trainer card (`3starMakeDebutScout`). The fix is correspondingly minimal: 5 slots routed through one existing helper (`getChoiceCardId`). Cheap to land, cheap to fire, cheap to attribute.

3. **The two outcomes are both informative.** LIFT confirms the head-expressivity hypothesis and opens v6 vocab-head work. FLAT/SOFT-SHIP confirms the v3.8 capacity-ceiling generalises to head-expressivity too, and v5 ships as a structural cleanup with no lift expectation. Either way the answer comes in ~30 min and is decisive about action-axis head-degeneracy at h128/d2.

4. **What this doc deliberately does NOT bet:**
   - Does NOT bet on combat sub-parameter disambiguation (§13 corpus rules that out for the current deck pool).
   - Does NOT bet on choice-card identity via vocab head (held as v6).
   - Does NOT bet on type / weakness one-hots (held as v5.1 conditional).
   - Does NOT bet on state-axis additions.
   - Does NOT bet on trunk capacity.

The implementer lands code + smoke + init-parity ckpt + Python↔Rust parity smoke. A1 fire is **USER-GATED** and blocked on v3.8 A1 promotion.

---

## 10. Canonical file pointers

### Feature code
- `frontend/src/game/engine/ai-policy/actions.ts:50-51` — `ACTION_FEATURE_SCHEMA_VERSION = 4`, `ACTION_FEATURE_COUNT = 52` (bump to 5, 57).
- `frontend/src/game/engine/ai-policy/actions.ts:496-560` — v4 action-vector builder (extend to 57-d).
- `frontend/src/game/engine/ai-policy/actions.ts:704-709` — `getChoiceCardId` (route hub for all 5 v5 slots).
- `frontend/src/game/engine/ai-policy/actions.ts:711-739` — `cardRoleKind/Progression/Output/Utility` (existing role-bit helpers feeding slots 42-45; v5 slots are complementary stat helpers, not duplicates).
- `frontend/src/game/engine/core/playTypes.ts:1-6` — `PlayChoices` type (payload source of all 3 choice-card routes).
- `frontend/src/game/engine/core/catalog.ts` — `getCardById`, `getPrimaryAttack`, `getUmamusumeCard`.
- `engine-rs/crates/engine/src/policy/actions.rs:64-71` — Rust action-schema constants.
- `engine-rs/crates/engine/src/policy/actions.rs:1104-1746` — `build_features` Rust mirror (extend to 57-d).
- `engine-rs/crates/engine/src/policy/actions.rs:1778-1796` — `play_choices_to_value` (payload serializer; confirms `deckCardIndex` plumbing in worktree).
- `engine-rs/crates/engine/src/policy/featurize.rs:72` — `ACTION_DIM = 52` (bump to 57).
- `engine-rs/crates/engine/src/core/play_types.rs:5-12` — Rust `PlayChoices` mirror.
- `engine-rs/crates/engine/src/inference/mod.rs:380-405` — action-schema guard (extend to accept 5).
- `engine-rs/crates/engine/src/inference/mod.rs:636-688` — per-schema `action_dim_for_schema` + `pack_row` slicing (extend with v5 → v4 / v5 → v3 backward shims).
- `training/uma_ai/features.py:50` — Python action-schema constants.
- `training/uma_ai/model.py:9, 75, 393` — `action_encoder` (Linear(action_dim → hidden) — `action_dim` config-driven; expander §3.3 widens the first Linear weight).
- `training/serve_onnx.py:_SCHEMA_TABLE` — add v5 entry.
- `training/export_onnx.py` — schema sidecar stamping (no code change; reads constants).
- `training/make_v38_slim_init.py` — template for `make_v5_action_init.py` (action-tail zero-init pattern; v5 expander drops the state-tail expansion).

### Research docs
- `docs/ai-research/scoping/v38-slim-feature-add-scoping.md` — direct parent; obs-contract pattern + LOCKED definitions template + ckpt-expander pattern + parity-smoke convention.
- `docs/ai-research/scoping/v33-correctness-fix-scoping.md` — v2→v3 action-schema bump precedent (slot reinterpretation, no shape change).
- `docs/ai-research/progress/r110.md §4n` — schema-axis exhaustion verdict (the framing v5 explicitly does NOT re-test, see §1).
- `docs/ai-research/README.md` — write hygiene + canonical home routing.

### State files
- `docs/ai-agent-state/queue.json` — add `v5-action-disambiguation` entry blocked on v3.8 A1 verdict.
- `docs/ai-research/progress/r110.md` — pre-allocated section `§4o` for v5 verdict (write at A1 fire time, not at scope time).

---

## 11. What this doc is NOT

- **NOT a re-test of the schema-axis exhaustion verdict.** §1 explicitly defers to v3.8's outcome. v5's claim is narrower (head expressivity) and independent of the trunk-capacity bet.
- **NOT a vocab-embedding architecture change.** Held to action-vector-only additive tail.
- **NOT a combat sub-parameter fix.** §13 corpus rules that out for the current deck pool.
- **NOT a state-axis bump.** State schema stays at v3.8.
- **NOT compute-approved yet.** Pre-condition is v3.8 A1 verdict.

---

## 12. Status line

DRAFT — USER-GATED 2026-05-26. Blocked on v3.8 A1 promotion. Implementer instruction: do not fire A1 until (a) v3.8 A1 verdict is in r110.md §4o-pre, (b) v5 init-parity smoke (Δlogits ≤ 1e-5 vs the v3.8-promoted ckpt) passes, (c) v5 Python↔Rust parity smoke Gate C passes. User-gate confirms A1 fire post-pre-condition checks.

---

## 13. Evidence audit (action-feature degeneracy corpus check)

### Corpus

Sampled `runs/R16-P3-v36-az-5k-nobuffer-cuda/iter-19/selfplay.jsonl` (newest large recent self-play shard, May 2026; 20,000 rows / 67,934 action slots). Cross-checked against `iter-0/selfplay.jsonl` — numbers within ±0.1pp.

### Headline numbers

| Metric | iter-19 | iter-0 |
|---|---|---|
| States with ≥1 degenerate group | **5.62%** (1,123 / 20k) | 5.54% |
| Degenerate groups total | 4,631 | 4,695 |
| Action slots inside a degen group | **13.63%** (9,262 / 67,934) | 13.75% |
| Selected action lands in a degen group | **3.21%** of rows | 3.29% |
| Group size distribution | 100% size-2 (no triples+) | same |

### Breakdown by kind

| Pairing | Count | % of degen groups |
|---|---|---|
| `(playTrainer, playTrainer)` in `trainerBefore`/`trainerAfter` | 4,605 | **99.4%** |
| `(attack, retreatAttack)` in `combat` | 26 | 0.6% |
| `(useAbility, *)` | **0** | — |
| Other combat sub-parameter pairings | **0** | — |

100% of `playTrainer` degeneracy in the corpus is `3starMakeDebutScout` deck-search rows. Sample IDs: `trainerBefore:trainer:5:3starMakeDebutScout:0:0:x:x` vs `trainerBefore:trainer:5:3starMakeDebutScout:0:6:x:x` — same handIndex (5), same discardHandIndex (0), only `deckCardIndex` differs (0 vs 6), and that does not propagate into the v4 48-d feature vector (the searched cardId is fed only to slots 42-45 cardRole bits, which collide across multiple basic umamusume).

### Visit spread inside degenerate groups (does MCTS break the tie?)

Visits are essentially **uniform across the degenerate pair** — search does *not* break the tie:

- Mean max-share within a degen group: **0.578** (uniform would be 0.5; peaked would be ~1.0)
- p50 max-share = 0.556, p90 = 0.700
- 0.0% of groups are peaked (max ≥ 0.95); 68.3% are diffuse (max ≤ 0.6)
- Mean normalized entropy = **0.964** (uniform = 1.0)

Gradient is smeared across feature-identical members. Policy target itself is near-uniform across them. **The model literally cannot learn to prefer one variant over the other from the current v4 feature set.**

### Why v5 covers this

`getChoiceCardId` at `actions.ts:704-709` ALREADY resolves `choices.deckCardIndex → side.deck[deckCardIndex]` to a cardId. The cardId is fed into slots 42-45 (4 role bits). For the 4,605 `3starMakeDebutScout` pairs, both rows fetch basic umamusume with **identical cardRole bits** (kind=1, progression=0, output bucket within rounding, utility=ability-presence). v5 slots 53-56 (HP / damage / cost / ability) provide per-card-instance granularity that role bits compress out. The model now has 5 additional discriminating dimensions per choice-card row.

### What v5 does NOT cover (corpus evidence)

- **Combat sub-parameter cross-products** (`useShuffleSelfIntoDeck`, `randomDiscardIndex`, `evolutionDeckCardIndex`, `combat_discardHandIndex`, `switchTargetUid`): **0 observed degenerate pairs** in this corpus on these axes. Either the v4 slot 49 (`expected_damage_norm`) already differentiates across the cross-product, OR the current R16-P3-v36 deck pool under-exercises these mechanics. Re-audit after deck-pool change; defer to v6.
- **`useAbility` sub-shape variants**: **0 observed degenerate pairs** in this corpus. Same deferral.
- **`(attack, retreatAttack)` pairs in combat**: 26 pairs, 0.6% of degeneracy. v4's slot 48 (`swap_in_attack_ready`) partially handles this — non-zero only on retreat-flavored actions — but the 26 residual pairs imply edge cases where the swap-in is energy-equivalent. Not a v5 concern; the 0.6% slice is below the cost of additional slots.

### Why one corpus is enough for this scope

- Two shards from the same recipe family (iter-0 and iter-19) replicate to ±0.1pp; the prevalence is stable.
- 99.4% concentration on one trainer card is structural, not stochastic — the deck-search trainer's enumeration pattern guarantees feature-identical rows whenever the searched deck has ≥2 candidates of the same role type.
- The choice-card route hub (`getChoiceCardId`) is decoupled from deck pool. The v5 fix lands the same regardless of deck variation; it does not depend on which trainers are present.

Confidence: **high** on prevalence numbers, on `3starMakeDebutScout` mechanism, and on the choice-card-stat fix erasing 99%+ of empirical degeneracy. **Medium** on whether residual degeneracy after v5 lands clusters on same-stat-different-id basics (would justify v6 vocab head) or fades into Wilson noise — answered by post-A1 re-audit.

---

## End

v5 is the smallest action-axis slice scoped to date and the cheapest to fire. The risk-reward profile: low cost to disambiguate whether choice-card head-expressivity is bottlenecking policy learning at h128/d2. Pre-condition: v3.8 A1 verdict must land first; v5 is blocked on that gate.
