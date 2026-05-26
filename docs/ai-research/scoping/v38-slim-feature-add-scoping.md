# v3.8 Slim Feature Add (per-bench ETA + gust-swing catastrophe + action v4 cross-bits) — Scoping

- **Date:** 2026-05-26 (rewritten 2026-05-26 — earlier draft proposed a triple-axis state+action+trunk bundle; replaced after evidence audit below).
- **Status:** **LANDED-SOFT-SHIP 2026-05-26**. A1 fired iter-1 ckpt n=10k tight-gate wl_lower=**0.5882** (wilson95 [0.5882, 0.6074]); Δ vs v3.7 anchor (0.5870) = +0.0012, within Wilson noise. LIFT band (≥0.6040) NOT crossed. SOFT-SHIP per §5 rule 2: v3.8 ships as new schema baseline; v3.7 retires. Per-side Δ = +0.062 (player 0.5534 vs opp 0.6149) — independent third corroboration of the §4m/§4n per-side asymmetry signature. Canonical verdict writeup: `docs/ai-research/progress/r110.md §4o`. Slim additive scope; deliberately small. Evidence audit (see §13) trimmed the original thick-bundle proposal by ~70% after finding most candidate features either redundant with existing v3.0–v3.7 encodings, derivable from the action card_id embedding, or contraindicated by overfitting + flat-prior evidence (trunk bump).
- **Parent:** `v37-combat-arith-and-catalog-scoping.md` (LANDED-SOFT-SHIP, state_dim=296) + `v33-feature-gap-brainstorm-handoff.md` §2.A item 5 (bench-Uma temporal), §2.C item 2 (gust-swing catastrophe), §2.C item 4 deep (per-bench ETA), §2.B item 4 (retreat-swap-in), §2.B item 1 (expected damage), §2.B item 2 (attach-typed-completion).
- **Predecessor schema:** v3.7 (296-d, no slots, action schema v3, hidden=128/depth=2). n=10k tight-gate wl_lower=0.5870 at cap128 iter-6 (`runs/R16-P1-v37-cap128-iter6-tight-gate/gate.manifest.json`).
- **Scope:** A **slim** two-axis tail. **+8 state bits** at `[296:304]` (per-bench primary-attack ETA × 3 bench × 2 sides + gust-swing catastrophe × 2 bits) + **+4 action slots** at `[48:52]` (cross-bits the action embedding genuinely cannot derive from card_id pairs). **No trunk bump** (h128/d2 unchanged — overfitting evidence in §13 contraindicates h256/d4). **No obs-contract bump** (all features derive from existing v3.6/v3.7 obs surface + static catalog lookup).

---

## TL;DR

- **STATE_DIM_V3_8 = 304** (+8 bits over v3.7). v3.7 head `[0:296]` byte-stable.
- **ACTION_FEATURE_COUNT = 52** (+4 slots over v3 = 48). **ACTION_FEATURE_SCHEMA_VERSION = 4.** First action-schema bump since the 2026-05-25 cleanup (commit `5ea1758`).
- **Trunk unchanged.** Evidence in §13 shows h128/d2 is OVERFITTING per-iter distill (train acc 0.89 / val acc 0.65, value-loss train 0.13 / val 0.57 = ~4.4× gap) — bigger trunk worsens generalization. The directly analogous cap64→cap128 transition lifted +0.0004 (flat), so cap128→cap256 has no prior to expect lift.
- **Single ablation:** A0 = v3.7 baseline (REUSE `runs/R16-P1-v37-cap128-iter6-tight-gate/gate.manifest.json` wl_lower=0.5870). A1 = v3.8 bundle. **Wall ~2 hrs** (vs the 30-40 hr triple-axis bundle the original draft scoped).
- **Bet:** the v3.7 §4n verdict ("schema axis exhausted at h128/d2") was scoped to channel CLASSES already tried. v3.8 admits two genuinely new classes (per-bench temporal lookahead + cross-bit attach/retreat decisions) that v3.x has never encoded. If A1 also FLAT, the §4n verdict generalises to per-bench temporal + action cross-bit classes too, and the schema axis closes definitively across BOTH state and action surfaces at this trunk. Off-axis pivot becomes unambiguous: per-side asymmetry diagnostic primary (highest ex-ante leverage per §4n), set-attention Slice 3 secondary, history features tertiary.

**Why slim + low-risk now:** Evidence audit (§13) eliminated 70% of the originally-proposed bundle. What remains is the set of features where the validation explicitly confirmed **no existing encoding** AND **discriminative across action candidates or temporal frames the model cannot derive from existing inputs**. The bundle is intentionally cheap to fire so the answer comes fast.

---

## 1. The v3.7 verdict and what this scope respects

`progress/r110.md §4n` — v3.7 cap128 iter-6 wl_lower=0.5870 vs v3.6 0.5880 (Δ=−0.0010). The schema-axis-exhaustion claim from §4m HARDENS across three orthogonal axes (info-density / recipe / qualitatively-new representational classes).

§4n's forward implications (verbatim, r110.md:1476-1490):
> Schema axis is closed. Future bumps require an off-axis lever first: trunk capacity (h256+/d4+), architecture (set-attention Slice 3), or representation (history/sequence features). Stacking more bits at the current trunk is empirically wasted compute.

v3.8 **explicitly does not defy this directive lightly.** The evidence audit (§13) shows the trunk-capacity-bump lever §4n names is contraindicated by overfitting evidence on the v3.7 train/val curves. The remaining levers (set-attention, history features) are larger separate scopes outside the v3.x lineage. What v3.8 commits to is a **deliberately small** schema-axis add scoped to the narrow class of features that:
1. Are not yet encoded in v3.0–v3.7 (verified via §13 audit).
2. Are not derivable from the action card_id embedding (verified per-slot in §13).
3. Have per-game frequency high enough that gradient is non-trivial (per-bench ETA: every turn; gust-swing: ~once per few turns when opp has gust in hand; action cross-bits: every action of the relevant kind).

If §4n is correct in the strong form (schema axis truly exhausted at h128/d2 across ALL feature classes), v3.8 lands FLAT and the verdict generalises. If §4n was scoped to channels v3.5/v3.6/v3.7 actually tried (info-density, qualitatively-new arith/catalog), v3.8 may lift by catching a slim slice of genuinely unencoded signal.

### v3.7 lessons preserved

- **Channel orthogonality (v3.4→v3.7 doctrine):** per-bench ETA extends v3.7 Ch.4 (active-only) into a new temporal surface; gust-swing catastrophe is a fundamentally new catastrophe-EV channel (bench-refill catastrophe at v3.5 was bench-EMPTY-only, not gust-vulnerability). Action cross-bits ([48:52]) are scoped to predicates that REQUIRE cross-referencing source-action-state with target-Uma-state, NOT decomposable into card_id embedding lookups.
- **Slot tokens stay retired (v3.4 doctrine).** UMA_SLOT_FEATURE_DIM unchanged at 23.
- **Locked vocabs at scope time** — gust-swing predicates are pure-function definitions (§4.5 below). No vocab risk.
- **No n=120 verdict-grade comparisons.** A1 arbitrated at n=10k tight-gate.

---

## 2. Pre-registered hypotheses

### H1 — Per-bench Uma primary-attack ETA surfaces actionable signal

- **Falsifiable signal:** wl_lower(A1) ≥ wl_lower(A0) + 0.013 = 0.6000 → bundle lifts above v3.7 anchor. Per-bench ETA is the load-bearing channel for "should I promote a bencher next turn?" planning that the model cannot today reason about (v3.7 Ch.4 covers active-only).
- **Theory:** v3.7 Ch.4 exposes "primary attack usable next turn" for active Umas only. After a gust or KO, a bencher becomes active — its energy ETA matters NOW because the decision to bench-attach vs active-attach is gated on future-active-attack-readiness. Today the model has no calibrated bench-ETA bit; it must learn it indirectly from energy counts × cost × bench-position embedding interaction at h128.

### H2 — Gust-swing catastrophe predicates close a specific loss class

- **Falsifiable signal:** within the A1 result, the catastrophe-bit ablation (see §5) shows per-bench-Uma KO loss rate drops by ≥2% when the gust-swing bits are present.
- **Theory:** "opp has gust in hand AND own weakest bench Uma at low HP AND KO would deplete prizes" is a specific catastrophe v33 §2.C item 2 cited as "single-feature explanation for a whole class of catastrophic losses." Today the model has hand-trainer counts (gust included) but no synthesised cross-bit linking hand-gust × bench-vulnerability × prize-state.

### H3 — Action cross-bits surface signal not derivable from card_id embedding

- **Falsifiable signal:** wl_lower(A1) lift on `attachEnergy` actions specifically (measured by win-rate-on-turns-with-attach-decisions) exceeds the lift on non-attach turns by ≥0.005.
- **Theory:** `attach_color_matches_typed_need` (slot 50) and `attach_completes_typed_threshold` (slot 51) are cross-bits requiring (attach color from `side.energyZone[0]`) × (target's primary-attack typed cost) × (target's current energies). The model SEES the inputs (state has color one-hot, hand-trainer counts, target Uma energies) but at h128/d2 may not learn this 3-way interaction. Explicit pre-computed cross-bits short-circuit.

### H4 — Action-axis is not exhausted (separate from state-axis exhaustion verdict)

- **Falsifiable signal:** wl_lower(A1) ≥ wl_lower(A0) + 0.013 with the action-axis contribution attributable via §5 sub-arm (if needed).
- **Theory:** The v3.7 §4n schema-axis exhaustion verdict was measured on the STATE vector. Action vector has been frozen at v3 since 2026-05-25; the exhaustion claim has never been tested on the action surface. v3.8's 4 action slots are the first thick-add on this surface and test "is the ceiling state-vector-specific or feature-surface-wide?"

### H5 — Parity surfaces hold

- **Risk:** per-bench ETA reads `side.bench[i].energies` typed vs `attack[0].cost` typed; gust-swing reads own bench HP/damage + opp hand gust-trainer flag + own prize count. Action slot 49 expected-damage restricted to base + activeAttackDamageBonus + weakness (per v3.7 Ch.1 precedent).
- **Falsifiable signal:** Python↔Rust parity smoke (`v38_python_rust_parity_smoke.py`, Gate A STRICT on state[296:304] + Gate B STRICT on action[48:52]) fails on any fixture → block A1 fire.

---

## 3. Freeze contract

| Surface | v3.7 (current) | v3.8 (proposed) | Compatibility |
|---|---|---|---|
| `state_features` dim | 296 | **304** | Additive tail at indices [296:304]. Slots [0:296] byte-stable (v3.7 head frozen). |
| `STATE_FEATURE_SCHEMA_VERSION` (Python latest) | 3.7 | **3.8** | Bump LATEST marker; legacy dispatch unchanged. |
| `STATE_DIM_V3_8` (new constant) | n/a | **304** | New constant; dispatch entry `_SCHEMA_BY_STATE_DIM[304]`. |
| `observation_to_features_v3_8` (new builder) | n/a | **NEW** | Layered on `observation_to_features_v3_7` then appends 8 bits. Bit-exact Rust mirror in `featurize.rs`. |
| `ACTION_FEATURE_COUNT` | 48 | **52** | Action-vector thick bump (slim); +4 slots at [48:52]. |
| `ACTION_FEATURE_SCHEMA_VERSION` | 3 | **4** | First bump since 2026-05-25 `5ea1758` cleanup. |
| `PublicSideObservation` | unchanged | **NO NEW FIELDS** | All v3.8 features derive from existing v3.6/v3.7 obs surface + static catalog. Gust-swing reads `opp.hand_card_ids` (already public when `include_opp_hand_for_catastrophe_predicates` is False — gust-trainer COUNT is in v3.7 hand-role at [71:75]); per-bench ETA reads `side.bench[i].energies` + catalog. |
| ONNX input set | 5 inputs | **5 inputs unchanged** | `state_features` last dim 296 → 304; `legal_action_features` last dim 48 → 52. |
| `uma_slot_features[10, 23]` | Not used | **Still not used** | Slot tokens stay retired (v3.4 falsified). |
| Trunk dims (`hidden_dim`, `depth`) | 128 / 2 | **128 / 2 (unchanged)** | Trunk bump explicitly excluded per §13.3 overfitting evidence. |
| `_SCHEMA_BY_STATE_DIM` dispatch | {96, 110, 164, 167, 212, 246, 296} | **{96, 110, 164, 167, 212, 246, 296, 304}** | New entry → `observation_to_features_v3_8`. Fail-fast preserved. |
| Init parity | v3.7 anchor | **v3.7 → v3.8 zero-init residual** | A v3.7 ckpt loaded into v3.8 produces bit-identical iter-0 outputs IFF (a) state tail [296:304] zero-init, (b) action tail [48:52] zero-init. NO trunk widening (h128/d2 preserved). |

### 3.1 Bit-exact mirror requirement

`observation_to_features_v3_8` lands in BOTH:
- `training/uma_ai/features.py` (Python spec).
- `engine-rs/crates/engine/src/policy/featurize.rs` (Rust mirror).

Action-vector v4 builder lands in BOTH:
- `frontend/src/game/engine/ai-policy/actions.ts` (TS canonical).
- `engine-rs/crates/engine/src/policy/featurize.rs` (Rust mirror).
- `training/uma_ai/features.py:42` (Python constant).

Parity smoke: `v38_python_rust_parity_smoke.py` mirrors v3.7's cadence:
- Gate A STRICT on state[296:304] across 10 fixtures (mid-game, early-game, terminal, gust-trigger, bench-attack-ready).
- Gate B STRICT on action[48:52] across one fixture per kind (attack, retreatAttack, attachEnergy × 3 targets, evolve, useAbility).
- Existing v3.7 [246:296] STRICT band preserved; v3 [0:48] action STRICT preserved.

### 3.2 Obs-contract surface — NO NEW FIELDS

Per §13 audit: all v3.8 features derive from existing v3.6/v3.7 obs + static catalog lookup. The original draft's deck-residual channel (which required adding `own.deck_residual_counts` to `PublicSideObservation`) is **DEFERRED to v3.9** when more state-axis items could share the same obs-contract bump.

- **Per-bench ETA** reads `side.bench[i].energies: [u8; 10]` + catalog `getPrimaryAttack(bench[i].card_id).cost`. Both already in obs.
- **Gust-swing catastrophe** reads `opp.hand_trainer_counts.gust` (already aggregated in v3.7 hand-role at state [71:75]) + `own.bench[i].hp - damage` + own prize count (already in v3.0 head). NO new obs fields.
- **Action cross-bits** all read action `kind` + `target` + `side.energyZone[0]` + catalog. All in v3 action vector + v3.6 obs.

### 3.3 Existing checkpoint compatibility

- Pre-v3.8 ckpts (v3.0–v3.7) unchanged — `_SCHEMA_BY_STATE_DIM[304]` routes only v3.8-shaped inputs.
- Loading a v3.7 ckpt into v3.8 graph requires `training/make_v38_slim_init.py`:
  1. Reads v3.7 lineage best (`runs/R16-P1-v37-cap128-iter6-tight-gate/`-source ckpt).
  2. Expands first `state_features → hidden` Linear by 8 columns at indices 296-303 (zero-init).
  3. Expands action-feature input projection by 4 columns at indices 48-51 (zero-init).
  4. Bumps `model_config.state_dim → 304`, `model_config.action_feature_count → 52`, `feature_schema.state_feature_schema_version → 3.8`, `feature_schema.action_feature_schema_version → 4`.
  5. Trunk dims UNCHANGED (`hidden_dim=128, depth=2` preserved).
  6. Saves a v3.8-shaped ckpt.
  7. Asserts bit-identical iter-0 outputs vs v3.7 source on a fixed observation batch (tolerance `1e-5`).
- Refuses non-v3.7 sources, refuses ckpts with non-default trunk dims (defense against accidentally trunk-bumping in this slice).

### 3.4 Action-schema-version guard

Action loader guard (commit `6a7dd84`) currently pins schema 3. v3.8 bumps to ACCEPT 4 when state_dim=304. Old v3.7 ckpts continue to load via the v3-action path.

---

## 4. Implementation order

**Three phases. Per `feedback_bigger_fixes_at_once`, related fixes bundle into single commits.**

1. **Phase A — Featurizer additions (Python + Rust, both axes together).**
   - Add `STATE_DIM_V3_8 = 304`, `STATE_FEATURE_SCHEMA_VERSION_V3_8 = 3.8` to `training/uma_ai/features.py`.
   - Write `observation_to_features_v3_8(...)` that calls `observation_to_features_v3_7(...)` then appends 8 bits per §4.5.
   - Bump `ACTION_FEATURE_COUNT = 52`, `ACTION_FEATURE_SCHEMA_VERSION = 4` in `training/uma_ai/features.py:42` AND `frontend/src/game/engine/ai-policy/actions.ts:31-32`.
   - Extend TS action builder at `actions.ts:487-560` with 4 new slot computations per §4.5.
   - Mirror in `engine-rs/crates/engine/src/policy/featurize.rs`:
     - `observation_state_features_v3_8` (state tail).
     - Action-feature v4 builder (4 new slots).
   - `_SCHEMA_BY_STATE_DIM[304]` dispatch entry; `serve_onnx.py` schema table; `engine-rs/crates/engine/src/inference/mod.rs` `GraphSchema::V3_8` variant + action-schema guard accepts schema 4 when state_dim=304.
   - Smoke: `v38_state_v38_smoke.py` truth-table per [296:304] bit. `v38_action_v4_smoke.py` truth-table per [48:52] slot (one fixture per kind).

2. **Phase B — Init parity + ckpt expander.**
   - Write `training/make_v38_slim_init.py` per §3.3 above. Templates: `make_v37_combat_arith_init.py` (state-tail zero-init pattern); extend with action-tail zero-init for the +4 action columns. **NO trunk widening.**
   - Smoke: `v38_slim_init_smoke.py` asserts Δlogits ≤ 1e-5 vs v3.7 source ckpt on fixed observation batch.

3. **Phase C — Cross-language parity smoke + recipe + queue entry.**
   - `v38_python_rust_parity_smoke.py` per §3.1.
   - Rust parity fixtures: `engine-rs/crates/engine/tests/v38_python_parity_fixtures.rs` mirrors v3.7's pattern; 10 fixtures spanning every v3.8 bit/slot.
   - Recipe: clone v3.6 cap128 mirror (KL=0.05, W6-fix ON, 8×240 selfplay, MCTS sims=100, hidden=128, depth=2) — same as v3.7 except init from `make_v38_slim_init.py`.
   - Queue entry id: `v38-slim-feature-add`. A0 (v3.7 anchor REUSE) + A1 (v3.8 bundle) only. NO multi-arm bundle.

### 4.5 Bit / slot definitions (LOCKED here to prevent parity drift)

**Locked at scope time. Any changes after A1 fire require a fresh scoping doc (v3.8.1).**

#### State channel — per-bench Uma primary-attack ETA + gust-swing catastrophe (slots [296:304], 8 bits)

| Slot | Field | Definition | Source |
|---|---|---|---|
| 296 | `own_bench0_primary_usable_next_turn` | Catalog-derived `catalog[own.bench[0].card_id].attacks[0].cost: EnergyCost` minus `own.bench[0].energies` typed coverage minus next-turn attach budget (=1) is fully coverable from `own.energy_pool` typed multiset. Structural feasibility per v3.7 Ch.4 pattern. **0 if bench[0] absent.** | `engine-rs/crates/engine/src/policy/types.rs` `PublicSideObservation.bench` + `core/effects.rs Attack.cost` |
| 297 | `own_bench1_primary_usable_next_turn` | Same predicate, `bench[1]`. **0 if bench[1] absent.** | same |
| 298 | `own_bench2_primary_usable_next_turn` | Same predicate, `bench[2]`. **0 if bench[2] absent.** | same |
| 299 | `opp_bench0_primary_usable_next_turn` | Symmetric for opp.bench[0] (opp side IS public per v3.7 Ch.4 precedent). | same |
| 300 | `opp_bench1_primary_usable_next_turn` | Symmetric for opp.bench[1]. | same |
| 301 | `opp_bench2_primary_usable_next_turn` | Symmetric for opp.bench[2]. | same |
| 302 | `own_lose_if_opp_gusts_weakest_bench` | 1 iff ALL of: (a) `opp.hand_trainer_counts.gust > 0` (i.e., v3.7 hand-role slot [71:75] gust-count > 0), (b) `min(own.bench[i].hp - own.bench[i].damage) ≤ catalog[opp.active.card_id].attacks[0].damage` (weakness-adjusted per v3.7 Ch.1 formula), (c) `own.prizes_remaining ≤ 1` (taking one more prize loses the game). Pure cross-bit predicate. | v3.7 hand-role + opp active attack catalog + own prize state |
| 303 | `own_can_gust_win_prize_race` | 1 iff ALL of: (a) `own.hand_trainer_counts.gust > 0`, (b) ANY `opp.bench[i]` has `hp - damage ≤ catalog[own.active.card_id].attacks[0].damage` weakness-adjusted, (c) `opp.prizes_remaining ≤ 1`. Symmetric. | same surfaces |

**Defaults:** all 8 bits = 0 when the predicate's prerequisite state is absent (e.g., bench slot empty, no gust in hand, prize state not in winning range).

**Why bench[0..2] explicit instead of slot-token:** slot tokens were retired by v3.4 doctrine. Per-bench scalar bits at fixed positions preserve channel orthogonality (each bench position is its own slot) without re-opening the slot-token channel that v3.4 falsified.

#### Action channel — v3→v4 slot definitions (slots [48:52], 4 slots)

| Slot | Field | Definition | Source |
|---|---|---|---|
| 48 | `swap_in_attack_ready` | 1 iff kind∈{retreat, retreatAttack} AND `hasEnoughEnergy(target, getPrimaryAttack(getUmamusumeCard(target)).cost)`. **0 otherwise.** | `flow/energy.ts hasEnoughEnergy`; `core/catalog.ts getPrimaryAttack` |
| 49 | `expected_damage_norm` | For kind∈{attack, retreatAttack, useAbility-damage}: `damage_after_weakness / 300` where `damage = attack.damage + activeAttackDamageBonus + (defenderCard.weakness.type == attackerCard.type ? weakness.amount : 0)`. **RESTRICTED**: NO per-attack conditional bonuses (`attackBonusIfAttachedEnergy`, `attackDamageBonusIfEvolvedLastTurn`, `coinBonus`, `attackDamageBonusIfDiscardHandCard`, `damageDoubledIfAllOpponentBenchHaveEnergy`). | `flow/combat.ts:117`; weakness arithmetic per v3.7 Ch.1 |
| 50 | `attach_color_matches_typed_need` | For kind=attachEnergy: 1 iff `side.energyZone[0]` (the color being attached) reduces a typed deficit on `target`'s primary-attack cost. 0 otherwise. | `flow/energy.ts:7`; `attack.cost` typed counts |
| 51 | `attach_completes_typed_threshold` | For kind=attachEnergy: 1 iff post-attach `target` meets primary-attack typed cost (colorless still allowed unmet). 0 otherwise. | post-attach simulated typed count vs cost |

**Action slot defaults:** all slots emit `0` when the action's kind doesn't apply.

**Locked exclusions (dropped per §13 usefulness audit, defer to v3.8.1 or later):**
- `retreat_cost_norm` (slot 48 in original draft) — already in state `effectiveRetreatCostReduction` (`features.py:367-388`); embedding learns from card_id.
- `expected_damage_kos_target` (slot 51 in original draft) — duplicates existing v3 slot 26 `lethalTarget`.
- `attach_color_idx_norm` (slot 52 in original draft) — same value across all attach actions in a turn (non-discriminative).
- Trainer effect flags (gust/search/recover/extra-attach/retreat-reduction, slots 55-59 in original draft) — each maps to literally 1 catalog card; action embedding learns trivially.
- Evolution Δ slots (60-63 in original draft) — catalog-derivable from (source, target) card_id pair embeddings; ~1-3 evolves per game = tiny gradient.
- Deck-residual searchable-targets channel — DEFER to v3.9 when obs-contract bump can amortize across more items.
- Hand-trainer effect-class multihot — already PARTIAL as normalized /10 counts at v3.0 state [71:75]; multihot adds only marginal calibration.
- Trunk capacity bump (h128/d2 → h256/d4) — contraindicated by overfitting evidence per §13.3.

---

## 5. Ablation plan

**Recipe:** v3.6 cap128 mirror (hidden_dim=128, depth=2, KL=0.05, W6-fix flags ON, 8×240 selfplay, MCTS sims=100). Same recipe that produced v3.6 cap128 cont-iter2 wl=0.5880 and v3.7 cap128 iter-6 wl=0.5870 at n=10k.

**Arms:**
- **A0** (baseline): v3.7 cap128 iter-6 — REUSE the §4n v3.7 wl=0.5870 measurement at n=10k. NO new training. Manifest at `runs/R16-P1-v37-cap128-iter6-tight-gate/gate.manifest.json` IS the A0 reference.
- **A1** (slim bundle): v3.8 retrain (v3.7 head + 8-bit state tail + 4-slot action tail). Built from `make_v38_slim_init.py` applied to v3.7 cap128 iter-6 ckpt. Iter-0 parity contract verified (Δlogits ≤ 1e-5). Then run v3.6 cap128 recipe mirror at state_dim=304, action_count=52, 5-input ONNX.

**Sub-arms (CONDITIONAL — fire only if A1 lands in regression band wl_lower < 0.5750):**
- **A2: v3.8 state-only** — A1 minus action [48:52] tail. Tests whether action-v4 hurts.
- **A3: v3.8 action-only** — A1 minus state [296:304] tail. Tests whether state extension hurts.
- **A4: v3.8 minus gust-swing** — A1 with state [302:304] zero'd. Tests whether catastrophe predicates are parity-fragile drag.

**Sample size:** n=10k tight gate for A1 (mirrors v3.7 cadence). Smoke at n=240 first as sanity.

**Decision rules:**
1. **Hard ship (LIFT band):** wl_lower(A1) ≥ 0.6040 → v3.8 promoted as new schema candidate, decisively clears v3.0 upper bound 0.6004. Production-pin discussion opens.
2. **Soft ship (non-regression):** 0.5750 ≤ wl_lower(A1) < 0.6040 → v3.8 ships as new state+action baseline; v3.7 retires. Schema axis stays CLOSED at the verdict bands; v3.8 ships because a slim add is cheap, not because it lifted.
3. **Hard fail:** wl_lower(A1) < 0.5750 → fire A2-A4 to decompose. If a single sub-arm clears 0.5750 cleanly, drop the offending sub-bundle and re-fire trimmed. If ALL sub-arms regress, the verdict "schema axis closed extends to per-bench temporal + cross-bit predicates" hardens further; next pivot is unambiguously off-axis (per-side asymmetry probe per §4n recommendation).

**Estimated wall:**
- Per-arm wall per v3.7 ground truth: ~18 min loop + ~9 min n=10k gate = **~27 min wall for A1.**
- A0 reused = 0 min.
- A2-A4 conditional only on A1 regression; +3 × ~27 min = ~80 min IF needed.
- **Total: ~30 min (no regression) or ~110 min (full attribution).**

This is the smallest v3.x slice scoped to date and the cheapest to fire. The risk-reward profile: low cost to disambiguate whether per-bench temporal + cross-bit predicates carry signal at h128/d2.

---

## 6. Out of scope (explicit)

Items deferred past v3.8 even though earlier draft considered them:

- **Trunk capacity bump (h256/d4).** Contraindicated by §13.3 overfitting evidence + cap64→cap128 flat prior. Held pending direct evidence that h128/d2 is capacity-bound on v3.7 features (no such evidence exists today).
- **Deck-residual searchable-targets.** Obs-contract bump amortizes poorly at 2-3 bits. Defer to v3.9 when bundled with hand-trainer multihot, opponent-deck-known-cards, or energy-zone forward queue depth-2/3.
- **Hand-trainer effect-class multihot.** PARTIAL as normalized counts at v3.0 [71:75]; multihot adds only marginal calibration. Defer to v3.9 if other state-axis items reopen the channel.
- **Opp `usedSupporter/Retreat/Stadium` flags.** ALREADY in v3.3 at [164:167] per `features.py:491-493, 511-514`.
- **Opp-side discard role buckets.** ALREADY in v3.5 at [207:210].
- **Phase one-hot, special-conditions per-type one-hot.** ALREADY in v3.5 at [167:177], [177:187].
- **Per-bench Uma secondary-attack ETA.** Deferred for budget; primary covers most decisions. v3.8.1 candidate if signal lifts.
- **Trainer effect magnitudes on action vector (heal/draw/gust-target/etc).** Action embedding learns from card_id 1:1; not worth bits.
- **Evolution Δ on action vector.** Catalog-derivable from (source, target) embedding pair; low frequency.
- **Attach-color one-hot on action vector.** Same color across all attach actions in a turn = non-discriminative.
- **Per-attack conditional damage bonuses on slot 49 expected-damage.** Parity surface too large.
- **Per-side asymmetry probe.** Different axis (queue id `per-side-asymmetry-probe`). Runs in parallel; v3.8 does not block or supersede. Per §4n + §13.6 this is the highest-leverage off-axis line in the queue.
- **Set-attention Slice 3, history features.** Different axes; held.

---

## 7. Constraints honored

Per `v33-feature-gap-brainstorm-handoff.md` §6 + `v35-multichannel-tail-scoping.md` §7 + `v36-priors-and-arithmetic-scoping.md` §7 + `v37-combat-arith-and-catalog-scoping.md` §7:

- **No opponent hand IDs.** Gust-swing predicate reads opp hand-trainer COUNT (`hand_trainer_counts.gust`), which is already public-aggregated in v3.7 hand-role; identity NOT exposed.
- **No raw turn-stamps.** All temporal predicates (per-bench ETA, gust-swing) derive from public state.
- **No ability-name strings.** Gust-swing predicate uses pre-existing v3.7 hand-role gust-count slot.
- **v3.7 byte freeze** (slots [0:296]) preserved. v3.8 is strictly additive at [296:304].
- **v3 action-vector backward read** (slots [0:48]) preserved. v4 is strictly additive at [48:52].
- **`_SCHEMA_BY_STATE_DIM` fail-fast dispatch** preserved — new entry at 304, no silent fallback.
- **Init parity / zero-init residual** — `make_v38_slim_init.py` ensures bit-identical iter-0 outputs vs v3.7 source (Δlogits ≤ 1e-5).
- **Action-schema-version guard** rebumped to accept 4 when state_dim=304.
- **No process-level frozen vocabs needed** — gust-swing and per-bench ETA are pure-function predicates over engine fields; no vocab risk.

---

## 8. Conditional escalation paths

**If A1 lands in the LIFT band (rule 1):**
- v3.8 becomes a real lift over v3.7. Promotion-ready against the 96-d production pin pending: (a) deck-pair-sampling Slice 3 verdict on v3.8, (b) Rust observation builder emitting state_dim=304 + action_dim=52 in self-play binary, (c) export-onnx parity smoke pass.
- Open v3.9 scoping for the now-justified next-level bundle (e.g., deck-residual + hand-trainer multihot + per-bench secondary ETA bundled under one obs-contract bump).
- The §4n "schema axis exhausted" verdict was wrong in the strong form; rev §4n to "exhausted on channel-classes-already-tried but not on per-bench-temporal + cross-bit predicates."

**If A1 SOFT-SHIPS (rule 2):** ship v3.8 because the slim add is cheap and the parity machinery is reusable. Schema-axis verdict stays as §4n stated. Next pivot is unambiguously per-side asymmetry probe.

**If A1 HARD-FAILS (rule 3):** fire A2-A4 to attribute. If a single sub-arm clears 0.5750, ship the trimmed bundle. If all regress, the §4n verdict GENERALISES to per-bench temporal + cross-bit predicates; the schema-axis program at h128/d2 closes across ALL feature surfaces and pivot becomes unambiguously off-axis.

**Comparison anchors at decision time:**
- v3.0: wl_lower 0.5811.
- v3.3 (4-iter): wl_lower 0.5910 (historical peak).
- v3.5-cap128: wl_lower 0.5912.
- v3.6 cap128 cont-iter2: wl_lower 0.5880.
- v3.6-depth=3: wl_lower 0.5910.
- v3.7 cap128 iter-6: wl_lower 0.5870 (current).
- v3.8 NON-REGRESSION band: wl_lower ≥ 0.5750.
- v3.8 LIFT band: wl_lower ≥ 0.6040.

---

## 9. The bet (explicit framing)

User-stated framing: "thick slice and to take risk."

What this scope actually commits to AFTER the §13 evidence audit:

1. **Validation cycle changed the bet.** The original draft proposed state + action + trunk bundle (~32 new bits + 4× parameter count). Evidence found:
   - Trunk bump CONTRAINDICATED (overfitting evidence + flat cap64→cap128 prior).
   - 12 of 16 action slots derivable from card_id embedding.
   - 16 of 16 deck-residual bits either redundant or requiring obs-contract bump for 2-3 useful bits.
   - 6 of 10 originally-listed state items already done in v3.3/v3.5/v3.7.

2. **The actually-defensible bet:** v3.7's §4n schema-axis verdict was measured on channel CLASSES already tried. v3.8 admits two genuinely new classes (per-bench temporal lookahead + action cross-bit predicates) that have HIGH-CONFIDENCE non-redundancy per §13 audit. Cheap to fire (~30 min), cheap to attribute (~110 min worst case).

3. **The "risk" the user asked for is preserved in axis novelty, not in compute spend.** The state-axis per-bench temporal channel and the action-axis cross-bit channel have never been tested. If they're FLAT too, the §4n verdict generalises further. If they LIFT, the verdict was scoped too narrowly. Either way the answer comes in ~30 min and is decisive about per-bench-temporal + cross-bit-action signal at h128/d2.

4. **What was preserved from the user's original thick framing:** ALL of v3.8's bits/slots are evidence-audited for non-redundancy. The original "thick = many bits" framing was replaced by "thick = every bit has high-confidence claim to useful signal." Quality over quantity.

5. **If A1 also fails LIFT, the case for further schema-axis iteration at h128/d2 closes DEFINITIVELY** across state + action surfaces. Off-axis pivot becomes unambiguous per §4n.

The implementer is to land code + smoke + init-parity ckpt + Python↔Rust parity smoke. A1 fire is **USER-GATED**. Same cadence as v3.7.

---

## 10. Canonical file pointers

### Feature code
- `training/uma_ai/features.py:149-150` — `STATE_DIM_V3_7 = 296`, `STATE_FEATURE_SCHEMA_VERSION_V3_7 = 3.7` (anchor).
- `training/uma_ai/features.py:42` — `ACTION_FEATURE_SCHEMA_VERSION = 3` (to bump to 4).
- `training/uma_ai/features.py:1268` — `observation_to_features_v3_7` (template for layered builder).
- `training/uma_ai/features.py:1416-1448` — `_SCHEMA_BY_STATE_DIM` dispatch (add entry 304).
- `training/uma_ai/features.py:983-985` — v3.7 Ch.4 active-only ETA template (per-bench extension).
- `training/uma_ai/features.py:71-75` — hand-role normalized counts (gust-count source for state slot 302/303).
- `training/uma_ai/features.py:491-493, 511-514` — v3.3 opp-flags tail (CONFIRM NOT TO DUPLICATE per §13.2).
- `engine-rs/crates/engine/src/policy/featurize.rs` — `observation_state_features_v3_7` template; add `_v3_8`.
- `engine-rs/crates/engine/src/policy/types.rs` — `PublicSideObservation.bench[i].energies` (per-bench ETA source).
- `engine-rs/crates/engine/src/core/effects.rs:133` — `Attack.cost: EnergyCost` (per-bench ETA cost source).
- `frontend/src/game/engine/ai-policy/actions.ts:31-32` — `ACTION_FEATURE_SCHEMA_VERSION = 3`, `ACTION_FEATURE_COUNT = 48` (bump to 4, 52).
- `frontend/src/game/engine/ai-policy/actions.ts:487-560` — 48-d action vector builder (extend to 52-d).
- `frontend/src/game/engine/flow/energy.ts:7` — attach-color queue head (slot 50/51).
- `frontend/src/game/engine/flow/combat.ts:117` — expected-damage formula (slot 49).
- `training/make_v37_combat_arith_init.py` — zero-init tail-append template for `make_v38_slim_init.py`.
- `training/serve_onnx.py` — schema dispatch (add 304).
- `engine-rs/crates/engine/src/inference/mod.rs` — `GraphSchema::V3_8` variant + action-schema guard.

### Research docs
- `docs/ai-research/scoping/v37-combat-arith-and-catalog-scoping.md` — direct parent; obs-contract pattern + LOCKED definitions template.
- `docs/ai-research/scoping/v33-feature-gap-brainstorm-handoff.md` §2 — gap inventory.
- `docs/ai-research/progress/r110.md §4n` — v3.7 verdict + schema-axis-exhaustion HARDEN + off-axis pivot mandate.
- `docs/ai-research/progress/r110.md §4m` — recipe-axis exhaustion + per-side asymmetry surfaced.

### State files
- `docs/ai-agent-state/queue.json` — A0–A1 queue entry. Queue id: `v38-slim-feature-add`.
- `docs/ai-research-backlog.md` — top-of-frontier update when v3.8 ships.

---

## 11. What this doc is NOT

- Not a verdict on any v3.8 feature. Pre-registered hypotheses with falsifiable bands.
- Not authorization to land code. Scope-only DRAFT; **USER-GATED**.
- Not a contradiction of §4n. §4n's strong-form claim (schema axis exhausted at h128/d2 across all classes) remains testable; v3.8 admits two narrow class extensions to test it.
- Not a substitute for per-side asymmetry probe. Per §4n + §13.6 that probe remains the highest-leverage off-axis line; v3.8 explicitly does not preempt it.
- Not a trunk-bump scope. The original draft included h256/d4; §13.3 evidence audit REMOVED it.

---

## 12. Status line

**`LANDED-SOFT-SHIP 2026-05-26`** — Slim two-axis additive scope after §13 evidence audit reduced original triple-axis draft by ~70%. STATE_DIM_V3_8=304 (+8 bits per-bench ETA + gust-swing). ACTION_FEATURE_COUNT=52, schema v4 (+4 cross-bit slots). Trunk UNCHANGED h128/d2. Single-arm fire (A0 reused v3.7 wl=0.5870, A1 bundle); ~10 min total wall (loop halted at iter-3 by orchestrator safeguard). A1 n=10k verdict: wl_lower=0.5882, SOFT-SHIP. Schema-axis exhaustion HARDENS three times now per `progress/r110.md §4o`. Canonical destination: this doc. Queue entry id: `v38-slim-feature-add`.

---

## 13. Evidence audit (pre-implementation, replaces original triple-axis framing)

Three validation investigators ran before any code landed: (a) action-vector v3→v4 slot usefulness audit, (b) deck-residual channel usefulness audit, (c) trunk capacity bet plausibility audit + (d) final redundancy verification. **The user explicitly requested high-confidence validation that proposed features would actually be useful for the model.** Findings forced the §1 scope rewrite:

### 13.1 Action-vector v3→v4 — 12 of 16 slots dropped

Per-slot verdicts (usefulness 1-5):

- **KEEP (4 slots = §4.5 above):** `swap_in_attack_ready` (4/5 — discriminative across retreat candidates), `expected_damage_norm` (4/5 — per-target weakness-multiplied), `attach_color_matches_typed_need` (4/5 — true cross-bit), `attach_completes_typed_threshold` (4/5 — threshold crossing predicate).
- **DROP (10 slots):** `retreat_cost_norm` (state already has `effectiveRetreatCostReduction` at `features.py:367-388`), `expected_damage_kos_target` (duplicates existing v3 slot 26 `lethalTarget` per `combatPlanner.ts:232`), `attach_color_idx_norm` (state already one-hot encodes own `energyZone[0]` at `features.py:582`; per-candidate identical → non-discriminative), `trainer_gust_flag` (1 catalog card has gustOpponent: yayoiAkikawa at `cards.json:1589`; card_id embedding learns 1:1), `trainer_search_class_idx_norm` (3 search effects map to 3 distinct cards 1:1), `trainer_recover_special_conditions_flag` (1 card: takoyakiBox at `cards.json:1491`), `trainer_extra_energy_attach_norm` (2 cards), `trainer_retreat_reduction_norm` (2 cards), `evolve_adds_ability_bit` (catalog-derivable from card_id pair), `evolve_adds_secondary_attack_bit` (same).
- **DEFER (2 slots):** `evolve_delta_hp_norm`, `evolve_delta_primary_damage_norm` — low frequency, catalog-derivable; revisit in v3.8.1 if evolve-action under-use surfaces.

Net new: **4 action slots (not 16).** Trimmed action vector → 48 + 4 = **52 slots** (vs original 64).

### 13.2 Deck-residual searchable-targets — DEFER to v3.9

Catalog reality:
- Only **3 deck-search trainers** in catalog (`engine-rs/crates/engine/src/core/effects.rs:333-337`); all target Umamusume.
- Player default deck = matikanetannhauser self-play (`engine-rs/crates/engine/src/deck_sampling.rs:228, 287`) with 2 basic Umas + 4 evolution Umas in 20 cards (`shared/src/data/premadeDecks.json:12-64`).
- Catastrophe condition ("search-X-in-hand AND deck has 0 valid targets") fires ~once every 2-3 games and almost always late.
- Bucketed counts add nothing — no catalog effect has a count threshold; binary "has-any" already covers the decision boundary.
- Deck-size scalar duplicates existing `deckCount/50` at `features.py:193`.

Required obs-contract bump for 2-3 actually-useful bits amortizes poorly. **Defer to v3.9** when bundled with hand-trainer multihot, opponent-deck-known-cards, or energy-zone forward queue depth-2/3 (8-15 bit batch).

### 13.3 Trunk capacity bump (h128/d2 → h256/d4) — REMOVED

LOW confidence the bump produces measurable lift. Evidence against:

- **Overfitting signature on v3.7 train/val curves.** From `runs/R16-P1-v37-cap128-A1/loop/iter-{0,2,4,6}/manifest.json`:
  | iter | train_loss | val_loss | train_acc | val_acc | val/train value-loss |
  |---|---|---|---|---|---|
  | 0 | 0.84 | 1.88 | 0.88 | 0.66 | 4.4× |
  | 2 | 0.82 | 1.73 | 0.89 | 0.68 | 4.0× |
  | 4 | 0.90 | 1.33 | 0.88 | 0.64 | 3.5× |
  | 6 | 0.88 | 1.47 | 0.89 | 0.65 | 4.1× |
  
  Train acc 0.89 / val acc 0.65 = **+24-point gap**. 4× parameter count makes overfitting WORSE on the same ~2.3k-row per-iter distill set.

- **cap64→cap128 transition was flat (+0.0004).** `r110.md:1146-1147` — v3.5-extended-cap64 vs v3.5-cap128 = 0.5908 vs 0.5912. The directly analogous prior doubling returned flat; no prior to expect cap128→cap256 to behave differently.

- **`project_model_size_transition_h256_d4` memory note is throughput-only.** It re-baselines B5 CUDA, B6 wave, value-only payoff. Zero performance-lift evidence; cannot justify a feature-axis scoping bet.

- **Per-side asymmetry (player 0.55 vs opp 0.62, σ=0.0018 across 9 runs, 35× cross-recipe noise) is the structural signature of a featurizer perspective-swap bug**, not a capacity ceiling. §4m ranks it as top-1 root cause for the off-axis probe. Bigger trunk doesn't fix featurizer bugs.

- **Wall projection corrected.** Per `runs/R16-P1-v37-cap128-A1/loop/events.jsonl`: actual v3.7 wall = **1075s (~18 min) for 8 iters + 537s (~9 min) tight-gate**. Original draft's "30-40 hrs for 5 arms" was off ~10×; bundle would have been ~3-4 hrs but on the wrong hypothesis. The right cheap test is **A0 alone, ~45 min, single-arm probe of trunk-only lift** — not a 5-arm pre-registered bundle.

**Verdict:** trunk bump REMOVED from v3.8. If user wants to fire the cheap A0 probe (~45 min) as a separate slice, that becomes its own scoping doc (queue id suggestion: `trunk-capacity-cheap-probe-h256d4`).

### 13.4 Redundancy verification (final pass)

- **Per-bench Uma primary-attack ETA** — OPEN. v3.7 Ch.4 ETA is ACTIVE-only (`features.py:983-985`). Per-bench is genuinely new.
- **Opp `usedSupporter/Retreat/Stadium`** — **ALREADY in v3.3** at state [164:167] (`features.py:491-493, 511-514`). Original draft would have duplicated; DROPPED.
- **Gust-swing catastrophe** — OPEN. No `gust*catastrophe`/`gust*bench`/`lose_if_gust*` hits in `features.py`. Synthesised predicate genuinely absent.
- **Hand-trainer effect-class multihot** — **PARTIAL.** Already encoded as normalized /10 counts at v3.0 state [71:75] (`features.py:2002-2036`); multihot adds only marginal calibration. DROPPED from v3.8; held for v3.9.
- **Action cross-bits (49/50/53/54 in original-draft numbering)** — OPEN. None present in current 48-d (`actions.ts:496-561`).

#### 13.4.1 Implementation-time adaptation: opp gust availability proxy (LANDED 2026-05-26)

The original slot [302] definition `own_lose_if_opp_gusts_weakest_bench` requires "opp has gust in hand." Implementation discovered that **`opp.handCardIds` is private** per the obs-contract:

- `engine-rs/crates/engine/src/policy/types.rs:145-147` — `pub hand_card_ids: Option<Vec<String>>` annotated "Present only when the side is `own` (model side)."
- `training/uma_ai/features.py:1980` — `_card_awareness_features` reads `own.handCardIds` only; opp side does not expose this field.
- The scoping doc §3.2 claim that "gust-trainer COUNT is in v3.7 hand-role at [71:75]" is correct ONLY for own side — `_hand_role_features` (`features.py:2002-2036`) receives `own.handCardIds` exclusively. Opp side gust counts are NOT exposed via this path.

**Adapted predicate (LANDED in commit `<phase-A+B>`):**

Per the implementer brief's "must use only PUBLIC info" instruction, the opp-side gust-availability predicate substitutes a public-info proxy. Component (a) "opp could play a gust this opp turn" becomes:

```
opp_gust_playable_proxy = (
    NOT opp.usedSupporterThisTurn      # supporter slot still open
    AND opp.handCount > 0               # opp has something in hand
    AND ANY gust-trainer in opp.discard # opp has demonstrated gust play
)
```

Where a "gust trainer" is any catalog card with `kind == "trainer"` and `effect.gustOpponent == True` or `effect.discardRandomOpponentActiveEnergy == True` (matches the v3.7 hand-role gust-bit semantics at `_hand_role_features` line `features.py:2032`).

Components (b) "weakest own bench KO-able by opp.active primary, weakness-adjusted" and (c) "own.points >= 2 (remaining <= 1 prize)" are unchanged from the original definition.

**Trade-offs:**

- **STRICTLY WEAKER** than "opp has gust in hand right now" — opp may have played a gust earlier without holding another → proxy fires false-positive. But: the proxy still picks up the catastrophe pattern v33 §2.C item 2 cites ("opp's known to gust + I'm vulnerable + I'm at 1 prize"), and the false-positive direction is the SAFE one for a catastrophe predicate (over-cautious is better than missed-warning).
- Component (b') for slot [303] `own_can_gust_win_prize_race` does NOT need adaptation — own.handCardIds IS public on own perspective, so the direct catalog lookup over own.handCardIds for gust trainers is correct (mirrors `_hand_role_features` exactly).
- **Asymmetry note:** the two predicates are now informationally asymmetric (own side reads hand IDs directly; opp side reads a public proxy). This is a known limitation. Tightening would require an obs-contract bump (private → public aggregate of opp hand-role bits) deferred to v3.9 if the v3.8 signal warrants it.

**Verification:**

- Python `_v38_opp_gust_playable_proxy` at `training/uma_ai/features.py` reads opp.discard + opp.usedSupporterThisTurn + opp.handCount only.
- Rust `v38_opp_gust_playable_proxy` at `engine-rs/crates/engine/src/policy/featurize.rs` mirrors exactly.
- Parity smoke: 10/10 fixtures bit-identical on slots [302] and [303] including the `09_opp_supporter_used` negative case (supporter used → proxy fails → bit clear).
- Smoke fixture `v38_state_v38_smoke.py` exercises both positive (gust in discard + supporter open) and negative (supporter used) cases.

### 13.5 Why the slim scope is the right answer

The user asked for a thick risky bundle. Evidence shows:
- The schema-axis exhaustion verdict at h128/d2 is robust across MULTIPLE channel-class rotations (info-density v3.6 / qualitatively-new v3.7).
- The available state-axis additions that are GENUINELY unencoded AND independent of the action card_id embedding AND don't require obs-contract bumps are few: per-bench ETA, gust-swing catastrophe.
- The action-axis additions that are GENUINELY unencoded AND not derivable from card_id 1:1 are few: 4 cross-bits.
- The trunk lever §4n names is contraindicated.

The slim scope is therefore not "lacking ambition" — it's "exhausting the genuinely-untested feature space at this trunk in a single cheap fire." The verdict comes fast (~30 min) and is decisive. If A1 LIFTS, the §4n verdict was wrong; if A1 FLAT, §4n hardens across per-bench temporal + cross-bit predicates and the next pivot is unambiguously per-side asymmetry probe.

### 13.6 The per-side asymmetry probe is the right next bet REGARDLESS of v3.8 outcome

§4m + §4n + investigator-c independently agree: the **player=0.55 / opp=0.62** structural gap is the only signal above noise floor, reproducible σ=0.0018 across 9 hidden=128 runs (35× cross-recipe noise). This pattern matches a featurizer perspective-swap bug — not a capacity, schema, or recipe issue. Diagnostic: `featurize(state)` vs `featurize(swap(state))` parity check on 100+ fixtures. Expected payoff: +0.02-0.04 wl_lower from collapsing the gap. Cost: hours, not days.

v3.8 explicitly does NOT preempt this. If v3.8 flat, per-side asymmetry probe IS the next slice. If v3.8 lifts, the gap may still be present and the probe still lifts on top.
