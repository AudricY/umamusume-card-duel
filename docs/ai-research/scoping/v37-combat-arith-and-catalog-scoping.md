# v3.7 Combat-Arithmetic-Depth + Catalog-Lookup Thick Bundle — Scoping

- **Date:** 2026-05-26
- **Status:** **APPROVED-IMPLEMENTING-CORRECTED 2026-05-26** — user approved code + compute runs in same turn. Pre-impl recon corrected three engine-API mismatches: (a) weakness is ADDITIVE not multiplicative and fields are catalog-derived; (b) no `AttackEffect` enum exists — `Attack` is flat-struct of `Option<...>` fields, so coin-flip / per-energy / per-bench detection re-anchors to `coin_bonus`/`damage_per_attached_energy`/`damage_per_umamusume_in_play`; (c) `paralysis_recovery_pending` is already a typed bool on `PublicUmaTurnState`. Catalog reality tightened vocabs: 4-variant ToolEffectKind, 8-variant AbilityEffectKind. NET RESULT: **no new obs-contract fields required**; all 7 channels derive from existing v3.6 obs + static catalog lookup. State_dim stays 296 (50 bits, reserved [272:274] dropped to repack). See §13 RECON-CORRECTIONS for details.
- **Parent:** `v36-priors-and-arithmetic-scoping.md` (LANDED-SOFT-SHIP, state_dim=246) + `v33-feature-gap-brainstorm-handoff.md` §2.B/§2.C (action-axis cleanup + synthesised-feature gap inventory) + `progress/r110.md §4l`/`§4m` (v3.6 SOFT-SHIP, recipe-axis exhausted, per-side asymmetry surfaced as off-axis probe).
- **Predecessor schema:** v3.6 (246-d, no slots, v3-action). n=10k tight-gate wl_lower=0.5880 at cap128 cont-iter2. Accepted as new schema baseline; v3.5-cap128 retired as prior candidate-of-record.
- **Scope:** A deliberately **thick** additive tail on top of v3.6 that **admits the parity-fragile arithmetic and catalog-lookup channels** that v3.3/v3.5/v3.6 explicitly locked out. Seven channels, ~50 bits net. The user has asked for risk-taking; this scoping doc inventories the risks taken and the falsifiable bands that catch them.

---

## TL;DR

- **Thick additive tail** appended to v3.6's 246-d state vector → new **STATE_DIM 296**.
- **Seven channels, 50 bits, deliberately admitting previously-LOCKED-OUT signal classes:**
  1. **Weakness-adjusted lethal & secondary-KO (4 bits)** — extends v3.6's face-value lethal with the weakness multiplier the simulator already applies (`flow/combat.rs:303-305`). v3.6's lethal-face-value LOCKED OUT this multiplier on parity grounds.
  2. **Coin-flip-expected damage indicators (8 bits)** — for both attacks of both actives: "has coin-flip component", "expected damage ≥ defender remaining HP". v3.6 §4.5 EXPLICITLY DEFERRED.
  3. **Conditional-damage-bonus type indicators (8 bits)** — "primary has per-attached-energy bonus", "primary has per-bench bonus", "secondary has per-attached-energy bonus", "secondary has per-bench bonus", both sides. v3.6 §4.5 EXPLICITLY DEFERRED.
  4. **Energy ETA + paralysis exploit window (8 bits)** — "primary attack usable next turn (incl. energy_pool draw + own attach budget)", "secondary attack usable next turn", "own paralysis_recovery_pending", "opp paralysis_recovery_pending", both sides. v3.6 §4.5 EXPLICITLY DEFERRED ("multi-step lookahead").
  5. **Tool effect-kind one-hot (8 bits)** — 4 frozen abstract effect classes × 2 active slots (vocab tightened per §13 recon: catalog has only 3 tools today; proposed 6-variant vocab over-allocated). v3.6 §6 EXPLICITLY OUT OF SCOPE ("catalog-fragile against future additions").
  6. **Active ability effect-kind one-hot (16 bits)** — 8 frozen abstract effect classes × 2 active slots (vocab tightened per §13 recon: catalog has 17 abilities across 8 distinct effect classes including HP-bonus, retreat-modifier, conditional-attack-bonus that the original 5-variant vocab would have mis-bucketed). v3.6 §6 EXPLICITLY OUT OF SCOPE.
  7. ~~Bench-aggregate restoration~~ — CUT at §4 impl reconciliation (kept out of scope; restorable in v3.7.1 at [296:306] if signal lifts).

- **Net change:** +50 bits (4 + 8 + 8 + 6 + 8 + 16 — repacked from original 4+8+8+8+12+10 after vocab tightening and dropping reserved-for-v3.7.1 bits). v3.6 keeps slots `[0:246]` byte-stable. New tail at `[246:296]`.
- **Slot tokens** stay out (v3.4 falsified, doctrine unchanged).
- **Action vector** untouched (separate axis, separate scoping doc).
- **Zero-init residual** warm-start: v3.6 ckpt → v3.7 graph yields bit-identical iter-0 outputs via `make_v37_combat_arith_init.py` (50 zero-init columns appended at `[246:296]`).

**Why thick + risky now:** §4m exhausted recipe-axis at hidden=128. §4l falsified the "info-density-per-bit" rotation. The remaining open lever on the schema axis is **qualitatively new representational classes** — channels that v3.x has explicitly never tried because of parity-fragility concerns. v3.7 is the deliberate "buy the risk" bet: admit the LOCKED-OUT channels as a thick bundle. If v3.7 also fails the LIFT band, the schema axis closes definitively — not just at one bit-density rotation but across the full feature-class space, and the next pivot becomes off-axis (per-side asymmetry diagnostic, architecture).

The user explicitly endorses thick + risk-tolerant here. This scoping doc is the contract that says "we tried the risky bundle responsibly" — channel-orthogonality preserved, parity surface bounded by the v3.6 fixture template, vocabularies frozen at scoping time, falsifiable bands pre-registered.

---

## 1. The v3.6 lessons and how this bundle leans into vs against them

`progress/r110.md §4l` — v3.6 SOFT-SHIP at wl=0.5880 vs v3.5-cap128 0.5912 (Δ=−0.0032, within Wilson noise). The "info-density-per-bit" bet was FLAT: trimming 10 dead bits and substituting prior-rich bits on the same energy-color channel did not crack ~0.59.

`progress/r110.md §4m` — recipe-axis sweep across depth 2→3, sims 100/400/1600, HP grid, cold-start, 100-iter — 9 hidden=128 runs ALL land in [0.5731, 0.5912] at n=10k. σ across recipes = 0.0057 ≈ Wilson noise floor. **Recipe + schema axis combined are exhausted at this trunk capacity.**

The §4m forward-implications conclusion was: "the bet for v3.7 cannot be 'more bits.'" v3.7 **DELIBERATELY ARGUES BACK** against that conclusion, with this distinction:

- v3.5 (45 channel-orthogonal bits, mixed dead/sparse) — FLAT.
- v3.6 (info-density rotation on the same energy-color channel) — FLAT.
- **v3.7 (qualitatively new representational classes that v3.x has NEVER ENCODED)** — UNTESTED.

The §4l/§4m verdict was that schema-axis is exhausted **for the channel classes we've tried**. v3.7 tests **classes we've never tried**: weakness-arithmetic, probabilistic-attack-expected-damage, conditional-damage-bonus-presence, multi-turn energy-readiness, tool/ability effect-kind enums.

The risk is real: if these classes also fail to lift, the verdict "schema-axis exhausted" generalises beyond the bit-density framing — and we'll know that more deeply than any prior bundle has shown.

### v3.4 lesson preserved

v3.4 falsified that *overlapping* channel stacking hurts (slot-tokens shadowed opp-flags). v3.5 and v3.6 enshrined channel-orthogonality. v3.7 PRESERVES this — §1 channel table below confirms each new channel touches a representation axis v3.6 cannot express.

### v3.5 lesson preserved

§4j dead-bit Diagnostic B. v3.7 does NOT trim further — no surfaced dead bits since v3.6's drop. If v3.6 §4l's flat result generalises further at v3.7 fire time (e.g., a freshly diagnosed dead-bit set), trim happens in v3.7.1 (separate slice).

### v3.6 lesson partially DISREGARDED

The §4m forward direction was per-side asymmetry diagnosis (queue id `per-side-asymmetry-probe`). v3.7 does NOT block that line — they run in parallel. v3.7 is the schema-axis-doubled-down bet; the asymmetry probe is the off-axis line. Both can land independently. If the asymmetry probe reveals a featurizer perspective-swap bug AND that bug uses v3.6's contract, v3.7's tail inherits the same bug; the smoke-and-fixture chain catches it.

---

## 2. Pre-registered hypotheses

### H1 — Qualitatively new feature classes lift v3.6 by ≥ Wilson noise

- **Falsifiable signal:** wl_lower(v3.7) ≥ wl_lower(v3.6) − 0.013 = 0.5750 (non-regression band on v3.6 anchor 0.5880). Hard fail below this.
- **Stretch (LIFT band):** wl_lower(v3.7) ≥ 0.6040 (clears v3.0 upper bound 0.6004 decisively → production-promotion candidate; same LIFT band v3.5/v3.6 missed).
- **Theory:** v3.5/v3.6 channels were all *information already present in obs that the model could in principle derive from raw inputs* (own energy color totals, opp discard role counts, prize numerals, face-value attack damage). v3.7 channels are *computations the model cannot derive at v3.6 capacity*: weakness multiplier requires per-Uma weakness lookup the head doesn't carry; coin-flip expected damage requires probabilistic interpretation of attack-effect strings; tool/ability effect-kind requires catalog lookup the head also doesn't carry. The bet: making these computations directly available will surface signal that hidden=128/depth=2 trunks have been *capacity-bound on the derivation*, not capacity-bound on the representation.

### H2 — Combat-arithmetic-depth (channels C1-C4) lifts independently of catalog channels (C5-C6)

- **Theory:** combat arithmetic and effect-kind lookup are pre-registered as **orthogonal sub-bundles**. A1 (full v3.7) is the bundle test. If A1 regresses while a sub-bundle lifts, we can ship the lifting sub-bundle as v3.7.0.
- **Falsifiable signal:** A2 (v3.7 minus C1-C4) vs A3 (v3.7 minus C5-C6) wl_lower differ by ≥ 0.013, indicating one sub-bundle carries while the other drags.

### H3 — Catalog vocabularies are stable enough across the training corpus to not parity-drift

- **Risk:** v3.7 freezes effect-kind enums (§4.5 below). If the engine adds a new card with an effect-class not in the v3.7 vocab between scope time and A1 fire time, the featurizer maps it to "generic" (bit 0) and the model under-weights it. A bigger risk: Python's vocab and Rust's vocab disagree on which abstract class a given concrete card belongs to.
- **Falsifiable signal:** Python↔Rust parity smoke (`v37_python_rust_parity_smoke.py`, mirroring `v36_python_rust_parity_smoke.py`) fails on any random seed → block A1 fire; rebuild vocabs; defer the offending channel to v3.8.
- **Recovery rule:** if A1 lands in the regression band AND A4 (v3.7 minus all catalog channels = v3.7-minus-C5-minus-C6) clears the soft-ship band, the catalog admission was premature; v3.7 ships in arithmetic-only form and catalog lookups stay deferred.

### H4 — Parity surface for combat arithmetic is small enough to land bit-exact

- **Risk:** weakness multiplier requires Python and Rust both to read `defender.weakness.type` and `attacker.card.type` and apply `damage += weakness.amount` (the simulator's `flow/combat.rs:303-305` formula). Coin-flip expected damage requires parsing attack-effect strings the same way in both runtimes. Conditional-damage-bonus requires per-effect-class scanning the same way in both runtimes.
- **Falsifiable signal:** parity-smoke Gate A STRICT (bit-identical on all 50 v3.7-contract bits) fails on any of 10 fixtures → block A1 fire; rebuild the offending bit's definition; defer to v3.8.

---

## 3. Freeze contract

| Surface | v3.6 (current) | v3.7 (proposed) | Compatibility |
|---|---|---|---|
| `state_features` dim | 246 | **296** | Additive tail at indices [246:296]. Slots [0:246] byte-stable (v3.6 head frozen). |
| `STATE_FEATURE_SCHEMA_VERSION` (Python latest) | 3.6 | **3.7** | Bump LATEST marker; legacy 3.0/3.1/3.3/3.5/3.6 dispatch unchanged. |
| `STATE_DIM_V3_7` (new constant) | n/a | **296** | New constant; dispatch entry `_SCHEMA_BY_STATE_DIM[296]`. |
| `observation_to_features_v3_7` (new builder) | n/a | **NEW** | Layered on top of `observation_to_features_v3_6` then appends 50 bits. Bit-exact Rust mirror in `featurize.rs`. |
| `PublicObservation` / `PublicSideView` | v3.6 has `energy_pool` | **NO NEW FIELDS** | Per §13 recon: all v3.7 channels derive from existing v3.6 obs + static catalog lookup (which Python and Rust both load from `shared/src/data/cards.json`). v3.6 builders silently ignore the catalog-derived computations. |
| ONNX input set | 5 inputs | **5 inputs unchanged** | v3.7 keeps the v3.3/v3.5/v3.6 5-input contract. Only `state_features` last dim grows 246 → 296. |
| `uma_slot_features[10, 23]` | Not used | **Still not used** | Slot tokens stay retired (v3.4 falsified). |
| `action_features[A, 48]` | 48-d, schema 3 | **48-d, schema 3 unchanged** | Action vector untouched. v3.7 does NOT bump action schema. |
| `_SCHEMA_BY_STATE_DIM` dispatch | {96, 110, 164, 167, 212, 246} | **{96, 110, 164, 167, 212, 246, 296}** | New entry → `observation_to_features_v3_7`. Fail-fast preserved. |
| Init parity | v3.6 anchor | **v3.6 → v3.7 zero-init residual** | A v3.6 ckpt loaded into the v3.7 graph (new state_features Linear columns 246-295 zero-init) produces bit-identical iter-0 outputs to v3.6. |

### 3.1 Bit-exact mirror requirement

`observation_to_features_v3_7` lands in BOTH:
- `training/uma_ai/features.py` (Python spec).
- `engine-rs/crates/engine/src/policy/featurize.rs` (Rust mirror).

Parity smoke: extend the chain (`v3_6_head_is_byte_identical_to_v3_5_excluding_opp_energy_zone_band`) with `v3_7_head_is_byte_identical_to_v3_6`. The first 246 bytes of v3.7 output must equal v3.6's output on the same observation; bytes [246:296] hold the new tail.

### 3.2 Existing checkpoint compatibility

- Pre-v3.7 ckpts (v3.0/v3.1/v3.3/v3.5/v3.6) unchanged — `_SCHEMA_BY_STATE_DIM[296]` routes only v3.7-shaped inputs.
- Loading a v3.6 ckpt into the v3.7 graph requires `training/make_v37_combat_arith_init.py`:
  1. Reads v3.6 lineage best (`runs/R16-P1-v36-cap128-A1-continue/loop/iter-2/checkpoint.pt` or equivalent — the cont-iter2 ckpt that the §4l verdict was measured against).
  2. Expands the first Linear `state_features → hidden` by 50 columns at indices 246-295, zero-init.
  3. Bumps `model_config.state_dim → 296` and `feature_schema.state_feature_schema_version → 3.7`.
  4. Saves a v3.7-shaped ckpt.
  5. Asserts bit-identical iter-0 outputs vs v3.6 source on a fixed observation batch (tolerance `1e-5` — the strict v3.5-style band, because zero-init append has no column-drop unlike v3.6).
- Refuses non-v3.6 sources (state_dim ≠ 246 → error), refuses slot-token ckpts (uma_slot_features columns non-zero → error), refuses action_schema_version ≠ 3.
- Mirrors `make_v36_priors_init.py` (template) but drops the column-drop logic absent the v3.6-style dead-band repurpose.

### 3.3 Action-schema-version guard

Unchanged. Action schema stays at 3. Rust loader guard (commit `6a7dd84`) fires consistently across v3.6 → v3.7.

### 3.4 Obs-contract surface (CORRECTED PER §13 RECON — NO NEW FIELDS)

Pre-impl recon confirmed: **all 7 v3.7 channels derive from existing v3.6 observation + static catalog lookup**. No new fields on `PublicUmaObservation` or `PublicSideView` are required. Channels read from:

1. **Weakness arithmetic (Channel 1)** — `card_id` (already on `PublicUmaObservation`) → catalog → `Card::Umamusume(u).weakness.{r#type, amount}` (engine-rs `core/effects.rs:382-386`). Python featurizer mirrors via catalog-derived helper. **Weakness is ADDITIVE**: `if defender.weakness.r#type == attacker.r#type { damage += defender.weakness.amount }` (matches `flow/combat.rs:303-305` exactly). The scope-doc draft incorrectly called it "the weakness multiplier" — corrected throughout §4.5 Ch.1.
2. **Coin-flip / per-energy / per-bench detection (Channels 2 + 3)** — `card_id` → catalog → `Card::Umamusume(u).attacks[i]` (`Attack` struct in `core/effects.rs:133-196`). The `Attack` struct is a flat collection of `Option<...>` fields, NOT a discriminated enum (the scope-doc draft incorrectly referenced a non-existent `Attack.effects: Vec<AttackEffect>`). Flat-field detection per §4.5 Ch.2/3.
3. **Energy ETA + paralysis-window (Channel 4)** — energy ETA: `own.energy_pool` (v3.6 field) + `attacks[i].cost: EnergyCost` (catalog-derived). Paralysis-window: `turn_state.paralysis_recovery_pending: bool` is **already a typed boolean** on `PublicUmaTurnState` (`policy/types.rs:118`, computed at `policy/observation.rs:240-242`). The scope-doc draft incorrectly placed it in `special_conditions: Vec<String>`; corrected at §4.5 Ch.4.
4. **Tool effect-kind (Channel 5)** — `tool_card_id: Option<String>` (already on `PublicUmaObservation`) → catalog lookup of the matching `TrainerEffect` → classifier returns `ToolEffectKind` (4-variant frozen enum per §4.5 Ch.5). Tool effects encoded as flat `Option<T>` fields under `TrainerEffect` struct (`core/effects.rs:323`); classifier is a derived dominant-class selector cached at catalog-load time.
5. **Ability effect-kind (Channel 6)** — `card_id` + `used_ability_this_turn: bool` (both already on `PublicUmaObservation`) → catalog lookup of `Ability` (`core/effects.rs:243`) → classifier returns `AbilityEffectKind` (8-variant frozen enum per §4.5 Ch.6). Same flat `Option<T>` structure; classifier cached.

### 3.4.1 What v3.7 DOES add (Rust + Python)

- **`engine-rs/crates/engine/src/core/effect_kinds.rs`** (NEW) — `ToolEffectKind` (4 variants) and `AbilityEffectKind` (8 variants) enums + `classify_tool_effect(&TrainerEffect) -> ToolEffectKind` + `classify_active_ability(&Ability) -> AbilityEffectKind` pure functions. Variant lists locked at §4.5.
- **`training/uma_ai/effect_kinds.py`** (NEW) — Python `IntEnum` mirrors of the two enums + identical classifier functions reading the same catalog JSON. Byte-identical mapping enforced by §13's catalog-coverage smoke.
- **`engine-rs/crates/engine/tests/catalog_effect_kinds_parity.rs`** (NEW) — cross-language golden table test: enumerate every catalog card, assert Rust-classifier(card) == Python-classifier(card) for both kinds.
- **`training/v37_catalog_coverage_smoke.py`** (NEW) — iterates every card in `shared/src/data/cards.json`, asserts each maps to a known variant, prints per-class card counts, fails LOUD on unmapped or on classifier disagreement between languages.

### 3.4.2 v3.6 byte-identity preservation

Because no `PublicUmaObservation` / `PublicSideView` fields are added, v3.6 builders are GUARANTEED byte-identical post-v3.7 (the existing 246-d output uses only v3.6-era obs fields; adding catalog-derived computations downstream does not perturb upstream serialization). The existing `v36_python_parity_fixtures.rs` test continues to pass without modification.

---

## 4. Implementation order

1. **Effect-kind enums + classifiers** (§3.4.1): land `engine-rs/crates/engine/src/core/effect_kinds.rs` with `ToolEffectKind` (4 variants), `AbilityEffectKind` (8 variants), `classify_tool_effect()`, `classify_active_ability()`. Land Python mirror at `training/uma_ai/effect_kinds.py` with `IntEnum` + identical classifiers reading `shared/src/data/cards.json`. NO featurizer change yet; v3.6 byte-identity preserved by construction (no obs-contract fields added). **Smoke:** `v37_catalog_coverage_smoke.py` enumerates every card, asserts 100% coverage with explicit "Other" buckets accepted only for the known one-off ability cards (Tachyon coin-draw / U=ma2 disable / Rudolf shuffle). Cross-language parity test `catalog_effect_kinds_parity.rs` asserts Rust-classifier(card) == Python-classifier(card) for every card.
2. ~~Frozen vocab definitions step~~ — FOLDED into step 1 above per §3.4 simplification.
3. **Python `observation_to_features_v3_7` + `STATE_DIM_V3_7 = 296`** in `training/uma_ai/features.py`.
   - Call `observation_to_features_v3_6(...)` to seed slots [0:246].
   - Append 50 bits at indices [246:296] in this **CORRECTED LOCKED LAYOUT** (per §13 recon: 4-tool / 8-ability vocab, [272:274] reserved bits dropped to repack):
     - `[246:248]` (2) — `own_weakness_adjusted_lethal`, `opp_weakness_adjusted_lethal`. **§4.5 Channel 1.**
     - `[248:250]` (2) — `own_weakness_adjusted_secondary_ko`, `opp_weakness_adjusted_secondary_ko`. **§4.5 Channel 1.**
     - `[250:254]` (4) — `own_primary_has_coin_flip`, `own_primary_coin_expected_ko`, `opp_primary_has_coin_flip`, `opp_primary_coin_expected_ko`. **§4.5 Channel 2.**
     - `[254:258]` (4) — `own_secondary_has_coin_flip`, `own_secondary_coin_expected_ko`, `opp_secondary_has_coin_flip`, `opp_secondary_coin_expected_ko`. **§4.5 Channel 2.**
     - `[258:262]` (4) — `own_primary_per_energy_bonus`, `own_primary_per_bench_bonus`, `opp_primary_per_energy_bonus`, `opp_primary_per_bench_bonus`. **§4.5 Channel 3.**
     - `[262:266]` (4) — `own_secondary_per_energy_bonus`, `own_secondary_per_bench_bonus`, `opp_secondary_per_energy_bonus`, `opp_secondary_per_bench_bonus`. **§4.5 Channel 3.**
     - `[266:270]` (4) — `own_primary_usable_next_turn`, `own_secondary_usable_next_turn`, `opp_primary_usable_next_turn`, `opp_secondary_usable_next_turn`. **§4.5 Channel 4.**
     - `[270:272]` (2) — `own_paralysis_window_open`, `opp_paralysis_window_open`. **§4.5 Channel 4.**
     - `[272:276]` (4) — own active tool effect-kind one-hot, 4 frozen classes. **§4.5 Channel 5.**
     - `[276:280]` (4) — opp active tool effect-kind one-hot, 4 frozen classes. **§4.5 Channel 5.**
     - `[280:288]` (8) — own active ability effect-kind one-hot, 8 frozen classes. **§4.5 Channel 6.**
     - `[288:296]` (8) — opp active ability effect-kind one-hot, 8 frozen classes. **§4.5 Channel 6.**
   - **Reconciliation note (post-recon):** the original §1 TL;DR proposed (a) 7 channels including own-bench-typed restoration; (b) 6-tool/5-ability vocab; (c) reserved-for-v3.7.1 bits at [272:274]. All three CUT at recon-reconciliation: (a) own-bench-typed deferred to v3.7.1 at [296:306]; (b) vocabs tightened to 4-tool/8-ability per catalog reality; (c) reserved bits dropped to repack C5+C6 within state_dim=296. Final bit count: 50.
4. **Rust mirror in `featurize.rs`** — new `observation_state_features_v3_7` function, called when policy/dispatch picks v3.7. Bit-exact parity. Helpers: `v37_weakness_adjusted_lethal`, `v37_coin_flip_expected_damage`, `v37_conditional_bonus_bits`, `v37_energy_eta`, `v37_paralysis_window`, `v37_tool_effect_kind`, `v37_ability_effect_kind`.
5. **Schema dispatch updates** — `_SCHEMA_BY_STATE_DIM[296] = STATE_FEATURE_SCHEMA_VERSION_V3_7` in `features.py`; `serve_onnx.py` schema table; Rust `inference/mod.rs` ONNX-graph signature check accepts state_dim=296 for v3.7 dispatch (`GraphSchema::V3_7` enum variant).
6. **`make_v37_combat_arith_init.py`** — ckpt expander. Iter-0 parity assertion (`Δlogits ≤ 1e-5`) on a fixed observation batch. Refuses non-v3.6 sources, slot-token ckpts, action_schema_version != 3.
7. **Smoke tests** —
   - `v37_combat_arith_smoke.py` — Python builder layout + truth-table coverage for each channel (weakness arithmetic, coin-flip parsing, conditional-bonus detection, energy-ETA predicate, paralysis window, tool effect-kind, ability effect-kind).
   - `v37_tail_init_smoke.py` — init-parity (Δlogits ≤ 1e-5 from v3.6 source).
   - `v37_python_rust_parity_smoke.py` — 10 fixtures, Gate A STRICT bit-identity on [246:296], Gate B 4-ULP head bound (mirror v3.6 cadence). Fixtures cover: face-value-only matchup (weakness off), weakness-on matchup, coin-flip primary, coin-flip secondary, conditional-bonus active, paralysis-window open, tool present (each effect class once), ability present (each effect class once), mid-game energy ETA truth case, terminal-frame.
   - `v37_catalog_coverage_smoke.py` — iterate every card in `data/cards/*.json`, assert `tool_effect_kind` / `ability_effect_kind` enum classification returns a known variant; fail-loud on unmapped.
   - Rust parity smoke: extend `tests/v37_python_parity_fixtures.rs` analogous to v3.6.
8. **Queue entry** — A1 = v3.7 full bundle from v3.6 init; A0 = v3.6 cont-iter2 anchor (REUSE wl=0.5880 at n=10k); per-sub-bundle arms A2-A5 only fire if A1 regresses.

### 4.5 Synthesis-bit definitions (LOCKED here to prevent parity drift)

**Locked at scope time. Any changes after A1 fire require a fresh scoping doc (v3.7.1).**

#### Channel 1 — Weakness-adjusted lethal & secondary-KO (CORRECTED: additive, not multiplicative)

Per side, 2 bits:
- `own_weakness_adjusted_lethal`: Let `def = own.active`, `att = opp.active`. Catalog-derived: `att_attack = catalog[att.card_id].attacks[0]` (the engine's primary attack source). Bit = 1 iff `att_attack.damage + (def_weakness.amount if def_weakness.r#type == att.r#type else 0) ≥ def.hp - def.damage_counters`. Mirrors `flow/combat.rs:303-305` exactly (additive weakness bonus).
- `own_weakness_adjusted_secondary_ko`: analogous for `att.attacks[1]`, predicated on `opp_secondary_attack_usable` (v3.6's bit at slot [244]). If [244] = 0, this bit = 0.
- Symmetric bits for opp.

**Excluded modifiers** (KEPT LOCKED OUT — defer to v3.8):
- Damage-reduction tools (`toolDamageReduction` flag on `TrainerEffect`). Parity surface ≈ tool catalog × per-attack interaction matrix; admitted in v3.8.
- Status conditions affecting damage (e.g., burned defender). Engine special-conditions × per-attack matrix.
- Coin-bonus damage stacked on weakness (Channel 2 handles coin-flip separately at face-value lethal; Channel 1 ignores `coin_bonus` to keep the four predicates orthogonal).

#### Channel 2 — Coin-flip-expected damage (CORRECTED to flat-field detection)

Per side, 4 bits (2 per attack). `Attack` is a flat struct, not an enum; coin-flip detection reads boolean / scalar fields:
- `own_primary_has_coin_flip` = ANY of these is set on `att.attacks[0]`:
  - `coin_bonus: Option<i32>` (single-flip damage-on-heads bonus), OR
  - `knock_out_active_if_all_coin_heads: Option<i32>` (multi-flip boolean KO), OR
  - `draw_on_heads: Option<u32>` (heads-conditional card draw), OR
  - `discard_random_opponent_hand_on_heads: Option<u32>` (heads-conditional hand discard).
- `own_primary_coin_expected_ko` = `att.attacks[0].damage + expected_coin_damage ≥ def.hp - def.damage_counters` where `expected_coin_damage = 0.5 * att.attacks[0].coin_bonus.unwrap_or(0)`. **Single-flip formula only**: `knock_out_active_if_all_coin_heads` contributes 0 to expected-damage (its KO is boolean, not damage-scaling); other coin variants contribute 0. Keeps parity surface to a single multiplication per attack.
- Symmetric for `own_secondary_*`, `opp_primary_*`, `opp_secondary_*`.

#### Channel 3 — Conditional damage bonus type indicators (CORRECTED to flat-field detection)

Per side, 4 bits (2 per attack):
- `own_primary_per_energy_bonus` = `att.attacks[0].damage_per_attached_energy.is_some() OR att.attacks[0].damage_per_unique_attached_energy.is_some()`. Structural-only flag.
- `own_primary_per_bench_bonus` = `att.attacks[0].damage_per_umamusume_in_play.is_some()`. Structural-only flag (the engine's per-uma-in-play scope can be own-or-all; v3.7 treats both as "per-bench" for the bit since the model has bench counts at v3.6 and can disambiguate via interaction with side bench counts).
- Symmetric for secondary.

**Why structural-only:** computing the actual bonus damage requires the per-Uma attached-energy count and per-side bench count (both already in v3.6). Letting the model combine "bonus-present flag" × "energy count" / "bench count" is cheaper than baking the multiplication into a parity-fragile bit.

#### Channel 4 — Energy ETA + paralysis exploit window (CORRECTED paralysis source)

Per side, 5 bits each (4 ETA at [266:270] + 2 paralysis at [270:272] — wait, that's 6 total; correction: ETA uses 2 bits per side at [266:270] for primary+secondary respectively, paralysis uses 1 bit per side at [270:272]):
- `own_primary_usable_next_turn` = catalog-derived `att_attack[0].cost: EnergyCost` (the `[u8; 10]` typed multiset) minus current `own.active.energies` typed coverage minus the next-turn attach budget (= 1, default) is fully coverable from `own.energy_pool` typed multiset. **LOCKED DEFINITION**: this is a *structural feasibility* predicate, not a probability — if EVERY shortfall color has at least one matching entry in `own.energy_pool`, bit = 1. Otherwise 0. NO expected-value calculation; NO opp-attach modeling. Single Python ↔ Rust set-difference operation per attack.
- `own_secondary_usable_next_turn`: analogous for `attacks[1]` (predicated on attacks vector length ≥ 2).
- `own_paralysis_window_open` = `opp.active.turn_state.paralysis_recovery_pending` (already a typed boolean on `PublicUmaTurnState` at `policy/types.rs:118`). When set, opp.active cannot attack next turn → own has a free-attack window. **No `special_conditions` string token lookup** (the original scope-doc draft was wrong about this).
- Symmetric for opp.

**Excluded** (KEPT LOCKED OUT — defer to v3.8):
- Lethal-in-N for N > 1 (search-tree shape).
- Energy ETA accounting for opp.attach (would require opp-attach-color distribution modeling).
- Status windows for asleep / frozen / confused (deferred to v3.7.1 if signal lifts; would append at [296:306]).

#### Channel 5 — Tool effect-kind one-hot (CORRECTED to 4-variant vocab)

**4 frozen abstract classes** (one-hot, exactly one bit set per active iff `tool_card_id` is non-None):
0. **Heal-at-turn-end** — matches `TrainerEffect.toolEndTurnHealActive: Option<i32>` set. Today: `leftoverCarrot` (heal 10).
1. **Damage-reduction** — matches `TrainerEffect.toolDamageReduction: Option<i32>` set. Today: `trainingHelmet` (reduce 10).
2. **Counter-damage** — matches `TrainerEffect.toolCounterDamage: Option<i32>` set (counter-damage when attacked). Today: `boxingGloves` (counter 20).
3. **Other** — generic catch-all; the catalog-coverage smoke flags ANY new tool that falls here and demands a vocab review (v3.7.1 scoping doc required, NOT a silent patch).

If `tool_card_id == None`, all 4 bits = 0. Multi-effect tools assign to the dominant class by classifier precedence (Heal > Damage-reduction > Counter-damage > Other). Today no multi-effect tool exists; precedence is preventative.

**Why shrunk from 6:** catalog inventory (per §13 recon) shows ZERO cards for the originally-proposed "energy-cost-modifier", "retreat-cost-modifier", "attack-bonus" tool variants, and adds `toolCounterDamage` which the original vocab missed. The 4-variant lock matches current catalog reality with one "Other" buffer for one-off cards.

#### Channel 6 — Active ability effect-kind one-hot (CORRECTED to 8-variant vocab)

**8 frozen abstract classes** (one-hot, exactly one bit set per active iff `used_ability_this_turn = true` AND classifier returns a known kind):
0. **Heal-on-turn-start** — `Ability.heal: Option<i32>`. Today: `matikanetannhauserStage2`.
1. **Damage-reduction** — `Ability.damageReduction: Option<i32>`. Today: `mihonoBourbon` line (4 cards).
2. **HP-bonus** — `Ability.activeHpBonus: Option<i32>`. Today: `niceNature` line (2 cards). *Persistent stat buff, semantically distinct from healing.*
3. **Energy-acceleration** — `Ability.moveBenchedEnergyToActive: Option<u32>` (or analogous attach-outside-normal-budget flag). Today: `haruUrara` Basic.
4. **Conditional-attack-bonus** — `Ability.attackDamageBonusIfAttachedEnergy: Option<_>` OR `attackDamageBonusIfEvolvedLastTurn: Option<_>`. Today: `agnesDigital` Stage1, `tamamoCross` Stage2.
5. **Direct-damage** (ping) — `Ability.damageOpponent: Option<_>`. Today: `manhattanCafe` Stage1.
6. **Retreat-modifier** — `Ability.retreatCostZeroIfX: Option<_>` (any retreat-cost-zero-on-condition flag). Today: `twinTurbo` BasicEx, `daiwaScarlet` Stage1/Stage2.
7. **Other** — catch-all for genuine one-offs. Today: `agnesTachyon` Stage1 (coin-flip card draw), `agnesTachyon` Stage1Ex (disable-abilities aura), `symboliRudolf` Stage2Ex (shuffle-discard).

If `used_ability_this_turn == false`, all 8 bits = 0. The bit only fires for ABILITIES USED THIS TURN; matches the v3.6 `used_ability_this_turn` semantics.

**Why expanded from 5:** catalog inventory (per §13 recon) shows 17 ability cards across 7 distinct effect classes the original 5-variant vocab would have collapsed into "Other" (18% Other rate, well above the 5% threshold). Expansion to 8 classes (including HP-bonus, Retreat-modifier, Conditional-attack-bonus, Direct-damage as their own variants) preserves discrimination; Other now holds only 3 genuine one-offs (Tachyon coin-draw, U=ma2 disable, Rudolf shuffle) at an irreducible 18% floor.

#### Locked exclusions (kept LOCKED OUT — defer to v3.8 or later)

- **Damage-reduction tools applied to weakness arithmetic** (Channel 1 extension).
- **Status-affected damage** (burned defender +10).
- **Lethal-in-N for N > 1** (search-tree).
- **Opp-attach color distribution** (energy ETA conditional on opp behaviour).
- **Effect magnitudes for tools/abilities** (heal amount, draw count) — only kind one-hot here, not magnitude.
- **Hand-trainer effect-class multihot** (gap §2.C item 5) — held for v3.8.
- **Searchable-targets-remaining-in-deck multihot** (gap §2.C item 5) — held for v3.8.
- **Bench-temporal-promotion-cost arithmetic** (gap §2.C item 6 deep extension) — held.

---

## 5. Ablation plan

**Recipe baseline:** v3.6 cap128 mirror (hidden_dim=128, depth=2, KL=0.05, W6-fix flags ON, 8×240 selfplay, MCTS sims=100). The recipe that produced v3.6 cont-iter2 wl=0.5880 at n=10k.

**Arms:**
- **A0** (baseline): v3.6 cont-iter2 retrain — REUSE the §4l v3.6 wl=0.5880 measurement at n=10k. NO new training. Manifest at `runs/R16-P1-v36-cap128-cont-iter2-tight-gate/gate.manifest.json` IS the A0 reference.
- **A1** (full bundle): v3.7 retrain (v3.6 head + 50-bit thick tail). Built from `make_v37_combat_arith_init.py` applied to v3.6 cont-iter2 ckpt. Iter-0 parity contract verified (Δlogits ≤ 1e-5). Then run v3.6 cap128 recipe mirror at state_dim=296, 5-input ONNX.
- **A2-A5 (CONDITIONAL):** fire ONLY if A1 lands in the regression band (wl_lower(A1) < 0.5750). Each strips one sub-bundle from A1 to localize the drag:
  - **A2: v3.7 minus Channels 1-4 (combat arithmetic)** — keep only catalog channels [274:296]. Tests H2 sub-bundle independence.
  - **A3: v3.7 minus Channels 5-6 (catalog)** — keep only combat arithmetic [246:274]. Tests H2 sub-bundle independence.
  - **A4: v3.7 minus Channels 1+3 (weakness + conditional structural)** — keep coin-flip + ETA + paralysis + catalog. Tests whether structural-only flags help and the parity-fragile arithmetic hurts.
  - **A5: v3.7 minus Channel 2 (coin-flip arithmetic)** — most parity-fragile sub-channel; tests H4 parity-poison hypothesis.

**Sample size:** n=10k tight gate for A1 (mirrors v3.6 cadence). Smoke at n=240 first as sanity. A0 reference already at n=10k.

**Decision rules:**
1. **Hard ship (LIFT band):** wl_lower(A1) ≥ 0.6040 → v3.7 promoted as the new schema candidate, decisively clears v3.0 upper bound 0.6004. Production-pin discussion. Schema axis NOT closed.
2. **Soft ship (non-regression):** 0.5750 ≤ wl_lower(A1) < 0.6040 → v3.7 ships as the new state-vector schema; v3.6 retires the way v3.5 did post-§4l. Next slice scoped against v3.7. Schema axis CLOSED at thick-bundle scale (v3.5/v3.6/v3.7 all soft-ship without lift — verdict generalises).
3. **Hard fail:** wl_lower(A1) < 0.5750 → fire A2-A5 to decompose. If a single sub-bundle clears 0.5750 cleanly, drop the offending sub-bundle and re-fire with the trimmed bundle. If ALL sub-bundles regress, the verdict "thick + risky also fails" closes the schema axis DEFINITIVELY at this trunk; next pivot is off-axis (per-side asymmetry diagnostic + recipe-axis recipe-axis closed → architecture-axis = set-attention Slice 2 re-baseline OR history features R7.b.4).

**Estimated wall:**
- v3.6 cap128 recipe wall: ~70-90 min for the 8×240 loop (matches v3.5-extended / v3.6 cadence).
- n=10k tight-gate: ~11 min at v3.7's projected 5-input throughput (similar to v3.6).
- **Total: ~85-105 min wall for A1.**
- **A2-A5 budget: ~7 hours wall if all four fire.** Run A2 + A3 first (highest-prior diagnostic split: arithmetic-vs-catalog).

---

## 6. Out of scope (explicit)

Items deferred past v3.7 even though gap audit lists them:

- **Action-vector items** (attach-energy type, trainer effect magnitudes, retreat cost) — action axis, separate scoping doc. Action-schema-version stays at 3.
- **Bench-temporal in widened slot tokens** — touches the slot-token channel v3.4 falsified. Held until set-attention Slice 3 verdicts.
- **Species embedding per Uma** — card vocab axis. Held.
- **Slot tokens** (per-Uma 23-d row). Falsified at v3.4. Doctrine unchanged.
- **Damage-reduction tools applied to weakness arithmetic.** Held — too much per-tool catalog scan; v3.8 candidate.
- **Status-affected damage modifiers** (burned +10). Held — per-condition × per-attack matrix.
- **Lethal-in-N for N > 1.** Search-tree shape. Held permanently or moved to MCTS-side computation.
- **Opp-attach-color modeling for energy ETA.** Hidden-info-fragile. Held.
- **Effect magnitudes** (heal amount, draw count, gust target index). Held — magnitude axis is a separate scoping doc.
- **Hand-trainer effect-class multihot** + **searchable-targets-in-deck multihot.** Both gap §2.C items, both candidates for v3.8 if v3.7 lifts.
- **Own bench typed energy aggregate restoration.** Pulled from v3.7 at §4 impl reconciliation. If signal lifts, append at [296:306] in v3.7.1.
- **Per-side asymmetry diagnostic.** Different axis (queue id `per-side-asymmetry-probe`). Runs in parallel; v3.7 does not block or supersede.

---

## 7. Constraints honored

Per `v33-feature-gap-brainstorm-handoff.md` §6 + `v35-multichannel-tail-scoping.md` §7 + `v36-priors-and-arithmetic-scoping.md` §7:

- **No opponent hand IDs.** All v3.7 channels read from public observation only.
- **No raw turn-stamps.** All temporal predicates (energy ETA, paralysis window) derive from public state.
- **No ability-name strings.** Channels 5 + 6 use frozen abstract effect-kind vocabularies (6 + 5 variants); names are NOT featurized.
- **v3.6 byte freeze** (slots [0:246]) preserved. v3.7 is strictly additive at [246:296].
- **`_SCHEMA_BY_STATE_DIM` fail-fast dispatch** preserved — new entry, no silent fallback.
- **Init parity / zero-init residual** — `make_v37_combat_arith_init.py` ensures bit-identical iter-0 outputs vs v3.6 (Δlogits ≤ 1e-5 — strict band, same as v3.5; no column-drop here unlike v3.6).
- **Action-schema-version guard** (commit `6a7dd84`) — unaffected; action schema stays at 3.
- **Process-level frozen vocabs** — Channels 5 + 6 effect-kind enums are LOCKED at scope time, with `v37_catalog_coverage_smoke.py` enforcing 100% catalog coverage. Future card additions that don't map to a v3.7 variant FAIL THE SMOKE and require a fresh scoping doc (v3.7.1 or v3.8), not a silent vocab patch.

---

## 8. Conditional escalation paths

**If A1 lands in the LIFT band (rule 1, wl_lower ≥ 0.6040):**
- v3.7 becomes the strongest schema candidate ever measured.
- Promotion-ready against the 96-d production pin pending: (a) deck-pair-sampling Slice 3 verdict on v3.7, (b) Rust observation builder confirmed emitting state_dim=296 in self-play binary, (c) export-onnx parity smoke pass, (d) catalog-coverage smoke green on the production card set.
- Open v3.8 scoping: held items (damage-reduction × weakness, status-affected damage, magnitude axis, hand-trainer multihot, deck-searchable multihot) using v3.7's parity discipline as the template.

**If A1 lands in non-regression band but NOT lift (rule 2):**
- v3.7 ships as the new schema baseline. v3.6 retires.
- **Major implication: schema axis verdict generalises.** v3.5 (orthogonal-bit), v3.6 (info-density rotation), v3.7 (qualitatively new representational classes including parity-fragile arithmetic and catalog lookup) ALL soft-ship without lift. The schema-axis closes definitively at this trunk capacity (hidden=128/depth=2/state_dim≤296). Next pivot is recommended to be off-axis: per-side asymmetry probe primary, set-attention Slice 2 re-baseline secondary, history/sequence features tertiary. Future schema work would require either (a) compounding axes (set-attention trunk + schema) or (b) a recipe-axis breakthrough first.

**If A1 fails (rule 3):**
- Fire A2 (catalog-only) + A3 (arithmetic-only) FIRST — highest-prior diagnostic split. Each ~85-105 min wall.
- If A2 lifts but A3 regresses (catalog channels carry, arithmetic drags): ship v3.7.0 in catalog-only form (drop [246:274], renumber); v3.8 sequel attempts the arithmetic with stricter parity discipline.
- If A3 lifts but A2 regresses (arithmetic carries, catalog drags): ship v3.7.0 in arithmetic-only form (drop [274:296], renumber); revisit catalog vocabs.
- If BOTH A2 and A3 regress (both sub-bundles harmful): falsify H1 and H2 simultaneously. Verdict: schema-axis closed not just at "thick + risky" but at any sub-bundle of these classes. Pivot directly to off-axis without further v3.7-class attempts.

**Comparison anchors at decision time:**
- v3.0: wl_lower 0.5811.
- v3.3 (4-iter): wl_lower 0.5910 (peak, original sweet-spot).
- v3.5-cap128: wl_lower 0.5912.
- v3.6 cap128 cont-iter2: wl_lower 0.5880.
- v3.6 depth=3: wl_lower 0.5910.
- v3.7 NON-REGRESSION band: wl_lower ≥ 0.5750.
- v3.7 LIFT band: wl_lower ≥ 0.6040.

---

## 9. The bet (explicit framing)

User-stated framing: "i want it to be thick addition and take risks."

What this scoping doc commits to:

1. **§4l explicitly concluded "the bet for v3.7 cannot be 'more bits.'"** v3.7 ARGUES BACK with: the v3.5/v3.6 bundles ran more *quantity-of-bits* on the *same kinds of channels* (energy color, prize, role-residual). v3.7 introduces channel CLASSES the v3.x lineage has NEVER ENCODED (weakness arithmetic, probabilistic damage, conditional bonuses, effect-kind one-hot). The bet is **NOT** "more bits will help." The bet is **"the remaining unencoded representational classes are qualitatively different and the schema-axis verdict has been measured only on classes we've actually tried."**

2. **§4m exhausted the recipe-axis at hidden=128.** v3.7 does NOT touch the recipe axis. If v3.7 fails the LIFT band, the verdict "schema + recipe combined are exhausted at this trunk" hardens. v3.7 is therefore designed AS A FALSIFIABLE TEST OF THE STRONG VERSION OF THE SCHEMA-EXHAUSTION CLAIM. A clean failure of v3.7 is itself useful evidence — schema-axis truly closes, not just "the bit-density rotation didn't crack it."

3. **The risks taken are inventoried explicitly:**
   - Parity-fragile arithmetic (Channels 1, 2, 3 partly) — guarded by `v37_python_rust_parity_smoke.py` Gate A STRICT and H4 falsifiable band.
   - Catalog-vocab fragility (Channels 5, 6) — guarded by `v37_catalog_coverage_smoke.py` and H3 falsifiable band. Vocabs are LOCKED at scope time; future card additions that don't map require a fresh scoping doc.
   - Channel-density risk (5 sub-channels on top of v3.6's 6) — addressed by §1 channel-orthogonality table; each new channel touches a representation axis v3.6 cannot express; ablation arms A2-A5 catch any unexpected drag.
   - Meta-risk (going schema-axis when §4m says off-axis is the highest-leverage probe) — addressed by NOT BLOCKING the per-side asymmetry probe; v3.7 runs in parallel as a schema-axis bet, asymmetry diagnostic continues independently.

4. **If v3.7 also fails the LIFT band, the case for further schema-axis iteration at this trunk capacity closes DEFINITIVELY.** Next pivot becomes off-axis: per-side asymmetry diagnostic (queue id `per-side-asymmetry-probe`), set-attention architecture-axis (Slice 2 re-baseline → Slice 3), history/sequence features (R7.b.4).

The implementer is to land code + smoke + init-parity ckpt + Python↔Rust parity smoke + catalog-coverage smoke. A1 fire is **USER-GATED**. Same cadence as v3.5 / v3.6.

---

## 10. Canonical file pointers

### Feature code
- `training/uma_ai/features.py:125` — `STATE_DIM_V3_6 = 246` (anchor constant).
- `training/uma_ai/features.py:786` — `_v36_lethal_face_value` (template for `_v37_weakness_adjusted_lethal`).
- `training/uma_ai/features.py:806` — `_v36_secondary_attack_bits` (template for `_v37_weakness_adjusted_secondary_ko`).
- `training/uma_ai/features.py:833` — `observation_to_features_v3_6` (template for layered builder).
- `engine-rs/crates/engine/src/policy/featurize.rs` — `observation_state_features_v3_6` (template for Rust mirror); add `observation_state_features_v3_7`.
- `engine-rs/crates/engine/src/policy/types.rs:121` — `PublicObservation` / `PublicUmaObservation`; extend with weakness fields (verify already present), `PublicAttackSummary`, `tool_effect_kind`, `active_ability_kind`.
- `engine-rs/crates/engine/src/flow/combat.rs:130-146,303-305` — weakness lookup truth source; mirror in v3.7 Channel 1 builder.
- `engine-rs/crates/engine/src/core/effects.rs:133` — `Attack` struct + `AttackEffect` enum; extend with `ToolEffectKind` and `AbilityEffectKind` enums (6 + 5 frozen variants).
- `training/make_v36_priors_init.py` — template for `make_v37_combat_arith_init.py` (append-only, no column-drop).
- `training/serve_onnx.py` — adds dispatch entry for state_dim=296.
- `engine-rs/crates/engine/src/inference/mod.rs` — `GraphSchema::V3_7` enum variant + 5-input contract dispatch.

### Research docs
- `docs/ai-research/scoping/v33-feature-gap-brainstorm-handoff.md` §2.B / §2.C — gap inventory (combat-arithmetic-depth + catalog-lookup items).
- `docs/ai-research/scoping/v35-multichannel-tail-scoping.md` — channel-orthogonality discipline.
- `docs/ai-research/scoping/v36-priors-and-arithmetic-scoping.md` §3.4 / §4.5 / §12 — obs-contract extension pattern + LOCKED definitions template + 6-phase commit chain pattern.
- `docs/ai-research/progress/r110.md §4l` — v3.6 verdict (the v3.7 baseline).
- `docs/ai-research/progress/r110.md §4m` — recipe-axis exhaustion + per-side asymmetry probe (the verdict v3.7 deliberately argues back against).

### State files
- `docs/ai-agent-state/queue.json` — A1 queue entry to add when this scope flips to APPROVED. Queue id suggestion: `v37-combat-arith-and-catalog`.
- `docs/ai-research-backlog.md` — top-of-frontier update when v3.7 ships.

---

## 11. What this doc is NOT

- Not a verdict on any v3.7 channel. The seven channels are pre-registered hypotheses with falsifiable bands.
- Not authorization to land code. Scope-only DRAFT; **USER-GATED** for approval before implementation begins.
- Not a contradiction of the §4m verdict — see §9 for the explicit "argues back" framing. If v3.7 also fails, §4m verdict hardens; if v3.7 lifts, §4m's "more bits won't help" claim was scoped only to channel classes we'd already tried.
- Not a defense of indefinite schema-axis iteration. If A1 fails AND ablation arms also regress, v3.7 is the last schema-axis bet at this trunk capacity; the next pivot is off-axis.
- Not a substitute for the per-side asymmetry probe. Both run independently; the asymmetry probe is the highest-leverage off-axis line in the queue.

---

## 12. Status line

**`APPROVED-IMPLEMENTING-CORRECTED 2026-05-26`** — user approved code + compute in same turn ("ok implement e2e. i approve compute runs"). Two parallel investigators ran pre-impl recon and surfaced three engine-API corrections + catalog vocab tightening (§13). Implementation chain in flight on the corrected scope.

---

## 13. Recon corrections (pre-impl audit findings)

Two parallel investigators ran before any code landed: (a) catalog-vocab enumeration, (b) obs-contract / `Attack` / `AbilityEffect` / `TrainerEffect` surface verification. They surfaced corrections to the original DRAFT that affect §3.4 + §4.5. **The conceptual bet (§9) is unchanged.** Corrections:

### 13.1 No new `PublicUmaObservation` / `PublicSideView` fields

The DRAFT §3.4 proposed adding `weakness_type`, `weakness_amount`, `PublicAttackSummary`, `tool_effect_kind`, `active_ability_kind` to `PublicUmaObservation`. Recon found ALL of these are catalog-derivable from `card_id` + the existing v3.6 obs surface. Both Python and Rust share `shared/src/data/cards.json` as the canonical catalog; the v3.6 featurizer already uses catalog lookup (`_v36_active_attacks(active)` at `training/uma_ai/features.py:719-735`). v3.7 extends this pattern; no obs-contract change.

**Implication:** Phase A simplifies from "land obs-contract fields + ignore in v3.6" to "land Rust enums + classifier functions + Python mirror + catalog-coverage smoke." The 6-phase v3.6 cadence collapses to 3 phases (effect-kinds + featurizers, ckpt expander + parity smoke, A1 fire).

### 13.2 Weakness is ADDITIVE, not multiplicative

The DRAFT §4.5 Channel 1 referenced "the weakness multiplier (2× damage if type matches weakness)." Engine reality (`flow/combat.rs:303-305`): `if damage > 0 && def_weakness.r#type == att.r#type { damage += def_weakness.amount }`. Additive bonus, not 2× multiplier. The corrected §4.5 Channel 1 uses `damage += amount`.

### 13.3 `AttackEffect` enum does not exist

The DRAFT §3.4 item 2 + §4.5 Channels 2 + 3 referenced `Attack.effects: Vec<AttackEffect>` with variants like `FlipCoins { count, damage_per_head }`, `BonusDamagePerAttachedEnergy`, `BonusDamagePerBenchedUma`. **None of this type exists.** `Attack` (`core/effects.rs:133-196`) is a flat struct with ~30 `Option<...>` fields. The corrected §4.5 Channels 2 + 3 use the actual field names:
- Coin-flip detection: `coin_bonus`, `knock_out_active_if_all_coin_heads`, `draw_on_heads`, `discard_random_opponent_hand_on_heads`.
- Per-energy bonus: `damage_per_attached_energy`, `damage_per_unique_attached_energy`.
- Per-bench bonus: `damage_per_umamusume_in_play`.
- Expected-coin formula simplified to `0.5 * coin_bonus.unwrap_or(0)` (single flip; multi-flip variants contribute 0 to expected-damage).

### 13.4 `paralysis_recovery_pending` is already typed

The DRAFT §4.5 Channel 4 referenced a `PARALYSIS_RECOVERY_PENDING` token in `special_conditions: Vec<String>`. Engine reality: it's a typed boolean `paralysis_recovery_pending: bool` on `PublicUmaTurnState` (`policy/types.rs:118`, computed at `policy/observation.rs:240-242`). The corrected §4.5 Channel 4 reads from `active.turn_state.paralysis_recovery_pending`.

### 13.5 Vocab tightening: 4-tool / 8-ability

Catalog enumeration (`shared/src/data/cards.json`): 3 tools across 3 distinct effect classes (Heal-turn-end, Damage-reduction, Counter-damage); 17 abilities across 7 distinct effect classes + 3 genuine one-offs.

- **ToolEffectKind shrinks from 6 → 4.** Dropped: `EnergyCostModifier`, `RetreatCostModifier`, `AttackBonus` (0 cards each). Added: `CounterDamage` (the originally-missed `toolCounterDamage` field, present on `boxingGloves`). Variants: Heal-turn-end, Damage-reduction, Counter-damage, Other.
- **AbilityEffectKind expands from 5 → 8.** Original 5 (EnergyAccel, HandSearch, Draw, Status/damage-mod, Other) would have collapsed HP-bonus (2 cards), Retreat-modifier (3 cards), Conditional-attack-bonus (2 cards), Direct-damage (1 card) into "Other" at 18% rate (above the 5% threshold). Expansion to 8 keeps Other at 3 genuine one-offs (Tachyon coin-draw, U=ma2 disable, Rudolf shuffle) at an irreducible floor.

### 13.6 Layout repacked

Original DRAFT layout reserved [272:274] for v3.7.1 status-window overflow + allocated [274:286] to 12-bit tool one-hot (6 × 2) + [286:296] to 10-bit ability one-hot (5 × 2). With the corrected 4-tool / 8-ability vocab, the [272:274] reserved bits are DROPPED to repack:
- C5 tool: [272:280] (4 bits × 2 sides = 8 bits).
- C6 ability: [280:296] (8 bits × 2 sides = 16 bits).

Net bit count unchanged at 50; state_dim unchanged at 296. v3.7.1 status windows would append at [296:306] if signal lifts (non-destructive on [0:296]).

### 13.7 Effect-encoding shape — flat structs, derived classifier

Both `Ability` (`core/effects.rs:243-279`) and `TrainerEffect` (`core/effects.rs:323-372`) are flat structs with `Option<T>` fields. A card can in principle set multiple flags. **The classifier is a derived pure function** that picks the dominant flag per card (precedence rules locked in `core/effect_kinds.rs` at impl time). Parity fragility is LOW because mapping is pure-function over the static catalog (no game-state input); a golden table of `(card_id → ToolEffectKind | AbilityEffectKind)` per language is sufficient and asserted equal by `catalog_effect_kinds_parity.rs`.

---
