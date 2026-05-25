# v3.6 Trimmed-Priors + Minimal-Arithmetic Additive Tail — Scoping

- **Date:** 2026-05-25
- **Status:** **PROPOSED 2026-05-25** — user-gated. Awaiting approval to land code + smoke + init-parity ckpt; A1 fire then queued for separate user approval per v3.5 cadence.
- **Parent:** `v35-multichannel-tail-scoping.md` (LANDED-EXTENDED-OUTPERFORMS-CONTROL) + `v33-feature-gap-brainstorm-handoff.md` §2.A/§2.C (gap inventory) + `progress/r110.md §4j` (column-norm + activation-rate diagnostics on v3.5 tail).
- **Predecessor schema:** v3.5 (212-d, no slots, v3-action). At extended training (8 iters × 240 selfplay-games) wl=0.5908 vs v3.3 0.5812 at same budget. Single seed, Wilson intervals overlap; replication pending.
- **Scope:** A single thick additive tail that (a) TRIMS the v3.5 bits §4j confirmed dead, (b) ADDS three structural-prior channels v3.5 cannot express, (c) ADDS two minimal-parity-surface arithmetic predicates. Six channels total, each pre-registered against the v3.5 channel-orthogonality rule.

---

## TL;DR

- **Trim then thick-add** on top of v3.5's 212-d state vector → new STATE_DIM 246.
  - **Drop:** 10-bit `[197:207]` opp.energy_zone front-of-queue typed one-hot (structurally dead per `progress/r110.md §4j` Diagnostic B + `flow/turn.rs:110-111` + `flow/energy.rs:18`).
  - **Add (six channels, 44 bits):**
    1. **Both-side `energy_pool` typed multihot (20 bits, 10×2)** — *future-roll* color distribution; replaces the dead `energy_zone.front` channel with the actually-prior-rich form of the same signal class.
    2. **Both-side prize one-hot ×4 (8 bits, 4×2)** — non-linear win-condition urgency, complements `f[3]=points/3` ratio.
    3. **Opp-only bench typed energy aggregate (10 bits)** — already-committed energies on opp bench Umas; opp threat-by-color, a known coverage gap (head-110 collapses this to scalar `energy_total`). Own bench dropped at impl reconciliation: 6-channel sum was 54, but TL;DR locked at +34 net (STATE_DIM 246, 44 bits added). Own bench typed cut as the lowest-priority redundancy (v3.5 head already exposes own.active typed energies; own.bench energies are partly inferable from the rest of the own-side state).
    4. **Both-side lethal-next-turn face-value predicate (2 bits, 1×2)** — minimal arithmetic: opp.active can KO own.active *at face value* (max single-attack base damage, no weakness, no bonuses).
    5. **Both-side active secondary-attack readiness + would-KO (4 bits, 2×2)** — addresses `featurize.rs:709` only-consults-`attacks[0]` gap.
- **Net change:** −10 + 44 = **+34 bits**. v3.5 keeps slots `[0:197]` byte-stable; new bits at `[197:246]` (and slot `[197:207]` is the freshly-vacated opp.energy_zone band — repurposed for the new energy-pool channel).
- **Channel-orthogonality table** vs v3.5: §1 below shows each new channel touches a representation-axis v3.5 cannot express.
- **Synthesised arithmetic admitted (carefully):** lethal-next-turn face-value (one comparison, no multipliers) and secondary-attack would-KO (same comparison shape, applied to `attacks[1]`). Definitions LOCKED at §4.5 to prevent parity drift.
- **Obs-contract extension required:** `PublicSideView` (Rust) + `PublicSideObservation` (Python) gain a typed `energy_pool` field. Heuristic AI (`flow/ai/public_info.rs:13-19`) explicitly omits this today; v3.6 makes it visible per game-public-knowledge semantics.
- **Slot tokens stay out** (falsified v3.4 + scoping doctrine).
- **Zero-init residual** warm-start preserved: v3.5 ckpt → v3.6 graph yields bit-identical iter-0 outputs via `make_v36_priors_init.py` (drop column 197-206 then append 44 zero-init columns; the dropped band is replaced 1-for-1 with own.energy_pool typed, so iter-0 output differs only by the value that was provably ≈0-rate at v3.5 activation).

**Why thick now:** v3.5 §4j shows the trunk IS using the tail (column-norms grew 0 → 0.20 frob) but per-bit yield is diluted by 15 dead/sparse bits out of 45. v3.6 trades dead bits for prior-rich bits, ON THE SAME CHANNEL CLASS (energy-color), so this is the cleanest possible A/B for the "info density beats bit count" hypothesis. The user explicitly endorses a feature-axis bet here, even though v3.5's extended-training lift is single-seed.

---

## 1. The v3.5 lesson and how this bundle applies it

`progress/r110.md §4j` Diagnostic A confirmed the v3.3→v3.5 head Linear DID learn the tail (column-norms grew 0 → 0.20 frobenius across iters; per-channel strongest: opp.discard.buckets, bench-refill, phase one-hot). Diagnostic B identified two failure-modes per channel:

1. **Structurally dead** (read-from-nonexistent-state): 10 opp.energy_zone.front bits. Not fixable by longer training — feature reads empty state at every player decision point (`flow/turn.rs:110-111` clears opp.energy_zone at opp turn start; `flow/energy.rs:18` consumes during opp turn).
2. **Sample-dead** (early-game corpus bias): 5 opp.cond bits. Activates with longer-game corpus; keep pending.

v3.5's hard channel-orthogonality rule (§1 of `v35-multichannel-tail-scoping.md`: "don't bundle features that touch the same signal channel") is **preserved**. Each v3.6 channel touches a representation axis v3.5 cannot express:

| v3.6 item | Channel | Already in v3.3 / v3.5? | Overlap risk |
|---|---|---|---|
| Both-side energy_pool typed | Future-energy distribution | v3.5 `energy_zone.front` is **point-in-time depth-1**, structurally dead on opp. Pool is **multinomial prior over the rest of the game**. | None — orthogonal to depth-1 boolean and to attached-energy typed |
| Both-side prize one-hot ×4 | Win-condition urgency (non-linear) | v3.1 has `points/3` ratio (slots 3-4). Linear. | Low — non-linearity is the new signal; ratio retained |
| Opp-only bench typed energy aggregate | Already-committed energy threat (off-active) | v3.5 has active.energies typed via head-110. Bench typed is via slot tokens, which v3.3+v3.5 EXCLUDE. | None — recovers a signal v3.4 falsified at the slot-token axis but admits at the head-aggregate axis (different representation). Own bench cut at impl reconciliation; opp side carries the threat-by-color signal. |
| Both-side lethal-next-turn face-value | Terminal-state arithmetic | v3.5 has `would_lose_on_active_KO` = (bench empty). v3.6 extends to "active dies even with bench non-empty" via direct HP-vs-damage comparison. | Low — extends, does not duplicate |
| Both-side secondary attack readiness + KO | `active.attacks[1]` representation | v3.5 + v3.3 + v3.1 head ALL only consult `attacks[0]` (`featurize.rs:709`). | None — completely new |

**Slot-token channel** (v3.4-falsified) remains EXCLUDED. **Opp-flag channel** (v3.3-locked) remains UNTOUCHED.

---

## 2. Pre-registered hypotheses

### H1 — Trimmed-priors + minimal-arithmetic compound additively into v3.6

- **Falsifiable signal:** wl_lower(v3.6) ≥ wl_lower(v3.5-extended) − 0.013 = 0.5778 (non-regression band on v3.5 extended-training anchor 0.5908). Hard fail below this.
- **Stretch:** wl_lower(v3.6) ≥ 0.6040 (clears v3.0 upper bound 0.6004 decisively → production-promotion candidate; same lift band v3.5 missed).
- **Theory:** v3.5 §4j proved per-bit yield is bottlenecked by dead/sparse bits, not by the additive-tail strategy. Trimming 10 dead bits and replacing them with a higher-prior channel (energy_pool typed) on the same axis isolates "info density per bit" as the lever.

### H2 — Energy_pool typed multihot lifts independently

- **Theory:** energy_pool is the prior over every future opp attach; encoding it gives the model a permanent forecast that v3.3/v3.5 currently has to estimate from hand+discard residuals.
- **Falsifiable signal:** A2 (v3.6 minus energy_pool, keep prize + bench + lethal + secondary-attack) wl_lower < A1 wl_lower − 0.013.

### H3 — Arithmetic predicates don't poison the bundle via parity-bug drag

- **Risk:** Python and Rust must produce identical bits for lethal-next-turn and secondary-attack readiness. A divergence yields a feature the model learns to over-trust in selfplay (Rust path) but mis-evaluates at eval (where served-from-ONNX-via-Python-batched-encoder paths still align, but parity drift between selfplay traces and tight-gate runs would silently corrupt distillation targets).
- **Falsifiable signal:** Python↔Rust featurizer-parity smoke (`v36_parity_smoke.py`, mirroring `v35_multichannel_tail_smoke.py`) fails on any random seed → block A1 fire; rebuild definitions; defer the offending bit to v3.7.
- **Recovery rule:** if A1 lands in the regression band AND A6 (v3.6 minus arithmetic bits) clears the soft-ship band, the arithmetic admission was premature; v3.6 ships in structural-only form and arithmetic stays deferred.

---

## 3. Freeze contract

| Surface | v3.5 (current) | v3.6 (proposed) | Compatibility |
|---|---|---|---|
| `state_features` dim | 212 | **246** | Drop slots [197:207]; append 44 new bits. Slots [0:197] byte-stable. New tail at [197:246]. |
| `STATE_FEATURE_SCHEMA_VERSION` (Python latest) | 3.5 | **3.6** | Bump LATEST marker; legacy 3.0/3.1/3.3/3.5 dispatch unchanged. |
| `STATE_DIM_V3_6` (new constant) | n/a | **246** | New constant; dispatch entry `_SCHEMA_BY_STATE_DIM[246]`. |
| `observation_to_features_v3_6` (new builder) | n/a | **NEW** | Layered on top of `observation_to_features_v3_5` then (a) zero-overwrites slots [197:207] and (b) appends 44 bits. Bit-exact Rust mirror in `featurize.rs`. |
| `PublicSideView` / `PublicSideObservation` | No `energy_pool` field | **+ `energy_pool: Vec<EnergyType>` (≤3)** | New field; serialised in PublicObservation contract. Heuristic AI (`flow/ai/public_info.rs`) extended to populate it. v3.5 schema still works because builder ignores the new field. |
| ONNX input set | 5 inputs (state, action, action_mask, deck_targets, action_index) | **5 inputs unchanged** | v3.6 keeps the v3.3+v3.5 5-input ONNX shape. Only `state_features` last dim grows 212 → 246. |
| `uma_slot_features[10, 23]` | Not used (post-v3.4) | **Still not used** | Slot tokens stay retired. |
| `action_features[A, 48]` | 48-d, schema 3 | **48-d, schema 3 unchanged** | Action vector untouched. |
| `_SCHEMA_BY_STATE_DIM` dispatch | {96, 110, 164, 167, 212} | **{96, 110, 164, 167, 212, 246}** | New entry → `observation_to_features_v3_6`. Fail-fast preserved. |
| Init parity | v3.5 anchor | **v3.5 → v3.6 zero-init-with-band-replacement** | See §3.1. |

### 3.1 Bit-exact mirror requirement

`observation_to_features_v3_6` lands in BOTH:
- `training/uma_ai/features.py` (Python spec).
- `engine-rs/crates/engine/src/policy/featurize.rs` (Rust mirror).

Parity smoke: extend the chain (`v3_5_head_is_byte_identical_to_v3_3`) with `v3_6_head_is_byte_identical_to_v3_5_excluding_opp_energy_zone_band`. The first 197 bytes of v3.6 output must equal v3.5's output on the same observation; bytes [197:207] in v3.6 hold the new own.energy_pool typed multihot (no v3.5 equivalent — band repurposed); v3.5's opp.energy_zone.front goes to /dev/null.

### 3.2 Existing checkpoint compatibility

- Pre-v3.6 ckpts (v3.0/v3.1/v3.3/v3.5) unchanged — `_SCHEMA_BY_STATE_DIM[246]` routes only v3.6-shaped inputs.
- Loading a v3.5 ckpt into the v3.6 graph requires `training/make_v36_priors_init.py`:
  1. Reads v3.5 lineage best (`runs/<v3.5-extended-best>/checkpoint.pt`).
  2. **Drops** columns 197-206 of the first Linear `state_features → hidden` (the dead opp.energy_zone.front band). At v3.5 activation rate ≈0 these columns contribute ≈0 to outputs already, so dropping them yields ≤ Δlogits ≈ activation_threshold × ||column||.
  3. **Appends** 44 zero-init columns at indices 197-240 (10 own.energy_pool typed) and 207-246 (other new channels), reordering so the drop is in-place at [197:207].
  4. Saves a v3.6-shaped ckpt.
  5. Asserts iter-0 output drift vs v3.5 source below tolerance `1e-3` on a fixed observation batch (looser than v3.5's `1e-5` because column-drop is not strictly bit-identical — opp.energy_zone activation rate is ≈0 but not exactly 0).
- Mirrors `make_v35_tail_init.py` (template). Adds drop-column logic absent from previous expanders.

### 3.3 Action-schema-version guard

Unchanged. Action schema stays at 3. Rust loader guard (commit `6a7dd84`) fires consistently across v3.5 → v3.6.

### 3.4 Obs-contract extension (THE INFRA WORK)

This is the only non-featurizer piece. Order of operations:

1. **Rust:** add `energy_pool: Vec<EnergyType>` to `PublicSideView` (`engine-rs/crates/engine/src/flow/ai/public_info.rs:13-19`). Populate from `SideState.energy_pool` (`engine-rs/crates/engine/src/core/state.rs:104`). Per-game-setup it's a fixed ≤3-element vector chosen at deck-build (`flow/setup.rs:143-168`); public from setup onward.
2. **Python types:** mirror in `PublicSideObservation` (whatever dataclass mirrors `PublicSideView` — locate via `engine-rs/crates/engine/src/policy/types.rs` → `training/uma_ai/observation.py` if it exists, or the equivalent NamedTuple Python uses).
3. **ObservationSnapshot JSON contract:** schema-version bump on the trace-jsonl path; existing v3.5 traces remain consumable by v3.5 builders, new traces carry the `energy_pool` field consumed by v3.6 builder.
4. **Backward compat:** v3.5 / v3.3 / v3.1 / v3.0 builders IGNORE the new field. They do not break.

Risk: this surface touches the observation contract that selfplay → distill → tight-gate ALL depend on. Plan to land 3.4 BEFORE any other v3.6 code, smoke independently (existing v3.5 builders still produce 212-d), THEN layer the v3.6 builder.

---

## 4. Implementation order

1. **Obs-contract extension** (`flow/ai/public_info.rs` + Python mirror + trace JSONL schema bump). Smoke: existing v3.5 builders still produce byte-identical 212-d output on a fixed observation batch (the new field is silently ignored). NO new features yet.
2. **Python `observation_to_features_v3_6` + `STATE_DIM_V3_6 = 246`** in `training/uma_ai/features.py`.
   - Call `observation_to_features_v3_5(...)` to seed slots [0:212].
   - Zero-overwrite slots [197:207] (the dead opp.energy_zone.front band).
   - Repurpose [197:207] in-band for `own.energy_pool` typed multihot (10 bits).
   - Append 34 bits at indices [212:246] in this LOCKED layout:
     - `[212:222]` (10) — opp.energy_pool typed multihot.
     - `[222:226]` (4) — own prize one-hot over remaining-prize `{3,2,1,0}`.
     - `[226:230]` (4) — opp prize one-hot over remaining-prize `{3,2,1,0}`.
     - `[230:240]` (10) — opp bench typed energy aggregate (multihot over the 10 energy types; bit set iff any opp bench Uma has ≥1 attached energy of that type).
     - `[240]` (1) — own_lethal_next_turn (face-value, §4.5 Channel 4).
     - `[241]` (1) — opp_lethal_next_turn (face-value, §4.5 Channel 4).
     - `[242]` (1) — own_secondary_attack_usable (§4.5 Channel 5).
     - `[243]` (1) — own_secondary_attack_would_KO (§4.5 Channel 5).
     - `[244]` (1) — opp_secondary_attack_usable (§4.5 Channel 5).
     - `[245]` (1) — opp_secondary_attack_would_KO (§4.5 Channel 5).
   - **Reconciliation note (impl phase):** the §4.5 channel breakdown sums to 54 bits across BOTH sides for all channels. The TL;DR locks STATE_DIM at 246 (net +34 bits). To hit 246, **own bench typed energy aggregate (10 bits) was DROPPED** at impl; opp bench typed kept since opp-threat-by-color is the stated rationale and own bench is partly redundant with v3.5's head energies. If a later replication wants own bench typed back, bump STATE_DIM_V3_6 to 256 and append at [246:256] — non-destructive of the [0:246] layout.
   - **Alternative:** write fresh layered builder from v3.3 to avoid the in-place zero-overwrite. **Final choice:** layered-on-v3.5 with zero-overwrite, to keep slot-stability invariant tight and the column-drop visible in code review.
3. **Rust mirror in `featurize.rs`** — new `observation_state_features_v3_6` function, called when policy/dispatch picks v3.6. Bit-exact parity.
4. **Schema dispatch updates** — `_SCHEMA_BY_STATE_DIM[246] = STATE_FEATURE_SCHEMA_VERSION_V3_6` in `features.py`; `serve_onnx.py` schema table; Rust `inference/mod.rs` ONNX-graph signature check accepts state_dim=246 for v3.6 dispatch.
5. **`make_v36_priors_init.py`** — ckpt expander with column-drop + append-zero-init logic. Iter-0 parity assertion (`Δlogits ≤ 1e-3`) on a fixed observation batch.
6. **Smoke tests** — `v36_priors_arithmetic_smoke.py` (Python builder layout + bench-energy-pool multihot bag-of-3 invariant + lethal-predicate truth-table); `v36_tail_init_smoke.py` (init parity); Rust parity smoke. Run at n=240 first for sanity.
7. **Queue entry** — A1 = v3.6 full bundle from v3.5 init; A0 = v3.5-extended anchor (REUSE wl=0.5908 at n=10k); per-channel arms A2-A6 only fire if A1 regresses.

### 4.5 Synthesis-bit definitions (LOCKED here to prevent parity drift)

#### Channel 1 — Energy_pool typed multihot

Per side, 10 bits, one per energy color in `_UMA_SLOT_ENERGY_TYPES` (grass, fire, water, lightning, psychic, fighting, darkness, steel, colorless, dragon). Bit set iff the side's `energy_pool` contains that type. Multihot (typically 1-3 bits set). Bag semantics: duplicates collapse (a pool of `[fire, fire, water]` sets only the fire and water bits).

#### Channel 2 — Prize one-hot ×4

Per side, 4 bits, one-hot over remaining-prize-counts `{3, 2, 1, 0}` derived as `3 − points_taken`. (Points are 0-indexed at start, increment on opponent KO; game ends at 3.) The "0" bit means "game would be over" — should never fire at a player-decision point; included for completeness and for terminal-frame featurization paths.

#### Channel 3 — Bench typed energy aggregate (OPP-ONLY at v3.6)

10 bits, one per energy color. Bit set iff the **opp** side's bench Umas have **at least one attached energy of that type** across all bench slots (does NOT include the active). Aggregate multihot, same vocab as Channel 1. Own bench aggregate dropped at impl-phase reconciliation to keep STATE_DIM at 246 (see §4 step 2); own.active typed energies already in v3.5 head. Symmetry can be restored at v3.7 if signal lifts and bench-aggregate-own becomes desired.

#### Channel 4 — Lethal-next-turn face-value

Per side, 1 bit. The bit `own_lethal_next_turn` is set iff:
- `opp.active.attacks` is non-empty, AND
- `max(attack.base_damage for attack in opp.active.attacks) ≥ own.active.hp − own.active.damage_counters`.
- "Base damage" is the static `Attack.base_damage` field — NOT adjusted for weakness, NOT adjusted for energy availability, NOT adjusted for status conditions, NOT adjusted for damage_reduction tools, NOT adjusted for coin flips. The face-value form is deliberately under-powered to keep parity surface tiny.
- If `opp.active.attacks` is empty (rare; should not happen with a non-null active), bit is 0.
- `opp_lethal_next_turn` is the symmetric bit (own.active.attacks → opp.active HP).

#### Channel 5 — Secondary attack readiness + KO

Per side, 2 bits:
- `own_secondary_attack_usable`: `len(own.active.attacks) ≥ 2 AND energy_attached(own.active) covers attacks[1].energy_cost`. "Covers" = the same logic the engine uses for attack-legality on `attacks[0]` (multiset matching, colorless tokens consume any color). If `attacks[1]` does not exist, bit is 0.
- `own_secondary_attack_would_KO`: `own_secondary_attack_usable AND attacks[1].base_damage ≥ opp.active.hp − opp.active.damage_counters`. Same face-value rule as Channel 4.
- `opp_secondary_attack_usable` and `opp_secondary_attack_would_KO` are the symmetric bits.

**Locked exclusions** (these parity-fragile derivations stay OUT of v3.6):
- Weakness multipliers (2× damage if type matches weakness). Deferred — touches per-Uma weakness lookup; admit in v3.7 if v3.6 lifts.
- Coin-flip expected damage (`damage * coin_heads_count` style attacks). Deferred — requires probabilistic expectation, parity-fragile.
- Conditional damage bonuses (extra damage per attached energy, per benched Uma, etc.). Deferred — touches per-attack effect parsing.
- Energy ETA (turns-until-attack-X-becomes-usable). Deferred — multi-step lookahead.
- Lethal-in-N for N > 1. Deferred — search-tree shape, not a feature.

---

## 5. Ablation plan

**Recipe baseline:** v3.5 extended-training recipe (8 iters × 240 selfplay-games, the recipe that produced the §4j v3.5 wl=0.5908). Same recipe parameters except for the state-feature schema bump.

**Arms:**
- **A0** (baseline): v3.5-extended retrain — REUSE the §4j v3.5 extended wl=0.5908 measurement at n=10k. NO new training. Manifest at `runs/<v3.5-extended-best>/gate.manifest.json` IS the A0 reference.
- **A1** (full bundle): v3.6 retrain (v3.5 head + −10/+44 priors-and-arithmetic tail). Built from `make_v36_priors_init.py` applied to v3.5 lineage best. Iter-0 drift contract verified (Δlogits ≤ 1e-3). Then run v3.5-extended recipe mirror at state_dim=246, 5-input ONNX.
- **A2-A6 (CONDITIONAL):** fire ONLY if A1 lands in the regression band (wl_lower(A1) < 0.5778). Each strips one channel from A1 to localize the drag:
  - A2: v3.6 minus energy_pool (state_dim 246 but pool slots zero-masked) — tests H2 directly.
  - A3: v3.6 minus prize one-hot — tests if non-linear urgency overlaps `points/3` ratio.
  - A4: v3.6 minus bench typed energy — tests if bench-aggregate overlaps the slot-token channel v3.4 falsified.
  - A5: v3.6 minus lethal-next-turn — tests H3-arithmetic-poison hypothesis.
  - A6: v3.6 minus secondary attack — tests if `attacks[1]` is corpus-rare and the bit is sample-dead.

**Sample size:** n=10k tight gate for A1 (mirrors §4j extended cadence). Smoke at n=240 first as sanity. A0 reference already at n=10k.

**Decision rules:**
1. **Hard ship (lift band):** wl_lower(A1) ≥ 0.6040 → v3.6 promoted as the new schema candidate, decisively clears v3.0 upper bound 0.6004. Production-pin discussion.
2. **Soft ship (non-regression):** 0.5778 ≤ wl_lower(A1) < 0.6040 → v3.6 ships as the new state-vector schema; v3.5 retires the way v3.3 did post-§4j. Next slice scoped against v3.6.
3. **Hard fail:** wl_lower(A1) < 0.5778 → fire A2-A6 to decompose. If a single arm clears 0.5778 cleanly, drop the offending channel and re-fire with the 5-item bundle. If all arms regress, falsify H3 (parameter saturation OR arithmetic-bug-poisoning) and pause this slice. Pivot to history-features axis (R7.b.4, untested under loop).

**Estimated wall:** v3.5-extended recipe wall is ~80-90 min for the 8×240 loop. Plus n=10k tight-gate ~11 min at v3.6's projected 5-input throughput (similar to v3.5's). **Total: ~95-105 min wall for A1.** A2-A6 budget: ~9 hours wall if all five fire (so consider firing A2 + A5 first, the two highest-prior diagnostic arms).

---

## 6. Out of scope (explicit)

The handoff doc §2 still has items that v3.6 does NOT touch. Deferred:

- **Weakness-adjusted KO threats.** Listed in early v3.6 brainstorm but cut at §4.5 due to per-Uma weakness lookup parity surface. Admit in v3.7 if v3.6 lifts.
- **Tool effect-kind one-hot.** Promising (4-bit one-hot per tool slot × 2 sides = 8 bits) but requires catalog lookup on the featurization path — parity-fragile against future catalog additions. Deferred.
- **Active ability-kind one-hot.** Same reason as above (~18 ability-kind toggles per active slot).
- **Energy_zone depth 2-3.** v3.5 only encoded depth-1 and trimmed in v3.6. Depth-2+ remains unexplored but has the same "structurally dead on opp" problem and is deprioritised relative to energy_pool.
- **History/sequence features (R7.b.4).** Different axis (sequence model). Listed as "next axis to consider if v3.6 fails" in §5 escalation.
- **Action-vector items** (attach-energy type, trainer effect magnitudes, retreat cost, evolution deltas). All on the action axis, separate scoping doc.
- **Species embedding per Uma.** Card vocab embedding axis, not state-feature axis.
- **Slot tokens** (per-Uma 23-d row). EXPLICITLY EXCLUDED. Falsified at v3.4 and the v3.4 lesson still holds.

---

## 7. Constraints honored

Per `v33-feature-gap-brainstorm-handoff.md` §6 + `v35-multichannel-tail-scoping.md` §7:

- **No opponent hand IDs.** All v3.6 channels read from public observation. `energy_pool` is public per game setup (`flow/setup.rs:143-168`).
- **No raw turn-stamps.** Prize one-hot derives from `points_taken`, a public counter.
- **No ability-name strings.** All channels use fixed vocabularies (energy types: 10, prize counts: 4) or boolean predicates.
- **v3.5 byte freeze** (slots [0:197]) preserved. v3.6 zero-overwrites [197:207] (the dead opp.energy_zone band) and appends at [212:246].
- **`_SCHEMA_BY_STATE_DIM` fail-fast dispatch** preserved — new entry, no silent fallback.
- **Init parity / near-zero-init residual** — `make_v36_priors_init.py` ensures Δlogits ≤ 1e-3 iter-0 vs v3.5 (looser than v3.5's 1e-5 because of the column-drop; justified by the §4j Diagnostic B activation-rate proof that the dropped band contributes ≈0).
- **Action-schema-version guard** (commit `6a7dd84`) — unaffected; action schema stays at 3.

---

## 8. Conditional escalation paths

**If A1 lands in the lift band (rule 1, wl_lower ≥ 0.6040):**
- v3.6 becomes the strongest schema candidate ever measured; promotion-ready against the 96-d production pin.
- Promotion still requires: (a) deck-pair-sampling Slice 3 verdict on v3.6, (b) Rust observation builder confirmed emitting state_dim=246 in self-play binary, (c) export-onnx parity smoke pass, (d) `PublicSideView.energy_pool` extension confirmed populated correctly across all setup paths.
- Open the v3.7 scoping: weakness-adjusted KO + tool/ability-kind one-hot, with the v3.6 parity discipline as the template.

**If A1 lands in non-regression band but not lift (rule 2):**
- v3.6 ships as the new schema baseline.
- Next candidate: choose between (a) admit weakness-adjusted arithmetic (low-cost on top of v3.6's lethal-next-turn infrastructure) or (b) tool/ability-kind one-hot (different axis, needs catalog-lookup harness).

**If A1 fails (rule 3):**
- Fire A2 (energy_pool isolated) and A5 (arithmetic isolated) FIRST — highest-prior diagnostic arms. Each ~95 min wall.
- If A2 lifts cleanly (energy_pool was the gain, the other 4 channels are net-negative): drop the other 4, ship a v3.6.1 with just trim + energy_pool (+10 bits net).
- If A5 lifts cleanly (arithmetic was the gain): re-scope a structural-only v3.6.0 dropping the arithmetic; the arithmetic admission was premature.
- If neither A2 nor A5 clears the gate, fall through to A3/A4/A6 to localize before pausing the additive-tail strategy on top of v3.5.
- **Fallback axis:** history/sequence features (R7.b.4) — never tested under loop, lives at a different representational axis (recurrence/attention over action history), not contaminated by the additive-tail data-limitedness pattern.

**Comparison anchors at decision time:**
- v3.0: wl_lower 0.5811.
- v3.3 (4-iter): wl_lower 0.5910 (peak, original sweet-spot).
- v3.3 (extended, 8×240): wl_lower 0.5812 (collapsed).
- v3.5 (4-iter): wl_lower 0.5856 (original verdict).
- v3.5 (extended, 8×240): wl_lower 0.5908 (current best at extended budget, single seed).
- v3.6 lift target: wl_lower ≥ 0.6040.

---

## 9. The bet (explicit framing)

User-stated framing: "im betting that the features will be needed even though we haven't proved that previous thick addition provides value."

What this scoping doc commits to:
1. v3.5 has soft single-seed evidence at extended training. Replication is still pending and is NOT a prerequisite for v3.6. The bet on v3.6 is **on top of an unreplicated v3.5 lift** by deliberate choice.
2. v3.6 is designed so that even if v3.5's lift is noise (the replication fails), v3.6 still tests a coherent feature-design hypothesis: trim what diagnostics proved dead, add what gap analysis proved unencoded. The result is interpretable independently of whether v3.5 stands.
3. If v3.6 also fails the lift band, the case for further additive-tail iteration on the v3.3 trunk becomes weak. Next pivot is to a different axis (history/sequence, or back to the synthesised-arithmetic axis with stricter per-bit scoping, or set-attention architecture).

The implementer should land code + smoke + init-parity ckpt and queue A1, then **wait for user approval before firing A1**. Same cadence as v3.5.

---

## 10. Canonical file pointers

### Feature code
- `training/uma_ai/features.py:562` — `observation_to_features_v3_5` (template for layered builder).
- `training/uma_ai/features.py:493` — `_UMA_SLOT_ENERGY_TYPES` (10 energy types → energy_pool / bench typed vocab).
- `engine-rs/crates/engine/src/policy/featurize.rs:440` — `observation_state_features_v3_5` (template for Rust mirror).
- `engine-rs/crates/engine/src/policy/types.rs:121` — `PublicObservation` (locate `PublicSideObservation` for energy_pool extension).
- `engine-rs/crates/engine/src/flow/ai/public_info.rs:13` — `PublicSideView` (Rust struct for energy_pool extension).
- `engine-rs/crates/engine/src/core/state.rs:104` — `SideState.energy_pool` source of truth.
- `engine-rs/crates/engine/src/flow/setup.rs:143` — `energy_pool` set at game setup (proof of public-from-start semantics).
- `engine-rs/crates/engine/src/core/state.rs:46` — `UmamusumeInstance` (hp, damage_counters, attacks fields).
- `engine-rs/crates/engine/src/core/effects.rs:133` — `Attack` (base_damage, energy_cost fields for lethal/secondary predicates).
- `training/make_v35_tail_init.py` — template for `make_v36_priors_init.py` (44-column zero-init + 10-column drop).
- `training/serve_onnx.py` — adds dispatch entry for state_dim=246.

### Research docs
- `docs/ai-research/scoping/v33-feature-gap-brainstorm-handoff.md` §2.A/§2.C — gap inventory.
- `docs/ai-research/scoping/v35-multichannel-tail-scoping.md` — predecessor (LANDED-EXTENDED-OUTPERFORMS-CONTROL).
- `docs/ai-research/progress/r110.md §4i` — v3.5 initial falsification (4-iter, since reversed).
- `docs/ai-research/progress/r110.md §4j` — v3.5 extended-training reversal + column-norm + activation-rate diagnostics (the source of the dead-bit finding).

### State files
- `docs/ai-agent-state/queue.json` — A1 queue entry to add when this scope flips to APPROVED.
- `docs/ai-research-backlog.md` — top-of-frontier update when v3.6 ships.

---

## 11. What this doc is NOT

- Not a verdict on any v3.6 channel. The six channels are pre-registered hypotheses with falsifiable bands.
- Not authorization to fire A1. User-gated. Implementer lands code + smoke + init-parity ckpt, then queues A1 and waits for user approval.
- Not a replication of v3.5 single-seed result. v3.5 replication is a separate concern, parallel to v3.6.
- Not a defense of the additive-tail strategy. If v3.6 also fails the lift band, the next pivot is off-axis (history, architecture, or arithmetic with stricter scoping).
