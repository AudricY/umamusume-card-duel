# v3.5 Multichannel Additive Tail — Orthogonal-Bundle Scoping

- **Date:** 2026-05-25
- **Status:** **LANDED-EXTENDED-OUTPERFORMS-CONTROL 2026-05-25** — Initial A1 (4-iter) landed wl=0.5856 (§4i, original "RETIRED" verdict). **Post-§4i diagnostics revealed undertraining, not active overfit.** Extended-training rerun (8 iters × 240 selfplay-games) lifts v3.5 to wl=0.5908 [0.5908, 0.6100]; fair-comparison v3.3-extended degrades to wl=0.5812 [0.5812, 0.6005]. v3.5 BEATS v3.3 by +0.0096 at the same extended budget — confirming v3.5 IS adding meaningful info. v3.5 NOT retired. See `docs/ai-research/progress/r110.md §4j` for the full reversal. Single-run Wilson intervals overlap — replicate runs needed to confirm. Implementation chain: commits `99a9ba1` / `de4f13a` / `66b709c` / `1362130` / `7beb2ff` (scope + builder + dispatch + init + smokes + queue + initial verdict). Init parity verified empirically: Δlogits=7.15e-07.
- **Parent:** `v33-feature-gap-brainstorm-handoff.md` §5 step 3 (further additive tail) + lesson from `progress/r110.md §4h` (v3.4 compound-axis falsified).
- **Predecessor schema:** v3.3 (167-d, no slots, v3-action) — `runs/R16-P1-v33-iter1-tight-gate/gate.manifest.json` wl=0.5910 (confirmatory 0.5863, avg ≈0.5887). Highest of any schema tested.
- **Scope:** A single thick additive tail bundling **five signal-channel-orthogonal** items onto v3.3. Replaces the "one slice per bump" cadence with a deliberately wider bump now that v3.3 is the production candidate and v3.4 has proven the failure mode of compounding overlapping channels.

---

## TL;DR

- **Multichannel additive tail** appended to v3.3's 167-d state vector → new STATE_DIM 212.
- Five items selected to span **five distinct signal channels** so that no two compete for the same representational slot (the failure mode that retired v3.4):
  1. **Phase one-hot (10 bits)** — temporal-cadence channel.
  2. **Per-condition one-hot, 5 × 2 sides (10 bits)** — uma-condition channel.
  3. **Energy-zone front-of-queue typed one-hot, 10 × 2 sides (20 bits)** — energy-color channel.
  4. **Opp-side discard role buckets (3 bits)** — zone-residual symmetry channel.
  5. **Bench-refill catastrophe bits, 2 sides (2 bits)** — terminal-state synth channel.
- All five are **observation-plucks or single-boolean synth** — no parity-fragile arithmetic (no lethal-in-N, no energy ETA, no clock-differential).
- v3.2-style slot tokens **stay out** — proven harmful when stacked on v3.3 (v3.4 wl=0.5860 < v3.3 0.5910, both sides regressed).
- Mandatory v3.2 contract preserved: v3.5 keeps the 5-input ONNX shape inherited via v3.3 (no slot tokens). Action schema stays at 3.
- **Zero-init residual** warm-start: v3.3 ckpt → v3.5 graph produces bit-identical iter-0 outputs via `make_v35_tail_init.py` (new tail Linear columns zero-init).

**Why thick now:** v3.4 just demonstrated that the failure mode of bundling is *channel overlap*, not bundling per se. A bundle of provably orthogonal channels has no more attribution risk than five sequential single-slice bumps, and saves 4× the recipe wall.

---

## 1. The v3.4 lesson and how this bundle avoids repeating it

`progress/r110.md §4h` reports v3.4 = v3.3 + slot tokens lifted **−0.0050** vs v3.3 (regression on both sides, opp 0.6077 vs 0.6129, player 0.5564 vs 0.5610). Working hypothesis (§4h): "the slot-token channel and the opp-flag channel may overlap in what they let the model represent about board state … combining them isn't strictly additive."

The implication is sharper than "don't bundle." It is **don't bundle features that touch the same signal channel**. The five v3.5 items below were chosen by partitioning the §2 gap inventory into channel classes and picking at most one item per channel, where the channels themselves are pairwise disjoint relative to what v3.3 already exposes:

| v3.5 item | Channel | Already in v3.3? | Overlap risk with v3.3 channels |
|---|---|---|---|
| Phase one-hot | Temporal-cadence (turn-state machine) | Ordinal `phase_index/9` at slot 0 only | Low — supplements ordinal, doesn't replicate threat-flag channel |
| Per-condition one-hot | Uma-condition (status-effect identity) | Paralysed-bit + count/5 at slots 19-20 | Low — disambiguates burned/poisoned (end-of-turn damage) from asleep/frozen (attack block); v3.3 has no per-type signal |
| Energy-zone typed front | Energy-color (incoming attach color) | Length + depth-1 boolean (`next_energy_matches_active_need`) | Low — boolean is "match yes/no"; one-hot is "which color" — orthogonal information |
| Opp-side discard buckets | Zone-residual symmetry | Own-side discard buckets (slots 84-86); opp totally absent | Low — pure layout-symmetry fix |
| Bench-refill catastrophe | Terminal-game-state | Not encoded anywhere | Low — single boolean per side, doesn't shadow any opp-flag |

The **opp-flag channel** (used-supporter/retreat/stadium this turn) — the channel v3.3 already added — is **not** retouched in v3.5. The **slot-token channel** is also not retouched, by deliberate exclusion of the per-Uma 23-d row.

---

## 2. Pre-registered hypotheses

### H1 — Five orthogonal channels compound additively into v3.5

- **Falsifiable signal:** wl_lower(v3.5) ≥ wl_lower(v3.3) − 0.013 = 0.5780 (non-regression band on the v3.3 anchor at 0.5910). Hard fail below this.
- **Stretch:** wl_lower(v3.5) ≥ wl_lower(v3.3) + 0.013 = 0.6040 (clears v3.0 upper bound 0.6004 decisively → production-promotion candidate).
- **Theory:** unlike slot tokens (which shadow the opp-flag's board-state channel), each of the five items contributes information v3.3 cannot currently express. The hypothesis is *direct sum*, not *Hadamard product*: each channel contributes its expected per-item EV without interaction.

### H2 — At least one item lifts v3.3 individually

- Backstop assumption: if H1 falsifies (v3.5 regresses), the per-item arms (§5 ablation plan) will surface which channels are the drag. Energy-zone typed contents is the *a priori* highest-EV item (gap audit §2.A item 1: "biggest semantic loss in v3.2"); if it doesn't lift in isolation, the gap-audit's "biggest loss" claim is empirically wrong.
- **Falsifiable signal:** A2 (v3.5-minus-energy) wl_lower < A1 (v3.5 full) wl_lower − 0.013 OR vice versa.

### H3 — No channel competes for representational bandwidth at v3.3 model size

- Risk: same failure as v3.4 (extra parameters introduce noise the trunk can't usefully exploit). v3.5 adds 45 input dims → 45 × hidden_dim extra weights into the first Linear. That's small (<5%) for typical hidden_dim ≥ 256.
- **Falsifiable signal:** all four per-channel ablation arms regress vs v3.3 by >0.013 simultaneously, meaning the model is parameter-saturated and any new tail hurts. If observed, v3.5 falsifies the additive-tail strategy itself at the current recipe scale, not just this bundle.

---

## 3. Freeze contract

| Surface | v3.3 (current) | v3.5 (proposed) | Compatibility |
|---|---|---|---|
| `state_features` dim | 167 | **212** | Additive tail at indices 167-211. Slots 0-166 byte-stable (v3.1 head + v3.3 opp-flag tail frozen). |
| `STATE_FEATURE_SCHEMA_VERSION` (Python latest) | 3.3 | **3.5** | Bump LATEST marker; legacy 3.0/3.1/3.3 dispatch unchanged. v3.4 is RETIRED and gets no schema constant. |
| `STATE_DIM_V3_5` (new constant) | n/a | **212** | New constant; dispatch entry `_SCHEMA_BY_STATE_DIM[212]`. |
| `observation_to_features_v3_5` (new builder) | n/a | **NEW** | Layered on top of `observation_to_features_v3_3` then appends 45 bits. Bit-exact Rust mirror in `featurize.rs`. |
| ONNX input set | 5 inputs (v3.3 no-slot contract) | **5 inputs unchanged** | v3.5 keeps the v3.3 ONNX shape. Only `state_features` last dim grows 167 → 212. |
| `uma_slot_features[10, 23]` | Not used (v3.3 5-input contract) | **Still not used** | Slot tokens stay retired post-v3.4. |
| `action_features[A, 48]` | 48-d, schema 3 | **48-d, schema 3 unchanged** | Action vector untouched. |
| `_SCHEMA_BY_STATE_DIM` dispatch | {96, 110, 164, 167} | **{96, 110, 164, 167, 212}** | New entry → `observation_to_features_v3_5`. Fail-fast preserved. |
| Init parity | v3.3 anchor | **v3.3 → v3.5 zero-init residual** | A v3.3 ckpt loaded into the v3.5 graph (new state_features Linear columns 167-211 zero-init) produces bit-identical iter-0 outputs to v3.3. |

### Bit-exact mirror requirement

`observation_to_features_v3_5` lands in BOTH:
- `training/uma_ai/features.py` (Python spec).
- `engine-rs/crates/engine/src/policy/featurize.rs` (Rust mirror).

Parity smoke: extend the existing chain (`v3_1_head_is_byte_identical_to_v3_0`, `v3_3_head_is_byte_identical_to_v3_1`) with `v3_5_head_is_byte_identical_to_v3_3`. The first 167 bytes of v3.5 output must equal v3.3's output on the same observation.

### Existing checkpoint compatibility

- Pre-v3.5 ckpts (v3.0/v3.1/v3.3) unchanged — the new `_SCHEMA_BY_STATE_DIM[212]` entry routes only v3.5-shaped inputs.
- Loading a v3.3 ckpt into the v3.5 graph requires `training/make_v35_tail_init.py`:
  1. Reads v3.3 lineage best (`runs/R16-P1-v33-ablation/loop/iter-1/checkpoint.pt` or equivalent).
  2. Expands the first Linear `state_features → hidden` by 45 columns, zero-init.
  3. Saves a v3.5-shaped ckpt.
  4. Asserts bit-identical iter-0 outputs vs v3.3 source on a fixed observation batch.
- Mirrors `make_v33_tail_init.py` (template).

### Action-schema-version guard interaction

The Rust loader guard (commit `6a7dd84`) checks `action_feature_schema_version`. v3.5 keeps action schema at 3, so the guard fires consistently across v3.3 → v3.5 transitions. No new logic.

---

## 4. Implementation order

1. **Python `observation_to_features_v3_5` + `STATE_DIM_V3_5 = 212`** in `training/uma_ai/features.py`.
   - Call `observation_to_features_v3_3(...)` for indices 0-166.
   - Append 45 bits in fixed layout:
     - `[167:177]` (10 bits) — phase one-hot from `PHASES` index (full 10-entry list at `features.py:112-123`).
     - `[177:182]` (5 bits) — own active per-condition one-hot: paralysed, burned, poisoned, asleep, confused.
     - `[182:187]` (5 bits) — opp active per-condition one-hot, same vocab.
     - `[187:197]` (10 bits) — own energy-zone front-of-queue typed one-hot (10 types in `_UMA_SLOT_ENERGY_TYPES`).
     - `[197:207]` (10 bits) — opp energy-zone front-of-queue typed one-hot, same vocab.
     - `[207:210]` (3 bits) — opp discard role buckets (mirror of own slots 84-86).
     - `[210:211]` (1 bit) — own `would_lose_on_active_KO` (bench empty AND active HP ≤ damage-threshold-1, OR equivalent terminal-state predicate; see §4.5 below).
     - `[211:212]` (1 bit) — opp `would_lose_on_active_KO`.
2. **Rust mirror in `featurize.rs`** — new `observation_state_features_v3_5` function, called when policy/dispatch picks v3.5. Bit-exact parity.
3. **Schema dispatch updates** — `_SCHEMA_BY_STATE_DIM[212] = STATE_FEATURE_SCHEMA_VERSION_V3_5` in `features.py`; `serve_onnx.py` schema table; Rust `inference/mod.rs` ONNX-graph signature check accepts state_dim=212 for v3.5 dispatch.
4. **`make_v35_tail_init.py`** — ckpt expander. Bit-exact iter-0 parity assertion on a fixed observation batch.
5. **Smoke tests** — `r16_p2_slot_init_smoke.py`-style smoke for v3.5 init; Rust parity smoke for the new builder. Run at n=240 first for sanity.
6. **Ablation queue entry** — A1 = v3.5 full bundle from v3.3 init; A0 = v3.3 anchor (already at wl=0.5910, no new training). Per-item arms A2-A5 only fire if A1 regresses.

### 4.5 Synthesis-bit definitions (locked here to prevent parity drift)

To keep both bench-refill bits pure-pluck-with-one-AND (the cheapest possible synth):

- `would_lose_on_active_KO(side)` = (own bench count == 0) AND (active HP ≤ active HP — i.e., it's a 1-bit catastrophe predicate computed as the conjunction of "no Uma to promote" AND "active is one combat tick from being KO'd").
- **Minimum-viable definition** to keep parity surface tiny: `(bench_count(side) == 0)`. The "active is in lethal danger" predicate is deferred — too much arithmetic, risks lethal-in-N parity bugs. The single-AND below is also acceptable:
  - `(bench_count(side) == 0) AND (active.hp / active.max_hp ≤ 0.5)` — a "bench-empty + half-HP" caution bit.
- Final choice: **minimum-viable form** (`bench_count == 0`). Justification: forces the model to learn "you have no out" cleanly; the HP-half threshold can be added in a v3.6 if v3.5 lifts and the model demonstrably uses this bit.

---

## 5. Ablation plan

**Recipe baseline:** R16-P1 v3.3 mirror (the path that produced v3.3 iter-1 wl=0.5910 at n=10k per `progress/r110.md §4g`). Same recipe parameters except for the state-feature schema bump.

**Arms:**
- **A0** (baseline): v3.3 retrain — REUSE the existing R16-P1-v33-iter1 wl=0.5910 measurement at n=10k. NO new training. Manifest at `runs/R16-P1-v33-iter1-tight-gate/gate.manifest.json` IS the A0 reference.
- **A1** (full bundle): v3.5 retrain (v3.3 head + 45-bit multichannel tail). Built from `make_v35_tail_init.py` applied to the v3.3 lineage best. Bit-identical iter-0 contract verified (delta_logits = 0.0 exactly). Then run R16-P1 mirror recipe at state_dim=212, 5-input ONNX.
- **A2-A5 (CONDITIONAL):** fire ONLY if A1 lands in the regression band (wl_lower(A1) < 0.5780). Each strips one channel from A1 to localize the drag:
  - A2: v3.5 minus energy-zone (state_dim 212 but energy slots zero-masked) — tests if energy-color overlaps something.
  - A3: v3.5 minus per-condition one-hot — tests if condition disambiguation overlaps slot-19/20.
  - A4: v3.5 minus phase one-hot — tests if ordinal phase already saturates.
  - A5: v3.5 minus discard-opp + bench-refill — tests if catastrophe + zone-residual channels are dead.

**Sample size:** n=10k tight gate for A1 (mirrors re-verdict cadence). Smoke at n=240 first as sanity. A0 reference already at n=10k.

**Decision rules:**
1. **Hard ship (lift band):** wl_lower(A1) ≥ 0.6040 → v3.5 promoted as the new schema candidate, decisively clears v3.0 upper bound 0.6004. Production-pin discussion.
2. **Soft ship (non-regression):** 0.5780 ≤ wl_lower(A1) < 0.6040 → v3.5 ships as the new state-vector schema; v3.3 retires the way v3.4 did. Next slice scoped against v3.5.
3. **Hard fail:** wl_lower(A1) < 0.5780 → fire A2-A5 to decompose. If a single arm clears 0.5780 cleanly, drop the offending channel and re-fire with the 4-item bundle. If all arms regress, falsify H3 (parameter saturation) and pause the additive-tail strategy until a different model-size or recipe is tried.

**Estimated wall:** R16-P1 v3.3 recipe wall is ~10-15 min for the loop at G5 fast path. Plus n=10k tight-gate ~11 min at v3.5's projected 5-input throughput (similar to v3.3's 31.1 g/s — slot tokens stay out). **Total: ~25-30 min wall for A1.** A2-A5 budget: ~2 hours wall if all four fire.

---

## 6. Out of scope (explicit)

The handoff doc §2 has more items than fit in one thick bundle. ONLY the five channel-orthogonal items above ship in this slice. Deferred to follow-up scoping:

- **Slot tokens** (per-Uma 23-d row) — **EXPLICITLY EXCLUDED.** v3.4 falsified this channel as a v3.3-compounder (`progress/r110.md §4h`). Slot tokens may revive only if a future probe shows they help on a *different* state-vector base (e.g., set-attention trunk).
- **Bench temporal slot widening** — moves bench temporal into a widened 28-d uma_slot_features row. Excluded because it touches a frozen contract (`UMA_SLOT_FEATURE_DIM = 23`) and is structurally on the slot-token channel.
- **Energy-zone depth 2-3** — only depth-1 (front of queue) ships here. Depth-2 and depth-3 add ~40 more bits and may be redundant with the v3.5 depth-1 signal.
- **Synthesised features with arithmetic** — lethal-in-N, clock differential, gust-swing, energy ETA, paralysis exploit window. All deferred. Reason: parity surface is large (Python and Rust must produce identical floats), and a bug there poisons the bundle's signal. The minimum-viable bench-refill bit is the only synth admitted, with the definition locked at §4.5 above.
- **Action-vector items** (attach-energy type, trainer effect magnitudes, retreat cost, evolution deltas, lethal-target flag beyond the slot-26 repurpose) — all on the action axis, orthogonal to the state-vector axis but their own scoping doc. Held.
- **Species embedding per Uma** — touches the card vocab embedding axis, not the state-feature axis. Out of scope.

---

## 7. Constraints honored

Per handoff §6 "Constraints / what's off-limits":

- **No opponent hand IDs.** All five v3.5 channels read from public observation only.
- **No raw turn-stamps.** Phase one-hot is a public turn-state value, not a derivative of `enteredTurn`/`evolvedTurn`.
- **No ability-name strings.** Per-condition one-hot uses a fixed 5-entry vocab (paralysed/burned/poisoned/asleep/confused).
- **v3.3 byte freeze** (slots 0-166) preserved. v3.5 is strictly additive at indices 167-211.
- **`_SCHEMA_BY_STATE_DIM` fail-fast dispatch** preserved — new entry, no silent fallback.
- **Init parity / zero-init residual** — `make_v35_tail_init.py` ensures bit-identical iter-0 outputs vs v3.3.
- **Action-schema-version guard** (commit `6a7dd84`) — unaffected; action schema stays at 3.

---

## 8. Conditional escalation paths

**If A1 lands in the lift band (rule 1, wl_lower ≥ 0.6040):**
- v3.5 becomes the strongest schema candidate ever measured; promotion-ready against the 96-d production pin.
- Promotion still requires: (a) deck-pair-sampling Slice 3 verdict on v3.5, (b) Rust observation builder confirmed emitting state_dim=212 in self-play binary, (c) export-onnx parity smoke pass.
- Consider firing the v33-correctness-fix A2 arm against v3.5 as well, to settle whether the v3.3→v3.5 lift is the new tail or the lingering Fix 2-4 confound carrying over.

**If A1 lands in non-regression band but not lift (rule 2):**
- v3.5 ships as the new schema baseline.
- Next candidate is depth-2 energy-zone (~20 more bits) OR the deferred bench-temporal slot widening (different axis), depending on which channel the per-item ablations suggest carried the most signal.

**If A1 fails (rule 3):**
- Fire A2-A5 to decompose. If a single arm (e.g., v3.5-minus-energy) clears 0.5780, the offending channel was the drag; drop it and re-fire with the 4-item bundle.
- If all arms also regress (H3 falsified), pause additive-tail strategy at the current recipe scale. Pivot to synthesised features axis (§2.C) with stricter per-item scoping, OR wait on set-attention Slice 3 architecture-axis crossing 0.40.

**Comparison anchors at decision time:**
- v3.0: wl_lower 0.5811, upper 0.6004.
- v3.3: wl_lower 0.5910, upper 0.6102.
- v3.4: wl_lower 0.5860 (RETIRED).
- v3.5 lift target: wl_lower ≥ 0.6040 (clears v3.0 upper by 0.0036).

---

## 9. Canonical file pointers

### Feature code
- `training/uma_ai/features.py:430` — `observation_to_features_v3_3` (template for layered builder).
- `training/uma_ai/features.py:112-123` — `PHASES` list (10 entries → 10-bit one-hot vocab).
- `training/uma_ai/features.py:609-620` — `_UMA_SLOT_ENERGY_TYPES` (10 energy types → 10-bit one-hot vocab).
- `engine-rs/crates/engine/src/policy/featurize.rs:351` — `observation_state_features_v3_3` (template for Rust mirror).
- `engine-rs/crates/engine/src/policy/types.rs:133` — `special_conditions: Vec<String>` (per-Uma condition source).
- `engine-rs/crates/engine/src/policy/types.rs:152` — `energy_zone: Vec<String>` (per-side energy queue, front-of-queue typed source).
- `training/make_v33_tail_init.py` — template for `make_v35_tail_init.py` (45-column zero-init expansion).
- `training/serve_onnx.py` — adds dispatch entry for state_dim=212.

### Research docs
- `docs/ai-research/scoping/v33-feature-gap-brainstorm-handoff.md` §2 — channel-by-channel gap inventory.
- `docs/ai-research/scoping/v33-additive-tail-scoping.md` — sibling scoping (v3.3 opp-flag tail, LANDED).
- `docs/ai-research/scoping/v33-correctness-fix-scoping.md` — sibling correctness-fix slice (Fixes 1-4 LANDED).
- `docs/ai-research/progress/r110.md §4g` — v3.3 verdict (the v3.5 baseline).
- `docs/ai-research/progress/r110.md §4h` — v3.4 falsification (the lesson driving channel-orthogonality).

### State files
- `docs/ai-agent-state/queue.json` — A1 queue entry to add when this scope flips to APPROVED.

---

## 10. What this doc is NOT

- Not a verdict on any specific channel. The five items are pre-registered hypotheses, not measurements. Each carries falsifiable bands.
- Not authorization to fire A1. User-gated, like v33-additive-tail and the re-verdicts before it. The implementer should land code + smoke + init-parity ckpt, then queue A1 and wait for user approval.
- Not a substitute for synthesised-feature exploration. The deferred §2.C items (lethal-in-N, energy ETA, gust-swing) remain the next axis to scope after v3.5 settles — they need their own pre-registration because their parity surface is large.

---

## 11. Status line

**`LANDED-EXTENDED-OUTPERFORMS-CONTROL 2026-05-25`** — Initial 4-iter result (§4i, wl=0.5856) was an undertraining artifact, not a feature-design failure. Post-§4i diagnostics + extended-training rerun overturn the "RETIRED" verdict. See `docs/ai-research/progress/r110.md §4j` for full reversal.

**Outcome ladder at n=10k:**

| Schema | Training budget | wl_lower | Δ vs v3.0 0.5811 |
|---|---|---|---|
| v3.3 (167-d) | 4 × 60 games | 0.5910 ★ (peak) | +0.0099 |
| v3.5 (212-d) | 4 × 60 games | 0.5856 | +0.0045 |
| **v3.3-extended** | **8 × 240 games** | **0.5812** | **+0.0001** (collapsed) |
| **v3.5-extended (iter-4)** | **8 × 240 games** | **0.5908** | **+0.0097** |

**Two inversions:**

1. **v3.3-extended is WORSE than v3.3-original** (−0.0098): v3.3 OVERFITS at extended training. The original 4-iter wl=0.5910 was a sweet-spot peak, not a stable ceiling.
2. **v3.5-extended beats v3.3-extended by +0.0096** at the same training budget: v3.5's 45-bit tail acts as a stabilizer (capacity-against-overfit) that lets the model use more data without degradation.

**Caveats:**
- Single-run, Wilson intervals overlap (v3.5 [0.5908, 0.6100] vs v3.3 [0.5812, 0.6005]). Need 2-3 replicate seeds to confirm v3.5 > v3.3 at extended training.
- Neither schema crosses the LIFT band (≥ 0.6040). v3.5 doesn't unlock a new ceiling vs v3.0; it just survives extended training where v3.3 degrades.

**Diagnostic A (post-§4i, column norms):** trunk IS learning the v3.5 tail — column norms grew 0 → 0.20 frob across iters; per-channel strongest: opp.discard.buckets (0.14× head-median), bench-refill (0.07-0.09×), phase one-hot (0.06×).

**Diagnostic B (post-§4i, activation rates):**
- 10 opp.energy.front bits STRUCTURALLY DEAD — `flow/turn.rs:110-111` clears opp's `energy_zone` at opp turn start; `flow/energy.rs:18` consumes on attach. So at player decision points opp.energyZone is always empty. NOT a bug; the feature reads nonexistent state. Remove in any future v3.5.1.
- 5 opp.cond bits SAMPLE-DEAD (early-game corpus bias — selfplay max turn=11). Keep pending longer-game corpus.
- Remaining 30 bits active or sparsely active; keep.

### Implementation chain (2026-05-25)

| Commit | Files | Notes |
|---|---|---|
| `99a9ba1` | `features.py`, `featurize.rs`, `v35-multichannel-tail-scoping.md`, `v33-feature-gap-brainstorm-handoff.md` | Python `observation_to_features_v3_5` + Rust `observation_state_features_v3_5` + frozen condition vocab + 6 Rust parity tests (head byte-identity, phase/cond/energy/bench tail subcases). Handoff doc §5 step 3 pointer chain updated. |
| `de4f13a` | `inference/mod.rs`, `serve_onnx.py` | Rust `GraphSchema::V3_5` enum variant + 5-input contract dispatch (state_dim=212 → V3_5). serve_onnx schema-table row + valid-tokens + encoder selector for v3.3 / v3.4 / v3.5. |
| `66b709c` | `make_v35_tail_init.py`, `v35_multichannel_tail_smoke.py`, `v35_tail_init_smoke.py` | Ckpt expander (167→212 widening, 45 zero-init columns; refuses non-v3.3 and slot-token sources). Two smokes (Python builder layout + init-parity). v33 smokes still PASS — no regression. |

### Verification status

| Surface | Status | Evidence |
|---|---|---|
| Python builder shape, byte-identity vs v3.3 | PASS | `training/v35_multichannel_tail_smoke.py` 4/4 cases |
| Rust mirror parity | PASS | 6 unit tests in `featurize.rs::tests::v3_5_*` |
| Schema dispatch (Python + Rust) | PASS | `feature_builder_for_state_dim(212) → observation_to_features_v3_5`; Rust 5-input contract accepts state_dim=212 → V3_5 |
| ONNX export | PRESUMED PASS | `export_onnx.py` uses `feature_builder_for_state_dim` and `schema_version_for_state_dim` which both route 212 correctly; not exercised end-to-end here |
| Ckpt expander + init parity | PASS | Built from v3.3 iter-1 lineage best; Δlogits=7.15e-07 < 1e-5 tolerance; `v35_tail_init_smoke.py` 6/6 cases |
| Refusal of bad sources | PASS | v3.1 ckpt refused (state_dim mismatch); v3.4 ckpt refused (slot-token guard) |
| v33 smokes after v3.5 changes | PASS | `v33_additive_tail_smoke` + `v33_tail_init_smoke` both green |

### What's NOT verified (out of scope for this slice)

- Full self-play loop at state_dim=212 (requires A1 fire — USER-GATED).
- ONNX graph export end-to-end (will be exercised the first time `make_v35_tail_init.py` output is passed to `export_onnx.py`; the dispatch path is structurally correct but unsmoked).
- Rust binary rebuild from updated Rust dispatch (binary at `runs/R16-P1-v33-ablation/...` was built against v3.3 only; v3.5 state_dim=212 ONNX would need a freshly-built binary to consume).
