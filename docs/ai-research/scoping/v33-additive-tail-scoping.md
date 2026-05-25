# v3.3 Additive Tail — Opp-Side Flags Slice

- **Date:** 2026-05-25
- **Status:** SCOPING — pre-registered, not yet launched. Justification gate: re-verdict #3 outcome (v3.2 wl=0.5877 at n=10k, +0.0066 over v3.0 0.5811) cleared the handoff §5 step 3 conditional ("if schema axis shows movement").
- **Parent:** `v33-feature-gap-brainstorm-handoff.md` §5 step 3 (additive v3.3 tail). Sibling: `v33-correctness-fix-scoping.md` (Fixes 1-4 LANDED 2026-05-25). Promoted from "deferred" to "scoped" by the re-verdict #3 result.
- **Scope:** Smallest defensible additive tail — three opp-side flags. Other §2.A candidates (phase one-hot, energy-zone typed contents, per-condition one-hot, bench-temporal slot widening) are explicitly **out of scope** for this slice; each gets its own scoping pass if v3.3 step-1 holds non-regression.

## TL;DR

- **Three-bit additive tail** appended to the v3.1/v3.2 164-d state vector → new STATE_DIM 167.
- New slots: `opp.usedSupporterThisTurn`, `opp.usedRetreatThisTurn`, `opp.usedStadiumThisTurn` (booleans normalized 0/1).
- Featurizer currently reads `own.*` for these three at slots 110-163 (per the v3.1 temporal block); the opponent-side mirror was omitted in v3.1 — a layout asymmetry the gap audit flagged in handoff §2.A item 2.
- Hypothesis: opp-side flags give the model threat-awareness ("opp already played a supporter this turn → no more attach/draw bombs coming") and post-attach safety inference. Falsifiable as non-regression vs v3.2 at n=10k; lift target is +0.013 wl_lower (clears noise band).
- Mandatory v3.2 contract preserved: v3.3 keeps slot tokens (7-input ONNX) and the per-Uma 23-d row. Only the scalar state vector widens.
- **Zero-init residual** for warm-start parity: a v3.2 ckpt loaded into v3.3 graph produces bit-identical outputs at iter-0 (the 3 new slots feed through a `Linear(167, hidden)` weight whose new columns are zero-init).

## 1. Pre-registered hypotheses

### H1 — Opp-side flags expose threat-window signal

- **Current state:**
  - `engine-rs/crates/engine/src/policy/types.rs` `PublicSideTurnState` carries both `own` and `opp` side fields (the struct is symmetric).
  - `training/uma_ai/features.py:_observation_to_features_v3_1` emits 54 temporal slots (slots 110-163), and the `usedSupporterThisTurn / usedRetreatThisTurn / usedStadiumThisTurn` slots are populated for **own only**. The opp side is not surfaced.
  - Per handoff doc §2.A item 2: "Opponent `usedSupporterThisTurn` / `usedRetreatThisTurn` / `usedStadiumThisTurn` — featurizer reads `own.*` only. 3 free bits, high-value for threat assessment."
- **Hypothesis:** these three booleans gate a class of opponent threats (supporter-bomb sequences, gust+attach combos, stadium swap) the model currently can't see. Adding them lets the policy head learn "opp already burned X this turn → fewer threats this turn" without needing to infer it from action history.
- **Falsifiable signal:** wl_lower(v3.3) ≥ wl_lower(v3.2) − 0.013 (non-regression band). Stretch: wl_lower(v3.3) ≥ wl_lower(v3.2) + 0.013 (lift past noise).

### H2 — Lift compounds with v3.2's +0.0066

- **Reasoning:** re-verdict #3 established v3.2 sits at wl_lower=0.5877, +0.0066 over v3.0 0.5811. If opp-side flags add another +0.005-0.013, v3.3 could land at wl_lower ≈ 0.59-0.60, **decisively over the v3.0 ceiling** (clears v3.0's upper bound 0.6004). Promotion-ready.
- **Falsifiable signal:** if v3.3 wl_lower < v3.2 wl_lower − 0.013, the additive-tail strategy is falsified for this slice; the gap audit's "3 free bits" claim was wrong about magnitude.

## 2. Freeze contract

| Surface | v3.2 (current) | v3.3 (proposed) | Compatibility |
|---|---|---|---|
| `state_features` dim | 164 | **167** | Additive tail at indices 164-166. Slots 0-163 byte-stable (v3.1/v3.2 head frozen). |
| `STATE_FEATURE_SCHEMA_VERSION` (Python latest) | 3.1 | **3.3** | Bump the LATEST-version marker; legacy 3.0/3.1 dispatch unchanged. |
| `STATE_DIM_V3_3` (new constant) | n/a | **167** | New constant; dispatch entry in `_SCHEMA_BY_STATE_DIM[167]`. |
| `observation_to_features_v3_3` (new builder) | n/a | **NEW** | Python: layered on top of v3.1 (calls `observation_to_features_v3_1` then appends 3 bits). Rust mirror in `featurize.rs`. |
| ONNX input set | 7 inputs (v3.2 slot tokens mandatory) | **7 inputs unchanged** | v3.3 keeps slot tokens. Only the `state_features` tensor's last dim changes 164 → 167. |
| `uma_slot_features[10, 23]` | Frozen | **Unchanged** | Per-Uma slot row stays 23-d. |
| `action_features[A, 48]` | 48-d, schema 3 | **48-d, schema 3 unchanged** | Action vector untouched by this slice. |
| `_SCHEMA_BY_STATE_DIM` dispatch | {96, 110, 164} | **{96, 110, 164, 167}** | New entry routes to `observation_to_features_v3_3`. Fail-fast (no silent fallback) preserved. |
| Init parity | v3.0 ↔ v3.1 ↔ v3.2 | **v3.2 → v3.3 zero-init residual** | A v3.2 ckpt loaded into the v3.3 graph (with new state_features Linear columns zero-init) produces bit-identical iter-0 outputs to v3.2. The v3.1 `delta=0.0` pattern generalizes — explicitly noted in r16-model-feature-backlog-refinement.md §C5 and v33-feature-gap-brainstorm-handoff.md §6. |

### Bit-exact mirror requirement

`observation_to_features_v3_3` lands in BOTH:
- `training/uma_ai/features.py` (Python spec; canonical).
- `engine-rs/crates/engine/src/policy/featurize.rs` (Rust mirror).

Parity smoke: extend the existing v3.1 head-bit-identical test (`v3_1_head_is_byte_identical_to_v3_0`) with a v3.3 variant: `v3_3_head_is_byte_identical_to_v3_1`. The first 164 bytes of v3.3 output must equal v3.1's output on the same observation.

### Existing checkpoint compatibility

- Pre-v3.3 ckpts (v3.0/v3.1/v3.2) are state_dim=110/164. The new `_SCHEMA_BY_STATE_DIM[167]` entry does not affect them.
- Loading a v3.2 ckpt into the v3.3 graph requires a tail-init script `make_v33_tail_init.py` that:
  1. Reads v3.2 `policy.pt`.
  2. Expands the first Linear of `state_features → hidden` by 3 columns, zero-init.
  3. Saves a v3.3-shaped ckpt.
  4. Asserts bit-identical iter-0 outputs vs v3.2.
- This mirrors `make_v32_slot_token_init.py` (R16-P2 C6 — already in repo).

### Action-schema-version guard interaction

The new Rust loader guard (commit `6a7dd84`) checks `action_feature_schema_version`. v3.3 keeps action schema at 3, so the guard fires consistently across v3.2 → v3.3 transitions. No new logic needed.

## 3. Implementation order

1. **Python `observation_to_features_v3_3` + `STATE_DIM_V3_3`** in `training/uma_ai/features.py`. Read `opp.usedSupporterThisTurn / usedRetreatThisTurn / usedStadiumThisTurn` from the same `PublicSideTurnState` already used for the own-side slots. Append 3 booleans (cast to f32) at indices 164-166.
2. **Rust mirror in `featurize.rs`**: new `observation_state_features_v3_3` function, called when the policy/dispatch picks v3.3. Bit-exact parity with Python.
3. **Schema dispatch updates**: `_SCHEMA_BY_STATE_DIM[167] = STATE_FEATURE_SCHEMA_VERSION_V3_3` in features.py; `serve_onnx.py` schema table; Rust `inference/mod.rs` ONNX-graph signature check accepts state_dim=167 for v3.3 dispatch.
4. **`make_v33_tail_init.py`** ckpt expander script. Bit-exact iter-0 parity assertion.
5. **Smoke tests**: `r16_p2_slot_init_smoke.py`-style smoke for v3.3 init; Rust parity smoke for the new builder.
6. **Ablation queue entry**: A0 baseline = v3.2 lineage best at wl=0.5877; A1 = v3.3 (v3.2 + opp-side flag tail) trained from v3.2 init. User-gated firing.

## 4. Ablation plan

**Recipe baseline:** v3.2 uniform-retrain recipe (the path that produced the lineage best wl=0.5877). Same recipe parameters except for the state-feature schema bump.

**Arms:**
- **A0** (baseline): v3.2 retrain (no change). Must reproduce the lineage best wl=0.5877 ± n=10k CI as a sanity check.
- **A1**: v3.3 retrain (v3.2 head + 3-bit opp-side flag tail). Trained from `make_v33_tail_init.py` output (bit-identical to v3.2 init at iter-0).

**Sample size:** n=10k tight gate per arm (mirrors the re-verdict cadence). Smoke at n=240 first as sanity.

**Decision rules:**
1. **Non-regression (ship gate):** wl_lower(A1) ≥ wl_lower(A0) − 0.013.
2. **Lift bonus (promotion case):** wl_lower(A1) ≥ wl_lower(A0) + 0.013 AND wl_lower(A1) ≥ 0.5811 + 0.013 (i.e., clears v3.0 ceiling+upper bound combined). Triggers production-candidacy discussion.
3. **Hard fail:** wl_lower(A1) < wl_lower(A0) − 0.013 falsifies the "free 3 bits" claim — pause additive-tail strategy and re-investigate the gap audit's threat-assessment assumption.

**Estimated wall:** v3.2 uniform-retrain wall was ~5.5 min for 4 iters. v3.3 should match. Plus n=10k tight-gate ~11 min per arm. **Total: ~30-40 min wall** assuming fresh training runs cleanly to a halt-after-2 outcome.

## 5. Out of scope (explicit)

The handoff doc §2.A flagged 5 additional structural fixes as v3.3 tail candidates. ONLY opp-side flags ship in this slice; the others are deferred to follow-up scoping docs:

- **Phase one-hot** (10 bits) — supplement slot 0 ordinal. Deferred.
- **Energy-zone typed contents** (~80 bits both sides, depth 2-3) — biggest single-feature win per handoff §2.A item 1. Deferred — large enough to warrant its own scoping pass.
- **Per-condition one-hot** (5 bits) — replace paralysed-bit + count/5. Deferred.
- **Bench temporal slot widening** — move bench temporal into widened `uma_slot_features` (23 → e.g. 28-d). Deferred — touches a frozen contract (uma_slot_feature_dim).
- **Synthesised features** (lethal-in-N, clock differential, weakness-adjusted damage on damage features, energy ETA, paralysis exploit window) — handoff §2.C. Deferred — each requires synthesis arithmetic and its own falsifiable hypothesis.

Sequencing intent: **ship opp-side flags first as the smallest provable additive-tail item**, then use that signal to decide whether to follow up with phase one-hot, energy-zone typed contents, etc. If opp-side flags alone don't move the needle, the bigger items are also unlikely to pay off without an architecture-axis change.

## 6. Constraints honored

Per handoff §6 "Constraints / what's off-limits":

- **No opponent hand IDs.** Hidden-info regression guard intact — opp-side flags are public observation booleans, not hand contents.
- **No raw turn-stamps.** The flags are booleans, not derivatives of `enteredTurn` or `evolvedTurn`.
- **No ability-name strings.** None added.
- **v3.2 byte freeze** (slots 0-163) preserved. v3.3 is strictly additive at indices 164-166.
- **`_SCHEMA_BY_STATE_DIM` fail-fast dispatch** preserved — new entry, no silent fallback.
- **Init parity / zero-init residual** — `make_v33_tail_init.py` ensures bit-identical iter-0 outputs vs v3.2 (the `delta=0.0` trick from v3.1).
- **Action-schema-version guard** (commit `6a7dd84`) — unaffected; action schema stays at 3.

## 7. Conditional escalation paths

If A1 lands in the lift band (decision rule 2):
- v3.3 becomes the strongest schema-axis candidate; consider production promotion (rolls into a v3.3 production pin discussion).
- Promotion still requires: (a) deck-pair-sampling Slice 3 verdict on v3.3, (b) Rust observation builder extension to emit state_dim=167, (c) v3.3 vocab compatibility check (no change expected, but the export-onnx parity smoke must pass).
- W6-fix HP sweep (`w6-loop-anti-degradation`) should re-baseline against v3.3 if v3.3 lands.

If A1 lands in non-regression band but not lift (decision rule 1, not 2):
- v3.3 ships as the new schema baseline, replacing v3.2.
- Next additive-tail candidate (likely energy-zone typed contents per handoff §2.A item 1) gets scoped against v3.3.

If A1 fails (decision rule 3):
- Pause additive-tail strategy. Pivot to synthesised-features axis (handoff §2.C) or wait on architecture axis (set-attention Slice 3).
- Re-investigate the "3 free bits" assumption — maybe the model already infers opp-side action context from action history embeddings; maybe the test recipe doesn't surface opp-action threats often enough.

## 8. Canonical file pointers

- `training/uma_ai/features.py:_observation_to_features_v3_1` — current v3.1 builder (54-d temporal block at slots 110-163). v3.3 layers on top.
- `engine-rs/crates/engine/src/policy/types.rs:PublicSideTurnState` — both sides already in the observation struct.
- `engine-rs/crates/engine/src/policy/featurize.rs:observation_state_features_v3_1` — Rust mirror.
- `training/make_v32_slot_token_init.py` — template for `make_v33_tail_init.py`.
- `training/serve_onnx.py:_resolve_feature_schema` — adds dispatch entry for state_dim=167.
- `docs/ai-research/scoping/v33-feature-gap-brainstorm-handoff.md` §2.A item 2 — original gap call.
- `docs/ai-research/scoping/v33-correctness-fix-scoping.md` — sibling correctness-fix slice (LANDED).
- `docs/ai-research/progress/r110.md §4f` — re-verdict #3 outcome that unblocked this scope.

## 9. Status line

`SCOPING — pre-registered`. Flips to IN-FLIGHT at training-arm-fire; flips to LANDED when the progress writeup lands at `docs/ai-research/progress/v33-additive-tail.md`.
