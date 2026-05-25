# v3.3 Correctness-Fix Slice

- **Date:** 2026-05-25
- **Status:** SCOPING — pre-registered, not yet launched.
- **Parent:** `v33-feature-gap-brainstorm-handoff.md` §5 step 1 (correctness fixes — compute-independent, ship at any scale).
- **Scope:** Three correctness/layout fixes that do **not** require a state-dim bump and are not gated on re-verdict #3. Opp-side state flags (the additive v3.3 tail) are explicitly deferred to a follow-up scoping doc once re-verdict #3 movement justifies a new dispatch entry.

## TL;DR

- Three fixes, all compute-independent, all bit-exact across Python/Rust:
  1. **Weakness-bonus correction** — featurizer ignores `weakness_bonus`; simulator uses it. Featurizer/simulator disagree on ~30% of matchups. Single multiply.
  2. **Action-slot cleanup** — slot 8 == slot 26 (duplicate), slot 28 dead bit (uid vs slot-idx mismatch), slot 10 polysemic (3 different meanings by kind). All in `frontend/src/game/engine/ai-policy/actions.ts`. Bumps `ACTION_FEATURE_SCHEMA_VERSION` 2 → 3.
  3. **State-dim and ONNX input set unchanged** — v3.2 byte freeze respected; no `_SCHEMA_BY_STATE_DIM` dispatch entry added.
- Hypothesis framing is **non-regression at minimum** (these are bugs, not features). Lift is a stretch goal, not the gate.
- Ablation: A0 baseline, A1 weakness only, A2 action-slot only, A3 combined. Gate is wl_lower non-regression vs A0 at n=10k.
- Defers: opp-side `usedSupporter/Retreat/Stadium` flags (additive tail, blocked on re-verdict #3 outcome).

## 1. Pre-registered hypotheses

### H1 — Weakness-bonus correction reduces Q-target noise on weakness-relevant matchups

- **Current state:** `training/uma_ai/features.py:1066-1073` `_can_ko` computes `damage = readiness[2] * 150.0` then compares to `defender.hp`. No weakness lookup. `_uma_readiness_features:1038` uses `attack.damage / 150.0` — pre-weakness printed damage only. Rust mirror `engine-rs/crates/engine/src/policy/featurize.rs` matches the Python bug. Simulator core (`engine-rs/crates/engine/src/flow/combat.rs:130-146`) applies weakness correctly: `+30` typed bonus or `*2` depending on the rule variant in effect.
- **Featurizer/simulator disagreement:** in any matchup where defender's `weakness` type matches attacker's primary type, featurizer reports `can_ko=0` and undamage when the simulator will actually deal lethal. Population of disagreement ≈ 30% per the handoff inventory.
- **Hypothesis:** correcting the featurizer to match simulator weakness logic will reduce Q-target prediction MSE on weakness-relevant rollouts. Population-level wl impact uncertain; framing is **non-regression**.
- **Falsifiable signal:** training-loss curves on the same recipe with A1 (corrected) vs A0 (current) should show A1 tracking Q-targets at lower or equal validation loss on a held-out weakness-relevant subset. If A1 loss is *strictly higher* than A0 on that subset, the hypothesis fails.

### H2 — Action-slot cleanup frees model capacity for legitimate signal

- **Current state in `actions.ts:487-551` (`ACTION_FEATURE_SCHEMA_VERSION = 2`):**
  - Slot 8 and slot 26 are exact duplicates (gap audit confirmed).
  - Slot 28 compares a uid to a slot index (different ID spaces) — near-always 0; dead bit.
  - Slot 10 is polysemic: energy count for attach-energy actions, targetValue for combat actions, unused for trainer actions. The model must learn to gate on kind (slot 2) to interpret slot 10 correctly.
  - Slots 11/29-31/12 partially re-encode the kind index from slot 2.
- **Hypothesis:** disambiguating slot 10 and repurposing slots 8/26/28 into orthogonal signals (TBD in §3) frees ~3 bits of effective input dimensionality. Model capacity is fixed; freeing wasted bits should be at-least-neutral at fixed budget and plausibly lift at variance-limited recipes.
- **Falsifiable signal:** non-regression on wl_lower at n=10k. Strict failure: A2 wl_lower < A0 wl_lower by more than the n=10k confidence interval (~0.013 wide at p=0.05 per R16 conventions).

### H3 — Combined A3 has no destructive interaction

- **Hypothesis:** A1 and A2 are orthogonal (one touches the state vector, one touches the action vector), so A3 = A1+A2 should perform at least as well as max(A1, A2) on wl_lower.
- **Falsifiable signal:** A3 wl_lower ≥ max(A1, A2) wl_lower − 0.013.

## 2. Freeze contract

| Surface | Pre-slice | Post-slice | Compatibility |
|---|---|---|---|
| `state_features` dim | 164 | **164 unchanged** | v3.2 byte-stable; slots 0-163 identical bytes. Weakness-bonus correction changes **the semantics** of slot 84 (`can_ko`) and the damage features in `_uma_readiness_features`. Old checkpoints' weights still apply — the input distribution shifts, which is fundamentally an input correction. |
| `state_features` schema version (`STATE_FEATURE_SCHEMA_VERSION`) | 3.1 (164-d marker) | **3.1.1** patch bump | Documents the semantic correction without changing dispatch. `_SCHEMA_BY_STATE_DIM[164]` resolution unchanged. |
| `action_features` dim | 48 | **48 unchanged** | Width preserved. |
| `ACTION_FEATURE_SCHEMA_VERSION` | 2 | **3** | Per-slot semantics changed for slots 8, 10, 26, 28. Bumped per the canonical pointer at `frontend/src/game/engine/ai-policy/actions.ts:487-551`. |
| ONNX input set (7 inputs) | Unchanged | Unchanged | No new tensors. |
| `_SCHEMA_BY_STATE_DIM` | Unchanged | Unchanged | No new dispatch entry. |
| `q_head` output | Present (added 2026-05-22) | Unchanged | Orthogonal. |
| `uma_slot_features[10, 23]` layout | Frozen | Unchanged | The frozen per-slot 23-d row is untouched. |
| Zero-init residual pattern | N/A | N/A | No tail added. Warm-start from v3.2 checkpoints loads weights verbatim; only input semantics shift. |

### Bit-exact mirror requirement

All state-feature changes land in BOTH:
- `training/uma_ai/features.py` (Python spec)
- `engine-rs/crates/engine/src/policy/featurize.rs` (Rust mirror)

A parity smoke must show byte-identical state vectors across the two paths on a fixed seed-pair before commit. The existing parity test should be extended to cover the weakness branch; if no such test exists, one is added in this slice.

### Existing checkpoint compatibility

- Weakness correction shifts the **input distribution** for slot 84 (some rows that were 0 become 1) and for the underlying damage features (some are doubled). Model weights stay loadable. Inference behavior changes by design.
- Action schema bump invalidates rows produced by ACTION_FEATURE_SCHEMA_VERSION=2 datasets at training time — the export pipeline must dispatch on the action-schema-version per row. If the dataset format does not currently embed `action_feature_schema_version`, this slice adds it (a one-field metadata bump).
- Production serving (`serve_onnx.py`) must be updated to refuse mismatched action-schema-version requests with a fail-fast error (consistent with the C5 "LANDMINE" pattern at `r16-model-feature-backlog-refinement.md:633-642`).

## 3. Per-fix specifications

### Fix 1 — Weakness-bonus correction

**Files:**
- `training/uma_ai/features.py` (`_can_ko`, `_uma_readiness_features`, damage-related helpers)
- `engine-rs/crates/engine/src/policy/featurize.rs` (Rust mirror)
- `engine-rs/crates/engine/src/policy/types.rs` (verify weakness type is in `PublicUmaObservation`; if absent, raise)

**Change:**
1. Extend the per-Uma observation type (if not already present) to expose `weakness: Option<EnergyType>` for the defender.
2. In `_can_ko`, before comparing damage to defender hp:
   ```python
   damage = readiness[2] * 150.0
   if defender.weakness is not None and defender.weakness == attacker.primary_type:
       damage = apply_weakness(damage)  # per-rule-variant: +30 or *2
   ```
3. In `_uma_readiness_features`, the damage feature emits weakness-adjusted damage divided by the existing normalization constant (150.0).
4. Mirror exactly in Rust featurize.rs.
5. Determine which weakness rule (+30 vs *2) is active by inspecting `engine-rs/crates/engine/src/flow/combat.rs:130-146` — featurizer matches simulator literal.

**Parity smoke:** add `tests/state_feature_weakness_parity_smoke.py` (or extend the existing parity test) that runs both Python and Rust featurizers on a fixed pair of weakness-relevant board states and asserts byte-identical state vectors.

**Risk:** if the simulator weakness rule is conditioned on stadium / item / trainer effects we didn't enumerate, the featurizer can drift again. Mitigation: featurizer queries the same `combat.rs` weakness function via a thin wrapper rather than reimplementing; if not feasible cross-language, the Rust path reuses `combat.rs::weakness_multiplier` directly and the Python path mirrors via FFI or hard-codes the documented rule with a comment pointing to the simulator source.

### Fix 2 — Action-slot 10 disambiguation

**File:** `frontend/src/game/engine/ai-policy/actions.ts:487-551`

**Change:** slot 10 becomes **kind-agnostic**, holding only the energy-count semantic. The combat targetValue and trainer signals previously sharing slot 10 move to repurposed slots (see Fix 3 and Fix 4 below).

### Fix 3 — Action-slot 8/26 dedup

**File:** `frontend/src/game/engine/ai-policy/actions.ts:487-551`

**Change:** slot 26 is repurposed (was a duplicate of slot 8). New use: **combat targetValue** (formerly the polysemic occupant of slot 10 for combat actions). Slot 8 retains its current semantic (the canonical owner of the duplicated feature).

### Fix 4 — Action-slot 28 repurpose

**File:** `frontend/src/game/engine/ai-policy/actions.ts:487-551`

**Change:** slot 28 (was a dead uid-vs-slot-idx comparison) becomes the **trainer-signal slot** previously sharing slot 10 for trainer actions. Concretely, slot 28 = a trainer-specific feature TBD by inspecting which trainer signals are currently invisible. First-cut candidate: heal amount or gust-target indicator (both flagged in handoff §B item 3). Final choice picked at implementation time after auditing what the 4-bit trainer flag block (slots 33-36) already covers.

**Note on slots 11/29-31/12:** these partially re-encode the kind index from slot 2. The handoff doc flags this as redundant but does not classify it as a correctness bug. **Out of scope for this slice** — left for a future schema audit. Keeping the cleanup focused: 4 slots, not 8.

## 4. Ablation plan

**Recipe baseline:** R16-P1 (v3.1 production recipe, since v3.2 ceiling is officially UNKNOWN and re-verdict #3 has not fired). Holding the baseline at v3.1 also avoids interaction with the v3.2 per-Uma-slot featurizer until re-verdict #3 establishes a v3.2 wl number.

**Arms:**
- **A0** (baseline): v3.1 features unchanged, ACTION_FEATURE_SCHEMA_VERSION=2.
- **A1**: v3.1 features + weakness correction. ACTION_FEATURE_SCHEMA_VERSION=2.
- **A2**: v3.1 features unchanged + ACTION_FEATURE_SCHEMA_VERSION=3 (slot 8/10/26/28 cleanup).
- **A3** (combined): v3.1 features + weakness correction + ACTION_FEATURE_SCHEMA_VERSION=3.

**Sample size:** n=10k gate per arm. Per the user's lean-validation memory, do NOT also run n=20k unless n=10k reports a band-straddle. Cheap smoke (n=240, eval-gate-only) precedes each n=10k run to catch wiring breakage.

**Decision rules:**
1. **Ship gate (non-regression):** all of A1, A2, A3 satisfy `wl_lower(arm) ≥ wl_lower(A0) - 0.013` at n=10k.
2. **Lift bonus (non-blocking):** if any arm shows `wl_lower(arm) ≥ wl_lower(A0) + 0.013`, write a lift finding into the progress doc.
3. **Hard fail:** if A3 fails the ship gate while A1 or A2 individually passes, investigate negative interaction before shipping anything.
4. **A0 reproduction sanity:** A0 must reproduce R16-P1's recorded wl=0.5810 within the n=10k confidence interval. If A0 itself drifts, the slice is paused until baseline reproducibility is restored.

**Estimated wall time** (workers=16, 12.4× post-speedup): each n=10k arm projects ~11 min from queue.json:59. Four arms × 11 min = ~44 min total. Smokes add ~5 min. **Total slice ablation budget: ~50 min wall.**

## 5. Implementation order

1. **Fix 1 (weakness) first** — single multiply, smallest blast radius, parity smoke is the hardest part. Lands as one commit.
2. **Fixes 2/3/4 (action slots) batched** — all in `actions.ts`, all schema-version-coupled. Lands as one commit with the schema bump and serve_onnx fail-fast guard.
3. **Smokes pass** — npm `test:train`, Python feature parity, action schema-guard smoke.
4. **Ablation queue entry** — add `v33-correctness-fix-ablation` to `docs/ai-agent-state/queue.json` with A0–A3 arms pre-registered. User-gated firing.
5. **STATUS flip** — this doc moves from SCOPING to IN-FLIGHT at queue-entry-added; flips to LANDED when the progress writeup lands at `docs/ai-research/progress/v33-correctness-fix.md`.

## 6. Out of scope (explicit)

- **Opp-side `usedSupporter/Retreat/Stadium` flags** — additive 3-bit state tail. Requires `_SCHEMA_BY_STATE_DIM[167]` entry and a v3.3 dispatch. Deferred to follow-up scoping doc, gated on re-verdict #3 movement per handoff §5 step 3.
- **Phase one-hot, per-condition one-hot, bench-temporal slot widening** — all flagged as v3.3 tail candidates in handoff §2.A. Deferred.
- **Synthesised features (lethal-in-N, clock differential, bench-refill safety, paralysis exploit window)** — all flagged in handoff §2.C. Higher conviction than the additive tail items, but require synthesis arithmetic and a separate scoping pass with their own falsifiable hypotheses.
- **Slots 11/29-31/12 redundancy** — kind-index re-encoding cleanup. Possibly a free win but not a correctness bug; left for a future action-schema audit.
- **Set-attention architecture** — orthogonal trunk-shape line, gated on Slice 2 ≥ 0.40.

## 7. Constraints honored

Per handoff §6 "Constraints / what's off-limits":
- No opponent hand IDs added (hidden-info regression guard).
- No raw turn-stamps emitted (bounded-norm booleans only).
- No ability-name strings (counts only).
- v3.2 byte freeze (slots 0-163) respected — semantics-only change on slot 84, no layout change.
- `_SCHEMA_BY_STATE_DIM` fail-fast dispatch unchanged — no silent fallback.
- Init parity preserved — no tail added means zero-init residual is N/A; existing checkpoints load verbatim with shifted input distribution.

## 8. Canonical file pointers

- `training/uma_ai/features.py:1038-1073` — `_can_ko`, `_uma_readiness_features`, damage features.
- `engine-rs/crates/engine/src/policy/featurize.rs` — Rust mirror.
- `engine-rs/crates/engine/src/policy/types.rs` — `PublicUmaObservation`; verify `weakness` field exposure.
- `engine-rs/crates/engine/src/flow/combat.rs:130-146` — simulator weakness logic (the canonical truth).
- `frontend/src/game/engine/ai-policy/actions.ts:487-551` — 48-d action vector and ACTION_FEATURE_SCHEMA_VERSION.
- `training/serve_onnx.py` — `_resolve_feature_schema` and schema-version guard.
- `docs/ai-research/scoping/v33-feature-gap-brainstorm-handoff.md` — parent gap audit.
- `docs/ai-agent-state/queue.json` — re-verdict #3 status (user-gated).
