# v3.3 Feature-Gap Brainstorm — Agent Handoff

**Status:** Brainstorm-stage. Section 7 open questions RESOLVED 2026-05-25 (see §7). Spawning a correctness-fix scoping doc (see `v33-correctness-fix-scoping.md`).
**Date:** 2026-05-25 (refreshed 2026-05-25 — same day; speedup landed concurrently).
**Purpose:** Pick this up to continue a feature-schema review for the AI policy.
**Routing:** This doc is the synthesis of a four-investigator gap audit + a strategic discussion about *revamp vs additive growth*. It is **not** a scoping doc for a specific bump — it is the input a scoping doc would draw from. The correctness-fix scoping doc (referenced above) is the first concrete spinoff.

---

## TL;DR

- v3.2's feature contract has real gaps, but most are **additively fixable** under the existing freeze-and-zero-init-residual pattern.
- The "schema axis is closed" verdict from R16 was **conditional on the pre-speedup compute regime**. The 12.4× MCTS speedup landed 2026-05-25 (`f75b347`), so re-verdict #3 (v3.2 at n=10k) is now compute-affordable — but it remains user-gated. The v3.2 ceiling claim "officially UNKNOWN" stands until the user fires it.
- Recommended sequencing (unchanged in shape, advanced in state): **correctness fixes first** (only step actionable without user gate) → user fires re-verdict #3 → if schema axis moves, land additive v3.3 tail → only then consider a token-first v4 revamp.
- A clean-slate revamp is **not** recommended today. It only becomes attractive if set-attention crosses its 0.40 gate (currently MARGINAL at Slice 2 wl=0.3205, Slice 3 never run).
- **Next concrete action:** correctness-fix slice (weakness-bonus correction, action-slot dedupe/disambiguate, opp-side flags). See `v33-correctness-fix-scoping.md`.

---

## 1. v3.2 current feature contract (recap)

v3.2 is the production schema, mandatory per directive B (`docs/ai-research-backlog.md:22-31`). It is distinguished from v3.0/v3.1 by **ONNX input-set** (5 → 7 inputs), not by state-vector width.

### ONNX feed
1. `state_features: f32[164]` — v3.0 110-d head + v3.1 54-d temporal tail.
2. `legal_action_features: f32[A, 48]` — pre-computed TS-side, carried verbatim.
3. `card_ids_by_zone: i64[8, 30]` — zones: ownActive, oppActive, ownBench, oppBench, ownHand, ownDiscard, oppDiscard, stadium.
4. `action_card_idx_pair: i64[A, 2]` — `(source, target)` vocab idx per action.
5. `uma_slot_card_ids: i64[10]` — **new in v3.2**.
6. `uma_slot_features: f32[10, 23]` — **new in v3.2**.
7. (`q_head` output added 2026-05-22 — additive scalar output, not a state-feature bump.)

### Per-Uma slot order (FROZEN, 10 slots)
- 0: own active
- 1-3: own bench[0..2]
- 4: own bench[3] (RESERVED — engine `MAX_BENCH=3`, always absent; kept stable for future growth)
- 5: opp active
- 6-8: opp bench[0..2]
- 9: opp bench[3] (reserved)

### Per-slot 23-d feature row (FROZEN)
| Col | Field |
|---|---|
| 0 | polarity (+1 own, -1 opp) |
| 1 | role_active (1 if active) |
| 2 | slot_idx_norm (-1 for active, bench i/3) |
| 3 | present mask |
| 4 | hp / max_hp |
| 5 | (max_hp - hp) / max_hp |
| 6 | stage / 2 |
| 7 | energy_total / 6 |
| 8-17 | typed energies / 4 (grass, fire, water, lightning, psychic, fighting, darkness, steel, colorless, dragon) |
| 18 | tool present (0/1) |
| 19 | paralysed condition present |
| 20 | special_conditions count / 5 |
| 21 | used_ability_this_turn |
| 22 | evolved (stage > 0) |

### Canonical source files
- `training/uma_ai/features.py` — Python spec (canonical).
- `engine-rs/crates/engine/src/policy/featurize.rs` — bit-exact Rust mirror.
- `engine-rs/crates/engine/src/policy/types.rs` — `PublicObservation` struct.
- `frontend/src/game/engine/ai-policy/actions.ts:487-551` — 48-d action vector definition (`ACTION_FEATURE_SCHEMA_VERSION = 2`).

---

## 2. Gap inventory (from four parallel investigators)

### A. Observation fields the featurizer ignores or lossy-encodes

Source: `policy/types.rs` vs `features.py` / `featurize.rs`.

1. **`energy_zone` typed contents** — captured only as `len/4` (slot 28) plus a depth-1 boolean (`next_energy_matches_active_need`). Both the upcoming queue's colors AND the opponent's energy zone are completely ignored. **Biggest semantic loss in v3.2.**
2. **Opponent `usedSupporterThisTurn` / `usedRetreatThisTurn` / `usedStadiumThisTurn`** — featurizer reads `own.*` only. 3 free bits, high-value for threat assessment.
3. **Phase as ordinal scalar (slot 0 = `phase_index/9`)** — non-linear ordering forced onto a scalar. Cheap fix: 10-bit one-hot.
4. **Special conditions collapsed to "paralysed bit + count/5"** (slot cols 19-20) — burned/poisoned (end-of-turn damage) vs asleep/frozen (attack block) differ mechanically. Identity matters, not just cardinality.
5. **Bench per-Uma temporal is mean-aggregated** (v3.1 slots 137-145 / 155-163). Which bench Uma took damage last turn is blurred. Natural home: move bench temporal into the v3.2 slot tokens (widen `UMA_SLOT_FEATURE_DIM`).
6. **`species` identity per Uma** — currently only learnable via cardId vocab co-occurrence; explicit species embedding would shortcut evolution-chain reasoning.
7. **Discard role buckets are own-side only** (slots 84-86). Symmetric opp-side extension is a 3-bit drop-in.

### B. 48-d action vector gaps

Source: `frontend/src/game/engine/ai-policy/actions.ts:487-551`.

1. **Attack expected damage / KO flag / post-attack effect bits** — currently folded into the heuristic prior (slot 0) and one conditional bit (slot 41). Combat planner already computes lethality; not exposed.
2. **Attach-energy type + "completes typed threshold" bit** — no slot for *which* energy is being attached; only a post-state deficit count.
3. **Trainer effect *magnitudes*, not just flags** — slots 33-36 collapse ~12 effect classes into 4 bits; heal amount, gust target, search-filter class invisible.
4. **Retreat cost + swap-in readiness** — retreat candidates differ only in slot 0 (the heuristic).
5. **Evolution Δhp / Δdamage / Δability** — slots 14-17 describe the evolve card in isolation, not the upgrade delta.

**Free cleanup wins (no width growth):**
- Slot 8 and slot 26 are exact duplicates.
- Slot 28 compares a uid to a slot index (different ID spaces) — near-always 0; dead bit.
- Slot 10 is polysemic (energy count for attach, targetValue for combat, unused for trainer) — disambiguate or split.
- Slots 11/29-31/12 partially re-encode the kind index (slot 2). Redundant.

### C. Game-mechanic-driven synthesised features (forward arithmetic)

These require synthesis across observation fields, not just plucking.

1. **Lethal-in-N / clock differential** — turns-to-KO own & opp active given current energy + realised attach budget. The dominant race-vs-stabilise decision.
2. **Net point swing if opp gusts my weakest bencher** — single-feature explanation for a whole class of catastrophic losses (`gust_opponent` trainers).
3. **Weakness-adjusted effective damage** — `can_ko` (slot 84) and `damage/150` ignore `weakness_bonus` when defender type matches attacker's weakness. Systematic bias on ~30% of matchups. **Cheapest "free win" candidate — single multiply.**
4. **Energy ETA per Uma** — turns-to-attack-ready accounting for energy-zone color mismatch.
5. **Searchable targets remaining in deck** — own deck composition is fully known (decklist − hand − discard − in-play). A search trainer with zero valid targets is dead, but the model can't tell.
6. **Bench-refill safety** — `would_lose_on_active_KO` (last Uma + no promote available). Binary catastrophe the policy should never blunder.
7. **Paralysis exploit window** — if opp paralysis_recovery_pending, free attack turn; symmetric for self. Highest single-turn EV swing in the game.

Full ranked list (15 candidates) is in the investigator transcripts; these are the top-of-funnel.

### D. Already-considered / deferred ideas worth reviving

From `docs/ai-research/scoping/r16-model-feature-backlog-refinement.md`:
- **Slot-token integration of per-Uma temporal fields** (lines 388-394) — scoped P1, rejected in v3.2 ("ship scalar-only first"). Natural fit for a v3.3 widening of `UMA_SLOT_FEATURE_DIM` from 23.
- **C3 `schemaVersion 3→4` escape hatch** (line 631) — already reserved for any TS observation bump.
- **Board-zone double-count cleanup** (lines 498-505) — currently use conservative "ignore in projection," cleaner split deferred.

---

## 3. The strategic question — revamp vs additive

### Why NOT a clean-slate revamp now

The empirical pin in the backlog (`docs/ai-research/progress/r16.md:42-115`) is that **schema-axis changes have not moved the n=10k ceiling since v3.0** at the pre-speedup compute scale:
- v3.1 ablated NO-GO (wl=0.5810 vs v3.0 0.5811).
- v3.2 W6 iter-2 wobble reproduces independent of schema.

A revamp pays a large parallel-training tax against weak evidence of payoff, and it throws away the *additive-tail + zero-init residual + byte-freeze + `_SCHEMA_BY_STATE_DIM` dispatch* pattern that makes v3.x changes nearly risk-free to land.

### What additive growth CAN handle

Most of the gap inventory is reachable additively:
- **Correctness bugs** (weakness bonus, slot 8/26 duplicate, slot 10 polysemic, opp-side flag asymmetry) — these aren't features, they're fixes.
- **Layout problems** that look structural (ordinal phase, mean-aggregated bench temporal, collapsed conditions) can be repaired by emitting corrected slots in a new tail, freezing the old slots, and letting the model migrate via zero-init residual (the v3.1 `delta=0.0` trick generalizes — explicitly noted in the scoping doc).

### When a revamp WOULD make architectural sense

If **set-attention crosses its 0.40 gate at Slice 3+**. That trunk wants tokenized inputs end-to-end — a v4 where "everything is a slot token" (hand tokens, discard tokens, deck-residual token, action tokens) would dovetail with the architecture already being explored (`docs/ai-research/scoping/set-attention-architecture-probe.md`). The current v3.2 per-Uma slots are a half-step in that direction.

But set-attention is **MARGINAL at Slice 2 (wl=0.3205 < 0.40 gate)**. Redesigning the feature contract around a trunk that may not ship is putting the cart before the horse.

---

## 4. Compute-conditional update — RESOLVED 2026-05-25

**Status:** The "≥10x soon" framing this section was written under is now **landed**. Section retained for historical reasoning trace; current state inlined below.

### What landed

- **12.4× cumulative wall-clock speedup** on the sim-mcts-selfplay path vs the 0.60 g/s baseline (closeout commit `f75b347`, table at `mcts-selfplay-throughput-handoff.md:283`, scoping `docs/ai-research/scoping/r12-selfplay-gate-throughput.md:5`).
- Path was R7.b throughput-spike Slices 1-3i: Rust in-process ort, orchestrator wiring, v3.2/v3.1 Rust featurizer fast path, real `--workers` game-level parallelism via `std::thread::scope`, GPU EP G5 lock-free dispatch, work-stealing worker pool, plus hot-path micro-opts.
- 12.4× comfortably exceeds the ≥10× framing the original gap audit assumed.

### What the post-landing picture actually is

1. **v3.2 n=10k re-verdict is no longer compute-gated — it's user-gated.** Queue id `tight-gate-reverdict-program` (`docs/ai-agent-state/queue.json:55+`) holds re-verdict #3 (v3.2 at n=10k, projected ~11 min wall at workers=16) explicitly user-gated. Re-verdicts #1 (v3.0) and #2 (v3.1) at n=10k are DONE — both PARTIAL, identical to 4 decimals (wl ≈ 0.5810/0.5811). The v3.2 ceiling claim "officially UNKNOWN" remains literally true until the user fires #3.
2. **Set-attention Slice 2 verdict — still MARGINAL.** `set-attention-architecture-probe.md:5-13,203-209` reports wl=0.3205 (n=1000, side-balanced raw-policy gate, landed 2026-05-22). Slice 3 NEVER run, gated on Slice 2 ≥ 0.40 AND re-verdict #3 reporting. Schema-revamp justification has not moved.
3. **Weakness-bonus correction — still wide open.** Zero touches: no `"weakness"` hits in `training/uma_ai/` or `engine-rs/crates/engine/src/policy/`. `_can_ko` (`features.py:1066-1073`) reads `damage = readiness[2] * 150.0` with no weakness lookup; Rust mirror matches. Simulator core (`engine-rs/.../flow/combat.rs:130-146`) uses weakness — so featurizer and simulator disagree on ~30% of matchups. Cheapest free win remains untaken.

### What does NOT change

- **Correctness bugs first** gets *stronger*, not weaker. Weakness-bonus correction is still a single multiply; EV is compute-independent on the downside and scales with compute on the upside.
- **Additive growth as default** still wins because of the freeze contract — even with compute open, additive tails are still the cheap way to ride the schema axis.
- **Hidden-info / vocab-drift guards** — orthogonal, don't move.

### The honest framing (updated)

The R16 verdict was "schema closed *at the compute scale we trained at*." The 12.4× speedup didn't automatically reopen it — it made the experiments that *would* reopen it cheap. Those experiments are now waiting on **user permission** (re-verdict #3), not throughput. The revamp question is downstream of those re-verdicts; in the meantime, correctness fixes ship at any scale.

---

## 5. Recommended sequencing

1. **Land correctness fixes** (no compute risk, gains compound at any scale):
   - Weakness-bonus adjustment in `_card_awareness_features` (`can_ko`, damage features). **LANDED 2026-05-25 commit `8141772`.**
   - Dedupe action slots 8/26, repurpose slot 28, disambiguate slot 10. **LANDED 2026-05-25 commit `5ea1758` (ACTION_FEATURE_SCHEMA_VERSION 2 → 3).**
   - Add opp-side `usedSupporter/Retreat/Stadium` flags (3 bits). **DEFERRED to v3.3 additive tail (step 3) per scoping doc `v33-correctness-fix-scoping.md` §6** — width bump requires re-verdict #3 movement to justify.
   - Combined ablation A0/A1/A2/A3 queued and user-approved 2026-05-25; see queue id `v33-correctness-fix`.
2. **Once the MCTS speedup lands, run the queued re-verdicts** before committing to anything bigger:
   - v3.2 n=10k tight-gate re-verdict. **FIRED 2026-05-25 (user-approved)** — manifest at `runs/v32-uniform-retrain-iter1-tight-gate/gate.manifest.json`; verdict writeup at `progress/r110.md § 4f`.
   - Set-attention Slice 2/3 re-verdict. **Slice 2 still MARGINAL (wl=0.3205 < 0.40 gate)**; Slice 3 not run, gated on Slice 2 ≥ 0.40 AND re-verdict #3 outcome.
3. **If schema axis shows movement** at the new scale → land an **additive v3.3 tail** with the high-conviction structural fixes:
   - Phase one-hot (10 bits) supplementing slot 0.
   - Energy-zone typed contents (~80 bits for both sides, depth 2-3).
   - Per-condition one-hot (5 bits) replacing/supplementing paralysed+count.
   - Bench temporal moved into widened slot tokens (drop the mean aggregate).
   - **Opp-side flags (3 bits)** — deferred from step 1, lands as part of the additive tail with a new `_SCHEMA_BY_STATE_DIM[167]` dispatch entry.
4. **Only if set-attention crosses 0.40** → seriously scope a token-first v4 revamp. Until then, hold it as a contingency.

---

## 6. Constraints / what's off-limits

- **Opponent hand IDs** — hidden-info regression guard. Stay absent. (`r16-model-feature-backlog-refinement.md:405,418`)
- **Raw turn-stamps** (`enteredTurn`, `evolvedTurn`) — always derive bounded-norm booleans. (`r16-model-feature-backlog-refinement.md:277-283`)
- **Ability-name strings** — counts only (vocab-drift guard). (`r16-model-feature-backlog-refinement.md:290-291`)
- **Byte-layout freeze:** slots 0-109 (v3.0 head) and 0-163 (v3.1 layout) are byte-stable; new bumps must be additive tails.
- **`_SCHEMA_BY_STATE_DIM` fail-fast dispatch** — no silent fallback. v3.2 distinguishes by input-set count, not state_dim. The C5 "LANDMINE" warning is at `r16-model-feature-backlog-refinement.md:633-642`.
- **Init-parity / zero-init residual** — any new tail uses zero-init Linear so warm-start matches the previous schema's behavior exactly (the v3.1 `delta=0.0` trick).

---

## 7. Open questions — RESOLVED 2026-05-25

All five resolved by investigator pass; original questions retained below for traceability, with answers inlined.

1. **Has the MCTS speedup landed yet?** **YES.** Closeout commit `f75b347` (2026-05-25). Slices 1-3i landed under R7.b throughput-spike. See `docs/ai-research/scoping/r12-selfplay-gate-throughput.md` and untracked `mcts-selfplay-throughput-handoff.md`.
2. **What's the actual speedup ratio?** **12.4× cumulative** vs the 0.60 g/s baseline. Sub-figures along the way: 4.18× (Slice 2 orchestrator wiring), 5.6× at workers=8 (Slice 3d worker pool), 7.11× at workers=16 (GPU EP G5).
3. **Has v3.2's n=10k re-verdict been re-queued or run?** **Re-verdicts #1 and #2 DONE — both PARTIAL** (v3.0 wl=0.5811, v3.1 wl=0.5810 — identical to 4 decimals at n=10k). **Re-verdict #3 (v3.2) USER-GATED**, "fire only on user request," ~11 min projected wall at workers=16 (`docs/ai-agent-state/queue.json:55-73`).
4. **Is the weakness-bonus correction already on someone's plate?** **NO.** Zero hits for `"weakness"` in `training/uma_ai/` or `engine-rs/crates/engine/src/policy/`. `_can_ko` and damage features in `features.py:1066-1073` and Rust mirror ignore `weakness_bonus`; simulator core (`engine-rs/.../flow/combat.rs:130-146`) uses it. **Cheapest first slice confirmed open.**
5. **Has set-attention Slice 3 been gated open?** **NO.** Slice 2 MARGINAL at wl=0.3205 (run `runs/r7b3-set-attention-slice2/gate.manifest.json`, landed 2026-05-22). 0.40 acceptance gate not crossed. Slice 3 never run; explicitly "DO NOT proceed without user gate."

### What these answers imply

- **Compute is no longer the gate** for re-verdicts; user permission is. Anything in §5 step 2 ("once the speedup lands") is now in "run them" state, not "wait."
- **§5 step 1 (correctness fixes) is the only step actionable without user gating.** Weakness-bonus correction is the cheapest, most leveraged slice (compute-independent EV downside, scales upward with compute).
- **§5 step 3 (additive v3.3 tail) requires re-verdict #3 movement to justify.** Holding until that ships.
- **§5 step 4 (token-first v4 revamp) requires Slice 3 set-attention crossing 0.40.** Holding until Slice 2 is re-baselined and crosses the gate.

---

## 8. Canonical file pointers

### Feature code
- `training/uma_ai/features.py` — Python spec (canonical for state, action, slot, card_ids).
- `engine-rs/crates/engine/src/policy/featurize.rs` — Rust mirror (bit-exact).
- `engine-rs/crates/engine/src/policy/types.rs` — `PublicObservation`, `PublicSideObservation`, `PublicUmaObservation` structs.
- `engine-rs/crates/engine/src/policy/observation.rs` — observation builder.
- `frontend/src/game/engine/ai-policy/actions.ts:487-551` — 48-d action vector.

### Research docs
- `docs/ai-research/README.md` — routing map.
- `docs/ai-research-backlog.md` — master backlog (directive B = v3.2 mandatory).
- `docs/ai-research/scoping/r16-model-feature-backlog-refinement.md` — the canonical feature-schema scoping doc; constraints, deferrals, freeze contract.
- `docs/ai-research/progress/r16.md` and `r16-archive.md` — feature-schema verdicts.
- `docs/ai-research/scoping/set-attention-architecture-probe.md` — trunk-shape probe.
- `docs/ai-research/scoping/post-throughput-scale-up-directions.md` — what to do with new compute headroom.
- `docs/ai-agent-state/queue.json` — currently queued work.

### Model / serving
- `training/serve_onnx.py` — `_resolve_feature_schema` and `_SCHEMA_BY_STATE_DIM` (the fail-fast schema dispatch).

---

## 9. What this doc is NOT

- Not a v3.3 scoping proposal. A scoping doc requires a specific feature set, a falsifiable lift hypothesis, an ablation plan, and a freeze contract. This is the input to that.
- Not a verdict on any specific feature. The investigators ranked candidates; the rankings are hypotheses, not measurements.
- Not authorization to implement. The next agent should confirm sequencing with the user before writing code.
