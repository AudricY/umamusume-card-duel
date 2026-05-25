# v3.3 Feature-Gap Brainstorm — Agent Handoff

**Status:** Brainstorm-stage. No code changes proposed. No verdict reached.
**Date:** 2026-05-25
**Purpose:** Pick this up to continue a feature-schema review for the AI policy.
**Routing:** This doc is the synthesis of a four-investigator gap audit + a strategic discussion about *revamp vs additive growth*. It is **not** a scoping doc for a specific bump — it is the input a scoping doc would draw from.

---

## TL;DR

- v3.2's feature contract has real gaps, but most are **additively fixable** under the existing freeze-and-zero-init-residual pattern.
- The "schema axis is closed" verdict from R16 is **conditional on the pre-speedup compute regime**. A ≥10x MCTS speedup re-opens it because (a) v3.2's own n=10k ceiling is officially UNKNOWN, (b) deeper rollouts produce crisper Q targets that richer schemas need to express their asymptote.
- Recommended sequencing: correctness fixes first → additive v3.3 tail for high-conviction structural fixes → re-run schema-axis and set-attention verdicts at the new compute scale → only then consider a token-first v4 revamp.
- A clean-slate revamp is **not** recommended today. It only becomes attractive if set-attention crosses its 0.40 gate (currently MARGINAL at Slice 2 wl=0.3205).

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

## 4. Compute-conditional update (KEY UPDATE)

A ≥10x MCTS speedup landing soon **partially overturns the "schema axis closed" verdict**.

### Why it changes things

1. **v3.2's own n=10k ceiling is officially UNKNOWN** (`docs/ai-research-backlog.md:30`). The re-verdict is queued and user-gated. Cheaper compute means that re-verdict actually gets run. If v3.2 at the new scale beats v3.0's 0.5811, the "schema closed" claim collapses to "v3.0 vs v3.1 didn't move at the *old* scale."
2. **Deeper MCTS → crisper Q targets.** Slot tokens, weakness correction, and energy-zone typing all want clean per-state value differences. They show as noise at shallow rollouts and as lift at deep rollouts. The richer the schema, the more rollouts needed to find its asymptote.
3. **Set-attention Slice 2 verdict re-opens** at the new scale too. If it crosses 0.40, the trunk that wants tokenized inputs becomes live, and a token-first revamp acquires a real consumer.

### What does NOT change

- **Correctness bugs first** gets *stronger*, not weaker. Weakness-bonus correction is a single multiply; EV is compute-independent on the downside and scales with compute on the upside.
- **Additive growth as default** still wins because of the freeze contract — even if compute opens the schema axis, additive tails are still the cheap way to ride it.
- **Hidden-info regression guards, vocab-drift guards** — orthogonal, don't move.

### The honest framing

The R16 verdict was "schema closed *at the compute scale we trained at*." A 10x speedup doesn't automatically reopen it — it makes the experiments that would reopen it cheap enough to run. The revamp question is downstream of those re-verdicts.

---

## 5. Recommended sequencing

1. **Land correctness fixes** (no compute risk, gains compound at any scale):
   - Weakness-bonus adjustment in `_card_awareness_features` (`can_ko`, damage features).
   - Dedupe action slots 8/26, repurpose slot 28, disambiguate slot 10.
   - Add opp-side `usedSupporter/Retreat/Stadium` flags (3 bits).
2. **Once the MCTS speedup lands, run the queued re-verdicts** before committing to anything bigger:
   - v3.2 n=10k tight-gate re-verdict (already user-gated in the queue).
   - Set-attention Slice 2/3 re-verdict.
3. **If schema axis shows movement** at the new scale → land an **additive v3.3 tail** with the high-conviction structural fixes:
   - Phase one-hot (10 bits) supplementing slot 0.
   - Energy-zone typed contents (~80 bits for both sides, depth 2-3).
   - Per-condition one-hot (5 bits) replacing/supplementing paralysed+count.
   - Bench temporal moved into widened slot tokens (drop the mean aggregate).
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

## 7. Open questions for the next agent

1. **Has the MCTS speedup landed yet?** Check `git log --since="2026-05-25" --oneline -- engine-rs/crates/engine/src/mcts/` and recent digests under `docs/ai-agent-state/digests/`. The strategic verdict here is conditional on that.
2. **What's the actual speedup ratio?** "≥10x" is the framing the user used in conversation; the realised ratio matters for whether the schema-axis re-verdicts are likely to find anything.
3. **Has v3.2's n=10k re-verdict been re-queued or run?** Check `docs/ai-agent-state/queue.json` for `tight-gate-reverdict-program` status.
4. **Is the weakness-bonus correction already on someone's plate?** A quick `rg "weakness" docs/ai-research training/uma_ai engine-rs/crates/engine/src/policy/` before scoping. If absent, it's the cheapest first slice.
5. **Has set-attention Slice 3 been gated open?** Check `docs/ai-research/scoping/set-attention-architecture-probe.md` and recent progress logs.

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
