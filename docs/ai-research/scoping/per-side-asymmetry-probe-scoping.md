# Per-Side Asymmetry Probe — Scoping

- **Date:** 2026-05-26
- **Status:** **PROPOSED** — DRAFT, USER-GATED. Diagnostic pyramid only; no training-run commitment until Phase D1 result.
- **Queue entry:** `per-side-asymmetry-probe` (`docs/ai-agent-state/queue.json:160-164`). This scoping doc OPERATIONALISES that entry; the queue entry is unchanged and remains canonical for status. Do not duplicate.
- **Parent:** `docs/ai-research/progress/r110.md` §4m (asymmetry surfaced; recipe-axis exhausted) + §4n (v3.7 SOFT-SHIP; asymmetry unchanged at player=0.5556 / opp=0.6105 / Δ=0.0549, reaffirms structural gap). `v37-combat-arith-and-catalog-scoping.md` §8 rule 2 (off-axis pivot priority list, asymmetry probe #1).

---

## TL;DR

Across 10 hidden=128 runs spanning v3.0 → v3.7 (9 recipes pre-v3.7 in §4m + v3.7 A1), every n=10k tight-gate exhibits a structural per-side wl_lower gap:

- player-side wl_lower ∈ [0.5500, 0.5556] (σ=0.0018)
- opp-side wl_lower ∈ [0.6105, 0.6242] (σ=0.0023)
- Δ ≈ 0.055–0.074 in every run; **35× the cross-recipe noise floor (σ overall=0.002)**

This is the ONLY signal above noise floor in the entire schema-axis-exhausted lineage. Closing half the gap (player 0.55 → 0.59) lifts overall wl_lower to ≈0.605 — **clears the LIFT band (≥0.6040)** no recipe or schema lever has crossed.

The probe is a **diagnostic-pyramid investigation, not a training-iteration arm**. Cheapest test first (featurizer parity, hours); escalate only if the cheap test does not localise the cause.

---

## 1. Pre-registered hypotheses

Each candidate root cause gets a binary signal and an expected lift band conditional on a fix.

### H1 — Featurizer perspective-swap bug (Python or Rust mirror)
- **Theory:** `observation_to_features_v3_7` (and its v3.0/v3.3/v3.5/v3.6 ancestors) was authored side-by-side as own/opp slot pairs; a single mis-mirrored slot (e.g., a copy-paste using `own` where `opp` is meant, or a bit-flip in a side-relative scalar) would produce systematically worse value/policy when the model plays one side.
- **Falsifiable signal (D1):** `featurize(build_public_observation(state, Player))` vs `featurize(build_public_observation(state, Opponent))` on 100+ deterministic fixtures, after applying a known own↔opp slot-permutation map and inverting any side-tag scalars (slot 1 `SIDES index`, slot 96 `_first_player_polarity`). Any non-permutation byte delta → bug located. Must hold for BOTH Python (`training/uma_ai/features.py`) and Rust mirror (`engine-rs/crates/engine/src/policy/featurize.rs`); a Python↔Rust disagreement on the swap result is itself a class-2 bug.
- **Expected lift if fixed:** +0.04 – 0.07 wl_lower (closes the full gap, clears LIFT band).

### H2 — MCTS root or backup perspective drift (single-sided mode)
- **Theory:** The v3.7 gate uses `mctsTwoSided: false` (`gate.manifest.json:16`). Backup unconditionally adds in `model_side` frame (`mcts/driver.rs:289-303`); if a child expansion or terminal evaluation accidentally injects a `next_side`-frame scalar without sign-flip, the model side gets a systematically biased Q.
- **Falsifiable signal (D2):** With featurizer parity confirmed in D1, run a fixed seed twice — once with `model_side=Player`, once with `model_side=Opponent` — on a deterministic mid-game position and compare root visit distributions + value backups. In single-sided mode they should mirror under own/opp permutation. Any non-mirror visit shift > 1 visit on a clean fixture → MCTS-side asymmetry.
- **Expected lift if fixed:** +0.02 – 0.05 wl_lower (partial; MCTS-side fixes typically narrower than featurizer fixes).

### H3 — Selfplay-side label imbalance in training data
- **Theory:** Selfplay rollouts emit rows tagged by `model_side`; if iteration training accidentally over-weights one side (e.g., shuffle bias, drop-last in an unequal batch, or asymmetric label vs target convention), the value head undertrains on player frames.
- **Falsifiable signal (D3):** Histogram of `model_side` over the last loop's emitted rows (`runs/R16-P1-v37-cap128-A1/iter-*/selfplay/*`). Expected 50/50 ± Wilson noise at row count. >2σ skew → label imbalance. Secondary: per-side value-loss curves during training — if player-side value-loss converges meaningfully slower or to a higher floor than opp-side, points at data-side cause.
- **Expected lift if fixed:** +0.01 – 0.03 wl_lower.

### H4 — Eval-gate harness side-balance / config bias
- **Theory:** Eval-gate may apply asymmetric MCTS config or selection per side, or compute wilsonLower over an off-balance task set.
- **Falsifiable signal (D4):** Confirm `eval_gate.rs:530-541` builds 5000 player + 5000 opponent tasks for `--model-side both` (manifest confirms: playerSide.games=5000, opponentSide.games=5000). Confirm `MctsConfig` is invariant across `model_side` in `drive_one_game`. Already mostly ruled out by inspection; D4 is a 30-min code-read + golden-state assert, NOT a re-run.
- **Expected lift if fixed:** +0.00 – 0.02 wl_lower (this hypothesis is the lowest-prior).

### H5 — Intrinsic first-player rule asymmetry (game-design baseline)
- **Theory:** The Pokémon-style first-player rule may give one seat a structural winrate edge even under optimal play. If heuristic-AI shows the same gap, part of Δ=0.067 is unfix-able.
- **Falsifiable signal (D5):** Heuristic AI vs heuristic AI, model_side=both, n=2000. If `player_wr ≈ opp_wr ± Wilson noise`, no intrinsic gap → all 0.07 model-attributable. If heuristic shows e.g. Δ=0.03, intrinsic baseline soaks half; the model-attributable residual is ≤0.04.
- **Note:** `first_player` is coin-flipped at setup (`dispatcher.rs:220-233`) — `modelSide=Player` does NOT mean "goes first." H5 measures the seat asymmetry, NOT the first-mover asymmetry. A follow-on probe (D5b) can stratify wl_lower by `first_player == model_side` to isolate first-mover effect.
- **Expected lift if fixed:** N/A (caps the addressable ceiling rather than producing lift).

---

## 2. Diagnostic contract (pyramid order)

The cheapest test runs first; each step's outcome decides whether the next step fires.

### D1 — Featurizer perspective parity smoke (~1–2 hours wall)
**Land:** `training/per_side_featurizer_parity_smoke.py` + Rust mirror `tests/per_side_featurizer_parity.rs`.

**Method:**
1. Generate 100 deterministic mid-game fixtures via `setup_ai_vs_ai_game` over seeds 0..99, advancing N≈8 turns each (covers prize 0–2, both phases).
2. For each fixture: build `obs_p = build_public_observation(state, Player)` and `obs_o = build_public_observation(state, Opponent)`.
3. Compute `f_p = observation_to_features_v3_7(obs_p)` and `f_o = observation_to_features_v3_7(obs_o)`.
4. Compute the **expected swap permutation**: a fixed mapping from v3.7's 296-slot layout that swaps own↔opp pairs across all paired channels (slot pairs documented in `features.py` headers + `v37-combat-arith-and-catalog-scoping.md` §4 layout) and inverts known polarity scalars (slot 1 SIDES index, slot 96 `_first_player_polarity`).
5. Assert `apply_swap(f_p) == f_o` byte-identically. Allowed tolerance: 0 ULP on integer/categorical slots, ≤1 ULP on derived hash slots (mirror v3.6 parity-smoke convention).

**Cross-language step (D1b):** Repeat the same check with Rust's `observation_state_features_v3_7` on identical fixtures. Python↔Rust agreement on swap result is a third assertion.

**Decision:**
- Any non-permutation delta → **STOP**. Root cause is featurizer. Implement fix; ckpt warm-start is preserved (no schema change); re-fire v3.7 tight-gate (~9 min wall). Expected lift +0.04–0.07 per H1.
- Pass → escalate to D2.

### D2 — MCTS perspective parity smoke (~2–4 hours wall, code-only)
**Land:** `engine-rs/crates/engine/tests/mcts_side_symmetry.rs`.

**Method:** Pick 10 fixed mid-game states. For each, run `mcts_search(state, model_side=Player, sims=100)` and `mcts_search(state, model_side=Opponent, sims=100)` with the same seed. With featurizer parity confirmed, root visit distributions should be permutation-mirror images. Sign-flip the backup Q in the comparison.

**Decision:**
- Visit-count shift > 1 visit on >=2 fixtures → MCTS-side bug. Implementer fix in `engine-rs/crates/engine/src/mcts/` (driver / node / sample). Re-fire tight-gate. Expected lift +0.02–0.05 per H2.
- Pass → escalate to D3.

### D3 — Selfplay-side label histogram + per-side value-loss (~1 hour wall, post-hoc analysis)
**Method:** Read `runs/R16-P1-v37-cap128-A1/iter-*/selfplay/*.jsonl`, histogram `model_side` field. Read per-iter training metrics from existing `loop/iter-*/train_metrics.json` (if logged) and compute per-side value-loss separately if grouped tags exist.

**Decision:**
- |player_rows − opp_rows| > 2σ Wilson → data-side imbalance. Recipe adjustment (re-shuffle / balanced sampler) in `training/orchestrator/*`. Re-fire base loop + tight-gate. Expected lift +0.01–0.03 per H3.
- Pass → escalate to D4 + D5.

### D4 — Eval-gate side-balance audit (~30 min code-read)
**Method:** Inspect `eval_gate.rs:530-541` task generation and `drive_one_game` config (`eval_gate.rs:334-370`) for any `model_side`-conditional branch. Confirm v3.7 gate manifest `playerSide.games = opponentSide.games = 5000`.

**Decision:** Already mostly resolved by inspection (counts balanced). Any side-conditional config branch → gate-harness fix.

### D5 — Heuristic-AI per-side baseline (~30–60 min wall)
**Method:** `cargo run -p sim-cli --bin eval_gate -- --selection random --model-side both --seeds 1000 --sims 0` (or analogous heuristic-vs-heuristic config). Measure per-side wl_lower of a model-free baseline.

**Decision:**
- If Δ(player − opp) ≈ 0 ± Wilson → no intrinsic seat asymmetry; full 0.07 is model-attributable.
- If Δ < 0.067 with same sign → some intrinsic residual; model-attributable ceiling shrinks.
- Run D5b stratification by `first_player == model_side` to disentangle seat vs first-mover effect.

---

## 3. Implementation order

1. **D1 featurizer parity smoke** — Python smoke + Rust mirror test. NO training run. (USER-GATED START.)
2. **D5 heuristic-AI baseline** — parallel with D1 (independent compute, ~1 hour wall). Sets the model-attributable ceiling.
3. If D1 fails: implement fix, re-fire v3.7 tight-gate. STOP pyramid. Report verdict.
4. If D1 passes: D2 MCTS parity smoke + D3 selfplay histogram + D4 eval-gate audit, in parallel (all read-only / cheap).
5. Whichever Dn returns a positive signal: hand off to `implementer` with the bounded fix scope.
6. If ALL of D1–D5 pass with the model showing the asymmetry: VERDICT = asymmetry is real but localised outside featurizer / MCTS / data / eval. Escalate to off-axis architecture probe (set-attention re-baseline) with the asymmetry as a known unexplained residual.

---

## 4. Ablation / decision rules

| Dn outcome | Action | Expected wl_lower lift | Implementer scope |
|---|---|---|---|
| D1 fails | Fix featurizer slot; re-fire v3.7 tight-gate. | +0.04 – 0.07 | hours; `features.py` + `featurize.rs` |
| D2 fails | Fix MCTS perspective; re-fire tight-gate. | +0.02 – 0.05 | days; `mcts/` |
| D3 fails | Recipe adjustment (balanced sampler); re-launch base loop + tight-gate. | +0.01 – 0.03 | recipe re-launch (~90 min wall) |
| D4 fails | Gate-harness fix; re-fire gate (no re-train). | +0.00 – 0.02 | hours; `eval_gate.rs` |
| D5 shows Δ_heuristic > 0 | Tighten the model-attributable ceiling; do NOT pursue lift past the residual. | n/a | docs-only |
| All pass, model still shows Δ | Unexplained residual; close probe; escalate to architecture-axis. | n/a | docs-only verdict |

**Composite outcome:** If multiple Dn fire, run fixes serially (cheapest first) and re-gate between each.

---

## 5. Out of scope (explicit)

- **No schema bump.** State-vector layout stays at v3.7 296-d. Any featurizer fix preserves slot positions; only mis-set slot values change.
- **No new training-axis lever.** This probe does not bump hidden_dim, depth, KL, or sims. (Those are the recipe-axis already exhausted in §4m.)
- **No two-sided MCTS toggle.** If D2 surfaces an MCTS-side bug, fix in single-sided mode first; two-sided is a separate scope (`two-sided-mcts-scoping.md`).
- **No new feature classes.** v3.7's bet on qualitatively-new classes already fired (§4n SOFT-SHIP). Adding more bits here would re-test §4n.
- **No deck-pair-sampling change.** Asymmetry is invariant to deck choice (reproduces across 9 distinct recipes including HP grid and cold-start) — sampling is not the lever.
- **First-mover vs seat disentanglement (D5b)** is in-scope but DEFERRED until D5 main result lands.

---

## 6. Constraints honored

- **No schema bump.** v3.7 byte freeze preserved unless D1 fix requires a slot-value change (which is by construction byte-stable on the layout, only changes the values written into existing slots).
- **`_SCHEMA_BY_STATE_DIM` dispatch unchanged.** No new state_dim entries.
- **Action-schema-version guard unchanged** (action schema stays at 3).
- **No mutation of historical `runs/` artifacts.** D3 is read-only over existing selfplay traces.
- **No orchestrator changes.** D1, D2, D4, D5 are bounded smokes / read-only audits; only D3 + a positive Dn might motivate a recipe-axis tweak.
- **No `mctsTwoSided` flip.** D2 uses single-sided mode (matches v3.7 gate config).

---

## 7. Conditional escalation paths

**If D1 fails:** Featurizer bug located. Implementer fixes the offending slot pair; v3.7 ckpt is preserved (slot values change at inference, no warm-start invalidation). Re-fire `runs/R16-P1-v37-cap128-iter6-tight-gate/` recipe with the fixed featurizer (~9 min wall). If wl_lower ≥ 0.6040 → **clears LIFT band** → v3.7 promotion-ready against 96-d production pin per `v37-combat-arith-and-catalog-scoping.md` §8 rule 1 conditions.

**If D2 fails:** MCTS-side bug. Implementer scope larger (mcts/driver.rs backup + child expansion). Re-fire tight-gate. Lift band same as above; partial lift more likely than full closure.

**If D3 fails:** Recipe-axis fix (sampler balance). Re-launch v3.7 cap128 base loop (~90 min wall) + tight-gate. Lift conservative.

**If D4 fails:** Gate-harness fix only; re-gate existing ckpt; no re-train. Lift conservative.

**If D5 shows Δ_heuristic ≥ 0.02:** Model-attributable ceiling shrinks. The lift band assumption of +0.04–0.07 (H1) tightens to +(0.04 − Δ_heuristic). Update bands and re-rank pyramid.

**If ALL pass:** Asymmetry is real but escapes the four candidate root-cause classes. Document as known unexplained structural residual in `progress/r110.md §4o`. Pivot to architecture-axis (set-attention Slice 2 re-baseline) per `v37` §8 rule 2 ordering, with asymmetry recorded as a known background variable.

---

## 8. Canonical file pointers

### Evidence of the asymmetry
- `runs/R16-P1-v37-cap128-iter6-tight-gate/gate.manifest.json` lines 55–75 — playerSide.wilsonLower=0.5556, opponentSide.wilsonLower=0.6105, balanced 5000+5000 tasks.
- `docs/ai-research/progress/r110.md:1300-1335` — §4m per-side asymmetry section + candidate ranking table.
- `docs/ai-research/progress/r110.md:1444-1450` — §4n confirmation (v3.7 does not move the gap).

### Featurizer code (D1 surface)
- `training/uma_ai/features.py:202-244` — `observation_to_features` v3.0 head, sets the side-tag conventions.
- `training/uma_ai/features.py:1912-1926` — `_first_player_polarity` (the one explicit polarity scalar on the v3.0 head).
- `training/uma_ai/features.py:233` — slot 96 polarity write.
- `engine-rs/crates/engine/src/policy/featurize.rs:208,1221-1223` — Rust mirror's side-tag handling.
- `engine-rs/crates/engine/src/policy/observation.rs:24-71` — `build_public_observation(state, side_id)` perspective construction (already takes `side_id` and swaps own↔opp).
- `engine-rs/crates/engine/src/policy/observation.rs:363-380` — existing `observation_swaps_perspective_when_side_changes` test (asserts JSON differs but does NOT byte-permute). D1 EXTENDS this with the permutation-symmetry assertion.

### MCTS code (D2 surface)
- `engine-rs/crates/engine/src/mcts/driver.rs:40-65` — root construction with `model_side`.
- `engine-rs/crates/engine/src/mcts/driver.rs:289-403` — backup loop sign-flip logic (single-sided vs two-sided branches).

### Eval-gate code (D4 surface)
- `engine-rs/crates/sim-cli/src/bin/eval_gate.rs:530-541` — task generation for `--model-side both`.
- `engine-rs/crates/sim-cli/src/bin/eval_gate.rs:334-385` — `drive_one_game` config invariance over `model_side`.

### Setup (intrinsic first-player rule)
- `engine-rs/crates/engine/src/dispatcher.rs:210-234` — coin-flip first-player selection. (Decouples `model_side` from `first_player`.)

### Research docs
- `docs/ai-research/scoping/v37-combat-arith-and-catalog-scoping.md` §8 rule 2 — explicit "off-axis pivot, asymmetry probe primary" guidance.
- `docs/ai-agent-state/queue.json:160-164` — queue entry `per-side-asymmetry-probe`; status remains there.
- `docs/ai-research/scoping/two-sided-mcts-scoping.md` — adjacent scope if D2 escalates beyond a single-sided fix.

### Verdict writeup destination (post-probe)
- `docs/ai-research/progress/r110.md` §4o (NEW) — when probe lands, append verdict section there per `docs/ai-research/README.md` routing.

---

## 9. What this doc is NOT

- **Not authorisation to land code.** Scope-only DRAFT; USER-GATED for approval before D1 fires.
- **Not a training-iteration arm.** No new training run is committed unless a Dn surfaces a fixable cause AND a re-fire is needed to confirm the lift.
- **Not a contradiction of the v3.7 SOFT-SHIP verdict.** v3.7 ships as the schema baseline per `v37-combat-arith-and-catalog-scoping.md` §12; this probe is off-axis and runs independently of any further schema work.
- **Not a substitute for the architecture-axis (set-attention Slice 2 re-baseline) or history-features (R7.b.4) lines.** Both remain queued; this probe is sequenced FIRST because of the ex-ante leverage table in `progress/r110.md:1324-1335` (~30–50% actionable × +0.02–0.04 wl_lower).
- **Not a commitment to find the bug.** "All Dn pass with model still showing Δ" is a valid landing state; the probe's value includes that falsification.

---

## 10. Status line

**`PROPOSED 2026-05-26`** — DRAFT scoping doc. Queue entry `per-side-asymmetry-probe` (`docs/ai-agent-state/queue.json:160-164`) remains canonical for status; this doc operationalises the diagnostic pyramid (D1 → D5). USER-GATED start at D1 (featurizer parity smoke). No training-run commitment. Verdict writeup destination: `docs/ai-research/progress/r110.md` §4o. Implementer handoff scope per §3 / §4 conditional table.
