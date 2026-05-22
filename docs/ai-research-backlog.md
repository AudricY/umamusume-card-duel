# AI Research Backlog

This is the forward-looking model-strength research backlog. Keep it compact:
landed or failed work shrinks to a pointer, and detailed mechanisms/numbers live
in progress, scoping, or analysis docs. See `docs/ai-research/README.md`.

## Current Anchor

**Two user directives 2026-05-22 — both must hold for any new training.**

**(A) No more AI research or training without deck variety.** Every post-R110
measurement was taken on a single near-mirror Matikane vs Matikane matchup;
`defaultAiOpponentDeckId='riceShowerHaruUrara'` does not match any id in
`aiPremadeDecks`, silently falling back to AI-flavor Matikane. 11 AI decks and
2 player decks otherwise unused in sim. Featurizer is deck-agnostic (verified
zero OOV across all decks). Training-bearing forward lines (W6-fix HP sweep,
per-game-PFSP league retry, value-head data-program Stages 2/3, any new
self-play loop) are BLOCKED until `deck-pair-sampling` Slices P0+1+2 land.
Canonical scope: `docs/ai-research/scoping/deck-pair-sampling.md`; queue
`deck-pair-sampling` (P1).

**(B) All training and research uses per-Uma slot tokens (v3.2) from now on.**
`uses_uma_slot_tokens=True` is mandatory for any new training; v3.0 (96-d /
110-d) and v3.1 (164-d) become historical baselines only. v3.2 infra is
landed end-to-end (C1-C7 bit-exact, Rust featurizer Slice 3, `serve_onnx`
schema 3.2 dispatch). The existing 0.5811 tight-gate ceiling is a v3.0
reference; **the v3.2 ceiling is unknown**. Tight-gate re-verdict #3 (C8
iter-0, v3.2, n=120 `wl=0.5955`) is now the highest-leverage next measurement
— it establishes the v3.2 ceiling under the current production rollout-leaf
recipe. Production pin **stays at 96-d v3.0 (0.6479)** until a v3.2 ckpt
clears it on the diverse-matchup gate.

**Combined gate for any new training:** v3.2 architecture **AND**
deck-variety sampling active. Measurement-only work on existing checkpoints
(re-verdict #3 on v3.2; high-sim regime probe on v3.2 ckpts) remains
unblocked.

As of 2026-05-18:

- **Production stays pinned 96-d.** Rollout-leaf MCTS at W6 iter-2 is the
  current production claim: Wilson lower 0.6479 vs rule bot at the
  side-balanced n=120 gate. Canonical details:
  `docs/ai-research/progress/r15.md` and the R1-R14 historical record in
  `docs/ai-performance-research-progress.md`.
- **The iter-2-peak-then-regress is now a characterized schema-independent
  recipe property.** Reproduced across BOTH 96-d and 110-d v3; 0.6479 is
  itself W6's *transient iter-2 peak*, not a stable optimum. The R110 v3
  reproduction was MARGINAL (best-promoted iter-2 0.6042 at n=120). Tight-gate
  re-verdict 2026-05-21 at n=10,000 resolved the v3 ceiling to **Wilson-lower
  0.5811 / point 0.5908 / upper 0.6004** — 0.6042 was n=120 high-tail noise.
  Pin position unchanged; strength claim downgrades to MARGINAL-WEAK. Where
  this backlog reads "0.6042 iter-2 ceiling" below, the honest comparator is
  now 0.5811. Canonical: `docs/ai-research/progress/r110.md` §1 + §4d.
- **The search-free/raw-policy SL line is closed.** Labels, objective,
  representation, and label-shape all failed to approach the 0.40 gate
  (R7/R8/R7.b.2/mcts-distill v1). Do not spend compute on another isolated
  raw-policy tweak without explicit new evidence.
- **Cheap inference remains a fallback, not the production claim.** Value-head
  leaf plus adaptive-ratio=1.5 sits around the 0.39-0.45 Wilson-lower envelope
  across seed ranges and is only a latency fallback.
- **Side asymmetry is real enough to disclose.** Current rollout-leaf evidence
  is aggregate side-balanced strength, not a per-side guarantee.
- **User reprioritization (2026-05-19): hyperparameter tuning is
  deprioritized.** Land ALL model-feature-upgrade and training-data-upgrade
  items first, then return to the W6 regularization-dose sweep. The active
  frontier is now (1) the cheap contested-state coverage pilot, then (2) the
  R16-P2 per-Uma slot-token model-feature migration, then downstream data
  items; the W6 loop anti-degradation **regularization-dose sweep is the
  deprioritized HP tuning** and runs only after that block lands (still
  user-gated). R16-P1 temporal/turn-state features were implemented and
  **ablated NO-GO** (v3.1 ≈ v3.0, no Wilson-lower win;
  `docs/ai-research/progress/r16.md`). Live operational state and forward
  order are in `docs/ai-agent-state/queue.json`.
- **Post-data-work ceiling path (2026-05-21 sequencing refinement).** The
  iter-2-peak-then-rot ceiling has two layered fixes, cheapest first:
  **(A)** W6 recipe-fix HP sweep — already-landed flags (cross-iter replay
  + fixed KL anchor, commit `d44de3f`) at tuned-down strengths; cheap,
  in-methodology, R110 A/B chain intact. **(B)** Value-head leaf at MCTS
  (item 0d below) — R111 recipe-axis change; heavy, breaks the R110 A/B
  chain. Run (B) only if (A) cannot clear the 0.6042 iter-2 ceiling — pay
  the methodology-break cost at most once. This sequence sits behind the
  active data block; nothing changes about the current forward order, only
  what comes after it.

## Active Search-Wrapped Frontier

0. **Contested-state coverage pilot — DONE-FALSIFIED 2026-05-21** (chunk 5j
   n=1000; monotonicity broken, margin sign-flipped −1.8pp). Canonical:
   `docs/ai-research/progress/r16.md`.

0a. **R16-P2 per-Uma slot tokens — DONE-MIXED-SIGNAL 2026-05-21.** C1-C7
   landed bit-exact; C8 acceptance NO-GO at best v3.2 wl 0.5955. C8 iter-0 is
   the re-verdict #3 candidate (+0.0144 above 0.5811 tight-gate ceiling,
   within sampling noise). Queue `r16-p2-per-uma-slot-tokens`; canonical
   `docs/ai-research/progress/r16.md`.

0b. **W6 loop anti-degradation recipe — DEPRIORITIZED ordering, REFRAMED
   role (ceiling-path step A).** Demoted from TOP PRIORITY to P3 by the
   2026-05-19 user reprioritization: runs only AFTER all three core
   predecessors land (coverage pilot → R16-P2 → deep data program: corpus
   recipe → preference/value-data), and stays user-gated. **Strategic role
   refined 2026-05-21**: this is not just deprioritized HP tuning — it is
   the *cheap, in-methodology, R110-A/B-preserving* first attempt at the
   iter-2-peak-then-rot ceiling. One or two HP runs decides whether
   containment alone clears 0.6042; if it does NOT, escalate to item 0d
   (value-head leaf at MCTS, the heavy recipe-axis fix). Gate-depth = option
   B: it is **NOT** gated on the conditional side-balancing item or the P3
   manual mistake catalog (those fire on their own triggers; gating W6
   behind the conditional item could block it indefinitely). Operational
   predecessor set is canonical in `docs/ai-agent-state/queue.json`.
   Mechanism still CONFIRMED via R111 (iter-3 rot eliminated; recipe-fix
   landed commit `d44de3f`, cross-iter replay buffer + fixed iter-0/SL KL
   anchor in `r12_orchestrator.py`) but over-damps at default HP — policy
   froze then decayed (0.5527→0.5358), below the 0.6042 baseline iter-2
   peak. Reopen knobs: lower/anneal the fixed-KL-anchor weight from 0.05
   and/or reduce replay old-fraction (0.40) / window (3). Canonical:
   `docs/ai-research/progress/r110.md` §4c; queue `w6-loop-anti-degradation`.

0d. **Value-head leaf at MCTS — CLOSED as strength lever 2026-05-21.**
   vhleaf loop on the C8-W6FIX-ON seed descended after iter-1, best
   below R14.A 0.452 baseline; "calibration was the missing piece"
   falsified. Queue `value-head-leaf-recipe-axis`; canonical
   `docs/ai-research/progress/r16.md`.

0c. **Fork B — label-quality test matrix.** Umbrella over W6 dose sweep;
   deprioritized with the sweep, gated behind a near-zero-cost read-only
   label-quality probe. Scope:
   `docs/ai-research/scoping/forkb-label-quality-loop-recipe.md`.

1. **R110 W6 reproduction — DONE-MARGINAL.** Best-promoted iter-2 wl=0.6042
   (n=120), tightened to 0.5811 (n=10,000) by re-verdict #1; pin unchanged.
   Canonical: `docs/ai-research/progress/r110.md`.

2. **R16-P1 temporal/turn-state features — ABLATED NO-GO.** v3.1 164-d
   produced no Wilson-lower win; re-verdict #2 confirmed v3.1 = v3.0 at
   n=10,000. Representation axis CLOSED. Canonical:
   `docs/ai-research/progress/r16.md`.

3. **GPU-fed stronger MCTS — P1 scoping, not implementation.**
   First prove which search configs can use GPU inference for strength:
   batched predict, hybrid learned leaves, larger sim budgets, ensembles, or
   root-search variants. Promotion requires a search-wrapped gain over
   production, not raw-policy WR. Seed:
   `docs/ai-research/scoping/gpu-fed-stronger-mcts.md`.

3b. **Set-attention architecture probe (R7.b.3) — P3 SCOPED 2026-05-22,
    user-gated.** Architecture-axis sibling to the recipe-axis lines
    (0b W6 HP sweep, 0d value-head-leaf closed). Tests a 1-2 layer MHA
    trunk over per-card+per-slot tokens vs the current sum-pool prior;
    raw-policy SL plateau at wl≤0.33 across R7/R8/R7.b.2/mcts-distill-v1
    and the schema-independent iter-2-peak-then-rot across v3.0/v3.1/v3.2
    sum-pool variants motivate it. NOT a capacity probe (R6 closes
    capacity-from-above; param delta ≤ +80K). Gated behind
    deck-pair-sampling Slice 2 + re-verdict #3 + W6 HP sweep + PFSP
    league retry. Canonical scope:
    `docs/ai-research/scoping/set-attention-architecture-probe.md`;
    queue `set-attention-architecture-probe`.

4. **Value/action-value data program — P2 (3-stage ladder).**
   Data/training prerequisite for item 0d (value-head leaf at MCTS). Three
   sub-stages, cheapest first; each falsifies one hypothesis about why the
   value head is currently too noisy for MCTS leaves. Historical evidence
   that constrains the ladder: R13.W3 retrain on R4-era rollout means moved
   Brier 0.243→0.084 but value-head-leaf MCTS strength stayed below the
   production bar (PARTIAL); R14 crossover probe across W6 iter-1→iter-4
   never satisfied `val_mse ≤ 1.10 × W3-floor AND pearson ≥ 0.7` (pearson
   stuck ~0.5, ratio 1.18–1.29); the +18pp rollout-leaf iter-1→iter-2 gain
   (0.452→0.6479) did **not** transfer to value-head-leaf inference (held
   flat 0.404–0.452). Throwing epochs at the existing head/objective/corpus
   recipe is closed (`docs/ai-performance-research-progress.md:1803, 1818,
   1870, 1946`).
   - **Stage 1 — post-W6 corpus retrain — DONE-POSITIVE 2026-05-21** via
     C8-W6FIX-ON iter-1 (R14 crossover gate accidentally satisfied; ratio
     0.722 ≤ 1.10 AND pearson 0.725 ≥ 0.70). Downstream vhleaf consumer
     hit 0.40 production bar but fell short of R14.A 0.452 baseline.
   - **Stages 2 (action-value Q-head) and 3 (architectural change)** —
     user-gated for scope; both now also bound by directives A+B (deck
     variety + v3.2). Queue: `value-head-data-program`.

### 4b. Rust engine port follow-ups (post Phase-2-merge, 2026-05-21)

Engine-rust-port branch merged into feat/ai as commit `4e86ea7`. Phases 0/1/2
all complete (canonical home: `docs/ai-research/scoping/rust-engine-port-handoff.md`).
Forward items, ordered by user prioritization 2026-05-21:

- **Orchestrator wiring** (P1, user-approved next-up): `r12_orchestrator.py` (+ppo+dagger) still invoke TS `npm run sim:*`. Rust sim CLIs offer 140-220× via flag-compatible aliases. Add opt-in `--engine rust|ts` with A/B parity gate (1-iter R110 in both engines, compare trajectory rows + eval-gate verdict). Sequenced ahead of P1 research items because the speedup compounds across tight-gate-reverdict-program, high-sim-mcts-regime-probe, side-conditioned eval (item 5), PPO/RL feasibility (item 6), and W6 HP-sweep density (item 0b).
- **v3.2 schema verification** (P2, step-1 pre-C8 feasible): Rust `/predict` client emits `schemaVersion=3` + `cardIdsByZone` only, no `uma_slot_*`. Step-1 (read whether `serve_onnx.py` synthesizes slots server-side from `PublicObservation`) is a cheap code-check doable now as insurance. Step-2 (extend Rust observation builder) gated on C8 promoting v3.2.
- **Backend NAPI consumer** (P3, exploratory): no `backend/src/` consumer yet. Defer.
- **Engine residual TODO** (P4): `has_consecutive_no_attack_streak: u8` on `SideState` (V4 RNG-gap reframed as benign behavioral variance).

5. **Side-conditioned sim budget — P2, throughput-unlocked.**
   Side-split CI separation was the throughput-driven gate. Rust 140x makes
   side-conditioned eval routine; flips from 'blocked on evidence' to
   'actionable post `rust-port-orchestrator-wiring`'.

6. **RL/PPO from a strong search-wrapped checkpoint — P3.**
   Later strategic bet. Do not use it to bypass search-wrapped gates above.
   Throughput-unlock note: Rust 140x partially lifts the self-play barrier
   that made PPO Phase J prohibitive. Worth a fresh scoping pass after
   `rust-port-orchestrator-wiring` lands.

6b. **Deck-pair sampling for self-play and eval — P1, BLOCKING all new training (user directive 2026-05-22A).** Combines with directive 2026-05-22B (v3.2-only): any new training runs under v3.2 architecture AND uniform deck sampling.
   Surfaced 2026-05-22 via deck-selection investigation: every post-R110
   measurement (the 0.5811 tight-gate ceiling, the iter-2-peak-then-rot
   characterization, the v3.0/v3.1/v3.2 representation verdicts) was taken on
   a single near-mirror Matikane vs Matikane matchup. The opponent default
   `defaultAiOpponentDeckId='riceShowerHaruUrara'` does not match any id in
   `aiPremadeDecks`; the fallback silently lands on `ai_decks.first()`
   (AI-flavor Matikane). 11 AI decks and 2 player decks otherwise unused in
   sim. Featurizer is deck-agnostic (global card vocab covers every card in
   every deck, zero OOV verified). Re-surfaces archived items §6 + §13 from
   `docs/archive/ai-research/ai-performance-research-backlog.md`. Lands
   ADDITIVE — fixed-matchup tight-gates stay; diverse-matchup gate is a second
   gate. Scope: `docs/ai-research/scoping/deck-pair-sampling.md`; queue
   `deck-pair-sampling`. P0 (defaults-bug fix + deck-legality smoke) is the
   cheapest autonomous step; Slices 1-3 are user-gated.

7. **Per-game PFSP league retry — P3, deferred-revisit.**
   Infra already built (`training/opponent_pool.py`: snapshots, PFSP weights,
   retention cap, cycling alarm, JSON persistence; wired into
   `ppo_orchestrator.py` + `dagger_orchestrator.py`). Two prior negatives:
   R5 weak-pool missed gate; PPO Phase J strong-pool regressed iter-2 and
   missed the 0.40 gate by 18pp (`docs/ai-performance-research-progress.md:2114-2127`).
   Diagnosis attributed the Phase J failure to a **per-RUN sampler with
   self-promoted-at-mode** (`progress.md:808-811`); open hypothesis is that
   per-GAME re-sampling + strict PFSP weight enforcement would rescue it. Do
   NOT auto-launch. Current dominant failure mode (W6 iter-2-peak-then-rot) is
   representation drift, not non-transitivity (`r12_orchestrator.py:53-69`);
   `cycling_alarm` has not fired in any committed run log — no measured
   league-shaped symptom in-tree. Revisit only after state-coverage line
   (items 0/0a + TD 2-3) resolves AND explicit user go-ahead. Throughput
   note (2026-05-21): the 'no league-shaped symptom in-tree' justification
   was partly throughput-driven. At Rust 140x, thousand-game tournaments
   become cheap; revisit only as part of a deliberate strategic call, still
   user-gated.

## Training-Data / State-Coverage Backlog

**Fork A — contested-state coverage** frames items 1–2 below as its
sub-tasks. Metric = retained `>=4`-legal fraction, floor `>=30%` (audit's
existing `legal_action_count` target); cheap-experiment design and
closed-axes non-goals (no capacity, no generic volume, no `min_actions`
relaxation) in
`docs/ai-research/scoping/r16-training-data-backlog-refinement.md` (Fork A
section).

1. **Corpus retention + state-overlap audit — DONE.** Finding: retained rows
   bottlenecked by contested decision-state coverage. Report:
   `docs/ai-research/analysis/training-data-coverage-audit.md`.

2. **Rule-bot-covered corpus + DPO preference pairs — SUPERSEDED 2026-05-21**
   by `training-data-deep-program` done-negative. High-sim-regime caveat
   feeds `high-sim-mcts-regime-probe`. Canonical:
   `docs/ai-research/progress/r16.md`.

4. **Side-conditioned retained-data balancing — P2, conditional.** Fires
   only if side weakness proves data-linked; balance retained player/opponent
   decision states within major phase/action buckets. Not a W6 predecessor.

5. **Rule-bot mistake catalog + forced-state suite — P3 manual.** Build
   hand-audited tactical states explaining where MCTS beats the rule bot;
   feeds fixture/eval tooling. Not a W6 predecessor.

## Guardrails

- Do not relax `min_actions=1`; forced single-action states carry no policy
  gradient and would dilute training.
- Do not schedule self-play-only data regeneration for raw-policy SL; the
  mcts-distill v1 failure is the canonical negative example.
- Do not re-open raw-policy SL unless a new coverage result explicitly
  falsifies the current diagnosis.
- Do not auto-launch the W6 regularization-dose sweep. Per the 2026-05-19
  user reprioritization it is DEPRIORITIZED (P3): it runs only AFTER all
  three core predecessors land (coverage pilot → R16-P2 → deep data
  program) AND with explicit go-ahead. Gate-depth = option B: do NOT extend
  the W6 gate to the conditional side-balancing item or the P3 manual
  mistake catalog.
  (The R16-P1 v3.1 ablation is now CLOSED = NO-GO; no auto-launch remains on
  that line — `docs/ai-research/progress/r16.md`.)
- R16-P2 per-Uma slot tokens: the prior "do not start unless the user
  explicitly calls for it" gate is now SATISFIED by the 2026-05-19
  reprioritization. Still sequence it AFTER the cheap coverage pilot
  (leverage-per-cost); it remains a 3–4-day model/ONNX migration.
- Do not re-open representation/capacity tuning off the R110 MARGINAL band;
  the forward line is the loop-recipe axis only (`docs/ai-research/progress/r110.md`).
- Carve-out from "do not re-open representation/capacity tuning":
  set-attention architecture probe (item 3b) is the *trunk-shape* axis
  (inductive bias), not capacity tuning and not feature-schema; user-gated,
  does not auto-launch. Canonical scope:
  `docs/ai-research/scoping/set-attention-architecture-probe.md`.

## Historical Pointers

- R1-R14 evidence: `docs/ai-performance-research-progress.md`.
- R15 result blocks: `docs/ai-research/progress/r15.md` and
  `docs/ai-research/progress/r15-archive.md`.
- R110 W6-repro verdict + recipe mechanism: `docs/ai-research/progress/r110.md`.
- R16-P0/P1 results (P1 v3.1 ablation NO-GO): `docs/ai-research/progress/r16.md`.
- Closed scoping docs: `docs/ai-research/scoping/archive/`.
- Closed sprint/design/proposal docs: `docs/archive/ai-research/`.
