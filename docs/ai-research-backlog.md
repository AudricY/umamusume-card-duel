# AI Research Backlog

This is the forward-looking model-strength research backlog. Keep it compact:
landed or failed work shrinks to a pointer, and detailed mechanisms/numbers live
in progress, scoping, or analysis docs. See `docs/ai-research/README.md`.

## Current Anchor

As of 2026-05-18:

- **Production stays pinned 96-d.** Rollout-leaf MCTS at W6 iter-2 is the
  current production claim: Wilson lower 0.6479 vs rule bot at the
  side-balanced n=120 gate. Canonical details:
  `docs/ai-research/progress/r15.md` and the R1-R14 historical record in
  `docs/ai-performance-research-progress.md`.
- **The iter-2-peak-then-regress is now a characterized schema-independent
  recipe property.** Reproduced across BOTH 96-d and 110-d v3; 0.6479 is
  itself W6's *transient iter-2 peak*, not a stable optimum. The R110 v3
  reproduction was MARGINAL (best-promoted iter-2 0.6042) and not promoted.
  Canonical: `docs/ai-research/progress/r110.md`.
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

0. **Contested-state coverage pilot (R16-TD)** — DONE-FALSIFIED 2026-05-21
   (chunk 5j n=1000 confirmation, monotonicity broken + margin sign-flipped
   −1.8pp). See `docs/ai-research/progress/r16.md`. **Tight-gate re-verdict
   candidate**: kill verdict held at the throughput-constrained n; not on the
   immediate re-verdict shortlist but worth flagging.

0a. **R16-P2 per-Uma slot tokens — model-feature upgrade, user-gate now
   OPEN.** The 2026-05-19 "land all model feature upgrade items first"
   directive is the explicit user call the prior guardrail required. A 3–4-day
   model/ONNX migration; sequence *after* the cheap coverage pilot
   (leverage-per-cost). Scope: `docs/ai-research/scoping/r16-model-feature-backlog-refinement.md`
   § "P2 - Per-Uma Slot Tokens"; queue `r16-p2-per-uma-slot-tokens`.

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

0d. **Value-head leaf at MCTS — ceiling-path step B (R111 recipe-axis).**
   Surfaced 2026-05-21 as its own forward-line entry (previously buried as a
   one-line pointer inside the DONE `r12-throughput` queue entry). Root
   mechanism behind the iter-2-peak-then-rot ceiling: in rollout-leaf mode
   the trained value head is never used at gate/self-play
   (`backend/src/sim/mcts.ts:552-555`), so the loop has no improving signal
   feeding back into MCTS across iters (rule-bot rollouts at leaves are
   fixed quality). Re-enabling the value head at leaves gives the loop a
   positive gradient toward ground-truth `z`, which is what lets visits
   ratchet *sharper* across iters instead of softer. Trigger condition:
   item 0b HP sweep ran and did NOT clear the 0.6042 ceiling. COST: changes
   the learning targets, breaks the R110 A/B chain (96-d 0.6479 production
   claim is rollout-leaf-based); any run on this axis establishes a new
   baseline. **Predecessor data program (item 4 below) is the binding
   constraint, not in flight.** All prior value-head-leaf attempts (R13.W3
   PARTIAL, R13.W8 loop regressed, R14 crossover never crossed) used the
   shared-trunk single-scalar head trained on R4-era data; throwing epochs
   at that recipe is closed. Item 4 enumerates the 3-stage ladder
   (post-W6 corpus retrain → action-value head → architectural change) that
   produces a value head this item can consume. The `training-data-deep-
   program` 3b arm tested only the *policy* side (DPO + soft-CE BC) and is
   now done-negative; the value side never ran. DEFERRED + USER-GATED.
   Scope pointer: `docs/ai-research/scoping/r12-selfplay-gate-throughput.md`
   item 4; queue `value-head-leaf-recipe-axis`.

0c. **Fork B — label-quality test matrix (umbrella over the W6 dose
   sweep).** Raises target/teacher quality at *fixed* capacity and volume
   (both axes closed); the M6 anti-drift arm is the W6 dose sweep, not a
   re-plan. Deprioritized with the W6 sweep above; gated behind a
   near-zero-cost read-only label-quality probe before any user-gated loop
   compute. Canonical:
   `docs/ai-research/scoping/forkb-label-quality-loop-recipe.md`.

1. **R110 W6 reproduction — DONE.**
   Verdict MARGINAL (best-promoted iter-2 Wilson lower 0.6042, in the
   0.60-0.6479 band); reproduced-but-not-superior, not promoted; pinned 96-d
   stays production. R16-P0 embedding fix confirmed working.
   `docs/ai-research/progress/r110.md`.

2. **R16-P1 temporal/turn-state features — ABLATED NO-GO.**
   v3.1 164-d implemented end-to-end (commits `357f0d6`/`3ce1404`); the
   faithful strength ablation produced **no Wilson-lower win** (v3.1 ≈ v3.0,
   best-promoted 0.5955 < v3.0 0.6042). v3.1 NOT promoted; production stays
   pinned 96-d. Null strength signal. Canonical:
   `docs/ai-research/progress/r16.md`.

3. **GPU-fed stronger MCTS — P1 scoping, not implementation.**
   First prove which search configs can use GPU inference for strength:
   batched predict, hybrid learned leaves, larger sim budgets, ensembles, or
   root-search variants. Promotion requires a search-wrapped gain over
   production, not raw-policy WR. Seed:
   `docs/ai-research/scoping/gpu-fed-stronger-mcts.md`.

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
   - **Stage 1 — post-W6 corpus retrain (cheapest, ~4h compute).** Same head
     architecture, same objective (regress to rollout-CRN means), same
     freeze-trunk-and-policy protocol as R13.W3, but on the W6 corpus
     (sharper rollout targets than R4). Gate = R14 crossover probe
     (`training/r14_value_crossover_probe.py`) crossed = true on a
     held-out slice. Outcome falsifies "the W3 PARTIAL result was data-
     limited." If crossed, attempt the value-head-leaf gate; if not,
     escalate to Stage 2.
   - **Stage 2 — action-value (Q) head + per-action rollout targets.**
     Changes the *target*: corpus emits per-legal-action rollout-CRN means
     instead of one state-value scalar, and the model gains a Q(s,a) head
     consumed at MCTS leaves directly (one rollout per legal action at
     corpus-generation time; head is fed per-action embeddings or
     action-conditioned). This is the AlphaZero→MuZero-style move and is
     the closest match to what MCTS leaves actually want. Outcome
     falsifies "the state-value objective was the bottleneck." Bigger
     scope: corpus regeneration + head architecture + serve_onnx schema
     bump.
   - **Stage 3 — architectural change (separate value tower / deeper head
     / inference-time variance reduction).** All prior heads were
     shared-trunk single-scalar with the same depth. Possible moves:
     dedicated value tower, deeper value head, dropout-at-inference as
     ensemble proxy, or auxiliary calibration loss. Outcome falsifies
     "shared-trunk capacity was the bottleneck." Largest scope and most
     speculative — only fires if Stages 1 and 2 both fail to lift
     value-head-leaf MCTS strength to the 0.40 production bar.

   **Stage 1 PROMOTED TO AUTONOMOUS P1 (2026-05-21)** per user lift —
   cleanest properties of any line in the queue (cheap, falsifiable via the
   R14 crossover probe, A/B-comparable, no R110 chain break, no new code
   beyond a corpus-pointer flag); ~4h GPU. Stages 2 and 3 remain
   user-gated for scope. Queue: `value-head-data-program`.

### 4b. Rust engine port follow-ups (post Phase-2-merge, 2026-05-21)

Engine-rust-port branch merged into feat/ai as commit `4e86ea7`. Phases 0/1/2
all complete (canonical home: `docs/ai-research/scoping/rust-engine-port-handoff.md`).
Forward items:

- **v3.2 schema verification** (P2, gated on C8 promotion): Rust `/predict` client emits `schemaVersion=3` with `cardIdsByZone` only; no `uma_slot_*`. Verify whether `serve_onnx.py` v3.2 path synthesizes slot tensors server-side from `PublicObservation`; if not, extend Rust observation builder.
- **Orchestrator wiring** (P2, user-gated pilot): `r12_orchestrator.py` (+ppo+dagger) still invoke TS `npm run sim:*`. Flag-compatible Rust CLIs offer 140-220× speedup; add opt-in `--engine rust|ts` with A/B parity gate.
- **Backend NAPI consumer** (P3, exploratory): no `backend/src/` consumer yet; pick a target after orchestrator wiring lands.
- **Engine residual TODO** (P4): `has_consecutive_no_attack_streak: u8` field on `SideState` (V4 RNG-gap reframed as benign behavioral variance, but field still missing if anyone wants exact parity).

5. **Side-conditioned sim budget — P2, throughput-unlocked.**
   Side-split CI separation was the throughput-driven gate. Rust 140x makes
   side-conditioned eval routine; flips from 'blocked on evidence' to
   'actionable post `rust-port-orchestrator-wiring`'.

6. **RL/PPO from a strong search-wrapped checkpoint — P3.**
   Later strategic bet. Do not use it to bypass search-wrapped gates above.
   Throughput-unlock note: Rust 140x partially lifts the self-play barrier
   that made PPO Phase J prohibitive. Worth a fresh scoping pass after
   `rust-port-orchestrator-wiring` lands.

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

1. **Corpus retention + state-overlap audit — DONE.**
   Canonical report:
   `docs/ai-research/analysis/training-data-coverage-audit.md`. Finding:
   retained rows are bottlenecked by contested decision-state coverage, not
   schema/card-id loss or generic row count.

2. **Rule-bot-covered state corpus recipe** — SUPERSEDED 2026-05-21 by
   `training-data-deep-program` done-negative (chunks 5a-i). See
   `docs/ai-research/progress/r16.md`. **High-sim-regime caveat**: the
   deep-program kill criterion (argmax-flip at 4x sims) was measured at
   400-800 sims; at 50,000 sims the answer may differ — see queue
   `high-sim-mcts-regime-probe`.

3. **Preference pairs on rule-bot-covered states (DPO/BT)** — SUPERSEDED
   2026-05-21 by `training-data-deep-program` done-negative (chunks 5a-i).
   See `docs/ai-research/progress/r16.md`. **Same caveat as item 2 above.**

4. **Side-conditioned retained-data balancing — P2.**
   If side weakness is data-linked, balance retained player/opponent decision
   states within major phase/action buckets and require side-split gate
   reporting. **Conditional + NOT a W6 predecessor** (option B): it fires
   only if side weakness proves data-linked; gating the W6 sweep behind it
   could block W6 indefinitely.

5. **Rule-bot mistake catalog + forced-state suite — P3.**
   Build hand-audited tactical states that explain where MCTS beats the rule
   bot and feed the fixture/eval tooling backlog. **P3 manual + NOT a W6
   predecessor** (option B): fires on its own track, not in the HP-gate
   chain.

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

## Historical Pointers

- R1-R14 evidence: `docs/ai-performance-research-progress.md`.
- R15 result blocks: `docs/ai-research/progress/r15.md` and
  `docs/ai-research/progress/r15-archive.md`.
- R110 W6-repro verdict + recipe mechanism: `docs/ai-research/progress/r110.md`.
- R16-P0/P1 results (P1 v3.1 ablation NO-GO): `docs/ai-research/progress/r16.md`.
- Closed scoping docs: `docs/ai-research/scoping/archive/`.
- Closed sprint/design/proposal docs: `docs/archive/ai-research/`.
