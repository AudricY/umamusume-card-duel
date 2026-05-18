# mcts-distill — Scoping

- **Date:** 2026-05-14
- **Status:** CLOSED / FAIL as standalone raw-policy scoping. Status update
  2026-05-15: mcts-distill v1 landed at raw-policy Wilson 0.1470 and is part
  of the closed raw-policy SL line. Keep this file as historical scoping only.
  Future use of `mcts-distill` is allowed only inside an iterated search-
  wrapped loop or a coverage-backed reopen proposal; it is not a standalone
  raw-policy north-star path.
- **One-liner:** Distill rollout-leaf MCTS soft-visit-distribution policy targets into the BC trainer to test whether the F1 raw-policy ceiling (Wilson 0.30-0.37 across labels/objective/representation axes) is binding on label *shape* (argmax vs soft visits). Pre-existing W6 corpus is reusable; experiment cost <30 min compute.
- **Forward brief:** queue entry `mcts-distill-scoping`.
- **Pick rationale:** F1 line exhausted across three axes (R7 labels 0.2921, R8 objective 0.3318, R7.b.2 representation 0.3045). Rollout-leaf MCTS at R12-W6/iter-2 reaches search-wrapped Wilson 0.6479 — the network family plays strong with search-time compute. `train_bc.py --data-mode mcts-distill` already wired; W6 selfplay corpus on disk.

## 1. Question (Q)

> Does training BC on rollout-leaf MCTS soft-visit-distribution labels — instead of single/multi-teacher hard-argmax labels — clear the F1 raw-policy iter-0 SL gate (Wilson lower ≥ 0.40)?

**Falsifiable proposition.** A `train_bc.py --data-mode mcts-distill` retrain from `runs/item17-2026-05-11/iter-002/model/checkpoint.pt` on the combined `runs/R13-W6-phase-d/iter-{0,1,2}/selfplay.jsonl` corpus (~6216 rows), at phase-H-scale (75 epochs, hidden 64, depth 2, KL anchor 0.05), evaluated at the F1 standard iter-0 SL gate (n=1000 side-balanced, rule-bot, seed-start 9000) via `training/r8_gate_eval.py`, reaches Wilson lower ≥ 0.40.

## 2. Hypothesis + Motivation

**Hypothesis.** F1 raw-policy caps at 0.30-0.37 because the *shape* of the supervised target (one-hot argmax of a moderate-strength teacher's chosen action) is too coarse for the policy to absorb good move-ordering. A normalized visit-count distribution from rollout-leaf MCTS gives the policy access to relative confidence across legal actions, not just argmax winners.

**Why now.** R7 (labels), R8 (objective), R7.b.2 (representation) all measured iter-0 Wilson in 0.29-0.33. None cleared 0.37. The surviving axis distinct from those three is **label-shape**: soft visit distribution vs argmax hard target. MCTS-distill is the natural test.

**Strength.** Rollout-leaf MCTS at W6/iter-2 reaches search-wrapped Wilson 0.6479. The network *family* can be very strong if labels carry search-derived ordering. Selfplay corpus already exists from R13's W6 Phase D run (6216 rows × 100 sims × rollout-leaf K=3).

**Weakness.** W6/iter-2 raw-policy (no MCTS at eval) was Wilson 0.1455 at Phase J (`docs/ai-performance-research-progress.md:763`), but **n=20 only** — that's a soft anchor, not a ceiling. The real n=1000 raw-policy Wilson could plausibly land anywhere in 0.15-0.40. The experiment resolves that uncertainty.

## 3. Pre-existing Evidence

- W6 Phase D (R13) ran mcts-distill 5 iterations: iter-{0,1,2} search-wrapped Wilson = **0.536 / 0.570 / 0.6479**; iter-3/4 regressed to 0.5783 / 0.5612 (co-adaptation). Production pick is iter-2 ckpt.
- W6/iter-2 raw-policy Wilson @ n=20 (Phase J PPO warm-start eval): **0.1455**.
- R12 Phase C SL smoke landed mcts-distill at 72% policy accuracy (2 epochs, smoke only).
- **No n=1000 raw-policy gate has been run on a fresh BC retrain from item17 using mcts-distill labels.** That is the actual unknown this scoping doc resolves.

## 4. Corpus Source

- **Primary (v1):** `runs/R13-W6-phase-d/iter-{0,1,2}/selfplay.jsonl` (~6216 rows combined). `kind=mcts-selfplay`, soft visit distributions over legal actions, 100 sims × rollout-leaf K=3, temperatureMoves=6 + Dirichlet root noise α=0.3 ε=0.25.
- **Optional 12k-row extension (gated on v1 marginal):** `mctsSelfPlay.ts` regen from `runs/R13-W6-phase-d/iter-2/checkpoint.pt` (~55 min, 4 workers).
- **No rule-bot mixing for v1.** The mcts-selfplay corpus has zero rule-bot-vs-model trajectory rows. State-coverage mismatch with the SL gate is a known risk (§ 5(c)); v1 measures it.

## 5. Risks

- **(a) Self-distill / fixed-point.** PUCT prior is model-driven (`mcts.ts:182,396-399`); leaf value is rule-bot rollout (`mcts.ts:589-590,597-610`). Leaf signal is model-independent — distillation can pull genuine ordering from search. Empirical W6 iter-2→3→4 regression shows the corrective signal fades over iterations; single iter from item17 should still be productive.
- **(b) Soft vs argmax target shape.** `selfplay_dataset.py:90-95` produces normalized visit distribution. F1 corpora used hard-argmax (R7 used 3-teacher uniform mix, also one-hot-shaped). Soft visits is a different loss surface. `train_bc.py:362-369` soft-CE branch handles it.
- **(c) State coverage mismatch — biggest risk.** Mcts-selfplay rows are MCTS-vs-MCTS trajectories with temperature + Dirichlet noise. The SL gate evaluates raw-policy vs rule-bot. The W6/iter-2 search-wrapped 0.6479 vs raw 0.1455 (n=20) gap is consistent with this biting. **If v1 lands < 0.30, state-coverage is likely the binding constraint** and the next step is rule-bot-mixed corpus rather than a different label shape.

## 6. Exit Gate

- **PASS** (Wilson lower ≥ 0.40): ship raw-policy mcts-distill warm-start as the new SL baseline. Adopt as the warm-start for any future F1 follow-ups.
- **MARGINAL** (0.37 ≤ Wilson < 0.40): one follow-up — regenerate 12k rows from W6/iter-2 ckpt, mix in R7-vintage rule-bot replay rows (~3000 rows from R7 baseline `mixed.jsonl`), retrain.
- **FAIL** (Wilson < 0.37): close MCTS-distill SL branch. The next direction must change either (i) the warm-start (init from W6/iter-2 directly rather than item17), or (ii) the supervision distribution (e.g., DPO on MCTS-derived pairs). Escalate to user.

## 7. Out of Scope

- Multi-iteration DAgger from MCTS-distill warm-start (v1 is one SL pass + one gate).
- Generating an entirely fresh MCTS-selfplay corpus from item17 — the W6/iter-2 ckpt is the stronger player; reuse its existing corpora.
- Changing the MCTS search config (sims/K/depth/rollout type) — out of distillation scope.
- PPO from the MCTS-distill warm-start — gated on SL gate PASS first.
