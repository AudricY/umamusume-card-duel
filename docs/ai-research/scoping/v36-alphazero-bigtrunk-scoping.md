# v3.6 AlphaZero Big-Trunk + High-Sim + Cold-Init Recipe — Scoping

- **Date:** 2026-05-26
- **Status:** **FIRING** — cold-init checkpoint synthesized, orchestrator launched at hidden=256/depth=4/sims=400/iters=40 under the AZ recipe (value-head leaf + two-sided + no KL anchor + replay window=15).
- **Parent:** `v36-alphazero-recipe-scoping.md` (LANDED-AZ-RECIPE-INFERIOR-AT-V36-SCALE). This is the explicit §10.5 follow-up: AZ canon at v3.6 scale was falsified because we tested AZ at a budget AZ wasn't designed for. This run hits the joint capacity × sims × data × iters threshold where AZ self-improvement is supposed to kick in.
- **Sibling:** `v36-priors-and-arithmetic-scoping.md §9` (the schema-axis-exhausted verdict that motivates the capacity bump).

---

## TL;DR

Five compounded improvements over the failed AZ baseline (`v36-alphazero-recipe-scoping.md`):

| Lever | Failed AZ baseline | This run | Why |
|---|---|---|---|
| Trunk capacity | hidden=128, depth=2 | **hidden=256, depth=4** | v3.6 §9 said trunk is capacity-bound; AZ value head needs more capacity to learn the stochastic integral a rollout computes for free |
| MCTS sims | 100 | **400** | Search signal needs to outweigh value-head noise; AZ used 800 — 400 is the compute-feasible compromise |
| Iterations | 4 (halted) | **40** | AZ self-improvement is asymptotic; 4 iters can't bootstrap |
| Init | v3.6 cap128 cont-iter2 (rollout-trained, OOD for value-head leaf) | **cold (random)** | Search and net co-evolve from scratch — AZ canon |
| Replay window | 3 iters | **15 iters** | Bigger buffer smooths per-iter variance the search distribution introduces |

Other AZ recipe knobs unchanged from `v36-alphazero-recipe-scoping.md`: `--mcts-leaf value-head`, `--mcts-two-sided`, `--kl-anchor-weight 0.0`, `--no-w6-fix-fixed-kl-anchor`, Dirichlet noise default, temperature schedule default. Self-play games bumped to **480/iter** (2× baseline) to feed the bigger trunk.

## Pre-registered hypotheses

### H1 — Joint-threshold lift
Cumulative effect of all 5 levers should cross the AZ-self-improvement regime. **Falsifiable:** wl_lower(best iter, n=10k matched-recipe gate) ≥ 0.5778 (v3.6 soft-ship band). Anything below this is "AZ still didn't kick in at v3.6 scale even with the big-trunk + high-sim recipe."

### H2 — Iter-N+ improvement, not iter-1-peak
Bigger replay window (15) plus more data per iter (480 games) plus more capacity should resist the iter-2-peak-then-regress pattern that fired in the failed AZ baseline (the cap128/depth2 trunk overfit per-iter). **Falsifiable:** trajectory shape across iters 0-39 shows monotone or near-monotone improvement past iter-5 (not a single early peak).

### H3 — Cold-init breaks the value-head OOD calibration mismatch
A cold-init value head learns its calibration UNDER AZ-search self-play from the start, instead of inheriting a calibration trained on rule-bot-collapse trajectories. **Falsifiable:** matched-recipe gate from iter-1 onward should not collapse below random-baseline (~0.50); the Phase 1 P3 catastrophe (wl=0.38 on warm-init) should not reproduce.

## Recipe

```
python3 training/r12_orchestrator.py \
  --out-dir runs/R16-P3-v36-az-bigtrunk-cold \
  --init-checkpoint runs/R16-P3-v36-az-bigtrunk-cold/init-cold/checkpoint.pt \
  --iterations 40 \
  --selfplay-games 480 \
  --eval-games 60 \
  --mcts-simulations 400 \
  --mcts-c-puct 1.5 \
  --mcts-leaf value-head \
  --mcts-two-sided \
  --mcts-wave-size 16 \
  --epochs 25 --batch-size 64 --lr 3e-4 \
  --hidden-dim 256 --depth 4 \
  --state-dim 246 \
  --kl-anchor-weight 0.0 \
  --no-w6-fix-fixed-kl-anchor \
  --w6-replay-window 15 \
  --w6-replay-old-fraction 0.5 \
  --halt-after-consecutive-failures 0 \
  --engine rust
```

Cold-init helper: `python3 training/make_v36_cold_init.py --output <init>/checkpoint.pt --hidden-dim 256 --depth 4 --seed 0`.

## Decision rules

- **Lift band (H1 confirmed):** wl_lower(best iter, n=10k matched recipe) ≥ 0.5778 → AZ-as-designed works at v3.6 scale with the joint-threshold recipe. Promote to candidate. Open scoping for the next axis (e.g., hidden=512, sims=800).
- **Partial lift (asymmetry-only or under-band):** matched-recipe wl_lower in [0.50, 0.578) but trajectory clearly trending → halted by compute, not by ceiling. Restart with more iters and/or sims.
- **Null/regression (H1 falsified):** matched-recipe wl_lower < 0.50 at iter-40 → joint-threshold isn't the bottleneck; AZ canon doesn't transfer to this game's stochasticity/hidden-info structure. Pivot to AZ-with-rollout-leaf (just two-sided), or accept that rule-bot-collapse + rollout is the game's strength ceiling.

## Wall budget

Per-iter estimate (~2-3 min):
- Selfplay: 480 games × sims=400 × value-head + two-sided + wave-size=16 ≈ 30-60s (wave-size batches inference across worker threads)
- Distill: 25 epochs × ~3-4k rows (current + 15-iter replay × old-fraction 0.5) × hidden=256/depth=4 ≈ 30-90s
- Gate: 120 games × sims=400 ≈ 5-10s

Total: **~1.5-2.5 hours wall for 40 iters**.

## Risks

| Risk | Mitigation |
|---|---|
| Cold-init iter-0 self-play data is pure noise (random net) → first few iters have garbage targets | Halt-after=0 lets the loop run through the noisy phase; replay window=15 dilutes early-iter noise once buffer fills |
| Bigger trunk + more iters = potentially much slower than estimated | --mcts-wave-size 16 amortizes per-game inference; if per-iter wall >> 5 min, halt and downsize iters |
| Cold-init at hidden=256 vs v3.6 anchor (also hidden=128) makes comparison apples-to-oranges | Documented up front; the comparison is "did joint-threshold AZ recipe ever beat anything"; for fairness we'd also need a hidden=256 rollout-recipe ckpt — separate follow-up |
| sims=400 cold-init has been tried (R16-P1-v36-cold-sims400 hit wl=0.4939 at iter-19) — but that was rollout-recipe at hidden=128 | This run differs on (a) AZ recipe (value-head + two-sided), (b) bigger trunk (hidden=256 depth=4), (c) bigger replay; explicit AZ-with-capacity test |

## Out of scope

- sims=800 (compute) — if H1 confirms at 400, sims=800 is the obvious next axis
- hidden=512 — same reasoning
- From-scratch with longer iters (100+) — only if H1 trends positive at 40
- Deck-sampling variation — inherits orchestrator default (uniform, satisfies directive A)

## Status

**`FIRING 2026-05-26`** — cold-init synthesized; orchestrator launched in background.
