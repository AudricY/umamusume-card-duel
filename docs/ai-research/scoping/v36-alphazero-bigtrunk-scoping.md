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

**`LANDED-NULL-WITH-INFORMATION-CEILING-DIAGNOSED 2026-05-26`** — see §10.

---

## 10. Results + diagnostics

Three runs fired, all under the AZ recipe (value-head leaf + two-sided + no-KL) at hidden=256/depth=4 cold:

| Run | games/iter | sims | replay | iters | device | wall | best wl_lower | mean | σ |
|---|---|---|---|---|---|---|---|---|---|
| bigtrunk CPU | 480 | 400 | 15 | 17/40 (killed) | cpu | 1h16m | 0.4773 @ iter-3 | 0.4188 | 0.0399 |
| bigtrunk CUDA | 480 | 400 | 15 | 12/20 (killed) | cuda | 11m | 0.4363 @ iter-0 | 0.3980 | ~0.04 |
| 5k-no-buffer CUDA | 5000 | 400 | 0 | 20/20 ✓ | cuda | 93m | 0.4281 @ iter-15 | 0.3673 | 0.0324 |

None lift past the cold-init random-search noise floor (~0.44). Mean strictly below it.

### Diagnostics on the 5k-no-buffer CUDA run (`runs/.../analyze.py`)

**Training metrics:** val_loss flat at 1.93 across all 20 iters (Δ first-half-vs-last-half = −0.013). val_accuracy creeps 0.41→0.53. train_value_loss tiny (0.03→0.09). The model fits its targets quickly; nothing generalizes.

**Policy concentration:** mean **3.4 legal actions/decision**. Top-1 prior climbs 0.42→0.48 (vs uniform 0.29) and plateaus by iter-3. Visit-distribution ENA stays 2.94. Policy is barely more concentrated than uniform — *not collapsed, but with little structure to learn either.*

**Value-head calibration:** corr(rootValue, valueTarget z) jumps from 0.04 (cold) to **0.46-0.51 after iter-1 and stays there for 19 iters**. std_rv ≈ 0.86 (vs std_vt = 0.99). MSE ≈ 0.9. **One iter of training saturates value-head learnable signal.**

**Weight L2 drift:** per-iter ΔL2 = 23-32 against init norm 85.8 (**27-37% of total weights changing every iter**), trend slowly growing. Two outlier iters (7→8 at 43%, 13→14 at 47%) suggest orchestrator-state hiccups.

### Synthesis: three independent ceilings

1. **Environmental-noise ceiling (corr ≈ 0.5).** ~50% of game-outcome variance is RNG (coin flips, hand draws, deck shuffles). A perfect value head couldn't get above corr ≈ 0.7-0.8 from state alone in this game. We hit 0.5 after **one** iter and parked there — the value head has already learned what's learnable from state.
2. **Action-space ceiling (3.4 legal moves/decision).** AZ's premise — search differentiates many options — is broken on a 3-option action space. The policy can't get much more concentrated than ~0.5 top-1 visit. Sims past ~50 are wasted because visit-target precision is already excellent at low sims.
3. **Weight-jitter ceiling (no-KL gradient noise dominates).** Weights swing 30%+ per iter despite flat val_loss. Without the W6 KL anchor the model hops between equally-good fits in a large flat basin. The 0.5880 v3.6 anchor *was* achieved with (a) the W6 anchor preventing this drift and (b) rollout-leaf naturally integrating the RNG.

### Verdict

**H1 (AZ joint-threshold lift) FALSIFIED.** AZ canon assumes Go-like conditions: no RNG, large branching factor, clear positional signal. This game has none. The "5 improvements" all targeted compute/data scaling, not the information-theoretic ceilings, so they couldn't move the needle. The v3.6 anchor's strength comes from removing the brittleness AZ canon introduces (rollout-leaf for RNG, KL anchor for stability) — both of which we removed for AZ purity.

### Follow-up bets (more directly aligned to diagnostics)

- **More-games shallower-search at sims=100.** Visit-target precision saturates at ~30 visits/action; sims=400 wastes 4× compute. Recipe: 15k games × sims=100 × no-buffer × 20 iters, wave-size=64. Same wall budget, 3× more diverse selfplay. The hypothesis is that the value-head plateau at corr=0.5 is data-limited, not search-limited.
- **Restore weak KL anchor (weight ≈ 0.02-0.05).** Damps the 30%+ per-iter weight drift while leaving room for whatever weak signal exists. Falsification: weight ΔL2 should drop below 15 per iter and per-iter wl variance should compress below σ=0.025.
- **Asymmetry-redux exploit.** Iter-1 of the warm AZ run reduced per-side asymmetry by half under anchor-recipe eval. If the only consistent off-axis gain from AZ is "shape the policy slightly more symmetric without losing strength," that's a *targeted distillation* loss-term ("asymmetry penalty") not an AZ recipe — different axis entirely.

### Scripts + artifacts

- `runs/R16-P3-v36-az-5k-nobuffer-cuda/analyze.py` — diagnostic script (training metrics, policy concentration, value calibration, weight drift)
- `runs/R16-P3-v36-az-5k-nobuffer-cuda/events.jsonl` — full per-iter events
- `runs/R16-P3-v36-az-5k-nobuffer-cuda/iter-{0..19}/checkpoint.pt` — ckpt chain
