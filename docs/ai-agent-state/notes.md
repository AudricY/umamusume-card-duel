# Claude Harness Notes

The Claude Code harness is intentionally lightweight.

- Default command: `/work`
- Default execution path: one general `worker` subagent
- Purpose of the worker: context isolation, not parallel throughput
- No hooks in the initial implementation
- No Python harness in the initial implementation

Keep this file for durable harness notes that do not belong in sprint plans or AI research progress docs.

## F1 self-play readiness — scoping (2026-05-14)

**What R5 actually did.** R5 ran PPO from the DAgger iter-2 warm-start with rollouts against the
**item17 opponent pool, sampled uniformly** (see `docs/ai-research-backlog.md` line 252: "PPO from
DAgger iter-2 warm-start, rollouts vs the item17 opponent pool (DAgger iter-002 sampled uniformly)").
That pool was DAgger-era checkpoints from `runs/item17-2026-05-11/` whose strongest member topped
out at Wilson lower **0.3109** — the same number as the warm-start. R5 ran 3 iters × 800 games and
landed at Wilson 0.3109 (iter-2), matching the gate cap exactly. Mean_return moved -0.20 → -0.14
(reward distribution did shift), gate WR did not (`docs/ai-research-backlog.md` lines 254-260).

**What item-12 added on top.** `training/opponent_pool.py` defines `OpponentPool` with PFSP weights
`w_i = max(0.05, 1 - p_i)` (lines 122-159), strict-monotone cycling alarm (lines 164-180), and
retention "last 8 + every 4th historical, hard cap 24" (lines 56-102). This is the infra that lets
the orchestrator maintain a *versioned* pool of promoted checkpoints across iterations and weight
sampling by per-opponent win-rate signal.

**Current wiring state.** `ppo_orchestrator.py` has `--rollout-vs-pool` (line 829) and `--pool-path`
(line 831). When `--rollout-vs-pool` is set, `_select_pool_opponent` (lines 836-863) loads the pool
and picks **one opponent per run** via `rng.choice(list(pool.entries))` (line 862, seed
`trace_seed_start + 9973` — explicitly "stable per-run, doesn't depend on iter"). The selected
checkpoint is exported to ONNX and served as `--opponent-model-url` for all rollout games this
iteration (lines 538-555). **Two real gaps vs the post-mortem's "PFSP self-play" framing:**
(a) sampling is one-shot uniform per run, not per-game and not PFSP-weighted; (b)
`opponent_pool.pfsp_weights()` exists but is never called from `ppo_orchestrator`.

**The diagnostic question.** Is "PFSP pool of W6/iter-2 + earlier promoted checkpoints" a different
experiment from R5? **Yes, materially.** R5's pool was DAgger iter-002 checkpoints (Wilson ~0.31).
W6/iter-2 is Wilson **0.6479** (R13-W6-phase-d) — a fundamentally stronger opponent set. R5 closed
the "self-play against a weak pool" question; it did NOT close the "self-play against a strong
pool" question.

**Go / no-go: GO, but with plumbing-clarity caveat.** The proposed sweep is a genuinely different
experiment from R5 because the pool members are ~30pp stronger. Cost is cheap (~10-15 min compute
per F1 wall-clock datapoint). However, the operator should know: today's `--rollout-vs-pool` picks
one opponent per run and does NOT exercise PFSP weights. For an honest "PFSP self-play" test the
sweep needs either (i) to accept "one strong opponent sampled once per run" as the actual
experiment (simplest, recommended for first attempt) or (ii) a small plumbing patch to call
`pfsp_weights()` with rolling history and re-sample per game (deeper, defer to iteration 2 if
iteration 1 promotes).

**Proposed sweep config (first attempt, "one strong opponent per run"):**
- Pool composition: hand-built `opponent-pool.json` with three entries — `runs/R13-W6-phase-d/iter-0/checkpoint.pt`,
  `runs/R13-W6-phase-d/iter-1/checkpoint.pt`, `runs/R13-W6-phase-d/iter-2/checkpoint.pt`. Iteration
  numbers 0/1/2 to match retain() ordering.
- Warm-start: `runs/R13-W6-phase-d/iter-2/checkpoint.pt` (same checkpoint also in pool; the
  one-per-run sampler may pick a self-game, that's fine).
- HPs: aggressive (mirror F1 phase H so the comparison axis is opponent only) — lr=3e-4,
  clip=0.3, entropy_coef=0.01, ppo_epochs=4.
- Rollouts: 3 iters × 800 games/update; ratio of self-play vs vs-rule-bot rollouts = 100% self-play
  (no mix; the gate already evaluates vs rule-bot so we don't double-up).
- Wall-clock: F1 datapoint is ~5-6 min per 800-game update + a few minutes for ONNX export + gate.
  3 iterations ≈ 30-45 min total.
- Exit / gate: same R15.S2 criterion — Wilson lower ≥ 0.40 on the rule-bot gate at any iter.
  Stretch: if iter-1 promotes, run an iteration 2 with the PFSP patch in place.

## F1 better warm-start — scoping (2026-05-14)

**Hypothesis (pre-registered).** The DAgger warm-start at `hidden_dim=64 / depth=2` is
*compute-starved, not capacity-starved*: holding model size fixed and scaling games + epochs by
~3× will lift the warm-start's greedy-eval Wilson lower from item17-2026-05-11's **0.311** (iter-2,
n=200, [0.311, 0.444], `docs/ai-performance-research-progress.md:582`) to **≥ 0.45**, giving PPO an
actually-movable starting point. Prediction: ≥ +14pp Wilson-lower lift over item17 iter-2.
Falsification: Wilson lower stays within ±2pp of 0.311 → the cap is upstream of compute (label
noise / representation / capacity), and additional epochs only sharpen the same imitation error.

**What we know (current item17 config + result).**
- Canonical command pre-registered for the take-1 sweep: `dagger_orchestrator.py --iterations 3
  --games 250 --max-steps 500 --teacher rollout --rollout-steps 500 --rollout-crn-samples 3
  --replay-games 100 --epochs 30 --batch-size 32 --hidden-dim 64 --depth 2 --eval-games 250
  --kl-anchor-weight 0.1 --kl-anchor-weights "0.0,0.1,0.5"`
  (`docs/ai-performance-research-progress.md:511-528`).
- **What actually ran at item17-2026-05-11 (take-2, the bedrock warm-start).** Compute-downscaled:
  30 trace games, 100-game eval, 25 epochs (`docs/ai-performance-research-progress.md:571-573,
  618-619`). This is the run whose iter-2 checkpoint sits at
  `runs/item17-2026-05-11/iter-002/checkpoint.pt` and is the warm-start every F1 PPO phase
  loaded from.
- `dagger_orchestrator.py` defaults (`training/dagger_orchestrator.py:1024-1056`): `--iterations 3`,
  `--games 20`, `--rollout-steps 200`, `--rollout-crn-samples 3`, `--epochs 8`, `--batch-size 32`,
  `--hidden-dim 64`, `--depth 2`, `--kl-anchor-weight 0.0`.
- `train_bc.py` defaults fed in (`training/train_bc.py:722-766`): `--lr 3e-4` cosine
  (`--lr-warmup-steps 0`), `--weight-decay 1e-4`, `--dropout 0.05`, `--value-weight 0.1`,
  `--entropy-bonus 0.0`, `--split-by episode`, `--data-mode bc`, `--policy-weight 1.0`.
- Result: iter-2 Wilson lower **0.311** at WR 37.5% (n=200). All F1 PPO phases stuck within ±0.5pp
  of this number (`docs/ai-performance-research-progress.md:725-735`).

**What has already been tried (so this run is genuinely new).**
- **R3** added entropy bonus β ∈ {0.05, 0.2} to the BC loss; β=0.05 hit entropy 0.527 nats but
  gate WR stayed at 38.5% — statistically equivalent to the warm-start
  (`docs/ai-research-backlog.md:147`).
- **R4** retrained value head with `--epochs 50 --value-weight 1.0`; value Brier dropped but gate
  WR fell to 34.5% (`docs/ai-research-backlog.md:233-235`).
- **R6 (capacity)** trained `--hidden-dim 128 --depth 3 --epochs 50 --value-weight 1.0`; train
  accuracy hit 0.97, argmax-match 0.832, gate WR **33.0%** — *worse* than the 64/2 warm-start
  (`docs/ai-research-backlog.md:227-236`). Standing diagnosis: "the cap is
  imitation-target-quality, not capacity."
- **R3-b020 + PPO** combined β=0.2 entropy + value retrain + aggressive PPO, still stuck at
  Wilson lower 0.30 (`docs/ai-research-backlog.md:238-248`).
- **Not yet tried: same size, same data recipe, more games + more epochs (no other axis moved).**
  R6 covaried capacity and epochs; R4 covaried value-weight and epochs. The pure "more compute,
  same shape" axis at the canonical 64/2 size has not been run as a controlled follow-up to
  item17-2026-05-11. The pre-registered 250-game/30-epoch sweep was *documented* but never run
  (take-2 was downscaled to 30/25 for wall-clock).

**Proposed config (refines queue's "3x data, 3x epochs").** Launch
`training/dagger_orchestrator.py` with:

- `--games 90` (3× item17 take-2's 30; the documented full-scale 250 is 8× and out of scope here).
- `--epochs 75` (3× item17 take-2's 25).
- `--iterations 3` (unchanged — single DAgger inner loop on the same teacher).
- `--max-steps 500`, `--rollout-steps 500`, `--rollout-crn-samples 3` (unchanged from take-2).
- `--replay-games 100`, `--relabeled-weight 0.6`, `--replay-weight 0.4` (unchanged).
- `--batch-size 32` (unchanged; bumping to 64 is allowed only if memory headroom — not required).
- `--hidden-dim 64 --depth 2` (UNCHANGED — load-bearing pre-registered constraint; this is what
  isolates compute-starved from capacity-starved).
- `--kl-anchor-weight 0.1 --kl-anchor-weights "0.0,0.1,0.5"` (unchanged off/low/high schedule).
- `--eval-games 250` (the originally-documented eval budget; keeps n=500 side-balanced per iter so
  Wilson half-width is ~4pp — required to resolve a +14pp lift against the noise floor).
- `train_bc.py` flags via orchestrator pass-through stay at default: lr=3e-4 cosine,
  weight-decay 1e-4, dropout 0.05, value-weight 0.1, **entropy-bonus 0.0** (we are NOT redoing R3;
  this run isolates the SL-compute axis). If train loss bottoms out but Wilson lower stalls, a
  follow-up may layer `--entropy-bonus 0.05` as a single-axis ablation.

**Exit criteria.**
- *Primary success:* any iter's greedy-eval Wilson lower **≥ 0.45** at n ≥ 500 side-balanced.
  Promotes to a tier-1 F1 PPO re-attempt from this checkpoint.
- *Secondary:* SL train loss strictly decreasing across the 75 epochs (no plateau before
  epoch 50). If the loss plateaus by epoch ~25 (matching take-2), that itself confirms
  "compute is saturated at the current size" — falsification path is supported.
- *Falsification:* iter-2 Wilson lower in 0.311 ± 2pp (i.e. 0.29–0.33). Closes the
  compute-starved branch; the cap is then *not* compute. Remaining attention routes to R7
  (multi-teacher labels), R8 (DPO), or the R12-line evidence that the cap is downstream of
  value-head leaf noise rather than SL labels.

**Expected cost.** Item17 take-2 (30 games × 25 epochs) ran in <10 min wall-clock
(`docs/ai-research-backlog.md:638`). Scaling games 3× (more rollout-CRN teacher invocations, the
dominant cost) and epochs 3× gives a ~9× wall-clock estimate → **~1.5h compute** for the
3-iteration sweep at the proposed config. Within the "1–2h" envelope at
`docs/ai-research-backlog.md:638`. n=500 eval × 3 iters adds ~20–30 min.

**Risk register.**
- *Overfit to noisy teacher labels.* R6 already showed 128/3 + 50 epochs over-imitates the
  teacher's 25% argmax errors and drops gate WR. The same risk applies at 64/2 + 75 epochs; the
  secondary "loss-still-declining" check is the early warning.
- *Label-distribution drift.* Each DAgger iter relabels with the same rollout-CRN×3 teacher; at 3×
  games per iter the trace distribution shifts (more states visited by the iter-0 policy rather
  than rule-bot baseline). The KL anchor schedule (0.0/0.1/0.5) mitigates but does not eliminate;
  monitor per-iter Wilson rather than only iter-2.
- *Compute under-spend.* If `--games 90 / --epochs 75` is still below saturation, a negative
  result is uninformative about the hypothesis. Mitigation: log train loss per epoch and verify
  the curve is still declining at epoch 75. If it is, the hypothesis is *not* falsified — only
  the specific 3× scale is.

**Decision gate (human-owned, before paying compute).** The /work harness must NOT launch this
sweep autonomously (warn-before-launch rule in `CLAUDE.md`). Before running, the human should
confirm: (a) the diagnosis-quality argument above is acceptable — specifically that R6's capacity
result does not subsume this experiment because R6 covaried capacity with epochs and this run
isolates the compute axis at fixed capacity; (b) the ~1.5h compute budget is available; (c) the
falsification path (Wilson lower stays at 0.31 ± 2pp) is acceptable as a research outcome and
will close the branch rather than spawn further SL ablations. On green light, launch with the
command in the "Proposed config" section; on red light, demote to R15.S3 / R15.S4 instead.
