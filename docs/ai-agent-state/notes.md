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
