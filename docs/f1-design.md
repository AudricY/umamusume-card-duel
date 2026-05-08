# F1 Design — Online PPO On Top Of The DAgger Warm-Start

This doc fixes the F1 (Option B) defaults so the PPO smoke at item 7's exit
gate doesn't relitigate every knob from scratch. Item 18 (F1 plumbing prep)
landed the unblocked-today pieces: behavior-policy logging in `serve_onnx`,
the persistence path through `evaluateModelVsHeuristic` and
`relabelDecisionTrace`, and these defaults. Once item 17's larger-scale
DAgger sweep produces a warm-start checkpoint, the F1 smoke runs against
this spec.

The backlog item that owns the *implementation* of F1 lives at
`docs/ai-performance-research-backlog.md` under "F1. Online PPO Plumbing".
This doc owns the *defaults* it points at.

## Reward shaping

- **Per-step shaping:** `r_t = α · Δpoints_t` where `Δpoints_t` is the change
  in `own_points - opponent_points` between successive *acting-side*
  decisions for the model's side. Default `α = 1/3` so a single point swing
  ≈ 0.33 per-step reward (3 points wins; the magnitude here is calibrated so
  the cumulative shaped signal across a 6-7 turn game is ≤2.0).
- **Terminal:** `r_T = β · win_indicator` where `win_indicator ∈ {-1, 0, +1}`
  (+1 win, 0 unfinished, -1 loss). Default `β = 1.0`.
- **Mix:** total return is the discounted sum of per-step `r_t` plus the
  terminal `r_T` at the absorbing state. Per-step shaping carries the
  Δpoints signal; terminal carries the win/loss outcome. `α/β` may be tuned
  in the HP sweep but the prior keeps `β >> α · steps_per_game` so winning
  outweighs intra-game point churn.
- **Where the signal lives:** `Δpoints` is computed from the public
  observation already persisted on every decision trace row
  (`observation.own.points`, `observation.opponent.points`). No new
  exporter or feature is required.

## Episode boundary

- One *episode* = one game.
- One *PPO time step* = one model-side decision (matches the DAgger trace
  granularity). Opponent-side and rule-bot intermediate steps are
  collapsed.
- *Absorbing state:* the final decision row of the model side; carries
  `r_T` and `done = True`.
- Trace rows that the model never visited (opponent-side rows) are not
  PPO time steps — they are state transitions but not decision points.

## Behavior-policy logging (landed in item 18)

- `serve_onnx` /predict returns `actionLogProbs`, `actionProbs`,
  `selectedLogProb`, and `behaviorPolicy: {kind, temperature}` per decision.
- `evaluateModelVsHeuristic.ts` persists these into decision traces
  whenever `--selection policy` is in use.
- `relabelDecisionTrace.ts` forwards `behaviorPolicy` into the relabeled
  training row so PPO can recover importance ratios on warm-start data
  without re-running rollouts.
- Single-action decisions emit a degenerate `behaviorPolicy: {kind:
  "single-action", selectedLogProb: 0}` so importance ratios stay defined
  on every row.
- Greedy serving still records the softmax distribution as the behavior
  policy. At PPO iteration 0 the target equals the behavior so the
  importance ratio is exactly 1; drift accumulates as PPO updates the
  target. A future stochastic-sampling mode will replace argmax selection
  with Gumbel-max from `actionLogProbs` and the same plumbing carries the
  non-trivial distribution through.

## Legal-action mask handling

- Masks are *recomputed* at PPO update time from each row's stored
  `legalActions` array, not stored as a frozen mask. Stale masks from
  rollout time would diverge if the action enumeration changes between
  data generation and PPO update; recompute is cheap (O(legalActions)) and
  correctness-protective.
- Masked logit positions use the existing `-1e9` sentinel; downstream
  consumers expect the same encoding as the trained policy.

## Buffer sizing

- **On-policy buffer:** N games per PPO update, where N is set so
  `N × decisions_per_game ≈ 32K transitions`. Item 0's throughput probe
  measured ~40 decisions/game model-side, so default `N = 800 games per
  update`.
- **Update count per buffer:** 1 epoch over the buffer with 4 minibatches
  (default minibatch size 8K transitions); standard PPO 1-epoch-per-batch
  setting kept until a HP sweep flags it.
- **Replay:** none. F1 is on-policy. The DAgger replay buffer is the
  *warm-start* data, not the F1 update data.

## Advantage estimator

- **GAE-grade gate cleared:** GAE with `λ = 0.95`, `γ = 0.99`. Critic is
  the value head from item 7 (state-only, Tanh-bounded ∈ [-1,1]).
- **GAE-grade gate not cleared (only tiebreaker-grade):** point-margin
  advantage estimator: `A_t = sign(Δpoints_t) - V̂(s_t)` where `V̂(s_t)` is
  fitted to a normalized point margin. Same data spine, drops the
  multi-step credit assignment but still corrects the on-policy bias that
  raw `Δpoints` would carry. Recorded in the manifest.
- **Both options computed** at runtime so an iteration that flips between
  GAE and point-margin advantages produces comparable manifests.

## Stability controls (logged on every update)

- **KL clipping:** standard PPO ratio clip at ε = 0.2. Adaptive KL
  controller off by default; enabled if smoke run shows KL drift > 0.05
  per update for 3 consecutive updates.
- **KL penalty coefficient:** `c_KL = 0.01` if adaptive KL is enabled,
  else 0.
- **Entropy bonus:** `c_H = 0.005`. Targets entropy floor of `0.5 ·
  log(legal_actions)` averaged over the buffer; below this the bonus
  doubles.
- **Gradient clip:** `‖g‖₂ ≤ 0.5` (current DAgger training uses 2.0 — F1
  is tighter because the policy gradient is noisier than imitation loss).
- **Value-loss clip:** `c_V = 0.5`, mirrored against the policy clip.

All five must be in the per-update manifest line so a regression can be
diffed mechanically.

## League sampling (item 12 prerequisite)

PPO rollouts sample opponents from the snapshot pool with PFSP weighting:
`w_i ∝ max(0.05, 1 - p_i)` where `p_i` is the current model's empirical
win rate against snapshot `i` over a fixed evaluation window. Falls back
to uniform if the pool is degenerate (fewer than 2 snapshots). Same
mechanism as item 12 — F1 does not introduce a separate sampler.

## HP sweep methodology (run *before* the first multi-iteration F1 run)

- **Grid (4 axes × 2 values each = 16 cells):**
  - `lr` ∈ `{3e-5, 1e-4}` (default 3e-5)
  - `KL coef` ∈ `{0.0, 0.01}` (default 0.0; enabled only if drift)
  - `entropy coef` ∈ `{0.001, 0.005}` (default 0.005)
  - `GAE λ` ∈ `{0.9, 0.95}` (default 0.95)
- **Seeds per cell:** 3 (so the cheapest variance estimate per cell is on
  3-sample bootstrap).
- **Wall-clock budget:** ≤ 30 min/cell on the user's box, capping total
  sweep at 24 cell-hours.
- **Selection rule:** Wilson lower bound on win rate vs. the warm-start
  parent over 100 side-balanced games per cell. Pick the cell with the
  highest Wilson lower bound; record the runner-up for ablation context.
- The sweep runs as `training/f1_hp_sweep.py` (not yet implemented;
  blocked on the F1 smoke landing).

## Smoke acceptance (≥10 PPO updates)

- KL per update stays in [0, 0.05] (no blow-up).
- Entropy never drops below 30% of the warm-start initial entropy (no
  collapse).
- Pool-aggregate Wilson lower bound at update 10 ≥ Wilson lower at update
  0 - 5pp (proves the smoke didn't just diverge).
- All five stability controls are recorded on every update.
- Promoted F1 checkpoint beats DAgger parent at Wilson lower bound on the
  corrected suite *and* on the opponent pool.

## What remains for the F1 PPO implementation

- A `training/ppo_orchestrator.py` analogous to `dagger_orchestrator.py`
  but on-policy: rollout → buffer → update → evaluate → promote/reject.
  Reuses item 12's pool, item 13's per-matchup gate, item 11's
  reproducibility tiering.
- A stochastic serving mode in `serve_onnx` (Gumbel-max sampling from
  `actionLogProbs` with optional temperature scaling).
- The HP sweep script.

These are tracked under F1 in the active backlog. None are blocked by
this doc.
