# R15 Sprint — Archived Result Blocks

Rolled out of `r15.md` per its ≤500-line cap. Canonical, read-mostly.

---

## R7 — multi-teacher BC blend (single iter, FAIL) — 2026-05-14

**Verdict: FAIL — closes the SL-warm-start teacher-target axis. Iter-0 Wilson
lower 0.2921 (n=500, rule-bot, side-balanced) on
`runs/R7-multi-teacher-warmstart/iter-000/`, +0.0077 over R15.S1 iter-0's 0.2844
at the same SL-compute budget — statistically indistinguishable at the
Wilson half-width (~4.2pp at n=500), and ~4pp below the pre-registered SL gate
≥0.35 from the scoping doc § 4.** No PPO sweep launched; the SL gate close-out
condition (scoping § 4 "if either gate fails, close-out with the new SL ceiling
number") fires here.

**Run setup.** R7 step 4 (executed 2026-05-14, PID 3211541) ran
`training/dagger_orchestrator.py --iterations 1` with `--trace-teachers
rollout,search,planner` and otherwise byte-for-byte matched R15.S1's iter-0
SL-compute budget: `--games 90 --epochs 75 --hidden-dim 64 --depth 2
--rollout-steps 500 --rollout-crn-samples 3 --max-steps 500 --replay-games 100
--eval-games 250`. Rule-bot replay produced 4432 examples (identical to
R15.S1's iter-0 replay corpus). Mixed-teacher trace produced 8826 rows (vs
R15.S1 iter-0's 8917 single-teacher rows — equal SL compute). Relabel emitted
`policyTargets` over `legalActions` with uniform 1/K weights per scoping § 3
"Per-action distribution" pick. Mixed corpus 12387 rows = 8826 mixed-teacher +
3561 sampled rule-bot-replay. KL anchor irrelevant (iter-0, no parent).

**Iter-0 gate result.** From
`runs/R7-multi-teacher-warmstart/iter-000/gate.manifest.json`:

| Metric | Value |
| --- | --- |
| games | 500 (250 player + 250 opponent) |
| modelWins | 166 |
| modelWinRate | 33.2% |
| Wilson 95% | [**0.2921**, 0.3744] |
| averageModelPoints | 0.858 |
| averageHeuristicPoints | 1.380 |
| heuristicFallbacks | 0 |
| selectedNoOps | 0 |
| terminalReasons | gameOver: 500 |

Side split: player side WR 28.0% Wilson [0.228, 0.339] (n=250); opponent side
WR 38.4% Wilson [0.326, 0.446] (n=250). Same +10pp opponent-side gap that
replicates across every F1 / R15 phase at value-head-leaf inference (backlog
§ "Current state" bullet 5).

**Comparison anchors.**

| Anchor | iter | Wilson lower | Δ vs R7 | Source |
| --- | --- | --- | --- | --- |
| R15.S1 iter-0 (single-teacher rollout, same SL compute) | 0 | **0.2844** | **+0.0077** (wash, within Wilson half-width) | `progress.md` § Phase K |
| R15.S1 iter-2 (3-iter DAgger from same warm-start) | 2 | 0.3269 | -0.0348 (R7 ran only iter-0) | `progress.md` § Phase K |
| R15.S1 falsification band | — | [0.291, 0.331] | R7 iter-0 0.2921 sits at the band's lower edge | `progress.md` § Phase K |
| item17 take-2 (rollout-only DAgger ref) | 2 | 0.311 | -0.019 | `progress.md` L582 |
| Phase H (PPO from item17 warm-start) | 2 | 0.3109 | -0.019 | `progress.md` § Phase K table |

**Falsification mechanism — plateau-then-overfit signature replicated.** The
R15.S1 closeout (`progress.md` § Phase K) pre-registered val_acc plateau as the
secondary diagnostic for SL-compute saturation. R7's iter-0 train log shows
the same signature even more clearly than R15.S1 did, despite the broader
mixture target:

- `train_completed` event: `final_train_loss=0.1636`, `final_train_accuracy=
  0.9651`, `final_val_loss=2.2104`, `final_val_accuracy=0.7473`. Train/val
  divergence: train_loss → 0.16 while val_loss climbs to 2.21.
- Per-epoch trajectory (`events.jsonl`): peak val_acc **0.7855 at ep16**;
  val_acc plateaus then drifts down to 0.7473 by ep75 (ep45 0.7402, ep60
  0.7449, ep75 0.7473). Val_loss climbs monotonically from 0.93 (ep1) to
  ~1.59 (ep45) to **2.21 (ep75)** while train_acc climbs from 0.68 (ep1) to
  0.93 (ep75). The val-loss climb is **+1.28 nats over the last ~30 epochs
  with val_acc essentially flat** — the textbook overfit shape R15.S1's
  diagnosis predicted.
- R15.S1 reference for the same epoch budget: iter-0 final_val_loss 1.588,
  final_val_acc 0.7382, final_train_acc 0.958. R7 has slightly higher
  final_train_acc (0.965 vs 0.958, expected from the smoother soft target)
  and **notably worse final_val_loss (2.21 vs 1.59)** — the mixture target
  did not improve generalisation; it produced a *more* overfit run at the
  same compute.

**Mechanism interpretation.** The R7 hypothesis (`scoping § 2`) bet that
teacher diversity would regularise away each teacher's idiosyncratic mistakes
and produce a broader-competence warm-start. The pre-flight teacher-agreement
probe (`scoping § 8`) confirmed genuine per-state disagreement (rollout↔planner
0.180, all-three-agree 0.133), so the diversity input was real. But at this
SL-compute budget the multi-teacher target **(a)** did not lift the gate above
the single-teacher baseline (+0.008 wash) and **(b)** produced a *worse*
val_loss trajectory than single-teacher rollout at the same epoch count. The
binding constraint at the 64-hidden/depth-2/75-epoch budget is not teacher
target quality — it is some combination of capacity, data volume, and
gate-relevant signal that no label-axis intervention (single-teacher rollout
in R15.S1 nor uniform 3-teacher mixture in R7) can move.

**Limits of this falsification.** Non-uniform teacher weights and recent-action
history were deliberately out-of-scope per `scoping § 7` ("v1 is uniform 1/3";
recent-action history is R7.b.4). The R7 result does not rule out a re-weighted
mixture (e.g. up-weight planner, the structurally-different teacher per the
probe) — but with the uniform-weight gate result at 0.0077 from the single-
teacher baseline, a non-uniform re-weight is a third-order intervention and
not promoted into the queue here. If anyone resurrects this axis in the
future, the parked entry would be R7-followup-reweight; the F1 SL-line moves
on to R8 DPO next.

**Cost.** End-to-end ~16 min wall-clock (`events.jsonl`: rule-bot replay
completed @ 06:45:24Z trace-gen → train → gate completed @ 07:01:25Z; iter-0
train 58s, gate 22s; trace-gen ~14.5 min as the planner-dominated cost
expected from the pre-flight probe's 14× planner-vs-rollout per-state ratio).
Below the scoping doc § 5 estimate (~3 h total) because we only ran one iter
and skipped the PPO sweep on SL-gate failure.

**Code residue.** Multi-teacher trace + relabel + Python dataset loader (steps
2 + 3) landed clean and remain in the codebase; they are reusable for any
future multi-teacher experiment (re-weighted, recent-action-history-augmented,
etc.) without re-implementing the schema. Orchestrator `--trace-teachers`
flag also remains, separately gated from `--teacher`.

**Next direction picked.** R8 (DPO objective replacement) per the cost-rank in
`docs/ai-agent-state/queue.json` rationale: R8 is ~3 h code + ~10 min training
per backlog § R8; R7.b.2 (card embedding) is ~1.5 + 0.5 days. Cheaper-to-
falsify rule fires R8 next. R7.b.2 promotes only if R8 also closes negative.

**Files & artifacts.**

- `runs/R7-multi-teacher-warmstart/orchestrator-state.json` —
  `promoted_wilson_lower: 0.2921`, single iter, `halted=false`.
- `runs/R7-multi-teacher-warmstart/iter-000/gate.manifest.json` — summary +
  byModelSide + terminalReasons; git sha 6be1e6a clean.
- `runs/R7-multi-teacher-warmstart/events.jsonl` — per-epoch train metrics
  (75 rows for iter-0).
- `docs/ai-research/scoping/r7-multi-teacher-warmstart.md` §§ 1–7 (closed
  design + pre-registered gates), § 8 (pre-flight probe GO result), § 9
  (step 4 launch record). § 4 close-out condition fired by this block.
- Backlog pointer: `docs/ai-research-backlog.md` § Tier 2 R7 line (DONE / FAIL).

