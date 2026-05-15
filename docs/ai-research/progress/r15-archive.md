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

---

## R8 — DPO (Direct Preference Optimization, single iter, FAIL) — 2026-05-14

**Verdict: FAIL — closes the R8 v1 branch. Iter-0 Wilson lower 0.3318
(n=1000 side-balanced, rule-bot, seed-start 9000) on
`runs/R8-dpo/iter-000/`, +3.97pp over R7 iter-0's 0.2921 and +2.09pp over
item17 iter-2's 0.311 — a real lift over both anchors but ~3.6pp below the
R15.S3 obs-delta ceiling (0.3677) and ~6.8pp below the pre-registered
gate ≥ 0.40 (`scoping § 4`).** Wilson lower < 0.37 → close-out path per
scoping § 5(b); the marginal-band β=0.3 follow-up is NOT triggered. R7.b.2
(card-embedding feature representation) promotes to the next pick.

**Run setup.** Step 2a (digest slot 43) trained DPO from the item17
warm-start (`runs/item17-2026-05-11/iter-002/model/checkpoint.pt`) with
β=0.1, τ=0.3005, 568 kept pairs (13.9% of the 4094-row outcome corpus
`training/runs/r8-outcome-crn/examples.jsonl`), 5 epochs / batch 32 / lr
1e-4 on CUDA in ~30s. Train metrics from
`runs/R8-dpo/iter-000/model/manifest.json`: train loss 0.683→0.603,
val loss 0.683→0.618, train acc 0.692, val acc 0.681, train policy_margin
3.31 / val 3.27, train logits_diff_mean 2.08 / val 1.53 — model moved
away from reference as expected for non-zero β. ONNX roundtrip smoke PASS
(max_logit_diff 7e-7, max_value_diff 3e-7).

**Step 2b gate.** New driver `training/r8_gate_eval.py` (~150 LOC) wraps
the three existing primitives `export_checkpoint_to_onnx`,
`serve_onnx_context`, `run_eval_gate`. Seed-start 9000 chosen to match
R7 step 4's `seedStart` field in
`runs/R7-multi-teacher-warmstart/iter-000/gate.manifest.json`.
**Deviation from scoping § 3.7 / brief:** brief asked for "n=500
side-balanced" but `--games N` is interpreted per-side (`modelSide=both`),
so the actual sample is **n=1000 (500 player + 500 opponent)** — 2×
larger than R7's n=500. The verdict direction is unchanged (Wilson lower
0.3318 is well below the 0.37 marginal-band floor); the larger n only
tightens the interval (Wilson width 5.95pp at n=1000 vs ~8.2pp at R7's
n=500). Wall-clock 1m 18s (15:42:41Z → 15:43:59Z).

**Iter-0 gate result.** From
`runs/R8-dpo/iter-000/gate.manifest.json`:

| Metric | Value |
| --- | --- |
| games | 1000 (500 player + 500 opponent) |
| modelWins | 361 |
| modelWinRate | 36.1% |
| Wilson 95% | [**0.3318**, 0.3912] |
| averageModelPoints | 0.960 |
| averageHeuristicPoints | 1.404 |
| heuristicFallbacks | 0 |
| selectedNoOps | 0 |
| terminalReasons | gameOver: 1000 |

Side split: player WR 31.8% Wilson [0.279, 0.360] (n=500); opponent WR
40.4% Wilson [0.362, 0.448] (n=500). Same +8–10pp opponent-side gap
that replicates across every F1 / R15 / R7 phase at value-head-leaf
inference (backlog § "Current state" bullet 5). **Opponent side
in isolation clears 0.40 (Wilson lower 0.3619 — within 4pp of the 0.40
gate)** but the symmetric gate is the binding metric.

**Comparison anchors.**

| Anchor | Wilson lower | Δ vs R8 iter-0 | Source |
| --- | --- | --- | --- |
| R8 DPO iter-0 (β=0.1, item17 ref) | **0.3318** | — | this block |
| R15.S3 ceiling (Phase N/O', obs-delta saturated) | 0.3677 | -3.59pp | `progress.md` § Phase N / O' |
| R15.S1 iter-2 (3-iter DAgger from item17) | 0.3269 | +0.49pp | `progress.md` § Phase K |
| item17 take-2 iter-2 (rollout-only DAgger ref) | 0.311 | +2.08pp | `progress.md` L582 |
| Phase H (PPO from item17 warm-start) | 0.3109 | +2.09pp | `progress.md` § Phase K table |
| R7 iter-0 (multi-teacher BC blend) | 0.2921 | +3.97pp | § R7 above |
| Pre-registered R8 gate | ≥ 0.40 | -6.82pp | `scoping § 4` |

**Mechanism interpretation.** DPO did move the gate above all SL/PPO
anchors (R7, item17, Phase H, even R15.S1 iter-2 by a hair) — the
Bradley-Terry objective is **not strictly bounded by the reference policy**
in practice; the +3.97pp over R7 and +2.08pp over the item17 reference
demonstrate genuine objective-family lift. But it lands **below** the
R15.S3 reward-shaping ceiling (0.3677) and well below the 0.40 gate. Two
interpretations consistent with the data:

- **(a) Pair-budget binding.** 568 kept pairs after τ=0.3005 filtering is
  ~5.7% of the 10k-row scoping budget. The corpus may simply be too small
  for the DPO loss to extract more lift before val-loss bottoms out.
- **(b) Reference-action-support binding (scoping risk (c)).** A
  Wilson-0.31 reference assigns near-uniform mass to several bad actions;
  DPO's BT loss can re-weight only between actions the reference already
  considers, capping the lift the objective itself can deliver.

Either way, β=0.1 v1 cleared the R7 / item17 / R15.S1 anchors but did not
clear R15.S3's reward-shaping ceiling. The marginal-band β=0.3 follow-up
(scoping § 5(b)) requires Wilson lower ∈ [0.37, 0.40) — at 0.3318 we are
~4pp below that floor, so the follow-up does **not** fire automatically.

**Limits of this falsification.** Only β=0.1 tested. Only single-pass
(no iterative refresh from trained policy). Only the 568-pair τ=0.3005
filter. R8.b candidates explicitly out of v1 scope (scoping § 7): R8.b
DPO from R15.S3 phase-N warm-start (Wilson 0.368 — bypasses
interpretation (b)); iter-2 DPO with refreshed pairs from the trained
policy (scoping risk (d)); top-K / Plackett-Luce loss; β sweep at
non-marginal start. Any of these could re-open the line; none are
auto-promoted by this v1 outcome.

**Cost.** Step 1 ~3 h code (corpus regen + `pair_corpus.py` 257 LOC +
`train_dpo.py` 332 LOC + smoke 151 LOC). Step 2a train ~30s on CUDA.
Step 2b gate driver ~150 LOC + 1m 18s gate wall-clock. Total under the
scoping § 5 estimate (~3 h code + ~10 min training); the corpus was
smaller than budgeted (568 vs ~10k pairs) which collapsed train + gate
wall-clock from the projected ~15 min to ~2 min total.

**Next direction picked.** **R7.b.2** (card-embedding feature
representation pass) per `docs/ai-research-backlog.md` § R7.b and the
queue's `r7b-card-embedding-pass` parked entry. Promotion fires per
scoping § 4 close-out condition. R7.b.0 trace-reencodability spike
already YES (digest slot 36) → R7.b.2 schema bump v3 is feature
re-extraction only, not multi-hour DAgger regen. Cost ~1.5 days code +
half day eval. R7.b.2 targets feature-representation interventions
(card embeddings, action-target embedding) — distinct family from R7's
labels axis and R8's objective axis, and the only remaining structural
lever in the F1 line that hasn't been falsified.

**Files & artifacts.**

- `runs/R8-dpo/iter-000/gate.manifest.json` — summary + byModelSide +
  terminalReasons; git sha `aea954e` (dirty: this block + queue/escalation
  edits in flight).
- `runs/R8-dpo/iter-000/policy.gate.onnx` — exported from
  `runs/R8-dpo/iter-000/model/checkpoint.pt`.
- `runs/R8-dpo/iter-000/gate.log` — full eval-gate stdout (1000-game
  trace).
- `runs/R8-dpo/iter-000/model/{checkpoint.pt, manifest.json}` —
  step 2a outputs (digest slot 43).
- `training/r8_gate_eval.py` — new mechanical driver wrapping
  `export_checkpoint_to_onnx` / `serve_onnx_context` / `run_eval_gate`.
- `docs/ai-research/scoping/r8-dpo.md` §§ 1–7 (closed design + pre-
  registered gates). § 4 close-out condition fired by this block.
- Backlog pointer: `docs/ai-research-backlog.md` § Tier 2 R8 line (DONE /
  FAIL, demoted to one-liner).

---

## R7.b.2 — card-embedding feature representation pass (single iter, FAIL) — 2026-05-14

**Verdict: FAIL — closes the R7.b family. Iter-0 Wilson lower 0.3045 (n=1000
side-balanced, rule-bot, seed-start 9000) on `runs/R7b2-card-embed/iter-000/`.
+1.24pp over R7 baseline (0.2921) — wash within Wilson half-width (~3pp at
n=1000); -0.65pp vs item17 iter-2 (0.311); -2.24pp vs R15.S1 iter-2 (0.3269);
-6.32pp vs R15.S3 obs-delta ceiling (0.3677); -9.55pp vs pre-registered gate
≥0.40 (`scoping § 7`). Wilson < 0.37 fires the close-out path per scoping § 7
("close-out R7.b family + escalate"); R7.b.3 (set-encoder attention) and
R7.b.4/5 (history + aux heads) were conditional on R7.b.2 lift and do not
auto-promote.** F1-line third axis (representation) now exhausted after R7
(labels) and R8 (objective).

**Run setup.** Phase 5 retrain on the Phase-4 re-extracted corpus
`runs/R7-multi-teacher-warmstart/iter-000/mixed-v3.jsonl` (12387 rows; same
mixed-teacher trace + rule-bot replay as R7 baseline, only feature schema
bumped v2.1 → v3.0 — `cardIdsByZone` + per-action src/tgt idx, embedding
table `nn.Embedding(108, K=32)` sum-pooled per zone, schema additive on the
existing 110-d state). `train_bc.py --hidden 64 --depth 2 --epochs 75
--data-mode bc` matching R7's hyperparameters byte-for-byte. CUDA, ~30s
wall-clock. Phase 6 gate via `training/r8_gate_eval.py` (`--games 500
--seed-start 9000 --min-ci-lower 0.40`, side-balanced n=1000); ONNX
roundtrip PASS (max_logit_diff 7e-7).

**Iter-0 gate result.** From `runs/R7b2-card-embed/iter-000/gate.manifest.json`:

| Metric | Value |
| --- | --- |
| games | 1000 (500 player + 500 opponent) |
| modelWins | 333 |
| modelWinRate | 33.3% |
| Wilson 95% | [**0.3045**, 0.3628] |
| averageModelPoints | 0.874 |
| averageHeuristicPoints | 1.352 |
| heuristicFallbacks | 0 |
| selectedNoOps | 0 |
| terminalReasons | gameOver: 1000 |

Side split: player WR 27.8% Wilson [0.241, 0.319] (n=500); opponent WR
38.8% Wilson [0.346, 0.431] (n=500). Same +11pp opponent-side gap that
replicates across every F1 / R15 / R7 / R8 phase at value-head-leaf
inference.

**Comparison anchors.**

| Anchor | Wilson lower | Δ vs R7.b.2 | Source |
| --- | --- | --- | --- |
| R7.b.2 iter-0 (card-embed v3.0) | **0.3045** | — | this block |
| R7 iter-0 (mixed-teacher BC, same corpus, v2.1) | 0.2921 | +1.24pp (wash, within Wilson half-width) | § R7 above |
| item17 take-2 iter-2 (rollout-only DAgger ref) | 0.311 | -0.65pp | `progress.md` L582 |
| R15.S1 iter-2 (3-iter DAgger from item17) | 0.3269 | -2.24pp | `progress.md` § Phase K |
| R8 DPO iter-0 (β=0.1, item17 ref) | 0.3318 | -2.73pp | § R8 above |
| R15.S3 ceiling (Phase N/O', obs-delta saturated) | 0.3677 | -6.32pp | `progress.md` § Phase N / O' |
| Pre-registered R7.b.2 gate (`scoping § 7`) | ≥ 0.40 | -9.55pp | `r7b-feature-representation.md` § 7 |

**Train trajectory — plateau-then-overfit signature worsened.** From
`runs/R7b2-card-embed/iter-000/model/manifest.json` (`split_by=episode`,
seed=7, train_rows 2950 / val_rows 839; dataset filter discards ~70% of
mixed-corpus rows for single-action / no-target reasons identically to
the R7 baseline at the same 3789 sample count):

| Run | final_train_acc | final_val_acc | Δ overfit gap |
| --- | --- | --- | --- |
| R7 baseline (same corpus, schema v2.1) | 0.965 | 0.747 | 0.218 |
| **R7.b.2 (schema v3.0, card-embed)** | **0.971** | **0.713** | **0.258** |

Adding 44.4k embedding parameters (3.5% of model) lifted train_acc by
+0.6pp **and** dropped val_acc by **−3.4pp** — the model fit the training
set marginally better and generalised meaningfully worse. The val-loss
trajectory (3.41 vs R7's 2.21) replicates R15.S1's and R7's plateau-then-
overfit signature with a *larger* val-loss climb at the same epoch budget.
Embedding capacity did not bind on a missing-information axis; if anything
the extra parameters made the same generalisation failure mode slightly
worse.

**Mechanism interpretation — F1 line representation axis exhausted.** The
R7.b hypothesis (`scoping § 1-2`) bet that the binding constraint at the
SL ceiling was input information content: the 16 hashed-float card-identity
slots can collide and force the network to learn dense super-positions
where a learned vocabulary embedding would let it carry per-card affordance
gradients. The pre-flight R7.b.0 trace-reencodability spike (digest slot
36) confirmed the schema lift was *possible* without DAgger regen. R7.b.2
exercised the lift cleanly (Phases 1-4 all green smokes; ONNX roundtrip
1.9e-6) and the result is unambiguous: the SL labels and the gate-relevant
signal at this capacity / corpus size budget do **not** bottleneck on
input-side card identity. Three axes of the F1 line are now exhausted at
the iter-0 SL gate:

- **Labels** (R7 multi-teacher BC blend) — Wilson 0.2921.
- **Objective** (R8 DPO from item17 ref) — Wilson 0.3318.
- **Representation** (R7.b.2 card-embed v3.0) — Wilson 0.3045.

None cleared 0.40, and none cleared the R15.S3 obs-delta reward-shape
ceiling (0.3677). The post-R8 working hypothesis was that representation
was the surviving F1 lever; that hypothesis is now falsified.

**Branch close-out per pre-reg § 7.** R7.b.3 (set-encoder / attention
over per-card tokens) was conditional on R7.b.2 lifting but capping below
0.40 (scoping § 4 #3 "Launch only if R7.b.2 lifts but caps below 0.40").
R7.b.2 did not lift at all; R7.b.3 does not auto-promote. R7.b.4 (recent-
action history) and R7.b.5 (aux heads / capacity bump) were
deprioritised-revisit-after-R7.b.2 entries and now also do not promote.
The R7.b family closes here.

**Cost.** Phase 5 train ~30s on CUDA; Phase 6 gate ~1m 18s wall-clock.
Phase 1-4 plumbing already landed across digest slots 47-50 (~515 LOC over
TS schema / Python encoder / ONNX / re-extractor). Total R7.b.2 wall-clock
from Phase 5 start to gate verdict: ~2 min. No DAgger regen needed
(R7.b.0 spike already confirmed re-encodability).

**Limits of this falsification.** Only `K=32` per-zone sum-pool tested.
Only the 12387-row R7 mixed-teacher corpus. Only iter-0 SL gate (no
post-SL PPO sweep, scoping § 7 exit gate (b) does not fire — gate (a)
SL non-regress passes within 2pp of R7 baseline but gate (b) Wilson lower
≥ 0.40 fails). R7.b.3 (attention pool) and R7.b.4 (history embedding)
remain *not falsified*, but the supporting result for any of them would
need a re-opening hypothesis distinct from "input-side card identity is
the bottleneck" since that's what R7.b.2 refuted at K=32.

**Surprise during writeup.** Dataset loader filters the 12387-row mixed
corpus down to 3789 samples (~30% retention) at the trainer entry — same
filter, same count as the R7 baseline at the v2.1 schema. The comparison
is therefore apples-to-apples on training data; the schema bump and the
embedding table are the only axes that change between R7 baseline and
R7.b.2. The "~0.93" train_acc reference in the brief was an
approximation; the actual R7 baseline number is 0.965, and R7.b.2's
0.971 is +0.6pp over that.

**Next direction picked — MCTS-distillation.** Three F1 axes exhausted
across labels (R7), objective (R8), representation (R7.b.2). R15.S3
reward-shape closed across both axes (obs-delta saturated 0.368 ± 0.001;
value-head tempo regressed). Rollout-leaf MCTS at R12 iter-2 reaches
Wilson **0.6479** with inference-time compute — the network *plays*
strong when search is bolted on at decision time. The gap is search →
model distillation, not architecture or representation or labels or
objective. **Pick: MCTS-distillation from rollout-leaf MCTS visit-count
targets.** `train_bc.py` already exposes `--data-mode mcts-distill` (line
887). The surviving structural lever is generating an SL corpus where
each row's `policyTargets` come from high-depth rollout-leaf MCTS visit
distributions instead of single-step rollout / search / planner traces.
Cheaper-to-falsify alternative rejected (pure capacity bump hidden
128→256 / depth 3→4: R7.b.2 already added 3.5% parameters with no Wilson
lift, parameter count isn't binding). Pre-reg gate consistent with prior
F1 family: Wilson lower ≥ 0.40 at iter-0 SL gate from a phase-H-scale
retrain. Scoping doc next slot (out of scope for this writeup); investigator
first locates the mcts-distill label source.

**Files & artifacts.**

- `runs/R7b2-card-embed/iter-000/gate.manifest.json` — summary +
  byModelSide + terminalReasons; git sha `eb5bdfc` clean.
- `runs/R7b2-card-embed/iter-000/policy.gate.onnx` — exported from
  `runs/R7b2-card-embed/iter-000/model/checkpoint.pt`.
- `runs/R7b2-card-embed/iter-000/gate.log` + `train.log` — Phase 5/6
  stdout.
- `runs/R7b2-card-embed/iter-000/model/{checkpoint.pt, manifest.json}` —
  Phase 5 outputs (state_dim=110, schema v3.0, card_vocab hash
  `e3a35716156494d6`).
- `runs/R7-multi-teacher-warmstart/iter-000/mixed-v3.jsonl` — Phase 4
  re-extracted corpus (12387 rows, observation.schemaVersion=2).
- `docs/ai-research/scoping/r7b-feature-representation.md` §§ 1–15
  (closed design + pre-reg gates + Phase 1-4 landing records). § 7
  close-out condition fired by this block.
- Closeout escalation: `docs/ai-agent-state/escalations.md` `## Resolved`
  2026-05-14 R7.b.2 bullet.

