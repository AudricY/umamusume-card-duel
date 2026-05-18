# R8. Direct Preference Optimization (DPO) — Scoping

- **Date:** 2026-05-14
- **Status:** scoping in progress (no code yet)
- **One-liner:** Replace PPO with a Bradley-Terry pairwise loss over (top-1, runner-up) preference pairs sourced from rollout-CRN per-candidate rewards, trained against a frozen item17 warm-start as the reference policy.
- **Forward brief:** `docs/ai-research-backlog.md` § "R8. DPO (Direct Preference Optimization)" (lines 106–122).
- **Pick rationale:** R7 closed FAIL (`docs/ai-research/progress/r15.md` § "R7 — multi-teacher BC blend"). Same SL-warm-start teacher-target axis exhausted; R15.S3 already closed both reward-shape axes. Cheaper-to-falsify rule selects R8 (~3 h code + ~10 min training) over R7.b.2 card-embedding (~1.5+0.5 days).

## 1. Question (Q)

> Given rollout-CRN gives us (state, top-1-action, runner-up-action) tuples with score margins, can DPO push past the F1 cap on pairs alone without per-step reward shaping (which R15.S3 saturated)?

Falsifiable proposition: a DPO pass from the item17 warm-start either (a) lifts Wilson lower ≥ 0.40 on the rule-bot gate, or (b) documents a new ceiling and the branch closes.

## 2. Hypothesis + Motivation

**Hypothesis.** The dense pairwise signal "rollout-CRN ranks A above B by margin m" is more learnable than the sparse ±1 terminal reward PPO trains on, and bypasses the SL-imitation cap that BC-style objectives inherit from the teacher's argmax-quality.

**Prior-art evidence chain (all close at Wilson 0.31–0.37).**

- R5 (PPO self-play) — FAIL (`progress.md` § R5).
- R15.S2 W6 PFSP — FAIL (`progress.md` L754, "W6/iter-2 warm-start, ~30pp stronger pool" without lift).
- R15.S3 axis 1 (observation-delta shaping, 4 phases L/M/N/O') — converged at iter-2 Wilson 0.368 ± 0.001 (`progress.md` § Phase O' closeout L736).
- R15.S3 axis 2 (value-head tempo, phases O+P) — regressed to 0.350 / 0.346 (`progress.md` § Phase P).
- R7 (multi-teacher BC blend) — FAIL iter-0 Wilson 0.2921 (`progress.md` § R7, `r15.md` § R7 result block).

Five branches across two SL axes (single + multi teacher) and two PPO axes (reward shape + warm-start pool) all cap at ~0.31–0.37. DPO is the first move that changes the **objective family** itself: it is not BC (cross-entropy on argmax) and not PPO (advantage from terminal reward); it is direct optimisation of `P(y_w ≻ y_l | x)` under a Bradley-Terry log-odds, anchored to a reference policy by a KL-like penalty.

**Why the bet is alive.** The R12 rollout-leaf MCTS GO (Wilson 0.6479 at eval time) shows the rollout signal is *informationally sufficient* to play strongly — the issue is the distillation objective, not the signal. R8 keeps the rollout signal but replaces both the objective and the training problem shape.

**Why the bet may be empty.** DPO from a Wilson-0.31 reference policy may inherit the reference's failure modes — the Bradley-Terry loss only re-weights between candidates the reference already considers, so if the reference assigns near-uniform probability to bad actions, DPO has limited gradient. See risk (c) below.

## 3. Design

### 3.1 Preference-pair source

**Status: existing outcome-export emits per-candidate rewards; the dagger-trace path used by R15.S1/R7 does not. The next implementer slot must pick one path.**

Two existing rollout-CRN paths score every candidate against shared CRN seeds:

- **`backend/src/sim/exportOutcomeTrainingExamples.ts` (preferred for v1).** Emits one `TrainingExample` per state with `oracle.candidates[].rewardMean` for every legal action (`exportOutcomeTrainingExamples.ts:176-184`), an explicit `oracle.selectedVsRunnerUpMargin` (line 168), `oracle.selectedOriginalRank` (line 173), and `oracle.sampleCount` (line 164). The runner-up is computed at line 140 as the highest-`reward` candidate ≠ best, margin at line 141. **This is the file the R8 forward brief refers to as "outcome-export's per-candidate reward records."** Schema is sufficient for DPO without extension.
- **`backend/src/sim/evaluateModelVsHeuristic.ts::chooseRolloutAction` (line 739–782).** Computes per-candidate `reward` across CRN seeds internally but returns only `bestIndex` via `rankedDecision` (line 781). The dagger relabel pass (`backend/src/sim/dagger/relabelDecisionTrace.ts:24-43`) writes `TeacherEntry` rows with `selectedActionId/Index` but **no per-candidate rewards and no runner-up margin**. R15.S1's and R7's trace JSONLs therefore *cannot* be re-used for DPO without re-running trace-gen or extending this emission.

**Pick: outcome-export.** Reasons: (i) schema is already DPO-compatible, no TS-side change needed; (ii) `oracle.candidates` carries the full per-candidate vector, so v1+ extensions (top-K pairs, full ranking loss) are unblocked; (iii) the `oracle.sampleCount` / `rewardVariance` fields let us implement the noise-threshold mitigation (risk (a)) cheaply.

**v1 pair-construction rule.** Per state, pair `y_w = oracle.candidates[0].action` (best by `rewardMean`) with `y_l =` the next-best candidate by `rewardMean` (i.e. the row already-computed runner-up). Discard rows where `oracle.selectedVsRunnerUpMargin < tau` (tau threshold per risk (a) below) or where `legalActions.length < 2`. One pair per state for v1; multi-pair per state (top-1 vs ranks 2…K) is out of scope.

### 3.2 Trainer location

**Pick: new file `training/train_dpo.py`.** Justifications:

- The DPO loss is structurally different from cross-entropy on labels — it consumes (state, y_w_action_features, y_l_action_features, action_mask) and computes log-probabilities for two specific actions per row, not a softmax over all actions. The existing `train_bc.py:355-371` `run_epoch` loop is built around `(state_features, action_features, action_mask, targets)` with a single forward pass + masked log-softmax + CE. Bolting a pair-loss branch onto that loop tangles two unrelated objectives; the file is already 750+ lines.
- The reference-policy forward pass (frozen item17 warm-start) is a second model invocation per batch — distinct from the optional `anchor_model` KL-divergence path at `train_bc.py:373-377`, which computes a soft KL to the anchor on the same logit set. DPO needs `log π(y_w|x)` and `log π(y_l|x)` from *both* policies, not a distribution-level KL.
- The dataset shape differs (see § 3.3): two action-feature slots per row, not the padded `(B, max_actions, ACTION_DIM)` matrix. A parallel collate is cleaner than overloading `collate_policy_batch` (`training/uma_ai/dataset.py:90-134`).

A small shared utilities module is fine — `train_dpo.py` may import `CandidatePolicyNet`, `masked_log_softmax_logits`, `load_init_from_checkpoint`, and the state/action feature builders from existing locations.

### 3.3 Loss function + hyperparams

Standard DPO loss against the reference policy `π_ref`, with policy `π_θ`:

```
loss = -log_sigmoid(β * ( (log π_θ(y_w|x) - log π_θ(y_l|x)) - (log π_ref(y_w|x) - log π_ref(y_l|x)) ))
```

- **β = 0.1.** DPO paper default (Rafailov et al. 2023 use β ∈ {0.1, 0.5}; 0.1 is the headline setting). Lower β = stronger anchor to reference; higher β = freer to move away. For v1 we want to confirm that the *objective* moves the needle; a small β reduces the chance of catastrophic divergence from a Wilson-0.31 reference policy. If v1 PASSes but caps below 0.40, β-sweep is the natural one-shot follow-up (risk (b)).
- **`log π(y|x)`** is `log_softmax(masked_logits)[action_index]` — same `masked_log_softmax_logits` (`train_bc.py:563`) used by the BC soft-CE branch. This guarantees masked positions are correctly handled.
- **Value head.** Not optimised in v1 — DPO is a policy-only objective. The value head from the item17 warm-start is preserved as-is (matches PPO phase-H behaviour). Inference-time pickers that read the value head are unchanged.

### 3.4 Dataset shape + collate

Per-row tensors:

- `state_features` — `(STATE_DIM,)` from `observation_to_features` (existing).
- `legal_action_features` — `(num_actions, ACTION_DIM)` from `legal_actions_to_features` (existing), kept so the model can compute the full action logits (needed for `log_softmax`).
- `action_mask` — `(num_actions,)`.
- `y_w_index`, `y_l_index` — int64 scalars, indices into `legal_action_features`.

Collate pads `legal_action_features` to `max_actions` like `collate_policy_batch` already does, and emits parallel `y_w_index` / `y_l_index` int tensors. Forward pass: one call to `model(state, action_features, mask)` per batch, take `log_softmax` over `mask`, gather at `y_w_index` and `y_l_index`. Reference policy: same forward + gather under `torch.no_grad()`. Memory cost is ~2× the BC forward (policy + reference).

### 3.5 Reference policy checkpoint

`runs/item17-2026-05-11/iter-002/model/checkpoint.pt`. This is the canonical F1 / R15.S3 PPO warm-start (`progress.md` L2266 cites the same path; `progress.md` § Phase H/L/M/N/O'/O/P all `--init-from-checkpoint` from this file). Iter-2 Wilson lower 0.311 (`progress.md` L582 / r15 § R7 anchor table). Initialised the same way as the BC trainer's `load_init_from_checkpoint` (`train_bc.py:268-280`): load `payload["model_state"]` into a fresh `CandidatePolicyNet` with the same `(hidden_dim=64, depth=2)` config, freeze (`requires_grad_(False)`, `.eval()`).

The trained DPO policy is *also* initialised from this checkpoint (warm-start = reference at step 0, log-ratio term starts at 0).

### 3.6 No `mix-sources`

Match R7 v1 (`r7-multi-teacher-warmstart.md` § 7): train on pure rollout-CRN preference pairs from a single outcome-export corpus. Mixed-corpus pairing semantics (how to pair across heterogeneous score scales) is out of scope. Rule-bot replay mixing — the dagger-orchestrator pattern — is also out of scope; rule-bot replay has no per-candidate rewards and the BC pretrain already absorbed that information into the warm-start.

### 3.7 Eval target

`npm run sim:eval-gate` against the rule-bot at **n=500 side-balanced** (250 player + 250 opponent), Wilson 95% lower bound. Seed range matches R15.S1's iter-1 SL gate evaluation seeds (also identical to R7 step 4's gate config per `r15.md` § R7 result block) so the comparison is honest. Same gate the entire F1/R15/R7 line is measured against.

## 4. Pre-Registered Exit Gates

Verbatim from `docs/ai-research-backlog.md` lines 119–120:

> Wilson lower ≥ 0.40 from the item17 warm-start on the rule-bot gate, or document the new ceiling and close.

**Gate evaluation procedure.**

- **n=500, side-balanced** (250 player + 250 opponent). Wilson 95% lower bound.
- **Reference anchors at gate time:** item17 take-2 iter-2 Wilson 0.311 (`progress.md` L582), F1 Phase H 0.3109 (`progress.md` § Phase K), R15.S3 ceiling 0.3677 (Phase N/O'). R8 lifts to PASS iff Wilson lower ≥ 0.40 — i.e. **+3.2pp over the R15.S3 ceiling** and **+9pp over the SL warm-start**.
- **Close-out condition.** If Wilson lower < 0.40, write the DPO ceiling number into `docs/ai-research/progress/r15.md` § "R8 — DPO" result block, demote backlog § R8 to a one-line pointer, and defer the next move to either R7.b.2 (card embeddings, the next-cheapest forward-brief) or escalation.

## 5. Cost Estimate + Risk Callouts

Per forward brief: **~3 h code + ~10 min training.** Breakdown:

- ~30 min: outcome-export pair-construction + tau-filter helper (Python-side reader for the existing JSONL).
- ~1.5 h: `training/train_dpo.py` (~250 LoC: dataset, collate, loss, train loop, checkpoint save).
- ~30 min: smoke test in `training/tests/dpo_smoke.py` (or extend `training/smoke_e2e.py`).
- ~10 min: train wall-clock (single epoch over ~10k preference pairs at the R15.S1 corpus size).
- ~10 min: gate eval (250-game pass per existing F1 cadence).
- ~30 min: writeup.

**Risks.**

- **(a) Preference pairs are noisy at rolloutCrnSamples=3.** Rollout-CRN with K=3 shared seeds (`dagger_orchestrator.py:1042` default; same K used by R15.S1 and R7) yields a per-candidate `rewardVariance` field (`exportOutcomeTrainingExamples.ts:184`) that is non-negligible relative to typical margins. If `selectedVsRunnerUpMargin < sqrt(rewardVariance / K)`, the pair could be the wrong ordering. **Mitigation:** drop rows where `oracle.selectedVsRunnerUpMargin < tau`. Pick `tau = max(0.02, sqrt(mean(rewardVariance) / 3))` as a starting heuristic, computed from a 100-row probe of the corpus. The existing tie threshold default is 0.02 (`exportOutcomeTrainingExamples.ts:234`); use that as a floor. Report `kept_pairs / total_states` in the result block.
- **(b) β tuning.** β=0.1 is the DPO paper default but β∈{0.3, 0.5} lifted some downstream evals in later literature (e.g. Llama-Guard variants). **v1 is β=0.1 only.** If v1 PASSes (Wilson lower ≥ 0.40), no β sweep needed. If v1 lifts but caps below 0.40 (e.g. lands at 0.37–0.39), a one-shot β=0.3 follow-up is the pre-registered next move; flag in the result block but do not auto-launch.
- **(c) Reference-policy ceiling.** If item17 (Wilson 0.31) is itself the binding constraint, DPO inherits the imitation cap — the Bradley-Terry loss only re-weights probabilities already non-zero under `π_ref`. **Mitigation:** DPO is by design an imitation-bypass (it changes the gradient direction even when the reference is weak), but interpret a Wilson-0.31–0.37 outcome as "DPO did not lift this warm-start" rather than "DPO is the wrong objective." A failed v1 leaves the door open for R8.b: DPO from the R15.S3 phase-N warm-start (Wilson 0.368) as a stronger reference. R8.b is *not* part of v1's pre-registered exit — only flagged here.
- **(d) Single-pass vs iterative DPO.** v1 is one DPO pass from the existing preference corpus. Iterative DPO (refresh pairs from the trained policy's visited states, DAgger-style) is out of scope and would convert the cost from ~10 min to ~30 min × iters. If v1 PASSes, iter-2 DPO is the natural next move; not pre-registered here.

## 6. Ordered Execution Plan

1. **(Pre-flight, ~20 min) Audit outcome-export schema.** Confirm `oracle.candidates[].rewardMean`, `selectedVsRunnerUpMargin`, `rewardVariance`, `sampleCount` are present on a recent `exportOutcomeTrainingExamples` run (any `training/runs/outcome*/examples.jsonl`). If the existing corpus is stale, run `npm run sim:export-training` once with `--rollout-crn-samples 3 --samples 3 --games 90 --max-examples 10000` to match R15.S1's compute budget. Compute the tau-threshold heuristic from a 100-row sample.
2. **(~30 min) Write `training/pair_corpus.py`** — a small loader that reads outcome-export JSONL, filters by tau, and yields `(state_features, action_features, mask, y_w_index, y_l_index)` rows. Reuse `observation_to_features` and `legal_actions_to_features` from `uma_ai/dataset.py`. Skip the existing `JsonlPolicyDataset` plumbing — DPO has its own collate.
3. **(~1 h) Write `training/train_dpo.py`** with the Bradley-Terry loss in § 3.3, the dataset/collate in § 3.4, the reference-policy load in § 3.5. CLI mirrors `train_bc.py` argparse for `--epochs / --batch-size / --hidden-dim / --depth / --device / --out-dir`. Add `--beta 0.1`, `--reference-checkpoint`, `--init-from-checkpoint` (same checkpoint as reference for v1). Save `checkpoint.pt` + ONNX export via the existing `export_onnx.py` helper.
4. **(~30 min) Smoke test `training/tests/dpo_smoke.py`** — synthetic 5-row preference batch, assert (a) loss is finite, (b) gradients flow on policy params, (c) reference params have no grad, (d) loss strictly decreases over 10 SGD steps on the same batch. Wire into the same harness as `training/dagger_smoke.py` or extend `training/smoke_e2e.py`.
5. **(~10 min) Train.** `python training/train_dpo.py --init-from-checkpoint runs/item17-2026-05-11/iter-002/model/checkpoint.pt --reference-checkpoint runs/item17-2026-05-11/iter-002/model/checkpoint.pt --beta 0.1 --epochs 5 --out-dir runs/R8-dpo/iter-000/model/`. Single epoch should suffice; 5 epochs is a safety margin still under the 10-min budget.
6. **(~5 min) Gate eval.** `npm run sim:eval-gate` n=500 side-balanced vs rule-bot, same seed range as R7 step 4. Record Wilson interval.
7. **(~30 min) Writeup.** Result block in `docs/ai-research/progress/r15.md` § "R8 — DPO" with the verdict, kept-pairs fraction, β value, reference-policy anchor Wilson, and comparison-anchor table. Backlog § R8 collapses to one-line pointer per CLAUDE.md documentation discipline. If FAIL, log the tau threshold used + per-margin-bucket Wilson and close.

## 7. Out of Scope (deliberately deferred)

- **Mixed-source preference corpus.** Single outcome-export source only for v1.
- **Multi-iter / iterative DPO** (DAgger-style refresh from the trained policy's visited states).
- **β sweep** (v1 = β=0.1 only; sweep flagged as one-shot follow-up under risk (b)).
- **Alternate reference policies** (e.g. R15.S3 phase-N warm-start at Wilson 0.368) — R8.b candidate, not part of v1.
- **Value-head fine-tuning under DPO** — policy-only loss in v1.
- **Reward-shape signals** — R15.S3 closed both axes; not re-opened.
- **Top-K / full ranking loss** (e.g. ListNet, Plackett-Luce over the full `oracle.candidates`) — one pair per state in v1.
- **R7.b family** (feature representation expansion) — parked behind R8 result per backlog § R7.b.
