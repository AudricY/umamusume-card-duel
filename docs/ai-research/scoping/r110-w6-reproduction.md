# r110-w6-reproduction — Scoping

- **Date:** 2026-05-15
- **Status:** scoping
- **One-liner:** Re-run the *exact W6 Phase D rollout-leaf MCTS self-play + mcts-distill loop* that produced the pinned 96-d production model (search-wrapped Wilson 0.6479), but at the current 110-d / state-schema-v3.0 feature representation, to produce a 110-d model non-inferior (ideally superior) to the pinned 96-d model under the same rollout-leaf search wrap — thereby unlocking the `serve_onnx --feature-schema v3` "promote later" path.
- **Forward brief:** queue entry `r110-w6-reproduction`.
- **Pick rationale:** User-commissioned next-arc. The 96-d production strength is a *search* property, not an SL-net property; the loop that produced it is schema-portable and the pipeline is already 110-d end-to-end. Strictly-richer features through the identical pipeline has no principled reason for inferiority.

## 1. Question (Q)

> Does the W6 Phase D loop — rollout-leaf MCTS self-play → soft-visit `mcts-distill` BC, iterated with relative-improvement promotion — reproduce its 96-d strength when run at the 110-d / schema-v3.0 representation, evaluated **search-wrapped** at the F1-standard gate?

**Falsifiable proposition.** Running `training/r12_orchestrator.py --mcts-leaf rollout` from a 110-d init for ≥3 iterations at the W6 iter-2 hyper-config (§4), the best-promoted 110-d checkpoint, exported via `export_onnx.py` (5-input v3 graph) and served via `serve_onnx --feature-schema v3`, evaluated at the iter-2 gate protocol (n=120 = 60 player + 60 opponent, rule-bot opponent, rollout-leaf MCTS both sides, 100 sims, K=3, seed-start 9000), reaches **Wilson lower ≥ 0.6479**.

## 2. Hypothesis + Motivation

**Hypothesis.** The pinned model's strength is supplied by the rollout-leaf MCTS search signal (model-independent rule-bot rollout leaf value + model-driven PUCT prior), not by the SL net's raw policy. The W6 loop is the procedure that converts that search signal into a usable prior/value net. That procedure is representation-agnostic: feeding it the strictly-richer 110-d / v3.0 features (96-d v2 superset + card-embedding inputs) should produce an at-least-equivalent search-wrapped player.

**Why this is NOT a re-run of the closed F1 SL line.** The naive "SL-train a 110-d net up to the pinned one" was already executed and falsified: **R7.b.2** is exactly the 110-d card-embedding *representation* SL pass — iter-0 Wilson lower **0.3045** (n=1000, side-balanced, rule-bot, seed-start 9000) vs 96-d R7 0.2921 (+1.2pp, wash inside the Wilson half-width); val_acc *dropped* 3.4pp (the embedding increased overfitting on the F1 corpus). Detail: `docs/ai-research/progress/r15.md` § R7.b.2 (rolled to `r15-archive.md`). A further single-pass SL variant, **MCTS-distill v1** (soft-visit `mcts-distill` BC at 110-d from item17), also failed at raw-policy Wilson 0.1470 — state-coverage mismatch, the pre-registered failure mode (`docs/ai-research/progress/r15.md` § "MCTS-distill v1"; `docs/ai-research/scoping/mcts-distill.md`). **Both falsified results are single SL passes evaluated raw-policy.** This arc is categorically different: the full *iterated self-play + distillation loop*, evaluated **search-wrapped**. That is the pipeline that already cleared 0.6479 at 96-d; nothing in it was 96-d-specific.

## 3. Pre-existing Evidence

- **96-d W6 provenance (the target to match).** `runs/R13-W6-phase-d/orchestrator-state.json` + `runs/R13-W6-phase-d/iter-2/{manifest,selfplay.manifest,gate.manifest}.json`: iter-2 search-wrapped Wilson lower **0.6479** (WR 0.7333, n=120, 0 fallbacks/no-ops, gameOver 120/120), git SHA `bc6db85`. Iter trajectory 0.536 → 0.570 → **0.6479** → 0.578 → 0.561 (auto-halted iter-4 after 2 consecutive promotion misses). Pinned production artifact: `runs/R13-W6-phase-d/iter-2/policy.onnx` (96-d, schema v2).
- **W8 control — the loop's load-bearing half.** Substituting value-head leaf for rollout leaf in self-play regressed every iteration below baseline (`docs/ai-performance-research-progress.md` § R13.W8). Rollout-leaf self-play is mandatory; this arc keeps `--mcts-leaf rollout`.
- **R7.b.2 (110-d SL representation) falsified the SL route** — see §2. Pointer, numbers not re-pasted: `docs/ai-research/progress/r15.md` § R7.b.2.
- **Pipeline already 110-d / v3.0 end-to-end** — see §5 readiness audit.

## 4. Loop / Sweep Recipe (reconstructed from W6 iter-2 manifests)

Identical to the W6 Phase D run, schema-portable. Source: `runs/R13-W6-phase-d/iter-2/selfplay.manifest.json` (self-play args), `iter-2/manifest.json` + `iter-2/distill.log` (distill), `iter-2/gate.manifest.json` (gate), `orchestrator-state.json` (per-iter cost).

**Orchestrator:** `training/r12_orchestrator.py`, `--mcts-leaf rollout` (NOT the `value-head` default).

**Per-iteration self-play (`sim:mcts-selfplay`):**
- 60 games/iter, `mctsSimulations=100`, `mctsLeaf=rollout`, `mctsRolloutCrnSamples=3` (K=3), `mctsRolloutSteps=200`, `mctsCPuct=1.5`, `mctsPrior=policy`, `mctsCollapseMaxSteps=64`, `mctsMaxNodes=5000`
- root Dirichlet on: α=0.3, ε=0.25; `temperatureMoves=6`, `temperatureValue=1.0`
- `--selfplay-seed-start 30000` (W6 used 30120; pick a fresh non-overlapping range), 4 workers; ~2000 rows/iter

**Distillation (`train_bc.py --data-mode mcts-distill`):**
- 20 epochs (W6 `distill.log` shows 20 epoch records; pass `--epochs 20`, not the orchestrator's 25 default), `--batch-size 64`, `--lr 3e-4`, `--value-weight 1.0`, `--policy-weight 1.0`, `--split-by seed`
- KL anchor 0.05 vs the *previous iteration's* checkpoint (`--kl-anchor-weight 0.05`); cosine LR; AMP off
- **Model config — the one explicit delta vs W6:** W6 auto-inferred `hidden_dim=64 / depth=2 / dropout=0.05` from its W3-retrained 96-d warm-start. This arc has **no cross-schema warm-start** (see §5 blocker note), so set `--hidden-dim 64 --depth 3` *or* `--hidden-dim 64 --depth 2` explicitly and hold it fixed across iterations. Recommendation: match W6's promoted `64/2` to keep the only variable the feature schema; revisit capacity only if the recipe is otherwise faithful and falls short.

**Per-iteration gate (`sim:eval-gate`):** n=120 (60 player + 60 opponent, `modelSide=both`), `opponentSelection=rule`, rollout-leaf MCTS both sides (`mctsLeaf=rollout`, sims 100, K=3, steps 200), `--eval-seed-start 9000`, `requireZeroFallbacks`, `requireZeroNoOps`. Promotion floor `--eval-min-ci-lower 0.30` (relative-improvement promotion; final non-inferiority bar is §6, not the per-iter floor). Crossover probe runs after distill (informational; never satisfied at 96-d, not a gate here).

**Iteration count.** W6 promoted at iter-2 and auto-halted at iter-4 (2 consecutive misses). Run ≥3 iters; let the orchestrator's halt-after-2-failures rule terminate. Take the best-promoted checkpoint.

**Cost (from `orchestrator-state.json` per-iter elapsed):** self-play ≈ 545–630 s, distill ≈ 5–7 s, gate ≈ 580–730 s ⇒ **≈ 19–23 min wall per iteration** at 4 workers. A faithful 3–5 iter run ≈ **1–2 h wall** + one final §6 confirmation gate at n≥120 (≈ 12–15 min). Single GPU; no new infra. This is a real (not trivial) compute job and must be flagged before launch per CLAUDE.md.

## 5. 110-d Pipeline Readiness (audited 2026-05-15)

**Verdict (revised 2026-05-18): READY end-to-end ONLY after R16-P0. The 2026-05-15 "no code blocker" verdict was wrong — the mcts-distill card-embedding data-path bug below confounded the first run. With R16-P0 landed and self-play corpora regenerated at v3, the only remaining delta vs W6 is config (no warm-start), not missing capability.**

- `training/uma_ai/features.py:22,31` — `STATE_DIM = 110`, `STATE_FEATURE_SCHEMA_VERSION = 3.0`. `train_bc.py` extracts 110-d / v3.0 *by default now*; W6's `manifest.json` shows 96-d/v2 only because it ran at SHA `bc6db85` predating the schema bump (R7.b.2 Phase 2, commit `301759e`).
- `training/train_bc.py` — `--data-mode mcts-distill` soft-visit branch unchanged; `feature_schema_metadata()` reports current 110-d/v3.0 + card vocab. **CORRECTION (R16-P0, commit pending, 2026-05-18):** the "Card-embedding inputs handled" claim audited here on 2026-05-15 was FALSE for the mcts-distill path and confounded the first R110-W6 run (`runs/R110-W6-repro-CONFOUNDED-p0bug-iter0only/`). `MctsSelfPlaySample` / `collate_mcts_selfplay_batch` / `load_mcts_selfplay_samples` never emitted `card_ids_by_zone` / `action_card_idx`, so `train_bc.py`'s `batch.get(...)` returned `None` and the 110-d model trained with the card-embedding branch **inert** under `--data-mode mcts-distill` (BC `--data-mode bc` was fine — `JsonlPolicyDataset` always emitted them). Fixed by R16-P0: the self-play dataset now emits the v3 embedding tensors (strict fail-loud on missing `cardIdsByZone`; named `allow_missing_card_ids` compat path for old corpora), `evaluate_grouped()` no longer hardcodes `collate_policy_batch`, and value-target consumers forward the optional tensors. Proven live by `training/r16_mcts_embedding_smoke.py` (nonzero grad on `card_embed.weight` and `zone_projection.weight` from one mcts-distill batch). **R110-W6 must be relaunched on v3-regenerated self-play corpora** — any mcts-selfplay JSONL produced before this fix lacks per-row embedding inputs in usable form for the loader's strict default.
- `training/export_onnx.py:60,67-68,76-94` — exports the 5-input v3 graph (`state`, `actions`, `mask`, `card_ids_by_zone`, `action_card_idx`) with dynamic axes; asserts `config.state_dim == STATE_DIM` (110).
- `training/serve_onnx.py:46,103-148` — `--feature-schema {auto,v2,v3}`; `auto` inspects the ONNX graph (`card_ids_by_zone` input + state last-dim) and resolves v3 for a 110-d card-embedding graph; explicit `v3` is asserted graph-consistent and fails loud on mismatch. `request_to_arrays(payload, schema)` encodes features **server-side** from game state.
- `backend/src/sim/mctsSelfPlay.ts:42,312,458` and `backend/src/sim/evalGate.ts` — both query the model purely via `--model-url` HTTP `/predict`; they send game state, not encoded features. **The TS self-play/gate path is schema-agnostic** — all v2/v3 encoding lives in `serve_onnx.py`. No TS change needed.
- Gate: `training/r8_gate_eval.py` / `sim:eval-gate` operate through the same server; schema-agnostic.

**The mcts-distill-v1 warm-start blocker does NOT apply here.** v1 failed in part on an arch/schema incompatibility when warm-starting a 110-d net from a 96-d (item17) checkpoint. This arc runs the W6 loop **from a 110-d init (random or a 110-d SL ckpt) with no cross-schema warm-start** — exactly because W6 itself only warm-started within its own schema (W3-retrained 96-d → 96-d loop). Same-schema-from-scratch sidesteps the incompatibility entirely. (Pick the 110-d init pragmatically: random-init is the cleanest faithful analogue since W6's warm-start was a *same-schema* SL ckpt; a 110-d SL ckpt e.g. R7.b.2's may speed convergence but is not required and adds a confound.)

## 6. Exit Gate

Final evaluation: best-promoted 110-d checkpoint, exported v3, served `serve_onnx --feature-schema v3`, gated at the **iter-2 protocol** — n≥120 (≥60 player + ≥60 opponent, `modelSide=both`), rule-bot opponent, rollout-leaf MCTS both sides (100 sims, K=3, 200 steps), seed-start 9000, 0 fallbacks / 0 no-ops required. Report aggregate Wilson lower + per-side split (W6 iter-2 reference: aggregate 0.6479, player 0.5577 lower, opponent 0.6638 lower).

- **PASS (non-inferior)** — aggregate Wilson lower **≥ 0.6479**: the 110-d model matches/beats the pinned 96-d model search-wrapped. Promote: re-export, switch the served model to the 110-d v3 artifact via `serve_onnx --feature-schema v3`, re-point the webapp opponent. Record as the new production model.
- **MARGINAL** — 0.60 ≤ Wilson lower < 0.6479: the loop reproduced at v3 but slightly under the pinned point. One faithful follow-up only: extend iterations to the orchestrator halt and/or take the single best-promoted iter; if still < 0.6479, do **not** promote (pinned 96-d stays production) and record v3 as "reproduced-but-not-superior". No representation/capacity tuning under this band (would re-open a closed axis).
- **FAIL** — Wilson lower < 0.60: the W6 loop does not reproduce at v3. This would be a genuinely new negative (the pipeline that worked at 96-d failed at strictly-richer features) → escalate to user with the per-iter trajectory; likely mechanism = the same state-coverage / overfitting dynamics that bit R7.b.2 and mcts-distill v1, now also biting the iterated loop. Do not auto-pivot.

## 7. Out of Scope

- Any raw-policy (no-MCTS) SL gate as the success metric — that is the closed F1 line; the metric here is **search-wrapped** only.
- Representation / capacity / objective tuning of the net (R7.b.* axes are closed; changing them re-opens a falsified line).
- Cross-schema warm-start (96-d → 110-d) — explicitly avoided; init is same-schema.
- Changing the MCTS search config (sims / K / leaf type / rollout steps) — the recipe is a faithful W6 reproduction; tuning search is a different experiment.
- PPO / DPO from the 110-d checkpoint — downstream, gated on PASS first.
- Multi-iteration tuning beyond the orchestrator's built-in halt-after-2-failures policy.
