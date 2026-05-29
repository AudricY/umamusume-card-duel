# v6 Relational-Trunk Model Scheme

- **Date:** 2026-05-29
- **Status:** IMPLEMENTED + SMOKE-GREEN + E2E-VALIDATED, UNTRAINED. The model,
  training-CLI wiring, and an ONNX/signature/trainability smoke landed first.
  On the `feature-improvement → feat/ai` merge the full ReBeL self-improving
  loop was run end-to-end on `--model-variant relational` (2-iteration CPU
  smoke: self-play → validate → distill → export → R19 head-to-head gate →
  promote-decision, clean halt). Two integration blockers were fixed there
  (commit `7c94157`): the dynamo ONNX exporter baked a static batch dim when
  traced at batch=1 (broke Rust batched leaf inference for the relational value
  head — now traced at batch=2 for relational only); and `rebel_orchestrator`'s
  `validate_rebel_rows` had a stale belief-dim literal (16 vs the live 123). No
  strength result yet — the pre-registered gate (below) has not been run.
- **Routing:** canonical home for the v6 *model/trunk* scheme until it
  resolves. New backlog entry under "Active Search-Wrapped Frontier". This is
  a **trunk-architecture** scheme, NOT a feature-schema bump — the state /
  action *feature* schema stays v3.x (action v5); "v6" names the model scheme.
- **Relationship to the set-attention probe
  (`set-attention-architecture-probe.md`):** v6 is the unconstrained version
  of that bet. The R7.b.3 probe added a **1-layer zero-init self-attention
  residual** bolted on top of the frozen v3.2 sum-pool trunk, so a v3.2
  checkpoint warm-started into it stayed bit-stable. That parity constraint
  capped it at d_model=64, 1 layer, and "additive lens on top of sum-pool".
  It landed MARGINAL (wl=0.3205 raw-policy SL, below the 0.40 bar). v6 drops
  the parity constraint entirely (the user directive is "backward
  compatibility doesn't matter") and makes attention the **primary trunk**.

## Why now — historical verdicts are stale

Two facts reframe the "the strength ceiling is ~0.59 and feature-tail
engineering is dead" reading of the backlog:

1. **The training paradigm is mid-shift (MCTS → ReBeL).** The corrected ReBeL
   line (R20, commit `b07c2d3`) supersedes R18/R19, which trained on a **buggy
   value target + an info-incorrect rollout leaf** (`project_rebel_correctness_r20`
   memory; R17-R19 postmortem in `docs/ai-research/progress/`). The
   `~0.59` tight-gate ceiling and the "v3.6/v3.7/v3.8/v5 all flat" verdicts
   were measured on the MCTS sum-pool line, much of it before the correctness
   fixes. They are **not load-bearing** for a forward bet and should be
   re-baselined, not cited as a hard ceiling.
2. **The additive-tail ladder and the trunk are different axes.** v3.3-v3.8
   added hand-engineered scalar channels (lethal/ETA/KO/coin-flip) to a frozen
   head consumed by a sum-pool MLP. That is the *featurization* axis. v6 holds
   featurization fixed (reuses the v3.2 tokenized signature) and changes the
   *trunk*. The case for v6 rests on a **structural** argument, not on the
   stale empirical ceiling.

## Hypothesis

The strength bottleneck is the trunk's weak relational prior, not the scalar
feature tail. In `training/uma_ai/model.py` the sum-pool trunk:

- sum-pools 32-d card embeddings *within* each of 8 zones, then linearly
  projects — cards within a zone are a commutative bag; cards across zones
  interact only through the joint MLP;
- adds a per-slot `uma_slot_encoder` whose cross-slot interaction is again a
  sum-pool into `state_encoded`;
- gives the action branch `[source_embed, target_embed]` with no mechanism for
  "which of my board cards threatens which of their targets."

A card game is relational (threat/trade/lethal reasoning is about *pairs and
sets* of board entities). The conjecture: a real multi-layer attention trunk
expresses those relations directly, and can **derive** much of the
hand-computed v3.6/v3.7 arithmetic tail (attack-damage-vs-HP lethal, energy
ETA) from card-id tokens + per-slot HP/energy features — making that tail the
sum-pool's crutch rather than a strength lever. **Falsifiable**: if v6 fails to
clear its gate, the arithmetic crutch (or the featurization itself) was
load-bearing, and the v6.1 fork (below) re-adds a richer signature.

## Design (landed)

`model_variant="relational"` in `ModelConfig`; implemented as `RelationalTrunk`
in `training/uma_ai/model.py`, dispatched at the top of
`CandidatePolicyNet.forward` (the sum-pool path is skipped entirely; the
relational checkpoint carries **no dead legacy params** — every sum-pool
submodule is `None`).

**Tokenization** (consumes the existing v3.2 / belief input tensors — no new
featurizer):
- `CLS` token = learned vector + `Linear(110-d state_features)`. Carries the
  global scalars (points/turn/phase/energy) the card tokens do not.
- Card tokens from `card_ids_by_zone [B,8,30]`: `card_embed(id)` projected to
  d_model, + zone-type embed + within-zone-position embed + **polarity embed**.
  Pad-masked where `id==0`.
- Uma slot tokens from `uma_slot_card_ids/features [B,10]/[B,10,23]`:
  `card_embed` + `Linear(slot features)` + slot-position embed + polarity
  embed. Pad-masked where `id==0`.
- **Belief token** (iff `uses_belief_features`): `Linear(belief_features)` +
  learned belief-type marker. This is the ReBeL-native move — the public-belief
  summary is a first-class token the attention mixes with board state
  relationally, replacing the inert zero-init additive `belief_encoder`
  residual the sum-pool used.

**Polarity embedding** (3 rows: neutral/own/opp), added to every token from a
frozen per-position polarity buffer derived from `ZONE_ORDER` / `UMA_SLOT_ORDER`
name prefixes. This is the relational analog of the per-side-asymmetry probe:
the two sides share weights and are distinguished by an explicit polarity
signal rather than separate sum-pool lanes, so the trunk reasons about "my
board vs their board" symmetrically. (Secondary hypothesis — does NOT claim to
resolve the +0.06 opponent-side gap, which may be intrinsic/featurizer; see
`per-side-asymmetry-probe-scoping.md`.)

**Trunk:** `relational_layers` (default 4) pre-LN `TransformerEncoderLayer`
(GELU, `batch_first`), d_model = `hidden_dim`, `relational_heads` (default 8),
FFN = `relational_ffn_mult * d_model` (default 2). CLS always unmasked → no
fully-masked attention row → no NaN.

**Candidate-conditioned policy head (cross-attention):** each action is a query
token (`Linear([action_features(57) ⊕ source_embed ⊕ target_embed])`) that
cross-attends over the encoded board tokens (`key_padding_mask` = the same pad
mask). `logit = MLP([attended ⊕ query ⊕ attended*query])`, masked to
`finfo.min` on illegal actions. Each candidate gets a board summary filtered
through its own lens — strictly stronger than a single shared pooled context,
and ONNX-clean (the action axis A stays a dynamic axis).

**Value head:** `tanh(MLP(CLS))` — whole public-state value; for a belief graph
the CLS has already attended to the belief token.

Sizes: hidden=128 ≈ 0.70M params, hidden=256 ≈ 2.70M params (verified by the
smoke). This is a *structural* change, not a capacity probe (R6 closes
capacity-from-above); the inductive bias is the point.

## Integration — the cheap-integration win

v6 emits `(logits, value)` positionally and consumes the **same** v3.2 (7-input)
/ belief (8-input) tensors as the sum-pool trunk, so:

- **Zero Rust / featurizer changes.** Rust `inference/mod.rs` and `serve_onnx`
  dispatch key on `state_dim` + the input-name set, NOT on `model_variant`
  (which is sidecar-only metadata). A relational graph is detected as
  `GraphSchema::V3_2` (or the belief variant), fed the same 7/8 inputs, and
  read as `[logits, value]`. Verified: the smoke's exported graph input-name
  set is byte-equal to the v3.2 contract (7-input) and v3.2+`belief_features`
  (8-input).
- **Re-extracts from existing self-play traces** (raw `PublicObservation` JSONL)
  with no resimulation — v6 reads only existing public-observation-derived
  v3.2 features.
- **Export sidecar** already projects `model_variant != "mlp"`; the relational
  knobs ride `model_config` in the checkpoint via `ModelConfig.to_dict`.

Trainable today through the existing orchestrators (no new framework):
`train_bc.py --model-variant relational --uma-slot-tokens --state-dim 110`,
threaded by `r12_orchestrator.py` and `rebel_orchestrator.py`
(`--model-variant relational`).

## Pre-registered gate

- **Train under the corrected ReBeL R20 loop** (leaf=1.0, grounded value
  target), `--data-mode rebel --belief-features` (so the belief token is live),
  uniform deck sampling for self-play.
- **Promotion = R19/R20 head-to-head vs the champion** (challenger beats
  champion at 95% Wilson-lower + margin), the gate already wired in
  `rebel_orchestrator.py`. Report the fixed tight-gate number too, but do NOT
  treat the historical `~0.59` as a pass/fail line — it is a stale-baseline
  reference, not a ceiling.
- **Apples-to-apples control:** also run the same recipe with
  `--model-variant mlp` from the same seed/data so the verdict isolates the
  trunk axis from the corrected-loop lift.
- **Re-baseline obligation:** because the ceiling was measured on a buggy loop,
  the first deliverable is the mlp-vs-relational delta under R20, not an
  absolute number.

## Non-goals / guardrails

- No feature-schema bump in v6 (state stays v3.x / action v5). v6 is the trunk
  axis; do not fold a new featurization in without a separate gate.
- No Q-head / value-adapter under relational (rejected in `__init__`) — those
  are MCTS-era sum-pool add-ons outside v6 scope.
- Throughput is an open risk: 251 tokens × `relational_layers` per leaf is
  heavier than sum-pool. Use the batched predict path; a throughput re-baseline
  is a follow-up (cf. `project_model_size_transition_h256_d4`), not a blocker
  for the strength gate.

## Representation & feature-improvement brainstorm (2026-05-29)

**Implementation status (2026-05-29): all Tier-1 + Tier-2 items below are
IMPLEMENTED, validated, and committed** (untrained — these are ablation knobs
for the v6 gate, not yet a strength result). Commits `8dcb556` (catalog
embeddings #1, contextual policy head #4, belief per-card range #2),
`13c1ed1` (BCE value loss #3, side-swap #6, belief complement resampling #5),
`a6b5460` (v6 deck-composition featurizer #7a). Validation: `v6_relational_smoke`
13/13, `v6_python_rust_parity_smoke` ALL GATES PASS, full `cargo test -p engine`
green, capstone full-stack ONNX export ~1e-7. Caveats: (#7a) deck-composition is
wired + parity-tested but emits ZEROS until `PublicObservation` carries the own
deck-card-id list (single hook `*_v6_own_deck_card_ids`); (#7b) un-gate
ability/attack-roster was folded into #1 (catalog mechanics ride every token).
All knobs are config/flag-gated and default to off/ablatable except the v6
contextual policy head (default on). Next: gate them under corrected ReBeL R20.

Four-lens subagent brainstorm (info-completeness / tokenization geometry /
ReBeL belief / training objectives). Two cross-cutting findings dominate:

- **Convergence: the card-embedding table is the #1 lever.** Three lenses
  independently landed on it — `card_embed` is `nn.Embedding(108, 32,
  padding_idx=0)`, randomly initialized, supervised only by RL signal, and the
  relational trunk leans on it for EVERY token (card/slot/action). 107 cards ×
  32-d with a skewed self-play distribution → rare cards under-learned. The
  static mechanics (attack damage/cost, HP, weakness, effect-kind) live in
  `cards.json` and are vocab-id-keyed, so they can be baked in.
- **Correctness smell in the belief (echoes the stale-verdict caveat):** the
  per-card hand range is ALREADY computed in `build_belief_features`
  (`engine-rs/.../belief/mod.rs:329`) and then DISCARDED — only 16 scalar
  moments reach the model. Worse, particles are a permutation of the
  opponent's TRUE hand+deck multiset (`belief/mod.rs:216-235`), so the
  "belief" is a near-deterministic point mass, not a range, and the true hand
  is always in support. ReBeL is the paradigm; this is a real ceiling.

### Tier 1 — cheap pre-gate ablations (model/loss only, no serving/resim change)

Test as BC / short-ReBeL ablations on the v6 path BEFORE the expensive R20
strength gate; all ride the existing ONNX signature.

1. **Catalog-grounded card embeddings.** Init `card_embed` rows from `cards.json`
   mechanics and/or add a mechanics-prediction aux head (pretrain →
   `--init-from-checkpoint`, or a joint anchor loss). Optionally factor the
   table (type ⊕ stage ⊕ role ⊕ residual). De-risks the v6 gate itself — a
   random 32-d table is a weak prior for a trunk whose thesis is *relating*
   cards. Cost: model.py + train_bc, ONNX-unchanged.
2. **Belief per-card hand range.** Emit the already-computed presence vector
   onto `belief_features`; bump `BELIEF_FEATURE_DIM` in `belief/mod.rs:23` +
   `model.py` lockstep. Rides the 8-input belief signature; the v6 belief token
   becomes a real range. Highest ReBeL lift per cost. (Stronger variant:
   per-card belief *tokens* via the shared `card_embed`.)
3. **Value-as-win-probability (BCE) calibration.** Reparam the value loss as
   BCE on `(value+1)/2`; keeps the exported scalar in [-1,1] (ONNX-unchanged).
   Targets the documented value-head noise (`progress/r16.md:141-272`).
4. **Fix the v6 policy-head leak.** The cross-attention head re-reads the RAW
   `card_embed(action_card_idx)` for source/target (`model.py` policy head)
   instead of the CONTEXTUALIZED encoder token for that card — discarding the
   board-aware representation the trunk just computed. Gather the encoded token
   (needs an action→token-position index). Model-side polish on shipped code.

### Tier 2 — deeper bets (higher ceiling / cost; after Tier 1 + corrected-loop baseline)

5. **Belief real-range resampling.** Resample hidden zones from the unseen
   deck-list complement (deck list is public) + public-history filtering, not
   the true multiset. The structural ReBeL correctness fix; needs revealed-card
   bookkeeping. Canonical home: `rebel-e2e-scoping.md`.
6. **Side-swap symmetry.** Own↔opp augmentation + a value-negation consistency
   loss (zero-sum ⇒ `value(s) = -value(swap(s))`). Attacks the +~0.06
   opponent-side gap the v6 polarity embed only addresses architecturally.
   Training-loop only; cross-link `per-side-asymmetry-probe-scoping.md`.
7. **Selected featurizer re-extracts that fill genuine blind spots** the trunk
   cannot derive: own-side deck-composition inference (the heuristic AI uses
   `deck_inference.rs`; the model sees only scalar `deckCount`), and un-gating
   the backward-looking `usedAbilityThisTurn` feature into a forward
   opp-threat / attack-roster summary. Featurizer-only re-extract (no resim).
   Hold the broader scalar-tail set unless Tier-1 embedding grounding
   underperforms (the trunk should derive much of it).

### Tier 3 — defer / spike-first

- **Action/turn-history tokens (R7.b.4).** Never tested; needs a trace
  cross-row reconstruction spike first, then a new ONNX input.
- **Threat-matrix attention bias.** The data-dependent `[S,S]` version is a
  heavy new input; the static role-pair bias variant is model-only and cheap —
  do that first.
- **Evolution-chain field / opponent-next-action / terminal-reason heads.**
  Need new obs fields or trace relabels (resim or backfill).
- **Token pruning / present-only packing.** Throughput lever (60 of 251 tokens
  are discards); featurizer-side fixed-shape change. Gate on a measured
  per-leaf wall, not assumed.

### Recommended sequence

Tier 1 in order 1 → 2 → 3 → 4 (each a clean A/B that also sharpens the v6 gate
itself), then re-evaluate Tier 2 against the corrected-R20 mlp-vs-relational
baseline. Belief items (2, 5) are the highest-ceiling line given ReBeL is the
paradigm; embedding grounding (1) is the highest-conviction-per-cost.

## Implementation pointers

- Model: `training/uma_ai/model.py` (`RelationalTrunk`, `CandidatePolicyNet`
  relational dispatch, `ModelConfig.relational_*`).
- Smoke: `training/v6_relational_smoke.py` (12 cases: forward/mask/value-range,
  no-dead-params, ONNX roundtrip ×2, signature parity ×2, mlp/set_attention
  regression, invariants). Run:
  `training/.venv/bin/python training/v6_relational_smoke.py`.
- CLI: `--model-variant relational` in `train_bc.py`, `r12_orchestrator.py`,
  `rebel_orchestrator.py` (all require `--uma-slot-tokens`).
- Cold start: the relational trunk has no init-expander and the ReBeL loop has
  no heuristic-only self-play path, so iteration 0 needs a bootstrap ONNX. Mint
  a config-correct random checkpoint with `training/mint_relational_init.py`
  then `export_onnx.py` it. E2E loop recipe (CPU smoke): `rebel_orchestrator.py
  --model-variant relational --uma-slot-tokens --state-dim 110 --hidden-dim 128
  --use-release-binary --device cpu --selfplay-onnx-path <init.onnx>
  --init-from-checkpoint <init.pt> --iterations 2 --smoke` (small games/sims).
