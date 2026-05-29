# v6 Relational-Trunk Model Scheme

- **Date:** 2026-05-29
- **Status:** IMPLEMENTED + SMOKE-GREEN, UNTRAINED. The model, training-CLI
  wiring, and an ONNX/signature/trainability smoke landed this commit. No
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

## Open forks (post-gate)

- **v6.1 richer signature** — if v6 clears structurally but the arithmetic
  crutch proves load-bearing, re-add the v3.7 lethal/ETA channels as explicit
  *derived-fact tokens* (not a scalar tail). Pre-register before running.
- **Token pruning** — drop the 60 discard tokens (or pack present-only) if
  throughput is the binding constraint; ONNX needs a fixed shape so this is a
  featurizer-side change.
- **Per-side asymmetry** — if the polarity-shared trunk measurably narrows the
  opponent-side gap, feed that into `per-side-asymmetry-probe-scoping.md`.

## Implementation pointers

- Model: `training/uma_ai/model.py` (`RelationalTrunk`, `CandidatePolicyNet`
  relational dispatch, `ModelConfig.relational_*`).
- Smoke: `training/v6_relational_smoke.py` (12 cases: forward/mask/value-range,
  no-dead-params, ONNX roundtrip ×2, signature parity ×2, mlp/set_attention
  regression, invariants). Run:
  `training/.venv/bin/python training/v6_relational_smoke.py`.
- CLI: `--model-variant relational` in `train_bc.py`, `r12_orchestrator.py`,
  `rebel_orchestrator.py` (all require `--uma-slot-tokens`).
