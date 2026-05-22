# Set-Attention Architecture Probe (R7.b.3-Style)

- **Date:** 2026-05-22
- **Status:** USER-GATED scoping. Sits behind the v3.2-mandatory + deck-variety
  gates AND behind the current recipe-axis forward lines
  (`w6-loop-anti-degradation`, `per-game-pfsp-league-retry`,
  `value-head-data-program` Stage 2/3). P3 in
  `docs/ai-agent-state/queue.json`. Not auto-launchable; explicit user gate
  required before any code lands beyond P0a.
- **Routing:** new backlog entry under "Active Search-Wrapped Frontier" (item
  3b, the architecture axis sibling to item 0d's recipe axis); this scoping
  is the canonical home until the line resolves.
- **Re-opens** the R7.b axis. `docs/ai-research/scoping/archive/r7b-feature-representation.md`
  §4 #3 pre-registered "set-encoder / attention over per-card tokens (1–2 layer
  self-attention, 4 heads, hidden=64)" as a conditional follow-up; R7.b.2's
  failure (sum-pool axis) was framed at `r15-archive.md:386-393` as leaving
  R7.b.3 (attention pool) and R7.b.4 (history embedding) **not falsified**.
  This probe executes R7.b.3.
- **Guardrail relaxation:** `docs/ai-research-backlog.md` Guardrails says "do
  not re-open representation/capacity tuning off the R110 MARGINAL band; the
  forward line is the loop-recipe axis only." This probe relaxes that
  carve-out: attention is an *inductive-bias axis*, not a capacity-tuning
  axis (R6 closes capacity-from-above; see Prior datapoints). The guardrail
  was written when both representation (v3.0→v3.1→v3.2) and recipe (W6/PFSP)
  were active within sum-pool; with the representation axis converging on
  the 0.5811 ceiling and slot-tokens not unblocking it cleanly, the next
  untested axis within representation is trunk shape.

## Hypothesis

The current trunk has a *weak relational prior* for a card game. Concretely
in `training/uma_ai/model.py`:

- Per-zone state encoding sum-pools 32-d card embeddings within each of 8
  zones, then linearly projects the 256-d zone-flat vector to `hidden`
  (`model.py:120-126,230-256`). Cards within a zone are summed (commutative
  bag); cards across zones do not interact except through the joint MLP.
- v3.2 adds a per-slot encoder (`uma_slot_encoder`, 10 slots ×
  `Linear-GELU-Linear`; `model.py:170-303`). Cross-slot interactions exist
  only via sum-pool into `state_encoded`.
- The action branch concatenates `[source_embed, target_embed]` onto
  `state_encoded`; there is no mechanism for "which board card threatens
  which target."

Two symptoms point at this being load-bearing:

1. **Raw-policy SL plateaus at Wilson ≤ 0.33 across all four axes tested.**
   R7 (labels, wl 0.2921), R8 (DPO objective, wl 0.3318), R7.b.2
   (sum-pool representation, wl 0.3045), mcts-distill-v1 (label-shape,
   wl 0.1470) — see `docs/ai-research/progress/r15-archive.md:278-287,
   361-366`. **Attention has never been tested.** The R7.b scoping doc
   explicitly reserved R7.b.3 attention-pool as the next-untested
   sub-step.
2. **iter-2-peak-then-rot is schema-independent.** Tight-gate re-verdicts
   #1 (R110-W6 v3.0 wl=0.5811) and #2 (R16-P1 v3.1 wl=0.5810) match to four
   decimals at n=10,000 (`docs/ai-research/progress/r110.md` §4d, §4e). v3.2
   slot-tokens add a small positive but bit-exact-init delta=0.0 and
   recipe-induced regression at iter-1 (`docs/ai-research/progress/r16.md`).
   Feature-set churn within sum-pool is exhausted; trunk shape is the one
   untouched dimension *within* representation.

Together: the model may be unable to amortize MCTS search into a faster
policy because the trunk lacks the inductive bias to reason about
card-to-card relationships. AlphaZero-style work on combinatorial games
typically uses architectures with explicit relational structure.

**What this probe tests:** does a small set-attention trunk over per-card
tokens — replacing the additive zone-sum and additive slot-sum branches —
(a) lift raw-policy SL above the F1 0.40 gate, AND (b) lift search-wrapped
strength above the v3.0 0.5811 ceiling (or re-verdict-#3 v3.2 ceiling if
landed)?

## Why now

- The v3.0/v3.1/v3.2 representation-axis sequence has been exhausted *within
  sum-pool architectures*. All three share the same trunk shape; only the
  feature schema differed.
- Throughput is no longer binding (Slice 3c gives ~11-min tight-gate walls
  at n=10,000; Rust featurizer Slice 3 handles v3.2 in-process).
- R7.b.3 is *pre-registered* — executing it is the lowest-novelty way to
  test an architectural axis vs proposing a wholly new design not in any
  prior scope.
- The deck-variety + v3.2 user directives constrain *training data and
  feature shape*; they do not constrain the trunk shape. An attention-trunk
  probe can satisfy both directives by training on uniform-deck v3.2 data.

## Prior negative-ish datapoints (to navigate around)

- **R6 capacity** (`docs/ai-performance-research-progress.md:1640-1651`).
  `hidden_dim=128 depth=3` reached 97% train acc / 83% argmax-match vs
  teacher / best Brier but WORST gate WR (33.0%, vs 37.5% at hidden=64/d=2).
  Capacity-from-above is closed — bigger MLP overfits teacher noise.
  **Design constraint:** the attention probe must change the inductive bias
  (relational reasoning), not the parameter count. Target trunk delta
  ≤ +100K params vs current production trunk; do NOT also widen hidden.
- **R7.b.2 sum-pool** (`docs/ai-research/progress/r15-archive.md:278-287`).
  K=32 additive per-zone sum-pool failed Wilson ≥ 0.40 gate at 0.3045
  (-9.55pp). Attention replaces sum-pool but reuses the SAME `card_embed`
  table — the embedding lookup is not the lever; the *aggregation* is.
- **mcts-distill v1** (`docs/ai-research/scoping/archive/mcts-distill.md`,
  `docs/ai-research/progress/r15.md:84-95`). Failure attributed to
  state-coverage mismatch (training corpus = MCTS-vs-MCTS w/ Dirichlet;
  eval = raw-policy vs rule-bot). **Design constraint:** Slice 2 SL training
  must use the same corpus type as R7.b.2 (R4-era rollout corpus from
  `runs/item17-2026-05-11/iter-002/`) — apples-to-apples vs the sum-pool
  baseline, not the failed mcts-distill corpus.
- **R110 tight-gate ceiling 0.5811 is a v3.0 reference.** The v3.2 ceiling
  is pending re-verdict #3 (C8 iter-0). Slice 3 must compare against
  whichever is current at the time of acceptance.

## Plumbing

### Architecture sketch

Replace the current `state_encoder + zone_projection + uma_slot_encoder`
pre-trunk with a set-attention encoder over a token sequence. Token sequence
per state:

- 1 CLS token (learned).
- Up to `Σ zones max_cards_per_zone` card tokens. Each =
  `card_embed(card_id) + zone_pos_embed(zone_idx)`. Pad card_ids mask out.
- Up to `UMA_SLOT_COUNT=10` slot tokens. Each =
  `card_embed(slot_card_id) + slot_feature_proj(uma_slot_features) +
  slot_pos_embed(slot_idx)`. Absent slots mask out.

Encoder: 1–2 layers of `MHA + LayerNorm + FFN + LayerNorm` (pre-LN), with
`d_model=hidden_dim=64, n_heads=4, ffn_dim=128, dropout=0.05`. CLS output
replaces `state_encoded`. The action branch is unchanged for the first probe
(keep `[source_embed, target_embed]` concat into the joint trunk); a
cross-attention action branch is reserved for a follow-up if Slice 3 passes.

Param budget for the new attention block: `~ 4 · d_model² · n_layers +
2 · d_model · ffn_dim · n_layers` ≈ 33K (n_layers=1) to 66K (n_layers=2),
plus position embeddings (~3K). Net trunk delta ≤ +80K params vs current
production (which is ~hidden=64/depth=2 with embed+zone proj). Stays well
below the R6 128/3 trunk total.

**Init contract.** Zero-init the encoder's output projection (the path from
attention out → `state_encoded`). At init, the attention contribution is
structurally null; `card_embed` weights copy bit-exact from a v3.2 ckpt; the
joint trunk + heads copy bit-exact from the v3.2 ckpt. A fresh
`set_attention` model thus produces identical logits+value to its v3.2
warm-start until training perturbs the zero-init projection. This is the
direct analog of the v3.1 `delta=0.0` parity trick and C7's
`make_v32_slot_token_init.py` (`model.py:170-179`).

### Predecessor (cheap, lands first)

**P0a. ONNX MHA opset-17 roundtrip smoke.** R7.b.2 only validated `Gather`
for embedding lookup. MHA needs `MatMul + Softmax + Gather` at opset-17
(well-supported) but pad-masking via `-inf` round-trips need verification.
Build a 1-layer MHA module with random weights, ONNX-export at opset-17,
roundtrip through `training/serve_onnx.py`-equivalent inference, check
`max_abs_diff < 1e-5` vs PyTorch eager on `N=100` random
`(state, legal_actions)` inputs. ~half day.

**KILL CRITERION:** if export fails or `max_abs_diff > 1e-4` after standard
fixups (e.g. attn-mask dtype/device), the attention probe is infra-blocked.
Pause and triage before any further investment.

**P0b. Rust featurizer shape check.** Confirm by code-read only (no code
change): `engine-rs/crates/engine/src/policy/featurize.rs` Slice 3 emits
`state_features` + `card_ids_by_zone` + `uma_slot_card_ids` +
`uma_slot_features` for v3.2. The attention probe consumes the SAME tensors
— only the model's trunk consumes them differently. Confirm dispatch
(`inference/mod.rs`) detects v3.2 input set unchanged.

### Slice 1 — model + serve_onnx + Rust dispatch (~1-2 days)

- Add `model_variant: str = "mlp"` to `ModelConfig`; new value
  `"set_attention"` builds the attention trunk; default unchanged (v3.0 /
  v3.1 / v3.2 sum-pool callers are byte-identical).
- In `CandidatePolicyNet.__init__`, branch on `model_variant`; for
  `set_attention`, construct the encoder as sketched above with zero-init
  output projection.
- ONNX export: bump schema dispatch key (e.g. add `model_variant` to the
  ONNX metadata + serve_onnx dispatch). Reuse existing v3.2 input tensors;
  no new featurizer.
- Rust inference dispatch: `inference/mod.rs` v3.2 graph load is unchanged;
  the new ONNX graph still has 7 inputs. No Rust code change required for
  Slices 1–2.

### Slice 2 — SL smoke vs R7.b.2 gate (~1 day wall)

Train `train_bc.py --data-mode rollout` from `runs/item17-2026-05-11/iter-002/`
(the same R4-era rollout corpus R7.b.2 used). Same SL recipe (50 epochs,
hidden=64, depth=2, dropout=0.05). Eval gate at `n=1,000` side-balanced
rule-bot seed-start 9000 via `training/r8_gate_eval.py`.

**Acceptance:** `wilson_lower ≥ 0.40` (the F1 / R7.b.3 pre-registered gate)
AND `wilson_lower ≥ 0.30` strict-non-regression vs R7.b.2 0.3045.

**Falsification:** `wilson_lower < 0.30` ⇒ attention is no better than
sum-pool at the SL gate; close the line and document.

**Marginal:** `0.30 ≤ wl < 0.40` ⇒ attention helps but does not clear F1;
do NOT proceed to Slice 3 without explicit user gate.

### Slice 3 — search-wrapped acceptance under v3.2 + deck-variety (~1 day wall)

GATED on Slice 2 ≥ 0.40 AND on `deck-pair-sampling` Slice 2 landing AND on
re-verdict #3 reporting (so the v3.2 ceiling is known).

Train one C8-style 5-iter W6 selfplay loop with the attention trunk
warm-started from the Slice 2 ckpt:

- v3.2 architecture (mandatory per directive B).
- `--deck-sampling=uniform` in selfplay (mandatory per directive A).
- R110-W6 recipe-faithful (rollout-leaf, `--w6-replay-window 3`,
  `--kl-anchor-weight 0.05`, fixed iter-0/SL KL anchor).
- Tight-gate eval at `n=10,000` on the best-promoted iter via the Rust
  parallel path (~11 min wall) on BOTH gates: fixed-matchup and
  diverse-matchup (uniform).

**Acceptance:**
- Fixed-matchup `wilson_lower` beats the current ceiling (v3.0 0.5811 if
  re-verdict #3 has not improved on it, OR the re-verdict-#3 v3.2 ceiling
  if it has).
- Diverse-matchup `wilson_lower` within ±5pp of fixed-matchup AND not below
  the v3.0 ceiling.

**Falsification:**
- `wilson_lower ≥ 5pp below` the current ceiling on either gate ⇒
  attention trunk under-performs; close the line and treat the
  architectural-axis question as closed for the current corpus quality.

## Out of scope

- Wider MLPs / deeper MLPs. R6 closes capacity-from-above; this probe is
  strictly the inductive-bias lever.
- New features. The attention probe consumes existing v3.2 tensors only.
- New ONNX runtime or new Rust featurizer slice. Reuse the v3.2 dispatch
  end-to-end.
- Cross-attention to action tokens. Reserved as a follow-up if Slice 3
  passes.
- Graph-net / GNN variants. Single-axis test; transformer is the canonical
  small-set encoder baseline. If transformer fails, GNN does not get a free
  pass — close the architectural axis.
- New decks, new label sources, new recipes. Everything else is held
  constant vs R110-W6 baseline.

## Falsification (top-level)

- Slice 2 FAIL (`wl < 0.30`) ⇒ the raw-policy SL line stays closed; the
  architectural axis is not the lever either. Near-zero-residual-claim
  outcome; the existing recipe-axis program continues unchanged.
- Slice 2 PASS, Slice 3 FAIL ⇒ attention helps SL but does not amortize
  search at this scale; the production rollout-leaf path remains the
  strength claim; attention is at most a fast-policy fallback.
- Slice 3 PASS ⇒ model architecture *was* a load-bearing constraint; new
  strength-claim ckpt under attention + v3.2 + uniform decks; new
  methodology baseline.

## Relation to active queue items

- **Parallel/contingent sibling:** `rich-data-bank-at-throughput-variety-regime`
  (P2 SCOPED 2026-05-22, queue
  `rich-data-bank-at-throughput-variety-regime`). Tests the
  *corpus-distribution* axis vs this probe's *trunk-shape* axis. If
  this probe's Slice 2 falsifies (`wl < 0.30` on R4-era rollout corpus),
  the corpus-distribution line promotes P2 -> P1 as the parallel-axis
  falsification candidate. If this probe's Slice 2 passes, the
  corpus-distribution line stays P2 — architecture-axis is the active
  forward line. Scope:
  `docs/ai-research/scoping/rich-data-bank-at-throughput-variety-regime.md`.
- **GATED behind:**
  - `deck-pair-sampling` (Slice 2 must land so selfplay can run with
    uniform sampling).
  - `tight-gate-reverdict-program` re-verdict #3 (establishes v3.2 ceiling
    so Slice 3 has an apples-to-apples promotion target).
  - `w6-loop-anti-degradation` HP sweep (the *cheap, in-methodology,
    R110-A/B-preserving* attempt). If the HP sweep clears the ceiling, the
    methodology-break case for attention weakens.
  - `per-game-pfsp-league-retry` (opponent-diversity recipe lever). Same
    logic — if PFSP clears the ceiling, attention drops further in
    priority.
- **Architecturally orthogonal to:** the recipe axes (W6, PFSP, vhleaf).
  Attention is the *trunk shape* axis; recipe is the *loss/optimization*
  axis. Verdicts on either don't preempt the other.
- **Touches:** `training/uma_ai/model.py`, `training/serve_onnx.py`, and
  optionally `engine-rs/crates/engine/src/inference/mod.rs` (dispatch
  metadata only — no new featurizer). No `r12_orchestrator.py` change.

## Open questions

- **Is ONNX MHA at opset-17 pad-mask-safe through ort+CUDA?** Answered by
  P0a. If not, attention probe is infra-blocked until ORT/opset bump.
- **Does the SL gate at 0.40 probe the right capability?** R7.b.2 closed
  the sum-pool axis at this gate; if SL turns out to be a state-coverage
  problem (mcts-distill-v1 framing), Slice 2 might mislead Slice 3.
  Mitigation: train on the same R4-era corpus as R7.b.2 (apples-to-apples
  vs the sum-pool baseline), not on mcts-distill.
- **Param-count parity sanity check.** Attention adds ~50–80K params vs
  current production trunk. Well below R6's 1.27M total — but the prior is
  weak ("R6 was wider AND deeper AND on bad labels"). Mitigation: report
  param count + train/val curve in Slice 2 and abort if val_acc regresses
  > 3pp vs R7.b.2 sum-pool baseline (echoing the R7.b.2 −3.4pp val_acc
  finding) before spending Slice 3 compute.
- **Does zero-init output projection actually preserve init parity on
  CUDA?** Verify in P0a's roundtrip — `make_v32_slot_token_init.py`-style
  smoke comparing logits/value vs the warm-start ckpt at zero-init.
