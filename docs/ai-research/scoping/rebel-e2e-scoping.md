# ReBeL E2E Scoping

- **Date:** 2026-05-28
- **Status:** SCOPED-AWAITING-IMPLEMENTATION
- **Parent evidence:** `docs/ai-research/analysis/alphazero-style-training-postmortem.md`
- **Intent:** Start an end-to-end ReBeL-inspired line for this game: public-belief state construction, belief-aware search targets, self-play data capture, and a train/eval loop.

## Summary

ReBeL is the best long-term research direction for this game's AI shape, but it should be implemented as a staged end-to-end system rather than as a one-shot full reproduction of the poker paper.

The reason is structural. The failed AlphaZero-style line searched exact simulator states and asked a value head to summarize stochastic hidden-card futures. That recipe was mismatched to this game:

- hidden hands/decks mean exact-state search can accidentally condition on information a player does not know;
- shuffles, draws, and coin flips make value-head leaves compress a high-variance future into one scalar;
- the observed legal-action branching factor is small, so higher MCTS sims mostly sharpen already-small visit targets;
- no-anchor self-play produced large per-iteration weight drift without a positive strength curve.

ReBeL's core move is the right theoretical correction: search over a **public belief state** instead of a fully known game state. For this repo, the first useful version should be "ReBeL-lite": particle public-belief state + rollout/value search + belief-conditioned training rows. True CFR subgame search can come after that substrate proves useful.

## Relevant Literature

### ReBeL

Brown et al., **Combining Deep Reinforcement Learning and Search for Imperfect-Information Games**, NeurIPS 2020. [`arXiv:2007.13544`](https://arxiv.org/abs/2007.13544).

ReBeL introduces Recursive Belief-based Learning. It combines self-play, search, and neural policy/value learning for two-player zero-sum imperfect-information games. Its central abstraction is the **public belief state**: public history plus beliefs over each player's private information. Search runs from public belief states and the neural net evaluates public-belief leaves.

Transferable idea: AlphaZero's "search improves policy, policy/value net learns from search" loop remains useful, but the search object must be a belief over hidden states rather than a sampled fully observed state.

### Student of Games

Schmid et al., **Student of Games: A unified learning algorithm for both perfect and imperfect information games**, 2023. [`arXiv:2112.03178`](https://arxiv.org/abs/2112.03178), [`Science Advances`](https://www.science.org/doi/10.1126/sciadv.adg3256).

This is the stronger long-term north star. It generalizes AlphaZero-style search/learning and imperfect-information CFR-style search in one framework. It reinforces the same inference as ReBeL: hidden-information games need public-belief / information-set reasoning, not plain perfect-information MCTS.

### Deep CFR / Single Deep CFR / NFSP

- Brown et al., **Deep Counterfactual Regret Minimization**, ICML 2019. [`arXiv:1811.00164`](https://arxiv.org/abs/1811.00164), [`PMLR`](https://proceedings.mlr.press/v97/brown19b.html).
- Steinberger, **Single Deep Counterfactual Regret Minimization**, 2019. [`arXiv:1901.07621`](https://arxiv.org/abs/1901.07621).
- Heinrich and Silver, **Deep Reinforcement Learning from Self-Play in Imperfect-Information Games**, 2016. [`arXiv:1603.01121`](https://arxiv.org/abs/1603.01121).

These are the equilibrium-learning branch. They are less directly AlphaZero-like than ReBeL but relevant if exploitability becomes the target metric. NFSP is simpler; Deep CFR and Single Deep CFR are closer to scalable imperfect-information game solving.

### ISMCTS / Determinization

- Cowling, Powley, Whitehouse, **Information Set Monte Carlo Tree Search**, 2012. [`DOI:10.1109/TCIAIG.2012.2200894`](https://doi.org/10.1109/TCIAIG.2012.2200894).
- Cowling, Ward, Powley, **Ensemble Determinization in Monte Carlo Tree Search for the Imperfect Information Card Game Magic: The Gathering**, 2012. [`PDF`](https://eprints.whiterose.ac.uk/id/eprint/75050/1/EnsDetMagic.pdf).
- Frank and Basin, **A theoretical and empirical investigation of search in imperfect information games**, 2001. [`DOI:10.1016/S0304-3975(00)00083-9`](https://doi.org/10.1016/S0304-3975%2800%2900083-9).

These are the practical bridge. They do not give the same guarantees as ReBeL/CFR, but they are implementable against the current engine. They also define the key failure modes to guard against: strategy fusion and non-locality.

### Card-game engineering references

- Zha et al., **DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning**, ICML 2021. [`arXiv:2106.06135`](https://arxiv.org/abs/2106.06135), [`PMLR`](https://proceedings.mlr.press/v139/zha21a.html).
- Xi et al., **Mastering Strategy Card Game (Legends of Code and Magic) via End-to-End Policy and Optimistic Smooth Fictitious Play**, 2023. [`arXiv:2303.04096`](https://arxiv.org/abs/2303.04096).
- Xiao et al., **Mastering Strategy Card Game (Hearthstone) with Improved Techniques**, 2023. [`arXiv:2303.05197`](https://arxiv.org/abs/2303.05197).
- Zha et al., **RLCard: A Toolkit for Reinforcement Learning in Card Games**, 2019. [`arXiv:1910.04376`](https://arxiv.org/abs/1910.04376).
- Lanctot et al., **OpenSpiel: A Framework for Reinforcement Learning in Games**, 2019. [`arXiv:1908.09453`](https://arxiv.org/abs/1908.09453).

These point away from pure AlphaZero for CCGs and toward self-play populations, legal-action masks, careful action representations, large-scale parallel actors, and robust evals against exploiters or checkpoint pools.

## Inference For This Game

### ReBeL is the right long-term shape

The game is a two-player adversarial card duel with public board state, private hand/deck state, and chance events. That is closer to poker/CCG imperfect-information games than to chess/Go. ReBeL addresses the exact mismatch by making the state of search:

```text
public board/history + belief over private worlds
```

instead of:

```text
one exact sampled full state
```

This matters because a legal policy must choose the same action for all hidden worlds that look identical to the acting player. A search that can pick different actions for different sampled opponent hands is cheating, even if it never explicitly exposes those cards to the model.

### Full ReBeL is too large for the first patch

A faithful ReBeL implementation needs:

- public-belief state representation;
- belief updates over private hands/decks after public actions and reveals;
- subgame search over public belief states;
- counterfactual values or private-state-indexed values;
- training targets from belief-aware search;
- exploitability or best-response-style evaluation.

That is a large system. Implementing all of it at once would make failures hard to diagnose. The right approach is to land an end-to-end ReBeL-shaped minimum viable loop first, then replace approximations with more principled pieces.

### Rollouts remain important

The AlphaZero postmortem found that rollout leaf outperformed pure value-head leaf. ReBeL does not require deleting rollouts immediately. In this game, rollouts are an integration tool for stochastic futures. The first ReBeL-lite version should keep rollout-backed evaluation and use the belief machinery to fix hidden-information conditioning before trying pure value leaves again.

## Working Definitions

### Public observation

The repo already builds `PublicObservation` in `engine-rs/crates/engine/src/policy/observation.rs`. It exposes own private hand, public board zones, public discard zones, counts for hidden opponent hand/deck, and packed card ids by zone. This is the current policy-net input.

### Public belief state

For this line, a public belief state is:

```text
PublicObservation
+ public action/history metadata needed to constrain hidden zones
+ particles representing possible full private worlds
+ per-particle weight
```

The initial version can use uniformly weighted particles.

### Particle

A particle is a full `GameState` consistent with the acting player's public observation and known private information. For the acting side, own hand is known. For the opponent side, private hand/deck order is sampled from the remaining card multiset subject to public history and visible zones.

### Belief search target

A belief search target is an action distribution and optional action-value vector produced by aggregating search over particles rooted at the same public belief state. It must be indexed only by actions legal from the acting player's observation.

## E2E Implementation Plan

### Slice 0: Evidence and contract locks

Goal: make the ReBeL line hard to confuse with the closed pure-AZ line.

Deliverables:

- Add a manifest mode name, e.g. `rebel-lite-v0`.
- Every generated row records:
  - `kind: "belief-selfplay"`;
  - `beliefSchemaVersion`;
  - `particleCount`;
  - `beliefSampler`;
  - `searchAggregator`;
  - deck pair ids;
  - state/action schema metadata.
- Add a small smoke fixture proving rows from two different hidden particles share one public observation/action target frame.

Exit gate:

- A 1-game smoke writes parseable belief-selfplay rows with legal actions, target distribution, value target, and belief metadata.

### Slice 1: Public-belief particle sampler

Goal: sample hidden worlds consistent with the current public observation.

MVP algorithm:

1. Start from the real `GameState` during self-play.
2. For the acting side, preserve known private hand and all public zones.
3. For the non-acting side, hide private hand/deck order from the search policy, then resample from the side's remaining decklist multiset minus public zones.
4. Generate `N` full `GameState` particles with deterministic seed labels.
5. Reject particles that violate counts, public active/bench/discard/stadium, points, turn, pending choice, or energy zones.

Initial simplification:

- It is acceptable for the sampler to use the true remaining hidden-card multiset while randomizing assignment/order, because self-play infrastructure has access to the full simulator state. The row must not expose the sampled private cards to the model input. Later slices can replace this with stricter public-history reconstruction.

Key risk:

- If search uses full sampled hidden state too freely, strategy fusion can remain. Slice 2 mitigates by aggregating one action target per public observation rather than training per-particle policies.

Exit gate:

- For fixed seeds, particle generation is deterministic.
- Particle public observations match the root public observation for the acting side.
- Hidden opponent hand/deck vary across particles when enough unknown cards exist.

### Slice 2: Belief-sampled search aggregation

Goal: produce one legal action distribution per public belief state.

MVP algorithm:

```text
for each decision:
  root_obs = build_public_observation(real_state, side_to_act)
  root_legal = enumerate_legal_actions(real_state, side_to_act)
  particles = sample_particles(real_state, side_to_act, K)

  aggregate_visits = zeros(len(root_legal))
  aggregate_q = zeros(len(root_legal))

  for particle in particles:
    particle_legal = enumerate_legal_actions(particle, side_to_act)
    map particle actions back to root legal actions by stable action identity
    run existing MCTS on the particle
    add visits/Q into aggregate slots

  target = normalize(aggregate_visits)
  selected_action = sample target early / argmax late
  apply selected_action to the real state
```

Use rollout leaf first. Use policy prior only if the policy input is the acting player's public observation, not the hidden particle internals.

Action mapping requirement:

- Do not rely on legal-action index equality across particles. Use stable action identity serialization. If an action references an unknown opponent-private card, it should not be a legal root action for the acting player anyway.

Exit gate:

- Aggregated visit distributions sum to 1.
- Selected action is legal in the real root state.
- Same public root with different particle order produces either identical targets or differences bounded to RNG labels.

### Slice 3: Belief-selfplay row loader

Goal: train the current policy/value model from belief-search rows.

MVP row fields:

```json
{
  "kind": "belief-selfplay",
  "schemaVersion": 1,
  "beliefSchemaVersion": 1,
  "observation": {},
  "legalActions": [],
  "visitDistribution": [],
  "rootMeanQ": [],
  "valueTarget": 1.0,
  "particleCount": 16,
  "beliefSampler": "hidden-zone-resample-v0",
  "searchAggregator": "particle-rollout-mcts-v0"
}
```

Implementation path:

- Mirror `training/uma_ai/selfplay_dataset.py`, but keep a distinct loader class so schema errors identify belief rows explicitly.
- Reuse existing feature builders and legal-action featurization.
- Keep value target as terminal outcome initially.
- Optional: include aggregate root mean-Q as an auxiliary target after the loader is stable.

Exit gate:

- Loader smoke constructs a batch with state features, action features, action mask, policy target, and value target.

### Slice 4: ReBeL-lite orchestrator

Goal: run the full loop end-to-end.

First recipe:

```text
self-play:
  engine: rust
  deck-sampling: uniform
  model-side: both
  belief particles: 8 or 16
  per-particle sims: 25-50
  leaf: rollout
  prior: policy after smoke, uniform for first invariants
  root dirichlet: on

training:
  data-mode: belief-distill
  value target: terminal z
  policy target: aggregate belief-search visits
  KL anchor: weak on, not zero
  replay: checkpoint/windowed replay preserved

eval:
  fixed gate for continuity
  uniform/deck-diverse gate for directive A
  side split reported
```

Compute note:

- `K particles * S sims` should initially match or stay below the current rollout-MCTS budget. Example: `K=8, S=25` is 200 particle-sims per decision, but each particle search has lower per-tree depth/statistical strength. The point of the first run is target quality, not raw eval strength.

Exit gate:

- One full iteration completes: belief self-play, distill, ONNX export, fixed gate, uniform gate.
- Manifest records all belief and schema knobs.

### Slice 5: Replace approximations

Only after slices 0-4 produce stable artifacts:

- Replace true-state hidden multiset access with public-history belief reconstruction.
- Add particle weights from likelihood under observed public actions.
- Add a belief encoder: aggregate hidden-zone probabilities/card-class histograms into model features.
- Add per-private-state value heads or counterfactual-value targets.
- Prototype CFR-style subgame search for high-value decision states.

This is the transition from ReBeL-lite to fuller ReBeL.

## Initial Experiment Matrix

### E0: Sampler and target sanity

- `K=8`, `sims=16`, `prior=uniform`, `leaf=rollout`, no training.
- Compare target entropy and selected action agreement against current rollout MCTS.
- Acceptance: no illegal actions, deterministic with fixed seed, public-observation equality across particles.

### E1: First training-bearing loop

- `K=8`, `sims=25`, `prior=policy`, `leaf=rollout`, weak KL anchor.
- 240-480 games, one iteration.
- Acceptance: completes E2E, no schema failures, non-degenerate policy targets, value/policy losses finite.

### E2: Strength smoke

- Same as E1 but 3-5 iterations.
- Acceptance: does not reproduce the pure-AZ collapse band. It need not beat production yet; it must show stable targets and no rapid drift.

### E3: Particle count axis

- Compare `K=4/8/16` at fixed total sim budget.
- Acceptance: identify whether diversity of hidden worlds or per-particle search depth matters more.

## Evaluation Gates

Report all of:

- fixed-matchup Wilson lower for continuity with old gates;
- uniform deck-sampling gate;
- side split;
- matchup matrix when available;
- target diagnostics:
  - legal-action count histogram;
  - visit top-1 share;
  - target entropy;
  - particle action agreement;
  - aggregate Q variance across particles;
  - selected-action disagreement vs current rollout MCTS;
  - weight L2 drift per iteration.

Promotion bar:

- Do not require first ReBeL-lite run to beat production. Require it to beat the pure-AZ failure mode: stable loop, non-degenerate belief targets, no fast value-head collapse, no no-KL drift.

Strength bar:

- Once stable, it must eventually clear both fixed and uniform/deck-diverse gates before replacing production rollout-leaf search.

## Risks

### Strategy fusion remains

Particle aggregation reduces direct per-hidden-world training leakage, but per-particle MCTS can still plan with sampled private facts. This is why this is ReBeL-lite, not full ReBeL. The follow-up fix is information-set/shared-policy search or CFR subgame search over public belief states.

### Belief sampler encodes impossible histories

Sampling from remaining hidden cards can produce worlds that are count-consistent but history-inconsistent. The MVP accepts this to get the loop running. Later slices should reconstruct beliefs from public action history and revealed cards.

### Compute multiplies quickly

Particle count times sims can exceed current MCTS cost. Keep first runs small, and optimize only after target diagnostics look better than current rollout MCTS.

### Current observation features may be too thin

The current `PublicObservation` mostly contains visible state, own hand, and opponent counts/discards. ReBeL-like learning likely needs belief summaries: possible opponent hand classes, card-presence probabilities, draw odds, and public-history features.

### Exploitability can hide behind win rate

Self-play mirror strength can improve while the policy becomes exploitable. Once the line is stable, add checkpoint-pool and adversarial/best-response-style probes.

## Implementation Entry Points

- Rust public observation: `engine-rs/crates/engine/src/policy/observation.rs`
- Rust MCTS config/result shape: `engine-rs/crates/engine/src/mcts/config.rs`
- Rust MCTS driver: `engine-rs/crates/engine/src/mcts/driver.rs`
- Rust self-play binary: `engine-rs/crates/sim-cli/src/bin/mcts_selfplay.rs`
- Rust eval gate: `engine-rs/crates/sim-cli/src/bin/eval_gate.rs`
- Python MCTS dataset pattern: `training/uma_ai/selfplay_dataset.py`
- Python model/features: `training/uma_ai/model.py`, `training/uma_ai/features.py`
- Training orchestrator: `training/r12_orchestrator.py`
- Existing closed-AZ evidence: `docs/ai-research/analysis/alphazero-style-training-postmortem.md`

## Recommended First Patch

Implement the smallest artifact-producing loop:

1. Add `engine-rs/crates/engine/src/belief/` with:
   - `PublicBeliefConfig`;
   - `BeliefParticle`;
   - deterministic hidden-zone sampler;
   - public-observation consistency checks.
2. Add `sim-belief-selfplay` or a `--belief-particles` mode to `sim-mcts-selfplay`.
3. Emit `kind="belief-selfplay"` rows with aggregate visit targets.
4. Add `BeliefSelfPlayDataset` in Python.
5. Add an orchestrator data mode `belief-distill`.
6. Add smoke tests before any long run:
   - sampler determinism;
   - particle public-observation equality;
   - aggregate target legality;
   - Python loader batch shape.

The first goal is not to prove ReBeL wins. It is to land a correct end-to-end public-belief training loop whose failures are diagnosable.
