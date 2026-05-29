# ReBeL E2E Scoping

- **Date:** 2026-05-28
- **Status:** IMPLEMENTED — loop runs e2e (R17–R19); self-improvement OPEN. Results + verdicts: `docs/ai-research/progress/r17.md`.
- **Parent evidence:** `docs/ai-research/analysis/alphazero-style-training-postmortem.md`
- **Intent:** Start an end-to-end ReBeL-inspired line for this game: public-belief state construction, belief-aware search targets, self-play data capture, and a train/eval loop.

## Summary

ReBeL is the best long-term research direction for this game's AI shape, and the implementation target should be a full end-to-end ReBeL system rather than another AlphaZero variant with belief sampling bolted on.

The reason is structural. The failed AlphaZero-style line searched exact simulator states and asked a value head to summarize stochastic hidden-card futures. That recipe was mismatched to this game:

- hidden hands/decks mean exact-state search can accidentally condition on information a player does not know;
- shuffles, draws, and coin flips make value-head leaves compress a high-variance future into one scalar;
- the observed legal-action branching factor is small, so higher MCTS sims mostly sharpen already-small visit targets;
- no-anchor self-play produced large per-iteration weight drift without a positive strength curve.

ReBeL's core move is the right theoretical correction: search over a **public belief state** instead of a fully known game state. For this repo, the big-bang target is public-history belief reconstruction, public-belief subgame search, neural belief-state policy/value prediction, self-play data generation, training, export, and fixed/diverse evaluation as one coherent system.

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

### Full ReBeL is the implementation target

A faithful ReBeL implementation needs:

- public-belief state representation;
- belief updates over private hands/decks after public actions and reveals;
- subgame search over public belief states;
- counterfactual values or private-state-indexed values;
- training targets from belief-aware search;
- exploitability or best-response-style evaluation.

That is the scope. Intermediate experiments are allowed, but only as validation probes inside the full build. They should not redefine the target into a phased rollout or a ReBeL-lite endpoint.

### Rollouts remain important

The AlphaZero postmortem found that rollout leaf outperformed pure value-head leaf. ReBeL does not require deleting rollouts from the codebase, but full ReBeL should make the primary search backup a public-belief value/search procedure rather than ordinary rollout MCTS. Rollouts remain useful as a baseline, debugging oracle, and auxiliary target source.

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

Particles carry weights. Uniform weights are acceptable only where public history gives no basis to distinguish private worlds; otherwise weights should reflect consistency and likelihood under the observed public sequence.

### Particle

A particle is a full `GameState` consistent with the acting player's public observation and known private information. For the acting side, own hand is known. For the opponent side, private hand/deck order is sampled from the remaining card multiset subject to public history and visible zones.

### Belief search target

A belief search target is an action distribution and optional action-value vector produced by public-belief search rooted at one public information set. Particles support belief evaluation and sampled traversal, but the returned root target must be indexed only by actions legal from the acting player's observation.

## Full E2E Implementation Scope

The target is a complete ReBeL-style system with all major components present in the first full implementation. Validation probes can run along the way, but the implementation should be scoped as one integrated architecture.

### 1. Public history and belief reconstruction

Add a public-history layer that records enough information to reconstruct legal hidden-state beliefs from the perspective of either side:

- initial deck identities and decklists for both players;
- public setup events and all public zone transitions;
- cards revealed from deck/hand/prize-equivalent hidden zones;
- search, draw, discard, shuffle, promote, evolution, and trainer effects with public/private visibility tags;
- chance events such as coin flips;
- current public board, discard, stadium, points, turn, phase, pending choices, and visible energy zones.

The belief constructor should take `(public_history, observer_side, known_private_state_for_observer)` and produce a `PublicBeliefState`.

The first implementation may use exact simulator access during self-play to audit the belief, but the belief itself should be generated from public history and legal private information. Avoid making true hidden state the primary sampler input, because that preserves the wrong abstraction.

Required outputs:

- `PublicBeliefState`
- hidden-zone card probability summaries
- weighted private-state particles
- consistency/audit diagnostics
- deterministic serialization for replay and dataset rows

### 2. Public-belief state representation

Represent each search/train root as:

```text
PublicBeliefState:
  public_observation
  public_history_digest
  observer_side
  legal_actions
  particles: [PrivateWorldParticle]
  particle_weights
  belief_features
```

Each `PrivateWorldParticle` is a full private-world assignment consistent with public history and the observer's legal knowledge:

```text
PrivateWorldParticle:
  game_state
  weight
  hidden_zone_assignment
  likelihood_features
  rng_seed_label
```

`belief_features` should be model-facing and stable:

- opponent hand card-presence probabilities;
- opponent deck composition probabilities by card id and coarse card class;
- own deck draw probabilities when own deck order is unknown to the player;
- hidden-zone entropy/count features;
- reveal-history features;
- per-action belief annotations when relevant, e.g. probability that an action's target line is punished by known card classes.

The model should not receive sampled hidden card identities as ordinary visible cards. It receives public observation plus belief summaries.

### 3. Neural ReBeL model

Extend the current policy/value model into a belief-conditioned network:

```text
inputs:
  public state features
  card zone embeddings
  per-Uma slot tokens
  belief summary features
  legal action features
  optional public-history features

outputs:
  action policy logits over legal actions
  public-belief scalar value
  private-state / counterfactual value estimates
  optional action-value head
```

The key difference from the current value head is that value prediction is conditioned on public belief, not a single exact simulator state. If private-state value heads are too expensive for every hidden world, bucket them initially by particle/sample index at training time and distill to aggregate belief value plus optional action-Q.

Training losses:

- policy loss against public-belief search policy;
- value loss against search value and/or terminal outcome;
- counterfactual/private-state value loss where search supplies it;
- optional action-Q loss from subgame search;
- weak anchor/regularization to prevent the no-KL drift seen in pure AZ.

### 4. Public-belief subgame search

Implement a ReBeL-style search module separate from ordinary MCTS:

```text
run_public_belief_search(public_belief_state, config, model) -> BeliefSearchResult
```

`BeliefSearchResult` should include:

- root policy over legal public actions;
- root action values;
- public-belief value;
- per-particle/private-state values where available;
- regret/search diagnostics;
- sampled action;
- search tree/subgame summary suitable for debugging.

Search should operate over public belief states. At player decision nodes, actions are chosen from the public legal action set. At chance/private-update nodes, particle states and weights update according to draw/search/shuffle/reveal mechanics. At opponent decision nodes, the policy is conditioned on that opponent's own information set/public belief, not on the root player's hidden sample.

The principled target is CFR-style subgame search:

```text
initialize strategy from neural policy
for iteration in search_iters:
  traverse public-belief subgame
  use neural value at depth/leaf public-belief states
  update cumulative regrets per public information set
  update average strategy
return average root strategy + values
```

Practical implementation can use sampled particles and sampled traversals, but the search state must remain an information-set/public-belief state, not independent perfect-information MCTS per particle.

### 5. Belief-aware self-play

Add a new self-play driver, preferably a distinct Rust binary:

```text
sim-rebel-selfplay
```

Per decision:

1. Append public event history from the real game.
2. Build public belief state for side to act.
3. Run public-belief subgame search.
4. Sample/select action from the search policy with a temperature schedule.
5. Apply selected action to the real simulator state.
6. Record a `rebel-selfplay` row.

Rows should include:

```json
{
  "kind": "rebel-selfplay",
  "schemaVersion": 1,
  "beliefSchemaVersion": 1,
  "observation": {},
  "beliefFeatures": {},
  "publicHistoryDigest": {},
  "legalActions": [],
  "searchPolicy": [],
  "searchActionValues": [],
  "beliefValue": 0.0,
  "privateStateValues": [],
  "valueTarget": 1.0,
  "particleCount": 64,
  "searchIterations": 64,
  "searchAlgorithm": "public-belief-cfr-v1",
  "beliefSampler": "public-history-particles-v1",
  "playerDeckId": "...",
  "opponentDeckId": "..."
}
```

Use `kind="rebel-selfplay"` rather than overloading `mcts-selfplay`; this is a different algorithm and should fail loudly if accidentally loaded by the old dataset path.

### 6. Training and orchestration

Add a full training mode:

```text
data-mode: rebel
engine: rust
selfplay-binary: sim-rebel-selfplay
search: public-belief-cfr-v1
deck-sampling: uniform
model-side: both
```

The orchestrator should run:

1. self-play with public-belief search;
2. dataset validation and schema guard;
3. policy/value/counterfactual training;
4. ONNX export with belief-input metadata;
5. fixed gate;
6. uniform/deck-diverse gate;
7. side-split and matchup diagnostics;
8. checkpoint pool update.

This is a new line, not a patch on the old AlphaZero recipe. Manifest names should reflect that, e.g. `R17-rebel-e2e`.

### 7. Evaluation and exploitability probes

ReBeL's point is robust imperfect-information play, so evaluation cannot only be latest-vs-heuristic:

- fixed historical gate for continuity;
- uniform deck-diverse gate;
- side-conditioned gate;
- checkpoint league;
- policy-vs-rollout-MCTS comparison;
- belief-search-vs-ordinary-MCTS comparison;
- exploitability-style probe where a policy or search agent trains/responds against frozen ReBeL checkpoints;
- forced-state tactical benchmark with hidden-information cases.

The first full run is successful only if it produces valid artifacts across self-play, training, export, and eval. Strength promotion is separate.

## Validation Probes Inside The Big Build

These are not rollout phases. They are correctness and risk probes to run while implementing the full system.

### V0: Belief reconstruction audit

- For fixed seeds, reconstruct public beliefs at every decision.
- Assert the true hidden state is inside the support when history permits.
- Assert sampled particles match public observation and hidden-zone counts.
- Report belief entropy and particle diversity.

### V1: Information-set policy audit

- Create multiple private worlds with the same public observation.
- Run root search.
- Assert the root policy is one public action distribution, not one policy per hidden world.
- Flag any code path that indexes root action choice by hidden opponent private cards.

### V2: Search target audit

- Compare public-belief search policy against ordinary rollout MCTS on the same states.
- Track policy entropy, action-Q variance, particle disagreement, and selected-action disagreement.
- This can use small search iterations; it validates target shape, not strength.

### V3: Loader/model audit

- Load `rebel-selfplay` rows.
- Verify belief features, public features, legal action features, policy targets, scalar values, and private-state/counterfactual values collate correctly.
- Export ONNX and verify inference parity for belief-input tensors.

### V4: One-iteration system audit

- Run one full ReBeL iteration end-to-end.
- Validate manifests, artifact paths, schema metadata, and eval outputs.
- Do not interpret this as a phased milestone; it is a full-system smoke.

## Evaluation Gates

Report all of:

- fixed-matchup Wilson lower for continuity with old gates;
- uniform deck-sampling gate;
- side split;
- matchup matrix when available;
- target diagnostics:
  - legal-action count histogram;
  - search-policy top-1 share;
  - search-policy entropy;
  - particle action agreement;
  - aggregate Q variance across particles;
  - selected-action disagreement vs current rollout MCTS;
  - weight L2 drift per iteration.

Promotion bar:

- Do not require the first full ReBeL run to beat production. Require it to beat the pure-AZ failure mode: stable loop, non-degenerate public-belief search targets, no fast value-head collapse, no no-KL drift, and no evidence that private hidden state is leaking into root policy selection.

Strength bar:

- Once stable, it must eventually clear both fixed and uniform/deck-diverse gates before replacing production rollout-leaf search.

## Risks

### Strategy fusion remains

The main implementation hazard is accidentally falling back to per-particle perfect-information MCTS while calling it ReBeL. The mitigation is architectural: keep root policy/regret tables keyed by public information set, not by full hidden state, and make V1 fail if root action choice varies by hidden world that is invisible to the acting player.

### Belief sampler encodes impossible histories

Sampling from remaining hidden cards can produce worlds that are count-consistent but history-inconsistent. The big-bang scope includes public-history reconstruction specifically to avoid making count-only sampling the final belief model. The audit should still report how often particles are rejected and why.

### Compute multiplies quickly

Public-belief CFR/search can exceed current MCTS cost. The implementation should include search-iteration, particle-count, and depth/depth-value controls from the start, plus profiling output per decision. Small validation runs are fine, but the code path should be the same full ReBeL search path.

### Current observation features may be too thin

The current `PublicObservation` mostly contains visible state, own hand, and opponent counts/discards. Full ReBeL should add belief summaries as first-class model inputs: possible opponent hand classes, card-presence probabilities, draw odds, hidden-zone entropy, and public-history features.

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

## Big-Bang Work Packages

These work packages should be implemented against the full ReBeL target. They can be developed in parallel where file ownership is clean, but they should converge into one end-to-end branch/run rather than a series of reduced algorithm releases.

### A. Rust belief core

Owns:

- `engine-rs/crates/engine/src/belief/`
- public-history event types;
- hidden-zone constraint solver;
- weighted particle sampler;
- belief feature builder;
- belief serialization and replay;
- audits for support, consistency, entropy, and determinism.

Acceptance:

- Given a replayed public history and observer side, the belief builder returns deterministic weighted particles and model-facing belief features.
- The true simulator private state is either in support or the audit explains which public-history abstraction made it unrecoverable.

### B. Rust public-belief search

Owns:

- `engine-rs/crates/engine/src/rebel/` or `engine-rs/crates/engine/src/search/rebel/`;
- CFR-style public-belief subgame search;
- neural policy/value priors over belief features;
- leaf public-belief value calls;
- regret/average-strategy tables keyed by public information set;
- `BeliefSearchResult` diagnostics.

Acceptance:

- Root search returns one policy over public legal actions.
- The same public information set with different hidden particles does not produce separate root policies.
- Search can run with bounded iterations/depth and deterministic seed labels.

### C. ReBeL self-play binary

Owns:

- `engine-rs/crates/sim-cli/src/bin/rebel_selfplay.rs`;
- manifest output;
- `rebel-selfplay` JSONL rows;
- deck sampling integration;
- side/model selection;
- temperature schedule;
- public-history capture during real simulator advancement.

Acceptance:

- A one-game run produces valid `rebel-selfplay` rows with public belief metadata, search policy, action values, belief values, terminal value targets, and deck ids.

### D. Python belief dataset and model

Owns:

- `training/uma_ai/rebel_dataset.py`;
- belief feature packing;
- model input expansion;
- policy/value/counterfactual/action-Q losses;
- ONNX export metadata for belief-input tensors;
- smoke tests for collate and ONNX parity.

Acceptance:

- `data-mode=rebel` batches public state, card ids, slot tokens, belief features, legal action features, search policy targets, belief values, and optional private/counterfactual values.
- Exported ONNX carries enough metadata for Rust inference to reject schema mismatches.

### E. Orchestrator and gates

Owns:

- `training/r12_orchestrator.py` or a dedicated `training/rebel_orchestrator.py`;
- self-play/train/export/eval loop;
- checkpoint pool update;
- fixed and uniform/deck-diverse gates;
- side split and matchup diagnostics;
- schema preflight.

Acceptance:

- One full `R17-rebel-e2e` iteration runs self-play, trains, exports, evaluates, and writes a manifest that records every ReBeL-specific knob.

### F. Test and diagnostic suite

Owns:

- Rust unit tests for belief reconstruction and public-information-set invariants;
- sim smoke for `sim-rebel-selfplay`;
- Python dataset/model/export smokes;
- deterministic replay of a tiny public-history fixture;
- target diagnostics report.

Acceptance:

- All V0-V4 validation probes pass on small fixtures before any long run is trusted.
