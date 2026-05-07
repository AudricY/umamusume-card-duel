# Proposal: Trainable AI for Umamusume Card Duel

## Goal

Replace the current rule-driven AI with a trained policy that can play full games through legal actions, while keeping the TypeScript game engine as the source of truth for rules.

The target is not just "a neural net that picks attacks." The trained AI should eventually handle setup, benching, trainers, evolution, energy attachment, ability use, retreat, attack targeting, and end-turn decisions under imperfect information.

## Current Codebase Fit

The current AI is already separated into phase-specific decision functions:

- `frontend/src/game/engine.ts`
  - `advanceAiTurnStep` orchestrates the AI turn by phase.
  - `advanceOpponentTurnStep` and `advancePlayerAiTurnStep` expose AI stepping for opponent and AI-vs-AI play.
- `frontend/src/game/engine/flow/ai/core.ts`
  - `aiPlayOneBasic`
  - `aiPlayOneTrainer`
  - `aiEvolveOne`
  - `aiAttachOneEnergy`
  - `aiUseOneAbility`
  - `aiResolveCombatDecision`
- `frontend/src/game/engine/flow/ai/combatPlanner.ts`
  - `buildCombatCandidates` already enumerates combat/retreat/targeting candidates and scores them.

This structure is a good foundation. The trained model can replace phase selectors incrementally without rewriting card rules or combat resolution.

The main blockers are:

- Randomness is not fully injectable. `Math.random()` is still used by shuffle, energy rolls, trainer effects, combat effects, setup coin flip, and some ability paths.
- The AI functions choose and mutate in one step. Training needs a split between "enumerate legal actions", "choose action", and "apply action".
- The current AI often has access to full `GameState`, including hidden information. A trained agent should use a public observation unless intentionally training a cheating/debug baseline.
- There is no batch simulator or episode logger.
- The existing backend AI scenario command currently fails locally because `tsx` is missing from installed dependencies in this workspace.

Keep the current scenario runner useful, but do not build training around ad hoc backend imports of the frontend engine. Training, evaluation, and replay should use a dedicated headless simulator boundary with a stable API.

## Recommended Approach

Use a staged training strategy:

1. Build a deterministic simulator and legal-action API.
2. Convert the current heuristic AI into a policy implementation behind that API.
3. Generate supervised data from heuristic self-play and targeted scenarios.
4. Train a behavior cloning model as the first useful neural policy.
5. Improve through self-play reinforcement learning.
6. Export the trained model to an app-friendly inference format.

This avoids starting from random play, keeps rules in TypeScript, and gives measurable improvement checkpoints.

## Architecture

### 1. Policy Interface

Add a policy boundary around AI decisions:

```ts
export type AiPhase =
  | "setup"
  | "pendingChoice"
  | "bench"
  | "trainerBefore"
  | "evolve"
  | "attach"
  | "trainerAfter"
  | "ability"
  | "combat"
  | "stadiumOrEnd";

export type LegalAiAction = {
  id: string;
  phase: AiPhase;
  kind: string;
  payload: Record<string, unknown>;
};

export type AiPolicyInput = {
  observation: PublicObservation;
  legalActions: LegalAiAction[];
  sideId: "player" | "opponent";
  phase: AiPhase;
};

export type AiPolicy = {
  selectAction(input: AiPolicyInput): LegalAiAction;
};
```

The current AI should become `HeuristicPolicy`. The model-backed AI should become `ModelPolicy`.

The important rule is that policies select only from legal actions. They do not directly mutate `GameState`.

Pending player choices are part of the same policy surface. Promotion after knockout and switch-after-gust decisions should be represented as `pendingChoice` actions, not handled as special UI-only cases. The current player-oriented pending-choice resolver should be made side-parametric before the trained policy is expected to complete full games.

### 2. Legal Action Enumeration

Create legal-action builders for each phase:

- `enumerateSetupActions(state, sideId)`
  - active hand index
  - bench hand indexes
- `enumerateBenchActions(state, sideId)`
  - each playable basic in hand
  - pass
- `enumerateTrainerActions(state, sideId, pass)`
  - each playable trainer
  - all required trainer choices:
    - deck search choice when the acting player is legitimately searching their own deck
    - discard index
    - target UID
    - Rainbow Uncap target/evolution choice
  - pass
- `enumerateEvolutionActions(state, sideId)`
  - each valid evolution hand index and target UID
  - pass
- `enumerateAttachActions(state, sideId)`
  - each legal energy target
  - pass
- `enumerateAbilityActions(state, sideId)`
  - each usable ability and all legal targets/payment choices
  - pass
- `enumerateCombatActions(state, sideId)`
  - reuse and extend `buildCombatCandidates`
  - include controllable decisions such as attack target, heal target, retreat target, optional shuffle choice, optional discard payment choice, switch target, and pass/end turn
- `enumerateStadiumOrEndActions(state, sideId)`
  - use stadium if legal
  - end turn
- `enumeratePendingChoiceActions(state, sideId)`
  - promotion after knockout
  - switch-after-gust target selection

Legal actions must include only player-controllable decisions. Random discard indexes, coin results, shuffle permutations, random searched cards, generated energy, and any other stochastic outcomes are RNG events recorded after action application.

For high-branching effects, prefer multi-step pending-choice actions over one enormous action list. For example, a trainer can first be selected, then its required search/discard/target choices can be resolved through explicit pending choices if the branch count is too high.

Each legal action needs a stable `id` so logs, labels, and training examples remain consistent across TypeScript and Python. Do not rely only on mutable array indexes or module-level UIDs. Use episode-scoped card-instance identity and semantic references where possible, plus the current legal-action index as a tie-breaker for model targets.

### 3. Action Application

Add a single action applier:

```ts
export function applyAiAction(
  state: GameState,
  sideId: SideId,
  action: LegalAiAction,
  deps: AiActionDeps,
): GameState;
```

This function should call existing engine functions wherever possible. The current rule implementation remains the authority.

The legal action API should not duplicate rules. It should expose choices that are already legal according to existing rule functions, then apply the selected action through shared command handlers.

The current public action APIs cannot be reused blindly because many are hard-coded to `player`, while AI helpers choose and mutate in one step. First extract side-parametric command handlers that return an explicit result:

```ts
export type CommandResult =
  | { status: "applied"; state: GameState }
  | { status: "illegal"; reason: string }
  | { status: "needsChoice"; choices: LegalAiAction[] }
  | { status: "needsRandom"; event: RandomEventSpec };
```

The UI and AI should both wrap these handlers. Invalid or missing choices should not silently default in simulation mode; they should produce `illegal` or `needsChoice`.

### 4. Deterministic Simulator

Self-play requires reproducibility. Add a seeded RNG object:

```ts
export type Rng = {
  next: () => number;
  fork: (label: string) => Rng;
};
```

Thread it through:

- setup opening-hand shuffle
- setup coin flip
- turn-start energy-zone roll
- trainer random search/discard/energy effects
- combat coin flips
- combat random discard
- shuffle effects
- AI sampling/tie-breaking

Replace direct `Math.random()` calls in engine code with the injected RNG. Keep default wrappers for UI calls so existing gameplay still works.

Enumeration, observation building, and heuristic/model scoring must not consume the episode RNG. The main RNG stream is reserved for committed actions only. Candidate scoring should use forced expected outcomes, deterministic evaluator forks, or separate labeled RNG streams that cannot alter the real episode sequence.

Deterministic runs should record:

- seed
- deck IDs or exact deck lists
- policy versions
- action IDs
- random events
- final winner

### 5. Training Environment Contract

Before starting PPO or any other self-play RL, benchmark a headless simulator.

Required environment contract:

- `reset(seed, matchup, policyConfig)` returns the initial observation and legal actions.
- `step(actionId)` applies one policy decision and returns the next observation, legal actions, reward, terminal flag, and logged RNG events.
- episodes have a max-turn cap and explicit timeout/draw result.
- a replay command can reproduce an episode from seed, deck lists, policy versions, and action IDs.
- batch simulation runs in Node worker processes, not through React.
- Python training talks to the simulator through a batch worker protocol or generated datasets, not one slow subprocess call per decision.

Do not begin RL until this runner has measured decisions/second, full games/hour, worker scaling, and replay determinism. If throughput is too low, optimize the simulator before changing the algorithm.

### 5.1 Simulator Batching and Throughput

Training must use persistent Node simulator workers. Do not spawn one Node process per game or one subprocess call per decision.

Use a vectorized worker API:

- `init(config)` loads card data, rules, encoder schema, and policy hooks once.
- `resetBatch(envIds, seeds, matchups)` resets many games in one request.
- `stepBatch(envIds, actionIndexes)` advances many active games in one request.
- `replayEpisode(trace)` replays one failed game with strict determinism checks.
- `getMetrics()` reports decisions/sec, games/hour, terminal rate, timeout rate, memory, and mean legal-action count.

Each Node worker should own multiple environments, for example 32-256 games depending on memory and latency. Python should run several workers in parallel and batch model inference across all ready decisions.

The simulator should support two data modes:

- `debugJson`: complete JSON observations, legal actions, RNG events, and logs for replay/debugging.
- `tensorBatch`: compact numeric tensors and action masks for training throughput.

JSON is acceptable for the first prototype. Promotion to RL should require measured throughput. If JSON serialization dominates runtime, switch the hot path to a schema-checked binary format such as typed arrays, MessagePack, or Arrow IPC.

Batching invariants:

- no React imports
- no browser globals
- no wall-clock timers in simulation
- no UI delays
- no hidden dependency on module-level mutable state unless reset per environment
- deterministic seed partitioning by `rootSeed`, `workerId`, `envId`, and `episodeIndex`

### 5.2 Python/Node Bridge

Keep rules in TypeScript and learning in Python. The bridge should be a persistent protocol, not a loose collection of scripts.

Initial bridge:

- Python launches `node dist/training/simWorker.js`.
- Communication uses stdin/stdout with length-prefixed JSON messages.
- Every request has `requestId`, `schemaVersion`, and `command`.
- Every response has `requestId`, `ok`, and either `payload` or structured `error`.

Required commands:

- `handshake`
  - returns simulator version, rules hash, cards hash, observation schema, action schema, and encoder version
- `resetBatch`
  - inputs seeds and matchup specs
  - returns observation tensors, legal-action tensors, masks, and debug handles
- `stepBatch`
  - inputs selected action indexes
  - returns next tensors, rewards, terminal flags, winner, and compact step logs
- `exportReplay`
  - writes a failing episode trace to disk
- `validateReplay`
  - replays an episode trace and checks state/action/RNG hashes at every step

Python should never infer legal actions independently. It receives legal-action masks/features from Node and returns selected legal-action indexes. This prevents Python and TypeScript rule drift.

### 6. Public Observation

The production model should not receive hidden opponent hand/deck order. Define `PublicObservation` as a compact, serializable view:

- own side:
  - active card/evolution line
  - bench
  - hand card IDs
  - discard
  - deck size and known remaining counts if available
  - energy zone
  - points
  - per-card HP, max HP, energy counts, status, tool, ability-used flags
- opponent side:
  - active card/evolution line
  - bench
  - discard
  - hand size
  - deck size
  - points
  - public status/tool/energy/HP data
- shared:
  - stadium
  - turn number
  - current phase
  - first player
  - pending choice type
  - known rule flags such as supporter used, retreat used, stadium used

For experiments, also allow a `DebugFullStateObservation`, but do not use it for production-strength evaluation.

Define `PublicObservationV1` as a strict schema, not an ad hoc object copied from `GameState`.

Every observation should include:

- `schemaVersion`
- `rulesetVersion`
- `sideToAct`
- `phase`
- `turnNumber`
- `firstPlayer`
- `pendingChoiceKind`
- `own`, `opponent`, and `shared` blocks
- visibility metadata for fields that are public, own-private, inferred, search-revealed, or debug-only

Use fixed slot schemas for board positions:

- active: nullable `BoardUmaSlot`
- bench: fixed-length array padded to `MAX_BENCH`
- hand: own card IDs only
- opponent hand: count only
- deck: own count plus known composition summary, not order
- discard: public card IDs
- energy zone: own exact visible energy IDs/counts
- attached energy: public per-type counts
- tools, stadium, status, HP, max HP, ability-used flags, entered/evolved turn age

Do not expose raw `GameState`, `SideState`, deck arrays, or mutable object references to policies.

### 7. Information Boundaries and Feature Parity

Define every field as one of:

- public
- private to acting player
- inferred from public history
- debug-only

Production features must not include:

- opponent hand card IDs
- opponent deck order
- own deck order, except during a legal own-deck search choice
- RNG state
- simulator-only fields
- raw module-level UID creation order when it creates hidden leakage

Known remaining counts should distinguish decklist prior, own private hand, public discard, cards revealed by search effects, and cards inferred from public play. Add tests that fail if hidden fields enter `PublicObservation`.

TypeScript and Python feature encoders need golden fixtures. A small set of serialized observations/actions should produce byte-stable feature tensors in both environments, so training and inference use the same schema.

For own deck search, expose a separate `ChoiceObservation` scoped to that pending action. It may contain legal searchable candidates but should not permanently add whole-deck order to later observations.

Versioned training artifacts should include:

- `observation_schema_v1.json`
- `action_schema_v1.json`
- `card_metadata_v1.jsonl`
- `deck_metadata_v1.jsonl`
- `feature_encoder_manifest.json`
- `dataset_manifest.json`
- `golden_fixtures/*.json`

### 7.1 Action Schema

Every legal action should have two representations:

1. A stable semantic action for logs/replay.
2. A numeric action feature row for model scoring.

Suggested action fields:

- `schemaVersion`
- `id`
- `phase`
- `kind`
- `sourceZone`: hand, active, bench, discard, deck, stadium, none
- `sourceCardId`
- `sourceInstanceRef`
- `targetSide`
- `targetZone`
- `targetSlot`
- `targetCardId`
- `targetInstanceRef`
- `choiceCardId`
- `choiceCardIds`
- `attackIndex`
- `abilityName`
- `trainerEffectKind`
- `energyType`
- `numericAmount`
- `endsTurn`

Action IDs should be stable enough for replay, but the model target should still be `selectedActionIndex` into the legal-action list for that state. The feature encoder should never assume a global fixed action space.

### 7.2 Card and Deck Metadata

Generate a versioned card metadata table from `shared/src/data/cards.json`, separate from runtime art fields.

Suggested card features:

- card kind and trainer type
- species, stage, evolves-from species
- type, weakness, HP, and parsed retreat cost
- per-attack energy cost vector
- base/max printed damage
- target mode
- effect tags for draw, heal, gust, search, discard, energy attach/discard, switch, shuffle, status, damage prevention, coin flip, and bench damage
- ability tags using the same effect vocabulary
- implemented flag
- print-equivalence group, so full-art variants share gameplay embeddings

Deck metadata should include:

- deck ID
- exact card list hash
- energy pool
- archetype/style label when known
- primary species lines
- trainer counts by type
- evolution line completeness
- energy color count
- setup consistency estimates such as basic count, mulligan risk, and average attack cost

### 8. Data Logging

Use JSONL episodes:

```json
{
  "schemaVersion": 1,
  "episodeId": "seed-1234-game-000019",
  "step": 42,
  "seed": 1234,
  "sideId": "opponent",
  "phase": "attach",
  "observation": {},
  "legalActions": [],
  "selectedActionId": "attach:active:uid-7",
  "selectedActionIndex": 2,
  "policy": "heuristic-hard@2026-05-07",
  "split": "train",
  "rulesHash": "sha256:...",
  "cardsHash": "sha256:...",
  "simulatorCommit": "...",
  "rng": { "algorithm": "xoshiro128ss", "version": 1 },
  "result": {
    "winner": null,
    "points": { "player": 1, "opponent": 1 }
  }
}
```

For terminal examples, include final outcome and optionally backfill every step with:

- `gameResult`: win/loss/draw from acting side perspective
- `finalTurn`
- `finalPointDiff`
- `deckOutOccurred`

For RL and offline RL, also log:

- behavior action probability/log probability
- full legal-action logits when generated by a model
- value estimate
- shaped reward components
- unshaped terminal return
- opponent policy version
- league role of each policy
- curriculum stage
- whether the state came from normal play, DAgger, puzzle generation, or rollback replay

Each dataset should include a manifest:

- schema versions for observation, action, and feature encoders
- generator git commit
- rules/card content hashes
- RNG algorithm/version
- source seed ranges
- deck lists and matchup sampling weights
- policy/model manifests used to generate labels
- train/validation/test split files
- command to regenerate the same dataset hash from a clean checkout

Split datasets by seed and matchup. Promotion/evaluation seeds must not appear in training or tuning data. Weight or sample phases so pass/end-turn actions do not dominate behavior cloning.

Split by deterministic episode identity, not by individual rows, so the same game cannot leak across train/validation/test.

Use these split types:

- `train`: large self-play corpus
- `validation`: held-out seeds and same matchup distribution
- `test`: held-out seeds and side-swapped paired games
- `promotion`: never used for tuning
- `scenario`: hand-authored tactical states
- `regression`: fixed episodes used only for replay determinism and schema checks

Hold out at least some entire deck matchups or deck families if the goal is generalization beyond the current premade decks.

Training labels should include:

- selected legal action index
- selected semantic action ID
- teacher policy ID/version
- teacher confidence or score if available
- action-kind label
- phase label
- value target from final result, from acting side perspective
- final point differential
- turns remaining until terminal result
- optional tactical tags from heuristic telemetry, such as lethal, retreat, heal, setup, search, or deny-lethal

Raw self-play will overrepresent pass/end-turn, simple attacks, and common phases. Dataset builders should report and optionally rebalance rows by phase, action kind, deck matchup, turn bucket, terminal outcome, legal-action count bucket, teacher policy, and tactical tag. Prefer weighting over deleting too much data.

Use only game-symmetry-preserving augmentations:

- side swap with corresponding perspective rewrite
- bench slot permutation if action targets and instance refs are remapped consistently
- hand order permutation if legal action source indexes are remapped consistently
- equivalent print variant normalization
- action-list order randomization during training, with selected index remapped

Do not augment by changing public card IDs, damage, energy, HP, turn flags, or random outcomes unless the transformed state is revalidated by the engine and receives new legal actions.

### 9. Training Plan

#### Phase A: Behavior Cloning

Train a model to imitate `HeuristicPolicy`.

Purpose:

- Learn legal game flow quickly.
- Produce a neural baseline that can play coherent games.
- Validate observation encoding, action masking, export, and inference before RL complexity.

Inputs:

- public observation tensor
- legal action feature tensor
- action mask

Targets:

- selected legal action index
- optional value target from final winner

Loss:

- cross-entropy for policy
- optional binary/value loss for win probability
- optional phase/action-kind weighting to handle class imbalance

Expected limitation:

- The model will mostly inherit heuristic weaknesses.

Behavior cloning should use more than hard argmax labels where the heuristic has internal scores.

Training examples should include teacher scores when available:

- combat: `CombatCandidate.score`, `keepsSafe`, `lethalTarget`, `targetValue`
- trainer: playable decision plus trainer priority/search/discard choices
- evolution/attach/ability: extracted heuristic scores where practical

Prefer soft imitation targets for scored phases:

- use teacher scores as a Boltzmann distribution over legal actions
- keep hard cross-entropy as fallback when only one action is labeled
- train with phase-balanced sampling
- add action-kind reweighting for rare but important actions such as retreat, gust, promotion, and search choice

Behavior cloning success criteria:

- completes at least 99.5% of held-out games without timeout
- matches teacher action kind on held-out states by phase
- stays close to teacher win rate against current hard heuristic
- does not collapse to pass/end-turn in rare phases

#### Phase B: DAgger

After the behavior-cloned model can complete games, use DAgger to fix distribution shift.

Loop:

1. Roll out games with the current model policy.
2. At each visited state, record the model action distribution and selected action.
3. Query a teacher for the same public observation and legal action list.
4. Add teacher-labeled visited states to the dataset.
5. Retrain or fine-tune the model on the aggregate dataset.

Teacher choices can come from:

- current `HeuristicPolicy`
- heuristic plus shallow rollout for high-value states
- tactical puzzle oracle labels
- previous best model when it clearly outperforms the heuristic in a matchup

Prioritize DAgger examples where:

- model entropy is high
- model disagrees with teacher
- chosen action causes fast loss, no-bench loss, missed lethal, or fallback
- state is rare, such as pending choices, retreat, gust, search, or promotion

DAgger labels must be generated from the same `PublicObservation` and legal actions seen by the model. Do not let the teacher use hidden information unless the label is explicitly marked `debugTeacher`.

#### Phase C: Offline RL and Offline Improvement

Offline training should be conservative because the dataset will initially be generated by heuristics.

Useful approaches, in increasing complexity:

- advantage-weighted behavior cloning from final returns
- ranking loss that prefers actions leading to better shallow rollout outcomes
- IQL or CQL-style conservative offline RL over legal action candidates
- value head trained from terminal outcome and point differential

Do not trust offline RL promotion by training loss alone. Use it only to produce candidate checkpoints, then promote through fixed held-out simulator evaluation.

Avoid extrapolation:

- train Q/value only over enumerated legal actions
- penalize high value assigned to actions far outside the behavior distribution
- keep a behavior cloning loss mixed in during offline RL
- evaluate per-phase action distribution drift from the source dataset

#### Phase D: PPO Self-Play

Use self-play with legal action masking.

Start PPO from a behavior-cloned checkpoint, not from random initialization. PPO is a good first RL choice because it is straightforward with stochastic environments and dynamic action lists. Consider AlphaZero-style MCTS later only after the simulator is fast and deterministic.

PPO environment requirements:

- vectorized Node workers
- action masking before sampling
- stored log probability of the selected legal action
- value prediction from the same public observation
- GAE advantage estimates
- terminal result from acting side perspective
- max-turn truncation marked separately from true draw

Reward:

- terminal win: `+1`
- terminal loss: `-1`
- draw/timeout: `0`

Early training may use light shaping, but final evaluation should rely on terminal outcomes:

- point gain/loss
- avoiding immediate no-bench loss

Avoid heavy hand-written shaping because it can recreate the current heuristic AI under a different name. Do not reward improvement in the live model's own value estimate. If shaping is needed, use fixed potential-based shaping from a frozen evaluator or explicit game events only. Log shaped and unshaped returns separately, and promote checkpoints using unshaped terminal results.

Initial PPO recipe:

- terminal win/loss reward as main reward
- small entropy bonus, tracked by phase
- KL penalty or auxiliary BC loss to prevent immediate policy collapse
- value loss clipped or normalized by matchup
- reward normalization per rollout batch
- curriculum and opponent sampling controlled by config

### 9.1 Curriculum

Use curriculum only to improve sample efficiency, not as the final evaluation distribution.

Curriculum stages:

1. Combat-only tactical states using `buildCombatCandidates`.
2. Attach-plus-combat turns.
3. Trainer/search/discard choice states.
4. Full single-turn optimization from fixed board states.
5. Short games from midgame snapshots.
6. Full games across all premade decks.
7. League self-play with held-out matchups and seeds.

Curriculum should be mixed, not strictly sequential. Keep a replay buffer of earlier stages so the model does not forget basic tactical competence.

Scenario generators should randomize deck matchup, side to move, point totals, hand composition, bench size, damage counters, energy zone, public discard, and whether lethal, retreat, gust, or survival is available.

Do not train only on hand-authored tactical puzzles. They are valuable for coverage but easy to overfit.

### 9.2 League Training

Self-play should use a league instead of only latest-policy mirror matches.

League members:

- current main learner
- frozen snapshots of promoted models
- current hard heuristic
- noisy heuristic variants
- behavior-cloned baseline
- exploiters trained against the current main
- specialist policies for weak matchups, if needed

Opponent sampling:

- sample some games uniformly from the league
- oversample opponents near the learner's Elo
- oversample known bad matchups
- keep a fixed percentage against hard heuristic for continuity
- include side-swapped paired seeds

Keep frozen snapshots immutable. Training against only the latest model risks cyclic strategies and forgetting.

### 9.3 Avoiding Heuristic Overfit

The heuristic is a bootstrap teacher, not the objective.

Anti-overfitting controls:

- train on multiple teacher variants, including noisy tie-breaking
- use soft labels where candidate scores exist
- inject action dropout among near-equal teacher actions
- keep held-out seeds, matchups, and tactical puzzles
- evaluate against previous models, not only the teacher
- report model-vs-teacher disagreement by phase
- monitor whether the model inherits known heuristic failure modes
- include human-authored or rollout-derived corrections for known bad decisions

Useful diagnostics:

- missed lethal rate
- unnecessary retreat rate
- no-bench loss rate
- bad search/discard choice rate
- trainer waste rate
- overattachment to doomed active rate
- pass/end-turn rate by phase
- matchup-specific collapse

A model that imitates the heuristic perfectly is not final success. It is a working initialization.

### 10. Model Design

Use a candidate-scoring architecture:

- embed card IDs
- encode own active, own bench, opponent active, opponent bench
- encode hand as a set/sequence of card embeddings
- encode discard and known deck counts as pooled summaries
- encode scalar features:
  - HP ratios
  - energy counts by type
  - points
  - turn number
  - status flags
  - supporter/retreat/stadium usage
- encode each legal action with:
  - phase
  - action kind
  - referenced card ID
  - referenced target slot/card
  - numeric choice fields
- score each legal action independently against the state embedding
- output:
  - masked action logits
  - state value estimate

This is better than a fixed global action space because legal moves depend heavily on hand contents, board UIDs, attack text, and trainer effects.

#### 10.1 Architecture Options and Recommendation

Use candidate-action scoring as the default architecture. The current code already thinks in candidates: `buildCombatCandidates`, trainer choice helpers, attach target scoring, evolution scoring, and phase-specific AI steps all create small dynamic choice sets. A fixed global action vocabulary would fight the codebase because many legal actions refer to episode-local UIDs, hand indexes, deck-search indexes, and temporary pending choices.

Recommended first model:

- state encoder: small set/slot encoder, not a full transformer
- action encoder: per-legal-action feature encoder
- scorer: shared MLP that scores each `(state, action)` pair
- outputs: masked legal-action logits plus one or more value heads

A pragmatic first version should be intentionally small enough for ONNX Runtime Web. The current catalog is modest, so architectural complexity is more likely to fail on data/simulator issues than to unlock strength.

#### 10.2 Card and Zone Embeddings

Use a learned card embedding table keyed by stable `cardId`, combined with static numeric features from the card catalog.

Represent each in-play Umamusume instance as:

- current card ID
- evolution-line pooled embedding
- attached tool card ID or null token
- zone/slot embedding
- stage, HP ratio, HP missing, max HP
- attached energy counts by type
- total attached energy
- status flags
- damage/attack-block/ability-used flags
- entered/evolved turn age features, bucketed

Represent hand as a set of private card embeddings for the acting side. Represent opponent hand as count plus public/inferred summaries only. Represent discard as pooled card counts, not a long ordered sequence. Represent deck as size plus known/inferred count summaries; do not expose hidden order.

Use zone embeddings rather than relying on array position alone. Bench order may matter for UI references, but strategy mostly cares about active versus bench and target identity.

#### 10.3 State Encoder Options

Option A: pooled set encoder

Encode each entity independently, then pool by zone:

- own active
- own bench pooled plus max/top-k pooled
- opponent active
- opponent bench pooled plus max/top-k pooled
- own hand pooled
- own discard pooled
- opponent discard pooled
- known deck-count vector
- scalar game features

This is the recommended MVP. It is simple, fast, handles variable sizes, exports cleanly to ONNX, and matches the small board size.

Option B: slot encoder

Use fixed slots for active, bench, hand, discard summary, and deck summary. This is easiest for TypeScript/Python tensor parity. The risk is accidentally teaching hand/bench order artifacts. Use slot embeddings and randomize non-semantic ordering during training if order should not matter.

Option C: transformer encoder

A transformer over entity tokens is attractive later because it can model interactions such as "this hand evolution matches that benched species." It is not the best first production architecture. It adds export/runtime cost and needs more data to beat a well-built set encoder.

Option D: recurrent/history encoder

Do not make recurrence part of the first production model. Most immediate decisions can be made from public state plus inferred counts. Add a GRU/LSTM history encoder later only if hidden-information belief tracking becomes important. If added, feed compact public event tokens, not raw logs.

#### 10.4 Candidate Action Encoder

Each legal action should be encoded with structured fields, not just an opaque ID.

Action features should include:

- phase embedding
- action kind embedding
- source zone and source slot
- source card ID, if any
- target side, zone, and slot
- target card ID, if public
- referenced hand card ID for own hand choices
- referenced trainer/evolution/ability/attack ID
- energy type involved
- booleans for pass/end-turn/retreat/attack/search/discard/heal/gust/switch/shuffle
- numeric fields such as expected damage, expected healing, retreat cost paid, cards drawn, cards discarded

The action scorer should receive both the global state embedding and local entity embeddings for referenced source/target objects:

```txt
logit_i = MLP([
  state_embedding,
  action_embedding_i,
  source_entity_embedding_i,
  target_entity_embedding_i,
  source_target_interaction_features_i
])
```

This lets the model compare "attach to active" versus "attach to bench Stage 1" without requiring the global state vector to memorize every slot detail.

For early behavior cloning, optionally include existing heuristic candidate scores as auxiliary features or auxiliary regression targets. Do not depend on them permanently for production strength because that can cap the model at heuristic behavior.

#### 10.5 Action Masking

The model must never choose outside `legalActions`.

Implementation rule:

- model emits one score per legal action
- invalid actions are not present
- if fixed padding is used for batching, padded entries get very negative logits
- if all logits are invalid/NaN, fall back to heuristic policy and log it

For PPO, compute entropy only over legal actions. For behavior cloning, train cross-entropy over the legal-action list index, not over a global action ID. Keep `selectedActionId` for logging/replay, but use the current legal-action index as the supervised target after verifying it maps to the same stable action ID.

#### 10.6 Value Heads

Start with one phase-conditioned state value head:

```txt
V(s) = expected terminal result from acting side perspective
```

Targets:

- win: `+1`
- loss: `-1`
- draw/timeout: `0`

Useful auxiliary heads:

- point differential at game end
- probability of winning by points versus no-bench/deck-out
- immediate tactical value for combat candidates, distilled from `buildCombatCandidates`
- phase-specific "will attack this turn" or "will be KO'd before next turn" heads

Do not use the model's own value estimate as reward shaping. Value heads are for training stability, analysis, and optional search.

A later research model can add action-value heads `Q(s, a)` for candidate ranking and shallow lookahead. For the first production model, policy logits plus `V(s)` are enough.

#### 10.7 Uncertainty and Fallback

Treat uncertainty as a rollout and safety signal, not proof of correctness.

Track:

- policy entropy over legal actions
- top-1 versus top-2 logit margin
- value head confidence/calibration
- disagreement between small ensemble checkpoints, if affordable
- model versus heuristic disagreement in shadow mode

Use uncertainty for telemetry, training-scenario mining, heavier offline evaluation, and optional heuristic fallback during early rollout. Do not silently override high-entropy choices in final evaluation, or evaluation will measure the fallback system rather than the model.

#### 10.8 Recommended First Production Model

Use this architecture first:

```txt
PublicObservation
  -> card/static feature lookup
  -> entity encoders for active/bench/hand/discard/deck summaries
  -> pooled set/slot state encoder
  -> state embedding

LegalAction[i]
  -> action feature encoder
  -> source/target entity lookup
  -> candidate scorer MLP(state, action, source, target)
  -> logit[i]

state embedding
  -> value head
```

Suggested scale:

- card embedding: 32-64 dims
- entity embedding: 64-128 dims
- state embedding: 128-256 dims
- action embedding: 64-128 dims
- scorer MLP: 2-3 layers

This is the best fit for the current TypeScript engine because it preserves legal-action enumeration as the rules authority, keeps inference cheap, handles dynamic UIDs through action/entity references, and can be trained first from the existing heuristic before self-play is ready.

### 11. Model Manifest

Every model bundle should include an immutable manifest:

- model artifact ID
- training git commit
- training data manifest hash
- observation schema version
- action schema version
- feature encoder version
- rules/card content hashes
- RNG algorithm/version used for training data
- ONNX export version
- expected ONNX Runtime version
- evaluation report path/hash
- compatible app version or ruleset version

The app should verify manifest compatibility before loading a model. If compatibility fails, fall back to the heuristic AI and log the failure in dev telemetry.

### 11.1 Model Artifact Management

Treat every checkpoint as an immutable bundle:

- `model.onnx`
- `manifest.json`
- `encoder.json`
- `training_config.json`
- `eval_report.json`
- `dataset_manifest.json`
- optional `calibration.json`
- optional `debug_examples/`

Use content-addressed artifact IDs:

```txt
uma-policy-{rulesHashShort}-{dataHashShort}-{trainRunId}-{checkpointStep}
```

Never overwrite a promoted model. Promotion should create an alias such as `latest-dev`, `candidate`, or `production`, pointing to an immutable artifact.

### 12. Inference

Train in Python. Export to ONNX.

Runtime options:

- Browser inference with ONNX Runtime Web.
- Backend inference endpoint for heavier models.
- Heuristic fallback if model load or inference fails.

The first production model should be small enough for browser inference. A larger research model can run offline for generating improved labels or checkpoint evaluation.

Initial runtime targets:

- lazy-load the model only when a model-AI match starts
- keep compressed model bundle under an agreed budget before rollout
- cold-load within the match setup window
- p95 decision latency comfortably below the current AI step delay
- hard timeout per decision with heuristic fallback
- no visible UI lockup while the model loads or runs

Exact numbers should be set after the first ONNX prototype, but the promotion gate should include bundle size, cold-load time, p95 inference latency, timeout rate, and fallback rate.

### 12.1 ONNX Export Constraints

Design the model for ONNX before training.

Avoid export-hostile model behavior:

- Python-side loops over legal actions
- dynamic dictionaries
- ragged tensors without padding
- unsupported custom ops
- runtime string/card-ID processing
- control flow dependent on individual card text

Use fixed padded tensors:

- `state_features: float32[B, S]`
- `action_features: float32[B, A_MAX, F]`
- `action_mask: bool[B, A_MAX]`

The model outputs:

- `logits: float32[B, A_MAX]`
- `value: float32[B, 1]`

Set masked logits to a large negative value inside or immediately outside the model. Browser inference should not need Python-only postprocessing.

Pick `A_MAX` from measured legal-action counts with headroom. If a position exceeds `A_MAX`, the simulator should fail loudly in training/evaluation and use heuristic fallback in production.

### 13. Evaluation

Track:

- win rate against current hard heuristic
- win rate against previous model checkpoint
- mirror-match deck win rates
- per-deck matchup matrix
- average game length
- illegal-action rate, which should be impossible if masking is correct
- timeout/loop rate
- setup failure rate
- action distribution by phase
- model load/fallback/timeout rate
- p95 and p99 decision latency

Evaluation suites:

- existing AI combat scenarios
- deterministic full-game smoke tests
- fixed-seed regression games
- targeted tactical puzzles:
  - lethal target selection
  - bench survival
  - energy attachment planning
  - evolution timing
  - trainer search/discard choices
  - retreat decisions
  - special condition recovery

Use side-swapped paired seeds, held-out random seeds, and confidence intervals for win rates. Do not promote a model based on a tiny sample. Report aggregate results and per-matchup results; aggregate win rate must not hide a deck that collapses.

Suggested gate for first replacement:

- 55% or better win rate versus hard heuristic across at least 2,000 held-out paired-seed games.
- No illegal actions.
- No material regression on scenario tests.
- Reasonable performance on all premade AI decks, not only one favored deck.
- Per-matchup confidence intervals and side-swapped results are published with the checkpoint.
- Promotion seeds are separate from training and tuning seeds.

Consider an Elo or SPRT-style promotion protocol once checkpoint iteration becomes frequent.

### 13.1 Checkpoint Evaluation Protocol

Every checkpoint should run through the same evaluation harness before comparison.

Required evaluation tiers:

- smoke: 20-50 games to catch crashes and schema errors
- regression: fixed replay seeds from past bugs and tactical scenarios
- candidate: thousands of paired-seed games versus hard heuristic and previous promoted model
- matrix: per-deck and side-swapped matchup table
- stress: max-turn games, unusual pending choices, deck-out, no-bench losses, high-branching trainer choices

Use paired seeds:

- checkpoint A and checkpoint B both play each seed from both sides
- report win rate, draw rate, average turns, confidence interval, and per-matchup deltas
- flag any severe matchup regression even if aggregate win rate improves

Promotion should fail on:

- any illegal action
- replay nondeterminism
- simulator crash
- model timeout above threshold
- material scenario regression
- large per-deck collapse hidden by aggregate win rate

### 13.2 Experiment Tracking

Use a minimal experiment tracker from the start. A local file-based tracker is enough initially; it can later move to MLflow, Weights & Biases, or another system.

Each run should record:

- run ID
- git commit
- dirty worktree flag
- rules/cards hashes
- dataset manifest hash
- training config
- random seeds
- model architecture parameters
- loss curves
- policy entropy
- value loss
- invalid-mask diagnostics
- evaluation report paths
- artifact ID

For self-play, also track:

- opponent sampling distribution
- checkpoint pool contents
- win rates against each pool member
- draw/timeout rate
- mean legal actions per phase
- action-kind distribution drift

Without this, model changes will be impossible to attribute once simulator, encoder, data, and training code are all changing.

### 13.3 Golden Tests

Add golden fixtures before training starts.

Required golden tests:

- `PublicObservation` contains no opponent hand IDs or deck order.
- Observation serialization is byte-stable for fixed states.
- Legal action enumeration is deterministic and does not consume RNG.
- Every enumerated action can be applied or intentionally returns `needsChoice`.
- Model feature encoder produces identical tensors for the same fixture in TypeScript and Python.
- Action feature encoder preserves selected-action index after action-list shuffling.
- Replay from seed, deck lists, action IDs, and RNG events reproduces final winner and key board hashes.
- Card metadata generation is stable and changes hash when gameplay-relevant card data changes.
- Full-art/variant cards map to the expected print-equivalence group.
- Scenario fixtures cover lethal targeting, retreat, energy attach, evolution, trainer search/discard, gust, promotion, special conditions, and deck search visibility.

Each golden fixture should store:

- input state or replay prefix
- acting side
- observation JSON
- legal actions JSON
- selected action if applicable
- encoded tensor hash
- expected post-action board hash when applying a chosen action

### 13.4 Debugging Failed Games

Every failed evaluation game should produce a compact failure bundle:

- episode trace JSONL
- simulator config
- model artifact manifest
- deck lists
- seed
- worker/env IDs
- terminal reason
- final log
- first divergent replay step, if replay validation fails
- last N observations/actions before failure

Failure categories:

- illegal action selected
- legal action missing from enumeration
- action application rejected
- replay nondeterminism
- hidden-information leak detected
- ONNX/PyTorch output mismatch
- timeout or infinite loop
- terminal state inconsistency

Add a reducer command that attempts to minimize a failing trace by replaying prefixes and preserving the first failing condition. This makes model and rules bugs debuggable without reading a 200-turn game log.

### 14. Rollout Plan

Use staged rollout rather than replacing the heuristic immediately.

Rollout controls:

- feature flag for model AI by environment
- local/dev switch between heuristic, model, and shadow mode
- kill switch that forces heuristic fallback
- model manifest compatibility check before activation
- timeout fallback per decision
- telemetry for model load errors, decision latency, selected action, fallback reason, and illegal-action guard failures

Rollout stages:

1. Offline evaluation only.
2. Local dev model-vs-heuristic matches.
3. Shadow mode where the heuristic still acts, but model choices are logged for comparison.
4. Opt-in user-facing model AI.
5. Default model AI only after promotion gates are stable.

Rollback criteria:

- increased crash or unhandled-error rate
- repeated decision timeouts
- model load failures above the agreed threshold
- illegal action generated by the model/policy bridge
- visible latency regression during AI turns
- evaluation regression after card/rules changes

## Implementation Milestones

### Milestone 1: Simulator Foundations

- Add seeded RNG.
- Remove direct `Math.random()` from engine paths used by simulation.
- Add fast headless full-game runner.
- Restore/fix AI scenario test execution.
- Add replay by seed and action log.

Deliverable:

- `npm` script that runs deterministic AI-vs-AI games by seed.

### Milestone 2: Side-Parametric Commands

- Extract command handlers for setup, play-card, attach, ability, retreat, attack, stadium, end-turn, and pending-choice resolution.
- Make command handlers side-parametric.
- Return explicit `applied`, `illegal`, `needsChoice`, or `needsRandom` results.
- Wrap current UI and AI calls around these handlers without changing behavior.

Deliverable:

- player and AI actions share command handlers instead of separate mutation paths.

### Milestone 3: Legal Action API

- Define `LegalAiAction`, `AiPolicy`, `PublicObservation`.
- Implement legal action enumeration phase by phase.
- Implement `HeuristicPolicy` using existing scoring logic.
- Keep UI behavior unchanged.
- Add public-observation leakage tests.
- Add legal-action apply/enumeration parity tests.

Deliverable:

- current heuristic AI runs through the policy/action interface.

### Milestone 4: Episode Logging

- Add JSONL logger.
- Generate heuristic self-play data.
- Add schema versioning and validation.
- Add dataset manifest generation.

Deliverable:

- reproducible dataset from fixed seeds and deck matchups.

### Milestone 5: Behavior Cloning Baseline

- Add Python training project under `training/`.
- Train policy/value model from JSONL.
- Export ONNX.
- Add offline evaluator.
- Add golden feature parity fixtures between TypeScript and Python.

Deliverable:

- model can complete games through the legal-action interface.

### Milestone 6: App Integration

- Add `ModelPolicy`.
- Add model loading and heuristic fallback.
- Add config switch for heuristic/model AI.
- Add telemetry comparing model logits to heuristic choices in dev mode.
- Add model manifest validation.
- Add decision timeout fallback.

Deliverable:

- user can play against the trained model.

### Milestone 7: Self-Play RL

- Add self-play training loop.
- Evaluate every checkpoint against heuristic and prior champions.
- Promote checkpoints only through fixed evaluation gates.
- Benchmark simulator throughput before committing to PPO-scale runs.

Deliverable:

- model demonstrably beats hard heuristic across the deck pool.

### Milestone 8: DAgger and Offline Improvement

- Roll out BC policy and collect visited states.
- Label visited states with heuristic/rollout teachers.
- Add disagreement and high-entropy state mining.
- Train an improved imitation/offline RL checkpoint.

Deliverable:

- model beats the pure BC checkpoint on held-out games without increasing failure rate.

### Milestone 9: League Training

- Add league opponent registry.
- Save immutable policy snapshots.
- Add Elo/matchup matrix reporting.
- Train PPO against mixed opponents.
- Promote only through fixed gates.

Deliverable:

- model improves through self-play without forgetting earlier baselines.

## Risks and Mitigations

### Risk: Hidden Information Leakage

If the model sees full opponent state, evaluation will overstate strength.

Mitigation:

- Make `PublicObservation` the default.
- Add tests that verify hidden opponent hand/deck order is absent.
- Keep full-state observations behind an explicit debug flag.

### Risk: Action Space Bugs

If legal action enumeration omits valid moves, the model cannot learn them. If it includes invalid moves, training and inference become noisy.

Mitigation:

- Build legal actions from existing eligibility and play-rule functions.
- Add property tests: every enumerated action applies successfully or intentionally passes.
- Compare human UI-available actions against enumerated actions where possible.

### Risk: Simulator Too Slow

Self-play may require millions of decisions.

Mitigation:

- Run simulation in Node without React.
- Disable logs/telemetry during bulk runs.
- Avoid deep clone where action application can mutate an isolated episode state.
- Parallelize by seed across worker processes.

### Risk: Model Overfits One Deck

The model may learn a narrow deck or matchup.

Mitigation:

- Train across all premade decks.
- Balance matchup sampling.
- Report a matchup matrix, not just aggregate win rate.

### Risk: Heuristic Imitation Ceiling

Behavior cloning cannot reliably exceed the teacher.

Mitigation:

- Use behavior cloning only as a warm start.
- Add self-play and checkpoint league evaluation.
- Add tactical puzzle datasets for known weaknesses.

### Risk: Frontend Bundle Size or Latency

ONNX Runtime and model weights may be too heavy for initial page load.

Mitigation:

- Lazy-load model policy only when AI match starts.
- Keep heuristic fallback.
- Start with a small model.
- Consider backend inference for larger models.

## Suggested File Layout

```txt
frontend/src/game/engine/ai-policy/
  actions.ts
  applyAction.ts
  enumerateActions.ts
  heuristicPolicy.ts
  modelPolicy.ts
  observation.ts
  rng.ts
  types.ts

backend/src/sim/
  runEpisode.ts
  runBatch.ts
  writeEpisodes.ts
  simWorker.ts
  replayEpisode.ts

training/
  README.md
  pyproject.toml
  train_bc.py
  train_dagger.py
  train_ppo.py
  evaluate.py
  export_onnx.py
  uma_ai/
    dataset.py
    model.py
    features.py
    self_play.py
    node_bridge.py
    league.py

training/artifacts/
  observation_schema_v1.json
  action_schema_v1.json
  card_metadata_v1.jsonl
  deck_metadata_v1.jsonl
  golden_fixtures/
```

## First Concrete PR

The first PR should not train a model. It should make training possible.

Scope:

- Add seeded RNG support.
- Replace simulation-path `Math.random()` usage with injected RNG or documented compatibility wrappers.
- Add a deterministic headless full-game smoke runner.
- Add replay metadata: seed, deck lists, RNG algorithm/version, action log, and winner.
- Fix or document the missing dependency that prevents the AI scenario command from running.

Acceptance criteria:

- Existing game UI still works.
- Existing hard AI behavior is unchanged or explainably equivalent.
- Fixed seed produces byte-identical action/RNG logs on repeated runs.
- Different seeds change expected random events.
- No simulation-path direct `Math.random()` remains without an explicit compatibility wrapper.
- Scenario tests can run from `npm`.

The second PR should extract side-parametric command handlers. The third PR should add the first legal-action enumerators, starting with combat because `buildCombatCandidates` already exists.

## Decision

Proceed incrementally. Do not replace the current AI in one jump.

The right first milestone is a deterministic, legal-action simulator. Once that exists, behavior cloning and self-play become engineering work rather than guesswork.
