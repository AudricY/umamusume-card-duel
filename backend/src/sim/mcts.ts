// R12 day-1 spike: PUCT MCTS over the modeled simulator.
//
// Design choices (see docs/r12-sprint-plan.md for rationale):
// - Tree nodes are *model-decision* states only. Opponent moves are
//   collapsed via the rule-bot between expansions, mirroring how the
//   existing `search`/`planner` selectors handle the same problem. This
//   removes the need for side-relative sign flips during backup, because
//   every node represents an outcome already in modelSide's frame.
// - Leaf evaluation queries `serve_onnx`'s `/predict` and uses the model's
//   side-relative tanh `value` directly. No mixing with the point-margin
//   `rewardForRollout` scaling — that mismatch is the trap the plan
//   warns against in the risk register.
// - Uniform prior in the day-1 spike. Phase A will swap in the policy
//   softmax (`actionProbs[0]`) without changing the tree structure.
// - Terminal value is clean ±1/0 (NO point-margin scaling) to stay
//   consistent with `_value_target` in `training/uma_ai/dataset.py`.

import { advancePlayerAiTurnStep, advanceOpponentTurnStep } from "../../../frontend/src/game/engine";
import { cloneGame } from "../../../frontend/src/game/engine/core/stateClone";
import { createSeededRng, type Rng } from "../../../frontend/src/game/engine/core/random";
import { enumerateLegalAiActions } from "../../../frontend/src/game/engine/ai-policy/actions";
import { buildPublicObservation } from "../../../frontend/src/game/engine/ai-policy/observation";
import type { LegalAiAction } from "../../../frontend/src/game/engine/ai-policy/types";
import type { GameState, SideId } from "../../../shared/src/types";
import { advanceModeledTurnStep, getForcedAttackCoinResults, stateHash } from "./evaluateModelVsHeuristic";

export type MctsLeaf = "value-head" | "rollout";
export type MctsPrior = "uniform" | "policy";

export type MctsConfig = {
  simulations: number;
  cPuct: number;
  leaf: MctsLeaf;
  prior: MctsPrior;
  // For `leaf: "rollout"`: number of CRN rollouts averaged at each leaf and
  // depth limit per rollout. The rollout-CRN-3 teacher we use elsewhere
  // averages 3 shared seeds; we mirror that as a default. Higher counts buy
  // less leaf variance at proportional latency cost.
  rolloutCrnSamples: number;
  rolloutSteps: number;
  // Phase A: at the root only, mix in Dirichlet noise (α applied uniformly,
  // ε weight on the noise) to encourage exploration. Off by default; the
  // self-play data generation phase enables it via the CLI flag.
  addRootDirichlet: boolean;
  dirichletAlpha: number;
  dirichletEpsilon: number;
  maxNodes: number;
  // Cap on how far we will collapse rule-bot turns after the model's move
  // before treating the leaf as the next model-decision state. Same idea
  // as `advanceHeuristicUntilModelTurnOrTerminal` in evaluateModelVsHeuristic.
  collapseMaxSteps: number;
  // R13.W2 adaptive halting: stop the simulation loop early once the root's
  // top action dominates the runner-up. Threshold is the ratio
  // (maxVisits / (secondMaxVisits + 1)). Once it exceeds `adaptiveRatio`
  // AND at least `adaptiveMinSims` sims have run, the remaining budget is
  // skipped. Disabled when adaptiveRatio <= 0 (the default). The "+1" in
  // the denominator avoids division-by-zero when the runner-up has 0
  // visits (which would otherwise force an early halt after sim 1).
  adaptiveRatio: number;
  adaptiveMinSims: number;
};

export type MctsDiagnostics = {
  rootValue: number;
  rootVisitDistribution: number[];
  rootMeanQ: number[];
  rootPriors: number[];
  expansions: number;
  leafEvaluations: number;
  terminalLeafs: number;
  visitedHashes: number;
  // Phase A diagnostics: entropy of the (pre-Dirichlet) policy prior at
  // the root, and whether the visit-count argmax agrees with the prior's
  // argmax. Both are policy-vs-search signals that surface to the
  // observability stack.
  rootPriorEntropy: number;
  rootPriorArgmax: number;
  // R13.W2 adaptive halting: how many sims actually ran, and whether
  // the loop was cut short by the (maxVisits / secondMax) ratio rule.
  simulationsRun: number;
  haltedEarly: boolean;
};

export type MctsResult = {
  selectedIndex: number;
  visits: number[];
  diagnostics: MctsDiagnostics;
};

type MctsNode = {
  state: GameState;
  modelSide: SideId;
  legalActions: LegalAiAction[];
  priors: number[];
  visits: number[];
  wsum: number[];
  children: (MctsNode | null)[];
  // null = non-terminal interior node. number = value already determined
  // by terminal state, in modelSide frame, in [-1, 1].
  terminalValue: number | null;
  // Phase A: when buildModelDecisionNode calls /predict to fetch the
  // policy prior, it harvests `value[0]` from the same response and caches
  // it here so backup doesn't need a second /predict round trip.
  cachedLeafValue: number | null;
};

export function defaultMctsConfig(overrides?: Partial<MctsConfig>): MctsConfig {
  return {
    simulations: 100,
    cPuct: 1.5,
    leaf: "value-head",
    prior: "uniform",
    rolloutCrnSamples: 3,
    rolloutSteps: 200,
    addRootDirichlet: false,
    dirichletAlpha: 0.3,
    dirichletEpsilon: 0.25,
    maxNodes: 5000,
    collapseMaxSteps: 64,
    adaptiveRatio: 0,
    adaptiveMinSims: 20,
    ...overrides,
  };
}

export function mctsTerminalValue(state: GameState, modelSide: SideId): number {
  // Clean ±1/0 contract that mirrors `_value_target` in dataset.py.
  // The point-margin scaling used by `rewardForRollout` is intentionally
  // omitted to keep training and inference value semantics aligned.
  if (!state.gameOver || state.winner === null) return 0;
  return state.winner === modelSide ? 1 : -1;
}

export async function runMcts(
  rootState: GameState,
  modelSide: SideId,
  config: MctsConfig,
  modelUrl: string,
  seed: string,
): Promise<MctsResult> {
  const rootRng = createSeededRng(seed, "mcts-root");
  const root = await buildModelDecisionNode(rootState, modelSide, modelUrl, config);
  const diagnostics: MctsDiagnostics = {
    rootValue: 0,
    rootVisitDistribution: [],
    rootMeanQ: [],
    rootPriors: [],
    expansions: 1,
    leafEvaluations: 0,
    terminalLeafs: 0,
    visitedHashes: 0,
    rootPriorEntropy: 0,
    rootPriorArgmax: 0,
    simulationsRun: 0,
    haltedEarly: false,
  };

  if (root.terminalValue !== null) {
    diagnostics.rootValue = root.terminalValue;
    return {
      selectedIndex: 0,
      visits: [],
      diagnostics,
    };
  }

  // Snapshot the policy prior *before* any Dirichlet noise — diagnostics
  // (entropy, argmax) describe the network's belief, not the noisy
  // exploration shim. The visit-vs-prior argmax-match downstream is
  // therefore a clean network-vs-search agreement signal.
  diagnostics.rootPriors = root.priors.slice();
  diagnostics.rootPriorEntropy = entropy(diagnostics.rootPriors);
  diagnostics.rootPriorArgmax = argmax(diagnostics.rootPriors);

  if (config.addRootDirichlet && root.legalActions.length > 1) {
    const noise = sampleDirichlet(root.legalActions.length, config.dirichletAlpha, rootRng.fork("dirichlet"));
    const eps = config.dirichletEpsilon;
    root.priors = root.priors.map((p, i) => (1 - eps) * p + eps * (noise[i] ?? 0));
  }

  // Cache the root value. Only reuse the harvested value if we're using
  // value-head leaves; in rollout mode the cached scalar is the policy's
  // value estimate, not what we want to seed the diagnostic with.
  if (config.leaf === "value-head" && root.cachedLeafValue !== null) {
    diagnostics.rootValue = root.cachedLeafValue;
  } else {
    diagnostics.rootValue = await leafValue(rootState, modelSide, modelUrl, config, rootRng.fork("root-leaf"));
    diagnostics.leafEvaluations += 1;
  }

  let totalNodes = 1;

  for (let sim = 0; sim < config.simulations; sim += 1) {
    const simRng = rootRng.fork(`sim${sim}`);
    const path: { node: MctsNode; actionIndex: number }[] = [];
    let node: MctsNode = root;

    // Selection: walk down the tree via PUCT until we hit a leaf
    // (either an unexpanded child or a terminal node).
    while (true) {
      if (node.terminalValue !== null) break;
      const actionIndex = puctSelect(node, config.cPuct);
      path.push({ node, actionIndex });
      const child = node.children[actionIndex];
      if (child === null || child === undefined) break;
      node = child;
    }

    let leafValueScalar: number;
    if (node.terminalValue !== null) {
      // Leaf is terminal — back up the deterministic value directly.
      leafValueScalar = node.terminalValue;
      diagnostics.terminalLeafs += 1;
    } else {
      // Leaf is the deepest node we got to without finding a child for
      // the chosen action. Expand it.
      const last = path[path.length - 1]!;
      const parent = last.node;
      const actionIndex = last.actionIndex;
      const action = parent.legalActions[actionIndex]!;
      const nextState = stepFromModelDecision(
        parent.state,
        parent.modelSide,
        action,
        config,
        simRng.fork(`expand:a${actionIndex}`),
      );
      if (nextState === null) {
        // Action didn't change state — treat as a terminal value of 0
        // and discourage re-selection by recording a visit with neutral Q.
        leafValueScalar = 0;
      } else if (totalNodes >= config.maxNodes) {
        // Memory cap reached — evaluate the leaf without storing a node.
        leafValueScalar = nextState.gameOver
          ? mctsTerminalValue(nextState, parent.modelSide)
          : await leafValue(nextState, parent.modelSide, modelUrl, config, simRng.fork(`leaf:cap:a${actionIndex}`));
        diagnostics.leafEvaluations += nextState.gameOver ? 0 : 1;
        diagnostics.terminalLeafs += nextState.gameOver ? 1 : 0;
      } else {
        const newChild = await buildModelDecisionNode(
          nextState,
          parent.modelSide,
          modelUrl,
          config,
        );
        parent.children[actionIndex] = newChild;
        totalNodes += 1;
        diagnostics.expansions += 1;
        if (newChild.terminalValue !== null) {
          leafValueScalar = newChild.terminalValue;
          diagnostics.terminalLeafs += 1;
        } else if (config.leaf === "value-head" && newChild.cachedLeafValue !== null) {
          // Policy-prior mode: value was harvested from the same /predict
          // call that produced the priors; no extra fetch needed.
          leafValueScalar = newChild.cachedLeafValue;
        } else {
          leafValueScalar = await leafValue(newChild.state, newChild.modelSide, modelUrl, config, simRng.fork(`leaf:expand:a${actionIndex}`));
          diagnostics.leafEvaluations += 1;
        }
      }
    }

    // Backup: every node in the path is a modelSide-decision state, so
    // the leaf value (in modelSide frame) is added unmodified.
    for (const step of path) {
      const i = step.actionIndex;
      step.node.visits[i] = (step.node.visits[i] ?? 0) + 1;
      step.node.wsum[i] = (step.node.wsum[i] ?? 0) + leafValueScalar;
    }

    diagnostics.simulationsRun = sim + 1;

    // Adaptive halt: once the top action's visit count dominates the
    // runner-up by the configured ratio AND a minimum number of sims
    // have run (so the early-noise phase doesn't trip the rule),
    // remaining budget is skipped. Cheap O(legalActions) check.
    if (config.adaptiveRatio > 0 && diagnostics.simulationsRun >= config.adaptiveMinSims && root.visits.length >= 2) {
      let topVisits = 0;
      let secondVisits = 0;
      for (const v of root.visits) {
        if (v > topVisits) {
          secondVisits = topVisits;
          topVisits = v;
        } else if (v > secondVisits) {
          secondVisits = v;
        }
      }
      if (topVisits / (secondVisits + 1) >= config.adaptiveRatio) {
        diagnostics.haltedEarly = true;
        break;
      }
    }
  }

  const visits = root.visits.slice();
  const totalVisits = visits.reduce((sum, n) => sum + n, 0);
  diagnostics.rootVisitDistribution = totalVisits > 0
    ? visits.map((n) => n / totalVisits)
    : visits.map(() => 0);
  diagnostics.rootMeanQ = root.visits.map((n, i) => (n > 0 ? root.wsum[i]! / n : 0));
  diagnostics.visitedHashes = totalNodes;

  // Argmax visits with tiebreak by mean Q.
  let bestIndex = 0;
  let bestVisits = -1;
  let bestQ = -Infinity;
  for (let i = 0; i < visits.length; i += 1) {
    const n = visits[i]!;
    const q = diagnostics.rootMeanQ[i]!;
    if (n > bestVisits || (n === bestVisits && q > bestQ)) {
      bestVisits = n;
      bestQ = q;
      bestIndex = i;
    }
  }

  return { selectedIndex: bestIndex, visits, diagnostics };
}

function puctSelect(node: MctsNode, cPuct: number): number {
  const sumN = node.visits.reduce((sum, n) => sum + n, 0);
  const sqrtSum = Math.sqrt(Math.max(1, sumN));
  let bestIndex = 0;
  let bestScore = -Infinity;
  for (let i = 0; i < node.legalActions.length; i += 1) {
    const n = node.visits[i]!;
    const q = n > 0 ? node.wsum[i]! / n : 0;
    const u = cPuct * (node.priors[i] ?? 0) * sqrtSum / (1 + n);
    const score = q + u;
    if (score > bestScore) {
      bestScore = score;
      bestIndex = i;
    }
  }
  return bestIndex;
}

async function buildModelDecisionNode(
  state: GameState,
  modelSide: SideId,
  modelUrl: string,
  config: MctsConfig,
): Promise<MctsNode> {
  // If the state is terminal, return a sentinel node with no actions and
  // the terminal value baked in.
  if (state.gameOver) {
    return {
      state,
      modelSide,
      legalActions: [],
      priors: [],
      visits: [],
      wsum: [],
      children: [],
      terminalValue: mctsTerminalValue(state, modelSide),
      cachedLeafValue: mctsTerminalValue(state, modelSide),
    };
  }

  // Should always be a model-decision state by construction. Defensive:
  // if somehow we land on the opponent's turn (e.g., due to a turn-end
  // action that flips currentSide), collapse forward until model's turn.
  let collapsedState = state;
  if (collapsedState.currentSide !== modelSide) {
    collapsedState = collapseUntilModelOrTerminal(collapsedState, modelSide, config.collapseMaxSteps, createSeededRng(`${stateHash(state)}:precollapse`, "mcts-collapse"));
    if (collapsedState.gameOver) {
      return {
        state: collapsedState,
        modelSide,
        legalActions: [],
        priors: [],
        visits: [],
        wsum: [],
        children: [],
        terminalValue: mctsTerminalValue(collapsedState, modelSide),
        cachedLeafValue: mctsTerminalValue(collapsedState, modelSide),
      };
    }
  }

  const legalActions = enumerateLegalAiActions(collapsedState, modelSide);
  if (legalActions.length === 0) {
    return {
      state: collapsedState,
      modelSide,
      legalActions: [],
      priors: [],
      visits: [],
      wsum: [],
      children: [],
      terminalValue: 0,
      cachedLeafValue: 0,
    };
  }

  let priors: number[];
  let cachedLeafValue: number | null = null;
  if (config.prior === "policy") {
    const { actionProbs, value } = await predictPolicyAndValue(modelUrl, collapsedState, modelSide, legalActions);
    priors = legalActions.map((_, i) => Math.max(1e-8, actionProbs[i] ?? 0));
    cachedLeafValue = value;
  } else {
    const uniform = 1 / legalActions.length;
    priors = legalActions.map(() => uniform);
  }

  return {
    state: collapsedState,
    modelSide,
    legalActions,
    priors,
    visits: legalActions.map(() => 0),
    wsum: legalActions.map(() => 0),
    children: legalActions.map(() => null),
    terminalValue: null,
    cachedLeafValue,
  };
}

async function predictPolicyAndValue(
  modelUrl: string,
  state: GameState,
  modelSide: SideId,
  legalActions: LegalAiAction[],
): Promise<{ actionProbs: number[]; value: number }> {
  const response = await fetch(`${modelUrl.replace(/\/$/, "")}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      observation: buildPublicObservation(state, modelSide),
      legalActions,
      sampling: "greedy",
    }),
  });
  if (!response.ok) {
    throw new Error(`MCTS prior+value request failed: ${response.status} ${await response.text()}`);
  }
  const payload = await response.json() as { actionProbs?: number[][]; value?: number[] };
  const probs = (payload.actionProbs?.[0] ?? []).slice(0, legalActions.length);
  // Renormalize masked-by-legal slice in case the server returned the full
  // action-vocabulary head; legal-only fragment should already sum to ~1
  // because the server masks before softmax.
  const sum = probs.reduce((s, p) => s + (Number.isFinite(p) ? Math.max(0, p) : 0), 0);
  const normalized = sum > 0 ? probs.map((p) => Math.max(0, p) / sum) : legalActions.map(() => 1 / legalActions.length);
  return {
    actionProbs: normalized,
    value: Number(payload.value?.[0] ?? 0),
  };
}

function stepFromModelDecision(
  state: GameState,
  modelSide: SideId,
  action: LegalAiAction,
  config: MctsConfig,
  rng: Rng,
): GameState | null {
  // Apply the model's action, then advance the heuristic opponent until
  // it's the model's turn again (or the game ends).
  const forcedCoins = getForcedAttackCoinResults(state, rng);
  const afterModel = advanceModeledTurnStep(state, modelSide, action, forcedCoins, rng);
  if (stateHash(afterModel) === stateHash(state)) return null;
  if (afterModel.gameOver) return afterModel;
  if (afterModel.currentSide === modelSide) return afterModel;
  return collapseUntilModelOrTerminal(afterModel, modelSide, config.collapseMaxSteps, rng);
}

function collapseUntilModelOrTerminal(
  state: GameState,
  modelSide: SideId,
  maxSteps: number,
  rng: Rng,
): GameState {
  let current = cloneGame(state);
  for (let step = 0; step < maxSteps; step += 1) {
    if (current.gameOver) break;
    if (current.currentSide === modelSide) break;
    const before = stateHash(current);
    const sideId: SideId = current.currentSide === "player" ? "player" : "opponent";
    const forcedCoins = getForcedAttackCoinResults(current, rng);
    current = sideId === "player"
      ? advancePlayerAiTurnStep(current, forcedCoins, rng.next)
      : advanceOpponentTurnStep(current, forcedCoins, rng.next);
    if (stateHash(current) === before) break;
  }
  return current;
}

function entropy(probs: number[]): number {
  let h = 0;
  for (const p of probs) {
    if (p > 0) h -= p * Math.log(p);
  }
  return h;
}

function argmax(values: number[]): number {
  let best = 0;
  let bestVal = -Infinity;
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i] ?? -Infinity;
    if (v > bestVal) {
      bestVal = v;
      best = i;
    }
  }
  return best;
}

function sampleDirichlet(n: number, alpha: number, rng: Rng): number[] {
  // Sample n i.i.d. Gamma(α, 1) variates via Marsaglia–Tsang and normalize.
  // Stays valid for α >= 0.1; with α=0.3 (plan default) the sampler is stable.
  const samples = Array.from({ length: n }, () => sampleGamma(Math.max(0.05, alpha), rng));
  const total = samples.reduce((sum, v) => sum + v, 0);
  if (total <= 0) return samples.map(() => 1 / n);
  return samples.map((v) => v / total);
}

function sampleGamma(alpha: number, rng: Rng): number {
  // Marsaglia–Tsang for shape >= 1; boost-and-discard wrapper for shape < 1.
  if (alpha < 1) {
    const u = Math.max(1e-12, rng.next());
    return sampleGamma(alpha + 1, rng) * Math.pow(u, 1 / alpha);
  }
  const d = alpha - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  while (true) {
    let x = 0;
    let v = 0;
    // Box–Muller on uniform pair → standard normal
    do {
      const u1 = Math.max(1e-12, rng.next());
      const u2 = rng.next();
      x = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rng.next();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

async function leafValue(
  state: GameState,
  modelSide: SideId,
  modelUrl: string,
  config: MctsConfig,
  rng: Rng,
): Promise<number> {
  if (state.gameOver) return mctsTerminalValue(state, modelSide);
  if (config.leaf === "rollout") {
    return rolloutLeafValue(state, modelSide, config, rng);
  }
  return valueHeadLeafValue(state, modelSide, modelUrl);
}

async function valueHeadLeafValue(state: GameState, modelSide: SideId, modelUrl: string): Promise<number> {
  const legalActions = enumerateLegalAiActions(state, modelSide);
  const response = await fetch(`${modelUrl.replace(/\/$/, "")}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      observation: buildPublicObservation(state, modelSide),
      legalActions,
      sampling: "greedy",
    }),
  });
  if (!response.ok) {
    throw new Error(`MCTS leaf value request failed: ${response.status} ${await response.text()}`);
  }
  const payload = await response.json() as { value?: number[] };
  return Number(payload.value?.[0] ?? 0);
}

function rolloutLeafValue(
  state: GameState,
  modelSide: SideId,
  config: MctsConfig,
  rng: Rng,
): number {
  // K shared CRN seeds → run the rule-bot heuristic from the leaf state
  // forward to game-over (or rollout-steps cap), score by side-relative
  // ±1/0 terminal value. Mean across K samples. NO point-margin scaling,
  // to stay consistent with the value-head leaf semantics.
  const k = Math.max(1, config.rolloutCrnSamples);
  let sum = 0;
  let counted = 0;
  for (let i = 0; i < k; i += 1) {
    const sample = rolloutHeuristic(state, rng.fork(`rollout-crn-${i}`), config.rolloutSteps);
    sum += mctsTerminalValue(sample, modelSide);
    counted += 1;
  }
  return counted > 0 ? sum / counted : 0;
}

function rolloutHeuristic(state: GameState, rng: Rng, maxSteps: number): GameState {
  let next = cloneGame(state);
  for (let step = 0; step < maxSteps; step += 1) {
    if (next.gameOver) break;
    const before = stateHash(next);
    const sideId: SideId = next.currentSide === "player" ? "player" : "opponent";
    const forcedCoins = getForcedAttackCoinResults(next, rng);
    next = sideId === "player"
      ? advancePlayerAiTurnStep(next, forcedCoins, rng.next)
      : advanceOpponentTurnStep(next, forcedCoins, rng.next);
    if (stateHash(next) === before) break;
  }
  return next;
}
