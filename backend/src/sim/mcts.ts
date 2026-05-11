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

export type MctsLeaf = "value-head";

export type MctsConfig = {
  simulations: number;
  cPuct: number;
  leaf: MctsLeaf;
  maxNodes: number;
  // Cap on how far we will collapse rule-bot turns after the model's move
  // before treating the leaf as the next model-decision state. Same idea
  // as `advanceHeuristicUntilModelTurnOrTerminal` in evaluateModelVsHeuristic.
  collapseMaxSteps: number;
};

export type MctsDiagnostics = {
  rootValue: number;
  rootVisitDistribution: number[];
  rootMeanQ: number[];
  expansions: number;
  leafEvaluations: number;
  terminalLeafs: number;
  visitedHashes: number;
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
};

export function defaultMctsConfig(overrides?: Partial<MctsConfig>): MctsConfig {
  return {
    simulations: 100,
    cPuct: 1.5,
    leaf: "value-head",
    maxNodes: 5000,
    collapseMaxSteps: 64,
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
    expansions: 1,
    leafEvaluations: 0,
    terminalLeafs: 0,
    visitedHashes: 0,
  };

  if (root.terminalValue !== null) {
    diagnostics.rootValue = root.terminalValue;
    diagnostics.rootVisitDistribution = [];
    diagnostics.rootMeanQ = [];
    return {
      selectedIndex: 0,
      visits: [],
      diagnostics,
    };
  }

  // Cache the root value via a single leaf eval so diagnostics can report
  // what the network thinks before any search ran. This is "free" since
  // the first simulation will descend through the same path.
  diagnostics.rootValue = await leafValue(rootState, modelSide, modelUrl);
  diagnostics.leafEvaluations += 1;

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
          : await leafValue(nextState, parent.modelSide, modelUrl);
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
        } else {
          leafValueScalar = await leafValue(newChild.state, newChild.modelSide, modelUrl);
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
    };
  }
  const uniform = 1 / legalActions.length;
  const priors = legalActions.map(() => uniform);
  void modelUrl; // unused in day-1 (uniform prior). Phase A will use it.
  return {
    state: collapsedState,
    modelSide,
    legalActions,
    priors,
    visits: legalActions.map(() => 0),
    wsum: legalActions.map(() => 0),
    children: legalActions.map(() => null),
    terminalValue: null,
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

async function leafValue(state: GameState, modelSide: SideId, modelUrl: string): Promise<number> {
  if (state.gameOver) return mctsTerminalValue(state, modelSide);
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
