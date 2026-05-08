import {
  advanceOpponentTurnStep,
  advancePlayerAiTurnStep,
  autoCompleteOpponentSetup,
  canAttack,
  canAttachEnergyToUmamusume,
  chooseOpeningCoin,
  completePregameSetup,
  createGame,
  dealOpeningHands,
  getCard,
  getPlayableAction,
  getPrimaryAttack,
  getUmamusumeCard,
  tickSetupCountdown,
} from "../../../frontend/src/game/engine";
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { chooseAiSetupSelection } from "../../../frontend/src/app/gameUiHelpers";
import { enumerateLegalAiActions, chooseHighestScoredAction } from "../../../frontend/src/game/engine/ai-policy/actions";
import { buildPublicObservation } from "../../../frontend/src/game/engine/ai-policy/observation";
import type { LegalAiAction } from "../../../frontend/src/game/engine/ai-policy/types";
import { getAiPhase } from "../../../frontend/src/game/engine/ai-policy/phase";
import { cloneGame } from "../../../frontend/src/game/engine/core/stateClone";
import { createSeededRng, randomFloat, withRng, type Rng } from "../../../frontend/src/game/engine/core/random";
import { attachEnergy } from "../../../frontend/src/game/engine/flow/energy";
import { canUseUmamusumeAbility } from "../../../frontend/src/game/engine/flow/eligibility";
import { performAttack, knockOutUmamusume } from "../../../frontend/src/game/engine/flow/combat";
import { choosePreferredActiveIndex, normalizeBoardState, refreshContinuousHp, switchOutOpponentActive } from "../../../frontend/src/game/engine/flow/board";
import { endTurn, startTurn } from "../../../frontend/src/game/engine/flow/turn";
import { adjustHandChoices, resolveCardPlay } from "../../../frontend/src/game/engine/flow/playRules";
import { canUseStadium, useStadium } from "../../../frontend/src/game/engine/flow/trainers";
import { aiRetreatToTarget } from "../../../frontend/src/game/engine/flow/ai/combatPlanner";
import { aiUseCoinFlipDrawAbility, aiUseDamageAbility, aiUseMoveBenchedEnergyAbility } from "../../../frontend/src/game/engine/flow/ai/abilityUtils";
import { estimateAttackDamageOutput, markAbilityUsed, withEnergyShift } from "../../../frontend/src/game/engine/flow/ai/attachUtils";
import type { AiCombatDecision } from "../../../frontend/src/game/engine/flow/ai/types";
import { findOwnUmamusumeByUid, getAllUmamusume } from "../../../frontend/src/game/engine/core/umamusume";
import { getUmamusumeAbility } from "../../../frontend/src/game/engine/flow/abilityRules";
import { getAbilityMoveEnergyTypes } from "../../../frontend/src/game/engine/flow/energy";
import { drawCards } from "../../../frontend/src/game/engine/flow/turn";
import type { PlayChoices } from "../../../frontend/src/game/engine/core/playTypes";
import type { CoinFlipResult, EnergyType, GameState, SideId, SideState, UmamusumeInstance } from "../../../shared/src/types";
import { stateFingerprint } from "./stateFingerprint";
import { rankLegalActions, type CandidateRankerMode } from "./candidateRanker";
import { withGitMetadata } from "./manifest";

export type EvaluateModelArgs = {
  modelUrl: string;
  games: number;
  seedStart: number;
  maxSteps: number;
  modelSide: SideId | "both";
  details: boolean;
  selection: "policy" | "baseline" | "value" | "rollout" | "search" | "planner";
  rolloutSteps: number;
  searchDepth: number;
  searchTopK: number;
  searchSamples: number;
  ranker: CandidateRankerMode;
  decisionTraceOut: string | null;
  manifestOut: string | null;
  plannerTopK: number;
  plannerMaxSequences: number;
  plannerMaxDepth: number;
};

type GameResult = {
  seed: string;
  modelSide: SideId;
  winner: SideId | null;
  modelWon: boolean;
  terminalReason: "gameOver" | "maxSteps" | "stalled";
  steps: number;
  turnNumber: number;
  points: Record<SideId, number>;
  modelActions: number;
  heuristicFallbacks: number;
  selectedCandidateRanks: number[];
  decisionTraces: DecisionTraceRow[];
};

type DecisionTraceRow = {
  schemaVersion: 1;
  source: "model-visited";
  seed: string;
  modelSide: SideId;
  step: number;
  sideId: SideId;
  selection: EvaluateModelArgs["selection"];
  observation: ReturnType<typeof buildPublicObservation>;
  legalActions: LegalAiAction[];
  selectedActionId: string;
  selectedActionIndex: number;
  selectedOriginalRank?: number;
  heuristicSelectedActionId: string;
  heuristicSelectedActionIndex: number;
  fallback: boolean;
  result: {
    winner: SideId | null;
    modelWon: boolean;
    points: Record<SideId, number>;
    terminalReason: GameResult["terminalReason"];
  } | null;
};

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const sides: SideId[] = args.modelSide === "both" ? ["player", "opponent"] : [args.modelSide];
  const results: GameResult[] = [];
  if (args.decisionTraceOut) {
    mkdirSync(dirname(args.decisionTraceOut), { recursive: true });
    writeFileSync(args.decisionTraceOut, "", "utf8");
  }
  for (const modelSide of sides) {
    for (let index = 0; index < args.games; index += 1) {
      const seed = String(args.seedStart + index);
      const result = await runModelVsHeuristicGame(args, seed, modelSide);
      if (args.decisionTraceOut && result.decisionTraces.length) {
        appendFileSync(args.decisionTraceOut, result.decisionTraces.map((row) => JSON.stringify(row)).join("\n") + "\n", "utf8");
      }
      results.push(result);
    }
  }
  const summary = summarize(results);
  const output = args.details ? { args, summary, results } : { args, summary };
  if (args.manifestOut) {
    mkdirSync(dirname(args.manifestOut), { recursive: true });
    writeFileSync(args.manifestOut, JSON.stringify(withGitMetadata(output), null, 2) + "\n", "utf8");
  }
  console.log(JSON.stringify(output, null, 2));
}

export async function runModelVsHeuristicGame(args: EvaluateModelArgs, seed: string, modelSide: SideId): Promise<GameResult> {
  const rng = createSeededRng(`${seed}:${modelSide}`, "model-vs-heuristic");
  return withRng(rng, () => runModelVsHeuristicGameWithRng(args, seed, modelSide, rng));
}

async function runModelVsHeuristicGameWithRng(args: EvaluateModelArgs, seed: string, modelSide: SideId, rng: Rng): Promise<GameResult> {
  let state = setupAiVsAiGame();
  let terminalReason: GameResult["terminalReason"] = "maxSteps";
  let modelActions = 0;
  let heuristicFallbacks = 0;
  const selectedCandidateRanks: number[] = [];
  const decisionTraces: DecisionTraceRow[] = [];

  for (let step = 0; step < args.maxSteps; step += 1) {
    if (state.gameOver) {
      terminalReason = "gameOver";
      break;
    }
    const beforeHash = stateHash(state);
    const sideId = state.currentSide === "player" || state.currentSide === "opponent" ? state.currentSide : "player";
    const forcedCoinResults = getForcedAttackCoinResults(state, rng);
    if (sideId === modelSide) {
      const legalActions = enumerateLegalAiActions(state, sideId);
      const heuristic = chooseHighestScoredAction(legalActions);
      const heuristicSelectedActionIndex = Math.max(0, legalActions.findIndex((action) => action.id === heuristic.id));
      const decision = args.selection === "value"
        ? await chooseValueAction(args.modelUrl, state, sideId, rng)
        : args.selection === "baseline"
          ? chooseBaselineAction(state, sideId)
        : args.selection === "rollout"
          ? chooseRolloutAction(args, state, sideId, rng)
        : args.selection === "search"
          ? chooseSearchAction(args, state, sideId, `${seed}:${modelSide}:${step}`)
        : args.selection === "planner"
          ? choosePlannerAction(args, state, sideId, `${seed}:${modelSide}:${step}`)
        : await chooseModelAction(args.modelUrl, state, sideId);
      modelActions += 1;
      if (decision.selectedOriginalRank !== undefined) selectedCandidateRanks.push(decision.selectedOriginalRank);
      const next = advanceModeledTurnStep(state, sideId, decision.action, forcedCoinResults, rng);
      const fallback = stateHash(next) === beforeHash;
      if (args.decisionTraceOut) {
        const trace: DecisionTraceRow = {
          schemaVersion: 1,
          source: "model-visited",
          seed,
          modelSide,
          step,
          sideId,
          selection: args.selection,
          observation: buildPublicObservation(state, sideId),
          legalActions,
          selectedActionId: decision.action.id,
          selectedActionIndex: decision.selectedIndex,
          heuristicSelectedActionId: heuristic.id,
          heuristicSelectedActionIndex,
          fallback,
          result: null,
        };
        if (decision.selectedOriginalRank !== undefined) trace.selectedOriginalRank = decision.selectedOriginalRank;
        decisionTraces.push(trace);
      }
      if (fallback) {
        heuristicFallbacks += 1;
        state = sideId === "player"
          ? advancePlayerAiTurnStep(state, forcedCoinResults, rng.next)
          : advanceOpponentTurnStep(state, forcedCoinResults, rng.next);
      } else {
        state = next;
      }
    } else {
      state = sideId === "player"
        ? advancePlayerAiTurnStep(state, forcedCoinResults, rng.next)
        : advanceOpponentTurnStep(state, forcedCoinResults, rng.next);
    }
    if (stateHash(state) === beforeHash) {
      terminalReason = "stalled";
      break;
    }
    if (step === args.maxSteps - 1 && state.gameOver) terminalReason = "gameOver";
  }

  const result: GameResult = {
    seed,
    modelSide,
    winner: state.winner,
    modelWon: state.winner === modelSide,
    terminalReason,
    steps: modelActions,
    turnNumber: state.turnNumber,
    points: { player: state.sides.player.points, opponent: state.sides.opponent.points },
    modelActions,
    heuristicFallbacks,
    selectedCandidateRanks,
    decisionTraces,
  };
  decisionTraces.forEach((trace) => {
    trace.result = {
      winner: result.winner,
      modelWon: result.modelWon,
      points: result.points,
      terminalReason: result.terminalReason,
    };
  });
  return result;
}

async function chooseModelAction(modelUrl: string, state: GameState, sideId: SideId): Promise<{ action: LegalAiAction; selectedIndex: number; selectedOriginalRank?: number }> {
  const legalActions = enumerateLegalAiActions(state, sideId);
  if (legalActions.length <= 1) return { action: legalActions[0] ?? chooseHighestScoredAction(legalActions), selectedIndex: 0 };
  const response = await fetch(`${modelUrl.replace(/\/$/, "")}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      observation: buildPublicObservation(state, sideId),
      legalActions,
    }),
  });
  if (!response.ok) throw new Error(`Model server returned ${response.status}: ${await response.text()}`);
  const payload = await response.json() as { selectedIndex?: number[] };
  const selectedIndex = Math.max(0, Math.min(legalActions.length - 1, Number(payload.selectedIndex?.[0] ?? 0)));
  return { action: legalActions[selectedIndex] ?? legalActions[0]!, selectedIndex };
}

function chooseBaselineAction(state: GameState, sideId: SideId): { action: LegalAiAction; selectedIndex: number; selectedOriginalRank?: number } {
  const legalActions = enumerateLegalAiActions(state, sideId);
  const selected = chooseHighestScoredAction(legalActions);
  const selectedIndex = Math.max(0, legalActions.findIndex((action) => action.id === selected.id));
  return rankedDecision(legalActions, selectedIndex);
}

async function chooseValueAction(modelUrl: string, state: GameState, sideId: SideId, rng: Rng): Promise<{ action: LegalAiAction; selectedIndex: number; selectedOriginalRank?: number }> {
  const legalActions = enumerateLegalAiActions(state, sideId);
  if (legalActions.length <= 1) return { action: legalActions[0] ?? chooseHighestScoredAction(legalActions), selectedIndex: 0 };
  let bestIndex = 0;
  let bestScore = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < legalActions.length; index += 1) {
    const action = legalActions[index]!;
    const next = advanceModeledTurnStep(state, sideId, action, getForcedAttackCoinResults(state, rng), rng);
    const score = stateHash(next) === stateHash(state)
      ? Number.NEGATIVE_INFINITY
      : next.gameOver
        ? terminalValue(next, sideId)
        : await predictStateValue(modelUrl, next, sideId);
    if (score > bestScore) {
      bestScore = score;
      bestIndex = index;
    }
  }
  return { action: legalActions[bestIndex]!, selectedIndex: bestIndex };
}

async function predictStateValue(modelUrl: string, state: GameState, sideId: SideId): Promise<number> {
  const legalActions = enumerateLegalAiActions(state, sideId);
  const response = await fetch(`${modelUrl.replace(/\/$/, "")}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      observation: buildPublicObservation(state, sideId),
      legalActions,
    }),
  });
  if (!response.ok) throw new Error(`Model server returned ${response.status}: ${await response.text()}`);
  const payload = await response.json() as { value?: number[] };
  return Number(payload.value?.[0] ?? 0);
}

function terminalValue(state: GameState, sideId: SideId): number {
  const opponentId: SideId = sideId === "player" ? "opponent" : "player";
  const pointMargin = (state.sides[sideId].points - state.sides[opponentId].points) / 3;
  if (state.winner === sideId) return 1 + pointMargin * 0.2;
  if (state.winner === opponentId) return -1 + pointMargin * 0.2;
  return pointMargin;
}

function chooseRolloutAction(args: EvaluateModelArgs, state: GameState, sideId: SideId, rng: Rng): { action: LegalAiAction; selectedIndex: number; selectedOriginalRank?: number } {
  const legalActions = enumerateLegalAiActions(state, sideId);
  if (legalActions.length <= 1) return { action: legalActions[0] ?? chooseHighestScoredAction(legalActions), selectedIndex: 0 };
  let bestIndex = 0;
  let bestReward = Number.NEGATIVE_INFINITY;
  legalActions.forEach((action, index) => {
    const next = advanceModeledTurnStep(state, sideId, action, getForcedAttackCoinResults(state, rng), rng);
    const reward = stateHash(next) === stateHash(state)
      ? Number.NEGATIVE_INFINITY
      : rewardForRollout(rolloutHeuristic(next, rng, args.rolloutSteps), sideId);
    if (reward > bestReward) {
      bestReward = reward;
      bestIndex = index;
    }
  });
  return rankedDecision(legalActions, bestIndex);
}

function chooseSearchAction(args: EvaluateModelArgs, state: GameState, sideId: SideId, seed: string): { action: LegalAiAction; selectedIndex: number; selectedOriginalRank?: number } {
  const legalActions = enumerateLegalAiActions(state, sideId);
  if (legalActions.length <= 1) return { action: legalActions[0] ?? chooseHighestScoredAction(legalActions), selectedIndex: 0 };
  const memo = new Map<string, number>();
  let bestIndex = 0;
  let bestReward = Number.NEGATIVE_INFINITY;
  legalActions.forEach((action, index) => {
    const rewards = Array.from({ length: Math.max(1, args.searchSamples) }, (_, sample) => (
      scoreRootSearchAction(args, state, sideId, action, args.searchDepth, `${seed}:root:s${sample}`, memo)
    ));
    const reward = rewards.reduce((sum, value) => sum + value, 0) / rewards.length;
    if (reward > bestReward) {
      bestReward = reward;
      bestIndex = index;
    }
  });
  return rankedDecision(legalActions, bestIndex);
}

function choosePlannerAction(args: EvaluateModelArgs, state: GameState, sideId: SideId, seed: string): { action: LegalAiAction; selectedIndex: number; selectedOriginalRank?: number } {
  const legalActions = enumerateLegalAiActions(state, sideId);
  if (legalActions.length <= 1) return { action: legalActions[0] ?? chooseHighestScoredAction(legalActions), selectedIndex: 0 };
  const bundles = enumerateTurnBundles(args, state, sideId, seed);
  let best = bundles[0];
  let bestReward = Number.NEGATIVE_INFINITY;
  bundles.forEach((bundle, index) => {
    const reward = rewardForRollout(rolloutHeuristic(bundle.state, createSeededRng(`${seed}:bundle:${index}:leaf`, "planner-leaf"), args.rolloutSteps), sideId);
    if (reward > bestReward) {
      bestReward = reward;
      best = bundle;
    }
  });
  return best?.first ?? rankedDecision(legalActions, 0);
}

type TurnBundle = {
  first: { action: LegalAiAction; selectedIndex: number; selectedOriginalRank?: number };
  state: GameState;
};

function enumerateTurnBundles(args: EvaluateModelArgs, state: GameState, sideId: SideId, seed: string): TurnBundle[] {
  type PartialBundle = {
    first: TurnBundle["first"] | null;
    state: GameState;
    depth: number;
  };
  const complete: TurnBundle[] = [];
  const queue: PartialBundle[] = [{ first: null, state: cloneGame(state), depth: 0 }];
  while (queue.length && complete.length < args.plannerMaxSequences) {
    const current = queue.shift()!;
    if (current.state.gameOver || current.state.currentSide !== sideId || current.depth >= args.plannerMaxDepth) {
      if (current.first) complete.push({ first: current.first, state: current.state });
      continue;
    }
    const legalActions = enumerateLegalAiActions(current.state, sideId);
    const candidates = rankLegalActions(legalActions, {
      topK: Math.max(1, args.plannerTopK),
      mode: args.ranker,
      baseline: chooseHighestScoredAction(legalActions),
      rng: createSeededRng(`${seed}:d${current.depth}:ranker`, "planner-ranker"),
    });
    candidates.forEach(({ action, index, originalRank }) => {
      if (complete.length + queue.length >= args.plannerMaxSequences) return;
      const rng = createSeededRng(`${seed}:d${current.depth}:a${index}`, "planner-action");
      const before = stateHash(current.state);
      const next = advanceModeledTurnStep(current.state, sideId, action, getForcedAttackCoinResults(current.state, rng), rng);
      if (stateHash(next) === before) return;
      const first = current.first ?? firstDecision(action, index, originalRank);
      if (next.gameOver || next.currentSide !== sideId || action.kind === "attack" || action.kind === "endTurn") {
        complete.push({ first, state: next });
      } else {
        queue.push({ first, state: next, depth: current.depth + 1 });
      }
    });
  }
  return complete;
}

function firstDecision(action: LegalAiAction, selectedIndex: number, originalRank: number): TurnBundle["first"] {
  return { action, selectedIndex, selectedOriginalRank: originalRank };
}

function rankedDecision(
  legalActions: LegalAiAction[],
  selectedIndex: number,
): { action: LegalAiAction; selectedIndex: number; selectedOriginalRank?: number } {
  const originalRank = rankLegalActions(legalActions, { topK: legalActions.length, mode: "heuristic" }).find((entry) => entry.index === selectedIndex)?.originalRank;
  const result: { action: LegalAiAction; selectedIndex: number; selectedOriginalRank?: number } = {
    action: legalActions[selectedIndex]!,
    selectedIndex,
  };
  if (originalRank !== undefined) result.selectedOriginalRank = originalRank;
  return result;
}

function scoreRootSearchAction(
  args: EvaluateModelArgs,
  state: GameState,
  modelSide: SideId,
  action: LegalAiAction,
  depth: number,
  seed: string,
  memo: Map<string, number>,
): number {
  const rng = createSeededRng(seed, "root-search-action");
  const next = advanceModeledTurnStep(state, modelSide, action, getForcedAttackCoinResults(state, rng), rng);
  if (stateHash(next) === stateHash(state)) return Number.NEGATIVE_INFINITY;
  return evaluateSearchState(args, next, modelSide, Math.max(0, depth - 1), `${seed}:after`, memo);
}

function evaluateSearchState(
  args: EvaluateModelArgs,
  state: GameState,
  modelSide: SideId,
  depth: number,
  seed: string,
  memo: Map<string, number>,
): number {
  if (state.gameOver) return rewardForRollout(state, modelSide);
  const key = `${depth}:${stateHash(state)}`;
  const cached = memo.get(key);
  if (cached !== undefined) return cached;

  const modelDecisionState = advanceHeuristicUntilModelTurnOrTerminal(state, modelSide, seed, args.rolloutSteps);
  if (modelDecisionState.gameOver || depth <= 0) {
    const reward = rewardForRollout(rolloutHeuristic(modelDecisionState, createSeededRng(`${seed}:leaf`, "search-leaf"), args.rolloutSteps), modelSide);
    memo.set(key, reward);
    return reward;
  }

  const legalActions = enumerateLegalAiActions(modelDecisionState, modelSide);
  const candidates = rankLegalActions(legalActions, {
    topK: Math.max(1, args.searchTopK),
    mode: args.ranker,
    baseline: chooseHighestScoredAction(legalActions),
    rng: createSeededRng(`${seed}:ranker`, "search-ranker"),
  });
  if (candidates.length === 0) return rewardForRollout(modelDecisionState, modelSide);

  let best = Number.NEGATIVE_INFINITY;
  candidates.forEach(({ action, index }) => {
    const rng = createSeededRng(`${seed}:d${depth}:a${index}`, "search-action");
    const next = advanceModeledTurnStep(modelDecisionState, modelSide, action, getForcedAttackCoinResults(modelDecisionState, rng), rng);
    const reward = stateHash(next) === stateHash(modelDecisionState)
      ? Number.NEGATIVE_INFINITY
      : evaluateSearchState(args, next, modelSide, depth - 1, `${seed}:d${depth}:a${index}`, memo);
    if (reward > best) best = reward;
  });
  memo.set(key, best);
  return best;
}

function advanceHeuristicUntilModelTurnOrTerminal(state: GameState, modelSide: SideId, seed: string, maxSteps: number): GameState {
  let next = cloneGame(state);
  const rng = createSeededRng(`${seed}:heuristic`, "search-heuristic");
  for (let step = 0; step < maxSteps; step += 1) {
    if (next.gameOver || next.currentSide === modelSide) break;
    const before = stateHash(next);
    const sideId = next.currentSide === "player" || next.currentSide === "opponent" ? next.currentSide : "player";
    const forcedCoinResults = getForcedAttackCoinResults(next, rng);
    next = sideId === "player"
      ? advancePlayerAiTurnStep(next, forcedCoinResults, rng.next)
      : advanceOpponentTurnStep(next, forcedCoinResults, rng.next);
    if (stateHash(next) === before) break;
  }
  return next;
}

function rolloutHeuristic(state: GameState, rng: Rng, maxSteps: number): GameState {
  let next = cloneGame(state);
  for (let step = 0; step < maxSteps; step += 1) {
    if (next.gameOver) break;
    const before = stateHash(next);
    const sideId = next.currentSide === "player" || next.currentSide === "opponent" ? next.currentSide : "player";
    const forcedCoinResults = getForcedAttackCoinResults(next, rng);
    next = sideId === "player"
      ? advancePlayerAiTurnStep(next, forcedCoinResults, rng.next)
      : advanceOpponentTurnStep(next, forcedCoinResults, rng.next);
    if (stateHash(next) === before) break;
  }
  return next;
}

function rewardForRollout(state: GameState, sideId: SideId): number {
  const opponentId: SideId = sideId === "player" ? "opponent" : "player";
  const pointMargin = (state.sides[sideId].points - state.sides[opponentId].points) / 3;
  if (state.winner === sideId) return 1 + pointMargin * 0.2;
  if (state.winner === opponentId) return -1 + pointMargin * 0.2;
  return pointMargin;
}

export function advanceModeledTurnStep(
  state: GameState,
  sideId: SideId,
  action: LegalAiAction,
  forcedAttackCoinResult: CoinFlipResult | CoinFlipResult[] | undefined,
  rng: Rng,
): GameState {
  const next = cloneGame(state);
  if (next.phase !== "play" || next.pendingPlayerChoice || next.gameOver || next.currentSide !== sideId) return next;
  const side = next.sides[sideId];
  if (!side.active) return next;

  if (action.kind === "pass") {
    advanceModeledPhase(next, sideId);
    return next;
  }

  if (action.kind === "playBasic" || action.kind === "playTrainer") {
    playSelectedHandCard(next, side, Number(action.payload.handIndex ?? -1), readPlayChoices(action.payload));
    return next;
  }

  if (action.kind === "evolve") {
    playSelectedHandCard(next, side, Number(action.payload.handIndex ?? -1), { umamusumeTargetUid: Number(action.payload.targetUid) });
    return next;
  }

  if (action.kind === "attachEnergy") {
    const target = findOwnUmamusumeByUid(side, Number(action.payload.targetUid));
    if (target && canAttachEnergyToUmamusume(next, side, target)) {
      attachEnergy(next, side, target);
      normalizeBoardState(next);
    }
    return next;
  }

  if (action.kind === "useAbility") {
    const used = useSelectedAbility(next, side, action.payload, rng);
    if (!used) next.opponentTurnStep = "attack";
    return next;
  }

  if (action.kind === "attack" || action.kind === "retreatAttack" || action.kind === "retreat" || action.kind === "endTurn") {
    resolveSelectedCombat(next, side, action.payload.decision as AiCombatDecision | undefined, forcedAttackCoinResult);
    return next;
  }

  if (action.kind === "useStadium") {
    if (canUseStadium(next, side) && useStadium(next, side)) finishTurn(next);
    return next;
  }

  advanceModeledPhase(next, sideId);
  return next;
}

function playSelectedHandCard(state: GameState, side: SideState, handIndex: number, choices: PlayChoices = {}): void {
  const cardId = side.hand[handIndex];
  if (!cardId) return;
  const card = getCard(cardId);
  const play = getPlayableAction(state, side, cardId);
  if (!play.canPlay) return;
  side.hand.splice(handIndex, 1);
  resolveCardPlay(state, side, card, play, adjustHandChoices(choices, handIndex), switchOutOpponentActive);
  normalizeBoardState(state);
  refreshContinuousEffects(state);
}

function useSelectedAbility(state: GameState, side: SideState, payload: Record<string, unknown>, rng: Rng): boolean {
  const sourceUid = Number(payload.sourceUid);
  const source = getAllUmamusume(side).find((umamusume) => umamusume.uid === sourceUid);
  if (!source || !canUseUmamusumeAbility(state, side, source.uid)) return false;
  const ability = getUmamusumeAbility(state, side.id, source);
  if (!ability) return false;
  const deps = { refreshContinuousEffects, choosePreferredActiveIndex };
  if (ability.damageOpponent) {
    const opponentId: SideId = side.id === "player" ? "opponent" : "player";
    const opponent = state.sides[opponentId];
    const targetUid = payload.targetUid === undefined ? undefined : Number(payload.targetUid);
    const target = ability.damageOpponentTarget === "any"
      ? (targetUid !== undefined ? getAllUmamusume(opponent).find((umamusume) => umamusume.uid === targetUid) : undefined) ?? opponent.active
      : opponent.active;
    if (!target) return false;
    if (ability.discardEnergy) {
      const canPay = Object.entries(ability.discardEnergy).every(([type, amount]) => source.energies[type as EnergyType] >= (amount ?? 0));
      if (!canPay) return false;
      Object.entries(ability.discardEnergy).forEach(([type, amount]) => {
        source.energies[type as EnergyType] = Math.max(0, source.energies[type as EnergyType] - (amount ?? 0));
      });
    }
    target.hp = Math.max(0, target.hp - ability.damageOpponent);
    target.tookDamageThisTurn = ability.damageOpponent > 0;
    markSelectedAbilityUsed(side, source, ability.name, Boolean(ability.oncePerGame));
    if (target.hp <= 0) {
      if (knockOutUmamusume(state, side.id, opponentId, target, choosePreferredActiveIndex)) {
        if (!state.gameOver) refreshContinuousEffects(state);
      }
    }
    return true;
  }
  if (ability.moveBenchedEnergyToActive) {
    if (!side.active) return false;
    const energySourceUid = Number(payload.energySourceUid);
    const energyType = payload.energyType as EnergyType | undefined;
    if (!energyType || !getAbilityMoveEnergyTypes(ability).includes(energyType)) return false;
    const energySource = side.bench.find((umamusume) => umamusume.uid === energySourceUid);
    if (!energySource || energySource.energies[energyType] <= 0) return false;
    energySource.energies[energyType] -= 1;
    side.active.energies[energyType] += 1;
    markSelectedAbilityUsed(side, source, ability.name, Boolean(ability.oncePerGame));
    return true;
  }
  if (ability.discardToDraw) {
    if (side.hand.length < ability.discardToDraw.discard) return false;
    const discardHandIndex = payload.discardHandIndex === undefined ? 0 : Number(payload.discardHandIndex);
    if (discardHandIndex < 0 || discardHandIndex >= side.hand.length) return false;
    for (let count = 0; count < ability.discardToDraw.discard; count += 1) {
      const index = Math.min(discardHandIndex, side.hand.length - 1);
      const [discardedCardId] = side.hand.splice(index, 1);
      if (discardedCardId) side.discard.push(discardedCardId);
    }
    drawCards(state, side, ability.discardToDraw.draw);
    markSelectedAbilityUsed(side, source, ability.name, Boolean(ability.oncePerGame));
    return true;
  }
  if (ability.coinFlipDrawOrActiveDamageCounter) {
    return aiUseCoinFlipDrawAbility(state, side, source, rng.next, deps, state.aiDifficulty);
  }
  if (ability.shuffleRandomDiscardIntoDeck) {
    const count = Math.min(ability.shuffleRandomDiscardIntoDeck, side.discard.length);
    if (count <= 0) return false;
    const picked: string[] = [];
    for (let index = 0; index < count; index += 1) {
      const discardIndex = Math.floor(rng.next() * side.discard.length);
      const [cardId] = side.discard.splice(discardIndex, 1);
      if (cardId) picked.push(cardId);
    }
    side.deck.push(...picked);
    markSelectedAbilityUsed(side, source, ability.name, Boolean(ability.oncePerGame));
    return true;
  }
  if (ability.damageOpponent && aiUseDamageAbility(state, side, source, deps)) return true;
  if (ability.moveBenchedEnergyToActive && aiUseMoveBenchedEnergyAbility(state, side, source, state.aiDifficulty, {
    estimateAttackDamageOutput,
    withEnergyShift,
    markAbilityUsed,
  })) return true;
  return false;
}

function readPlayChoices(payload: Record<string, unknown>): PlayChoices {
  const raw = payload.choices;
  if (!raw || typeof raw !== "object") return {};
  const record = raw as Record<string, unknown>;
  const choices: PlayChoices = {};
  if (record.discardHandIndex !== undefined) choices.discardHandIndex = Number(record.discardHandIndex);
  if (record.deckCardIndex !== undefined) choices.deckCardIndex = Number(record.deckCardIndex);
  if (record.umamusumeTargetUid !== undefined) choices.umamusumeTargetUid = Number(record.umamusumeTargetUid);
  if (record.rainbowEvolutionHandIndex !== undefined) choices.rainbowEvolutionHandIndex = Number(record.rainbowEvolutionHandIndex);
  return choices;
}

function markSelectedAbilityUsed(side: SideState, source: UmamusumeInstance, abilityName: string, oncePerGame: boolean): void {
  source.usedAbilityThisTurn = true;
  if (!side.usedAbilityNamesThisTurn.includes(abilityName)) side.usedAbilityNamesThisTurn.push(abilityName);
  if (oncePerGame && !side.usedAbilityNamesThisGame.includes(abilityName)) side.usedAbilityNamesThisGame.push(abilityName);
}

function resolveSelectedCombat(
  state: GameState,
  side: SideState,
  decision: AiCombatDecision | undefined,
  forcedAttackCoinResult: CoinFlipResult | CoinFlipResult[] | undefined,
): void {
  if (!decision || decision.kind === "endTurn") {
    finishTurn(state);
    return;
  }

  if (decision.retreatTargetUid !== undefined) {
    const retreated = aiRetreatToTarget(state, side, decision.retreatTargetUid);
    if (!retreated) return;
    if (decision.kind !== "attack") return;
  }

  performAttack(
    state,
    side.id,
    { refreshContinuousEffects, choosePreferredActiveIndex },
    decision.attackTargetUid,
    decision.healTargetUid,
    forcedAttackCoinResult,
    decision.evolutionDeckCardIndex,
    decision.attackIndex,
    decision.discardHandIndex,
    decision.randomDiscardIndex,
    decision.switchTargetUid,
    decision.useShuffleSelfIntoDeck,
  );
  if (state.pendingPlayerChoice) {
    state.opponentTurnStep = "finish";
    return;
  }
  finishTurn(state);
}

function advanceModeledPhase(state: GameState, sideId: SideId): void {
  const phase = getAiPhase(state, sideId);
  if (phase === "bench") state.opponentTurnStep = "trainerBefore";
  else if (phase === "trainerBefore") state.opponentTurnStep = "evolve";
  else if (phase === "evolve") state.opponentTurnStep = "attach";
  else if (phase === "attach") state.opponentTurnStep = "trainerAfter";
  else if (phase === "trainerAfter") state.opponentTurnStep = "ability";
  else if (phase === "ability") state.opponentTurnStep = "attack";
  else finishTurn(state);
}

function finishTurn(state: GameState): void {
  state.opponentTurnStep = null;
  if (!state.gameOver) {
    endTurn(state, (turnState, sideId) => startTurn(turnState, sideId, refreshContinuousEffects), refreshContinuousEffects);
  }
}

export function setupAiVsAiGame(): GameState {
  let state = createGame(undefined, undefined, "Opponent", "hard", false, "Player AI");
  state.humanBySide.player = false;
  state.humanBySide.opponent = false;
  state = chooseOpeningCoin(state, randomFloat() >= 0.5 ? "heads" : "tails");
  state = dealOpeningHands(state);
  const setup = chooseAiSetupSelection(state);
  if (!setup) throw new Error("Unable to choose AI setup for player.");
  state = completePregameSetup(state, setup.activeIndex, setup.benchIndexes);
  state = autoCompleteOpponentSetup(state);
  for (let tick = 0; tick < 5 && state.phase === "setup"; tick += 1) state = tickSetupCountdown(state);
  if (state.phase !== "play") throw new Error("Headless setup did not enter play phase.");
  return state;
}

export function getForcedAttackCoinResults(state: GameState, rng: Rng): CoinFlipResult | CoinFlipResult[] | undefined {
  if (state.phase !== "play") return undefined;
  if (state.currentSide !== "player" && state.currentSide !== "opponent") return undefined;
  if (state.opponentTurnStep !== "attack") return undefined;
  const side = state.sides[state.currentSide];
  if (!side.active || !canAttack(state, side)) return undefined;
  const attack = getPrimaryAttack(getUmamusumeCard(side.active));
  const flipCount = attack.knockOutActiveIfAllCoinHeads ?? ((attack.coinBonus || attack.drawOnHeads || attack.discardRandomOpponentHandOnHeads) ? 1 : 0);
  if (flipCount <= 0) return undefined;
  const results = Array.from({ length: flipCount }, (_, index): CoinFlipResult => {
    if (index < (side.guaranteedCoinFlipHeads ?? 0)) return "heads";
    return rng.next() >= 0.5 ? "heads" : "tails";
  });
  return results.length === 1 ? results[0] : results;
}

function refreshContinuousEffects(state: GameState): void {
  refreshContinuousHp(state);
  resolveContinuousKnockouts(state);
}

function resolveContinuousKnockouts(state: GameState): void {
  let resolvedKnockout = true;
  while (resolvedKnockout && !state.gameOver) {
    resolvedKnockout = false;
    (["player", "opponent"] as SideId[]).forEach((sideId) => {
      if (resolvedKnockout || state.gameOver) return;
      const side = state.sides[sideId];
      const knockedOut = getAllUmamusume(side).find((umamusume: UmamusumeInstance) => umamusume.hp <= 0);
      if (!knockedOut) return;
      const scoringSideId: SideId = sideId === "player" ? "opponent" : "player";
      resolvedKnockout = knockOutUmamusume(state, scoringSideId, sideId, knockedOut, choosePreferredActiveIndex);
      if (resolvedKnockout) refreshContinuousHp(state);
    });
  }
}

export function stateHash(state: GameState): string {
  return stateFingerprint(state);
}

function summarize(results: GameResult[]) {
  const modelWins = results.filter((result) => result.modelWon).length;
  const completed = results.filter((result) => result.terminalReason === "gameOver").length;
  const totalModelPoints = results.reduce((sum, result) => sum + result.points[result.modelSide], 0);
  const totalHeuristicPoints = results.reduce((sum, result) => sum + result.points[result.modelSide === "player" ? "opponent" : "player"], 0);
  return {
    games: results.length,
    completed,
    modelWins,
    modelWinRate: results.length ? modelWins / results.length : 0,
    averageModelPoints: results.length ? totalModelPoints / results.length : 0,
    averageHeuristicPoints: results.length ? totalHeuristicPoints / results.length : 0,
    heuristicFallbacks: results.reduce((sum, result) => sum + result.heuristicFallbacks, 0),
    averageSelectedCandidateRank: averageSelectedCandidateRank(results),
    byModelSide: {
      player: summarizeSide(results.filter((result) => result.modelSide === "player")),
      opponent: summarizeSide(results.filter((result) => result.modelSide === "opponent")),
    },
  };
}

function summarizeSide(results: GameResult[]) {
  const modelWins = results.filter((result) => result.modelWon).length;
  const totalModelPoints = results.reduce((sum, result) => sum + result.points[result.modelSide], 0);
  const totalHeuristicPoints = results.reduce((sum, result) => sum + result.points[result.modelSide === "player" ? "opponent" : "player"], 0);
  return {
    games: results.length,
    modelWins,
    modelWinRate: results.length ? modelWins / results.length : 0,
    averageModelPoints: results.length ? totalModelPoints / results.length : 0,
    averageHeuristicPoints: results.length ? totalHeuristicPoints / results.length : 0,
    heuristicFallbacks: results.reduce((sum, result) => sum + result.heuristicFallbacks, 0),
    averageSelectedCandidateRank: averageSelectedCandidateRank(results),
  };
}

function averageSelectedCandidateRank(results: GameResult[]): number | null {
  const ranks = results.flatMap((result) => result.selectedCandidateRanks);
  if (ranks.length === 0) return null;
  return ranks.reduce((sum, rank) => sum + rank, 0) / ranks.length;
}

function parseArgs(argv: string[]): EvaluateModelArgs {
  const get = (name: string, fallback: string) => {
    const index = argv.indexOf(name);
    return index >= 0 ? argv[index + 1] ?? fallback : fallback;
  };
  const modelSide = get("--model-side", "both");
  if (modelSide !== "player" && modelSide !== "opponent" && modelSide !== "both") {
    throw new Error("--model-side must be player, opponent, or both");
  }
  return {
    modelUrl: get("--model-url", "http://127.0.0.1:8765"),
    games: Number(get("--games", "24")),
    seedStart: Number(get("--seed-start", "8000")),
    maxSteps: Number(get("--max-steps", "500")),
    modelSide,
    details: argv.includes("--details"),
    selection: parseSelection(get("--selection", "policy")),
    rolloutSteps: Number(get("--rollout-steps", "500")),
    searchDepth: Number(get("--search-depth", "2")),
    searchTopK: Number(get("--search-top-k", "4")),
    searchSamples: Number(get("--search-samples", "1")),
    ranker: parseRanker(get("--ranker", "heuristic")),
    decisionTraceOut: get("--decision-trace-out", ""),
    manifestOut: get("--manifest-out", ""),
    plannerTopK: Number(get("--planner-top-k", get("--search-top-k", "4"))),
    plannerMaxSequences: Number(get("--planner-max-sequences", "64")),
    plannerMaxDepth: Number(get("--planner-max-depth", "8")),
  };
}

function parseSelection(raw: string): EvaluateModelArgs["selection"] {
  if (raw === "baseline" || raw === "value" || raw === "rollout" || raw === "search" || raw === "planner") return raw;
  return "policy";
}

function parseRanker(raw: string): CandidateRankerMode {
  if (raw === "phase-diverse" || raw === "epsilon") return raw;
  return "heuristic";
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
