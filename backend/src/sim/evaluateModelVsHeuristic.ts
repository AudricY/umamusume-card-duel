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
import { enumerateLegalAiActions, chooseHighestScoredAction, chooseLowestScoredAction } from "../../../frontend/src/game/engine/ai-policy/actions";
import { buildPublicObservation } from "../../../frontend/src/game/engine/ai-policy/observation";
import type { LegalAiAction } from "../../../frontend/src/game/engine/ai-policy/types";
import { getAiPhase } from "../../../frontend/src/game/engine/ai-policy/phase";
import { cloneGame } from "../../../frontend/src/game/engine/core/stateClone";
import "./rngAsyncStore";
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
import type { CoinFlipResult, EnergyCost, EnergyType, GameState, SideId, SideState, UmamusumeInstance } from "../../../shared/src/types";
import { stateFingerprint } from "./stateFingerprint";
import { rankLegalActions, type CandidateRankerMode } from "./candidateRanker";
import { withGitMetadata } from "./manifest";
import { runMcts, defaultMctsConfig, type MctsConfig } from "./mcts";

export type EvaluateModelArgs = {
  modelUrl: string;
  games: number;
  seedStart: number;
  maxSteps: number;
  modelSide: SideId | "both";
  details: boolean;
  selection: "policy" | "baseline" | "inverted-baseline" | "value" | "rollout" | "search" | "planner" | "mcts";
  // R12 day-1: MCTS knobs surface up to the gate via these args. Defaults
  // come from `defaultMctsConfig()`; CLI flags override per run.
  mctsSimulations: number;
  mctsCPuct: number;
  mctsLeaf: "value-head" | "rollout";
  mctsRolloutCrnSamples: number;
  mctsRolloutSteps: number;
  mctsCollapseMaxSteps: number;
  mctsMaxNodes: number;
  mctsPrior: "uniform" | "policy";
  mctsRootDirichlet: boolean;
  mctsDirichletAlpha: number;
  mctsDirichletEpsilon: number;
  // R13.W2 adaptive halting (model side). 0 disables. Typical: 3.0 with
  // adaptiveMinSims around 20–30. Latency drops most on forced/near-forced
  // root decisions which are common (R12 found 62% of root decisions are
  // 1-action; another large fraction has a runaway top action by sim 30).
  mctsAdaptiveRatio: number;
  mctsAdaptiveMinSims: number;
  // Optional JSONL progress stream. When set, the per-game loop writes one
  // line per completed game (game index, side, winner, wall-clock elapsed)
  // so `tail -f` is meaningful while a 200-game gate runs. The TS process
  // also mirrors a short text heartbeat to stderr so dropping `--progress-out`
  // still surfaces liveness without buffer-starving stderr consumers.
  progressOut: string | null;
  cycleWindow: number;
  cycleMinVisits: number;
  plannerCrnSamples: number;
  plannerLeafAggregate: "mean" | "max" | "median";
  plannerFirstActionAggregate: "max" | "mean";
  rolloutCrnSamples: number;
  rolloutSteps: number;
  searchDepth: number;
  searchTopK: number;
  searchSamples: number;
  ranker: CandidateRankerMode;
  decisionTraceOut: string | null;
  manifestOut: string | null;
  traceTeacher: "none" | "rollout" | "search" | "planner";
  plannerTopK: number;
  plannerMaxSequences: number;
  plannerMaxDepth: number;
  // Item 12: when set, the non-model side consults a second served
  // checkpoint instead of advancing through the rule bot. Used by the
  // orchestrator for pool matchup eval. Falls back silently to the rule
  // bot on any error so eval-time robustness is preserved.
  opponentModelUrl: string | null;
  // R13.W4 MCTS strength ladder: the non-model side can also run MCTS
  // with its own config so we can compare MCTS@K1 vs MCTS@K2 etc. Defaults
  // to "rule" (the existing rule-bot path) unless --opponent-selection is
  // set. When "mcts", the opponent uses opponentMcts* settings, falling
  // back to the model's settings for any field not overridden. The
  // opponent's /predict goes to opponentModelUrl if set, else modelUrl.
  opponentSelection: "rule" | "policy" | "mcts";
  opponentMctsSimulations: number;
  opponentMctsCPuct: number;
  opponentMctsLeaf: "value-head" | "rollout";
  opponentMctsRolloutCrnSamples: number;
  opponentMctsRolloutSteps: number;
  opponentMctsCollapseMaxSteps: number;
  opponentMctsMaxNodes: number;
  opponentMctsPrior: "uniform" | "policy";
  opponentMctsAdaptiveRatio: number;
  opponentMctsAdaptiveMinSims: number;
};

type GameResult = {
  seed: string;
  modelSide: SideId;
  winner: SideId | null;
  modelWon: boolean;
  terminalReason: "gameOver" | "maxSteps" | "stalled" | "cycleStalled";
  steps: number;
  turnNumber: number;
  points: Record<SideId, number>;
  modelActions: number;
  heuristicFallbacks: number;
  selectedNoOps: number;
  selectedExplicitPasses: number;
  selectedCandidateRanks: number[];
  decisionTraces: DecisionTraceRow[];
};

type BehaviorPolicySnapshot = {
  kind: string;
  temperature: number;
  actionLogProbs: number[];
  actionProbs?: number[];
  selectedLogProb: number | null;
  valueEstimate?: number;
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
  // Item 18: behavior-policy snapshot at decision time so PPO can
  // recompute importance-sampling ratios on the warm-start rollouts. Only
  // populated when the selection actually consults the model server (policy
  // / value); rule/heuristic/rollout/planner selections leave this field
  // off because there is no parametric behavior policy to log.
  behaviorPolicy?: BehaviorPolicySnapshot;
  teacher?: {
    selection: "rollout" | "search" | "planner";
    selectedActionId: string;
    selectedActionIndex: number;
    selectedOriginalRank?: number;
  };
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
  if (args.progressOut) {
    mkdirSync(dirname(args.progressOut), { recursive: true });
    writeFileSync(args.progressOut, "", "utf8");
  }
  const runStartedAt = Date.now();
  const totalGames = args.games * sides.length;
  let gamesCompleted = 0;
  let modelWinsSoFar = 0;
  for (const modelSide of sides) {
    for (let index = 0; index < args.games; index += 1) {
      const seed = String(args.seedStart + index);
      const gameStart = Date.now();
      const result = await runModelVsHeuristicGame(args, seed, modelSide);
      if (args.decisionTraceOut && result.decisionTraces.length) {
        appendFileSync(args.decisionTraceOut, result.decisionTraces.map((row) => JSON.stringify(row)).join("\n") + "\n", "utf8");
      }
      results.push(result);
      gamesCompleted += 1;
      if (result.modelWon) modelWinsSoFar += 1;
      const elapsedSec = (Date.now() - runStartedAt) / 1000;
      const gameSec = (Date.now() - gameStart) / 1000;
      const runningWr = modelWinsSoFar / gamesCompleted;
      const etaSec = gamesCompleted > 0 ? (elapsedSec / gamesCompleted) * (totalGames - gamesCompleted) : 0;
      if (args.progressOut) {
        const row = {
          event: "game_completed",
          gameIndex: gamesCompleted,
          totalGames,
          seed,
          modelSide,
          winner: result.winner,
          modelWon: result.modelWon,
          turnNumber: result.turnNumber,
          modelActions: result.modelActions,
          heuristicFallbacks: result.heuristicFallbacks,
          terminalReason: result.terminalReason,
          gameElapsedSec: gameSec,
          totalElapsedSec: elapsedSec,
          etaSec,
          runningWinRate: runningWr,
          ts: Date.now() / 1000,
        };
        appendFileSync(args.progressOut, JSON.stringify(row) + "\n", "utf8");
      }
      // stderr heartbeat: short, line-buffered, safe to drop. Useful when
      // the caller did not pass --progress-out.
      process.stderr.write(
        `[eval ${gamesCompleted}/${totalGames}] side=${modelSide} winner=${result.winner ?? "none"} wr=${runningWr.toFixed(3)} game=${gameSec.toFixed(1)}s eta=${(etaSec / 60).toFixed(1)}min\n`,
      );
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
  let selectedNoOps = 0;
  let selectedExplicitPasses = 0;
  const selectedCandidateRanks: number[] = [];
  const decisionTraces: DecisionTraceRow[] = [];
  const recentHashes: string[] = [];
  const cycleWindow = Math.max(0, args.cycleWindow);
  // Require a hash to appear cycleMinVisits times in the window before declaring
  // a cycle stall. Default 3 (= 2 repeats) so transient state aliasing — e.g. a
  // turn-end fingerprint that recurs at a later draw step — does not falsely
  // trip the gate. Reviewer 2 #5.
  const cycleMinVisits = Math.max(2, args.cycleMinVisits);
  const hashVisitCounts = new Map<string, number>();

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
        : args.selection === "inverted-baseline"
          ? chooseInvertedBaselineAction(state, sideId)
        : args.selection === "rollout"
          ? chooseRolloutAction(args, state, sideId, `${seed}:${modelSide}:${step}`, rng)
        : args.selection === "search"
          ? chooseSearchAction(args, state, sideId, `${seed}:${modelSide}:${step}`)
        : args.selection === "planner"
          ? choosePlannerAction(args, state, sideId, `${seed}:${modelSide}:${step}`)
        : args.selection === "mcts"
          ? await chooseMctsAction(args, state, sideId, `${seed}:${modelSide}:${step}:mcts`)
        : await chooseModelAction(args.modelUrl, state, sideId);
      modelActions += 1;
      if (decision.selectedOriginalRank !== undefined) selectedCandidateRanks.push(decision.selectedOriginalRank);
      const next = advanceModeledTurnStep(state, sideId, decision.action, forcedCoinResults, rng);
      const fallback = stateHash(next) === beforeHash;
      if (args.decisionTraceOut) {
        const teacher = chooseTraceTeacher(args, state, sideId, `${seed}:${modelSide}:${step}:trace-teacher`);
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
        if (teacher) trace.teacher = teacher;
        const behavior = (decision as { behavior?: BehaviorPolicySnapshot }).behavior;
        if (behavior) trace.behaviorPolicy = behavior;
        decisionTraces.push(trace);
      }
      if (fallback) {
        heuristicFallbacks += 1;
        if (decision.action.kind !== "pass") selectedNoOps += 1;
        state = sideId === "player"
          ? advancePlayerAiTurnStep(state, forcedCoinResults, rng.next)
          : advanceOpponentTurnStep(state, forcedCoinResults, rng.next);
      } else {
        if (decision.action.kind === "pass") selectedExplicitPasses += 1;
        state = next;
      }
    } else {
      // Item 12 (orig) + R13.W4: the non-model side may delegate to a served
      // checkpoint (one-step policy) or run its own MCTS for the ladder. On
      // any error or single-action shortcut we fall back to the rule-bot
      // advance so the evaluator never crashes mid-game.
      let advanced = false;
      if (args.opponentSelection === "mcts") {
        const opponentDecision = await chooseOpponentMctsAction(args, state, sideId, `${seed}:${modelSide}:${step}:opp-mcts`);
        if (opponentDecision) {
          const next = advanceModeledTurnStep(state, sideId, opponentDecision.action, forcedCoinResults, rng);
          if (stateHash(next) !== beforeHash) {
            state = next;
            advanced = true;
          }
        }
      } else if (args.opponentSelection === "policy" || args.opponentModelUrl) {
        try {
          const url = args.opponentModelUrl ?? args.modelUrl;
          const opponentDecision = await chooseOpponentModelAction(url, state, sideId);
          if (opponentDecision) {
            const next = advanceModeledTurnStep(state, sideId, opponentDecision.action, forcedCoinResults, rng);
            if (stateHash(next) !== beforeHash) {
              state = next;
              advanced = true;
            }
          }
        } catch {
          advanced = false;
        }
      }
      if (!advanced) {
        state = sideId === "player"
          ? advancePlayerAiTurnStep(state, forcedCoinResults, rng.next)
          : advanceOpponentTurnStep(state, forcedCoinResults, rng.next);
      }
    }
    const afterHash = stateHash(state);
    if (afterHash === beforeHash) {
      terminalReason = "stalled";
      break;
    }
    if (cycleWindow > 0) {
      const nextCount = (hashVisitCounts.get(afterHash) ?? 0) + 1;
      hashVisitCounts.set(afterHash, nextCount);
      recentHashes.push(afterHash);
      if (recentHashes.length > cycleWindow) {
        const evicted = recentHashes.shift()!;
        const remaining = (hashVisitCounts.get(evicted) ?? 0) - 1;
        if (remaining <= 0) hashVisitCounts.delete(evicted);
        else hashVisitCounts.set(evicted, remaining);
      }
      if (nextCount >= cycleMinVisits) {
        terminalReason = "cycleStalled";
        break;
      }
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
    selectedNoOps,
    selectedExplicitPasses,
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

function chooseTraceTeacher(
  args: EvaluateModelArgs,
  state: GameState,
  sideId: SideId,
  seed: string,
): DecisionTraceRow["teacher"] | undefined {
  if (args.traceTeacher === "none") return undefined;
  const decision = args.traceTeacher === "rollout"
    ? chooseRolloutAction(args, state, sideId, seed, createSeededRng(seed, "trace-teacher-rollout"))
    : args.traceTeacher === "search"
      ? chooseSearchAction(args, state, sideId, seed)
      : choosePlannerAction(args, state, sideId, seed);
  const teacher: DecisionTraceRow["teacher"] = {
    selection: args.traceTeacher,
    selectedActionId: decision.action.id,
    selectedActionIndex: decision.selectedIndex,
  };
  if (decision.selectedOriginalRank !== undefined) teacher.selectedOriginalRank = decision.selectedOriginalRank;
  return teacher;
}

async function chooseOpponentModelAction(
  opponentModelUrl: string,
  state: GameState,
  sideId: SideId,
): Promise<{ action: LegalAiAction; selectedIndex: number } | null> {
  // Mirrors chooseModelAction's transport but is failure-tolerant: any
  // malformed response yields null so the caller can fall back to the
  // rule bot. The opponent side only needs an action; behavior-policy
  // snapshots are deliberately not threaded through (these games do not
  // produce decision traces for the non-model side).
  const legalActions = enumerateLegalAiActions(state, sideId);
  if (legalActions.length <= 1) return null;
  let response: Response;
  try {
    response = await fetch(`${opponentModelUrl.replace(/\/$/, "")}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        observation: buildPublicObservation(state, sideId),
        legalActions,
      }),
    });
  } catch {
    return null;
  }
  if (!response.ok) return null;
  let payload: { selectedIndex?: number[] };
  try {
    payload = await response.json() as { selectedIndex?: number[] };
  } catch {
    return null;
  }
  const rawIndex = Number(payload.selectedIndex?.[0] ?? NaN);
  if (!Number.isFinite(rawIndex)) return null;
  const selectedIndex = Math.max(0, Math.min(legalActions.length - 1, rawIndex));
  const action = legalActions[selectedIndex];
  if (!action) return null;
  return { action, selectedIndex };
}

async function chooseModelAction(modelUrl: string, state: GameState, sideId: SideId): Promise<{ action: LegalAiAction; selectedIndex: number; selectedOriginalRank?: number; behavior?: BehaviorPolicySnapshot }> {
  const legalActions = enumerateLegalAiActions(state, sideId);
  if (legalActions.length <= 1) {
    // Single-action shortcut still emits a degenerate behavior snapshot so
    // PPO importance ratios are well-defined for every policy-source row,
    // not only the multi-candidate ones. log P(forced action) = 0.
    const action = legalActions[0] ?? chooseHighestScoredAction(legalActions);
    const behavior: BehaviorPolicySnapshot = {
      kind: "single-action",
      temperature: 0,
      actionLogProbs: legalActions.length === 1 ? [0] : [],
      actionProbs: legalActions.length === 1 ? [1] : [],
      selectedLogProb: legalActions.length === 1 ? 0 : null,
    };
    return { action, selectedIndex: 0, behavior };
  }
  const response = await fetch(`${modelUrl.replace(/\/$/, "")}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      observation: buildPublicObservation(state, sideId),
      legalActions,
    }),
  });
  if (!response.ok) throw new Error(`Model server returned ${response.status}: ${await response.text()}`);
  const payload = await response.json() as {
    selectedIndex?: number[];
    actionLogProbs?: number[][];
    actionProbs?: number[][];
    selectedLogProb?: number[];
    value?: number[];
    behaviorPolicy?: { kind?: string; temperature?: number };
  };
  const selectedIndex = Math.max(0, Math.min(legalActions.length - 1, Number(payload.selectedIndex?.[0] ?? 0)));
  // Item 18: capture behavior-policy log-probabilities so PPO can recover
  // importance-sampling ratios from the warm-start rollouts. The serving
  // policy is greedy, so under the warm-start checkpoint the chosen-action
  // log-prob collapses to 0 (probability 1). Once a stochastic serving mode
  // lands the same plumbing carries the non-trivial distribution through.
  let behavior: BehaviorPolicySnapshot | undefined;
  if (payload.actionLogProbs?.[0]) {
    const snapshot: BehaviorPolicySnapshot = {
      kind: payload.behaviorPolicy?.kind ?? "greedy",
      temperature: typeof payload.behaviorPolicy?.temperature === "number" ? payload.behaviorPolicy.temperature : 0,
      actionLogProbs: payload.actionLogProbs[0].slice(0, legalActions.length),
      selectedLogProb: typeof payload.selectedLogProb?.[0] === "number" ? payload.selectedLogProb[0] : null,
    };
    if (payload.actionProbs?.[0]) snapshot.actionProbs = payload.actionProbs[0].slice(0, legalActions.length);
    if (typeof payload.value?.[0] === "number") snapshot.valueEstimate = payload.value[0];
    behavior = snapshot;
  }
  const result: { action: LegalAiAction; selectedIndex: number; behavior?: BehaviorPolicySnapshot } = {
    action: legalActions[selectedIndex] ?? legalActions[0]!,
    selectedIndex,
  };
  if (behavior) result.behavior = behavior;
  return result;
}

function chooseBaselineAction(state: GameState, sideId: SideId): { action: LegalAiAction; selectedIndex: number; selectedOriginalRank?: number } {
  const legalActions = enumerateLegalAiActions(state, sideId);
  const selected = chooseHighestScoredAction(legalActions);
  const selectedIndex = Math.max(0, legalActions.findIndex((action) => action.id === selected.id));
  return rankedDecision(legalActions, selectedIndex);
}

function chooseInvertedBaselineAction(state: GameState, sideId: SideId): { action: LegalAiAction; selectedIndex: number; selectedOriginalRank?: number } {
  const legalActions = enumerateLegalAiActions(state, sideId);
  const selected = chooseLowestScoredAction(legalActions);
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

async function chooseMctsAction(
  args: EvaluateModelArgs,
  state: GameState,
  sideId: SideId,
  seed: string,
): Promise<{ action: LegalAiAction; selectedIndex: number; selectedOriginalRank?: number }> {
  return runMctsForSide(state, sideId, seed, args.modelUrl, {
    simulations: args.mctsSimulations,
    cPuct: args.mctsCPuct,
    leaf: args.mctsLeaf,
    rolloutCrnSamples: args.mctsRolloutCrnSamples,
    rolloutSteps: args.mctsRolloutSteps,
    prior: args.mctsPrior,
    addRootDirichlet: args.mctsRootDirichlet,
    dirichletAlpha: args.mctsDirichletAlpha,
    dirichletEpsilon: args.mctsDirichletEpsilon,
    collapseMaxSteps: args.mctsCollapseMaxSteps,
    maxNodes: args.mctsMaxNodes,
    adaptiveRatio: args.mctsAdaptiveRatio,
    adaptiveMinSims: args.mctsAdaptiveMinSims,
  });
}

async function chooseOpponentMctsAction(
  args: EvaluateModelArgs,
  state: GameState,
  sideId: SideId,
  seed: string,
): Promise<{ action: LegalAiAction; selectedIndex: number } | null> {
  // The opponent's /predict goes to opponentModelUrl when set; otherwise it
  // shares the model's serve_onnx (typical for self-vs-self ladder runs).
  // Wrapped in try/catch so the caller can fall back to the rule bot on
  // any transport error, matching chooseOpponentModelAction's contract.
  try {
    const url = args.opponentModelUrl ?? args.modelUrl;
    const decision = await runMctsForSide(state, sideId, seed, url, {
      simulations: args.opponentMctsSimulations,
      cPuct: args.opponentMctsCPuct,
      leaf: args.opponentMctsLeaf,
      rolloutCrnSamples: args.opponentMctsRolloutCrnSamples,
      rolloutSteps: args.opponentMctsRolloutSteps,
      prior: args.opponentMctsPrior,
      addRootDirichlet: false,
      dirichletAlpha: args.mctsDirichletAlpha,
      dirichletEpsilon: args.mctsDirichletEpsilon,
      collapseMaxSteps: args.opponentMctsCollapseMaxSteps,
      maxNodes: args.opponentMctsMaxNodes,
      adaptiveRatio: args.opponentMctsAdaptiveRatio,
      adaptiveMinSims: args.opponentMctsAdaptiveMinSims,
    });
    return { action: decision.action, selectedIndex: decision.selectedIndex };
  } catch {
    return null;
  }
}

type RunMctsForSideOverrides = {
  simulations: number;
  cPuct: number;
  leaf: "value-head" | "rollout";
  rolloutCrnSamples: number;
  rolloutSteps: number;
  prior: "uniform" | "policy";
  addRootDirichlet: boolean;
  dirichletAlpha: number;
  dirichletEpsilon: number;
  collapseMaxSteps: number;
  maxNodes: number;
  adaptiveRatio: number;
  adaptiveMinSims: number;
};

async function runMctsForSide(
  state: GameState,
  sideId: SideId,
  seed: string,
  modelUrl: string,
  overrides: RunMctsForSideOverrides,
): Promise<{ action: LegalAiAction; selectedIndex: number; selectedOriginalRank?: number }> {
  const legalActions = enumerateLegalAiActions(state, sideId);
  if (legalActions.length <= 1) {
    return { action: legalActions[0] ?? chooseHighestScoredAction(legalActions), selectedIndex: 0 };
  }
  const config: MctsConfig = defaultMctsConfig({
    simulations: Math.max(1, overrides.simulations),
    cPuct: overrides.cPuct,
    leaf: overrides.leaf,
    rolloutCrnSamples: Math.max(1, overrides.rolloutCrnSamples),
    rolloutSteps: Math.max(1, overrides.rolloutSteps),
    prior: overrides.prior,
    addRootDirichlet: overrides.addRootDirichlet,
    dirichletAlpha: overrides.dirichletAlpha,
    dirichletEpsilon: overrides.dirichletEpsilon,
    collapseMaxSteps: Math.max(1, overrides.collapseMaxSteps),
    maxNodes: Math.max(64, overrides.maxNodes),
    adaptiveRatio: Math.max(0, overrides.adaptiveRatio),
    adaptiveMinSims: Math.max(1, overrides.adaptiveMinSims),
  });
  const result = await runMcts(state, sideId, config, modelUrl, seed);
  const selectedIndex = Math.min(Math.max(0, result.selectedIndex), legalActions.length - 1);
  return rankedDecision(legalActions, selectedIndex);
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

function chooseRolloutAction(args: EvaluateModelArgs, state: GameState, sideId: SideId, seed: string, fallbackRng: Rng): { action: LegalAiAction; selectedIndex: number; selectedOriginalRank?: number } {
  const legalActions = enumerateLegalAiActions(state, sideId);
  if (legalActions.length <= 1) return { action: legalActions[0] ?? chooseHighestScoredAction(legalActions), selectedIndex: 0 };
  const samples = Math.max(1, args.rolloutCrnSamples);
  // CRN: every candidate is scored against the same K shared seeds. The
  // pairing reduces between-candidate variance — differences in returns
  // come from the choice, not from RNG drift between calls.
  const sharedSeeds = Array.from({ length: samples }, (_, i) => `${seed}:rollout-crn-${i}`);
  let bestIndex = 0;
  let bestReward = Number.NEGATIVE_INFINITY;
  legalActions.forEach((action, index) => {
    const rewards = sharedSeeds.map((sharedSeed) => {
      const advanceRng = createSeededRng(`${sharedSeed}:advance:${index}`, "rollout-crn-advance");
      const next = advanceModeledTurnStep(state, sideId, action, getForcedAttackCoinResults(state, advanceRng), advanceRng);
      if (stateHash(next) === stateHash(state)) return Number.NEGATIVE_INFINITY;
      const rolloutRng = createSeededRng(`${sharedSeed}:rollout`, "rollout-crn-rollout");
      return rewardForRollout(rolloutHeuristic(next, rolloutRng, args.rolloutSteps), sideId);
    });
    const allInvalid = rewards.every((r) => !Number.isFinite(r));
    const reward = allInvalid
      ? Number.NEGATIVE_INFINITY
      : rewards.filter((r) => Number.isFinite(r)).reduce((sum, r) => sum + r, 0) / rewards.filter((r) => Number.isFinite(r)).length;
    if (reward > bestReward) {
      bestReward = reward;
      bestIndex = index;
    }
  });
  if (bestReward === Number.NEGATIVE_INFINITY) {
    // No CRN sample produced a state change; fall back to the original
    // single-rollout path on the live rng so we still pick the best option
    // observed under the running game seed.
    legalActions.forEach((action, index) => {
      const next = advanceModeledTurnStep(state, sideId, action, getForcedAttackCoinResults(state, fallbackRng), fallbackRng);
      const reward = stateHash(next) === stateHash(state)
        ? Number.NEGATIVE_INFINITY
        : rewardForRollout(rolloutHeuristic(next, fallbackRng, args.rolloutSteps), sideId);
      if (reward > bestReward) {
        bestReward = reward;
        bestIndex = index;
      }
    });
  }
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
  if (bundles.length === 0) return rankedDecision(legalActions, 0);
  const samples = Math.max(1, args.plannerCrnSamples);
  // CRN: every bundle is scored against the same K shared rollout seeds. The
  // pairing reduces between-bundle variance — differences in returns now come
  // from the choices, not from RNG.
  const sharedSeeds = Array.from({ length: samples }, (_, i) => `${seed}:crn-seed-${i}`);
  const bundleScores = bundles.map((bundle, bundleIndex) => {
    const rewards = sharedSeeds.map((rolloutSeed) => {
      const rng = createSeededRng(rolloutSeed, "planner-leaf-crn");
      return rewardForRollout(rolloutHeuristic(bundle.state, rng, args.rolloutSteps), sideId);
    });
    return { bundle, bundleIndex, rewards, leafScore: aggregateScores(rewards, args.plannerLeafAggregate) };
  });

  // Group bundles by their first action; pick the first action whose grouped
  // score is best. This avoids letting one lucky deep continuation overrule
  // the actual first-action question.
  const grouped = new Map<string, { first: TurnBundle["first"]; scores: number[] }>();
  bundleScores.forEach(({ bundle, leafScore }) => {
    const key = bundle.first.action.id;
    const entry = grouped.get(key);
    if (entry) entry.scores.push(leafScore);
    else grouped.set(key, { first: bundle.first, scores: [leafScore] });
  });

  let bestKey: string | null = null;
  let bestScore = Number.NEGATIVE_INFINITY;
  grouped.forEach((entry, key) => {
    const aggregated = aggregateScores(entry.scores, args.plannerFirstActionAggregate);
    if (aggregated > bestScore) {
      bestScore = aggregated;
      bestKey = key;
    }
  });
  if (bestKey === null) return bundles[0]?.first ?? rankedDecision(legalActions, 0);
  return grouped.get(bestKey)!.first;
}

function aggregateScores(scores: number[], mode: "mean" | "max" | "median"): number {
  if (scores.length === 0) return Number.NEGATIVE_INFINITY;
  if (mode === "max") return Math.max(...scores);
  if (mode === "median") {
    const sorted = [...scores].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ? (sorted[mid - 1]! + sorted[mid]!) / 2 : sorted[mid]!;
  }
  return scores.reduce((sum, value) => sum + value, 0) / scores.length;
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
  if (!side.active || side.active.specialConditions.includes("paralysed")) return undefined;
  if (side.active.attackBlockedUntilOwnTurn === state.turnsTakenBySide[side.id]) return undefined;
  const flipCount = Math.max(0, ...getUmamusumeCard(side.active).attacks
    .filter((attack) => hasEnoughEnergyForAttack(side.active!, attack.cost))
    .map((attack) => attack.knockOutActiveIfAllCoinHeads ?? ((attack.coinBonus || attack.drawOnHeads || attack.discardRandomOpponentHandOnHeads) ? 1 : 0)));
  if (flipCount <= 0) return undefined;
  const results = Array.from({ length: flipCount }, (_, index): CoinFlipResult => {
    if (index < (side.guaranteedCoinFlipHeads ?? 0)) return "heads";
    return rng.next() >= 0.5 ? "heads" : "tails";
  });
  return results.length === 1 ? results[0] : results;
}

function hasEnoughEnergyForAttack(umamusume: UmamusumeInstance, cost: EnergyCost): boolean {
  const totalAttached = Object.values(umamusume.energies).reduce((sum, value) => sum + value, 0);
  const typedRequired = Object.entries(cost)
    .filter(([type]) => type !== "colorless")
    .reduce((sum, [type, amount]) => sum + Math.max(0, (amount ?? 0) - umamusume.energies[type as keyof typeof umamusume.energies]), 0);
  const totalRequired = Object.values(cost).reduce((sum, amount) => sum + (amount ?? 0), 0);
  return typedRequired === 0 && totalAttached >= totalRequired;
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
    selectedNoOps: results.reduce((sum, result) => sum + result.selectedNoOps, 0),
    selectedExplicitPasses: results.reduce((sum, result) => sum + result.selectedExplicitPasses, 0),
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
    selectedNoOps: results.reduce((sum, result) => sum + result.selectedNoOps, 0),
    selectedExplicitPasses: results.reduce((sum, result) => sum + result.selectedExplicitPasses, 0),
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
    traceTeacher: parseTraceTeacher(get("--trace-teacher", "none")),
    plannerTopK: Number(get("--planner-top-k", get("--search-top-k", "4"))),
    plannerMaxSequences: Number(get("--planner-max-sequences", "64")),
    plannerMaxDepth: Number(get("--planner-max-depth", "8")),
    cycleWindow: Number(get("--cycle-window", "8")),
    cycleMinVisits: Number(get("--cycle-min-visits", "3")),
    plannerCrnSamples: Number(get("--planner-crn-samples", "3")),
    plannerLeafAggregate: parseAggregate(get("--planner-leaf-aggregate", "mean")),
    plannerFirstActionAggregate: parseFirstActionAggregate(get("--planner-first-action-aggregate", "max")),
    rolloutCrnSamples: Number(get("--rollout-crn-samples", "1")),
    opponentModelUrl: get("--opponent-model-url", "") || null,
    mctsSimulations: Number(get("--mcts-simulations", "100")),
    mctsCPuct: Number(get("--mcts-c-puct", "1.5")),
    mctsLeaf: parseMctsLeaf(get("--mcts-leaf", "value-head")),
    mctsRolloutCrnSamples: Number(get("--mcts-rollout-crn-samples", "3")),
    mctsRolloutSteps: Number(get("--mcts-rollout-steps", "200")),
    mctsCollapseMaxSteps: Number(get("--mcts-collapse-max-steps", "64")),
    mctsMaxNodes: Number(get("--mcts-max-nodes", "5000")),
    mctsPrior: parseMctsPrior(get("--mcts-prior", "uniform")),
    mctsRootDirichlet: argv.includes("--mcts-root-dirichlet"),
    mctsDirichletAlpha: Number(get("--mcts-dirichlet-alpha", "0.3")),
    mctsDirichletEpsilon: Number(get("--mcts-dirichlet-epsilon", "0.25")),
    mctsAdaptiveRatio: Number(get("--mcts-adaptive-ratio", "0")),
    mctsAdaptiveMinSims: Number(get("--mcts-adaptive-min-sims", "20")),
    progressOut: get("--progress-out", "") || null,
    opponentSelection: parseOpponentSelection(get("--opponent-selection", "rule")),
    opponentMctsSimulations: Number(get("--opponent-mcts-simulations", get("--mcts-simulations", "100"))),
    opponentMctsCPuct: Number(get("--opponent-mcts-c-puct", get("--mcts-c-puct", "1.5"))),
    opponentMctsLeaf: parseMctsLeaf(get("--opponent-mcts-leaf", get("--mcts-leaf", "value-head"))),
    opponentMctsRolloutCrnSamples: Number(get("--opponent-mcts-rollout-crn-samples", get("--mcts-rollout-crn-samples", "3"))),
    opponentMctsRolloutSteps: Number(get("--opponent-mcts-rollout-steps", get("--mcts-rollout-steps", "200"))),
    opponentMctsCollapseMaxSteps: Number(get("--opponent-mcts-collapse-max-steps", get("--mcts-collapse-max-steps", "64"))),
    opponentMctsMaxNodes: Number(get("--opponent-mcts-max-nodes", get("--mcts-max-nodes", "5000"))),
    opponentMctsPrior: parseMctsPrior(get("--opponent-mcts-prior", get("--mcts-prior", "uniform"))),
    opponentMctsAdaptiveRatio: Number(get("--opponent-mcts-adaptive-ratio", get("--mcts-adaptive-ratio", "0"))),
    opponentMctsAdaptiveMinSims: Number(get("--opponent-mcts-adaptive-min-sims", get("--mcts-adaptive-min-sims", "20"))),
  };
}

function parseOpponentSelection(raw: string): EvaluateModelArgs["opponentSelection"] {
  if (raw === "policy" || raw === "mcts") return raw;
  return "rule";
}

function parseMctsPrior(raw: string): "uniform" | "policy" {
  if (raw === "uniform" || raw === "policy") return raw;
  throw new Error(`--mcts-prior must be uniform or policy, got ${raw}`);
}

function parseMctsLeaf(raw: string): "value-head" | "rollout" {
  if (raw === "value-head" || raw === "rollout") return raw;
  throw new Error(`--mcts-leaf must be value-head or rollout, got ${raw}`);
}

function parseAggregate(raw: string): "mean" | "max" | "median" {
  if (raw === "max" || raw === "median") return raw;
  return "mean";
}

function parseFirstActionAggregate(raw: string): "max" | "mean" {
  if (raw === "mean") return raw;
  return "max";
}

function parseSelection(raw: string): EvaluateModelArgs["selection"] {
  if (raw === "baseline" || raw === "inverted-baseline" || raw === "value" || raw === "rollout" || raw === "search" || raw === "planner" || raw === "mcts") return raw;
  return "policy";
}

function parseRanker(raw: string): CandidateRankerMode {
  if (raw === "phase-diverse" || raw === "epsilon") return raw;
  return "heuristic";
}

function parseTraceTeacher(raw: string): EvaluateModelArgs["traceTeacher"] {
  if (raw === "rollout" || raw === "search" || raw === "planner") return raw;
  return "none";
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
