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
import { getAiRainbowUncapChoice, getAiTrainerChoices } from "../../../frontend/src/game/engine/flow/ai/trainerUtils";
import type { AiCombatDecision } from "../../../frontend/src/game/engine/flow/ai/types";
import { findOwnUmamusumeByUid, getAllUmamusume } from "../../../frontend/src/game/engine/core/umamusume";
import { getUmamusumeAbility } from "../../../frontend/src/game/engine/flow/abilityRules";
import type { CoinFlipResult, GameState, SideId, SideState, UmamusumeInstance } from "../../../shared/src/types";

type Args = {
  modelUrl: string;
  games: number;
  seedStart: number;
  maxSteps: number;
  modelSide: SideId | "both";
  details: boolean;
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
};

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const sides: SideId[] = args.modelSide === "both" ? ["player", "opponent"] : [args.modelSide];
  const results: GameResult[] = [];
  for (const modelSide of sides) {
    for (let index = 0; index < args.games; index += 1) {
      const seed = String(args.seedStart + index);
      results.push(await runModelVsHeuristicGame(args, seed, modelSide));
    }
  }
  const summary = summarize(results);
  console.log(JSON.stringify(args.details ? { summary, results } : { summary }, null, 2));
}

async function runModelVsHeuristicGame(args: Args, seed: string, modelSide: SideId): Promise<GameResult> {
  const rng = createSeededRng(`${seed}:${modelSide}`, "model-vs-heuristic");
  return withRng(rng, () => runModelVsHeuristicGameWithRng(args, seed, modelSide, rng));
}

async function runModelVsHeuristicGameWithRng(args: Args, seed: string, modelSide: SideId, rng: Rng): Promise<GameResult> {
  let state = setupAiVsAiGame();
  let terminalReason: GameResult["terminalReason"] = "maxSteps";
  let modelActions = 0;
  let heuristicFallbacks = 0;

  for (let step = 0; step < args.maxSteps; step += 1) {
    if (state.gameOver) {
      terminalReason = "gameOver";
      break;
    }
    const beforeHash = stateHash(state);
    const sideId = state.currentSide === "player" || state.currentSide === "opponent" ? state.currentSide : "player";
    const forcedCoinResults = getForcedAttackCoinResults(state, rng);
    if (sideId === modelSide) {
      const decision = await chooseModelAction(args.modelUrl, state, sideId);
      modelActions += 1;
      const next = advanceModeledTurnStep(state, sideId, decision.action, forcedCoinResults, rng);
      if (stateHash(next) === beforeHash) {
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

  return {
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
  };
}

async function chooseModelAction(modelUrl: string, state: GameState, sideId: SideId): Promise<{ action: LegalAiAction; selectedIndex: number }> {
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

function advanceModeledTurnStep(
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
    playSelectedHandCard(next, side, Number(action.payload.handIndex ?? -1));
    return next;
  }

  if (action.kind === "evolve") {
    playSelectedHandCard(next, side, Number(action.payload.handIndex ?? -1), Number(action.payload.targetUid));
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
    const used = useSelectedAbility(next, side, Number(action.payload.sourceUid), rng);
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

function playSelectedHandCard(state: GameState, side: SideState, handIndex: number, targetUid?: number): void {
  const cardId = side.hand[handIndex];
  if (!cardId) return;
  const card = getCard(cardId);
  const play = getPlayableAction(state, side, cardId);
  if (!play.canPlay) return;
  const turnGoal = "maximize_progress";
  const trainerChoices = card.kind === "trainer" ? getAiTrainerChoices(state, side, card, handIndex, turnGoal) : {};
  const rainbowChoice = card.kind === "trainer" && card.effect.rainbowUncapCrystal ? getAiRainbowUncapChoice(state, side) : null;
  const choices = {
    ...trainerChoices,
    ...(targetUid !== undefined ? { umamusumeTargetUid: targetUid } : {}),
    ...(rainbowChoice ? { umamusumeTargetUid: rainbowChoice.targetUid, rainbowEvolutionHandIndex: rainbowChoice.evolutionHandIndex } : {}),
  };
  side.hand.splice(handIndex, 1);
  resolveCardPlay(state, side, card, play, adjustHandChoices(choices, handIndex), switchOutOpponentActive);
  normalizeBoardState(state);
  refreshContinuousEffects(state);
}

function useSelectedAbility(state: GameState, side: SideState, sourceUid: number, rng: Rng): boolean {
  const source = getAllUmamusume(side).find((umamusume) => umamusume.uid === sourceUid);
  if (!source || !canUseUmamusumeAbility(state, side, source.uid)) return false;
  const ability = getUmamusumeAbility(state, side.id, source);
  if (!ability) return false;
  const deps = { refreshContinuousEffects, choosePreferredActiveIndex };
  if (ability.damageOpponent && aiUseDamageAbility(state, side, source, deps)) return true;
  if (ability.moveBenchedEnergyToActive && aiUseMoveBenchedEnergyAbility(state, side, source, state.aiDifficulty, {
    estimateAttackDamageOutput,
    withEnergyShift,
    markAbilityUsed,
  })) return true;
  if (ability.coinFlipDrawOrActiveDamageCounter) {
    return aiUseCoinFlipDrawAbility(state, side, source, rng.next, deps, state.aiDifficulty);
  }
  return false;
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

  if (decision.usesCoinFlip && !forcedAttackCoinResult) return;
  performAttack(
    state,
    side.id,
    { refreshContinuousEffects, choosePreferredActiveIndex },
    decision.attackTargetUid,
    decision.healTargetUid,
    forcedAttackCoinResult,
    undefined,
    0,
    undefined,
    undefined,
    undefined,
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

function setupAiVsAiGame(): GameState {
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

function getForcedAttackCoinResults(state: GameState, rng: Rng): CoinFlipResult | CoinFlipResult[] | undefined {
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

function stateHash(state: GameState): string {
  return JSON.stringify({
    phase: state.phase,
    currentSide: state.currentSide,
    step: state.opponentTurnStep,
    pending: state.pendingPlayerChoice,
    turn: state.turnNumber,
    gameOver: state.gameOver,
    winner: state.winner,
    points: { player: state.sides.player.points, opponent: state.sides.opponent.points },
    active: {
      player: state.sides.player.active?.uid ?? null,
      opponent: state.sides.opponent.active?.uid ?? null,
    },
    handSizes: { player: state.sides.player.hand.length, opponent: state.sides.opponent.hand.length },
    deckSizes: { player: state.sides.player.deck.length, opponent: state.sides.opponent.deck.length },
    logHead: state.log[0] ?? null,
  });
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
  };
}

function parseArgs(argv: string[]): Args {
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
  };
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
