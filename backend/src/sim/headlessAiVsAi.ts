import {
  advanceOpponentTurnStep,
  advancePlayerAiTurnStep,
  autoCompleteOpponentSetup,
  chooseOpeningCoin,
  completePregameSetup,
  createGame,
  dealOpeningHands,
  getPrimaryAttack,
  getUmamusumeCard,
  tickSetupCountdown,
} from "../../../frontend/src/game/engine";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { createSeededRng, randomFloat, withRng, type Rng } from "../../../frontend/src/game/engine/core/random";
import { chooseAiSetupSelection } from "../../../frontend/src/app/gameUiHelpers";
import type { CoinFlipResult, EnergyCost, GameState, SideId, UmamusumeInstance } from "../../../shared/src/types";
import { enumerateLegalAiActions, chooseHighestScoredAction } from "../../../frontend/src/game/engine/ai-policy/actions";
import { buildPublicObservation } from "../../../frontend/src/game/engine/ai-policy/observation";
import type { TrainingExample } from "../../../frontend/src/game/engine/ai-policy/types";
import { stateFingerprint } from "./stateFingerprint";

export type HeadlessRunOptions = {
  seed: string;
  maxSteps: number;
  episodeId?: string;
  collectExamples?: boolean;
};

export type HeadlessRunResult = {
  seed: string;
  episodeId: string;
  steps: number;
  turnNumber: number;
  winner: SideId | null;
  points: Record<SideId, number>;
  terminalReason: "gameOver" | "maxSteps" | "stalled";
  examples: TrainingExample[];
  finalLog: string[];
};

export function runHeadlessAiVsAi(options: HeadlessRunOptions): HeadlessRunResult {
  const rng = createSeededRng(options.seed, "headless");
  return withRng(rng, () => runHeadlessAiVsAiWithActiveRng(options, rng));
}

function runHeadlessAiVsAiWithActiveRng(options: HeadlessRunOptions, rng: Rng): HeadlessRunResult {
  const episodeId = options.episodeId ?? `seed-${options.seed}`;
  let state = setupAiVsAiGame();
  const examples: TrainingExample[] = [];
  let terminalReason: HeadlessRunResult["terminalReason"] = "maxSteps";

  for (let step = 0; step < options.maxSteps; step += 1) {
    if (state.gameOver) {
      terminalReason = "gameOver";
      break;
    }
    const beforeHash = stateHash(state);
    const sideId = state.currentSide === "player" || state.currentSide === "opponent" ? state.currentSide : "player";
    if (options.collectExamples !== false) {
      examples.push(makeTrainingExample(state, sideId, episodeId, options.seed, step));
    }

    state = advanceOneAiStep(state, rng);
    if (stateHash(state) === beforeHash) {
      terminalReason = "stalled";
      break;
    }
    if (step === options.maxSteps - 1 && state.gameOver) terminalReason = "gameOver";
  }

  const points = { player: state.sides.player.points, opponent: state.sides.opponent.points };
  examples.forEach((example) => {
    example.result.winner = state.winner;
    example.result.points = points;
  });

  return {
    seed: options.seed,
    episodeId,
    steps: examples.length,
    turnNumber: state.turnNumber,
    winner: state.winner,
    points,
    terminalReason,
    examples,
    finalLog: state.log.slice(0, 12),
  };
}

export function runHeadlessBatch(seeds: string[], maxSteps: number): HeadlessRunResult[] {
  return seeds.map((seed) => runHeadlessAiVsAi({ seed, maxSteps, collectExamples: true }));
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
  for (let tick = 0; tick < 5 && state.phase === "setup"; tick += 1) {
    state = tickSetupCountdown(state);
  }
  if (state.phase !== "play") throw new Error("Headless setup did not enter play phase.");
  return state;
}

function advanceOneAiStep(state: GameState, rng: Rng): GameState {
  const forcedCoinResults = getForcedAttackCoinResults(state, rng);
  if (state.currentSide === "player") return advancePlayerAiTurnStep(state, forcedCoinResults, rng.next);
  if (state.currentSide === "opponent") return advanceOpponentTurnStep(state, forcedCoinResults, rng.next);
  return state;
}

function getForcedAttackCoinResults(state: GameState, rng: Rng): CoinFlipResult | CoinFlipResult[] | undefined {
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

function hasEnoughEnergyForAttack(
  umamusume: UmamusumeInstance,
  cost: EnergyCost,
): boolean {
  const totalAttached = Object.values(umamusume.energies).reduce((sum, value) => sum + value, 0);
  const typedRequired = Object.entries(cost)
    .filter(([type]) => type !== "colorless")
    .reduce((sum, [type, amount]) => sum + Math.max(0, (amount ?? 0) - umamusume.energies[type as keyof typeof umamusume.energies]), 0);
  const totalRequired = Object.values(cost).reduce((sum, amount) => sum + (amount ?? 0), 0);
  return typedRequired === 0 && totalAttached >= totalRequired;
}

function makeTrainingExample(state: GameState, sideId: SideId, episodeId: string, seed: string, step: number): TrainingExample {
  const legalActions = enumerateLegalAiActions(state, sideId);
  const selected = chooseHighestScoredAction(legalActions);
  const selectedActionIndex = Math.max(0, legalActions.findIndex((action) => action.id === selected.id));
  return {
    schemaVersion: 1,
    episodeId,
    step,
    seed,
    sideId,
    phase: selected.phase,
    observation: buildPublicObservation(state, sideId),
    legalActions,
    selectedActionId: selected.id,
    selectedActionIndex,
    policy: "heuristic-candidate-v1",
    result: {
      winner: null,
      points: { player: state.sides.player.points, opponent: state.sides.opponent.points },
    },
  };
}

function stateHash(state: GameState): string {
  return stateFingerprint(state);
}

function parseArgs(argv: string[]): HeadlessRunOptions {
  const get = (name: string, fallback: string) => {
    const index = argv.indexOf(name);
    return index >= 0 ? argv[index + 1] ?? fallback : fallback;
  };
  return {
    seed: get("--seed", "1"),
    maxSteps: Number(get("--max-steps", "300")),
    collectExamples: !argv.includes("--no-examples"),
  };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const argv = process.argv.slice(2);
  const result = runHeadlessAiVsAi(parseArgs(argv));
  const examplesOut = getArg(argv, "--examples-out");
  if (examplesOut) {
    mkdirSync(dirname(examplesOut), { recursive: true });
    writeFileSync(examplesOut, result.examples.map((example) => JSON.stringify(example)).join("\n") + "\n", "utf8");
  }
  const output = {
    seed: result.seed,
    episodeId: result.episodeId,
    terminalReason: result.terminalReason,
    winner: result.winner,
    turns: result.turnNumber,
    points: result.points,
    steps: result.steps,
    examples: result.examples.length,
    finalLog: result.finalLog,
  };
  console.log(JSON.stringify(output, null, 2));
}

function getArg(argv: string[], name: string): string | null {
  const index = argv.indexOf(name);
  return index >= 0 ? argv[index + 1] ?? null : null;
}
