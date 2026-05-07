import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import {
  advanceOpponentTurnStep,
  advancePlayerAiTurnStep,
} from "../../../frontend/src/game/engine";
import { enumerateLegalAiActions, chooseHighestScoredAction } from "../../../frontend/src/game/engine/ai-policy/actions";
import { buildPublicObservation } from "../../../frontend/src/game/engine/ai-policy/observation";
import type { LegalAiAction, TrainingExample } from "../../../frontend/src/game/engine/ai-policy/types";
import { createSeededRng, withRng, type Rng } from "../../../frontend/src/game/engine/core/random";
import { cloneGame } from "../../../frontend/src/game/engine/core/stateClone";
import type { GameState, SideId } from "../../../shared/src/types";
import {
  advanceModeledTurnStep,
  getForcedAttackCoinResults,
  setupAiVsAiGame,
  stateHash,
} from "./evaluateModelVsHeuristic";

type Args = {
  out: string;
  seedStart: number;
  games: number;
  maxSteps: number;
  rolloutSteps: number;
  maxExamples: number;
  maxActions: number;
};

type ScoredAction = {
  action: LegalAiAction;
  index: number;
  reward: number;
};

const args = parseArgs(process.argv.slice(2));
const examples: TrainingExample[] = [];
const metadata = {
  games: 0,
  statesConsidered: 0,
  oracleExamples: 0,
  skippedSingleAction: 0,
};

for (let gameIndex = 0; gameIndex < args.games && examples.length < args.maxExamples; gameIndex += 1) {
  const seed = String(args.seedStart + gameIndex);
  const rng = createSeededRng(`collect:${seed}`, "outcome-collect");
  withRng(rng, () => collectGame(seed, rng));
  metadata.games += 1;
}

mkdirSync(dirname(args.out), { recursive: true });
writeFileSync(args.out, examples.map((example) => JSON.stringify(example)).join("\n") + "\n", "utf8");
console.log(JSON.stringify({ out: args.out, examples: examples.length, ...metadata }, null, 2));

function collectGame(seed: string, rng: Rng): void {
  let state = setupAiVsAiGame();
  for (let step = 0; step < args.maxSteps && examples.length < args.maxExamples; step += 1) {
    if (state.gameOver) break;
    const before = cloneGame(state);
    const sideId = state.currentSide === "player" || state.currentSide === "opponent" ? state.currentSide : "player";
    const legalActions = enumerateLegalAiActions(before, sideId);
    metadata.statesConsidered += 1;
    if (legalActions.length <= 1) {
      metadata.skippedSingleAction += 1;
    } else {
      const example = buildOutcomeExample(before, legalActions, sideId, seed, step);
      if (example) {
        examples.push(example);
        metadata.oracleExamples += 1;
      }
    }

    const forcedCoinResults = getForcedAttackCoinResults(state, rng);
    const baselineAction = chooseHighestScoredAction(legalActions);
    const modeled = advanceModeledTurnStep(state, sideId, baselineAction, forcedCoinResults, rng);
    state = stateHash(modeled) === stateHash(before)
      ? sideId === "player"
        ? advancePlayerAiTurnStep(state, forcedCoinResults, rng.next)
        : advanceOpponentTurnStep(state, forcedCoinResults, rng.next)
      : modeled;
    if (stateHash(state) === stateHash(before)) break;
  }
}

function buildOutcomeExample(
  state: GameState,
  legalActions: LegalAiAction[],
  sideId: SideId,
  seed: string,
  step: number,
): TrainingExample | null {
  const baseline = chooseHighestScoredAction(legalActions);
  const actionPool = legalActions
    .map((action, index) => ({ action, index }))
    .sort((left, right) => (right.action.features[0] ?? 0) - (left.action.features[0] ?? 0))
    .slice(0, args.maxActions);
  if (!actionPool.some((entry) => entry.action.id === baseline.id)) {
    const baselineIndex = legalActions.findIndex((action) => action.id === baseline.id);
    if (baselineIndex >= 0) actionPool.push({ action: baseline, index: baselineIndex });
  }

  const rolloutSeed = `${seed}:${step}`;
  const scored = actionPool.map(({ action, index }): ScoredAction => {
    const reward = withRng(createSeededRng(rolloutSeed, "outcome-action"), () => scoreAction(state, sideId, action, rolloutSeed));
    return { action, index, reward };
  }).sort((left, right) => right.reward - left.reward);

  const best = scored[0];
  if (!best || !Number.isFinite(best.reward)) return null;
  const baselineReward = scored.find((entry) => entry.action.id === baseline.id)?.reward ?? best.reward;
  const margin = best.reward - baselineReward;
  const sampleWeight = 1 + Math.min(4, Math.max(0, margin) * 2);
  return {
    schemaVersion: 1,
    episodeId: `outcome-${seed}`,
    step,
    seed,
    sideId,
    phase: best.action.phase,
    observation: buildPublicObservation(state, sideId),
    legalActions,
    selectedActionId: best.action.id,
    selectedActionIndex: best.index,
    policy: "rollout-outcome-v1",
    result: {
      winner: null,
      points: { player: state.sides.player.points, opponent: state.sides.opponent.points },
    },
    valueTarget: Math.max(-1, Math.min(1, best.reward)),
    sampleWeight,
  } as TrainingExample & { valueTarget: number; sampleWeight: number };
}

function scoreAction(state: GameState, sideId: SideId, action: LegalAiAction, seed: string): number {
  const rng = createSeededRng(seed, "outcome-rollout");
  let next = advanceModeledTurnStep(state, sideId, action, getForcedAttackCoinResults(state, rng), rng);
  if (stateHash(next) === stateHash(state)) return Number.NEGATIVE_INFINITY;
  next = rolloutHeuristic(next, rng);
  return rewardFor(next, sideId);
}

function rolloutHeuristic(state: GameState, rng: Rng): GameState {
  let next = cloneGame(state);
  for (let step = 0; step < args.rolloutSteps; step += 1) {
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

function rewardFor(state: GameState, sideId: SideId): number {
  const opponentId: SideId = sideId === "player" ? "opponent" : "player";
  const pointMargin = (state.sides[sideId].points - state.sides[opponentId].points) / 3;
  if (state.winner === sideId) return 1 + pointMargin * 0.2;
  if (state.winner === opponentId) return -1 + pointMargin * 0.2;
  return pointMargin;
}

function parseArgs(argv: string[]): Args {
  const get = (name: string, fallback: string) => {
    const index = argv.indexOf(name);
    return index >= 0 ? argv[index + 1] ?? fallback : fallback;
  };
  return {
    out: get("--out", "training/runs/outcome/examples.jsonl"),
    seedStart: Number(get("--seed-start", "20000")),
    games: Number(get("--games", "64")),
    maxSteps: Number(get("--max-steps", "500")),
    rolloutSteps: Number(get("--rollout-steps", "500")),
    maxExamples: Number(get("--max-examples", "2500")),
    maxActions: Number(get("--max-actions", "6")),
  };
}
