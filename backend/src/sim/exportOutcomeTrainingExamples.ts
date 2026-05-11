import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import {
  advanceOpponentTurnStep,
  advancePlayerAiTurnStep,
} from "../../../frontend/src/game/engine";
import { ACTION_FEATURE_COUNT, ACTION_FEATURE_SCHEMA_VERSION, enumerateLegalAiActions, chooseHighestScoredAction } from "../../../frontend/src/game/engine/ai-policy/actions";
import { buildPublicObservation } from "../../../frontend/src/game/engine/ai-policy/observation";
import type { LegalAiAction, TrainingExample } from "../../../frontend/src/game/engine/ai-policy/types";
import "./rngAsyncStore";
import { createSeededRng, withRng, type Rng } from "../../../frontend/src/game/engine/core/random";
import { cloneGame } from "../../../frontend/src/game/engine/core/stateClone";
import type { GameState, SideId } from "../../../shared/src/types";
import {
  advanceModeledTurnStep,
  getForcedAttackCoinResults,
  setupAiVsAiGame,
  stateHash,
} from "./evaluateModelVsHeuristic";
import { writeManifestFor } from "./manifest";
import { rankLegalActions, type CandidateRankerMode } from "./candidateRanker";

type Args = {
  out: string;
  seedStart: number;
  games: number;
  maxSteps: number;
  rolloutSteps: number;
  maxExamples: number;
  maxActions: number;
  samples: number;
  tieThreshold: number;
  ranker: CandidateRankerMode;
  candidateOrder: "normal" | "reverse" | "shuffle";
};

type ScoredAction = {
  action: LegalAiAction;
  index: number;
  reward: number;
  rewards: number[];
  variance: number;
  originalRank: number;
};

const args = parseArgs(process.argv.slice(2));
const examples: TrainingExample[] = [];
const metadata = {
  games: 0,
  statesConsidered: 0,
  oracleExamples: 0,
  skippedSingleAction: 0,
  baselineFallbacks: 0,
};

for (let gameIndex = 0; gameIndex < args.games && examples.length < args.maxExamples; gameIndex += 1) {
  const seed = String(args.seedStart + gameIndex);
  const rng = createSeededRng(`collect:${seed}`, "outcome-collect");
  withRng(rng, () => collectGame(seed, rng));
  metadata.games += 1;
}

mkdirSync(dirname(args.out), { recursive: true });
writeFileSync(args.out, examples.map((example) => JSON.stringify(example)).join("\n") + "\n", "utf8");
const manifestOut = writeManifestFor(args.out, {
  artifact: args.out,
  sourceTaxonomy: { trajectorySource: "ai-policy-baseline-visited", labelSource: "rollout-labeled" },
  args,
  examples: examples.length,
  ...metadata,
  phaseCounts: countBy(examples, (example) => example.phase),
  actionKindCounts: countBy(examples, (example) => example.legalActions[example.selectedActionIndex]?.kind ?? "unknown"),
  marginBuckets: countMargins(examples),
  featureSchemas: { observationSchemaVersion: 1, actionFeatureSchemaVersion: ACTION_FEATURE_SCHEMA_VERSION, actionFeatureDimensions: ACTION_FEATURE_COUNT, stateFeatureDimensions: 96 },
});
console.log(JSON.stringify({ out: args.out, manifest: manifestOut, examples: examples.length, ...metadata }, null, 2));

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
    if (stateHash(modeled) === stateHash(before)) {
      metadata.baselineFallbacks += 1;
      state = sideId === "player"
        ? advancePlayerAiTurnStep(state, forcedCoinResults, rng.next)
        : advanceOpponentTurnStep(state, forcedCoinResults, rng.next);
    } else {
      state = modeled;
    }
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
  const candidateActions = orderCandidateActions(legalActions, `${seed}:${step}:candidate-order`);
  const actionPool = rankLegalActions(candidateActions, {
    topK: args.maxActions,
    mode: args.ranker,
    baseline,
    rng: createSeededRng(`${seed}:${step}:ranker`, "outcome-ranker"),
  });

  const sampleSeedIds = Array.from({ length: Math.max(1, args.samples) }, (_, sample) => `${seed}:${step}:s${sample}`);
  const scored = actionPool.map(({ action, index, originalRank }): ScoredAction => {
    const rewards = sampleSeedIds.map((sampleSeed) => withRng(
      createSeededRng(sampleSeed, "outcome-action"),
      () => scoreAction(state, sideId, action, sampleSeed),
    ));
    const reward = mean(rewards);
    return { action, index, reward, rewards, variance: variance(rewards), originalRank };
  }).sort((left, right) => (right.reward - left.reward) || left.action.id.localeCompare(right.action.id));

  const best = scored[0];
  if (!best || !Number.isFinite(best.reward)) return null;
  const baselineReward = scored.find((entry) => entry.action.id === baseline.id)?.reward ?? best.reward;
  const runnerUp = scored.find((entry) => entry.action.id !== best.action.id);
  const margin = best.reward - (runnerUp?.reward ?? best.reward);
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
    policy: "rollout-outcome-v2",
    result: {
      winner: null,
      points: { player: state.sides.player.points, opponent: state.sides.opponent.points },
    },
    valueTarget: Math.max(-1, Math.min(1, best.reward)),
    sampleWeight,
    source: "ai-policy-baseline-visited",
    labelSource: "rollout-labeled",
    oracle: {
      sampleCount: sampleSeedIds.length,
      sampleSeedIds,
      rewardMean: best.reward,
      rewardVariance: best.variance,
      selectedVsRunnerUpMargin: margin,
      selectedVsBaselineMargin: best.reward - baselineReward,
      tiePolicy: "action-id-lexicographic",
      tieThreshold: args.tieThreshold,
      lowMargin: margin < args.tieThreshold,
      selectedOriginalRank: best.originalRank,
      candidateCount: actionPool.length,
      legalActionCount: legalActions.length,
      candidates: scored.map((entry) => ({
        actionId: entry.action.id,
        index: entry.index,
        kind: entry.action.kind,
        originalRank: entry.originalRank,
        rewardMean: entry.reward,
        rewardVariance: entry.variance,
        rewards: entry.rewards,
      })),
    },
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
    samples: Number(get("--samples", "1")),
    tieThreshold: Number(get("--tie-threshold", "0.02")),
    ranker: parseRanker(get("--ranker", "heuristic")),
    candidateOrder: parseCandidateOrder(get("--candidate-order", "normal")),
  };
}

function parseRanker(raw: string): CandidateRankerMode {
  if (raw === "phase-diverse" || raw === "epsilon") return raw;
  return "heuristic";
}

function parseCandidateOrder(raw: string): Args["candidateOrder"] {
  if (raw === "reverse" || raw === "shuffle") return raw;
  return "normal";
}

function orderCandidateActions(actions: LegalAiAction[], seed: string): LegalAiAction[] {
  if (args.candidateOrder === "reverse") return [...actions].reverse();
  if (args.candidateOrder !== "shuffle") return actions;
  const rng = createSeededRng(seed, "outcome-candidate-order");
  const shuffled = [...actions];
  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const swap = Math.floor(rng.next() * (index + 1));
    [shuffled[index], shuffled[swap]] = [shuffled[swap]!, shuffled[index]!];
  }
  return shuffled;
}

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function variance(values: number[]): number {
  if (values.length <= 1) return 0;
  const average = mean(values);
  return values.reduce((sum, value) => sum + (value - average) ** 2, 0) / values.length;
}

function countBy<T>(items: T[], keyOf: (item: T) => string): Record<string, number> {
  return items.reduce<Record<string, number>>((counts, item) => {
    const key = keyOf(item);
    counts[key] = (counts[key] ?? 0) + 1;
    return counts;
  }, {});
}

function countMargins(rows: TrainingExample[]): Record<string, number> {
  const buckets: Record<string, number> = {};
  rows.forEach((row) => {
    const margin = Number((row as TrainingExample & { oracle?: { selectedVsRunnerUpMargin?: number } }).oracle?.selectedVsRunnerUpMargin ?? 0);
    const key = margin < 0.02 ? "<0.02" : margin < 0.1 ? "0.02-0.1" : ">=0.1";
    buckets[key] = (buckets[key] ?? 0) + 1;
  });
  return buckets;
}
