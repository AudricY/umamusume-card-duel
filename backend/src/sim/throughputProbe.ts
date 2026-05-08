import { performance } from "node:perf_hooks";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import {
  advanceOpponentTurnStep,
  advancePlayerAiTurnStep,
} from "../../../frontend/src/game/engine";
import {
  chooseHighestScoredAction,
  enumerateLegalAiActions,
} from "../../../frontend/src/game/engine/ai-policy/actions";
import { cloneGame } from "../../../frontend/src/game/engine/core/stateClone";
import { createSeededRng, withRng, type Rng } from "../../../frontend/src/game/engine/core/random";
import type { GameState, SideId } from "../../../shared/src/types";
import {
  advanceModeledTurnStep,
  getForcedAttackCoinResults,
  setupAiVsAiGame,
  type EvaluateModelArgs,
} from "./evaluateModelVsHeuristic";
import { stateFingerprint } from "./stateFingerprint";
import { withGitMetadata } from "./manifest";
import { rankLegalActions } from "./candidateRanker";

type ProbeArgs = {
  games: number;
  seedStart: number;
  maxSteps: number;
  rolloutSteps: number;
  plannerTopK: number;
  plannerMaxSequences: number;
  plannerMaxDepth: number;
  microSamples: number;
  manifestOut: string | null;
};

type GameTimings = {
  seed: string;
  side: SideId;
  steps: number;
  baselineDecisions: number;
  baselineDecisionMs: number;
  enumerateMs: number;
  chooseMs: number;
  fingerprintMs: number;
  cloneMs: number;
  plannerDecisions: number;
  plannerDecisionMs: number;
  plannerEnumeratedBundles: number;
};

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const games = args.games;
  const seeds = Array.from({ length: games }, (_, i) => String(args.seedStart + i));

  const startWall = performance.now();
  const baselineGames: GameTimings[] = [];
  for (let i = 0; i < seeds.length; i += 1) {
    const seed = seeds[i]!;
    const side: SideId = i % 2 === 0 ? "player" : "opponent";
    baselineGames.push(runBaselineGame(seed, side, args));
  }
  const baselineWall = performance.now() - startWall;

  const microSeed = seeds[0] ?? "0";
  const micro = microbenchmark(microSeed, args.microSamples);

  const startPlanner = performance.now();
  const plannerGames: GameTimings[] = [];
  const plannerArgs = makePlannerArgs(args);
  for (let i = 0; i < seeds.length; i += 1) {
    const seed = seeds[i]!;
    const side: SideId = i % 2 === 0 ? "player" : "opponent";
    plannerGames.push(runPlannerGame(seed, side, args, plannerArgs));
  }
  const plannerWall = performance.now() - startPlanner;

  const baseline = aggregate(baselineGames);
  const planner = aggregatePlanner(plannerGames);

  const baselineDecisionsPerSec = baseline.totalDecisions / (baselineWall / 1000);
  const baselineDecisionsPerGame = baseline.totalDecisions / games;
  const plannerDecisionsPerSec = planner.totalDecisions / (plannerWall / 1000);
  const plannerDecisionsPerGame = planner.totalDecisions / games;

  const projected = projectIteration(plannerDecisionsPerSec, plannerDecisionsPerGame);

  const summary = {
    games,
    seeds: { start: args.seedStart, end: args.seedStart + games - 1 },
    baseline: {
      wallSeconds: baselineWall / 1000,
      totalDecisions: baseline.totalDecisions,
      totalSteps: baseline.totalSteps,
      decisionsPerSec: baselineDecisionsPerSec,
      decisionsPerGame: baselineDecisionsPerGame,
      avgEnumerateMs: baseline.avgEnumerateMs,
      avgChooseMs: baseline.avgChooseMs,
      avgFingerprintMs: baseline.avgFingerprintMs,
      avgCloneMs: baseline.avgCloneMs,
    },
    planner: {
      wallSeconds: plannerWall / 1000,
      totalDecisions: planner.totalDecisions,
      decisionsPerSec: plannerDecisionsPerSec,
      decisionsPerGame: plannerDecisionsPerGame,
      avgPlannerDecisionMs: planner.avgPlannerDecisionMs,
      avgBundlesEnumerated: planner.avgBundlesEnumerated,
      plannerConfig: plannerArgs,
    },
    micro,
    projection: projected,
    targets: {
      iterationWallHoursMax: 4,
      plannerDecisionsPerSecAtFullWorkers: 200,
    },
    notes: notes(plannerDecisionsPerSec, projected),
  };
  const output = { args, summary };

  if (args.manifestOut) {
    mkdirSync(dirname(args.manifestOut), { recursive: true });
    writeFileSync(args.manifestOut, JSON.stringify(withGitMetadata(output), null, 2) + "\n", "utf8");
  }

  console.log(JSON.stringify(output, null, 2));
}

function makePlannerArgs(args: ProbeArgs): EvaluateModelArgs {
  return {
    modelUrl: "",
    games: args.games,
    seedStart: args.seedStart,
    maxSteps: args.maxSteps,
    modelSide: "both",
    details: false,
    selection: "planner",
    rolloutSteps: args.rolloutSteps,
    searchDepth: 0,
    searchTopK: 0,
    searchSamples: 0,
    ranker: "heuristic",
    decisionTraceOut: null,
    manifestOut: null,
    traceTeacher: "none",
    plannerTopK: args.plannerTopK,
    plannerMaxSequences: args.plannerMaxSequences,
    plannerMaxDepth: args.plannerMaxDepth,
    cycleWindow: 0,
  };
}

function runBaselineGame(seed: string, modelSide: SideId, args: ProbeArgs): GameTimings {
  const rng = createSeededRng(`${seed}:${modelSide}`, "throughput-baseline");
  return withRng(rng, () => runBaselineGameWithRng(seed, modelSide, args, rng));
}

function runBaselineGameWithRng(seed: string, modelSide: SideId, args: ProbeArgs, rng: Rng): GameTimings {
  let state = setupAiVsAiGame();
  const timings: GameTimings = {
    seed,
    side: modelSide,
    steps: 0,
    baselineDecisions: 0,
    baselineDecisionMs: 0,
    enumerateMs: 0,
    chooseMs: 0,
    fingerprintMs: 0,
    cloneMs: 0,
    plannerDecisions: 0,
    plannerDecisionMs: 0,
    plannerEnumeratedBundles: 0,
  };

  for (let step = 0; step < args.maxSteps; step += 1) {
    if (state.gameOver) break;
    const sideId = state.currentSide === "player" || state.currentSide === "opponent" ? state.currentSide : "player";
    const forcedCoinResults = getForcedAttackCoinResults(state, rng);
    if (sideId === modelSide) {
      const t0 = performance.now();
      const legalActions = enumerateLegalAiActions(state, sideId);
      const t1 = performance.now();
      const selected = chooseHighestScoredAction(legalActions);
      const t2 = performance.now();
      timings.enumerateMs += t1 - t0;
      timings.chooseMs += t2 - t1;
      timings.baselineDecisions += 1;
      timings.baselineDecisionMs += t2 - t0;

      const fp0 = performance.now();
      stateFingerprint(state);
      timings.fingerprintMs += performance.now() - fp0;

      const c0 = performance.now();
      cloneGame(state);
      timings.cloneMs += performance.now() - c0;

      state = advanceModeledTurnStep(state, sideId, selected, forcedCoinResults, rng);
    } else {
      state = sideId === "player"
        ? advancePlayerAiTurnStep(state, forcedCoinResults, rng.next)
        : advanceOpponentTurnStep(state, forcedCoinResults, rng.next);
    }
    timings.steps += 1;
  }

  return timings;
}

function runPlannerGame(seed: string, modelSide: SideId, args: ProbeArgs, plannerArgs: EvaluateModelArgs): GameTimings {
  const rng = createSeededRng(`${seed}:${modelSide}`, "throughput-planner");
  return withRng(rng, () => runPlannerGameWithRng(seed, modelSide, args, plannerArgs, rng));
}

function runPlannerGameWithRng(seed: string, modelSide: SideId, args: ProbeArgs, plannerArgs: EvaluateModelArgs, rng: Rng): GameTimings {
  let state = setupAiVsAiGame();
  const timings: GameTimings = {
    seed,
    side: modelSide,
    steps: 0,
    baselineDecisions: 0,
    baselineDecisionMs: 0,
    enumerateMs: 0,
    chooseMs: 0,
    fingerprintMs: 0,
    cloneMs: 0,
    plannerDecisions: 0,
    plannerDecisionMs: 0,
    plannerEnumeratedBundles: 0,
  };

  for (let step = 0; step < args.maxSteps; step += 1) {
    if (state.gameOver) break;
    const sideId = state.currentSide === "player" || state.currentSide === "opponent" ? state.currentSide : "player";
    const forcedCoinResults = getForcedAttackCoinResults(state, rng);
    if (sideId === modelSide) {
      const t0 = performance.now();
      const decision = inlinePlannerDecision(plannerArgs, state, sideId, `${seed}:${modelSide}:${step}`, timings);
      const t1 = performance.now();
      timings.plannerDecisions += 1;
      timings.plannerDecisionMs += t1 - t0;
      state = advanceModeledTurnStep(state, sideId, decision.action, forcedCoinResults, rng);
    } else {
      state = sideId === "player"
        ? advancePlayerAiTurnStep(state, forcedCoinResults, rng.next)
        : advanceOpponentTurnStep(state, forcedCoinResults, rng.next);
    }
    timings.steps += 1;
  }

  return timings;
}

function inlinePlannerDecision(
  args: EvaluateModelArgs,
  state: GameState,
  sideId: SideId,
  seed: string,
  timings: GameTimings,
): { action: ReturnType<typeof enumerateLegalAiActions>[number]; selectedIndex: number } {
  const legalActions = enumerateLegalAiActions(state, sideId);
  if (legalActions.length <= 1) {
    return { action: legalActions[0] ?? chooseHighestScoredAction(legalActions), selectedIndex: 0 };
  }
  const bundles = enumerateBundlesForProbe(args, state, sideId, seed);
  timings.plannerEnumeratedBundles += bundles.length;
  if (bundles.length === 0) {
    const fallback = chooseHighestScoredAction(legalActions);
    const idx = Math.max(0, legalActions.findIndex((a) => a.id === fallback.id));
    return { action: fallback, selectedIndex: idx };
  }
  let bestReward = Number.NEGATIVE_INFINITY;
  let best = bundles[0]!;
  bundles.forEach((bundle, index) => {
    const rng = createSeededRng(`${seed}:bundle:${index}:leaf`, "planner-leaf-probe");
    const reward = leafScore(bundle.state, sideId, rng, args.rolloutSteps);
    if (reward > bestReward) {
      bestReward = reward;
      best = bundle;
    }
  });
  return { action: best.first.action, selectedIndex: best.first.selectedIndex };
}

function enumerateBundlesForProbe(
  args: EvaluateModelArgs,
  state: GameState,
  sideId: SideId,
  seed: string,
) {
  type Bundle = { first: { action: ReturnType<typeof enumerateLegalAiActions>[number]; selectedIndex: number }; state: GameState };
  type Partial = { first: Bundle["first"] | null; state: GameState; depth: number };
  const complete: Bundle[] = [];
  const queue: Partial[] = [{ first: null, state: cloneGame(state), depth: 0 }];
  while (queue.length && complete.length < args.plannerMaxSequences) {
    const current = queue.shift()!;
    if (current.state.gameOver || current.state.currentSide !== sideId || current.depth >= args.plannerMaxDepth) {
      if (current.first) complete.push({ first: current.first, state: current.state });
      continue;
    }
    const legalActions = enumerateLegalAiActions(current.state, sideId);
    const ranked = rankLegalActions(legalActions, {
      topK: Math.max(1, args.plannerTopK),
      mode: args.ranker,
      baseline: chooseHighestScoredAction(legalActions),
      rng: createSeededRng(`${seed}:d${current.depth}:ranker`, "planner-ranker-probe"),
    });
    ranked.forEach(({ action, index }) => {
      if (complete.length + queue.length >= args.plannerMaxSequences) return;
      const rng = createSeededRng(`${seed}:d${current.depth}:a${index}`, "planner-action-probe");
      const beforeFp = stateFingerprint(current.state);
      const next = advanceModeledTurnStep(current.state, sideId, action, getForcedAttackCoinResults(current.state, rng), rng);
      if (stateFingerprint(next) === beforeFp) return;
      const first = current.first ?? { action, selectedIndex: index };
      if (next.gameOver || next.currentSide !== sideId || action.kind === "attack" || action.kind === "endTurn") {
        complete.push({ first, state: next });
      } else {
        queue.push({ first, state: next, depth: current.depth + 1 });
      }
    });
  }
  return complete;
}

function leafScore(state: GameState, sideId: SideId, rng: Rng, maxSteps: number): number {
  let next = cloneGame(state);
  for (let step = 0; step < maxSteps; step += 1) {
    if (next.gameOver) break;
    const sId = next.currentSide === "player" || next.currentSide === "opponent" ? next.currentSide : "player";
    const forced = getForcedAttackCoinResults(next, rng);
    next = sId === "player"
      ? advancePlayerAiTurnStep(next, forced, rng.next)
      : advanceOpponentTurnStep(next, forced, rng.next);
  }
  const opponentId: SideId = sideId === "player" ? "opponent" : "player";
  const margin = (next.sides[sideId].points - next.sides[opponentId].points) / 3;
  if (next.winner === sideId) return 1 + margin * 0.2;
  if (next.winner === opponentId) return -1 + margin * 0.2;
  return margin;
}

function microbenchmark(seed: string, samples: number) {
  const rng = createSeededRng(seed, "throughput-micro");
  const state = withRng(rng, () => setupAiVsAiGame());
  const sideId: SideId = state.currentSide === "player" || state.currentSide === "opponent" ? state.currentSide : "player";

  // warm
  for (let i = 0; i < 50; i += 1) {
    enumerateLegalAiActions(state, sideId);
    chooseHighestScoredAction(enumerateLegalAiActions(state, sideId));
    stateFingerprint(state);
    cloneGame(state);
  }

  const enumerateNs = timeRepeat(samples, () => {
    enumerateLegalAiActions(state, sideId);
  });

  const cachedActions = enumerateLegalAiActions(state, sideId);
  const chooseNs = timeRepeat(samples, () => {
    chooseHighestScoredAction(cachedActions);
  });

  const fingerprintNs = timeRepeat(samples, () => {
    stateFingerprint(state);
  });

  const cloneNs = timeRepeat(samples, () => {
    cloneGame(state);
  });

  return {
    samples,
    legalActionCount: cachedActions.length,
    enumerateNsPerCall: enumerateNs,
    chooseNsPerCall: chooseNs,
    fingerprintNsPerCall: fingerprintNs,
    cloneNsPerCall: cloneNs,
  };
}

function timeRepeat(samples: number, fn: () => void): number {
  const start = performance.now();
  for (let i = 0; i < samples; i += 1) fn();
  const elapsed = performance.now() - start;
  return (elapsed * 1_000_000) / samples;
}

function aggregate(games: GameTimings[]) {
  const totalSteps = games.reduce((sum, g) => sum + g.steps, 0);
  const totalDecisions = games.reduce((sum, g) => sum + g.baselineDecisions, 0);
  const totalDecisionMs = games.reduce((sum, g) => sum + g.baselineDecisionMs, 0);
  const totalEnumerateMs = games.reduce((sum, g) => sum + g.enumerateMs, 0);
  const totalChooseMs = games.reduce((sum, g) => sum + g.chooseMs, 0);
  const totalFingerprintMs = games.reduce((sum, g) => sum + g.fingerprintMs, 0);
  const totalCloneMs = games.reduce((sum, g) => sum + g.cloneMs, 0);
  return {
    totalSteps,
    totalDecisions,
    avgDecisionMs: totalDecisions ? totalDecisionMs / totalDecisions : 0,
    avgEnumerateMs: totalDecisions ? totalEnumerateMs / totalDecisions : 0,
    avgChooseMs: totalDecisions ? totalChooseMs / totalDecisions : 0,
    avgFingerprintMs: totalDecisions ? totalFingerprintMs / totalDecisions : 0,
    avgCloneMs: totalDecisions ? totalCloneMs / totalDecisions : 0,
  };
}

function aggregatePlanner(games: GameTimings[]) {
  const totalDecisions = games.reduce((sum, g) => sum + g.plannerDecisions, 0);
  const totalDecisionMs = games.reduce((sum, g) => sum + g.plannerDecisionMs, 0);
  const totalBundles = games.reduce((sum, g) => sum + g.plannerEnumeratedBundles, 0);
  return {
    totalDecisions,
    avgPlannerDecisionMs: totalDecisions ? totalDecisionMs / totalDecisions : 0,
    avgBundlesEnumerated: totalDecisions ? totalBundles / totalDecisions : 0,
  };
}

function projectIteration(plannerDecisionsPerSec: number, plannerDecisionsPerGame: number) {
  const games = 500;
  const decisionsPerIteration = games * plannerDecisionsPerGame;
  const wallSecs = (cores: number) => decisionsPerIteration / Math.max(1e-9, plannerDecisionsPerSec * cores);
  return {
    games,
    plannerDecisionsPerIteration: decisionsPerIteration,
    wallHours_1core: wallSecs(1) / 3600,
    wallHours_8cores: wallSecs(8) / 3600,
    wallHours_32cores: wallSecs(32) / 3600,
  };
}

function notes(plannerDecisionsPerSec: number, projected: { wallHours_8cores: number; wallHours_32cores: number }) {
  const flags: string[] = [];
  if (plannerDecisionsPerSec < 25) flags.push(`single-core planner throughput ${plannerDecisionsPerSec.toFixed(1)} dec/s is below the 200 dec/s target / 8 ideal workers`);
  if (projected.wallHours_8cores > 24) flags.push(`projected ≥24h iteration on 8 cores — escalate to optimization sub-task`);
  if (projected.wallHours_32cores > 4) flags.push(`projected >4h iteration on 32 cores — outside item 14 budget target`);
  return flags;
}

function parseArgs(argv: string[]): ProbeArgs {
  const get = (name: string, fallback: string) => {
    const index = argv.indexOf(name);
    return index >= 0 ? argv[index + 1] ?? fallback : fallback;
  };
  return {
    games: Number(get("--games", "10")),
    seedStart: Number(get("--seed-start", "9000")),
    maxSteps: Number(get("--max-steps", "500")),
    rolloutSteps: Number(get("--rollout-steps", "120")),
    plannerTopK: Number(get("--planner-top-k", "4")),
    plannerMaxSequences: Number(get("--planner-max-sequences", "32")),
    plannerMaxDepth: Number(get("--planner-max-depth", "6")),
    microSamples: Number(get("--micro-samples", "2000")),
    manifestOut: get("--manifest-out", "") || null,
  };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
