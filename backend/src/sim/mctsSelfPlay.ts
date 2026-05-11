// R12 phase B: MCTS self-play data generator.
//
// Both sides play with MCTS using the served policy as prior and value head
// at the leaf. For each real decision we emit one row containing:
//   - observation (PublicObservation for the side to move)
//   - legalActions
//   - visitDistribution (π_target for distillation; soft, sums to 1)
//   - rootPriors (network's prior at the root — diagnostic, not a label)
//   - rootValue (network's leaf value at the root — diagnostic)
//   - sampleAction (sampled or argmax index — for trace fidelity, not a label)
//   - z_target (filled in at game end: +1 if this side won, -1 lost, 0 draw)
//
// The sampling policy is: temperature 1 over visit counts for the first
// `temperatureMoves` per-side moves (default 6) and argmax thereafter, in
// the spirit of AlphaZero's exploration / exploitation schedule.

import {
  advanceOpponentTurnStep,
  advancePlayerAiTurnStep,
} from "../../../frontend/src/game/engine";
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { enumerateLegalAiActions, chooseHighestScoredAction } from "../../../frontend/src/game/engine/ai-policy/actions";
import { buildPublicObservation } from "../../../frontend/src/game/engine/ai-policy/observation";
import { createSeededRng, withRng, type Rng } from "../../../frontend/src/game/engine/core/random";
import type { LegalAiAction } from "../../../frontend/src/game/engine/ai-policy/types";
import type { GameState, SideId } from "../../../shared/src/types";
import {
  advanceModeledTurnStep,
  getForcedAttackCoinResults,
  setupAiVsAiGame,
  stateHash,
} from "./evaluateModelVsHeuristic";
import { defaultMctsConfig, runMcts, type MctsConfig, type MctsResult } from "./mcts";
import { withGitMetadata } from "./manifest";

const ROW_SCHEMA_VERSION = 1;

type SelfPlayArgs = {
  modelUrl: string;
  games: number;
  seedStart: number;
  maxSteps: number;
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
  temperatureMoves: number;
  temperatureValue: number;
  outPath: string;
  manifestOut: string | null;
};

type SelfPlayRow = {
  schemaVersion: 1;
  kind: "mcts-selfplay";
  seed: string;
  sideId: SideId;
  step: number;
  turnNumber: number;
  observation: ReturnType<typeof buildPublicObservation>;
  legalActions: LegalAiAction[];
  selectedActionIndex: number;
  visitDistribution: number[];
  rootPriors: number[];
  rootValue: number;
  rootPriorEntropy: number;
  rootPriorArgmax: number;
  visitedHashes: number;
  expansions: number;
  leafEvaluations: number;
  // Soft, sums to 1. The distillation loss is
  //   -Σ_a visitDistribution[a] · log π_θ(a|s).
  // Filled at end of game.
  valueTarget: number | null;
  result: { winner: SideId | null; pointsP: number; pointsO: number } | null;
};

type GameRecord = {
  seed: string;
  winner: SideId | null;
  points: Record<SideId, number>;
  rows: SelfPlayRow[];
  modelDecisions: number;
  terminalReason: "gameOver" | "maxSteps" | "stalled";
};

async function main() {
  const args = parseArgs(process.argv.slice(2));
  mkdirSync(dirname(args.outPath), { recursive: true });
  writeFileSync(args.outPath, "", "utf8");

  const games: GameRecord[] = [];
  for (let index = 0; index < args.games; index += 1) {
    const seed = String(args.seedStart + index);
    const record = await runSelfPlayGame(args, seed);
    games.push(record);
    if (record.rows.length) {
      appendFileSync(args.outPath, record.rows.map((row) => JSON.stringify(row)).join("\n") + "\n", "utf8");
    }
  }
  const summary = summarize(games);
  const output = { args, summary };
  if (args.manifestOut) {
    mkdirSync(dirname(args.manifestOut), { recursive: true });
    writeFileSync(args.manifestOut, JSON.stringify(withGitMetadata(output), null, 2) + "\n", "utf8");
  }
  console.log(JSON.stringify(output, null, 2));
}

export async function runSelfPlayGame(args: SelfPlayArgs, seed: string): Promise<GameRecord> {
  const rng = createSeededRng(`${seed}:selfplay`, "selfplay");
  return withRng(rng, () => runSelfPlayGameWithRng(args, seed, rng));
}

async function runSelfPlayGameWithRng(args: SelfPlayArgs, seed: string, rng: Rng): Promise<GameRecord> {
  let state = setupAiVsAiGame();
  let terminalReason: GameRecord["terminalReason"] = "maxSteps";
  let modelDecisions = 0;
  const rows: SelfPlayRow[] = [];
  const movesPerSide: Record<SideId, number> = { player: 0, opponent: 0 };

  for (let step = 0; step < args.maxSteps; step += 1) {
    if (state.gameOver) {
      terminalReason = "gameOver";
      break;
    }
    const before = stateHash(state);
    const sideId: SideId = state.currentSide === "player" || state.currentSide === "opponent" ? state.currentSide : "player";
    const legalActions = enumerateLegalAiActions(state, sideId);
    const forcedCoins = getForcedAttackCoinResults(state, rng);

    if (legalActions.length <= 1) {
      // Single-action shortcut: don't record a training row (no choice to
      // distill), just advance.
      const action = legalActions[0];
      if (action) {
        const next = advanceModeledTurnStep(state, sideId, action, forcedCoins, rng);
        state = stateHash(next) === before
          ? (sideId === "player" ? advancePlayerAiTurnStep(state, forcedCoins, rng.next) : advanceOpponentTurnStep(state, forcedCoins, rng.next))
          : next;
      } else {
        state = sideId === "player"
          ? advancePlayerAiTurnStep(state, forcedCoins, rng.next)
          : advanceOpponentTurnStep(state, forcedCoins, rng.next);
      }
      if (stateHash(state) === before) {
        terminalReason = "stalled";
        break;
      }
      continue;
    }

    const mctsConfig: MctsConfig = defaultMctsConfig({
      simulations: Math.max(1, args.mctsSimulations),
      cPuct: args.mctsCPuct,
      leaf: args.mctsLeaf,
      rolloutCrnSamples: Math.max(1, args.mctsRolloutCrnSamples),
      rolloutSteps: Math.max(1, args.mctsRolloutSteps),
      prior: args.mctsPrior,
      addRootDirichlet: args.mctsRootDirichlet,
      dirichletAlpha: args.mctsDirichletAlpha,
      dirichletEpsilon: args.mctsDirichletEpsilon,
      collapseMaxSteps: Math.max(1, args.mctsCollapseMaxSteps),
      maxNodes: Math.max(64, args.mctsMaxNodes),
    });
    const mctsResult = await runMcts(state, sideId, mctsConfig, args.modelUrl, `${seed}:${sideId}:${step}:mcts`);
    const totalVisits = mctsResult.visits.reduce((sum, n) => sum + n, 0);
    const visitDistribution = totalVisits > 0
      ? mctsResult.visits.map((n) => n / totalVisits)
      : legalActions.map(() => 1 / legalActions.length);

    // Temperature schedule: exploratory on the first `temperatureMoves`
    // decisions per side; greedy thereafter. Matches AlphaZero's standard
    // self-play protocol.
    const sideMoves = movesPerSide[sideId];
    const useTemperature = sideMoves < args.temperatureMoves;
    const temperature = useTemperature ? args.temperatureValue : 0;
    const selectedIndex = pickFromVisits(mctsResult, visitDistribution, temperature, rng.fork(`pick:${step}`));
    movesPerSide[sideId] += 1;
    modelDecisions += 1;

    const row: SelfPlayRow = {
      schemaVersion: ROW_SCHEMA_VERSION,
      kind: "mcts-selfplay",
      seed,
      sideId,
      step,
      turnNumber: state.turnNumber,
      observation: buildPublicObservation(state, sideId),
      legalActions,
      selectedActionIndex: selectedIndex,
      visitDistribution,
      rootPriors: mctsResult.diagnostics.rootPriors,
      rootValue: mctsResult.diagnostics.rootValue,
      rootPriorEntropy: mctsResult.diagnostics.rootPriorEntropy,
      rootPriorArgmax: mctsResult.diagnostics.rootPriorArgmax,
      visitedHashes: mctsResult.diagnostics.visitedHashes,
      expansions: mctsResult.diagnostics.expansions,
      leafEvaluations: mctsResult.diagnostics.leafEvaluations,
      valueTarget: null,
      result: null,
    };
    rows.push(row);

    const action = legalActions[selectedIndex] ?? legalActions[0]!;
    const next = advanceModeledTurnStep(state, sideId, action, forcedCoins, rng);
    if (stateHash(next) === before) {
      // MCTS picked a no-op (rare): fall back to the rule bot's move so
      // the game advances. Drop the just-recorded row since the action
      // didn't actually change state.
      rows.pop();
      modelDecisions -= 1;
      state = sideId === "player"
        ? advancePlayerAiTurnStep(state, forcedCoins, rng.next)
        : advanceOpponentTurnStep(state, forcedCoins, rng.next);
    } else {
      state = next;
    }
    if (stateHash(state) === before) {
      terminalReason = "stalled";
      break;
    }
  }

  const winner: SideId | null = state.winner;
  const points = { player: state.sides.player.points, opponent: state.sides.opponent.points };
  const result = { winner, pointsP: points.player, pointsO: points.opponent };
  for (const row of rows) {
    row.valueTarget = winner === null ? 0 : (row.sideId === winner ? 1 : -1);
    row.result = result;
  }

  return {
    seed,
    winner,
    points,
    rows,
    modelDecisions,
    terminalReason: state.gameOver ? "gameOver" : terminalReason,
  };
}

function pickFromVisits(
  result: MctsResult,
  visitDistribution: number[],
  temperature: number,
  rng: Rng,
): number {
  if (temperature <= 0 || visitDistribution.length <= 1) {
    return result.selectedIndex;
  }
  // Resample proportional to visits^(1/T). T=1 → straight visit proportion.
  const weights = result.visits.map((n) => Math.pow(Math.max(0, n), 1 / temperature));
  const total = weights.reduce((sum, w) => sum + w, 0);
  if (total <= 0) return result.selectedIndex;
  const r = rng.next() * total;
  let cum = 0;
  for (let i = 0; i < weights.length; i += 1) {
    cum += weights[i]!;
    if (r <= cum) return i;
  }
  void visitDistribution;
  return weights.length - 1;
}

function summarize(games: GameRecord[]) {
  const totalRows = games.reduce((sum, g) => sum + g.rows.length, 0);
  const playerWins = games.filter((g) => g.winner === "player").length;
  const opponentWins = games.filter((g) => g.winner === "opponent").length;
  const draws = games.filter((g) => g.winner === null).length;
  const meanLen = games.length ? games.reduce((s, g) => s + g.modelDecisions, 0) / games.length : 0;
  const meanVisitEntropy = games.length
    ? games.reduce((s, g) => s + g.rows.reduce((rs, r) => rs + visitEntropy(r.visitDistribution), 0), 0) / Math.max(1, totalRows)
    : 0;
  return {
    games: games.length,
    totalRows,
    playerWins,
    opponentWins,
    draws,
    meanGameLength: meanLen,
    meanVisitEntropy,
  };
}

function visitEntropy(dist: number[]): number {
  let h = 0;
  for (const p of dist) {
    if (p > 0) h -= p * Math.log(p);
  }
  return h;
}

function parseArgs(argv: string[]): SelfPlayArgs {
  const get = (name: string, fallback: string) => {
    const i = argv.indexOf(name);
    return i >= 0 ? argv[i + 1] ?? fallback : fallback;
  };
  return {
    modelUrl: get("--model-url", "http://127.0.0.1:8765"),
    games: Number(get("--games", "200")),
    seedStart: Number(get("--seed-start", "30000")),
    maxSteps: Number(get("--max-steps", "500")),
    mctsSimulations: Number(get("--mcts-simulations", "100")),
    mctsCPuct: Number(get("--mcts-c-puct", "1.5")),
    mctsLeaf: (get("--mcts-leaf", "value-head") === "rollout" ? "rollout" : "value-head"),
    mctsRolloutCrnSamples: Number(get("--mcts-rollout-crn-samples", "3")),
    mctsRolloutSteps: Number(get("--mcts-rollout-steps", "200")),
    mctsCollapseMaxSteps: Number(get("--mcts-collapse-max-steps", "64")),
    mctsMaxNodes: Number(get("--mcts-max-nodes", "5000")),
    mctsPrior: get("--mcts-prior", "policy") === "uniform" ? "uniform" : "policy",
    mctsRootDirichlet: !argv.includes("--no-root-dirichlet"),
    mctsDirichletAlpha: Number(get("--mcts-dirichlet-alpha", "0.3")),
    mctsDirichletEpsilon: Number(get("--mcts-dirichlet-epsilon", "0.25")),
    temperatureMoves: Number(get("--temperature-moves", "6")),
    temperatureValue: Number(get("--temperature-value", "1.0")),
    outPath: get("--out", "runs/R12-selfplay/selfplay.jsonl"),
    manifestOut: get("--manifest-out", "") || null,
  };
}

void chooseHighestScoredAction; // currently unused; kept for parity with the evaluator's fallback path.

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
