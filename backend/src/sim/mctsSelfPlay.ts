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
import { appendFileSync, mkdirSync, readFileSync, unlinkSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { fork, type ChildProcess } from "node:child_process";
import { enumerateLegalAiActions, chooseHighestScoredAction } from "../../../frontend/src/game/engine/ai-policy/actions";
import { buildPublicObservation } from "../../../frontend/src/game/engine/ai-policy/observation";
import "./rngAsyncStore";
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
import { MCTS_WORK_STEALING_ENABLED, WORK_STEALING_PREFETCH } from "./workStealing";

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
  mctsAdaptiveRatio: number;
  mctsAdaptiveMinSims: number;
  temperatureMoves: number;
  temperatureValue: number;
  outPath: string;
  manifestOut: string | null;
  workers: number;
  workerMode: boolean;
};

type GameSummary = {
  seed: string;
  winner: SideId | null;
  points: Record<SideId, number>;
  modelDecisions: number;
  rowCount: number;
  visitEntropySum: number;
  terminalReason: GameRecord["terminalReason"];
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
  rootMeanQ: number[];
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
  if (args.workerMode) return runWorker(args);
  mkdirSync(dirname(args.outPath), { recursive: true });
  writeFileSync(args.outPath, "", "utf8");

  const summaries: GameSummary[] = [];
  if (args.workers > 1 && args.games > 0) {
    const seeds: string[] = [];
    for (let index = 0; index < args.games; index += 1) seeds.push(String(args.seedStart + index));
    await runOrchestrator(args, seeds, summaries);
  } else {
    const runStartedAt = Date.now();
    for (let index = 0; index < args.games; index += 1) {
      const seed = String(args.seedStart + index);
      const record = await runSelfPlayGame(args, seed);
      if (record.rows.length) {
        appendFileSync(args.outPath, record.rows.map((row) => JSON.stringify(row)).join("\n") + "\n", "utf8");
      }
      summaries.push(gameRecordToSummary(seed, record));
      const elapsedSec = (Date.now() - runStartedAt) / 1000;
      process.stderr.write(
        `[selfplay ${index + 1}/${args.games}] seed=${seed} winner=${record.winner ?? "none"} rows=${record.rows.length} elapsed=${elapsedSec.toFixed(1)}s\n`,
      );
    }
  }
  const summary = summarizeFromSummaries(summaries);
  const output = { args, summary };
  if (args.manifestOut) {
    mkdirSync(dirname(args.manifestOut), { recursive: true });
    writeFileSync(args.manifestOut, JSON.stringify(withGitMetadata(output), null, 2) + "\n", "utf8");
  }
  console.log(JSON.stringify(output, null, 2));
}

function buildWorkerArgv(parentArgv: string[]): string[] {
  const out: string[] = [];
  for (let i = 0; i < parentArgv.length; i += 1) {
    if (parentArgv[i] === "--workers") { i += 1; continue; }
    if (parentArgv[i] === "--out") { i += 1; continue; }
    if (parentArgv[i] === "--manifest-out") { i += 1; continue; }
    out.push(parentArgv[i]!);
  }
  out.push("--worker-mode");
  return out;
}

function partitionSeeds(seeds: string[], workers: number): string[][] {
  const sliceSize = Math.ceil(seeds.length / workers);
  const out: string[][] = [];
  for (let w = 0; w < workers; w += 1) {
    const slice = seeds.slice(w * sliceSize, (w + 1) * sliceSize);
    if (slice.length > 0) out.push(slice);
  }
  return out;
}

// Static contiguous chunking. Worker w owns seed-index slice
// [w*size,(w+1)*size); concatenating shards in worker order is therefore
// already global seed order. Kept verbatim behind UMA_MCTS_WORK_STEALING=0.
async function runOrchestratorStatic(args: SelfPlayArgs, seeds: string[], summaries: GameSummary[]): Promise<void> {
  const slices = partitionSeeds(seeds, args.workers);
  const workerArgv = buildWorkerArgv(process.argv.slice(2));
  const shardPaths: string[] = [];
  const runStartedAt = Date.now();
  let gamesCompleted = 0;
  await Promise.all(slices.map((slice, workerId) => new Promise<void>((resolve, reject) => {
    const shardPath = `${args.outPath}.w${workerId}`;
    shardPaths.push(shardPath);
    const workerArgvForChild = ["--out", shardPath, ...workerArgv];
    const child: ChildProcess = fork(process.argv[1]!, workerArgvForChild, {
      stdio: ["inherit", "inherit", "inherit", "ipc"],
    });
    let workerReady = false;
    let workerDone = false;
    child.on("message", (msg: unknown) => {
      const m = msg as { kind: string; summary?: GameSummary };
      if (m.kind === "ready") {
        workerReady = true;
        child.send({ kind: "seeds", workerId, seeds: slice });
      } else if (m.kind === "game_completed" && m.summary) {
        summaries.push(m.summary);
        gamesCompleted += 1;
        const elapsedSec = (Date.now() - runStartedAt) / 1000;
        process.stderr.write(
          `[selfplay ${gamesCompleted}/${seeds.length} w${workerId}] seed=${m.summary.seed} winner=${m.summary.winner ?? "none"} rows=${m.summary.rowCount} elapsed=${elapsedSec.toFixed(1)}s\n`,
        );
      } else if (m.kind === "done") {
        workerDone = true;
      }
    });
    child.on("error", (err) => reject(err));
    child.on("exit", (code) => {
      if (!workerReady) {
        reject(new Error(`selfplay worker ${workerId} exited before becoming ready (code=${code})`));
      } else if (!workerDone) {
        reject(new Error(`selfplay worker ${workerId} exited before completing seeds (code=${code})`));
      } else if (code !== 0 && code !== null) {
        reject(new Error(`selfplay worker ${workerId} exited non-zero (code=${code})`));
      } else {
        resolve();
      }
    });
  })));
  // Concatenate shard files into the canonical outPath (worker order ==
  // global seed order under contiguous chunking).
  for (const shardPath of shardPaths) {
    try {
      const data = readFileSync(shardPath, "utf8");
      if (data.length > 0) appendFileSync(args.outPath, data, "utf8");
      unlinkSync(shardPath);
    } catch {
      // Shard may not exist if its slice was empty; ignore.
    }
  }
}

// Work-stealing dispatch (docs/ai-research/scoping/r12-selfplay-gate-throughput.md).
// Shared seed-index queue: seed each worker with WORK_STEALING_PREFETCH tasks,
// refill one task on every completion. Each game's rows are written to a
// per-seed-index shard `${outPath}.s${idx}`; the orchestrator concatenates
// shards in ASCENDING SEED-INDEX ORDER (not completion order), and pushes
// summaries into a slot array indexed by seed-index. Output is therefore
// byte-identical to the static path regardless of which worker finished when.
async function runOrchestratorWorkStealing(args: SelfPlayArgs, seeds: string[], summaries: GameSummary[]): Promise<void> {
  const workerArgv = buildWorkerArgv(process.argv.slice(2));
  const workerCount = Math.min(seeds.length, args.workers);
  const summarySlots: (GameSummary | null)[] = new Array(seeds.length).fill(null);
  const runStartedAt = Date.now();
  let nextSeedIndex = 0;
  let gamesCompleted = 0;

  await Promise.all(Array.from({ length: workerCount }, (_unused, workerId) => new Promise<void>((resolve, reject) => {
    const child: ChildProcess = fork(process.argv[1]!, workerArgv, {
      stdio: ["inherit", "inherit", "inherit", "ipc"],
    });
    let workerReady = false;
    let workerDone = false;
    let inFlight = 0;

    const dispatchNext = (): void => {
      if (nextSeedIndex >= seeds.length) {
        if (inFlight === 0) child.send({ kind: "no_more_tasks" });
        return;
      }
      const seedIndex = nextSeedIndex;
      nextSeedIndex += 1;
      inFlight += 1;
      child.send({
        kind: "task",
        seedIndex,
        seed: seeds[seedIndex]!,
        shardPath: `${args.outPath}.s${seedIndex}`,
      });
    };

    child.on("message", (msg: unknown) => {
      const m = msg as { kind: string; summary?: GameSummary; seedIndex?: number };
      if (m.kind === "ready") {
        workerReady = true;
        for (let k = 0; k < WORK_STEALING_PREFETCH; k += 1) dispatchNext();
      } else if (m.kind === "game_completed" && m.summary && m.seedIndex !== undefined) {
        summarySlots[m.seedIndex] = m.summary;
        inFlight -= 1;
        gamesCompleted += 1;
        const elapsedSec = (Date.now() - runStartedAt) / 1000;
        process.stderr.write(
          `[selfplay ${gamesCompleted}/${seeds.length} w${workerId}] seed=${m.summary.seed} winner=${m.summary.winner ?? "none"} rows=${m.summary.rowCount} elapsed=${elapsedSec.toFixed(1)}s\n`,
        );
        dispatchNext();
      } else if (m.kind === "done") {
        workerDone = true;
      }
    });
    child.on("error", (err) => reject(err));
    child.on("exit", (code) => {
      if (!workerReady) {
        reject(new Error(`selfplay worker ${workerId} exited before becoming ready (code=${code})`));
      } else if (!workerDone) {
        reject(new Error(`selfplay worker ${workerId} exited before completing tasks (code=${code})`));
      } else if (code !== 0 && code !== null) {
        reject(new Error(`selfplay worker ${workerId} exited non-zero (code=${code})`));
      } else {
        resolve();
      }
    });
  })));

  // Concatenate per-seed-index shards in ascending seed order (deterministic,
  // independent of completion order) then push summaries in the same order.
  for (let seedIndex = 0; seedIndex < seeds.length; seedIndex += 1) {
    const shardPath = `${args.outPath}.s${seedIndex}`;
    try {
      const data = readFileSync(shardPath, "utf8");
      if (data.length > 0) appendFileSync(args.outPath, data, "utf8");
      unlinkSync(shardPath);
    } catch {
      // Shard may not exist if the game produced zero rows; ignore.
    }
    const slot = summarySlots[seedIndex];
    if (slot) summaries.push(slot);
  }
}

async function runOrchestrator(args: SelfPlayArgs, seeds: string[], summaries: GameSummary[]): Promise<void> {
  if (MCTS_WORK_STEALING_ENABLED) {
    return runOrchestratorWorkStealing(args, seeds, summaries);
  }
  return runOrchestratorStatic(args, seeds, summaries);
}

async function runWorker(args: SelfPlayArgs): Promise<void> {
  if (!process.send) {
    process.stderr.write("mctsSelfPlay worker has no IPC channel\n");
    process.exit(2);
    return;
  }
  if (!MCTS_WORK_STEALING_ENABLED) {
    // Static contiguous chunking: worker writes its whole slice to the
    // single `--out` shard the orchestrator assigned it.
    mkdirSync(dirname(args.outPath), { recursive: true });
    writeFileSync(args.outPath, "", "utf8");
    process.send({ kind: "ready" });
    process.on("message", async (msg: unknown) => {
      const m = msg as { kind: string; workerId?: number; seeds?: string[] };
      if (m.kind !== "seeds" || !m.seeds) return;
      try {
        for (const seed of m.seeds) {
          const record = await runSelfPlayGame(args, seed);
          if (record.rows.length) {
            appendFileSync(args.outPath, record.rows.map((row) => JSON.stringify(row)).join("\n") + "\n", "utf8");
          }
          const summary = gameRecordToSummary(seed, record);
          process.send!({ kind: "game_completed", summary });
        }
        process.send!({ kind: "done" });
        setTimeout(() => process.exit(0), 25);
      } catch (err) {
        process.stderr.write(`selfplay worker error: ${(err as Error).message}\n`);
        process.exit(1);
      }
    });
    return;
  }

  // Work-stealing: the orchestrator prefetches WORK_STEALING_PREFETCH tasks
  // per worker to hide the IPC round-trip, so two `task` messages can be
  // in this worker's mailbox at once. They MUST NOT run concurrently: a
  // single game owns the process-wide AsyncLocalStorage RNG context
  // (rngAsyncStore.ts) and the shared serve_onnx client; interleaving two
  // games on the event loop perturbs both and breaks bit-identity. So
  // buffer prefetched tasks in an internal FIFO and drain it STRICTLY
  // SEQUENTIALLY — exactly one game in flight at any time, matching the
  // static path's per-worker `for`-loop semantics. Load balancing comes
  // from the orchestrator pulling the next task on completion, not from
  // intra-worker concurrency.
  type Task = { seedIndex: number; seed: string; shardPath: string };
  const pending: Task[] = [];
  let draining = false;
  let noMoreTasks = false;
  const finish = (): void => {
    process.send!({ kind: "done" });
    setTimeout(() => process.exit(0), 25);
  };
  const drain = async (): Promise<void> => {
    if (draining) return;
    draining = true;
    try {
      while (pending.length > 0) {
        const task = pending.shift()!;
        const record = await runSelfPlayGame(args, task.seed);
        if (record.rows.length) {
          mkdirSync(dirname(task.shardPath), { recursive: true });
          writeFileSync(task.shardPath, record.rows.map((row) => JSON.stringify(row)).join("\n") + "\n", "utf8");
        }
        const summary = gameRecordToSummary(task.seed, record);
        process.send!({ kind: "game_completed", summary, seedIndex: task.seedIndex });
      }
    } catch (err) {
      process.stderr.write(`selfplay worker error: ${(err as Error).message}\n`);
      process.exit(1);
    }
    draining = false;
    if (noMoreTasks && pending.length === 0) finish();
  };
  process.send({ kind: "ready" });
  process.on("message", (msg: unknown) => {
    const m = msg as { kind: string; seedIndex?: number; seed?: string; shardPath?: string };
    if (m.kind === "no_more_tasks") {
      noMoreTasks = true;
      if (!draining && pending.length === 0) finish();
      return;
    }
    if (m.kind !== "task" || m.seed === undefined || m.seedIndex === undefined || !m.shardPath) return;
    pending.push({ seedIndex: m.seedIndex, seed: m.seed, shardPath: m.shardPath });
    void drain();
  });
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
      adaptiveRatio: Math.max(0, args.mctsAdaptiveRatio),
      adaptiveMinSims: Math.max(1, args.mctsAdaptiveMinSims),
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
      rootMeanQ: mctsResult.diagnostics.rootMeanQ,
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

function summarizeFromSummaries(summaries: GameSummary[]) {
  const totalRows = summaries.reduce((sum, g) => sum + g.rowCount, 0);
  const playerWins = summaries.filter((g) => g.winner === "player").length;
  const opponentWins = summaries.filter((g) => g.winner === "opponent").length;
  const draws = summaries.filter((g) => g.winner === null).length;
  const meanLen = summaries.length ? summaries.reduce((s, g) => s + g.modelDecisions, 0) / summaries.length : 0;
  const totalEntropy = summaries.reduce((s, g) => s + g.visitEntropySum, 0);
  const meanVisitEntropy = totalRows > 0 ? totalEntropy / totalRows : 0;
  return {
    games: summaries.length,
    totalRows,
    playerWins,
    opponentWins,
    draws,
    meanGameLength: meanLen,
    meanVisitEntropy,
  };
}

function gameRecordToSummary(seed: string, record: GameRecord): GameSummary {
  const visitEntropySum = record.rows.reduce((s, r) => s + visitEntropy(r.visitDistribution), 0);
  return {
    seed,
    winner: record.winner,
    points: record.points,
    modelDecisions: record.modelDecisions,
    rowCount: record.rows.length,
    visitEntropySum,
    terminalReason: record.terminalReason,
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
    mctsAdaptiveRatio: Number(get("--mcts-adaptive-ratio", "0")),
    mctsAdaptiveMinSims: Number(get("--mcts-adaptive-min-sims", "20")),
    temperatureMoves: Number(get("--temperature-moves", "6")),
    temperatureValue: Number(get("--temperature-value", "1.0")),
    outPath: get("--out", "runs/R12-selfplay/selfplay.jsonl"),
    manifestOut: get("--manifest-out", "") || null,
    workers: Math.max(1, Number(get("--workers", "1"))),
    workerMode: argv.includes("--worker-mode"),
  };
}

void chooseHighestScoredAction; // currently unused; kept for parity with the evaluator's fallback path.

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
