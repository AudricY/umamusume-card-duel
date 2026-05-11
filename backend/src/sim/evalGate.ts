import { runModelVsHeuristicGame, type EvaluateModelArgs } from "./evaluateModelVsHeuristic";
import type { SideId } from "../../../shared/src/types";
import type { CandidateRankerMode } from "./candidateRanker";
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { fork, type ChildProcess } from "node:child_process";
import { withGitMetadata } from "./manifest";

type Args = EvaluateModelArgs & {
  minGames: number;
  minWinRate: number;
  minCiLower: number;
  maxCiLower: number | null;
  requireZeroFallbacks: boolean;
  requireZeroNoOps: boolean;
  requireManifest: boolean;
  expectFail: boolean;
  manifestOut: string | null;
  workers: number;
  workerMode: boolean;
};

type WorkerTask = { seed: string; side: SideId };

class EvalGateError extends Error {
  constructor(public readonly errorCode: string, message: string) {
    super(message);
    this.name = "EvalGateError";
  }
}

type GateResult = Awaited<ReturnType<typeof runModelVsHeuristicGame>>;

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.workerMode) return runWorker(args);
  if (args.requireManifest && !args.manifestOut) {
    const error = { error: "EvalGateError", code: "missing_manifest_path", message: "--require-manifest demands --manifest-out <path>" };
    console.error(JSON.stringify(error));
    process.exit(2);
  }
  const sides: SideId[] = args.modelSide === "both" ? ["player", "opponent"] : [args.modelSide];
  const tasks: WorkerTask[] = [];
  for (const side of sides) {
    for (let index = 0; index < args.games; index += 1) {
      tasks.push({ seed: String(args.seedStart + index), side });
    }
  }
  if (args.progressOut) {
    mkdirSync(dirname(args.progressOut), { recursive: true });
    writeFileSync(args.progressOut, "", "utf8");
  }
  const runStartedAt = Date.now();
  let gamesCompleted = 0;
  let modelWinsSoFar = 0;

  const handleResult = (result: GateResult, workerId: number, gameSec: number) => {
    gamesCompleted += 1;
    if (result.modelWon) modelWinsSoFar += 1;
    const elapsedSec = (Date.now() - runStartedAt) / 1000;
    const runningWr = modelWinsSoFar / gamesCompleted;
    const etaSec = (elapsedSec / gamesCompleted) * (tasks.length - gamesCompleted);
    if (args.progressOut) {
      appendFileSync(args.progressOut, JSON.stringify({
        event: "game_completed",
        gameIndex: gamesCompleted,
        totalGames: tasks.length,
        seed: result.seed,
        modelSide: result.modelSide,
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
        workerId,
        ts: Date.now() / 1000,
      }) + "\n", "utf8");
    }
    const wTag = args.workers > 1 ? ` w${workerId}` : "";
    process.stderr.write(
      `[gate ${gamesCompleted}/${tasks.length}${wTag}] side=${result.modelSide} winner=${result.winner ?? "none"} wr=${runningWr.toFixed(3)} game=${gameSec.toFixed(1)}s eta=${(etaSec / 60).toFixed(1)}min\n`,
    );
  };

  let results: GateResult[];
  if (args.workers > 1 && tasks.length > 0) {
    results = await runOrchestrator(args, tasks, handleResult);
  } else {
    results = [];
    for (const task of tasks) {
      const gameStart = Date.now();
      const result = await runModelVsHeuristicGame(args, task.seed, task.side);
      results.push(result);
      handleResult(result, 0, (Date.now() - gameStart) / 1000);
    }
  }

  const summary = summarize(results);
  const failures: string[] = [];
  if (summary.games < args.minGames) failures.push(`games ${summary.games} < minGames ${args.minGames}`);
  if (summary.modelWinRate < args.minWinRate) failures.push(`winRate ${summary.modelWinRate} < minWinRate ${args.minWinRate}`);
  if (summary.wilson95.lower < args.minCiLower) failures.push(`wilsonLower ${summary.wilson95.lower} < minCiLower ${args.minCiLower}`);
  if (args.maxCiLower !== null && summary.wilson95.lower > args.maxCiLower) failures.push(`wilsonLower ${summary.wilson95.lower} > maxCiLower ${args.maxCiLower}`);
  if (args.requireZeroFallbacks && summary.heuristicFallbacks !== 0) failures.push(`heuristicFallbacks ${summary.heuristicFallbacks} != 0`);
  if (args.requireZeroNoOps && summary.selectedNoOps !== 0) failures.push(`selectedNoOps ${summary.selectedNoOps} != 0`);

  const passed = failures.length === 0;
  const status = args.expectFail
    ? (passed ? "FAIL_UNEXPECTED_PASS" : "PASS")
    : (passed ? "PASS" : "FAIL");
  const exitNonZero = status !== "PASS";
  const output = { status, failures, expectFail: args.expectFail, args, summary };
  if (args.manifestOut) {
    mkdirSync(dirname(args.manifestOut), { recursive: true });
    writeFileSync(args.manifestOut, JSON.stringify(withGitMetadata(output), null, 2) + "\n", "utf8");
  }
  console.log(JSON.stringify(output, null, 2));
  if (exitNonZero) process.exit(1);
}

function partitionTasks(tasks: WorkerTask[], workers: number): WorkerTask[][] {
  const sliceSize = Math.ceil(tasks.length / workers);
  const slices: WorkerTask[][] = [];
  for (let w = 0; w < workers; w += 1) {
    const slice = tasks.slice(w * sliceSize, (w + 1) * sliceSize);
    if (slice.length > 0) slices.push(slice);
  }
  return slices;
}

function buildWorkerArgv(parentArgv: string[]): string[] {
  const out: string[] = [];
  for (let i = 0; i < parentArgv.length; i += 1) {
    if (parentArgv[i] === "--workers") { i += 1; continue; }
    if (parentArgv[i] === "--progress-out") { i += 1; continue; }
    if (parentArgv[i] === "--manifest-out") { i += 1; continue; }
    out.push(parentArgv[i]!);
  }
  out.push("--worker-mode");
  return out;
}

async function runOrchestrator(
  args: Args,
  tasks: WorkerTask[],
  onResult: (result: GateResult, workerId: number, gameSec: number) => void,
): Promise<GateResult[]> {
  const slices = partitionTasks(tasks, args.workers);
  const workerArgv = buildWorkerArgv(process.argv.slice(2));
  const collected: GateResult[] = [];
  await Promise.all(slices.map((slice, workerId) => new Promise<void>((resolve, reject) => {
    const child: ChildProcess = fork(process.argv[1]!, workerArgv, {
      stdio: ["inherit", "inherit", "inherit", "ipc"],
    });
    let workerReady = false;
    let workerDone = false;
    child.on("message", (msg: unknown) => {
      const m = msg as { kind: string; result?: GateResult; gameSec?: number };
      if (m.kind === "ready") {
        workerReady = true;
        child.send({ kind: "tasks", workerId, tasks: slice });
      } else if (m.kind === "game_completed" && m.result) {
        collected.push(m.result);
        onResult(m.result, workerId, m.gameSec ?? 0);
      } else if (m.kind === "done") {
        workerDone = true;
      }
    });
    child.on("error", (err) => reject(err));
    child.on("exit", (code) => {
      if (!workerReady) {
        reject(new Error(`worker ${workerId} exited before becoming ready (code=${code})`));
      } else if (!workerDone) {
        reject(new Error(`worker ${workerId} exited before completing tasks (code=${code})`));
      } else if (code !== 0 && code !== null) {
        reject(new Error(`worker ${workerId} exited non-zero (code=${code})`));
      } else {
        resolve();
      }
    });
  })));
  return collected;
}

async function runWorker(args: Args): Promise<void> {
  if (!process.send) {
    process.stderr.write("evalGate worker has no IPC channel\n");
    process.exit(2);
    return;
  }
  process.send({ kind: "ready" });
  process.on("message", async (msg: unknown) => {
    const m = msg as { kind: string; workerId?: number; tasks?: WorkerTask[] };
    if (m.kind !== "tasks" || !m.tasks) return;
    try {
      for (const task of m.tasks) {
        const gameStart = Date.now();
        const result = await runModelVsHeuristicGame(args, task.seed, task.side);
        const gameSec = (Date.now() - gameStart) / 1000;
        process.send!({ kind: "game_completed", result, gameSec });
      }
      process.send!({ kind: "done" });
      setTimeout(() => process.exit(0), 25);
    } catch (err) {
      process.stderr.write(`worker error: ${(err as Error).message}\n`);
      process.exit(1);
    }
  });
}

function summarize(results: GateResult[]) {
  const modelWins = results.filter((result) => result.modelWon).length;
  const totalModelPoints = results.reduce((sum, result) => sum + result.points[result.modelSide], 0);
  const totalHeuristicPoints = results.reduce((sum, result) => sum + result.points[result.modelSide === "player" ? "opponent" : "player"], 0);
  return {
    games: results.length,
    modelWins,
    modelWinRate: results.length ? modelWins / results.length : 0,
    wilson95: wilsonInterval(modelWins, results.length),
    averageModelPoints: results.length ? totalModelPoints / results.length : 0,
    averageHeuristicPoints: results.length ? totalHeuristicPoints / results.length : 0,
    heuristicFallbacks: results.reduce((sum, result) => sum + result.heuristicFallbacks, 0),
    selectedNoOps: results.reduce((sum, result) => sum + result.selectedNoOps, 0),
    selectedExplicitPasses: results.reduce((sum, result) => sum + result.selectedExplicitPasses, 0),
    averageSelectedCandidateRank: averageSelectedCandidateRank(results),
    terminalReasons: countBy(results, (result) => result.terminalReason),
    byModelSide: {
      player: summarizeSide(results.filter((result) => result.modelSide === "player")),
      opponent: summarizeSide(results.filter((result) => result.modelSide === "opponent")),
    },
  };
}

function summarizeSide(results: GateResult[]) {
  const modelWins = results.filter((result) => result.modelWon).length;
  const totalModelPoints = results.reduce((sum, result) => sum + result.points[result.modelSide], 0);
  const totalHeuristicPoints = results.reduce((sum, result) => sum + result.points[result.modelSide === "player" ? "opponent" : "player"], 0);
  return {
    games: results.length,
    modelWins,
    modelWinRate: results.length ? modelWins / results.length : 0,
    wilson95: wilsonInterval(modelWins, results.length),
    averageModelPoints: results.length ? totalModelPoints / results.length : 0,
    averageHeuristicPoints: results.length ? totalHeuristicPoints / results.length : 0,
    heuristicFallbacks: results.reduce((sum, result) => sum + result.heuristicFallbacks, 0),
    selectedNoOps: results.reduce((sum, result) => sum + result.selectedNoOps, 0),
    selectedExplicitPasses: results.reduce((sum, result) => sum + result.selectedExplicitPasses, 0),
    averageSelectedCandidateRank: averageSelectedCandidateRank(results),
    terminalReasons: countBy(results, (result) => result.terminalReason),
  };
}

function averageSelectedCandidateRank(results: GateResult[]): number | null {
  const ranks = results.flatMap((result) => result.selectedCandidateRanks);
  if (ranks.length === 0) return null;
  return ranks.reduce((sum, rank) => sum + rank, 0) / ranks.length;
}

function wilsonInterval(successes: number, total: number) {
  if (total <= 0) return { lower: 0, upper: 0 };
  const z = 1.96;
  const phat = successes / total;
  const denominator = 1 + z * z / total;
  const center = (phat + z * z / (2 * total)) / denominator;
  const halfWidth = z * Math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denominator;
  return {
    lower: Math.max(0, center - halfWidth),
    upper: Math.min(1, center + halfWidth),
  };
}

function countBy<T>(items: T[], keyOf: (item: T) => string): Record<string, number> {
  return items.reduce<Record<string, number>>((counts, item) => {
    const key = keyOf(item);
    counts[key] = (counts[key] ?? 0) + 1;
    return counts;
  }, {});
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
    games: Number(get("--games", "250")),
    seedStart: Number(get("--seed-start", "8000")),
    maxSteps: Number(get("--max-steps", "500")),
    modelSide,
    details: false,
    selection: parseSelection(get("--selection", "policy")),
    rolloutSteps: Number(get("--rollout-steps", "500")),
    searchDepth: Number(get("--search-depth", "2")),
    searchTopK: Number(get("--search-top-k", "4")),
    searchSamples: Number(get("--search-samples", "1")),
    ranker: parseRanker(get("--ranker", "heuristic")),
    decisionTraceOut: get("--decision-trace-out", ""),
    plannerTopK: Number(get("--planner-top-k", get("--search-top-k", "4"))),
    plannerMaxSequences: Number(get("--planner-max-sequences", "64")),
    plannerMaxDepth: Number(get("--planner-max-depth", "8")),
    cycleWindow: Number(get("--cycle-window", "8")),
    cycleMinVisits: Number(get("--cycle-min-visits", "3")),
    plannerCrnSamples: Number(get("--planner-crn-samples", "3")),
    plannerLeafAggregate: parsePlannerLeafAggregate(get("--planner-leaf-aggregate", "mean")),
    plannerFirstActionAggregate: parsePlannerFirstActionAggregate(get("--planner-first-action-aggregate", "max")),
    rolloutCrnSamples: Number(get("--rollout-crn-samples", "1")),
    traceTeacher: parseTraceTeacher(get("--trace-teacher", "none")),
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
    minGames: Number(get("--min-games", "500")),
    minWinRate: Number(get("--min-win-rate", "0")),
    minCiLower: Number(get("--min-ci-lower", "0")),
    maxCiLower: argv.includes("--max-ci-lower") ? Number(get("--max-ci-lower", "1")) : null,
    requireZeroFallbacks: !argv.includes("--allow-fallbacks"),
    requireZeroNoOps: !argv.includes("--allow-no-ops"),
    requireManifest: argv.includes("--require-manifest"),
    expectFail: argv.includes("--expect-fail"),
    manifestOut: get("--manifest-out", ""),
    workers: Math.max(1, Number(get("--workers", "1"))),
    workerMode: argv.includes("--worker-mode"),
  };
}

function parseSelection(raw: string): EvaluateModelArgs["selection"] {
  if (raw === "baseline" || raw === "inverted-baseline" || raw === "value" || raw === "rollout" || raw === "search" || raw === "planner" || raw === "mcts") return raw;
  return "policy";
}

function parseMctsLeaf(raw: string): "value-head" | "rollout" {
  if (raw === "value-head" || raw === "rollout") return raw;
  throw new Error(`--mcts-leaf must be value-head or rollout, got ${raw}`);
}

function parseOpponentSelection(raw: string): EvaluateModelArgs["opponentSelection"] {
  if (raw === "policy" || raw === "mcts") return raw;
  return "rule";
}

function parseMctsPrior(raw: string): "uniform" | "policy" {
  if (raw === "uniform" || raw === "policy") return raw;
  throw new Error(`--mcts-prior must be uniform or policy, got ${raw}`);
}

function parseRanker(raw: string): CandidateRankerMode {
  if (raw === "phase-diverse" || raw === "epsilon") return raw;
  return "heuristic";
}

function parsePlannerLeafAggregate(raw: string): "mean" | "max" | "median" {
  if (raw === "max" || raw === "median") return raw;
  return "mean";
}

function parsePlannerFirstActionAggregate(raw: string): "max" | "mean" {
  if (raw === "mean") return raw;
  return "max";
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
