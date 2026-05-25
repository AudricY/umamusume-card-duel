import { parseTraceTeacherList, runModelVsHeuristicGame, type EvaluateModelArgs } from "./evaluateModelVsHeuristic";
import type { SideId } from "../../../shared/src/types";
import type { CandidateRankerMode } from "./candidateRanker";
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { fork, type ChildProcess } from "node:child_process";
import { withGitMetadata } from "./manifest";
import { MCTS_WORK_STEALING_ENABLED, WORK_STEALING_PREFETCH } from "./workStealing";

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
  // R16-TD 3a corpus harness: when --decision-trace-out is set the orchestrator
  // owns the trace file. Workers populate `result.decisionTraces` (because
  // they also see `--decision-trace-out`) but never touch disk; the
  // orchestrator buffers rows by taskIndex and flushes in ascending order
  // after all tasks complete. This gives deterministic output independent of
  // worker count or completion interleaving and matches the slot-ordered
  // GateResult contract already provided by the work-stealing path.
  if (args.decisionTraceOut) {
    mkdirSync(dirname(args.decisionTraceOut), { recursive: true });
    writeFileSync(args.decisionTraceOut, "", "utf8");
  }
  const traceBuffer = new Map<number, unknown[]>();
  const runStartedAt = Date.now();
  let gamesCompleted = 0;
  let modelWinsSoFar = 0;

  const handleResult = (result: GateResult, workerId: number, gameSec: number, taskIndex: number) => {
    gamesCompleted += 1;
    if (result.modelWon) modelWinsSoFar += 1;
    if (args.decisionTraceOut && result.decisionTraces && result.decisionTraces.length > 0) {
      traceBuffer.set(taskIndex, result.decisionTraces as unknown[]);
    }
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
    for (let taskIndex = 0; taskIndex < tasks.length; taskIndex += 1) {
      const task = tasks[taskIndex]!;
      const gameStart = Date.now();
      const result = await runModelVsHeuristicGame(args, task.seed, task.side);
      results.push(result);
      handleResult(result, 0, (Date.now() - gameStart) / 1000, taskIndex);
    }
  }

  // Flush buffered traces in ascending taskIndex order. Determinism contract:
  // the row stream is identical across --workers values because (a) taskIndex
  // is assigned deterministically from (seedStart, modelSide) pairing in the
  // task-construction loop above and (b) per-game RNG seeding ignores worker
  // identity / completion order. The work-stealing dispatch is verified
  // deterministic by r12_workstealing_determinism_gate.py.
  if (args.decisionTraceOut && traceBuffer.size > 0) {
    const orderedIndexes = Array.from(traceBuffer.keys()).sort((a, b) => a - b);
    const lines: string[] = [];
    for (const idx of orderedIndexes) {
      for (const row of traceBuffer.get(idx)!) lines.push(JSON.stringify(row));
    }
    if (lines.length > 0) appendFileSync(args.decisionTraceOut, lines.join("\n") + "\n", "utf8");
  }

  const summary = summarize(results);
  const failures: string[] = [];
  if (summary.games < args.minGames) failures.push(`games ${summary.games} < minGames ${args.minGames}`);
  if (summary.modelWinRate < args.minWinRate) failures.push(`winRate ${summary.modelWinRate} < minWinRate ${args.minWinRate}`);
  if (summary.wilson95.lower < args.minCiLower) failures.push(`wilsonLower ${summary.wilson95.lower} < minCiLower ${args.minCiLower}`);
  if (args.maxCiLower !== null && summary.wilson95.lower > args.maxCiLower) failures.push(`wilsonLower ${summary.wilson95.lower} > maxCiLower ${args.maxCiLower}`);
  if (args.requireZeroFallbacks && summary.heuristicFallbacks !== 0) failures.push(`heuristicFallbacks ${summary.heuristicFallbacks} != 0`);
  if (args.requireZeroNoOps && summary.selectedNoOps !== 0) failures.push(`selectedNoOps ${summary.selectedNoOps} != 0`);

  // R16-TD 3a corpus harness: when running as a corpus generator
  // (--relabel-mcts), gate floors are informational only. A weak model may
  // legitimately lose to the rule bot during early corpus collection; the
  // failure list is still reported in the summary so an orchestrator can
  // inspect it, but the process always exits 0 in this mode so downstream
  // scripts don't have to fork on win-rate.
  const corpusMode = args.relabelMcts;
  const passed = corpusMode || failures.length === 0;
  const status = args.expectFail
    ? (passed ? "FAIL_UNEXPECTED_PASS" : "PASS")
    : (passed ? "PASS" : "FAIL");
  const exitNonZero = status !== "PASS";
  const output = { status, failures, expectFail: args.expectFail, corpusMode, args, summary };
  if (args.manifestOut) {
    mkdirSync(dirname(args.manifestOut), { recursive: true });
    writeFileSync(args.manifestOut, JSON.stringify(withGitMetadata(output), null, 2) + "\n", "utf8");
  }
  console.log(JSON.stringify(output, null, 2));
  if (exitNonZero) process.exit(1);
}

function partitionTasks(tasks: WorkerTask[], workers: number): Array<Array<WorkerTask & { taskIndex: number }>> {
  const sliceSize = Math.ceil(tasks.length / workers);
  const indexed: Array<WorkerTask & { taskIndex: number }> = tasks.map((task, taskIndex) => ({ ...task, taskIndex }));
  const slices: Array<Array<WorkerTask & { taskIndex: number }>> = [];
  for (let w = 0; w < workers; w += 1) {
    const slice = indexed.slice(w * sliceSize, (w + 1) * sliceSize);
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
  onResult: (result: GateResult, workerId: number, gameSec: number, taskIndex: number) => void,
): Promise<GateResult[]> {
  if (MCTS_WORK_STEALING_ENABLED) {
    return runOrchestratorWorkStealing(args, tasks, onResult);
  }
  return runOrchestratorStatic(args, tasks, onResult);
}

// Static contiguous chunking. Kept verbatim behind UMA_MCTS_WORK_STEALING=0.
async function runOrchestratorStatic(
  args: Args,
  tasks: WorkerTask[],
  onResult: (result: GateResult, workerId: number, gameSec: number, taskIndex: number) => void,
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
      const m = msg as { kind: string; result?: GateResult; gameSec?: number; taskIndex?: number };
      if (m.kind === "ready") {
        workerReady = true;
        child.send({ kind: "tasks", workerId, tasks: slice });
      } else if (m.kind === "game_completed" && m.result) {
        collected.push(m.result);
        onResult(m.result, workerId, m.gameSec ?? 0, m.taskIndex ?? -1);
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

// Work-stealing dispatch (docs/ai-research/scoping/r12-selfplay-gate-throughput.md).
// Shared task-index queue: seed each worker with WORK_STEALING_PREFETCH tasks,
// refill one on every completion. Workers spun = min(tasks, workers) (fixes
// the dark-cores bug). Results are stored into a slot array indexed by task
// index, NOT push-order, so the returned GateResult[] is deterministic
// regardless of which worker finished when. Per-game RNG is seeded solely
// from (seed, modelSide) (evaluateModelVsHeuristic.ts `:modelSide`), so
// dispatch order cannot perturb trajectories or gate outcomes.
async function runOrchestratorWorkStealing(
  args: Args,
  tasks: WorkerTask[],
  onResult: (result: GateResult, workerId: number, gameSec: number, taskIndex: number) => void,
): Promise<GateResult[]> {
  const workerArgv = buildWorkerArgv(process.argv.slice(2));
  const workerCount = Math.min(tasks.length, args.workers);
  const slots: (GateResult | null)[] = new Array(tasks.length).fill(null);
  let nextTaskIndex = 0;

  await Promise.all(Array.from({ length: workerCount }, (_unused, workerId) => new Promise<void>((resolve, reject) => {
    const child: ChildProcess = fork(process.argv[1]!, workerArgv, {
      stdio: ["inherit", "inherit", "inherit", "ipc"],
    });
    let workerReady = false;
    let workerDone = false;
    let inFlight = 0;

    const dispatchNext = (): void => {
      if (nextTaskIndex >= tasks.length) {
        if (inFlight === 0) child.send({ kind: "no_more_tasks" });
        return;
      }
      const taskIndex = nextTaskIndex;
      nextTaskIndex += 1;
      inFlight += 1;
      const task = tasks[taskIndex]!;
      child.send({ kind: "task", taskIndex, seed: task.seed, side: task.side });
    };

    child.on("message", (msg: unknown) => {
      const m = msg as { kind: string; result?: GateResult; gameSec?: number; taskIndex?: number };
      if (m.kind === "ready") {
        workerReady = true;
        for (let k = 0; k < WORK_STEALING_PREFETCH; k += 1) dispatchNext();
      } else if (m.kind === "game_completed" && m.result && m.taskIndex !== undefined) {
        slots[m.taskIndex] = m.result;
        inFlight -= 1;
        onResult(m.result, workerId, m.gameSec ?? 0, m.taskIndex);
        dispatchNext();
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

  // Return results in ascending task-index order (deterministic, independent
  // of completion order). summarize() is itself order-independent, but slot
  // ordering keeps the contract identical to the static path.
  return slots.filter((r): r is GateResult => r !== null);
}

async function runWorker(args: Args): Promise<void> {
  if (!process.send) {
    process.stderr.write("evalGate worker has no IPC channel\n");
    process.exit(2);
    return;
  }
  if (!MCTS_WORK_STEALING_ENABLED) {
    process.send({ kind: "ready" });
    process.on("message", async (msg: unknown) => {
      const m = msg as { kind: string; workerId?: number; tasks?: Array<WorkerTask & { taskIndex?: number }> };
      if (m.kind !== "tasks" || !m.tasks) return;
      try {
        for (const task of m.tasks) {
          const gameStart = Date.now();
          const result = await runModelVsHeuristicGame(args, task.seed, task.side);
          const gameSec = (Date.now() - gameStart) / 1000;
          // Forward the globally-assigned taskIndex (added in the orchestrator's
          // partitionTasks step) so the orchestrator can sort trace rows
          // deterministically across the static-chunking path too.
          process.send!({ kind: "game_completed", result, gameSec, taskIndex: task.taskIndex });
        }
        process.send!({ kind: "done" });
        setTimeout(() => process.exit(0), 25);
      } catch (err) {
        process.stderr.write(`worker error: ${(err as Error).message}\n`);
        process.exit(1);
      }
    });
    return;
  }

  // Work-stealing: the orchestrator prefetches WORK_STEALING_PREFETCH tasks
  // per worker, so two `task` messages can sit in the mailbox at once. They
  // MUST NOT run concurrently — a single game owns the process-wide
  // AsyncLocalStorage RNG context (rngAsyncStore.ts) and the shared
  // serve_onnx client; interleaving breaks bit-identity. Buffer prefetched
  // tasks in an internal FIFO and drain STRICTLY SEQUENTIALLY (exactly one
  // game in flight), matching the static path's per-worker `for`-loop.
  type Pend = { taskIndex: number; seed: string; side: SideId };
  const pending: Pend[] = [];
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
        const gameStart = Date.now();
        const result = await runModelVsHeuristicGame(args, task.seed, task.side);
        const gameSec = (Date.now() - gameStart) / 1000;
        process.send!({ kind: "game_completed", result, gameSec, taskIndex: task.taskIndex });
      }
    } catch (err) {
      process.stderr.write(`worker error: ${(err as Error).message}\n`);
      process.exit(1);
    }
    draining = false;
    if (noMoreTasks && pending.length === 0) finish();
  };
  process.send({ kind: "ready" });
  process.on("message", (msg: unknown) => {
    const m = msg as { kind: string; taskIndex?: number; seed?: string; side?: SideId };
    if (m.kind === "no_more_tasks") {
      noMoreTasks = true;
      if (!draining && pending.length === 0) finish();
      return;
    }
    if (m.kind !== "task" || m.seed === undefined || m.side === undefined || m.taskIndex === undefined) return;
    pending.push({ taskIndex: m.taskIndex, seed: m.seed, side: m.side });
    void drain();
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
    // R16-TD 3a corpus harness: thread through the relabel-mcts payload so
    // evalGate can host the production corpus generator (was hardcoded false
    // pre-spike). When set, gate floors are treated as informational so a
    // weak-model corpus run does not exit non-zero — see `main()` below.
    relabelMcts: argv.includes("--relabel-mcts"),
    relabelStateSource: get("--relabel-state-source", "rule-bot-mirror"),
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
    mctsTwoSided: argv.includes("--mcts-two-sided"),
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

// R7 step 2: accept either a single teacher (back-compat with R3/R4/R6/R15.S1)
// or a comma-separated list ("rollout,search,planner") that fans out into a
// multi-teacher mixture target on each trace row. Delegates to the shared
// parser in evaluateModelVsHeuristic.ts so the CLI surface stays consistent
// across the two entry points.
function parseTraceTeacher(raw: string): EvaluateModelArgs["traceTeacher"] {
  return parseTraceTeacherList(raw);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
