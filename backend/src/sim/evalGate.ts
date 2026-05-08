import { runModelVsHeuristicGame, type EvaluateModelArgs } from "./evaluateModelVsHeuristic";
import type { SideId } from "../../../shared/src/types";
import type { CandidateRankerMode } from "./candidateRanker";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
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
};

class EvalGateError extends Error {
  constructor(public readonly errorCode: string, message: string) {
    super(message);
    this.name = "EvalGateError";
  }
}

type GateResult = Awaited<ReturnType<typeof runModelVsHeuristicGame>>;

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.requireManifest && !args.manifestOut) {
    const error = { error: "EvalGateError", code: "missing_manifest_path", message: "--require-manifest demands --manifest-out <path>" };
    console.error(JSON.stringify(error));
    process.exit(2);
  }
  const sides: SideId[] = args.modelSide === "both" ? ["player", "opponent"] : [args.modelSide];
  const results: GateResult[] = [];
  for (const side of sides) {
    for (let index = 0; index < args.games; index += 1) {
      results.push(await runModelVsHeuristicGame(args, String(args.seedStart + index), side));
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
    plannerCrnSamples: Number(get("--planner-crn-samples", "3")),
    plannerLeafAggregate: parsePlannerLeafAggregate(get("--planner-leaf-aggregate", "mean")),
    plannerFirstActionAggregate: parsePlannerFirstActionAggregate(get("--planner-first-action-aggregate", "max")),
    traceTeacher: parseTraceTeacher(get("--trace-teacher", "none")),
    minGames: Number(get("--min-games", "500")),
    minWinRate: Number(get("--min-win-rate", "0")),
    minCiLower: Number(get("--min-ci-lower", "0")),
    maxCiLower: argv.includes("--max-ci-lower") ? Number(get("--max-ci-lower", "1")) : null,
    requireZeroFallbacks: !argv.includes("--allow-fallbacks"),
    requireZeroNoOps: !argv.includes("--allow-no-ops"),
    requireManifest: argv.includes("--require-manifest"),
    expectFail: argv.includes("--expect-fail"),
    manifestOut: get("--manifest-out", ""),
  };
}

function parseSelection(raw: string): EvaluateModelArgs["selection"] {
  if (raw === "baseline" || raw === "inverted-baseline" || raw === "value" || raw === "rollout" || raw === "search" || raw === "planner") return raw;
  return "policy";
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
