import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import {
  advanceOpponentTurnStep,
  advancePlayerAiTurnStep,
} from "../../../frontend/src/game/engine";
import {
  enumerateLegalAiActions,
  chooseHighestScoredAction,
} from "../../../frontend/src/game/engine/ai-policy/actions";
import { createSeededRng, withRng, type Rng } from "../../../frontend/src/game/engine/core/random";
import type { GameState, SideId } from "../../../shared/src/types";
import { stateFingerprint } from "./stateFingerprint";
import { withGitMetadata } from "./manifest";
import {
  runModelVsHeuristicGame,
  setupAiVsAiGame,
  getForcedAttackCoinResults,
  type EvaluateModelArgs,
} from "./evaluateModelVsHeuristic";

type Selection = EvaluateModelArgs["selection"];

type RebaselineArgs = {
  games: number;
  seedStart: number;
  maxSteps: number;
  rolloutSteps: number;
  searchDepth: number;
  searchTopK: number;
  searchSamples: number;
  plannerTopK: number;
  plannerMaxSequences: number;
  plannerMaxDepth: number;
  plannerCrnSamples: number;
  plannerLeafAggregate: "mean" | "max" | "median";
  plannerFirstActionAggregate: "max" | "mean";
  rolloutCrnSamples: number;
  cycleWindow: number;
  modelUrl: string;
  methods: string;
  outDir: string;
};

type MethodKey =
  | "rule-mirror"
  | "baseline"
  | "inverted-baseline"
  | "rollout"
  | "search"
  | "planner"
  | "policy";

const ALL_METHODS: MethodKey[] = [
  "rule-mirror",
  "baseline",
  "inverted-baseline",
  "rollout",
  "search",
  "planner",
  "policy",
];

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const requested = args.methods === "all"
    ? ALL_METHODS.filter((m) => m !== "policy")
    : (args.methods.split(",").filter(Boolean) as MethodKey[]);

  mkdirSync(args.outDir, { recursive: true });
  const summaries: Array<{ method: MethodKey; summary: ReturnType<typeof summarizeMethod> }> = [];

  for (const method of requested) {
    const summary = method === "rule-mirror"
      ? await runRuleMirror(args)
      : await runMethodAsModel(args, method);
    summaries.push({ method, summary });
    const manifest = withGitMetadata({ method, args, summary });
    writeFileSync(join(args.outDir, `${method}.json`), JSON.stringify(manifest, null, 2) + "\n", "utf8");
  }

  const table = renderTable(summaries);
  writeFileSync(join(args.outDir, "rebaseline.json"), JSON.stringify(withGitMetadata({ args, summaries }), null, 2) + "\n", "utf8");
  writeFileSync(join(args.outDir, "rebaseline.md"), table, "utf8");
  console.log(table);
}

async function runMethodAsModel(args: RebaselineArgs, method: MethodKey) {
  const selection = method as Selection;
  const evaluateArgs: EvaluateModelArgs = {
    modelUrl: args.modelUrl,
    games: args.games,
    seedStart: args.seedStart,
    maxSteps: args.maxSteps,
    modelSide: "both",
    details: false,
    selection,
    rolloutSteps: args.rolloutSteps,
    searchDepth: args.searchDepth,
    searchTopK: args.searchTopK,
    searchSamples: args.searchSamples,
    ranker: "heuristic",
    decisionTraceOut: null,
    manifestOut: null,
    traceTeacher: "none",
    plannerTopK: args.plannerTopK,
    plannerMaxSequences: args.plannerMaxSequences,
    plannerMaxDepth: args.plannerMaxDepth,
    cycleWindow: args.cycleWindow,
    plannerCrnSamples: args.plannerCrnSamples,
    plannerLeafAggregate: args.plannerLeafAggregate,
    plannerFirstActionAggregate: args.plannerFirstActionAggregate,
    rolloutCrnSamples: args.rolloutCrnSamples,
  };
  const sides: SideId[] = ["player", "opponent"];
  const results = [];
  for (const side of sides) {
    for (let index = 0; index < args.games; index += 1) {
      results.push(await runModelVsHeuristicGame(evaluateArgs, String(args.seedStart + index), side));
    }
  }
  return summarizeMethod(results.map((r) => ({
    modelWon: r.modelWon,
    modelSide: r.modelSide,
    points: r.points,
    heuristicFallbacks: r.heuristicFallbacks,
    selectedNoOps: r.selectedNoOps,
    selectedExplicitPasses: r.selectedExplicitPasses,
    selectedRanks: r.selectedCandidateRanks,
    terminalReason: r.terminalReason,
  })));
}

async function runRuleMirror(args: RebaselineArgs) {
  const sides: SideId[] = ["player", "opponent"];
  const results = [];
  for (const side of sides) {
    for (let index = 0; index < args.games; index += 1) {
      const seed = String(args.seedStart + index);
      results.push(runRuleMirrorGame(seed, side, args));
    }
  }
  return summarizeMethod(results);
}

function runRuleMirrorGame(seed: string, scoringSide: SideId, args: RebaselineArgs) {
  const rng = createSeededRng(`${seed}:${scoringSide}`, "rule-mirror");
  return withRng(rng, () => runRuleMirrorGameWithRng(seed, scoringSide, args, rng));
}

function runRuleMirrorGameWithRng(seed: string, scoringSide: SideId, args: RebaselineArgs, rng: Rng) {
  let state = setupAiVsAiGame();
  let terminalReason: "gameOver" | "maxSteps" | "stalled" | "cycleStalled" = "maxSteps";
  const recentHashes: string[] = [];
  for (let step = 0; step < args.maxSteps; step += 1) {
    if (state.gameOver) {
      terminalReason = "gameOver";
      break;
    }
    const before = stateFingerprint(state);
    const sideId = state.currentSide === "player" || state.currentSide === "opponent" ? state.currentSide : "player";
    const forced = getForcedAttackCoinResults(state, rng);
    state = sideId === "player"
      ? advancePlayerAiTurnStep(state, forced, rng.next)
      : advanceOpponentTurnStep(state, forced, rng.next);
    const after = stateFingerprint(state);
    if (after === before) {
      terminalReason = "stalled";
      break;
    }
    if (args.cycleWindow > 0) {
      if (recentHashes.includes(after)) {
        terminalReason = "cycleStalled";
        break;
      }
      recentHashes.push(after);
      if (recentHashes.length > args.cycleWindow) recentHashes.shift();
    }
  }
  return {
    modelWon: state.winner === scoringSide,
    modelSide: scoringSide,
    points: { player: state.sides.player.points, opponent: state.sides.opponent.points },
    heuristicFallbacks: 0,
    selectedNoOps: 0,
    selectedExplicitPasses: 0,
    selectedRanks: [] as number[],
    terminalReason,
  };
}

function summarizeMethod(results: Array<{
  modelWon: boolean;
  modelSide: SideId;
  points: Record<SideId, number>;
  heuristicFallbacks: number;
  selectedNoOps: number;
  selectedExplicitPasses: number;
  selectedRanks: number[];
  terminalReason: string;
}>) {
  const wins = results.filter((r) => r.modelWon).length;
  const games = results.length;
  const wilson = wilson95(wins, games);
  const playerRows = results.filter((r) => r.modelSide === "player");
  const opponentRows = results.filter((r) => r.modelSide === "opponent");
  const ranks = results.flatMap((r) => r.selectedRanks);
  return {
    games,
    wins,
    winRate: games ? wins / games : 0,
    wilson95: wilson,
    averageModelPoints: games ? results.reduce((sum, r) => sum + r.points[r.modelSide], 0) / games : 0,
    averageOpponentPoints: games ? results.reduce((sum, r) => sum + r.points[r.modelSide === "player" ? "opponent" : "player"], 0) / games : 0,
    heuristicFallbacks: results.reduce((sum, r) => sum + r.heuristicFallbacks, 0),
    selectedNoOps: results.reduce((sum, r) => sum + r.selectedNoOps, 0),
    selectedExplicitPasses: results.reduce((sum, r) => sum + r.selectedExplicitPasses, 0),
    averageSelectedRank: ranks.length ? ranks.reduce((s, v) => s + v, 0) / ranks.length : null,
    terminalReasons: countBy(results.map((r) => r.terminalReason)),
    bySide: {
      player: sideSummary(playerRows),
      opponent: sideSummary(opponentRows),
    },
  };
}

function sideSummary(results: Array<{ modelWon: boolean; modelSide: SideId; points: Record<SideId, number> }>) {
  const wins = results.filter((r) => r.modelWon).length;
  const games = results.length;
  return {
    games,
    wins,
    winRate: games ? wins / games : 0,
    wilson95: wilson95(wins, games),
  };
}

function wilson95(successes: number, total: number) {
  if (total <= 0) return { lower: 0, upper: 0 };
  const z = 1.96;
  const phat = successes / total;
  const denom = 1 + z * z / total;
  const center = (phat + z * z / (2 * total)) / denom;
  const half = z * Math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denom;
  return { lower: Math.max(0, center - half), upper: Math.min(1, center + half) };
}

function countBy(items: string[]): Record<string, number> {
  return items.reduce<Record<string, number>>((acc, item) => {
    acc[item] = (acc[item] ?? 0) + 1;
    return acc;
  }, {});
}

function renderTable(summaries: Array<{ method: MethodKey; summary: ReturnType<typeof summarizeMethod> }>): string {
  const header = "| Method | Games | WR | Wilson95 | Player WR | Opp WR | Pts | Opp Pts | Fallbacks | NoOps | Passes | Avg Rank | Terminal |";
  const sep    = "|--------|-------|----|----------|-----------|--------|-----|---------|-----------|-------|--------|----------|----------|";
  const lines = summaries.map(({ method, summary }) => {
    const wilson = `[${(summary.wilson95.lower * 100).toFixed(1)}, ${(summary.wilson95.upper * 100).toFixed(1)}]`;
    const playerWr = `${(summary.bySide.player.winRate * 100).toFixed(1)}%`;
    const oppWr = `${(summary.bySide.opponent.winRate * 100).toFixed(1)}%`;
    const rank = summary.averageSelectedRank == null ? "—" : summary.averageSelectedRank.toFixed(2);
    const terminal = Object.entries(summary.terminalReasons).map(([k, v]) => `${k}=${v}`).join(" ");
    return `| ${method} | ${summary.games} | ${(summary.winRate * 100).toFixed(1)}% | ${wilson} | ${playerWr} | ${oppWr} | ${summary.averageModelPoints.toFixed(2)} | ${summary.averageOpponentPoints.toFixed(2)} | ${summary.heuristicFallbacks} | ${summary.selectedNoOps} | ${summary.selectedExplicitPasses} | ${rank} | ${terminal} |`;
  });
  return [header, sep, ...lines, ""].join("\n");
}

function parseArgs(argv: string[]): RebaselineArgs {
  const get = (name: string, fallback: string) => {
    const index = argv.indexOf(name);
    return index >= 0 ? argv[index + 1] ?? fallback : fallback;
  };
  return {
    games: Number(get("--games", "30")),
    seedStart: Number(get("--seed-start", "8000")),
    maxSteps: Number(get("--max-steps", "500")),
    rolloutSteps: Number(get("--rollout-steps", "500")),
    searchDepth: Number(get("--search-depth", "2")),
    searchTopK: Number(get("--search-top-k", "8")),
    searchSamples: Number(get("--search-samples", "1")),
    plannerTopK: Number(get("--planner-top-k", "4")),
    plannerMaxSequences: Number(get("--planner-max-sequences", "64")),
    plannerMaxDepth: Number(get("--planner-max-depth", "8")),
    plannerCrnSamples: Number(get("--planner-crn-samples", "3")),
    plannerLeafAggregate: parsePlannerLeafAggregate(get("--planner-leaf-aggregate", "mean")),
    plannerFirstActionAggregate: parsePlannerFirstActionAggregate(get("--planner-first-action-aggregate", "max")),
    rolloutCrnSamples: Number(get("--rollout-crn-samples", "1")),
    cycleWindow: Number(get("--cycle-window", "8")),
    modelUrl: get("--model-url", "http://127.0.0.1:8765"),
    methods: get("--methods", "rule-mirror,baseline,inverted-baseline,rollout,search,planner"),
    outDir: get("--out-dir", "runs/rebaseline"),
  };
}

function parsePlannerLeafAggregate(raw: string): "mean" | "max" | "median" {
  if (raw === "max" || raw === "median") return raw;
  return "mean";
}

function parsePlannerFirstActionAggregate(raw: string): "max" | "mean" {
  if (raw === "mean") return raw;
  return "max";
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
