import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { withGitMetadata } from "../manifest";

type BehaviorPolicySnapshot = {
  kind: string;
  temperature: number;
  actionLogProbs: number[];
  actionProbs?: number[];
  selectedLogProb: number | null;
};

type DecisionTraceRow = {
  schemaVersion: 1;
  source: "model-visited";
  seed: string;
  modelSide: "player" | "opponent";
  step: number;
  sideId: "player" | "opponent";
  selection: string;
  observation: { opponent: { handCardIds?: unknown; deck?: unknown; hand?: unknown } } & Record<string, unknown>;
  legalActions: Array<{ id: string }>;
  selectedActionId: string;
  selectedActionIndex: number;
  selectedOriginalRank?: number;
  heuristicSelectedActionId: string;
  heuristicSelectedActionIndex: number;
  fallback: boolean;
  behaviorPolicy?: BehaviorPolicySnapshot;
  teacher?: {
    selection: "rollout" | "search" | "planner";
    selectedActionId: string;
    selectedActionIndex: number;
    selectedOriginalRank?: number;
  };
  result: { winner: "player" | "opponent" | null; modelWon: boolean; points: Record<string, number>; terminalReason: string } | null;
};

type RelabelArgs = {
  in: string;
  out: string;
  source: string;
  labelSource: string;
  manifestOut: string | null;
  dropFallbacks: boolean;
};

function main() {
  const args = parseArgs(process.argv.slice(2));
  const rows = readJsonl<DecisionTraceRow>(args.in);
  let kept = 0;
  let skippedNoTeacher = 0;
  let skippedFallback = 0;
  let outOfRange = 0;
  let leakDetected = 0;
  const teacherByName: Record<string, number> = {};

  mkdirSync(dirname(args.out), { recursive: true });
  const out = [] as string[];

  for (const row of rows) {
    if (!row.teacher) {
      skippedNoTeacher += 1;
      continue;
    }
    if (args.dropFallbacks && row.fallback) {
      skippedFallback += 1;
      continue;
    }
    const targetIndex = row.teacher.selectedActionIndex;
    if (targetIndex < 0 || targetIndex >= row.legalActions.length) {
      outOfRange += 1;
      continue;
    }
    const leakFields = detectLeakFields(row);
    if (leakFields.length > 0) {
      leakDetected += 1;
      throw new Error(`Hidden-info leak in row seed=${row.seed} step=${row.step}: ${leakFields.join(",")}`);
    }
    teacherByName[row.teacher.selection] = (teacherByName[row.teacher.selection] ?? 0) + 1;
    const result = row.result ?? { winner: null, modelWon: false, points: { player: 0, opponent: 0 }, terminalReason: "unknown" };
    const trainingRow: Record<string, unknown> = {
      schemaVersion: 1,
      source: args.source,
      labelSource: args.labelSource,
      relabeledFrom: { source: row.source, selection: row.selection },
      teacherSelection: row.teacher.selection,
      episodeId: `${row.seed}:${row.modelSide}`,
      seed: row.seed,
      modelSide: row.modelSide,
      step: row.step,
      sideId: row.sideId,
      observation: row.observation,
      legalActions: row.legalActions,
      selectedActionId: row.teacher.selectedActionId,
      selectedActionIndex: targetIndex,
      selectedOriginalRank: row.teacher.selectedOriginalRank,
      heuristicSelectedActionId: row.heuristicSelectedActionId,
      heuristicSelectedActionIndex: row.heuristicSelectedActionIndex,
      modelChoseActionId: row.selectedActionId,
      modelChoseActionIndex: row.selectedActionIndex,
      modelFallback: row.fallback,
      result: { winner: result.winner, points: result.points, terminalReason: result.terminalReason },
    };
    // Item 18: forward the behavior-policy snapshot from the trace row into
    // the relabeled training row so PPO can recover importance ratios on
    // the warm-start data. Only present when the underlying selection was
    // policy/value (i.e. the model server was consulted at decision time).
    if (row.behaviorPolicy) trainingRow.behaviorPolicy = row.behaviorPolicy;
    out.push(JSON.stringify(trainingRow));
    kept += 1;
  }

  writeFileSync(args.out, out.join("\n") + (out.length ? "\n" : ""), "utf8");

  const summary = {
    inputRows: rows.length,
    kept,
    skippedNoTeacher,
    skippedFallback,
    outOfRange,
    leakDetected,
    teacherSelections: teacherByName,
    inputPath: args.in,
    outputPath: args.out,
  };

  if (args.manifestOut) {
    mkdirSync(dirname(args.manifestOut), { recursive: true });
    writeFileSync(args.manifestOut, JSON.stringify(withGitMetadata({ args, summary }), null, 2) + "\n", "utf8");
  }

  console.log(JSON.stringify({ status: "PASS", ...summary }, null, 2));
}

function detectLeakFields(row: DecisionTraceRow): string[] {
  const violations: string[] = [];
  const opponent = row.observation?.opponent ?? {};
  if ((opponent as { handCardIds?: unknown }).handCardIds !== undefined) violations.push("opponent.handCardIds");
  if ((opponent as { hand?: unknown }).hand !== undefined) violations.push("opponent.hand");
  if ((opponent as { deck?: unknown }).deck !== undefined) violations.push("opponent.deck");
  return violations;
}

function readJsonl<T>(path: string): T[] {
  const raw = readFileSync(path, "utf8").trim();
  if (!raw) return [];
  return raw.split("\n").filter(Boolean).map((line) => JSON.parse(line) as T);
}

function parseArgs(argv: string[]): RelabelArgs {
  const get = (name: string, fallback: string) => {
    const index = argv.indexOf(name);
    return index >= 0 ? argv[index + 1] ?? fallback : fallback;
  };
  return {
    in: get("--in", ""),
    out: get("--out", ""),
    source: get("--source", "model-visited-relabeled"),
    labelSource: get("--label-source", "teacher-relabeled"),
    manifestOut: get("--manifest-out", "") || null,
    dropFallbacks: !argv.includes("--keep-fallbacks"),
  };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    main();
  } catch (error) {
    console.error(error);
    process.exit(1);
  }
}
