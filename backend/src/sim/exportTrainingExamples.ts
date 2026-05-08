import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { runHeadlessBatch } from "./headlessAiVsAi";
import { writeManifestFor } from "./manifest";
import { ACTION_FEATURE_COUNT, ACTION_FEATURE_SCHEMA_VERSION } from "../../../frontend/src/game/engine/ai-policy/actions";

type Args = {
  out: string;
  seedStart: number;
  games: number;
  maxSteps: number;
};

const args = parseArgs(process.argv.slice(2));
const seeds = Array.from({ length: args.games }, (_, index) => String(args.seedStart + index));
const runs = runHeadlessBatch(seeds, args.maxSteps);
const examples = runs.flatMap((run) => run.examples);

mkdirSync(dirname(args.out), { recursive: true });
writeFileSync(args.out, examples.map((example) => JSON.stringify(example)).join("\n") + "\n", "utf8");
const manifestOut = writeManifestFor(args.out, {
  artifact: args.out,
  sourceTaxonomy: { source: "rule-bot" },
  args,
  seeds,
  games: runs.length,
  examples: examples.length,
  phaseCounts: countBy(examples, (example) => example.phase),
  actionKindCounts: countBy(examples, (example) => example.legalActions[example.selectedActionIndex]?.kind ?? "unknown"),
  terminalReasons: countBy(runs, (run) => run.terminalReason),
  featureSchemas: { observationSchemaVersion: 1, actionFeatureSchemaVersion: ACTION_FEATURE_SCHEMA_VERSION, actionFeatureDimensions: ACTION_FEATURE_COUNT, stateFeatureDimensions: 96 },
});

console.log(JSON.stringify({
  out: args.out,
  manifest: manifestOut,
  games: runs.length,
  examples: examples.length,
  terminalReasons: countBy(runs, (run) => run.terminalReason),
}, null, 2));

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
  return {
    out: get("--out", "training/runs/smoke/examples.jsonl"),
    seedStart: Number(get("--seed-start", "1000")),
    games: Number(get("--games", "16")),
    maxSteps: Number(get("--max-steps", "360")),
  };
}
