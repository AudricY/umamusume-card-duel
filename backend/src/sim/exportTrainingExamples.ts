import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { runHeadlessBatch } from "./headlessAiVsAi";

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

console.log(JSON.stringify({
  out: args.out,
  games: runs.length,
  examples: examples.length,
  terminalReasons: runs.reduce<Record<string, number>>((counts, run) => {
    counts[run.terminalReason] = (counts[run.terminalReason] ?? 0) + 1;
    return counts;
  }, {}),
}, null, 2));

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
