import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { withGitMetadata } from "../manifest";
import { createSeededRng } from "../../../../frontend/src/game/engine/core/random";

type MixComponent = {
  path: string;
  weight: number;
  sourceTag: string;
  rows: string[];
};

type MixArgs = {
  out: string;
  components: MixComponent[];
  seed: string;
  manifestOut: string | null;
  totalRows: number | null;
};

function main() {
  const args = parseArgs(process.argv.slice(2));
  const totalWeight = args.components.reduce((sum, c) => sum + c.weight, 0);
  if (totalWeight <= 0) throw new Error("Total mix weight must be > 0");

  const rng = createSeededRng(args.seed, "mix-sources");
  const requestedTotal = args.totalRows ?? args.components.reduce((sum, c) => sum + c.rows.length, 0);

  const componentTargets = args.components.map((c) => {
    const fairShare = (c.weight / totalWeight) * requestedTotal;
    let target = Math.min(c.rows.length, Math.round(fairShare));
    // A component with positive weight and at least one available row must
    // contribute at least one row; otherwise the manifest's requestedRatio
    // silently lies. v4.1 review #3.
    if (c.weight > 0 && c.rows.length > 0 && target === 0) target = 1;
    return { component: c, target };
  });

  args.components.forEach((c, index) => {
    const sampled = componentTargets[index]!.target;
    if (c.weight > 0 && c.rows.length > 0 && sampled === 0) {
      throw new Error(`mix component ${c.path} (weight ${c.weight}) sampled 0 rows from ${c.rows.length} available — bug`);
    }
  });

  const sampled = componentTargets.flatMap(({ component, target }) => {
    const indices = Array.from({ length: component.rows.length }, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i -= 1) {
      const j = Math.floor(rng.next() * (i + 1));
      [indices[i], indices[j]] = [indices[j]!, indices[i]!];
    }
    return indices.slice(0, target).map((idx) => ({
      sourceTag: component.sourceTag,
      row: component.rows[idx]!,
    }));
  });

  for (let i = sampled.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng.next() * (i + 1));
    [sampled[i], sampled[j]] = [sampled[j]!, sampled[i]!];
  }

  mkdirSync(dirname(args.out), { recursive: true });
  const lines = sampled.map((entry) => {
    const parsed = JSON.parse(entry.row);
    parsed.mixSourceTag = entry.sourceTag;
    return JSON.stringify(parsed);
  });
  writeFileSync(args.out, lines.join("\n") + (lines.length ? "\n" : ""), "utf8");

  const summary = {
    outputPath: args.out,
    requestedTotal,
    sampledRows: sampled.length,
    totalWeight,
    components: componentTargets.map(({ component, target }) => ({
      path: component.path,
      sourceTag: component.sourceTag,
      requestedWeight: component.weight,
      requestedRatio: component.weight / totalWeight,
      availableRows: component.rows.length,
      sampledRows: target,
    })),
  };

  if (args.manifestOut) {
    mkdirSync(dirname(args.manifestOut), { recursive: true });
    writeFileSync(args.manifestOut, JSON.stringify(withGitMetadata({ args: { out: args.out, seed: args.seed, totalRows: args.totalRows }, summary }), null, 2) + "\n", "utf8");
  }

  console.log(JSON.stringify({ status: "PASS", ...summary }, null, 2));
}

function parseArgs(argv: string[]): MixArgs {
  const get = (name: string, fallback: string) => {
    const index = argv.indexOf(name);
    return index >= 0 ? argv[index + 1] ?? fallback : fallback;
  };
  const out = get("--out", "");
  const seed = get("--seed", "mix-default");
  const totalRows = argv.includes("--total-rows") ? Number(get("--total-rows", "0")) : null;
  const sourceArgs: string[] = [];
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--source") {
      const spec = argv[i + 1];
      if (spec) sourceArgs.push(spec);
      i += 1;
    }
  }
  if (sourceArgs.length === 0) throw new Error("At least one --source path:weight:tag is required");
  const components = sourceArgs.map((spec) => {
    const parts = spec.split(":");
    if (parts.length < 2) throw new Error(`Bad --source spec ${spec}; expected path:weight or path:weight:tag`);
    const [path, weightStr, tag] = parts;
    if (!path) throw new Error(`--source missing path in ${spec}`);
    const weight = Number(weightStr);
    if (!Number.isFinite(weight) || weight < 0) throw new Error(`--source bad weight ${weightStr} in ${spec}`);
    const rows = readFileSync(path, "utf8").trim();
    return {
      path,
      weight,
      sourceTag: tag ?? path,
      rows: rows ? rows.split("\n").filter(Boolean) : [],
    };
  });
  return {
    out,
    components,
    seed,
    manifestOut: get("--manifest-out", "") || null,
    totalRows,
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
