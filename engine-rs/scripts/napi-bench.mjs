// Benchmark: NAPI in-process throughput vs subprocess-per-game.
//
// Two measurements:
//   1. NAPI loop: N games driven via bridge.driveHeuristicGameJson.
//      Single process; FFI calls only.
//   2. Subprocess loop: N games each spawned as a separate
//      sim-export-training invocation. Mirrors the historical pipeline
//      pattern where orchestrators spawned one binary per task.
//
// Reports games/sec, mean ms/game, and wall time for each. The point
// is to quantify the subprocess startup cost the Phase 2 bridge
// eliminates.

import { createRequire } from "node:module";
import { spawnSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const bridge = require(join(here, "..", "crates", "napi-bridge", "index.js"));

const N = parseInt(process.argv[2] ?? "20", 10);

console.log(`Benchmark: ${N} heuristic-vs-heuristic games each way.`);

// 1. NAPI loop.
{
  const t0 = process.hrtime.bigint();
  let totalSteps = 0;
  for (let i = 0; i < N; i += 1) {
    const r = JSON.parse(bridge.driveHeuristicGameJson(String(i), 1000));
    totalSteps += r.steps;
  }
  const t1 = process.hrtime.bigint();
  const wallMs = Number(t1 - t0) / 1e6;
  console.log(
    `  NAPI in-process:    ${(N / (wallMs / 1000)).toFixed(1).padStart(7)} games/s   `
      + `${(wallMs / N).toFixed(2).padStart(7)} ms/game   `
      + `wall ${wallMs.toFixed(0)} ms   total ${totalSteps} steps`,
  );
}

// 1b. NAPI loop, MCTS-vs-heuristic. Slower per game (MCTS sims) but
// still in-process — measures the MCTS pipeline cost without
// subprocess startup.
{
  const cfg = JSON.stringify({
    simulations: 30,
    cPuct: 1.5,
    rolloutCrnSamples: 2,
    rolloutSteps: 100,
    prior: "uniform",
    leaf: "rollout",
    maxNodes: 1000,
  });
  const t0 = process.hrtime.bigint();
  let totalSteps = 0, totalDecisions = 0;
  for (let i = 0; i < N; i += 1) {
    const r = JSON.parse(bridge.driveMctsGameJson(String(i), "player", 500, cfg));
    totalSteps += r.steps;
    totalDecisions += r.modelDecisions;
  }
  const t1 = process.hrtime.bigint();
  const wallMs = Number(t1 - t0) / 1e6;
  console.log(
    `  NAPI MCTS in-process:${(N / (wallMs / 1000)).toFixed(1).padStart(7)} games/s   `
      + `${(wallMs / N).toFixed(2).padStart(7)} ms/game   `
      + `wall ${wallMs.toFixed(0)} ms   total ${totalSteps} steps, ${totalDecisions} MCTS decisions`,
  );
}

// 2. Subprocess loop. Use sim-export-training which drives heuristic
// games to terminal; the closest binary to driveHeuristicGameJson.
const binDir = join(here, "..", "target", "release");
{
  const t0 = process.hrtime.bigint();
  let totalSteps = 0;
  for (let i = 0; i < N; i += 1) {
    const res = spawnSync(
      join(binDir, "sim-export-training"),
      ["--games", "1", "--seed-start", String(i), "--out", "/dev/null"],
      { encoding: "utf8" },
    );
    if (res.status !== 0) {
      throw new Error(`subprocess failed: ${res.stderr}`);
    }
    // sim-export-training doesn't report steps; we just count games.
    totalSteps += 1;
  }
  const t1 = process.hrtime.bigint();
  const wallMs = Number(t1 - t0) / 1e6;
  console.log(
    `  Subprocess-per-game:${(N / (wallMs / 1000)).toFixed(1).padStart(7)} games/s   `
      + `${(wallMs / N).toFixed(2).padStart(7)} ms/game   `
      + `wall ${wallMs.toFixed(0)} ms`,
  );
}
