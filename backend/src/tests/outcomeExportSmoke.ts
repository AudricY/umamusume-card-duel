import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

const runA = runOutcomeExport("a");
const runB = runOutcomeExport("b");

assert.deepEqual(
  compactRows(runA.rows),
  compactRows(runB.rows),
  "same outcome export seed should reproduce selected labels and oracle statistics",
);
assert.ok(existsSync(runA.manifestPath), "outcome export should write a sibling manifest");
assert.equal(runA.manifest.sourceTaxonomy.trajectorySource, "ai-policy-baseline-visited");
assert.equal(runA.manifest.sourceTaxonomy.labelSource, "rollout-labeled");
assert.equal(runA.manifest.baselineFallbacks, 0, "modeled baseline should not fall back in smoke export");
runA.rows.forEach((row) => {
  assert.equal(row.observation.opponent.handCardIds, undefined, "opponent hand IDs must not leak in outcome rows");
  assert.equal(row.oracle.sampleCount, 2, "oracle metadata should record sample count");
  assert.ok(Array.isArray(row.oracle.sampleSeedIds), "oracle metadata should record sample seed IDs");
  assert.ok(Number.isFinite(row.oracle.selectedVsRunnerUpMargin), "oracle metadata should record selected margin");
});

console.log(JSON.stringify({
  status: "PASS",
  examples: runA.rows.length,
  manifest: runA.manifestPath,
}, null, 2));

function runOutcomeExport(label: string) {
  const dir = mkdtempSync(join(tmpdir(), `uma-outcome-${label}-`));
  const out = join(dir, "examples.jsonl");
  execFileSync("tsx", [
    "src/sim/exportOutcomeTrainingExamples.ts",
    "--out", out,
    "--games", "1",
    "--seed-start", "95000",
    "--max-steps", "80",
    "--rollout-steps", "30",
    "--max-examples", "4",
    "--max-actions", "4",
    "--samples", "2",
  ], { cwd: process.cwd(), stdio: "pipe" });
  const manifestPath = join(dir, "examples.manifest.json");
  return {
    rows: readJsonl(out),
    manifestPath,
    manifest: JSON.parse(readFileSync(manifestPath, "utf8")),
  };
}

function readJsonl(path: string): any[] {
  return readFileSync(path, "utf8")
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function compactRows(rows: any[]) {
  return rows.map((row) => ({
    seed: row.seed,
    step: row.step,
    sideId: row.sideId,
    selectedActionId: row.selectedActionId,
    selectedActionIndex: row.selectedActionIndex,
    sampleWeight: row.sampleWeight,
    valueTarget: row.valueTarget,
    oracle: {
      sampleSeedIds: row.oracle.sampleSeedIds,
      rewardMean: row.oracle.rewardMean,
      rewardVariance: row.oracle.rewardVariance,
      selectedVsRunnerUpMargin: row.oracle.selectedVsRunnerUpMargin,
      selectedVsBaselineMargin: row.oracle.selectedVsBaselineMargin,
      selectedOriginalRank: row.oracle.selectedOriginalRank,
      candidateCount: row.oracle.candidateCount,
    },
  }));
}
