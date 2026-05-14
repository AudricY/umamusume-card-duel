import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

const root = mkdtempSync(join(tmpdir(), "uma-dagger-smoke-"));
const traceOut = join(root, "trace.jsonl");
const relabeledOut = join(root, "relabeled.jsonl");
const ruleBotOut = join(root, "rulebot.jsonl");
const mixedOut = join(root, "mixed.jsonl");
const traceManifest = join(root, "trace.manifest.json");
const relabelManifest = join(root, "relabel.manifest.json");
const mixManifest = join(root, "mix.manifest.json");

await execFileAsync("tsx", [
  "src/sim/evaluateModelVsHeuristic.ts",
  "--selection", "baseline",
  "--games", "2",
  "--model-side", "both",
  "--max-steps", "200",
  "--rollout-steps", "40",
  "--decision-trace-out", traceOut,
  "--trace-teacher", "rollout",
  "--manifest-out", traceManifest,
], { cwd: process.cwd(), maxBuffer: 1024 * 1024 * 16 });

const traceRows = readJsonl(traceOut);
assert.ok(traceRows.length >= 4, `decision trace should produce >=4 rows, got ${traceRows.length}`);

assertHiddenInfoSafe(traceRows);

await execFileAsync("tsx", [
  "src/sim/dagger/relabelDecisionTrace.ts",
  "--in", traceOut,
  "--out", relabeledOut,
  "--source", "model-visited-rollout-relabeled",
  "--label-source", "rollout-teacher",
  "--manifest-out", relabelManifest,
], { cwd: process.cwd(), maxBuffer: 1024 * 1024 * 16 });

const relabeled = readJsonl(relabeledOut);
assert.ok(relabeled.length > 0, "relabel must keep at least some rows");
relabeled.forEach((row) => {
  assert.equal(row.source, "model-visited-rollout-relabeled");
  assert.equal(row.labelSource, "rollout-teacher");
  // R7 step 2: teacherSelections is the new (list-shaped) field. For the
  // single-teacher recipe we hit here, it must be ["rollout"].
  assert.ok(Array.isArray(row.teacherSelections), "relabeled row must carry teacherSelections list");
  assert.deepEqual(row.teacherSelections, ["rollout"], "single-teacher recipe must emit length-1 teacherSelections");
  assert.ok(Array.isArray(row.teachers), "relabeled row must forward teachers list");
  assert.equal(row.teachers.length, 1);
  assert.equal(row.teachers[0].selection, "rollout");
  assert.ok(row.legalActions.length > 0);
  assert.ok(row.selectedActionIndex >= 0 && row.selectedActionIndex < row.legalActions.length);
  const targetLabel = row.legalActions[row.selectedActionIndex];
  assert.equal(row.selectedActionId, targetLabel.id, "selectedActionId must match teacher's chosen action");
  // R7 step 2: policyTargets is the per-state mixture distribution. With one
  // teacher it collapses to a one-hot at selectedActionIndex.
  assert.ok(Array.isArray(row.policyTargets), "relabeled row must carry policyTargets");
  assert.equal(row.policyTargets.length, row.legalActions.length, "policyTargets length must match legalActions");
  const sum = row.policyTargets.reduce((acc: number, v: number) => acc + v, 0);
  assert.ok(Math.abs(sum - 1.0) < 1e-6, `policyTargets should sum to 1, got ${sum}`);
  assert.equal(row.policyTargets[row.selectedActionIndex], 1.0, "single-teacher policyTargets must be one-hot on selectedActionIndex");
});
assertHiddenInfoSafe(relabeled);

// Item 18: the relabeler must forward behaviorPolicy when present on the
// upstream trace row. baseline-source rows have no behaviorPolicy, so plant
// one onto every trace row, re-relabel, and assert each output row carries it.
const plantedTracePath = join(root, "trace.with-behavior.jsonl");
const plantedRelabeledPath = join(root, "relabeled.with-behavior.jsonl");
const plantedTrace = traceRows.map((row) => ({
  ...row,
  behaviorPolicy: {
    kind: "planted-uniform",
    temperature: 1,
    actionLogProbs: row.legalActions.map(() => Math.log(1 / row.legalActions.length)),
    actionProbs: row.legalActions.map(() => 1 / row.legalActions.length),
    selectedLogProb: Math.log(1 / row.legalActions.length),
  },
}));
writeFileSync(plantedTracePath, plantedTrace.map((row) => JSON.stringify(row)).join("\n") + "\n", "utf8");
await execFileAsync("tsx", [
  "src/sim/dagger/relabelDecisionTrace.ts",
  "--in", plantedTracePath,
  "--out", plantedRelabeledPath,
  "--source", "model-visited-rollout-relabeled",
  "--label-source", "rollout-teacher",
], { cwd: process.cwd(), maxBuffer: 1024 * 1024 * 16 });
const relabeledWithBehavior = readJsonl(plantedRelabeledPath);
assert.ok(relabeledWithBehavior.length > 0, "behavior-planted relabel must keep rows");
relabeledWithBehavior.forEach((row) => {
  assert.ok(row.behaviorPolicy, "relabeler must forward behaviorPolicy when present upstream");
  assert.equal(row.behaviorPolicy.kind, "planted-uniform");
  assert.equal(row.behaviorPolicy.actionLogProbs.length, row.legalActions.length);
});

await execFileAsync("tsx", [
  "src/sim/exportTrainingExamples.ts",
  "--out", ruleBotOut,
  "--games", "2",
  "--max-steps", "200",
  "--seed-start", "9100",
], { cwd: process.cwd(), maxBuffer: 1024 * 1024 * 16 });
const ruleBotRows = readJsonl(ruleBotOut);
assert.ok(ruleBotRows.length > 0, "rule-bot exporter should produce rows");

await execFileAsync("tsx", [
  "src/sim/dagger/mixSources.ts",
  "--out", mixedOut,
  "--seed", "smoke-mix",
  "--source", `${relabeledOut}:0.4:model-visited-relabeled`,
  "--source", `${ruleBotOut}:0.6:rule-bot`,
  "--manifest-out", mixManifest,
], { cwd: process.cwd(), maxBuffer: 1024 * 1024 * 16 });

const mixed = readJsonl(mixedOut);
assert.ok(mixed.length > 0, "mixer should produce rows");
const sourceCounts = new Map<string, number>();
mixed.forEach((row) => {
  assert.ok(typeof row.mixSourceTag === "string", "mix tags row with mixSourceTag");
  sourceCounts.set(row.mixSourceTag, (sourceCounts.get(row.mixSourceTag) ?? 0) + 1);
});
assert.ok(sourceCounts.size >= 1, `mixed jsonl should include at least one source tag, got ${[...sourceCounts.keys()].join(",")}`);

const traceManifestPayload = JSON.parse(readFileSync(traceManifest, "utf8"));
assert.ok(traceManifestPayload.summary, "trace manifest should record summary");
const relabelManifestPayload = JSON.parse(readFileSync(relabelManifest, "utf8"));
assert.equal(relabelManifestPayload.summary.outputPath, relabeledOut);
assert.equal(relabelManifestPayload.summary.inputPath, traceOut);
const mixManifestPayload = JSON.parse(readFileSync(mixManifest, "utf8"));
assert.equal(mixManifestPayload.summary.outputPath, mixedOut);
assert.ok(Array.isArray(mixManifestPayload.summary.components), "mix manifest must record components");

const writableExample = mixed[0];
assert.ok(writableExample.legalActions.length >= 1, "mixed row must retain legalActions for trainer");
assert.ok(writableExample.observation, "mixed row must retain observation for trainer");

// Reviewer 2 #6: extend the leak fixture beyond opponent.handCardIds. Plant
// each known leak field independently and assert the relabeler aborts on each.
// Without this, a regression that allowed opponent.hand or opponent.deck through
// would slip past the smoke even though the production detector is supposed to
// catch them.
const leakFields: Array<{ field: string; mutate: (opp: Record<string, unknown>) => Record<string, unknown> }> = [
  { field: "opponent.handCardIds", mutate: (opp) => ({ ...opp, handCardIds: ["leak-card"] }) },
  { field: "opponent.hand", mutate: (opp) => ({ ...opp, hand: [{ id: "leak-card" }] }) },
  { field: "opponent.deck", mutate: (opp) => ({ ...opp, deck: ["leak-card"] }) },
];
for (const { field, mutate } of leakFields) {
  const planted = traceRows.map((row) => ({
    ...row,
    observation: { ...row.observation, opponent: mutate(row.observation.opponent) },
  }));
  const plantedPath = join(root, `leak-${field.replace(/\./g, "-")}.jsonl`);
  writeFileSync(plantedPath, planted.map((row) => JSON.stringify(row)).join("\n") + "\n", "utf8");
  let leakCaught = false;
  try {
    await execFileAsync("tsx", [
      "src/sim/dagger/relabelDecisionTrace.ts",
      "--in", plantedPath,
      "--out", join(root, `leak-${field.replace(/\./g, "-")}.relabeled.jsonl`),
    ], { cwd: process.cwd(), maxBuffer: 1024 * 1024 * 16 });
  } catch (error) {
    leakCaught = true;
    const stderr = String((error as { stderr?: unknown }).stderr ?? "");
    assert.ok(stderr.includes("Hidden-info leak"), `relabel must error on planted ${field} leak: ${stderr}`);
    assert.ok(stderr.includes(field), `relabel error must name the leaked field ${field}: ${stderr}`);
  }
  assert.equal(leakCaught, true, `planted ${field} leak must abort the relabel run`);
}

console.log(JSON.stringify({ status: "PASS", traceRows: traceRows.length, relabeled: relabeled.length, mixed: mixed.length, ruleBot: ruleBotRows.length, dir: root }, null, 2));

function readJsonl(path: string): any[] {
  return readFileSync(path, "utf8")
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function assertHiddenInfoSafe(rows: any[]): void {
  rows.forEach((row, index) => {
    const opponent = row.observation?.opponent ?? {};
    assert.equal(opponent.handCardIds, undefined, `row ${index}: opponent.handCardIds must not leak`);
    assert.equal(opponent.hand, undefined, `row ${index}: opponent.hand must not leak`);
    assert.equal(opponent.deck, undefined, `row ${index}: opponent.deck must not leak`);
  });
}
