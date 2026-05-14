// R7 step 2 smoke: assert the multi-teacher trace path emits a per-row
// `teachers` list and that the relabel pass produces a well-formed mixture
// distribution `policyTargets` over `legalActions`.
//
// This test exercises the *schema* change introduced in step 2 only; the SL
// loss / dataset loader changes belong to step 3 and are not exercised here.
//
// Coverage targets:
//   1. `--trace-teacher rollout,search,planner` populates `teachers.length===3`
//      on every model-visited row that has more than one legal action.
//   2. Each of the three teacher selections is present on every multi-teacher
//      row (the parser sorts + dedups so the order is deterministic).
//   3. The relabel pass turns the `teachers` list into a `policyTargets` array
//      whose length matches `legalActions.length`, sums to 1.0 ± 1e-6, and
//      puts nonzero mass on every teacher's `selectedActionIndex`.
//   4. Back-compat single-teacher CLI (`--trace-teacher rollout`) still emits a
//      length-1 `teachers` array, and the relabel pass produces a one-hot
//      `policyTargets` distribution from it.

import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

type TeacherEntry = {
  selection: "rollout" | "search" | "planner";
  selectedActionId: string;
  selectedActionIndex: number;
};

type TraceRow = {
  legalActions: Array<{ id: string }>;
  teachers?: TeacherEntry[];
  modelSide: "player" | "opponent";
};

type RelabeledRow = {
  legalActions: Array<{ id: string }>;
  teachers: TeacherEntry[];
  teacherSelections: Array<"rollout" | "search" | "planner">;
  policyTargets: number[];
  selectedActionIndex: number;
};

function readJsonl<T>(path: string): T[] {
  return readFileSync(path, "utf8")
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as T);
}

const root = mkdtempSync(join(tmpdir(), "uma-r7-multi-teacher-smoke-"));
const multiTraceOut = join(root, "trace.multi.jsonl");
const multiRelabeledOut = join(root, "relabeled.multi.jsonl");
const singleTraceOut = join(root, "trace.single.jsonl");
const singleRelabeledOut = join(root, "relabeled.single.jsonl");

// ----- Multi-teacher path -----
await execFileAsync("tsx", [
  "src/sim/evaluateModelVsHeuristic.ts",
  "--selection", "baseline",
  "--games", "1",
  "--model-side", "player",
  "--max-steps", "120",
  "--rollout-steps", "20",
  "--planner-max-sequences", "8",
  "--planner-max-depth", "4",
  "--decision-trace-out", multiTraceOut,
  "--trace-teacher", "rollout,search,planner",
], { cwd: process.cwd(), maxBuffer: 1024 * 1024 * 16 });

const multiTraceRows = readJsonl<TraceRow>(multiTraceOut);
assert.ok(multiTraceRows.length > 0, "multi-teacher trace should emit at least one row");

// Filter to rows with multiple legal actions — single-action rows skip the
// teacher path entirely (chooseRolloutAction et al short-circuit on
// legalActions.length <= 1), so they legitimately carry no `teachers` field.
const multiActionRows = multiTraceRows.filter((row) => (row.legalActions?.length ?? 0) > 1);
assert.ok(multiActionRows.length > 0, "smoke needs at least one multi-action state in the trace to test multi-teacher emission");

multiActionRows.forEach((row, idx) => {
  assert.ok(Array.isArray(row.teachers), `row ${idx}: must carry a teachers array`);
  assert.equal(row.teachers!.length, 3, `row ${idx}: --trace-teacher rollout,search,planner must produce 3 teachers`);
  const selections = new Set(row.teachers!.map((t) => t.selection));
  assert.ok(selections.has("rollout"), `row ${idx}: rollout teacher must be present`);
  assert.ok(selections.has("search"), `row ${idx}: search teacher must be present`);
  assert.ok(selections.has("planner"), `row ${idx}: planner teacher must be present`);
  row.teachers!.forEach((t) => {
    assert.ok(
      t.selectedActionIndex >= 0 && t.selectedActionIndex < row.legalActions.length,
      `row ${idx} teacher ${t.selection}: selectedActionIndex ${t.selectedActionIndex} out of range [0, ${row.legalActions.length})`,
    );
    assert.ok(typeof t.selectedActionId === "string" && t.selectedActionId.length > 0, "teacher selectedActionId must be non-empty");
  });
});

// Relabel the multi-teacher trace and assert policyTargets are well-formed.
await execFileAsync("tsx", [
  "src/sim/dagger/relabelDecisionTrace.ts",
  "--in", multiTraceOut,
  "--out", multiRelabeledOut,
  "--source", "model-visited-multi-teacher",
  "--label-source", "multi-teacher",
], { cwd: process.cwd(), maxBuffer: 1024 * 1024 * 16 });

const multiRelabeled = readJsonl<RelabeledRow>(multiRelabeledOut);
assert.ok(multiRelabeled.length > 0, "relabel should keep at least one multi-teacher row");

multiRelabeled.forEach((row, idx) => {
  assert.ok(Array.isArray(row.teachers), `row ${idx}: relabel must forward teachers list`);
  assert.equal(row.teachers.length, 3, `row ${idx}: multi-teacher relabel must keep all 3 teachers`);
  // Mixture target shape: one number per legal action, sums to 1.
  assert.ok(Array.isArray(row.policyTargets), `row ${idx}: relabel must emit policyTargets`);
  assert.equal(
    row.policyTargets.length,
    row.legalActions.length,
    `row ${idx}: policyTargets length ${row.policyTargets.length} must match legalActions.length ${row.legalActions.length}`,
  );
  const sum = row.policyTargets.reduce((acc, v) => acc + v, 0);
  assert.ok(
    Math.abs(sum - 1.0) < 1e-6,
    `row ${idx}: policyTargets must sum to 1.0 ± 1e-6, got ${sum}`,
  );
  // Every teacher's chosen index must carry nonzero mass.
  row.teachers.forEach((t) => {
    const mass = row.policyTargets[t.selectedActionIndex];
    assert.ok(
      mass !== undefined && mass > 0,
      `row ${idx}: teacher ${t.selection} chose index ${t.selectedActionIndex} but policyTargets has zero mass there (${mass})`,
    );
  });
  // teacherSelections is the parallel canonical list of selection names.
  assert.ok(Array.isArray(row.teacherSelections), `row ${idx}: teacherSelections must be present`);
  assert.equal(row.teacherSelections.length, 3, `row ${idx}: teacherSelections must be length 3`);
});

// ----- Back-compat: single-teacher path still emits length-1 -----
await execFileAsync("tsx", [
  "src/sim/evaluateModelVsHeuristic.ts",
  "--selection", "baseline",
  "--games", "1",
  "--model-side", "player",
  "--max-steps", "120",
  "--rollout-steps", "20",
  "--decision-trace-out", singleTraceOut,
  "--trace-teacher", "rollout",
], { cwd: process.cwd(), maxBuffer: 1024 * 1024 * 16 });

const singleTraceRows = readJsonl<TraceRow>(singleTraceOut);
const singleMultiActionRows = singleTraceRows.filter((row) => (row.legalActions?.length ?? 0) > 1);
assert.ok(singleMultiActionRows.length > 0, "single-teacher smoke needs at least one multi-action state");
singleMultiActionRows.forEach((row, idx) => {
  assert.ok(Array.isArray(row.teachers), `single-teacher row ${idx}: must carry teachers list`);
  assert.equal(row.teachers!.length, 1, `single-teacher row ${idx}: length-1 back-compat`);
  assert.equal(row.teachers![0]!.selection, "rollout", `single-teacher row ${idx}: selection must echo CLI`);
});

await execFileAsync("tsx", [
  "src/sim/dagger/relabelDecisionTrace.ts",
  "--in", singleTraceOut,
  "--out", singleRelabeledOut,
  "--source", "model-visited-rollout-relabeled",
  "--label-source", "rollout-teacher",
], { cwd: process.cwd(), maxBuffer: 1024 * 1024 * 16 });

const singleRelabeled = readJsonl<RelabeledRow>(singleRelabeledOut);
assert.ok(singleRelabeled.length > 0, "single-teacher relabel should keep rows");
singleRelabeled.forEach((row, idx) => {
  assert.equal(row.policyTargets.length, row.legalActions.length);
  const sum = row.policyTargets.reduce((acc, v) => acc + v, 0);
  assert.ok(Math.abs(sum - 1.0) < 1e-6, `single-teacher row ${idx}: policyTargets must sum to 1`);
  // Single teacher → one-hot mixture.
  assert.equal(
    row.policyTargets[row.selectedActionIndex],
    1.0,
    `single-teacher row ${idx}: policyTargets must be one-hot on selectedActionIndex`,
  );
  // Exactly one nonzero entry.
  const nonzero = row.policyTargets.filter((v) => v > 0).length;
  assert.equal(nonzero, 1, `single-teacher row ${idx}: must have exactly one nonzero policyTargets entry`);
});

// ----- evalGate parseTraceTeacher comma-list smoke -----
// Exercises the CLI surface of evalGate.ts so we know the parser there agrees
// with the parser in evaluateModelVsHeuristic.ts. We don't run a full gate
// here (too slow); the planted-bad-policy gate in evalGateSmoke.ts already
// covers gate machinery. Here we just confirm the CLI accepts the comma
// input without throwing.
let gateExit = 0;
try {
  await execFileAsync("tsx", [
    "src/sim/evalGate.ts",
    "--selection", "baseline",
    "--games", "1",
    "--model-side", "player",
    "--max-steps", "60",
    "--min-games", "1",
    "--allow-fallbacks",
    "--allow-no-ops",
    "--min-ci-lower", "0",
    "--trace-teacher", "rollout,search,planner",
  ], { cwd: process.cwd(), maxBuffer: 1024 * 1024 * 8 });
} catch (error) {
  // The gate may FAIL on a 1-game run for various reasons; what we care about
  // is that the multi-teacher CLI string parses without a parser-side throw.
  // A parser-side throw would appear on stderr as our explicit error message.
  const stderr = String((error as { stderr?: unknown }).stderr ?? "");
  assert.ok(
    !stderr.includes("--trace-teacher: unknown token"),
    `evalGate parseTraceTeacher must accept comma-list input, got parse error: ${stderr}`,
  );
  gateExit = Number((error as { code?: unknown }).code ?? 1);
}
// gateExit may be 0 or nonzero; either is fine — the assertion above already
// confirmed the parser accepted the input.
void gateExit;

console.log(JSON.stringify({
  status: "PASS",
  multiTeacherTraceRows: multiTraceRows.length,
  multiTeacherMultiActionRows: multiActionRows.length,
  multiTeacherRelabeled: multiRelabeled.length,
  singleTeacherRelabeled: singleRelabeled.length,
  dir: root,
}, null, 2));
