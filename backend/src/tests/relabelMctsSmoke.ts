// R16-TD 3a smoke: exercise the `--relabel-mcts` emit-path end-to-end so the
// source-recipe matrix (rule-bot-mirror / policy-vs-rule / search-vs-rule) is
// regression-locked. The matrix is achieved today via existing flags
// (`--selection` + `--relabel-state-source`); no new flag is needed.
//
// Sub-cases:
//   (a) --relabel-mcts ON, --selection baseline,
//       --relabel-state-source rule-bot-mirror — every emitted row carries a
//       well-formed `policyTargets` (length === legalActions.length, sums to
//       1 ± 1e-6, no negatives), a full `oracle` block per scoping § P1, and
//       `stateSource === "rule-bot-mirror"`. All emitted rows have
//       legalActions.length >= 2 (forced states suppressed by construction).
//       Also pipe the trace through `relabelDecisionTrace.ts` with
//       `--label-source rollout-leaf-mcts` and assert pass-through preserves
//       `policyTargets` length+sum and forwards `oracle.visitDistribution`
//       field-for-field.
//   (b) Same as (a) but --relabel-state-source policy-vs-rule. Proves the
//       state-source tag wires through independently of the played selection.
//   (c) --relabel-mcts OFF, otherwise same args. No row may carry
//       policyTargets / oracle / stateSource (no-op invariant).
//
// Knob rationale: --selection baseline keeps the smoke server-free (rule-bot
// played, MCTS only fires on the relabel path). --mcts-simulations 8 +
// --mcts-rollout-steps 20 + --mcts-rollout-crn-samples 1 is the smallest
// MCTS config that still emits ≥1 contested row in a 2-game,
// --max-steps 80 trace. If this smoke flakes (zero rows) raise --games or
// --max-steps first; do not lower MCTS knobs below these floors.

import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

type LegalAction = { id: string };

type OracleBlock = {
  simulationsRun: number;
  rolloutCrnSamples: number;
  rolloutSteps: number;
  rootValue: number;
  rootMeanQ: number[];
  rootPriors: number[];
  visitDistribution: number[];
  rootPriorEntropy: number;
  expansions: number;
  leafEvaluations: number;
  haltedEarly: boolean;
};

type TraceRow = {
  legalActions: LegalAction[];
  policyTargets?: number[];
  oracle?: OracleBlock;
  stateSource?: string;
  selection: string;
  seed: string;
  step: number;
};

type RelabeledMctsRow = {
  legalActions: LegalAction[];
  policyTargets: number[];
  oracle: OracleBlock | null;
  stateSource: string | null;
  labelSource: string;
};

function readJsonl<T>(path: string): T[] {
  const raw = readFileSync(path, "utf8").trim();
  if (!raw) return [];
  return raw.split("\n").filter(Boolean).map((line) => JSON.parse(line) as T);
}

const REQUIRED_ORACLE_KEYS: Array<keyof OracleBlock> = [
  "simulationsRun",
  "rolloutCrnSamples",
  "rolloutSteps",
  "rootValue",
  "rootMeanQ",
  "rootPriors",
  "visitDistribution",
  "rootPriorEntropy",
  "expansions",
  "leafEvaluations",
  "haltedEarly",
];

function assertWellFormedTraceRow(label: string, row: TraceRow, idx: number, expectedStateSource: string): void {
  // Every emitted relabel row must be contested by construction (forced
  // states are suppressed in runMctsRelabel).
  assert.ok(
    row.legalActions.length >= 2,
    `${label} row ${idx}: relabel-mcts must suppress forced states (legalActions.length=${row.legalActions.length})`,
  );
  // policyTargets shape + simplex.
  assert.ok(Array.isArray(row.policyTargets), `${label} row ${idx}: policyTargets must be present`);
  assert.equal(
    row.policyTargets!.length,
    row.legalActions.length,
    `${label} row ${idx}: policyTargets length ${row.policyTargets!.length} must match legalActions.length ${row.legalActions.length}`,
  );
  const sum = row.policyTargets!.reduce((acc, v) => acc + v, 0);
  assert.ok(
    Math.abs(sum - 1.0) < 1e-6,
    `${label} row ${idx}: policyTargets must sum to 1.0 ± 1e-6, got ${sum}`,
  );
  row.policyTargets!.forEach((v, vi) => {
    assert.ok(v >= 0, `${label} row ${idx}: policyTargets[${vi}] must be non-negative, got ${v}`);
  });
  // Full oracle schema (scoping § P1).
  assert.ok(row.oracle, `${label} row ${idx}: oracle block must be present`);
  for (const key of REQUIRED_ORACLE_KEYS) {
    assert.ok(
      Object.prototype.hasOwnProperty.call(row.oracle!, key),
      `${label} row ${idx}: oracle missing required key '${String(key)}'`,
    );
  }
  // State-source tag wires through verbatim.
  assert.equal(
    row.stateSource,
    expectedStateSource,
    `${label} row ${idx}: stateSource must be '${expectedStateSource}', got '${row.stateSource}'`,
  );
}

const root = mkdtempSync(join(tmpdir(), "uma-r16-relabel-mcts-smoke-"));
const ruleBotTraceOut = join(root, "trace.rule-bot-mirror.jsonl");
const ruleBotRelabeledOut = join(root, "relabeled.rule-bot-mirror.jsonl");
const policyVsRuleTraceOut = join(root, "trace.policy-vs-rule.jsonl");
const noOpTraceOut = join(root, "trace.no-op.jsonl");

const COMMON_ARGS = [
  "--selection", "baseline",
  "--games", "2",
  "--model-side", "player",
  "--max-steps", "80",
  "--mcts-simulations", "8",
  "--mcts-rollout-steps", "20",
  "--mcts-rollout-crn-samples", "1",
];

// ----- (a) --relabel-mcts ON + rule-bot-mirror -----
await execFileAsync("tsx", [
  "src/sim/evaluateModelVsHeuristic.ts",
  ...COMMON_ARGS,
  "--decision-trace-out", ruleBotTraceOut,
  "--relabel-mcts",
  "--relabel-state-source", "rule-bot-mirror",
], { cwd: process.cwd(), maxBuffer: 1024 * 1024 * 16 });

const ruleBotRows = readJsonl<TraceRow>(ruleBotTraceOut);
assert.ok(
  ruleBotRows.length > 0,
  "rule-bot-mirror smoke must emit at least one row; raise --games/--max-steps if this fails",
);
ruleBotRows.forEach((row, idx) => assertWellFormedTraceRow("rule-bot-mirror", row, idx, "rule-bot-mirror"));

// Relabel pass-through (case (a) extension): label-source rollout-leaf-mcts
// must forward policyTargets / oracle.visitDistribution verbatim.
await execFileAsync("tsx", [
  "src/sim/dagger/relabelDecisionTrace.ts",
  "--in", ruleBotTraceOut,
  "--out", ruleBotRelabeledOut,
  "--source", "model-visited-rollout-leaf-mcts",
  "--label-source", "rollout-leaf-mcts",
], { cwd: process.cwd(), maxBuffer: 1024 * 1024 * 16 });

const ruleBotRelabeled = readJsonl<RelabeledMctsRow>(ruleBotRelabeledOut);
assert.equal(
  ruleBotRelabeled.length,
  ruleBotRows.length,
  `relabel pass-through must keep every row (in=${ruleBotRows.length}, out=${ruleBotRelabeled.length})`,
);
ruleBotRelabeled.forEach((row, idx) => {
  const src = ruleBotRows[idx]!;
  assert.equal(row.labelSource, "rollout-leaf-mcts", `relabeled row ${idx}: labelSource must echo CLI`);
  assert.equal(
    row.policyTargets.length,
    row.legalActions.length,
    `relabeled row ${idx}: policyTargets length must match legalActions.length`,
  );
  const sum = row.policyTargets.reduce((acc, v) => acc + v, 0);
  assert.ok(
    Math.abs(sum - 1.0) < 1e-6,
    `relabeled row ${idx}: policyTargets must sum to 1 ± 1e-6, got ${sum}`,
  );
  // policyTargets forwarded element-for-element.
  src.policyTargets!.forEach((v, vi) => {
    assert.equal(
      row.policyTargets[vi],
      v,
      `relabeled row ${idx}: policyTargets[${vi}] must be forwarded verbatim (src=${v}, out=${row.policyTargets[vi]})`,
    );
  });
  // visitDistribution forwarded field-for-field.
  assert.ok(row.oracle, `relabeled row ${idx}: oracle must be forwarded`);
  assert.equal(
    row.oracle!.visitDistribution.length,
    src.oracle!.visitDistribution.length,
    `relabeled row ${idx}: oracle.visitDistribution length must match`,
  );
  src.oracle!.visitDistribution.forEach((v, vi) => {
    assert.equal(
      row.oracle!.visitDistribution[vi],
      v,
      `relabeled row ${idx}: oracle.visitDistribution[${vi}] must be forwarded verbatim`,
    );
  });
  assert.equal(
    row.stateSource,
    "rule-bot-mirror",
    `relabeled row ${idx}: stateSource must be forwarded`,
  );
});

// ----- (b) --relabel-mcts ON + policy-vs-rule tag -----
await execFileAsync("tsx", [
  "src/sim/evaluateModelVsHeuristic.ts",
  ...COMMON_ARGS,
  "--decision-trace-out", policyVsRuleTraceOut,
  "--relabel-mcts",
  "--relabel-state-source", "policy-vs-rule",
], { cwd: process.cwd(), maxBuffer: 1024 * 1024 * 16 });

const policyVsRuleRows = readJsonl<TraceRow>(policyVsRuleTraceOut);
assert.ok(policyVsRuleRows.length > 0, "policy-vs-rule smoke must emit at least one row");
policyVsRuleRows.forEach((row, idx) => assertWellFormedTraceRow("policy-vs-rule", row, idx, "policy-vs-rule"));

// ----- (c) --relabel-mcts OFF: no-op invariant -----
await execFileAsync("tsx", [
  "src/sim/evaluateModelVsHeuristic.ts",
  ...COMMON_ARGS,
  "--decision-trace-out", noOpTraceOut,
], { cwd: process.cwd(), maxBuffer: 1024 * 1024 * 16 });

const noOpRows = readJsonl<TraceRow>(noOpTraceOut);
assert.ok(noOpRows.length > 0, "no-op smoke must emit at least one trace row");
noOpRows.forEach((row, idx) => {
  assert.equal(
    row.policyTargets,
    undefined,
    `no-op row ${idx}: policyTargets must not be emitted when --relabel-mcts is off`,
  );
  assert.equal(
    row.oracle,
    undefined,
    `no-op row ${idx}: oracle must not be emitted when --relabel-mcts is off`,
  );
  assert.equal(
    row.stateSource,
    undefined,
    `no-op row ${idx}: stateSource must not be emitted when --relabel-mcts is off`,
  );
});

console.log(JSON.stringify({
  status: "PASS",
  ruleBotMirrorRows: ruleBotRows.length,
  ruleBotMirrorRelabeled: ruleBotRelabeled.length,
  policyVsRuleRows: policyVsRuleRows.length,
  noOpRows: noOpRows.length,
  dir: root,
}, null, 2));
