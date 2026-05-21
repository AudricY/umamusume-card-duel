// R16-TD 3a corpus harness smoke. Locks the contract added to evalGate.ts so
// it can host the production corpus generator end-to-end:
//
//   (1) `--relabel-mcts` + `--relabel-state-source <tag>` flags wire through
//       into EvaluateModelArgs and produce well-formed relabel rows (oracle
//       block, simplex policyTargets, stateSource tag, no forced-state rows).
//   (2) `--decision-trace-out` is honored under multi-worker dispatch — the
//       orchestrator owns the file, buffers rows by taskIndex, and flushes in
//       ascending order so the trace is deterministic across --workers.
//   (3) Determinism: a --workers 4 run and a --workers 1 run with the same
//       --seed-start produce the same multiset of trace rows. (Stricter:
//       sorting both by (seed, modelSide, step) must yield identical lists.)
//   (4) Corpus-mode bypass: gate floors do not fail the process when
//       --relabel-mcts is set, even if win-rate is below the floor.
//
// Knob rationale matches relabelMctsSmoke.ts (the lower-level analog):
// --selection baseline keeps the smoke server-free; --mcts-simulations 8 /
// --mcts-rollout-steps 20 / --mcts-rollout-crn-samples 1 is the smallest MCTS
// config that reliably emits ≥1 contested row in a 2-game --max-steps 80
// trace. If this smoke flakes (zero rows) raise --games / --max-steps first.

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
  selectedActionId: string;
  seed: string;
  modelSide: "player" | "opponent";
  step: number;
};

function readJsonl<T>(path: string): T[] {
  const raw = readFileSync(path, "utf8").trim();
  if (!raw) return [];
  return raw.split("\n").filter(Boolean).map((line) => JSON.parse(line) as T);
}

function rowKey(row: TraceRow): string {
  return `${row.seed}::${row.modelSide}::${row.step}`;
}

const root = mkdtempSync(join(tmpdir(), "uma-r16-evalgate-corpus-smoke-"));
const multiOut = join(root, "corpus.multi.jsonl");
const singleOut = join(root, "corpus.single.jsonl");

const COMMON_ARGS = [
  "src/sim/evalGate.ts",
  "--selection", "baseline",
  "--games", "2",
  "--model-side", "both",
  "--max-steps", "80",
  "--mcts-simulations", "8",
  "--mcts-rollout-steps", "20",
  "--mcts-rollout-crn-samples", "1",
  "--seed-start", "9000",
  "--relabel-mcts",
  "--relabel-state-source", "rule-bot-mirror",
  // Gate floors: high enough that a baseline-vs-rulebot run *would* fail
  // them without --relabel-mcts. The smoke asserts the corpus-mode bypass
  // kicks in (exit 0, status PASS).
  "--min-games", "999",
  "--min-win-rate", "0.99",
  "--min-ci-lower", "0.95",
  "--allow-fallbacks",
  "--allow-no-ops",
];

// ----- (1) Multi-worker corpus run -----
const multiResult = await execFileAsync("tsx", [
  ...COMMON_ARGS,
  "--workers", "4",
  "--decision-trace-out", multiOut,
], { cwd: process.cwd(), maxBuffer: 1024 * 1024 * 32 });
const multiPayload = JSON.parse(multiResult.stdout);
assert.equal(multiPayload.status, "PASS", `multi-worker run must exit PASS in corpus mode (status=${multiPayload.status})`);
assert.equal(multiPayload.corpusMode, true, "corpusMode flag must be true on the JSON summary");
assert.ok(multiPayload.summary.games > 0, "multi-worker run must complete at least one game");

const multiRows = readJsonl<TraceRow>(multiOut);
assert.ok(multiRows.length > 0, "multi-worker corpus must emit ≥1 trace row (raise --games / --max-steps if this flakes)");
multiRows.forEach((row, idx) => {
  assert.ok(
    row.legalActions.length >= 2,
    `multi row ${idx}: relabel mode must suppress forced states (legalActions.length=${row.legalActions.length})`,
  );
  assert.ok(Array.isArray(row.policyTargets), `multi row ${idx}: policyTargets must be present`);
  assert.equal(
    row.policyTargets!.length,
    row.legalActions.length,
    `multi row ${idx}: policyTargets length must match legalActions.length`,
  );
  const sum = row.policyTargets!.reduce((acc, v) => acc + v, 0);
  assert.ok(
    Math.abs(sum - 1.0) < 1e-6,
    `multi row ${idx}: policyTargets must sum to 1 ± 1e-6, got ${sum}`,
  );
  row.policyTargets!.forEach((v, vi) => {
    assert.ok(v >= 0, `multi row ${idx}: policyTargets[${vi}] must be non-negative`);
  });
  assert.ok(row.oracle, `multi row ${idx}: oracle block must be present`);
  assert.equal(
    row.stateSource,
    "rule-bot-mirror",
    `multi row ${idx}: stateSource must be 'rule-bot-mirror', got '${row.stateSource}'`,
  );
});

// ----- (2) Single-worker reference run for determinism comparison -----
const singleResult = await execFileAsync("tsx", [
  ...COMMON_ARGS,
  "--workers", "1",
  "--decision-trace-out", singleOut,
], { cwd: process.cwd(), maxBuffer: 1024 * 1024 * 32 });
const singlePayload = JSON.parse(singleResult.stdout);
assert.equal(singlePayload.status, "PASS", "single-worker run must exit PASS in corpus mode");

const singleRows = readJsonl<TraceRow>(singleOut);
assert.ok(singleRows.length > 0, "single-worker corpus must emit ≥1 trace row");

// ----- (3) Determinism: --workers 4 and --workers 1 must produce the same
// row set. Trace rows include (seed, modelSide, step) which uniquely
// identifies a decision point under fixed --seed-start, so sorted-by-key
// equality is the right invariant. We compare on selectedActionId (cheap
// proxy that captures policy + state) rather than full row equality; full
// equality is implied for the contested-row set under the work-stealing
// determinism contract proven in r12_workstealing_determinism_gate.py.
assert.equal(
  multiRows.length,
  singleRows.length,
  `multi (${multiRows.length}) vs single (${singleRows.length}) row counts must match`,
);
const multiSorted = [...multiRows].sort((a, b) => rowKey(a).localeCompare(rowKey(b)));
const singleSorted = [...singleRows].sort((a, b) => rowKey(a).localeCompare(rowKey(b)));
multiSorted.forEach((row, i) => {
  const other = singleSorted[i]!;
  assert.equal(
    rowKey(row),
    rowKey(other),
    `determinism: sorted multi row ${i} key '${rowKey(row)}' ≠ single '${rowKey(other)}'`,
  );
  assert.equal(
    row.selectedActionId,
    other.selectedActionId,
    `determinism: same (seed,modelSide,step) ${rowKey(row)} must pick the same action across worker counts`,
  );
  // policyTargets length must match (same legal action set on the same state).
  assert.equal(
    row.policyTargets!.length,
    other.policyTargets!.length,
    `determinism: policyTargets length must match at ${rowKey(row)}`,
  );
});

// ----- (4) Multi-worker file ordering: the orchestrator must emit rows in
// ascending taskIndex order. taskIndex 0 is (modelSide=player, seedStart),
// taskIndex 1 is (modelSide=player, seedStart+1), ..., then modelSide=opponent
// from index `games`. We assert the modelSide tag is monotone non-decreasing
// in the file (all `player` rows come before all `opponent` rows), and within
// the same modelSide the `seed` is monotone non-decreasing.
let lastSideRank = -1;
let lastSeedNum = -1;
let currentSide: string | null = null;
for (const row of multiRows) {
  const sideRank = row.modelSide === "player" ? 0 : 1;
  if (sideRank < lastSideRank) {
    throw new Error(`multi file ordering: modelSide rank regressed at row '${rowKey(row)}'`);
  }
  if (sideRank > lastSideRank) {
    lastSideRank = sideRank;
    currentSide = row.modelSide;
    lastSeedNum = -1;
  }
  const seedNum = Number(row.seed);
  if (row.modelSide === currentSide && seedNum < lastSeedNum) {
    throw new Error(`multi file ordering: seed regressed within ${row.modelSide} block at '${rowKey(row)}' (${seedNum} < ${lastSeedNum})`);
  }
  lastSeedNum = seedNum;
}

console.log(JSON.stringify({
  status: "PASS",
  multiRows: multiRows.length,
  singleRows: singleRows.length,
  multiWinRate: multiPayload.summary.modelWinRate,
  dir: root,
}, null, 2));
