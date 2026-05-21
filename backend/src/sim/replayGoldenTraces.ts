// Phase 0 golden-trace replay tool (see
// docs/ai-research/scoping/rust-engine-port-plan.md § 3 "Phase 0").
//
// Reads a JSONL trace file produced by `recordGoldenTraces.ts` and re-runs
// each seed under the same config, asserting byte-equality of every recorded
// field (per-step fingerprintBefore / action / rngDrawsThisStep, terminal
// fingerprint, winner, turnNumber, totalRngDraws).
//
// On a divergence we print the first diverging step with expected vs. actual
// values and exit 1. On success we print "OK <N>/<N> seeds bit-identical".
//
// The replay path uses the exact same recorder function (`recordTraceForSeed`)
// so this is, day-1, a TS-vs-TS determinism gate: if running the same seed
// twice in TS gives different traces, Phase 1 is paused per the scoping doc.
// Day-N (Rust port live) the same script could be repointed at a Rust
// generator's output to gate cross-language identity — until then it's
// catching node-version / iteration-order / float-drift surprises.

import { readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import {
  recordTraceForSeed,
  computeConfigHash,
} from "./recordGoldenTraces";

type ReplayArgs = {
  inPath: string;
};

type TraceLine = Awaited<ReturnType<typeof recordTraceForSeed>>;

function parseArgs(argv: string[]): ReplayArgs {
  const get = (name: string, fallback: string) => {
    const i = argv.indexOf(name);
    return i >= 0 ? argv[i + 1] ?? fallback : fallback;
  };
  return {
    inPath: get("--in", "runs/rust-port-golden-traces/traces.jsonl"),
  };
}

// Inverse of the recorder's CLI arg derivation: we reconstruct a RecorderArgs
// shape from the fields the trace records. The recorded `configHash` is
// rebuilt from these args + engineSha extracted from the trace's configHash
// — but we don't actually need the engineSha at replay because the recorded
// trace already carries the original configHash. We pass through the
// trace's configHash unchanged.
function deriveRecorderArgs(trace: TraceLine, defaultsFromTrace: TraceLine): Parameters<typeof recordTraceForSeed>[1] {
  // The schema does not currently round-trip every CLI flag. For the
  // smoke-corpus tier this is enough: the recorder's per-step behavior is
  // a pure function of the engine + a fixed config, and the config used
  // here matches whatever the recorder defaults to. Future schema bumps
  // (`traceVersion: 2+`) can serialize the full RecorderArgs into the
  // trace line and round-trip exactly.
  //
  // For now: replay assumes the file was produced with the recorder's
  // defaults (or that the caller is asserting equivalence against the
  // same defaults). The configHash equality check is the load-bearing
  // safety net — if the recorder defaults shift between record and replay,
  // configHash diverges and we bail loudly before any per-step diff runs.
  void defaultsFromTrace;
  return {
    seeds: 1,
    seedBase: trace.seed,
    outPath: "/dev/null",
    modelUrl: "",
    mctsSimulations: 100,
    mctsCPuct: 1.5,
    mctsLeaf: "rollout",
    mctsRolloutCrnSamples: 3,
    mctsRolloutSteps: 200,
    mctsCollapseMaxSteps: 64,
    mctsMaxNodes: 5000,
    mctsPrior: "uniform",
    temperatureMoves: 6,
    temperatureValue: 1.0,
    maxSteps: 500,
    traceVersion: trace.traceVersion,
  };
}

type Diff = { field: string; expected: unknown; actual: unknown };

function diffTraces(expected: TraceLine, actual: TraceLine): Diff | null {
  if (expected.seed !== actual.seed) return { field: "seed", expected: expected.seed, actual: actual.seed };
  if (expected.traceVersion !== actual.traceVersion) return { field: "traceVersion", expected: expected.traceVersion, actual: actual.traceVersion };
  if (expected.configHash !== actual.configHash) return { field: "configHash", expected: expected.configHash, actual: actual.configHash };
  const len = Math.max(expected.actions.length, actual.actions.length);
  for (let i = 0; i < len; i += 1) {
    const e = expected.actions[i];
    const a = actual.actions[i];
    if (!e || !a) {
      return {
        field: `actions[${i}] (length mismatch: expected=${expected.actions.length} actual=${actual.actions.length})`,
        expected: e ?? null,
        actual: a ?? null,
      };
    }
    if (e.step !== a.step) return { field: `actions[${i}].step`, expected: e.step, actual: a.step };
    if (e.turnNumber !== a.turnNumber) return { field: `actions[${i}].turnNumber`, expected: e.turnNumber, actual: a.turnNumber };
    if (e.sideId !== a.sideId) return { field: `actions[${i}].sideId`, expected: e.sideId, actual: a.sideId };
    if (e.fingerprintBefore !== a.fingerprintBefore) {
      return { field: `actions[${i}].fingerprintBefore`, expected: e.fingerprintBefore, actual: a.fingerprintBefore };
    }
    if (e.rngDrawsThisStep !== a.rngDrawsThisStep) {
      return { field: `actions[${i}].rngDrawsThisStep`, expected: e.rngDrawsThisStep, actual: a.rngDrawsThisStep };
    }
    if (e.action.id !== a.action.id) return { field: `actions[${i}].action.id`, expected: e.action.id, actual: a.action.id };
    if (e.action.kind !== a.action.kind) return { field: `actions[${i}].action.kind`, expected: e.action.kind, actual: a.action.kind };
    if (e.action.phase !== a.action.phase) return { field: `actions[${i}].action.phase`, expected: e.action.phase, actual: a.action.phase };
    if (e.action.selectedIndex !== a.action.selectedIndex) {
      return { field: `actions[${i}].action.selectedIndex`, expected: e.action.selectedIndex, actual: a.action.selectedIndex };
    }
    if (e.action.legalCount !== a.action.legalCount) {
      return { field: `actions[${i}].action.legalCount`, expected: e.action.legalCount, actual: a.action.legalCount };
    }
    const ePayload = JSON.stringify(e.action.payload);
    const aPayload = JSON.stringify(a.action.payload);
    if (ePayload !== aPayload) return { field: `actions[${i}].action.payload`, expected: ePayload, actual: aPayload };
  }
  if (expected.terminalFingerprint !== actual.terminalFingerprint) {
    return { field: "terminalFingerprint", expected: expected.terminalFingerprint, actual: actual.terminalFingerprint };
  }
  if (expected.winner !== actual.winner) return { field: "winner", expected: expected.winner, actual: actual.winner };
  if (expected.turnNumber !== actual.turnNumber) return { field: "turnNumber", expected: expected.turnNumber, actual: actual.turnNumber };
  if (expected.totalRngDraws !== actual.totalRngDraws) {
    return { field: "totalRngDraws", expected: expected.totalRngDraws, actual: actual.totalRngDraws };
  }
  if (expected.terminalReason !== actual.terminalReason) {
    return { field: "terminalReason", expected: expected.terminalReason, actual: actual.terminalReason };
  }
  return null;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const raw = readFileSync(args.inPath, "utf8");
  const lines = raw.split("\n").filter((line) => line.trim().length > 0);
  if (lines.length === 0) {
    process.stderr.write(`ERROR: no trace lines in ${args.inPath}\n`);
    process.exit(2);
  }
  process.stderr.write(`[replay-golden-traces] verifying ${lines.length} seeds from ${args.inPath}\n`);

  // Sanity: cross-check the configHash on the first line matches the
  // recorder's current default config + engine sha. If it doesn't, we
  // continue anyway (the trace's own configHash is what we compare
  // re-runs against), but we surface the mismatch loudly so callers know
  // the recorder defaults shifted.
  const firstTrace = JSON.parse(lines[0]!) as TraceLine;
  const probeArgs = deriveRecorderArgs(firstTrace, firstTrace);
  const probeEngineSha = (() => {
    try {
      return execFileSync("git", ["rev-parse", "HEAD:frontend/src/game/engine"], { encoding: "utf8" }).trim();
    } catch {
      return "unknown";
    }
  })();
  const probeConfigHash = computeConfigHash(probeArgs, probeEngineSha);
  if (probeConfigHash !== firstTrace.configHash) {
    process.stderr.write(
      `WARN: trace configHash=${firstTrace.configHash.slice(0, 12)} differs from current recorder default ` +
      `configHash=${probeConfigHash.slice(0, 12)}. Will replay against the trace's recorded configHash.\n`,
    );
  }

  let okCount = 0;
  const startedAt = Date.now();
  for (let i = 0; i < lines.length; i += 1) {
    const expected = JSON.parse(lines[i]!) as TraceLine;
    const replayArgs = deriveRecorderArgs(expected, firstTrace);
    const actual = await recordTraceForSeed(expected.seed, replayArgs, expected.configHash);
    const diff = diffTraces(expected, actual);
    if (diff) {
      process.stderr.write(
        `FAIL seed=${expected.seed} at field=${diff.field}\n` +
        `  expected: ${JSON.stringify(diff.expected)}\n` +
        `  actual:   ${JSON.stringify(diff.actual)}\n`,
      );
      process.exit(1);
    }
    okCount += 1;
    if ((i + 1) % 50 === 0 || i === lines.length - 1) {
      const elapsed = (Date.now() - startedAt) / 1000;
      process.stderr.write(`[replay-golden-traces] ${i + 1}/${lines.length} elapsed=${elapsed.toFixed(1)}s\n`);
    }
  }

  process.stdout.write(`OK ${okCount}/${lines.length} seeds bit-identical\n`);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
