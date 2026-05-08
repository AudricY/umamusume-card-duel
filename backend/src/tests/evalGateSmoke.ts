import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync } from "node:fs";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

const server = createServer(handlePredict);
await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
const address = server.address();
assert.ok(address && typeof address === "object", "fake model server should have a TCP address");
const modelUrl = `http://127.0.0.1:${address.port}`;

try {
  const evalRun = await execFileAsync("tsx", [
    "src/sim/evaluateModelVsHeuristic.ts",
    "--selection", "policy",
    "--model-url", modelUrl,
    "--games", "1",
    "--model-side", "player",
    "--max-steps", "120",
  ], { cwd: process.cwd(), maxBuffer: 1024 * 1024 * 8 });
  const evalPayload = JSON.parse(evalRun.stdout);
  assert.equal(evalPayload.summary.games, 1, "fake model eval should run one game");
  assert.equal(evalPayload.summary.heuristicFallbacks, 0, "fake pass-biased policy should not require heuristic fallbacks");

  let failed = false;
  try {
    await execFileAsync("tsx", [
      "src/sim/evalGate.ts",
      "--selection", "rollout",
      "--games", "1",
      "--model-side", "player",
      "--max-steps", "80",
      "--rollout-steps", "20",
      "--min-games", "999",
    ], { cwd: process.cwd(), maxBuffer: 1024 * 1024 * 8 });
  } catch (error) {
    failed = true;
    const stdout = String((error as { stdout?: unknown }).stdout ?? "");
    const payload = JSON.parse(stdout);
    assert.equal(payload.status, "FAIL", "eval gate should report FAIL on threshold miss");
    assert.ok(payload.failures.some((failure: string) => failure.includes("minGames")), "eval gate should explain minGames failure");
  }
  assert.equal(failed, true, "eval gate command should exit nonzero on threshold miss");

  // Planted bad-policy regression: inverted-baseline must lose to the rule bot.
  // The gate is configured with a 50% Wilson-lower-bound floor; an inverted
  // policy should trip it.
  let plantedFailed = false;
  try {
    await execFileAsync("tsx", [
      "src/sim/evalGate.ts",
      "--selection", "inverted-baseline",
      "--games", "8",
      "--model-side", "both",
      "--max-steps", "300",
      "--min-games", "16",
      "--min-ci-lower", "0.5",
      "--allow-fallbacks",
      "--allow-no-ops",
    ], { cwd: process.cwd(), maxBuffer: 1024 * 1024 * 8 });
  } catch (error) {
    plantedFailed = true;
    const stdout = String((error as { stdout?: unknown }).stdout ?? "");
    const payload = JSON.parse(stdout);
    assert.equal(payload.status, "FAIL", "inverted-baseline should fail strength gate");
    assert.ok(
      payload.failures.some((f: string) => f.includes("wilsonLower") || f.includes("minCiLower")),
      `inverted-baseline failure should reference Wilson lower bound floor: ${JSON.stringify(payload.failures)}`,
    );
  }
  assert.equal(plantedFailed, true, "planted bad-policy gate must exit nonzero");

  // The same inverted-baseline run should pass when expectFail is set.
  const plantedPass = await execFileAsync("tsx", [
    "src/sim/evalGate.ts",
    "--selection", "inverted-baseline",
    "--games", "4",
    "--model-side", "both",
    "--max-steps", "300",
    "--min-games", "8",
    "--min-ci-lower", "0.5",
    "--allow-fallbacks",
    "--allow-no-ops",
    "--expect-fail",
  ], { cwd: process.cwd(), maxBuffer: 1024 * 1024 * 8 });
  const plantedPassPayload = JSON.parse(plantedPass.stdout);
  assert.equal(plantedPassPayload.status, "PASS", "expectFail flips inverted-baseline FAIL into PASS");
  assert.ok(typeof plantedPassPayload.summary.selectedNoOps === "number", "summary should report selectedNoOps count");
  assert.ok(typeof plantedPassPayload.summary.selectedExplicitPasses === "number", "summary should report selectedExplicitPasses count");

  // --require-manifest with no --manifest-out must exit nonzero with a
  // structured error JSON.
  let manifestRejected = false;
  try {
    await execFileAsync("tsx", [
      "src/sim/evalGate.ts",
      "--selection", "baseline",
      "--games", "1",
      "--model-side", "player",
      "--max-steps", "60",
      "--min-games", "1",
      "--require-manifest",
    ], { cwd: process.cwd(), maxBuffer: 1024 * 1024 * 8 });
  } catch (error) {
    manifestRejected = true;
    const stderr = String((error as { stderr?: unknown }).stderr ?? "");
    const payload = JSON.parse(stderr);
    assert.equal(payload.error, "EvalGateError");
    assert.equal(payload.code, "missing_manifest_path");
  }
  assert.equal(manifestRejected, true, "--require-manifest without --manifest-out must error");

  // N-step cycle detection smoke: run a tight cycle window and assert
  // the field shows up in terminal reasons (or cycleStalled key absent if
  // no cycle hit). The smoke just verifies the field plumbing.
  const cycleRun = await execFileAsync("tsx", [
    "src/sim/evaluateModelVsHeuristic.ts",
    "--selection", "baseline",
    "--games", "2",
    "--model-side", "player",
    "--max-steps", "80",
    "--cycle-window", "4",
  ], { cwd: process.cwd(), maxBuffer: 1024 * 1024 * 8 });
  const cyclePayload = JSON.parse(cycleRun.stdout);
  assert.equal(cyclePayload.summary.games, 2, "cycle-window run should still complete games");

  // Reviewer 2 #5: cycle detection at the default --cycle-min-visits=3 must
  // tolerate transient hash aliasing — i.e. a single repeat in the window must
  // not trip cycleStalled. Compare against --cycle-min-visits=2 (legacy
  // strict) to confirm the looser default is actually behaving as intended.
  // We compare cycleStalled rates at a small game count; the strict mode
  // should be ≥ the loose mode by construction.
  async function cycleStalledRate(minVisits: string): Promise<number> {
    const run = await execFileAsync("tsx", [
      "src/sim/evaluateModelVsHeuristic.ts",
      "--selection", "baseline",
      "--games", "8",
      "--model-side", "both",
      "--max-steps", "300",
      "--cycle-window", "4",
      "--cycle-min-visits", minVisits,
    ], { cwd: process.cwd(), maxBuffer: 1024 * 1024 * 16 });
    const payload = JSON.parse(run.stdout);
    return Number(payload.summary.terminalReasons?.cycleStalled ?? 0);
  }
  const strictStalled = await cycleStalledRate("2");
  const looseStalled = await cycleStalledRate("3");
  assert.ok(
    looseStalled <= strictStalled,
    `--cycle-min-visits=3 should never report more cycleStalled games than =2: loose=${looseStalled} strict=${strictStalled}`,
  );

  const traceDir = mkdtempSync(join(tmpdir(), "uma-trace-smoke-"));
  const traceOut = join(traceDir, "trace.jsonl");
  await execFileAsync("tsx", [
    "src/sim/evaluateModelVsHeuristic.ts",
    "--selection", "rollout",
    "--games", "1",
    "--model-side", "both",
    "--max-steps", "100",
    "--rollout-steps", "30",
    "--decision-trace-out", traceOut,
    "--trace-teacher", "rollout",
  ], { cwd: process.cwd(), maxBuffer: 1024 * 1024 * 8 });
  assert.ok(existsSync(traceOut), "decision trace file should be written");
  const traceRows = readJsonl(traceOut);
  assert.ok(traceRows.length > 0, "decision trace should contain model-visited decisions");
  assert.ok(new Set(traceRows.map((row) => row.modelSide)).has("player"), "trace should include player-side model decisions");
  assert.ok(new Set(traceRows.map((row) => row.modelSide)).has("opponent"), "trace should include opponent-side model decisions");
  traceRows.forEach((row) => {
    assert.equal(row.source, "model-visited");
    assert.equal(row.observation.opponent.handCardIds, undefined, "trace observation must not leak opponent hand IDs");
    assert.ok(row.result?.winner === "player" || row.result?.winner === "opponent" || row.result?.winner === null, "trace rows should include final result");
    assert.equal(row.teacher?.selection, "rollout", "trace rows should include requested teacher labels");
    assert.ok(typeof row.teacher?.selectedActionId === "string", "trace teacher should include selected action ID");
    // Item 18 negative test: rollout selection does not consult the model
    // server, so behaviorPolicy must be absent on these trace rows.
    assert.equal(row.behaviorPolicy, undefined, "rollout-source trace must not carry behaviorPolicy");
  });

  // Item 18 positive test: policy-source traces *must* carry behaviorPolicy
  // so PPO can recover importance-sampling ratios on warm-start rollouts.
  const policyTraceOut = join(traceDir, "policy-trace.jsonl");
  await execFileAsync("tsx", [
    "src/sim/evaluateModelVsHeuristic.ts",
    "--selection", "policy",
    "--model-url", modelUrl,
    "--games", "1",
    "--model-side", "player",
    "--max-steps", "120",
    "--decision-trace-out", policyTraceOut,
  ], { cwd: process.cwd(), maxBuffer: 1024 * 1024 * 8 });
  const policyTraceRows = readJsonl(policyTraceOut);
  assert.ok(policyTraceRows.length > 0, "policy decision trace should produce rows");
  policyTraceRows.forEach((row) => {
    assert.ok(row.behaviorPolicy, `policy-source trace row must carry behaviorPolicy: ${JSON.stringify(row).slice(0, 120)}...`);
    assert.ok(["fake-uniform", "single-action"].includes(row.behaviorPolicy.kind), `unexpected kind ${row.behaviorPolicy.kind}`);
    assert.ok(Array.isArray(row.behaviorPolicy.actionLogProbs), "behaviorPolicy.actionLogProbs must be an array");
    if (row.behaviorPolicy.kind === "fake-uniform") {
      assert.equal(row.behaviorPolicy.actionLogProbs.length, row.legalActions.length, "actionLogProbs must align to legalActions");
      assert.ok(typeof row.behaviorPolicy.selectedLogProb === "number", "behaviorPolicy.selectedLogProb must be a number");
      const expectedLogP = Math.log(1 / row.legalActions.length);
      assert.ok(
        Math.abs(row.behaviorPolicy.selectedLogProb - expectedLogP) < 1e-6,
        `selectedLogProb ${row.behaviorPolicy.selectedLogProb} should be log(1/n) ${expectedLogP}`,
      );
    } else {
      assert.equal(row.behaviorPolicy.selectedLogProb, 0, "single-action selectedLogProb must be 0");
      assert.equal(row.legalActions.length, 1, "single-action snapshots imply legalActions.length == 1");
    }
  });
} finally {
  server.close();
}

console.log(JSON.stringify({ status: "PASS", modelUrl }, null, 2));

function handlePredict(request: IncomingMessage, response: ServerResponse): void {
  if (request.method === "GET" && request.url === "/health") {
    sendJson(response, 200, { status: "ok" });
    return;
  }
  if (request.method !== "POST" || request.url !== "/predict") {
    sendJson(response, 404, { error: "not found" });
    return;
  }
  let body = "";
  request.setEncoding("utf8");
  request.on("data", (chunk) => { body += chunk; });
  request.on("end", () => {
    const payload = JSON.parse(body);
    const legalActions = payload.legalActions ?? [];
    const passIndex = legalActions.findIndex((action: { kind?: string }) => action.kind === "pass" || action.kind === "endTurn");
    const selected = Math.max(0, passIndex);
    // Item 18: emit a fake but well-shaped behavior-policy distribution so the
    // evaluator's persistence path is exercised under tests. Uniform over
    // legal actions, log-prob = log(1/n).
    const n = Math.max(1, legalActions.length);
    const uniformLogP = Math.log(1 / n);
    const actionLogProbs = legalActions.map(() => uniformLogP);
    const actionProbs = legalActions.map(() => 1 / n);
    sendJson(response, 200, {
      selectedIndex: [selected],
      actionLogProbs: [actionLogProbs],
      actionProbs: [actionProbs],
      selectedLogProb: [uniformLogP],
      behaviorPolicy: { kind: "fake-uniform", temperature: 1 },
    });
  });
}

function sendJson(response: ServerResponse, status: number, payload: unknown): void {
  const body = JSON.stringify(payload);
  response.writeHead(status, { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(body) });
  response.end(body);
}

function readJsonl(path: string): any[] {
  return readFileSync(path, "utf8")
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}
