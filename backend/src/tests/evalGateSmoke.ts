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
    sendJson(response, 200, { selectedIndex: [Math.max(0, passIndex)] });
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
