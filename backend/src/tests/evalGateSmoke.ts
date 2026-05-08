import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
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
