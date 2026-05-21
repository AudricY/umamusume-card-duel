// Phase 2 NAPI smoke test. Loads the compiled libnapi_bridge.so and
// confirms the three scaffolded functions work end-to-end. Run with:
//
//   node engine-rs/scripts/napi-smoke.mjs
//
// Build the .so first: `cargo build -p napi-bridge --release --lib`
// (manifest at engine-rs/Cargo.toml).

import { createRequire } from "node:module";
import { existsSync, copyFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(here, "..", "..");
const built = join(repoRoot, "engine-rs", "target", "release", "libnapi_bridge.so");
if (!existsSync(built)) {
  console.error(`napi-bridge artifact missing: ${built}`);
  console.error("Build with: cargo build --manifest-path engine-rs/Cargo.toml -p napi-bridge --release --lib");
  process.exit(1);
}

// napi-rs expects the artifact to be loadable as a .node file. Create
// a symlink-style copy with .node extension if one doesn't exist.
const loadable = join(repoRoot, "engine-rs", "target", "release", "napi_bridge.node");
if (!existsSync(loadable)) {
  copyFileSync(built, loadable);
}

const require = createRequire(import.meta.url);
const bridge = require(loadable);

const version = bridge.engineVersion();
console.log("engineVersion ->", version);
if (!version.startsWith("rust-port") || !version.includes("catalog=")) {
  throw new Error(`unexpected version string: ${version}`);
}

// createGameJson now returns {stateJson, rngStateJson}.
const initRaw = bridge.createGameJson("0");
const init = JSON.parse(initRaw);
const state = JSON.parse(init.stateJson);
if (state.phase !== "play" && state.phase !== "Play") {
  throw new Error(`unexpected initial phase: ${state.phase}`);
}
console.log(`createGameJson("0") -> stateJson=${init.stateJson.length}B rngStateJson=${init.rngStateJson.length}B phase=${state.phase}`);

const hash = bridge.stateHashForJson(init.stateJson);
console.log("stateHashForJson ->", hash);
if (hash.length !== 32 || !/^[0-9a-f]+$/.test(hash)) {
  throw new Error(`unexpected hash shape: ${hash}`);
}

// Drive a full game via repeated advance_step_json. GameState serializes
// as snake_case (e.g., game_over, current_side, first_player) since the
// struct doesn't carry #[serde(rename_all = "camelCase")] at the top
// level — only nested types like PublicObservation do.
let curStateJson = init.stateJson;
let curRngJson = init.rngStateJson;
let steps = 0;
const maxSteps = 1000;
let priorHash = hash;
for (; steps < maxSteps; steps += 1) {
  const cs = JSON.parse(curStateJson);
  if (cs.game_over === true || cs.current_side === "done") break;
  const raw = bridge.advanceStepJson(curStateJson, curRngJson);
  const next = JSON.parse(raw);
  const nextHash = bridge.stateHashForJson(next.stateJson);
  if (nextHash === priorHash) {
    throw new Error(`step ${steps} did not change state — stall`);
  }
  priorHash = nextHash;
  curStateJson = next.stateJson;
  curRngJson = next.rngStateJson;
}
const final = JSON.parse(curStateJson);
console.log(`advanceStepJson drove ${steps} steps, game_over=${final.game_over} winner=${final.winner ?? null}`);
if (final.game_over !== true) {
  throw new Error(`expected game_over after ${steps} steps; got: ${curStateJson.slice(0, 200)}`);
}

// legal_actions_json at the initial state (before first advance).
const legalRaw = bridge.legalActionsJson(init.stateJson, init.rngStateJson);
const legal = JSON.parse(legalRaw);
if (!Array.isArray(legal) || legal.length === 0) {
  throw new Error(`expected non-empty legal-actions array; got: ${legalRaw.slice(0, 200)}`);
}
console.log(`legalActionsJson at t=0 -> ${legal.length} actions (kinds: ${[...new Set(legal.map(a => a.kind))].join(",")})`);

// Determinism: same seed → same final hash.
const init2 = JSON.parse(bridge.createGameJson("0"));
let s2 = init2.stateJson, r2 = init2.rngStateJson;
for (let i = 0; i < steps; i += 1) {
  const x = JSON.parse(bridge.advanceStepJson(s2, r2));
  s2 = x.stateJson; r2 = x.rngStateJson;
}
const replayHash = bridge.stateHashForJson(s2);
const finalHash = bridge.stateHashForJson(curStateJson);
if (replayHash !== finalHash) {
  throw new Error(`replay hash ${replayHash} != original ${finalHash} — non-determinism`);
}
console.log(`deterministic replay: ${steps} steps, final hash ${finalHash} matches`);

console.log("✅ napi-bridge smoke OK (engineVersion, createGameJson, advanceStepJson, legalActionsJson, stateHashForJson)");
