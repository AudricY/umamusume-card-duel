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

const stateJson = bridge.createGameJson("0");
const state = JSON.parse(stateJson);
if (state.phase !== "play" && state.phase !== "Play") {
  throw new Error(`unexpected initial phase: ${state.phase}`);
}
if (state.gameOver !== false && state.game_over !== false) {
  throw new Error(`fresh game should not be game_over: ${stateJson.slice(0, 200)}`);
}
console.log(`createGameJson("0") -> ${stateJson.length} bytes, phase=${state.phase}`);

const hash = bridge.stateHashForJson(stateJson);
if (typeof hash !== "string" || hash.length !== 32 || !/^[0-9a-f]+$/.test(hash)) {
  throw new Error(`unexpected hash shape: ${hash}`);
}
console.log("stateHashForJson ->", hash);

const hash2 = bridge.stateHashForJson(stateJson);
if (hash !== hash2) {
  throw new Error("hash must be deterministic for the same input");
}

// Cross-seed: hashes must differ for different seeds.
const s2 = bridge.createGameJson("1");
const h2 = bridge.stateHashForJson(s2);
if (h2 === hash) {
  throw new Error("hashes for different seeds should differ");
}

console.log("✅ napi-bridge smoke OK");
