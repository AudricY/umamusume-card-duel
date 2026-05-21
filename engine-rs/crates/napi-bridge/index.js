// Minimal loader for the Phase 2 NAPI bridge. Consumers do:
//
//   const bridge = require("/path/to/engine-rs/crates/napi-bridge");
//   const v = bridge.engineVersion();
//
// Or via npm install once published.

const { createRequire } = require("node:module");
const { existsSync, copyFileSync } = require("node:fs");
const { join, dirname } = require("node:path");

const here = __dirname;
// Build output path. Defaults to release; flip via NAPI_PROFILE=debug.
const profile = process.env.NAPI_PROFILE || "release";
const targetDir = join(here, "..", "..", "target", profile);
const built = join(targetDir, "libnapi_bridge.so");
const loadable = join(targetDir, "napi_bridge.node");

if (!existsSync(loadable)) {
  if (!existsSync(built)) {
    throw new Error(
      `napi-bridge artifact missing: ${built}\n` +
        `Build with: cargo build --manifest-path engine-rs/Cargo.toml -p napi-bridge --${profile} --lib`,
    );
  }
  copyFileSync(built, loadable);
}

const req = createRequire(__filename);
module.exports = req(loadable);
