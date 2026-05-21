# engine-rs-bridge

Phase 2 NAPI binding. Lets Node code call the bit-identical Rust port
of the umamusume-card-duel engine directly, without spawning a sim-cli
subprocess per task.

## Build

```bash
cargo build --manifest-path engine-rs/Cargo.toml -p napi-bridge --release --lib
```

That produces `engine-rs/target/release/libnapi_bridge.so`. The
package's `index.js` auto-copies it to `napi_bridge.node` on first
require — no manual step.

## Use

```ts
import { createRequire } from "node:module";
import type { NapiBridge } from "./engine-rs/crates/napi-bridge";

const require = createRequire(import.meta.url);
const bridge: NapiBridge = require("./engine-rs/crates/napi-bridge");

const init = JSON.parse(bridge.createGameJson("seed-0"));
// → { stateJson, rngStateJson }

const summary = JSON.parse(bridge.driveHeuristicGameJson("seed-0", 1000));
// → { finalStateHash, steps, winner, gameOver, terminalReason }
```

CommonJS:

```js
const bridge = require("./engine-rs/crates/napi-bridge");
const v = bridge.engineVersion();  // "rust-port v0.1.0 — catalog=106 cards"
```

## Surface (8 functions)

| Function | Purpose |
| --- | --- |
| `engineVersion()` | Sentinel; confirms bridge loaded + catalog ready. |
| `createGameJson(seed)` | Fresh AI-vs-AI game state. Returns `{stateJson, rngStateJson}`. |
| `advanceStepJson(state, rng)` | One heuristic-AI step for `state.currentSide`. |
| `legalActionsJson(state, rng)` | JSON-encoded `LegalAiAction[]` for the active side. |
| `runMctsJson(state, rng, args, seed)` | Run MCTS at the given state; state unchanged. |
| `mctsStepJson(state, rng, args, seed)` | Run MCTS + pick most-visited action + apply. |
| `stateHashForJson(state)` | 32-char xxh3 fingerprint of a packed state. |
| `driveHeuristicGameJson(seed, maxSteps)` | Full heuristic-vs-heuristic game in pure Rust; returns summary. |

See `index.d.ts` for the full TypeScript interface.

## Bulk vs composable surface — which to use

For **driving a full game** in pure Rust (matches sim-cli output bit-for-bit):

```ts
bridge.driveHeuristicGameJson(seed, maxSteps)
bridge.driveMctsGameJson(seed, "player", maxSteps, mctsArgsJson)
```

These return only a summary, never expose the GameState across the FFI per-step.

For **interactive use** — replaying a recorded trace, inspecting state mid-game, hand-picking actions, integrating with a UI — use the composable functions:

```ts
bridge.createGameJson(seed)
bridge.advanceStepJson(stateJson, rngStateJson)
bridge.legalActionsJson(stateJson, rngStateJson)
bridge.mctsStepJson(stateJson, rngStateJson, mctsArgsJson, mctsSeed)
bridge.runMctsJson(stateJson, rngStateJson, mctsArgsJson, mctsSeed)
bridge.stateHashForJson(stateJson)
```

⚠️ Per-step composable invocations **may produce slightly different traces than the bulk drivers for some seeds** (the multi-step JSON roundtrip can interact with the heuristic AI's log-reading checks). For training-data generation or eval-gate workloads where you need parity with sim-cli, prefer the bulk drivers.

## Performance

Measured at N=100 short heuristic games:

```
NAPI in-process:     3369 games/s   0.30 ms/game   wall 30 ms
Subprocess-per-game:  300 games/s   3.33 ms/game   wall 333 ms
```

**11.2× speedup** over spawning sim-cli binaries. Subprocess startup
(~3 ms/game on Linux) dominates short tasks. Per-step engine compute
is ~7 μs.

Pipelines that already spawn one process per worker and stream many
games (e.g. `sim-mcts-selfplay --games N`) amortize startup and would
see only the per-step FFI overhead.

## Run the smoke + benchmark

```bash
node engine-rs/scripts/napi-smoke.mjs        # ~1 second
node engine-rs/scripts/napi-bench.mjs 100    # ~0.5 seconds
```

The smoke also runs as the last stage of
`engine-rs/scripts/smoke-all-clis.sh`, so any NAPI regression breaks
the full regression net.

## Status

- Phase 2 NAPI binding: complete (this package).
- Phase 2 optional WASM: not started.
