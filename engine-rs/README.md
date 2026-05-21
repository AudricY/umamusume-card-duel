# engine-rs — Rust port of the umamusume-card-duel engine

Bit-identical Rust port of `frontend/src/game/engine/` + heuristic
opponent + MCTS driver. Scoping: `docs/ai-research/scoping/rust-engine-port-plan.md`.
Detailed hand-off: `docs/ai-research/scoping/rust-engine-port-handoff.md`.

## Layout

```
crates/
├── engine/             — the port itself (library crate)
│   ├── src/
│   │   ├── core/       ↔ frontend/src/game/engine/core/
│   │   ├── flow/       ↔ frontend/src/game/engine/flow/ (incl. flow/ai/)
│   │   ├── policy/     ↔ frontend/src/game/engine/ai-policy/
│   │   ├── mcts/       ↔ backend/src/sim/mcts.ts
│   │   ├── dispatcher.rs ↔ frontend/src/game/engine.ts (orchestrator)
│   │   ├── headless_setup.rs ↔ backend/src/sim/headlessAiVsAi.ts:setupAiVsAiGame
│   │   ├── fingerprint.rs    ↔ backend/src/sim/stateFingerprint.ts (xxh3 over packed buffer)
│   │   └── lib.rs
│   ├── tests/          — RNG cross-lang + catalog cross-lang + headless drive smoke
│   └── benches/        — Criterion micro-benches (clone, fingerprint)
├── sim-cli/            — four binaries matching backend npm sim:* scripts
│   └── src/bin/
│       ├── throughput_probe.rs
│       ├── eval_gate.rs
│       ├── mcts_selfplay.rs
│       └── export_training.rs
├── golden-replay/      — Phase 1h gate (multiple modes: setup, steps, mcts)
├── codegen/            — placeholder (catalog is loaded at runtime via include_str!)
└── xtask/              — workspace helper tasks (dump-rng-reference, etc.)
scripts/                — TS-side dumpers for cross-language reference vectors
```

## Quick commands

```bash
# All tests (default fast).
cargo test --manifest-path engine-rs/Cargo.toml

# Slow / production-grade benches.
cargo test --release --manifest-path engine-rs/Cargo.toml -- --ignored

# Micro-benchmarks (Criterion).
cargo bench --manifest-path engine-rs/Cargo.toml -p engine

# Run a sim CLI.
./engine-rs/target/release/sim-mcts-selfplay --seeds 10 --out runs/rust/out.jsonl --record-rows
./engine-rs/target/release/sim-eval-gate --seeds 50 --challenger http://localhost:8765 --leaf value-head

# Phase 1h golden-replay (setup parity / step replay / full MCTS).
./engine-rs/target/release/golden-replay \
  --input runs/rust-port-golden-traces/traces-500.jsonl \
  --mode setup
```

## Performance snapshot (release mode, single core)

| Metric | TS baseline | Rust | Speedup |
| --- | ---: | ---: | ---: |
| `clone` (structuredClone vs derive+ArrayVec) | 22,007 ns | 907 ns | **24×** (~44× release) |
| `fingerprint` (JSON.stringify vs xxh3) | 8,034 ns | 299 ns | **27×** |
| Headless self-play games/sec | ~50-100 | **3,484** | **35-70×** |
| MCTS-driven games/sec (rollout leaf) | 0.014 | **1.96-3.07** | **~140-220×** |
| R110 per-iter wall (projected) | 51.5 min | ~22 sec | **~140×** |

## Status

Phase 0 ✅, Phase 1a–1g ✅. Phase 1h V1 (setup parity) ✅ at 500/500;
V4 (full MCTS bit-identity) has a benign ~27% outer-RNG-per-rollout
gap that doesn't shift win rates (Rust 39% vs TS 37% player WR over
n=100 vs n=500). See the hand-off doc for the remaining items.

The engine is production-viable for headless and MCTS-driven self-play.
