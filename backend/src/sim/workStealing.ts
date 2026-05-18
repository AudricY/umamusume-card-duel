// Throughput work-stealing dispatch
// (docs/ai-research/scoping/r12-selfplay-gate-throughput.md).
//
// Static contiguous chunking (`sliceSize = ceil(N/workers)` then
// `.slice(w*size,(w+1)*size)`) had two defects on the 32-core box:
//   1. Selfplay 60 games / 24 workers → ceil(60/24)=3 → only 20 slices,
//      workers 20-23 got nothing (4 dark cores the entire selfplay stage).
//   2. Per-game length variance is ~100× (gate games 1.6s-168s). Static
//      assignment clusters long games onto unlucky workers → 2-3× per-worker
//      wall spread even when slice counts are equal. "Make games a multiple
//      of workers" does NOT fix this (gate is already 5 games/worker exactly
//      and still ~20% imbalanced) — only dynamic assignment does.
//
// Work-stealing replaces the pre-sliced one-shot assignment with a shared
// queue: each worker is seeded with a small initial batch, then pulls the
// next task on every completion. Workers spun = min(N, workers) naturally
// (fixes defect 1). Result aggregation is by task slot, never append order,
// so output rows/records are byte-identical regardless of which worker
// finished when — the per-game RNG is seeded solely from the game seed
// (mctsSelfPlay.ts `:selfplay`, evaluateModelVsHeuristic.ts `:modelSide`),
// never from worker id or completion order, so this is trajectory-neutral
// (proven bit-identical by training/r12_throughput_determinism_gate.py).
//
// Gated by a module constant defaulting ON, mirroring the W6 recipe-fix /
// throughput #2/#4 flag pattern. UMA_MCTS_WORK_STEALING=0 reverts to the
// static contiguous chunking path for an exact A/B.
export const MCTS_WORK_STEALING_ENABLED = process.env.UMA_MCTS_WORK_STEALING !== "0";

// Initial tasks handed to each worker before refill-on-completion kicks in.
// 2 keeps a worker's next game queued while it reports the previous one,
// hiding the IPC round-trip without creating a long static tail.
export const WORK_STEALING_PREFETCH = 2;
