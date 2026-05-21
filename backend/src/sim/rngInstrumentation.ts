// Phase 0 golden-trace harness (docs/ai-research/scoping/rust-engine-port-plan.md
// § 3 "Phase 0"): opt-in instrumentation that counts `Rng.next()` calls.
//
// Wraps a Rng so every `.next()` invocation bumps a shared counter and every
// `.fork(label)` returns another wrapped Rng that shares the same counter.
// Wrapped rngs are drop-in replacements for the originals — the engine API
// (`Rng.next: () => number`, `Rng.fork: (label) => Rng`) is preserved.
//
// Scope of the counter:
//   - Every direct `.next()` call on a wrapped Rng or any of its (transitively)
//     forked descendants is counted.
//   - When the wrapped Rng is installed via `withRng(rng, ...)`, the storage
//     provider returns the wrapped instance and `randomFloat()` therefore
//     counts too (it calls `storage.get()!.next()`).
//   - Pass-as-callback usage like `advancePlayerAiTurnStep(state, coins, rng.next)`
//     also counts, because the wrapped `next` is the bound function the engine
//     receives.
//   - Rngs created INDEPENDENTLY via `createSeededRng` inside callees (e.g. the
//     `mcts.ts` `rootRng = createSeededRng(seed, "mcts-root")` path) are NOT
//     counted by this counter — they live in a separate rng tree. The recorder
//     documents this so its per-turn count is interpreted as
//     "draws from the recorder-owned rng tree this turn", not
//     "every PRNG draw the engine made this turn".
//
// Public API is intentionally tiny: `createRngCounter()` returns a counter
// object, and `instrumentRng(rng, counter)` returns a wrapped Rng. The
// `core/random.ts` module is untouched.
import type { Rng } from "../../../frontend/src/game/engine/core/random";

export type RngCounter = {
  count: number;
  /** Reads the current count and resets to 0 in one step. */
  takeAndReset: () => number;
};

export function createRngCounter(): RngCounter {
  const counter: RngCounter = {
    count: 0,
    takeAndReset: () => {
      const value = counter.count;
      counter.count = 0;
      return value;
    },
  };
  return counter;
}

export function instrumentRng(rng: Rng, counter: RngCounter): Rng {
  const wrappedNext = (): number => {
    counter.count += 1;
    return rng.next();
  };
  const wrappedFork = (label: string): Rng => instrumentRng(rng.fork(label), counter);
  const wrapped: Rng = {
    next: wrappedNext,
    fork: wrappedFork,
  };
  if (rng.label !== undefined) wrapped.label = rng.label;
  return wrapped;
}
