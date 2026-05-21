// Dump TS-side rolloutHeuristic per-advance trace for a single rollout,
// matching the Rust set_verbose_first_rollout output format.
//
// Usage:
//   npx tsx engine-rs/scripts/dump-ts-rollout.ts <seed> <recorded-action-id> > ts-rollout.txt
//
// Reproduces the recorder's setupAiVsAiGame + one advance + one
// rolloutHeuristic (with K=3 CRN, takes the FIRST i=0 rollout), then
// prints per-advance details.
//
// Pair with Rust:
//   ./engine-rs/target/release/golden-replay --mode mcts --limit-seeds 1 --limit-steps 3
// and diff the two dumps to localize the heuristic-divergence step.

import {
  advancePlayerAiTurnStep,
  advanceOpponentTurnStep,
  cloneGame,
  createGame,
  chooseOpeningCoin,
  dealOpeningHands,
  completePregameSetup,
  autoCompleteOpponentSetup,
  tickSetupCountdown,
} from "../../frontend/src/game/engine.ts";
import { createSeededRng, randomFloat, withRng } from "../../frontend/src/game/engine/core/random";
import { advanceModeledTurnStep, getForcedAttackCoinResults } from "../../backend/src/sim/evaluateModelVsHeuristic";
import { stateFingerprint } from "../../backend/src/sim/stateFingerprint";
import { enumerateLegalAiActions } from "../../frontend/src/game/engine/ai-policy/actions";
import { chooseAiSetupSelection } from "../../frontend/src/app/gameUiHelpers";
import type { GameState, SideId } from "../../shared/src/types";

// Inlined setupAiVsAiGame (private in backend/src/sim/headlessAiVsAi.ts).
function setupAiVsAiGame(): GameState {
  let state = createGame(undefined, undefined, "Opponent", "hard", false, "Player AI");
  state.humanBySide.player = false;
  state.humanBySide.opponent = false;
  state = chooseOpeningCoin(state, randomFloat() >= 0.5 ? "heads" : "tails");
  state = dealOpeningHands(state);
  const setup = chooseAiSetupSelection(state);
  if (!setup) throw new Error("Unable to choose AI setup for player.");
  state = completePregameSetup(state, setup.activeIndex, setup.benchIndexes);
  state = autoCompleteOpponentSetup(state);
  for (let tick = 0; tick < 5 && state.phase === "setup"; tick += 1) {
    state = tickSetupCountdown(state);
  }
  return state;
}

// Replica of `stateHash` used inside mcts.ts.
function stateHash(s: GameState): string {
  return stateFingerprint(s);
}

function rolloutHeuristicInstrumented(
  state: GameState,
  rng: { next: () => number; fork: (label: string) => any },
  maxSteps: number,
  outerCounter: { count: number },
): GameState {
  let next = cloneGame(state);
  let before: string | null = null;
  for (let step = 0; step < maxSteps; step += 1) {
    if (next.gameOver) break;
    if (before === null) before = stateHash(next);
    const sideId: SideId = next.currentSide === "player" ? "player" : "opponent";
    const preStep = next.opponentTurnStep;
    const drawsBefore = outerCounter.count;
    const forcedCoins = getForcedAttackCoinResults(next, rng as any);
    next = sideId === "player"
      ? advancePlayerAiTurnStep(next, forcedCoins, rng.next)
      : advanceOpponentTurnStep(next, forcedCoins, rng.next);
    const drawsAfter = outerCounter.count;
    const delta = drawsAfter - drawsBefore;
    process.stderr.write(
      `    [rollout 1 advance ${step}] side=${sideId} step=${preStep} draws+=${delta} turn=${next.turnNumber}\n`,
    );
    if (stateHash(next) === before) break;
    before = stateHash(next);
  }
  return next;
}

async function main() {
  const seedRaw = process.argv[2] ?? "0";
  const seed = Number(seedRaw);

  // Instrument outer rng to count draws.
  const counter = { count: 0 };
  const baseRng = createSeededRng(`${seed}:selfplay`, "selfplay");
  const outerRng = {
    label: baseRng.label,
    next: () => {
      counter.count += 1;
      return baseRng.next();
    },
    fork: (label: string) => baseRng.fork(label),
  };

  await withRng(outerRng as any, async () => {
    let state: GameState = setupAiVsAiGame();
    const drawsAfterSetup = counter.count;
    process.stderr.write(`setup complete; outer draws=${drawsAfterSetup} (TS=59 expected)\n`);

    // Apply step 0 action (opponent pass — legalCount=1, single-action shortcut).
    const legal0 = enumerateLegalAiActions(state, "opponent");
    if (legal0.length === 1) {
      const action = legal0[0];
      if (action) {
        const forced = getForcedAttackCoinResults(state, outerRng as any);
        const next = advanceModeledTurnStep(state, "opponent", action, forced, outerRng as any);
        // recorder's stalled-fallback path
        state = stateHash(next) === stateHash(state)
          ? advanceOpponentTurnStep(state, forced, outerRng.next)
          : next;
      }
    }
    process.stderr.write(`after step 0; outer draws=${counter.count} (cumulative)\n`);

    // Now state is at step 1's pre-state. Run ONE rollout from here.
    // MCTS seed for step 1 (opponent) — first rollout uses inner-rng fork.
    const mctsSeed = `${seed}:opponent:1:mcts`;
    const innerRng = createSeededRng(mctsSeed, "mcts-root");
    const sampleRng = innerRng.fork("rollout-crn-0");

    process.stderr.write(`\n--- First rollout of step 1 (200 max steps) ---\n`);
    const drawsBeforeRollout = counter.count;
    rolloutHeuristicInstrumented(state, sampleRng as any, 200, counter);
    process.stderr.write(`\nRollout complete; outer draws this rollout=${counter.count - drawsBeforeRollout}\n`);
  });
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
