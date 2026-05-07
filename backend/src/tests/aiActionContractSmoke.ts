import assert from "node:assert/strict";
import {
  advanceOpponentTurnStep,
  advancePlayerAiTurnStep,
} from "../../../frontend/src/game/engine";
import { enumerateLegalAiActions } from "../../../frontend/src/game/engine/ai-policy/actions";
import { createSeededRng, withRng } from "../../../frontend/src/game/engine/core/random";
import type { SideId } from "../../../shared/src/types";
import {
  advanceModeledTurnStep,
  getForcedAttackCoinResults,
  setupAiVsAiGame,
  stateHash,
} from "../sim/evaluateModelVsHeuristic";

type Noop = {
  seed: string;
  step: number;
  sideId: SideId;
  actionId: string;
  kind: string;
};

const noops: Noop[] = [];
let checked = 0;

for (let game = 0; game < 24; game += 1) {
  const seed = String(41000 + game);
  const rng = createSeededRng(seed, "action-contract");
  withRng(rng, () => {
    let state = setupAiVsAiGame();
    for (let step = 0; step < 120 && !state.gameOver; step += 1) {
      const sideId = state.currentSide === "player" || state.currentSide === "opponent" ? state.currentSide : "player";
      const beforeHash = stateHash(state);
      const forcedCoinResults = getForcedAttackCoinResults(state, rng);
      const actions = enumerateLegalAiActions(state, sideId);
      actions.forEach((action, actionIndex) => {
        const actionRng = createSeededRng(`${seed}:${step}:${actionIndex}`, "action-contract-action");
        const result = advanceModeledTurnStep(state, sideId, action, forcedCoinResults, actionRng);
        checked += 1;
        if (stateHash(result) === beforeHash) {
          noops.push({ seed, step, sideId, actionId: action.id, kind: action.kind });
        }
      });
      state = sideId === "player"
        ? advancePlayerAiTurnStep(state, forcedCoinResults, rng.next)
        : advanceOpponentTurnStep(state, forcedCoinResults, rng.next);
      if (stateHash(state) === beforeHash) break;
    }
  });
}

assert.equal(noops.length, 0, `Found non-mutating legal actions: ${JSON.stringify(noops.slice(0, 20), null, 2)}`);

console.log(JSON.stringify({
  status: "PASS",
  checked,
}, null, 2));
