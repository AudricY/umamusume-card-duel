import assert from "node:assert/strict";
import { cloneGame } from "../../../frontend/src/game/engine/core/stateClone";
import { setupAiVsAiGame } from "../sim/evaluateModelVsHeuristic";
import { stateFingerprint } from "../sim/stateFingerprint";
import type { GameState } from "../../../shared/src/types";

const cases: Array<[string, (state: GameState) => void]> = [
  ["active hp", (state) => { ownActive(state).hp = Math.max(0, ownActive(state).hp - 10); }],
  ["active energy", (state) => { ownActive(state).energies.fire += 1; }],
  ["special condition", (state) => { ownActive(state).specialConditions.push("poisoned"); }],
  ["tool", (state) => { ownActive(state).toolCardId = "training-helmet"; }],
  ["ability usage", (state) => { ownActive(state).usedAbilityThisTurn = true; }],
  ["phase", (state) => { state.phase = "setup"; }],
  ["pending choice", (state) => { state.pendingPlayerChoice = { kind: "promoteAfterKnockout", sideId: "player", resume: "none" }; }],
  ["points", (state) => { state.sides.player.points += 1; }],
  ["turn flags", (state) => { state.sides.player.usedSupporterThisTurn = !state.sides.player.usedSupporterThisTurn; }],
  ["energy zone", (state) => { state.sides.player.energyZone.push("fire"); }],
  ["stadium", (state) => { state.stadium = { cardId: "tracen-gym", owner: "player" }; }],
  ["board movement", moveActiveToBench],
  ["same-count hand content", (state) => { replaceFirst(state.sides.player.hand, "fingerprint-hand-card"); }],
  ["same-count deck content", (state) => { replaceFirst(state.sides.player.deck, "fingerprint-deck-card"); }],
  ["same-count discard content", (state) => {
    if (state.sides.player.discard.length === 0) state.sides.player.discard.push("discard-before");
    replaceFirst(state.sides.player.discard, "fingerprint-discard-card");
  }],
];

for (const [label, mutate] of cases) {
  const before = setupAiVsAiGame();
  const after = cloneGame(before);
  mutate(after);
  assert.notEqual(
    stateFingerprint(after),
    stateFingerprint(before),
    `fingerprint should change for ${label}`,
  );
}

const repeated = setupAiVsAiGame();
assert.equal(
  stateFingerprint(repeated),
  stateFingerprint(cloneGame(repeated)),
  "fingerprint should be stable across clone-only copies",
);

console.log(JSON.stringify({ status: "PASS", cases: cases.length }, null, 2));

function ownActive(state: GameState) {
  const active = state.sides.player.active;
  assert.ok(active, "expected player active");
  return active;
}

function moveActiveToBench(state: GameState): void {
  const active = ownActive(state);
  if (state.sides.player.bench.length === 0) {
    state.sides.player.bench.push({ ...active, uid: active.uid + 1000, cardId: `${active.cardId}-bench` });
  }
  const [bench] = state.sides.player.bench.splice(0, 1, active);
  state.sides.player.active = bench ?? active;
}

function replaceFirst(cards: string[], replacement: string): void {
  if (cards.length === 0) cards.push("fingerprint-before-card");
  cards[0] = replacement;
}
