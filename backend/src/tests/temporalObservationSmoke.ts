// R16-P1 temporal / turn-state observation smoke.
//
// Asserts `buildPublicObservation` (schemaVersion 3) surfaces the temporal
// block + per-side/per-Uma turnState exactly, that derived fields key off
// the engine predicates, and that the hidden-info contract still holds
// (opponent hand ids absent). Mirrors the scope-doc § "Tests And Smokes"
// list: extra energy attach budget, active attack damage bonus,
// evolved-last-turn, blocked-attack, poison/paralysis recovery.

import assert from "node:assert/strict";
import { cloneGame } from "../../../frontend/src/game/engine/core/stateClone";
import { setupAiVsAiGame } from "../sim/evaluateModelVsHeuristic";
import { buildPublicObservation } from "../../../frontend/src/game/engine/ai-policy/observation";
import type { GameState } from "../../../shared/src/types";

function ownActive(state: GameState) {
  const a = state.sides.player.active;
  assert.ok(a, "expected player active");
  return a;
}

const base = setupAiVsAiGame();

// --- schema + structural shape -------------------------------------------
const obs0 = buildPublicObservation(base, "player");
assert.equal(obs0.schemaVersion, 3, "schemaVersion must be 3");
assert.ok(obs0.temporal, "temporal block present");
assert.equal(typeof obs0.temporal.ownTurnsTaken, "number");
assert.equal(typeof obs0.temporal.ownIsFirstTurn, "boolean");
assert.ok(obs0.own.turnState, "own side turnState present");
assert.ok(obs0.opponent.turnState, "opp side turnState present");
assert.ok(obs0.own.active && obs0.own.active.turnState, "own active uma turnState present");
// effectiveRetreatCostReduction must be present (omission-1 resolution).
assert.equal(
  typeof obs0.own.turnState.effectiveRetreatCostReduction,
  "number",
  "effectiveRetreatCostReduction must be emitted",
);
// Counts only — never name strings (hidden-info / public-safety).
assert.equal(typeof obs0.own.turnState.usedAbilityNameCountThisTurn, "number");
assert.equal(typeof obs0.own.turnState.usedAbilityNameCountThisGame, "number");
assert.ok(
  !("usedAbilityNamesThisTurn" in (obs0.own.turnState as unknown as object)),
  "ability NAME strings must not appear in turnState",
);

// --- hidden-info regression ----------------------------------------------
assert.equal(
  obs0.opponent.handCardIds,
  undefined,
  "opponent hand IDs must not be exposed (hidden-info contract)",
);
assert.equal(typeof obs0.opponent.handCount, "number", "opponent hand COUNT is exposed");
assert.equal(
  obs0.cardIdsByZone.oppActive.length >= 0,
  true,
  "opp active card-id zone present (board, public)",
);

// --- temporal cases ------------------------------------------------------

// extra energy attach budget (side turnState).
{
  const s = cloneGame(base);
  s.sides.player.energyAttachmentsThisTurn = 1;
  s.sides.player.bonusEnergyAttachments = 2;
  const o = buildPublicObservation(s, "player");
  assert.equal(o.own.turnState.energyAttachmentsThisTurn, 1, "energyAttachmentsThisTurn");
  assert.equal(o.own.turnState.bonusEnergyAttachments, 2, "bonusEnergyAttachments");
}

// active attack damage bonus (side-level, per scope).
{
  const s = cloneGame(base);
  s.sides.player.activeAttackDamageBonus = 30;
  const o = buildPublicObservation(s, "player");
  assert.equal(o.own.turnState.activeAttackDamageBonus, 30, "activeAttackDamageBonus side-level");
}

// effective retreat cost reduction folds in the stadium global term.
{
  const s = cloneGame(base);
  s.sides.player.retreatCostReduction = 1;
  // nakayamaTurf is a trainer stadium with globalRetreatCostReduction: 1
  // (shared/src/data/cards.json), so effective = raw(1) + global(1) = 2.
  s.stadium = { cardId: "nakayamaTurf", owner: "player" };
  const o = buildPublicObservation(s, "player");
  assert.equal(o.own.turnState.retreatCostReduction, 1, "raw retreatCostReduction unchanged");
  assert.equal(
    o.own.turnState.effectiveRetreatCostReduction,
    2,
    "effective = raw(1) + stadium global(1) = 2",
  );
  // No-stadium baseline: effective collapses to the raw per-side value.
  const sNo = cloneGame(base);
  sNo.sides.player.retreatCostReduction = 1;
  sNo.stadium = null;
  const oNo = buildPublicObservation(sNo, "player");
  assert.equal(
    oNo.own.turnState.effectiveRetreatCostReduction,
    1,
    "effective == raw when no stadium",
  );
}

// evolved-last-turn / evolved-this-turn / entered-this-turn.
{
  const s = cloneGame(base);
  const a = ownActive(s);
  s.turnNumber = 5;
  a.enteredTurn = 2;
  a.evolvedTurn = 4;
  const o = buildPublicObservation(s, "player");
  const ts = o.own.active!.turnState;
  assert.equal(ts.enteredThisTurn, false, "enteredThisTurn false (entered turn 2, now 5)");
  assert.equal(ts.evolvedThisTurn, false, "evolvedThisTurn false (evolved turn 4, now 5)");
  assert.equal(ts.evolvedLastTurn, true, "evolvedLastTurn true (evolved turn 4, now 5)");
  assert.equal(ts.turnsInPlay, 3, "turnsInPlay = turnNumber - enteredTurn");

  s.turnNumber = 4;
  const o2 = buildPublicObservation(s, "player");
  assert.equal(o2.own.active!.turnState.evolvedThisTurn, true, "evolvedThisTurn true (evolved == turn)");
}

// blocked-attack state (eligibility.ts:29 predicate:
// attackBlockedUntilOwnTurn === turnsTakenBySide[side]).
{
  const s = cloneGame(base);
  const a = ownActive(s);
  s.turnsTakenBySide.player = 3;
  a.attackBlockedUntilOwnTurn = 3;
  const o = buildPublicObservation(s, "player");
  assert.equal(o.own.active!.turnState.attackBlockedThisTurn, true, "attackBlockedThisTurn true");
  a.attackBlockedUntilOwnTurn = 4;
  const o2 = buildPublicObservation(s, "player");
  assert.equal(o2.own.active!.turnState.attackBlockedThisTurn, false, "attackBlockedThisTurn false (different turn)");
}

// poison / paralysis recovery pending (turn.ts:130-136: recovery fires when
// turnsTaken >= paralysedUntilOwnTurn; pending while strictly less).
{
  const s = cloneGame(base);
  const a = ownActive(s);
  s.turnsTakenBySide.player = 2;
  a.specialConditions.push("paralysed");
  a.paralysedUntilOwnTurn = 4;
  const o = buildPublicObservation(s, "player");
  assert.equal(o.own.active!.turnState.paralysisRecoveryPending, true, "paralysisRecoveryPending true (2 < 4)");
  s.turnsTakenBySide.player = 4;
  const o2 = buildPublicObservation(s, "player");
  assert.equal(
    o2.own.active!.turnState.paralysisRecoveryPending,
    false,
    "paralysisRecoveryPending false (4 >= 4, recovery would fire)",
  );

  // poison damage memory surfaces via tookDamageThisTurn.
  const s2 = cloneGame(base);
  const a2 = ownActive(s2);
  a2.tookDamageThisTurn = true;
  a2.tookDamageLastTurn = false;
  const o3 = buildPublicObservation(s2, "player");
  assert.equal(o3.own.active!.turnState.tookDamageThisTurn, true, "tookDamageThisTurn");
  assert.equal(o3.own.active!.turnState.tookDamageLastTurn, false, "tookDamageLastTurn");
}

// first-turn flag is the energy/setup-phase predicate (turnsTakenBySide===0).
{
  const s = cloneGame(base);
  s.turnsTakenBySide.player = 0;
  s.turnsTakenBySide.opponent = 1;
  const o = buildPublicObservation(s, "player");
  assert.equal(o.temporal.ownIsFirstTurn, true, "ownIsFirstTurn true at turnsTaken 0");
  assert.equal(o.temporal.opponentIsFirstTurn, false, "opponentIsFirstTurn false at turnsTaken 1");
  s.turnsTakenBySide.player = 1;
  const o2 = buildPublicObservation(s, "player");
  assert.equal(o2.temporal.ownIsFirstTurn, false, "ownIsFirstTurn false at turnsTaken 1 (not <=1)");
}

console.log(JSON.stringify({ status: "PASS", schemaVersion: 3 }, null, 2));
