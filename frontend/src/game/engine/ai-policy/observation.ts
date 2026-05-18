import { MAX_BENCH } from "../../../../../shared/src/gameData";
import { cardVocabIndex } from "../../../../../shared/src/cardVocab";
import type { GameState, SideId, SideState, UmamusumeInstance } from "../../../../../shared/src/types";
import { getGlobalRetreatCostReduction } from "../flow/retreat";
import { getAiPhase } from "./phase";
import type {
  PublicObservation,
  PublicSideObservation,
  PublicSideTurnState,
  PublicTemporalObservation,
  PublicUmaObservation,
  PublicUmaTurnState,
  ZoneKey,
} from "./types";

export function buildPublicObservation(state: GameState, sideId: SideId): PublicObservation {
  const opponentId: SideId = sideId === "player" ? "opponent" : "player";
  const own = state.sides[sideId];
  const opp = state.sides[opponentId];
  const ownTurnsTaken = state.turnsTakenBySide[sideId] ?? 0;
  const oppTurnsTaken = state.turnsTakenBySide[opponentId] ?? 0;
  // R16-P1: `ownIsFirstTurn` is the ENERGY/SETUP-PHASE flag only — the
  // exact `startTurn` energy-skip predicate `turnsTaken === 0`
  // (frontend/src/game/engine/flow/turn.ts:55-72). Evolution's `<= 1`
  // first-turn threshold (evolution.ts:42-44) is intentionally NOT
  // re-encoded: evolve-legality is already covered by the per-Uma
  // enteredThisTurn/evolvedThisTurn sickness booleans.
  const temporal: PublicTemporalObservation = {
    ownTurnsTaken,
    opponentTurnsTaken: oppTurnsTaken,
    ownIsFirstTurn: ownTurnsTaken === 0,
    opponentIsFirstTurn: oppTurnsTaken === 0,
  };
  return {
    schemaVersion: 3,
    sideToAct: sideId,
    phase: getAiPhase(state, sideId),
    turnNumber: state.turnNumber,
    firstPlayer: state.firstPlayer,
    pendingChoiceKind: state.pendingPlayerChoice?.kind ?? null,
    temporal,
    own: toPublicSideObservation(state, own, true),
    opponent: toPublicSideObservation(state, opp, false),
    shared: {
      stadiumCardId: state.stadium?.cardId ?? null,
      currentSide: state.currentSide,
      gameOver: state.gameOver,
    },
    cardIdsByZone: buildCardIdsByZone(own, opp, state.stadium?.cardId ?? null),
  };
}

function buildCardIdsByZone(own: SideState, opp: SideState, stadiumCardId: string | null): Record<ZoneKey, number[]> {
  const benchIdxs = (side: SideState): number[] => side.bench.map((uma) => cardVocabIndex(uma.cardId));
  // Opponent hand is hidden — emit an empty array. The Phase 2 Python collator
  // pads to the fixed hand-cap shape and the embedding's padding_idx=0 keeps
  // hidden info out of the gradient.
  return {
    ownActive: own.active ? [cardVocabIndex(own.active.cardId)] : [],
    oppActive: opp.active ? [cardVocabIndex(opp.active.cardId)] : [],
    ownBench: benchIdxs(own),
    oppBench: benchIdxs(opp),
    ownHand: own.hand.map((cardId) => cardVocabIndex(cardId)),
    ownDiscard: own.discard.map((cardId) => cardVocabIndex(cardId)),
    oppDiscard: opp.discard.map((cardId) => cardVocabIndex(cardId)),
    stadium: stadiumCardId ? [cardVocabIndex(stadiumCardId)] : [],
  };
}

// R16-P1: side-level temporal scalars. `effectiveRetreatCostReduction` folds
// in the stadium global term (getGlobalRetreatCostReduction, retreat.ts:13)
// so the encoded retreat-legality signal matches `effectiveRetreatCost`
// (retreat.ts:20-26). Ability NAME strings are public-info-sensitive and
// are never emitted — only the lengths (counts).
function toPublicSideTurnState(state: GameState, side: SideState): PublicSideTurnState {
  const globalRetreat = getGlobalRetreatCostReduction(state);
  return {
    energyAttachmentsThisTurn: side.energyAttachmentsThisTurn,
    bonusEnergyAttachments: side.bonusEnergyAttachments,
    retreatCostReduction: side.retreatCostReduction,
    effectiveRetreatCostReduction: side.retreatCostReduction + globalRetreat,
    activeAttackDamageBonus: side.activeAttackDamageBonus,
    usedAbilityNameCountThisTurn: side.usedAbilityNamesThisTurn.length,
    usedAbilityNameCountThisGame: side.usedAbilityNamesThisGame.length,
    guaranteedCoinFlipHeads: side.guaranteedCoinFlipHeads,
  };
}

function toPublicSideObservation(
  state: GameState,
  side: SideState,
  includePrivateHand: boolean,
): PublicSideObservation {
  const sideTurnsTaken = state.turnsTakenBySide[side.id] ?? 0;
  const bench: Array<PublicUmaObservation | null> = side.bench
    .slice(0, MAX_BENCH)
    .map((uma) => toPublicUmaObservation(state, uma, sideTurnsTaken));
  while (bench.length < MAX_BENCH) bench.push(null);
  const observation: PublicSideObservation = {
    id: side.id,
    points: side.points,
    handCount: side.hand.length,
    deckCount: side.deck.length,
    discard: [...side.discard],
    active: side.active ? toPublicUmaObservation(state, side.active, sideTurnsTaken) : null,
    bench,
    energyZone: [...side.energyZone],
    usedSupporterThisTurn: side.usedSupporterThisTurn,
    usedRetreatThisTurn: side.usedRetreatThisTurn,
    usedStadiumThisTurn: side.usedStadiumThisTurn,
    turnState: toPublicSideTurnState(state, side),
  };
  if (includePrivateHand) observation.handCardIds = [...side.hand];
  return observation;
}

// R16-P1: per-Uma temporal sickness/memory booleans, derived at build time
// from the engine's turn stamps so the model never sees a raw absolute turn
// number (overfit guard). Predicates mirror the engine exactly:
//   - enteredThisTurn:  enteredTurn === turnNumber  (evolution.ts:21)
//   - evolvedThisTurn:  evolvedTurn === turnNumber  (evolution.ts:22)
//   - evolvedLastTurn:  evolvedTurn === turnNumber - 1
//   - attackBlockedThisTurn:
//       attackBlockedUntilOwnTurn === turnsTakenBySide[side] (eligibility.ts:29)
//   - paralysisRecoveryPending: still paralysed AND recovery has not yet
//       fired, i.e. turnsTaken < paralysedUntilOwnTurn (turn.ts:130-136)
function toPublicUmaTurnState(
  state: GameState,
  umamusume: UmamusumeInstance,
  sideTurnsTaken: number,
): PublicUmaTurnState {
  const turnNumber = state.turnNumber;
  const evolvedTurn = umamusume.evolvedTurn;
  const paralysed = umamusume.specialConditions.includes("paralysed");
  const paralysisRecoveryPending =
    paralysed
    && umamusume.paralysedUntilOwnTurn !== null
    && sideTurnsTaken < umamusume.paralysedUntilOwnTurn;
  return {
    turnsInPlay: Math.max(0, turnNumber - umamusume.enteredTurn),
    enteredThisTurn: umamusume.enteredTurn === turnNumber,
    evolvedThisTurn: evolvedTurn !== null && evolvedTurn === turnNumber,
    evolvedLastTurn: evolvedTurn !== null && evolvedTurn === turnNumber - 1,
    tookDamageLastTurn: umamusume.tookDamageLastTurn,
    tookDamageThisTurn: umamusume.tookDamageThisTurn,
    nextTurnDamageReduction: umamusume.nextTurnDamageReduction,
    attackBlockedThisTurn:
      umamusume.attackBlockedUntilOwnTurn !== null
      && umamusume.attackBlockedUntilOwnTurn === sideTurnsTaken,
    paralysisRecoveryPending,
  };
}

function toPublicUmaObservation(
  state: GameState,
  umamusume: UmamusumeInstance,
  sideTurnsTaken: number,
): PublicUmaObservation {
  const energyTotal = Object.values(umamusume.energies).reduce<number>((sum, count) => sum + count, 0);
  return {
    uid: umamusume.uid,
    cardId: umamusume.cardId,
    species: umamusume.species,
    stage: umamusume.stage,
    hp: umamusume.hp,
    maxHp: umamusume.maxHp,
    energyTotal,
    energies: { ...umamusume.energies },
    specialConditions: [...umamusume.specialConditions],
    toolCardId: umamusume.toolCardId,
    usedAbilityThisTurn: umamusume.usedAbilityThisTurn,
    turnState: toPublicUmaTurnState(state, umamusume, sideTurnsTaken),
  };
}
