import { MAX_BENCH } from "../../../../../shared/src/gameData";
import type { GameState, SideId, SideState, UmamusumeInstance } from "../../../../../shared/src/types";
import { getAiPhase } from "./phase";
import type { PublicObservation, PublicSideObservation, PublicUmaObservation } from "./types";

export function buildPublicObservation(state: GameState, sideId: SideId): PublicObservation {
  const opponentId: SideId = sideId === "player" ? "opponent" : "player";
  return {
    schemaVersion: 1,
    sideToAct: sideId,
    phase: getAiPhase(state, sideId),
    turnNumber: state.turnNumber,
    firstPlayer: state.firstPlayer,
    pendingChoiceKind: state.pendingPlayerChoice?.kind ?? null,
    own: toPublicSideObservation(state.sides[sideId], true),
    opponent: toPublicSideObservation(state.sides[opponentId], false),
    shared: {
      stadiumCardId: state.stadium?.cardId ?? null,
      currentSide: state.currentSide,
      gameOver: state.gameOver,
    },
  };
}

function toPublicSideObservation(side: SideState, includePrivateHand: boolean): PublicSideObservation {
  const bench: Array<PublicUmaObservation | null> = side.bench.slice(0, MAX_BENCH).map(toPublicUmaObservation);
  while (bench.length < MAX_BENCH) bench.push(null);
  const observation: PublicSideObservation = {
    id: side.id,
    points: side.points,
    handCount: side.hand.length,
    deckCount: side.deck.length,
    discard: [...side.discard],
    active: side.active ? toPublicUmaObservation(side.active) : null,
    bench,
    energyZone: [...side.energyZone],
    usedSupporterThisTurn: side.usedSupporterThisTurn,
    usedRetreatThisTurn: side.usedRetreatThisTurn,
    usedStadiumThisTurn: side.usedStadiumThisTurn,
  };
  if (includePrivateHand) observation.handCardIds = [...side.hand];
  return observation;
}

function toPublicUmaObservation(umamusume: UmamusumeInstance): PublicUmaObservation {
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
  };
}
