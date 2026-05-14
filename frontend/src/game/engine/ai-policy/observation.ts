import { MAX_BENCH } from "../../../../../shared/src/gameData";
import { cardVocabIndex } from "../../../../../shared/src/cardVocab";
import type { GameState, SideId, SideState, UmamusumeInstance } from "../../../../../shared/src/types";
import { getAiPhase } from "./phase";
import type { PublicObservation, PublicSideObservation, PublicUmaObservation, ZoneKey } from "./types";

export function buildPublicObservation(state: GameState, sideId: SideId): PublicObservation {
  const opponentId: SideId = sideId === "player" ? "opponent" : "player";
  const own = state.sides[sideId];
  const opp = state.sides[opponentId];
  return {
    schemaVersion: 2,
    sideToAct: sideId,
    phase: getAiPhase(state, sideId),
    turnNumber: state.turnNumber,
    firstPlayer: state.firstPlayer,
    pendingChoiceKind: state.pendingPlayerChoice?.kind ?? null,
    own: toPublicSideObservation(own, true),
    opponent: toPublicSideObservation(opp, false),
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
