import type { GameState, SideState, UmamusumeInstance } from "../../../shared/src/types";

export const STATE_FINGERPRINT_VERSION = 1;

/**
 * Simulator fingerprint used for no-op/stall detection and search memoization.
 * It intentionally includes private hand/deck/discard contents because this is
 * not a model observation; it is an internal full-state mutation detector.
 */
export function stateFingerprint(state: GameState): string {
  return JSON.stringify({
    version: STATE_FINGERPRINT_VERSION,
    phase: state.phase,
    setup: state.setup,
    pending: state.pendingPlayerChoice,
    currentSide: state.currentSide,
    step: state.opponentTurnStep,
    stadium: state.stadium,
    turnNumber: state.turnNumber,
    firstPlayer: state.firstPlayer,
    turnsTakenBySide: state.turnsTakenBySide,
    gameOver: state.gameOver,
    winner: state.winner,
    points: { player: state.sides.player.points, opponent: state.sides.opponent.points },
    sides: {
      player: compactSide(state.sides.player),
      opponent: compactSide(state.sides.opponent),
    },
  });
}

function compactSide(side: SideState) {
  return {
    id: side.id,
    deck: [...side.deck],
    discard: [...side.discard],
    hand: [...side.hand],
    energyZone: [...side.energyZone],
    energyAttachmentsThisTurn: side.energyAttachmentsThisTurn,
    bonusEnergyAttachments: side.bonusEnergyAttachments,
    retreatCostReduction: side.retreatCostReduction,
    activeAttackDamageBonus: side.activeAttackDamageBonus,
    usedSupporterThisTurn: side.usedSupporterThisTurn,
    usedRetreatThisTurn: side.usedRetreatThisTurn,
    usedStadiumThisTurn: side.usedStadiumThisTurn,
    usedAbilityNamesThisTurn: [...side.usedAbilityNamesThisTurn],
    usedAbilityNamesThisGame: [...side.usedAbilityNamesThisGame],
    guaranteedCoinFlipHeads: side.guaranteedCoinFlipHeads,
    active: side.active ? compactUmamusume(side.active) : null,
    bench: side.bench.map(compactUmamusume),
  };
}

function compactUmamusume(umamusume: UmamusumeInstance) {
  return {
    uid: umamusume.uid,
    cardId: umamusume.cardId,
    evolutionCardIds: [...umamusume.evolutionCardIds],
    species: umamusume.species,
    stage: umamusume.stage,
    hp: umamusume.hp,
    maxHp: umamusume.maxHp,
    energies: { ...umamusume.energies },
    specialConditions: [...umamusume.specialConditions],
    enteredTurn: umamusume.enteredTurn,
    evolvedTurn: umamusume.evolvedTurn,
    tookDamageLastTurn: umamusume.tookDamageLastTurn,
    tookDamageThisTurn: umamusume.tookDamageThisTurn,
    nextTurnDamageReduction: umamusume.nextTurnDamageReduction,
    usedAbilityThisTurn: umamusume.usedAbilityThisTurn,
    attackBlockedUntilOwnTurn: umamusume.attackBlockedUntilOwnTurn,
    paralysedUntilOwnTurn: umamusume.paralysedUntilOwnTurn,
    toolCardId: umamusume.toolCardId,
  };
}
