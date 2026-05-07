import { MAX_BENCH } from "../../../../../shared/src/gameData";
import type { GameState, SideId, SideState, UmamusumeInstance } from "../../../../../shared/src/types";
import { getCard, getPrimaryAttack, getUmamusumeCard } from "../core/catalog";
import { attachedEnergyCount, getAllUmamusume } from "../core/umamusume";
import { choosePreferredActiveIndex, refreshContinuousHp } from "../flow/board";
import { canAttachEnergy, canAttachEnergyToUmamusume, canAttack, canRetreat, canUseUmamusumeAbility } from "../flow/eligibility";
import { findEvolutionTarget } from "../flow/evolution";
import { getPlayableAction, getToolTargets } from "../flow/playRules";
import { canUseStadium } from "../flow/trainers";
import { buildCombatCandidates } from "../flow/ai/combatPlanner";
import { getAiPhase } from "./phase";
import type { AiPhase, LegalAiAction } from "./types";

const FEATURE_COUNT = 18;

export function enumerateLegalAiActions(state: GameState, sideId: SideId): LegalAiAction[] {
  const side = state.sides[sideId];
  const phase = getAiPhase(state, sideId);
  switch (phase) {
    case "setup":
      return enumerateSetupActions(state, sideId);
    case "pendingChoice":
      return enumeratePendingChoiceActions(state, sideId);
    case "bench":
      return withPass(phase, enumerateBenchActions(state, side));
    case "trainerBefore":
    case "trainerAfter":
      return withPass(phase, enumerateTrainerActions(state, side, phase));
    case "evolve":
      return withPass(phase, enumerateEvolutionActions(state, side));
    case "attach":
      return withPass(phase, enumerateAttachActions(state, side));
    case "ability":
      return withPass(phase, enumerateAbilityActions(state, side));
    case "combat":
      return enumerateCombatActions(state, side);
    case "stadiumOrEnd":
      return enumerateStadiumOrEndActions(state, side);
    default:
      return [passAction(phase)];
  }
}

export function chooseHighestScoredAction(actions: LegalAiAction[]): LegalAiAction {
  return [...actions].sort((left, right) => (right.features[0] ?? 0) - (left.features[0] ?? 0))[0] ?? passAction("stadiumOrEnd");
}

function enumerateSetupActions(state: GameState, sideId: SideId): LegalAiAction[] {
  const setup = state.setup;
  const side = state.sides[sideId];
  if (!setup?.openingHandsDealt && side.hand.length === 0) return [passAction("setup")];
  const basics = side.hand
    .map((cardId, handIndex) => ({ cardId, handIndex }))
    .flatMap(({ cardId, handIndex }) => {
      const card = getCard(cardId);
      if (card.kind !== "umamusume" || card.stage !== 0) return [];
      const attack = getPrimaryAttack(card);
      return [{ cardId, handIndex, score: card.hp + attack.damage * 1.8 }];
    })
    .sort((left, right) => right.score - left.score);
  const active = basics[0];
  if (!active) return [passAction("setup")];
  const benchHandIndexes = basics.slice(1, MAX_BENCH + 1).map((entry) => entry.handIndex);
  return [{
    id: `setup:${sideId}:active:${active.handIndex}:bench:${benchHandIndexes.join(",")}`,
    phase: "setup",
    kind: "setupChooseBoard",
    payload: { activeHandIndex: active.handIndex, benchHandIndexes },
    features: features({
      score: active.score + benchHandIndexes.length * 12,
      phase: "setup",
      kind: "setupChooseBoard",
      sourceCardId: active.cardId,
      amount: benchHandIndexes.length,
    }),
  }];
}

function enumeratePendingChoiceActions(state: GameState, sideId: SideId): LegalAiAction[] {
  const pending = state.pendingPlayerChoice;
  if (!pending || pending.sideId !== sideId) return [passAction("pendingChoice")];
  const side = state.sides[sideId];
  const actions = side.bench.map((target, slot) => ({
    id: `pending:${pending.kind}:${target.uid}`,
    phase: "pendingChoice" as const,
    kind: "resolvePendingChoice",
    payload: { targetUid: target.uid },
    features: features({
      score: scoreUmamusume(target),
      phase: "pendingChoice",
      kind: "resolvePendingChoice",
      target,
      targetSlot: slot,
    }),
  }));
  return actions.length > 0 ? actions : [passAction("pendingChoice")];
}

function enumerateBenchActions(state: GameState, side: SideState): LegalAiAction[] {
  if (side.bench.length >= MAX_BENCH) return [];
  return side.hand.flatMap((cardId, handIndex) => {
    const card = getCard(cardId);
    if (card.kind !== "umamusume" || card.stage !== 0) return [];
    const attack = getPrimaryAttack(card);
    return [{
      id: `bench:${handIndex}:${cardId}`,
      phase: "bench" as const,
      kind: "playBasic",
      payload: { handIndex },
      features: features({
        score: card.hp * 0.5 + attack.damage + 20,
        phase: "bench",
        kind: "playBasic",
        sourceCardId: cardId,
        sourceHandIndex: handIndex,
      }),
    }];
  });
}

function enumerateTrainerActions(state: GameState, side: SideState, phase: Extract<AiPhase, "trainerBefore" | "trainerAfter">): LegalAiAction[] {
  return side.hand.flatMap((cardId, handIndex) => {
    const card = getCard(cardId);
    if (card.kind !== "trainer") return [];
    const play = getPlayableAction(state, side, cardId);
    if (!play.canPlay) return [];
    let score = 24;
    if (card.effect.draw) score += card.effect.draw * 8;
    if (card.effect.searchUmamusume || card.effect.searchEvolutionUmamusume || card.effect.searchRandomBasicUmamusume) score += 34;
    if (card.effect.gustOpponent) score += 30;
    if (card.effect.extraEnergyAttach || card.effect.attachEnergyFromZoneToBench) score += 36;
    if (card.effect.heal) score += 16;
    if (card.trainerType === "tool" && getToolTargets(side).length > 0) score += 20;
    return [{
      id: `${phase}:trainer:${handIndex}:${cardId}`,
      phase,
      kind: "playTrainer",
      payload: { handIndex },
      features: features({ score, phase, kind: "playTrainer", sourceCardId: cardId, sourceHandIndex: handIndex }),
    }];
  });
}

function enumerateEvolutionActions(state: GameState, side: SideState): LegalAiAction[] {
  return side.hand.flatMap((cardId, handIndex) => {
    const card = getCard(cardId);
    if (card.kind !== "umamusume" || card.stage <= 0) return [];
    const target = findEvolutionTarget(state, side, card);
    if (!target) return [];
    const hpGain = Math.max(0, card.hp - target.maxHp);
    return [{
      id: `evolve:${handIndex}:${cardId}:${target.uid}`,
      phase: "evolve" as const,
      kind: "evolve",
      payload: { handIndex, targetUid: target.uid },
      features: features({
        score: 50 + hpGain + card.stage * 18 + attachedEnergyCount(target) * 8,
        phase: "evolve",
        kind: "evolve",
        sourceCardId: cardId,
        sourceHandIndex: handIndex,
        target,
      }),
    }];
  });
}

function enumerateAttachActions(state: GameState, side: SideState): LegalAiAction[] {
  if (!canAttachEnergy(state, side)) return [];
  return getAllUmamusume(side).flatMap((target, slot) => {
    if (!canAttachEnergyToUmamusume(state, side, target)) return [];
    return [{
      id: `attach:${target.uid}`,
      phase: "attach" as const,
      kind: "attachEnergy",
      payload: { targetUid: target.uid },
      features: features({
        score: 30 + scoreUmamusume(target) * 0.1 + (target.uid === side.active?.uid ? 18 : 0),
        phase: "attach",
        kind: "attachEnergy",
        target,
        targetSlot: slot,
        amount: attachedEnergyCount(target),
      }),
    }];
  });
}

function enumerateAbilityActions(state: GameState, side: SideState): LegalAiAction[] {
  return getAllUmamusume(side).flatMap((source, slot) => {
    if (!canUseUmamusumeAbility(state, side, source.uid)) return [];
    const card = getUmamusumeCard(source);
    const ability = card.ability;
    if (!ability) return [];
    let score = 20 + source.stage * 10;
    if (ability.damageOpponent) score += ability.damageOpponent * 2;
    if (ability.moveBenchedEnergyToActive) score += 38;
    if (ability.discardToDraw) score += ability.discardToDraw.draw * 8;
    if (ability.coinFlipDrawOrActiveDamageCounter) score += 18;
    return [{
      id: `ability:${source.uid}:${ability.name}`,
      phase: "ability" as const,
      kind: "useAbility",
      payload: { sourceUid: source.uid, abilityName: ability.name },
      features: features({ score, phase: "ability", kind: "useAbility", sourceCardId: source.cardId, target: source, targetSlot: slot }),
    }];
  });
}

function enumerateCombatActions(state: GameState, side: SideState): LegalAiAction[] {
  const actions: LegalAiAction[] = [];
  if (canAttack(state, side) || canRetreat(state, side)) {
    const candidates = buildCombatCandidates(state, side, {
      refreshContinuousEffects: refreshContinuousHp,
      choosePreferredActiveIndex,
    });
    candidates.forEach((candidate, index) => {
      const targetUid = candidate.decision.kind === "attack" ? candidate.decision.attackTargetUid : undefined;
      const target = targetUid !== undefined
        ? getAllUmamusume(state.sides[side.id === "player" ? "opponent" : "player"]).find((umamusume) => umamusume.uid === targetUid)
        : undefined;
      const featureInput = {
        score: candidate.score + (candidate.lethalTarget ? 100 : 0) + (candidate.keepsSafe ? 20 : 0),
        phase: "combat" as const,
        kind: candidate.decision.kind,
        amount: candidate.targetValue,
        endsTurn: candidate.decision.kind === "attack",
      };
      actions.push({
        id: `combat:${candidate.id}:${index}`,
        phase: "combat",
        kind: candidate.decision.kind === "attack" && candidate.decision.retreatTargetUid !== undefined ? "retreatAttack" : candidate.decision.kind,
        payload: { decision: candidate.decision },
        features: features(target ? { ...featureInput, target } : featureInput),
      });
    });
  }
  return actions.length > 0 ? actions : [passAction("combat")];
}

function enumerateStadiumOrEndActions(state: GameState, side: SideState): LegalAiAction[] {
  const actions: LegalAiAction[] = [];
  if (canUseStadium(state, side)) {
    actions.push({
      id: "stadium:use",
      phase: "stadiumOrEnd",
      kind: "useStadium",
      payload: {},
      features: features({ score: 35, phase: "stadiumOrEnd", kind: "useStadium", endsTurn: true }),
    });
  }
  actions.push({
    id: "turn:end",
    phase: "stadiumOrEnd",
    kind: "endTurn",
    payload: {},
    features: features({ score: 1, phase: "stadiumOrEnd", kind: "endTurn", endsTurn: true }),
  });
  return actions;
}

function withPass(phase: AiPhase, actions: LegalAiAction[]): LegalAiAction[] {
  return actions.length > 0 ? [...actions, passAction(phase)] : [passAction(phase)];
}

function passAction(phase: AiPhase): LegalAiAction {
  return {
    id: `${phase}:pass`,
    phase,
    kind: "pass",
    payload: {},
    features: features({ score: 0, phase, kind: "pass" }),
  };
}

function features(input: {
  score: number;
  phase: AiPhase;
  kind: string;
  sourceCardId?: string;
  sourceHandIndex?: number;
  target?: UmamusumeInstance;
  targetSlot?: number;
  amount?: number;
  endsTurn?: boolean;
}): number[] {
  const vector = Array.from({ length: FEATURE_COUNT }, () => 0);
  vector[0] = input.score / 100;
  vector[1] = phaseIndex(input.phase) / 10;
  vector[2] = kindIndex(input.kind) / 16;
  vector[3] = input.sourceCardId ? hashToUnit(input.sourceCardId) : 0;
  vector[4] = input.sourceHandIndex === undefined ? -1 : input.sourceHandIndex / 10;
  vector[5] = input.target ? hashToUnit(input.target.cardId) : 0;
  vector[6] = input.target?.stage ?? 0;
  vector[7] = input.target ? input.target.hp / Math.max(1, input.target.maxHp) : 0;
  vector[8] = input.target ? attachedEnergyCount(input.target) / 6 : 0;
  vector[9] = input.targetSlot === undefined ? -1 : input.targetSlot / 4;
  vector[10] = input.amount ?? 0;
  vector[11] = input.endsTurn ? 1 : 0;
  return vector;
}

function scoreUmamusume(umamusume: UmamusumeInstance): number {
  const attack = getPrimaryAttack(getUmamusumeCard(umamusume));
  return umamusume.hp + attack.damage + attachedEnergyCount(umamusume) * 18 + umamusume.stage * 24;
}

function phaseIndex(phase: AiPhase): number {
  return ["setup", "pendingChoice", "bench", "trainerBefore", "evolve", "attach", "trainerAfter", "ability", "combat", "stadiumOrEnd"].indexOf(phase);
}

function kindIndex(kind: string): number {
  return ["pass", "setupChooseBoard", "resolvePendingChoice", "playBasic", "playTrainer", "evolve", "attachEnergy", "useAbility", "retreat", "retreatAttack", "attack", "useStadium", "endTurn"].indexOf(kind);
}

function hashToUnit(text: string): number {
  let hash = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) / 4294967295;
}
