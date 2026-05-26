import { MAX_BENCH } from "../../../../../shared/src/gameData";
import { cardVocabIndex } from "../../../../../shared/src/cardVocab";
import type { Card, EnergyType, GameState, SideId, SideState, TrainerCard, UmamusumeInstance } from "../../../../../shared/src/types";
import { getCard, getPrimaryAttack, getUmamusumeCard } from "../core/catalog";
import { attachedEnergyCount, getAllUmamusume } from "../core/umamusume";
import { choosePreferredActiveIndex, refreshContinuousHp } from "../flow/board";
import { canAttachEnergy, canAttachEnergyToUmamusume, canUseUmamusumeAbility } from "../flow/eligibility";
import { findEvolutionTarget } from "../flow/evolution";
import { getPlayableAction, getRainbowUncapEvolutionHandOptions, getRainbowUncapTargets, getToolTargets } from "../flow/playRules";
import { canUseStadium } from "../flow/trainers";
import { buildCombatCandidates } from "../flow/ai/combatPlanner";
import type { AiCombatDecision } from "../flow/ai/types";
import { scoreAiAttachTarget } from "../flow/ai/attachUtils";
import { chooseAiTurnGoal } from "../flow/ai/turnPlan";
import { scoreEvolutionTarget, shouldAiPlayTrainer } from "../flow/ai/trainerUtils";
import { getAbilityMoveEnergyTypes, hasEnoughEnergy } from "../flow/energy";
import { getAiPhase } from "./phase";
import type { AiPhase, LegalAiAction } from "./types";
import type { PlayChoices } from "../core/playTypes";

// v33-correctness-fix Fix 2-4 bumped 2 → 3. Slot 10 was polysemic
// (setup/attach/combat overloads on `amount`); slot 26 was an exact
// duplicate of slot 8; slot 28 compared `target.uid` to `targetSlot`
// (different ID spaces — near-always 0). New layout:
//   slot 10 = combat targetValue / 200 (0 outside combat)
//   slot 26 = combat lethalTarget flag (0/1, outside combat = 0)
//   slot 28 = trainer effect.heal magnitude / 100 (0 outside trainer)
// Rust mirror (`engine-rs/.../policy/actions.rs::build_features`) and
// Python ACTION_FEATURE_SCHEMA_VERSION constant in
// `training/uma_ai/features.py` bumped to match.
// v38-slim-feature-add bumped 3 → 4. Extends v3 with 4 new slots at
// [48:52]:
//   slot 48 = swap_in_attack_ready (1 iff kind∈{retreat, retreatAttack}
//             AND swap target has enough energy for its primary attack)
//   slot 49 = expected_damage_norm = damage / 300 where damage =
//             attack.damage + activeAttackDamageBonus +
//             (weakness.amount if attacker.type ==
//             defender.weakness.type AND damage > 0 else 0). RESTRICTED
//             — no per-attack conditional bonuses, no coin-flip, no
//             discard-conditioned bonuses.
//   slot 50 = attach_color_matches_typed_need (1 iff attachEnergy with
//             side.energyZone[0] color reduces typed deficit on target)
//   slot 51 = attach_completes_typed_threshold (1 iff post-attach
//             target meets primary-attack typed cost; colorless still
//             allowed unmet)
// Rust mirror (`engine-rs/.../policy/actions.rs::build_features`),
// Python ACTION_FEATURE_SCHEMA_VERSION, and Rust ACTION_FEATURE_COUNT
// in `engine-rs/.../policy/featurize.rs` + `policy/actions.rs` bumped
// to match. v3 slots [0:48] BYTE-STABLE.
export const ACTION_FEATURE_SCHEMA_VERSION = 4;
export const ACTION_FEATURE_COUNT = 52;
const ENERGY_TYPES: EnergyType[] = ["grass", "fire", "water", "lightning", "psychic", "fighting", "darkness", "steel", "colorless", "dragon"];

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

export function chooseLowestScoredAction(actions: LegalAiAction[]): LegalAiAction {
  return [...actions].sort((left, right) => (left.features[0] ?? 0) - (right.features[0] ?? 0))[0] ?? passAction("stadiumOrEnd");
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
    }),
    actionSourceCardIdx: cardVocabIndex(active.cardId),
    actionTargetCardIdx: null,
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
    actionSourceCardIdx: null,
    actionTargetCardIdx: cardVocabIndex(target.cardId),
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
      actionSourceCardIdx: cardVocabIndex(cardId),
      actionTargetCardIdx: null,
    }];
  });
}

function enumerateTrainerActions(state: GameState, side: SideState, phase: Extract<AiPhase, "trainerBefore" | "trainerAfter">): LegalAiAction[] {
  const turnGoal = chooseAiTurnGoal(state, side);
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
    score += shouldAiPlayTrainer(state, side, card, handIndex, turnGoal) ? 28 : -90;
    return enumerateTrainerChoices(state, side, card, handIndex).map((choices) => {
      const target = choices.umamusumeTargetUid !== undefined
        ? getAllUmamusume(side).find((umamusume) => umamusume.uid === choices.umamusumeTargetUid)
        : undefined;
      const choiceCardId = getChoiceCardId(side, choices);
      return {
        id: `${phase}:trainer:${handIndex}:${cardId}:${choiceKey(choices)}`,
        phase,
        kind: "playTrainer",
        payload: { handIndex, choices },
        features: features({
          score: score + scoreTrainerChoices(side, choices) * 0.05,
          phase,
          kind: "playTrainer",
          sourceCardId: cardId,
          sourceHandIndex: handIndex,
          ...(choiceCardId ? { choiceCardId } : {}),
          ...(target ? { target } : {}),
        }),
        actionSourceCardIdx: cardVocabIndex(cardId),
        actionTargetCardIdx: target ? cardVocabIndex(target.cardId) : null,
      };
    });
  });
}

function enumerateEvolutionActions(state: GameState, side: SideState): LegalAiAction[] {
  return side.hand.flatMap((cardId, handIndex) => {
    const card = getCard(cardId);
    if (card.kind !== "umamusume" || card.stage <= 0) return [];
    const target = findEvolutionTarget(state, side, card);
    if (!target) return [];
    return [{
      id: `evolve:${handIndex}:${cardId}:${target.uid}`,
      phase: "evolve" as const,
      kind: "evolve",
      payload: { handIndex, targetUid: target.uid },
      features: features({
        score: scoreEvolutionTarget(state, side, target, card),
        phase: "evolve",
        kind: "evolve",
        sourceCardId: cardId,
        sourceHandIndex: handIndex,
        target,
      }),
      actionSourceCardIdx: cardVocabIndex(cardId),
      actionTargetCardIdx: cardVocabIndex(target.cardId),
    }];
  });
}

function enumerateAttachActions(state: GameState, side: SideState): LegalAiAction[] {
  if (!canAttachEnergy(state, side)) return [];
  const turnGoal = chooseAiTurnGoal(state, side);
  const attachColor = side.energyZone[0];
  return getAllUmamusume(side).flatMap((target, slot) => {
    if (!canAttachEnergyToUmamusume(state, side, target)) return [];
    return [{
      id: `attach:${target.uid}`,
      phase: "attach" as const,
      kind: "attachEnergy",
      payload: { targetUid: target.uid },
      features: features({
        score: scoreAiAttachTarget(state, side, target, turnGoal),
        phase: "attach",
        kind: "attachEnergy",
        target,
        targetSlot: slot,
        ...(attachColor !== undefined ? { attachColor } : {}),
      }),
      actionSourceCardIdx: null,
      actionTargetCardIdx: cardVocabIndex(target.cardId),
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
    const opponent = state.sides[side.id === "player" ? "opponent" : "player"];
    if (ability.moveBenchedEnergyToActive) {
      return side.bench.flatMap((energySource) => getAbilityMoveEnergyTypes(ability)
        .filter((energyType) => energySource.energies[energyType] > 0)
        .map((energyType) => ({
          id: `ability:${source.uid}:${ability.name}:move:${energySource.uid}:${energyType}`,
          phase: "ability" as const,
          kind: "useAbility",
          payload: { sourceUid: source.uid, abilityName: ability.name, energySourceUid: energySource.uid, energyType },
          features: features({
            score: score + (side.active ? scoreUmamusume(side.active) * 0.04 : 0),
            phase: "ability",
            kind: "useAbility",
            sourceCardId: source.cardId,
            target: energySource,
            targetSlot: slot,
          }),
          actionSourceCardIdx: cardVocabIndex(source.cardId),
          actionTargetCardIdx: cardVocabIndex(energySource.cardId),
        })));
    }
    if (ability.damageOpponent && ability.damageOpponentTarget === "any") {
      return getAllUmamusume(opponent).map((target, targetSlot) => ({
        id: `ability:${source.uid}:${ability.name}:damage:${target.uid}`,
        phase: "ability" as const,
        kind: "useAbility",
        payload: { sourceUid: source.uid, abilityName: ability.name, targetUid: target.uid },
        features: features({
          score: score + (target.hp <= ability.damageOpponent! ? 80 : 0) + scoreUmamusume(target) * 0.05,
          phase: "ability",
          kind: "useAbility",
          sourceCardId: source.cardId,
          target,
          targetSlot,
        }),
        actionSourceCardIdx: cardVocabIndex(source.cardId),
        actionTargetCardIdx: cardVocabIndex(target.cardId),
      }));
    }
    if (ability.discardToDraw) {
      return discardChoiceIndexes(side, -1).map((discardHandIndex) => ({
        id: `ability:${source.uid}:${ability.name}:discard:${discardHandIndex}`,
        phase: "ability" as const,
        kind: "useAbility",
        payload: { sourceUid: source.uid, abilityName: ability.name, discardHandIndex },
        features: features({
          score: score - discardHandIndex * 0.2,
          phase: "ability",
          kind: "useAbility",
          sourceCardId: source.cardId,
          ...(side.hand[discardHandIndex] ? { choiceCardId: side.hand[discardHandIndex] } : {}),
          target: source,
          targetSlot: slot,
        }),
        actionSourceCardIdx: cardVocabIndex(source.cardId),
        actionTargetCardIdx: cardVocabIndex(source.cardId),
      }));
    }
    return [{
      id: `ability:${source.uid}:${ability.name}`,
      phase: "ability" as const,
      kind: "useAbility",
      payload: { sourceUid: source.uid, abilityName: ability.name },
      features: features({ score, phase: "ability", kind: "useAbility", sourceCardId: source.cardId, target: source, targetSlot: slot }),
      actionSourceCardIdx: cardVocabIndex(source.cardId),
      actionTargetCardIdx: cardVocabIndex(source.cardId),
    }];
  });
}

function enumerateTrainerChoices(state: GameState, side: SideState, card: TrainerCard, handIndex: number): PlayChoices[] {
  let choices: PlayChoices[] = [{}];
  if (card.effect.discardOtherCard) {
    choices = expandChoices(choices, discardChoiceIndexes(side, handIndex).map((discardHandIndex) => ({ discardHandIndex })));
  }
  if (card.effect.searchUmamusume || card.effect.searchEvolutionUmamusume) {
    choices = expandChoices(choices, searchDeckIndexes(side, card).map((deckCardIndex) => ({ deckCardIndex })));
  }
  if (card.effect.attachEnergyFromZoneToBench && side.bench.length > 0) {
    choices = expandChoices(choices, side.bench.map((target) => ({ umamusumeTargetUid: target.uid })));
  }
  if (card.effect.heal && card.effect.healTarget === "any") {
    const targets = getAllUmamusume(side).filter((target) => target.hp < target.maxHp);
    if (targets.length > 0) choices = expandChoices(choices, targets.map((target) => ({ umamusumeTargetUid: target.uid })));
  }
  if (card.trainerType === "tool") {
    choices = expandChoices(choices, getToolTargets(side).map((target) => ({ umamusumeTargetUid: target.uid })));
  }
  if (card.effect.rainbowUncapCrystal) {
    const rainbowChoices = getRainbowUncapTargets(state, side).flatMap((target) => (
      getRainbowUncapEvolutionHandOptions(side, target).map((option) => ({
        umamusumeTargetUid: target.uid,
        rainbowEvolutionHandIndex: option.handIndex,
      }))
    ));
    choices = expandChoices(choices, rainbowChoices);
  }
  return choices.length > 0 ? choices : [{}];
}

function expandChoices(base: PlayChoices[], additions: PlayChoices[]): PlayChoices[] {
  if (additions.length === 0) return [];
  return base.flatMap((choice) => additions.map((addition) => ({ ...choice, ...addition })));
}

function discardChoiceIndexes(side: SideState, playedHandIndex: number): number[] {
  return side.hand
    .map((cardId, handIndex) => ({ handIndex, score: scoreDiscardCandidate(cardId) + (handIndex === playedHandIndex ? 10000 : 0) }))
    .filter(({ handIndex }) => handIndex !== playedHandIndex)
    .sort((left, right) => left.score - right.score)
    .slice(0, 4)
    .map(({ handIndex }) => handIndex);
}

function searchDeckIndexes(side: SideState, trainer: TrainerCard): number[] {
  return side.deck
    .map((cardId, deckCardIndex) => ({ cardId, deckCardIndex, card: getCard(cardId) }))
    .filter(({ card }) => {
      if (trainer.effect.searchEvolutionUmamusume) return card.kind === "umamusume" && card.stage > 0;
      return trainer.effect.searchUmamusume ? card.kind === "umamusume" : false;
    })
    .sort((left, right) => scoreSearchCandidate(right.cardId) - scoreSearchCandidate(left.cardId))
    .slice(0, 8)
    .map(({ deckCardIndex }) => deckCardIndex);
}

function scoreSearchCandidate(cardId: string): number {
  const card = getCard(cardId);
  if (card.kind !== "umamusume") return 0;
  return card.hp + getPrimaryAttack(card).damage * 1.5 + card.stage * 24;
}

function scoreDiscardCandidate(cardId: string): number {
  const card = getCard(cardId);
  if (card.kind === "trainer") return card.trainerType === "stadium" ? 18 : 28;
  return card.hp + getPrimaryAttack(card).damage + card.stage * 35;
}

function scoreTrainerChoices(side: SideState, choices: PlayChoices): number {
  const target = choices.umamusumeTargetUid !== undefined
    ? getAllUmamusume(side).find((umamusume) => umamusume.uid === choices.umamusumeTargetUid)
    : undefined;
  return (target ? scoreUmamusume(target) : 0) - (choices.discardHandIndex ?? 0);
}

function choiceKey(choices: PlayChoices): string {
  return [
    choices.discardHandIndex ?? "x",
    choices.deckCardIndex ?? "x",
    choices.umamusumeTargetUid ?? "x",
    choices.rainbowEvolutionHandIndex ?? "x",
  ].join(":");
}

function enumerateCombatActions(state: GameState, side: SideState): LegalAiAction[] {
  const actions: LegalAiAction[] = [];
  if (side.active) {
    const candidates = buildCombatCandidates(state, side, {
      refreshContinuousEffects: refreshContinuousHp,
      choosePreferredActiveIndex,
    });
    const opponent = state.sides[side.id === "player" ? "opponent" : "player"];
    candidates.forEach((candidate, index) => {
      const targetUid = candidate.decision.kind === "attack" ? candidate.decision.attackTargetUid : undefined;
      const target = targetUid !== undefined
        ? getAllUmamusume(opponent).find((umamusume) => umamusume.uid === targetUid)
        : undefined;
      const sourceCardId = combatSourceCardId(side, candidate.decision);
      // v3.8: pass cross-bit context for slots [48:49]. For retreatAttack
      // the swap-in is the bench Uma referenced by `retreatTargetUid`; for
      // plain attack the swap target is the existing active. `defenderActive`
      // = opponent's active Uma (slot 49 expected_damage_norm computes
      // attacker→opponent.active damage). NOTE: `kind` below preserves the
      // v3 byte-stable semantics — TS source historically passes the RAW
      // decision kind ("attack" or "endTurn") to `features()`, NOT the
      // action-emit kind ("retreatAttack"). Slots that depend on action
      // kind name (e.g. v3 slot 2 = kindIndex/16) MUST remain byte-stable.
      const retreatTargetUid = candidate.decision.kind === "attack" ? candidate.decision.retreatTargetUid : undefined;
      const retreatSwapTarget = retreatTargetUid !== undefined
        ? side.bench.find((u) => u.uid === retreatTargetUid)
        : undefined;
      const featureInput = {
        score: candidate.score + (candidate.lethalTarget ? 100 : 0) + (candidate.keepsSafe ? 20 : 0),
        phase: "combat" as const,
        kind: candidate.decision.kind,
        targetValue: candidate.targetValue,
        lethalTarget: candidate.lethalTarget,
        endsTurn: candidate.decision.kind === "attack",
        attackerSide: side,
        ...(opponent.active ? { defenderActive: opponent.active } : {}),
        ...(retreatSwapTarget ? { retreatSwapTarget } : {}),
      };
      actions.push({
        id: `combat:${candidate.id}:${index}`,
        phase: "combat",
        kind: candidate.decision.kind === "attack" && candidate.decision.retreatTargetUid !== undefined ? "retreatAttack" : candidate.decision.kind,
        payload: { decision: candidate.decision },
        features: features({
          ...featureInput,
          ...(sourceCardId ? { sourceCardId } : {}),
          ...(target ? { target } : {}),
        }),
        actionSourceCardIdx: sourceCardId ? cardVocabIndex(sourceCardId) : null,
        actionTargetCardIdx: target ? cardVocabIndex(target.cardId) : null,
      });
    });
  }
  return actions.length > 0 ? actions : [passAction("combat")];
}

function combatSourceCardId(side: SideState, decision: AiCombatDecision): string | undefined {
  if (decision.kind !== "attack") return undefined;
  if (decision.retreatTargetUid !== undefined) {
    return side.bench.find((umamusume) => umamusume.uid === decision.retreatTargetUid)?.cardId;
  }
  return side.active?.cardId;
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
      actionSourceCardIdx: null,
      actionTargetCardIdx: null,
    });
  }
  actions.push({
    id: "turn:end",
    phase: "stadiumOrEnd",
    kind: "endTurn",
    payload: {},
    features: features({ score: 1, phase: "stadiumOrEnd", kind: "endTurn", endsTurn: true }),
    actionSourceCardIdx: null,
    actionTargetCardIdx: null,
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
    actionSourceCardIdx: null,
    actionTargetCardIdx: null,
  };
}

// R7.b.2 Phase 1: each call site sets `actionSourceCardIdx` and
// `actionTargetCardIdx` on its `LegalAiAction` literal alongside the
// `features` array. `null` is the sentinel for "no clear source/target"
// (endTurn / pass / useStadium / setup). Phase 2 Python collator maps
// `null` → 0 (the shared padding_idx of the embedding table).
//
// v38-slim-feature-add: the `attackerSide` / `defenderActive` /
// `attachColor` inputs supply cross-bit context for slots [48:52]. When
// absent the new slots emit 0 (slot 48 only fires for retreat/retreatAttack;
// slot 49 for attack/retreatAttack/useAbility; slots 50/51 for attachEnergy).
// Call sites pass these only on the action kinds where they're meaningful.
function features(input: {
  score: number;
  phase: AiPhase;
  kind: string;
  sourceCardId?: string;
  sourceHandIndex?: number;
  choiceCardId?: string;
  target?: UmamusumeInstance;
  targetSlot?: number;
  targetValue?: number;
  lethalTarget?: boolean;
  endsTurn?: boolean;
  // v3.8 cross-bit context (see slot definitions in the header above).
  attackerSide?: SideState;
  defenderActive?: UmamusumeInstance;
  attachColor?: EnergyType;
  retreatSwapTarget?: UmamusumeInstance;
}): number[] {
  const vector = Array.from({ length: ACTION_FEATURE_COUNT }, () => 0);
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
  vector[10] = input.targetValue !== undefined ? input.targetValue / 200 : 0;
  vector[11] = input.endsTurn ? 1 : 0;
  const sourceCard = input.sourceCardId ? getCard(input.sourceCardId) : null;
  vector[12] = input.kind === "pass" ? 1 : 0;
  vector[13] = sourceCard?.kind === "umamusume" ? 1 : sourceCard?.kind === "trainer" ? 0.5 : 0;
  vector[14] = sourceCard?.kind === "umamusume" ? sourceCard.stage / 2 : 0;
  vector[15] = sourceCard?.kind === "umamusume" ? sourceCard.hp / 180 : 0;
  vector[16] = sourceCard?.kind === "umamusume" ? getPrimaryAttack(sourceCard).damage / 150 : 0;
  vector[17] = sourceCard?.kind === "umamusume" ? Object.values(getPrimaryAttack(sourceCard).cost).reduce((sum, cost) => sum + (cost ?? 0), 0) / 5 : 0;
  vector[18] = sourceCard?.kind === "trainer" && sourceCard.trainerType === "supporter" ? 1 : 0;
  vector[19] = sourceCard?.kind === "trainer" && sourceCard.trainerType === "stadium" ? 1 : 0;
  vector[20] = sourceCard?.kind === "trainer" && sourceCard.trainerType === "tool" ? 1 : 0;
  vector[21] = sourceCard?.kind === "trainer" ? (sourceCard.effect.draw ?? 0) / 5 : 0;
  vector[22] = sourceCard?.kind === "trainer" && (sourceCard.effect.searchUmamusume || sourceCard.effect.searchEvolutionUmamusume || sourceCard.effect.searchRandomBasicUmamusume) ? 1 : 0;
  vector[23] = sourceCard?.kind === "trainer" && (sourceCard.effect.extraEnergyAttach || sourceCard.effect.attachEnergyFromZoneToBench) ? 1 : 0;
  vector[24] = sourceCard?.kind === "trainer" ? (sourceCard.effect.activeAttackDamageBonus ?? 0) / 100 : 0;
  vector[25] = input.target ? input.target.maxHp / 180 : 0;
  vector[26] = input.lethalTarget ? 1 : 0;
  vector[27] = input.target ? input.target.specialConditions.length / 4 : 0;
  vector[28] = sourceCard?.kind === "trainer" ? (sourceCard.effect.heal ?? 0) / 100 : 0;
  vector[29] = input.kind === "attack" || input.kind === "retreatAttack" ? 1 : 0;
  vector[30] = input.kind === "useAbility" ? 1 : 0;
  vector[31] = input.kind === "endTurn" ? 1 : 0;
  vector[32] = sourceCard?.kind === "trainer" && sourceCard.trainerType === "item" ? 1 : 0;
  vector[33] = sourceCard?.kind === "trainer" && (sourceCard.effect.gustOpponent || sourceCard.effect.discardRandomOpponentActiveEnergy || sourceCard.effect.disableTools) ? 1 : 0;
  vector[34] = sourceCard?.kind === "trainer" && (sourceCard.effect.heal || sourceCard.effect.recoverActiveSpecialConditions) ? 1 : 0;
  vector[35] = sourceCard?.kind === "trainer" && (sourceCard.effect.discardOtherCard || sourceCard.effect.randomBasicUmamusumeFromDiscard) ? 1 : 0;
  vector[36] = sourceCard?.kind === "trainer" && (sourceCard.effect.shuffleHandIntoDeckDraw || sourceCard.effect.rainbowUncapCrystal || sourceCard.effect.basicHpBonus || sourceCard.effect.globalRetreatCostReduction) ? 1 : 0;
  vector[37] = sourceCard?.kind === "umamusume" && sourceCard.ability ? 1 : 0;
  vector[38] = sourceCard?.kind === "umamusume" ? typedAttackCost(sourceCard) / 4 : 0;
  vector[39] = sourceCard?.kind === "umamusume" ? colorlessAttackCost(sourceCard) / 4 : 0;
  vector[40] = sourceCard?.kind === "umamusume" && hasFlexibleAttackTarget(sourceCard) ? 1 : 0;
  vector[41] = sourceCard?.kind === "umamusume" && hasConditionalAttackOrAbility(sourceCard) ? 1 : 0;
  const choiceCard = input.choiceCardId ? getCard(input.choiceCardId) : null;
  vector[42] = choiceCard ? cardRoleKind(choiceCard) : 0;
  vector[43] = choiceCard ? cardRoleProgression(choiceCard) : 0;
  vector[44] = choiceCard ? cardRoleOutput(choiceCard) : 0;
  vector[45] = choiceCard ? cardRoleUtility(choiceCard) : 0;
  vector[46] = input.target ? attackReadiness(input.target) : 0;
  vector[47] = input.target ? typedEnergyDeficit(input.target) / 4 : 0;
  // v38-slim-feature-add slots [48:52]. See header comment for the
  // LOCKED slot definitions. All four default to 0 when context is
  // absent or the action kind doesn't apply.
  vector[48] = v38SwapInAttackReady(input);
  vector[49] = v38ExpectedDamageNorm(input);
  vector[50] = v38AttachColorMatchesTypedNeed(input);
  vector[51] = v38AttachCompletesTypedThreshold(input);
  return vector;
}

// v3.8 slot 48 — swap_in_attack_ready. 1 iff this action involves a
// retreat-style swap (i.e. `retreatSwapTarget` is supplied by the
// retreat/retreatAttack enumerator) AND the swap-in target has enough
// energy for its primary attack. Per scoping §4.5 slot 48 def.
// NOTE: TS combat enumerator passes `kind: candidate.decision.kind`
// (raw "attack"/"endTurn", v3-byte-stable contract), so the predicate
// keys off `retreatSwapTarget` presence rather than the action's emitted
// kind name — the enumerator supplies `retreatSwapTarget` iff
// `decision.retreatTargetUid !== undefined` (== retreatAttack semantics).
function v38SwapInAttackReady(input: { kind: string; target?: UmamusumeInstance; retreatSwapTarget?: UmamusumeInstance }): number {
  const swapIn = input.retreatSwapTarget;
  if (!swapIn) return 0;
  const card = getUmamusumeCard(swapIn);
  const attack = getPrimaryAttack(card);
  return hasEnoughEnergy(swapIn, attack.cost) ? 1 : 0;
}

// v3.8 slot 49 — expected_damage_norm. Fires when defender + attacker
// are known (combat enumerator supplies both — slot fires for raw kind
// "attack" since combat passes raw decision.kind). RESTRICTED to base +
// activeAttackDamageBonus + weakness (no per-attack conditional bonuses,
// no coin-flip, no discard). Returns damage / 300 (300 ≈ engine max
// face-value damage cap with weakness). Per scoping §4.5 slot 49 def.
function v38ExpectedDamageNorm(input: { kind: string; sourceCardId?: string; attackerSide?: SideState; defenderActive?: UmamusumeInstance }): number {
  if (input.kind !== "attack" && input.kind !== "retreatAttack" && input.kind !== "useAbility") return 0;
  if (!input.sourceCardId || !input.defenderActive) return 0;
  const attackerCard = getCard(input.sourceCardId);
  if (attackerCard.kind !== "umamusume") return 0;
  const primary = getPrimaryAttack(attackerCard);
  let damage = primary.damage;
  if (input.attackerSide) damage += input.attackerSide.activeAttackDamageBonus;
  if (damage > 0) {
    const defenderCard = getCard(input.defenderActive.cardId);
    if (defenderCard.kind === "umamusume" && defenderCard.weakness.type === attackerCard.type) {
      damage += defenderCard.weakness.amount;
    }
  }
  return Math.max(0, damage) / 300;
}

// v3.8 slot 50 — attach_color_matches_typed_need. 1 iff kind=attachEnergy
// AND attachColor (= side.energyZone[0]) reduces a typed deficit on the
// target's primary attack cost. Per scoping §4.5 slot 50 def.
function v38AttachColorMatchesTypedNeed(input: { kind: string; target?: UmamusumeInstance; attachColor?: EnergyType }): number {
  if (input.kind !== "attachEnergy") return 0;
  if (!input.target || !input.attachColor) return 0;
  if (input.attachColor === "colorless") return 0;
  const card = getUmamusumeCard(input.target);
  const attack = getPrimaryAttack(card);
  const need = attack.cost[input.attachColor] ?? 0;
  const have = input.target.energies[input.attachColor];
  return need > have ? 1 : 0;
}

// v3.8 slot 51 — attach_completes_typed_threshold. 1 iff kind=attachEnergy
// AND post-attach target meets primary-attack TYPED cost (colorless still
// allowed unmet). Per scoping §4.5 slot 51 def.
function v38AttachCompletesTypedThreshold(input: { kind: string; target?: UmamusumeInstance; attachColor?: EnergyType }): number {
  if (input.kind !== "attachEnergy") return 0;
  if (!input.target || !input.attachColor) return 0;
  const card = getUmamusumeCard(input.target);
  const attack = getPrimaryAttack(card);
  for (const type of ENERGY_TYPES) {
    if (type === "colorless") continue;
    const need = attack.cost[type] ?? 0;
    let have = input.target.energies[type];
    if (type === input.attachColor) have += 1;
    if (have < need) return 0;
  }
  return 1;
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

function getChoiceCardId(side: SideState, choices: PlayChoices): string | undefined {
  if (choices.deckCardIndex !== undefined) return side.deck[choices.deckCardIndex];
  if (choices.discardHandIndex !== undefined) return side.hand[choices.discardHandIndex];
  if (choices.rainbowEvolutionHandIndex !== undefined) return side.hand[choices.rainbowEvolutionHandIndex];
  return undefined;
}

function cardRoleKind(card: Card): number {
  if (card.kind === "umamusume") return 1;
  if (card.trainerType === "supporter") return 0.75;
  if (card.trainerType === "item") return 0.55;
  if (card.trainerType === "tool") return 0.35;
  return 0.2;
}

function cardRoleProgression(card: Card): number {
  if (card.kind === "umamusume") return card.stage / 2;
  if (card.trainerType === "supporter") return 0.75;
  if (card.trainerType === "item") return 0.5;
  if (card.trainerType === "tool") return 0.35;
  return 0.2;
}

function cardRoleOutput(card: Card): number {
  if (card.kind === "umamusume") return Math.max(card.hp / 180, getPrimaryAttack(card).damage / 150);
  const effect = card.effect;
  return Math.max(
    (effect.draw ?? effect.shuffleHandIntoDeckDraw ?? 0) / 5,
    (effect.heal ?? 0) / 80,
    (effect.activeAttackDamageBonus ?? 0) / 100,
  );
}

function cardRoleUtility(card: Card): number {
  if (card.kind === "umamusume") return card.ability ? 1 : (hasFlexibleAttackTarget(card) || hasConditionalAttackOrAbility(card) ? 0.5 : 0);
  const effect = card.effect;
  if (effect.searchUmamusume || effect.searchEvolutionUmamusume || effect.searchRandomBasicUmamusume || effect.rainbowUncapCrystal) return 1;
  if (effect.extraEnergyAttach || effect.attachEnergyFromZoneToBench || effect.gustOpponent || effect.discardRandomOpponentActiveEnergy) return 0.85;
  if (effect.heal || effect.retreatCostReduction || effect.globalRetreatCostReduction || card.trainerType === "tool") return 0.55;
  return 0.25;
}

function typedAttackCost(card: Extract<Card, { kind: "umamusume" }>): number {
  const cost = getPrimaryAttack(card).cost;
  return ENERGY_TYPES.reduce((sum, type) => type === "colorless" ? sum : sum + (cost[type] ?? 0), 0);
}

function colorlessAttackCost(card: Extract<Card, { kind: "umamusume" }>): number {
  return getPrimaryAttack(card).cost.colorless ?? 0;
}

function hasFlexibleAttackTarget(card: Extract<Card, { kind: "umamusume" }>): boolean {
  const attack = getPrimaryAttack(card);
  return attack.targetOpponent === "any" || Boolean(attack.benchDamage || attack.healTarget === "any");
}

function hasConditionalAttackOrAbility(card: Extract<Card, { kind: "umamusume" }>): boolean {
  const attack = getPrimaryAttack(card);
  return Boolean(
    card.ability
      || attack.coinBonus
      || attack.drawOnHeads
      || attack.discardEnergy
      || attack.damagePerAttachedEnergy
      || attack.damagePerUniqueAttachedEnergy
      || attack.damagePerUmamusumeInPlay
      || attack.attackDamageBonusIfToolAttached
      || attack.attackDamageBonusIfDiscardHandCard
      || attack.attackDamageBonusPerDiscardedHandCard
      || attack.shuffleSelfIntoDeck
      || attack.switchSelfAfterAttack
      || attack.preventDamageNextTurn
      || attack.bonusIfTookDamageLastTurn
      || attack.cannotAttackNextTurn
      || attack.inflictSpecialCondition,
  );
}

function attackReadiness(target: UmamusumeInstance): number {
  const attack = getPrimaryAttack(getUmamusumeCard(target));
  const totalCost = Object.values(attack.cost).reduce((sum, cost) => sum + (cost ?? 0), 0);
  const energyRatio = attachedEnergyCount(target) / Math.max(1, totalCost);
  return hasEnoughEnergy(target, attack.cost) ? Math.max(1, energyRatio) : Math.min(0.99, energyRatio);
}

function typedEnergyDeficit(target: UmamusumeInstance): number {
  const attack = getPrimaryAttack(getUmamusumeCard(target));
  return ENERGY_TYPES.reduce((sum, type) => {
    if (type === "colorless") return sum;
    return sum + Math.max(0, (attack.cost[type] ?? 0) - target.energies[type]);
  }, 0);
}

function hashToUnit(text: string): number {
  let hash = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) / 4294967295;
}
