import type { GameState, SideId } from "../../../../../shared/src/types";

export type AiPhase =
  | "setup"
  | "pendingChoice"
  | "bench"
  | "trainerBefore"
  | "evolve"
  | "attach"
  | "trainerAfter"
  | "ability"
  | "combat"
  | "stadiumOrEnd";

// R7.b.2 Phase 1: 8-zone enumeration for the per-zone card-id arrays consumed
// by the Phase 2 Python embedding table. Member order matches scoping
// § 4 #1's grouping.
export type ZoneKey =
  | "ownActive"
  | "oppActive"
  | "ownBench"
  | "oppBench"
  | "ownHand"
  | "ownDiscard"
  | "oppDiscard"
  | "stadium";

export type LegalAiAction = {
  id: string;
  phase: AiPhase;
  kind: string;
  payload: Record<string, unknown>;
  features: number[];
  // R7.b.2 Phase 1: per-action card-vocab idx for the action's source
  // (umamusume performing or hand card being played) and target (umamusume
  // being targeted). `null` means "no clear source/target" (e.g. endTurn,
  // pass). Phase 2 Python collator converts `null` → 0 (padding_idx).
  actionSourceCardIdx: number | null;
  actionTargetCardIdx: number | null;
};

// R16-P1 temporal / turn-state scalar block. Bumped schemaVersion 2 → 3.
// `ownIsFirstTurn`/`opponentIsFirstTurn` are the ENERGY/SETUP-PHASE flag
// only: `(turnsTakenBySide[side] ?? 0) === 0` (the `startTurn` energy-skip
// predicate, frontend/src/game/engine/flow/turn.ts:55-72). Evolution's
// `<= 1` first-turn threshold (evolution.ts:42-44) is deliberately NOT
// re-encoded here — evolve-legality is already covered by the per-Uma
// `enteredThisTurn`/`evolvedThisTurn` sickness booleans below. See
// docs/ai-research/scoping/r16-model-feature-backlog-refinement.md
// § P1 "Resolved" for the full omission resolution.
export type PublicTemporalObservation = {
  ownTurnsTaken: number;
  opponentTurnsTaken: number;
  ownIsFirstTurn: boolean;
  opponentIsFirstTurn: boolean;
};

export type PublicSideTurnState = {
  energyAttachmentsThisTurn: number;
  bonusEnergyAttachments: number;
  // Raw per-side reduction (`SideState.retreatCostReduction`). Kept for
  // completeness/debug; the legality-relevant quantity is the derived
  // `effectiveRetreatCostReduction` below (raw + stadium global term).
  retreatCostReduction: number;
  // Derived: side.retreatCostReduction + getGlobalRetreatCostReduction(state)
  // (the stadium `globalRetreatCostReduction` effect; retreat.ts:13,20-26).
  // The Python v3.1 encoder maps the retreat slot to THIS value.
  effectiveRetreatCostReduction: number;
  activeAttackDamageBonus: number;
  // Counts ONLY — the ability NAME strings are public-info-sensitive and
  // are never emitted (hidden-info contract).
  usedAbilityNameCountThisTurn: number;
  usedAbilityNameCountThisGame: number;
  guaranteedCoinFlipHeads: number;
};

export type PublicUmaTurnState = {
  turnsInPlay: number;
  enteredThisTurn: boolean;
  evolvedThisTurn: boolean;
  evolvedLastTurn: boolean;
  tookDamageLastTurn: boolean;
  tookDamageThisTurn: boolean;
  nextTurnDamageReduction: number;
  attackBlockedThisTurn: boolean;
  paralysisRecoveryPending: boolean;
};

export type PublicObservation = {
  // R7.b.2 Phase 1 bumped 1 → 2 (added `cardIdsByZone`). R16-P1 bumps
  // 2 → 3: observation now also carries `temporal` + per-side/per-Uma
  // `turnState`. The Python encoder bumps `STATE_FEATURE_SCHEMA_VERSION`
  // 3.0 → 3.1 and adds a NEW 164-d builder when it consumes these.
  schemaVersion: 3;
  sideToAct: SideId;
  phase: AiPhase;
  turnNumber: number;
  firstPlayer: SideId;
  pendingChoiceKind: "promoteAfterKnockout" | "switchAfterGust" | null;
  temporal: PublicTemporalObservation;
  own: PublicSideObservation;
  opponent: PublicSideObservation;
  shared: {
    stadiumCardId: string | null;
    currentSide: GameState["currentSide"];
    gameOver: boolean;
  };
  // R7.b.2 Phase 1: per-zone card-vocab idx arrays for the embedding pass.
  // Lengths are variable TS-side (no padding here); Phase 2 Python collator
  // pads to the fixed shapes documented in scoping § 11 Phase 2.
  cardIdsByZone: Record<ZoneKey, number[]>;
};

export type PublicSideObservation = {
  id: SideId;
  points: number;
  handCount: number;
  handCardIds?: string[];
  deckCount: number;
  discard: string[];
  active: PublicUmaObservation | null;
  bench: Array<PublicUmaObservation | null>;
  energyZone: string[];
  usedSupporterThisTurn: boolean;
  usedRetreatThisTurn: boolean;
  usedStadiumThisTurn: boolean;
  turnState: PublicSideTurnState;
};

export type PublicUmaObservation = {
  uid: number;
  cardId: string;
  species: string;
  stage: number;
  hp: number;
  maxHp: number;
  energyTotal: number;
  energies: Record<string, number>;
  specialConditions: string[];
  toolCardId: string | null;
  usedAbilityThisTurn: boolean;
  turnState: PublicUmaTurnState;
};

export type AiPolicyInput = {
  state: GameState;
  sideId: SideId;
  observation: PublicObservation;
  legalActions: LegalAiAction[];
  phase: AiPhase;
};

export type AiPolicy = {
  id: string;
  selectAction: (input: AiPolicyInput) => LegalAiAction;
};

export type TrainingExample = {
  schemaVersion: 1;
  episodeId: string;
  step: number;
  seed: string;
  sideId: SideId;
  phase: AiPhase;
  observation: PublicObservation;
  legalActions: LegalAiAction[];
  selectedActionId: string;
  selectedActionIndex: number;
  policy: string;
  result: {
    winner: SideId | null;
    points: Record<SideId, number>;
  };
};
