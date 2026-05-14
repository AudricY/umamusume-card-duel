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

export type PublicObservation = {
  // R7.b.2 Phase 1 bumped 1 → 2; observation now also carries
  // `cardIdsByZone`. Phase 2 Python encoder will bump its own
  // `STATE_FEATURE_SCHEMA_VERSION` once it consumes the new field.
  schemaVersion: 2;
  sideToAct: SideId;
  phase: AiPhase;
  turnNumber: number;
  firstPlayer: SideId;
  pendingChoiceKind: "promoteAfterKnockout" | "switchAfterGust" | null;
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
