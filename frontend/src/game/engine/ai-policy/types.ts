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

export type LegalAiAction = {
  id: string;
  phase: AiPhase;
  kind: string;
  payload: Record<string, unknown>;
  features: number[];
};

export type PublicObservation = {
  schemaVersion: 1;
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
