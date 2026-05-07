import type { GameState, SideId } from "../../../../../shared/src/types";
import type { AiPhase } from "./types";

export function getAiPhase(state: GameState, sideId: SideId): AiPhase {
  if (state.phase === "setup") return "setup";
  if (state.pendingPlayerChoice?.sideId === sideId) return "pendingChoice";
  if (state.currentSide !== sideId) return "stadiumOrEnd";
  const step = state.opponentTurnStep ?? "bench";
  switch (step) {
    case "bench":
      return "bench";
    case "trainerBefore":
      return "trainerBefore";
    case "evolve":
      return "evolve";
    case "attach":
      return "attach";
    case "trainerAfter":
      return "trainerAfter";
    case "ability":
      return "ability";
    case "attack":
      return "combat";
    case "finish":
      return "stadiumOrEnd";
    default:
      return "stadiumOrEnd";
  }
}
