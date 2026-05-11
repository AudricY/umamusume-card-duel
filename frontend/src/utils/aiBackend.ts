// R14.E: dev-flag toggle for which AI policy drives the opponent loop.
// `rule-bot` (default) → frontend's heuristic in advanceOpponentTurnStep.
// `mcts` → request a decision from the backend's /ai/decide endpoint and
// apply the server-returned nextState. On timeout / error / engine fallback
// the caller is responsible for routing back to the rule-bot advance step.
//
// Stored under a separate localStorage key so it doesn't pollute the
// customisation settings (visual chrome). Intentionally not exposed in the
// settings UI yet — toggle via:
//   window.localStorage.setItem("umamusume-card-duel-ai-backend", "mcts")
// and reload, or use setAiBackend() at runtime.

export type AiBackend = "rule-bot" | "mcts";

const AI_BACKEND_STORAGE_KEY = "umamusume-card-duel-ai-backend";
const ALL_BACKENDS: AiBackend[] = ["rule-bot", "mcts"];

export function readAiBackend(): AiBackend {
  if (typeof window === "undefined") return "rule-bot";
  const stored = window.localStorage.getItem(AI_BACKEND_STORAGE_KEY);
  return ALL_BACKENDS.includes(stored as AiBackend) ? (stored as AiBackend) : "rule-bot";
}

export function setAiBackend(backend: AiBackend): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(AI_BACKEND_STORAGE_KEY, backend);
}
