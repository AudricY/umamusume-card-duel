// R13.W5 client for backend /ai/decide.
//
// The UI calls this from the opponent turn loop (and optionally the
// AI-vs-AI player loop) when `aiBackend === "mcts"`. The handler awaits
// the server's decision with a configurable timeout and, on any failure
// (HTTP error, network drop, timeout), falls back gracefully via the
// returned `fallback` flag so the caller can advance with the rule bot.
//
// Latency expectations (post R13.W2 dials, K=1 + adaptive halt):
//   - p50 ≈ 0.4–1.0 s for most decisions.
//   - p95 ≈ 2–3 s on positions where MCTS doesn't dominate quickly.
// The 5 s default timeout below leaves margin for both.

import type { GameState, SideId } from "../../../../../shared/src/types";
import type { LegalAiAction } from "./types";

export type MctsDecisionResponse = {
  actionIndex: number;
  selectedActionId: string;
  decisionMs: number;
  simulationsRun: number;
  haltedEarly: boolean;
  reason?: string;
};

export type MctsDecisionResult =
  | { ok: true; action: LegalAiAction | null; response: MctsDecisionResponse }
  | { ok: false; reason: "timeout" | "http_error" | "transport_error" | "bad_response"; message: string };

export type MctsClientOptions = {
  endpointUrl?: string;
  timeoutMs?: number;
  modelUrl?: string;
  seed?: string;
  mctsConfig?: {
    simulations?: number;
    cPuct?: number;
    leaf?: "value-head" | "rollout";
    rolloutCrnSamples?: number;
    rolloutSteps?: number;
    prior?: "uniform" | "policy";
    adaptiveRatio?: number;
    adaptiveMinSims?: number;
    collapseMaxSteps?: number;
    maxNodes?: number;
  };
};

const DEFAULT_ENDPOINT = "/ai/decide";
const DEFAULT_TIMEOUT_MS = 5000;

export async function requestMctsDecision(
  state: GameState,
  modelSide: SideId,
  legalActions: readonly LegalAiAction[],
  options: MctsClientOptions = {},
): Promise<MctsDecisionResult> {
  const endpoint = options.endpointUrl ?? DEFAULT_ENDPOINT;
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        state,
        modelSide,
        modelUrl: options.modelUrl,
        seed: options.seed,
        mctsConfig: options.mctsConfig,
      }),
      signal: controller.signal,
    });
    if (!response.ok) {
      const message = await response.text().catch(() => `HTTP ${response.status}`);
      return { ok: false, reason: "http_error", message: `HTTP ${response.status}: ${message}` };
    }
    const payload = (await response.json()) as MctsDecisionResponse;
    if (typeof payload.actionIndex !== "number" || !Number.isFinite(payload.actionIndex)) {
      return { ok: false, reason: "bad_response", message: "missing actionIndex" };
    }
    const action = payload.actionIndex >= 0 && payload.actionIndex < legalActions.length
      ? legalActions[payload.actionIndex] ?? null
      : null;
    return { ok: true, action, response: payload };
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      return { ok: false, reason: "timeout", message: `MCTS decision exceeded ${timeoutMs}ms` };
    }
    const message = error instanceof Error ? error.message : "transport failed";
    return { ok: false, reason: "transport_error", message };
  } finally {
    window.clearTimeout(timeoutId);
  }
}
