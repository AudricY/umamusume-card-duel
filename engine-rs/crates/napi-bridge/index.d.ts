// TypeScript declarations for the Phase 2 NAPI bridge.
//
// Load via:
//   import { createRequire } from "node:module";
//   const require = createRequire(import.meta.url);
//   const bridge: NapiBridge = require("/path/to/napi_bridge.node");
//
// Build the .node first:
//   cargo build --manifest-path engine-rs/Cargo.toml -p napi-bridge --release --lib
//   cp target/release/libnapi_bridge.so target/release/napi_bridge.node

/** Opaque JSON string of a serialized GameState. Parse with JSON.parse. */
export type StateJson = string;

/** Opaque JSON string of a serialized Rng state. Treat as a cookie. */
export type RngStateJson = string;

/** Result of setup or step calls. */
export interface StepBundle {
  /** JSON of the new GameState. snake_case fields at top level. */
  stateJson: StateJson;
  /** RNG snapshot to pass into the next call to preserve determinism. */
  rngStateJson: RngStateJson;
}

/** Result of MCTS calls without state advancement. */
export interface MctsBundle extends StepBundle {
  /** JSON of MctsResult { selectedIndex, visits, diagnostics }. */
  mctsResult: string;
}

/** Result of mcts-step (search + apply). */
export interface MctsStepBundle extends MctsBundle {
  chosenActionIndex: number;
  /** JSON of the chosen LegalAiAction. */
  chosenActionJson: string;
}

/** Result of driveHeuristicGameJson. */
export interface HeuristicGameSummary {
  finalStateHash: string;
  steps: number;
  winner: "player" | "opponent" | null;
  gameOver: boolean;
  terminalReason: "gameOver" | "stalled" | "maxSteps";
}

/** MCTS config — all fields optional, defaults match sim-mcts-selfplay. */
export interface McTsArgs {
  simulations?: number;
  cPuct?: number;
  leaf?: "rollout" | "value-head";
  prior?: "uniform" | "policy";
  rolloutCrnSamples?: number;
  rolloutSteps?: number;
  addRootDirichlet?: boolean;
  dirichletAlpha?: number;
  dirichletEpsilon?: number;
  maxNodes?: number;
  collapseMaxSteps?: number;
  modelUrl?: string;
}

export interface NapiBridge {
  /** Sentinel: `rust-port vN — catalog=N cards`. */
  engineVersion(): string;

  /**
   * Seed a fresh AI-vs-AI game.
   * @returns JSON of `StepBundle`.
   */
  createGameJson(seed: string): string;

  /**
   * Drive one heuristic-AI step for `state.currentSide`.
   * Errors if game is already over.
   * @returns JSON of `StepBundle`.
   */
  advanceStepJson(stateJson: StateJson, rngStateJson: RngStateJson): string;

  /**
   * Legal actions for `state.currentSide`. Errors if game is over.
   * @returns JSON of `LegalAiAction[]`.
   */
  legalActionsJson(stateJson: StateJson, rngStateJson: RngStateJson): string;

  /** 32-char hex xxh3 fingerprint of a packed state. */
  stateHashForJson(stateJson: StateJson): string;

  /**
   * Run MCTS at the given state. State is unchanged.
   * @returns JSON of `MctsBundle`.
   */
  runMctsJson(
    stateJson: StateJson,
    rngStateJson: RngStateJson,
    mctsArgsJson: string,
    mctsSeed: string,
  ): string;

  /**
   * Run MCTS, pick most-visited action, apply it.
   * @returns JSON of `MctsStepBundle`.
   */
  mctsStepJson(
    stateJson: StateJson,
    rngStateJson: RngStateJson,
    mctsArgsJson: string,
    mctsSeed: string,
  ): string;

  /**
   * Drive a full heuristic-vs-heuristic game in pure Rust.
   * Avoids JS↔Rust roundtrip per step.
   * @returns JSON of `HeuristicGameSummary`.
   */
  driveHeuristicGameJson(seed: string, maxSteps: number): string;
}
