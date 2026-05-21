// Phase 0 golden-trace recorder for the Rust engine port (see
// docs/ai-research/scoping/rust-engine-port-plan.md § 3 "Phase 0").
//
// Replays the current TS simulator self-play loop for N seeds and writes one
// JSONL line per seed: `(seed, configHash, per-step (fingerprintBefore,
// action, rngDrawsThisStep), terminalFingerprint, winner, turnNumber,
// totalRngDraws)`. This file is the gate Phase 1 hangs on — the Rust port
// must reproduce every recorded field byte-for-byte at every seed.
//
// Determinism notes:
//   - Single-process by design. `mctsSelfPlay.ts` ships a worker fan-out
//     orchestrator; we deliberately do NOT use it here. Cross-process
//     fork/IPC ordering is irrelevant once each game runs serially in one
//     process, and the worker shim depends on the parent script being
//     `mctsSelfPlay.ts` (its `process.argv[1]` is hardcoded as the worker
//     entrypoint). A serial loop also gives us a clean
//     "compute fingerprint → record → step → record rng count" sequence
//     per step with no IPC interleaving.
//   - Each seed gets its own seeded rng (`createSeededRng(${seed}:selfplay)`)
//     wrapped with the `rngInstrumentation.ts` counter, and the wrapped rng
//     is installed via `withRng(...)`. All `randomFloat()` reads, all
//     `rng.next` callbacks passed into the engine, and all forks are
//     therefore counted under the same counter.
//   - MCTS internally creates its own rng (`createSeededRng(seed, "mcts-root")`
//     in `mcts.ts:177`) with a string seed derived from
//     `${seed}:${sideId}:${step}:mcts`. Those draws are deterministic from
//     that seed but DO NOT contribute to `rngDrawsThisStep` (different rng
//     tree). This is documented in `rngInstrumentation.ts`; the per-step
//     count therefore captures "draws from the recorder-owned rng tree"
//     this step. A Rust port that reproduces the same MCTS seed string
//     trivially reproduces the same internal draws too — they're a function
//     of the same input.
//   - `prior=uniform` + `leaf=rollout` is the default so the recorder needs
//     NO `serve_onnx` HTTP dependency. The R110 production config uses
//     `prior=policy`, which requires a model server; that path is supported
//     by passing `--model-url http://... --prior policy` explicitly. The
//     golden traces themselves only have to be self-consistent — what
//     matters is that the Rust port reproduces whatever config the corpus
//     was recorded at, byte-identical.
//
// CLI:
//   --seeds N                 number of seeds (default 500)
//   --seed-base K             seeds are K..K+N-1 (default 0)
//   --out PATH                JSONL output (default runs/rust-port-golden-traces/traces.jsonl)
//   --sims N                  --mcts-simulations (default 100)
//   --rollout-steps N         --mcts-rollout-steps (default 200)
//   --k N                     --mcts-rollout-crn-samples (default 3)
//   --collapse-max N          --mcts-collapse-max-steps (default 64)
//   --c-puct F                MCTS exploration constant (default 1.5)
//   --max-nodes N             MCTS tree node cap (default 5000)
//   --leaf value-head|rollout (default rollout)
//   --prior uniform|policy    (default uniform)
//   --model-url URL           required when --prior policy (default "")
//   --temperature-moves N     (default 6) — matches mctsSelfPlay
//   --temperature-value F     (default 1.0) — matches mctsSelfPlay
//   --max-steps N             per-game step cap (default 500)
//   --trace-version N         override trace schema version (default 1)

import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { dirname } from "node:path";
import { execFileSync } from "node:child_process";
import {
  advanceOpponentTurnStep,
  advancePlayerAiTurnStep,
} from "../../../frontend/src/game/engine";
import { enumerateLegalAiActions } from "../../../frontend/src/game/engine/ai-policy/actions";
import "./rngAsyncStore";
import { createSeededRng, withRng } from "../../../frontend/src/game/engine/core/random";
import type { LegalAiAction } from "../../../frontend/src/game/engine/ai-policy/types";
import type { GameState, SideId } from "../../../shared/src/types";
import {
  advanceModeledTurnStep,
  getForcedAttackCoinResults,
  setupAiVsAiGame,
  stateHash,
} from "./evaluateModelVsHeuristic";
import { defaultMctsConfig, runMcts, type MctsConfig, type MctsResult } from "./mcts";
import { stateFingerprint } from "./stateFingerprint";
import { createRngCounter, instrumentRng } from "./rngInstrumentation";

const DEFAULT_TRACE_VERSION = 1;

type RecorderArgs = {
  seeds: number;
  seedBase: number;
  outPath: string;
  modelUrl: string;
  mctsSimulations: number;
  mctsCPuct: number;
  mctsLeaf: "value-head" | "rollout";
  mctsRolloutCrnSamples: number;
  mctsRolloutSteps: number;
  mctsCollapseMaxSteps: number;
  mctsMaxNodes: number;
  mctsPrior: "uniform" | "policy";
  temperatureMoves: number;
  temperatureValue: number;
  maxSteps: number;
  traceVersion: number;
};

type ActionRecord = {
  step: number;
  turnNumber: number;
  sideId: SideId;
  fingerprintBefore: string;
  // Compact, JSON-stable action representation. `id` is the
  // human-readable per-action identifier (`actions.ts`); `kind`, `phase`,
  // and `payload` give the Rust port a structured handle. `selectedIndex`
  // and `legalCount` let the replay tool re-pick deterministically and
  // catch divergence in the legal-action enumerator's ordering.
  action: {
    id: string;
    kind: string;
    phase: string;
    payload: Record<string, unknown>;
    selectedIndex: number;
    legalCount: number;
  };
  rngDrawsThisStep: number;
};

type TraceLine = {
  seed: number;
  traceVersion: number;
  configHash: string;
  actions: ActionRecord[];
  terminalFingerprint: string;
  winner: SideId | null;
  turnNumber: number;
  totalRngDraws: number;
  terminalReason: "gameOver" | "maxSteps" | "stalled";
};

function parseArgs(argv: string[]): RecorderArgs {
  const get = (name: string, fallback: string) => {
    const i = argv.indexOf(name);
    return i >= 0 ? argv[i + 1] ?? fallback : fallback;
  };
  const leafRaw = get("--leaf", "rollout");
  const priorRaw = get("--prior", "uniform");
  return {
    seeds: Math.max(1, Number(get("--seeds", "500"))),
    seedBase: Number(get("--seed-base", "0")),
    outPath: get("--out", "runs/rust-port-golden-traces/traces.jsonl"),
    modelUrl: get("--model-url", ""),
    mctsSimulations: Math.max(1, Number(get("--sims", "100"))),
    mctsCPuct: Number(get("--c-puct", "1.5")),
    mctsLeaf: leafRaw === "value-head" ? "value-head" : "rollout",
    mctsRolloutCrnSamples: Math.max(1, Number(get("--k", "3"))),
    mctsRolloutSteps: Math.max(1, Number(get("--rollout-steps", "200"))),
    mctsCollapseMaxSteps: Math.max(1, Number(get("--collapse-max", "64"))),
    mctsMaxNodes: Math.max(64, Number(get("--max-nodes", "5000"))),
    mctsPrior: priorRaw === "policy" ? "policy" : "uniform",
    temperatureMoves: Number(get("--temperature-moves", "6")),
    temperatureValue: Number(get("--temperature-value", "1.0")),
    maxSteps: Math.max(1, Number(get("--max-steps", "500"))),
    traceVersion: Math.max(1, Number(get("--trace-version", String(DEFAULT_TRACE_VERSION)))),
  };
}

function engineGitSha(): string {
  // We hash flags + the engine source tree's HEAD git SHA into configHash.
  // `git rev-parse HEAD:frontend/src/game/engine` would give the tree sha
  // for the engine subtree only, but that fails in a dirty/uncommitted
  // working copy. Fall back to a stable marker so the recorder still runs;
  // the replay tool only checks configHash equality between traces and
  // re-runs, not absolute identity to git.
  try {
    return execFileSync("git", ["rev-parse", "HEAD:frontend/src/game/engine"], { encoding: "utf8" }).trim();
  } catch {
    try {
      return execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim() + ":fallback";
    } catch {
      return "unknown";
    }
  }
}

export function computeConfigHash(args: RecorderArgs, engineSha: string): string {
  // Stable hash over the CLI flag set that affects per-step outputs, plus
  // the engine git sha. Order is canonical: keys sorted alphabetically.
  // `seedBase`/`seeds`/`outPath` are NOT part of the hash — they choose
  // which seeds to run, not what one seed produces.
  const payload = {
    engineSha,
    maxSteps: args.maxSteps,
    mctsCPuct: args.mctsCPuct,
    mctsCollapseMaxSteps: args.mctsCollapseMaxSteps,
    mctsLeaf: args.mctsLeaf,
    mctsMaxNodes: args.mctsMaxNodes,
    mctsPrior: args.mctsPrior,
    mctsRolloutCrnSamples: args.mctsRolloutCrnSamples,
    mctsRolloutSteps: args.mctsRolloutSteps,
    mctsSimulations: args.mctsSimulations,
    modelUrl: args.modelUrl,
    temperatureMoves: args.temperatureMoves,
    temperatureValue: args.temperatureValue,
    traceVersion: args.traceVersion,
  };
  return createHash("sha256").update(JSON.stringify(payload)).digest("hex");
}

function mctsConfigFromArgs(args: RecorderArgs): MctsConfig {
  return defaultMctsConfig({
    simulations: args.mctsSimulations,
    cPuct: args.mctsCPuct,
    leaf: args.mctsLeaf,
    rolloutCrnSamples: args.mctsRolloutCrnSamples,
    rolloutSteps: args.mctsRolloutSteps,
    prior: args.mctsPrior,
    addRootDirichlet: false, // golden traces deliberately disable Dirichlet
    collapseMaxSteps: args.mctsCollapseMaxSteps,
    maxNodes: args.mctsMaxNodes,
    adaptiveRatio: 0,
    adaptiveMinSims: args.mctsSimulations,
  });
}

export async function recordTraceForSeed(
  seed: number,
  args: RecorderArgs,
  configHash: string,
): Promise<TraceLine> {
  const counter = createRngCounter();
  const rawRng = createSeededRng(`${seed}:selfplay`, "selfplay");
  const rng = instrumentRng(rawRng, counter);
  const mctsConfig = mctsConfigFromArgs(args);

  return withRng(rng, async () => {
    let state: GameState = setupAiVsAiGame();
    const actions: ActionRecord[] = [];
    let terminalReason: TraceLine["terminalReason"] = "maxSteps";

    for (let step = 0; step < args.maxSteps; step += 1) {
      if (state.gameOver) {
        terminalReason = "gameOver";
        break;
      }
      const before = stateHash(state);
      const sideId: SideId = state.currentSide === "player" || state.currentSide === "opponent" ? state.currentSide : "player";
      const legalActions = enumerateLegalAiActions(state, sideId);
      const forcedCoins = getForcedAttackCoinResults(state, rng);

      // Drain the counter so it's about-to-cover this step's draws only.
      // We harvest counter once at the very end of the step. The pre-step
      // setupAiVsAiGame draws (opening coin + opening hands) are attributed
      // to step 0's count, which is the natural place: it's the first
      // "Δrng during this step" the per-game stream sees.
      let selectedIndex = -1;
      let actionPicked: LegalAiAction | null = null;

      if (legalActions.length <= 1) {
        // Single-action shortcut: still record the (forced) action and step
        // forward. This is what `mctsSelfPlay.ts` calls the
        // "no choice to distill" branch, but the trace still needs it for
        // bit-identity since it advances state and consumes rng draws.
        const action = legalActions[0];
        if (action) {
          const next = advanceModeledTurnStep(state, sideId, action, forcedCoins, rng);
          state = stateHash(next) === before
            ? (sideId === "player"
                ? advancePlayerAiTurnStep(state, forcedCoins, rng.next)
                : advanceOpponentTurnStep(state, forcedCoins, rng.next))
            : next;
          actionPicked = action;
          selectedIndex = 0;
        } else {
          state = sideId === "player"
            ? advancePlayerAiTurnStep(state, forcedCoins, rng.next)
            : advanceOpponentTurnStep(state, forcedCoins, rng.next);
          // No action chosen — record a sentinel pseudo-action.
          actionPicked = null;
          selectedIndex = -1;
        }
      } else {
        const mctsResult: MctsResult = await runMcts(
          state,
          sideId,
          mctsConfig,
          args.modelUrl,
          `${seed}:${sideId}:${step}:mcts`,
        );
        const totalVisits = mctsResult.visits.reduce((sum, n) => sum + n, 0);
        const visitDistribution = totalVisits > 0
          ? mctsResult.visits.map((n) => n / totalVisits)
          : legalActions.map(() => 1 / legalActions.length);

        // Temperature schedule mirroring `mctsSelfPlay.ts`. For golden
        // traces we still want to exercise the temperature-sampled branch
        // because that's what a Rust port would have to reproduce.
        const useTemperature = step < args.temperatureMoves * 2;
        const temperature = useTemperature ? args.temperatureValue : 0;
        selectedIndex = pickFromVisits(mctsResult, visitDistribution, temperature, rng);

        const action = legalActions[selectedIndex] ?? legalActions[0]!;
        const next = advanceModeledTurnStep(state, sideId, action, forcedCoins, rng);
        if (stateHash(next) === before) {
          state = sideId === "player"
            ? advancePlayerAiTurnStep(state, forcedCoins, rng.next)
            : advanceOpponentTurnStep(state, forcedCoins, rng.next);
        } else {
          state = next;
        }
        actionPicked = action;
      }

      const rngDrawsThisStep = counter.takeAndReset();
      const actionRecord: ActionRecord = {
        step,
        turnNumber: state.turnNumber,
        sideId,
        fingerprintBefore: before,
        action: {
          id: actionPicked?.id ?? "__none__",
          kind: actionPicked?.kind ?? "__none__",
          phase: actionPicked?.phase ?? "__none__",
          payload: actionPicked?.payload ?? {},
          selectedIndex,
          legalCount: legalActions.length,
        },
        rngDrawsThisStep,
      };
      actions.push(actionRecord);

      if (stateHash(state) === before) {
        terminalReason = "stalled";
        break;
      }
    }

    if (state.gameOver) terminalReason = "gameOver";

    const totalRngDraws = actions.reduce((sum, a) => sum + a.rngDrawsThisStep, 0);
    const trace: TraceLine = {
      seed,
      traceVersion: args.traceVersion,
      configHash,
      actions,
      terminalFingerprint: stateFingerprint(state),
      winner: state.winner,
      turnNumber: state.turnNumber,
      totalRngDraws,
      terminalReason,
    };
    return trace;
  });
}

function pickFromVisits(
  result: MctsResult,
  visitDistribution: number[],
  temperature: number,
  rng: { next: () => number },
): number {
  if (temperature <= 0 || visitDistribution.length <= 1) {
    return result.selectedIndex;
  }
  const weights = result.visits.map((n) => Math.pow(Math.max(0, n), 1 / temperature));
  const total = weights.reduce((sum, w) => sum + w, 0);
  if (total <= 0) return result.selectedIndex;
  const r = rng.next() * total;
  let cum = 0;
  for (let i = 0; i < weights.length; i += 1) {
    cum += weights[i]!;
    if (r <= cum) return i;
  }
  void visitDistribution;
  return weights.length - 1;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  if (args.mctsPrior === "policy" && !args.modelUrl) {
    process.stderr.write("ERROR: --prior policy requires --model-url <serve_onnx URL>\n");
    process.exit(2);
  }
  const engineSha = engineGitSha();
  const configHash = computeConfigHash(args, engineSha);

  mkdirSync(dirname(args.outPath), { recursive: true });
  writeFileSync(args.outPath, "", "utf8");

  const startedAt = Date.now();
  process.stderr.write(
    `[record-golden-traces] start seeds=${args.seeds} seed-base=${args.seedBase} ` +
    `sims=${args.mctsSimulations} leaf=${args.mctsLeaf} prior=${args.mctsPrior} ` +
    `out=${args.outPath} configHash=${configHash.slice(0, 12)}\n`,
  );

  for (let i = 0; i < args.seeds; i += 1) {
    const seed = args.seedBase + i;
    const trace = await recordTraceForSeed(seed, args, configHash);
    appendFileSync(args.outPath, JSON.stringify(trace) + "\n", "utf8");
    if ((i + 1) % 50 === 0 || i === args.seeds - 1) {
      const elapsed = (Date.now() - startedAt) / 1000;
      process.stderr.write(
        `[record-golden-traces] ${i + 1}/${args.seeds} elapsed=${elapsed.toFixed(1)}s ` +
        `(last seed=${seed} winner=${trace.winner ?? "none"} steps=${trace.actions.length} ` +
        `rng=${trace.totalRngDraws})\n`,
      );
    }
  }

  const elapsed = (Date.now() - startedAt) / 1000;
  process.stderr.write(
    `[record-golden-traces] done seeds=${args.seeds} elapsed=${elapsed.toFixed(1)}s out=${args.outPath}\n`,
  );
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
