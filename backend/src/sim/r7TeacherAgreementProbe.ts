// R7 pre-flight teacher-agreement probe.
//
// Step 1 of the R7 multi-teacher BC blend execution plan (see
// `docs/ai-research/scoping/r7-multi-teacher-warmstart.md` § 6).
//
// Hypothesis under test: the three trace-teachers (rollout-CRN, search,
// planner) make different mistakes per state, so a per-state mixture target
// is a meaningfully different SL target than any single teacher. We probe
// pairwise argmax-agreement + mixture-entropy on a small corpus of
// rule-bot-visited states. If teachers agree too much (mixture collapses to
// single-teacher) OR too little (search/planner are mixing in junk against
// rollout), R7's diversity assumption fails and the schema change is wasted
// compute.
//
// Read-only against the existing teacher selectors. Does NOT touch the
// trace schema, relabel pass, dataset loader, or training pipeline. Spends
// no model-server calls — all three selectors are heuristic / rollout-CRN
// and need only the game engine.

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

import {
  type EvaluateModelArgs,
  advanceModeledTurnStep,
  chooseRolloutAction,
  chooseSearchAction,
  choosePlannerAction,
  getForcedAttackCoinResults,
  setupAiVsAiGame,
  stateHash,
} from "./evaluateModelVsHeuristic";
import { createSeededRng, withRng, type Rng } from "../../../frontend/src/game/engine/core/random";
import { enumerateLegalAiActions, chooseHighestScoredAction } from "../../../frontend/src/game/engine/ai-policy/actions";
import { advanceOpponentTurnStep, advancePlayerAiTurnStep } from "../../../frontend/src/game/engine";
import type { GameState, SideId } from "../../../shared/src/types";

type ProbeArgs = {
  n: number;
  seedStart: number;
  maxStepsPerGame: number;
  out: string;
};

type PerStateRecord = {
  gameIndex: number;
  step: number;
  modelSide: SideId;
  legalActions: number;
  rolloutIndex: number;
  searchIndex: number;
  plannerIndex: number;
  rolloutMs: number;
  searchMs: number;
  plannerMs: number;
  agree_rollout_search: boolean;
  agree_rollout_planner: boolean;
  agree_search_planner: boolean;
  allThreeAgree: boolean;
  allThreeDisagree: boolean;
  mixtureEntropyNats: number;
  mixtureEntropyNormalised: number;
};

function parseArgs(argv: string[]): ProbeArgs {
  const get = (name: string, fallback: string) => {
    const index = argv.indexOf(name);
    return index >= 0 ? argv[index + 1] ?? fallback : fallback;
  };
  return {
    n: Number(get("--n", "300")),
    seedStart: Number(get("--seed-start", "70000")),
    maxStepsPerGame: Number(get("--max-steps-per-game", "120")),
    out: get("--out", ""),
  };
}

// Minimal-surface `EvaluateModelArgs` used purely to satisfy the three
// selectors' function signatures. modelUrl is never dereferenced by any of
// the three selectors (all three are heuristic / rollout-CRN, no /predict
// calls). Hyperparameters match `parseArgs` defaults in
// `evaluateModelVsHeuristic.ts` so the agreement number is honest against
// the trace-gen recipe R7 § 3 commits to.
function buildSelectorArgs(): EvaluateModelArgs {
  return {
    modelUrl: "http://127.0.0.1:0",
    games: 0,
    seedStart: 0,
    maxSteps: 500,
    modelSide: "both",
    details: false,
    selection: "rollout",
    mctsSimulations: 100,
    mctsCPuct: 1.5,
    mctsLeaf: "value-head",
    mctsRolloutCrnSamples: 3,
    mctsRolloutSteps: 200,
    mctsCollapseMaxSteps: 64,
    mctsMaxNodes: 5000,
    mctsPrior: "uniform",
    mctsRootDirichlet: false,
    mctsDirichletAlpha: 0.3,
    mctsDirichletEpsilon: 0.25,
    mctsAdaptiveRatio: 0,
    mctsAdaptiveMinSims: 20,
    progressOut: null,
    cycleWindow: 8,
    cycleMinVisits: 3,
    plannerCrnSamples: 3,
    plannerLeafAggregate: "mean",
    plannerFirstActionAggregate: "max",
    rolloutCrnSamples: 1,
    rolloutSteps: 500,
    searchDepth: 2,
    searchTopK: 4,
    searchSamples: 1,
    ranker: "heuristic",
    decisionTraceOut: null,
    manifestOut: null,
    traceTeacher: [],
    plannerTopK: 4,
    plannerMaxSequences: 64,
    plannerMaxDepth: 8,
    opponentModelUrl: null,
    opponentSelection: "rule",
    opponentMctsSimulations: 100,
    opponentMctsCPuct: 1.5,
    opponentMctsLeaf: "value-head",
    opponentMctsRolloutCrnSamples: 3,
    opponentMctsRolloutSteps: 200,
    opponentMctsCollapseMaxSteps: 64,
    opponentMctsMaxNodes: 5000,
    opponentMctsPrior: "uniform",
    opponentMctsAdaptiveRatio: 0,
    opponentMctsAdaptiveMinSims: 20,
  };
}

// One-hot mixture (uniform 1/3 weights) → entropy in nats and normalised to
// [0, 1] by dividing by log(legalActions.length). Matches the R7 v1 design
// (uniform weights) at scoping § 3.
function mixtureEntropy(indices: number[], legalActions: number): { nats: number; normalised: number } {
  const counts = new Map<number, number>();
  indices.forEach((idx) => counts.set(idx, (counts.get(idx) ?? 0) + 1));
  const total = indices.length;
  let nats = 0;
  counts.forEach((c) => {
    const p = c / total;
    if (p > 0) nats -= p * Math.log(p);
  });
  const maxNats = legalActions > 1 ? Math.log(legalActions) : 0;
  const normalised = maxNats > 0 ? nats / maxNats : 0;
  return { nats, normalised };
}

function runOneGame(
  selectorArgs: EvaluateModelArgs,
  modelSide: SideId,
  gameIndex: number,
  seedStart: number,
  maxStepsPerGame: number,
  budgetRemaining: number,
): { records: PerStateRecord[] } {
  const seed = String(seedStart + gameIndex);
  const driverRng = createSeededRng(`${seed}:${modelSide}`, "r7-probe-driver");
  return withRng(driverRng, () => runOneGameInner(selectorArgs, modelSide, gameIndex, seed, driverRng, maxStepsPerGame, budgetRemaining));
}

function runOneGameInner(
  selectorArgs: EvaluateModelArgs,
  modelSide: SideId,
  gameIndex: number,
  seed: string,
  driverRng: Rng,
  maxStepsPerGame: number,
  budgetRemaining: number,
): { records: PerStateRecord[] } {
  const records: PerStateRecord[] = [];
  let state: GameState = setupAiVsAiGame();
  for (let step = 0; step < maxStepsPerGame; step += 1) {
    if (state.gameOver) break;
    if (records.length >= budgetRemaining) break;
    const beforeHash = stateHash(state);
    const sideId: SideId = state.currentSide === "player" || state.currentSide === "opponent" ? state.currentSide : "player";
    const forcedCoinResults = getForcedAttackCoinResults(state, driverRng);
    if (sideId === modelSide) {
      const legalActions = enumerateLegalAiActions(state, sideId);
      if (legalActions.length > 1) {
        // Probe all three teachers on this state. Each teacher gets its own
        // sub-seed so stochastic rollouts/searches are reproducible *and*
        // independent across teachers — matches the per-teacher seed recipe
        // sketched in R7 § 3 "Mixed-corpus generation recipe".
        const baseSeed = `${seed}:${modelSide}:${step}:trace-teacher`;
        const t0 = Date.now();
        const rolloutFallback = createSeededRng(`${baseSeed}:rollout-fallback`, "r7-probe-rollout-fallback");
        const rollout = chooseRolloutAction(selectorArgs, state, sideId, `${baseSeed}:rollout`, rolloutFallback);
        const t1 = Date.now();
        const search = chooseSearchAction(selectorArgs, state, sideId, `${baseSeed}:search`);
        const t2 = Date.now();
        const planner = choosePlannerAction(selectorArgs, state, sideId, `${baseSeed}:planner`);
        const t3 = Date.now();
        const indices = [rollout.selectedIndex, search.selectedIndex, planner.selectedIndex];
        const { nats, normalised } = mixtureEntropy(indices, legalActions.length);
        const distinct = new Set(indices);
        records.push({
          gameIndex,
          step,
          modelSide,
          legalActions: legalActions.length,
          rolloutIndex: rollout.selectedIndex,
          searchIndex: search.selectedIndex,
          plannerIndex: planner.selectedIndex,
          rolloutMs: t1 - t0,
          searchMs: t2 - t1,
          plannerMs: t3 - t2,
          agree_rollout_search: rollout.selectedIndex === search.selectedIndex,
          agree_rollout_planner: rollout.selectedIndex === planner.selectedIndex,
          agree_search_planner: search.selectedIndex === planner.selectedIndex,
          allThreeAgree: distinct.size === 1,
          allThreeDisagree: distinct.size === 3,
          mixtureEntropyNats: nats,
          mixtureEntropyNormalised: normalised,
        });
      }
    }
    // Advance state with the rule bot regardless of side. Using the rule
    // bot for both sides means the state distribution we sample from is
    // "states a rule-bot game visits" — slightly different from R15.S1's
    // rollout-led trace-gen distribution, but cheap and faithful enough for
    // a pre-flight diversity check. (The rollout/search/planner question is
    // about argmax disagreement *given a state*; the state distribution
    // matters less than the per-state spread.)
    const next = sideId === "player"
      ? advancePlayerAiTurnStep(state, forcedCoinResults, driverRng.next)
      : advanceOpponentTurnStep(state, forcedCoinResults, driverRng.next);
    if (stateHash(next) === beforeHash) {
      // Stalled — bail to avoid infinite loop on degenerate states.
      // (Mirrors the `stalled` short-circuit in
      // runModelVsHeuristicGameWithRng.)
      break;
    }
    state = next;
    // touch chooseHighestScoredAction once per game so tree-shaking doesn't
    // drop it from the dev bundle (defensive; not load-bearing).
    void chooseHighestScoredAction;
  }
  return { records };
}

function aggregate(records: PerStateRecord[]) {
  const n = records.length;
  if (n === 0) {
    return {
      n: 0,
      pairwise_agreement: { rollout_search: 0, rollout_planner: 0, search_planner: 0 },
      all_three_agree_freq: 0,
      all_three_disagree_freq: 0,
      mixture_entropy_normalised: { min: 0, p25: 0, mean: 0, p50: 0, p75: 0, max: 0 },
      legal_actions: { min: 0, mean: 0, max: 0 },
      teacher_wall_clock_ms: { rollout_mean: 0, search_mean: 0, planner_mean: 0 },
    };
  }
  const agreeRS = records.filter((r) => r.agree_rollout_search).length / n;
  const agreeRP = records.filter((r) => r.agree_rollout_planner).length / n;
  const agreeSP = records.filter((r) => r.agree_search_planner).length / n;
  const allAgree = records.filter((r) => r.allThreeAgree).length / n;
  const allDisagree = records.filter((r) => r.allThreeDisagree).length / n;
  const entropies = records.map((r) => r.mixtureEntropyNormalised).sort((a, b) => a - b);
  const legalActions = records.map((r) => r.legalActions).sort((a, b) => a - b);
  const meanLA = legalActions.reduce((s, v) => s + v, 0) / n;
  const pct = (sorted: number[], q: number) => {
    if (sorted.length === 0) return 0;
    const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor(q * sorted.length)));
    return sorted[idx]!;
  };
  const meanE = entropies.reduce((s, v) => s + v, 0) / n;
  const rolloutMs = records.reduce((s, r) => s + r.rolloutMs, 0) / n;
  const searchMs = records.reduce((s, r) => s + r.searchMs, 0) / n;
  const plannerMs = records.reduce((s, r) => s + r.plannerMs, 0) / n;
  return {
    n,
    pairwise_agreement: { rollout_search: agreeRS, rollout_planner: agreeRP, search_planner: agreeSP },
    all_three_agree_freq: allAgree,
    all_three_disagree_freq: allDisagree,
    mixture_entropy_normalised: {
      min: entropies[0]!,
      p25: pct(entropies, 0.25),
      mean: meanE,
      p50: pct(entropies, 0.50),
      p75: pct(entropies, 0.75),
      max: entropies[entropies.length - 1]!,
    },
    legal_actions: {
      min: legalActions[0]!,
      mean: meanLA,
      max: legalActions[legalActions.length - 1]!,
    },
    teacher_wall_clock_ms: {
      rollout_mean: rolloutMs,
      search_mean: searchMs,
      planner_mean: plannerMs,
    },
  };
}

// Decision rules from the brief / scoping § 5 risks (a) and (b).
function verdict(aggregates: ReturnType<typeof aggregate>): { decision: "GO" | "NO-GO-reweight" | "NO-GO-close"; rule: string } {
  const rs = aggregates.pairwise_agreement.rollout_search;
  const rp = aggregates.pairwise_agreement.rollout_planner;
  const allAgree = aggregates.all_three_agree_freq;
  const meanE = aggregates.mixture_entropy_normalised.mean;
  // NO-GO-close: teachers collapse to ~same actions.
  if (allAgree >= 0.50 && meanE < 0.10) {
    return {
      decision: "NO-GO-close",
      rule: `all_three_agree_freq=${allAgree.toFixed(3)} >= 0.50 AND mean_mixture_entropy_normalised=${meanE.toFixed(3)} < 0.10 — teachers converge to similar actions; mixture target collapses to single-teacher. Close R7; R8 (DPO) becomes natural next move.`,
    };
  }
  // NO-GO-reweight: rollout disagrees with both search and planner heavily.
  if (rs < 0.30 && rp < 0.30) {
    return {
      decision: "NO-GO-reweight",
      rule: `rollout_search=${rs.toFixed(3)} < 0.30 AND rollout_planner=${rp.toFixed(3)} < 0.30 — search/planner mostly disagree with rollout; risk (a) "mixing in junk" fires. Recommend re-weighting (e.g. 0.6 rollout / 0.2 search / 0.2 planner) before launching v1.`,
    };
  }
  // GO band.
  const rsOk = rs >= 0.30 && rs <= 0.85;
  const rpOk = rp >= 0.30 && rp <= 0.85;
  if ((rsOk || rpOk) && allAgree < 0.50) {
    return {
      decision: "GO",
      rule: `at least one of rollout_search=${rs.toFixed(3)}, rollout_planner=${rp.toFixed(3)} is in [0.30, 0.85] AND all_three_agree_freq=${allAgree.toFixed(3)} < 0.50 — meaningful per-state disagreement exists. Proceed to R7 step 2 (schema change).`,
    };
  }
  // Borderline → default NO-GO-close per brief.
  return {
    decision: "NO-GO-close",
    rule: `borderline: rs=${rs.toFixed(3)}, rp=${rp.toFixed(3)}, allAgree=${allAgree.toFixed(3)}, meanE=${meanE.toFixed(3)} — does not clearly satisfy GO band and does not trigger NO-GO-reweight. Per brief default, treat as NO-GO-close (we want cheap falsification, not a marginal launch).`,
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const selectorArgs = buildSelectorArgs();
  const startedAt = Date.now();
  const records: PerStateRecord[] = [];
  // Alternate modelSide per game to balance the per-side state distribution.
  let gameIndex = 0;
  const sides: SideId[] = ["player", "opponent"];
  while (records.length < args.n) {
    const modelSide = sides[gameIndex % sides.length]!;
    const remaining = args.n - records.length;
    const { records: gameRecords } = runOneGame(
      selectorArgs,
      modelSide,
      gameIndex,
      args.seedStart,
      args.maxStepsPerGame,
      remaining,
    );
    records.push(...gameRecords);
    process.stderr.write(`[r7-probe] game=${gameIndex} side=${modelSide} +${gameRecords.length} states (total=${records.length}/${args.n})\n`);
    gameIndex += 1;
    // Hard cap on games to avoid runaway loops if states-per-game is sparse.
    if (gameIndex > args.n * 4) {
      process.stderr.write(`[r7-probe] WARN: hit gameIndex hard cap (${gameIndex}); stopping at ${records.length}/${args.n} states.\n`);
      break;
    }
  }
  const wallClockMs = Date.now() - startedAt;
  const aggregates = aggregate(records);
  const v = verdict(aggregates);
  const output = {
    probe: "r7-pre-flight-teacher-agreement",
    timestamp: new Date().toISOString(),
    args,
    wall_clock_ms: wallClockMs,
    wall_clock_sec: wallClockMs / 1000,
    games_played: gameIndex,
    aggregates,
    verdict: v,
    selector_hyperparameters: {
      rolloutCrnSamples: selectorArgs.rolloutCrnSamples,
      rolloutSteps: selectorArgs.rolloutSteps,
      searchDepth: selectorArgs.searchDepth,
      searchTopK: selectorArgs.searchTopK,
      searchSamples: selectorArgs.searchSamples,
      plannerCrnSamples: selectorArgs.plannerCrnSamples,
      plannerTopK: selectorArgs.plannerTopK,
      plannerMaxSequences: selectorArgs.plannerMaxSequences,
      plannerMaxDepth: selectorArgs.plannerMaxDepth,
      plannerLeafAggregate: selectorArgs.plannerLeafAggregate,
      plannerFirstActionAggregate: selectorArgs.plannerFirstActionAggregate,
      ranker: selectorArgs.ranker,
    },
    per_state_records: records,
  };
  if (args.out) {
    mkdirSync(dirname(args.out), { recursive: true });
    writeFileSync(args.out, JSON.stringify(output, null, 2) + "\n", "utf8");
    process.stderr.write(`[r7-probe] wrote ${args.out}\n`);
  }
  // stdout: summary block for the implementer / queue update.
  const stdout = {
    probe: output.probe,
    n: aggregates.n,
    wall_clock_sec: output.wall_clock_sec,
    games_played: output.games_played,
    pairwise_agreement: aggregates.pairwise_agreement,
    all_three_agree_freq: aggregates.all_three_agree_freq,
    all_three_disagree_freq: aggregates.all_three_disagree_freq,
    mixture_entropy_normalised_mean: aggregates.mixture_entropy_normalised.mean,
    teacher_wall_clock_ms: aggregates.teacher_wall_clock_ms,
    verdict: v,
  };
  console.log(JSON.stringify(stdout, null, 2));
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
