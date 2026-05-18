import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { withGitMetadata } from "../manifest";

type BehaviorPolicySnapshot = {
  kind: string;
  temperature: number;
  actionLogProbs: number[];
  actionProbs?: number[];
  selectedLogProb: number | null;
  valueEstimate?: number;
};

// R7 step 2: teacher labels on a trace row are now a *list* (length 1–3). The
// singular `teacher` field is gone — single-teacher recipes emit a length-1
// `teachers` array so back-compat is structural rather than flag-driven.
type TeacherEntry = {
  selection: "rollout" | "search" | "planner";
  selectedActionId: string;
  selectedActionIndex: number;
  selectedOriginalRank?: number;
};

type DecisionTraceRow = {
  schemaVersion: 1;
  source: "model-visited";
  seed: string;
  modelSide: "player" | "opponent";
  step: number;
  sideId: "player" | "opponent";
  selection: string;
  observation: { opponent: { handCardIds?: unknown; deck?: unknown; hand?: unknown } } & Record<string, unknown>;
  legalActions: Array<{ id: string }>;
  selectedActionId: string;
  selectedActionIndex: number;
  selectedOriginalRank?: number;
  heuristicSelectedActionId: string;
  heuristicSelectedActionIndex: number;
  fallback: boolean;
  behaviorPolicy?: BehaviorPolicySnapshot;
  teachers?: TeacherEntry[];
  // R16-TD 3a: online rollout-leaf MCTS relabel payload emitted by
  // evaluateModelVsHeuristic.ts under --relabel-mcts. When present the
  // --label-source rollout-leaf-mcts branch forwards these directly instead
  // of recomputing a mixture from `teachers`.
  policyTargets?: number[];
  oracle?: Record<string, unknown>;
  stateSource?: string;
  result: { winner: "player" | "opponent" | null; modelWon: boolean; points: Record<string, number>; terminalReason: string } | null;
};

type RelabelArgs = {
  in: string;
  out: string;
  source: string;
  labelSource: string;
  manifestOut: string | null;
  dropFallbacks: boolean;
};

function main() {
  const args = parseArgs(process.argv.slice(2));
  const rows = readJsonl<DecisionTraceRow>(args.in);
  let kept = 0;
  let skippedNoTeacher = 0;
  let skippedFallback = 0;
  let outOfRange = 0;
  let leakDetected = 0;
  const teacherByName: Record<string, number> = {};
  // R7 step 2 telemetry: count rows by their per-row teacher-count so the
  // operator can confirm at a glance whether a corpus is single-teacher
  // (length-1 back-compat) or multi-teacher (length 2–3).
  const teachersPerRowHistogram: Record<string, number> = {};

  mkdirSync(dirname(args.out), { recursive: true });
  const out = [] as string[];

  // R16-TD 3a: rollout-leaf-mcts pass-through. The evaluator already computed
  // the soft target online on the live GameState; forward policyTargets/oracle
  // verbatim and reuse the SAME fallback/leak/range/sum gates (no new audit
  // logic, no mixture recomputation from teachers).
  const mctsPassThrough = args.labelSource === "rollout-leaf-mcts";

  for (const row of rows) {
    const teachers = row.teachers ?? [];
    if (!mctsPassThrough && teachers.length === 0) {
      skippedNoTeacher += 1;
      continue;
    }
    if (args.dropFallbacks && row.fallback) {
      skippedFallback += 1;
      continue;
    }
    const numActions = row.legalActions.length;
    if (mctsPassThrough) {
      const targets = row.policyTargets ?? [];
      if (targets.length !== numActions) {
        outOfRange += 1;
        continue;
      }
      const leakFieldsMcts = detectLeakFields(row);
      if (leakFieldsMcts.length > 0) {
        leakDetected += 1;
        throw new Error(`Hidden-info leak in row seed=${row.seed} step=${row.step}: ${leakFieldsMcts.join(",")}`);
      }
      const sumMcts = targets.reduce((acc, v) => acc + v, 0);
      if (!(Math.abs(sumMcts - 1.0) < 1e-6)) {
        outOfRange += 1;
        continue;
      }
      let argmaxIdx = 0;
      let argmaxVal = targets[0]!;
      for (let i = 1; i < targets.length; i += 1) {
        if (targets[i]! > argmaxVal) {
          argmaxVal = targets[i]!;
          argmaxIdx = i;
        }
      }
      const argmaxAct = row.legalActions[argmaxIdx]!;
      const res = row.result ?? { winner: null, modelWon: false, points: { player: 0, opponent: 0 }, terminalReason: "unknown" };
      const mctsRow: Record<string, unknown> = {
        schemaVersion: 1,
        source: args.source,
        labelSource: args.labelSource,
        stateSource: row.stateSource ?? null,
        relabeledFrom: { source: row.source, selection: row.selection },
        episodeId: `${row.seed}:${row.modelSide}`,
        seed: row.seed,
        modelSide: row.modelSide,
        step: row.step,
        sideId: row.sideId,
        observation: row.observation,
        legalActions: row.legalActions,
        selectedActionId: argmaxAct.id,
        selectedActionIndex: argmaxIdx,
        policyTargets: targets,
        oracle: row.oracle ?? null,
        heuristicSelectedActionId: row.heuristicSelectedActionId,
        heuristicSelectedActionIndex: row.heuristicSelectedActionIndex,
        modelChoseActionId: row.selectedActionId,
        modelChoseActionIndex: row.selectedActionIndex,
        modelFallback: row.fallback,
        result: { winner: res.winner, points: res.points, terminalReason: res.terminalReason },
      };
      if (row.behaviorPolicy) mctsRow.behaviorPolicy = row.behaviorPolicy;
      out.push(JSON.stringify(mctsRow));
      kept += 1;
      teacherByName["rollout-leaf-mcts"] = (teacherByName["rollout-leaf-mcts"] ?? 0) + 1;
      teachersPerRowHistogram["mcts"] = (teachersPerRowHistogram["mcts"] ?? 0) + 1;
      continue;
    }
    // Validate every teacher's chosen index lies inside legalActions. A single
    // bad teacher invalidates the entire row — the mixture target is only
    // well-defined when all teachers vote in-range.
    const anyOutOfRange = teachers.some((t) => t.selectedActionIndex < 0 || t.selectedActionIndex >= numActions);
    if (anyOutOfRange) {
      outOfRange += 1;
      continue;
    }
    const leakFields = detectLeakFields(row);
    if (leakFields.length > 0) {
      leakDetected += 1;
      throw new Error(`Hidden-info leak in row seed=${row.seed} step=${row.step}: ${leakFields.join(",")}`);
    }
    teachers.forEach((t) => {
      teacherByName[t.selection] = (teacherByName[t.selection] ?? 0) + 1;
    });
    const key = String(teachers.length);
    teachersPerRowHistogram[key] = (teachersPerRowHistogram[key] ?? 0) + 1;
    // R7 step 2: build the per-state mixture target. Uniform 1/K weights for
    // v1 (matches scoping doc § 3 "Trace-schema change"). Each teacher
    // contributes 1/K to its chosen action's bucket; ties on the same action
    // sum correctly, so a length-3 list where two teachers agree puts 2/3 on
    // one action and 1/3 on the other.
    const policyTargets = new Array<number>(numActions).fill(0);
    const weight = 1 / teachers.length;
    for (const t of teachers) {
      policyTargets[t.selectedActionIndex]! += weight;
    }
    // Argmax of the mixture distribution is the hard label kept for telemetry
    // and for downstream consumers that only want a single action (the
    // existing accuracy accumulator path). Ties broken by lowest index, which
    // matches numpy.argmax's deterministic behavior in selfplay_dataset.py.
    let argmaxIndex = 0;
    let argmaxValue = policyTargets[0]!;
    for (let i = 1; i < policyTargets.length; i += 1) {
      if (policyTargets[i]! > argmaxValue) {
        argmaxValue = policyTargets[i]!;
        argmaxIndex = i;
      }
    }
    const argmaxAction = row.legalActions[argmaxIndex]!;
    const result = row.result ?? { winner: null, modelWon: false, points: { player: 0, opponent: 0 }, terminalReason: "unknown" };
    const trainingRow: Record<string, unknown> = {
      schemaVersion: 1,
      source: args.source,
      labelSource: args.labelSource,
      relabeledFrom: { source: row.source, selection: row.selection },
      // R7 step 2: forward the full teacher list and the per-row mixture
      // selections so the dataset loader (step 3 of the execution plan) can
      // build padded policy_targets tensors and so the trainer reporting can
      // break down loss by teacher.
      teachers,
      teacherSelections: teachers.map((t) => t.selection),
      episodeId: `${row.seed}:${row.modelSide}`,
      seed: row.seed,
      modelSide: row.modelSide,
      step: row.step,
      sideId: row.sideId,
      observation: row.observation,
      legalActions: row.legalActions,
      // selectedActionId / selectedActionIndex point at the *argmax of the
      // mixture distribution* so legacy consumers that only read the hard
      // label still get a sensible target. The soft target is in
      // policyTargets below.
      selectedActionId: argmaxAction.id,
      selectedActionIndex: argmaxIndex,
      policyTargets,
      heuristicSelectedActionId: row.heuristicSelectedActionId,
      heuristicSelectedActionIndex: row.heuristicSelectedActionIndex,
      modelChoseActionId: row.selectedActionId,
      modelChoseActionIndex: row.selectedActionIndex,
      modelFallback: row.fallback,
      result: { winner: result.winner, points: result.points, terminalReason: result.terminalReason },
    };
    // Item 18: forward the behavior-policy snapshot from the trace row into
    // the relabeled training row so PPO can recover importance ratios on
    // the warm-start data. Only present when the underlying selection was
    // policy/value (i.e. the model server was consulted at decision time).
    if (row.behaviorPolicy) trainingRow.behaviorPolicy = row.behaviorPolicy;
    out.push(JSON.stringify(trainingRow));
    kept += 1;
  }

  writeFileSync(args.out, out.join("\n") + (out.length ? "\n" : ""), "utf8");

  const summary = {
    inputRows: rows.length,
    kept,
    skippedNoTeacher,
    skippedFallback,
    outOfRange,
    leakDetected,
    teacherSelections: teacherByName,
    teachersPerRowHistogram,
    inputPath: args.in,
    outputPath: args.out,
  };

  if (args.manifestOut) {
    mkdirSync(dirname(args.manifestOut), { recursive: true });
    writeFileSync(args.manifestOut, JSON.stringify(withGitMetadata({ args, summary }), null, 2) + "\n", "utf8");
  }

  console.log(JSON.stringify({ status: "PASS", ...summary }, null, 2));
}

function detectLeakFields(row: DecisionTraceRow): string[] {
  const violations: string[] = [];
  const opponent = row.observation?.opponent ?? {};
  if ((opponent as { handCardIds?: unknown }).handCardIds !== undefined) violations.push("opponent.handCardIds");
  if ((opponent as { hand?: unknown }).hand !== undefined) violations.push("opponent.hand");
  if ((opponent as { deck?: unknown }).deck !== undefined) violations.push("opponent.deck");
  return violations;
}

function readJsonl<T>(path: string): T[] {
  const raw = readFileSync(path, "utf8").trim();
  if (!raw) return [];
  return raw.split("\n").filter(Boolean).map((line) => JSON.parse(line) as T);
}

function parseArgs(argv: string[]): RelabelArgs {
  const get = (name: string, fallback: string) => {
    const index = argv.indexOf(name);
    return index >= 0 ? argv[index + 1] ?? fallback : fallback;
  };
  return {
    in: get("--in", ""),
    out: get("--out", ""),
    source: get("--source", "model-visited-relabeled"),
    labelSource: get("--label-source", "teacher-relabeled"),
    manifestOut: get("--manifest-out", "") || null,
    dropFallbacks: !argv.includes("--keep-fallbacks"),
  };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    main();
  } catch (error) {
    console.error(error);
    process.exit(1);
  }
}
