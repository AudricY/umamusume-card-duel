import type { LegalAiAction } from "../../../frontend/src/game/engine/ai-policy/types";
import type { Rng } from "../../../frontend/src/game/engine/core/random";

export type CandidateRankerMode = "heuristic" | "phase-diverse" | "epsilon";

export type RankedAction = {
  action: LegalAiAction;
  index: number;
  originalRank: number;
};

export function rankLegalActions(
  legalActions: LegalAiAction[],
  options: {
    topK: number;
    mode: CandidateRankerMode;
    baseline?: LegalAiAction;
    rng?: Rng;
    epsilon?: number;
  },
): RankedAction[] {
  const ranked = legalActions
    .map((action, index) => ({ action, index }))
    .sort((left, right) => ((right.action.features[0] ?? 0) - (left.action.features[0] ?? 0)) || left.action.id.localeCompare(right.action.id))
    .map((entry, rank) => ({ ...entry, originalRank: rank + 1 }));

  const selected: RankedAction[] = [];
  const add = (entry: RankedAction | undefined) => {
    if (entry && !selected.some((candidate) => candidate.action.id === entry.action.id)) selected.push(entry);
  };

  if (options.mode === "phase-diverse") {
    const groups = new Map<string, RankedAction[]>();
    ranked.forEach((entry) => {
      const key = `${entry.action.phase}:${entry.action.kind}`;
      groups.set(key, [...(groups.get(key) ?? []), entry]);
    });
    Array.from(groups.values()).forEach((entries) => add(entries[0]));
  }

  if (options.mode === "epsilon" && options.rng && (options.epsilon ?? 0.1) > 0) {
    ranked.forEach((entry) => {
      if (selected.length < options.topK && options.rng!.next() < (options.epsilon ?? 0.1)) add(entry);
    });
  }

  ranked.forEach((entry) => {
    if (selected.length < Math.max(1, options.topK)) add(entry);
  });

  add(options.baseline ? ranked.find((entry) => entry.action.id === options.baseline?.id) : undefined);
  ranked.forEach((entry) => {
    if (entry.action.kind === "pass" || entry.action.kind === "endTurn") add(entry);
  });
  return selected;
}
