import { mkdirSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { dirname } from "node:path";
import { allCards } from "../../../shared/src/gameData";
import { toBaseCardId } from "../../../shared/src/cardRarity";

type VocabArgs = {
  out: string;
  pretty: boolean;
};

const SUFFIX_TOKENS = ["UncommonPlus", "FullArtGold", "FullArt", "Ex"] as const;

function suffixOf(cardId: string): string {
  for (const suffix of SUFFIX_TOKENS) {
    if (cardId.endsWith(suffix)) return suffix;
  }
  return "";
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const allIds = Object.keys(allCards);
  // Group by base id; sort base ids alphabetically; within a base id sort
  // suffix variants in a fixed order so the index assignment is deterministic
  // across machines and runs.
  const baseToVariants = new Map<string, string[]>();
  for (const id of allIds) {
    const base = toBaseCardId(id);
    const list = baseToVariants.get(base) ?? [];
    list.push(id);
    baseToVariants.set(base, list);
  }
  const sortedBases = [...baseToVariants.keys()].sort();
  const suffixOrder: Record<string, number> = { "": 0, UncommonPlus: 1, FullArt: 2, FullArtGold: 3, Ex: 4 };

  const orderedIds: string[] = [];
  for (const base of sortedBases) {
    const variants = (baseToVariants.get(base) ?? []).sort((left, right) => {
      const lp = suffixOrder[suffixOf(left)] ?? 99;
      const rp = suffixOrder[suffixOf(right)] ?? 99;
      if (lp !== rp) return lp - rp;
      return left.localeCompare(right);
    });
    orderedIds.push(...variants);
  }

  const indexById: Record<string, number> = {};
  orderedIds.forEach((id, index) => {
    indexById[id] = index;
  });

  // Reserve the 0th slot for unknown / out-of-vocab; shift everyone else by 1.
  const indexByIdWithReserved: Record<string, number> = {};
  Object.entries(indexById).forEach(([id, idx]) => {
    indexByIdWithReserved[id] = idx + 1;
  });

  const baseIdByVariant: Record<string, string> = {};
  for (const id of orderedIds) baseIdByVariant[id] = toBaseCardId(id);

  const vocab = {
    schemaVersion: 1,
    unknownIndex: 0,
    vocabSize: orderedIds.length + 1,
    indexById: indexByIdWithReserved,
    baseIdByVariant,
    orderedIds,
    suffixOrder,
  };

  const serialized = args.pretty ? JSON.stringify(vocab, null, 2) : JSON.stringify(vocab);
  const hash = createHash("sha256").update(serialized).digest("hex").slice(0, 16);
  const out = { ...vocab, hash };

  mkdirSync(dirname(args.out), { recursive: true });
  writeFileSync(args.out, JSON.stringify(out, null, args.pretty ? 2 : 0) + "\n", "utf8");

  console.log(JSON.stringify({
    status: "PASS",
    out: args.out,
    vocabSize: out.vocabSize,
    hash: out.hash,
    sampleIds: orderedIds.slice(0, 5),
  }, null, 2));
}

function parseArgs(argv: string[]): VocabArgs {
  const get = (name: string, fallback: string) => {
    const index = argv.indexOf(name);
    return index >= 0 ? argv[index + 1] ?? fallback : fallback;
  };
  return {
    out: get("--out", "shared/src/cardVocab.json"),
    pretty: !argv.includes("--compact"),
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
