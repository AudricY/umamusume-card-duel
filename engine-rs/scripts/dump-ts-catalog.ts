// Dump the TS catalog's expanded card-id set + per-card identity fields
// for the Rust catalog parity test.
//
// Run from repo root:
//   npx tsx engine-rs/scripts/dump-ts-catalog.ts > engine-rs/crates/engine/tests/catalog-reference.json

import { cards } from "../../shared/src/gameData";

type CardSummary = {
  id: string;
  kind: "umamusume" | "trainer";
  stage?: number;
  hp?: number;
  type?: string;
  trainerType?: string;
};

const summaries: CardSummary[] = Object.entries(cards)
  .map(([id, card]) => {
    if (card.kind === "umamusume") {
      return {
        id,
        kind: "umamusume" as const,
        stage: card.stage,
        hp: card.hp,
        type: card.type,
      };
    }
    return {
      id,
      kind: "trainer" as const,
      trainerType: card.trainerType,
    };
  })
  // Sort by id for stable comparison — the runtime order is insertion order
  // which is what both implementations preserve, but for the test we only
  // care about set equality + per-id-field equality.
  .sort((a, b) => a.id.localeCompare(b.id));

process.stdout.write(JSON.stringify({ count: summaries.length, cards: summaries }, null, 2));
