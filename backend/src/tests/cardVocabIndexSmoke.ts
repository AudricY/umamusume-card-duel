// R7.b.2 Phase 1 parity smoke for `shared/src/cardVocab.ts`.
// Asserts that `cardVocabIndex` mirrors the Python `card_vocab_index`
// at `training/uma_ai/features.py:459-471`:
//
//   - empty / null cardId → 0 (unknownIndex)
//   - direct hit in `indexById` → that integer
//   - suffix fallback in priority order FullArtGold > FullArt > UncommonPlus > Ex
//     (FullArtGold first because "FullArt" is a strict suffix of it)
//   - completely unknown id → 0
//
// Concrete fixtures here come straight from `shared/src/cardVocab.json`
// so a vocab regen surfaces as a typed mismatch rather than silent drift.

import assert from "node:assert/strict";
import { cardVocabIndex, cardVocabSize, cardVocabUnknownIndex } from "../../../shared/src/cardVocab";

const expected: Array<[string | null | undefined, number, string]> = [
  // Empty / null sentinel cases — must hit the `0` early return.
  [null, 0, "null cardId → 0"],
  [undefined, 0, "undefined cardId → 0"],
  ["", 0, "empty cardId → 0"],
  // Direct hits sampled from cardVocab.json.
  ["3starMakeDebutScout", 1, "direct id 1"],
  ["3starMakeDebutScoutFullArtGold", 2, "direct id 2 (FullArtGold suffix as direct hit)"],
  ["agnesDigitalBasic", 3, "direct id 3"],
  ["agnesTachyonStage1Ex", 11, "direct id 11 (Ex suffix as direct hit)"],
  ["yayoiAkikawaFullArt", 106, "direct id 106 (last entry)"],
  // Suffix fallback: unknown variant id whose base is in vocab. Constructed
  // from a known base by appending an unrecognised variant string the vocab
  // doesn't contain. We use real variants that DO exist as direct entries
  // (above), so for fallback we synthesise by re-suffixing a known *base*.
  // `boxingGloves` (id 14) is a base with only an UncommonPlus variant —
  // FullArt + Ex are not direct hits, so they should fall back to base 14.
  ["boxingGlovesFullArt", 14, "suffix FullArt fallback to base 14"],
  ["boxingGlovesEx", 14, "suffix Ex fallback to base 14"],
  // Unknown id → unknownIndex (0).
  ["definitelyNotARealCard", 0, "unknown id → 0"],
];

for (const [cardId, want, label] of expected) {
  const got = cardVocabIndex(cardId ?? null);
  assert.equal(got, want, `${label}: cardVocabIndex(${JSON.stringify(cardId)}) → ${got}, want ${want}`);
}

const size = cardVocabSize();
assert.ok(size === 107, `vocabSize should be 107, got ${size}`);
const unk = cardVocabUnknownIndex();
assert.ok(unk === 0, `unknownIndex should be 0, got ${unk}`);

console.log(JSON.stringify({ status: "PASS", cases: expected.length, vocabSize: size, unknownIndex: unk }, null, 2));
