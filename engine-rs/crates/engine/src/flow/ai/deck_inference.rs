//! Bit-identical port of `frontend/src/game/engine/flow/ai/deckInference.ts`.

use crate::core::catalog::{catalog, Card};
use crate::core::state::SideState;
use crate::core::umamusume::get_all_umamusume;

#[derive(Debug, Clone, Copy)]
pub struct KnownRemainingDeckCounts {
    pub basic_umamusume: i32,
    pub evolution_umamusume: i32,
    pub trainer: i32,
}

pub fn get_known_remaining_deck_counts(side: &SideState) -> KnownRemainingDeckCounts {
    let cat = catalog();
    let mut basic_umamusume = 0i32;
    let mut evolution_umamusume = 0i32;
    let mut trainer = 0i32;
    for &cid in side.deck.iter() {
        match cat.get(cid) {
            Some(Card::Trainer(_)) => trainer += 1,
            Some(Card::Umamusume(u)) => {
                if u.stage == 0 {
                    basic_umamusume += 1;
                } else {
                    evolution_umamusume += 1;
                }
            }
            None => {}
        }
    }
    KnownRemainingDeckCounts {
        basic_umamusume,
        evolution_umamusume,
        trainer,
    }
}

pub fn count_consumed_basics(side: &SideState) -> i32 {
    // Slice 3i: hot-loop classifier — `catalog().is_basic_umamusume(cid)`
    // is a `Vec<bool>` index, replacing the `Card` variant match + `u.stage`
    // load this used to perform on every iteration. Identical result, but
    // skips the catalog struct field touch (and the L1 hit it took to
    // resolve `Card::Umamusume(_)` past the enum tag).
    let cat = catalog();
    let mut count = 0i32;
    // Iteration order: hand → discard → all-umamusume.
    for &cid in side.hand.iter() {
        if cat.is_basic_umamusume(cid) {
            count += 1;
        }
    }
    for &cid in side.discard.iter() {
        if cat.is_basic_umamusume(cid) {
            count += 1;
        }
    }
    for inst in get_all_umamusume(side) {
        if cat.is_basic_umamusume(inst.card_id) {
            count += 1;
        }
    }
    count
}
