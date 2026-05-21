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
    let cat = catalog();
    let mut count = 0i32;
    // Iteration order: hand → discard → all-umamusume.
    for &cid in side.hand.iter() {
        if let Some(Card::Umamusume(u)) = cat.get(cid) {
            if u.stage == 0 {
                count += 1;
            }
        }
    }
    for &cid in side.discard.iter() {
        if let Some(Card::Umamusume(u)) = cat.get(cid) {
            if u.stage == 0 {
                count += 1;
            }
        }
    }
    for inst in get_all_umamusume(side) {
        if let Some(Card::Umamusume(u)) = cat.get(inst.card_id) {
            if u.stage == 0 {
                count += 1;
            }
        }
    }
    count
}
