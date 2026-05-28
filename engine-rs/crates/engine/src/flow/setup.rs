//! Bit-identical port of `frontend/src/game/engine/flow/setup.ts`.
//!
//! RNG sites:
//! - `draw_opening_hand`: `shuffle(deckList)` in a loop until at least one
//!   basic umamusume is in the dealt hand. The loop's draw count is
//!   bounded in practice by the deck composition, but the RNG advance
//!   PER LOOP ITERATION is load-bearing for fingerprint identity.
//!
//! `nextUmamusumeId` is module-level state in TS; in Rust we expose it as
//! a thread-local counter so the simulator stays single-threaded
//! (matching the harness's single-process determinism contract).

use std::cell::Cell;

use arrayvec::ArrayVec;

use crate::core::card_id::CardId;
use crate::core::catalog::{catalog, Card};
use crate::core::constants::{EnergyType, SideId, UmamusumeType, MAX_BENCH, OPENING_HAND};
use crate::core::random::shuffle;
use crate::core::state::{SideState, UmamusumeInstance};

thread_local! {
    static NEXT_UMAMUSUME_ID: Cell<u32> = const { Cell::new(1) };
}

pub fn reset_umamusume_id_counter() {
    NEXT_UMAMUSUME_ID.with(|c| c.set(1));
}

fn next_umamusume_id() -> u32 {
    NEXT_UMAMUSUME_ID.with(|c| {
        let v = c.get();
        c.set(v + 1);
        v
    })
}

/// `setup.ts:13` `createUmamusume`. Panics if the cardId resolves to a
/// non-umamusume (mirrors the TS throw).
pub fn create_umamusume(card_id: CardId, turn_number: u32) -> UmamusumeInstance {
    let cat = catalog();
    let card = match cat.get(card_id) {
        Some(Card::Umamusume(u)) => u,
        _ => panic!(
            "Expected Umamusume card: {:?}",
            cat.interner.resolve(card_id)
        ),
    };
    UmamusumeInstance {
        uid: next_umamusume_id(),
        card_id,
        evolution_card_ids: ArrayVec::new(),
        stage: card.stage,
        hp: card.hp,
        max_hp: card.hp,
        energies: [0u16; EnergyType::COUNT],
        special_conditions: ArrayVec::new(),
        entered_turn: turn_number,
        evolved_turn: None,
        took_damage_last_turn: false,
        took_damage_this_turn: false,
        next_turn_damage_reduction: 0,
        used_ability_this_turn: false,
        attack_blocked_until_own_turn: None,
        paralysed_until_own_turn: None,
        tool_card_id: None,
    }
}

/// `setup.ts:110` `drawOpeningHand`. Retries `shuffle` until at least one
/// basic umamusume is in the opening hand. Returns `(deck, hand)` where
/// `deck` is the post-draw deck and `hand` is the opening 5.
///
/// Must be called inside a `with_rng` scope.
pub fn draw_opening_hand(deck_list: &[CardId]) -> (Vec<CardId>, Vec<CardId>) {
    let cat = catalog();
    loop {
        let mut shuffled = shuffle(deck_list);
        let hand: Vec<CardId> = shuffled.drain(..OPENING_HAND.min(shuffled.len())).collect();
        if hand.iter().any(|&cid| cat.is_basic_umamusume(cid)) {
            return (shuffled, hand);
        }
    }
}

/// `setup.ts:38` `buildOpeningSide`.
pub fn build_opening_side(
    id: SideId,
    title: &str,
    deck_list: &[CardId],
    auto_setup_active: bool,
    selected_energy_types: Option<&[EnergyType]>,
) -> SideState {
    let (deck_vec, hand_vec) = draw_opening_hand(deck_list);
    let energy_pool = get_deck_energy_pool(deck_list, selected_energy_types);
    let mut side = make_side(id, title, deck_vec, energy_pool);
    // `hand_vec` was OPENING_HAND items.
    for c in &hand_vec {
        let _ = side.hand.try_push(*c);
    }

    if auto_setup_active {
        auto_setup_basic_umamusume(&mut side);
    }
    side
}

/// `setup.ts:69` `autoSetupBasicUmamusume`. Picks up to `MAX_BENCH + 1`
/// basics out of hand to seed active + bench in source order.
pub fn auto_setup_basic_umamusume(side: &mut SideState) {
    let cat = catalog();
    let cap = MAX_BENCH + 1;
    let basics: Vec<(CardId, usize)> = side
        .hand
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, cid)| cat.is_basic_umamusume(*cid))
        .take(cap)
        .map(|(idx, cid)| (cid, idx))
        .collect();
    if basics.is_empty() {
        return;
    }
    side.active = Some(create_umamusume(basics[0].0, 0));
    let mut new_bench: ArrayVec<UmamusumeInstance, MAX_BENCH> = ArrayVec::new();
    for (cid, _idx) in basics.iter().skip(1) {
        let _ = new_bench.try_push(create_umamusume(*cid, 0));
    }
    side.bench = new_bench;

    let taken: std::collections::BTreeSet<usize> = basics.iter().map(|(_, idx)| *idx).collect();
    let mut new_hand: ArrayVec<CardId, { crate::core::constants::MAX_HAND }> = ArrayVec::new();
    for (idx, c) in side.hand.iter().copied().enumerate() {
        if !taken.contains(&idx) {
            let _ = new_hand.try_push(c);
        }
    }
    side.hand = new_hand;
}

fn get_deck_energy_pool(
    deck_list: &[CardId],
    selected_energy_types: Option<&[EnergyType]>,
) -> Vec<EnergyType> {
    let selected = normalize_selected_energy_types(selected_energy_types);
    if !selected.is_empty() {
        return selected;
    }
    let cat = catalog();
    // Use a vec instead of HashSet to preserve TS Set insertion order
    // (which IS load-bearing: iteration over the pool happens in
    // construction order).
    let mut pool: Vec<EnergyType> = Vec::new();
    for &cid in deck_list {
        if let Some(Card::Umamusume(u)) = cat.get(cid) {
            let e = u.r#type.energy();
            if !pool.contains(&e) {
                pool.push(e);
            }
        }
    }
    if pool.is_empty() {
        vec![EnergyType::Psychic]
    } else {
        pool
    }
}

fn normalize_selected_energy_types(
    selected_energy_types: Option<&[EnergyType]>,
) -> Vec<EnergyType> {
    let Some(input) = selected_energy_types else {
        return Vec::new();
    };
    // Mirror of `.filter(type !== colorless && selected.has(type)).slice(0, 3)`.
    let mut out = Vec::new();
    for t in EnergyType::ALL {
        if t == EnergyType::Colorless {
            continue;
        }
        if input.iter().any(|&i| i == t) {
            out.push(t);
            if out.len() == 3 {
                break;
            }
        }
    }
    out
}

fn make_side(
    id: SideId,
    title: &str,
    deck_vec: Vec<CardId>,
    energy_pool: Vec<EnergyType>,
) -> SideState {
    let mut deck: ArrayVec<CardId, { crate::core::constants::DECK_CARD_COUNT }> = ArrayVec::new();
    for c in &deck_vec {
        let _ = deck.try_push(*c);
    }
    let mut pool: ArrayVec<EnergyType, { EnergyType::COUNT }> = ArrayVec::new();
    for e in &energy_pool {
        let _ = pool.try_push(*e);
    }
    SideState {
        id,
        title: title.to_string(),
        energy_pool: pool,
        deck,
        discard: ArrayVec::new(),
        hand: ArrayVec::new(),
        active: None,
        bench: ArrayVec::new(),
        points: 0,
        energy_zone: ArrayVec::new(),
        energy_attachments_this_turn: 0,
        bonus_energy_attachments: 0,
        retreat_cost_reduction: 0,
        active_attack_damage_bonus: 0,
        used_supporter_this_turn: false,
        used_retreat_this_turn: false,
        used_stadium_this_turn: false,
        used_ability_names_this_turn: ArrayVec::new(),
        used_ability_names_this_game: ArrayVec::new(),
        guaranteed_coin_flip_heads: 0,
    }
}

// Silence unused import warnings when UmamusumeType is unreferenced in
// downstream modules.
#[allow(dead_code)]
fn _kw_referenced() {
    let _ = UmamusumeType::Grass;
}
