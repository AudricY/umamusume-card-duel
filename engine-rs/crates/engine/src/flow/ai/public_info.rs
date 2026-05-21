//! Bit-identical port of `frontend/src/game/engine/flow/ai/publicInfo.ts`.
//!
//! Snapshots the publicly-knowable view of the opponent. TS uses spread
//! copies (`[...side.bench]`, `[...side.discard]`); the Rust port returns
//! cloned `Vec`s so the caller can mutate them freely.

use crate::core::card_id::CardId;
use crate::core::constants::SideId;
use crate::core::state::{GameState, SideState, UmamusumeInstance};
use crate::flow::board::get_opposing_side;

#[derive(Debug, Clone)]
pub struct PublicSideView {
    pub id: SideId,
    pub active: Option<UmamusumeInstance>,
    pub bench: Vec<UmamusumeInstance>,
    pub discard: Vec<CardId>,
    pub points: u8,
    pub stadium_card_id: Option<CardId>,
}

pub fn get_public_opponent_view(state: &GameState, side_id: SideId) -> PublicSideView {
    let opponent = get_opposing_side(state, side_id);
    to_public_side_view(state, opponent)
}

pub fn to_public_side_view(state: &GameState, side: &SideState) -> PublicSideView {
    PublicSideView {
        id: side.id,
        active: side.active.clone(),
        bench: side.bench.iter().cloned().collect(),
        discard: side.discard.iter().copied().collect(),
        points: side.points,
        stadium_card_id: state.stadium.as_ref().map(|s| s.card_id),
    }
}
