//! Packed serialization of `GameState` for clone and fingerprinting.
//!
//! TS uses `structuredClone` (≈22 μs/call per the throughput probe) and
//! `JSON.stringify` of a compact view (≈8 μs/call) for the fingerprint. The
//! Rust port collapses both costs by:
//!
//! 1. Defining `GameState` with fixed-cap collections (`ArrayVec`,
//!    `[T; 2]`, `[u16; 10]`) so it implements `Clone` cheaply (no
//!    allocation, just memcpy).
//! 2. Hashing a deterministic byte serialization of the state directly.
//!
//! The byte serialization defined here is **not** intended to round-trip
//! into a `GameState` — it exists only so xxh3 has a stable byte stream
//! to digest. Round-trip serialization, if ever needed, should use serde
//! against the `GameState` struct in `state.rs`.
//!
//! Stability contract: byte order, field order, and length-prefixing are
//! all fixed by this module. If the schema changes, bump
//! `PACKED_SCHEMA_VERSION` so fingerprints stored on disk become
//! invalidatable.

use crate::core::state::{
    GameState, PendingPlayerChoice, PromoteResume, SetupState, SideState, StadiumState,
    SwitchResume, UmamusumeInstance,
};

/// Bump when the byte schema changes. Stored as the first 4 bytes of every
/// packed buffer so older fingerprints can be invalidated.
pub const PACKED_SCHEMA_VERSION: u32 = 1;

pub fn pack(state: &GameState) -> Vec<u8> {
    let mut buf = Vec::with_capacity(512);
    pack_into(state, &mut buf);
    buf
}

pub fn pack_into(state: &GameState, buf: &mut Vec<u8>) {
    buf.clear();
    buf.extend_from_slice(&PACKED_SCHEMA_VERSION.to_le_bytes());

    // phase + currentSide + opponentTurnStep + game over/winner — small
    // discriminants packed up front so a quick reject is cheap.
    buf.push(state.phase as u8);
    buf.push(state.current_side as u8);
    buf.push(state.opponent_turn_step.map(|s| s as u8).unwrap_or(0xFF));
    buf.push(state.game_over as u8);
    buf.push(state.winner.map(|s| s as u8).unwrap_or(0xFF));
    buf.push(state.first_player as u8);
    buf.push(state.ai_difficulty as u8);
    buf.push(state.human_by_side[0] as u8);
    buf.push(state.human_by_side[1] as u8);
    buf.push(state.ai_deck_style_by_side[0] as u8);
    buf.push(state.ai_deck_style_by_side[1] as u8);
    buf.extend_from_slice(&state.turn_number.to_le_bytes());
    buf.extend_from_slice(&state.turns_taken_by_side[0].to_le_bytes());
    buf.extend_from_slice(&state.turns_taken_by_side[1].to_le_bytes());
    buf.extend_from_slice(&state.turn_deadline_ms.unwrap_or(u64::MAX).to_le_bytes());

    // pendingPlayerChoice tag + payload
    match &state.pending_player_choice {
        None => buf.push(0),
        Some(PendingPlayerChoice::PromoteAfterKnockout { side_id, resume }) => {
            buf.push(1);
            buf.push(*side_id as u8);
            buf.push(match resume {
                PromoteResume::FinishOpponentTurn => 0,
                PromoteResume::None => 1,
            });
        }
        Some(PendingPlayerChoice::SwitchAfterGust { side_id, resume }) => {
            buf.push(2);
            buf.push(*side_id as u8);
            buf.push(match resume {
                SwitchResume::ResumeOpponentAfterFirstTrainerPass => 0,
                SwitchResume::ResumeOpponentAfterSecondTrainerPass => 1,
                SwitchResume::None => 2,
            });
        }
    }

    // stadium
    match &state.stadium {
        None => buf.push(0),
        Some(StadiumState { card_id, owner }) => {
            buf.push(1);
            buf.extend_from_slice(&card_id.0.to_le_bytes());
            buf.push(*owner as u8);
        }
    }

    // setup
    match &state.setup {
        None => buf.push(0),
        Some(s) => {
            buf.push(1);
            pack_setup(s, buf);
        }
    }

    // sides — iteration order is fixed by SideId::ALL = [Player, Opponent]
    // and matches Record<SideId, …> declaration order in TS.
    pack_side(&state.sides[0], buf);
    pack_side(&state.sides[1], buf);
}

fn pack_setup(setup: &SetupState, buf: &mut Vec<u8>) {
    buf.push(setup.coin_choice.map(|c| c as u8).unwrap_or(0xFF));
    buf.push(setup.coin_flip_result.map(|c| c as u8).unwrap_or(0xFF));
    buf.push(setup.opening_hands_dealt as u8);
    buf.push(setup.ready_by_side[0] as u8);
    buf.push(setup.ready_by_side[1] as u8);
    buf.push(setup.opponent_revealed as u8);
    buf.extend_from_slice(
        &setup
            .countdown_seconds_remaining
            .unwrap_or(u32::MAX)
            .to_le_bytes(),
    );
    for hand in &setup.opening_hands {
        push_card_list(buf, hand);
    }
}

fn pack_side(side: &SideState, buf: &mut Vec<u8>) {
    buf.push(side.id as u8);
    buf.push(side.points);
    buf.push(side.energy_attachments_this_turn);
    buf.push(side.bonus_energy_attachments);
    buf.push(side.retreat_cost_reduction);
    buf.extend_from_slice(&side.active_attack_damage_bonus.to_le_bytes());
    buf.push(side.used_supporter_this_turn as u8);
    buf.push(side.used_retreat_this_turn as u8);
    buf.push(side.used_stadium_this_turn as u8);
    buf.push(side.guaranteed_coin_flip_heads);

    // Energy pool & energy zone preserved in push order.
    buf.push(side.energy_pool.len() as u8);
    for &e in &side.energy_pool {
        buf.push(e as u8);
    }
    buf.push(side.energy_zone.len() as u8);
    for &e in &side.energy_zone {
        buf.push(e as u8);
    }

    // Card lists.
    push_card_list(buf, &side.deck);
    push_card_list(buf, &side.discard);
    push_card_list(buf, &side.hand);

    // Active + bench instances.
    match &side.active {
        None => buf.push(0),
        Some(u) => {
            buf.push(1);
            pack_instance(u, buf);
        }
    }
    buf.push(side.bench.len() as u8);
    for u in &side.bench {
        pack_instance(u, buf);
    }

    // Used-ability strings: hash a stable digest of each entry so we don't
    // pull arbitrary-length strings into the fingerprint. xxh3_64 over each
    // name keeps the buffer compact and order-sensitive.
    buf.push(side.used_ability_names_this_turn.len() as u8);
    for name in &side.used_ability_names_this_turn {
        buf.extend_from_slice(&xxhash_rust::xxh3::xxh3_64(name.as_bytes()).to_le_bytes());
    }
    buf.push(side.used_ability_names_this_game.len() as u8);
    for name in &side.used_ability_names_this_game {
        buf.extend_from_slice(&xxhash_rust::xxh3::xxh3_64(name.as_bytes()).to_le_bytes());
    }
}

fn pack_instance(inst: &UmamusumeInstance, buf: &mut Vec<u8>) {
    buf.extend_from_slice(&inst.uid.to_le_bytes());
    buf.extend_from_slice(&inst.card_id.0.to_le_bytes());
    buf.push(inst.evolution_card_ids.len() as u8);
    for c in &inst.evolution_card_ids {
        buf.extend_from_slice(&c.0.to_le_bytes());
    }
    // species is a denormalized lookup key; treat as a hashed digest.
    buf.extend_from_slice(&xxhash_rust::xxh3::xxh3_64(inst.species.as_bytes()).to_le_bytes());
    buf.push(inst.stage);
    buf.extend_from_slice(&inst.hp.to_le_bytes());
    buf.extend_from_slice(&inst.max_hp.to_le_bytes());
    for &count in &inst.energies {
        buf.extend_from_slice(&count.to_le_bytes());
    }
    buf.push(inst.special_conditions.len() as u8);
    for &c in &inst.special_conditions {
        buf.push(c as u8);
    }
    buf.extend_from_slice(&inst.entered_turn.to_le_bytes());
    buf.extend_from_slice(&inst.evolved_turn.unwrap_or(u32::MAX).to_le_bytes());
    buf.push(inst.took_damage_last_turn as u8);
    buf.push(inst.took_damage_this_turn as u8);
    buf.extend_from_slice(&inst.next_turn_damage_reduction.to_le_bytes());
    buf.push(inst.used_ability_this_turn as u8);
    buf.extend_from_slice(
        &inst
            .attack_blocked_until_own_turn
            .unwrap_or(u32::MAX)
            .to_le_bytes(),
    );
    buf.extend_from_slice(
        &inst
            .paralysed_until_own_turn
            .unwrap_or(u32::MAX)
            .to_le_bytes(),
    );
    buf.extend_from_slice(
        &inst
            .tool_card_id
            .map(|c| c.0)
            .unwrap_or(u16::MAX)
            .to_le_bytes(),
    );
}

fn push_card_list(buf: &mut Vec<u8>, ids: &[crate::core::card_id::CardId]) {
    buf.extend_from_slice(&(ids.len() as u16).to_le_bytes());
    for cid in ids {
        buf.extend_from_slice(&cid.0.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::card_id::CardId;
    use crate::core::constants::{AiDeckStyle, AiDifficulty, SideId};
    use crate::core::state::{CurrentSide, Phase, SideState};
    use crate::fingerprint::fingerprint;
    use arrayvec::ArrayVec;

    fn empty_side(id: SideId) -> SideState {
        SideState {
            id,
            title: String::new(),
            energy_pool: ArrayVec::new(),
            deck: ArrayVec::new(),
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

    fn empty_state() -> GameState {
        GameState {
            phase: Phase::Setup,
            setup: None,
            pending_player_choice: None,
            sides: [empty_side(SideId::Player), empty_side(SideId::Opponent)],
            current_side: CurrentSide::Player,
            opponent_turn_step: None,
            stadium: None,
            turn_deadline_ms: None,
            turn_number: 0,
            first_player: SideId::Player,
            turns_taken_by_side: [0, 0],
            ai_difficulty: AiDifficulty::Normal,
            human_by_side: [false, false],
            ai_deck_style_by_side: [AiDeckStyle::Balanced, AiDeckStyle::Balanced],
            game_over: false,
            winner: None,
            log: std::collections::VecDeque::new(),
        }
    }

    #[test]
    fn pack_is_stable() {
        let s = empty_state();
        let a = pack(&s);
        let b = pack(&s);
        assert_eq!(a, b);
    }

    #[test]
    fn pack_changes_when_turn_advances() {
        let mut s = empty_state();
        let a = pack(&s);
        s.turn_number = 1;
        let b = pack(&s);
        assert_ne!(a, b, "turn change must perturb the packed buffer");
        assert_ne!(fingerprint(&a), fingerprint(&b));
    }

    #[test]
    fn fingerprint_matches_pack() {
        let s = empty_state();
        let buf = pack(&s);
        let fp = fingerprint(&buf);
        // Trivially: fingerprint is deterministic.
        assert_eq!(fp, fingerprint(&pack(&s)));
    }

    #[test]
    fn pack_changes_with_card_id() {
        let mut s = empty_state();
        let a = pack(&s);
        s.sides[0].hand.push(CardId(7));
        let b = pack(&s);
        assert_ne!(a, b);
    }
}
