//! Bit-identical port of `frontend/src/game/engine/ai-policy/observation.ts`.
//!
//! Builds the `PublicObservation` consumed by the model `/predict` endpoint
//! and by training-example recording. JSON shape stays byte-stable with the
//! TS source.

use indexmap::IndexMap;

use crate::core::catalog::catalog;
use crate::core::constants::{EnergyType, SideId, MAX_BENCH, SpecialCondition};
use crate::core::state::{
    CurrentSide, GameState, PendingPlayerChoice, SideState, UmamusumeInstance,
};
use crate::flow::retreat::get_global_retreat_cost_reduction;
use crate::policy::card_vocab::card_vocab_index;
use crate::policy::phase::get_ai_phase;
use crate::policy::types::{
    PublicObservation, PublicSharedObservation, PublicSideObservation, PublicSideTurnState,
    PublicTemporalObservation, PublicUmaObservation, PublicUmaTurnState,
};

/// Mirror of `observation.ts:16` `buildPublicObservation`. Observation
/// schema version is `3` per R16-P1.
pub fn build_public_observation(state: &GameState, side_id: SideId) -> PublicObservation {
    let opponent_id = side_id.opposite();
    let own = state.side(side_id);
    let opp = state.side(opponent_id);
    let own_turns_taken = state.turns_taken_by_side[side_id as usize];
    let opp_turns_taken = state.turns_taken_by_side[opponent_id as usize];

    let temporal = PublicTemporalObservation {
        own_turns_taken,
        opponent_turns_taken: opp_turns_taken,
        own_is_first_turn: own_turns_taken == 0,
        opponent_is_first_turn: opp_turns_taken == 0,
    };

    let pending_choice_kind = state.pending_player_choice.as_ref().map(|p| match p {
        PendingPlayerChoice::PromoteAfterKnockout { .. } => "promoteAfterKnockout".to_string(),
        PendingPlayerChoice::SwitchAfterGust { .. } => "switchAfterGust".to_string(),
    });

    let stadium_card_id_str = state
        .stadium
        .as_ref()
        .and_then(|s| catalog().interner.resolve(s.card_id).map(|x| x.to_string()));

    let shared = PublicSharedObservation {
        stadium_card_id: stadium_card_id_str.clone(),
        current_side: match state.current_side {
            CurrentSide::Player => "player".to_string(),
            CurrentSide::Opponent => "opponent".to_string(),
            CurrentSide::Done => "done".to_string(),
        },
        game_over: state.game_over,
    };

    PublicObservation {
        schema_version: 3,
        side_to_act: side_id,
        phase: get_ai_phase(state, side_id),
        turn_number: state.turn_number,
        first_player: state.first_player,
        pending_choice_kind,
        temporal,
        own: to_public_side_observation(state, own, true),
        opponent: to_public_side_observation(state, opp, false),
        shared,
        card_ids_by_zone: build_card_ids_by_zone(own, opp, stadium_card_id_str.as_deref()),
    }
}

fn build_card_ids_by_zone(
    own: &SideState,
    opp: &SideState,
    stadium_card_id: Option<&str>,
) -> IndexMap<String, Vec<u32>> {
    let cat = catalog();
    let resolve = |opt: Option<crate::core::card_id::CardId>| -> Option<String> {
        opt.and_then(|cid| cat.interner.resolve(cid).map(|s| s.to_string()))
    };
    let bench_idxs = |side: &SideState| -> Vec<u32> {
        side.bench
            .iter()
            .map(|uma| {
                let s = cat.interner.resolve(uma.card_id);
                card_vocab_index(s)
            })
            .collect()
    };
    let mut map: IndexMap<String, Vec<u32>> = IndexMap::new();
    map.insert(
        "ownActive".into(),
        own.active
            .as_ref()
            .map(|a| vec![card_vocab_index(cat.interner.resolve(a.card_id))])
            .unwrap_or_default(),
    );
    map.insert(
        "oppActive".into(),
        opp.active
            .as_ref()
            .map(|a| vec![card_vocab_index(cat.interner.resolve(a.card_id))])
            .unwrap_or_default(),
    );
    map.insert("ownBench".into(), bench_idxs(own));
    map.insert("oppBench".into(), bench_idxs(opp));
    map.insert(
        "ownHand".into(),
        own.hand
            .iter()
            .map(|cid| card_vocab_index(cat.interner.resolve(*cid)))
            .collect(),
    );
    map.insert(
        "ownDiscard".into(),
        own.discard
            .iter()
            .map(|cid| card_vocab_index(cat.interner.resolve(*cid)))
            .collect(),
    );
    map.insert(
        "oppDiscard".into(),
        opp.discard
            .iter()
            .map(|cid| card_vocab_index(cat.interner.resolve(*cid)))
            .collect(),
    );
    map.insert(
        "stadium".into(),
        match stadium_card_id {
            Some(id) => vec![card_vocab_index(Some(id))],
            None => Vec::new(),
        },
    );
    // Silence unused-resolve to keep the helper future-proof.
    let _ = resolve;
    map
}

fn to_public_side_turn_state(state: &GameState, side: &SideState) -> PublicSideTurnState {
    let global_retreat = get_global_retreat_cost_reduction(state);
    PublicSideTurnState {
        energy_attachments_this_turn: side.energy_attachments_this_turn,
        bonus_energy_attachments: side.bonus_energy_attachments,
        retreat_cost_reduction: side.retreat_cost_reduction,
        effective_retreat_cost_reduction: side.retreat_cost_reduction as u32 + global_retreat,
        active_attack_damage_bonus: side.active_attack_damage_bonus,
        used_ability_name_count_this_turn: side.used_ability_names_this_turn.len(),
        used_ability_name_count_this_game: side.used_ability_names_this_game.len(),
        guaranteed_coin_flip_heads: side.guaranteed_coin_flip_heads,
    }
}

fn to_public_side_observation(
    state: &GameState,
    side: &SideState,
    include_private_hand: bool,
) -> PublicSideObservation {
    let cat = catalog();
    let side_turns_taken = state.turns_taken_by_side[side.id as usize];

    // Bench: take up to MAX_BENCH, then pad with None.
    let mut bench: Vec<Option<PublicUmaObservation>> = side
        .bench
        .iter()
        .take(MAX_BENCH)
        .map(|uma| Some(to_public_uma_observation(state, uma, side_turns_taken)))
        .collect();
    while bench.len() < MAX_BENCH {
        bench.push(None);
    }

    let discard: Vec<String> = side
        .discard
        .iter()
        .filter_map(|cid| cat.interner.resolve(*cid).map(|s| s.to_string()))
        .collect();

    let energy_zone: Vec<String> = side
        .energy_zone
        .iter()
        .map(|e| energy_label_camel(*e).to_string())
        .collect();

    let hand_card_ids = if include_private_hand {
        Some(
            side.hand
                .iter()
                .filter_map(|cid| cat.interner.resolve(*cid).map(|s| s.to_string()))
                .collect::<Vec<String>>(),
        )
    } else {
        None
    };

    PublicSideObservation {
        id: side.id,
        points: side.points,
        hand_count: side.hand.len(),
        hand_card_ids,
        deck_count: side.deck.len(),
        discard,
        active: side
            .active
            .as_ref()
            .map(|a| to_public_uma_observation(state, a, side_turns_taken)),
        bench,
        energy_zone,
        used_supporter_this_turn: side.used_supporter_this_turn,
        used_retreat_this_turn: side.used_retreat_this_turn,
        used_stadium_this_turn: side.used_stadium_this_turn,
        turn_state: to_public_side_turn_state(state, side),
    }
}

fn to_public_uma_turn_state(
    state: &GameState,
    umamusume: &UmamusumeInstance,
    side_turns_taken: u32,
) -> PublicUmaTurnState {
    let turn_number = state.turn_number;
    let evolved_turn = umamusume.evolved_turn;
    let paralysed = umamusume
        .special_conditions
        .iter()
        .any(|c| *c == SpecialCondition::Paralysed);
    let paralysis_recovery_pending = paralysed
        && umamusume.paralysed_until_own_turn.is_some()
        && side_turns_taken < umamusume.paralysed_until_own_turn.unwrap();

    PublicUmaTurnState {
        turns_in_play: turn_number.saturating_sub(umamusume.entered_turn),
        entered_this_turn: umamusume.entered_turn == turn_number,
        evolved_this_turn: evolved_turn == Some(turn_number),
        evolved_last_turn: evolved_turn == Some(turn_number.saturating_sub(1))
            && evolved_turn.is_some()
            && turn_number > 0,
        took_damage_last_turn: umamusume.took_damage_last_turn,
        took_damage_this_turn: umamusume.took_damage_this_turn,
        next_turn_damage_reduction: umamusume.next_turn_damage_reduction,
        attack_blocked_this_turn: umamusume
            .attack_blocked_until_own_turn
            == Some(side_turns_taken),
        paralysis_recovery_pending,
    }
}

fn to_public_uma_observation(
    state: &GameState,
    umamusume: &UmamusumeInstance,
    side_turns_taken: u32,
) -> PublicUmaObservation {
    let cat = catalog();
    let energy_total: u32 = umamusume.energies.iter().map(|n| *n as u32).sum();
    let mut energies: IndexMap<String, u16> = IndexMap::new();
    for et in EnergyType::ALL {
        energies.insert(
            energy_label_camel(et).to_string(),
            umamusume.energies[et as usize],
        );
    }
    let card_id_str = cat
        .interner
        .resolve(umamusume.card_id)
        .map(|s| s.to_string())
        .unwrap_or_default();
    let tool_card_id = umamusume
        .tool_card_id
        .and_then(|cid| cat.interner.resolve(cid).map(|s| s.to_string()));
    let special_conditions: Vec<String> = umamusume
        .special_conditions
        .iter()
        .map(|c| special_condition_str(*c).to_string())
        .collect();

    PublicUmaObservation {
        uid: umamusume.uid,
        card_id: card_id_str,
        species: umamusume.species.clone(),
        stage: umamusume.stage,
        hp: umamusume.hp,
        max_hp: umamusume.max_hp,
        energy_total,
        energies,
        special_conditions,
        tool_card_id,
        used_ability_this_turn: umamusume.used_ability_this_turn,
        turn_state: to_public_uma_turn_state(state, umamusume, side_turns_taken),
    }
}

fn energy_label_camel(e: EnergyType) -> &'static str {
    // Lowercase JSON keys matching `EnergyType` TS string enum.
    match e {
        EnergyType::Grass => "grass",
        EnergyType::Fire => "fire",
        EnergyType::Water => "water",
        EnergyType::Lightning => "lightning",
        EnergyType::Psychic => "psychic",
        EnergyType::Fighting => "fighting",
        EnergyType::Darkness => "darkness",
        EnergyType::Steel => "steel",
        EnergyType::Colorless => "colorless",
        EnergyType::Dragon => "dragon",
    }
}

fn special_condition_str(c: SpecialCondition) -> &'static str {
    match c {
        SpecialCondition::Asleep => "asleep",
        SpecialCondition::Burned => "burned",
        SpecialCondition::Frozen => "frozen",
        SpecialCondition::Paralysed => "paralysed",
        SpecialCondition::Poisoned => "poisoned",
    }
}
