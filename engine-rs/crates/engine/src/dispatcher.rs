//! Bit-identical port of `frontend/src/game/engine.ts` (top-level game
//! orchestrator) + the `evaluateModelVsHeuristic.ts` helpers that MCTS
//! consumes (`advance_modeled_turn_step`, `get_forced_attack_coin_results`,
//! `state_hash`).
//!
//! Conformance contract:
//! - RNG sites are preserved in the same TS source order. The two state-
//!   aware `flip_coin` helpers (one in `flow::combat`, one here) consume
//!   one `random_float()` call per non-guaranteed-heads flip.
//! - `advance_ai_turn_step` mirrors the TS 8-iteration state machine
//!   verbatim, including the `aiPlayOneBasic` re-check at the
//!   `trainerBefore` and `trainerAfter` steps (early-return on success).
//! - `refresh_continuous_effects` and `resolve_continuous_knockouts` are
//!   the canonical wiring used by every public entry point.

use crate::core::card_id::CardId;
use crate::core::catalog::{catalog, Card};
use crate::core::constants::{
    AiDeckStyle, AiDifficulty, CoinFlipResult, EnergyType, OpponentTurnStep, SideId, MAX_BENCH,
    MAX_POINTS,
};
use crate::core::effects::{AttackTarget, EnergyCost};
use crate::core::play_types::{PlayActionKind, PlayActionOutcome, PlayChoices};
use crate::core::random::{random_float, random_int, shuffle};
use crate::core::state::{
    CurrentSide, GameState, PendingPlayerChoice, Phase, PromoteResume, SetupState, SideState,
    StadiumState, SwitchResume, UmamusumeInstance,
};
use crate::core::umamusume::{find_own_umamusume_by_uid, get_all_umamusume};
use crate::flow::ability_rules::get_umamusume_ability;
use crate::flow::ai::ability_utils::{
    ai_use_coin_flip_draw_ability, ai_use_damage_ability, ai_use_move_benched_energy_ability,
    AbilityHeuristicDeps,
};
use crate::flow::ai::attach_utils::{
    estimate_attack_damage_output, mark_ability_used, with_energy_shift,
};
use crate::flow::ai::combat_planner::ai_retreat_to_target;
use crate::flow::ai::core::{
    ai_attach_one_energy, ai_evolve_one, ai_play_one_basic, ai_play_one_trainer,
    ai_resolve_combat_decision, ai_use_one_ability, AiCombatDepsAi, AiTrainerDeps,
};
use crate::flow::ai::telemetry::clear_ai_telemetry;
use crate::flow::ai::turn_plan::choose_ai_turn_goal;
use crate::flow::ai::types::PendingSwitchAfterGustResume;
use crate::flow::board::{
    choose_preferred_active_index, normalize_board_state, refresh_continuous_hp,
    switch_out_opponent_active,
};
use crate::flow::combat::{knock_out_umamusume, perform_attack, CombatDeps};
use crate::flow::eligibility::{
    can_attach_energy_to_umamusume, can_attack, can_retreat, can_use_umamusume_ability,
    is_player_turn,
};
use crate::flow::energy::{attach_energy as flow_attach_energy, get_ability_move_energy_types};
use crate::flow::evolution::is_valid_evolution_target;
use crate::flow::play_rules::{
    get_playable_action, get_rainbow_uncap_evolution_hand_options, get_rainbow_uncap_targets,
    resolve_card_play,
};
use crate::flow::retreat::{effective_retreat_cost, pay_retreat_cost, pay_retreat_cost_by_selection};
use crate::flow::setup::{
    auto_setup_basic_umamusume, build_opening_side, create_umamusume, reset_umamusume_id_counter,
};
use crate::flow::special_conditions::clear_special_conditions;
use crate::flow::trainers::{can_use_stadium, use_stadium};
use crate::flow::turn::{draw_cards, end_turn, start_turn};
use crate::policy::types::{AiPhase, LegalAiAction};

// ---------------------------------------------------------------------------
// Continuous-effect helpers (mirror of `engine.ts:742,747`).
// ---------------------------------------------------------------------------

pub fn refresh_continuous_effects(state: &mut GameState) {
    refresh_continuous_hp(state);
    resolve_continuous_knockouts(state);
}

pub fn resolve_continuous_knockouts(state: &mut GameState) {
    let choose: &dyn Fn(&SideState) -> i32 = &choose_preferred_active_index;
    loop {
        if state.game_over {
            return;
        }
        let mut resolved = false;
        for sid in SideId::ALL {
            let knocked_uid: Option<u32> = {
                let side = state.side(sid);
                get_all_umamusume(side)
                    .into_iter()
                    .find(|u| u.hp <= 0)
                    .map(|u| u.uid)
            };
            let Some(uid) = knocked_uid else { continue };
            let scoring = sid.opposite();
            // We need to pass the instance to knock_out_umamusume.
            // Clone the instance first since the helper may need it after
            // the side mutates.
            let target_inst = {
                let side = state.side(sid);
                get_all_umamusume(side)
                    .into_iter()
                    .find(|u| u.uid == uid)
                    .cloned()
            };
            let Some(target_inst) = target_inst else { continue };
            let did = knock_out_umamusume(state, scoring, sid, &target_inst, choose);
            if did {
                refresh_continuous_hp(state);
                resolved = true;
                break;
            }
        }
        if !resolved {
            return;
        }
    }
}

pub fn advance_to_next_turn(state: &mut GameState) {
    state.turn_deadline_ms = None;
    end_turn(
        state,
        |s, side_id| {
            start_turn(s, side_id, refresh_continuous_effects, false);
        },
        refresh_continuous_effects,
    );
}

// ---------------------------------------------------------------------------
// Game construction (mirror of `engine.ts:76`).
// ---------------------------------------------------------------------------

/// Build the initial GameState. Caller must be inside a `with_rng` scope
/// because `build_opening_side` draws opening hands via `shuffle`.
pub fn create_game(
    player_deck: &[CardId],
    opponent_deck: &[CardId],
    opponent_name: &str,
    _ai_difficulty: AiDifficulty,
    opponent_is_human: bool,
    player_name: &str,
    player_energy_types: Option<&[EnergyType]>,
    opponent_energy_types: Option<&[EnergyType]>,
) -> GameState {
    clear_ai_telemetry();
    reset_umamusume_id_counter();
    let player_side = build_opening_side(
        SideId::Player,
        player_name,
        player_deck,
        false,
        player_energy_types,
    );
    let opponent_side = build_opening_side(
        SideId::Opponent,
        opponent_name,
        opponent_deck,
        false,
        opponent_energy_types,
    );

    let mut opening_hands_player: arrayvec::ArrayVec<CardId, { crate::core::constants::OPENING_HAND }> =
        arrayvec::ArrayVec::new();
    for c in player_side.hand.iter().take(crate::core::constants::OPENING_HAND) {
        let _ = opening_hands_player.try_push(*c);
    }
    let mut opening_hands_opponent: arrayvec::ArrayVec<CardId, { crate::core::constants::OPENING_HAND }> =
        arrayvec::ArrayVec::new();
    for c in opponent_side.hand.iter().take(crate::core::constants::OPENING_HAND) {
        let _ = opening_hands_opponent.try_push(*c);
    }

    let setup = SetupState {
        coin_choice: None,
        coin_flip_result: None,
        opening_hands: [opening_hands_player, opening_hands_opponent],
        opening_hands_dealt: false,
        ready_by_side: [false, false],
        opponent_revealed: false,
        countdown_seconds_remaining: None,
    };

    GameState {
        phase: Phase::Setup,
        setup: Some(setup),
        pending_player_choice: None,
        sides: [player_side, opponent_side],
        current_side: CurrentSide::Player,
        opponent_turn_step: None,
        stadium: None,
        turn_deadline_ms: None,
        turn_number: 1,
        first_player: SideId::Player,
        turns_taken_by_side: [0, 0],
        ai_difficulty: AiDifficulty::Hard,
        human_by_side: [true, opponent_is_human],
        ai_deck_style_by_side: [AiDeckStyle::Balanced, AiDeckStyle::Balanced],
        game_over: false,
        winner: None,
        log: std::collections::VecDeque::new(),
    }
}

// ---------------------------------------------------------------------------
// Setup-phase helpers (mirror of `engine.ts:587-695`).
// ---------------------------------------------------------------------------

pub fn choose_opening_coin(state: &mut GameState, choice: CoinFlipResult) {
    if state.phase != Phase::Setup {
        return;
    }
    let Some(setup) = state.setup.as_mut() else {
        return;
    };
    if setup.coin_flip_result.is_some() {
        return;
    }
    let result = if random_float() >= 0.5 {
        CoinFlipResult::Heads
    } else {
        CoinFlipResult::Tails
    };
    let first_player = if result == choice {
        SideId::Player
    } else {
        SideId::Opponent
    };
    setup.coin_choice = Some(choice);
    setup.coin_flip_result = Some(result);
    state.first_player = first_player;
    state.current_side = CurrentSide::from_side(first_player);
}

pub fn deal_opening_hands(state: &mut GameState) {
    if state.phase != Phase::Setup {
        return;
    }
    let Some(setup) = state.setup.as_ref() else {
        return;
    };
    if setup.coin_flip_result.is_none() || setup.opening_hands_dealt {
        return;
    }
    // The opening hands were built into `setup.opening_hands` at game
    // construction; mirror the TS copy-into-side step.
    let player_hand: Vec<CardId> = setup.opening_hands[0].iter().copied().collect();
    let opp_hand: Vec<CardId> = setup.opening_hands[1].iter().copied().collect();
    {
        let p = state.side_mut(SideId::Player);
        p.hand.clear();
        for c in &player_hand {
            let _ = p.hand.try_push(*c);
        }
    }
    {
        let o = state.side_mut(SideId::Opponent);
        o.hand.clear();
        for c in &opp_hand {
            let _ = o.hand.try_push(*c);
        }
    }
    if let Some(setup) = state.setup.as_mut() {
        setup.opening_hands_dealt = true;
    }
}

pub fn auto_complete_opponent_setup(state: &mut GameState) {
    if state.phase != Phase::Setup {
        return;
    }
    let Some(setup) = state.setup.as_ref() else {
        return;
    };
    if setup.ready_by_side[SideId::Opponent as usize] || !setup.opening_hands_dealt {
        return;
    }
    auto_setup_basic_umamusume(state.side_mut(SideId::Opponent));
    if let Some(setup) = state.setup.as_mut() {
        setup.ready_by_side[SideId::Opponent as usize] = true;
        if setup.ready_by_side[SideId::Player as usize] {
            setup.countdown_seconds_remaining = Some(3);
            setup.opponent_revealed = false;
        }
    }
}

pub fn complete_pregame_setup(
    state: &mut GameState,
    active_hand_index: usize,
    bench_hand_indexes: &[usize],
) {
    if state.phase != Phase::Setup {
        return;
    }
    let Some(setup) = state.setup.as_ref() else {
        return;
    };
    if setup.ready_by_side[SideId::Player as usize] || !setup.opening_hands_dealt {
        return;
    }
    let cat = catalog();
    let player = state.side(SideId::Player);
    let Some(&active_card_id) = player.hand.get(active_hand_index) else {
        return;
    };
    if !cat.is_basic_umamusume(active_card_id) {
        return;
    }

    // Dedupe + filter bench indexes, preserving first-occurrence order.
    let mut taken: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    let mut bench_kept: Vec<usize> = Vec::new();
    for &idx in bench_hand_indexes {
        if idx == active_hand_index {
            continue;
        }
        if idx >= player.hand.len() {
            continue;
        }
        if taken.contains(&idx) {
            continue;
        }
        let cid = player.hand[idx];
        if !cat.is_basic_umamusume(cid) {
            continue;
        }
        if bench_kept.len() >= MAX_BENCH {
            break;
        }
        bench_kept.push(idx);
        taken.insert(idx);
    }

    let bench_ids: Vec<CardId> = bench_kept.iter().map(|&i| player.hand[i]).collect();

    let player_mut = state.side_mut(SideId::Player);
    player_mut.active = Some(create_umamusume(active_card_id, 0));
    player_mut.bench.clear();
    for cid in bench_ids {
        let _ = player_mut.bench.try_push(create_umamusume(cid, 0));
    }

    // Filter out the taken indices from the hand.
    let mut all_taken = taken.clone();
    all_taken.insert(active_hand_index);
    let mut new_hand = arrayvec::ArrayVec::new();
    for (i, c) in player_mut.hand.iter().copied().enumerate() {
        if !all_taken.contains(&i) {
            let _ = new_hand.try_push(c);
        }
    }
    player_mut.hand = new_hand;

    state.pending_player_choice = None;
    if let Some(setup) = state.setup.as_mut() {
        setup.ready_by_side[SideId::Player as usize] = true;
        if setup.ready_by_side[SideId::Opponent as usize] {
            setup.countdown_seconds_remaining = Some(3);
            setup.opponent_revealed = false;
        }
    }
}

pub fn tick_setup_countdown(state: &mut GameState) {
    if state.phase != Phase::Setup {
        return;
    }
    let Some(setup) = state.setup.as_ref() else {
        return;
    };
    if !setup.ready_by_side[SideId::Player as usize]
        || !setup.ready_by_side[SideId::Opponent as usize]
    {
        return;
    }
    let remaining = setup.countdown_seconds_remaining;
    match remaining {
        None => {
            if let Some(s) = state.setup.as_mut() {
                s.countdown_seconds_remaining = Some(3);
                s.opponent_revealed = false;
            }
        }
        Some(r) if r > 1 => {
            if let Some(s) = state.setup.as_mut() {
                s.countdown_seconds_remaining = Some(r - 1);
                s.opponent_revealed = false;
            }
        }
        _ => {
            if let Some(s) = state.setup.as_mut() {
                s.countdown_seconds_remaining = Some(0);
                s.opponent_revealed = true;
            }
            state.phase = Phase::Play;
            let first = state.first_player;
            start_turn(state, first, refresh_continuous_effects, true);
        }
    }
}

// ---------------------------------------------------------------------------
// `flipCoin` for the player-ability path (mirror of `engine.ts:579`). One
// `random_float()` call per non-guaranteed-heads invocation.
// ---------------------------------------------------------------------------

pub fn flip_coin_for_side(side: &mut SideState) -> CoinFlipResult {
    if side.guaranteed_coin_flip_heads > 0 {
        side.guaranteed_coin_flip_heads -= 1;
        return CoinFlipResult::Heads;
    }
    if random_float() >= 0.5 {
        CoinFlipResult::Heads
    } else {
        CoinFlipResult::Tails
    }
}

// ---------------------------------------------------------------------------
// Player-side public API (mirror of `engine.ts:171-567`).
//
// MCTS does not call these — they're the human-side legal-move API — but
// completeness keeps the dispatcher honest and gives the `sim-cli` binaries
// the same shape as the TS exports.
// ---------------------------------------------------------------------------

pub fn play_hand_card(state: &mut GameState, hand_index: usize, choices: &PlayChoices) {
    let side_id = SideId::Player;
    if !is_player_turn(state) || state.pending_player_choice.is_some() {
        return;
    }
    let card_id = {
        let side = state.side(side_id);
        match side.hand.get(hand_index).copied() {
            Some(c) => c,
            None => return,
        }
    };
    let cat = catalog();
    let card = match cat.get(card_id).cloned() {
        Some(c) => c,
        None => return,
    };
    let play = {
        let side = state.side(side_id);
        get_playable_action(state, side, card_id)
    };
    let play_kind = match play {
        PlayActionOutcome::CanPlay(k) => k,
        PlayActionOutcome::CannotPlay { .. } => return,
    };

    // Pre-flight validations from TS:
    match (&card, &play_kind, &choices.umamusume_target_uid) {
        (Card::Umamusume(c), PlayActionKind::Evolve { .. }, Some(uid)) => {
            let valid = {
                let side = state.side(side_id);
                find_own_umamusume_by_uid(side, *uid)
                    .map(|u| is_valid_evolution_target(state, side_id, u, c))
                    .unwrap_or(false)
            };
            if !valid {
                return;
            }
        }
        (_, PlayActionKind::AttachTool { .. }, Some(uid)) => {
            let target_ok = {
                let side = state.side(side_id);
                find_own_umamusume_by_uid(side, *uid)
                    .map(|u| u.tool_card_id.is_none())
                    .unwrap_or(false)
            };
            if !target_ok {
                return;
            }
        }
        _ => {}
    }
    if let Card::Trainer(t) = &card {
        if t.effect.rainbow_uncap_crystal == Some(true) && choices.umamusume_target_uid.is_some() {
            let uid = choices.umamusume_target_uid.unwrap();
            let ok = {
                let side = state.side(side_id);
                get_rainbow_uncap_targets(state, side)
                    .iter()
                    .any(|u| u.uid == uid)
            };
            if !ok {
                return;
            }
            if let Some(hand_idx) = choices.rainbow_evolution_hand_index {
                let target_inst = {
                    let side = state.side(side_id);
                    get_all_umamusume(side)
                        .into_iter()
                        .find(|u| u.uid == uid)
                        .cloned()
                };
                if let Some(target) = target_inst {
                    let options_present = {
                        let side = state.side(side_id);
                        get_rainbow_uncap_evolution_hand_options(side, &target)
                            .iter()
                            .any(|(idx, _)| *idx == hand_idx)
                    };
                    if !options_present {
                        return;
                    }
                }
            }
        }
    }

    // Splice the card out of hand.
    {
        let side = state.side_mut(side_id);
        if hand_index >= side.hand.len() {
            return;
        }
        let _ = side.hand.remove(hand_index);
    }
    let mut shifted = choices.clone();
    shifted.adjust_for_hand_removal(hand_index);
    resolve_card_play(state, side_id, card_id, &card, &play_kind, &shifted);
    normalize_board_state(state);
    refresh_continuous_effects(state);
}

pub fn attach_player_energy(state: &mut GameState, umamusume_uid: Option<u32>) {
    let side_id = SideId::Player;
    if state.pending_player_choice.is_some() {
        return;
    }
    let target_uid = {
        let side = state.side(side_id);
        match umamusume_uid {
            Some(uid) => find_own_umamusume_by_uid(side, uid).map(|u| u.uid),
            None => side.active.as_ref().map(|a| a.uid),
        }
    };
    let Some(target_uid) = target_uid else { return };
    {
        let side = state.side(side_id);
        if find_own_umamusume_by_uid(side, target_uid).is_none() {
            return;
        }
        if !can_attach_energy_to_umamusume(state, side, target_uid) {
            return;
        }
    }
    let side_mut = state.side_mut(side_id);
    flow_attach_energy(side_mut, target_uid);
    normalize_board_state(state);
}

#[allow(clippy::too_many_arguments)]
pub fn player_attack(
    state: &mut GameState,
    attack_target_uid: Option<u32>,
    heal_target_uid: Option<u32>,
    forced_coin: Option<Vec<CoinFlipResult>>,
    evolution_deck_card_index: Option<usize>,
    attack_index: usize,
    discard_hand_index: Option<usize>,
    random_discard_index: Option<usize>,
    switch_target_uid: Option<u32>,
    use_shuffle_self_into_deck: Option<bool>,
) {
    if !can_attack(state, state.side(SideId::Player)) {
        return;
    }
    let mut combat_deps = CombatDeps {
        refresh_continuous_effects: &mut refresh_continuous_effects,
        choose_preferred_active_index: &choose_preferred_active_index,
    };
    perform_attack(
        state,
        SideId::Player,
        &mut combat_deps,
        attack_target_uid,
        heal_target_uid,
        forced_coin,
        evolution_deck_card_index,
        attack_index,
        discard_hand_index,
        random_discard_index,
        switch_target_uid,
        use_shuffle_self_into_deck,
    );
    if let Some(PendingPlayerChoice::PromoteAfterKnockout {
        side_id: SideId::Player,
        resume,
    }) = state.pending_player_choice.as_mut()
    {
        *resume = PromoteResume::FinishOpponentTurn;
        return;
    }
    if state.pending_player_choice.is_some() {
        return;
    }
    if !state.game_over {
        advance_to_next_turn(state);
    }
}

pub fn player_end_turn(state: &mut GameState) {
    if !is_player_turn(state) || state.pending_player_choice.is_some() {
        return;
    }
    advance_to_next_turn(state);
}

pub fn player_use_stadium(state: &mut GameState) {
    let side_id = SideId::Player;
    if !can_use_stadium(state, side_id) {
        return;
    }
    if !use_stadium(state, side_id) {
        return;
    }
    if !state.game_over {
        advance_to_next_turn(state);
    }
}

pub fn player_retreat(
    state: &mut GameState,
    bench_umamusume_uid: Option<u32>,
    discard_energy_types: Option<&[EnergyType]>,
) {
    let side_id = SideId::Player;
    if state.pending_player_choice.is_some() {
        return;
    }
    {
        let side = state.side(side_id);
        if side.active.is_none() || !can_retreat(state, side) {
            return;
        }
    }
    let cost = effective_retreat_cost(state, state.side(side_id));
    let side = state.side_mut(side_id);
    let Some(active) = side.active.as_mut() else {
        return;
    };
    let paid = match discard_energy_types {
        Some(types) => pay_retreat_cost_by_selection(active, types, cost),
        None => {
            pay_retreat_cost(active, cost);
            true
        }
    };
    if !paid {
        return;
    }
    // Resolve promotion target.
    let target_index = match bench_umamusume_uid {
        Some(uid) => side.bench.iter().position(|u| u.uid == uid),
        None => Some(0),
    };
    let Some(target_index) = target_index else {
        return;
    };
    if target_index >= side.bench.len() {
        return;
    }
    let promoted = side.bench.remove(target_index);
    let switched_out = side.active.take();
    if let Some(mut out) = switched_out {
        clear_special_conditions(&mut out);
        let _ = side.bench.try_push(out);
    }
    side.active = Some(promoted);
    side.used_retreat_this_turn = true;
    normalize_board_state(state);
    refresh_continuous_effects(state);
}

pub fn player_surrender(state: &mut GameState) {
    if state.game_over || state.current_side == CurrentSide::Done {
        return;
    }
    state.pending_player_choice = None;
    state.opponent_turn_step = None;
    state.game_over = true;
    state.winner = Some(SideId::Opponent);
    state.current_side = CurrentSide::Done;
}

pub fn opponent_abandoned_match(state: &mut GameState) {
    if state.game_over || state.current_side == CurrentSide::Done {
        return;
    }
    state.pending_player_choice = None;
    state.opponent_turn_step = None;
    state.game_over = true;
    state.winner = Some(SideId::Player);
    state.current_side = CurrentSide::Done;
    state.turn_deadline_ms = None;
}

pub fn timeout_end_turn(state: &mut GameState) {
    if state.phase != Phase::Play || state.game_over || state.pending_player_choice.is_some() {
        return;
    }
    if state.current_side == CurrentSide::Done {
        return;
    }
    advance_to_next_turn(state);
}

pub fn resolve_pending_player_choice(state: &mut GameState, umamusume_uid: u32) {
    let pending = state.pending_player_choice.clone();
    let Some(pending) = pending else { return };
    let pending_side = match &pending {
        PendingPlayerChoice::PromoteAfterKnockout { side_id, .. } => *side_id,
        PendingPlayerChoice::SwitchAfterGust { side_id, .. } => *side_id,
    };
    if pending_side != SideId::Player {
        return;
    }
    let player = state.side_mut(SideId::Player);
    // De-dupe and prune dead bench (TS does this implicitly via filter).
    let mut seen: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    let mut new_bench = arrayvec::ArrayVec::new();
    for u in player.bench.drain(..) {
        if u.hp <= 0 {
            continue;
        }
        if seen.contains(&u.uid) {
            continue;
        }
        seen.insert(u.uid);
        let _ = new_bench.try_push(u);
    }
    player.bench = new_bench;
    let requires_promotion = matches!(pending, PendingPlayerChoice::PromoteAfterKnockout { .. })
        || player.active.is_none()
        || player.active.as_ref().map(|a| a.hp <= 0).unwrap_or(true);

    if requires_promotion {
        if let Some(active) = &player.active {
            if active.hp <= 0 {
                player.active = None;
            }
        }
        let replacement_index = player.bench.iter().position(|u| u.uid == umamusume_uid);
        let Some(idx) = replacement_index else { return };
        let replacement = player.bench.remove(idx);
        player.active = Some(replacement);
    } else {
        let Some(_active) = player.active.as_ref() else {
            return;
        };
        let replacement_index = player.bench.iter().position(|u| u.uid == umamusume_uid);
        let Some(idx) = replacement_index else { return };
        let replacement = player.bench.remove(idx);
        let switched_out = player.active.take();
        if let Some(mut out) = switched_out {
            clear_special_conditions(&mut out);
            let _ = player.bench.try_push(out);
        }
        player.active = Some(replacement);
    }
    state.pending_player_choice = None;
    normalize_board_state(state);
    refresh_continuous_effects(state);

    if let PendingPlayerChoice::PromoteAfterKnockout { resume, .. } = pending {
        if matches!(resume, PromoteResume::FinishOpponentTurn) && !state.game_over {
            advance_to_next_turn(state);
        }
    }
}

// ---------------------------------------------------------------------------
// Per-step AI turn driver (mirror of `engine.ts:303-405`).
// ---------------------------------------------------------------------------

pub fn advance_player_ai_turn_step(
    state: &mut GameState,
    forced_attack_coin_result: Option<Vec<CoinFlipResult>>,
) {
    advance_ai_turn_step(state, SideId::Player, forced_attack_coin_result);
}

pub fn advance_opponent_turn_step(
    state: &mut GameState,
    forced_attack_coin_result: Option<Vec<CoinFlipResult>>,
) {
    advance_ai_turn_step(state, SideId::Opponent, forced_attack_coin_result);
}

fn advance_ai_turn_step(
    state: &mut GameState,
    acting_side_id: SideId,
    forced_attack_coin_result: Option<Vec<CoinFlipResult>>,
) {
    if state.phase != Phase::Play
        || state.pending_player_choice.is_some()
        || state.game_over
        || state.current_side != CurrentSide::from_side(acting_side_id)
    {
        return;
    }
    if state.side(acting_side_id).active.is_none() {
        return;
    }

    let trainer_before_resume = if acting_side_id == SideId::Opponent {
        PendingSwitchAfterGustResume::ResumeOpponentAfterFirstTrainerPass
    } else {
        PendingSwitchAfterGustResume::None
    };
    let trainer_after_resume = if acting_side_id == SideId::Opponent {
        PendingSwitchAfterGustResume::ResumeOpponentAfterSecondTrainerPass
    } else {
        PendingSwitchAfterGustResume::None
    };
    let mut ability_phase_resolved = false;

    for _ in 0..8 {
        let step = state
            .opponent_turn_step
            .unwrap_or(OpponentTurnStep::Bench);
        match step {
            OpponentTurnStep::Bench => {
                if ai_play_one_basic(state, acting_side_id) {
                    return;
                }
                state.opponent_turn_step = Some(OpponentTurnStep::TrainerBefore);
            }
            OpponentTurnStep::TrainerBefore => {
                if ai_play_one_basic(state, acting_side_id) {
                    return;
                }
                let mut deps = AiTrainerDeps {
                    refresh_continuous_effects: &mut refresh_continuous_effects,
                };
                if ai_play_one_trainer(state, acting_side_id, trainer_before_resume, &mut deps) {
                    return;
                }
                state.opponent_turn_step = Some(OpponentTurnStep::Evolve);
            }
            OpponentTurnStep::Evolve => {
                if ai_evolve_one(state, acting_side_id) {
                    refresh_continuous_effects(state);
                    return;
                }
                state.opponent_turn_step = Some(OpponentTurnStep::Attach);
            }
            OpponentTurnStep::Attach => {
                if ai_attach_one_energy(state, acting_side_id) {
                    return;
                }
                state.opponent_turn_step = Some(OpponentTurnStep::TrainerAfter);
            }
            OpponentTurnStep::TrainerAfter => {
                if ai_play_one_basic(state, acting_side_id) {
                    return;
                }
                let mut deps = AiTrainerDeps {
                    refresh_continuous_effects: &mut refresh_continuous_effects,
                };
                if ai_play_one_trainer(state, acting_side_id, trainer_after_resume, &mut deps) {
                    return;
                }
                state.opponent_turn_step = Some(OpponentTurnStep::Ability);
            }
            OpponentTurnStep::Ability => {
                refresh_continuous_effects(state);
                let turn_goal = choose_ai_turn_goal(state, state.side(acting_side_id));
                let mut deps = AiCombatDepsAi {
                    refresh_continuous_effects: &mut refresh_continuous_effects,
                    choose_preferred_active_index: &choose_preferred_active_index,
                };
                if ai_use_one_ability(state, acting_side_id, &mut deps, turn_goal) {
                    return;
                }
                ability_phase_resolved = true;
                state.opponent_turn_step = Some(OpponentTurnStep::Attack);
            }
            OpponentTurnStep::Attack => {
                refresh_continuous_effects(state);
                if !ability_phase_resolved {
                    let turn_goal = choose_ai_turn_goal(state, state.side(acting_side_id));
                    let mut deps = AiCombatDepsAi {
                        refresh_continuous_effects: &mut refresh_continuous_effects,
                        choose_preferred_active_index: &choose_preferred_active_index,
                    };
                    if ai_use_one_ability(state, acting_side_id, &mut deps, turn_goal) {
                        return;
                    }
                }
                let mut deps = AiCombatDepsAi {
                    refresh_continuous_effects: &mut refresh_continuous_effects,
                    choose_preferred_active_index: &choose_preferred_active_index,
                };
                let combat = ai_resolve_combat_decision(
                    state,
                    acting_side_id,
                    forced_attack_coin_result.clone(),
                    &mut deps,
                );
                if !combat.resolved {
                    return;
                }
                if combat.did_retreat == Some(true) {
                    return;
                }
                if combat.used_attack {
                    if state.pending_player_choice.is_some() {
                        state.opponent_turn_step = Some(OpponentTurnStep::Finish);
                        return;
                    }
                } else if can_use_stadium(state, acting_side_id)
                    && use_stadium(state, acting_side_id)
                {
                    state.opponent_turn_step = None;
                    if !state.game_over {
                        advance_to_next_turn(state);
                    }
                    return;
                } else {
                    // Mirror TS engine.ts:391 — emit the "did not attack"
                    // log line that turn_plan::has_consecutive_no_attack_turns
                    // matches.
                    let title = state.side(acting_side_id).title.clone();
                    crate::core::log::log(state, format!("{} did not attack.", title));
                }
                state.opponent_turn_step = None;
                if !state.game_over {
                    advance_to_next_turn(state);
                }
                return;
            }
            OpponentTurnStep::Finish => {
                state.opponent_turn_step = None;
                if !state.game_over {
                    advance_to_next_turn(state);
                }
                return;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// `use_player_ability` (mirror of `engine.ts:432-567`). Ported for
// completeness; not used by MCTS directly.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn use_player_ability(
    state: &mut GameState,
    ability_umamusume_uid: u32,
    source_umamusume_uid: u32,
    selected_energy_type: Option<EnergyType>,
    discard_hand_index: Option<usize>,
    opponent_target_umamusume_uid: Option<u32>,
) {
    let side_id = SideId::Player;
    if !can_use_umamusume_ability(state, state.side(side_id), ability_umamusume_uid) {
        return;
    }
    let ability_inst = {
        let side = state.side(side_id);
        find_own_umamusume_by_uid(side, ability_umamusume_uid).cloned()
    };
    let Some(ability_inst) = ability_inst else {
        return;
    };
    let ability = match get_umamusume_ability(state, side_id, &ability_inst) {
        Some(a) => a.clone(),
        None => return,
    };

    if ability.move_benched_energy_to_active.is_some() {
        if state.side(side_id).active.is_none() {
            return;
        }
        let energy_types = get_ability_move_energy_types(Some(&ability));
        let source_uid_for_energy = source_umamusume_uid;
        let source_idx = state
            .side(side_id)
            .bench
            .iter()
            .position(|u| u.uid == source_uid_for_energy);
        let Some(source_idx) = source_idx else {
            return;
        };
        // Determine energy type — choose based on TS rules.
        let available_types: Vec<EnergyType> = {
            let source = &state.side(side_id).bench[source_idx];
            energy_types
                .iter()
                .copied()
                .filter(|t| source.energies[*t as usize] > 0)
                .collect()
        };
        let chosen = match selected_energy_type {
            Some(t) if available_types.iter().any(|x| *x == t) => Some(t),
            None if available_types.len() == 1 => Some(available_types[0]),
            _ => None,
        };
        let Some(energy_type) = chosen else { return };
        let side = state.side_mut(side_id);
        side.bench[source_idx].energies[energy_type as usize] -= 1;
        if let Some(active) = side.active.as_mut() {
            active.energies[energy_type as usize] += 1;
        }
        mark_ability_used(side, &ability_inst, &ability.name);
        return;
    }

    if let Some(cfg) = &ability.coin_flip_draw_or_active_damage_counter {
        if state.side(side_id).active.is_none() {
            return;
        }
        let heads = {
            let side = state.side_mut(side_id);
            flip_coin_for_side(side) == CoinFlipResult::Heads
        };
        mark_ability_used(state.side_mut(side_id), &ability_inst, &ability.name);
        if heads {
            let _ = draw_cards(state.side_mut(side_id), cfg.draw as u32);
            return;
        }
        let damage = cfg.damage_on_tails;
        let active_uid = state.side(side_id).active.as_ref().map(|a| a.uid);
        let Some(active_uid) = active_uid else { return };
        if let Some(active) = state.side_mut(side_id).active.as_mut() {
            active.hp = (active.hp - damage).max(0);
            active.took_damage_this_turn = damage > 0;
        }
        let active_hp = state.side(side_id).active.as_ref().map(|a| a.hp).unwrap_or(0);
        if active_hp <= 0 {
            let scoring = side_id.opposite();
            let active_inst = state.side(side_id).active.clone();
            if let Some(inst) = active_inst {
                let _ = knock_out_umamusume(
                    state,
                    scoring,
                    side_id,
                    &inst,
                    &choose_preferred_active_index,
                );
                if !state.game_over {
                    refresh_continuous_effects(state);
                }
            }
            let _ = active_uid;
            return;
        }
        normalize_board_state(state);
        refresh_continuous_effects(state);
        return;
    }

    if let Some(cfg) = &ability.discard_to_draw {
        if state.side(side_id).hand.len() < cfg.discard as usize {
            return;
        }
        let resolved_discard_index = match discard_hand_index {
            Some(idx) if idx < state.side(side_id).hand.len() => idx,
            _ => 0,
        };
        let discarded = {
            let side = state.side_mut(side_id);
            if resolved_discard_index >= side.hand.len() {
                return;
            }
            let cid = side.hand.remove(resolved_discard_index);
            let _ = side.discard.try_push(cid);
            cid
        };
        let _ = draw_cards(state.side_mut(side_id), cfg.draw as u32);
        let _ = discarded;
        mark_ability_used(state.side_mut(side_id), &ability_inst, &ability.name);
        return;
    }

    if let Some(damage) = ability.damage_opponent {
        let opponent_id = side_id.opposite();
        let target_uid = if ability.damage_opponent_target == Some(AttackTarget::Any) {
            opponent_target_umamusume_uid.or_else(|| {
                state
                    .side(opponent_id)
                    .active
                    .as_ref()
                    .map(|a| a.uid)
            })
        } else {
            state.side(opponent_id).active.as_ref().map(|a| a.uid)
        };
        let Some(target_uid) = target_uid else { return };
        // Pre-check discard-energy payment.
        if let Some(disc) = &ability.discard_energy {
            let can_pay = EnergyType::ALL.iter().all(|t| {
                let want = disc.get(*t) as u16;
                ability_inst.energies[*t as usize] >= want
            });
            if !can_pay {
                return;
            }
        }
        // Apply damage.
        let target_inst = {
            let opp = state.side(opponent_id);
            get_all_umamusume(opp)
                .into_iter()
                .find(|u| u.uid == target_uid)
                .cloned()
        };
        let Some(_) = target_inst else { return };
        {
            let opp = state.side_mut(opponent_id);
            if let Some(a) = opp.active.as_mut() {
                if a.uid == target_uid {
                    a.hp = (a.hp - damage).max(0);
                    a.took_damage_this_turn = damage > 0;
                }
            }
            for b in opp.bench.iter_mut() {
                if b.uid == target_uid {
                    b.hp = (b.hp - damage).max(0);
                    b.took_damage_this_turn = damage > 0;
                }
            }
        }
        if let Some(disc) = &ability.discard_energy {
            let side = state.side_mut(side_id);
            // Mutate the ability source instance.
            let touch = |u: &mut UmamusumeInstance| {
                for t in EnergyType::ALL {
                    let need = disc.get(t) as u16;
                    if need == 0 {
                        continue;
                    }
                    let have = u.energies[t as usize];
                    let take = have.min(need);
                    u.energies[t as usize] = have - take;
                }
            };
            if let Some(active) = side.active.as_mut() {
                if active.uid == ability_inst.uid {
                    touch(active);
                }
            }
            for b in side.bench.iter_mut() {
                if b.uid == ability_inst.uid {
                    touch(b);
                }
            }
        }
        mark_ability_used(state.side_mut(side_id), &ability_inst, &ability.name);
        // Knock out check.
        let target_dead = {
            let opp = state.side(opponent_id);
            get_all_umamusume(opp)
                .into_iter()
                .find(|u| u.uid == target_uid)
                .map(|u| u.hp <= 0)
                .unwrap_or(false)
        };
        if target_dead {
            let target_inst = {
                let opp = state.side(opponent_id);
                get_all_umamusume(opp)
                    .into_iter()
                    .find(|u| u.uid == target_uid)
                    .cloned()
            };
            if let Some(inst) = target_inst {
                let _ = knock_out_umamusume(
                    state,
                    side_id,
                    opponent_id,
                    &inst,
                    &choose_preferred_active_index,
                );
                if !state.game_over {
                    refresh_continuous_effects(state);
                }
            }
        }
        return;
    }

    if let Some(amount) = ability.shuffle_random_discard_into_deck {
        let count = (amount as usize).min(state.side(side_id).discard.len());
        if count == 0 {
            return;
        }
        let mut picked = Vec::with_capacity(count);
        for _ in 0..count {
            let len = state.side(side_id).discard.len();
            if len == 0 {
                break;
            }
            let idx = random_int(len as u32) as usize;
            let cid = state.side_mut(side_id).discard.remove(idx);
            picked.push(cid);
        }
        // Shuffle deck + picked.
        let combined: Vec<CardId> = state
            .side(side_id)
            .deck
            .iter()
            .copied()
            .chain(picked.into_iter())
            .collect();
        let shuffled = shuffle(&combined);
        let side = state.side_mut(side_id);
        side.deck.clear();
        for c in shuffled {
            let _ = side.deck.try_push(c);
        }
        mark_ability_used(side, &ability_inst, &ability.name);
        let _ = selected_energy_type;
    }
}

// ---------------------------------------------------------------------------
// MCTS-support helpers (mirror of `evaluateModelVsHeuristic.ts:1129+`).
// ---------------------------------------------------------------------------

/// `evaluateModelVsHeuristic.ts:1368` `getForcedAttackCoinResults`. Returns
/// the forced coin-flip sequence for an upcoming attack, consuming RNG via
/// `random_float()` for each flip beyond the side's guaranteed-heads count.
///
/// Must be called inside a `with_rng` scope.
pub fn get_forced_attack_coin_results(state: &GameState) -> Option<Vec<CoinFlipResult>> {
    if state.phase != Phase::Play {
        return None;
    }
    let current_side = state.current_side.as_side()?;
    if state.opponent_turn_step != Some(OpponentTurnStep::Attack) {
        return None;
    }
    let side = state.side(current_side);
    let active = side.active.as_ref()?;
    if active
        .special_conditions
        .iter()
        .any(|c| *c == crate::core::constants::SpecialCondition::Paralysed)
    {
        return None;
    }
    if active.attack_blocked_until_own_turn == Some(state.turns_taken_by_side[current_side as usize])
    {
        return None;
    }
    let cat = catalog();
    let card = match cat.get(active.card_id) {
        Some(Card::Umamusume(u)) => u,
        _ => return None,
    };
    let mut max_flip_count: u32 = 0;
    for attack in &card.attacks {
        if !has_enough_energy_for_attack(active, &attack.cost) {
            continue;
        }
        let flips = attack
            .knock_out_active_if_all_coin_heads
            .map(|v| v as u32)
            .unwrap_or_else(|| {
                if attack.coin_bonus.is_some()
                    || attack.draw_on_heads.is_some()
                    || attack.discard_random_opponent_hand_on_heads.is_some()
                {
                    1
                } else {
                    0
                }
            });
        if flips > max_flip_count {
            max_flip_count = flips;
        }
    }
    if max_flip_count == 0 {
        return None;
    }
    let mut results: Vec<CoinFlipResult> = Vec::with_capacity(max_flip_count as usize);
    for i in 0..max_flip_count {
        if (i as u8) < side.guaranteed_coin_flip_heads {
            results.push(CoinFlipResult::Heads);
        } else if random_float() >= 0.5 {
            results.push(CoinFlipResult::Heads);
        } else {
            results.push(CoinFlipResult::Tails);
        }
    }
    Some(results)
}

fn has_enough_energy_for_attack(u: &UmamusumeInstance, cost: &EnergyCost) -> bool {
    let mut typed_required: i32 = 0;
    for t in EnergyType::ALL {
        if t == EnergyType::Colorless {
            continue;
        }
        let need = cost.get(t) as i32;
        let have = u.energies[t as usize] as i32;
        let short = (need - have).max(0);
        typed_required += short;
    }
    let total_required: i32 = EnergyType::ALL.iter().map(|t| cost.get(*t) as i32).sum();
    let total_attached: i32 = u.energies.iter().map(|n| *n as i32).sum();
    typed_required == 0 && total_attached >= total_required
}

/// Structural hash of the engine state — replaces the TS string fingerprint
/// for MCTS's no-progress loop break. Bit-identity with TS is not required
/// because both the recompute-per-step and the carry-forward versions in
/// `mcts.ts` only compare two Rust-side hashes for equality.
pub fn state_hash(state: &GameState) -> String {
    let packed = crate::core::packed::pack(state);
    format!("{:032x}", crate::fingerprint::fingerprint(&packed))
}

/// `evaluateModelVsHeuristic.ts:1129` `advanceModeledTurnStep`. Applies one
/// chosen action to the state, mirroring the TS dispatch by `action.kind`.
pub fn advance_modeled_turn_step(
    state: &GameState,
    side_id: SideId,
    action: &LegalAiAction,
    forced_attack_coin_result: Option<Vec<CoinFlipResult>>,
) -> GameState {
    let mut next = state.clone();
    if next.phase != Phase::Play
        || next.pending_player_choice.is_some()
        || next.game_over
        || next.current_side != CurrentSide::from_side(side_id)
    {
        return next;
    }
    if next.side(side_id).active.is_none() {
        return next;
    }

    let kind = action.kind.as_str();
    match kind {
        "pass" => {
            advance_modeled_phase(&mut next, side_id);
        }
        "playBasic" | "playTrainer" => {
            let hand_index = read_usize(&action.payload, "handIndex").unwrap_or(usize::MAX);
            let choices = read_play_choices(&action.payload);
            play_selected_hand_card(&mut next, side_id, hand_index, &choices);
        }
        "evolve" => {
            let hand_index = read_usize(&action.payload, "handIndex").unwrap_or(usize::MAX);
            let target_uid = read_u32(&action.payload, "targetUid").unwrap_or(0);
            let choices = PlayChoices {
                discard_hand_index: None,
                deck_card_index: None,
                umamusume_target_uid: Some(target_uid),
                rainbow_evolution_hand_index: None,
            };
            play_selected_hand_card(&mut next, side_id, hand_index, &choices);
        }
        "attachEnergy" => {
            let target_uid = read_u32(&action.payload, "targetUid").unwrap_or(0);
            let target_ok = {
                let side = next.side(side_id);
                find_own_umamusume_by_uid(side, target_uid).is_some()
                    && can_attach_energy_to_umamusume(&next, side, target_uid)
            };
            if target_ok {
                let side = next.side_mut(side_id);
                flow_attach_energy(side, target_uid);
                normalize_board_state(&mut next);
            }
        }
        "useAbility" => {
            let used = use_selected_ability(&mut next, side_id, &action.payload);
            if !used {
                next.opponent_turn_step = Some(OpponentTurnStep::Attack);
            }
        }
        "attack" | "retreatAttack" | "retreat" | "endTurn" => {
            let decision = decode_combat_decision(&action.payload);
            resolve_selected_combat(&mut next, side_id, decision, forced_attack_coin_result);
        }
        "useStadium" => {
            if can_use_stadium(&next, side_id) && use_stadium(&mut next, side_id) {
                finish_turn(&mut next);
            }
        }
        _ => {
            advance_modeled_phase(&mut next, side_id);
        }
    }
    next
}

fn advance_modeled_phase(state: &mut GameState, side_id: SideId) {
    let phase = crate::policy::phase::get_ai_phase(state, side_id);
    let next = match phase {
        AiPhase::Bench => Some(OpponentTurnStep::TrainerBefore),
        AiPhase::TrainerBefore => Some(OpponentTurnStep::Evolve),
        AiPhase::Evolve => Some(OpponentTurnStep::Attach),
        AiPhase::Attach => Some(OpponentTurnStep::TrainerAfter),
        AiPhase::TrainerAfter => Some(OpponentTurnStep::Ability),
        AiPhase::Ability => Some(OpponentTurnStep::Attack),
        _ => None,
    };
    if let Some(step) = next {
        state.opponent_turn_step = Some(step);
    } else {
        finish_turn(state);
    }
}

fn finish_turn(state: &mut GameState) {
    state.opponent_turn_step = None;
    if !state.game_over {
        end_turn(
            state,
            |s, side_id| start_turn(s, side_id, refresh_continuous_effects, false),
            refresh_continuous_effects,
        );
    }
}

fn play_selected_hand_card(
    state: &mut GameState,
    side_id: SideId,
    hand_index: usize,
    choices: &PlayChoices,
) {
    let card_id = match state.side(side_id).hand.get(hand_index).copied() {
        Some(c) => c,
        None => return,
    };
    let cat = catalog();
    let card = match cat.get(card_id).cloned() {
        Some(c) => c,
        None => return,
    };
    let play = get_playable_action(state, state.side(side_id), card_id);
    let kind = match play {
        PlayActionOutcome::CanPlay(k) => k,
        PlayActionOutcome::CannotPlay { .. } => return,
    };
    {
        let side = state.side_mut(side_id);
        if hand_index >= side.hand.len() {
            return;
        }
        let _ = side.hand.remove(hand_index);
    }
    let mut shifted = choices.clone();
    shifted.adjust_for_hand_removal(hand_index);
    resolve_card_play(state, side_id, card_id, &card, &kind, &shifted);
    normalize_board_state(state);
    refresh_continuous_effects(state);
}

fn use_selected_ability(state: &mut GameState, side_id: SideId, payload: &serde_json::Value) -> bool {
    let source_uid = match read_u32(payload, "sourceUid") {
        Some(v) => v,
        None => return false,
    };
    let source_inst = {
        let side = state.side(side_id);
        get_all_umamusume(side)
            .into_iter()
            .find(|u| u.uid == source_uid)
            .cloned()
    };
    let Some(source_inst) = source_inst else {
        return false;
    };
    if !can_use_umamusume_ability(state, state.side(side_id), source_uid) {
        return false;
    }
    let ability = match get_umamusume_ability(state, side_id, &source_inst) {
        Some(a) => a.clone(),
        None => return false,
    };

    if let Some(damage) = ability.damage_opponent {
        let opponent_id = side_id.opposite();
        let target_uid = if ability.damage_opponent_target == Some(AttackTarget::Any) {
            read_u32(payload, "targetUid")
                .or_else(|| state.side(opponent_id).active.as_ref().map(|a| a.uid))
        } else {
            state.side(opponent_id).active.as_ref().map(|a| a.uid)
        };
        let Some(target_uid) = target_uid else {
            return false;
        };
        if let Some(disc) = &ability.discard_energy {
            let can_pay = EnergyType::ALL.iter().all(|t| {
                let want = disc.get(*t) as u16;
                source_inst.energies[*t as usize] >= want
            });
            if !can_pay {
                return false;
            }
            let side = state.side_mut(side_id);
            let touch = |u: &mut UmamusumeInstance| {
                for t in EnergyType::ALL {
                    let want = disc.get(t) as u16;
                    if want == 0 {
                        continue;
                    }
                    let have = u.energies[t as usize];
                    u.energies[t as usize] = have.saturating_sub(want);
                }
            };
            if let Some(a) = side.active.as_mut() {
                if a.uid == source_uid {
                    touch(a);
                }
            }
            for b in side.bench.iter_mut() {
                if b.uid == source_uid {
                    touch(b);
                }
            }
        }
        {
            let opp = state.side_mut(opponent_id);
            if let Some(a) = opp.active.as_mut() {
                if a.uid == target_uid {
                    a.hp = (a.hp - damage).max(0);
                    a.took_damage_this_turn = damage > 0;
                }
            }
            for b in opp.bench.iter_mut() {
                if b.uid == target_uid {
                    b.hp = (b.hp - damage).max(0);
                    b.took_damage_this_turn = damage > 0;
                }
            }
        }
        mark_ability_used(state.side_mut(side_id), &source_inst, &ability.name);
        let target_dead = {
            let opp = state.side(opponent_id);
            get_all_umamusume(opp)
                .into_iter()
                .find(|u| u.uid == target_uid)
                .map(|u| u.hp <= 0)
                .unwrap_or(false)
        };
        if target_dead {
            let target_inst = {
                let opp = state.side(opponent_id);
                get_all_umamusume(opp)
                    .into_iter()
                    .find(|u| u.uid == target_uid)
                    .cloned()
            };
            if let Some(inst) = target_inst {
                let _ = knock_out_umamusume(
                    state,
                    side_id,
                    opponent_id,
                    &inst,
                    &choose_preferred_active_index,
                );
                if !state.game_over {
                    refresh_continuous_effects(state);
                }
            }
        }
        return true;
    }

    if ability.move_benched_energy_to_active.is_some() {
        if state.side(side_id).active.is_none() {
            return false;
        }
        let energy_source_uid = match read_u32(payload, "energySourceUid") {
            Some(v) => v,
            None => return false,
        };
        let energy_type_str = payload.get("energyType").and_then(|v| v.as_str());
        let Some(energy_type) = energy_type_str.and_then(parse_energy_type) else {
            return false;
        };
        if !get_ability_move_energy_types(Some(&ability))
            .iter()
            .any(|t| *t == energy_type)
        {
            return false;
        }
        let source_idx = state
            .side(side_id)
            .bench
            .iter()
            .position(|u| u.uid == energy_source_uid);
        let Some(source_idx) = source_idx else {
            return false;
        };
        let have = state.side(side_id).bench[source_idx].energies[energy_type as usize];
        if have == 0 {
            return false;
        }
        let side = state.side_mut(side_id);
        side.bench[source_idx].energies[energy_type as usize] -= 1;
        if let Some(active) = side.active.as_mut() {
            active.energies[energy_type as usize] += 1;
        }
        mark_ability_used(side, &source_inst, &ability.name);
        return true;
    }

    if let Some(cfg) = &ability.discard_to_draw {
        if state.side(side_id).hand.len() < cfg.discard as usize {
            return false;
        }
        let raw_idx =
            read_usize(payload, "discardHandIndex").unwrap_or(0);
        if raw_idx >= state.side(side_id).hand.len() {
            return false;
        }
        for _ in 0..cfg.discard {
            let len = state.side(side_id).hand.len();
            if len == 0 {
                break;
            }
            let take_idx = raw_idx.min(len - 1);
            let cid = state.side_mut(side_id).hand.remove(take_idx);
            let _ = state.side_mut(side_id).discard.try_push(cid);
        }
        let _ = draw_cards(state.side_mut(side_id), cfg.draw as u32);
        mark_ability_used(state.side_mut(side_id), &source_inst, &ability.name);
        return true;
    }

    if ability.coin_flip_draw_or_active_damage_counter.is_some() {
        let ai_difficulty = state.ai_difficulty;
        let mut rng = || random_float();
        let mut combat_deps = CombatDeps {
            refresh_continuous_effects: &mut refresh_continuous_effects,
            choose_preferred_active_index: &choose_preferred_active_index,
        };
        return ai_use_coin_flip_draw_ability(
            state,
            side_id,
            source_uid,
            &mut rng,
            &mut combat_deps,
            ai_difficulty,
        );
    }

    if let Some(amount) = ability.shuffle_random_discard_into_deck {
        let count = (amount as usize).min(state.side(side_id).discard.len());
        if count == 0 {
            return false;
        }
        let mut picked = Vec::with_capacity(count);
        for _ in 0..count {
            let len = state.side(side_id).discard.len();
            if len == 0 {
                break;
            }
            // TS: Math.floor(rng.next() * length)
            let r = random_float();
            let idx = ((r * len as f64).floor() as usize).min(len - 1);
            let cid = state.side_mut(side_id).discard.remove(idx);
            picked.push(cid);
        }
        let side = state.side_mut(side_id);
        for c in picked {
            let _ = side.deck.try_push(c);
        }
        mark_ability_used(side, &source_inst, &ability.name);
        return true;
    }

    // damageOpponent fallback already handled above; move-benched-energy
    // fallback wired through ai_use_move_benched_energy_ability in the AI
    // pipeline — the evaluator path doesn't re-enter it here.
    let _ = ai_use_move_benched_energy_ability;
    let _ = ai_use_damage_ability;
    let _ = estimate_attack_damage_output;
    let _ = with_energy_shift;
    let _: Option<AbilityHeuristicDeps<'_>> = None;
    false
}

fn resolve_selected_combat(
    state: &mut GameState,
    side_id: SideId,
    decision: Option<DecodedCombatDecision>,
    forced_attack_coin_result: Option<Vec<CoinFlipResult>>,
) {
    let Some(decision) = decision else {
        finish_turn(state);
        return;
    };
    if !decision.attack {
        // EndTurn-shaped decision.
        finish_turn(state);
        return;
    }
    if let Some(retreat_uid) = decision.retreat_target_uid {
        let retreated = ai_retreat_to_target(state, side_id, retreat_uid);
        if !retreated {
            return;
        }
        if decision.kind != "attack" {
            return;
        }
    }
    let mut deps = CombatDeps {
        refresh_continuous_effects: &mut refresh_continuous_effects,
        choose_preferred_active_index: &choose_preferred_active_index,
    };
    perform_attack(
        state,
        side_id,
        &mut deps,
        decision.attack_target_uid,
        decision.heal_target_uid,
        forced_attack_coin_result,
        decision.evolution_deck_card_index,
        decision.attack_index.unwrap_or(0),
        decision.discard_hand_index,
        decision.random_discard_index,
        decision.switch_target_uid,
        decision.use_shuffle_self_into_deck,
    );
    if state.pending_player_choice.is_some() {
        return;
    }
    if !state.game_over {
        finish_turn(state);
    }
}

struct DecodedCombatDecision {
    kind: String,
    attack: bool,
    retreat_target_uid: Option<u32>,
    attack_target_uid: Option<u32>,
    heal_target_uid: Option<u32>,
    attack_index: Option<usize>,
    discard_hand_index: Option<usize>,
    evolution_deck_card_index: Option<usize>,
    random_discard_index: Option<usize>,
    switch_target_uid: Option<u32>,
    use_shuffle_self_into_deck: Option<bool>,
}

fn decode_combat_decision(payload: &serde_json::Value) -> Option<DecodedCombatDecision> {
    let decision = payload.get("decision")?;
    let kind = decision.get("kind").and_then(|v| v.as_str())?.to_string();
    Some(DecodedCombatDecision {
        attack: kind == "attack",
        retreat_target_uid: decision
            .get("retreatTargetUid")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32),
        attack_target_uid: decision
            .get("attackTargetUid")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32),
        heal_target_uid: decision
            .get("healTargetUid")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32),
        attack_index: decision
            .get("attackIndex")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize),
        discard_hand_index: decision
            .get("discardHandIndex")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize),
        evolution_deck_card_index: decision
            .get("evolutionDeckCardIndex")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize),
        random_discard_index: decision
            .get("randomDiscardIndex")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize),
        switch_target_uid: decision
            .get("switchTargetUid")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32),
        use_shuffle_self_into_deck: decision
            .get("useShuffleSelfIntoDeck")
            .and_then(|v| v.as_bool()),
        kind,
    })
}

fn read_play_choices(payload: &serde_json::Value) -> PlayChoices {
    let mut out = PlayChoices::default();
    let Some(choices) = payload.get("choices") else {
        return out;
    };
    if let Some(v) = choices.get("discardHandIndex").and_then(|v| v.as_u64()) {
        out.discard_hand_index = Some(v as usize);
    }
    if let Some(v) = choices.get("deckCardIndex").and_then(|v| v.as_u64()) {
        out.deck_card_index = Some(v as usize);
    }
    if let Some(v) = choices.get("umamusumeTargetUid").and_then(|v| v.as_u64()) {
        out.umamusume_target_uid = Some(v as u32);
    }
    if let Some(v) = choices
        .get("rainbowEvolutionHandIndex")
        .and_then(|v| v.as_u64())
    {
        out.rainbow_evolution_hand_index = Some(v as usize);
    }
    out
}

fn read_u32(payload: &serde_json::Value, key: &str) -> Option<u32> {
    payload.get(key).and_then(|v| v.as_u64()).map(|v| v as u32)
}

fn read_usize(payload: &serde_json::Value, key: &str) -> Option<usize> {
    payload.get(key).and_then(|v| v.as_u64()).map(|v| v as usize)
}

fn parse_energy_type(s: &str) -> Option<EnergyType> {
    match s {
        "grass" => Some(EnergyType::Grass),
        "fire" => Some(EnergyType::Fire),
        "water" => Some(EnergyType::Water),
        "lightning" => Some(EnergyType::Lightning),
        "psychic" => Some(EnergyType::Psychic),
        "fighting" => Some(EnergyType::Fighting),
        "darkness" => Some(EnergyType::Darkness),
        "steel" => Some(EnergyType::Steel),
        "colorless" => Some(EnergyType::Colorless),
        "dragon" => Some(EnergyType::Dragon),
        _ => None,
    }
}

// Silence unused imports kept for forward-compatibility / parity.
#[allow(dead_code)]
fn _kw_unused() {
    let _ = switch_out_opponent_active;
    let _ = AttackTarget::Active;
    let _ = MAX_POINTS;
    let _ = StadiumState {
        card_id: CardId(0),
        owner: SideId::Player,
    };
    let _ = SwitchResume::None;
    let _: fn(&UmamusumeInstance, EnergyType, i32) -> UmamusumeInstance = with_energy_shift;
}
