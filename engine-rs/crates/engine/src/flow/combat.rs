//! Bit-identical port of `frontend/src/game/engine/flow/combat.ts`.
//!
//! Bit-identity invariants preserved here:
//! - **RNG site order** (see `docs/ai-research/scoping/rust-engine-port-plan.md`
//!   §4): every `randomInt` / `randomFloat` / `shuffle` in the TS source is
//!   replayed in exact order via the thread-local `core::random` provider.
//! - **Integer math**: damage, HP, energy counts are i32 with `.max(0)` /
//!   `.min(max_hp)` clamps.
//! - **Iteration order**: `get_all_umamusume` returns `[active, ...bench]`;
//!   `EnergyType::ALL` is the canonical order for `discardEnergy` / cost
//!   iteration.
//! - **Logging skipped**: `state.log` is excluded from the fingerprint; all
//!   `log(state, ...)` calls in TS are no-ops here.
//! - **`flipCoin`**: forced results consume from the front; a
//!   guaranteed-heads counter decrements AND consumes one forced result.
//!
//! Borrow-checker patterns:
//! - `GameState::sides_mut_for(attacker_id)` splits the sides into
//!   `(attacker, defender)` disjoint mutable refs.
//! - Catalog lookups (`Attack`, `UmamusumeCard`) are snapshotted into local
//!   `Clone` variables before mutable borrows so we never alias the
//!   immutable catalog reference with `&mut` on state.

use crate::core::catalog::{catalog, Card, UmamusumeCard};
use crate::core::constants::{
    CoinFlipResult, EnergyType, SideId, SpecialCondition, MAX_POINTS,
};
use crate::core::effects::{
    Attack, AttackTarget, DamagePerUmamusumeSide, HealTarget, ShuffleSelfIntoDeck,
};
use crate::core::random::{random_float, random_int, shuffle};
use crate::core::state::{
    CurrentSide, GameState, PendingPlayerChoice, PromoteResume, SideState, UmamusumeInstance,
};
use crate::core::umamusume::{
    find_most_damaged_umamusume, find_own_umamusume_by_uid, get_all_umamusume,
};
use crate::flow::ability_rules::get_umamusume_ability;
use crate::flow::evolution::evolve_umamusume;
use crate::flow::special_conditions::clear_special_conditions;
use crate::flow::turn::draw_cards;

/// Combat callback bundle. Mirrors the TS `CombatDeps` shape:
/// `refresh_continuous_effects` is invoked after switches / knockouts;
/// `choose_preferred_active_index` selects the bench card to promote when
/// an AI side must do so.
///
/// Modeled as a struct of `&mut dyn FnMut` so we can re-borrow it in
/// multiple call sites within one `perform_attack` invocation without
/// having to thread two generic type parameters through every helper.
pub struct CombatDeps<'a> {
    pub refresh_continuous_effects: &'a mut dyn FnMut(&mut GameState),
    pub choose_preferred_active_index: &'a dyn Fn(&SideState) -> i32,
}

/// `combat.ts:18` `performAttack`. Argument order matches the TS source.
#[allow(clippy::too_many_arguments)]
pub fn perform_attack(
    state: &mut GameState,
    attacker_id: SideId,
    deps: &mut CombatDeps<'_>,
    attack_target_uid: Option<u32>,
    heal_target_uid: Option<u32>,
    forced_coin_result: Option<Vec<CoinFlipResult>>,
    evolution_deck_card_index: Option<usize>,
    attack_index: usize,
    discard_hand_index: Option<usize>,
    random_discard_index: Option<usize>,
    switch_target_uid: Option<u32>,
    use_shuffle_self_into_deck: Option<bool>,
) {
    let defender_id = attacker_id.opposite();
    let points_before_attacker = state.side(attacker_id).points;
    let points_before_defender = state.side(defender_id).points;

    // Bail if either active is missing — TS does the same.
    if state.side(attacker_id).active.is_none() || state.side(defender_id).active.is_none() {
        return;
    }

    // Snapshot the catalog data — these references stay valid because the
    // catalog is `'static`. The cloned Attack object lets us drop the
    // borrow before we take `&mut GameState`.
    let cat = catalog();
    let attacker_card: UmamusumeCard = {
        let active = state.side(attacker_id).active.as_ref().unwrap();
        match cat.get(active.card_id) {
            Some(Card::Umamusume(u)) => u.clone(),
            _ => return,
        }
    };
    let attack: Attack = match attacker_card
        .attacks
        .get(attack_index)
        .or_else(|| attacker_card.attacks.first())
        .cloned()
    {
        Some(a) => a,
        None => return,
    };

    // Snapshot the starting-active uid for post-attack switch logic.
    let starting_active_uid = state.side(attacker_id).active.as_ref().unwrap().uid;

    // Resolve switch target uid (does not yet remove from bench).
    let switch_target_uid_resolved = resolve_switch_target_uid(
        state,
        attacker_id,
        switch_target_uid,
        &attack,
        deps.choose_preferred_active_index,
    );

    // Resolve attack-target uid: TS uses `defender.active` as fallback.
    let attack_target_uid_resolved: u32 = match attack.target_opponent {
        Some(AttackTarget::Any) => {
            let defender = state.side(defender_id);
            attack_target_uid
                .and_then(|uid| {
                    get_all_umamusume(defender)
                        .into_iter()
                        .find(|u| u.uid == uid)
                        .map(|u| u.uid)
                })
                .unwrap_or_else(|| defender.active.as_ref().unwrap().uid)
        }
        _ => state.side(defender_id).active.as_ref().unwrap().uid,
    };

    // Snapshot the defender card's printed weakness for the weakness check.
    let (defender_weakness_match_type, defender_weakness_amount): (
        crate::core::constants::UmamusumeType,
        i32,
    ) = {
        let defender = state.side(defender_id);
        let target = if defender.active.as_ref().map(|a| a.uid) == Some(attack_target_uid_resolved)
        {
            defender.active.as_ref().unwrap()
        } else if let Some(b) = defender.bench.iter().find(|u| u.uid == attack_target_uid_resolved)
        {
            b
        } else {
            return;
        };
        match cat.get(target.card_id) {
            Some(Card::Umamusume(u)) => (u.weakness.r#type, u.weakness.amount),
            _ => return,
        }
    };

    let non_damaging_attack = is_non_damaging_attack(&attack);

    // ----- Damage calculation -----
    let mut damage: i32 = attack.damage
        + if non_damaging_attack {
            0
        } else {
            state.side(attacker_id).active_attack_damage_bonus as i32
        };
    let mut coin_flip_heads: Option<bool> = None;
    let mut forced_coin_results: Vec<CoinFlipResult> = forced_coin_result.unwrap_or_default();

    if let Some(bonus) = attack.bonus_if_took_damage_last_turn {
        let active = state.side(attacker_id).active.as_ref().unwrap();
        if active.took_damage_last_turn {
            damage += bonus;
        }
    }
    if let Some(d) = &attack.damage_per_attached_energy {
        let active = state.side(attacker_id).active.as_ref().unwrap();
        let bonus_energy_count: i32 = d
            .types
            .iter()
            .map(|t| active.energies[*t as usize] as i32)
            .sum();
        damage += bonus_energy_count * d.amount;
    }
    if let Some(d) = &attack.damage_per_umamusume_in_play {
        let in_play_count = match d.side {
            DamagePerUmamusumeSide::All => {
                get_all_umamusume(state.side(attacker_id)).len() as i32
                    + get_all_umamusume(state.side(defender_id)).len() as i32
            }
            DamagePerUmamusumeSide::Own => {
                get_all_umamusume(state.side(attacker_id)).len() as i32
            }
        };
        damage += in_play_count * d.amount;
    }
    if let Some(bonus) = attack.attack_damage_bonus_if_tool_attached {
        let attacker_has_tool = state
            .side(attacker_id)
            .active
            .as_ref()
            .unwrap()
            .tool_card_id
            .is_some();
        if attacker_has_tool && !are_tools_disabled(state) {
            damage += bonus;
        }
    }
    if switch_target_uid_resolved.is_some() {
        if let Some(sw) = &attack.switch_self_after_attack {
            if let Some(b) = sw.bonus_damage {
                damage += b;
            }
        }
    }
    if let Some(bonus) = attack.attack_damage_bonus_if_discard_hand_card {
        let attacker = state.side_mut(attacker_id);
        if !attacker.hand.is_empty() {
            let requested_discard_index = match discard_hand_index {
                Some(idx) if idx < attacker.hand.len() => Some(idx),
                _ => None,
            };
            let resolved_discard_index = if requested_discard_index.is_some() {
                requested_discard_index
            } else if !state.human_by_side[attacker_id as usize] {
                Some(0usize)
            } else {
                None
            };
            if let Some(resolved) = resolved_discard_index {
                let attacker = state.side_mut(attacker_id);
                let discarded_card_id = attacker.hand.remove(resolved);
                let _ = attacker.discard.try_push(discarded_card_id);
                damage += bonus;
            }
        }
    }
    // Ability conditional attack bonus (energy-thresholded).
    let attacker_active_energy_for_ability_bonus = {
        let active = state.side(attacker_id).active.as_ref().unwrap();
        // The Ability lookup needs &GameState; both calls are immutable
        // here so this is safe.
        let ability = get_umamusume_ability(state, attacker_id, active);
        ability
            .and_then(|a| a.attack_damage_bonus_if_attached_energy.as_ref())
            .map(|c| (c.r#type, c.min, c.amount))
    };
    if !non_damaging_attack {
        if let Some((etype, min, amount)) = attacker_active_energy_for_ability_bonus {
            let active = state.side(attacker_id).active.as_ref().unwrap();
            if (active.energies[etype as usize] as i32) >= min {
                damage += amount;
            }
        }
    }
    if let Some(per_unique) = attack.damage_per_unique_attached_energy {
        let active = state.side(attacker_id).active.as_ref().unwrap();
        let unique_energy_count =
            active.energies.iter().filter(|&&count| count > 0).count() as i32;
        damage += unique_energy_count * per_unique;
    }
    if let Some(per_discard) = attack.attack_damage_bonus_per_discarded_hand_card.clone() {
        let attacker = state.side_mut(attacker_id);
        if !attacker.hand.is_empty() {
            let discard_count =
                (per_discard.max_discard as usize).min(attacker.hand.len()) as i32;
            for _ in 0..discard_count {
                if attacker.hand.is_empty() {
                    break;
                }
                let discarded = attacker.hand.remove(0);
                let _ = attacker.discard.try_push(discarded);
            }
            if discard_count > 0 {
                damage += discard_count * per_discard.bonus_per_card;
            }
        }
    }
    // Evolved-last-turn bonus.
    let evolved_last_turn_bonus_amount = {
        let active = state.side(attacker_id).active.as_ref().unwrap();
        get_umamusume_ability(state, attacker_id, active)
            .and_then(|a| a.attack_damage_bonus_if_evolved_last_turn)
            .unwrap_or(0)
    };
    if !non_damaging_attack && evolved_last_turn_bonus_amount > 0 {
        let active = state.side(attacker_id).active.as_ref().unwrap();
        let evolved_turn = active.evolved_turn;
        let target_turn = state.turn_number.checked_sub(1);
        if evolved_turn.is_some() && evolved_turn == target_turn {
            damage += evolved_last_turn_bonus_amount;
        }
    }

    // Coin-flip bonus (also used by drawOnHeads / discardRandomOpponentHandOnHeads).
    if attack.coin_bonus.is_some()
        || attack.draw_on_heads.is_some()
        || attack.discard_random_opponent_hand_on_heads.is_some()
    {
        let result = flip_coin_with_state(state, attacker_id, &mut forced_coin_results);
        let heads = result == CoinFlipResult::Heads;
        coin_flip_heads = Some(heads);
        if heads {
            if let Some(b) = attack.coin_bonus {
                damage += b;
            }
        }
    }

    // Weakness: applies only if damage > 0 (TS).
    if damage > 0 && defender_weakness_match_type == attacker_card.r#type {
        damage += defender_weakness_amount;
    }

    // Damage reduction.
    let reduction = {
        let defender = state.side(defender_id);
        let target = if defender.active.as_ref().map(|a| a.uid) == Some(attack_target_uid_resolved)
        {
            defender.active.as_ref().unwrap()
        } else {
            defender
                .bench
                .iter()
                .find(|u| u.uid == attack_target_uid_resolved)
                .unwrap()
        };
        attack_damage_reduction_for(state, target).min(damage)
    };
    damage = (damage - reduction).max(0);

    // guaranteeNextCoinFlipHeads — applies before knockOutActiveIfAllCoinHeads.
    if let Some(n) = attack.guarantee_next_coin_flip_heads {
        let attacker = state.side_mut(attacker_id);
        attacker.guaranteed_coin_flip_heads = attacker.guaranteed_coin_flip_heads.saturating_add(n as u8);
    }

    // knockOutActiveIfAllCoinHeads — multiple coin flips, zero target HP if all heads.
    if let Some(n) = attack.knock_out_active_if_all_coin_heads {
        let mut results = Vec::with_capacity(n as usize);
        for _ in 0..n {
            results.push(flip_coin_with_state(state, attacker_id, &mut forced_coin_results));
        }
        if results.iter().all(|r| *r == CoinFlipResult::Heads) {
            let defender = state.side_mut(defender_id);
            if let Some(active) = defender.active.as_mut() {
                if active.uid == attack_target_uid_resolved {
                    active.hp = 0;
                }
            }
            for u in defender.bench.iter_mut() {
                if u.uid == attack_target_uid_resolved {
                    u.hp = 0;
                }
            }
        }
    }

    // Apply main damage to the resolved attack target.
    {
        let defender = state.side_mut(defender_id);
        let target_mut: Option<&mut UmamusumeInstance> =
            if defender.active.as_ref().map(|a| a.uid) == Some(attack_target_uid_resolved) {
                defender.active.as_mut()
            } else {
                defender
                    .bench
                    .iter_mut()
                    .find(|u| u.uid == attack_target_uid_resolved)
            };
        if let Some(target) = target_mut {
            target.hp = (target.hp - damage).max(0);
            if damage > 0 {
                target.took_damage_this_turn = true;
            }
        }
    }

    // Mirror TS combat.ts:124 — emit the "attacked with" log line that
    // turn_plan::has_consecutive_no_attack_turns matches. Only the
    // damage > 0 branch produces this exact prefix.
    if damage > 0 {
        let attacker_card_name = format!(
            "{}'s {}",
            crate::core::labels::format_umamusume_card_name(&attacker_card),
            attack.name
        );
        let actor = crate::core::labels::actor_name(state.side(attacker_id));
        let msg = format!("{} attacked with {} for {} damage.", actor, attacker_card_name, damage);
        crate::core::log::log(state, msg);
    }

    // Counter damage from defender's active tool (only when the attacked
    // target was the defender's active).
    let counter_damage = if damage > 0 {
        let defender = state.side(defender_id);
        if defender.active.as_ref().map(|a| a.uid) == Some(attack_target_uid_resolved) {
            if let Some(active) = &defender.active {
                active_tool_counter_damage(state, active)
            } else {
                0
            }
        } else {
            0
        }
    } else {
        0
    };
    if counter_damage > 0 {
        let attacker = state.side_mut(attacker_id);
        if let Some(active) = attacker.active.as_mut() {
            active.hp = (active.hp - counter_damage).max(0);
            active.took_damage_this_turn = true;
        }
    }

    // preventDamageNextTurn.
    if let Some(prevent) = attack.prevent_damage_next_turn {
        let attacker = state.side_mut(attacker_id);
        if let Some(active) = attacker.active.as_mut() {
            active.next_turn_damage_reduction = active.next_turn_damage_reduction.max(prevent);
        }
    }

    // cannotAttackNextTurn — bind on the starting-active uid (not whoever
    // is active after a switch).
    if attack.cannot_attack_next_turn == Some(true) {
        let until = state.turns_taken_by_side[attacker_id as usize] + 1;
        let attacker = state.side_mut(attacker_id);
        let bind = |u: &mut UmamusumeInstance| {
            u.attack_blocked_until_own_turn = Some(until);
        };
        if let Some(active) = attacker.active.as_mut() {
            if active.uid == starting_active_uid {
                bind(active);
            }
        }
        for b in attacker.bench.iter_mut() {
            if b.uid == starting_active_uid {
                bind(b);
            }
        }
    }

    // Draw.
    if let Some(n) = attack.draw {
        let attacker = state.side_mut(attacker_id);
        let _ = draw_cards(attacker, n as u32);
    }
    // drawOnHeads.
    if let Some(n) = attack.draw_on_heads {
        if coin_flip_heads == Some(true) {
            let attacker = state.side_mut(attacker_id);
            let _ = draw_cards(attacker, n as u32);
        }
    }
    // Heal.
    if let Some(heal) = attack.heal {
        let heal_target_kind = attack.heal_target;
        let target_uid: Option<u32> = match heal_target_kind {
            Some(HealTarget::Self_) => state.side(attacker_id).active.as_ref().map(|a| a.uid),
            Some(HealTarget::Any) => {
                let attacker = state.side(attacker_id);
                let chosen = heal_target_uid
                    .and_then(|uid| find_own_umamusume_by_uid(attacker, uid))
                    .map(|u| u.uid);
                chosen.or_else(|| find_most_damaged_umamusume(attacker).map(|u| u.uid))
            }
            None => find_most_damaged_umamusume(state.side(attacker_id)).map(|u| u.uid),
        };
        if let Some(uid) = target_uid {
            let recover_conditions = attack.recover_special_conditions == Some(true);
            let attacker = state.side_mut(attacker_id);
            let touch = |u: &mut UmamusumeInstance| {
                u.hp = (u.hp + heal).min(u.max_hp);
                if recover_conditions {
                    clear_special_conditions(u);
                }
            };
            if let Some(active) = attacker.active.as_mut() {
                if active.uid == uid {
                    touch(active);
                }
            }
            for b in attacker.bench.iter_mut() {
                if b.uid == uid {
                    touch(b);
                }
            }
        }
    }

    // Inflict special condition (only if target hp > 0).
    if let Some(condition) = attack.inflict_special_condition {
        // Check the resolved attack target's HP after main damage.
        let defender = state.side(defender_id);
        let target_hp = if defender.active.as_ref().map(|a| a.uid) == Some(attack_target_uid_resolved)
        {
            defender.active.as_ref().map(|a| a.hp).unwrap_or(0)
        } else {
            defender
                .bench
                .iter()
                .find(|u| u.uid == attack_target_uid_resolved)
                .map(|u| u.hp)
                .unwrap_or(0)
        };
        if target_hp > 0 {
            apply_special_condition(state, defender_id, attack_target_uid_resolved, condition);
        }
    }

    // discardRandomOpponentHandOnHeads.
    if let Some(d) = attack.discard_random_opponent_hand_on_heads.clone() {
        if coin_flip_heads == Some(true) {
            // Decide whether the optional discard fires.
            let (defender_hand_len, attacker_hp) = {
                let attacker = state.side(attacker_id);
                let defender = state.side(defender_id);
                (
                    defender.hand.len(),
                    attacker.active.as_ref().map(|a| a.hp).unwrap_or(0),
                )
            };
            let should = defender_hand_len > 0 && attacker_hp > d.self_damage;
            if should {
                let idx = random_int(defender_hand_len as u32) as usize;
                let defender = state.side_mut(defender_id);
                let discarded = defender.hand.remove(idx);
                let _ = defender.discard.try_push(discarded);
                let attacker = state.side_mut(attacker_id);
                if let Some(active) = attacker.active.as_mut() {
                    active.hp = (active.hp - d.self_damage).max(0);
                    active.took_damage_this_turn = true;
                }
            }
        }
    }

    // Bench damage.
    if let Some(bench_damage) = attack.bench_damage {
        if bench_damage > 0 {
            let defender = state.side_mut(defender_id);
            for b in defender.bench.iter_mut() {
                b.hp = (b.hp - bench_damage).max(0);
                b.took_damage_this_turn = true;
            }
        }
    }

    // discardEnergy — iterate EnergyType::ALL (matches source-declaration
    // order in TS).
    if let Some(discard_cost) = &attack.discard_energy {
        let attacker = state.side_mut(attacker_id);
        if let Some(active) = attacker.active.as_mut() {
            for energy_type in EnergyType::ALL.iter().copied() {
                let amount = discard_cost.get(energy_type) as i32;
                if amount > 0 {
                    let cur = active.energies[energy_type as usize] as i32;
                    let next = (cur - amount).max(0);
                    active.energies[energy_type as usize] = next as u16;
                }
            }
        }
    }

    // evolveFromDeck.
    if attack.evolve_from_deck == Some(true) && state.side(attacker_id).active.is_some() {
        evolve_active_from_deck(state, attacker_id, evolution_deck_card_index);
    }

    // shuffleSelfIntoDeck (the `useShuffleSelfIntoDeck` flag defaults to true).
    let should_shuffle_self_into_deck = use_shuffle_self_into_deck.unwrap_or(true);
    if let Some(effect) = attack.shuffle_self_into_deck.clone() {
        if state.side(attacker_id).active.is_some() && should_shuffle_self_into_deck {
            shuffle_active_into_deck_if_paid(state, attacker_id, &effect, deps);
        }
    }

    // shuffleRandomDiscardIntoDeck.
    if attack.shuffle_random_discard_into_deck.is_some() {
        shuffle_random_discard_into_deck(state, attacker_id, random_discard_index);
    }

    // Resolve knockout of the defender's active (if HP <= 0 after damage).
    resolve_knockout(state, attacker_id, defender_id, deps);

    // Iterate the bench-damaged knockouts in push order. TS calls
    // refreshContinuousEffects after each successful knockout.
    let bench_dead_uids: Vec<u32> = {
        let defender = state.side(defender_id);
        defender
            .bench
            .iter()
            .filter(|u| u.hp <= 0)
            .map(|u| u.uid)
            .collect()
    };
    for uid in bench_dead_uids {
        // Re-read the umamusume each iteration (knockout may have moved
        // bench members).
        let inst_opt: Option<UmamusumeInstance> = state
            .side(defender_id)
            .bench
            .iter()
            .find(|u| u.uid == uid)
            .cloned();
        if let Some(inst) = inst_opt {
            let did = knock_out_umamusume(
                state,
                attacker_id,
                defender_id,
                &inst,
                deps.choose_preferred_active_index,
            );
            if did && !state.game_over {
                (deps.refresh_continuous_effects)(state);
            }
        }
    }

    // Post-attack switch: only if attacker's active is the original one
    // and still alive.
    if let Some(switch_uid) = switch_target_uid_resolved {
        let still_starting = {
            let attacker = state.side(attacker_id);
            attacker.active.as_ref().map(|a| a.uid) == Some(starting_active_uid)
                && attacker.active.as_ref().map(|a| a.hp).unwrap_or(0) > 0
        };
        if still_starting {
            let switch_index = state
                .side(attacker_id)
                .bench
                .iter()
                .position(|u| u.uid == switch_uid);
            if let Some(idx) = switch_index {
                let attacker = state.side_mut(attacker_id);
                let promoted = attacker.bench.remove(idx);
                if let Some(mut current_active) = attacker.active.take() {
                    clear_special_conditions(&mut current_active);
                    let _ = attacker.bench.try_push(current_active);
                }
                attacker.active = Some(promoted);
                (deps.refresh_continuous_effects)(state);
            }
        }
    }

    // Simultaneous-KO check: if both sides were at match-point before this
    // attack and only the attacker breached 3 points, declare attacker
    // winner.
    let preserve_attacker_win = should_preserve_attacker_win_on_simultaneous_ko(
        state,
        attacker_id,
        defender_id,
        points_before_attacker,
        points_before_defender,
    );
    if preserve_attacker_win && !state.game_over {
        state.game_over = true;
        state.winner = Some(attacker_id);
        state.current_side = CurrentSide::Done;
    }

    // Attacker self-KO (e.g. counter-damage killed them).
    if !state.game_over && !preserve_attacker_win {
        let attacker_active_dead = {
            let attacker = state.side(attacker_id);
            attacker
                .active
                .as_ref()
                .map(|a| a.hp <= 0)
                .unwrap_or(false)
        };
        if attacker_active_dead {
            let active_clone = state.side(attacker_id).active.clone().unwrap();
            let did = knock_out_umamusume(
                state,
                defender_id,
                attacker_id,
                &active_clone,
                deps.choose_preferred_active_index,
            );
            if did {
                if let Some(PendingPlayerChoice::PromoteAfterKnockout {
                    side_id,
                    resume,
                }) = state.pending_player_choice.as_mut()
                {
                    if *side_id == attacker_id {
                        *resume = PromoteResume::FinishOpponentTurn;
                    }
                }
                (deps.refresh_continuous_effects)(state);
            }
        }
    }

}

/// `combat.ts:285` `shuffleRandomDiscardIntoDeck`. RNG: one `randomInt`
/// (if no caller-provided index) + one `shuffle`.
fn shuffle_random_discard_into_deck(
    state: &mut GameState,
    side_id: SideId,
    random_discard_index: Option<usize>,
) {
    let side = state.side(side_id);
    if side.discard.is_empty() {
        return;
    }
    let discard_index = match random_discard_index {
        Some(idx) if idx < side.discard.len() => idx,
        _ => random_int(side.discard.len() as u32) as usize,
    };
    let side = state.side_mut(side_id);
    let card_id = side.discard.remove(discard_index);
    let mut deck_vec: Vec<_> = side.deck.iter().copied().collect();
    deck_vec.push(card_id);
    let shuffled = shuffle(&deck_vec);
    side.deck.clear();
    for c in shuffled {
        let _ = side.deck.try_push(c);
    }
}

/// `combat.ts:296` `evolveActiveFromDeck`. No RNG.
fn evolve_active_from_deck(
    state: &mut GameState,
    side_id: SideId,
    evolution_deck_card_index: Option<usize>,
) {
    let cat = catalog();
    let (active_species, active_stage) = {
        let active = match state.side(side_id).active.as_ref() {
            Some(a) => a,
            None => return,
        };
        (active.species().to_string(), active.stage)
    };

    let is_active_evolution = |cid: crate::core::card_id::CardId| -> bool {
        match cat.get(cid) {
            Some(Card::Umamusume(u)) => {
                u.evolves_from.as_deref() == Some(active_species.as_str())
                    && u.stage as i32 == active_stage as i32 + 1
            }
            _ => false,
        }
    };

    let side = state.side(side_id);
    let deck_index: Option<usize> = match evolution_deck_card_index {
        Some(idx) if idx < side.deck.len() && is_active_evolution(side.deck[idx]) => Some(idx),
        _ => side.deck.iter().position(|&cid| is_active_evolution(cid)),
    };
    let Some(idx) = deck_index else {
        return;
    };
    // Snapshot the evolution card before mutating.
    let card_id = state.side(side_id).deck[idx];
    let evolution_card: UmamusumeCard = match cat.get(card_id) {
        Some(Card::Umamusume(u)) => u.clone(),
        _ => return,
    };
    let turn_number = state.turn_number;
    let side = state.side_mut(side_id);
    let _ = side.deck.remove(idx);
    if let Some(active) = side.active.as_mut() {
        evolve_umamusume(turn_number, active, card_id, &evolution_card);
    }
}

/// `combat.ts:315` `shuffleActiveIntoDeckIfPaid`. RNG: one `shuffle`.
fn shuffle_active_into_deck_if_paid(
    state: &mut GameState,
    side_id: SideId,
    effect: &ShuffleSelfIntoDeck,
    deps: &mut CombatDeps<'_>,
) {
    {
        let side = state.side(side_id);
        let Some(active) = &side.active else { return };
        if effect.requires_bench && side.bench.is_empty() {
            return;
        }
        let can_pay = EnergyType::ALL.iter().copied().all(|t| {
            (active.energies[t as usize] as i32) >= (effect.discard_energy.get(t) as i32)
        });
        if !can_pay {
            return;
        }
    }

    // Pay the discard energy cost.
    {
        let side = state.side_mut(side_id);
        if let Some(active) = side.active.as_mut() {
            for t in EnergyType::ALL.iter().copied() {
                let cur = active.energies[t as usize] as i32;
                let amount = effect.discard_energy.get(t) as i32;
                active.energies[t as usize] = (cur - amount).max(0) as u16;
            }
        }
    }

    // Build the list of card ids to shuffle back (evolution chain + base + tool).
    let shuffled_card_ids: Vec<crate::core::card_id::CardId> = {
        let active = state.side(side_id).active.as_ref().unwrap();
        let mut v: Vec<_> = active.evolution_card_ids.iter().copied().collect();
        v.push(active.card_id);
        if let Some(tool) = active.tool_card_id {
            v.push(tool);
        }
        v
    };

    // Clear active, shuffle ids into the deck.
    {
        let side = state.side_mut(side_id);
        side.active = None;
        let mut deck_vec: Vec<_> = side.deck.iter().copied().collect();
        deck_vec.extend(shuffled_card_ids.iter().copied());
        let shuffled = shuffle(&deck_vec);
        side.deck.clear();
        for c in shuffled {
            let _ = side.deck.try_push(c);
        }
    }

    // Promote a bench card via the deps callback (or the front of bench).
    let promoted_index = (deps.choose_preferred_active_index)(state.side(side_id));
    let promoted: Option<UmamusumeInstance> = {
        let side = state.side_mut(side_id);
        if promoted_index >= 0 && (promoted_index as usize) < side.bench.len() {
            Some(side.bench.remove(promoted_index as usize))
        } else if !side.bench.is_empty() {
            Some(side.bench.remove(0))
        } else {
            None
        }
    };
    if let Some(p) = promoted {
        state.side_mut(side_id).active = Some(p);
    }
}

/// `combat.ts:351` `applySpecialCondition`. Replaces all conditions with
/// the new one (rule: only one at a time). Sets the paralysis recovery
/// turn when applicable.
fn apply_special_condition(
    state: &mut GameState,
    affected_side_id: SideId,
    umamusume_uid: u32,
    condition: SpecialCondition,
) {
    let recovery_turn = state.turns_taken_by_side[affected_side_id as usize] + 1;
    let side = state.side_mut(affected_side_id);

    let touch = |u: &mut UmamusumeInstance| {
        if u.special_conditions.len() == 1 && u.special_conditions[0] == condition {
            return;
        }
        u.special_conditions.clear();
        let _ = u.special_conditions.try_push(condition);
        if condition == SpecialCondition::Paralysed {
            u.paralysed_until_own_turn = Some(recovery_turn);
        } else {
            u.paralysed_until_own_turn = None;
        }
    };

    if let Some(active) = side.active.as_mut() {
        if active.uid == umamusume_uid {
            touch(active);
            return;
        }
    }
    for b in side.bench.iter_mut() {
        if b.uid == umamusume_uid {
            touch(b);
            return;
        }
    }
}

/// `combat.ts:369` `knockOutUmamusume`. Returns true if the knock-out was
/// applied (mirrors TS return).
pub fn knock_out_umamusume(
    state: &mut GameState,
    scoring_side_id: SideId,
    knocked_side_id: SideId,
    knocked_out: &UmamusumeInstance,
    choose_preferred_active_index: &dyn Fn(&SideState) -> i32,
) -> bool {
    let active_knockout = state
        .side(knocked_side_id)
        .active
        .as_ref()
        .map(|a| a.uid == knocked_out.uid)
        .unwrap_or(false);
    let bench_index = state
        .side(knocked_side_id)
        .bench
        .iter()
        .position(|u| u.uid == knocked_out.uid);
    if !active_knockout && bench_index.is_none() {
        return false;
    }

    // Discard the card + chain + tool. Take ownership of the knocked-out
    // instance for its card-id list.
    {
        let defender = state.side_mut(knocked_side_id);
        if active_knockout {
            defender.active = None;
        }
        if let Some(idx) = bench_index {
            defender.bench.remove(idx);
        }
        // TS also filters bench by uid to drop any lingering duplicates.
        defender.bench.retain(|u| u.uid != knocked_out.uid);
        let _ = defender.discard.try_push(knocked_out.card_id);
        for c in &knocked_out.evolution_card_ids {
            let _ = defender.discard.try_push(*c);
        }
        if let Some(tool) = knocked_out.tool_card_id {
            let _ = defender.discard.try_push(tool);
        }
    }

    let attacker_points = {
        let attacker = state.side_mut(scoring_side_id);
        attacker.points = attacker.points.saturating_add(1);
        attacker.points
    };

    if attacker_points as u32 >= MAX_POINTS {
        state.game_over = true;
        state.winner = Some(scoring_side_id);
        state.current_side = CurrentSide::Done;
        return true;
    }

    // Defender has no umamusume at all → game over.
    let defender_has_anything = {
        let defender = state.side(knocked_side_id);
        defender.active.is_some() || !defender.bench.is_empty()
    };
    if !defender_has_anything {
        state.game_over = true;
        state.winner = Some(scoring_side_id);
        state.current_side = CurrentSide::Done;
        return true;
    }

    // Bench-only knockout — no active-promotion bookkeeping.
    if !active_knockout {
        return true;
    }

    // Active knockout: if defender is human, queue a pending choice. Else
    // auto-promote via the deps callback (or front of bench).
    if state.human_by_side[knocked_side_id as usize] {
        let should_advance = CurrentSide::from_side(knocked_side_id) != state.current_side;
        state.pending_player_choice = Some(PendingPlayerChoice::PromoteAfterKnockout {
            side_id: knocked_side_id,
            resume: if should_advance {
                PromoteResume::FinishOpponentTurn
            } else {
                PromoteResume::None
            },
        });
        return true;
    }

    let promoted_index = choose_preferred_active_index(state.side(knocked_side_id));
    let defender = state.side_mut(knocked_side_id);
    let promoted: Option<UmamusumeInstance> = if promoted_index >= 0
        && (promoted_index as usize) < defender.bench.len()
    {
        Some(defender.bench.remove(promoted_index as usize))
    } else if !defender.bench.is_empty() {
        Some(defender.bench.remove(0))
    } else {
        None
    };
    if let Some(p) = promoted {
        defender.active = Some(p);
    }
    true
}

/// `combat.ts:433` `attackDamageReductionFor`.
fn attack_damage_reduction_for(state: &GameState, umamusume: &UmamusumeInstance) -> i32 {
    let owner_side: SideId = if state
        .side(SideId::Player)
        .active
        .as_ref()
        .map(|a| a.uid)
        .unwrap_or(u32::MAX)
        == umamusume.uid
        || state
            .side(SideId::Player)
            .bench
            .iter()
            .any(|u| u.uid == umamusume.uid)
    {
        SideId::Player
    } else {
        SideId::Opponent
    };
    let ability_reduction = get_umamusume_ability(state, owner_side, umamusume)
        .and_then(|a| a.damage_reduction)
        .unwrap_or(0);
    ability_reduction + umamusume.next_turn_damage_reduction + active_tool_damage_reduction(state, umamusume)
}

fn active_tool_damage_reduction(state: &GameState, umamusume: &UmamusumeInstance) -> i32 {
    if are_tools_disabled(state) {
        return 0;
    }
    let Some(tool_id) = umamusume.tool_card_id else {
        return 0;
    };
    match catalog().get(tool_id) {
        Some(Card::Trainer(t)) => t.effect.tool_damage_reduction.unwrap_or(0),
        _ => 0,
    }
}

fn active_tool_counter_damage(state: &GameState, umamusume: &UmamusumeInstance) -> i32 {
    if are_tools_disabled(state) {
        return 0;
    }
    let Some(tool_id) = umamusume.tool_card_id else {
        return 0;
    };
    match catalog().get(tool_id) {
        Some(Card::Trainer(t)) => t.effect.tool_counter_damage.unwrap_or(0),
        _ => 0,
    }
}

fn are_tools_disabled(state: &GameState) -> bool {
    let Some(stadium) = &state.stadium else {
        return false;
    };
    matches!(
        catalog().get(stadium.card_id),
        Some(Card::Trainer(t)) if t.effect.disable_tools == Some(true)
    )
}

/// `combat.ts:456` `resolveKnockout`. Wraps `knockOutUmamusume` for the
/// defender's active; calls `refresh_continuous_effects` on success.
fn resolve_knockout(
    state: &mut GameState,
    attacker_id: SideId,
    defender_id: SideId,
    deps: &mut CombatDeps<'_>,
) {
    let target: Option<UmamusumeInstance> = {
        let defender = state.side(defender_id);
        match defender.active.as_ref() {
            Some(a) if a.hp <= 0 => Some(a.clone()),
            _ => None,
        }
    };
    let Some(target) = target else {
        return;
    };
    let did = knock_out_umamusume(
        state,
        attacker_id,
        defender_id,
        &target,
        deps.choose_preferred_active_index,
    );
    if did && !state.game_over {
        (deps.refresh_continuous_effects)(state);
    }
}

/// `combat.ts:473` `isNonDamagingAttack`.
fn is_non_damaging_attack(attack: &Attack) -> bool {
    attack.damage <= 0
        && attack.coin_bonus.is_none()
        && attack.bonus_if_took_damage_last_turn.is_none()
        && attack.damage_per_attached_energy.is_none()
        && attack.damage_per_umamusume_in_play.is_none()
        && attack.attack_damage_bonus_if_tool_attached.is_none()
        && attack.attack_damage_bonus_if_discard_hand_card.is_none()
        && match &attack.switch_self_after_attack {
            Some(s) => s.bonus_damage.is_none(),
            None => true,
        }
}

/// `combat.ts:484` `resolveSwitchTarget`. Returns the uid of the bench
/// umamusume to switch to, or None.
fn resolve_switch_target_uid(
    state: &GameState,
    attacker_id: SideId,
    switch_target_uid: Option<u32>,
    attack: &Attack,
    choose_preferred_active_index: &dyn Fn(&SideState) -> i32,
) -> Option<u32> {
    if attack.switch_self_after_attack.is_none() {
        return None;
    }
    let attacker = state.side(attacker_id);
    if let Some(uid) = switch_target_uid {
        return attacker
            .bench
            .iter()
            .find(|u| u.uid == uid)
            .map(|u| u.uid);
    }
    if state.human_by_side[attacker_id as usize] || attacker.bench.is_empty() {
        return None;
    }
    let preferred_index = choose_preferred_active_index(attacker);
    if preferred_index >= 0 {
        attacker
            .bench
            .get(preferred_index as usize)
            .map(|u| u.uid)
            .or_else(|| attacker.bench.first().map(|u| u.uid))
    } else {
        attacker.bench.first().map(|u| u.uid)
    }
}

/// `combat.ts:501` `flipCoin`. The thin wrapper over the state-aware
/// version below; tests may use it directly.
pub fn flip_coin(
    guaranteed_heads_remaining: &mut u8,
    forced_results: &mut Vec<CoinFlipResult>,
) -> CoinFlipResult {
    if *guaranteed_heads_remaining > 0 {
        *guaranteed_heads_remaining -= 1;
        if !forced_results.is_empty() {
            forced_results.remove(0);
        }
        return CoinFlipResult::Heads;
    }
    if !forced_results.is_empty() {
        return forced_results.remove(0);
    }
    if random_float() >= 0.5 {
        CoinFlipResult::Heads
    } else {
        CoinFlipResult::Tails
    }
}

/// State-aware wrapper used internally so we can mutate the side's
/// `guaranteed_coin_flip_heads` counter.
fn flip_coin_with_state(
    state: &mut GameState,
    side_id: SideId,
    forced_results: &mut Vec<CoinFlipResult>,
) -> CoinFlipResult {
    let side = state.side_mut(side_id);
    flip_coin(&mut side.guaranteed_coin_flip_heads, forced_results)
}

/// `combat.ts:517` `shouldPreserveAttackerWinOnSimultaneousKo`.
fn should_preserve_attacker_win_on_simultaneous_ko(
    state: &GameState,
    attacker_id: SideId,
    defender_id: SideId,
    points_before_attacker: u8,
    points_before_defender: u8,
) -> bool {
    let max_points_minus_one = (MAX_POINTS - 1) as u8;
    let attacker_at_match_point_before = points_before_attacker == max_points_minus_one;
    let defender_at_match_point_before = points_before_defender == max_points_minus_one;
    if !attacker_at_match_point_before || !defender_at_match_point_before {
        return false;
    }
    let attacker_points_after = state.side(attacker_id).points;
    let defender_points_after = state.side(defender_id).points;
    let attacker_reached_max_first = (attacker_points_after as u32) >= MAX_POINTS
        && (points_before_attacker as u32) < MAX_POINTS;
    let defender_had_not_reached_max_before_resolution =
        (defender_points_after as u32) < MAX_POINTS;
    attacker_reached_max_first && defender_had_not_reached_max_before_resolution
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::card_id::CardId;
    use crate::core::constants::{
        AiDeckStyle, AiDifficulty, EnergyType, SideId, SpecialCondition,
    };
    use crate::core::state::{CurrentSide, Phase, SideState};
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
            phase: Phase::Play,
            setup: None,
            pending_player_choice: None,
            sides: [empty_side(SideId::Player), empty_side(SideId::Opponent)],
            current_side: CurrentSide::Player,
            opponent_turn_step: None,
            stadium: None,
            turn_deadline_ms: None,
            turn_number: 1,
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

    fn dummy_instance(uid: u32) -> UmamusumeInstance {
        UmamusumeInstance {
            uid,
            card_id: CardId(0),
            evolution_card_ids: ArrayVec::new(),
            stage: 0,
            hp: 60,
            max_hp: 60,
            energies: [0u16; EnergyType::COUNT],
            special_conditions: ArrayVec::new(),
            entered_turn: 0,
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

    #[test]
    fn flip_coin_guaranteed_consumes_counter_and_forced_result() {
        let mut counter: u8 = 1;
        let mut forced = vec![CoinFlipResult::Tails, CoinFlipResult::Heads];
        // First call: guaranteed. Returns Heads, decrements counter to 0,
        // AND consumes the first forced result (the Tails).
        let r1 = flip_coin(&mut counter, &mut forced);
        assert_eq!(r1, CoinFlipResult::Heads);
        assert_eq!(counter, 0);
        assert_eq!(forced.len(), 1);
        assert_eq!(forced[0], CoinFlipResult::Heads);
        // Second call: counter is 0, forced has one entry; return it.
        let r2 = flip_coin(&mut counter, &mut forced);
        assert_eq!(r2, CoinFlipResult::Heads);
        assert!(forced.is_empty());
    }

    #[test]
    fn apply_special_condition_replaces_existing_and_sets_paralysis_turn() {
        let mut state = empty_state();
        state.turns_taken_by_side = [3, 0];
        let mut active = dummy_instance(42);
        // Pre-existing condition: Burned.
        let _ = active.special_conditions.try_push(SpecialCondition::Burned);
        state.sides[SideId::Player as usize].active = Some(active);

        apply_special_condition(&mut state, SideId::Player, 42, SpecialCondition::Paralysed);
        let active = state.side(SideId::Player).active.as_ref().unwrap();
        assert_eq!(active.special_conditions.len(), 1);
        assert_eq!(active.special_conditions[0], SpecialCondition::Paralysed);
        // recovery turn = turns_taken_by_side[player] + 1 = 4.
        assert_eq!(active.paralysed_until_own_turn, Some(4));

        // Switching to a non-paralysis condition clears the recovery turn.
        apply_special_condition(&mut state, SideId::Player, 42, SpecialCondition::Asleep);
        let active = state.side(SideId::Player).active.as_ref().unwrap();
        assert_eq!(active.special_conditions[0], SpecialCondition::Asleep);
        assert_eq!(active.paralysed_until_own_turn, None);
    }

    #[test]
    fn knock_out_umamusume_triggers_game_over_at_max_points() {
        let mut state = empty_state();
        // Attacker at MAX_POINTS - 1; one more point wins.
        state.sides[SideId::Player as usize].points = (MAX_POINTS - 1) as u8;
        // Opponent has an active and a bench (so removal doesn't auto-end).
        state.sides[SideId::Opponent as usize].active = Some(dummy_instance(7));
        let bench_member = dummy_instance(8);
        state.sides[SideId::Opponent as usize]
            .bench
            .push(bench_member);

        // Snapshot the knocked-out instance.
        let knocked = state
            .side(SideId::Opponent)
            .active
            .clone()
            .unwrap();
        let did = knock_out_umamusume(
            &mut state,
            SideId::Player,
            SideId::Opponent,
            &knocked,
            &|_side| -1, // no preference; falls back to first bench
        );
        assert!(did);
        assert!(state.game_over);
        assert_eq!(state.winner, Some(SideId::Player));
        assert_eq!(state.current_side, CurrentSide::Done);
    }

    #[test]
    fn knock_out_with_no_remaining_umamusume_ends_game() {
        let mut state = empty_state();
        // Attacker at 0 points; defender has only the active.
        state.sides[SideId::Opponent as usize].active = Some(dummy_instance(7));
        let knocked = state.side(SideId::Opponent).active.clone().unwrap();
        let did = knock_out_umamusume(
            &mut state,
            SideId::Player,
            SideId::Opponent,
            &knocked,
            &|_side| -1,
        );
        assert!(did);
        assert!(state.game_over);
        assert_eq!(state.winner, Some(SideId::Player));
        // Attacker scored 1 (< MAX_POINTS), but the bench is empty.
        assert_eq!(state.side(SideId::Player).points, 1);
    }

    #[test]
    fn is_non_damaging_attack_handles_switch_bonus_as_damaging() {
        let mut a = Attack {
            name: String::new(),
            cost: Default::default(),
            damage: 0,
            text: String::new(),
            target_opponent: None,
            bench_damage: None,
            coin_bonus: None,
            draw_on_heads: None,
            draw: None,
            heal: None,
            heal_target: None,
            discard_energy: None,
            evolve_from_deck: None,
            recover_special_conditions: None,
            shuffle_self_into_deck: None,
            switch_self_after_attack: None,
            prevent_damage_next_turn: None,
            bonus_if_took_damage_last_turn: None,
            damage_per_attached_energy: None,
            damage_per_unique_attached_energy: None,
            damage_per_umamusume_in_play: None,
            attack_damage_bonus_if_tool_attached: None,
            attack_damage_bonus_if_discard_hand_card: None,
            attack_damage_bonus_per_discarded_hand_card: None,
            discard_random_opponent_hand_on_heads: None,
            shuffle_random_discard_into_deck: None,
            guarantee_next_coin_flip_heads: None,
            knock_out_active_if_all_coin_heads: None,
            cannot_attack_next_turn: None,
            inflict_special_condition: None,
        };
        assert!(is_non_damaging_attack(&a));
        // A switch effect that grants no bonus is still non-damaging.
        a.switch_self_after_attack = Some(crate::core::effects::SwitchSelfAfterAttack {
            bonus_damage: None,
        });
        assert!(is_non_damaging_attack(&a));
        // A switch effect with a bonus tips into damaging.
        a.switch_self_after_attack = Some(crate::core::effects::SwitchSelfAfterAttack {
            bonus_damage: Some(10),
        });
        assert!(!is_non_damaging_attack(&a));
    }
}
