//! Bit-identical port of `frontend/src/game/engine/ai-policy/actions.ts`.
//!
//! The legal-action enumerator + per-action feature vector for the AI
//! policy interface. MCTS rollouts pick actions by index, and the recorded
//! golden traces hash the action list — iteration order over hand / bench
//! / deck is **load-bearing**.
//!
//! Float math (`* 0.5`, `* 1.8`, `* 0.05`, `* 0.04`, `card.hp / 180`, …)
//! is preserved verbatim. The 48-element `features` vector is consumed
//! element-by-element by `training/uma_ai/features.py`, so per-slot
//! arithmetic must match TS source-text op order.

use serde_json::{json, Map, Value};

use crate::core::card_id::CardId;
use crate::core::catalog::{catalog, Card, TrainerCard, UmamusumeCard};
use crate::core::constants::{EnergyType, SideId, TrainerType, MAX_BENCH};
use crate::core::effects::{Attack, AttackTarget, EnergyCost, HealTarget};
use crate::core::play_types::{PlayActionOutcome, PlayChoices};
use crate::core::state::{GameState, PendingPlayerChoice, SideState, UmamusumeInstance};
use crate::core::umamusume::{attached_energy_count, get_all_umamusume};
use crate::flow::board::{choose_preferred_active_index, refresh_continuous_hp};
use crate::flow::combat::CombatDeps;
use crate::flow::eligibility::{
    can_attach_energy, can_attach_energy_to_umamusume, can_use_umamusume_ability,
};
use crate::flow::energy::{get_ability_move_energy_types, has_enough_energy};
use crate::flow::evolution::find_evolution_target;
use crate::flow::play_rules::{
    get_playable_action, get_rainbow_uncap_evolution_hand_options, get_rainbow_uncap_targets,
    get_tool_targets,
};
use crate::flow::trainers::can_use_stadium;

use crate::flow::ai::attach_utils::score_ai_attach_target;
use crate::flow::ai::combat_planner::build_combat_candidates;
use crate::flow::ai::trainer_utils::{score_evolution_target, should_ai_play_trainer};
use crate::flow::ai::turn_plan::choose_ai_turn_goal;
use crate::flow::ai::types::AiCombatDecision;

use crate::policy::card_vocab::card_vocab_index;
use crate::policy::phase::get_ai_phase;
use crate::policy::types::{AiPhase, LegalAiAction};

/// `ai-policy/actions.ts:21`. Per-action feature schema version. Bump if
/// the layout of the 48-element `features` vector changes.
///
/// v33-correctness-fix Fix 2-4 bumped 2 → 3. Slot 10 was polysemic
/// (setup/attach/combat overloads on `amount`); slot 26 was an exact
/// duplicate of slot 8; slot 28 compared `target.uid` to `target_slot`
/// (different ID spaces — near-always 0). New layout:
///   slot 10 = combat `target_value` / 200 (0 outside combat)
///   slot 26 = combat `lethal_target` flag (0/1, outside combat = 0)
///   slot 28 = trainer `effect.heal` magnitude / 100 (0 outside trainer)
///
/// v38-slim-feature-add bumped 3 → 4. Adds 4 new slots at [48:52]:
///   slot 48 = `swap_in_attack_ready` (retreat/retreatAttack only)
///   slot 49 = `expected_damage_norm` (base + activeAttackDamageBonus +
///             weakness, normalized /300; RESTRICTED — no conditional
///             bonuses, no coin-flip, no discard)
///   slot 50 = `attach_color_matches_typed_need` (attachEnergy only)
///   slot 51 = `attach_completes_typed_threshold` (attachEnergy only)
/// v3 slots [0:48] BYTE-STABLE.
pub const ACTION_FEATURE_SCHEMA_VERSION: u32 = 4;

/// `ai-policy/actions.ts:22`. Per-action feature count — locks the Python
/// collator's input width.
///
/// v38-slim-feature-add bumped 48 → 52. v3 slots [0:48] BYTE-STABLE; the
/// 4 new slots [48:52] are pure additions per the v4 schema.
pub const ACTION_FEATURE_COUNT: usize = 52;

/// `ai-policy/actions.ts:23`. Source-declaration order. Matches
/// `EnergyType::ALL`.
const ENERGY_TYPES: [EnergyType; 10] = EnergyType::ALL;

/// `ai-policy/actions.ts:25` `enumerateLegalAiActions`.
pub fn enumerate_legal_ai_actions(state: &GameState, side_id: SideId) -> Vec<LegalAiAction> {
    let phase = get_ai_phase(state, side_id);
    let side = state.side(side_id);
    match phase {
        AiPhase::Setup => enumerate_setup_actions(state, side_id),
        AiPhase::PendingChoice => enumerate_pending_choice_actions(state, side_id),
        AiPhase::Bench => with_pass(phase, enumerate_bench_actions(state, side)),
        AiPhase::TrainerBefore | AiPhase::TrainerAfter => {
            with_pass(phase, enumerate_trainer_actions(state, side, phase))
        }
        AiPhase::Evolve => with_pass(phase, enumerate_evolution_actions(state, side)),
        AiPhase::Attach => with_pass(phase, enumerate_attach_actions(state, side)),
        AiPhase::Ability => with_pass(phase, enumerate_ability_actions(state, side)),
        AiPhase::Combat => enumerate_combat_actions(state, side),
        AiPhase::StadiumOrEnd => enumerate_stadium_or_end_actions(state, side_id),
    }
}

/// `ai-policy/actions.ts:53` `chooseHighestScoredAction`.
///
/// TS uses `Array.prototype.sort`, which is stable. Rust `slice::sort_by`
/// is also stable — that property is load-bearing for ties.
pub fn choose_highest_scored_action(actions: &[LegalAiAction]) -> LegalAiAction {
    let mut sorted: Vec<LegalAiAction> = actions.to_vec();
    sorted.sort_by(|left, right| {
        let lv = left.features.first().copied().unwrap_or(0.0);
        let rv = right.features.first().copied().unwrap_or(0.0);
        rv.partial_cmp(&lv).unwrap_or(std::cmp::Ordering::Equal)
    });
    sorted
        .into_iter()
        .next()
        .unwrap_or_else(|| pass_action(AiPhase::StadiumOrEnd))
}

/// `ai-policy/actions.ts:57` `chooseLowestScoredAction`.
pub fn choose_lowest_scored_action(actions: &[LegalAiAction]) -> LegalAiAction {
    let mut sorted: Vec<LegalAiAction> = actions.to_vec();
    sorted.sort_by(|left, right| {
        let lv = left.features.first().copied().unwrap_or(0.0);
        let rv = right.features.first().copied().unwrap_or(0.0);
        lv.partial_cmp(&rv).unwrap_or(std::cmp::Ordering::Equal)
    });
    sorted
        .into_iter()
        .next()
        .unwrap_or_else(|| pass_action(AiPhase::StadiumOrEnd))
}

// ---------------------------------------------------------------------------
// Phase-specific enumerators.
// ---------------------------------------------------------------------------

fn enumerate_setup_actions(state: &GameState, side_id: SideId) -> Vec<LegalAiAction> {
    let setup = state.setup.as_ref();
    let side = state.side(side_id);
    let opening_dealt = setup.map(|s| s.opening_hands_dealt).unwrap_or(false);
    if !opening_dealt && side.hand.is_empty() {
        return vec![pass_action(AiPhase::Setup)];
    }
    let cat = catalog();
    // Collect (cardId, handIndex, score) for basics, preserving hand order.
    let mut basics: Vec<(CardId, usize, f64)> = Vec::new();
    for (hand_index, &cid) in side.hand.iter().enumerate() {
        let Some(Card::Umamusume(card)) = cat.get(cid) else {
            continue;
        };
        if card.stage != 0 {
            continue;
        }
        let Some(attack) = card.attacks.first() else {
            continue;
        };
        let score = card.hp as f64 + attack.damage as f64 * 1.8;
        basics.push((cid, hand_index, score));
    }
    // TS uses stable `Array.prototype.sort`; preserve stability.
    basics.sort_by(|left, right| {
        right
            .2
            .partial_cmp(&left.2)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let Some(&(active_card_id, active_hand_index, active_score)) = basics.first() else {
        return vec![pass_action(AiPhase::Setup)];
    };

    // Bench at most `MAX_BENCH` more basics.
    let bench_hand_indexes: Vec<usize> = basics
        .iter()
        .skip(1)
        .take(MAX_BENCH)
        .map(|&(_, hand_index, _)| hand_index)
        .collect();

    let bench_indexes_str = bench_hand_indexes
        .iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let side_tag = match side_id {
        SideId::Player => "player",
        SideId::Opponent => "opponent",
    };
    let id = format!(
        "setup:{}:active:{}:bench:{}",
        side_tag, active_hand_index, bench_indexes_str
    );

    let payload = json!({
        "activeHandIndex": active_hand_index,
        "benchHandIndexes": bench_hand_indexes,
    });

    let feature_score = active_score + bench_hand_indexes.len() as f64 * 12.0;
    let feats = build_features(FeatureInput {
        score: feature_score,
        phase: AiPhase::Setup,
        kind: "setupChooseBoard",
        source_card_id: Some(active_card_id),
        ..Default::default()
    });

    vec![LegalAiAction {
        id,
        phase: AiPhase::Setup,
        kind: "setupChooseBoard".to_string(),
        payload,
        features: feats,
        action_source_card_idx: Some(card_vocab_index_of(active_card_id) as u16),
        action_target_card_idx: None,
    }]
}

fn enumerate_pending_choice_actions(state: &GameState, side_id: SideId) -> Vec<LegalAiAction> {
    let pending = match &state.pending_player_choice {
        Some(p) => p,
        None => return vec![pass_action(AiPhase::PendingChoice)],
    };
    let pending_side = match pending {
        PendingPlayerChoice::PromoteAfterKnockout { side_id, .. } => *side_id,
        PendingPlayerChoice::SwitchAfterGust { side_id, .. } => *side_id,
    };
    if pending_side != side_id {
        return vec![pass_action(AiPhase::PendingChoice)];
    }
    let pending_kind_tag = match pending {
        PendingPlayerChoice::PromoteAfterKnockout { .. } => "promoteAfterKnockout",
        PendingPlayerChoice::SwitchAfterGust { .. } => "switchAfterGust",
    };
    let side = state.side(side_id);
    let actions: Vec<LegalAiAction> = side
        .bench
        .iter()
        .enumerate()
        .map(|(slot, target)| {
            let id = format!("pending:{}:{}", pending_kind_tag, target.uid);
            let payload = json!({ "targetUid": target.uid });
            let score = score_umamusume(target);
            let feats = build_features(FeatureInput {
                score,
                phase: AiPhase::PendingChoice,
                kind: "resolvePendingChoice",
                target: Some(target),
                target_slot: Some(slot as f64),
                ..Default::default()
            });
            LegalAiAction {
                id,
                phase: AiPhase::PendingChoice,
                kind: "resolvePendingChoice".to_string(),
                payload,
                features: feats,
                action_source_card_idx: None,
                action_target_card_idx: Some(card_vocab_index_of(target.card_id) as u16),
            }
        })
        .collect();
    if actions.is_empty() {
        vec![pass_action(AiPhase::PendingChoice)]
    } else {
        actions
    }
}

fn enumerate_bench_actions(_state: &GameState, side: &SideState) -> Vec<LegalAiAction> {
    if side.bench.len() >= MAX_BENCH {
        return Vec::new();
    }
    let cat = catalog();
    let mut out: Vec<LegalAiAction> = Vec::new();
    for (hand_index, &card_id) in side.hand.iter().enumerate() {
        let Some(Card::Umamusume(card)) = cat.get(card_id) else {
            continue;
        };
        if card.stage != 0 {
            continue;
        }
        let Some(attack) = card.attacks.first() else {
            continue;
        };
        let id_str = catalog_card_id_str(card_id).unwrap_or_default();
        let id = format!("bench:{}:{}", hand_index, id_str);
        let payload = json!({ "handIndex": hand_index });
        let score = card.hp as f64 * 0.5 + attack.damage as f64 + 20.0;
        let feats = build_features(FeatureInput {
            score,
            phase: AiPhase::Bench,
            kind: "playBasic",
            source_card_id: Some(card_id),
            source_hand_index: Some(hand_index as f64),
            ..Default::default()
        });
        out.push(LegalAiAction {
            id,
            phase: AiPhase::Bench,
            kind: "playBasic".to_string(),
            payload,
            features: feats,
            action_source_card_idx: Some(card_vocab_index_of(card_id) as u16),
            action_target_card_idx: None,
        });
    }
    out
}

fn enumerate_trainer_actions(
    state: &GameState,
    side: &SideState,
    phase: AiPhase,
) -> Vec<LegalAiAction> {
    let turn_goal = choose_ai_turn_goal(state, side);
    let cat = catalog();
    let phase_tag = match phase {
        AiPhase::TrainerBefore => "trainerBefore",
        AiPhase::TrainerAfter => "trainerAfter",
        _ => "trainerBefore",
    };
    let mut out: Vec<LegalAiAction> = Vec::new();
    for (hand_index, &card_id) in side.hand.iter().enumerate() {
        let Some(Card::Trainer(trainer)) = cat.get(card_id) else {
            continue;
        };
        let play = get_playable_action(state, side, card_id);
        if !matches!(play, PlayActionOutcome::CanPlay(_)) {
            continue;
        }
        let mut score: f64 = 24.0;
        if let Some(d) = trainer.effect.draw {
            score += d as f64 * 8.0;
        }
        if trainer.effect.search_umamusume == Some(true)
            || trainer.effect.search_evolution_umamusume == Some(true)
            || trainer.effect.search_random_basic_umamusume == Some(true)
        {
            score += 34.0;
        }
        if trainer.effect.gust_opponent == Some(true) {
            score += 30.0;
        }
        if trainer.effect.extra_energy_attach.is_some()
            || trainer.effect.attach_energy_from_zone_to_bench.is_some()
        {
            score += 36.0;
        }
        if trainer.effect.heal.is_some() {
            score += 16.0;
        }
        if trainer.trainer_type == TrainerType::Tool && !get_tool_targets(side).is_empty() {
            score += 20.0;
        }
        if should_ai_play_trainer(state, side, card_id, hand_index, turn_goal) {
            score += 28.0;
        } else {
            score += -90.0;
        }

        let trainer_card_id_str = catalog_card_id_str(card_id).unwrap_or_default();
        let trainer_choices = enumerate_trainer_choices(state, side, trainer, hand_index);
        for choices in trainer_choices {
            let target = choices
                .umamusume_target_uid
                .and_then(|uid| {
                    get_all_umamusume(side)
                        .into_iter()
                        .find(|u| u.uid == uid)
                });
            let choice_card_id = get_choice_card_id(side, &choices);
            let payload_choices = play_choices_to_value(&choices);
            let payload = json!({
                "handIndex": hand_index,
                "choices": payload_choices,
            });
            let id = format!(
                "{}:trainer:{}:{}:{}",
                phase_tag,
                hand_index,
                trainer_card_id_str,
                choice_key(&choices)
            );
            let trainer_choice_bonus = score_trainer_choices(side, &choices) * 0.05;
            let feats = build_features(FeatureInput {
                score: score + trainer_choice_bonus,
                phase,
                kind: "playTrainer",
                source_card_id: Some(card_id),
                source_hand_index: Some(hand_index as f64),
                choice_card_id,
                target,
                ..Default::default()
            });
            out.push(LegalAiAction {
                id,
                phase,
                kind: "playTrainer".to_string(),
                payload,
                features: feats,
                action_source_card_idx: Some(card_vocab_index_of(card_id) as u16),
                action_target_card_idx: target
                    .map(|t| card_vocab_index_of(t.card_id) as u16),
            });
        }
    }
    out
}

fn enumerate_evolution_actions(state: &GameState, side: &SideState) -> Vec<LegalAiAction> {
    let cat = catalog();
    let mut out: Vec<LegalAiAction> = Vec::new();
    for (hand_index, &card_id) in side.hand.iter().enumerate() {
        let Some(Card::Umamusume(card)) = cat.get(card_id) else {
            continue;
        };
        if card.stage == 0 {
            continue;
        }
        let Some(target) = find_evolution_target(state, side, card) else {
            continue;
        };
        let card_id_str = catalog_card_id_str(card_id).unwrap_or_default();
        let id = format!("evolve:{}:{}:{}", hand_index, card_id_str, target.uid);
        let payload = json!({
            "handIndex": hand_index,
            "targetUid": target.uid,
        });
        let score = score_evolution_target(state, side, target, card);
        let feats = build_features(FeatureInput {
            score,
            phase: AiPhase::Evolve,
            kind: "evolve",
            source_card_id: Some(card_id),
            source_hand_index: Some(hand_index as f64),
            target: Some(target),
            ..Default::default()
        });
        out.push(LegalAiAction {
            id,
            phase: AiPhase::Evolve,
            kind: "evolve".to_string(),
            payload,
            features: feats,
            action_source_card_idx: Some(card_vocab_index_of(card_id) as u16),
            action_target_card_idx: Some(card_vocab_index_of(target.card_id) as u16),
        });
    }
    out
}

fn enumerate_attach_actions(state: &GameState, side: &SideState) -> Vec<LegalAiAction> {
    if !can_attach_energy(state, side) {
        return Vec::new();
    }
    let turn_goal = choose_ai_turn_goal(state, side);
    // v3.8 cross-bit context: `side.energy_zone[0]` is the color about
    // to be attached (`flow/energy.ts:7` head-of-queue). Pass to
    // `FeatureInput.attach_color` so slots 50/51 can compute.
    let attach_color: Option<EnergyType> = side.energy_zone.first().copied();
    let mut out: Vec<LegalAiAction> = Vec::new();
    for (slot, target) in get_all_umamusume(side).into_iter().enumerate() {
        if !can_attach_energy_to_umamusume(state, side, target.uid) {
            continue;
        }
        let id = format!("attach:{}", target.uid);
        let payload = json!({ "targetUid": target.uid });
        let score = score_ai_attach_target(state, side.id, target, turn_goal);
        let feats = build_features(FeatureInput {
            score,
            phase: AiPhase::Attach,
            kind: "attachEnergy",
            target: Some(target),
            target_slot: Some(slot as f64),
            attach_color,
            ..Default::default()
        });
        out.push(LegalAiAction {
            id,
            phase: AiPhase::Attach,
            kind: "attachEnergy".to_string(),
            payload,
            features: feats,
            action_source_card_idx: None,
            action_target_card_idx: Some(card_vocab_index_of(target.card_id) as u16),
        });
    }
    out
}

fn enumerate_ability_actions(state: &GameState, side: &SideState) -> Vec<LegalAiAction> {
    let cat = catalog();
    let opponent = state.side(side.id.opposite());
    let mut out: Vec<LegalAiAction> = Vec::new();
    let sources: Vec<(usize, UmamusumeInstance)> = get_all_umamusume(side)
        .into_iter()
        .enumerate()
        .map(|(slot, u)| (slot, u.clone()))
        .collect();
    for (slot, source) in sources {
        if !can_use_umamusume_ability(state, side, source.uid) {
            continue;
        }
        let Some(Card::Umamusume(card)) = cat.get(source.card_id) else {
            continue;
        };
        let Some(ability) = card.ability.as_ref() else {
            continue;
        };
        let mut score: f64 = 20.0 + source.stage as f64 * 10.0;
        if let Some(d) = ability.damage_opponent {
            score += d as f64 * 2.0;
        }
        if ability.move_benched_energy_to_active.is_some() {
            score += 38.0;
        }
        if let Some(dd) = ability.discard_to_draw.as_ref() {
            score += dd.draw as f64 * 8.0;
        }
        if ability.coin_flip_draw_or_active_damage_counter.is_some() {
            score += 18.0;
        }
        if ability.move_benched_energy_to_active.is_some() {
            // Sub-actions: for each benched umamusume × move-energy-type that
            // it has > 0 of.
            let energy_types = get_ability_move_energy_types(Some(ability));
            for energy_source in side.bench.iter() {
                for energy_type in energy_types.iter().copied() {
                    if energy_source.energies[energy_type as usize] == 0 {
                        continue;
                    }
                    let id = format!(
                        "ability:{}:{}:move:{}:{}",
                        source.uid,
                        ability.name,
                        energy_source.uid,
                        energy_type_tag(energy_type),
                    );
                    let payload = json!({
                        "sourceUid": source.uid,
                        "abilityName": ability.name,
                        "energySourceUid": energy_source.uid,
                        "energyType": energy_type_tag(energy_type),
                    });
                    let active_bonus = side
                        .active
                        .as_ref()
                        .map(|a| score_umamusume(a) * 0.04)
                        .unwrap_or(0.0);
                    let feats = build_features(FeatureInput {
                        score: score + active_bonus,
                        phase: AiPhase::Ability,
                        kind: "useAbility",
                        source_card_id: Some(source.card_id),
                        target: Some(energy_source),
                        target_slot: Some(slot as f64),
                        ..Default::default()
                    });
                    out.push(LegalAiAction {
                        id,
                        phase: AiPhase::Ability,
                        kind: "useAbility".to_string(),
                        payload,
                        features: feats,
                        action_source_card_idx: Some(card_vocab_index_of(source.card_id) as u16),
                        action_target_card_idx: Some(
                            card_vocab_index_of(energy_source.card_id) as u16,
                        ),
                    });
                }
            }
            continue;
        }
        if let (Some(damage_opponent), Some(target_kind)) =
            (ability.damage_opponent, ability.damage_opponent_target)
        {
            if matches!(target_kind, AttackTarget::Any) {
                for (target_slot, target) in get_all_umamusume(opponent).into_iter().enumerate() {
                    let id = format!(
                        "ability:{}:{}:damage:{}",
                        source.uid, ability.name, target.uid
                    );
                    let payload = json!({
                        "sourceUid": source.uid,
                        "abilityName": ability.name,
                        "targetUid": target.uid,
                    });
                    let lethal_bonus = if target.hp <= damage_opponent { 80.0 } else { 0.0 };
                    let feats = build_features(FeatureInput {
                        score: score + lethal_bonus + score_umamusume(target) * 0.05,
                        phase: AiPhase::Ability,
                        kind: "useAbility",
                        source_card_id: Some(source.card_id),
                        target: Some(target),
                        target_slot: Some(target_slot as f64),
                        ..Default::default()
                    });
                    out.push(LegalAiAction {
                        id,
                        phase: AiPhase::Ability,
                        kind: "useAbility".to_string(),
                        payload,
                        features: feats,
                        action_source_card_idx: Some(card_vocab_index_of(source.card_id) as u16),
                        action_target_card_idx: Some(card_vocab_index_of(target.card_id) as u16),
                    });
                }
                continue;
            }
        }
        if ability.discard_to_draw.is_some() {
            for discard_hand_index in discard_choice_indexes(side, -1) {
                let id = format!(
                    "ability:{}:{}:discard:{}",
                    source.uid, ability.name, discard_hand_index
                );
                let payload = json!({
                    "sourceUid": source.uid,
                    "abilityName": ability.name,
                    "discardHandIndex": discard_hand_index,
                });
                let chosen_card_id = side.hand.get(discard_hand_index).copied();
                let feats = build_features(FeatureInput {
                    score: score - discard_hand_index as f64 * 0.2,
                    phase: AiPhase::Ability,
                    kind: "useAbility",
                    source_card_id: Some(source.card_id),
                    choice_card_id: chosen_card_id,
                    target: Some(&source),
                    target_slot: Some(slot as f64),
                    ..Default::default()
                });
                out.push(LegalAiAction {
                    id,
                    phase: AiPhase::Ability,
                    kind: "useAbility".to_string(),
                    payload,
                    features: feats,
                    action_source_card_idx: Some(card_vocab_index_of(source.card_id) as u16),
                    action_target_card_idx: Some(card_vocab_index_of(source.card_id) as u16),
                });
            }
            continue;
        }
        // Generic fallback — no choices.
        let id = format!("ability:{}:{}", source.uid, ability.name);
        let payload = json!({
            "sourceUid": source.uid,
            "abilityName": ability.name,
        });
        let feats = build_features(FeatureInput {
            score,
            phase: AiPhase::Ability,
            kind: "useAbility",
            source_card_id: Some(source.card_id),
            target: Some(&source),
            target_slot: Some(slot as f64),
            ..Default::default()
        });
        out.push(LegalAiAction {
            id,
            phase: AiPhase::Ability,
            kind: "useAbility".to_string(),
            payload,
            features: feats,
            action_source_card_idx: Some(card_vocab_index_of(source.card_id) as u16),
            action_target_card_idx: Some(card_vocab_index_of(source.card_id) as u16),
        });
    }
    out
}

fn enumerate_combat_actions(state: &GameState, side: &SideState) -> Vec<LegalAiAction> {
    let mut out: Vec<LegalAiAction> = Vec::new();
    if side.active.is_some() {
        // build_combat_candidates needs CombatDeps; supply the same refs the
        // TS source uses.
        let mut refresh = refresh_continuous_hp;
        let choose_fn: fn(&SideState) -> i32 = choose_preferred_active_index;
        let mut deps = CombatDeps {
            refresh_continuous_effects: &mut refresh,
            choose_preferred_active_index: &choose_fn,
        };
        let candidates = build_combat_candidates(state, side.id, &mut deps, None);
        let opponent = state.side(side.id.opposite());
        for (index, candidate) in candidates.iter().enumerate() {
            let target_uid_opt = match &candidate.decision {
                AiCombatDecision::Attack(a) => a.attack_target_uid,
                AiCombatDecision::EndTurn => None,
            };
            let target: Option<&UmamusumeInstance> = match target_uid_opt {
                Some(uid) => get_all_umamusume(opponent)
                    .into_iter()
                    .find(|u| u.uid == uid),
                None => None,
            };
            let source_card_id = combat_source_card_id(side, &candidate.decision);
            // Decision kind tag.
            let decision_kind = match &candidate.decision {
                AiCombatDecision::Attack(_) => "attack",
                AiCombatDecision::EndTurn => "endTurn",
            };
            // `kind` for the LegalAiAction; TS replaces "attack" with
            // "retreatAttack" when retreat_target_uid is set.
            let kind = match &candidate.decision {
                AiCombatDecision::Attack(a) if a.retreat_target_uid.is_some() => "retreatAttack",
                _ => decision_kind,
            };
            let lethal_bonus = if candidate.lethal_target { 100.0 } else { 0.0 };
            let safe_bonus = if candidate.keeps_safe { 20.0 } else { 0.0 };
            let score = candidate.score + lethal_bonus + safe_bonus;
            let ends_turn = matches!(candidate.decision, AiCombatDecision::Attack(_));
            // v3.8 cross-bit context: pass attacker side, opponent active,
            // and the bench Uma that becomes active on retreatAttack swap.
            let retreat_swap_target: Option<&UmamusumeInstance> = match &candidate.decision {
                AiCombatDecision::Attack(a) => a
                    .retreat_target_uid
                    .and_then(|uid| side.bench.iter().find(|u| u.uid == uid)),
                AiCombatDecision::EndTurn => None,
            };
            // Payload: serialize the decision under the "decision" key with
            // the TS-equivalent shape. AiCombatDecision serialization uses
            // `tag = "kind"` already.
            let decision_value =
                serde_json::to_value(&candidate.decision).unwrap_or(Value::Null);
            let payload = json!({ "decision": decision_value });
            let id = format!("combat:{}:{}", candidate.id, index);
            let feats = build_features(FeatureInput {
                score,
                phase: AiPhase::Combat,
                // TS sets `kind: candidate.decision.kind` — that's the raw
                // decision kind ("attack" / "endTurn"), NOT the action
                // kind (which can be "retreatAttack"). features feed off
                // the decision kind.
                kind: decision_kind,
                target_value: Some(candidate.target_value),
                lethal_target: Some(candidate.lethal_target),
                ends_turn: Some(ends_turn),
                source_card_id,
                target,
                attacker_side: Some(side),
                defender_active: opponent.active.as_ref(),
                retreat_swap_target,
                ..Default::default()
            });
            out.push(LegalAiAction {
                id,
                phase: AiPhase::Combat,
                kind: kind.to_string(),
                payload,
                features: feats,
                action_source_card_idx: source_card_id.map(|c| card_vocab_index_of(c) as u16),
                action_target_card_idx: target.map(|t| card_vocab_index_of(t.card_id) as u16),
            });
        }
    }
    if out.is_empty() {
        vec![pass_action(AiPhase::Combat)]
    } else {
        out
    }
}

fn combat_source_card_id(side: &SideState, decision: &AiCombatDecision) -> Option<CardId> {
    let attack = match decision {
        AiCombatDecision::Attack(a) => a,
        AiCombatDecision::EndTurn => return None,
    };
    if let Some(retreat_uid) = attack.retreat_target_uid {
        return side
            .bench
            .iter()
            .find(|u| u.uid == retreat_uid)
            .map(|u| u.card_id);
    }
    side.active.as_ref().map(|a| a.card_id)
}

fn enumerate_stadium_or_end_actions(state: &GameState, side_id: SideId) -> Vec<LegalAiAction> {
    let mut out: Vec<LegalAiAction> = Vec::new();
    if can_use_stadium(state, side_id) {
        let feats = build_features(FeatureInput {
            score: 35.0,
            phase: AiPhase::StadiumOrEnd,
            kind: "useStadium",
            ends_turn: Some(true),
            ..Default::default()
        });
        out.push(LegalAiAction {
            id: "stadium:use".to_string(),
            phase: AiPhase::StadiumOrEnd,
            kind: "useStadium".to_string(),
            payload: json!({}),
            features: feats,
            action_source_card_idx: None,
            action_target_card_idx: None,
        });
    }
    let feats_end = build_features(FeatureInput {
        score: 1.0,
        phase: AiPhase::StadiumOrEnd,
        kind: "endTurn",
        ends_turn: Some(true),
        ..Default::default()
    });
    out.push(LegalAiAction {
        id: "turn:end".to_string(),
        phase: AiPhase::StadiumOrEnd,
        kind: "endTurn".to_string(),
        payload: json!({}),
        features: feats_end,
        action_source_card_idx: None,
        action_target_card_idx: None,
    });
    out
}

// ---------------------------------------------------------------------------
// Trainer choice expansion.
// ---------------------------------------------------------------------------

fn enumerate_trainer_choices(
    state: &GameState,
    side: &SideState,
    card: &TrainerCard,
    hand_index: usize,
) -> Vec<PlayChoices> {
    let mut choices: Vec<PlayChoices> = vec![PlayChoices::default()];
    if card.effect.discard_other_card == Some(true) {
        let additions: Vec<PlayChoices> = discard_choice_indexes(side, hand_index as i32)
            .into_iter()
            .map(|discard_hand_index| PlayChoices {
                discard_hand_index: Some(discard_hand_index),
                ..PlayChoices::default()
            })
            .collect();
        choices = expand_choices(&choices, &additions);
    }
    if card.effect.search_umamusume == Some(true)
        || card.effect.search_evolution_umamusume == Some(true)
    {
        let additions: Vec<PlayChoices> = search_deck_indexes(side, card)
            .into_iter()
            .map(|deck_card_index| PlayChoices {
                deck_card_index: Some(deck_card_index),
                ..PlayChoices::default()
            })
            .collect();
        choices = expand_choices(&choices, &additions);
    }
    if card.effect.attach_energy_from_zone_to_bench.is_some() && !side.bench.is_empty() {
        let additions: Vec<PlayChoices> = side
            .bench
            .iter()
            .map(|target| PlayChoices {
                umamusume_target_uid: Some(target.uid),
                ..PlayChoices::default()
            })
            .collect();
        choices = expand_choices(&choices, &additions);
    }
    if card.effect.heal.is_some()
        && matches!(
            card.effect.heal_target,
            Some(crate::core::effects::TrainerHealTarget::Any)
        )
    {
        let targets: Vec<&UmamusumeInstance> = get_all_umamusume(side)
            .into_iter()
            .filter(|u| u.hp < u.max_hp)
            .collect();
        if !targets.is_empty() {
            let additions: Vec<PlayChoices> = targets
                .iter()
                .map(|target| PlayChoices {
                    umamusume_target_uid: Some(target.uid),
                    ..PlayChoices::default()
                })
                .collect();
            choices = expand_choices(&choices, &additions);
        }
    }
    if card.trainer_type == TrainerType::Tool {
        let additions: Vec<PlayChoices> = get_tool_targets(side)
            .into_iter()
            .map(|target| PlayChoices {
                umamusume_target_uid: Some(target.uid),
                ..PlayChoices::default()
            })
            .collect();
        choices = expand_choices(&choices, &additions);
    }
    if card.effect.rainbow_uncap_crystal == Some(true) {
        let rainbow_targets = get_rainbow_uncap_targets(state, side);
        let mut rainbow_choices: Vec<PlayChoices> = Vec::new();
        for target in rainbow_targets {
            let options = get_rainbow_uncap_evolution_hand_options(side, target);
            for (option_hand_index, _) in options {
                rainbow_choices.push(PlayChoices {
                    umamusume_target_uid: Some(target.uid),
                    rainbow_evolution_hand_index: Some(option_hand_index),
                    ..PlayChoices::default()
                });
            }
        }
        choices = expand_choices(&choices, &rainbow_choices);
    }
    if choices.is_empty() {
        vec![PlayChoices::default()]
    } else {
        choices
    }
}

fn expand_choices(base: &[PlayChoices], additions: &[PlayChoices]) -> Vec<PlayChoices> {
    if additions.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<PlayChoices> = Vec::with_capacity(base.len() * additions.len());
    for choice in base {
        for addition in additions {
            out.push(merge_choices(choice, addition));
        }
    }
    out
}

/// Mirror of TS `{ ...choice, ...addition }` — later fields win.
fn merge_choices(base: &PlayChoices, addition: &PlayChoices) -> PlayChoices {
    PlayChoices {
        discard_hand_index: addition.discard_hand_index.or(base.discard_hand_index),
        deck_card_index: addition.deck_card_index.or(base.deck_card_index),
        umamusume_target_uid: addition.umamusume_target_uid.or(base.umamusume_target_uid),
        rainbow_evolution_hand_index: addition
            .rainbow_evolution_hand_index
            .or(base.rainbow_evolution_hand_index),
    }
}

fn discard_choice_indexes(side: &SideState, played_hand_index: i32) -> Vec<usize> {
    let mut scored: Vec<(usize, f64)> = side
        .hand
        .iter()
        .enumerate()
        .map(|(hand_index, &card_id)| {
            let extra = if hand_index as i32 == played_hand_index {
                10000.0
            } else {
                0.0
            };
            (hand_index, score_discard_candidate(card_id) + extra)
        })
        .collect();
    // Drop the played hand index from candidates.
    let mut filtered: Vec<(usize, f64)> = scored
        .drain(..)
        .filter(|&(hand_index, _)| hand_index as i32 != played_hand_index)
        .collect();
    // Ascending sort, stable.
    filtered.sort_by(|left, right| {
        left.1
            .partial_cmp(&right.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    filtered
        .into_iter()
        .take(4)
        .map(|(hand_index, _)| hand_index)
        .collect()
}

fn search_deck_indexes(side: &SideState, trainer: &TrainerCard) -> Vec<usize> {
    let cat = catalog();
    let mut scored: Vec<(usize, CardId, f64)> = Vec::new();
    for (deck_card_index, &card_id) in side.deck.iter().enumerate() {
        let Some(card) = cat.get(card_id) else {
            continue;
        };
        let keep = if trainer.effect.search_evolution_umamusume == Some(true) {
            matches!(card, Card::Umamusume(u) if u.stage > 0)
        } else if trainer.effect.search_umamusume == Some(true) {
            matches!(card, Card::Umamusume(_))
        } else {
            false
        };
        if !keep {
            continue;
        }
        scored.push((deck_card_index, card_id, score_search_candidate(card_id)));
    }
    scored.sort_by(|left, right| {
        right
            .2
            .partial_cmp(&left.2)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored
        .into_iter()
        .take(8)
        .map(|(deck_card_index, _, _)| deck_card_index)
        .collect()
}

fn score_search_candidate(card_id: CardId) -> f64 {
    let cat = catalog();
    let Some(Card::Umamusume(card)) = cat.get(card_id) else {
        return 0.0;
    };
    let attack_damage = card.attacks.first().map(|a| a.damage).unwrap_or(0);
    card.hp as f64 + attack_damage as f64 * 1.5 + card.stage as f64 * 24.0
}

fn score_discard_candidate(card_id: CardId) -> f64 {
    let cat = catalog();
    match cat.get(card_id) {
        Some(Card::Trainer(t)) => {
            if t.trainer_type == TrainerType::Stadium {
                18.0
            } else {
                28.0
            }
        }
        Some(Card::Umamusume(c)) => {
            let attack_damage = c.attacks.first().map(|a| a.damage).unwrap_or(0);
            c.hp as f64 + attack_damage as f64 + c.stage as f64 * 35.0
        }
        None => 0.0,
    }
}

fn score_trainer_choices(side: &SideState, choices: &PlayChoices) -> f64 {
    let target = choices
        .umamusume_target_uid
        .and_then(|uid| get_all_umamusume(side).into_iter().find(|u| u.uid == uid));
    let target_score = target.map(score_umamusume).unwrap_or(0.0);
    let discard_hand_index = choices.discard_hand_index.unwrap_or(0) as f64;
    target_score - discard_hand_index
}

fn choice_key(choices: &PlayChoices) -> String {
    fn opt_or_x<T: ToString>(opt: Option<T>) -> String {
        opt.map(|v| v.to_string()).unwrap_or_else(|| "x".to_string())
    }
    [
        opt_or_x(choices.discard_hand_index),
        opt_or_x(choices.deck_card_index),
        opt_or_x(choices.umamusume_target_uid),
        opt_or_x(choices.rainbow_evolution_hand_index),
    ]
    .join(":")
}

fn get_choice_card_id(side: &SideState, choices: &PlayChoices) -> Option<CardId> {
    if let Some(idx) = choices.deck_card_index {
        return side.deck.get(idx).copied();
    }
    if let Some(idx) = choices.discard_hand_index {
        return side.hand.get(idx).copied();
    }
    if let Some(idx) = choices.rainbow_evolution_hand_index {
        return side.hand.get(idx).copied();
    }
    None
}

// ---------------------------------------------------------------------------
// Pass action + with_pass helper.
// ---------------------------------------------------------------------------

fn with_pass(phase: AiPhase, mut actions: Vec<LegalAiAction>) -> Vec<LegalAiAction> {
    if actions.is_empty() {
        vec![pass_action(phase)]
    } else {
        actions.push(pass_action(phase));
        actions
    }
}

fn pass_action(phase: AiPhase) -> LegalAiAction {
    let feats = build_features(FeatureInput {
        score: 0.0,
        phase,
        kind: "pass",
        ..Default::default()
    });
    LegalAiAction {
        id: format!("{}:pass", phase_tag(phase)),
        phase,
        kind: "pass".to_string(),
        payload: json!({}),
        features: feats,
        action_source_card_idx: None,
        action_target_card_idx: None,
    }
}

fn phase_tag(phase: AiPhase) -> &'static str {
    match phase {
        AiPhase::Setup => "setup",
        AiPhase::PendingChoice => "pendingChoice",
        AiPhase::Bench => "bench",
        AiPhase::TrainerBefore => "trainerBefore",
        AiPhase::Evolve => "evolve",
        AiPhase::Attach => "attach",
        AiPhase::TrainerAfter => "trainerAfter",
        AiPhase::Ability => "ability",
        AiPhase::Combat => "combat",
        AiPhase::StadiumOrEnd => "stadiumOrEnd",
    }
}

// ---------------------------------------------------------------------------
// Feature vector builder.
// ---------------------------------------------------------------------------

#[derive(Default)]
struct FeatureInput<'a> {
    score: f64,
    phase: AiPhase,
    kind: &'a str,
    source_card_id: Option<CardId>,
    source_hand_index: Option<f64>,
    choice_card_id: Option<CardId>,
    target: Option<&'a UmamusumeInstance>,
    target_slot: Option<f64>,
    target_value: Option<f64>,
    lethal_target: Option<bool>,
    ends_turn: Option<bool>,
    // v3.8 cross-bit context (slots [48:52]). Optional — call sites
    // populate only on action kinds where the slot is meaningful.
    attacker_side: Option<&'a SideState>,
    defender_active: Option<&'a UmamusumeInstance>,
    attach_color: Option<EnergyType>,
    retreat_swap_target: Option<&'a UmamusumeInstance>,
}

impl Default for AiPhase {
    fn default() -> Self {
        AiPhase::Setup
    }
}

fn build_features(input: FeatureInput<'_>) -> Vec<f64> {
    let cat = catalog();
    let mut v = vec![0.0f64; ACTION_FEATURE_COUNT];
    v[0] = input.score / 100.0;
    v[1] = phase_index(input.phase) as f64 / 10.0;
    v[2] = kind_index(input.kind) as f64 / 16.0;
    v[3] = match input.source_card_id {
        Some(cid) => catalog_card_id_str(cid).map(hash_to_unit).unwrap_or(0.0),
        None => 0.0,
    };
    v[4] = match input.source_hand_index {
        Some(idx) => idx / 10.0,
        None => -1.0,
    };
    v[5] = match input.target {
        Some(t) => catalog_card_id_str(t.card_id).map(hash_to_unit).unwrap_or(0.0),
        None => 0.0,
    };
    v[6] = input.target.map(|t| t.stage as f64).unwrap_or(0.0);
    v[7] = match input.target {
        Some(t) => t.hp as f64 / (t.max_hp as f64).max(1.0),
        None => 0.0,
    };
    v[8] = match input.target {
        Some(t) => attached_energy_count(t) as f64 / 6.0,
        None => 0.0,
    };
    v[9] = match input.target_slot {
        Some(slot) => slot / 4.0,
        None => -1.0,
    };
    v[10] = input.target_value.map(|tv| tv / 200.0).unwrap_or(0.0);
    v[11] = if input.ends_turn.unwrap_or(false) { 1.0 } else { 0.0 };
    let source_card: Option<&Card> = input.source_card_id.and_then(|cid| cat.get(cid));
    v[12] = if input.kind == "pass" { 1.0 } else { 0.0 };
    v[13] = match source_card {
        Some(Card::Umamusume(_)) => 1.0,
        Some(Card::Trainer(_)) => 0.5,
        None => 0.0,
    };
    v[14] = match source_card {
        Some(Card::Umamusume(u)) => u.stage as f64 / 2.0,
        _ => 0.0,
    };
    v[15] = match source_card {
        Some(Card::Umamusume(u)) => u.hp as f64 / 180.0,
        _ => 0.0,
    };
    v[16] = match source_card {
        Some(Card::Umamusume(u)) => primary_attack(u)
            .map(|a| a.damage as f64 / 150.0)
            .unwrap_or(0.0),
        _ => 0.0,
    };
    v[17] = match source_card {
        Some(Card::Umamusume(u)) => primary_attack(u)
            .map(|a| energy_cost_total_sum(&a.cost) as f64 / 5.0)
            .unwrap_or(0.0),
        _ => 0.0,
    };
    v[18] = matches_trainer_type(source_card, TrainerType::Supporter);
    v[19] = matches_trainer_type(source_card, TrainerType::Stadium);
    v[20] = matches_trainer_type(source_card, TrainerType::Tool);
    v[21] = match source_card {
        Some(Card::Trainer(t)) => t.effect.draw.unwrap_or(0) as f64 / 5.0,
        _ => 0.0,
    };
    v[22] = match source_card {
        Some(Card::Trainer(t)) => {
            if t.effect.search_umamusume == Some(true)
                || t.effect.search_evolution_umamusume == Some(true)
                || t.effect.search_random_basic_umamusume == Some(true)
            {
                1.0
            } else {
                0.0
            }
        }
        _ => 0.0,
    };
    v[23] = match source_card {
        Some(Card::Trainer(t)) => {
            if t.effect.extra_energy_attach.is_some()
                || t.effect.attach_energy_from_zone_to_bench.is_some()
            {
                1.0
            } else {
                0.0
            }
        }
        _ => 0.0,
    };
    v[24] = match source_card {
        Some(Card::Trainer(t)) => t.effect.active_attack_damage_bonus.unwrap_or(0) as f64 / 100.0,
        _ => 0.0,
    };
    v[25] = input.target.map(|t| t.max_hp as f64 / 180.0).unwrap_or(0.0);
    v[26] = if input.lethal_target.unwrap_or(false) { 1.0 } else { 0.0 };
    v[27] = input
        .target
        .map(|t| t.special_conditions.len() as f64 / 4.0)
        .unwrap_or(0.0);
    v[28] = match source_card {
        Some(Card::Trainer(t)) => t.effect.heal.unwrap_or(0) as f64 / 100.0,
        _ => 0.0,
    };
    v[29] = if input.kind == "attack" || input.kind == "retreatAttack" {
        1.0
    } else {
        0.0
    };
    v[30] = if input.kind == "useAbility" { 1.0 } else { 0.0 };
    v[31] = if input.kind == "endTurn" { 1.0 } else { 0.0 };
    v[32] = matches_trainer_type(source_card, TrainerType::Item);
    v[33] = match source_card {
        Some(Card::Trainer(t)) => {
            if t.effect.gust_opponent == Some(true)
                || t.effect.discard_random_opponent_active_energy == Some(true)
                || t.effect.disable_tools == Some(true)
            {
                1.0
            } else {
                0.0
            }
        }
        _ => 0.0,
    };
    v[34] = match source_card {
        Some(Card::Trainer(t)) => {
            if t.effect.heal.is_some() || t.effect.recover_active_special_conditions == Some(true)
            {
                1.0
            } else {
                0.0
            }
        }
        _ => 0.0,
    };
    v[35] = match source_card {
        Some(Card::Trainer(t)) => {
            if t.effect.discard_other_card == Some(true)
                || t.effect.random_basic_umamusume_from_discard == Some(true)
            {
                1.0
            } else {
                0.0
            }
        }
        _ => 0.0,
    };
    v[36] = match source_card {
        Some(Card::Trainer(t)) => {
            if t.effect.shuffle_hand_into_deck_draw.is_some()
                || t.effect.rainbow_uncap_crystal == Some(true)
                || t.effect.basic_hp_bonus.is_some()
                || t.effect.global_retreat_cost_reduction.is_some()
            {
                1.0
            } else {
                0.0
            }
        }
        _ => 0.0,
    };
    v[37] = match source_card {
        Some(Card::Umamusume(u)) if u.ability.is_some() => 1.0,
        _ => 0.0,
    };
    v[38] = match source_card {
        Some(Card::Umamusume(u)) => typed_attack_cost(u) as f64 / 4.0,
        _ => 0.0,
    };
    v[39] = match source_card {
        Some(Card::Umamusume(u)) => colorless_attack_cost(u) as f64 / 4.0,
        _ => 0.0,
    };
    v[40] = match source_card {
        Some(Card::Umamusume(u)) => {
            if has_flexible_attack_target(u) {
                1.0
            } else {
                0.0
            }
        }
        _ => 0.0,
    };
    v[41] = match source_card {
        Some(Card::Umamusume(u)) => {
            if has_conditional_attack_or_ability(u) {
                1.0
            } else {
                0.0
            }
        }
        _ => 0.0,
    };
    let choice_card: Option<&Card> = input.choice_card_id.and_then(|cid| cat.get(cid));
    v[42] = choice_card.map(card_role_kind).unwrap_or(0.0);
    v[43] = choice_card.map(card_role_progression).unwrap_or(0.0);
    v[44] = choice_card.map(card_role_output).unwrap_or(0.0);
    v[45] = choice_card.map(card_role_utility).unwrap_or(0.0);
    v[46] = input.target.map(attack_readiness).unwrap_or(0.0);
    v[47] = input
        .target
        .map(|t| typed_energy_deficit(t) as f64 / 4.0)
        .unwrap_or(0.0);
    // v3.8 slots [48:52] — mirrors TS `v38SwapInAttackReady`,
    // `v38ExpectedDamageNorm`, `v38AttachColorMatchesTypedNeed`,
    // `v38AttachCompletesTypedThreshold`. See
    // `frontend/src/game/engine/ai-policy/actions.ts` and scoping doc
    // §4.5 for slot definitions.
    v[48] = v38_swap_in_attack_ready(&input);
    v[49] = v38_expected_damage_norm(&input);
    v[50] = v38_attach_color_matches_typed_need(&input);
    v[51] = v38_attach_completes_typed_threshold(&input);
    v
}

/// v3.8 slot 48 — `swap_in_attack_ready`. Fires when the action carries
/// a `retreat_swap_target` (= retreat/retreatAttack semantics; the combat
/// enumerator supplies the target iff `decision.retreat_target_uid` is
/// set). 1 iff the swap-in Uma has enough energy for its primary attack.
fn v38_swap_in_attack_ready(input: &FeatureInput<'_>) -> f64 {
    let Some(swap_in) = input.retreat_swap_target else {
        return 0.0;
    };
    let cat = catalog();
    let Some(Card::Umamusume(card)) = cat.get(swap_in.card_id) else {
        return 0.0;
    };
    let Some(attack) = primary_attack(card) else {
        return 0.0;
    };
    if has_enough_energy(swap_in, &attack.cost) {
        1.0
    } else {
        0.0
    }
}

/// v3.8 slot 49 — `expected_damage_norm`. RESTRICTED to base +
/// `activeAttackDamageBonus` + weakness (no conditional bonuses, no
/// coin-flip, no discard). Fires for kinds in {attack, retreatAttack,
/// useAbility} — but the combat enumerator passes the RAW decision kind
/// (`"attack"` regardless of retreatAttack), so we accept all three kind
/// strings and require both `source_card_id` and `defender_active`.
fn v38_expected_damage_norm(input: &FeatureInput<'_>) -> f64 {
    if input.kind != "attack" && input.kind != "retreatAttack" && input.kind != "useAbility" {
        return 0.0;
    }
    let Some(source_id) = input.source_card_id else {
        return 0.0;
    };
    let Some(defender) = input.defender_active else {
        return 0.0;
    };
    let cat = catalog();
    let Some(Card::Umamusume(attacker_card)) = cat.get(source_id) else {
        return 0.0;
    };
    let Some(primary) = primary_attack(attacker_card) else {
        return 0.0;
    };
    let mut damage: f64 = primary.damage as f64;
    if let Some(side) = input.attacker_side {
        damage += side.active_attack_damage_bonus as f64;
    }
    if damage > 0.0 {
        if let Some(Card::Umamusume(def_card)) = cat.get(defender.card_id) {
            if def_card.weakness.r#type == attacker_card.r#type {
                damage += def_card.weakness.amount as f64;
            }
        }
    }
    if damage < 0.0 {
        damage = 0.0;
    }
    damage / 300.0
}

/// v3.8 slot 50 — `attach_color_matches_typed_need`. 1 iff
/// kind=attachEnergy AND attach_color reduces a typed deficit on the
/// target's primary attack cost.
fn v38_attach_color_matches_typed_need(input: &FeatureInput<'_>) -> f64 {
    if input.kind != "attachEnergy" {
        return 0.0;
    }
    let (Some(target), Some(color)) = (input.target, input.attach_color) else {
        return 0.0;
    };
    if color == EnergyType::Colorless {
        return 0.0;
    }
    let cat = catalog();
    let Some(Card::Umamusume(card)) = cat.get(target.card_id) else {
        return 0.0;
    };
    let Some(attack) = primary_attack(card) else {
        return 0.0;
    };
    let need = attack.cost.get(color) as i32;
    let have = target.energies[color as usize] as i32;
    if need > have {
        1.0
    } else {
        0.0
    }
}

/// v3.8 slot 51 — `attach_completes_typed_threshold`. 1 iff
/// kind=attachEnergy AND post-attach target meets primary-attack typed
/// cost (colorless still allowed unmet).
fn v38_attach_completes_typed_threshold(input: &FeatureInput<'_>) -> f64 {
    if input.kind != "attachEnergy" {
        return 0.0;
    }
    let (Some(target), Some(color)) = (input.target, input.attach_color) else {
        return 0.0;
    };
    let cat = catalog();
    let Some(Card::Umamusume(card)) = cat.get(target.card_id) else {
        return 0.0;
    };
    let Some(attack) = primary_attack(card) else {
        return 0.0;
    };
    for t in ENERGY_TYPES {
        if t == EnergyType::Colorless {
            continue;
        }
        let need = attack.cost.get(t) as i32;
        let mut have = target.energies[t as usize] as i32;
        if t == color {
            have += 1;
        }
        if have < need {
            return 0.0;
        }
    }
    1.0
}

fn matches_trainer_type(card: Option<&Card>, t: TrainerType) -> f64 {
    match card {
        Some(Card::Trainer(c)) if c.trainer_type == t => 1.0,
        _ => 0.0,
    }
}

fn primary_attack(card: &UmamusumeCard) -> Option<&Attack> {
    card.attacks.first()
}

fn score_umamusume(umamusume: &UmamusumeInstance) -> f64 {
    let cat = catalog();
    let card = match cat.get(umamusume.card_id) {
        Some(Card::Umamusume(c)) => c,
        _ => return 0.0,
    };
    let damage = primary_attack(card).map(|a| a.damage).unwrap_or(0);
    umamusume.hp as f64
        + damage as f64
        + attached_energy_count(umamusume) as f64 * 18.0
        + umamusume.stage as f64 * 24.0
}

fn phase_index(phase: AiPhase) -> usize {
    // Same source-declaration order as TS `phaseIndex`. Matches
    // `AiPhase::index`.
    phase.index()
}

fn kind_index(kind: &str) -> i32 {
    // TS source-text order:
    // ["pass","setupChooseBoard","resolvePendingChoice","playBasic",
    //  "playTrainer","evolve","attachEnergy","useAbility","retreat",
    //  "retreatAttack","attack","useStadium","endTurn"]
    match kind {
        "pass" => 0,
        "setupChooseBoard" => 1,
        "resolvePendingChoice" => 2,
        "playBasic" => 3,
        "playTrainer" => 4,
        "evolve" => 5,
        "attachEnergy" => 6,
        "useAbility" => 7,
        "retreat" => 8,
        "retreatAttack" => 9,
        "attack" => 10,
        "useStadium" => 11,
        "endTurn" => 12,
        _ => -1, // TS `Array.indexOf` returns -1 on miss.
    }
}

// ---------------------------------------------------------------------------
// Card-role helpers — TS lines 573-606.
// ---------------------------------------------------------------------------

fn card_role_kind(card: &Card) -> f64 {
    match card {
        Card::Umamusume(_) => 1.0,
        Card::Trainer(t) => match t.trainer_type {
            TrainerType::Supporter => 0.75,
            TrainerType::Item => 0.55,
            TrainerType::Tool => 0.35,
            // TS falls through `return 0.2` after the three explicit
            // trainer branches — that's the "stadium" branch.
            TrainerType::Stadium => 0.2,
        },
    }
}

fn card_role_progression(card: &Card) -> f64 {
    match card {
        Card::Umamusume(u) => u.stage as f64 / 2.0,
        Card::Trainer(t) => match t.trainer_type {
            TrainerType::Supporter => 0.75,
            TrainerType::Item => 0.5,
            TrainerType::Tool => 0.35,
            TrainerType::Stadium => 0.2,
        },
    }
}

fn card_role_output(card: &Card) -> f64 {
    match card {
        Card::Umamusume(u) => {
            let hp_norm = u.hp as f64 / 180.0;
            let dmg_norm = primary_attack(u)
                .map(|a| a.damage as f64 / 150.0)
                .unwrap_or(0.0);
            hp_norm.max(dmg_norm)
        }
        Card::Trainer(t) => {
            // TS uses `effect.draw ?? effect.shuffleHandIntoDeckDraw ?? 0`.
            let draw_or_shuffle: f64 = match (t.effect.draw, t.effect.shuffle_hand_into_deck_draw) {
                (Some(d), _) => d as f64,
                (None, Some(s)) => s as f64,
                _ => 0.0,
            } / 5.0;
            let heal = t.effect.heal.unwrap_or(0) as f64 / 80.0;
            let bonus = t.effect.active_attack_damage_bonus.unwrap_or(0) as f64 / 100.0;
            draw_or_shuffle.max(heal).max(bonus)
        }
    }
}

fn card_role_utility(card: &Card) -> f64 {
    match card {
        Card::Umamusume(u) => {
            if u.ability.is_some() {
                1.0
            } else if has_flexible_attack_target(u) || has_conditional_attack_or_ability(u) {
                0.5
            } else {
                0.0
            }
        }
        Card::Trainer(t) => {
            let effect = &t.effect;
            if effect.search_umamusume == Some(true)
                || effect.search_evolution_umamusume == Some(true)
                || effect.search_random_basic_umamusume == Some(true)
                || effect.rainbow_uncap_crystal == Some(true)
            {
                return 1.0;
            }
            if effect.extra_energy_attach.is_some()
                || effect.attach_energy_from_zone_to_bench.is_some()
                || effect.gust_opponent == Some(true)
                || effect.discard_random_opponent_active_energy == Some(true)
            {
                return 0.85;
            }
            if effect.heal.is_some()
                || effect.retreat_cost_reduction.is_some()
                || effect.global_retreat_cost_reduction.is_some()
                || t.trainer_type == TrainerType::Tool
            {
                return 0.55;
            }
            0.25
        }
    }
}

fn typed_attack_cost(card: &UmamusumeCard) -> i32 {
    let Some(attack) = primary_attack(card) else {
        return 0;
    };
    let mut sum = 0i32;
    for t in ENERGY_TYPES {
        if t == EnergyType::Colorless {
            continue;
        }
        sum += attack.cost.get(t) as i32;
    }
    sum
}

fn colorless_attack_cost(card: &UmamusumeCard) -> i32 {
    primary_attack(card)
        .map(|a| a.cost.colorless() as i32)
        .unwrap_or(0)
}

fn has_flexible_attack_target(card: &UmamusumeCard) -> bool {
    let Some(attack) = primary_attack(card) else {
        return false;
    };
    matches!(attack.target_opponent, Some(AttackTarget::Any))
        || attack.bench_damage.is_some()
        || matches!(attack.heal_target, Some(HealTarget::Any))
}

fn has_conditional_attack_or_ability(card: &UmamusumeCard) -> bool {
    if card.ability.is_some() {
        return true;
    }
    let Some(attack) = primary_attack(card) else {
        return false;
    };
    attack.coin_bonus.is_some()
        || attack.draw_on_heads.is_some()
        || attack.discard_energy.is_some()
        || attack.damage_per_attached_energy.is_some()
        || attack.damage_per_unique_attached_energy.is_some()
        || attack.damage_per_umamusume_in_play.is_some()
        || attack.attack_damage_bonus_if_tool_attached.is_some()
        || attack.attack_damage_bonus_if_discard_hand_card.is_some()
        || attack.attack_damage_bonus_per_discarded_hand_card.is_some()
        || attack.shuffle_self_into_deck.is_some()
        || attack.switch_self_after_attack.is_some()
        || attack.prevent_damage_next_turn.is_some()
        || attack.bonus_if_took_damage_last_turn.is_some()
        || attack.cannot_attack_next_turn.is_some()
        || attack.inflict_special_condition.is_some()
}

fn attack_readiness(target: &UmamusumeInstance) -> f64 {
    let cat = catalog();
    let Some(Card::Umamusume(card)) = cat.get(target.card_id) else {
        return 0.0;
    };
    let Some(attack) = primary_attack(card) else {
        return 0.0;
    };
    let total_cost = energy_cost_total_sum(&attack.cost) as f64;
    let energy_ratio = attached_energy_count(target) as f64 / total_cost.max(1.0);
    if has_enough_energy(target, &attack.cost) {
        1.0_f64.max(energy_ratio)
    } else {
        0.99_f64.min(energy_ratio)
    }
}

fn typed_energy_deficit(target: &UmamusumeInstance) -> i32 {
    let cat = catalog();
    let Some(Card::Umamusume(card)) = cat.get(target.card_id) else {
        return 0;
    };
    let Some(attack) = primary_attack(card) else {
        return 0;
    };
    let mut sum = 0i32;
    for t in ENERGY_TYPES {
        if t == EnergyType::Colorless {
            continue;
        }
        let cost_amt = attack.cost.get(t) as i32;
        let have = target.energies[t as usize] as i32;
        sum += (cost_amt - have).max(0);
    }
    sum
}

fn energy_cost_total_sum(cost: &EnergyCost) -> i32 {
    let mut sum = 0i32;
    for t in EnergyType::ALL {
        sum += cost.get(t) as i32;
    }
    sum
}

// ---------------------------------------------------------------------------
// Catalog id lookup helpers.
// ---------------------------------------------------------------------------

fn catalog_card_id_str(card_id: CardId) -> Option<&'static str> {
    catalog().interner.resolve(card_id)
}

fn card_vocab_index_of(card_id: CardId) -> u32 {
    let s = catalog_card_id_str(card_id);
    card_vocab_index(s)
}

fn energy_type_tag(t: EnergyType) -> &'static str {
    match t {
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

// ---------------------------------------------------------------------------
// `hashToUnit` — TS lines 659-666.
// ---------------------------------------------------------------------------

/// FNV-1a style hash mapped to `[0, 1)`. Matches TS `hashToUnit` byte-for-
/// byte: starts at `2166136261`, XOR `charCodeAt(i)` per UTF-16 code unit,
/// `Math.imul(hash, 16777619)` per step, then `(hash >>> 0) / 4294967295`.
///
/// **Important**: TS `text.charCodeAt(i)` iterates UTF-16 code units. For
/// ASCII card ids (which is what the engine uses), this matches Rust
/// `str.chars().map(|c| c as u32)` exactly. If a card id ever contains a
/// non-BMP char this would need to be re-checked; the catalog is all-ASCII
/// today.
fn hash_to_unit(text: &str) -> f64 {
    let mut hash: u32 = 2166136261;
    // Iterate UTF-16 code units to mirror TS `charCodeAt`.
    for unit in text.encode_utf16() {
        hash ^= unit as u32;
        hash = hash.wrapping_mul(16777619);
    }
    hash as f64 / 4294967295.0
}

// ---------------------------------------------------------------------------
// Payload helpers.
// ---------------------------------------------------------------------------

fn play_choices_to_value(choices: &PlayChoices) -> Value {
    // TS `PlayChoices` payload only includes set fields; serde's
    // `skip_serializing_if = "Option::is_none"` would normally do this but
    // PlayChoices lacks that directive. Build the JSON map manually.
    let mut m = Map::new();
    if let Some(idx) = choices.discard_hand_index {
        m.insert("discardHandIndex".into(), json!(idx));
    }
    if let Some(idx) = choices.deck_card_index {
        m.insert("deckCardIndex".into(), json!(idx));
    }
    if let Some(uid) = choices.umamusume_target_uid {
        m.insert("umamusumeTargetUid".into(), json!(uid));
    }
    if let Some(idx) = choices.rainbow_evolution_hand_index {
        m.insert("rainbowEvolutionHandIndex".into(), json!(idx));
    }
    Value::Object(m)
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::{AiDeckStyle, AiDifficulty, SideId};
    use crate::core::state::{CurrentSide, Phase, SetupState, SideState};
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

    fn play_state() -> GameState {
        GameState {
            phase: Phase::Play,
            setup: None,
            pending_player_choice: None,
            sides: [empty_side(SideId::Player), empty_side(SideId::Opponent)],
            current_side: CurrentSide::Player,
            opponent_turn_step: None,
            stadium: None,
            turn_deadline_ms: None,
            turn_number: 2,
            first_player: SideId::Player,
            turns_taken_by_side: [1, 1],
            ai_difficulty: AiDifficulty::Normal,
            human_by_side: [false, false],
            ai_deck_style_by_side: [AiDeckStyle::Balanced, AiDeckStyle::Balanced],
            game_over: false,
            winner: None,
            log: std::collections::VecDeque::new(),
        }
    }

    fn setup_state() -> GameState {
        let mut s = play_state();
        s.phase = Phase::Setup;
        s.setup = Some(SetupState {
            coin_choice: None,
            coin_flip_result: None,
            opening_hands: [ArrayVec::new(), ArrayVec::new()],
            opening_hands_dealt: true,
            ready_by_side: [false, false],
            opponent_revealed: false,
            countdown_seconds_remaining: None,
        });
        s
    }

    fn first_basic_card_id() -> Option<CardId> {
        let cat = catalog();
        for (idx, card) in cat.cards.iter().enumerate() {
            if let Card::Umamusume(u) = card {
                if u.stage == 0 && !u.attacks.is_empty() {
                    return Some(CardId(idx as u16));
                }
            }
        }
        None
    }

    #[test]
    fn enumerate_legal_ai_actions_in_setup_returns_one_action() {
        let mut state = setup_state();
        let Some(basic_id) = first_basic_card_id() else {
            return; // No basic card available; nothing to test.
        };
        state.sides[SideId::Player as usize]
            .hand
            .push(basic_id);
        let actions = enumerate_legal_ai_actions(&state, SideId::Player);
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].kind, "setupChooseBoard");
        assert_eq!(actions[0].phase, AiPhase::Setup);
    }

    #[test]
    fn enumerate_returns_pass_when_no_legal_actions() {
        // Play phase, but the opponent is the current side — so AiPhase
        // resolves to StadiumOrEnd which always emits endTurn (+ stadium
        // if available). For Combat with no active and no bench, we should
        // get a single pass.
        let mut state = play_state();
        state.current_side = CurrentSide::Player;
        state.opponent_turn_step = Some(crate::core::constants::OpponentTurnStep::Attack);
        // No active, no bench → no candidates → pass.
        let actions = enumerate_legal_ai_actions(&state, SideId::Player);
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].kind, "pass");
        assert_eq!(actions[0].phase, AiPhase::Combat);
    }

    #[test]
    fn phase_index_matches_ai_phase_declaration_order() {
        assert_eq!(phase_index(AiPhase::Setup), 0);
        assert_eq!(phase_index(AiPhase::PendingChoice), 1);
        assert_eq!(phase_index(AiPhase::Bench), 2);
        assert_eq!(phase_index(AiPhase::TrainerBefore), 3);
        assert_eq!(phase_index(AiPhase::Evolve), 4);
        assert_eq!(phase_index(AiPhase::Attach), 5);
        assert_eq!(phase_index(AiPhase::TrainerAfter), 6);
        assert_eq!(phase_index(AiPhase::Ability), 7);
        assert_eq!(phase_index(AiPhase::Combat), 8);
        assert_eq!(phase_index(AiPhase::StadiumOrEnd), 9);
    }

    #[test]
    fn features_array_length_equals_count() {
        let pass = pass_action(AiPhase::Bench);
        assert_eq!(pass.features.len(), ACTION_FEATURE_COUNT);
    }

    #[test]
    fn card_vocab_index_used_correctly() {
        // Use the same setup-action machinery to land an action whose
        // source-card index should resolve to nonzero (because the basic
        // is in the vocab).
        let mut state = setup_state();
        let Some(basic_id) = first_basic_card_id() else {
            return;
        };
        state.sides[SideId::Player as usize]
            .hand
            .push(basic_id);
        let actions = enumerate_legal_ai_actions(&state, SideId::Player);
        assert_eq!(actions.len(), 1);
        let idx = actions[0]
            .action_source_card_idx
            .expect("setup action must record a source card");
        // The card may resolve to either a known vocab entry or
        // `unknown_index` (0) — we just assert the field is populated, not
        // its exact integer value (avoiding coupling tests to vocab churn).
        let _ = idx;
    }

    #[test]
    fn kind_index_table_matches_ts_order() {
        assert_eq!(kind_index("pass"), 0);
        assert_eq!(kind_index("setupChooseBoard"), 1);
        assert_eq!(kind_index("resolvePendingChoice"), 2);
        assert_eq!(kind_index("playBasic"), 3);
        assert_eq!(kind_index("playTrainer"), 4);
        assert_eq!(kind_index("evolve"), 5);
        assert_eq!(kind_index("attachEnergy"), 6);
        assert_eq!(kind_index("useAbility"), 7);
        assert_eq!(kind_index("retreat"), 8);
        assert_eq!(kind_index("retreatAttack"), 9);
        assert_eq!(kind_index("attack"), 10);
        assert_eq!(kind_index("useStadium"), 11);
        assert_eq!(kind_index("endTurn"), 12);
        assert_eq!(kind_index("nope"), -1);
    }

    #[test]
    fn hash_to_unit_is_deterministic_and_bounded() {
        let a = hash_to_unit("hello");
        let b = hash_to_unit("hello");
        assert_eq!(a, b);
        assert!(a >= 0.0 && a <= 1.0);
        // Distinct strings → distinct (with vanishing collision prob).
        assert_ne!(hash_to_unit("foo"), hash_to_unit("bar"));
    }

    // ----------------------------------------------------------------
    // v33-correctness-fix Fix 2-4: action-slot layout (10/26/28).
    // ----------------------------------------------------------------

    #[test]
    fn schema_version_bumped_to_four() {
        // v38-slim-feature-add bumped 3 → 4 (added slots [48:52]).
        assert_eq!(ACTION_FEATURE_SCHEMA_VERSION, 4);
    }

    #[test]
    fn action_feature_count_is_fifty_two() {
        // v38-slim-feature-add bumped 48 → 52 (4 new slots at [48:52]).
        assert_eq!(ACTION_FEATURE_COUNT, 52);
    }

    #[test]
    fn slot_10_holds_combat_target_value_normalised_by_200() {
        let feats = build_features(FeatureInput {
            score: 0.0,
            phase: AiPhase::Combat,
            kind: "attack",
            target_value: Some(150.0),
            ..Default::default()
        });
        assert_eq!(feats[10], 150.0 / 200.0);
        // Outside combat (no target_value): slot 10 == 0.
        let feats_setup = build_features(FeatureInput {
            score: 0.0,
            phase: AiPhase::Setup,
            kind: "setupChooseBoard",
            ..Default::default()
        });
        assert_eq!(feats_setup[10], 0.0);
    }

    #[test]
    fn slot_26_is_combat_lethal_flag() {
        let feats_lethal = build_features(FeatureInput {
            score: 0.0,
            phase: AiPhase::Combat,
            kind: "attack",
            lethal_target: Some(true),
            ..Default::default()
        });
        assert_eq!(feats_lethal[26], 1.0);
        let feats_nonlethal = build_features(FeatureInput {
            score: 0.0,
            phase: AiPhase::Combat,
            kind: "attack",
            lethal_target: Some(false),
            ..Default::default()
        });
        assert_eq!(feats_nonlethal[26], 0.0);
        // Outside combat / not provided: 0.
        let feats_unset = build_features(FeatureInput {
            score: 0.0,
            phase: AiPhase::Attach,
            kind: "attachEnergy",
            ..Default::default()
        });
        assert_eq!(feats_unset[26], 0.0);
    }

    #[test]
    fn slot_28_is_trainer_heal_amount_normalised_by_100() {
        // Find a trainer with a heal effect from the catalog. If none
        // exists in this catalog snapshot the test is vacuous, but the
        // build_features path is still exercised.
        let cat = catalog();
        let mut heal_card_id: Option<CardId> = None;
        let mut heal_amount: Option<i32> = None;
        for (idx, card) in cat.cards.iter().enumerate() {
            if let Card::Trainer(t) = card {
                if let Some(h) = t.effect.heal {
                    if h > 0 {
                        heal_card_id = Some(CardId(idx as u16));
                        heal_amount = Some(h);
                        break;
                    }
                }
            }
        }
        if let (Some(cid), Some(h)) = (heal_card_id, heal_amount) {
            let feats = build_features(FeatureInput {
                score: 0.0,
                phase: AiPhase::TrainerBefore,
                kind: "playTrainer",
                source_card_id: Some(cid),
                ..Default::default()
            });
            assert!(
                (feats[28] - h as f64 / 100.0).abs() < 1e-9,
                "expected slot28 = {} / 100 = {}, got {}",
                h, h as f64 / 100.0, feats[28]
            );
        }
        // Non-trainer source → 0.
        let feats_no_trainer = build_features(FeatureInput {
            score: 0.0,
            phase: AiPhase::Combat,
            kind: "attack",
            ..Default::default()
        });
        assert_eq!(feats_no_trainer[28], 0.0);
    }
}
