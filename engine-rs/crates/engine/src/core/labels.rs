//! Bit-identical port of `frontend/src/game/engine/core/labels.ts`.
//!
//! Display-only helpers. The Rust port carries them for completeness, but
//! the simulator's log buffer is out-of-band (not part of the fingerprint
//! contract — see `backend/src/sim/stateFingerprint.ts` lines 32–75 which
//! excludes `state.log`), so these strings only matter for human-readable
//! traces, not for state mutation.

use crate::core::catalog::{Card, UmamusumeCard};
use crate::core::constants::EnergyType;
use crate::core::state::{SideState, UmamusumeInstance};

pub fn energy_label(t: EnergyType) -> &'static str {
    t.label()
}

pub fn actor_name(side: &SideState) -> &'static str {
    match side.id {
        crate::core::constants::SideId::Player => "You",
        crate::core::constants::SideId::Opponent => "Opponent",
    }
}

pub fn actor_possessive(side: &SideState) -> &'static str {
    match side.id {
        crate::core::constants::SideId::Player => "Your",
        crate::core::constants::SideId::Opponent => "Opponent's",
    }
}

pub fn actor_lower_possessive(side: &SideState) -> &'static str {
    match side.id {
        crate::core::constants::SideId::Player => "your",
        crate::core::constants::SideId::Opponent => "opponent's",
    }
}

pub fn stage_name(stage: u8) -> String {
    if stage == 0 {
        "Basic".to_string()
    } else {
        format!("Stage {}", stage)
    }
}

pub fn format_umamusume_card_name(card: &UmamusumeCard) -> String {
    format!("{} ({})", card.name, stage_name(card.stage))
}

pub fn format_umamusume_instance_name(umamusume: &UmamusumeInstance) -> String {
    let cat = crate::core::catalog::catalog();
    match cat.get(umamusume.card_id) {
        Some(Card::Umamusume(u)) => format_umamusume_card_name(u),
        _ => String::new(),
    }
}

pub fn format_card_name(card: &Card) -> String {
    match card {
        Card::Umamusume(u) => format_umamusume_card_name(u),
        Card::Trainer(t) => t.name.clone(),
    }
}

pub fn pluralize(amount: i64, singular: &str, plural: Option<&str>) -> String {
    if amount == 1 {
        singular.to_string()
    } else {
        match plural {
            Some(p) => p.to_string(),
            None => format!("{}s", singular),
        }
    }
}
