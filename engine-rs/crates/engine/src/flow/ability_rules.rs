//! Bit-identical port of `frontend/src/game/engine/flow/abilityRules.ts`.
//!
//! Reads the active-side card's ability and gates it on the
//! `disableOtherUmamusumeAbilitiesWhileActive` suppressor, mirroring TS
//! sides-iteration order (`["player", "opponent"]`).

use crate::core::catalog::{catalog, Card};
use crate::core::effects::Ability;
use crate::core::state::{GameState, UmamusumeInstance};

pub fn get_umamusume_ability<'a>(
    state: &GameState,
    _side_for_ownership: crate::core::constants::SideId,
    umamusume: &UmamusumeInstance,
) -> Option<&'a Ability> {
    let cat = catalog();
    let card = match cat.get(umamusume.card_id)? {
        Card::Umamusume(u) => u,
        Card::Trainer(_) => return None,
    };
    let ability = card.ability.as_ref()?;
    if is_ability_suppressed(state, umamusume.uid) {
        return None;
    }
    Some(ability)
}

fn is_ability_suppressed(state: &GameState, umamusume_uid: u32) -> bool {
    match get_ability_suppressor_uid(state) {
        None => false,
        Some(suppressor) => suppressor != umamusume_uid,
    }
}

/// Walk both sides in `["player", "opponent"]` order to mirror TS source.
fn get_ability_suppressor_uid(state: &GameState) -> Option<u32> {
    let cat = catalog();
    for &side_id in &crate::core::constants::SideId::ALL {
        let side = state.side(side_id);
        let Some(active) = &side.active else {
            continue;
        };
        let Some(Card::Umamusume(card)) = cat.get(active.card_id) else {
            continue;
        };
        if let Some(ability) = &card.ability {
            if ability.disable_other_umamusume_abilities_while_active == Some(true) {
                return Some(active.uid);
            }
        }
    }
    None
}
