//! Bit-identical port of `frontend/src/game/engine/core/playTypes.ts`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlayChoices {
    pub discard_hand_index: Option<usize>,
    pub deck_card_index: Option<usize>,
    pub umamusume_target_uid: Option<u32>,
    pub rainbow_evolution_hand_index: Option<usize>,
}

impl PlayChoices {
    /// Mirror of `playRules.ts:63` `adjustHandChoices` — when a card at
    /// `hand_index` is removed, downshift any index-based field that
    /// referenced a later position; drop the field if it referenced the
    /// removed card itself.
    pub fn adjust_for_hand_removal(&mut self, hand_index: usize) {
        if let Some(idx) = self.discard_hand_index {
            if idx == hand_index {
                self.discard_hand_index = None;
            } else if idx > hand_index {
                self.discard_hand_index = Some(idx - 1);
            }
        }
        if let Some(idx) = self.rainbow_evolution_hand_index {
            if idx == hand_index {
                self.rainbow_evolution_hand_index = None;
            } else if idx > hand_index {
                self.rainbow_evolution_hand_index = Some(idx - 1);
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum PlayActionOutcome {
    CanPlay(PlayActionKind),
    CannotPlay { reason: String },
}

/// Mirror of `PlayAction` in `shared/src/types.ts:238-243`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum PlayActionKind {
    BenchBasic,
    Evolve { target_uid: u32 },
    AttachTool { target_uid: u32 },
    Trainer,
}
