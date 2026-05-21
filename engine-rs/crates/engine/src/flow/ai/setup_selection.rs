//! Bit-identical port of `frontend/src/app/gameUiHelpers.ts:chooseAiSetupSelection`.
//!
//! Selects the basic Umamusume that scores highest as `(hp + attack.damage * 1.8)`,
//! up to MAX_BENCH bench picks. Returns `None` if no basic is in hand.
//!
//! Used by the headless AI-vs-AI setup flow that the golden-trace recorder
//! depends on.

use crate::core::catalog::{catalog, Card};
use crate::core::constants::MAX_BENCH;
use crate::core::state::SideState;

pub struct SetupSelection {
    pub active_index: usize,
    pub bench_indexes: Vec<usize>,
}

/// Mirror of `chooseAiSetupSelection`. Float math is preserved verbatim
/// (`card.hp as f64 + attack.damage as f64 * 1.8`).
pub fn choose_ai_setup_selection(side: &SideState) -> Option<SetupSelection> {
    let cat = catalog();
    let mut basics: Vec<(usize, f64)> = side
        .hand
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(hand_index, cid)| {
            let Some(Card::Umamusume(u)) = cat.get(cid) else {
                return None;
            };
            if u.stage != 0 {
                return None;
            }
            let primary = u.attacks.first()?;
            let score = u.hp as f64 + primary.damage as f64 * 1.8;
            Some((hand_index, score))
        })
        .collect();

    // Stable sort, descending by score. TS uses Array.prototype.sort which
    // is stable; Rust slice::sort_by is stable too.
    basics.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let (active_index, _) = *basics.first()?;
    let bench_indexes: Vec<usize> = basics
        .iter()
        .skip(1)
        .take(MAX_BENCH)
        .map(|&(idx, _)| idx)
        .collect();

    Some(SetupSelection {
        active_index,
        bench_indexes,
    })
}
