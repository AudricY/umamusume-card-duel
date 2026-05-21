//! Pure-math MCTS helpers — bit-identical to `mcts.ts`.

use crate::core::constants::SideId;
use crate::core::state::GameState;

/// `mcts.ts:162` `mctsTerminalValue`. Clean ±1/0; no point-margin scaling.
pub fn mcts_terminal_value(state: &GameState, model_side: SideId) -> f64 {
    if !state.game_over || state.winner.is_none() {
        return 0.0;
    }
    if state.winner == Some(model_side) {
        1.0
    } else {
        -1.0
    }
}

/// `mcts.ts:356` `puctSelect`. Returns the index of the best child by
/// `Q + cPuct * P * sqrt(sumN) / (1 + N)`. Ties resolve to the lower
/// index because TS uses strict `>`.
///
/// `visits[i]`, `wsum[i]`, `priors[i]` must be the same length.
pub fn puct_select(visits: &[u32], wsum: &[f64], priors: &[f64], c_puct: f64) -> usize {
    debug_assert_eq!(visits.len(), wsum.len());
    debug_assert_eq!(visits.len(), priors.len());

    let sum_n: u32 = visits.iter().sum();
    let sqrt_sum = (sum_n.max(1) as f64).sqrt();

    let mut best_index = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    for i in 0..visits.len() {
        let n = visits[i];
        let q = if n > 0 { wsum[i] / n as f64 } else { 0.0 };
        let u = c_puct * priors[i] * sqrt_sum / (1.0 + n as f64);
        let score = q + u;
        if score > best_score {
            best_score = score;
            best_index = i;
        }
    }
    best_index
}

/// `mcts.ts:546` `entropy`.
pub fn entropy(probs: &[f64]) -> f64 {
    let mut h = 0.0;
    for &p in probs {
        if p > 0.0 {
            h -= p * p.ln();
        }
    }
    h
}

/// `mcts.ts:554` `argmax`. Ties → lowest index (TS uses strict `>`).
pub fn argmax(values: &[f64]) -> usize {
    let mut best = 0usize;
    let mut best_val = f64::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_resolves_ties_to_lowest_index() {
        assert_eq!(argmax(&[1.0, 1.0, 1.0]), 0);
        assert_eq!(argmax(&[0.5, 1.0, 1.0]), 1);
        assert_eq!(argmax(&[2.0, 1.0, 1.0]), 0);
    }

    #[test]
    fn entropy_of_uniform_is_log_n() {
        // Uniform over 4 outcomes: -4 * 0.25 * ln(0.25) = ln(4) ≈ 1.386
        let h = entropy(&[0.25, 0.25, 0.25, 0.25]);
        assert!((h - (4f64).ln()).abs() < 1e-12);
    }

    #[test]
    fn entropy_skips_zero_probs() {
        // -1*ln(1) = 0 for a degenerate distribution; zero-prob items
        // contribute zero (and avoid 0*log(0) NaN).
        assert_eq!(entropy(&[1.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn puct_select_with_zero_visits_uses_prior_only() {
        // sumN=0 → sqrtSum = sqrt(max(1,0)) = 1.
        // Q = 0 for each; U = cPuct * P[i] * 1 / (1 + 0) = cPuct * P[i].
        // Highest prior wins.
        let visits = [0u32, 0, 0];
        let wsum = [0.0, 0.0, 0.0];
        let priors = [0.1, 0.7, 0.2];
        assert_eq!(puct_select(&visits, &wsum, &priors, 1.5), 1);
    }

    #[test]
    fn puct_select_prefers_high_q_when_priors_equal() {
        let visits = [10u32, 10, 10];
        let wsum = [5.0, 9.0, 2.0]; // Q = 0.5, 0.9, 0.2
        let priors = [0.33, 0.33, 0.34];
        assert_eq!(puct_select(&visits, &wsum, &priors, 1.0), 1);
    }
}
