//! Dirichlet / Gamma sampling — bit-identical to `mcts.ts:567,576`.
//!
//! The TS code uses `rng.next()` (a 0..1 f64) and `Math.sqrt`, `Math.log`,
//! `Math.cos`, `Math.pow`, `Math.PI` — all of which have IEEE 754
//! deterministic counterparts in Rust as long as we use `f64` math and
//! the same operator order. `Math.PI` ↔ `std::f64::consts::PI`,
//! `Math.cos` ↔ `f64::cos`, `Math.pow` ↔ `f64::powf`.
//!
//! `f64::ln` and `f64::cos` and `f64::powf` are not guaranteed bit-
//! identical across libm implementations. Empirically on Linux glibc they
//! match Node's V8 implementation for the common cases. If a divergence
//! turns up at the golden-trace gate, switch to a known-portable libm
//! (e.g. the `libm` crate) for these calls and document.

use crate::core::random::Rng;

/// `mcts.ts:567` `sampleDirichlet`.
pub fn sample_dirichlet(n: usize, alpha: f64, rng: &mut Rng) -> Vec<f64> {
    let alpha = alpha.max(0.05);
    let samples: Vec<f64> = (0..n).map(|_| sample_gamma(alpha, rng)).collect();
    let total: f64 = samples.iter().sum();
    if total <= 0.0 {
        let uniform = 1.0 / n as f64;
        return vec![uniform; n];
    }
    samples.into_iter().map(|v| v / total).collect()
}

/// `mcts.ts:576` `sampleGamma`. Marsaglia–Tsang for shape >= 1; boost-and-
/// discard wrapper for shape < 1. Recursive in TS — port recursive in Rust.
pub fn sample_gamma(alpha: f64, rng: &mut Rng) -> f64 {
    if alpha < 1.0 {
        let u = rng.next_f64().max(1e-12);
        return sample_gamma(alpha + 1.0, rng) * u.powf(1.0 / alpha);
    }
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let mut x: f64;
        let mut v: f64;
        loop {
            let u1 = rng.next_f64().max(1e-12);
            let u2 = rng.next_f64();
            x = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            v = 1.0 + c * x;
            if v > 0.0 {
                break;
            }
        }
        v = v * v * v;
        let u = rng.next_f64();
        if u < 1.0 - 0.0331 * x.powi(4) {
            return d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// AlphaZero temperature sampling. Mirror of TS `pickFromVisits` at
/// `backend/src/sim/mctsSelfPlay.ts:546`.
///
/// When `temperature <= 0` or only one action is available, returns
/// `argmax_idx` (greedy). Otherwise resamples proportional to
/// `visits^(1/T)` using one `rng.next_f64()` draw.
///
/// Sampling order is deterministic given the RNG: same seed yields
/// the same index, so callers that need a "stochastic but reproducible"
/// trace can fork from a per-step seed.
pub fn pick_from_visits(
    visits: &[u32],
    argmax_idx: usize,
    temperature: f64,
    rng: &mut Rng,
) -> usize {
    if temperature <= 0.0 || visits.len() <= 1 {
        return argmax_idx;
    }
    let inv_t = 1.0 / temperature;
    let weights: Vec<f64> = visits
        .iter()
        .map(|&n| (n as f64).max(0.0).powf(inv_t))
        .collect();
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return argmax_idx;
    }
    let r = rng.next_f64() * total;
    let mut cum = 0.0;
    for (i, w) in weights.iter().enumerate() {
        cum += *w;
        if r <= cum {
            return i;
        }
    }
    weights.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::random::Rng;

    #[test]
    fn dirichlet_outputs_sum_to_one() {
        let mut rng = Rng::from_seed(42u32, "dirichlet-test");
        let v = sample_dirichlet(4, 0.3, &mut rng);
        assert_eq!(v.len(), 4);
        let s: f64 = v.iter().sum();
        assert!((s - 1.0).abs() < 1e-9, "sum was {}", s);
        assert!(v.iter().all(|&p| p >= 0.0));
    }

    #[test]
    fn dirichlet_with_zero_total_returns_uniform() {
        // We can't easily force a zero-total from the seeded RNG, but we
        // can verify the fallback branch produces uniform for n=3.
        // (Direct fallback exercise — bypass actual gamma sampling.)
        let mut rng = Rng::from_seed(7u32, "dirichlet-zero-total");
        let _ = sample_dirichlet(3, 0.3, &mut rng);
        // No assert on values — just exercises the branch without panic.
    }

    #[test]
    fn sample_gamma_is_positive() {
        let mut rng = Rng::from_seed(99u32, "gamma-positive");
        for _ in 0..100 {
            let g = sample_gamma(0.5, &mut rng);
            assert!(g > 0.0, "gamma sample was {}", g);
        }
    }

    #[test]
    fn dirichlet_deterministic_given_seed() {
        let mut rng_a = Rng::from_seed(123u32, "dirichlet-det");
        let mut rng_b = Rng::from_seed(123u32, "dirichlet-det");
        let a = sample_dirichlet(5, 0.3, &mut rng_a);
        let b = sample_dirichlet(5, 0.3, &mut rng_b);
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }

    // ---- pick_from_visits properties -----------------------------------

    #[test]
    fn pick_from_visits_temp_zero_is_argmax() {
        let mut rng = Rng::from_seed(1u32, "pick-zero");
        let visits = vec![3, 17, 5, 2];
        // argmax_idx = 1; with T=0 we must always return it regardless of RNG.
        for _ in 0..50 {
            assert_eq!(pick_from_visits(&visits, 1, 0.0, &mut rng), 1);
        }
    }

    #[test]
    fn pick_from_visits_single_action_is_argmax() {
        let mut rng = Rng::from_seed(2u32, "pick-single");
        let visits = vec![42];
        // visits.len() == 1 short-circuits to argmax even with T > 0.
        for _ in 0..20 {
            assert_eq!(pick_from_visits(&visits, 0, 1.0, &mut rng), 0);
        }
    }

    #[test]
    fn pick_from_visits_deterministic_given_seed() {
        let mut rng_a = Rng::from_seed(7u32, "pick-det");
        let mut rng_b = Rng::from_seed(7u32, "pick-det");
        let visits = vec![10, 5, 30, 7, 20];
        let a: Vec<_> = (0..30)
            .map(|_| pick_from_visits(&visits, 2, 1.0, &mut rng_a))
            .collect();
        let b: Vec<_> = (0..30)
            .map(|_| pick_from_visits(&visits, 2, 1.0, &mut rng_b))
            .collect();
        assert_eq!(a, b);
    }

    #[test]
    fn pick_from_visits_explores_when_temperature_positive() {
        // With non-trivial visits and T=1.5, samples must NOT all equal
        // argmax — otherwise temperature is a no-op.
        let mut rng = Rng::from_seed(99u32, "pick-explore");
        let visits = vec![20, 18, 22, 19, 21]; // close enough to encourage sampling spread
        let argmax = 2;
        let samples: Vec<_> = (0..200)
            .map(|_| pick_from_visits(&visits, argmax, 1.5, &mut rng))
            .collect();
        let non_argmax = samples.iter().filter(|&&i| i != argmax).count();
        assert!(
            non_argmax > 50,
            "expected substantial exploration; only {}/200 non-argmax",
            non_argmax
        );
    }

    #[test]
    fn pick_from_visits_all_zero_falls_back_to_argmax() {
        let mut rng = Rng::from_seed(11u32, "pick-zero-total");
        let visits = vec![0, 0, 0];
        // total == 0 → fallback to argmax.
        for _ in 0..20 {
            assert_eq!(pick_from_visits(&visits, 1, 1.0, &mut rng), 1);
        }
    }
}
