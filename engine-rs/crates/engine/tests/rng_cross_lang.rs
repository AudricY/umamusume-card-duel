//! Cross-language RNG bit-identity test.
//!
//! Loads reference vectors dumped from the TS engine by
//! `engine-rs/scripts/dump-ts-rng.ts` and asserts the Rust port produces
//! identical bit patterns.
//!
//! Refresh the reference data with:
//!
//! ```bash
//! npx tsx engine-rs/scripts/dump-ts-rng.ts > engine-rs/crates/engine/tests/rng-reference.json
//! ```

use std::fs;
use std::path::Path;

use engine::core::random::{normalize_seed, Rng, SeedSource};
use serde::Deserialize;

const REFERENCE_PATH: &str = "tests/rng-reference.json";

#[derive(Deserialize)]
struct Reference {
    mulberry32: Vec<Mulberry32Vec>,
    fnv1a: Vec<Fnv1aVec>,
    fork: Vec<ForkVec>,
}

#[derive(Deserialize)]
struct Mulberry32Vec {
    seed: u32,
    label: String,
    /// u32 pre-divide stream (see scripts/dump-ts-rng.ts). We compare in
    /// u32-space to avoid serde_json's decimal-to-f64 disagreeing with JS's
    /// shortest-decimal serialization at half-ULP boundaries. The f64
    /// division by 2^32 is exact in IEEE 754, so u32-identity implies
    /// f64-identity.
    first1000: Vec<u32>,
}

#[derive(Deserialize)]
struct Fnv1aVec {
    input: String,
    hash: u32,
}

#[derive(Deserialize)]
struct ForkVec {
    seed: u32,
    label: String,
    #[serde(rename = "forkLabel")]
    fork_label: String,
    first16: Vec<u32>,
}

/// Convert a u32 to the [0,1) f64 produced by the engine's `next()`.
fn u32_to_f64(value: u32) -> f64 {
    (value as f64) / 4_294_967_296.0
}

/// Inverse of `u32_to_f64`: given a `next()` return, recover the u32. Used
/// only on Rust side. Since `u32_to_f64` is exact, multiplying back and
/// rounding gives the original u32.
fn f64_to_u32(value: f64) -> u32 {
    (value * 4_294_967_296.0).round() as u32
}

fn load_reference() -> Option<Reference> {
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    let path = here.join(REFERENCE_PATH);
    if !path.exists() {
        eprintln!(
            "skipping cross-language RNG test: {} not found. Regenerate with `npx tsx engine-rs/scripts/dump-ts-rng.ts > {}`",
            path.display(),
            path.display()
        );
        return None;
    }
    let raw = fs::read_to_string(&path).expect("read reference json");
    let parsed: Reference = serde_json::from_str(&raw).expect("parse reference json");
    Some(parsed)
}

#[test]
fn mulberry32_matches_ts() {
    let Some(reference) = load_reference() else {
        return;
    };
    for vec in &reference.mulberry32 {
        let mut rng = Rng::from_seed(vec.seed, &vec.label);
        for (i, &expected_u32) in vec.first1000.iter().enumerate() {
            let actual = rng.next_f64();
            let actual_u32 = f64_to_u32(actual);
            assert_eq!(
                actual_u32, expected_u32,
                "seed={} label={} index={}: rust=0x{:08x} ts=0x{:08x}",
                vec.seed, vec.label, i, actual_u32, expected_u32,
            );
            // Also assert the reconstructed f64 matches what Rust returned.
            assert_eq!(
                actual.to_bits(),
                u32_to_f64(expected_u32).to_bits(),
                "seed={} index={}: f64-from-u32 disagrees",
                vec.seed,
                i,
            );
        }
    }
}

#[test]
fn fnv1a_matches_ts() {
    let Some(reference) = load_reference() else {
        return;
    };
    for vec in &reference.fnv1a {
        let actual = normalize_seed(SeedSource::Str(vec.input.clone()));
        assert_eq!(
            actual, vec.hash,
            "input={:?}: rust=0x{:08x} ts=0x{:08x}",
            vec.input, actual, vec.hash
        );
    }
}

#[test]
fn fork_matches_ts() {
    let Some(reference) = load_reference() else {
        return;
    };
    for vec in &reference.fork {
        let root = Rng::from_seed(vec.seed, &vec.label);
        let mut child = root.fork(&vec.fork_label);
        for (i, &expected_u32) in vec.first16.iter().enumerate() {
            let actual_u32 = f64_to_u32(child.next_f64());
            assert_eq!(
                actual_u32, expected_u32,
                "fork seed={} label={} forkLabel={} index={}: rust=0x{:08x} ts=0x{:08x}",
                vec.seed, vec.label, vec.fork_label, i, actual_u32, expected_u32,
            );
        }
    }
}
