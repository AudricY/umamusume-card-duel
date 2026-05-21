//! Bit-identical port of `frontend/src/game/engine/core/random.ts`.
//!
//! TS implementation (mulberry32 variant + FNV-1a seed hash):
//!
//! ```text
//! state += 0x6d2b79f5;
//! let value = state;
//! value = Math.imul(value ^ (value >>> 15), value | 1);
//! value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
//! return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
//! ```
//!
//! `Math.imul` returns the lower 32 bits of an `i32×i32` multiply, with
//! wraparound, treating inputs as signed 32-bit. In Rust we use
//! `u32::wrapping_mul` — algebraically equivalent on the low 32 bits.
//!
//! Output is f64 in [0,1) by dividing the unsigned 32-bit result by 2^32,
//! matching `(value >>> 0) / 4294967296` in TS.

use std::cell::RefCell;

#[derive(Debug, Clone)]
pub struct Rng {
    state: u32,
    initial: u32,
    label: String,
    /// Monotonically increasing draw count. Bit-identity-only test
    /// instrumentation; not load-bearing for engine semantics. Wraps
    /// at u64::MAX (unreachable in practice).
    draws: u64,
}

impl Rng {
    /// Mirror of `createSeededRng(seed, label = "root")`.
    pub fn from_seed<S: Into<SeedSource>>(seed: S, label: &str) -> Self {
        let initial = normalize_seed(seed.into());
        // TS: `state = initial || 0x6d2b79f5`. In JS `||` falls back when the
        // value is the falsy zero. Replicate exactly.
        let state = if initial == 0 { 0x6d2b79f5 } else { initial };
        Rng {
            state,
            initial,
            label: label.to_string(),
            draws: 0,
        }
    }

    /// Mirror of `rng.next()` — returns f64 in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x6d2b79f5);
        let mut value: u32 = self.state;
        // value = Math.imul(value ^ (value >>> 15), value | 1)
        value = (value ^ (value >> 15)).wrapping_mul(value | 1);
        // value ^= value + Math.imul(value ^ (value >>> 7), value | 61)
        let mix = (value ^ (value >> 7)).wrapping_mul(value | 61);
        value ^= value.wrapping_add(mix);
        // ((value ^ (value >>> 14)) >>> 0) / 4294967296
        let final_u32 = value ^ (value >> 14);
        self.draws = self.draws.saturating_add(1);
        (final_u32 as f64) / 4_294_967_296.0
    }

    /// Total `next_f64` calls since construction.
    pub fn draws(&self) -> u64 {
        self.draws
    }

    /// Mirror of `rng.fork(forkLabel)`.
    pub fn fork(&self, fork_label: &str) -> Rng {
        // TS: `createSeededRng(\`${initial}:${label}:${forkLabel}\`, \`${label}/${forkLabel}\`)`
        let seed_string = format!("{}:{}:{}", self.initial, self.label, fork_label);
        let new_label = format!("{}/{}", self.label, fork_label);
        Rng::from_seed(SeedSource::Str(seed_string), &new_label)
    }

    pub fn label(&self) -> &str {
        &self.label
    }

}

/// Mirror of TS `normalizeSeed`.
///
/// ```text
/// if (typeof seed === "number" && Number.isFinite(seed)) return seed >>> 0;
/// const text = String(seed);
/// let hash = 2166136261;
/// for (let i = 0; i < text.length; i++) {
///   hash ^= text.charCodeAt(i);
///   hash = Math.imul(hash, 16777619);
/// }
/// return hash >>> 0;
/// ```
///
/// `charCodeAt` returns a UTF-16 code unit (0..65535). To preserve
/// bit-identity we iterate UTF-16 code units, not Unicode scalar values.
pub fn normalize_seed(seed: SeedSource) -> u32 {
    match seed {
        SeedSource::U32(v) => v,
        SeedSource::I32(v) => v as u32,
        SeedSource::F64(v) => {
            if v.is_finite() {
                // TS `n >>> 0` casts via ToInt32 then to uint32. For finite f64
                // values, this is `(n as i32) as u32` after truncating toward
                // zero modulo 2^32. We replicate via the `as i32` chain Rust
                // uses for finite f64.
                (v as i64 as i32) as u32
            } else {
                // Non-finite f64 falls through to `String(seed)` in TS.
                let text = format!("{}", v);
                fnv1a_utf16(&text)
            }
        }
        SeedSource::Str(s) => fnv1a_utf16(&s),
    }
}

fn fnv1a_utf16(text: &str) -> u32 {
    let mut hash: u32 = 2_166_136_261;
    for unit in text.encode_utf16() {
        hash ^= unit as u32;
        hash = hash.wrapping_mul(16_777_619);
    }
    hash
}

#[derive(Debug, Clone)]
pub enum SeedSource {
    U32(u32),
    I32(i32),
    F64(f64),
    Str(String),
}

impl From<u32> for SeedSource {
    fn from(v: u32) -> Self {
        SeedSource::U32(v)
    }
}
impl From<i32> for SeedSource {
    fn from(v: i32) -> Self {
        SeedSource::I32(v)
    }
}
impl From<f64> for SeedSource {
    fn from(v: f64) -> Self {
        SeedSource::F64(v)
    }
}
impl From<&str> for SeedSource {
    fn from(v: &str) -> Self {
        SeedSource::Str(v.to_string())
    }
}
impl From<String> for SeedSource {
    fn from(v: String) -> Self {
        SeedSource::Str(v)
    }
}

// ---------------------------------------------------------------------------
// Thread-local storage provider (mirror of `installRngStorageProvider` +
// `withRng` / `randomFloat` / `randomInt` / `shuffle`).
// ---------------------------------------------------------------------------

thread_local! {
    static ACTIVE_RNG: RefCell<Option<Rng>> = const { RefCell::new(None) };
}

pub fn with_rng<T>(rng: Rng, f: impl FnOnce() -> T) -> (T, Rng) {
    ACTIVE_RNG.with(|cell| {
        let previous = cell.replace(Some(rng));
        let result = f();
        let used = cell.replace(previous);
        (result, used.expect("active rng disappeared mid-scope"))
    })
}

pub fn random_float() -> f64 {
    ACTIVE_RNG.with(|cell| {
        let mut borrow = cell.borrow_mut();
        match borrow.as_mut() {
            Some(rng) => rng.next_f64(),
            // TS falls back to Math.random() when no active RNG; in Rust we
            // panic instead — every sim path installs an RNG explicitly, and
            // a silent Math.random() would defeat the determinism contract.
            None => panic!("random_float() called outside with_rng scope"),
        }
    })
}

pub fn random_int(max_exclusive: u32) -> u32 {
    if max_exclusive == 0 {
        return 0;
    }
    let f = random_float();
    // TS: Math.floor(randomFloat() * maxExclusive).
    //
    // `f` is in [0, 1), `max_exclusive` is u32. The product is f64; flooring
    // gives an integer in [0, max_exclusive). We cast via i64 then u32 to
    // mirror the bit pattern JS produces (`Math.floor` then implicit int).
    (f * max_exclusive as f64).floor() as u32
}

/// Mirror of TS `shuffle<T>(items)`. Fisher–Yates top-down using
/// `randomInt(index + 1)` for the swap index.
pub fn shuffle<T: Clone>(items: &[T]) -> Vec<T> {
    let mut copy: Vec<T> = items.to_vec();
    if copy.len() < 2 {
        return copy;
    }
    let mut index = copy.len() - 1;
    while index > 0 {
        let swap_index = random_int((index + 1) as u32) as usize;
        copy.swap(index, swap_index);
        index -= 1;
    }
    copy
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference outputs captured from the TS impl via
    /// `engine-rs/scripts/dump-ts-rng.ts`. Seed 12345, first 16 outputs.
    /// (Filled in by the cross-language test runner; see
    /// `engine-rs/tests/rng_cross_lang.rs`.)
    #[test]
    fn rng_advances_deterministically() {
        let mut a = Rng::from_seed(12345u32, "root");
        let mut b = Rng::from_seed(12345u32, "root");
        for _ in 0..1000 {
            assert_eq!(a.next_f64().to_bits(), b.next_f64().to_bits());
        }
    }

    #[test]
    fn shuffle_in_with_rng_scope_is_deterministic() {
        let rng = Rng::from_seed(99u32, "root");
        let items: Vec<u32> = (0..20).collect();
        let (out, _) = with_rng(rng.clone(), || shuffle(&items));
        let (out2, _) = with_rng(rng, || shuffle(&items));
        assert_eq!(out, out2);
    }
}
