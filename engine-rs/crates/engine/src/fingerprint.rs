//! Structural fingerprinting over the packed `GameState` buffer.
//!
//! Replaces `backend/src/sim/stateFingerprint.ts`, which JSON-stringifies a
//! compact view of the state (≈8 μs/call per the throughput probe). The
//! Rust port hashes the packed buffer directly with xxhash3 (sub-microsecond).
//!
//! Conformance note: the TS fingerprint is JSON-shape-stable but not
//! algorithm-stable across languages — the Rust hash will not match TS
//! byte-for-byte. Bit-identity of *engine state* still holds; the
//! fingerprint is reduced to a structural identity check. The golden-trace
//! harness records both TS-JSON fingerprints and Rust xxhash3 digests so
//! both can be diffed independently.

use xxhash_rust::xxh3::xxh3_128;

/// Hash an opaque packed state buffer to a 128-bit digest.
///
/// Stable across runs and architectures (xxhash3 is endian-stable by
/// construction). Not stable across hash-library upgrades — pin
/// `xxhash-rust = 0.8` in `Cargo.toml`.
pub fn fingerprint(packed: &[u8]) -> u128 {
    xxh3_128(packed)
}
