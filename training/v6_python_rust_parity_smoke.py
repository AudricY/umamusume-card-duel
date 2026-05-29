"""v6 Python↔Rust featurizer parity smoke (own-deck-composition).

Validates two fixture families emitted by the Rust integration test
(`engine-rs/crates/engine/tests/v6_python_parity_fixtures.rs`) under
`engine-rs/crates/engine/tests/fixtures/v6_parity/`:

  1. OBSERVATION fixtures (`<name>.json` + `<name>.f32.bin`):
       Gate A (tail contract) — STRICT BIT IDENTITY on the v6 tail [110:126].
         Under the current obs contract this band is all-zeros in BOTH
         languages (the PublicObservation exposes only own.deck_count, not the
         deck card-id list). Any non-zero / mismatch fails loudly.
       Gate B (head precision) — 4-ULP TOLERANCE on the FROZEN v3.0 head
         [0:110] (hash-average chain rounds in f64 Python vs f32 Rust).

  2. DECK-COMPOSITION helper fixtures (`deck_ids/<name>.deck.json` +
     `deck_ids/<name>.tail.f32.bin`):
       Gate C (live bucketing) — STRICT BIT IDENTITY on the 16-slot
         `_v6_deck_composition_tail(deck_ids)` output vs the Rust
         `v6_deck_composition_tail`. This proves the kind + uma-type bucketing
         math is bit-exact, so the moment the obs contract exposes own deck
         card-ids BOTH builders light up identically.

Regenerate fixtures (after a deliberate Rust featurizer change):
    cargo test -p engine --test v6_python_parity_fixtures -- --ignored emit

Run the smoke:
    python training/v6_python_rust_parity_smoke.py

Exits 0 when Gates A + C are bit-identical on every fixture and Gate B reports
no greater-than-tolerance head drift. Exits 1 with per-slot diagnostics
otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from uma_ai.features import (  # noqa: E402
    STATE_DIM_V3,
    STATE_DIM_V6,
    _v6_deck_composition_tail,
    observation_to_features_v6,
)

_FIXTURE_DIR = (
    _HERE.parent / "engine-rs" / "crates" / "engine" / "tests" / "fixtures" / "v6_parity"
)
_DECK_DIR = _FIXTURE_DIR / "deck_ids"

_TAIL_BAND = (STATE_DIM_V3, STATE_DIM_V6)  # [110:126], 16 slots.
_HEAD_ABS_TOLERANCE = 4 * float(np.spacing(np.float32(1.0)))


def _load_f32(bin_path: Path, expected_len: int, label: str) -> np.ndarray:
    raw = bin_path.read_bytes()
    if len(raw) != expected_len * 4:
        raise SystemExit(
            f"FAIL: {bin_path.name} byte length {len(raw)} != {expected_len * 4} ({label})"
        )
    return np.frombuffer(raw, dtype="<f4").copy()


def _strict_band(
    name: str, label: str, lo: int, hi: int, rust: np.ndarray, py: np.ndarray
) -> list[str]:
    sub_rust = rust[lo:hi]
    sub_py = py[lo:hi]
    bad = np.where(sub_rust.view(np.uint32) != sub_py.view(np.uint32))[0]
    if bad.size == 0:
        return []
    lines = [f"FAIL[{label}]: {name} [{lo}:{hi}] diverges at {bad.size} slot(s):"]
    for offset in bad[:16]:
        abs_idx = lo + int(offset)
        lines.append(
            f"  slot {abs_idx}: rust={sub_rust[offset]!r} vs python={sub_py[offset]!r}"
        )
    return lines


def _head_drift(name: str, rust: np.ndarray, py: np.ndarray) -> list[str]:
    over_tol: list[str] = []
    for i in range(0, STATE_DIM_V3):
        if rust[i].view(np.uint32) == py[i].view(np.uint32):
            continue
        diff = abs(float(py[i]) - float(rust[i]))
        if diff > _HEAD_ABS_TOLERANCE:
            over_tol.append(
                f"FAIL[B]: {name} head slot {i}: rust={float(rust[i])!r} "
                f"python={float(py[i])!r} abs_diff={diff:.3e} > {_HEAD_ABS_TOLERANCE:.3e}"
            )
    return over_tol


def _check_obs_fixtures() -> tuple[int, list[str]]:
    failures: list[str] = []
    passed = 0
    json_paths = sorted(_FIXTURE_DIR.glob("*.json"))
    if not json_paths:
        return 0, [f"FAIL: no *.json obs fixtures in {_FIXTURE_DIR}"]
    lo, hi = _TAIL_BAND
    for json_path in json_paths:
        name = json_path.stem
        bin_path = json_path.with_suffix(".f32.bin")
        if not bin_path.exists():
            failures.append(f"FAIL: {name}: missing {bin_path.name}")
            continue
        observation = json.loads(json_path.read_text(encoding="utf-8"))
        py_vec = observation_to_features_v6(observation)
        if py_vec.shape != (STATE_DIM_V6,):
            failures.append(f"FAIL: {name}: python shape {py_vec.shape}")
            continue
        rust_vec = _load_f32(bin_path, STATE_DIM_V6, "obs")
        gate_a = _strict_band(name, "A", lo, hi, rust_vec, py_vec)
        gate_b = _head_drift(name, rust_vec, py_vec)
        if gate_a or gate_b:
            failures.extend(gate_a)
            failures.extend(gate_b)
            print(f"FAIL(obs): {name}", file=sys.stderr)
        else:
            print(f"PASS(obs): {name} — v6 tail bit-identical (zeros), head within 4 ULP")
            passed += 1
    return passed, failures


def _check_deck_fixtures() -> tuple[int, list[str]]:
    failures: list[str] = []
    passed = 0
    deck_paths = sorted(_DECK_DIR.glob("*.deck.json"))
    if not deck_paths:
        return 0, [f"FAIL: no *.deck.json fixtures in {_DECK_DIR}"]
    tail_len = STATE_DIM_V6 - STATE_DIM_V3
    for deck_path in deck_paths:
        name = deck_path.name[: -len(".deck.json")]
        bin_path = _DECK_DIR / f"{name}.tail.f32.bin"
        if not bin_path.exists():
            failures.append(f"FAIL: {name}: missing {bin_path.name}")
            continue
        deck = json.loads(deck_path.read_text(encoding="utf-8"))
        py_tail = _v6_deck_composition_tail(list(deck))
        rust_tail = _load_f32(bin_path, tail_len, "deck-tail")
        gate_c = _strict_band(name, "C", 0, tail_len, rust_tail, py_tail)
        if gate_c:
            failures.extend(gate_c)
            print(f"FAIL(deck): {name}", file=sys.stderr)
        else:
            print(f"PASS(deck): {name} — 16-slot composition tail bit-identical")
            passed += 1
    return passed, failures


def main() -> int:
    if not _FIXTURE_DIR.exists():
        print(
            f"FAIL: fixture dir {_FIXTURE_DIR} does not exist.\n"
            "      Regenerate: cargo test -p engine --test v6_python_parity_fixtures "
            "-- --ignored emit",
            file=sys.stderr,
        )
        return 1

    obs_passed, obs_failures = _check_obs_fixtures()
    deck_passed, deck_failures = _check_deck_fixtures()
    failures = obs_failures + deck_failures

    print("")
    print("--- v6_python_rust_parity_smoke summary ---")
    print(f"  obs fixtures passed:  {obs_passed}")
    print(f"  deck fixtures passed: {deck_passed}")
    print(f"  failures:             {len(failures)}")
    if failures:
        print("")
        for line in failures:
            print(line, file=sys.stderr)
        return 1
    print("  ALL GATES PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
