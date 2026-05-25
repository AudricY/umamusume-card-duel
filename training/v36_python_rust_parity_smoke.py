"""v3.6 Python↔Rust featurizer parity smoke.

Loads each `<name>.json` fixture under
`engine-rs/crates/engine/tests/fixtures/v36_parity/`, runs
`observation_to_features_v3_6` on the parsed dict, and compares the
result against the matching `<name>.f32.bin` emitted by the Rust
integration test
(`engine-rs/crates/engine/tests/v36_python_parity_fixtures.rs`).

Parity contract (two-tier — the v3.6 ADDITIVE contract is the gate)
--------------------------------------------------------------------
Gate A — STRICT BIT IDENTITY on the v3.6-new slots
  Slots [197:207]  own.energy_pool typed multihot (REPURPOSED in-place)
  Slots [212:246]  opp pool + prize-onehots + opp bench typed + lethal
                   + secondary-attack bits (the appended v3.6 tail)
  These slots are pure 0.0/1.0 booleans (no f32 arithmetic involved).
  ANY mismatch fails the smoke loudly.

Gate B — 4-ULP TOLERANCE on the inherited v3.5 head [0:197] + [207:212]
  The v3.0 builder pre-existing slot 42 = `hash_average(handCardIds)`
  sums in Python f64 then downcasts to f32, while Rust sums entirely in
  f32. This is a 1-ULP artifact of the original v3.0 builder
  (max_abs ≈ 3e-8) and was never part of the v3.6 contract. We
  TOLERATE up to 4 ULPs (~5e-7 at unit scale) on the head and REPORT
  per-slot drift so a real divergence (>4 ULP) still fails.

Regenerate fixtures (after a deliberate Rust featurizer change):
    cargo test -p engine --test v36_python_parity_fixtures -- --ignored emit

Run the smoke:
    python training/v36_python_rust_parity_smoke.py

Exits 0 when Gate A passes on every fixture and Gate B reports no
greater-than-tolerance drift. Exits 1 with per-fixture / per-slot
diagnostics on any failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Make `uma_ai` importable when invoked from the repo root.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from uma_ai.features import (  # noqa: E402
    STATE_DIM_V3_5,
    STATE_DIM_V3_6,
    observation_to_features_v3_6,
)

_FIXTURE_DIR = (
    _HERE.parent
    / "engine-rs"
    / "crates"
    / "engine"
    / "tests"
    / "fixtures"
    / "v36_parity"
)

# v3.6-new contract slots (frozen — mirror `_V36_*` offsets in features.py).
_V36_OWN_POOL_BAND = (197, 207)      # repurposed in-band, 10 bits
_V36_NEW_TAIL_BAND = (212, STATE_DIM_V3_6)  # appended tail, 34 bits

# Head precision tolerance. f32 ULP at magnitude 1 is ~1.19e-7; 4 ULPs
# covers the worst-case sum-then-cast drift in hash-average chains.
_HEAD_ULP_TOLERANCE = 4
_HEAD_ABS_TOLERANCE = 4 * float(np.spacing(np.float32(1.0)))  # ≈ 4.77e-7


def _load_rust_vector(bin_path: Path) -> np.ndarray:
    raw = bin_path.read_bytes()
    if len(raw) != STATE_DIM_V3_6 * 4:
        raise SystemExit(
            f"FAIL: {bin_path.name} byte length {len(raw)} != "
            f"{STATE_DIM_V3_6 * 4} (STATE_DIM_V3_6={STATE_DIM_V3_6} × 4 bytes)"
        )
    return np.frombuffer(raw, dtype="<f4").copy()


def _strict_band(name: str, lo: int, hi: int, rust: np.ndarray, py: np.ndarray
                 ) -> list[str]:
    """Bit-identity check over a closed-open slot range. Returns a list
    of human-readable failure lines (empty on PASS)."""
    sub_rust = rust[lo:hi]
    sub_py = py[lo:hi]
    rust_bits = sub_rust.view(np.uint32)
    py_bits = sub_py.view(np.uint32)
    bad = np.where(rust_bits != py_bits)[0]
    if bad.size == 0:
        return []
    lines = [
        f"FAIL[A]: {name} v3.6-new band [{lo}:{hi}] diverges at "
        f"{bad.size} slot(s):"
    ]
    for offset in bad[:16]:
        abs_idx = lo + int(offset)
        lines.append(
            f"  slot {abs_idx}: rust={sub_rust[offset]!r} "
            f"(bits=0x{rust_bits[offset]:08x}) vs "
            f"python={sub_py[offset]!r} (bits=0x{py_bits[offset]:08x})"
        )
    if bad.size > 16:
        lines.append(f"  ... and {bad.size - 16} more")
    return lines


def _head_drift(name: str, rust: np.ndarray, py: np.ndarray
                ) -> tuple[list[str], dict[int, float]]:
    """Report per-slot drift on the inherited head slots. Returns
    (failure lines for slots > tolerance, all-drift dict). The two head
    bands are [0:197] and [207:212] (the v3.5 tail preserved verbatim
    in v3.6)."""
    head_indices = list(range(0, 197)) + list(range(207, 212))
    drifts: dict[int, float] = {}
    over_tol_fail: list[str] = []
    for i in head_indices:
        if rust[i].view(np.uint32) == py[i].view(np.uint32):
            continue
        diff = abs(float(py[i]) - float(rust[i]))
        drifts[i] = diff
        if diff > _HEAD_ABS_TOLERANCE:
            over_tol_fail.append(
                f"FAIL[B]: {name} head slot {i}: rust={float(rust[i])!r} "
                f"python={float(py[i])!r} abs_diff={diff:.3e} > "
                f"tolerance {_HEAD_ABS_TOLERANCE:.3e}"
            )
    return over_tol_fail, drifts


def main() -> int:
    if not _FIXTURE_DIR.exists():
        print(
            f"FAIL: fixture dir {_FIXTURE_DIR} does not exist.\n"
            "      Regenerate with: cargo test -p engine "
            "--test v36_python_parity_fixtures -- --ignored emit",
            file=sys.stderr,
        )
        return 1

    json_paths = sorted(_FIXTURE_DIR.glob("*.json"))
    if not json_paths:
        print(f"FAIL: no *.json fixtures in {_FIXTURE_DIR}", file=sys.stderr)
        return 1

    failures: list[str] = []
    pass_count = 0
    aggregate_head_drift: dict[int, list[float]] = {}

    for json_path in json_paths:
        name = json_path.stem
        bin_path = json_path.with_suffix(".f32.bin")
        if not bin_path.exists():
            failures.append(f"FAIL: {name}: missing {bin_path.name}")
            continue

        try:
            observation = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            failures.append(f"FAIL: {name}: JSON decode error: {exc}")
            continue

        try:
            py_vec = observation_to_features_v3_6(observation)
        except Exception as exc:  # noqa: BLE001 — surface any exception
            failures.append(
                f"FAIL: {name}: observation_to_features_v3_6 raised "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        if py_vec.dtype != np.float32:
            failures.append(
                f"FAIL: {name}: python dtype {py_vec.dtype}, expected float32"
            )
            continue
        if py_vec.shape != (STATE_DIM_V3_6,):
            failures.append(
                f"FAIL: {name}: python shape {py_vec.shape}, "
                f"expected ({STATE_DIM_V3_6},)"
            )
            continue

        rust_vec = _load_rust_vector(bin_path)

        # Gate A — strict bit-identity on the v3.6-NEW bands.
        a_lo, a_hi = _V36_OWN_POOL_BAND
        gate_a_fail = _strict_band(name, a_lo, a_hi, rust_vec, py_vec)
        t_lo, t_hi = _V36_NEW_TAIL_BAND
        gate_a_fail += _strict_band(name, t_lo, t_hi, rust_vec, py_vec)

        # Gate B — head ULP-bounded report.
        gate_b_fail, head_drift = _head_drift(name, rust_vec, py_vec)
        for slot, diff in head_drift.items():
            aggregate_head_drift.setdefault(slot, []).append(diff)

        if gate_a_fail or gate_b_fail:
            failures.extend(gate_a_fail)
            failures.extend(gate_b_fail)
            print(f"FAIL: {name}", file=sys.stderr)
        else:
            head_note = (
                f" (head drift on {len(head_drift)} slot(s), "
                f"max={max(head_drift.values()):.3e})"
                if head_drift else " (head byte-identical)"
            )
            print(f"PASS: {name} — v3.6 bands bit-identical{head_note}")
            pass_count += 1

    summary_lines = [
        "",
        "--- v36_python_rust_parity_smoke summary ---",
        f"  fixtures:   {len(json_paths)}",
        f"  passed:     {pass_count}",
        f"  failed:     {len(failures)}",
        f"  state_dim:  {STATE_DIM_V3_6}",
        f"  Gate A:     STRICT bit-identity on slots "
        f"[{_V36_OWN_POOL_BAND[0]}:{_V36_OWN_POOL_BAND[1]}] + "
        f"[{_V36_NEW_TAIL_BAND[0]}:{_V36_NEW_TAIL_BAND[1]}] (v3.6-new contract)",
        f"  Gate B:     head [0:197]+[207:212] ULP-bounded, "
        f"tol={_HEAD_ABS_TOLERANCE:.3e} (~4 ULP@1)",
    ]
    if aggregate_head_drift:
        summary_lines.append(
            f"  head drift: {len(aggregate_head_drift)} unique slot(s) — "
            f"{sorted(aggregate_head_drift.keys())}"
        )
        for slot, diffs in sorted(aggregate_head_drift.items()):
            mx = max(diffs)
            summary_lines.append(
                f"    slot {slot}: count={len(diffs)} max_abs={mx:.3e} "
                + ("(within tolerance)" if mx <= _HEAD_ABS_TOLERANCE
                   else "(EXCEEDS tolerance)")
            )
    else:
        summary_lines.append("  head drift: none — head is fully byte-identical")
    print("\n".join(summary_lines))

    if failures:
        print("", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("OVERALL: PASS (v3.6 contract bands byte-identical; "
          "head drift within f32 ULP tolerance)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
