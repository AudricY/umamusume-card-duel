"""v3.8 Python↔Rust featurizer parity smoke (state surface).

Loads each `<name>.json` fixture under
`engine-rs/crates/engine/tests/fixtures/v38_parity/`, runs
`observation_to_features_v3_8` on the parsed dict, and compares the
result against the matching `<name>.f32.bin` emitted by the Rust
integration test
(`engine-rs/crates/engine/tests/v38_python_parity_fixtures.rs`).

Parity contract (three-tier — v3.8 tail is the new contract gate):

Gate A — STRICT BIT IDENTITY on the v3.8-new tail [296:304]
  All 8 v3.8 tail bits are pure 0.0/1.0 booleans (per-bench ETA + gust-
  swing catastrophe). ANY mismatch fails loudly. Mirrors v3.7's Gate A
  on [246:296].

Gate B (regression guard) — STRICT BIT IDENTITY on the v3.7 tail
[246:296]
  v3.8 inherits v3.7's 50-bit combat-arith + catalog tail unchanged.
  Any drift here means a v3.7 surface regression.

Gate C (head precision) — 4-ULP TOLERANCE on the v3.0/v3.6 head
[0:246]
  Same f32 ULP semantics as v3.7's Gate B (hash-average chain rounds
  in f64 Python vs f32 Rust at the head).

Regenerate fixtures (after a deliberate Rust featurizer change):
    cargo test -p engine --test v38_python_parity_fixtures -- --ignored emit

Run the smoke:
    python training/v38_python_rust_parity_smoke.py

Exits 0 when Gates A + B pass on every fixture and Gate C reports no
greater-than-tolerance drift. Exits 1 with per-fixture / per-slot
diagnostics on any failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from uma_ai.features import (  # noqa: E402
    STATE_DIM_V3_6,
    STATE_DIM_V3_7,
    STATE_DIM_V3_8,
    observation_to_features_v3_8,
)

_FIXTURE_DIR = (
    _HERE.parent
    / "engine-rs"
    / "crates"
    / "engine"
    / "tests"
    / "fixtures"
    / "v38_parity"
)

# v3.8-new contract slots (frozen — mirror `_V38_*` offsets in features.py).
_V38_NEW_TAIL_BAND = (STATE_DIM_V3_7, STATE_DIM_V3_8)  # [296:304], 8 bits.
# v3.7 tail regression band (v3.7 contract surface — must stay strict).
_V37_REGRESSION_BAND = (STATE_DIM_V3_6, STATE_DIM_V3_7)  # [246:296], 50 bits.

# Head precision tolerance (f32 ULP at magnitude 1 is ~1.19e-7; 4 ULPs
# covers worst-case sum-then-cast drift).
_HEAD_ULP_TOLERANCE = 4
_HEAD_ABS_TOLERANCE = 4 * float(np.spacing(np.float32(1.0)))


def _load_rust_vector(bin_path: Path) -> np.ndarray:
    raw = bin_path.read_bytes()
    if len(raw) != STATE_DIM_V3_8 * 4:
        raise SystemExit(
            f"FAIL: {bin_path.name} byte length {len(raw)} != "
            f"{STATE_DIM_V3_8 * 4} (STATE_DIM_V3_8={STATE_DIM_V3_8} × 4 bytes)"
        )
    return np.frombuffer(raw, dtype="<f4").copy()


def _strict_band(
    name: str, label: str, lo: int, hi: int, rust: np.ndarray, py: np.ndarray
) -> list[str]:
    sub_rust = rust[lo:hi]
    sub_py = py[lo:hi]
    rust_bits = sub_rust.view(np.uint32)
    py_bits = sub_py.view(np.uint32)
    bad = np.where(rust_bits != py_bits)[0]
    if bad.size == 0:
        return []
    lines = [f"FAIL[{label}]: {name} [{lo}:{hi}] diverges at {bad.size} slot(s):"]
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


def _head_drift(
    name: str, rust: np.ndarray, py: np.ndarray
) -> tuple[list[str], dict[int, float]]:
    drifts: dict[int, float] = {}
    over_tol: list[str] = []
    for i in range(0, STATE_DIM_V3_6):
        if rust[i].view(np.uint32) == py[i].view(np.uint32):
            continue
        diff = abs(float(py[i]) - float(rust[i]))
        drifts[i] = diff
        if diff > _HEAD_ABS_TOLERANCE:
            over_tol.append(
                f"FAIL[C]: {name} head slot {i}: rust={float(rust[i])!r} "
                f"python={float(py[i])!r} abs_diff={diff:.3e} > "
                f"tolerance {_HEAD_ABS_TOLERANCE:.3e}"
            )
    return over_tol, drifts


def main() -> int:
    if not _FIXTURE_DIR.exists():
        print(
            f"FAIL: fixture dir {_FIXTURE_DIR} does not exist.\n"
            "      Regenerate with: cargo test -p engine "
            "--test v38_python_parity_fixtures -- --ignored emit",
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
            py_vec = observation_to_features_v3_8(observation)
        except Exception as exc:  # noqa: BLE001
            failures.append(
                f"FAIL: {name}: observation_to_features_v3_8 raised "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        if py_vec.dtype != np.float32:
            failures.append(
                f"FAIL: {name}: python dtype {py_vec.dtype}, expected float32"
            )
            continue
        if py_vec.shape != (STATE_DIM_V3_8,):
            failures.append(
                f"FAIL: {name}: python shape {py_vec.shape}, "
                f"expected ({STATE_DIM_V3_8},)"
            )
            continue

        rust_vec = _load_rust_vector(bin_path)

        # Gate A — strict bit-identity on the v3.8-NEW tail [296:304].
        a_lo, a_hi = _V38_NEW_TAIL_BAND
        gate_a_fail = _strict_band(name, "A", a_lo, a_hi, rust_vec, py_vec)

        # Gate B — strict bit-identity on the inherited v3.7 tail
        # [246:296] (regression guard).
        b_lo, b_hi = _V37_REGRESSION_BAND
        gate_b_fail = _strict_band(name, "B", b_lo, b_hi, rust_vec, py_vec)

        # Gate C — head ULP-bounded report.
        gate_c_fail, head_drift = _head_drift(name, rust_vec, py_vec)
        for slot, diff in head_drift.items():
            aggregate_head_drift.setdefault(slot, []).append(diff)

        if gate_a_fail or gate_b_fail or gate_c_fail:
            failures.extend(gate_a_fail)
            failures.extend(gate_b_fail)
            failures.extend(gate_c_fail)
            print(f"FAIL: {name}", file=sys.stderr)
        else:
            head_note = (
                f" (head drift on {len(head_drift)} slot(s), "
                f"max={max(head_drift.values()):.3e})"
                if head_drift
                else " (head byte-identical)"
            )
            print(
                f"PASS: {name} — v3.8 tail bit-identical, v3.7 regression "
                f"band bit-identical{head_note}"
            )
            pass_count += 1

    summary_lines = [
        "",
        "--- v38_python_rust_parity_smoke summary ---",
        f"  fixtures:   {len(json_paths)}",
        f"  passed:     {pass_count}",
        f"  failed:     {len(failures)}",
        f"  state_dim:  {STATE_DIM_V3_8}",
        f"  Gate A:     STRICT bit-identity on slots "
        f"[{_V38_NEW_TAIL_BAND[0]}:{_V38_NEW_TAIL_BAND[1]}] (v3.8-new contract)",
        f"  Gate B:     STRICT bit-identity on slots "
        f"[{_V37_REGRESSION_BAND[0]}:{_V37_REGRESSION_BAND[1]}] (v3.7 regression guard)",
        f"  Gate C:     head [0:{STATE_DIM_V3_6}] ULP-bounded, "
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
                + (
                    "(within tolerance)"
                    if mx <= _HEAD_ABS_TOLERANCE
                    else "(EXCEEDS tolerance)"
                )
            )
    else:
        summary_lines.append("  head drift: none — head is fully byte-identical")
    print("\n".join(summary_lines))

    if failures:
        print("", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(
        "OVERALL: PASS (v3.8 tail byte-identical; v3.7 regression band "
        "byte-identical; head drift within f32 ULP tolerance)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
