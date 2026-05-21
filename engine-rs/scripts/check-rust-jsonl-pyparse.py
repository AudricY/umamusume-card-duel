#!/usr/bin/env python3
"""Verify Rust sim CLI JSONL output is consumable by the Python pipeline.

Mirrors the field access pattern in training/uma_ai/dataset.py and
value_target_dataset.py — both read camelCase keys like `legalActions`,
`visitDistribution`, `rootValue`, `selectedActionIndex`.

Usage:
  python3 engine-rs/scripts/check-rust-jsonl-pyparse.py <jsonl-path>

Run after sim-mcts-selfplay or sim-export-training to confirm Python
trainers can consume the output without an adapter layer.
"""

import json
import sys
from collections import Counter


def check_selfplay_rows(path: str) -> int:
    total_rows = 0
    kinds = Counter()
    legal_lens = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            d = json.loads(line)
            # Top-level GameRecord schema (mcts_selfplay):
            assert "seed" in d, f"line {lineno}: seed missing"
            assert "terminalReason" in d, f"line {lineno}: terminalReason missing (must be camelCase)"
            assert "turnNumber" in d, f"line {lineno}: turnNumber missing"
            assert "totalSteps" in d, f"line {lineno}: totalSteps missing"
            for r in d.get("rows", []):
                # Per-row SelfPlayRow schema (matches TS):
                actions = r.get("legalActions", [])
                visits = r.get("visitDistribution", [])
                rv = r.get("rootValue")
                sel = r.get("selectedActionIndex")
                schema_v = r.get("schemaVersion")
                kind = r.get("kind", "?")
                if not actions:
                    raise AssertionError(f"line {lineno}: legalActions empty")
                if len(visits) != len(actions):
                    raise AssertionError(
                        f"line {lineno}: visitDistribution len {len(visits)} != legalActions len {len(actions)}"
                    )
                if rv is None:
                    raise AssertionError(f"line {lineno}: rootValue missing")
                if sel is None:
                    raise AssertionError(f"line {lineno}: selectedActionIndex missing")
                if schema_v != 1:
                    raise AssertionError(f"line {lineno}: unexpected schemaVersion {schema_v}")
                obs = r.get("observation")
                if not isinstance(obs, dict):
                    raise AssertionError(f"line {lineno}: observation missing or wrong type")
                # Sanity-check a few camelCase nested keys.
                for required_obs_key in ("schemaVersion", "sideToAct", "turnNumber", "own", "opponent"):
                    if required_obs_key not in obs:
                        raise AssertionError(
                            f"line {lineno}: observation.{required_obs_key} missing"
                        )
                total_rows += 1
                kinds[kind] += 1
                legal_lens.append(len(actions))
    avg_legal = sum(legal_lens) / len(legal_lens) if legal_lens else 0.0
    print(
        f"OK parsed {total_rows} rows via Python-style dict access. "
        f"kinds={dict(kinds)}, avg legal_actions/row={avg_legal:.2f}"
    )
    return total_rows


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check-rust-jsonl-pyparse.py <jsonl-path>", file=sys.stderr)
        return 1
    n = check_selfplay_rows(sys.argv[1])
    return 0 if n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
