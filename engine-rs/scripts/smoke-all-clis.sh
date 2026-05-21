#!/usr/bin/env bash
# Smoke-test every Rust sim CLI binary in sequence.
# Each invocation is short (a few seconds); together they validate the
# four user-facing entry points still work after any code change.
#
# Usage:
#   bash engine-rs/scripts/smoke-all-clis.sh
#
# Run after edits that touch dispatcher, MCTS driver, flow modules, or
# the serialization layer.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

BIN_DIR="$REPO/engine-rs/target/release"
TMP_DIR="$(mktemp -d -t rust-cli-smoke-XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo ">>> Building release binaries"
cargo build --manifest-path engine-rs/Cargo.toml -p sim-cli --release >/dev/null 2>&1

echo
echo ">>> sim-throughput-probe (micro-bench only)"
"$BIN_DIR/sim-throughput-probe" --micro-samples 2000 --out "$TMP_DIR/probe.json" >/dev/null
python3 -c "
import json
d = json.load(open('$TMP_DIR/probe.json'))
m = d['summary']['micro']
print(f'  clone={m[\"clone_ns_per_call\"]:.0f}ns fingerprint={m[\"fingerprint_ns_per_call\"]:.0f}ns')
assert m['clone_ns_per_call'] < 5000, 'clone regressed'
assert m['fingerprint_ns_per_call'] < 5000, 'fingerprint regressed'
"

echo
echo ">>> sim-export-training (heuristic policy, 3 games)"
"$BIN_DIR/sim-export-training" --games 3 --seed-start 100 --out "$TMP_DIR/training.jsonl" >/dev/null
python3 -c "
import json
n = 0
with open('$TMP_DIR/training.jsonl') as f:
    for line in f:
        d = json.loads(line)
        assert 'episodeId' in d and 'legalActions' in d and 'observation' in d
        assert 'selectedActionIndex' in d
        n += 1
print(f'  {n} training examples emitted')
assert n > 0, 'no training examples produced'
"

echo
echo ">>> sim-mcts-selfplay (full MCTS, 3 games, --record-rows, ORCHESTRATOR FLAG SET)"
"$BIN_DIR/sim-mcts-selfplay" \
  --games 3 --seed-start 0 \
  --mcts-simulations 100 --mcts-c-puct 1.5 \
  --mcts-rollout-crn-samples 3 --mcts-rollout-steps 200 \
  --mcts-collapse-max-steps 64 --mcts-max-nodes 5000 \
  --mcts-prior uniform --mcts-leaf rollout \
  --mcts-dirichlet-alpha 0.3 --mcts-dirichlet-epsilon 0.25 \
  --temperature-moves 6 --temperature-value 1.0 \
  --workers 1 \
  --record-rows --out "$TMP_DIR/selfplay.jsonl" >/dev/null
python3 engine-rs/scripts/check-rust-jsonl-pyparse.py "$TMP_DIR/selfplay.jsonl"

echo
echo ">>> sim-eval-gate (MCTS-vs-heuristic, 4 seeds, ORCHESTRATOR FLAG SET)"
"$BIN_DIR/sim-eval-gate" \
  --selection mcts \
  --games 4 --seed-start 0 \
  --model-side both \
  --max-steps 500 \
  --mcts-simulations 100 --mcts-c-puct 1.5 \
  --mcts-rollout-crn-samples 3 --mcts-rollout-steps 200 \
  --mcts-collapse-max-steps 64 --mcts-max-nodes 5000 \
  --mcts-prior uniform --mcts-leaf rollout \
  --min-ci-lower 0.0 --min-games 4 \
  --progress-out "$TMP_DIR/gate-progress.jsonl" \
  --workers 1 \
  >"$TMP_DIR/gate.json"
python3 -c "
import json
d = json.load(open('$TMP_DIR/gate.json'))
ov = d['overall']
print(f'  overall: {ov[\"wins\"]}/{ov[\"games\"]} = {ov[\"winRate\"]:.0%} '
      f'(Wilson 95% CI {ov[\"wilsonLower\"]:.0%}-{ov[\"wilsonUpper\"]:.0%})')
assert d['terminalGameOver'] == d['config']['seeds'], 'some games did not reach game_over'
"

echo
echo "✅ All 4 sim CLIs healthy."
