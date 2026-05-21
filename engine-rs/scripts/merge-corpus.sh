#!/usr/bin/env bash
# Merge the 16 corpus chunks produced by the parallel recorder launch
# into a single 500-seed JSONL, then move it to the canonical path.
#
# Usage:
#   bash engine-rs/scripts/merge-corpus.sh
#
# After completion the file lives at
# runs/rust-port-golden-traces/traces-500.jsonl (from repo root). The
# chunks themselves were written to backend/runs/... because the npm
# script's cwd is the backend workspace; this script repositions them.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CHUNK_DIR="$REPO_ROOT/backend/runs/rust-port-golden-traces/chunks"
OUT_DIR="$REPO_ROOT/runs/rust-port-golden-traces"
OUT_FILE="$OUT_DIR/traces-500.jsonl"

if [ ! -d "$CHUNK_DIR" ]; then
  echo "no chunk dir at $CHUNK_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

# Verify all chunks are present and contain the expected line counts.
expected_total=500
total=0
for i in $(seq 0 15); do
  f="$CHUNK_DIR/chunk-$(printf %02d "$i").jsonl"
  if [ ! -f "$f" ]; then
    echo "missing chunk $f" >&2
    exit 1
  fi
  n=$(wc -l < "$f")
  total=$((total + n))
done

if [ "$total" -ne "$expected_total" ]; then
  echo "warning: total lines $total, expected $expected_total" >&2
fi

# Concatenate in chunk order. Seeds within a chunk are contiguous; chunks
# are ordered by seed-base ascending.
cat "$CHUNK_DIR"/chunk-*.jsonl > "$OUT_FILE"

echo "wrote $OUT_FILE ($total seeds)"
