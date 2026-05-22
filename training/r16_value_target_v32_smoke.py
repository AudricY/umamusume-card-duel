"""Smoke v3.2 slot-token loading for value-target datasets.

The historical value-head retrain/probe path uses `ValueTargetDataset`, not
`MctsSelfPlayDataset`. This smoke verifies that value-target rows now emit the
same v3.2 tensors (`uma_slot_card_ids`, `uma_slot_features`) that the main
MCTS distill path emits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uma_ai.features import STATE_DIM, UMA_SLOT_COUNT, UMA_SLOT_FEATURE_DIM  # noqa: E402
from uma_ai.selfplay_dataset import collate_mcts_selfplay_batch  # noqa: E402
from uma_ai.value_target_dataset import ValueTargetDataset  # noqa: E402


DEFAULT_DATA = (
    Path(__file__).resolve().parents[1]
    / "runs/v32-uniform-retrain-from-R110-W6/loop/iter-1/selfplay.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = Path(args.data)
    if not data.exists():
        raise SystemExit(f"smoke data not found: {data}")

    dataset = ValueTargetDataset(data, state_dim=STATE_DIM, uses_uma_slot_tokens=True)
    batch = collate_mcts_selfplay_batch([dataset[0]])
    if "uma_slot_card_ids" not in batch or "uma_slot_features" not in batch:
        raise SystemExit("missing v3.2 slot-token tensors in value-target batch")
    if tuple(batch["state_features"].shape) != (1, STATE_DIM):
        raise SystemExit(f"bad state_features shape: {tuple(batch['state_features'].shape)}")
    if tuple(batch["uma_slot_card_ids"].shape) != (1, UMA_SLOT_COUNT):
        raise SystemExit(f"bad uma_slot_card_ids shape: {tuple(batch['uma_slot_card_ids'].shape)}")
    if tuple(batch["uma_slot_features"].shape) != (1, UMA_SLOT_COUNT, UMA_SLOT_FEATURE_DIM):
        raise SystemExit(f"bad uma_slot_features shape: {tuple(batch['uma_slot_features'].shape)}")
    print(
        "[r16-value-target-v32-smoke] PASS "
        f"samples={len(dataset)} state_dim={STATE_DIM} "
        f"uma_slot_card_ids={tuple(batch['uma_slot_card_ids'].shape)} "
        f"uma_slot_features={tuple(batch['uma_slot_features'].shape)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
