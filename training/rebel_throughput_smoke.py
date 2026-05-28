from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rebel_orchestrator import build_selfplay_cmd, build_train_cmd


def _args(**overrides: object) -> argparse.Namespace:
    values = {
        "games": 1,
        "seed_start": 0,
        "particles": 2,
        "search_iterations": 4,
        "rollout_steps": 5,
        "max_steps": 20,
        "model_side": "player",
        "deck_sampling": "fixed",
        "workers": 8,
        "use_release_binary": False,
        "selfplay_onnx_path": "/tmp/policy.onnx",
        "selfplay_device": "auto",
        "selfplay_cuda_device_id": 0,
        "neural_policy_weight": 0.25,
        "neural_value_weight": 0.25,
        "neural_leaf_weight": 1.0,
        "selfplay_inference_batch_size": 32,
        "selfplay_inference_max_wait_us": 2000,
        "epochs": 1,
        "batch_size": None,
        "lr": None,
        "state_dim": 110,
        "device": "cuda",
        "amp": True,
        "dataloader_workers": 4,
        "compile": False,
        "hidden_dim": 128,
        "depth": 3,
        "dropout": 0.05,
        "grad_accum": 2,
        "init_from_checkpoint": "/tmp/init.pt",
        "policy_weight": 1.0,
        "value_weight": 0.1,
        "q_value_head": False,
        "q_value_weight": 0.0,
        "uma_slot_tokens": False,
        "kl_anchor_checkpoint": None,
        "kl_anchor_weight": 0.0,
        "entropy_bonus": 0.0,
        "smoke": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _contains_pair(cmd: list[str], flag: str, value: str) -> bool:
    return any(cmd[i] == flag and i + 1 < len(cmd) and cmd[i + 1] == value for i in range(len(cmd)))


def main() -> None:
    repo = Path("/repo")
    args = _args()
    train_cmd = build_train_cmd(args, repo, Path("/tmp/rebel.jsonl"), Path("/tmp/train"))
    selfplay_cmd = build_selfplay_cmd(args, repo, Path("/tmp/rebel.jsonl"), Path("/tmp/selfplay.json"))

    required_train_pairs = {
        "--batch-size": "256",
        "--lr": "0.0006",
        "--dataloader-workers": "4",
        "--grad-accum": "2",
        "--init-from-checkpoint": "/tmp/init.pt",
    }
    for flag, value in required_train_pairs.items():
        if not _contains_pair(train_cmd, flag, value):
            raise AssertionError(f"missing {flag} {value} in train command: {train_cmd}")
    for flag in ["--amp", "--hidden-dim", "--depth", "--dropout"]:
        if flag not in train_cmd:
            raise AssertionError(f"missing {flag} in train command: {train_cmd}")
    if not _contains_pair(selfplay_cmd, "--workers", "8"):
        raise AssertionError(f"missing --workers 8 in selfplay command: {selfplay_cmd}")
    required_selfplay_pairs = {
        "--onnx-path": "/tmp/policy.onnx",
        "--device": "cuda",
        "--neural-leaf-weight": "1.0",
        "--inference-batch-size": "32",
    }
    for flag, value in required_selfplay_pairs.items():
        if not _contains_pair(selfplay_cmd, flag, value):
            raise AssertionError(f"missing {flag} {value} in selfplay command: {selfplay_cmd}")

    smoke_cmd = build_train_cmd(
        _args(smoke=True, device="cpu", amp=None, dataloader_workers=None, init_from_checkpoint=None),
        repo,
        Path("/tmp/rebel.jsonl"),
        Path("/tmp/train"),
    )
    if not _contains_pair(smoke_cmd, "--batch-size", "16"):
        raise AssertionError(f"smoke command should keep tiny batch: {smoke_cmd}")
    if "--amp" in smoke_cmd or "--dataloader-workers" in smoke_cmd:
        raise AssertionError(f"smoke command should not enable CUDA throughput flags: {smoke_cmd}")

    smoke_selfplay_cmd = build_selfplay_cmd(
        _args(smoke=True, device="cpu", selfplay_device="auto"),
        repo,
        Path("/tmp/rebel.jsonl"),
        Path("/tmp/selfplay.json"),
    )
    if not _contains_pair(smoke_selfplay_cmd, "--device", "cpu"):
        raise AssertionError(f"selfplay auto device should follow CPU training device: {smoke_selfplay_cmd}")

    print("rebel_throughput_smoke: ok")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"rebel_throughput_smoke: FAIL: {exc}", file=sys.stderr)
        raise
