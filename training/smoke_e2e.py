from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort

from uma_ai.dataset import load_policy_samples
from uma_ai.features import legal_actions_to_features, observation_to_features
from uma_ai.node_bridge import export_training_examples


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = repo_root / "training" / "runs" / "smoke"
    model_dir = run_dir / "model"
    examples_path = run_dir / "examples.jsonl"
    onnx_path = model_dir / "policy.onnx"
    run_dir.mkdir(parents=True, exist_ok=True)

    export_training_examples(repo_root, examples_path, seed_start=7000, games=14, max_steps=360)
    train = [
        sys.executable,
        str(repo_root / "training" / "train_bc.py"),
        "--data",
        str(examples_path),
        "--out-dir",
        str(model_dir),
        "--epochs",
        "16",
        "--batch-size",
        "32",
        "--hidden-dim",
        "96",
        "--depth",
        "2",
    ]
    subprocess.run(train, cwd=repo_root, check=True)
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf8"))
    assert_grouped_split(manifest)
    subprocess.run([
        sys.executable,
        str(repo_root / "training" / "export_onnx.py"),
        "--checkpoint",
        str(model_dir / "checkpoint.pt"),
        "--out",
        str(onnx_path),
    ], cwd=repo_root, check=True)

    sample = next(iter(load_policy_samples(examples_path)))
    direct_prediction = run_onnx_prediction(onnx_path, sample.example)
    port = free_port()
    server = subprocess.Popen([
        sys.executable,
        str(repo_root / "training" / "serve_onnx.py"),
        "--model",
        str(onnx_path),
        "--port",
        str(port),
    ], cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        wait_for_health(port)
        served_prediction = post_json(
            f"http://127.0.0.1:{port}/predict",
            {
                "observation": sample.example["observation"],
                "legalActions": sample.example["legalActions"],
            },
        )
        selected = int(served_prediction["selectedIndex"][0])
        if selected < 0 or selected >= len(sample.example["legalActions"]):
            raise AssertionError(f"Server selected invalid index {selected}")
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()

    print(json.dumps({
        "status": "PASS",
        "examples": str(examples_path),
        "checkpoint": str(model_dir / "checkpoint.pt"),
        "onnx": str(onnx_path),
        "directSelectedIndex": direct_prediction["selectedIndex"],
        "servedSelectedIndex": served_prediction["selectedIndex"][0],
        "servedSelectedActionId": served_prediction.get("selectedActionId", [None])[0],
    }, indent=2))


def assert_grouped_split(manifest: dict) -> None:
    split = manifest.get("split", {})
    if split.get("split_by") != "episode":
        raise AssertionError(f"Expected episode split, got {split}")
    train_groups = set(split.get("train_groups", []))
    val_groups = set(split.get("val_groups", []))
    if not val_groups:
        raise AssertionError(f"Expected validation groups, got {split}")
    leaked = train_groups.intersection(val_groups)
    if leaked:
        raise AssertionError(f"Train/val group leakage: {sorted(leaked)}")


def run_onnx_prediction(model_path: Path, example: dict) -> dict[str, int]:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    state = observation_to_features(example["observation"])[None, :]
    actions = legal_actions_to_features(example["legalActions"])[None, :, :]
    mask = np.ones(actions.shape[:2], dtype=np.bool_)
    logits, _value = session.run(None, {
        "state_features": state.astype(np.float32),
        "action_features": actions.astype(np.float32),
        "action_mask": mask,
    })
    selected = int(logits.argmax(axis=1)[0])
    if selected < 0 or selected >= len(example["legalActions"]):
        raise AssertionError(f"ONNX selected invalid index {selected}")
    return {"selectedIndex": selected}


def wait_for_health(port: int) -> None:
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            post_json(f"http://127.0.0.1:{port}/health", None, method="GET")
            return
        except Exception:
            time.sleep(0.2)
    raise TimeoutError("ONNX server did not become healthy")


def post_json(url: str, payload: dict | None, *, method: str = "POST") -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf8")
    request = urllib.request.Request(url, data=data, method=method)
    if payload is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf8"))


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


if __name__ == "__main__":
    main()
