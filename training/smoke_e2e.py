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
from uma_ai.features import card_vocab_metadata, legal_actions_to_features, observation_to_features
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
    assert_card_vocab_recorded(manifest)
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

    assert_export_rejects_vocab_mismatch(repo_root, model_dir, run_dir)
    assert_dataset_rejects_bad_schema(repo_root, run_dir, examples_path)
    assert_resume_continues_training(repo_root, run_dir, examples_path)

    print(json.dumps({
        "status": "PASS",
        "examples": str(examples_path),
        "checkpoint": str(model_dir / "checkpoint.pt"),
        "onnx": str(onnx_path),
        "directSelectedIndex": direct_prediction["selectedIndex"],
        "servedSelectedIndex": served_prediction["selectedIndex"][0],
        "servedSelectedActionId": served_prediction.get("selectedActionId", [None])[0],
    }, indent=2))


def assert_resume_continues_training(repo_root: Path, run_dir: Path, source_jsonl: Path) -> None:
    seed_dir = run_dir / "resume" / "seed"
    resume_dir = run_dir / "resume" / "resumed"
    base_args = [
        sys.executable,
        str(repo_root / "training" / "train_bc.py"),
        "--data",
        str(source_jsonl),
        "--epochs",
        "4",
        "--batch-size",
        "16",
        "--hidden-dim",
        "32",
        "--depth",
        "1",
        "--lr-schedule",
        "cosine",
        "--lr-warmup-steps",
        "2",
        "--grad-accum",
        "2",
    ]
    seed_args = base_args + ["--out-dir", str(seed_dir)]
    subprocess.run(seed_args, cwd=repo_root, check=True, capture_output=True, text=True)
    seed_manifest = json.loads((seed_dir / "manifest.json").read_text(encoding="utf8"))
    if seed_manifest.get("onnx_roundtrip_smoke", {}).get("status") != "PASS":
        raise AssertionError(f"Seed run did not pass ONNX roundtrip smoke: {seed_manifest.get('onnx_roundtrip_smoke')}")

    resume_args = base_args + [
        "--out-dir",
        str(resume_dir),
        "--epochs",
        "8",
        "--resume",
        str(seed_dir / "checkpoint.pt"),
    ]
    subprocess.run(resume_args, cwd=repo_root, check=True, capture_output=True, text=True)
    resume_manifest = json.loads((resume_dir / "manifest.json").read_text(encoding="utf8"))
    if resume_manifest.get("training_kwargs", {}).get("resume_from") != str(seed_dir / "checkpoint.pt"):
        raise AssertionError(f"Resume manifest did not record resume_from path: {resume_manifest.get('training_kwargs')}")
    if resume_manifest.get("onnx_roundtrip_smoke", {}).get("status") != "PASS":
        raise AssertionError(f"Resume run did not pass ONNX roundtrip smoke: {resume_manifest.get('onnx_roundtrip_smoke')}")


def assert_dataset_rejects_bad_schema(repo_root: Path, run_dir: Path, source_jsonl: Path) -> None:
    from uma_ai.dataset import JsonlPolicyDataset, RowSchemaError

    rows = source_jsonl.read_text(encoding="utf8").strip().splitlines()
    if not rows:
        raise AssertionError(f"Source JSONL {source_jsonl} unexpectedly empty")

    missing = run_dir / "missing_schema.jsonl"
    bumped = run_dir / "bumped_schema.jsonl"
    missing_lines = []
    bumped_lines = []
    for raw in rows:
        payload = json.loads(raw)
        no_version = {key: value for key, value in payload.items() if key != "schemaVersion"}
        missing_lines.append(json.dumps(no_version))
        bumped_payload = dict(payload)
        bumped_payload["schemaVersion"] = 99
        bumped_lines.append(json.dumps(bumped_payload))
    missing.write_text("\n".join(missing_lines) + "\n", encoding="utf8")
    bumped.write_text("\n".join(bumped_lines) + "\n", encoding="utf8")

    try:
        JsonlPolicyDataset(missing)
    except RowSchemaError as exc:
        if "Missing schemaVersion" not in str(exc):
            raise AssertionError(f"Unexpected RowSchemaError: {exc}")
    else:
        raise AssertionError("Dataset must reject rows with no schemaVersion")

    try:
        JsonlPolicyDataset(bumped)
    except RowSchemaError as exc:
        if "Incompatible schemaVersion" not in str(exc):
            raise AssertionError(f"Unexpected RowSchemaError: {exc}")
    else:
        raise AssertionError("Dataset must reject rows with bumped schemaVersion")

    JsonlPolicyDataset(missing, strict_schema_version=False)


def assert_export_rejects_vocab_mismatch(repo_root: Path, model_dir: Path, run_dir: Path) -> None:
    import torch

    checkpoint_path = model_dir / "checkpoint.pt"
    tampered_path = run_dir / "tampered.pt"
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    schema = dict(raw.get("feature_schema", {}))
    vocab_meta = dict(schema.get("card_vocab", {}))
    vocab_meta["hash"] = "tampered-hash-deadbeef"
    schema["card_vocab"] = vocab_meta
    raw["feature_schema"] = schema
    torch.save(raw, tampered_path)

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "training" / "export_onnx.py"),
            "--checkpoint",
            str(tampered_path),
            "--out",
            str(run_dir / "tampered.onnx"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        raise AssertionError(
            f"export_onnx must reject mismatched card vocab hash, got returncode 0 stdout={result.stdout!r}"
        )
    if "Card vocab hash mismatch" not in result.stderr:
        raise AssertionError(
            f"Expected 'Card vocab hash mismatch' in stderr; got {result.stderr!r}"
        )


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


def assert_card_vocab_recorded(manifest: dict) -> None:
    schema = manifest.get("feature_schema", {})
    vocab = schema.get("card_vocab")
    if not vocab:
        raise AssertionError(f"Expected card_vocab metadata in manifest feature_schema, got {schema}")
    runtime = card_vocab_metadata()
    if vocab.get("hash") != runtime.get("hash"):
        raise AssertionError(
            f"Manifest vocab hash {vocab.get('hash')} != runtime {runtime.get('hash')}"
        )
    if vocab.get("vocabSize", 0) <= 0:
        raise AssertionError(f"Card vocab has unexpected size {vocab.get('vocabSize')}")


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
