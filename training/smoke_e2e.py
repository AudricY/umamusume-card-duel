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
import torch

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
        # Item 18: behavior-policy logging is in the serving path so the
        # warm-start checkpoint's PPO rollouts can recover importance ratios
        # without a serve-side change. /predict must return per-action
        # log-probs and the chosen-action log-prob.
        legal_count = len(sample.example["legalActions"])
        action_log_probs = served_prediction.get("actionLogProbs")
        if not action_log_probs or len(action_log_probs[0]) < legal_count:
            raise AssertionError(
                f"actionLogProbs missing or wrong length: {action_log_probs}"
            )
        selected_log_prob = served_prediction.get("selectedLogProb")
        if not selected_log_prob:
            raise AssertionError("selectedLogProb missing from /predict response")
        # The behavior distribution is softmax(logits), regardless of the
        # greedy/argmax selection rule. Chosen-action log-prob therefore lies
        # in (-inf, 0] and is the maximum over legal positions because the
        # selection is the argmax. Both invariants are checked here.
        chosen_lp = float(selected_log_prob[0])
        if chosen_lp > 1e-6:
            raise AssertionError(f"selectedLogProb must be <=0; got {chosen_lp}")
        max_legal_lp = max(action_log_probs[0][:legal_count])
        if chosen_lp < max_legal_lp - 1e-3:
            raise AssertionError(
                f"selectedLogProb {chosen_lp} should be argmax legal log-prob {max_legal_lp}"
            )
        # Sum of legal-action probs must be ~1.0; masked positions must
        # contribute zero.
        action_probs = served_prediction.get("actionProbs")
        if not action_probs:
            raise AssertionError("actionProbs missing from /predict response")
        legal_prob_sum = sum(action_probs[0][:legal_count])
        if not (0.999 <= legal_prob_sum <= 1.001):
            raise AssertionError(
                f"Sum of legal-action probs must be ~1.0; got {legal_prob_sum}"
            )
        behavior = served_prediction.get("behaviorPolicy")
        if not behavior or behavior.get("kind") != "greedy":
            raise AssertionError(f"behaviorPolicy.kind must be greedy; got {behavior}")

        # F1/PPO: stochastic sampling mode with Gumbel-max. Same request
        # body but with sampling=stochastic must return behaviorPolicy
        # {kind: 'stochastic', temperature: T}. Same samplingSeed produces
        # the same selection; legal-prob mass stays at 1.0. Across many
        # different seeds at T=1 the selected index must take at least
        # two distinct values when there are >= 2 legal actions, otherwise
        # the sampler is not stochastic.
        stoch_request_body = {
            "observation": sample.example["observation"],
            "legalActions": sample.example["legalActions"],
            "sampling": "stochastic",
            "temperature": 1.0,
        }
        stoch_a = post_json(
            f"http://127.0.0.1:{port}/predict",
            {**stoch_request_body, "samplingSeed": 42},
        )
        stoch_b = post_json(
            f"http://127.0.0.1:{port}/predict",
            {**stoch_request_body, "samplingSeed": 42},
        )
        if stoch_a["selectedIndex"] != stoch_b["selectedIndex"]:
            raise AssertionError(
                f"stochastic sampling with the same samplingSeed must be deterministic; "
                f"got {stoch_a['selectedIndex']} vs {stoch_b['selectedIndex']}"
            )
        if (stoch_a.get("behaviorPolicy") or {}).get("kind") != "stochastic":
            raise AssertionError(
                f"behaviorPolicy.kind must be stochastic in stochastic mode; got {stoch_a.get('behaviorPolicy')}"
            )
        if abs(float((stoch_a["behaviorPolicy"]).get("temperature", -1)) - 1.0) > 1e-6:
            raise AssertionError(
                f"behaviorPolicy.temperature must be 1.0; got {stoch_a['behaviorPolicy']}"
            )
        stoch_legal_prob_sum = sum(stoch_a["actionProbs"][0][:legal_count])
        if not (0.999 <= stoch_legal_prob_sum <= 1.001):
            raise AssertionError(
                f"stochastic actionProbs legal-sum must be ~1.0; got {stoch_legal_prob_sum}"
            )
        if legal_count >= 2:
            seen = set()
            for seed in range(1000, 1080):
                s = post_json(
                    f"http://127.0.0.1:{port}/predict",
                    {**stoch_request_body, "samplingSeed": seed},
                )
                seen.add(int(s["selectedIndex"][0]))
                if len(seen) >= 2:
                    break
            if len(seen) < 2:
                raise AssertionError(
                    f"stochastic sampling across 80 seeds must produce at least 2 distinct selections "
                    f"with {legal_count} legal actions; got {seen}"
                )
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()

    assert_export_rejects_vocab_mismatch(repo_root, model_dir, run_dir)
    assert_dataset_rejects_bad_schema(repo_root, run_dir, examples_path)
    assert_resume_continues_training(repo_root, run_dir, examples_path)
    assert_kl_anchor_smoke(repo_root, run_dir, examples_path, anchor_checkpoint=model_dir / "checkpoint.pt")

    print(json.dumps({
        "status": "PASS",
        "examples": str(examples_path),
        "checkpoint": str(model_dir / "checkpoint.pt"),
        "onnx": str(onnx_path),
        "directSelectedIndex": direct_prediction["selectedIndex"],
        "servedSelectedIndex": served_prediction["selectedIndex"][0],
        "servedSelectedActionId": served_prediction.get("selectedActionId", [None])[0],
    }, indent=2))


def assert_kl_anchor_smoke(repo_root: Path, run_dir: Path, source_jsonl: Path, *, anchor_checkpoint: Path) -> None:
    """Item 11/17: KL-anchor anti-forgetting plumbing smoke.

    Trains 2 epochs against a frozen anchor at weight=0.5; asserts the
    manifest's training_kwargs records the anchor checkpoint path and the
    weight, and that the per-epoch history carries a kl_loss field.
    """

    kl_dir = run_dir / "kl-anchor"
    cmd = [
        sys.executable,
        str(repo_root / "training" / "train_bc.py"),
        "--data",
        str(source_jsonl),
        "--out-dir",
        str(kl_dir),
        "--epochs",
        "2",
        "--batch-size",
        "16",
        "--hidden-dim",
        "32",
        "--depth",
        "1",
        "--kl-anchor-checkpoint",
        str(anchor_checkpoint),
        "--kl-anchor-weight",
        "0.5",
    ]
    subprocess.run(cmd, cwd=repo_root, check=True, capture_output=True, text=True)
    manifest = json.loads((kl_dir / "manifest.json").read_text(encoding="utf8"))
    kwargs = manifest.get("training_kwargs", {})
    if kwargs.get("kl_anchor_checkpoint") != str(anchor_checkpoint):
        raise AssertionError(f"manifest must record kl_anchor_checkpoint, got {kwargs}")
    if abs(float(kwargs.get("kl_anchor_weight", 0)) - 0.5) > 1e-6:
        raise AssertionError(f"manifest must record kl_anchor_weight=0.5, got {kwargs}")
    checkpoint = torch.load(kl_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    history = checkpoint.get("history", [])
    if not history or "kl_loss" not in history[0].get("train", {}):
        raise AssertionError(
            f"per-epoch history must record kl_loss when --kl-anchor-weight > 0; got {history[:1]}"
        )
    if float(history[0]["train"]["kl_loss"]) <= 0.0:
        # KL with anchor==current model would be 0; we resumed-from-scratch
        # against a *trained* anchor so the freshly-initialized model should
        # diverge from it. A 0 here means the anchor isn't actually being
        # consulted.
        raise AssertionError(
            f"kl_loss must be >0 when training a fresh model against a trained anchor; got {history[0]['train']['kl_loss']}"
        )


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
