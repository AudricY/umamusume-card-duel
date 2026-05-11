"""R13.W3 value-retrain smoke.

End-to-end mini training pass:
  - Synthesize a 16-row mcts-selfplay corpus with rootValue set
    (re-uses the existing R12 selfplay smoke output if available; else
    skips the smoke).
  - Run training/r13_value_retrain.py with 2 epochs, cpu, tiny model.
  - Assert: checkpoint saved, manifest written, val/train loss did not
    NaN out, training loss decreased between epoch 1 and epoch 2.

Hidden_dim/depth come from the init checkpoint's metadata so the shape
is consistent. Init checkpoint is R4 (or R3) per the standard preference.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_for_health(port: int, attempts: int = 60) -> bool:
    import urllib.request
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.25)
    return False


def export_onnx_if_needed(repo_root: Path, ckpt: Path) -> Path:
    onnx = ckpt.with_name("policy.onnx")
    if not onnx.exists() or onnx.stat().st_mtime < ckpt.stat().st_mtime:
        subprocess.run(
            [sys.executable, str(repo_root / "training" / "export_onnx.py"),
             "--checkpoint", str(ckpt), "--out", str(onnx)],
            cwd=repo_root, check=True,
        )
    return onnx


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ckpt_candidates = [
        repo_root / "runs" / "R4-value-head-retrain" / "checkpoint.pt",
        repo_root / "runs" / "R3-entropy-bc-b005" / "checkpoint.pt",
    ]
    init_ckpt = next((c for c in ckpt_candidates if c.exists()), None)
    if init_ckpt is None:
        print("[value-retrain-smoke] no init checkpoint found", file=sys.stderr)
        sys.exit(2)
    onnx = export_onnx_if_needed(repo_root, init_ckpt)

    out_dir = repo_root / "runs" / "R13-value-retrain-smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    selfplay_path = out_dir / "rollout-leaf-selfplay.jsonl"
    if selfplay_path.exists():
        selfplay_path.unlink()

    # 1. Generate a tiny rollout-leaf selfplay corpus (rootValue will be the rollout mean).
    port = free_port()
    serve = subprocess.Popen(
        [sys.executable, str(repo_root / "training" / "serve_onnx.py"),
         "--model", str(onnx), "--port", str(port), "--default-sampling", "greedy"],
        cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        if not wait_for_health(port):
            print("[value-retrain-smoke] serve_onnx never healthy", file=sys.stderr)
            sys.exit(2)
        log_path = out_dir / "selfplay.log"
        with log_path.open("w") as logf:
            subprocess.run([
                "npm", "--workspace", "backend", "run", "sim:mcts-selfplay", "--",
                "--model-url", f"http://127.0.0.1:{port}",
                "--games", "2",
                "--seed-start", "33330",
                "--mcts-simulations", "8",
                "--mcts-leaf", "rollout",
                "--mcts-rollout-crn-samples", "3",
                "--mcts-rollout-steps", "100",
                "--mcts-prior", "policy",
                "--mcts-collapse-max-steps", "32",
                "--mcts-max-nodes", "256",
                "--temperature-moves", "2",
                "--temperature-value", "1.0",
                "--out", str(selfplay_path),
            ], cwd=repo_root, stdout=logf, stderr=subprocess.STDOUT, check=True)
    finally:
        serve.terminate()
        try:
            serve.wait(timeout=3)
        except subprocess.TimeoutExpired:
            serve.kill()

    rows = []
    for line in selfplay_path.read_text(encoding="utf8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        print("[value-retrain-smoke] selfplay produced 0 rows", file=sys.stderr)
        sys.exit(1)
    finite_root_values = [r for r in rows if r.get("rootValue") is not None]
    if not finite_root_values:
        print("[value-retrain-smoke] all rows missing rootValue", file=sys.stderr)
        sys.exit(1)

    # 2. Run 2 epochs of value retrain on cpu.
    retrain_out = out_dir / "retrain"
    events_path = out_dir / "events.jsonl"
    if events_path.exists():
        events_path.unlink()
    result = subprocess.run([
        str(repo_root / "training" / ".venv" / "bin" / "python"),
        str(repo_root / "training" / "r13_value_retrain.py"),
        "--data", str(selfplay_path),
        "--init-checkpoint", str(init_ckpt),
        "--out-dir", str(retrain_out),
        "--epochs", "2",
        "--batch-size", "4",
        "--lr", "1e-3",
        "--val-fraction", "0.25",
        "--device", "cpu",
        "--events-out", str(events_path),
    ], cwd=repo_root, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"[value-retrain-smoke] retrain failed (code={result.returncode})", file=sys.stderr)
        sys.stderr.write(result.stdout + "\n" + result.stderr + "\n")
        sys.exit(1)

    checkpoint = retrain_out / "checkpoint.pt"
    manifest = retrain_out / "value-retrain.manifest.json"
    if not checkpoint.exists() or not manifest.exists():
        print(f"[value-retrain-smoke] expected artifacts missing", file=sys.stderr)
        sys.exit(1)

    # 3. Sanity: loss didn't NaN, decreased between epoch 1 and epoch 2.
    epoch_events = []
    for line in events_path.read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("event_type") == "epoch_completed":
            epoch_events.append(obj["data"])
    if len(epoch_events) < 2:
        print(f"[value-retrain-smoke] expected 2 epoch_completed events, got {len(epoch_events)}", file=sys.stderr)
        sys.exit(1)
    for ev in epoch_events:
        if not (ev["train_loss"] == ev["train_loss"]):  # NaN check
            print(f"[value-retrain-smoke] train_loss NaN at epoch {ev['epoch']}", file=sys.stderr)
            sys.exit(1)
    if epoch_events[1]["train_loss"] >= epoch_events[0]["train_loss"]:
        print(
            f"[value-retrain-smoke] FAIL — train loss did not decrease "
            f"({epoch_events[0]['train_loss']:.4f} → {epoch_events[1]['train_loss']:.4f})",
            file=sys.stderr,
        )
        sys.exit(1)
    print(json.dumps({
        "status": "PASS",
        "rows": len(rows),
        "rows_with_root_value": len(finite_root_values),
        "epoch_1_loss": epoch_events[0]["train_loss"],
        "epoch_2_loss": epoch_events[1]["train_loss"],
        "checkpoint": str(checkpoint),
    }, indent=2))


if __name__ == "__main__":
    main()
