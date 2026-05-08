"""Smoke for ``training/calibrate_value.py``.

Reuses the rule-bot dataset/checkpoint produced by ``smoke_e2e.py`` if they
already exist; otherwise it spins up a tiny rule-bot export + 1-epoch training
run so the smoke is self-contained when invoked standalone.

Verification is deliberately plumbing-level: load the output JSON, assert all
required keys are present, types are sane, and the status field is one of the
allowed values. The strength of the calibration is not tested here.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from uma_ai.node_bridge import export_training_examples


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = repo_root / "training" / "runs" / "calibrate-smoke"
    model_dir = run_dir / "model"
    examples_path = run_dir / "examples.jsonl"
    checkpoint_path = model_dir / "checkpoint.pt"
    out_path = run_dir / "calibration.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists() or not examples_path.exists():
        export_training_examples(repo_root, examples_path, seed_start=8000, games=14, max_steps=320)
        train_cmd = [
            sys.executable,
            str(repo_root / "training" / "train_bc.py"),
            "--data",
            str(examples_path),
            "--out-dir",
            str(model_dir),
            "--epochs",
            "4",
            "--batch-size",
            "16",
            "--hidden-dim",
            "32",
            "--depth",
            "1",
        ]
        subprocess.run(train_cmd, cwd=repo_root, check=True, capture_output=True, text=True)

    calibrate_cmd = [
        sys.executable,
        str(repo_root / "training" / "calibrate_value.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--data",
        str(examples_path),
        "--out",
        str(out_path),
        "--bootstrap",
        "200",
        "--device",
        "cpu",
    ]
    completed = subprocess.run(calibrate_cmd, cwd=repo_root, check=True, capture_output=True, text=True)

    summary = json.loads(completed.stdout.strip().splitlines()[-1])
    if summary.get("status") not in ("PASS", "FAIL"):
        raise AssertionError(f"Unexpected stdout status: {summary}")

    payload = json.loads(out_path.read_text(encoding="utf8"))
    required_keys = {
        "data", "checkpoint", "n_rows", "n_fit_rows", "point_margin_k",
        "value_head", "point_margin", "brier_lift", "monotone_calibration",
        "drift", "variance", "status",
    }
    missing = required_keys.difference(payload)
    if missing:
        raise AssertionError(f"Calibration JSON missing keys: {sorted(missing)}")
    if payload["status"] not in ("PASS", "FAIL"):
        raise AssertionError(f"Unexpected status: {payload['status']}")
    if payload["n_rows"] <= 0:
        raise AssertionError(f"Expected positive n_rows, got {payload['n_rows']}")

    for label in ("value_head", "point_margin"):
        block = payload[label]
        for sub in ("ece", "brier", "reliability_by_turn", "max_bucket_ece"):
            if sub not in block:
                raise AssertionError(f"{label} missing {sub}: {block}")
        if not 0.0 <= block["ece"] <= 1.0:
            raise AssertionError(f"{label} ECE out of range: {block['ece']}")
        if not 0.0 <= block["brier"] <= 1.0:
            raise AssertionError(f"{label} Brier out of range: {block['brier']}")

    lift = payload["brier_lift"]
    for sub in ("mean", "lower_2_5", "upper_97_5", "wilson_significant_lift"):
        if sub not in lift:
            raise AssertionError(f"brier_lift missing {sub}: {lift}")
    if lift["lower_2_5"] > lift["upper_97_5"]:
        raise AssertionError(f"Bootstrap CI inverted: {lift}")

    if payload["drift"]["status"] not in ("skipped", "computed"):
        raise AssertionError(f"Unexpected drift status: {payload['drift']}")

    print(json.dumps({
        "status": "PASS",
        "out": str(out_path),
        "n_rows": payload["n_rows"],
        "value_brier": payload["value_head"]["brier"],
        "point_margin_brier": payload["point_margin"]["brier"],
        "lift_mean": lift["mean"],
        "calibration_status": payload["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
