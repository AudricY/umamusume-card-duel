from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "rebel-selfplay.jsonl"
    selfplay_manifest = out_dir / "selfplay.manifest.json"
    train_dir = out_dir / "train"
    export_path = out_dir / "policy.onnx"

    run(build_selfplay_cmd(args, repo, data_path, selfplay_manifest), cwd=repo / "engine-rs", env=ort_env(repo))
    row_summary = validate_rebel_rows(data_path)

    train_cmd = build_train_cmd(args, repo, data_path, train_dir)
    run(train_cmd, cwd=repo)

    export_status: dict[str, Any] = {"status": "skipped", "reason": "--skip-export"}
    if not args.skip_export:
        run(
            [
                training_python(repo),
                str(repo / "training" / "export_onnx.py"),
                "--checkpoint",
                str(train_dir / "checkpoint.pt"),
                "--out",
                str(export_path),
            ],
            cwd=repo,
        )
        export_status = {"status": "ok", "onnx": str(export_path)}

    gates = run_gates(args, repo, out_dir, export_path) if not args.skip_gates and not args.skip_export else {
        "fixed": {"status": "not-run", "reason": "--skip-gates or --skip-export"},
        "uniform_deck_diverse": {"status": "not-run", "reason": "--skip-gates or --skip-export"},
        "side_split": {"status": "recorded-in-row-fields", "field": "sideId"},
    }

    training_settings = effective_training_settings(args)
    manifest = {
        "name": "R17-rebel-e2e",
        "status": "complete",
        "data_mode": "rebel",
        "engine": "rust",
        "selfplay_binary": "sim-rebel-selfplay",
        "search": "public-belief-cfr-v1",
        "deck_sampling": args.deck_sampling,
        "model_side": args.model_side,
        "artifacts": {
            "selfplay": str(data_path),
            "selfplay_manifest": str(selfplay_manifest),
            "train_dir": str(train_dir),
            "export": export_status,
        },
        "row_summary": row_summary,
        "knobs": {
            "games": args.games,
            "seed_start": args.seed_start,
            "particles": args.particles,
            "search_iterations": args.search_iterations,
            "rollout_steps": args.rollout_steps,
            "max_steps": args.max_steps,
            "workers": args.workers,
            "epochs": args.epochs,
            **training_settings,
            "state_dim": args.state_dim,
            "q_value_head": args.q_value_head,
            "q_value_weight": args.q_value_weight,
            "kl_anchor_weight": args.kl_anchor_weight,
            "entropy_bonus": args.entropy_bonus,
            "use_release_binary": args.use_release_binary,
            "selfplay_onnx_path": args.selfplay_onnx_path,
            "neural_policy_weight": args.neural_policy_weight,
            "neural_value_weight": args.neural_value_weight,
        },
        "gates": gates,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    print(json.dumps(manifest, indent=2))


def build_selfplay_cmd(
    args: argparse.Namespace,
    repo: Path,
    data_path: Path,
    selfplay_manifest: Path,
) -> list[str]:
    if args.use_release_binary:
        cmd = [str(repo / "engine-rs" / "target" / "release" / "sim-rebel-selfplay")]
    else:
        cmd = ["cargo", "run", "-p", "sim-cli", "--bin", "sim-rebel-selfplay", "--"]
    cmd.extend(
        [
            "--seeds",
            str(args.games),
            "--seed-start",
            str(args.seed_start),
            "--particles",
            str(args.particles),
            "--search-iterations",
            str(args.search_iterations),
            "--rollout-steps",
            str(args.rollout_steps),
            "--max-steps",
            str(args.max_steps),
            "--model-side",
            args.model_side,
            "--deck-sampling",
            args.deck_sampling,
            "--workers",
            str(args.workers),
            "--out",
            str(data_path),
            "--manifest-out",
            str(selfplay_manifest),
        ]
    )
    if args.selfplay_onnx_path:
        cmd.extend(
            [
                "--onnx-path",
                str(args.selfplay_onnx_path),
                "--device",
                args.selfplay_device,
                "--cuda-device-id",
                str(args.selfplay_cuda_device_id),
                "--neural-policy-weight",
                str(args.neural_policy_weight),
                "--neural-value-weight",
                str(args.neural_value_weight),
            ]
        )
    return cmd


def effective_training_settings(args: argparse.Namespace) -> dict[str, Any]:
    batch_size = args.batch_size if args.batch_size is not None else (16 if args.smoke else 256)
    lr = args.lr if args.lr is not None else (6e-4 if batch_size >= 256 else 3e-4)
    dataloader_workers = (
        args.dataloader_workers
        if args.dataloader_workers is not None
        else (0 if args.smoke or args.device != "cuda" else 4)
    )
    amp = args.amp if args.amp is not None else (False if args.smoke else args.device == "cuda")
    return {
        "batch_size": batch_size,
        "lr": lr,
        "amp": amp,
        "dataloader_workers": dataloader_workers,
        "compile": bool(args.compile),
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "dropout": args.dropout,
        "grad_accum": args.grad_accum,
        "init_from_checkpoint": args.init_from_checkpoint,
    }


def build_train_cmd(
    args: argparse.Namespace,
    repo: Path,
    data_path: Path,
    train_dir: Path,
) -> list[str]:
    settings = effective_training_settings(args)
    train_cmd = [
        training_python(repo),
        str(repo / "training" / "train_bc.py"),
        "--data",
        str(data_path),
        "--out-dir",
        str(train_dir),
        "--data-mode",
        "rebel",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(settings["batch_size"]),
        "--state-dim",
        str(args.state_dim),
        "--hidden-dim",
        str(settings["hidden_dim"]),
        "--depth",
        str(settings["depth"]),
        "--dropout",
        str(settings["dropout"]),
        "--lr",
        str(settings["lr"]),
        "--split-by",
        "row",
        "--device",
        args.device,
        "--value-weight",
        str(args.value_weight),
        "--policy-weight",
        str(args.policy_weight),
        "--kl-anchor-weight",
        str(args.kl_anchor_weight),
        "--entropy-bonus",
        str(args.entropy_bonus),
        "--grad-accum",
        str(settings["grad_accum"]),
    ]
    if settings["amp"]:
        train_cmd.append("--amp")
    if settings["compile"]:
        train_cmd.append("--compile")
    if settings["dataloader_workers"] > 0:
        train_cmd.extend(["--dataloader-workers", str(settings["dataloader_workers"])])
    if settings["init_from_checkpoint"]:
        train_cmd.extend(["--init-from-checkpoint", str(settings["init_from_checkpoint"])])
    if args.q_value_head:
        train_cmd.extend(["--q-value-head", "--q-value-weight", str(args.q_value_weight)])
    if args.uma_slot_tokens:
        train_cmd.append("--uma-slot-tokens")
    if args.kl_anchor_checkpoint:
        train_cmd.extend(["--kl-anchor-checkpoint", args.kl_anchor_checkpoint])
    return train_cmd


def training_python(repo: Path) -> str:
    venv_python = repo / "training" / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def run_gates(args: argparse.Namespace, repo: Path, out_dir: Path, onnx_path: Path) -> dict[str, Any]:
    fixed_manifest = out_dir / "gate-fixed.manifest.json"
    uniform_manifest = out_dir / "gate-uniform.manifest.json"
    fixed = run_gate(args, repo, onnx_path, fixed_manifest, deck_sampling="fixed", seed_start=args.gate_seed_start)
    uniform = run_gate(
        args,
        repo,
        onnx_path,
        uniform_manifest,
        deck_sampling="uniform",
        seed_start=args.gate_seed_start + args.gate_games,
    )
    return {
        "fixed": fixed,
        "uniform_deck_diverse": uniform,
        "side_split": {
            "status": "recorded-in-gate-manifests",
            "fields": ["summary.playerSide", "summary.opponentSide"],
        },
    }


def run_gate(
    args: argparse.Namespace,
    repo: Path,
    onnx_path: Path,
    manifest_path: Path,
    *,
    deck_sampling: str,
    seed_start: int,
) -> dict[str, Any]:
    cmd = [
        "cargo",
        "run",
        "-p",
        "sim-cli",
        "--bin",
        "sim-eval-gate",
        "--",
        "--games",
        str(args.gate_games),
        "--seed-start",
        str(seed_start),
        "--sims",
        str(args.gate_sims),
        "--mcts-prior",
        "policy",
        "--mcts-leaf",
        args.gate_leaf,
        "--onnx-path",
        str(onnx_path),
        "--model-side",
        "both",
        "--deck-sampling",
        deck_sampling,
        "--max-steps",
        str(args.gate_max_steps),
        "--workers",
        str(args.gate_workers),
        "--batch-size",
        str(args.gate_batch_size),
        "--manifest-out",
        str(manifest_path),
    ]
    env = ort_env(repo)
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(repo / "engine-rs"), check=True, env=env)
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    summary = manifest.get("summary") or {}
    return {
        "status": manifest.get("status", "unknown"),
        "passed": bool(manifest.get("passed", False)),
        "manifest": str(manifest_path),
        "deck_sampling": deck_sampling,
        "games": summary.get("overall", {}).get("games", summary.get("games")),
        "win_rate": summary.get("overall", {}).get("winRate", summary.get("modelWinRate")),
        "wilson_lower": (summary.get("wilson95") or {}).get("lower"),
        "player_side": summary.get("playerSide"),
        "opponent_side": summary.get("opponentSide"),
    }


def ort_env(repo: Path) -> dict[str, str]:
    import os

    env = os.environ.copy()
    if "ORT_DYLIB_PATH" not in env:
        capi = repo / "training" / ".venv" / "lib" / "python3.12" / "site-packages" / "onnxruntime" / "capi"
        candidate = capi / "libonnxruntime.so.1.22.0"
        if candidate.exists():
            env["ORT_DYLIB_PATH"] = str(candidate)
    return env


def validate_rebel_rows(path: Path) -> dict[str, Any]:
    required = {
        "kind",
        "schemaVersion",
        "beliefSchemaVersion",
        "observation",
        "beliefFeatures",
        "publicHistoryDigest",
        "legalActions",
        "searchPolicy",
        "searchActionValues",
        "beliefValue",
        "privateStateValues",
        "valueTarget",
        "particleCount",
        "searchIterations",
        "searchAlgorithm",
        "beliefSampler",
        "playerDeckId",
        "opponentDeckId",
    }
    rows = 0
    policy_entropy_sum = 0.0
    legal_hist: dict[str, int] = {}
    with path.open("r", encoding="utf8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = sorted(required - set(row))
            if missing:
                raise SystemExit(f"{path}:{line_number}: missing keys {missing}")
            if row["kind"] != "rebel-selfplay":
                raise SystemExit(f"{path}:{line_number}: bad kind {row['kind']!r}")
            if int(row["schemaVersion"]) != 1 or int(row["beliefSchemaVersion"]) != 1:
                raise SystemExit(f"{path}:{line_number}: bad schema versions")
            actions = row["legalActions"]
            policy = row["searchPolicy"]
            q_values = row["searchActionValues"]
            if len(actions) < 2 or len(policy) != len(actions) or len(q_values) != len(actions):
                raise SystemExit(f"{path}:{line_number}: action/search target length mismatch")
            total = sum(float(v) for v in policy)
            if abs(total - 1.0) > 1e-4:
                raise SystemExit(f"{path}:{line_number}: searchPolicy sums to {total}")
            belief_vector = (row.get("beliefFeatures") or {}).get("vector") or []
            if len(belief_vector) != 16:
                raise SystemExit(f"{path}:{line_number}: belief feature dim {len(belief_vector)} != 16")
            policy_entropy_sum += -sum(float(p) * safe_log(float(p)) for p in policy if float(p) > 0.0)
            legal_hist[str(len(actions))] = legal_hist.get(str(len(actions)), 0) + 1
            rows += 1
    if rows == 0:
        raise SystemExit(f"{path}: no rebel-selfplay rows")
    return {
        "rows": rows,
        "legal_action_count_histogram": legal_hist,
        "mean_search_policy_entropy": policy_entropy_sum / rows,
    }


def safe_log(value: float) -> float:
    import math

    return math.log(max(value, 1e-12))


def run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one R17 ReBeL E2E iteration.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--smoke", action="store_true", help="Use tiny training defaults for CPU smoke runs.")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--particles", type=int, default=16)
    parser.add_argument("--search-iterations", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--model-side", choices=["player", "opponent", "both"], default="both")
    parser.add_argument("--deck-sampling", default="fixed")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--use-release-binary", action="store_true")
    parser.add_argument("--selfplay-onnx-path", default=None)
    parser.add_argument("--selfplay-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--selfplay-cuda-device-id", type=int, default=0)
    parser.add_argument("--neural-policy-weight", type=float, default=0.25)
    parser.add_argument("--neural-value-weight", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--state-dim", type=int, default=110)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dataloader-workers", type=int, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--init-from-checkpoint", default=None)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=0.1)
    parser.add_argument("--q-value-head", action="store_true")
    parser.add_argument("--q-value-weight", type=float, default=0.0)
    parser.add_argument("--uma-slot-tokens", action="store_true")
    parser.add_argument("--kl-anchor-checkpoint", default=None)
    parser.add_argument("--kl-anchor-weight", type=float, default=0.0)
    parser.add_argument("--entropy-bonus", type=float, default=0.0)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-gates", action="store_true")
    parser.add_argument("--gate-games", type=int, default=20)
    parser.add_argument("--gate-seed-start", type=int, default=90000)
    parser.add_argument("--gate-sims", type=int, default=100)
    parser.add_argument("--gate-leaf", choices=["rollout", "value-head"], default="rollout")
    parser.add_argument("--gate-max-steps", type=int, default=500)
    parser.add_argument("--gate-workers", type=int, default=1)
    parser.add_argument("--gate-batch-size", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    main()
