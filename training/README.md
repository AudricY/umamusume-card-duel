# Uma AI Training

This package owns the long-term Python model path. TypeScript remains the game simulator and legal-action oracle; Python owns feature tensors, candidate-conditioned policy/value modeling, supervised training, ONNX export, and ONNX Runtime serving.

## Setup

```bash
npm run setup:python-train
```

For NVIDIA training or GPU-backed serving, install the GPU profile instead:

```bash
npm run setup:python-train:gpu
```

## End-to-End Smoke

```bash
TMPDIR=/tmp npm run test:python-train
```

The smoke test exports examples from the headless simulator, trains a shallow behavior-cloning run, exports ONNX, validates ONNX Runtime inference, starts the HTTP model server, and sends a raw observation/legal-actions prediction request.

## Manual Commands

```bash
TMPDIR=/tmp npm run sim:export-training -- --out training/runs/dev/examples.jsonl --games 32 --max-steps 360
training/.venv/bin/python training/train_bc.py --data training/runs/dev/examples.jsonl --out-dir training/runs/dev/model
training/.venv/bin/python training/export_onnx.py --checkpoint training/runs/dev/model/checkpoint.pt --out training/runs/dev/model/policy.onnx
training/.venv/bin/python training/serve_onnx.py --model training/runs/dev/model/policy.onnx --port 8765
```

Use `--device cuda` for GPU training, or `--device auto` to use CUDA when available and CPU otherwise. Use `--provider cuda` on the ONNX server when `onnxruntime-gpu` is installed.

## Artifact Layout

Exports and evaluations should be kept under `training/runs/<experiment>/`:

```text
training/runs/<experiment>/
  examples.jsonl
  examples.manifest.json
  outcome.jsonl
  outcome.manifest.json
  trace.jsonl
  eval.manifest.json
  model/
    checkpoint.pt
    manifest.json
    policy.onnx
```

The TypeScript exporters write sibling `*.manifest.json` files with command args, git SHA/dirty flag, seed/source taxonomy, phase/action-kind counts, feature schema dimensions, and terminal or margin summaries. `sim:evaluate-model` and `sim:eval-gate` can write eval artifacts with `--manifest-out`.

`train_bc.py` writes `model/manifest.json` and stores the same metadata in `checkpoint.pt`, including split mode, train/validation groups, feature schema, final metrics, and diagnostics by phase, selected action kind, source, and oracle margin bucket.
