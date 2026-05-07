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
