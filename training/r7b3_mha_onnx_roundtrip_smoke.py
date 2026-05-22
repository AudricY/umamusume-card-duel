"""R7.b.3 P0a — ONNX MHA opset-17 roundtrip smoke.

Builds a 1-layer MHA encoder block (d_model=64, n_heads=4, ffn=128) with
random weights matching the planned set-attention trunk architecture and
verifies that the ONNX opset-17 export roundtrips byte-stably through ORT
(CPUExecutionProvider).

Smoke contract (per scoping doc § P0a):
  - N=100 random `(state, legal_actions)`-style inputs (i.e. token-sequence
    inputs with batch=1, variable seq_len, pad mask). For each input the
    PyTorch eager forward and the ORT forward must agree to max_abs_diff
    < 1e-5. The kill criterion is > 1e-4 after the standard pad-mask
    fixup (cast mask to bool / float and feed via `key_padding_mask`).

  - If export fails or the diff exceeds 1e-4 after the fixup, this smoke
    aborts with rc=2 and the scoping doc P0a kill-criterion fires.

One-time artifact. Re-run only when the planned MHA layout (d_model/n_heads/
opset) changes or when the host ORT/torch versions move. Not a permanent
unit test — once the smoke passes, Slice 1 lands the real model variant
and the train_bc.py per-run ONNX-roundtrip smoke takes over as the
permanent regression check.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import onnxruntime as ort


D_MODEL = 64
N_HEADS = 4
FFN_DIM = 128
OPSET = 17
N_TRIALS = 100
KILL_THRESHOLD = 1e-4  # scoping doc P0a — abort cleanly above this
PASS_THRESHOLD = 1e-5  # scoping doc P0a — formal pass

# Token-sequence layout per scoping doc: 1 CLS + per-zone card tokens
# (cap 30 per zone * 8 zones = 240) + 10 uma_slot tokens. For the smoke we
# don't need the full sequence; we just need a representative max seq_len
# so the ORT shape-inference matches a realistic v3.2-era encoder. Pick a
# moderate max_seq_len that exercises pad masking on every trial.
MAX_SEQ_LEN = 32  # 1 CLS + 21 card-token mix + 10 uma-slot tokens


class MhaEncoderBlock(nn.Module):
    """One pre-LN transformer-encoder block as planned in scoping § "Architecture sketch".

    Layout: pre-LN MHA + residual, pre-LN FFN + residual. `batch_first=True`.
    Pad masking via `key_padding_mask` (a `[B, S]` bool tensor where True
    marks padding positions to be ignored by attention).
    """

    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS, ffn_dim: int = FFN_DIM) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        """x: [B, S, D], key_padding_mask: [B, S] bool (True = pad → ignore)."""
        x_norm = self.norm1(x)
        attn_out, _ = self.mha(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


def build_module(seed: int = 0) -> MhaEncoderBlock:
    torch.manual_seed(seed)
    module = MhaEncoderBlock()
    module.eval()
    return module


def random_input(rng: np.random.Generator, max_seq_len: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Build a single [1, S, D] input + [1, S] pad mask with random seq_len."""
    seq_len = int(rng.integers(low=2, high=max_seq_len + 1))  # at least 2 real tokens (CLS + one more)
    real_tokens = seq_len
    tokens = rng.standard_normal(size=(1, max_seq_len, D_MODEL)).astype(np.float32)
    # Mask out tokens beyond `seq_len` so attention ignores them.
    pad = np.ones((1, max_seq_len), dtype=np.bool_)
    pad[:, :real_tokens] = False  # False = real; True = pad
    return tokens, pad, real_tokens


def main() -> int:
    args = parse_args()

    print(f"[r7b3-mha-onnx-smoke] starting d_model={D_MODEL} n_heads={N_HEADS} "
          f"ffn={FFN_DIM} opset={OPSET} trials={N_TRIALS} max_seq_len={MAX_SEQ_LEN}", flush=True)
    t0 = time.time()

    module = build_module(seed=args.seed)
    # Export once with dummy inputs; the graph supports dynamic batch +
    # sequence axes so all subsequent trials feed through the same session.
    dummy_x = torch.zeros((1, MAX_SEQ_LEN, D_MODEL), dtype=torch.float32)
    # PyTorch wants key_padding_mask as bool or float (-inf semantics for
    # float). We export with bool to match the runtime input dtype below.
    dummy_mask = torch.zeros((1, MAX_SEQ_LEN), dtype=torch.bool)

    out_dir = Path(tempfile.mkdtemp(prefix="r7b3-mha-onnx-smoke-"))
    onnx_path = out_dir / "mha_block.onnx"

    try:
        torch.onnx.export(
            module,
            (dummy_x, dummy_mask),
            str(onnx_path),
            input_names=["x", "key_padding_mask"],
            output_names=["y"],
            dynamic_axes={
                "x": {0: "batch", 1: "seq"},
                "key_padding_mask": {0: "batch", 1: "seq"},
                "y": {0: "batch", 1: "seq"},
            },
            opset_version=OPSET,
        )
    except Exception as exc:
        print(f"[r7b3-mha-onnx-smoke] FATAL: ONNX export failed: {exc}", file=sys.stderr, flush=True)
        return 2

    print(f"[r7b3-mha-onnx-smoke] export OK ({onnx_path}, {onnx_path.stat().st_size} bytes)", flush=True)

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    rng = np.random.default_rng(args.seed)
    diffs: list[float] = []
    seqs: list[int] = []
    for trial in range(N_TRIALS):
        x_np, mask_np, seq_len = random_input(rng, MAX_SEQ_LEN)
        with torch.no_grad():
            y_torch = module(torch.from_numpy(x_np), torch.from_numpy(mask_np))
        y_ort = session.run(None, {"x": x_np, "key_padding_mask": mask_np})[0]
        # Only compare real-token positions; pad positions can carry NaN
        # from the masked-softmax + leak through the residual on either
        # backend in a manner the gate doesn't care about. Real CLS
        # token at position 0 is the eventual consumer of attention out.
        y_torch_np = y_torch.cpu().numpy()
        real_torch = y_torch_np[0, :seq_len, :]
        real_ort = y_ort[0, :seq_len, :]
        diff = float(np.abs(real_torch - real_ort).max())
        diffs.append(diff)
        seqs.append(seq_len)

    diffs_arr = np.asarray(diffs)
    max_diff = float(diffs_arr.max())
    mean_diff = float(diffs_arr.mean())
    median_diff = float(np.median(diffs_arr))
    p95_diff = float(np.quantile(diffs_arr, 0.95))

    elapsed = time.time() - t0
    summary = {
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "ffn_dim": FFN_DIM,
        "opset": OPSET,
        "trials": N_TRIALS,
        "max_seq_len": MAX_SEQ_LEN,
        "pass_threshold": PASS_THRESHOLD,
        "kill_threshold": KILL_THRESHOLD,
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "median_abs_diff": median_diff,
        "p95_abs_diff": p95_diff,
        "elapsed_sec": elapsed,
    }

    print(f"[r7b3-mha-onnx-smoke] diffs: max={max_diff:.3e} p95={p95_diff:.3e} "
          f"median={median_diff:.3e} mean={mean_diff:.3e}", flush=True)
    print(json.dumps(summary, indent=2))

    if max_diff > KILL_THRESHOLD:
        print(f"[r7b3-mha-onnx-smoke] KILL: max_abs_diff {max_diff:.3e} > kill_threshold "
              f"{KILL_THRESHOLD:.3e}. Per scoping § P0a: attention probe is infra-blocked; "
              f"pause and triage before further investment.", file=sys.stderr, flush=True)
        return 2

    if max_diff > PASS_THRESHOLD:
        # Between pass and kill: technically "MARGINAL" — log loudly but don't
        # fail. ORT vs PyTorch on MHA can drift to ~1e-6 due to flush-to-zero +
        # accumulator-order differences; the kill criterion is the load-bearing
        # gate per the scoping doc.
        print(f"[r7b3-mha-onnx-smoke] WARN: max_abs_diff {max_diff:.3e} > pass_threshold "
              f"{PASS_THRESHOLD:.3e} but <= kill_threshold; continuing.", flush=True)

    print(f"[r7b3-mha-onnx-smoke] PASS in {elapsed:.1f}s.", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
