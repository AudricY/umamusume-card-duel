"""B1 smoke for gpu-batched-inference-throughput.md.

Loads runs/R110-W6-repro/iter-0/policy.onnx on CPU and CUDA, runs B=1
baseline, then B in {2, 4, 8, 16, 32, 64} stacked from independent rows,
asserts each row's output matches its own B=1 call within 1e-3, and reports
per-call latency to inform the throughput-probe scoping doc.

Not bit-exact: ORT reduction order across batch members can shift FP outputs
by ~1e-6. The G4 envelope (wilson_lower within 0.02 at n=200) is the gate
that matters; this smoke uses 1e-3 as a "same model, no silent corruption"
sanity bar.
"""
from __future__ import annotations

import time

import numpy as np
import onnxruntime as ort

ONNX = "/home/audric/work/umamusume-card-duel/runs/R110-W6-repro/iter-0/policy.onnx"
STATE_DIM = 110
ACTION_DIM = 48
NUM_ZONES = 8
MAX_CARDS_PER_ZONE = 30
CARD_VOCAB_SIZE = 107
N_ACTIONS = 12  # realistic-ish legal action count
SEED = 20260525

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
TIMING_ITERS = 50
WARMUP_ITERS = 5


def make_batch(rng: np.random.Generator, b: int) -> dict[str, np.ndarray]:
    return {
        "state_features": rng.standard_normal((b, STATE_DIM)).astype(np.float32),
        "action_features": rng.standard_normal((b, N_ACTIONS, ACTION_DIM)).astype(np.float32),
        # All actions legal except last position — exercises the masked-softmax path.
        "action_mask": np.concatenate(
            [np.ones((b, N_ACTIONS - 1), dtype=bool), np.zeros((b, 1), dtype=bool)],
            axis=1,
        ),
        "card_ids_by_zone": rng.integers(
            0, CARD_VOCAB_SIZE, size=(b, NUM_ZONES, MAX_CARDS_PER_ZONE), dtype=np.int64
        ),
        "action_card_idx": rng.integers(
            0, CARD_VOCAB_SIZE, size=(b, N_ACTIONS, 2), dtype=np.int64
        ),
    }


def per_row(batch: dict[str, np.ndarray], i: int) -> dict[str, np.ndarray]:
    return {k: v[i : i + 1] for k, v in batch.items()}


def correctness_smoke(session: ort.InferenceSession, label: str) -> None:
    rng = np.random.default_rng(SEED)
    # Build B=64 once; we'll slice into B=1 calls and compare to B in {2,4,...,64}.
    full = make_batch(rng, 64)
    # Per-row B=1 reference outputs.
    ref_logits = np.zeros((64, N_ACTIONS), dtype=np.float32)
    ref_value = np.zeros((64,), dtype=np.float32)
    for i in range(64):
        out = session.run(None, per_row(full, i))
        ref_logits[i] = out[0][0]
        ref_value[i] = out[1][0]

    print(f"[{label}] correctness: B=1 reference rows computed (n=64)")
    worst_logit = 0.0
    worst_value = 0.0
    for b in BATCH_SIZES:
        if b == 1:
            continue
        sub = {k: v[:b] for k, v in full.items()}
        out = session.run(None, sub)
        d_logit = float(np.max(np.abs(out[0] - ref_logits[:b])))
        d_value = float(np.max(np.abs(out[1] - ref_value[:b])))
        worst_logit = max(worst_logit, d_logit)
        worst_value = max(worst_value, d_value)
        ok = "PASS" if (d_logit < 1e-3 and d_value < 1e-3) else "FAIL"
        print(f"[{label}] B={b:>2}: max|Δlogit|={d_logit:.2e}  max|Δvalue|={d_value:.2e}  {ok}")
    print(f"[{label}] correctness worst across all B: |Δlogit|={worst_logit:.2e} |Δvalue|={worst_value:.2e}")


def timing(session: ort.InferenceSession, label: str) -> list[tuple[int, float, float]]:
    rng = np.random.default_rng(SEED + 1)
    rows: list[tuple[int, float, float]] = []
    print(f"\n[{label}] timing ({TIMING_ITERS} iters/cell after {WARMUP_ITERS} warmup):")
    print(f"[{label}] {'B':>3}  {'ms/call':>10}  {'us/state':>10}  {'states/sec':>12}")
    for b in BATCH_SIZES:
        batch = make_batch(rng, b)
        for _ in range(WARMUP_ITERS):
            session.run(None, batch)
        t0 = time.perf_counter()
        for _ in range(TIMING_ITERS):
            session.run(None, batch)
        dt = (time.perf_counter() - t0) / TIMING_ITERS
        ms_call = dt * 1000.0
        us_state = (dt / b) * 1e6
        states_sec = b / dt
        rows.append((b, ms_call, states_sec))
        print(f"[{label}] {b:>3}  {ms_call:>10.3f}  {us_state:>10.1f}  {states_sec:>12.1f}")
    return rows


def main() -> None:
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1

    print("=== CPU ===")
    cpu = ort.InferenceSession(ONNX, sess_options=sess_opts, providers=["CPUExecutionProvider"])
    correctness_smoke(cpu, "cpu")
    cpu_rows = timing(cpu, "cpu")

    print("\n=== CUDA ===")
    try:
        cuda = ort.InferenceSession(
            ONNX,
            sess_options=sess_opts,
            providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
        )
        eps = cuda.get_providers()
        print(f"[cuda] active providers: {eps}")
        if "CUDAExecutionProvider" not in eps:
            print("[cuda] CUDAExecutionProvider not active — skipping CUDA section")
            return
        correctness_smoke(cuda, "cuda")
        cuda_rows = timing(cuda, "cuda")

        print("\n=== speedup CUDA vs CPU (states/sec) ===")
        for (b, _, sps_cpu), (_, _, sps_cuda) in zip(cpu_rows, cuda_rows):
            print(f"  B={b:>3}: cpu={sps_cpu:>8.1f}  cuda={sps_cuda:>8.1f}  cuda/cpu={sps_cuda/sps_cpu:.2f}x")
    except Exception as e:
        print(f"[cuda] failed to load CUDA EP: {e}")


if __name__ == "__main__":
    main()
