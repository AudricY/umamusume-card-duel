# Distill-Stage Throughput — Scoping

- **Date:** 2026-05-26
- **Status:** **SCOPING** — pre-registered, not launched.
- **Predecessors:**
  - `cuda-wave-sweep-validation.md` (LANDED-SHIP-STRENGTH-NEUTRAL: cuda-w256 + torch CUDA upgrade landed; explicit note at L110 "the iter-wall bottleneck on the bigger model was distill, not MCTS").
  - `cross-game-dispatcher-selfplay.md` (KILLSHOT-FALSIFIED: MCTS-side throughput ceiling reached at hidden=256/depth=4; forward implications pointed to distill as the next iter-wall lever).
- **Sibling:** none.

---

## TL;DR

Distill currently runs at ~125-135 s/iter (25 epochs × ~4.3 s/epoch, 32,991 samples per iter, batch_size=64) on the `R16-P3-v36-az-5k-nobuffer-cuda` recipe. Distill is ~45% of total iter-wall (selfplay ~155 s, distill ~130 s). GPU util during distill: **~38%** (observed live 2026-05-26). On an RTX 5000 Ada (compute_cap 8.9, 16 GB VRAM, tensor cores), this is dramatically under-utilized for a hidden=256/depth=4 model with batch=64 FP32.

Three knobs are off in `training/train_bc.py`:

1. **`--amp` flag never passed** by `r12_orchestrator.run_distill`. Training is full FP32. BF16 on Ada with tensor cores: **realistic 1.6-2.2×** at this model size.
2. **`DataLoader` defaults**: `num_workers=0, pin_memory=False, persistent_workers=False`. Data loading blocks GPU compute. Set `num_workers=4, pin_memory=True, persistent_workers=True`: **realistic 1.1-1.4×**.
3. **`batch_size=64`** is small for Ada at hidden=256. Larger batch (256-512) fits in 16 GB VRAM. **Realistic 1.3-1.8×** but changes optimizer dynamics — defer to Phase 2.

**Combined Phase 1 (knobs 1+2 + torch.compile bonus): 2-3× distill throughput**, saving ~60-85 s/iter (~20-30% off total iter-wall).

## Hypothesis

> Enabling AMP (bfloat16), async DataLoader (num_workers=4, pin_memory, persistent_workers), and torch.compile in `train_bc.py` will improve distill throughput by **≥1.5×** at iso-loss-trajectory (per-epoch train_loss within ±10% of baseline) and iso-wilson (Δwl < 0.02 at n=10k) on the current production recipe (hidden=256/depth=4, batch=64, lr=3e-4, 25 epochs).

## Phase 1 — Infra knobs (DataLoader + AMP + torch.compile)

**Changes in `training/train_bc.py`:**

1. **DataLoader** at L81 and L87: add `num_workers=4, pin_memory=True, persistent_workers=True` to both `train_loader` and `val_loader`. Defaults stay opt-in only via CLI flag for safety — but in practice we want this on by default for the orchestrator path.
2. **AMP**: no code change (`use_amp` already wired at L129-130). `r12_orchestrator.run_distill` at `r12_orchestrator.py:~1012-1075` needs to pass `--amp` to the subprocess by default (gated on `args.amp` which currently defaults to False).
3. **torch.compile**: wrap the model with `torch.compile(model, mode="reduce-overhead")` after construction. Gated on a new `--compile` flag, default True when `device.type == "cuda"`.

**Smoke recipe (no orchestrator changes yet).** Use an existing selfplay.jsonl as input, run train_bc.py twice with identical seed:

```
# Baseline (current settings)
python training/train_bc.py \
  --data runs/R16-P3-v36-az-5k-nobuffer-cuda/iter-11/selfplay.jsonl \
  --out-dir /tmp/distill-baseline --epochs 5 --batch-size 64 \
  --hidden-dim 256 --depth 4 --state-dim 246 --lr 3e-4 \
  --value-weight 1.0 --policy-weight 1.0 \
  --data-mode mcts-distill --split-by seed \
  --init-from-checkpoint runs/R16-P3-v36-az-5k-nobuffer-cuda/iter-10/checkpoint.pt \
  --seed 0 --verbose

# Challenger (all three knobs)
python training/train_bc.py \
  ...same args... \
  --amp --compile --dataloader-workers 4

# Measure: per-epoch wall, GPU util, train_loss trajectory
```

5 epochs is enough to detect any per-epoch wall change with cuDNN warmed up; bigger sample sizes are gated on the smoke clearing.

**Decision rubric.**

| Throughput delta | Loss trajectory similarity | Verdict |
| :-- | :-- | :-- |
| ≥1.5× | per-epoch train_loss within ±10% | **SHIP** — flip orchestrator defaults |
| 1.2-1.5× | within ±10% | **SHIP-with-acknowledgement** — flip with documented expectation |
| 1.0-1.2× | within ±10% | **PARTIAL** — instrument GPU util + investigate which knob underperformed |
| <1.0× | any | **REGRESSION** — investigate root cause |
| any | >10% loss-trajectory divergence | **STRENGTH RISK** — fall back to Phase 1.5 (Wilson n=10k gate before ship) |

**Effort.** `implementer`, ~2 hours code + 15 min smoke run.

## Phase 1.5 — Wilson validation gate (conditional on STRENGTH RISK)

Only if Phase 1 shows loss-trajectory divergence >10%. Run two full iters end-to-end (baseline vs challenger), gate the resulting checkpoints at n=10k with the existing tight-gate harness.

Acceptance bands (mirroring `cuda-wave-sweep-validation.md` Phase 2):

| Δwl_challenger_vs_baseline at n=10k | Verdict |
| :-- | :-- |
| \|Δwl\| < 0.02 | **strength-neutral** — ship |
| 0.02-0.05 | **measurable drift, in envelope** — ship with explicit documentation |
| 0.05-0.08 | **envelope breach** — stop, tune AMP/compile config |
| ≥0.08 | **regression** — reject AMP, ship DataLoader-only |

Skipped if Phase 1 loss-trajectory smoke is clean (within ±10%).

## Phase 2 — batch_size sweep (deferred, conditional on Phase 1 ship)

`batch_size=64` is small for Ada at hidden=256. Sweep at {128, 256, 512} with proportional LR scaling (`lr = 3e-4 × (batch/64)^0.5` per square-root rule, or `× (batch/64)` per linear rule — pick after smoking). Likely needs Wilson validation because optimizer dynamics shift.

Deferred to a separate phase because:
- Phase 1 has higher confidence + lower risk (pure infra, no training-dynamics change)
- batch_size is the riskiest knob (changes effective LR, gradient noise scale)
- Composes multiplicatively with Phase 1 — if Phase 1 lands cleanly, Phase 2 can stack on top

**Effort.** Half-day spike + Wilson gate.

## Phase 3 — Default flip in `r12_orchestrator.py` (conditional on Phase 1 ship)

If Phase 1 (and 1.5 if triggered) ship:

1. Flip `--amp` default in `r12_orchestrator.py` argparse to `True` (currently `action="store_true"` defaults to False at L1654). OR keep CLI default False but unconditionally add `--amp` to the `run_distill` cmd construction at `r12_orchestrator.py:~1012-1075`. The latter is cleaner (orchestrator-managed default vs CLI default).
2. Add `--compile` (similar pattern, default True).
3. DataLoader settings are train_bc.py-side defaults (already flipped in Phase 1 code change).
4. Single iter end-to-end on the in-flight orchestrator's resume to confirm no regression.

## Filing

- **Scoping doc:** this file.
- **Run dir:** `runs/distill-throughput-spike/` (created at Phase 1 smoke launch).
- **Queue entry:** `distill-throughput-spike` (added at scoping commit).
- **Progress writeup target:** new section in `docs/ai-research/progress/r16.md` at LANDED time.

## Notes

- **Risk-ordered knobs:** DataLoader (lowest risk, pure infra) → torch.compile (low risk, no math change) → AMP (low risk on Ada with BF16) → batch_size (medium risk, optimizer-affecting).
- **The orchestrator is currently live** on `R16-P3-v36-az-5k-nobuffer-cuda` (iter-12+). Smoke + wire-up must NOT disturb the running training. Smoke goes through standalone `train_bc.py` invocations on existing selfplay.jsonl data; default flip in `r12_orchestrator.py` only affects NEXT launches, not the running process.
- **Determinism:** BF16 introduces non-deterministic FP reduction order (similar to the cuda-w256 drift documented at `cuda-wave-sweep-validation.md`). Per-iter checkpoints will not be byte-identical to FP32 baseline, but loss trajectory should be — and that's what the rubric tests.
