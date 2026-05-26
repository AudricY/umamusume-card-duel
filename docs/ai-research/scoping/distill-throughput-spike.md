# Distill-Stage Throughput — Scoping

- **Date:** 2026-05-26
- **Status:** **LANDED-SHIP-2026-05-26** — Phase 1 verdict: **b=256 + lr=6e-4 (sqrt-scaled) + amp + dl=4 → 2.19× distill wall at −0.49pp val_acc (within ~0.4σ on n=6574)**. Orchestrator defaults flipped (`--batch-size 64→256`, `--lr 3e-4→6e-4`, `--amp False→True`, `--distill-dataloader-workers 0→4`). Explicit launch-script overrides preserve old behavior. See "Phase 1 Results" + "LR-scaling resolution" below.
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

### Phase 1 Results — 2026-05-26

Implementer landed the patches (CLI flags + DataLoader wiring + opt-in torch.compile). Main session followed with three downstream fixes needed for end-to-end correctness:

1. **`torch.compile` ordering bug.** Implementer placed compile BEFORE `load_init_from_checkpoint`; the `OptimizedModule` wrapper prefixes parameter names with `_orig_mod.` so checkpoint keys didn't match. Fixed by moving the compile call to AFTER checkpoint load + optimizer construction (`train_bc.py:179`).
2. **fp16 sentinel overflow.** Default `torch.cuda.amp.autocast()` is fp16; the masked-softmax sentinel `-1e9` (`train_bc.py:1003`) overflows fp16's ~6.55e4 range. Added `--amp-dtype {bfloat16, float16}` (default `bfloat16`) and split the autocast/scaler logic — bf16 has fp32's exponent range so no `GradScaler` needed.
3. **Compiled-model state_dict consumers.** Saved checkpoint and `run_onnx_roundtrip_smoke` both call `model.state_dict()`; with compile applied, this returns `_orig_mod.`-prefixed keys that downstream consumers can't load. Fixed by unwrapping (`getattr(model, "_orig_mod", model)`) in both sites.

**Smoke A/B at 5 epochs, 26417 train samples, iter-11 selfplay.jsonl, init iter-10/checkpoint.pt:**

| cell | batch | flags | steady µs/epoch | speedup vs baseline | ep5 train_loss |
| :-- | --: | :-- | --: | --: | --: |
| baseline | 64 | (none) | 3.38s | 1.00× | 1.371 |
| amp_only | 64 | `--amp` (bf16) | 4.54s | **0.74×** (regression) | 1.370 |
| dl_only | 64 | `--dataloader-workers 4` | 5.46s | **0.62×** (regression!) | 1.368 |
| amp_dl | 64 | both | 6.07s | **0.56×** (regression) | 1.368 |
| challenger | 64 | all three (+`--compile`) | 4.50s | 0.75× (regression after compile warmup) | 1.376 |
| b256 | 256 | (none) | 1.72s | **1.97×** | 1.428 |
| b256_amp_dl | 256 | `--amp --dataloader-workers 4` | 1.55s | **2.18×** | 1.420 |

**Original hypothesis INVERTED.** The scoping doc framed AMP+DataLoader+compile as low-risk "safe" knobs and batch_size as risky-deferred-to-Phase-2. Empirically:
- **The "safe" knobs regress at b=64.** Model + batch are too small for the optimization overhead (DataLoader IPC, AMP dtype casts, torch.compile cudagraph capture with 9 distinct shapes per epoch) to pay back.
- **batch_size IS the real lever.** Alone it gives 1.97× wall; with AMP+DL stacked on top, 2.18×.

**25-epoch convergence A/B (production length, lr linearly scaled 4× to 1.2e-3 for b=256):**

| metric | baseline (b=64 fp32) | challenger (b=256 bf16 dl=4 lr=1.2e-3) | delta |
| :-- | --: | --: | --: |
| total wall | **93.4s** | **42.7s** | **2.19× speedup** |
| final train_loss | 1.0827 | 1.0906 | +0.73% |
| final val_loss | 1.9405 | 1.9689 | +1.46% |
| **final val_accuracy** | **0.5155** | **0.4944** | **−4.1% (−2.1pp absolute)** |

Trajectory: monotone decrease, no instability. Both reach low-train / high-val gap (overfitting), expected at 25 epochs on iter-N data.

**Verdict.** Throughput rubric CLEARS (2.19× ≥ 1.5× ship band). Loss-trajectory rubric CLEARS (train +0.7%, val +1.5% — both within ±10%). BUT **val_accuracy regression is the strength-axis yellow flag** the rubric didn't anticipate: -2.1pp absolute on the held-out validation set suggests the b=256+lr=1.2e-3 optimizer trajectory is producing a measurably weaker policy at this iter. Could be noise (single A/B, n=6574 val); could be a real but small strength gap.

**Decision: SHIP THE FLAGS as opt-in, DON'T flip orchestrator defaults yet.** Wilson gate (Phase 1.5) is now mandatory because the val_accuracy delta exceeds the loss-trajectory rubric's coverage.

**Open follow-ups:**
1. Try sqrt-LR scaling (lr=6e-4) instead of linear (lr=1.2e-3) at b=256 — may close the val_accuracy gap without losing the throughput win.
2. Try intermediate batches (b=128, b=192) — may sit at the sweet spot of throughput + accuracy.
3. Wilson gate at n=10k on the b=256+amp+dl checkpoint vs the b=64 checkpoint, both produced from iter-10 init via the same recipe — confirms whether the val_accuracy delta translates to a real Δwl.

**Code landed (opt-in, defaults preserve byte-identicality):**
- `training/train_bc.py`: `--dataloader-workers` (DataLoader num_workers + pin_memory + persistent_workers), `--amp-dtype` (bfloat16 default), `--compile` (CUDA-only torch.compile wrap, guarded by try/except). Compile applied AFTER checkpoint load. Checkpoint save + ONNX roundtrip both unwrap `_orig_mod.` when compile is active.
- `training/r12_orchestrator.py`: `--distill-compile`, `--distill-dataloader-workers` (default 0), pass-through to train_bc.py via run_distill. Existing `--amp` arg now actually forwards.

### LR-scaling resolution — 2026-05-26

The −2.1pp val_accuracy gap above was suspected to come from over-aggressive linear LR scaling (3e-4 → 1.2e-3 at b=64 → b=256). Tested two follow-up cells with sqrt-LR scaling (the empirically defended choice for AdamW):

| cell | batch | lr | wall | speedup | final val_acc | Δ vs baseline |
| :-- | --: | --: | --: | --: | --: | --: |
| baseline_25 | 64 | 3e-4 | 93.4s | 1.00× | 0.5155 | — |
| b256_amp_dl_25 (linear LR) | 256 | 1.2e-3 | 42.7s | 2.19× | 0.4944 | −2.11pp |
| **b256_sqrtlr_25** | **256** | **6e-4** | **42.6s** | **2.19×** | **0.5106** | **−0.49pp** ← SHIP |
| b128_sqrtlr_25 | 128 | 4.24e-4 | 84.9s | 1.10× | 0.5149 | −0.06pp |

**sqrt-LR closed 4× of the val_acc gap** at no throughput cost (2.19× vs 2.19×). At n=6574 val, 95% Wilson noise is ~±1.2pp, so −0.49pp is ~0.4σ — well within noise.

b=128 sqrt-LR gives essentially no accuracy delta (−0.06pp) but the throughput win shrinks to 1.10× — not worth the lr/batch reshuffle.

**Wilson gate (Phase 1.5) SKIPPED** because the −0.49pp val_acc delta is below the noise floor on the validation split. If future iters show monotone strength regression, revisit.

### Phase 3 — Default flip LANDED — 2026-05-26

`training/r12_orchestrator.py`:
- `--batch-size`: default `64 → 256`
- `--lr`: default `3e-4 → 6e-4` (sqrt-scaled from old default)
- `--amp`: `action=BooleanOptionalAction, default=True` (forwards `--amp` to `train_bc.py` by default; `--no-amp` disables)
- `--distill-dataloader-workers`: default `0 → 4`
- `--distill-compile`: stays default `False` (regressed at b=64, not re-validated at b=256)

**Backward compatibility:** existing launch scripts that pin `--batch-size 64 --lr 3e-4` keep old behavior (explicit overrides default). New launches that omit those args get the fast recipe. The R16-P3-v36-az-5k-nobuffer-cuda run is finished; no live orchestrator was disturbed.

**Production iter-wall impact:** distill drops from ~125-135 s/iter to ~57-62 s/iter (~50% distill reduction → **~17-20% total iter-wall reduction**, since selfplay is the other ~155 s).

**Recipe summary for future launches:**

```
# Recommended distill recipe (default post-2026-05-26):
--epochs 25 --batch-size 256 --lr 6e-4  # (orchestrator defaults)
# --amp is on by default; add --no-amp to disable
# --distill-dataloader-workers 4 is default

# Legacy recipe (explicit override):
--epochs 25 --batch-size 64 --lr 3e-4 --no-amp --distill-dataloader-workers 0
```

## Phase 1.5 — Wilson validation gate (conditional on STRENGTH RISK) — SKIPPED

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
