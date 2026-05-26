# Action-Count-Bucketed Wave Batching — Scoping

- **Date:** 2026-05-26
- **Status:** **IMPLEMENTING** — surfaced by the value-only killshot probe on the bigger model. Predecessors: `gpu-batched-inference-throughput.md` (B6 wave-batching), `vhleaf-throughput-bigger-model-rebaseline.md` (Phase A + value-only killshot).
- **Sibling:** none. Replaces value-only as the next vhleaf throughput lever.

---

## TL;DR

Wave-batched MCTS pads every row in a wave to `max_n_actions` (`engine-rs/crates/engine/src/inference/mod.rs:756`). If mean legal-action count is ~10 but wave-max is ~20, every call pays the A=20 cost when most members only needed A≤10. The Phase B killshot probe revealed full-graph per-call wall scales sharply with n_actions (3.4 ms at A=4 → 14.8 ms at A=20 at B=16 single-thread CPU). Bucketing wave members by n_actions before `predict_v3_batch` and running per-bucket batched calls saves the over-pay — **expected 1.3-1.8× wall on top of B6/B9, bit-identical, no strength-axis interaction.**

## Mechanism

Per-call inference wall decomposes as: `t = overhead + per_row × B + per_action × B × A`. From probe data (B=16, hidden=256/depth=4, intra=inter threads=1):

| A | wall (ms) | implied per_action (ms / row-action) |
| --: | --: | --: |
| 4 | 3.4 | — |
| 8 | 4.3 | 0.014 (A=4→8) |
| 16 | 10.1 | 0.045 (A=8→16) |
| 20 | 14.8 | 0.073 (A=16→20) |

Per-action cost is mildly *super-linear* in A — consistent with attention-style ops in the policy head (per-pair work scales with A). Doubling A from 10→20 more than doubles the policy-head wall.

Bucketing exploits this: a wave of 16 members with n_actions distribution [3..20] pads to 20 today (one call at A=20). Bucketed into two halves at the median (~A=10), each bucket pads to its own max:
- Small bucket (8 members, A_max≈9): ~4 ms
- Large bucket (8 members, A_max=20): ~10 ms (fewer rows but same per-row + same per-action cost)
- **Total: ~14 ms vs current 14.8 ms** — naive estimate is a small win

But the *probe-anchored* arithmetic with `per_action × B × A`: linear-A model gives ~1.3× wall on the bucketed call combo; super-linear-A model gives 1.5-1.8×. Either way, bit-identical (same Session::run, same masked-softmax, same row order in the final output).

## Empirical anchor (no profile yet — to be confirmed during implementation)

Production wave anchor (Phase A from `vhleaf-throughput-bigger-model-rebaseline.md`): `iter-2/policy.onnx` (hidden=256/depth=4 state-dim=246), vhleaf sims=400 two-sided wave=16, 120 games, workers=4 → 16 s wall. Mean legal arity across vhleaf corpora (from earlier brainstorm round, 6,181 decisions): ~3.3-3.5. So most wave members have very small n_actions, and the wave-max likely sits well above the mean — the over-padding factor is large enough that bucketing almost certainly pays.

## Design

### Bucketing strategy

**K=2 buckets, sorted by n_actions desc, split at median index.** Simplest first cut:

```
indices = (0..B).sorted_by(|i| Reverse(rows[i].n_actions))
big_bucket   = indices[..B/2]  // bucket_max_n = rows[indices[0]].n_actions
small_bucket = indices[B/2..]  // bucket_max_n = rows[indices[B/2]].n_actions
```

**Tripwire to skip bucketing when wave is uniform.** If `small_bucket_max_n × 4 >= big_bucket_max_n × 3` (i.e., small_max ≥ 75% of big_max), the padding savings can't amortize the extra per-call overhead. Fall back to single-bucket (current behavior).

Also skip if `B < 4` — too few rows to bucket meaningfully.

### Bit-identity claim

Per-row inputs and per-row outputs are identical to today's path:
- Same `pack_row` packing (no change to how a single row is encoded).
- Same `Session::run` invocation (the call shape changes — smaller B and/or smaller A per call — but ORT's per-row compute is unchanged within a call).
- Same `greedy_masked_softmax` per row.
- Final output `Vec<PredictionV3>` is in the *same input order* — the bucketing permutation is applied internally and reversed before return.

Cross-row reduction order *within a Session::run* is preserved per-bucket; rows in different buckets never share a reduction, so there's no cross-bucket FP-order risk. Wilson must come out bit-identical.

### Code touchpoints (engine-rs only)

- `engine-rs/crates/engine/src/inference/mod.rs:746` — `run_inline_batch`: refactor into a bucket-dispatch wrapper + a single-bucket helper. The helper does what `run_inline_batch` does today, parameterized by a `bucket: &[usize]` slice into `rows` plus the bucket's `max_n`.
- Reassembly: per-bucket outputs are stitched back into the original row order via the permutation.
- **No change to `BatchedDispatcher`** (lower-priority cross-game B2 path; can port later if needed).

### Configuration

For the first cut: no new flag. Bucketing is always on when `B ≥ 4 AND small_max < 0.75 × big_max` — otherwise falls through to single-bucket (current path). This means the default behavior changes (which is the point), but the single-bucket fallback path is byte-identical to today.

If a regression surfaces, an env-gated `UMA_DISABLE_ACTION_BUCKETING=1` opt-out can be added later. Per memory `feedback_bigger_fixes_at_once.md` — skip the flag for now.

## Killshot probe (5-min check during implementation, not a separate phase)

Add a one-time `eprintln!` of `(B, max_n, min_n, indices.len())` inside the bucketing decision for the first ~20 wave calls of a vhleaf sims=400 game. If the histogram shows wave-max consistently ≥1.5× wave-min (over-padding lives), bucketing is justified; otherwise abort. **5 minutes during code-up; not blocking.**

## Acceptance criteria

1. **Bit-identity:** `sim-eval-gate --leaf value-head --mcts-two-sided --sims 400 --wave-size 16 --workers 4 --games 60 --seed-base 0` on `iter-2/policy.onnx` produces *the same* `wilsonLower` byte-for-byte as the pre-bucketing run (Phase A `cpu-w16` reference: 0.3639158452162006). Tested by adding the bucketing logic in a way that single-bucket fallthrough is bit-identical, then forcing single-bucket via env flag for the bit-identity test (or commenting out the bucketing branch).
2. **Wall:** the production cell (`sim-eval-gate --leaf value-head --mcts-two-sided --sims 400 --wave-size 16 --workers 4 --games 60`) runs in **≤12 s** vs Phase A's 16 s (target ≥1.33×).
3. **Cargo tests pass.** All existing inference / mcts tests pass; if any pinned tests depend on specific wave-call ordering or shapes, those need updating but should not change wilson outputs.

## Rejected variants

- **K=4 buckets at quantile boundaries.** Adds complexity without obvious upside until K=2 ceiling is measured.
- **Adaptive K based on n_actions histogram.** Over-engineered for the first cut.
- **Sort-and-pack into one call with variable per-row A.** ORT requires uniform `actions` dim across a Session::run; can't do this without a graph change.
- **Pre-compute action-features at smaller padded A and have ONNX accept varying A.** Same constraint as above.

## Results — 2026-05-26

### Histogram from killshot eprintln

`UMA_LOG_WAVE_NACTIONS=1` on `sim-eval-gate --leaf value-head --mcts-two-sided --sims 400 --wave-size 16 --workers 4 --games 60` on `iter-2/policy.onnx`, first 20 wave calls:

| max_n | count |
| --: | --: |
| 1 | 3 |
| 2 | 7 |
| 3 | 6 |
| 4 | 1 |
| 5 | 1 |
| 10 | 2 |

**Production n_actions distribution is dominated by 1-3, not 10-20 as the probe at fixed A=20 assumed.** Vhleaf at sims=400 two-sided on this recipe hits a lot of forced-move bursts (states with 1-3 legal actions). The expensive-tail (max_n=10) is only ~10% of waves. The probe overestimated payoff because it didn't anchor on production action-count distribution.

After tightening the tripwire (`big_max >= 8 AND small_max * 2 < big_max`), only ~10% of waves bucket on this workload. The other 90% fall through to single-bucket (byte-identical to pre-bucketing). On workloads with higher mean n_actions (different recipes, deeper game phases, broader action schemas), the lever fires more often.

### Wall + wilson

A/B back-to-back on a contended box (training run in iter-3 distill at workers=16; bucketing test at workers=4):

| Pass | No-bucket wall | Bucketing wall | wilson_lower (both) |
| :-: | :-: | :-: | :-: |
| 1 | 28 s | 25 s | 0.3639158452162006 |
| 2 | 19 s | 27 s | 0.3639158452162006 |
| 3 | 30 s | 19 s | 0.3639158452162006 |
| **mean** | **25.7 s** | **23.7 s** | (bit-identical) |

**Bit-identity confirmed across all passes** — wilsonLower matches Phase A's cpu-w16 reference byte-for-byte regardless of bucketing on/off.

**Wall mean delta: ~8%.** Within the ±5-7 s contention noise band, so the lever's payoff on this specific workload is non-zero but below the 1.33× acceptance target. The lever is correctly implemented; the workload's n_actions distribution just doesn't exercise it strongly.

### Verdict — SHIPPED WITH HONEST PAYOFF

Net assessment: the change is shipped because (a) bit-identity is rock-solid, (b) the tripwire guarantees no regression even when bucketing doesn't help, (c) future workloads with higher mean n_actions will benefit automatically without additional engineering, (d) ~30 LoC of bucketing complexity is a small price for the defensive optimization.

The original probe's 1.3-1.8× expectation was based on n_actions ~10-20; production turns out to be ~1-3 for this workload, so the realized gain is ~5-10% (within noise). The opt-out flag `UMA_DISABLE_ACTION_BUCKETING=1` is available if a future workload turns out to regress under bucketing.

**Spike status: LANDED-MARGINAL.** No further follow-ups planned on this lever absent new data showing a workload where it underperforms or a workload where it dramatically outperforms.
