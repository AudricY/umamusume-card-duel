# MCTS Selfplay Throughput — Sweep Handoff

- **Date:** 2026-05-25
- **Branch:** `feat/ai`
- **Status:** Slices 3d-3i LANDED. **12.4× wall-clock vs the 0.60 g/s
  baseline anchor** at workers=16 (7.438 g/s peak, 40-game sweep,
  prior=uniform leaf=rollout). All six slices preserve JSONL md5
  `65fc72a1…` and pass 133/133 engine tests. Evidence in
  `docs/ai-research/scoping/r12-selfplay-gate-throughput.md` §
  "Slice 3d-3i". The bigger algorithmic levers (subtree reuse,
  transposition cache, batched leaf eval) were *unlocked* by the user's
  later "drop bit-identity, don't regress strength" relaxation but
  remain unland for this-session-only reasons documented in the
  scoping doc — they need either ONNX staging (for transposition /
  value-head paths) or a head-to-head strength A/B harness (subtree
  reuse) that doesn't exist in this worktree.
- **Predecessor docs (read these in order if cold):**
  - `docs/ai-research/scoping/throughput-optimization-spike.md` — Slice 1–3 (Rust ORT in-process, 4.18× wall)
  - `docs/ai-research/scoping/gpu-inference-execution-provider.md` — G5 lock-free `UnsafeCell<Session>` for `sim-eval-gate` (7.11× at workers=16)
  - `docs/ai-research/scoping/r12-selfplay-gate-throughput.md` — earlier TS-era throughput recipe (this handoff extends as "Slice 3d")
  - `docs/ai-research/scoping/post-throughput-scale-up-directions.md` — what 4.18× unlocked

## Mission

Push `sim-mcts-selfplay` throughput well above the current **0.60 games/s at sims=800** baseline. The 32-core machine sits at ~3% utilisation during selfplay because the Rust binary ignores its own `--workers` flag.

## Baseline anchor

`runs/qhead-v32-highsim-label-stability-probe/selfplay-s800.{log,manifest.json}`:
- sims=800, K=3, rollout_steps=200, prior=policy, leaf=rollout
- model-side=both, seeds-per-side=20 → 40 games
- 66.5 s wall → **0.60 games/s**, single Rust process, workers=1

Other recent points for trend context (all Rust, `leaf=rollout` unless noted):
- sims=200, model-side=player: 2.16 g/s
- sims=100, model-side=player: 3.79 g/s
- sims=100, **leaf=value-head**, player only: 23.07 g/s (different cost shape — value head currently regresses strength per `docs/ai-agent-state/digests/2026-05-25.md:11`)

## Headline finding

**`sim-mcts-selfplay --workers` is a no-op.** `engine-rs/crates/sim-cli/src/bin/mcts_selfplay.rs:540` literally does `let _ = args.workers;` and the comment at `:63-65` admits it. The 50k-row corpus was produced by 8 hand-launched processes as workaround (`docs/ai-agent-state/digests/2026-05-25.md:5`).

The working pattern already exists in `engine-rs/crates/sim-cli/src/bin/eval_gate.rs:540-620`: `std::thread::scope` + atomic task cursor + lock-free shared `InferenceSession` (`engine-rs/crates/engine/src/inference/mod.rs:186-200, 668-682`). G5 measured **7.11× at workers=16** for the gate path; selfplay has the same cost profile (MCTS-dominated, ~7% predict share at batch=1).

## Ranked work plan

### Tier 1 — Infra unlock — LANDED `f75e556` (5.6× at workers=8)

**Port `eval_gate.rs:540-620` worker pool into `mcts_selfplay.rs:580-634`.**

Spec:
- Replace the sequential `for (task_index, (seed, model_side)) in tasks.iter().copied().enumerate()` loop with `std::thread::scope` + atomic cursor (mirror `eval_gate.rs` exactly).
- Share the existing `OnceLock`-backed `InferenceSession`; do not load per-worker.
- Per-task-index outcome/row shards (Vec indexed by `task_index`), concatenated at end to preserve deterministic JSONL row order by `(seed, side, step)`. See `eval_gate.rs:586-594` for the outcomes-indexed pattern.
- Default `--workers 0 ⇒ available_parallelism()` (match eval_gate).
- Required smoke before merge: `--workers 4` vs `--workers 1` bit-identity check. Mirror `training/r12_workstealing_determinism_gate.py`. The seed-keyed per-game RNG (`mcts_selfplay.rs:322, 353`) suggests determinism holds; verify.
- Follow-up cleanup (separate commit): rip the multi-process fanout from `training/r12_orchestrator._run_pool_selfplay` (`:812-962`) once the binary honors `--workers`. The 8-process bash hack in the digest can also be retired.

### Tier 2 — Algorithmic wins (multiplicative with Tier 1)

**Subtree reuse across moves.** `run_mcts` (`engine-rs/crates/engine/src/mcts/driver.rs:35`) is called from scratch every decision (`mcts_selfplay.rs:355-358`); the chosen child subtree (hundreds of visits) is discarded.
- ~80 LOC: extend `run_mcts` API to accept-or-return a root, swap-on-action in the selfplay loop.
- Dirichlet-noise gate: only mix noise once per fresh root (not on the reused subtree root).
- Expected: ~1.5–2× sims-equivalent at fixed wall.

**Transposition cache on `state_hash`.** Card-game states repeat across rollouts (bench reorderings, energy attaches). `predict_policy_and_value` (`driver.rs:418`) re-featurises + re-ORTs each time.
- ~30 LOC `HashMap<u64, (Vec<f64>, f64)>` per `run_mcts` call (or per game).
- `state_hash` already available via `dispatcher::state_hash`.

### Tier 3 — Hot-path micro-opts — LANDED (cumulative ~5x at workers=1, ~2x at workers=16 over Tier 1)

Batch into one commit per `feedback_bigger_fixes_at_once`.

- **`state_hash() -> u128` (not `String`).** Called ~600× per sim × 800 sims = ~480k allocs/decision; currently `format!("{:032x}", …)` over a freshly-allocated `Vec<u8>`. `pack_into(state, &mut buf)` (`engine-rs/crates/engine/src/core/packed.rs:37`) already exists for buffer reuse — wire a thread-local buffer through. Touches `dispatcher.rs:1319`, 6 driver call sites, and any tests asserting hex format.
- **`Rng::fork_idx(u32)` to replace `Rng::fork(&format!("sim{}", sim))`.** Add an integer-mixing variant (`wrapping_mul` + xor) at `engine-rs/crates/engine/src/core/random.rs:69-74`. Hot sites: `driver.rs:104, 151, 169, 207, 583`. Also drop `Rng.label: String` from the hot path (gate behind a debug feature if diagnostics still need it).
- **Drop `forced_coins.clone()` chain** (`driver.rs:469, 515, 517, 671, 673`) — pass by value once into `advance_*_turn_step` or use `SmallVec<[CoinFlipResult; 4]>`.

### Tier 4 — Structural, multi-day (only after Tier 1–3 + flamegraph evidence)

- **Batched leaf evaluation + virtual loss.** Run K parallel PUCT descents (K=8–32), apply `vloss=1` to in-flight edges, batch K leaf observations into one `predict_v3` call (change the hard-coded `(1, …)` shapes at `engine-rs/crates/engine/src/inference/mod.rs:344, 349, 358, 361, 369` to `(K, …)`). Reworks `driver.rs`'s raw-pointer descent (`:108-136`) which today assumes single-threaded mutation. 2–3 days. Highest absolute ceiling, especially if `leaf=value-head` becomes the default.
- **Compact `LegalAiActionLite` for MCTS-internal use.** Drop `String id`, `Vec<f64> features` (48 floats), `serde_json::Value` payload (`engine-rs/crates/engine/src/policy/actions.rs:262-281`). Keep the full struct only for the row recorder. ~3 allocs saved per action × ~8 actions per node × ~500 nodes/decision.
- **Inline `String` fields → interned IDs** (`UmamusumeInstance.species`, `SideState.title`, `used_ability_names_*`) — makes `GameState::clone` close to a memcpy.

## Critical pre-Tier-3 measurement — RESOLVED

samply replaced the broken `cargo flamegraph -> perf` chain (WSL2 ships
no `perf` binary; samply needs `perf_event_paranoid <= 1`). Build the
binary with `CARGO_PROFILE_RELEASE_DEBUG=line-tables-only` so addr2line
can resolve symbols; run `samply record --save-only -o profile.json.gz
--rate 999 …`; parse with the small Python in
`/tmp/profile_inclusive.py` (gist below if you need it). The
`--prior uniform --leaf rollout` measurement showed rollout dominates
(workers=1 baseline went from 0.60 → 0.636 g/s without the policy
prior — ORT call share was ~6%), so Tier 3 micro-opts were the right
branch. Slices 3e-3h delivered.

## Recommended sequencing

1. Tier 1 workers port + determinism smoke. (Task #1)
2. Flamegraph + `prior=uniform` measurement at workers=16. (Task #2)
3. Branch on (2):
   - Rollout-dominated → Tier 3 micro-opts as one commit. (Task #4)
   - NN-dominated → Tier 2 subtree reuse + transposition cache. (Tasks #3, #5)
4. Tier 4 only after Tier 1–3 pushes CPU near saturation at workers=16.

## Determinism / safety notes

- `inference/mod.rs:267-273` pins ORT to `intra_threads=1, inter_threads=1` for FP-determinism parity with `serve_onnx --ort-threads 1`. **Do not loosen** without also planning the parity-smoke recovery.
- ORT `Session::Run` is reentrant; `UnsafeCell<Session>` + `unsafe impl Sync for SessionGuard` is the documented contract (`inference/mod.rs:186-200`). Tier 1 relies on this; Tier 4 batched call also relies on this.
- The per-game RNG is seeded from `(seed, side)` (`mcts_selfplay.rs:322, 353`); multi-worker selfplay should remain bit-identical to single-worker provided the row-shard reassembly is stable-sorted by `(seed, side)`.
- NAPI hot loop is NOT exercised by `sim-mcts-selfplay` (it's pure Rust). `project_napi_vs_rust_rng_order` memory describes the bridge's MCTS→enumerate→advance ordering; that's a TS-bridge concern, irrelevant here.

## Doc write hygiene (per `docs/ai-research/README.md`)

- This handoff is a planning / scoping doc, not a results doc.
- When Tier 1 lands, append the measured speedup to `docs/ai-research/scoping/r12-selfplay-gate-throughput.md` under a new "Slice 3d" section. Do not restate evidence here.
- When the broader sweep finishes, write the analysis under `docs/ai-research/analysis/<topic>.md` and add the result row to `docs/ai-research/progress/r16.md`. Update `docs/ai-research/README.md` routing if a new canonical question appears.
- Keep this handoff doc terse — when Tier 1 lands, mark it done inline and let the new scoping section carry the evidence.

## Open questions / unknowns

- Whether bit-identical determinism holds for `--workers > 1` selfplay with `leaf=rollout` (which uses `with_rng` extensively). Plausible but unverified; the smoke gate above is the resolution.
- Per-call breakdown ORT-forward µs vs rollout-step µs vs featurize µs vs advance µs at sims=800. Flamegraph resolves.
- Whether `serde_json::Value` payloads on `LegalAiAction` are read after MCTS selection or only at row-write time. If only at write, Tier 4 LegalAiActionLite is cheaper than estimated.
- Whether CUDA EP (`inference/mod.rs:275-283`) becomes worth re-evaluating once batched inference (Tier 4) is in place. G4 was falsified for rollout-leaf; batch=K + value-head leaf was never measured.

## Conventions to respect (from auto-memory)

- `project_rust_first_class_default`: Rust is the default engine; `--engine ts` is escape hatch only. Do not regress this.
- `feedback_bigger_fixes_at_once`: batch Tier 3 micro-opts into one commit, not three.
- `feedback_lean_validation`: smoke-then-move-on; do not scope an over-thorough test matrix for the workers port. The `workers=4 vs workers=1` parity check is sufficient.
- `feedback_infra_first_compounding_payoff`: Tier 1 is exactly this pattern — it unlocks every downstream measurement and every other tier.
- `feedback_commit_regularly`: in a `/loop` session, commit at workstream checkpoints (e.g. after Tier 1, after flamegraph, after Tier 3 commit).
- `feedback_assume_approval`: describe-and-do in the same turn for low-risk in-repo edits.

## Key file paths (absolute)

- `engine-rs/crates/sim-cli/src/bin/mcts_selfplay.rs` — target binary; line 540 ignores `--workers`; sequential loop 580-634
- `engine-rs/crates/sim-cli/src/bin/eval_gate.rs` — reference parallel implementation, lines 540-620
- `engine-rs/crates/engine/src/inference/mod.rs` — shared `OnceLock` session (186-200, 668-682); batch-1 hard-coded shapes (344-369); ORT thread pin (267-273)
- `engine-rs/crates/engine/src/mcts/driver.rs` — `run_mcts` (35), PUCT select (117), expansion (298), backprop (222-227), action choice (272-283), rollout (634), raw-ptr path (108-136)
- `engine-rs/crates/engine/src/mcts/node.rs` — `cached_leaf_value` (30)
- `engine-rs/crates/engine/src/dispatcher.rs` — `state_hash` (1319)
- `engine-rs/crates/engine/src/core/{state,packed,random}.rs` — clone shape, `pack_into` (packed.rs:37), `Rng::fork` (random.rs:69-74)
- `engine-rs/crates/engine/src/policy/{actions,observation,types}.rs` — `LegalAiAction` allocations (actions.rs:262-281)
- `engine-rs/crates/engine/benches/clone_and_fingerprint.rs` — extend here for a `state_hash` bench
- `training/r12_orchestrator.py` — `--workers` plumbing (546, 1135, 1587), `_run_pool_selfplay` (812-962), engine resolution (408-444)
- `training/r12_workstealing_determinism_gate.py` — parity-smoke pattern to mirror
- `runs/qhead-v32-highsim-label-stability-probe/selfplay-s800.{log,manifest.json}` — baseline anchor
- `docs/ai-agent-state/digests/2026-05-22.md`, `2026-05-25.md` — G5 result; 8-process workaround note
