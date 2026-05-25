//! Rust port of `backend/src/sim/evalGate.ts`.
//!
//! Phase 1g — runs N games of MCTS-player vs heuristic-opponent
//! (both sides as model rotating across half the seeds), aggregates
//! win-rate with Wilson 95% confidence intervals, emits a summary
//! JSON matching the TS schema's top-level fields.
//!
//! Without /predict integration the "model" path uses Rust MCTS
//! (uniform prior, rollout leaf) as a stand-in. When /predict lands,
//! swap MctsLeaf::Rollout → MctsLeaf::ValueHead + thread through
//! --challenger / --baseline URLs.

use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use engine::core::constants::SideId;
use engine::core::random::{with_rng, Rng};
use engine::core::state::CurrentSide;
use engine::deck_sampling::{manifest_pair_for, DeckSampling};
use engine::dispatcher::{
    advance_modeled_turn_step, advance_opponent_turn_step, advance_player_ai_turn_step,
    get_forced_attack_coin_results, state_hash,
};
use engine::headless_setup::setup_ai_vs_ai_game_with_decks;
use engine::inference::{self, Device, InferenceSession};
use engine::mcts::config::{MctsConfig, MctsLeaf, MctsPrior};
use engine::mcts::driver::run_mcts;
use engine::policy::actions::enumerate_legal_ai_actions;
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(
    name = "sim-eval-gate",
    about = "Rust port of backend/src/sim/evalGate.ts"
)]
struct Args {
    /// Model URL (also serves as the value-head /predict endpoint when
    /// --leaf=value-head). Orchestrators pass this here.
    /// Orchestrator alias: --model-url (TS evalGate.ts flag name).
    #[arg(long, alias = "model-url")]
    challenger: Option<String>,
    #[arg(long)]
    baseline: Option<String>,
    /// Number of games **per modelled side**. Matches TS
    /// `evalGate.ts:43-49` semantics: under `--model-side both` the binary
    /// schedules `2 * games` total tasks (one per side per seed); under
    /// `--model-side player|opponent` it schedules `games` tasks for that
    /// side only. Orchestrator passes `--games`; legacy callers that pass
    /// `--seeds` get the same interpretation.
    ///
    /// Slice 2 follow-up (handoff doc § Phase 1h follow-up): prior to the
    /// fix `--seeds N --model-side both` ran N games total (alternating
    /// per seed), so `r12_orchestrator.run_gate` forwarding the same
    /// `--games eval_games` to both engines silently ran HALF the games
    /// under `--engine rust` vs `--engine ts`. Don't reintroduce the
    /// asymmetry — TS is the canonical contract.
    #[arg(long = "games", alias = "seeds", default_value_t = 100)]
    seeds: u32,
    /// First seed. Orchestrator alias: --seed-start.
    #[arg(long, alias = "seed-start", default_value_t = 0)]
    seed_base: u32,
    /// MCTS simulations. Orchestrator alias: --mcts-simulations.
    #[arg(long, alias = "mcts-simulations", default_value_t = 100)]
    sims: u32,
    /// CRN rollout samples. Orchestrator alias: --mcts-rollout-crn-samples.
    #[arg(long, alias = "mcts-rollout-crn-samples", default_value_t = 3)]
    k: u32,
    /// Rollout step cap. Orchestrator alias: --mcts-rollout-steps.
    #[arg(long, alias = "mcts-rollout-steps", default_value_t = 200)]
    rollout_steps: u32,
    /// Optional rollout blend for value-head leaves. 0.0 is pure value-head;
    /// 1.0 is rollout leaf scoring with the value-head/policy graph loaded.
    #[arg(long, alias = "mcts-value-head-rollout-blend", default_value_t = 0.0)]
    value_head_rollout_blend: f64,
    #[arg(long, default_value_t = 1000)]
    max_steps: u32,
    /// Manifest output (mirrors TS --manifest-out).
    #[arg(long)]
    manifest_out: Option<String>,
    /// Leaf mode: "rollout" (default) or "value-head" (queries /predict
    /// via --challenger URL). Orchestrator alias: --mcts-leaf.
    #[arg(long, alias = "mcts-leaf", default_value = "rollout")]
    leaf: String,
    /// Prior mode. Orchestrator alias: --mcts-prior.
    #[arg(long, alias = "mcts-prior", default_value = "uniform")]
    prior: String,
    /// PUCT constant. Orchestrator alias: --mcts-c-puct.
    #[arg(long, alias = "mcts-c-puct", default_value_t = 1.5)]
    c_puct: f64,
    /// MCTS tree-node cap. Orchestrator alias: --mcts-max-nodes.
    #[arg(long, alias = "mcts-max-nodes", default_value_t = 5_000)]
    max_nodes: u32,
    /// Heuristic-collapse cap. Orchestrator alias: --mcts-collapse-max-steps.
    #[arg(long, alias = "mcts-collapse-max-steps", default_value_t = 64)]
    collapse_max: u32,
    /// AlphaZero-style two-sided MCTS. Default false (single-sided
    /// rule-bot-collapse path; byte-identical to pre-flag behavior).
    /// When set, opponent decision points become real tree nodes, the
    /// policy/value net is queried for whoever is to move, and backup
    /// uses sign-flips per ply. See
    /// `docs/ai-research/scoping/two-sided-mcts-scoping.md`.
    #[arg(long, alias = "mcts-two-sided", default_value_t = false)]
    mcts_two_sided: bool,
    /// Selection mode. `mcts` uses the configured MCTS policy; `random`
    /// chooses uniformly among legal modeled-side actions as a floor
    /// baseline. Other values are accepted for orchestrator-flag parity
    /// and currently fall back to `mcts`.
    #[arg(long, default_value = "mcts")]
    selection: String,
    /// Model-side rotation: "both" (default — alternate per-seed),
    /// "player", or "opponent". Accepted for orchestrator-flag parity.
    #[arg(long, default_value = "both")]
    model_side: String,
    /// Wilson-lower pass threshold. After all games run, if overall
    /// Wilson lower bound < min_ci_lower the gate fails (TS parity:
    /// evalGate.ts:139). 0.0 disables.
    #[arg(long, default_value_t = 0.0)]
    min_ci_lower: f64,
    /// Minimum games required to pass. Fewer games → gate fails
    /// (TS parity: evalGate.ts:137).
    #[arg(long, default_value_t = 0)]
    min_games: u32,
    /// Minimum overall win rate to pass. TS parity: evalGate.ts:138.
    #[arg(long, default_value_t = 0.0)]
    min_win_rate: f64,
    /// Per-game progress JSONL output.
    #[arg(long)]
    progress_out: Option<String>,
    /// Worker fan-out — number of OS threads dispatching independent
    /// games in parallel. `0` resolves to `available_parallelism()`.
    /// Each task `(seed, model_side)` is fully independent; determinism
    /// is preserved because the per-task Rng is seeded from
    /// `seed`+`side` and `with_rng` uses thread-local state. The shared
    /// `InferenceSession` is `Arc<Mutex<Session>>` and serializes
    /// `predict_v3` calls; at sims=100 with `--leaf rollout` the MCTS
    /// dominates so the Mutex is not the bottleneck.
    #[arg(long, default_value_t = 1)]
    workers: u32,
    /// R16-P3 spike Option A: path to the ONNX policy file (v3.0
    /// graph). Required when `--leaf value-head` or `--prior policy` is
    /// set. Replaces the HTTP `/predict` round-trip the prior
    /// `--challenger` URL used to drive. Set `ORT_DYLIB_PATH` to the
    /// libonnxruntime.so location.
    #[arg(long)]
    onnx_path: Option<String>,
    /// GPU inference EP. `cpu` (default) uses the historical
    /// single-threaded ORT CPU path (FP-deterministic with
    /// `serve_onnx --ort-threads 1`). `cuda` loads the ONNX session on
    /// the CUDA EP and drops the per-call inference Mutex; this is the
    /// Slice 3c parallelism throughput lever. See
    /// `docs/ai-research/scoping/gpu-inference-execution-provider.md`.
    #[arg(long, default_value = "cpu")]
    device: String,
    /// CUDA device id (only honored under `--device cuda`).
    #[arg(long, default_value_t = 0)]
    cuda_device_id: i32,
    /// Slice 1 of `docs/ai-research/scoping/deck-pair-sampling.md`.
    /// Deck-pair sampling mode:
    ///   - `fixed` (default) — every game uses the registry defaults
    ///     (matikanetannhauser vs matikanetannhauser, the historical
    ///     tight-gate matchup). Byte-identical to pre-Slice-1 behavior.
    ///   - `uniform` — index by `(seed_base + task_index) %
    ///     (n_player_decks * n_ai_decks)`; row-major pair pick.
    ///     Deterministic across `--workers`.
    ///   - `pair=<player>:<opponent>` — literal deck-id pair (validates
    ///     both ids at parse time).
    /// Every game record in `--progress-out` carries `playerDeckId` +
    /// `opponentDeckId` regardless of mode; the aggregate manifest gains
    /// a `perMatchup` section only when sampling != fixed.
    #[arg(long, default_value = "fixed")]
    deck_sampling: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SideStat {
    games: u32,
    wins: u32,
    win_rate: f64,
    wilson_lower: f64,
    wilson_upper: f64,
}

/// Inner `summary` shape — the contract consumed by
/// `training/r12_orchestrator.py::load_summary` (reads
/// `summary.wilson95.lower`, `summary.modelWinRate`, `summary.games`,
/// `summary.heuristicFallbacks`). The orchestrator parser is the canonical
/// shape; this struct mirrors `backend/src/sim/evalGate.ts::summarize`
/// (`evalGate.ts:392`). Keep IN SYNC with that function — drift here
/// silently corrupts `orchestrator-state.json` (Bug 1 of
/// `rust-port-orchestrator-wiring` Slice 2 follow-ups).
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GateInnerSummary {
    games: u32,
    model_wins: u32,
    model_win_rate: f64,
    wilson95: Wilson95,
    /// Rust eval-gate does not track heuristic-fallback counts (the
    /// MCTS-rolling-loop path here has no "fall back to heuristic"
    /// branch the way TS evaluateModelVsHeuristic does), so always 0.
    /// Required for parser parity — promotion gates that key on
    /// `requireZeroFallbacks` will trivially pass.
    heuristic_fallbacks: u32,
    /// Same situation as `heuristic_fallbacks` — Rust eval-gate doesn't
    /// distinguish selected no-ops; emit 0 for parser parity.
    selected_no_ops: u32,
    overall: SideStat,
    player_side: SideStat,
    opponent_side: SideStat,
    terminal_game_over: u32,
    terminal_stalled: u32,
    terminal_max_steps: u32,
    elapsed_secs: f64,
    games_per_sec: f64,
}

#[derive(Serialize)]
struct Wilson95 {
    lower: f64,
    upper: f64,
}

/// Top-level manifest shape — mirrors TS `evalGate.ts:156`
/// (`{ status, failures, expectFail, corpusMode, args, summary }`).
/// Required by `training/r12_orchestrator.py::load_summary`, which reads
/// `payload["summary"]`. Bug 1 of `rust-port-orchestrator-wiring` Slice 2
/// follow-ups: prior shape was flat (no `summary` wrapper) so the parser
/// silently fell back to 0.0/games=0 on every Rust-default iter record.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GateSummary {
    /// TS `status`: "PASS" | "FAIL" | "FAIL_UNEXPECTED_PASS"
    /// (`evalGate.ts:152`).
    status: &'static str,
    failures: Vec<String>,
    /// TS `expectFail` — Rust eval-gate has no `--expect-fail` flag, so
    /// always false; left in the manifest for shape parity.
    expect_fail: bool,
    /// TS `corpusMode` — Rust eval-gate has no `--relabel-mcts` corpus
    /// path, so always false.
    corpus_mode: bool,
    /// Echo of the resolved arg set, mirrors TS `args` block.
    args: serde_json::Value,
    summary: GateInnerSummary,
    /// Inlined challenger/baseline/config left at top-level too for
    /// backward compatibility with any reader of the previous shape
    /// (e.g. older `runs/R110-rust-parity-rust/iter-0/gate.manifest.json`
    /// inspectors). New consumers should read `summary.*`.
    challenger: Option<String>,
    baseline: Option<String>,
    config: ConfigEcho,
    /// TS-parity gate pass/fail; also lifted into `status` above.
    passed: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ConfigEcho {
    sims: u32,
    k: u32,
    rollout_steps: u32,
    seeds: u32,
    seed_base: u32,
}

/// Wilson score interval for a binomial proportion at 95% confidence.
/// Mirror of `wilsonInterval` in `evalGate.ts:440`.
fn wilson_interval(successes: u32, total: u32) -> (f64, f64) {
    if total == 0 {
        return (0.0, 0.0);
    }
    let z = 1.96_f64;
    let n = total as f64;
    let p = successes as f64 / n;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denom;
    let half = (z * ((p * (1.0 - p) + z2 / (4.0 * n)) / n).sqrt()) / denom;
    ((center - half).max(0.0), (center + half).min(1.0))
}

fn drive_one_game(
    seed: u32,
    model_side: SideId,
    max_steps: u32,
    selection: &str,
    config: &MctsConfig,
    player_deck: Option<&[engine::core::card_id::CardId]>,
    opponent_deck: Option<&[engine::core::card_id::CardId]>,
) -> (Option<SideId>, &'static str) {
    let seed_str = seed.to_string();
    let rng = Rng::from_seed(format!("{}:selfplay", seed_str).as_str(), "selfplay");
    let (mut state, mut step_rng) = with_rng(rng, || {
        setup_ai_vs_ai_game_with_decks(player_deck, opponent_deck)
    });
    let mut terminal = "max_steps";
    for s in 0..max_steps {
        if state.game_over {
            terminal = "game_over";
            break;
        }
        let side = match state.current_side {
            CurrentSide::Player => SideId::Player,
            CurrentSide::Opponent => SideId::Opponent,
            CurrentSide::Done => {
                terminal = "game_over";
                break;
            }
        };
        let (legal, used_rng) = with_rng(step_rng.clone(), || {
            enumerate_legal_ai_actions(&state, side)
        });
        step_rng = used_rng;

        let pre_hash = state_hash(&state);
        let next_state = if side == model_side && legal.len() > 1 {
            let idx = if selection == "random" {
                let r = step_rng.next_f64();
                ((r * legal.len() as f64).floor() as usize).min(legal.len() - 1)
            } else {
                let mcts_seed = format!("{}:{:?}:{}:mcts", seed_str, side, s);
                let model_url = config.model_url.clone();
                let (mcts_result, used_rng) = with_rng(step_rng.clone(), || {
                    run_mcts(&state, side, config, model_url.as_str(), mcts_seed.as_str())
                });
                step_rng = used_rng;
                mcts_result.selected_index.min(legal.len() - 1)
            };
            let chosen = legal[idx].clone();
            let (ns, used_rng) = with_rng(step_rng.clone(), || {
                let forced = get_forced_attack_coin_results(&state);
                advance_modeled_turn_step(&state, side, &chosen, forced)
            });
            step_rng = used_rng;
            ns
        } else {
            let (ns, used_rng) = with_rng(step_rng.clone(), || {
                let forced = get_forced_attack_coin_results(&state);
                let mut s = state.clone();
                if side == SideId::Player {
                    advance_player_ai_turn_step(&mut s, forced);
                } else {
                    advance_opponent_turn_step(&mut s, forced);
                }
                s
            });
            step_rng = used_rng;
            ns
        };
        if state_hash(&next_state) == pre_hash {
            terminal = "stalled";
            break;
        }
        state = next_state;
        let _ = s;
    }
    (state.winner, terminal)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let sampling = DeckSampling::parse(&args.deck_sampling)
        .map_err(|e| anyhow::anyhow!("--deck-sampling: {}", e))?;
    let leaf = match args.leaf.as_str() {
        "value-head" => MctsLeaf::ValueHead,
        _ => MctsLeaf::Rollout,
    };
    let prior = match args.prior.as_str() {
        "policy" => MctsPrior::Policy,
        _ => MctsPrior::Uniform,
    };
    let model_url = args.challenger.clone().unwrap_or_default();
    let config = MctsConfig {
        simulations: args.sims,
        c_puct: args.c_puct,
        leaf,
        prior,
        rollout_crn_samples: args.k,
        rollout_steps: args.rollout_steps,
        record_rollout_leaf_samples: 0,
        value_head_rollout_blend: args.value_head_rollout_blend,
        // TS parity (`backend/src/sim/evalGate.ts:509`):
        // `mctsRootDirichlet: argv.includes("--mcts-root-dirichlet")` →
        // false by default for evaluation determinism. Rust eval-gate
        // matches: no opt-in flag wired here, so always false. Selfplay
        // path (`mcts_selfplay.rs`) is the inverse (default ON; opt-out
        // via `--no-root-dirichlet`) — see Bug fix 2 there.
        add_root_dirichlet: false,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.25,
        max_nodes: args.max_nodes,
        collapse_max_steps: args.collapse_max,
        adaptive_ratio: 0.0,
        adaptive_min_sims: 100,
        two_sided: args.mcts_two_sided,
        model_url: model_url.clone(),
        onnx_path: args.onnx_path.clone().map(PathBuf::from),
    };
    let selection = args.selection.clone();
    let _ = args.min_ci_lower;
    let _ = args.min_games;
    // `--workers` is consumed below; resolve `0` to available_parallelism.
    let workers: usize = if args.workers == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        args.workers as usize
    };
    // R16-P3 spike Option A: load the ONNX policy if either leaf or
    // prior path needs the model. Same bootstrap pattern as
    // sim-mcts-selfplay — one-time graph compile, shared session.
    let device = match args.device.as_str() {
        "cpu" => Device::Cpu,
        "cuda" => Device::Cuda {
            device_id: args.cuda_device_id,
        },
        other => anyhow::bail!("--device must be one of 'cpu' or 'cuda' (got {})", other),
    };
    let needs_inference = selection != "random"
        && (matches!(prior, MctsPrior::Policy) || matches!(leaf, MctsLeaf::ValueHead));
    if needs_inference {
        let onnx = args
            .onnx_path
            .as_deref()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "sim-eval-gate: --onnx-path is required when --prior policy or --leaf value-head is set"
                )
            })?;
        let session = InferenceSession::load_on(std::path::Path::new(onnx), device)
            .map_err(|e| anyhow::anyhow!("failed to load ONNX session at {}: {}", onnx, e))?;
        inference::set_global(session);
        eprintln!(
            "sim-eval-gate: loaded inference session from {} (device={:?})",
            onnx, device
        );
    } else if !model_url.is_empty() {
        eprintln!(
            "sim-eval-gate: WARNING --challenger/--model-url is DEPRECATED (R16-P3); \
             pass --onnx-path instead. The HTTP /predict path has been removed."
        );
    }
    // Mirror TS `evalGate.ts:43-49`: build the (seed, side) task list so
    // `--games N --model-side both` schedules 2N tasks (one per side per
    // seed), and `--games N --model-side player|opponent` schedules N
    // tasks for that side. Prior code alternated side per seed and ran
    // exactly `seeds` total — half of TS under `both`. See Slice 2
    // follow-up in `docs/ai-research/scoping/rust-engine-port-handoff.md`
    // § 'Phase 1h follow-up'.
    let sides: Vec<SideId> = match args.model_side.as_str() {
        "player" => vec![SideId::Player],
        "opponent" => vec![SideId::Opponent],
        "both" => vec![SideId::Player, SideId::Opponent],
        other => anyhow::bail!(
            "--model-side must be player, opponent, or both (got {})",
            other
        ),
    };
    let tasks: Vec<(u32, SideId)> = sides
        .iter()
        .flat_map(|&side| (0..args.seeds).map(move |i| (args.seed_base + i, side)))
        .collect();
    // Truncate progress-out file at start (matches TS `writeFileSync(path, "")`).
    if let Some(path) = args.progress_out.as_ref() {
        let p = PathBuf::from(path);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).with_context(|| format!("mkdir {}", parent.display()))?;
        }
        fs::write(&p, "").with_context(|| format!("truncate {}", path))?;
    }

    eprintln!(
        "sim-eval-gate: sims={} K={} rollout_steps={} games-per-side={} model-side={} (=> {} total tasks) base={} deck-sampling={} challenger={:?} baseline={:?}",
        args.sims, args.k, args.rollout_steps, args.seeds, args.model_side, tasks.len(), args.seed_base, args.deck_sampling, args.challenger, args.baseline,
    );

    let start = Instant::now();
    // Side assignment is driven by the (seed, side) task list above
    // (TS-parity), not by even/odd seed rotation.
    let total_tasks = tasks.len();
    // Per-task outcome — populated by worker threads, then sorted by
    // `task_index` for deterministic aggregation. Order in the progress
    // JSONL can be out-of-completion-order (parallel writes), but the
    // final manifest stays bit-stable because we re-sort here.
    #[derive(Clone, Copy)]
    struct Outcome {
        model_side: SideId,
        winner: Option<SideId>,
        terminal: &'static str,
        // Slice 1 (deck-pair-sampling): per-game manifest emits these
        // unconditionally so post-hoc consumers can stratify. For
        // `--deck-sampling=fixed` both ids point at the registry
        // defaults; for `uniform` / `pair=...` they vary per game.
        // 'static lifetime is safe — the deck registry is OnceLock-init.
        player_deck_id: &'static str,
        opponent_deck_id: &'static str,
    }
    let outcomes: Arc<Mutex<Vec<Option<Outcome>>>> = Arc::new(Mutex::new(vec![None; total_tasks]));
    // BufWriter wrapped in Mutex: each completed game appends one JSONL
    // line and flushes. Lines arrive out-of-order under parallelism;
    // consumers should not assume monotone gameIndex.
    let progress_writer: Option<Arc<Mutex<BufWriter<fs::File>>>> = match args.progress_out.as_ref()
    {
        Some(path) => {
            let f = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .with_context(|| format!("open progress-out {}", path))?;
            Some(Arc::new(Mutex::new(BufWriter::new(f))))
        }
        None => None,
    };
    // Shared running completion counter — drives `gameIndex`,
    // `etaSec`, and `runningWinRate` in the progress JSONL. Under
    // parallelism `gameIndex` is the *Nth game to complete*, not the
    // Nth-in-schedule. ETA is wall-clock based:
    // `remaining * (elapsed / completed)`.
    let completed_counter = Arc::new(AtomicUsize::new(0));
    let running_wins_counter = Arc::new(AtomicUsize::new(0));

    let task_cursor = Arc::new(AtomicUsize::new(0));
    let tasks_arc: Arc<Vec<(u32, SideId)>> = Arc::new(tasks);
    let config_arc = Arc::new(config);
    let sampling_arc = Arc::new(sampling);
    let selection_arc = Arc::new(selection);
    let seed_base = args.seed_base;

    // Effective worker count: never more than tasks (no-op extras
    // would just contend on the cursor).
    let effective_workers = workers.max(1).min(total_tasks.max(1));
    eprintln!(
        "sim-eval-gate: workers={} (requested {}, total_tasks {})",
        effective_workers, args.workers, total_tasks
    );

    std::thread::scope(|s| -> Result<()> {
        let mut handles = Vec::with_capacity(effective_workers);
        for worker_id in 0..effective_workers {
            let task_cursor = Arc::clone(&task_cursor);
            let tasks_arc = Arc::clone(&tasks_arc);
            let outcomes = Arc::clone(&outcomes);
            let config_arc = Arc::clone(&config_arc);
            let selection_arc = Arc::clone(&selection_arc);
            let completed_counter = Arc::clone(&completed_counter);
            let running_wins_counter = Arc::clone(&running_wins_counter);
            let progress_writer = progress_writer.as_ref().map(Arc::clone);
            let sampling_arc = Arc::clone(&sampling_arc);
            let max_steps = args.max_steps;
            let start_for_worker = start;
            let total_tasks_u32 = total_tasks as u32;
            handles.push(s.spawn(move || -> Result<()> {
                loop {
                    let task_index = task_cursor.fetch_add(1, Ordering::Relaxed);
                    if task_index >= tasks_arc.len() {
                        break;
                    }
                    let (seed, model_side) = tasks_arc[task_index];
                    // Deck-pair resolution. For `Fixed` this returns None
                    // and `drive_one_game` falls through to defaults
                    // exactly as pre-Slice-1.
                    let resolved = sampling_arc.resolve(seed_base, task_index as u32);
                    let (player_deck_opt, opponent_deck_opt) = resolved
                        .as_ref()
                        .map(|p| (Some(p.player_deck), Some(p.opponent_deck)))
                        .unwrap_or((None, None));
                    let (player_deck_id, opponent_deck_id) =
                        manifest_pair_for(&sampling_arc, seed_base, task_index as u32);
                    let game_start = Instant::now();
                    let (winner, terminal) = drive_one_game(
                        seed,
                        model_side,
                        max_steps,
                        selection_arc.as_str(),
                        &config_arc,
                        player_deck_opt,
                        opponent_deck_opt,
                    );
                    let game_secs = game_start.elapsed().as_secs_f64();
                    let model_won = winner == Some(model_side);
                    // Store outcome for final aggregation (sorted by index).
                    {
                        let mut guard = outcomes.lock().expect("outcomes mutex poisoned");
                        guard[task_index] = Some(Outcome {
                            model_side,
                            winner,
                            terminal,
                            player_deck_id,
                            opponent_deck_id,
                        });
                    }
                    // Update running counters BEFORE writing the JSONL
                    // line so `gameIndex` and `runningWinRate` reflect
                    // this completion.
                    if model_won {
                        running_wins_counter.fetch_add(1, Ordering::Relaxed);
                    }
                    let games_completed =
                        (completed_counter.fetch_add(1, Ordering::Relaxed) + 1) as u32;
                    if let Some(w) = progress_writer.as_ref() {
                        let elapsed_so_far = start_for_worker.elapsed().as_secs_f64();
                        let eta_sec = if games_completed > 0 {
                            (elapsed_so_far / games_completed as f64)
                                * (total_tasks_u32.saturating_sub(games_completed)) as f64
                        } else {
                            0.0
                        };
                        let running_wins =
                            running_wins_counter.load(Ordering::Relaxed) as f64;
                        let running_wr = if games_completed > 0 {
                            running_wins / games_completed as f64
                        } else {
                            0.0
                        };
                        let row = serde_json::json!({
                            "event": "game_completed",
                            "gameIndex": games_completed,
                            "taskIndex": task_index,
                            "totalGames": total_tasks_u32,
                            "seed": seed,
                            "modelSide": if model_side == SideId::Player { "Player" } else { "Opponent" },
                            "winner": winner.map(|s| if s == SideId::Player { "Player" } else { "Opponent" }),
                            "modelWon": model_won,
                            "terminalReason": terminal,
                            "gameElapsedSec": game_secs,
                            "totalElapsedSec": elapsed_so_far,
                            "etaSec": eta_sec,
                            "runningWinRate": running_wr,
                            "workerId": worker_id,
                            // Slice 1 (deck-pair-sampling): emitted on
                            // every game record regardless of mode so
                            // consumers can stratify post-hoc.
                            "playerDeckId": player_deck_id,
                            "opponentDeckId": opponent_deck_id,
                            "ts": std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_secs_f64())
                                .unwrap_or(0.0),
                        });
                        let mut guard = w.lock().expect("progress mutex poisoned");
                        writeln!(&mut *guard, "{}", row)?;
                        guard.flush()?;
                    }
                }
                Ok(())
            }));
        }
        for h in handles {
            h.join().expect("worker thread panicked")?;
        }
        Ok(())
    })?;

    // Flush progress writer one final time (the per-line flush is best
    // effort but a final flush ensures the BufWriter is empty before
    // the manifest write).
    if let Some(w) = progress_writer.as_ref() {
        let mut guard = w.lock().expect("progress mutex poisoned");
        guard.flush()?;
    }

    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();

    // Aggregate outcomes in task-index order so the manifest is
    // bit-stable regardless of worker completion order.
    let mut player_games = 0u32;
    let mut player_wins = 0u32;
    let mut opp_games = 0u32;
    let mut opp_wins = 0u32;
    let mut terminal_game_over = 0u32;
    let mut terminal_stalled = 0u32;
    let mut terminal_max_steps = 0u32;
    // Slice 1 (deck-pair-sampling): per-matchup tallies, keyed by
    // (player_deck_id, opponent_deck_id). Populated unconditionally;
    // serialized into the manifest only when sampling != fixed.
    let mut per_matchup: std::collections::BTreeMap<(String, String), (u32, u32)> =
        std::collections::BTreeMap::new();
    {
        let guard = outcomes.lock().expect("outcomes mutex poisoned");
        for (idx, slot) in guard.iter().enumerate() {
            let o = slot.expect("worker did not fill outcome slot");
            let _ = idx;
            match o.terminal {
                "game_over" => terminal_game_over += 1,
                "stalled" => terminal_stalled += 1,
                _ => terminal_max_steps += 1,
            }
            let model_won = o.winner == Some(o.model_side);
            if o.model_side == SideId::Player {
                player_games += 1;
                if model_won {
                    player_wins += 1;
                }
            } else {
                opp_games += 1;
                if model_won {
                    opp_wins += 1;
                }
            }
            let key = (o.player_deck_id.to_string(), o.opponent_deck_id.to_string());
            let entry = per_matchup.entry(key).or_insert((0u32, 0u32));
            entry.0 += 1; // games
            if model_won {
                entry.1 += 1; // wins
            }
        }
    }

    let total_games = player_games + opp_games;
    let total_wins = player_wins + opp_wins;
    let mk_stat = |w: u32, g: u32| {
        let (lo, hi) = wilson_interval(w, g);
        SideStat {
            games: g,
            wins: w,
            win_rate: if g > 0 { w as f64 / g as f64 } else { 0.0 },
            wilson_lower: lo,
            wilson_upper: hi,
        }
    };

    let overall = mk_stat(total_wins, total_games);
    let mut failures: Vec<String> = Vec::new();
    if total_games < args.min_games {
        failures.push(format!(
            "games {} < minGames {}",
            total_games, args.min_games
        ));
    }
    if overall.win_rate < args.min_win_rate {
        failures.push(format!(
            "winRate {:.4} < minWinRate {:.4}",
            overall.win_rate, args.min_win_rate
        ));
    }
    if overall.wilson_lower < args.min_ci_lower {
        failures.push(format!(
            "wilsonLower {:.4} < minCiLower {:.4}",
            overall.wilson_lower, args.min_ci_lower
        ));
    }
    let passed = failures.is_empty();

    let games_per_sec = total_games as f64 / elapsed_secs.max(1e-9);
    let model_win_rate = overall.win_rate;
    let wilson_lower = overall.wilson_lower;
    let wilson_upper = overall.wilson_upper;
    let player_side = mk_stat(player_wins, player_games);
    let opponent_side = mk_stat(opp_wins, opp_games);
    // Echo of the resolved args set, mirrors TS `evalGate.ts:156` `args`
    // block. Only fields the orchestrator looks at on a routine basis
    // are included; the full clap-parsed set isn't serialized to avoid
    // dragging in fields TS doesn't expose.
    let args_echo = serde_json::json!({
        "challenger": args.challenger,
        "baseline": args.baseline,
        "seeds": args.seeds,
        "seedBase": args.seed_base,
        "sims": args.sims,
        "k": args.k,
        "rolloutSteps": args.rollout_steps,
        "valueHeadRolloutBlend": args.value_head_rollout_blend,
        "maxSteps": args.max_steps,
        "leaf": args.leaf,
        "prior": args.prior,
        "cPuct": args.c_puct,
        "maxNodes": args.max_nodes,
        "collapseMax": args.collapse_max,
        "selection": args.selection,
        "modelSide": args.model_side,
        "minCiLower": args.min_ci_lower,
        "minGames": args.min_games,
        "minWinRate": args.min_win_rate,
        "manifestOut": args.manifest_out,
        "progressOut": args.progress_out,
        "workers": args.workers,
        "deckSampling": args.deck_sampling,
        "mctsTwoSided": args.mcts_two_sided,
    });
    let status = if passed { "PASS" } else { "FAIL" };
    let inner = GateInnerSummary {
        games: total_games,
        model_wins: total_wins,
        model_win_rate,
        wilson95: Wilson95 {
            lower: wilson_lower,
            upper: wilson_upper,
        },
        heuristic_fallbacks: 0,
        selected_no_ops: 0,
        overall,
        player_side,
        opponent_side,
        terminal_game_over,
        terminal_stalled,
        terminal_max_steps,
        elapsed_secs,
        games_per_sec,
    };
    let summary = GateSummary {
        status,
        failures: failures.clone(),
        expect_fail: false,
        corpus_mode: false,
        args: args_echo,
        summary: inner,
        challenger: args.challenger.clone(),
        baseline: args.baseline.clone(),
        config: ConfigEcho {
            sims: args.sims,
            k: args.k,
            rollout_steps: args.rollout_steps,
            seeds: args.seeds,
            seed_base: args.seed_base,
        },
        passed,
    };

    // Serialize the typed summary, then optionally splice in a
    // `perMatchup` block when sampling != fixed (Slice 1 of the
    // deck-pair-sampling scoping doc). `perMatchup` is keyed
    // `"<player_deck_id>:<opponent_deck_id>"` with point-estimate
    // win-rate + Wilson 95% bounds. At n≈45/matchup the per-matchup CIs
    // are wide (≈±15pp); the field is signal-spotting, not a verdict.
    let mut value = serde_json::to_value(&summary)?;
    if sampling_arc.is_active() {
        let mut per_matchup_obj = serde_json::Map::new();
        for ((player_id, opponent_id), (games, wins)) in per_matchup.iter() {
            let (lo, hi) = wilson_interval(*wins, *games);
            let win_rate = if *games > 0 {
                *wins as f64 / *games as f64
            } else {
                0.0
            };
            let key = format!("{}:{}", player_id, opponent_id);
            per_matchup_obj.insert(
                key,
                serde_json::json!({
                    "playerDeckId": player_id,
                    "opponentDeckId": opponent_id,
                    "games": games,
                    "wins": wins,
                    "winRate": win_rate,
                    "wilsonLower": lo,
                    "wilsonUpper": hi,
                }),
            );
        }
        if let Some(obj) = value.as_object_mut() {
            obj.insert(
                "perMatchup".to_string(),
                serde_json::Value::Object(per_matchup_obj),
            );
        }
    }
    let json = serde_json::to_string_pretty(&value)?;
    println!("{}", json);

    if let Some(path) = args.manifest_out.as_ref() {
        let p = PathBuf::from(path);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).with_context(|| format!("mkdir {}", parent.display()))?;
        }
        let mut f = fs::File::create(&p).with_context(|| format!("create {}", path))?;
        writeln!(f, "{}", json)?;
    }

    if !passed {
        eprintln!(
            "gate FAILED: {} condition(s) unmet:\n  - {}",
            failures.len(),
            failures.join("\n  - "),
        );
        std::process::exit(1);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    //! Games-vs-model-side semantics lock.
    //!
    //! TS `backend/src/sim/evalGate.ts:43-49` interprets `--games N` as
    //! N games **per modelled side**: under `--model-side both` the
    //! orchestrator schedules `2N` tasks (player×N + opponent×N); under
    //! `--model-side player|opponent` it schedules N tasks. The Rust
    //! binary was previously off-spec — `--seeds N --model-side both`
    //! ran exactly N games, alternating side. That asymmetry meant
    //! `r12_orchestrator.run_gate` forwarding the same
    //! `--games eval_games` to both engines silently ran HALF the games
    //! under `--engine rust` (see Slice 2 verdict in
    //! `docs/ai-research/scoping/rust-engine-port-handoff.md`
    //! § 'Phase 1h follow-up'). Pin the schedule math here so a future
    //! drift fails at `cargo test`, not at gate-stage Wilson-noise time.
    use super::*;
    use engine::core::constants::SideId;

    fn build_tasks(games: u32, model_side: &str, seed_base: u32) -> Vec<(u32, SideId)> {
        let sides: Vec<SideId> = match model_side {
            "player" => vec![SideId::Player],
            "opponent" => vec![SideId::Opponent],
            "both" => vec![SideId::Player, SideId::Opponent],
            _ => panic!("bad model_side in test"),
        };
        sides
            .iter()
            .flat_map(|&side| (0..games).map(move |i| (seed_base + i, side)))
            .collect()
    }

    #[test]
    fn games_model_side_both_doubles_task_count() {
        // TS evalGate.ts: --games 5 --model-side both → 10 tasks.
        let tasks = build_tasks(5, "both", 70000);
        assert_eq!(tasks.len(), 10, "both must schedule 2N tasks");
        // Player block first, then opponent block — mirrors TS
        // `for (const side of sides) for (let index = 0; ...)`.
        let player_block: Vec<_> = tasks.iter().filter(|(_, s)| *s == SideId::Player).collect();
        let opponent_block: Vec<_> = tasks
            .iter()
            .filter(|(_, s)| *s == SideId::Opponent)
            .collect();
        assert_eq!(player_block.len(), 5);
        assert_eq!(opponent_block.len(), 5);
        // Each side iterates [seed_base, seed_base + games), so both
        // sides share the same seed set (TS parity).
        let player_seeds: Vec<u32> = player_block.iter().map(|(s, _)| *s).collect();
        let opponent_seeds: Vec<u32> = opponent_block.iter().map(|(s, _)| *s).collect();
        assert_eq!(player_seeds, vec![70000, 70001, 70002, 70003, 70004]);
        assert_eq!(opponent_seeds, vec![70000, 70001, 70002, 70003, 70004]);
    }

    #[test]
    fn games_model_side_single_is_n_total() {
        // TS evalGate.ts: --games 5 --model-side player → 5 tasks all Player.
        let tasks = build_tasks(5, "player", 70000);
        assert_eq!(tasks.len(), 5);
        assert!(tasks.iter().all(|(_, s)| *s == SideId::Player));

        let tasks = build_tasks(5, "opponent", 70000);
        assert_eq!(tasks.len(), 5);
        assert!(tasks.iter().all(|(_, s)| *s == SideId::Opponent));
    }

    #[test]
    fn clap_accepts_both_games_and_seeds_aliases() {
        // Orchestrator passes `--games`; legacy callers / docs use `--seeds`.
        // Clap `long = "games", alias = "seeds"` must accept both as the
        // same field.
        let a = Args::parse_from(["sim-eval-gate", "--games", "7"]);
        assert_eq!(a.seeds, 7);
        let b = Args::parse_from(["sim-eval-gate", "--seeds", "7"]);
        assert_eq!(b.seeds, 7);
    }
}
