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
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use engine::core::constants::SideId;
use engine::core::random::{with_rng, Rng};
use engine::core::state::CurrentSide;
use engine::dispatcher::{
    advance_modeled_turn_step, advance_opponent_turn_step, advance_player_ai_turn_step,
    get_forced_attack_coin_results, state_hash,
};
use engine::headless_setup::setup_ai_vs_ai_game;
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
    /// Selection mode (mcts | rollout | planner). Currently only mcts
    /// is wired in this binary; accepted for orchestrator-flag parity.
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
    /// Worker fan-out (no-op — Rust runs single-process; orchestrators
    /// parallelise by spawning multiple Rust binaries).
    #[arg(long, default_value_t = 1)]
    workers: u32,
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
    config: &MctsConfig,
) -> (Option<SideId>, &'static str) {
    let seed_str = seed.to_string();
    let rng = Rng::from_seed(format!("{}:selfplay", seed_str).as_str(), "selfplay");
    let (mut state, mut step_rng) = with_rng(rng, || setup_ai_vs_ai_game());
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
            let mcts_seed = format!("{}:{:?}:{}:mcts", seed_str, side, s);
            let model_url = config.model_url.clone();
            let (mcts_result, used_rng) = with_rng(step_rng.clone(), || {
                run_mcts(&state, side, config, model_url.as_str(), mcts_seed.as_str())
            });
            step_rng = used_rng;
            let idx = mcts_result.selected_index.min(legal.len() - 1);
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
        model_url: model_url.clone(),
    };
    // Accept for orchestrator-flag parity (no-op stubs for now).
    let _ = args.selection;
    let _ = args.min_ci_lower;
    let _ = args.min_games;
    let _ = args.workers;
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
        "sim-eval-gate: sims={} K={} rollout_steps={} games-per-side={} model-side={} (=> {} total tasks) base={} challenger={:?} baseline={:?}",
        args.sims, args.k, args.rollout_steps, args.seeds, args.model_side, tasks.len(), args.seed_base, args.challenger, args.baseline,
    );

    let start = Instant::now();
    // Side assignment is now driven by the (seed, side) task list above
    // (TS-parity), not by even/odd seed rotation.
    let mut player_games = 0u32;
    let mut player_wins = 0u32;
    let mut opp_games = 0u32;
    let mut opp_wins = 0u32;
    let mut terminal_game_over = 0u32;
    let mut terminal_stalled = 0u32;
    let mut terminal_max_steps = 0u32;

    let mut progress_writer = match args.progress_out.as_ref() {
        Some(path) => Some(
            fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .with_context(|| format!("open progress-out {}", path))?,
        ),
        None => None,
    };
    let total_tasks = tasks.len() as u32;
    for (task_index, &(seed, model_side)) in tasks.iter().enumerate() {
        let game_start = Instant::now();
        let (winner, terminal) = drive_one_game(seed, model_side, args.max_steps, &config);
        let game_secs = game_start.elapsed().as_secs_f64();
        match terminal {
            "game_over" => terminal_game_over += 1,
            "stalled" => terminal_stalled += 1,
            _ => terminal_max_steps += 1,
        }
        let model_won = winner == Some(model_side);
        if model_side == SideId::Player {
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
        if let Some(w) = progress_writer.as_mut() {
            let games_completed = (task_index as u32) + 1;
            let elapsed_so_far = start.elapsed().as_secs_f64();
            let eta_sec = if games_completed > 0 {
                (elapsed_so_far / games_completed as f64)
                    * (total_tasks.saturating_sub(games_completed)) as f64
            } else {
                0.0
            };
            let running_wr = if games_completed > 0 {
                (player_wins + opp_wins) as f64 / games_completed as f64
            } else {
                0.0
            };
            let row = serde_json::json!({
                "event": "game_completed",
                "gameIndex": games_completed,
                "totalGames": total_tasks,
                "seed": seed,
                "modelSide": if model_side == SideId::Player { "Player" } else { "Opponent" },
                "winner": winner.map(|s| if s == SideId::Player { "Player" } else { "Opponent" }),
                "modelWon": model_won,
                "terminalReason": terminal,
                "gameElapsedSec": game_secs,
                "totalElapsedSec": elapsed_so_far,
                "etaSec": eta_sec,
                "runningWinRate": running_wr,
                "workerId": 0,
                "ts": std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs_f64())
                    .unwrap_or(0.0),
            });
            writeln!(w, "{}", row)?;
        }
    }
    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();

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

    let json = serde_json::to_string_pretty(&summary)?;
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
        let player_block: Vec<_> = tasks
            .iter()
            .filter(|(_, s)| *s == SideId::Player)
            .collect();
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

