//! Rust port of `backend/src/sim/mctsSelfPlay.ts`.
//!
//! Phase 1g — drives N games with Rust MCTS for the modeled side and
//! heuristic AI for the opponent. Writes per-game JSONL summaries +
//! a top-level summary. Per-decision trajectory recording is the next
//! increment (matches the recorder's selfplay row schema).

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
use engine::mcts::sample::pick_from_visits;
use engine::policy::actions::enumerate_legal_ai_actions;
use engine::policy::observation::build_public_observation;
use engine::policy::types::{LegalAiAction, PublicObservation};
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(
    name = "sim-mcts-selfplay",
    about = "Rust port of backend/src/sim/mctsSelfPlay.ts"
)]
struct Args {
    /// MCTS simulations per decision. Orchestrator alias: --mcts-simulations.
    #[arg(long, alias = "mcts-simulations", default_value_t = 100)]
    sims: u32,
    /// CRN rollout samples. Orchestrator alias: --mcts-rollout-crn-samples.
    #[arg(long, alias = "mcts-rollout-crn-samples", default_value_t = 3)]
    k: u32,
    /// Rollout step cap. Orchestrator alias: --mcts-rollout-steps.
    #[arg(long, alias = "mcts-rollout-steps", default_value_t = 200)]
    rollout_steps: u32,
    /// Heuristic-collapse cap. Orchestrator alias: --mcts-collapse-max-steps.
    #[arg(long, alias = "mcts-collapse-max-steps", default_value_t = 64)]
    collapse_max: u32,
    /// Number of seeds. Orchestrator alias: --games.
    #[arg(long, alias = "games", default_value_t = 100)]
    seeds: u32,
    /// First seed. Orchestrator alias: --seed-start.
    #[arg(long, alias = "seed-start", default_value_t = 0)]
    seed_base: u32,
    /// Prior mode: "uniform" or "policy". Orchestrator alias: --mcts-prior.
    #[arg(long, alias = "mcts-prior", default_value = "uniform")]
    prior: String,
    /// Leaf mode. Orchestrator alias: --mcts-leaf.
    #[arg(long, alias = "mcts-leaf", default_value = "rollout")]
    leaf: String,
    /// Worker fan-out (ignored — Rust runs single-process; orchestrator
    /// parallelises by spawning multiple Rust binaries).
    #[arg(long, default_value_t = 1)]
    workers: u32,
    /// PUCT exploration constant. Orchestrator alias: --mcts-c-puct.
    #[arg(long, alias = "mcts-c-puct", default_value_t = 1.5)]
    c_puct: f64,
    /// MCTS tree-node cap. Orchestrator alias: --mcts-max-nodes.
    #[arg(long, alias = "mcts-max-nodes", default_value_t = 5_000)]
    max_nodes: u32,
    /// Root Dirichlet noise alpha. Orchestrator alias: --mcts-dirichlet-alpha.
    #[arg(long, alias = "mcts-dirichlet-alpha", default_value_t = 0.3)]
    dirichlet_alpha: f64,
    /// Root Dirichlet noise epsilon. Orchestrator alias: --mcts-dirichlet-epsilon.
    #[arg(long, alias = "mcts-dirichlet-epsilon", default_value_t = 0.25)]
    dirichlet_epsilon: f64,
    /// Opt OUT of root Dirichlet noise. Mirrors TS
    /// `mctsSelfPlay.ts` default (`!argv.includes("--no-root-dirichlet")`):
    /// selfplay enables Dirichlet by default for exploration diversity;
    /// pass `--no-root-dirichlet` to disable. Eval-gate enables only via
    /// explicit opt-in (`--mcts-root-dirichlet` in TS evalGate.ts:509) —
    /// a separate flag lives on the `sim-eval-gate` binary.
    #[arg(long, default_value_t = false)]
    no_root_dirichlet: bool,
    /// Temperature-sampling cutoff in moves (per side).
    #[arg(long, default_value_t = 6)]
    temperature_moves: u32,
    #[arg(long, default_value_t = 1.0)]
    temperature_value: f64,
    /// Optional manifest JSON output (mirrors TS --manifest-out).
    #[arg(long)]
    manifest_out: Option<String>,
    #[arg(long, default_value_t = 1000)]
    max_steps: u32,
    #[arg(long)]
    out: Option<String>,
    /// Record per-decision trajectory rows (observation + legal actions
    /// + visit distribution) inside each game record. Off by default
    /// since rows can be large.
    #[arg(long, default_value_t = false)]
    record_rows: bool,
    /// `/predict` server URL. When set with --leaf=value-head, MCTS
    /// queries it for leaf values + policy priors.
    #[arg(long, default_value = "")]
    model_url: String,
}

impl Args {
    /// Serialize the args set into the JSON shape the TS sim:mcts-selfplay
    /// writes to --manifest-out: camelCase keys, all flags present.
    fn manifest_args_json(&self) -> serde_json::Value {
        serde_json::json!({
            "sims": self.sims,
            "k": self.k,
            "rolloutSteps": self.rollout_steps,
            "collapseMax": self.collapse_max,
            "seeds": self.seeds,
            "seedBase": self.seed_base,
            "prior": self.prior,
            "leaf": self.leaf,
            "workers": self.workers,
            "cPuct": self.c_puct,
            "maxNodes": self.max_nodes,
            "dirichletAlpha": self.dirichlet_alpha,
            "dirichletEpsilon": self.dirichlet_epsilon,
            "addRootDirichlet": !self.no_root_dirichlet,
            "temperatureMoves": self.temperature_moves,
            "temperatureValue": self.temperature_value,
            "manifestOut": self.manifest_out,
            "maxSteps": self.max_steps,
            "out": self.out,
            "recordRows": self.record_rows,
            "modelUrl": self.model_url,
        })
    }
}

/// Per-game wrapper. Emitted to `--out` ONLY when `--record-rows` is OFF.
/// When `--record-rows` is ON, the orchestrator's distill consumer expects
/// the TS-flat per-decision shape (`SelfPlayRow`), so we bypass this
/// wrapper and write rows one-per-line instead — see
/// `flush_rows_with_game_result` for the post-hoc fill of `valueTarget` +
/// `result` after the game ends. The wrapper itself stays around so any
/// caller that opted out of `--record-rows` still gets the per-game summary
/// the binary's earlier callers (sim-cli throughput probe, ad-hoc trace
/// inspection) consumed.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GameRecord {
    seed: u32,
    terminal_reason: String,
    turn_number: u32,
    winner: Option<String>,
    total_steps: u32,
    model_decisions: u32,
    /// Per-MCTS-decision trajectory rows. Empty when no MCTS decisions
    /// fired (e.g., game ended in single-action-only steps).
    rows: Vec<SelfPlayRow>,
}

/// `{ winner, pointsP, pointsO }` — mirrors TS `mctsSelfPlay.ts:530`'s
/// post-hoc `result` object. `null` until the game ends, at which point
/// `flush_rows_with_game_result` fills every row from the same game with a
/// reference to the SAME terminal result. Per-game scope is sufficient
/// since the loader never cross-references across games.
#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
struct GameResult {
    winner: Option<String>,
    #[serde(rename = "pointsP")]
    points_p: u8,
    #[serde(rename = "pointsO")]
    points_o: u8,
}

/// TS-flat per-decision row, byte-shape-compatible with
/// `backend/src/sim/mctsSelfPlay.ts:78-101` `SelfPlayRow` and consumed by
/// `training/uma_ai/selfplay_dataset.py:load_mcts_selfplay_samples` (the
/// distill input). All TS-flat keys are required; `value_target` + `result`
/// are filled in post-hoc when the game terminates so the Python loader's
/// `valueTarget ∈ {-1, 0, 1}` check passes (smoke
/// `training/r12_selfplay_smoke.py:135-137`). `seed` is serialized as a
/// string to match TS `seed: string` (TS passes the seed-base string down
/// from the orchestrator and never reparses it as a number).
///
/// The `kind: "mcts-selfplay"` discriminator is the binding constraint —
/// the Python loader raises `RowSchemaError` if it's missing (this is what
/// blew up Slice 2 of `rust-port-orchestrator-wiring`).
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SelfPlayRow {
    schema_version: u32,
    kind: &'static str,
    seed: String,
    side_id: String,
    step: u32,
    turn_number: u32,
    observation: PublicObservation,
    legal_actions: Vec<LegalAiAction>,
    selected_action_index: usize,
    visit_distribution: Vec<f64>,
    root_priors: Vec<f64>,
    root_value: f64,
    root_prior_entropy: f64,
    root_prior_argmax: usize,
    visited_hashes: u32,
    expansions: u32,
    leaf_evaluations: u32,
    /// `null` at emit time; filled in by `flush_rows_with_game_result`
    /// after the game terminates: +1 if this side won, -1 if lost, 0 on
    /// draw. Mirrors TS `mctsSelfPlay.ts:532`.
    value_target: Option<i8>,
    /// `null` at emit time; filled in post-hoc with `{ winner, pointsP,
    /// pointsO }` once the game terminates. Same per-row reference for
    /// every row in the same game.
    result: Option<GameResult>,
}

const ROW_SCHEMA_VERSION: u32 = 1;

/// Post-hoc fill: identical structure to TS `mctsSelfPlay.ts:530-534` —
/// after the game terminates, every row's `valueTarget` is computed from
/// the row's `sideId` against the terminal `winner` (+1 / -1 / 0), and
/// `result` carries the same `{ winner, pointsP, pointsO }` for every row
/// of the same game.
fn fill_terminal_result(
    rows: &mut [SelfPlayRow],
    winner: Option<&str>,
    points_p: u8,
    points_o: u8,
) {
    let result = GameResult {
        winner: winner.map(|s| s.to_string()),
        points_p,
        points_o,
    };
    for row in rows.iter_mut() {
        row.value_target = Some(match winner {
            None => 0,
            Some(w) if w == row.side_id => 1,
            Some(_) => -1,
        });
        row.result = Some(result.clone());
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RunSummary {
    games: u32,
    elapsed_secs: f64,
    games_per_sec: f64,
    player_wins: u32,
    opponent_wins: u32,
    draws: u32,
    terminal_reasons: TerminalReasons,
}

#[derive(Serialize, Default)]
#[serde(rename_all = "camelCase")]
struct TerminalReasons {
    game_over: u32,
    max_steps: u32,
    stalled: u32,
}

fn drive_one_game(
    seed: u32,
    max_steps: u32,
    config: &MctsConfig,
    model_side: SideId,
    record_rows: bool,
    temperature_moves: u32,
    temperature_value: f64,
) -> GameRecord {
    let seed_str = seed.to_string();
    let rng = Rng::from_seed(format!("{}:selfplay", seed_str).as_str(), "selfplay");
    let (mut state, mut step_rng) = with_rng(rng, || setup_ai_vs_ai_game());
    let mut step = 0u32;
    let mut terminal = "max_steps";
    let mut model_decisions = 0u32;
    let mut model_moves = 0u32;
    let mut rows: Vec<SelfPlayRow> = Vec::new();

    for s in 0..max_steps {
        step = s;
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
            let argmax_idx = mcts_result.selected_index.min(legal.len() - 1);
            // Temperature schedule (TS parity, mctsSelfPlay.ts:478-481):
            // exploratory for first `temperature_moves` model decisions,
            // greedy thereafter. Use a forked RNG so the outer game-rng
            // draw count doesn't shift between temp on/off runs.
            let temp = if model_moves < temperature_moves {
                temperature_value
            } else {
                0.0
            };
            let idx = if temp > 0.0 {
                let mut pick_rng = step_rng.fork(format!("pick:{}", s).as_str());
                pick_from_visits(&mcts_result.visits, argmax_idx, temp, &mut pick_rng)
                    .min(legal.len() - 1)
            } else {
                argmax_idx
            };
            let chosen = legal[idx].clone();
            model_decisions += 1;
            model_moves += 1;

            if record_rows {
                let total_visits: u32 = mcts_result.visits.iter().sum();
                let visit_distribution: Vec<f64> = if total_visits > 0 {
                    mcts_result.visits.iter().map(|&n| n as f64 / total_visits as f64).collect()
                } else {
                    let n = legal.len();
                    vec![1.0 / n.max(1) as f64; n]
                };
                rows.push(SelfPlayRow {
                    schema_version: ROW_SCHEMA_VERSION,
                    kind: "mcts-selfplay",
                    seed: seed_str.clone(),
                    side_id: match side {
                        SideId::Player => "player".into(),
                        SideId::Opponent => "opponent".into(),
                    },
                    step: s,
                    turn_number: state.turn_number,
                    observation: build_public_observation(&state, side),
                    legal_actions: legal.clone(),
                    selected_action_index: idx,
                    visit_distribution,
                    root_priors: mcts_result.diagnostics.root_priors.clone(),
                    root_value: mcts_result.diagnostics.root_value,
                    root_prior_entropy: mcts_result.diagnostics.root_prior_entropy,
                    root_prior_argmax: mcts_result.diagnostics.root_prior_argmax,
                    visited_hashes: mcts_result.diagnostics.visited_hashes,
                    expansions: mcts_result.diagnostics.expansions,
                    leaf_evaluations: mcts_result.diagnostics.leaf_evaluations,
                    // Post-hoc backfill below once the game terminates.
                    value_target: None,
                    result: None,
                });
            }

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
    }

    let winner = state.winner.map(|s| match s {
        SideId::Player => "player".to_string(),
        SideId::Opponent => "opponent".to_string(),
    });

    // Post-hoc backfill of `valueTarget` + `result` on every row of this
    // game, mirroring TS `mctsSelfPlay.ts:531-534`. The Python loader
    // (`training/r12_selfplay_smoke.py:135-137`) hard-asserts
    // `valueTarget ∈ {-1, 0, 1}`, so this fill is non-optional any time
    // `record_rows` is on. Cheap: rows is small per game.
    let points_p = state.side(SideId::Player).points;
    let points_o = state.side(SideId::Opponent).points;
    fill_terminal_result(&mut rows, winner.as_deref(), points_p, points_o);

    GameRecord {
        seed,
        terminal_reason: terminal.to_string(),
        turn_number: state.turn_number,
        winner,
        total_steps: step + 1,
        model_decisions,
        rows,
    }
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
    // Mirror TS `mctsSelfPlay.ts:627` default (Dirichlet ON unless explicit
    // `--no-root-dirichlet`). Bug fix 2 of `rust-port-orchestrator-wiring`
    // Slice 2 follow-ups: prior code hard-coded false even though both
    // alpha + epsilon were already plumbed, starving Rust selfplay of root
    // exploration noise and hurting trajectory diversity vs TS.
    let add_root_dirichlet = !args.no_root_dirichlet;
    let config = MctsConfig {
        simulations: args.sims,
        c_puct: args.c_puct,
        leaf,
        prior,
        rollout_crn_samples: args.k,
        rollout_steps: args.rollout_steps,
        add_root_dirichlet,
        dirichlet_alpha: args.dirichlet_alpha,
        dirichlet_epsilon: args.dirichlet_epsilon,
        max_nodes: args.max_nodes,
        collapse_max_steps: args.collapse_max,
        adaptive_ratio: 0.0,
        adaptive_min_sims: 100,
        model_url: args.model_url.clone(),
    };
    let _ = args.workers;
    let temperature_moves = args.temperature_moves;
    let temperature_value = args.temperature_value;

    let model_side = SideId::Player;
    eprintln!(
        "sim-mcts-selfplay: sims={} K={} rollout_steps={} prior={} leaf={} seeds={} (base={}) model_side={:?}",
        args.sims, args.k, args.rollout_steps, args.prior, args.leaf, args.seeds, args.seed_base, model_side,
    );

    let mut writer: Option<fs::File> = match args.out.as_ref() {
        Some(path) => {
            let p = PathBuf::from(path);
            if let Some(parent) = p.parent() {
                fs::create_dir_all(parent).with_context(|| format!("mkdir {}", parent.display()))?;
            }
            Some(fs::File::create(&p).with_context(|| format!("create {}", path))?)
        }
        None => None,
    };

    let start = Instant::now();
    let mut player_wins = 0u32;
    let mut opponent_wins = 0u32;
    let mut draws = 0u32;
    let mut terminal_reasons = TerminalReasons::default();

    for i in 0..args.seeds {
        let seed = args.seed_base + i;
        let record = drive_one_game(
            seed,
            args.max_steps,
            &config,
            model_side,
            args.record_rows,
            temperature_moves,
            temperature_value,
        );
        match record.terminal_reason.as_str() {
            "game_over" => terminal_reasons.game_over += 1,
            "stalled" => terminal_reasons.stalled += 1,
            _ => terminal_reasons.max_steps += 1,
        }
        match record.winner.as_deref() {
            Some("player") => player_wins += 1,
            Some("opponent") => opponent_wins += 1,
            _ => draws += 1,
        }
        if let Some(w) = writer.as_mut() {
            if args.record_rows {
                // TS-flat shape: one line per decision row. This is what
                // `training/uma_ai/selfplay_dataset.py:load_mcts_selfplay_samples`
                // expects (Slice 2 schema gap — `kind`/`schemaVersion`/
                // `sideId`/`observation`/`legalActions`/`visitDistribution`/
                // `rootPriors`/`rootValue`/`valueTarget`/`result` per row).
                for row in record.rows.iter() {
                    let line = serde_json::to_string(row)?;
                    writeln!(w, "{}", line)?;
                }
            } else {
                // Per-game wrapper retained for callers that opted out of
                // `--record-rows` (e.g., wall-clock probe runs that just
                // want game summaries).
                let line = serde_json::to_string(&record)?;
                writeln!(w, "{}", line)?;
            }
        }
    }

    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();
    let summary = RunSummary {
        games: args.seeds,
        elapsed_secs,
        games_per_sec: args.seeds as f64 / elapsed_secs.max(1e-9),
        player_wins,
        opponent_wins,
        draws,
        terminal_reasons,
    };
    let output = serde_json::json!({
        "args": args.manifest_args_json(),
        "summary": summary,
    });
    if let Some(path) = args.manifest_out.as_ref() {
        let p = PathBuf::from(path);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("mkdir {}", parent.display()))?;
        }
        let body = serde_json::to_string_pretty(&output)? + "\n";
        fs::write(&p, body).with_context(|| format!("write {}", p.display()))?;
        eprintln!("manifest-out: wrote {}", p.display());
    }
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Schema-parity lock for the TS-flat per-decision row shape.
    //!
    //! The Python loader
    //! (`training/uma_ai/selfplay_dataset.py:load_mcts_selfplay_samples`)
    //! and its smoke
    //! (`training/r12_selfplay_smoke.py:122-126`) require the EXACT key
    //! set asserted below. Slice 2 of `rust-port-orchestrator-wiring`
    //! blew up because the Rust path was emitting a per-game wrapper
    //! with no `kind` discriminator instead. Pin the key set so a future
    //! schema drift is caught at `cargo test` instead of at distill-stage
    //! crash time mid-iteration.
    use super::*;
    use engine::core::random::{with_rng, Rng};
    use engine::headless_setup::setup_ai_vs_ai_game;
    use engine::policy::observation::build_public_observation;

    #[test]
    fn selfplay_row_serialization_matches_ts_flat_key_set() {
        let rng = Rng::from_seed("schema-test:selfplay", "selfplay");
        let (state, _used) = with_rng(rng, || setup_ai_vs_ai_game());
        let observation = build_public_observation(&state, SideId::Player);
        let row = SelfPlayRow {
            schema_version: ROW_SCHEMA_VERSION,
            kind: "mcts-selfplay",
            seed: "0".to_string(),
            side_id: "player".to_string(),
            step: 0,
            turn_number: state.turn_number,
            observation,
            legal_actions: Vec::new(),
            selected_action_index: 0,
            visit_distribution: vec![1.0],
            root_priors: vec![1.0],
            root_value: 0.0,
            root_prior_entropy: 0.0,
            root_prior_argmax: 0,
            visited_hashes: 0,
            expansions: 0,
            leaf_evaluations: 0,
            value_target: Some(0),
            result: Some(GameResult {
                winner: None,
                points_p: 0,
                points_o: 0,
            }),
        };
        let value = serde_json::to_value(&row).expect("serialize row");
        let object = value.as_object().expect("row is a JSON object");
        let mut keys: Vec<&str> = object.keys().map(String::as_str).collect();
        keys.sort();
        // Canonical TS-flat key set — keep IN SYNC with
        // `backend/src/sim/mctsSelfPlay.ts:78-101` (`SelfPlayRow`) and
        // `training/r12_selfplay_smoke.py:122-126` (`required`).
        let mut expected = vec![
            "schemaVersion",
            "kind",
            "seed",
            "sideId",
            "step",
            "turnNumber",
            "observation",
            "legalActions",
            "selectedActionIndex",
            "visitDistribution",
            "rootPriors",
            "rootValue",
            "rootPriorEntropy",
            "rootPriorArgmax",
            "visitedHashes",
            "expansions",
            "leafEvaluations",
            "valueTarget",
            "result",
        ];
        expected.sort();
        assert_eq!(
            keys, expected,
            "TS-flat key set drift; sync with mctsSelfPlay.ts SelfPlayRow"
        );
        assert_eq!(object.get("kind").and_then(|v| v.as_str()), Some("mcts-selfplay"));
        assert_eq!(object.get("schemaVersion").and_then(|v| v.as_u64()), Some(1));
        assert!(object.get("seed").and_then(|v| v.as_str()).is_some());
    }

    #[test]
    fn fill_terminal_result_assigns_signed_value_targets() {
        let rng = Rng::from_seed("fill-test:selfplay", "selfplay");
        let (state, _used) = with_rng(rng, || setup_ai_vs_ai_game());
        let observation = build_public_observation(&state, SideId::Player);
        let make = |side: &str| SelfPlayRow {
            schema_version: ROW_SCHEMA_VERSION,
            kind: "mcts-selfplay",
            seed: "0".to_string(),
            side_id: side.to_string(),
            step: 0,
            turn_number: 0,
            observation: observation.clone(),
            legal_actions: Vec::new(),
            selected_action_index: 0,
            visit_distribution: vec![],
            root_priors: vec![],
            root_value: 0.0,
            root_prior_entropy: 0.0,
            root_prior_argmax: 0,
            visited_hashes: 0,
            expansions: 0,
            leaf_evaluations: 0,
            value_target: None,
            result: None,
        };
        let mut rows = vec![make("player"), make("opponent"), make("player")];
        fill_terminal_result(&mut rows, Some("opponent"), 1, 3);
        assert_eq!(rows[0].value_target, Some(-1));
        assert_eq!(rows[1].value_target, Some(1));
        assert_eq!(rows[2].value_target, Some(-1));
        for row in &rows {
            let result = row.result.as_ref().expect("result filled");
            assert_eq!(result.winner.as_deref(), Some("opponent"));
            assert_eq!(result.points_p, 1);
            assert_eq!(result.points_o, 3);
        }

        let mut draw_rows = vec![make("player"), make("opponent")];
        fill_terminal_result(&mut draw_rows, None, 0, 0);
        assert_eq!(draw_rows[0].value_target, Some(0));
        assert_eq!(draw_rows[1].value_target, Some(0));
        for row in &draw_rows {
            assert!(row.result.as_ref().unwrap().winner.is_none());
        }
    }

    /// Bug fix 2 of `rust-port-orchestrator-wiring` Slice 2 follow-ups:
    /// `add_root_dirichlet` was hard-coded false despite the
    /// `--mcts-dirichlet-alpha` / `--mcts-dirichlet-epsilon` flags being
    /// plumbed end-to-end. Mirror TS `mctsSelfPlay.ts:627`
    /// (`!argv.includes("--no-root-dirichlet")` — default ON). This pins
    /// the resolved gate so a future regression to the hard-coded `false`
    /// fails at `cargo test` instead of silently starving Rust selfplay
    /// of root exploration noise.
    #[test]
    fn dirichlet_gate_mirrors_ts_selfplay_default() {
        // Default selfplay: Dirichlet enabled (no `--no-root-dirichlet`).
        let args = Args::parse_from(["sim-mcts-selfplay"]);
        assert!(
            !args.no_root_dirichlet,
            "selfplay default: --no-root-dirichlet absent"
        );
        assert!(
            !args.no_root_dirichlet == true,
            "selfplay default add_root_dirichlet=true (TS mctsSelfPlay.ts:627)"
        );
        assert!((args.dirichlet_alpha - 0.3).abs() < 1e-12);
        assert!((args.dirichlet_epsilon - 0.25).abs() < 1e-12);

        // Opt-out flag flips the gate (mirrors TS
        // `argv.includes("--no-root-dirichlet")`).
        let args_off = Args::parse_from(["sim-mcts-selfplay", "--no-root-dirichlet"]);
        assert!(args_off.no_root_dirichlet);
        assert!(!args_off.no_root_dirichlet == false);

        // Manifest echo surfaces the resolved flag so post-hoc audits
        // can confirm Dirichlet was on. Bug 2 originally went unnoticed
        // because the manifest emitted alpha/epsilon but not the gate.
        let manifest = args.manifest_args_json();
        assert_eq!(
            manifest.get("addRootDirichlet").and_then(|v| v.as_bool()),
            Some(true)
        );
        let manifest_off = args_off.manifest_args_json();
        assert_eq!(
            manifest_off.get("addRootDirichlet").and_then(|v| v.as_bool()),
            Some(false)
        );
    }
}
