//! ReBeL public-belief self-play driver.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use engine::belief::{
    build_public_belief_state, BeliefBuildConfig, BeliefFeatures, PublicHistory, BELIEF_SCHEMA_VERSION,
};
use engine::core::constants::SideId;
use engine::core::random::{with_rng, Rng};
use engine::core::state::{CurrentSide, GameState};
use engine::deck_sampling::{manifest_pair_for, DeckSampling};
use engine::dispatcher::{
    advance_modeled_turn_step, advance_opponent_turn_step, advance_player_ai_turn_step,
    get_forced_attack_coin_results, state_hash,
};
use engine::headless_setup::setup_ai_vs_ai_game_with_decks;
use engine::policy::actions::enumerate_legal_ai_actions;
use engine::policy::observation::build_public_observation;
use engine::policy::types::{LegalAiAction, PublicObservation};
use engine::rebel::{run_public_belief_search, BeliefSearchResult, RebelSearchConfig};
use serde::Serialize;

const ROW_SCHEMA_VERSION: u32 = 1;

#[derive(Parser, Debug)]
#[command(
    name = "sim-rebel-selfplay",
    about = "Generate ReBeL public-belief self-play rows"
)]
struct Args {
    #[arg(long, alias = "games", default_value_t = 1)]
    seeds: u32,
    #[arg(long, alias = "seed-start", default_value_t = 0)]
    seed_base: u32,
    #[arg(long, default_value_t = 64)]
    particles: usize,
    #[arg(long, alias = "search-iterations", default_value_t = 64)]
    iterations: u32,
    #[arg(long, default_value_t = 120)]
    rollout_steps: u32,
    #[arg(long, default_value_t = 1)]
    max_depth: u32,
    #[arg(long, default_value = "player")]
    model_side: String,
    #[arg(long, default_value_t = 6)]
    temperature_moves: u32,
    #[arg(long, default_value_t = 1.0)]
    temperature_value: f64,
    #[arg(long, default_value_t = 200)]
    max_steps: u32,
    #[arg(long, default_value = "fixed")]
    deck_sampling: String,
    #[arg(long)]
    out: Option<String>,
    #[arg(long)]
    manifest_out: Option<String>,
    #[arg(long, default_value_t = 1)]
    workers: u32,
}

impl Args {
    fn manifest_args_json(&self) -> serde_json::Value {
        serde_json::json!({
            "seeds": self.seeds,
            "seedBase": self.seed_base,
            "particles": self.particles,
            "iterations": self.iterations,
            "rolloutSteps": self.rollout_steps,
            "maxDepth": self.max_depth,
            "modelSide": self.model_side,
            "temperatureMoves": self.temperature_moves,
            "temperatureValue": self.temperature_value,
            "maxSteps": self.max_steps,
            "deckSampling": self.deck_sampling,
            "out": self.out,
            "manifestOut": self.manifest_out,
            "workers": self.workers,
            "dataMode": "rebel",
            "engine": "rust",
            "selfplayBinary": "sim-rebel-selfplay",
            "search": "public-belief-cfr-v1",
        })
    }
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
struct GameResult {
    winner: Option<String>,
    #[serde(rename = "pointsP")]
    points_p: u8,
    #[serde(rename = "pointsO")]
    points_o: u8,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RebelSelfPlayRow {
    kind: &'static str,
    schema_version: u32,
    belief_schema_version: u32,
    seed: String,
    side_id: String,
    step: u32,
    turn_number: u32,
    observation: PublicObservation,
    belief_features: BeliefFeatures,
    public_history_digest: String,
    public_history: PublicHistory,
    legal_actions: Vec<LegalAiAction>,
    selected_action_index: usize,
    search_policy: Vec<f64>,
    search_action_values: Vec<f64>,
    belief_value: f64,
    private_state_values: Vec<f64>,
    value_target: Option<i8>,
    particle_count: usize,
    search_iterations: u32,
    search_algorithm: String,
    belief_sampler: &'static str,
    search_diagnostics: serde_json::Value,
    belief_audit: serde_json::Value,
    result: Option<GameResult>,
    player_deck_id: String,
    opponent_deck_id: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GameRecord {
    seed: u32,
    terminal_reason: String,
    turn_number: u32,
    winner: Option<String>,
    total_steps: u32,
    model_decisions: u32,
    player_deck_id: String,
    opponent_deck_id: String,
    rows: Vec<RebelSelfPlayRow>,
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
}

fn fill_terminal_result(rows: &mut [RebelSelfPlayRow], winner: Option<&str>, points_p: u8, points_o: u8) {
    let result = GameResult {
        winner: winner.map(|s| s.to_string()),
        points_p,
        points_o,
    };
    for row in rows {
        row.value_target = Some(match winner {
            None => 0,
            Some(w) if w == row.side_id => 1,
            Some(_) => -1,
        });
        row.result = Some(result.clone());
    }
}

#[allow(clippy::too_many_arguments)]
fn drive_one_game(
    seed: u32,
    max_steps: u32,
    search_config: &RebelSearchConfig,
    particle_count: usize,
    model_side: SideId,
    temperature_moves: u32,
    temperature_value: f64,
    player_deck: Option<&[engine::core::card_id::CardId]>,
    opponent_deck: Option<&[engine::core::card_id::CardId]>,
    player_deck_id: &str,
    opponent_deck_id: &str,
) -> GameRecord {
    let seed_str = seed.to_string();
    let rng = Rng::from_seed(format!("{}:rebel-selfplay", seed_str), "rebel-selfplay");
    let (mut state, mut step_rng) = with_rng(rng, || {
        setup_ai_vs_ai_game_with_decks(player_deck, opponent_deck)
    });
    let mut rows = Vec::new();
    let mut terminal = "max_steps";
    let mut step = 0u32;
    let mut model_decisions = 0u32;
    let mut model_moves = 0u32;

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
        let (legal, used_rng) = with_rng(step_rng.clone(), || enumerate_legal_ai_actions(&state, side));
        step_rng = used_rng;
        let pre_hash = state_hash(&state);

        let next_state = if side == model_side && legal.len() > 1 {
            let history = PublicHistory::from_state(&state, player_deck_id, opponent_deck_id, s);
            let mut belief_rng = step_rng.fork(format!("belief:{}", s).as_str());
            let belief = build_public_belief_state(
                &state,
                side,
                history,
                &BeliefBuildConfig {
                    particle_count,
                    seed_label: format!("{}:belief:{}", seed_str, s),
                },
                &mut belief_rng,
            );
            let mut search_rng = step_rng.fork(format!("search:{}", s).as_str());
            let search = run_public_belief_search(&belief, search_config, &mut search_rng);
            let greedy_index = argmax_f64(&search.root_action_values).min(legal.len() - 1);
            let selected_action_index = if model_moves < temperature_moves && temperature_value > 0.0 {
                search.sampled_action_index.min(legal.len() - 1)
            } else {
                greedy_index
            };
            let chosen = legal[selected_action_index].clone();
            rows.push(row_from_search(
                &seed_str,
                s,
                &state,
                side,
                selected_action_index,
                belief,
                search,
                player_deck_id,
                opponent_deck_id,
            ));
            model_decisions += 1;
            model_moves += 1;
            let (ns, used_rng) = with_rng(step_rng.clone(), || {
                let forced = get_forced_attack_coin_results(&state);
                advance_modeled_turn_step(&state, side, &chosen, forced)
            });
            step_rng = used_rng;
            ns
        } else {
            let (ns, used_rng) = with_rng(step_rng.clone(), || {
                let forced = get_forced_attack_coin_results(&state);
                let mut s2 = state.clone();
                if side == SideId::Player {
                    advance_player_ai_turn_step(&mut s2, forced);
                } else {
                    advance_opponent_turn_step(&mut s2, forced);
                }
                s2
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
        player_deck_id: player_deck_id.to_string(),
        opponent_deck_id: opponent_deck_id.to_string(),
        rows,
    }
}

#[allow(clippy::too_many_arguments)]
fn row_from_search(
    seed: &str,
    step: u32,
    state: &GameState,
    side: SideId,
    selected_action_index: usize,
    belief: engine::belief::PublicBeliefState,
    search: BeliefSearchResult,
    player_deck_id: &str,
    opponent_deck_id: &str,
) -> RebelSelfPlayRow {
    RebelSelfPlayRow {
        kind: "rebel-selfplay",
        schema_version: ROW_SCHEMA_VERSION,
        belief_schema_version: BELIEF_SCHEMA_VERSION,
        seed: seed.to_string(),
        side_id: side_label(side).to_string(),
        step,
        turn_number: state.turn_number,
        observation: build_public_observation(state, side),
        belief_features: belief.belief_features,
        public_history_digest: belief.public_history_digest,
        public_history: belief.public_history,
        legal_actions: search.legal_actions,
        selected_action_index,
        search_policy: search.root_policy,
        search_action_values: search.root_action_values,
        belief_value: search.public_belief_value,
        private_state_values: search.private_state_values,
        value_target: None,
        particle_count: search.diagnostics.particle_count,
        search_iterations: search.diagnostics.search_iterations,
        search_algorithm: search.search_algorithm,
        belief_sampler: "public-history-particles-v1",
        search_diagnostics: serde_json::to_value(search.diagnostics).unwrap_or(serde_json::Value::Null),
        belief_audit: serde_json::to_value(belief.audit).unwrap_or(serde_json::Value::Null),
        result: None,
        player_deck_id: player_deck_id.to_string(),
        opponent_deck_id: opponent_deck_id.to_string(),
    }
}

fn side_label(side: SideId) -> &'static str {
    match side {
        SideId::Player => "player",
        SideId::Opponent => "opponent",
    }
}

fn argmax_f64(values: &[f64]) -> usize {
    let mut best = 0usize;
    let mut best_value = f64::NEG_INFINITY;
    for (i, v) in values.iter().enumerate() {
        if *v > best_value {
            best = i;
            best_value = *v;
        }
    }
    best
}

fn main() -> Result<()> {
    let args = Args::parse();
    let sampling = DeckSampling::parse(&args.deck_sampling)
        .map_err(|e| anyhow::anyhow!("--deck-sampling: {}", e))?;
    if args.workers != 1 {
        eprintln!("sim-rebel-selfplay: --workers is accepted for orchestrator compatibility; this first driver runs sequentially");
    }
    let model_sides: Vec<SideId> = match args.model_side.as_str() {
        "player" => vec![SideId::Player],
        "opponent" => vec![SideId::Opponent],
        "both" => vec![SideId::Player, SideId::Opponent],
        other => anyhow::bail!("--model-side must be player, opponent, or both (got {})", other),
    };
    let search_config = RebelSearchConfig {
        iterations: args.iterations,
        max_depth: args.max_depth,
        rollout_steps: args.rollout_steps,
        algorithm: "public-belief-cfr-v1".to_string(),
    };
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
    let mut games = 0u32;

    for &side in &model_sides {
        for i in 0..args.seeds {
            let seed = args.seed_base + i;
            let task_index = games;
            let resolved = sampling.resolve(args.seed_base, task_index);
            let (player_deck_opt, opponent_deck_opt) = resolved
                .as_ref()
                .map(|p| (Some(p.player_deck), Some(p.opponent_deck)))
                .unwrap_or((None, None));
            let (player_deck_id, opponent_deck_id) =
                manifest_pair_for(&sampling, args.seed_base, task_index);
            let record = drive_one_game(
                seed,
                args.max_steps,
                &search_config,
                args.particles,
                side,
                args.temperature_moves,
                args.temperature_value,
                player_deck_opt,
                opponent_deck_opt,
                player_deck_id,
                opponent_deck_id,
            );
            match record.winner.as_deref() {
                Some("player") => player_wins += 1,
                Some("opponent") => opponent_wins += 1,
                _ => draws += 1,
            }
            if let Some(w) = writer.as_mut() {
                for row in &record.rows {
                    writeln!(w, "{}", serde_json::to_string(row)?)?;
                }
            }
            games += 1;
        }
    }

    let elapsed_secs = start.elapsed().as_secs_f64();
    let summary = RunSummary {
        games,
        elapsed_secs,
        games_per_sec: games as f64 / elapsed_secs.max(1e-9),
        player_wins,
        opponent_wins,
        draws,
    };
    let output = serde_json::json!({
        "args": args.manifest_args_json(),
        "summary": summary,
        "manifestName": "R17-rebel-e2e",
    });
    if let Some(path) = args.manifest_out.as_ref() {
        let p = PathBuf::from(path);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).with_context(|| format!("mkdir {}", parent.display()))?;
        }
        fs::write(&p, serde_json::to_string_pretty(&output)? + "\n")
            .with_context(|| format!("write {}", p.display()))?;
    }
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn rebel_args_default_manifest_identifies_new_line() {
        let args = Args::parse_from(["sim-rebel-selfplay"]);
        let manifest = args.manifest_args_json();
        assert_eq!(manifest.get("dataMode").and_then(Value::as_str), Some("rebel"));
        assert_eq!(
            manifest.get("selfplayBinary").and_then(Value::as_str),
            Some("sim-rebel-selfplay")
        );
        assert_eq!(manifest.get("search").and_then(Value::as_str), Some("public-belief-cfr-v1"));
    }
}
