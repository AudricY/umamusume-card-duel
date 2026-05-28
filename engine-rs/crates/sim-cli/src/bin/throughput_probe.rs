//! Rust port of `backend/src/sim/throughputProbe.ts` — microbenchmark
//! portion only (full sim throughput waits on Phase 1e/1f/1g).
//!
//! Output JSON mirrors the TS schema where the underlying op is ported;
//! emits `null` for fields that depend on the unported action enumerator.
//! Consumers of `runs/throughput-probe/probe-default.json` should treat
//! missing-or-null fields as "not yet available in Rust" and continue
//! using the TS probe for those.

use std::time::Instant;

use anyhow::Result;
use arrayvec::ArrayVec;
use clap::Parser;
use engine::core::card_id::CardId;
use engine::core::constants::{AiDeckStyle, AiDifficulty, EnergyType, SideId};
use engine::core::packed::pack;
use engine::core::random::{with_rng, Rng};
use engine::core::state::{CurrentSide, GameState, Phase, SideState, UmamusumeInstance};
use engine::fingerprint::fingerprint;
use serde::Serialize;
use serde_json::Value as JsonValue;

#[derive(Parser, Debug)]
#[command(
    name = "sim-throughput-probe",
    about = "Rust port of backend/src/sim/throughputProbe.ts (Phase 1 microbench only)"
)]
struct Args {
    #[arg(long, default_value = "default")]
    profile: String,

    /// Sample count for each microbenchmark. TS default is 5000.
    #[arg(long, default_value_t = 5_000)]
    micro_samples: usize,

    /// Seed for the micro-bench state setup (matches TS naming).
    #[arg(long, default_value = "throughput-micro-default")]
    micro_seed: String,

    #[arg(long, default_value_t = 1)]
    legal_action_count: u32,

    /// Output JSON path. If absent, prints to stdout.
    #[arg(long)]
    out: Option<String>,
}

#[derive(Serialize)]
struct ProbeOutput {
    args: ArgsView,
    summary: Summary,
}

#[derive(Serialize)]
struct ArgsView {
    profile: String,
    micro_samples: usize,
    micro_seed: String,
    legal_action_count: u32,
}

#[derive(Serialize)]
struct Summary {
    micro: Micro,
    notes: Vec<String>,
}

#[derive(Serialize)]
struct Micro {
    samples: usize,
    legal_action_count: u32,
    /// `null` until the legal-action enumerator lands (Phase 1e + 1f).
    enumerate_ns_per_call: Option<f64>,
    /// `null` until the action enumerator + scoring lands.
    choose_ns_per_call: Option<f64>,
    fingerprint_ns_per_call: f64,
    clone_ns_per_call: f64,
}

fn build_realistic_state() -> GameState {
    // Same fixture as the Criterion bench at
    // engine-rs/crates/engine/benches/clone_and_fingerprint.rs. Active +
    // bench + 14-deep deck + 5-deep hand + energy zone + points==1, all
    // sized to mirror a typical mid-game MCTS clone.
    let make_inst = |uid: u32| UmamusumeInstance {
        uid,
        card_id: CardId(uid as u16),
        evolution_card_ids: ArrayVec::new(),
        stage: 0,
        hp: 60,
        max_hp: 60,
        energies: {
            let mut a = [0u16; EnergyType::COUNT];
            a[EnergyType::Psychic as usize] = 2;
            a[EnergyType::Colorless as usize] = 1;
            a
        },
        special_conditions: ArrayVec::new(),
        entered_turn: 1,
        evolved_turn: None,
        took_damage_last_turn: false,
        took_damage_this_turn: false,
        next_turn_damage_reduction: 0,
        used_ability_this_turn: false,
        attack_blocked_until_own_turn: None,
        paralysed_until_own_turn: None,
        tool_card_id: None,
    };
    let make_side = |id: SideId, uid_base: u32| {
        let mut s = SideState {
            id,
            title: format!("Side{:?}", id),
            energy_pool: ArrayVec::from_iter([EnergyType::Psychic, EnergyType::Colorless]),
            deck: ArrayVec::new(),
            discard: ArrayVec::new(),
            hand: ArrayVec::new(),
            active: Some(make_inst(uid_base)),
            bench: ArrayVec::new(),
            points: 1,
            energy_zone: ArrayVec::from_iter([EnergyType::Psychic]),
            energy_attachments_this_turn: 1,
            bonus_energy_attachments: 0,
            retreat_cost_reduction: 0,
            active_attack_damage_bonus: 0,
            used_supporter_this_turn: false,
            used_retreat_this_turn: false,
            used_stadium_this_turn: false,
            used_ability_names_this_turn: ArrayVec::new(),
            used_ability_names_this_game: ArrayVec::new(),
            guaranteed_coin_flip_heads: 0,
        };
        for i in 0..14 {
            let _ = s.deck.try_push(CardId(i as u16));
        }
        for i in 0..5 {
            let _ = s.hand.try_push(CardId((i + 20) as u16));
        }
        let _ = s.bench.try_push(make_inst(uid_base + 1));
        s
    };
    GameState {
        phase: Phase::Play,
        setup: None,
        pending_player_choice: None,
        sides: [
            make_side(SideId::Player, 1),
            make_side(SideId::Opponent, 100),
        ],
        current_side: CurrentSide::Player,
        opponent_turn_step: None,
        stadium: None,
        turn_deadline_ms: None,
        turn_number: 3,
        first_player: SideId::Player,
        turns_taken_by_side: [2, 1],
        ai_difficulty: AiDifficulty::Normal,
        human_by_side: [false, false],
        ai_deck_style_by_side: [AiDeckStyle::Balanced, AiDeckStyle::Balanced],
        game_over: false,
        winner: None,
        log: std::collections::VecDeque::new(),
    }
}

fn time_repeat(samples: usize, mut f: impl FnMut()) -> f64 {
    // Match TS `timeRepeat`: elapsed-ms * 1e6 / samples → ns per call.
    let start = Instant::now();
    for _ in 0..samples {
        f();
    }
    let elapsed_ns = start.elapsed().as_nanos() as f64;
    elapsed_ns / samples as f64
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Build the state inside a with_rng scope so any RNG-consuming
    // operations during setup advance the seeded PRNG. The state itself
    // does not depend on RNG today, but the contract matches TS.
    let rng = Rng::from_seed(args.micro_seed.as_str(), "throughput-micro");
    let (state, _used_rng) = with_rng(rng, build_realistic_state);

    // Warm-up — match TS's 50-iteration warm.
    for _ in 0..50 {
        let _ = fingerprint(&pack(&state));
        let _ = state.clone();
    }

    let fingerprint_ns = time_repeat(args.micro_samples, || {
        let buf = pack(&state);
        let _ = fingerprint(&buf);
    });

    let clone_ns = time_repeat(args.micro_samples, || {
        let _ = state.clone();
    });

    let micro = Micro {
        samples: args.micro_samples,
        legal_action_count: args.legal_action_count,
        enumerate_ns_per_call: None,
        choose_ns_per_call: None,
        fingerprint_ns_per_call: fingerprint_ns,
        clone_ns_per_call: clone_ns,
    };

    let notes = vec![
        "Rust port: micro section only. Full sim throughput pending \
         Phase 1e (action enumerator) + Phase 1f (heuristic opponent) + \
         Phase 1g (MCTS port)."
            .to_string(),
        format!(
            "Compare to TS probe-default.json: cloneNsPerCall ~22007, \
             fingerprintNsPerCall ~8034. Current Rust clone={:.0}, \
             pack+fingerprint={:.0}.",
            clone_ns, fingerprint_ns
        ),
    ];

    let output = ProbeOutput {
        args: ArgsView {
            profile: args.profile.clone(),
            micro_samples: args.micro_samples,
            micro_seed: args.micro_seed.clone(),
            legal_action_count: args.legal_action_count,
        },
        summary: Summary { micro, notes },
    };

    let json = serde_json::to_string_pretty(&output)?;
    match args.out {
        Some(path) => std::fs::write(&path, json)?,
        None => println!("{}", json),
    }

    let _ = JsonValue::Null; // silence unused import in future stubs
    Ok(())
}
