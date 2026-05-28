//! Microbenchmarks paralleling `backend/src/sim/throughputProbe.ts`'s
//! micro section. Run with:
//!
//! ```bash
//! cargo bench --manifest-path engine-rs/Cargo.toml -p engine
//! ```
//!
//! Target numbers from the TS probe (single-core, legalActionCount=1)
//! recorded in `runs/throughput-probe/probe-default.json`:
//! - cloneNsPerCall: ~22,007 ns (TS `structuredClone`)
//! - fingerprintNsPerCall: ~8,034 ns (TS `JSON.stringify`)
//!
//! The Rust port's targets are sub-microsecond for both, since:
//! - `GameState` clone is a `Clone` derive over fixed-cap ArrayVecs +
//!   String fields (the heap-y bits) and primitive arrays elsewhere.
//! - Fingerprint is xxh3 over a packed buffer assembled in one pass.

use arrayvec::ArrayVec;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use engine::core::card_id::CardId;
use engine::core::constants::{AiDeckStyle, AiDifficulty, EnergyType, SideId};
use engine::core::packed::pack;
use engine::core::state::{CurrentSide, GameState, Phase, SideState, UmamusumeInstance};
use engine::fingerprint::fingerprint;

fn build_realistic_state() -> GameState {
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
        // Stuff in 14 deck + 5 hand + 1 bench to feel realistic.
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

fn bench_clone(c: &mut Criterion) {
    let state = build_realistic_state();
    c.bench_function("game_state_clone", |b| {
        b.iter(|| {
            let cloned = black_box(state.clone());
            black_box(cloned);
        })
    });
}

fn bench_pack(c: &mut Criterion) {
    let state = build_realistic_state();
    c.bench_function("pack_state", |b| {
        b.iter(|| {
            let buf = pack(black_box(&state));
            black_box(buf);
        })
    });
}

fn bench_fingerprint(c: &mut Criterion) {
    let state = build_realistic_state();
    let buf = pack(&state);
    c.bench_function("fingerprint_packed", |b| {
        b.iter(|| {
            let fp = fingerprint(black_box(&buf));
            black_box(fp);
        })
    });
}

fn bench_pack_plus_fingerprint(c: &mut Criterion) {
    let state = build_realistic_state();
    c.bench_function("pack_plus_fingerprint", |b| {
        b.iter(|| {
            let buf = pack(black_box(&state));
            let fp = fingerprint(&buf);
            black_box((buf, fp));
        })
    });
}

criterion_group!(
    benches,
    bench_clone,
    bench_pack,
    bench_fingerprint,
    bench_pack_plus_fingerprint,
);
criterion_main!(benches);
