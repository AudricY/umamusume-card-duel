//! v3.8 Python↔Rust byte-parity fixture emitter (state surface).
//!
//! Builds 10 diverse `PublicObservation` instances exercising v3.8's
//! 8-bit slim tail at [296:304], computes each one's
//! `observation_state_features_v3_8` vector, and writes both the
//! serialized JSON and the f32 little-endian feature vector to
//! `engine-rs/crates/engine/tests/fixtures/v38_parity/`. The companion
//! Python smoke (`training/v38_python_rust_parity_smoke.py`) loads the
//! JSON, calls `observation_to_features_v3_8`, and asserts byte-identity
//! on the v3.7 head [0:296] (regression guard) AND the v3.8 tail
//! [296:304] (new contract).
//!
//! Coverage matrix (one fixture per row — see scoping doc §4.5):
//!   01_empty_initial          — fresh-setup snapshot.
//!   02_own_bench_eta          — bench[0] ETA fires for own.
//!   03_opp_bench_eta          — bench[1] ETA fires for opp.
//!   04_gust_swing_catastrophe — bit [302] = 1 (own at 1 prize, low-HP
//!                                bench KO-able by opp.active, opp.discard
//!                                has gust, opp.usedSupporter=false).
//!   05_gust_win_race          — bit [303] = 1 (symmetric, own.handCardIds
//!                                has gust + opp at 1 prize).
//!   06_terminal_state         — both sides 1 prize left; verifies the
//!                                gust predicates don't double-fire.
//!   07_mid_game               — bench populated both sides, no
//!                                catastrophes.
//!   08_no_opp_active          — opp.active = None → gust predicates 0.
//!   09_opp_supporter_used     — gust catastrophe predicate fails on (a).
//!   10_full_bench             — all 3 bench slots populated both sides.
//!
//! Regenerate with: `cargo test -p engine --test v38_python_parity_fixtures -- --ignored emit`

use std::fs;
use std::path::{Path, PathBuf};

use engine::core::constants::SideId;
use engine::core::random::{with_rng, Rng};
use engine::headless_setup::setup_ai_vs_ai_game;
use engine::policy::featurize::observation_state_features_v3_8;
use engine::policy::observation::build_public_observation;
use engine::policy::types::{PublicObservation, PublicUmaObservation, PublicUmaTurnState};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("v38_parity")
}

fn base_obs() -> PublicObservation {
    let rng = Rng::from_seed("v38-parity-fixture:selfplay", "selfplay");
    let (state, _used) = with_rng(rng, || setup_ai_vs_ai_game());
    build_public_observation(&state, SideId::Player)
}

fn make_uma_obs(
    card_id: &str,
    hp: i32,
    max_hp: i32,
    energy_total: u32,
    energies: &[(&str, u16)],
) -> PublicUmaObservation {
    let mut e = indexmap::IndexMap::new();
    for (k, v) in energies {
        e.insert((*k).to_string(), *v);
    }
    PublicUmaObservation {
        uid: 1,
        card_id: card_id.to_string(),
        species: String::new(),
        stage: 0,
        hp,
        max_hp,
        energy_total,
        energies: e,
        special_conditions: Vec::new(),
        tool_card_id: None,
        used_ability_this_turn: false,
        turn_state: PublicUmaTurnState {
            turns_in_play: 1,
            entered_this_turn: false,
            evolved_this_turn: false,
            evolved_last_turn: false,
            took_damage_last_turn: false,
            took_damage_this_turn: false,
            next_turn_damage_reduction: 0,
            attack_blocked_this_turn: false,
            paralysis_recovery_pending: false,
        },
    }
}

fn build_fixtures() -> Vec<(&'static str, PublicObservation)> {
    let mut out: Vec<(&'static str, PublicObservation)> = Vec::new();

    // 01 — empty / initial. v3.7 ETA bits may fire from the trivial
    // setup; v3.8 tail should mostly be 0 (no bench, no gust signals).
    out.push(("01_empty_initial", base_obs()));

    // 02 — own bench[0] ETA. matikanetannhauserBasic on the bench →
    // its primary cost {psychic:1} reduces to vacuously-satisfied
    // after the +1 attach budget → own_bench[0]_eta=1.
    let mut o = base_obs();
    let bench_uma = make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[]);
    o.own.bench = vec![Some(bench_uma)];
    out.push(("02_own_bench_eta", o));

    // 03 — opp bench[1] ETA. Symmetric, slot 1 of opp.bench populated.
    let mut o = base_obs();
    let bench_uma = make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[]);
    o.opponent.bench = vec![None, Some(bench_uma)];
    out.push(("03_opp_bench_eta", o));

    // 04 — gust-swing catastrophe ([302]=1). All prerequisites:
    //   own.points=2 (1 prize left, losing one more loses the game),
    //   own.bench=[low-HP haruUraraBasic 10/90 — KO-able by attacker],
    //   opp.active=manhattanCafeStage1 (Darkness 40 damage),
    //   opp.discard=[yayoiAkikawa] (gust trainer per cards.json),
    //   opp.usedSupporterThisTurn=false,
    //   opp.handCount=5 (>0).
    let mut o = base_obs();
    o.own.points = 2;
    o.own.active = Some(make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[]));
    let low_hp_bench = make_uma_obs("haruUraraBasic", 10, 90, 0, &[]);
    o.own.bench = vec![Some(low_hp_bench)];
    o.opponent.active = Some(make_uma_obs(
        "manhattanCafeStage1",
        90,
        90,
        2,
        &[("darkness", 1), ("colorless", 1)],
    ));
    o.opponent.discard = vec!["yayoiAkikawa".to_string()];
    o.opponent.used_supporter_this_turn = false;
    o.opponent.hand_count = 5;
    out.push(("04_gust_swing_catastrophe", o));

    // 05 — gust-win race ([303]=1). Symmetric.
    let mut o = base_obs();
    o.opponent.points = 2;
    o.opponent.active = Some(make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[]));
    let low_hp_bench = make_uma_obs("haruUraraBasic", 10, 90, 0, &[]);
    o.opponent.bench = vec![Some(low_hp_bench)];
    o.own.active = Some(make_uma_obs(
        "manhattanCafeStage1",
        90,
        90,
        2,
        &[("darkness", 1), ("colorless", 1)],
    ));
    // own.hand_card_ids IS exposed (own-side perspective).
    o.own.hand_card_ids = Some(vec!["yayoiAkikawa".to_string()]);
    out.push(("05_gust_win_race", o));

    // 06 — both sides at 1 prize, symmetric vulnerability. Tests that
    // the predicates compute INDEPENDENTLY on each side without
    // cross-contamination.
    let mut o = base_obs();
    o.own.points = 2;
    o.opponent.points = 2;
    o.own.active = Some(make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[]));
    o.opponent.active = Some(make_uma_obs(
        "manhattanCafeStage1",
        90,
        90,
        2,
        &[("darkness", 1), ("colorless", 1)],
    ));
    o.own.bench = vec![Some(make_uma_obs("haruUraraBasic", 10, 90, 0, &[]))];
    o.opponent.bench = vec![Some(make_uma_obs("haruUraraBasic", 10, 90, 0, &[]))];
    o.opponent.discard = vec!["yayoiAkikawa".to_string()];
    o.opponent.used_supporter_this_turn = false;
    o.opponent.hand_count = 5;
    o.own.hand_card_ids = Some(vec!["yayoiAkikawa".to_string()]);
    out.push(("06_terminal_state", o));

    // 07 — mid-game: bench populated both sides, no catastrophes.
    let mut o = base_obs();
    o.own.bench = vec![
        Some(make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[])),
        Some(make_uma_obs("haruUraraBasic", 90, 90, 0, &[])),
    ];
    o.opponent.bench = vec![
        Some(make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[])),
        Some(make_uma_obs("haruUraraBasic", 90, 90, 0, &[])),
    ];
    out.push(("07_mid_game", o));

    // 08 — opp.active = None. Gust catastrophe predicate (b) requires
    // opp.active → bit [302] must be 0 even if other prerequisites are
    // met.
    let mut o = base_obs();
    o.own.points = 2;
    o.own.bench = vec![Some(make_uma_obs("haruUraraBasic", 10, 90, 0, &[]))];
    o.opponent.active = None;
    o.opponent.discard = vec!["yayoiAkikawa".to_string()];
    o.opponent.used_supporter_this_turn = false;
    o.opponent.hand_count = 5;
    out.push(("08_no_opp_active", o));

    // 09 — opp supporter already used. Gust proxy (a) fails → [302]=0
    // even with bench KO-able + own at 1 prize + gust in discard.
    let mut o = base_obs();
    o.own.points = 2;
    o.own.active = Some(make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[]));
    o.own.bench = vec![Some(make_uma_obs("haruUraraBasic", 10, 90, 0, &[]))];
    o.opponent.active = Some(make_uma_obs(
        "manhattanCafeStage1",
        90,
        90,
        2,
        &[("darkness", 1), ("colorless", 1)],
    ));
    o.opponent.discard = vec!["yayoiAkikawa".to_string()];
    o.opponent.used_supporter_this_turn = true; // disables the proxy
    o.opponent.hand_count = 5;
    out.push(("09_opp_supporter_used", o));

    // 10 — full bench on both sides (3 slots each).
    let mut o = base_obs();
    o.own.bench = vec![
        Some(make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[])),
        Some(make_uma_obs("haruUraraBasic", 90, 90, 0, &[])),
        Some(make_uma_obs("matikanefukukitaruStage1", 100, 100, 0, &[])),
    ];
    o.opponent.bench = vec![
        Some(make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[])),
        Some(make_uma_obs("haruUraraBasic", 90, 90, 0, &[])),
        Some(make_uma_obs("symboliRudolfStage2", 120, 120, 0, &[])),
    ];
    o.own.energy_pool = vec!["psychic".to_string(), "dragon".to_string()];
    o.opponent.energy_pool = vec!["psychic".to_string(), "water".to_string()];
    out.push(("10_full_bench", o));

    out
}

fn write_fixture(dir: &Path, name: &str, obs: &PublicObservation) {
    let vec = observation_state_features_v3_8(obs);
    assert_eq!(
        vec.len(),
        engine::policy::featurize::STATE_DIM_V3_8,
        "fixture {} vector width mismatch",
        name
    );

    let json_path = dir.join(format!("{}.json", name));
    let bin_path = dir.join(format!("{}.f32.bin", name));

    let json = serde_json::to_string_pretty(obs)
        .unwrap_or_else(|e| panic!("serialize {} obs: {}", name, e));
    fs::write(&json_path, json).unwrap_or_else(|e| panic!("write {:?}: {}", json_path, e));

    let mut bytes: Vec<u8> = Vec::with_capacity(vec.len() * 4);
    for v in &vec {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(&bin_path, bytes).unwrap_or_else(|e| panic!("write {:?}: {}", bin_path, e));
}

#[test]
#[ignore]
fn emit_v38_parity_fixtures() {
    let dir = fixtures_dir();
    fs::create_dir_all(&dir).expect("create fixtures dir");

    let readme = format!(
        "v3.8 Python↔Rust byte-parity fixtures.\n\
         Regenerate with: cargo test -p engine --test v38_python_parity_fixtures \
         -- --ignored emit\n\
         Validate with: python training/v38_python_rust_parity_smoke.py\n\
         STATE_DIM_V3_8 = {}\n",
        engine::policy::featurize::STATE_DIM_V3_8,
    );
    fs::write(dir.join("README.txt"), readme).expect("write README");

    let fixtures = build_fixtures();
    assert!(
        fixtures.len() >= 10,
        "v3.8 fixture set expected ≥10 fixtures, got {}",
        fixtures.len()
    );
    for (name, obs) in &fixtures {
        write_fixture(&dir, name, obs);
    }
    eprintln!("emitted {} fixtures to {:?}", fixtures.len(), dir);
}

#[test]
fn fixtures_roundtrip_against_current_rust_featurizer() {
    let dir = fixtures_dir();
    if !dir.exists() {
        eprintln!("fixtures dir {:?} does not exist; skipping roundtrip", dir);
        return;
    }
    let mut checked = 0usize;
    let mut entries: Vec<PathBuf> = fs::read_dir(&dir)
        .expect("read fixtures dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
        .collect();
    entries.sort();
    for json_path in entries {
        let name = json_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap()
            .to_string();
        let bin_path = dir.join(format!("{}.f32.bin", name));
        assert!(bin_path.exists(), "missing {:?}", bin_path);

        let json = fs::read_to_string(&json_path)
            .unwrap_or_else(|e| panic!("read {:?}: {}", json_path, e));
        let obs: PublicObservation =
            serde_json::from_str(&json).unwrap_or_else(|e| panic!("deserialize {}: {}", name, e));
        let live = observation_state_features_v3_8(&obs);
        let bytes = fs::read(&bin_path).unwrap_or_else(|e| panic!("read {:?}: {}", bin_path, e));
        assert_eq!(
            bytes.len(),
            live.len() * 4,
            "{}: bin byte-length mismatch",
            name
        );
        let saved: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(saved.len(), live.len(), "{}: vector width mismatch", name);
        for (i, (&a, &b)) in saved.iter().zip(live.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{}: slot {} drift saved=0x{:08x} live=0x{:08x} ({} vs {})",
                name,
                i,
                a.to_bits(),
                b.to_bits(),
                a,
                b
            );
        }
        checked += 1;
    }
    assert!(checked > 0, "no fixtures found to roundtrip-check");
    eprintln!(
        "roundtrip OK: {} fixtures match current Rust featurizer",
        checked
    );
}
