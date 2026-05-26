//! v3.6 Python↔Rust byte-parity fixture emitter.
//!
//! Builds N diverse `PublicObservation` instances, computes each one's
//! `observation_state_features_v3_6` vector, and writes both the
//! serialized JSON and the f32 little-endian feature vector to
//! `engine-rs/crates/engine/tests/fixtures/v36_parity/`. The companion
//! Python smoke (`training/v36_python_rust_parity_smoke.py`) loads the
//! JSON, calls `observation_to_features_v3_6`, and asserts byte-identity
//! against the Rust-emitted vector.
//!
//! Coverage matrix (one fixture per row):
//!   01_setup_baseline   — fresh-setup snapshot (no mutations).
//!   02_empty_pools      — both sides have empty energy_pool.
//!   03_one_type_pool    — own pool=[fire]; opp pool=[psychic].
//!   04_three_type_pool  — own pool=[fire,water,lightning]; opp pool=
//!                         [psychic,darkness,steel].
//!   05_dup_pool         — own pool=[fire,fire,water] (duplicate collapse).
//!   06_mid_game_prizes  — own.points=2, opp.points=1.
//!   07_lethal_true      — opp active 60-dmg vs own hp=60 → own_lethal=1.
//!   08_lethal_false     — opp active 20-dmg vs own hp=120 → own_lethal=0.
//!   09_secondary_attack — matikanefukukitaruStage1 (2 attacks) with
//!                         energy cover for the secondary; defender hp
//!                         tuned so would_KO=0 (secondary base=0 dmg).
//!   10_opp_bench_typed  — opp bench with 2 Umas carrying mixed typed
//!                         energies; own bench cleared (own bench is
//!                         dropped from v3.6 by reconciliation).
//!
//! Regenerate with: `cargo test -p engine --test v36_python_parity_fixtures -- --ignored emit`
//! The `emit` test is `#[ignore]` so it doesn't run in the default
//! `cargo test` cycle (it writes to the tree); to validate after a
//! featurizer change you re-emit + re-run the Python smoke. The
//! `roundtrip_check` test runs by default and re-verifies that the
//! committed fixtures still match the current Rust output (catches
//! accidental Rust-side drift).

use std::fs;
use std::path::{Path, PathBuf};

use engine::core::constants::SideId;
use engine::core::random::{with_rng, Rng};
use engine::headless_setup::setup_ai_vs_ai_game;
use engine::policy::featurize::observation_state_features_v3_6;
use engine::policy::observation::build_public_observation;
use engine::policy::types::{
    PublicObservation, PublicUmaObservation, PublicUmaTurnState,
};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("v36_parity")
}

fn base_obs() -> PublicObservation {
    let rng = Rng::from_seed("v36-parity-fixture:selfplay", "selfplay");
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

/// Build the full ordered fixture list. Each `(name, obs)` round-trips
/// through both serializers below.
fn build_fixtures() -> Vec<(&'static str, PublicObservation)> {
    let mut out: Vec<(&'static str, PublicObservation)> = Vec::new();

    // 01 — baseline straight from setup.
    out.push(("01_setup_baseline", base_obs()));

    // 02 — both pools explicitly empty.
    let mut o = base_obs();
    o.own.energy_pool.clear();
    o.opponent.energy_pool.clear();
    out.push(("02_empty_pools", o));

    // 03 — one-type pool both sides.
    let mut o = base_obs();
    o.own.energy_pool = vec!["fire".to_string()];
    o.opponent.energy_pool = vec!["psychic".to_string()];
    out.push(("03_one_type_pool", o));

    // 04 — three-type pool both sides.
    let mut o = base_obs();
    o.own.energy_pool = vec![
        "fire".to_string(),
        "water".to_string(),
        "lightning".to_string(),
    ];
    o.opponent.energy_pool = vec![
        "psychic".to_string(),
        "darkness".to_string(),
        "steel".to_string(),
    ];
    out.push(("04_three_type_pool", o));

    // 05 — duplicates collapse.
    let mut o = base_obs();
    o.own.energy_pool = vec![
        "fire".to_string(),
        "fire".to_string(),
        "water".to_string(),
    ];
    o.opponent.energy_pool = vec!["psychic".to_string(), "psychic".to_string()];
    out.push(("05_dup_pool", o));

    // 06 — mid-game prizes (one bit at non-zero index per side).
    let mut o = base_obs();
    o.own.points = 2;
    o.opponent.points = 1;
    out.push(("06_mid_game_prizes", o));

    // 07 — lethal-true case. Stage 2 attacker (60 dmg) vs hp=60 defender.
    let mut o = base_obs();
    let attacker = make_uma_obs("matikanetannhauserStage2", 120, 120, 3,
        &[("psychic", 2), ("colorless", 1)]);
    let defender = make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[]);
    o.opponent.active = Some(attacker);
    o.own.active = Some(defender);
    out.push(("07_lethal_true", o));

    // 08 — lethal-false case (basic 20-dmg attacker vs hp=120 defender).
    let mut o = base_obs();
    let weak_attacker = make_uma_obs("matikanetannhauserBasic", 60, 60, 1,
        &[("psychic", 1)]);
    let tough_defender = make_uma_obs("matikanetannhauserStage2", 120, 120, 0, &[]);
    o.opponent.active = Some(weak_attacker);
    o.own.active = Some(tough_defender);
    out.push(("08_lethal_false", o));

    // 09 — secondary attack present (matikanefukukitaruStage1 has 2
    // attacks; secondary needs psychic+colorless). Fully covered so
    // own_secondary_usable=1. Secondary damage=0, so would_KO=0.
    let mut o = base_obs();
    let attacker = make_uma_obs("matikanefukukitaruStage1", 100, 100, 2,
        &[("psychic", 1), ("colorless", 1)]);
    let defender = make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[]);
    o.own.active = Some(attacker);
    o.opponent.active = Some(defender);
    out.push(("09_secondary_attack", o));

    // 10 — opp-bench typed-energy aggregate. Two opp bench Umas with
    // disjoint typed energies; own bench wiped so it can't bleed.
    let mut o = base_obs();
    let mut a = make_uma_obs("matikanetannhauserBasic", 60, 60, 2,
        &[("fire", 1), ("water", 1)]);
    a.uid = 100;
    let mut b = make_uma_obs("matikanetannhauserBasic", 60, 60, 1,
        &[("darkness", 1)]);
    b.uid = 101;
    o.opponent.bench = vec![Some(a), Some(b), None];
    let mut own_b = make_uma_obs("matikanetannhauserBasic", 60, 60, 1,
        &[("steel", 1)]);
    own_b.uid = 200;
    o.own.bench = vec![Some(own_b), None, None];
    out.push(("10_opp_bench_typed", o));

    out
}

/// Serialize one fixture: write `<name>.json` (observation) and
/// `<name>.f32.bin` (little-endian f32 vector, length = STATE_DIM_V3_6).
fn write_fixture(dir: &Path, name: &str, obs: &PublicObservation) {
    let vec = observation_state_features_v3_6(obs);
    assert_eq!(
        vec.len(),
        engine::policy::featurize::STATE_DIM_V3_6,
        "fixture {} vector width mismatch",
        name
    );

    let json_path = dir.join(format!("{}.json", name));
    let bin_path = dir.join(format!("{}.f32.bin", name));

    let json = serde_json::to_string_pretty(obs)
        .unwrap_or_else(|e| panic!("serialize {} obs: {}", name, e));
    fs::write(&json_path, json)
        .unwrap_or_else(|e| panic!("write {:?}: {}", json_path, e));

    let mut bytes: Vec<u8> = Vec::with_capacity(vec.len() * 4);
    for v in &vec {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(&bin_path, bytes)
        .unwrap_or_else(|e| panic!("write {:?}: {}", bin_path, e));
}

#[test]
#[ignore]
fn emit_v36_parity_fixtures() {
    let dir = fixtures_dir();
    fs::create_dir_all(&dir).expect("create fixtures dir");

    // README sidecar so the directory is self-documenting.
    let readme = format!(
        "v3.6 Python↔Rust byte-parity fixtures.\n\
         Regenerate with: cargo test -p engine --test v36_python_parity_fixtures \
         -- --ignored emit\n\
         Validate with: python training/v36_python_rust_parity_smoke.py\n\
         STATE_DIM_V3_6 = {}\n",
        engine::policy::featurize::STATE_DIM_V3_6,
    );
    fs::write(dir.join("README.txt"), readme).expect("write README");

    let fixtures = build_fixtures();
    assert!(fixtures.len() >= 5, "need ≥5 fixtures, got {}", fixtures.len());
    for (name, obs) in &fixtures {
        write_fixture(&dir, name, obs);
    }
    eprintln!("emitted {} fixtures to {:?}", fixtures.len(), dir);
}

#[test]
fn fixtures_roundtrip_against_current_rust_featurizer() {
    // Default-cycle guard: re-derives the v3.6 vector from each fixture's
    // JSON and asserts byte-identity vs the on-disk `.f32.bin`. If a
    // Rust-side change drifts the featurizer, this fails locally before
    // the Python parity smoke would notice.
    let dir = fixtures_dir();
    if !dir.exists() {
        // Nothing to check — fixtures haven't been emitted yet. The
        // `emit_v36_parity_fixtures` test creates them.
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
        let obs: PublicObservation = serde_json::from_str(&json)
            .unwrap_or_else(|e| panic!("deserialize {}: {}", name, e));
        let live = observation_state_features_v3_6(&obs);
        let bytes = fs::read(&bin_path)
            .unwrap_or_else(|e| panic!("read {:?}: {}", bin_path, e));
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
        // Bit-exact equality (both written as IEEE-754 f32 LE).
        assert_eq!(
            saved.len(),
            live.len(),
            "{}: vector width mismatch",
            name
        );
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
    eprintln!("roundtrip OK: {} fixtures match current Rust featurizer", checked);
}
