//! v6 Python↔Rust byte-parity fixture emitter (own-deck-composition).
//!
//! v6 = the FROZEN v3.0 110-d head [0:110] + a 16-slot OWN remaining-deck
//! composition tail [110:126] (finding T2.7 "remaining-deck inference"). This
//! test emits two fixture families under
//! `engine-rs/crates/engine/tests/fixtures/v6_parity/`:
//!
//!   1. OBSERVATION fixtures (`<name>.json` + `<name>.f32.bin`): a diverse set
//!      of `PublicObservation`s and their `observation_state_features_v6`
//!      vectors. Because the current `PublicObservation` contract does NOT
//!      expose the own deck card-id list (only `own.deck_count`), the v6 tail
//!      [110:126] degrades to ZEROS on every observation fixture — and so does
//!      the Python `observation_to_features_v6`. The companion Python smoke
//!      asserts byte-identity on the tail (both zero) AND 4-ULP on the v3.0
//!      head [0:110]. This is the parity-by-construction regression guard.
//!
//!   2. DECK-COMPOSITION helper fixtures (`deck_ids/<name>.deck.json` +
//!      `deck_ids/<name>.tail.f32.bin`): explicit own-deck card-id lists and
//!      the 16-slot `v6_deck_composition_tail` output. This validates the LIVE
//!      bucketing math (kind + uma-type) bit-exact against the Python
//!      `_v6_deck_composition_tail`, so the moment the obs contract grows an
//!      own-deck-card-ids field BOTH builders light up identically.
//!
//! Regenerate with:
//!   cargo test -p engine --test v6_python_parity_fixtures -- --ignored emit
//! Validate with:
//!   python training/v6_python_rust_parity_smoke.py

use std::fs;
use std::path::{Path, PathBuf};

use engine::core::constants::SideId;
use engine::core::random::{with_rng, Rng};
use engine::headless_setup::setup_ai_vs_ai_game;
use engine::policy::featurize::{
    observation_state_features_v6, v6_deck_composition_tail, STATE_DIM_V6,
};
use engine::policy::observation::build_public_observation;
use engine::policy::types::{PublicObservation, PublicUmaObservation, PublicUmaTurnState};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("v6_parity")
}

fn deck_ids_dir() -> PathBuf {
    fixtures_dir().join("deck_ids")
}

fn base_obs() -> PublicObservation {
    let rng = Rng::from_seed("v6-parity-fixture:selfplay", "selfplay");
    let (state, _used) = with_rng(rng, || setup_ai_vs_ai_game());
    build_public_observation(&state, SideId::Player)
}

fn make_uma_obs(card_id: &str, hp: i32, max_hp: i32) -> PublicUmaObservation {
    PublicUmaObservation {
        uid: 1,
        card_id: card_id.to_string(),
        species: String::new(),
        stage: 0,
        hp,
        max_hp,
        energy_total: 0,
        energies: indexmap::IndexMap::new(),
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

/// Observation fixtures — the v6 tail is zeros on every one (no deck card-ids
/// in the obs contract), so coverage targets HEAD diversity (the parity
/// regression guard) + bench/active population.
fn build_obs_fixtures() -> Vec<(&'static str, PublicObservation)> {
    let mut out: Vec<(&'static str, PublicObservation)> = Vec::new();

    out.push(("01_empty_initial", base_obs()));

    let mut o = base_obs();
    o.own.active = Some(make_uma_obs("matikanetannhauserBasic", 60, 60));
    o.own.bench = vec![Some(make_uma_obs("haruUraraBasic", 90, 90))];
    out.push(("02_own_active_bench", o));

    let mut o = base_obs();
    o.own.points = 1;
    o.opponent.points = 2;
    o.own.active = Some(make_uma_obs("matikanefukukitaruStage1", 100, 100));
    o.opponent.active = Some(make_uma_obs("symboliRudolfStage2", 120, 120));
    out.push(("03_mid_game_prizes", o));

    let mut o = base_obs();
    o.own.deck_count = 0; // exhausted deck; tail still zeros (no id list).
    out.push(("04_empty_deck_count", o));

    out
}

/// Deck-composition helper fixtures — explicit own remaining-in-deck card-id
/// lists exercising every kind bucket + several uma-type buckets. The names
/// document the expected non-zero buckets.
fn build_deck_fixtures() -> Vec<(&'static str, Vec<String>)> {
    let s = |x: &str| x.to_string();
    vec![
        ("01_empty", vec![]),
        (
            // 2x basic Psychic uma + 1 supporter + 1 stage2 Dragon + 1 stage1 Psychic.
            "02_mixed",
            vec![
                s("matikanetannhauserBasic"),
                s("matikanetannhauserBasic"),
                s("tazunaHayakawa"),
                s("symboliRudolfStage2"),
                s("matikanefukukitaruStage1"),
            ],
        ),
        (
            // All-basic-uma, two types.
            "03_all_basic_uma",
            vec![
                s("matikanetannhauserBasic"),
                s("haruUraraBasic"),
                s("haruUraraBasic"),
            ],
        ),
        (
            // Trainer-only deck (supporter + item/tool buckets exercised via
            // whatever trainer types resolve). Unknown ids are skipped.
            "04_trainers_and_unknown",
            vec![
                s("tazunaHayakawa"),
                s("nakayamaTurf"),
                s("carrotHamburgerSteak"),
                s("not_a_real_card_id"),
            ],
        ),
        (
            // A full 20-card deck to exercise the /20 normalization at scale.
            "05_full_twenty",
            (0..20).map(|_| s("matikanetannhauserBasic")).collect(),
        ),
    ]
}

fn write_obs_fixture(dir: &Path, name: &str, obs: &PublicObservation) {
    let vec = observation_state_features_v6(obs);
    assert_eq!(vec.len(), STATE_DIM_V6, "fixture {} width mismatch", name);
    let json = serde_json::to_string_pretty(obs)
        .unwrap_or_else(|e| panic!("serialize {} obs: {}", name, e));
    fs::write(dir.join(format!("{}.json", name)), json)
        .unwrap_or_else(|e| panic!("write {}.json: {}", name, e));
    write_f32_bin(&dir.join(format!("{}.f32.bin", name)), &vec);
}

fn write_deck_fixture(dir: &Path, name: &str, deck: &[String]) {
    let tail = v6_deck_composition_tail(deck);
    let json = serde_json::to_string_pretty(deck)
        .unwrap_or_else(|e| panic!("serialize {} deck: {}", name, e));
    fs::write(dir.join(format!("{}.deck.json", name)), json)
        .unwrap_or_else(|e| panic!("write {}.deck.json: {}", name, e));
    write_f32_bin(&dir.join(format!("{}.tail.f32.bin", name)), &tail);
}

fn write_f32_bin(path: &Path, vec: &[f32]) {
    let mut bytes: Vec<u8> = Vec::with_capacity(vec.len() * 4);
    for v in vec {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, bytes).unwrap_or_else(|e| panic!("write {:?}: {}", path, e));
}

#[test]
#[ignore]
fn emit_v6_parity_fixtures() {
    let dir = fixtures_dir();
    let ddir = deck_ids_dir();
    fs::create_dir_all(&dir).expect("create fixtures dir");
    fs::create_dir_all(&ddir).expect("create deck_ids dir");

    let readme = format!(
        "v6 Python↔Rust byte-parity fixtures (own-deck-composition).\n\
         Regenerate: cargo test -p engine --test v6_python_parity_fixtures -- --ignored emit\n\
         Validate:   python training/v6_python_rust_parity_smoke.py\n\
         STATE_DIM_V6 = {}\n\
         NOTE: observation-fixture tails [110:126] are ZEROS (the obs contract\n\
         exposes only own.deck_count, not the deck card-id list). The live\n\
         bucketing math is validated by the deck_ids/ helper fixtures.\n",
        STATE_DIM_V6,
    );
    fs::write(dir.join("README.txt"), readme).expect("write README");

    let obs_fixtures = build_obs_fixtures();
    for (name, obs) in &obs_fixtures {
        write_obs_fixture(&dir, name, obs);
    }
    let deck_fixtures = build_deck_fixtures();
    for (name, deck) in &deck_fixtures {
        write_deck_fixture(&ddir, name, deck);
    }
    eprintln!(
        "emitted {} obs fixtures + {} deck fixtures",
        obs_fixtures.len(),
        deck_fixtures.len()
    );
}

#[test]
fn obs_fixtures_roundtrip_against_current_rust_featurizer() {
    let dir = fixtures_dir();
    if !dir.exists() {
        eprintln!("fixtures dir {:?} does not exist; skipping roundtrip", dir);
        return;
    }
    let mut entries: Vec<PathBuf> = fs::read_dir(&dir)
        .expect("read fixtures dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
        .collect();
    entries.sort();
    let mut checked = 0usize;
    for json_path in entries {
        let name = json_path.file_stem().and_then(|s| s.to_str()).unwrap();
        let bin_path = dir.join(format!("{}.f32.bin", name));
        assert!(bin_path.exists(), "missing {:?}", bin_path);
        let json = fs::read_to_string(&json_path)
            .unwrap_or_else(|e| panic!("read {:?}: {}", json_path, e));
        let obs: PublicObservation =
            serde_json::from_str(&json).unwrap_or_else(|e| panic!("deserialize {}: {}", name, e));
        let live = observation_state_features_v6(&obs);
        let saved = read_f32_bin(&bin_path);
        assert_eq!(saved.len(), live.len(), "{}: width mismatch", name);
        for (i, (&a, &b)) in saved.iter().zip(live.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{}: slot {} drift saved={} live={}",
                name,
                i,
                a,
                b
            );
        }
        // The tail [110:126] MUST be all zeros under the current obs contract.
        for (i, &x) in live[110..126].iter().enumerate() {
            assert_eq!(x, 0.0, "{}: tail slot {} not zero", name, i);
        }
        checked += 1;
    }
    assert!(checked > 0, "no obs fixtures found to roundtrip-check");
    eprintln!("obs roundtrip OK: {} fixtures", checked);
}

#[test]
fn deck_fixtures_roundtrip_against_current_helper() {
    let ddir = deck_ids_dir();
    if !ddir.exists() {
        eprintln!("deck_ids dir {:?} does not exist; skipping roundtrip", ddir);
        return;
    }
    let mut entries: Vec<PathBuf> = fs::read_dir(&ddir)
        .expect("read deck_ids dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.to_str().map(|s| s.ends_with(".deck.json")).unwrap_or(false))
        .collect();
    entries.sort();
    let mut checked = 0usize;
    for deck_path in entries {
        let fname = deck_path.file_name().and_then(|s| s.to_str()).unwrap();
        let name = fname.trim_end_matches(".deck.json");
        let bin_path = ddir.join(format!("{}.tail.f32.bin", name));
        assert!(bin_path.exists(), "missing {:?}", bin_path);
        let json = fs::read_to_string(&deck_path)
            .unwrap_or_else(|e| panic!("read {:?}: {}", deck_path, e));
        let deck: Vec<String> =
            serde_json::from_str(&json).unwrap_or_else(|e| panic!("deserialize {}: {}", name, e));
        let live = v6_deck_composition_tail(&deck);
        let saved = read_f32_bin(&bin_path);
        assert_eq!(saved.len(), live.len(), "{}: tail width mismatch", name);
        for (i, (&a, &b)) in saved.iter().zip(live.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{}: tail slot {} drift saved={} live={}",
                name,
                i,
                a,
                b
            );
        }
        checked += 1;
    }
    assert!(checked > 0, "no deck fixtures found to roundtrip-check");
    eprintln!("deck roundtrip OK: {} fixtures", checked);
}

fn read_f32_bin(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {:?}: {}", path, e));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
