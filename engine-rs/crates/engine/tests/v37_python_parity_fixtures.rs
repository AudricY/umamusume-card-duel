//! v3.7 Python↔Rust byte-parity fixture emitter.
//!
//! Builds 10 diverse `PublicObservation` instances exercising v3.7's
//! combat-arithmetic + catalog-lookup channels, computes each one's
//! `observation_state_features_v3_7` vector, and writes both the
//! serialized JSON and the f32 little-endian feature vector to
//! `engine-rs/crates/engine/tests/fixtures/v37_parity/`. The companion
//! Python smoke (`training/v37_python_rust_parity_smoke.py`) loads the
//! JSON, calls `observation_to_features_v3_7`, and asserts byte-identity
//! against the Rust-emitted vector.
//!
//! Coverage matrix (one fixture per row — see scoping doc §4 step 7):
//!   01_empty_initial           — fresh-setup snapshot (catalog defaults).
//!   02_weakness_on             — Channel 1 lethal bit fires (Darkness vs
//!                                 Psychic +20 weakness, dmg+amount ≥ hp).
//!   03_weakness_off            — Channel 1 clear despite damage near HP
//!                                 (no type match).
//!   04_coin_flip_primary       — Channel 2 primary has_cf + cf_eko fires.
//!   05_coin_flip_secondary     — Channel 2 secondary bits, depending on
//!                                 v3.6 secondary-usable bit.
//!   06_per_energy_bonus        — Channel 3 primary per_energy bit fires.
//!   07_per_bench_bonus         — Channel 3 primary per_bench bit fires.
//!   08_eta_feasible            — Channel 4 own_primary_eta=1 via pool.
//!   09_paralysis_window_open   — Channel 4 own_paralysis_window=1.
//!   10_tool_plus_ability       — Channel 5 + 6 both fire (own tool +
//!                                 own ability used this turn).
//!
//! Regenerate with: `cargo test -p engine --test v37_python_parity_fixtures -- --ignored emit`
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
use engine::policy::featurize::observation_state_features_v3_7;
use engine::policy::observation::build_public_observation;
use engine::policy::types::{
    PublicObservation, PublicUmaObservation, PublicUmaTurnState,
};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("v37_parity")
}

fn base_obs() -> PublicObservation {
    let rng = Rng::from_seed("v37-parity-fixture:selfplay", "selfplay");
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
/// through both serializers below. Channel coverage mirrors the
/// scoping doc §4 step 7 enumeration.
fn build_fixtures() -> Vec<(&'static str, PublicObservation)> {
    let mut out: Vec<(&'static str, PublicObservation)> = Vec::new();

    // 01 — empty / initial: fresh-setup snapshot. The v3.7 tail bits
    // should mostly be 0 except for whatever the trivial setup actives
    // force (ETA bits may fire because matikanetannhauserBasic's single
    // attack cost {psychic:1} is reduced to 0 by the +1 next-turn
    // attach budget regardless of pool).
    out.push(("01_empty_initial", base_obs()));

    // 02 — Channel 1 lethal-on (weakness fires). manhattanCafeStage1
    // (Darkness, 40 damage) on OWN vs matikanetannhauserBasic
    // (Psychic, hp=60, weakness Darkness +20) on OPP. Without weakness
    // 40 < 60; with: 40+20=60 → opp_weakness_lethal=1.
    let mut o = base_obs();
    let attacker = make_uma_obs(
        "manhattanCafeStage1",
        90,
        90,
        2,
        &[("darkness", 1), ("colorless", 1)],
    );
    let defender = make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[]);
    o.own.active = Some(attacker);
    o.opponent.active = Some(defender);
    out.push(("02_weakness_on", o));

    // 03 — Channel 1 lethal-off (no type match). Psychic-on-Psychic at
    // similar damage budget: matikanetannhauserStage1 (Psychic, 40
    // damage, coin_bonus=20) vs haruUraraBasic (Psychic, hp=90,
    // weakness Darkness +20). No type match → 40 < 90 → bit clear,
    // even though damage is near HP. Coin-flip bit ALSO fires
    // (channel-orthogonality check), but Channel 1 stays 0.
    let mut o = base_obs();
    let attacker = make_uma_obs(
        "matikanetannhauserStage1",
        90,
        90,
        2,
        &[("psychic", 1), ("colorless", 1)],
    );
    let defender = make_uma_obs("haruUraraBasic", 90, 90, 0, &[]);
    o.own.active = Some(attacker);
    o.opponent.active = Some(defender);
    out.push(("03_weakness_off", o));

    // 04 — Channel 2 primary coin-flip. matikanetannhauserStage1's
    // primary attack has coinBonus=20. Defender haruUraraBasic at hp=50
    // → expected = 40 + 0.5*20 = 50 ≥ 50 → own_primary_cf_eko=1.
    let mut o = base_obs();
    let attacker = make_uma_obs(
        "matikanetannhauserStage1",
        90,
        90,
        2,
        &[("psychic", 1), ("colorless", 1)],
    );
    let defender = make_uma_obs("haruUraraBasic", 50, 90, 0, &[]);
    o.own.active = Some(attacker);
    o.opponent.active = Some(defender);
    out.push(("04_coin_flip_primary", o));

    // 05 — Channel 2 secondary coin-flip. We need a card with a
    // secondary attack that has a coin-flip component. matikanetannhauserStage2
    // has two attacks; its secondary has coin-bonus semantics. Even if
    // not, Channel 2 bits are computed per-attack so we exercise the
    // secondary slot with whichever card has it. Use
    // matikanefukukitaruStage1 which has a 2-attack profile (its
    // secondary may or may not have coin_bonus — pick a card with two
    // attacks at a minimum and exercise the secondary slot). The bit
    // values are computed by the featurizer; the parity check asserts
    // Rust ≡ Python on whatever the catalog says.
    let mut o = base_obs();
    let attacker = make_uma_obs(
        "matikanefukukitaruStage1",
        100,
        100,
        2,
        &[("psychic", 1), ("colorless", 1)],
    );
    let defender = make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[]);
    o.own.active = Some(attacker);
    o.opponent.active = Some(defender);
    out.push(("05_coin_flip_secondary", o));

    // 06 — Channel 3 per-energy bonus. haruUraraBasic's primary attack
    // has damagePerAttachedEnergy → own_primary_per_energy=1.
    let mut o = base_obs();
    let attacker = make_uma_obs("haruUraraBasic", 90, 90, 0, &[]);
    let defender = make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[]);
    o.own.active = Some(attacker);
    o.opponent.active = Some(defender);
    out.push(("06_per_energy_bonus", o));

    // 07 — Channel 3 per-bench bonus. tamamoCrossStage2 has
    // damagePerUmamusumeInPlay on its attack → own_primary_per_bench=1.
    // (Catalog-verified by v37_combat_arith_smoke which dynamically
    // picks any card with the flag; we hardcode the card here for
    // fixture determinism.)
    let mut o = base_obs();
    let attacker = make_uma_obs("tamamoCrossStage2", 130, 130, 3,
        &[("fire", 1), ("colorless", 2)]);
    let defender = make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[]);
    o.own.active = Some(attacker);
    o.opponent.active = Some(defender);
    out.push(("07_per_bench_bonus", o));

    // 08 — Channel 4 ETA feasible. symboliRudolfStage2 has a 2-typed
    // cost {water:1, dragon:1, colorless:1}. attached={} → shortfall
    // {water:1, dragon:1}. After +1 attach to the lex-first
    // (water), surviving={dragon:1}. With pool=[dragon] → ETA=1.
    let mut o = base_obs();
    let attacker = make_uma_obs("symboliRudolfStage2", 120, 120, 0, &[]);
    let defender = make_uma_obs("haruUraraBasic", 90, 90, 0, &[]);
    o.own.active = Some(attacker);
    o.opponent.active = Some(defender);
    o.own.energy_pool = vec!["dragon".to_string()];
    out.push(("08_eta_feasible", o));

    // 09 — Channel 4 paralysis window. OPP active has
    // paralysis_recovery_pending=true → own_paralysis_window_open=1.
    let mut o = base_obs();
    let own_a = make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[]);
    let mut opp_a = make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[]);
    opp_a.turn_state.paralysis_recovery_pending = true;
    o.own.active = Some(own_a);
    o.opponent.active = Some(opp_a);
    out.push(("09_paralysis_window_open", o));

    // 10 — Channels 5 + 6 simultaneously. own active has
    // leftoverCarrot tool (HealAtTurnEnd, Channel 5 bit 0) AND
    // niceNatureBasic ability used (HpBonus, Channel 6 bit 2). We must
    // pick a card that BOTH carries the ability AND can hold the tool;
    // niceNatureBasic with usedAbilityThisTurn=true and tool attached
    // satisfies both.
    let mut o = base_obs();
    let mut own_a = make_uma_obs("niceNatureBasic", 70, 70, 0, &[]);
    own_a.tool_card_id = Some("leftoverCarrot".to_string());
    own_a.used_ability_this_turn = true;
    let opp_a = make_uma_obs("matikanetannhauserBasic", 60, 60, 0, &[]);
    o.own.active = Some(own_a);
    o.opponent.active = Some(opp_a);
    out.push(("10_tool_plus_ability", o));

    out
}

/// Serialize one fixture: write `<name>.json` (observation) and
/// `<name>.f32.bin` (little-endian f32 vector, length = STATE_DIM_V3_7).
fn write_fixture(dir: &Path, name: &str, obs: &PublicObservation) {
    let vec = observation_state_features_v3_7(obs);
    assert_eq!(
        vec.len(),
        engine::policy::featurize::STATE_DIM_V3_7,
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
fn emit_v37_parity_fixtures() {
    let dir = fixtures_dir();
    fs::create_dir_all(&dir).expect("create fixtures dir");

    // README sidecar so the directory is self-documenting.
    let readme = format!(
        "v3.7 Python↔Rust byte-parity fixtures.\n\
         Regenerate with: cargo test -p engine --test v37_python_parity_fixtures \
         -- --ignored emit\n\
         Validate with: python training/v37_python_rust_parity_smoke.py\n\
         STATE_DIM_V3_7 = {}\n",
        engine::policy::featurize::STATE_DIM_V3_7,
    );
    fs::write(dir.join("README.txt"), readme).expect("write README");

    let fixtures = build_fixtures();
    assert_eq!(
        fixtures.len(),
        10,
        "v3.7 fixture set expected 10 fixtures, got {}",
        fixtures.len()
    );
    for (name, obs) in &fixtures {
        write_fixture(&dir, name, obs);
    }
    eprintln!("emitted {} fixtures to {:?}", fixtures.len(), dir);
}

#[test]
fn fixtures_roundtrip_against_current_rust_featurizer() {
    // Default-cycle guard: re-derives the v3.7 vector from each fixture's
    // JSON and asserts byte-identity vs the on-disk `.f32.bin`. If a
    // Rust-side change drifts the featurizer, this fails locally before
    // the Python parity smoke would notice.
    let dir = fixtures_dir();
    if !dir.exists() {
        // Nothing to check — fixtures haven't been emitted yet. The
        // `emit_v37_parity_fixtures` test creates them.
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
        let live = observation_state_features_v3_7(&obs);
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
