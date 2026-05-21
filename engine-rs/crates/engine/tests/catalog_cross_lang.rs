//! Cross-language catalog parity test.
//!
//! Loads the TS-side catalog dump (every card id + identity fields after
//! variant expansion) and asserts the Rust catalog produces the same set.
//!
//! Regenerate the dump with:
//!   npx tsx engine-rs/scripts/dump-ts-catalog.ts > engine-rs/crates/engine/tests/catalog-reference.json

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use engine::core::catalog::{catalog, Card};
use serde::Deserialize;

const REFERENCE_PATH: &str = "tests/catalog-reference.json";

#[derive(Deserialize)]
struct Reference {
    count: usize,
    cards: Vec<CardSummary>,
}

#[derive(Deserialize, Debug, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
struct CardSummary {
    id: String,
    kind: String,
    #[serde(default)]
    stage: Option<u8>,
    #[serde(default)]
    hp: Option<i32>,
    #[serde(default)]
    r#type: Option<String>,
    #[serde(default)]
    trainer_type: Option<String>,
}

fn load_reference() -> Option<Reference> {
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    let path = here.join(REFERENCE_PATH);
    if !path.exists() {
        eprintln!(
            "skipping cross-language catalog test: {} not found. Regenerate with `npx tsx engine-rs/scripts/dump-ts-catalog.ts > {}`",
            path.display(),
            path.display()
        );
        return None;
    }
    let raw = fs::read_to_string(&path).ok()?;
    Some(serde_json::from_str(&raw).expect("parse catalog reference"))
}

fn build_rust_summaries() -> BTreeMap<String, CardSummary> {
    let cat = catalog();
    let mut out = BTreeMap::new();
    for card in &cat.cards {
        let (id, summary) = match card {
            Card::Umamusume(u) => (
                u.id.clone(),
                CardSummary {
                    id: u.id.clone(),
                    kind: "umamusume".to_string(),
                    stage: Some(u.stage),
                    hp: Some(u.hp),
                    r#type: Some(format!("{:?}", u.r#type)),
                    trainer_type: None,
                },
            ),
            Card::Trainer(t) => (
                t.id.clone(),
                CardSummary {
                    id: t.id.clone(),
                    kind: "trainer".to_string(),
                    stage: None,
                    hp: None,
                    r#type: None,
                    trainer_type: Some(format!("{:?}", t.trainer_type).to_lowercase()),
                },
            ),
        };
        out.insert(id, summary);
    }
    out
}

#[test]
fn catalog_card_set_matches_ts() {
    let Some(reference) = load_reference() else {
        return;
    };
    let rust = build_rust_summaries();
    assert_eq!(
        rust.len(),
        reference.count,
        "card count mismatch: rust={} ts={}",
        rust.len(),
        reference.count
    );
    let ts_ids: std::collections::BTreeSet<String> =
        reference.cards.iter().map(|c| c.id.clone()).collect();
    let rust_ids: std::collections::BTreeSet<String> = rust.keys().cloned().collect();

    let only_rust: Vec<_> = rust_ids.difference(&ts_ids).collect();
    let only_ts: Vec<_> = ts_ids.difference(&rust_ids).collect();
    if !only_rust.is_empty() || !only_ts.is_empty() {
        panic!(
            "id set diverges: only-rust={:?} only-ts={:?}",
            only_rust, only_ts
        );
    }
}

#[test]
fn catalog_identity_fields_match_ts() {
    let Some(reference) = load_reference() else {
        return;
    };
    let rust = build_rust_summaries();
    let mut diverged: Vec<String> = Vec::new();
    for ts_card in &reference.cards {
        let Some(rust_card) = rust.get(&ts_card.id) else {
            // Set-equality test above will catch this.
            continue;
        };
        if rust_card.kind != ts_card.kind {
            diverged.push(format!(
                "{}: kind rust={} ts={}",
                ts_card.id, rust_card.kind, ts_card.kind
            ));
            continue;
        }
        if rust_card.stage != ts_card.stage {
            diverged.push(format!(
                "{}: stage rust={:?} ts={:?}",
                ts_card.id, rust_card.stage, ts_card.stage
            ));
        }
        if rust_card.hp != ts_card.hp {
            diverged.push(format!(
                "{}: hp rust={:?} ts={:?}",
                ts_card.id, rust_card.hp, ts_card.hp
            ));
        }
        // Type comparison: Rust uses TitleCase ("Psychic"), TS uses
        // TitleCase too, so direct compare works. The format! debug print
        // gives "Psychic", which matches the TS dump field exactly.
        if rust_card.r#type != ts_card.r#type {
            diverged.push(format!(
                "{}: type rust={:?} ts={:?}",
                ts_card.id, rust_card.r#type, ts_card.r#type
            ));
        }
        if rust_card.trainer_type != ts_card.trainer_type {
            diverged.push(format!(
                "{}: trainerType rust={:?} ts={:?}",
                ts_card.id, rust_card.trainer_type, ts_card.trainer_type
            ));
        }
    }
    assert!(
        diverged.is_empty(),
        "{} divergences:\n  {}",
        diverged.len(),
        diverged.join("\n  ")
    );
}
