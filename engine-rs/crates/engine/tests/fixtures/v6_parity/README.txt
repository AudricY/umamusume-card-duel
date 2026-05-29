v6 Python↔Rust byte-parity fixtures (own-deck-composition).
Regenerate: cargo test -p engine --test v6_python_parity_fixtures -- --ignored emit
Validate:   python training/v6_python_rust_parity_smoke.py
STATE_DIM_V6 = 126
NOTE: observation-fixture tails [110:126] are ZEROS (the obs contract
exposes only own.deck_count, not the deck card-id list). The live
bucketing math is validated by the deck_ids/ helper fixtures.
