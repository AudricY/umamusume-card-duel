# Training-Data Coverage Audit (R7 mixed corpus)

Canonical home for the corpus-retention + state-overlap finding (backlog item `training-data-coverage-audit`, P1). Diagnostic-only: no training, no data regen, no GPU. Regenerate with `training/.venv/bin/python training/data_coverage_audit.py`. Machine-readable sidecar: `training-data-coverage-audit.json`.

## Headline

- Corpus: `/home/audric/work/umamusume-card-duel/runs/R7-multi-teacher-warmstart/iter-000/mixed.jsonl`
- **12387 rows -> 3789 retained (30.6%); 8598 dropped (69.4%)**.
- The ~70% drop is **not** schema/feature/card-id loss. It is **100% single-legal-action forced states** removed by the loader's `min_actions=2` predicate (`uma_ai.dataset.load_policy_samples`): a state with one legal action carries zero policy-decision signal.

## Filter-reason decomposition

| Reason | Count | Share of dropped |
| --- | ---: | ---: |
| `lt_min_actions(<2_legal)` | 8598 | 100.0% |

Loader invariant: every other loader failure mode (schema-version mismatch, bad feature shape, missing `cardIdsByZone`) *raises* rather than silently dropping. Retained + dropped == total with only the `min_actions` predicate firing confirms zero raise-class rows in this corpus (it trained successfully under the v2/96-d schema path with `strict_schema_version=False`).

## Per-source retention

| Source | Available | Retained | Dropped | Retained rate |
| --- | ---: | ---: | ---: | ---: |
| `model-visited-rollout-relabeled` | 7955 | 2596 | 5359 | 32.6% |
| `rule-bot-replay` | 4432 | 1193 | 3239 | 26.9% |

## Retained-vs-dropped slices

Full per-axis breakdown is in the JSON sidecar (`slices.<axis>`). Axes covered: source, side, seed range, phase, action kind, turn bucket, legal-action-count bucket, points, board stage, energy, hand/deck size, terminal distance, schema/card-id availability.

### Top 5 most-dropped buckets

| Axis | Bucket | Dropped | Retained | Drop rate |
| --- | --- | ---: | ---: | ---: |
| action_kind | `pass` | 7568 | 746 | 91.0% |
| points | `p0` | 6419 | 2970 | 68.4% |
| board_stage | `p0` | 6419 | 2970 | 68.4% |
| terminal_distance | `p0` | 6419 | 2970 | 68.4% |
| hand_size_bucket | `<=2` | 5515 | 1630 | 77.2% |

### Top 5 overrepresented buckets in the retained set

| Axis | Bucket | Retained | Share of axis retained |
| --- | --- | ---: | ---: |
| points | `p0` | 2970 | 78.4% |
| board_stage | `p0` | 2970 | 78.4% |
| terminal_distance | `p0` | 2970 | 78.4% |
| source | `model-visited-rollout-relabeled` | 2596 | 68.5% |
| seed_prefix | `140xxx` | 2596 | 68.5% |

## Train-vs-gate state overlap

- **Limitation:** Gate per-state observation traces were never persisted (decisionTraceOut empty in all runs/ gate manifests; no decision trace files exist). Exact retained-state vs gate-state observation overlap is UNCOMPUTABLE from existing artifacts — recorded as an environment gap. Structural comparison used instead.
- Gate structure: rule-bot opponent=True, side-balanced=True, seed_start=9000.
- The gate plays full games end-to-end, so it traverses ALL decision states including the 1-legal-action forced states the loader drops (min_actions=2). Training never sees that ~69% of the visited-state stream.
- Gate side split: {"player": {"games": 250, "winRate": 0.28, "wilson_lower": 0.22799348323513563}, "opponent": {"games": 250, "winRate": 0.384, "wilson_lower": 0.3258981738809542}}
- Retained side distribution: {"opponent": {"retained_rows": 1852, "retained_share": 0.4888}, "player": {"retained_rows": 1937, "retained_share": 0.5112}}

## Coverage-vs-loss buckets

### By phase (val loss, highest first)

| Bucket | Train n | Val n | Val loss | Val acc |
| --- | ---: | ---: | ---: | ---: |
| `unknown` | 1975.0 | 621.0 | 3.370 | 0.725 |
| `trainerAfter` | 24.0 | 4.0 | 1.342 | 0.500 |
| `trainerBefore` | 738.0 | 171.0 | 0.983 | 0.784 |
| `bench` | 67.0 | 17.0 | 0.237 | 0.941 |
| `evolve` | 33.0 | 10.0 | 0.224 | 1.000 |
| `combat` | 113.0 | 16.0 | 0.108 | 0.938 |

### By action_kind (val loss, highest first)

| Bucket | Train n | Val n | Val loss | Val acc |
| --- | ---: | ---: | ---: | ---: |
| `useAbility` | 39.0 | 9.0 | 3.507 | 0.778 |
| `pass` | 602.0 | 144.0 | 3.299 | 0.486 |
| `attachEnergy` | 539.0 | 170.0 | 3.002 | 0.900 |
| `playTrainer` | 1221.0 | 335.0 | 2.066 | 0.687 |
| `retreatAttack` | 19.0 | 3.0 | 0.721 | 1.000 |
| `evolve` | 218.0 | 78.0 | 0.651 | 0.962 |
| `playBasic` | 173.0 | 50.0 | 0.521 | 0.880 |
| `attack` | 139.0 | 50.0 | 0.379 | 0.900 |

### By source (val loss, highest first)

| Bucket | Train n | Val n | Val loss | Val acc |
| --- | ---: | ---: | ---: | ---: |
| `model-visited-rollout-relabeled` | 1975.0 | 621.0 | 3.370 | 0.725 |
| `heuristic-candidate-v1` | 975.0 | 218.0 | 0.833 | 0.812 |

Reading: the highest val-loss buckets coincide with the lowest-coverage / most-distribution-shifted slices (e.g. `attachEnergy`, `useAbility`, `unknown`-phase relabeled rows) — loss clusters where retained coverage is thin, consistent with the state-coverage-bottleneck hypothesis.

## Proposed source-mix target (handoff to backlog 2-4)

_The R7 corpus drop is ~100% single-legal-action forced states (no decision to imitate), not schema/feature loss. Row count is not the bottleneck; decision-state coverage is. The fix is a state-generation recipe (backlog item 2), not more self-play rows. Target percentages are floors over the RETAINED (>=2-legal-action) decision-state population._

| Slice | Floor / cap |
| --- | --- |
| side | player >= 45% and opponent >= 45% within each major phase bucket (gate is side-balanced; R7 retained is player 51% / opp 49% overall but skewed within phases) |
| source | rollout-leaf-MCTS-relabeled on rule-bot-covered states >= 60%; rule-bot-replay <= 40% (R7 had rollout 69% / rule-bot 31% of retained — keep rollout dominant but regenerate its STATES from the gate distribution) |
| phase | every non-trivial phase (combat, evolve, bench, trainerAfter) >= 8% each; cap trainerBefore <= 45% (R7 retained is trainerBefore-dominated) |
| action_kind | attack + retreatAttack combined >= 12%; evolve >= 6%; useAbility >= 3% (R7 retained: attack ~5%, retreatAttack ~0.6%, useAbility ~1.3% — combat-line under-covered, the exact contested decisions the gate rewards) |
| turn_bucket | t6-9 and t10+ combined >= 25% (late-game decisive states; R7 retained skews to early turns) |
| legal_action_count | contested states (>=4 legal) >= 30% of retained (R7 retained is dominated by exactly-2-legal binary choices) |

**Do not:** Do NOT relax min_actions to 1 — single-action states carry zero policy gradient and would dilute the loss. Do NOT regenerate self-play-only rows (mcts-distill v1 failed at 0.1470 on that exact mistake). Implement via backlog items 2-4 only.

## Environment gaps

- Gate per-state observation traces never persisted (decisionTraceOut empty across all runs/ gate manifests; no decision-trace files exist) — exact state-level overlap uncomputable; structural proxy used.
- Per-state eval loss not logged; loss is only bucketed by phase/action_kind/source in the train manifest diagnostics — coverage-vs-loss uses those buckets as the justified proxy.
- Exact terminal-distance per row not in corpus; own-points used as the monotone progress proxy.

---
_Generated by `training/data_coverage_audit.py` (deterministic, read-only). This document is the single canonical home for the coverage finding; other docs link here, never copy._
