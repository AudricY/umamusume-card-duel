//! Public-belief construction for the ReBeL research line.
//!
//! The first implementation builds weighted private-world particles from the
//! public observation plus the observer's legal private knowledge. During
//! self-play it may audit against the true simulator state, but model-facing
//! features expose only summaries over hidden zones.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::core::card_id::CardId;
use crate::core::catalog::{catalog, Card};
use crate::core::constants::SideId;
use crate::core::decks::deck_lists_by_id;
use crate::core::random::{with_rng, Rng};
use crate::core::state::{GameState, SideState};
use crate::dispatcher::state_hash;
use crate::policy::actions::enumerate_legal_ai_actions;
use crate::policy::card_vocab::card_vocab_index;
use crate::policy::observation::build_public_observation;
use crate::policy::types::{LegalAiAction, PublicObservation};

pub const BELIEF_SCHEMA_VERSION: u32 = 1;

/// Width of the per-card opponent-hand presence block appended to the
/// belief vector by [`build_belief_features`]. One slot per non-pad card
/// vocab index (`card_vocab().vocab_size`, currently 107). Index `i` holds
/// the marginal presence mass for the card whose `card_vocab_index` is
/// `i + 1` (the pad/unknown index 0 is never emitted).
pub const BELIEF_HAND_RANGE_DIM: usize = 107;

/// Number of scalar summary entries that precede the per-card presence
/// block in the belief vector. Kept byte-stable for graph compatibility.
const BELIEF_SUMMARY_DIM: usize = 16;

pub const BELIEF_FEATURE_DIM: usize = BELIEF_SUMMARY_DIM + BELIEF_HAND_RANGE_DIM;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PublicHistory {
    pub schema_version: u32,
    pub initial_player_deck_id: String,
    pub initial_opponent_deck_id: String,
    pub event_count: u32,
    pub digest: String,
}

impl PublicHistory {
    pub fn from_state(
        state: &GameState,
        player_deck_id: &str,
        opponent_deck_id: &str,
        event_count: u32,
    ) -> Self {
        PublicHistory {
            schema_version: BELIEF_SCHEMA_VERSION,
            initial_player_deck_id: player_deck_id.to_string(),
            initial_opponent_deck_id: opponent_deck_id.to_string(),
            event_count,
            digest: state_hash(state),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HiddenZoneProbability {
    pub card_id: String,
    pub probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BeliefFeatures {
    pub schema_version: u32,
    pub vector: Vec<f32>,
    pub opponent_hand_card_presence: Vec<HiddenZoneProbability>,
    pub opponent_deck_card_presence: Vec<HiddenZoneProbability>,
    pub own_deck_card_presence: Vec<HiddenZoneProbability>,
    pub hidden_zone_entropy: f64,
    pub particle_diversity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HiddenZoneAssignment {
    pub opponent_hand_card_ids: Vec<String>,
    pub opponent_deck_card_ids: Vec<String>,
    pub own_deck_card_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LikelihoodFeatures {
    pub count_consistent: bool,
    pub public_consistent: bool,
    pub log_likelihood: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PrivateWorldParticle {
    pub game_state: GameState,
    pub weight: f64,
    pub hidden_zone_assignment: HiddenZoneAssignment,
    pub likelihood_features: LikelihoodFeatures,
    pub rng_seed_label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BeliefAuditDiagnostics {
    pub true_state_in_support: bool,
    pub particle_count: usize,
    pub rejected_particles: u32,
    pub rejection_reasons: Vec<String>,
    pub hidden_zone_entropy: f64,
    pub particle_diversity: f64,
    pub public_observation_matches: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PublicBeliefState {
    pub schema_version: u32,
    pub public_observation: PublicObservation,
    pub public_history: PublicHistory,
    pub public_history_digest: String,
    pub observer_side: SideId,
    pub legal_actions: Vec<LegalAiAction>,
    pub particles: Vec<PrivateWorldParticle>,
    pub particle_weights: Vec<f64>,
    pub belief_features: BeliefFeatures,
    pub audit: BeliefAuditDiagnostics,
}

#[derive(Debug, Clone)]
pub struct BeliefBuildConfig {
    pub particle_count: usize,
    pub seed_label: String,
}

impl Default for BeliefBuildConfig {
    fn default() -> Self {
        BeliefBuildConfig {
            particle_count: 64,
            seed_label: "belief".to_string(),
        }
    }
}

pub fn build_public_belief_state(
    state: &GameState,
    observer_side: SideId,
    public_history: PublicHistory,
    config: &BeliefBuildConfig,
    rng: &mut Rng,
) -> PublicBeliefState {
    let legal_actions = with_rng_borrow(rng, || enumerate_legal_ai_actions(state, observer_side));
    build_public_belief_state_with_legal_actions(
        state,
        observer_side,
        public_history,
        legal_actions,
        config,
        rng,
    )
}

pub fn build_public_belief_state_with_legal_actions(
    state: &GameState,
    observer_side: SideId,
    public_history: PublicHistory,
    legal_actions: Vec<LegalAiAction>,
    config: &BeliefBuildConfig,
    rng: &mut Rng,
) -> PublicBeliefState {
    let particle_count = config.particle_count.max(1);
    // Candidate public deck lists for the opponent, resolved from the public
    // history. The same id can live in both the player-side and AI-side
    // registries with different lists, so we resolve ALL candidates and let
    // the conservation guard pick the one consistent with the observed state.
    // An empty list (unknown deck id) makes `resample_hidden_zones` fall back
    // to the true-multiset shuffle.
    let opponent_deck_candidates: Vec<&'static [CardId]> =
        deck_lists_by_id(&public_history.initial_opponent_deck_id);
    let mut particles = Vec::with_capacity(particle_count);
    for index in 0..particle_count {
        let mut particle_state = state.clone();
        resample_hidden_zones(
            &mut particle_state,
            observer_side,
            &opponent_deck_candidates,
            rng,
        );
        let weight = 1.0 / particle_count as f64;
        particles.push(PrivateWorldParticle {
            hidden_zone_assignment: hidden_assignment(&particle_state, observer_side),
            likelihood_features: LikelihoodFeatures {
                count_consistent: true,
                public_consistent: true,
                log_likelihood: 0.0,
            },
            rng_seed_label: format!("{}:particle:{}", config.seed_label, index),
            game_state: particle_state,
            weight,
        });
    }
    let particle_weights = particles.iter().map(|p| p.weight).collect::<Vec<_>>();
    let belief_features = build_belief_features(&particles, observer_side);
    let audit = BeliefAuditDiagnostics {
        true_state_in_support: true,
        particle_count: particles.len(),
        rejected_particles: 0,
        rejection_reasons: Vec::new(),
        hidden_zone_entropy: belief_features.hidden_zone_entropy,
        particle_diversity: belief_features.particle_diversity,
        public_observation_matches: true,
    };
    let digest = public_history.digest.clone();
    PublicBeliefState {
        schema_version: BELIEF_SCHEMA_VERSION,
        public_observation: build_public_observation(state, observer_side),
        public_history,
        public_history_digest: digest,
        observer_side,
        legal_actions,
        particles,
        particle_weights,
        belief_features,
        audit,
    }
}

fn with_rng_borrow<T>(rng: &mut Rng, f: impl FnOnce() -> T) -> T {
    let taken = rng.clone();
    let (out, used) = with_rng(taken, f);
    *rng = used;
    out
}

/// Resample the opponent's hidden zones (hand + deck) and reshuffle the
/// observer's own deck.
///
/// The opponent's hidden pool is derived from a *public* deck list
/// candidate (resolved from `public_history.initial_opponent_deck_id`)
/// minus the opponent's publicly-revealed cards — formalizing the belief
/// support as public-info-only and providing the hook for future
/// public-history conditioning.
///
/// `opponent_deck_candidates` holds every deck list registered under the
/// public deck id (the id can collide across the player-side and AI-side
/// registries with different lists); the conservation guard selects the
/// candidate consistent with the observed state.
///
/// Conservation safety net: a candidate's complement is only trusted when
/// it matches the simulator's TRUE hidden multiset
/// (`sort(opponent.hand ++ opponent.deck)`). If the revealed-card
/// enumeration is incomplete — or no candidate conserves — we fall back to
/// the existing true-multiset shuffle so the particle state is never
/// corrupted (e.g. cards invented or dropped). A `debug_assert!` flags the
/// divergence in tests; the release path degrades gracefully rather than
/// panicking.
///
/// Under conservation (the expected case) the complement equals the true
/// hidden multiset, so this is behavior-equivalent to the prior
/// true-multiset shuffle — that equivalence is intended. The value here is
/// correctness-hardening plus a public-derived belief support, not a change
/// in the sampled distribution.
fn resample_hidden_zones(
    state: &mut GameState,
    observer_side: SideId,
    opponent_deck_candidates: &[&[CardId]],
    rng: &mut Rng,
) {
    let opponent_side = observer_side.opposite();

    // True hidden multiset, used both as the fallback pool and as the
    // conservation oracle.
    let mut sorted_true: Vec<CardId> = {
        let opponent = state.side(opponent_side);
        opponent
            .hand
            .iter()
            .chain(opponent.deck.iter())
            .copied()
            .collect()
    };
    sorted_true.sort_unstable_by_key(|c| c.0);

    // Find the first candidate deck list whose public complement
    // (multiset(deck) - multiset(revealed)) conserves the true hidden
    // multiset. That candidate's complement becomes the sampling pool.
    let conserving_complement: Option<Vec<CardId>> =
        opponent_deck_candidates.iter().find_map(|deck_list| {
            let mut complement = public_complement_pool(state, opponent_side, deck_list)?;
            complement.sort_unstable_by_key(|c| c.0);
            (complement == sorted_true).then_some(complement)
        });
    debug_assert!(
        opponent_deck_candidates.is_empty() || conserving_complement.is_some(),
        "belief resample: no public deck candidate conserved the true hidden \
         multiset (revealed-card enumeration incomplete); true={:?}",
        sorted_true,
    );

    // Pool to partition into hand/deck: the conserving public complement
    // when one exists, otherwise the true hidden multiset (legacy behavior).
    let mut pool: Vec<CardId> = conserving_complement.unwrap_or(sorted_true);
    shuffle_in_place(&mut pool, rng);

    let opponent = state.side_mut(opponent_side);
    let hand_len = opponent.hand.len();
    let deck_len = opponent.deck.len();
    opponent.hand.clear();
    opponent.deck.clear();
    for cid in pool.iter().take(hand_len) {
        let _ = opponent.hand.try_push(*cid);
    }
    for cid in pool.iter().skip(hand_len).take(deck_len) {
        let _ = opponent.deck.try_push(*cid);
    }

    let own = state.side_mut(observer_side);
    let mut own_deck: Vec<CardId> = own.deck.iter().copied().collect();
    shuffle_in_place(&mut own_deck, rng);
    own.deck.clear();
    for cid in own_deck {
        let _ = own.deck.try_push(cid);
    }
}

/// Multiset subtraction: `multiset(deck_list) - multiset(revealed)` for the
/// given side. Returns `None` if any revealed card is absent from the deck
/// list (which would make the subtraction ill-defined and signals an
/// inconsistent public deck id); callers treat `None` as "cannot trust the
/// complement" and fall back to the true-multiset shuffle.
fn public_complement_pool(
    state: &GameState,
    side: SideId,
    deck_list: &[CardId],
) -> Option<Vec<CardId>> {
    let mut counts = BTreeMap::<u16, i64>::new();
    for cid in deck_list {
        *counts.entry(cid.0).or_default() += 1;
    }
    let mut revealed = revealed_cards(state.side(side));
    // A stadium card the opponent owns came from its deck, so subtract it
    // too. (`stadium.owner` distinguishes whose deck-list it belongs to.)
    if let Some(stadium) = state.stadium.as_ref() {
        if stadium.owner == side {
            revealed.push(stadium.card_id);
        }
    }
    for cid in revealed {
        let entry = counts.entry(cid.0).or_default();
        *entry -= 1;
        if *entry < 0 {
            // Revealed a card the public deck list does not contain — the
            // deck id is inconsistent with the observed state. Bail so the
            // caller falls back rather than fabricating a malformed pool.
            return None;
        }
    }
    let mut pool = Vec::new();
    for (id, count) in counts {
        for _ in 0..count.max(0) {
            pool.push(CardId(id));
        }
    }
    Some(pool)
}

/// Enumerate every card of `side` that sits in a publicly-visible
/// side-local zone: the active uma (its base card + evolution chain +
/// attached tool), each bench uma (same), and the discard pile. An owned
/// stadium also reveals a deck card, but it lives on `GameState` rather
/// than `SideState`, so the caller subtracts it separately. These are the
/// cards a fully-informed observer can subtract from the public deck list
/// to recover the hidden pool. (This engine models no prize zone, and the
/// energy zones do not consume deck cards.)
fn revealed_cards(side: &SideState) -> Vec<CardId> {
    let mut out = Vec::new();
    let push_uma = |uma: &crate::core::state::UmamusumeInstance, out: &mut Vec<CardId>| {
        out.push(uma.card_id);
        for ev in &uma.evolution_card_ids {
            out.push(*ev);
        }
        if let Some(tool) = uma.tool_card_id {
            out.push(tool);
        }
    };
    if let Some(active) = side.active.as_ref() {
        push_uma(active, &mut out);
    }
    for uma in &side.bench {
        push_uma(uma, &mut out);
    }
    for cid in &side.discard {
        out.push(*cid);
    }
    out
}

fn shuffle_in_place<T>(items: &mut [T], rng: &mut Rng) {
    if items.len() < 2 {
        return;
    }
    for index in (1..items.len()).rev() {
        let swap_index = (rng.next_f64() * (index as f64 + 1.0)).floor() as usize;
        items.swap(index, swap_index.min(index));
    }
}

fn hidden_assignment(state: &GameState, observer_side: SideId) -> HiddenZoneAssignment {
    let opponent = state.side(observer_side.opposite());
    let own = state.side(observer_side);
    HiddenZoneAssignment {
        opponent_hand_card_ids: card_labels(opponent.hand.iter().copied()),
        opponent_deck_card_ids: card_labels(opponent.deck.iter().copied()),
        own_deck_card_ids: card_labels(own.deck.iter().copied()),
    }
}

fn build_belief_features(
    particles: &[PrivateWorldParticle],
    observer_side: SideId,
) -> BeliefFeatures {
    let total_weight = particles.iter().map(|p| p.weight).sum::<f64>().max(1e-12);
    let mut opp_hand = BTreeMap::<CardId, f64>::new();
    let mut opp_deck = BTreeMap::<CardId, f64>::new();
    let mut own_deck = BTreeMap::<CardId, f64>::new();
    let mut hand_count_sum = 0.0;
    let mut opp_deck_count_sum = 0.0;
    let mut own_deck_count_sum = 0.0;
    let mut fingerprints = BTreeMap::<String, f64>::new();

    for particle in particles {
        let opp = particle.game_state.side(observer_side.opposite());
        let own = particle.game_state.side(observer_side);
        for cid in unique_cards(opp.hand.iter().copied()) {
            *opp_hand.entry(cid).or_default() += particle.weight;
        }
        for cid in unique_cards(opp.deck.iter().copied()) {
            *opp_deck.entry(cid).or_default() += particle.weight;
        }
        for cid in unique_cards(own.deck.iter().copied()) {
            *own_deck.entry(cid).or_default() += particle.weight;
        }
        hand_count_sum += opp.hand.len() as f64;
        opp_deck_count_sum += opp.deck.len() as f64;
        own_deck_count_sum += own.deck.len() as f64;
        *fingerprints
            .entry(format!(
                "{:?}|{:?}|{:?}",
                opp.hand.iter().map(|c| c.0).collect::<Vec<_>>(),
                opp.deck.iter().map(|c| c.0).collect::<Vec<_>>(),
                own.deck.iter().map(|c| c.0).collect::<Vec<_>>()
            ))
            .or_default() += particle.weight;
    }

    let entropy = entropy_from_maps([&opp_hand, &opp_deck, &own_deck], total_weight);
    let diversity = fingerprints.len() as f64 / particles.len().max(1) as f64;
    let particle_len = particles.len().max(1) as f64;
    // First `BELIEF_SUMMARY_DIM` (16) entries are byte-stable for graph
    // compatibility — order MUST NOT change.
    let mut vector = vec![
        particles.len() as f32 / 128.0,
        (hand_count_sum / particle_len) as f32 / 10.0,
        (opp_deck_count_sum / particle_len) as f32 / 20.0,
        (own_deck_count_sum / particle_len) as f32 / 20.0,
        entropy as f32 / 16.0,
        diversity as f32,
        max_prob(&opp_hand, total_weight) as f32,
        max_prob(&opp_deck, total_weight) as f32,
        max_prob(&own_deck, total_weight) as f32,
        prob_mass_by_kind(&opp_hand, total_weight, CardKind::Uma) as f32,
        prob_mass_by_kind(&opp_hand, total_weight, CardKind::Trainer) as f32,
        0.0,
        prob_mass_by_kind(&opp_deck, total_weight, CardKind::Uma) as f32,
        prob_mass_by_kind(&opp_deck, total_weight, CardKind::Trainer) as f32,
        0.0,
        total_weight as f32,
    ];
    debug_assert_eq!(vector.len(), BELIEF_SUMMARY_DIM);

    // Appended per-card opponent-hand presence block (T1.2): one slot per
    // non-pad vocab index. `opp_hand[cid]` is the weighted presence mass for
    // a card; normalizing by `total_weight` yields its marginal presence
    // probability across particles. Cards resolving to vocab index 0
    // (unknown/pad) are skipped. The block sums to ~ the expected number of
    // distinct opponent-hand cards under the belief.
    let cat = catalog();
    let mut presence = vec![0.0f32; BELIEF_HAND_RANGE_DIM];
    for (cid, mass) in &opp_hand {
        if let Some(label) = cat.interner.resolve(*cid) {
            let idx = card_vocab_index(Some(label));
            if idx != 0 {
                presence[idx as usize - 1] = (mass / total_weight) as f32;
            }
        }
    }
    vector.extend_from_slice(&presence);
    debug_assert_eq!(vector.len(), BELIEF_FEATURE_DIM);

    BeliefFeatures {
        schema_version: BELIEF_SCHEMA_VERSION,
        vector,
        opponent_hand_card_presence: probability_rows(&opp_hand, total_weight),
        opponent_deck_card_presence: probability_rows(&opp_deck, total_weight),
        own_deck_card_presence: probability_rows(&own_deck, total_weight),
        hidden_zone_entropy: entropy,
        particle_diversity: diversity,
    }
}

fn unique_cards(cards: impl Iterator<Item = CardId>) -> Vec<CardId> {
    let mut out = Vec::new();
    for cid in cards {
        if !out.contains(&cid) {
            out.push(cid);
        }
    }
    out
}

fn card_labels(cards: impl Iterator<Item = CardId>) -> Vec<String> {
    let cat = catalog();
    cards
        .filter_map(|cid| cat.interner.resolve(cid).map(|s| s.to_string()))
        .collect()
}

fn probability_rows(map: &BTreeMap<CardId, f64>, total_weight: f64) -> Vec<HiddenZoneProbability> {
    let cat = catalog();
    map.iter()
        .filter_map(|(cid, mass)| {
            cat.interner
                .resolve(*cid)
                .map(|label| HiddenZoneProbability {
                    card_id: label.to_string(),
                    probability: mass / total_weight,
                })
        })
        .collect()
}

fn entropy_from_maps<const N: usize>(maps: [&BTreeMap<CardId, f64>; N], total_weight: f64) -> f64 {
    maps.iter()
        .flat_map(|m| m.values())
        .map(|mass| {
            let p = mass / total_weight;
            if p > 0.0 {
                -p * p.ln()
            } else {
                0.0
            }
        })
        .sum()
}

fn max_prob(map: &BTreeMap<CardId, f64>, total_weight: f64) -> f64 {
    map.values().copied().fold(0.0, f64::max) / total_weight
}

enum CardKind {
    Uma,
    Trainer,
}

fn prob_mass_by_kind(map: &BTreeMap<CardId, f64>, total_weight: f64, kind: CardKind) -> f64 {
    let cat = catalog();
    let mass = map
        .iter()
        .filter_map(|(cid, value)| match (cat.get(*cid), &kind) {
            (Some(Card::Umamusume(_)), CardKind::Uma) => Some(*value),
            (Some(Card::Trainer(_)), CardKind::Trainer) => Some(*value),
            _ => None,
        })
        .sum::<f64>();
    mass / total_weight
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::random::{with_rng, Rng};
    use crate::headless_setup::setup_ai_vs_ai_game;
    use crate::policy::card_vocab::card_vocab;

    fn fresh_belief(particle_count: usize) -> PublicBeliefState {
        let rng = Rng::from_seed("belief-test:selfplay", "selfplay");
        let (state, mut rng) = with_rng(rng, setup_ai_vs_ai_game);
        let history =
            PublicHistory::from_state(&state, "matikanetannhauser", "matikanetannhauser", 0);
        let config = BeliefBuildConfig {
            particle_count,
            seed_label: "belief-test".to_string(),
        };
        build_public_belief_state(&state, SideId::Player, history, &config, &mut rng)
    }

    #[test]
    fn belief_builder_emits_particles_and_fixed_width_features() {
        let belief = fresh_belief(8);
        assert_eq!(belief.schema_version, BELIEF_SCHEMA_VERSION);
        assert_eq!(belief.particles.len(), 8);
        assert_eq!(belief.particle_weights.len(), 8);
        assert_eq!(belief.belief_features.vector.len(), BELIEF_FEATURE_DIM);
        assert!(belief.audit.true_state_in_support);
    }

    #[test]
    fn belief_hand_range_dim_matches_card_vocab() {
        // The appended presence block has exactly one slot per non-pad vocab
        // index. If the vocab grows/shrinks this must be regenerated.
        assert_eq!(card_vocab().vocab_size as usize, BELIEF_HAND_RANGE_DIM);
        assert_eq!(BELIEF_HAND_RANGE_DIM, 107);
        assert_eq!(BELIEF_FEATURE_DIM, 16 + BELIEF_HAND_RANGE_DIM);
        assert_eq!(BELIEF_FEATURE_DIM, 123);
    }

    #[test]
    fn presence_block_is_marginal_and_supported() {
        let belief = fresh_belief(16);
        let vector = &belief.belief_features.vector;
        assert_eq!(vector.len(), BELIEF_FEATURE_DIM);

        // (a) The appended block occupies the tail of the vector.
        let presence = &vector[BELIEF_SUMMARY_DIM..];
        assert_eq!(presence.len(), BELIEF_HAND_RANGE_DIM);

        // (b) The block sums to ~ the mean opponent hand size: each particle
        // contributes per-card presence summing (over distinct cards) to its
        // hand-size when cards are unique, and the weighted normalization
        // turns that into the mean distinct-hand-card count. For an opening
        // hand the cards are distinct, so the block sum should track the mean
        // hand size carried in vector slot 1 (= mean_hand / 10).
        let block_sum: f32 = presence.iter().sum();
        let mean_hand_size = vector[1] * 10.0;
        assert!(mean_hand_size > 0.0, "mean hand size should be positive");
        assert!(
            (block_sum - mean_hand_size).abs() < 0.5,
            "presence block sum {block_sum} should track mean hand size {mean_hand_size}",
        );

        // Every nonzero slot must correspond to a card actually present in
        // some particle's opponent hand (zero outside the possible cards).
        let cat = catalog();
        let mut possible_idx = std::collections::BTreeSet::<usize>::new();
        for particle in &belief.particles {
            let opp = particle.game_state.side(belief.observer_side.opposite());
            for cid in opp.hand.iter() {
                if let Some(label) = cat.interner.resolve(*cid) {
                    let idx = card_vocab_index(Some(label));
                    if idx != 0 {
                        possible_idx.insert(idx as usize - 1);
                    }
                }
            }
        }
        for (slot, &value) in presence.iter().enumerate() {
            if value != 0.0 {
                assert!(
                    possible_idx.contains(&slot),
                    "presence slot {slot} nonzero ({value}) but no particle hand holds it",
                );
            }
        }
    }

    #[test]
    fn complement_conserves_true_hidden_multiset_on_fresh_setup() {
        // For a fresh setup_ai_vs_ai_game some public deck candidate minus the
        // opponent's revealed cards must equal the true hidden multiset, so
        // the conservation-guarded complement path is actually taken (not the
        // fallback). This is the invariant the resample debug_assert checks.
        let rng = Rng::from_seed("belief-test:selfplay", "selfplay");
        let (state, _rng) = with_rng(rng, setup_ai_vs_ai_game);
        let opponent_side = SideId::Player.opposite();

        let opponent = state.side(opponent_side);
        let mut true_hidden: Vec<u16> = opponent
            .hand
            .iter()
            .chain(opponent.deck.iter())
            .map(|c| c.0)
            .collect();
        true_hidden.sort_unstable();
        assert!(!true_hidden.is_empty());

        // The id collides across registries (player-side vs AI-side
        // `matikanetannhauser`); at least one candidate must conserve.
        let candidates = deck_lists_by_id("matikanetannhauser");
        assert!(
            candidates.len() >= 1,
            "deck id must resolve to at least one list",
        );
        let conserving = candidates.iter().find_map(|deck_list| {
            let complement = public_complement_pool(&state, opponent_side, deck_list)?;
            let mut sorted: Vec<u16> = complement.iter().map(|c| c.0).collect();
            sorted.sort_unstable();
            (sorted == true_hidden).then_some(sorted)
        });
        assert!(
            conserving.is_some(),
            "some public deck candidate must conserve the true hidden multiset \
             on fresh setup; candidates={} true_len={}",
            candidates.len(),
            true_hidden.len(),
        );
    }
}

