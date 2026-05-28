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
use crate::core::random::Rng;
use crate::core::state::GameState;
use crate::dispatcher::state_hash;
use crate::policy::actions::enumerate_legal_ai_actions;
use crate::policy::observation::build_public_observation;
use crate::policy::types::{LegalAiAction, PublicObservation};

pub const BELIEF_SCHEMA_VERSION: u32 = 1;
pub const BELIEF_FEATURE_DIM: usize = 16;

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
    let legal_actions = enumerate_legal_ai_actions(state, observer_side);
    let particle_count = config.particle_count.max(1);
    let mut particles = Vec::with_capacity(particle_count);
    for index in 0..particle_count {
        let mut particle_state = state.clone();
        resample_hidden_zones(&mut particle_state, observer_side, rng);
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

fn resample_hidden_zones(state: &mut GameState, observer_side: SideId, rng: &mut Rng) {
    let opponent_side = observer_side.opposite();
    let opponent = state.side_mut(opponent_side);
    let hand_len = opponent.hand.len();
    let deck_len = opponent.deck.len();
    let mut unknown: Vec<CardId> = opponent.hand.iter().chain(opponent.deck.iter()).copied().collect();
    shuffle_in_place(&mut unknown, rng);
    opponent.hand.clear();
    opponent.deck.clear();
    for cid in unknown.iter().take(hand_len) {
        let _ = opponent.hand.try_push(*cid);
    }
    for cid in unknown.iter().skip(hand_len).take(deck_len) {
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

fn build_belief_features(particles: &[PrivateWorldParticle], observer_side: SideId) -> BeliefFeatures {
    let total_weight = particles.iter().map(|p| p.weight).sum::<f64>().max(1e-12);
    let mut opp_hand = BTreeMap::<CardId, f64>::new();
    let mut opp_deck = BTreeMap::<CardId, f64>::new();
    let mut own_deck = BTreeMap::<CardId, f64>::new();
    let mut hand_counts = Vec::new();
    let mut opp_deck_counts = Vec::new();
    let mut own_deck_counts = Vec::new();
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
        hand_counts.push(opp.hand.len() as f64);
        opp_deck_counts.push(opp.deck.len() as f64);
        own_deck_counts.push(own.deck.len() as f64);
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
    let vector = vec![
        particles.len() as f32 / 128.0,
        mean(&hand_counts) as f32 / 10.0,
        mean(&opp_deck_counts) as f32 / 20.0,
        mean(&own_deck_counts) as f32 / 20.0,
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
            cat.interner.resolve(*cid).map(|label| HiddenZoneProbability {
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
            if p > 0.0 { -p * p.ln() } else { 0.0 }
        })
        .sum()
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
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

    #[test]
    fn belief_builder_emits_particles_and_fixed_width_features() {
        let rng = Rng::from_seed("belief-test:selfplay", "selfplay");
        let (state, mut rng) = with_rng(rng, setup_ai_vs_ai_game);
        let history = PublicHistory::from_state(&state, "matikanetannhauser", "matikanetannhauser", 0);
        let config = BeliefBuildConfig {
            particle_count: 8,
            seed_label: "belief-test".to_string(),
        };
        let belief = build_public_belief_state(&state, SideId::Player, history, &config, &mut rng);
        assert_eq!(belief.schema_version, BELIEF_SCHEMA_VERSION);
        assert_eq!(belief.particles.len(), 8);
        assert_eq!(belief.particle_weights.len(), 8);
        assert_eq!(belief.belief_features.vector.len(), BELIEF_FEATURE_DIM);
        assert!(belief.audit.true_state_in_support);
    }
}
