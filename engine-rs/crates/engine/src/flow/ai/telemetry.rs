//! Bit-identical port of `frontend/src/game/engine/flow/ai/telemetry.ts`.
//!
//! TS keeps a module-global `seenTurnGoalKeys = new Set<string>()`,
//! `history: AiTelemetryRecord[]`, and `nextSequence: number`. It also
//! reads `globalThis.__UMA_AI_TELEMETRY__` — when unset, every emit is a
//! no-op.
//!
//! In Rust we mirror the contract with thread-local state. The fingerprint
//! contract excludes `state.log` and the telemetry buffer entirely; the
//! engine never reads back from this module. The dedup set bound (6000)
//! and history cap (1200) are preserved for parity with TS — they govern
//! the "long sessions" memory profile but do not affect game state.

use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;

use serde_json::{Map, Value};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AiTelemetryEvent {
    TurnGoal,
    TrainerBundleScores,
    CombatCandidates,
}

impl AiTelemetryEvent {
    pub fn tag(self) -> &'static str {
        match self {
            AiTelemetryEvent::TurnGoal => "turn_goal",
            AiTelemetryEvent::TrainerBundleScores => "trainer_bundle_scores",
            AiTelemetryEvent::CombatCandidates => "combat_candidates",
        }
    }
}

#[derive(Debug, Clone)]
pub struct AiTelemetryRecord {
    pub seq: u64,
    pub ts: i64,
    pub event: AiTelemetryEvent,
    pub payload: Map<String, Value>,
}

thread_local! {
    static SEEN_TURN_GOAL_KEYS: RefCell<BTreeSet<String>> = RefCell::new(BTreeSet::new());
    static HISTORY: RefCell<Vec<AiTelemetryRecord>> = const { RefCell::new(Vec::new()) };
    static NEXT_SEQUENCE: Cell<u64> = const { Cell::new(1) };
    /// Mirror of `globalThis.__UMA_AI_TELEMETRY__`. Disabled by default —
    /// every emit is a no-op until explicitly turned on.
    static ENABLED: Cell<bool> = const { Cell::new(false) };
}

pub fn set_enabled(enabled: bool) {
    ENABLED.with(|c| c.set(enabled));
}

pub fn is_enabled() -> bool {
    ENABLED.with(|c| c.get())
}

/// Mirror of `emitAiTelemetry(event, payload)`. No-op when telemetry is
/// disabled (the default), matching the TS contract.
pub fn emit_ai_telemetry(event: AiTelemetryEvent, payload: Map<String, Value>) {
    if !is_enabled() {
        return;
    }
    if event == AiTelemetryEvent::TurnGoal {
        let mut key_obj = Map::new();
        key_obj.insert("event".into(), Value::String(event.tag().to_string()));
        for field in ["turn", "side", "phase", "goal"] {
            key_obj.insert(field.into(), payload.get(field).cloned().unwrap_or(Value::Null));
        }
        // Note: TS keys this on `reasonTags` (not `tags`); preserve.
        key_obj.insert(
            "tags".into(),
            payload.get("reasonTags").cloned().unwrap_or(Value::Null),
        );
        let key = Value::Object(key_obj).to_string();
        let seen = SEEN_TURN_GOAL_KEYS.with(|cell| {
            let mut set = cell.borrow_mut();
            if set.contains(&key) {
                return true;
            }
            set.insert(key);
            // Keep memory bounded across long sessions.
            if set.len() > 6000 {
                set.clear();
            }
            false
        });
        if seen {
            return;
        }
    }

    let seq = NEXT_SEQUENCE.with(|c| {
        let v = c.get();
        c.set(v + 1);
        v
    });
    let record = AiTelemetryRecord {
        seq,
        ts: 0, // TS uses Date.now(); we keep deterministic 0.
        event,
        payload,
    };
    HISTORY.with(|cell| {
        let mut hist = cell.borrow_mut();
        hist.push(record);
        let len = hist.len();
        if len > 1200 {
            hist.drain(0..len - 1200);
        }
    });
}

pub fn get_ai_telemetry_snapshot() -> Vec<AiTelemetryRecord> {
    HISTORY.with(|cell| cell.borrow().clone())
}

pub fn clear_ai_telemetry() {
    HISTORY.with(|cell| cell.borrow_mut().clear());
    SEEN_TURN_GOAL_KEYS.with(|cell| cell.borrow_mut().clear());
}
