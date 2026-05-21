//! Card-id interning: string ids → `CardId(u16)`.
//!
//! TS state uses string card ids everywhere (`deck: string[]`, `hand:
//! string[]`, etc.). The Rust port interns them once at catalog load. With
//! ~600 cards in the catalog, a `u16` index is ample (and leaves 0xFFFF
//! as a sentinel if needed).
//!
//! The interner is build-time data — populated by the codegen step
//! (Phase 1b) and never mutated thereafter. Iteration order matches the
//! source-declaration order of the catalog JSON.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CardId(pub u16);

impl CardId {
    pub const NONE: CardId = CardId(u16::MAX);

    pub fn is_none(self) -> bool {
        self.0 == u16::MAX
    }

    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, Default, Clone)]
pub struct CardIdInterner {
    by_id: IndexMap<String, CardId>,
    ids: Vec<String>,
}

impl CardIdInterner {
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern a card id. Idempotent. Panics if the interner overflows u16.
    pub fn intern(&mut self, id: &str) -> CardId {
        if let Some(&existing) = self.by_id.get(id) {
            return existing;
        }
        let next = self.ids.len();
        assert!(next < u16::MAX as usize, "CardId interner overflow");
        let cid = CardId(next as u16);
        self.ids.push(id.to_string());
        self.by_id.insert(id.to_string(), cid);
        cid
    }

    pub fn get(&self, id: &str) -> Option<CardId> {
        self.by_id.get(id).copied()
    }

    pub fn resolve(&self, id: CardId) -> Option<&str> {
        self.ids.get(id.index()).map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_is_idempotent_and_preserves_order() {
        let mut interner = CardIdInterner::new();
        let a = interner.intern("alpha");
        let b = interner.intern("bravo");
        let a2 = interner.intern("alpha");
        assert_eq!(a, a2);
        assert_ne!(a, b);
        assert_eq!(interner.resolve(a), Some("alpha"));
        assert_eq!(interner.resolve(b), Some("bravo"));
    }
}
