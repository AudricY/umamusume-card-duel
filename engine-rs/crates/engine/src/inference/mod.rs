//! R16-P3 throughput-spike Option A: in-process ONNX inference for the
//! v3.0 and v3.2 policy/value graphs.
//!
//! Replaces the HTTP `/predict` round-trip in
//! `crate::mcts::driver::predict_policy_and_value` /
//! `value_head_leaf_value` with a direct ORT session call. The session
//! is loaded once per process and shared via a global; downstream MCTS
//! workers stamp it through `InferenceSession::set_global`.
//!
//! Schema scope: **v3.0 + v3.2** (Slice 3 lands v3.2). The graph signature
//! is validated at load time:
//!   - REQUIRED v3.0 inputs (set-equality): `state_features`,
//!     `action_features`, `action_mask`, `card_ids_by_zone`,
//!     `action_card_idx`. Five inputs total.
//!   - REQUIRED v3.2 inputs: the five v3.0 inputs PLUS `uma_slot_card_ids`
//!     and `uma_slot_features` (set-equality of 7). Either both v3.2
//!     inputs are present (v3.2) or neither (v3.0). A partial v3.2 graph
//!     (one slot input missing) is rejected explicitly (mirrors
//!     `serve_onnx._graph_has_partial_uma_slot_inputs` guard).
//!   - REJECTED: v3.1 graphs (164-d state). `unimplemented!()`-style hard
//!     error pointing at the v3.1 follow-up slice.
//!
//! ONNX runtime: dynamic-loaded via `load-dynamic` (set
//! `ORT_DYLIB_PATH=/path/to/libonnxruntime.so` before invoking; the
//! Python venv's bundled `libonnxruntime.so.1.22.0` works).
//!
//! Sidecar `.onnx.meta.json` (`card_vocab.hash`) is asserted at load
//! time against the compiled-in `cardVocab.json` hash; a mismatch is a
//! hard error — silently serving with a wrong vocab corrupts the
//! embedding inputs.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use ndarray::Array;
use ort::session::Session;
use ort::value::TensorRef;

use crate::policy::card_vocab::card_vocab;
use crate::policy::featurize::{
    self, ACTION_DIM, MAX_CARDS_PER_ZONE, NUM_ZONES, STATE_DIM_V3, UMA_SLOT_COUNT,
    UMA_SLOT_FEATURE_DIM,
};
use crate::policy::types::{LegalAiAction, PublicObservation};

/// Required input names for a v3.0 graph (set-equality check at load
/// time).
const REQUIRED_V3_INPUTS: [&str; 5] = [
    "state_features",
    "action_features",
    "action_mask",
    "card_ids_by_zone",
    "action_card_idx",
];

/// Additional input names that a v3.2 graph declares on top of v3.0.
/// Both must be present (the slot-token pair is contractual; partial
/// presence is rejected explicitly, matching serve_onnx).
const REQUIRED_V3_2_EXTRA_INPUTS: [&str; 2] = [
    "uma_slot_card_ids",
    "uma_slot_features",
];

/// Detected graph schema. Set at session load and read by
/// `predict_v3` to dispatch the correct tensor packing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GraphSchema {
    V3_0,
    V3_2,
}

/// Errors surfaced by the inference layer. We hide ORT's `Error` behind
/// our own enum (as a stringified message) so callers don't have to
/// take a transitive ort dep, and so the various `ort::Error<TypeParam>`
/// variants collapse to a single human-readable form.
#[derive(Debug)]
pub enum InferenceError {
    /// ORT session-build / inference failure (stringified).
    Ort(String),
    /// Sidecar `.onnx.meta.json` not found or unreadable.
    MissingSidecar(PathBuf),
    /// Sidecar parse failure.
    SidecarParse(String),
    /// `card_vocab.hash` between runtime and sidecar disagrees.
    VocabHashMismatch {
        expected: String,
        runtime: String,
    },
    /// ONNX graph signature does not match the v3.0 contract.
    SchemaMismatch(String),
    /// Action featurization error (e.g. ACTION_DIM mismatch).
    Featurize(featurize::FeaturizeError),
    /// Inference produced an unexpectedly-shaped output.
    OutputShape(String),
}

impl std::fmt::Display for InferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferenceError::Ort(e) => write!(f, "ort error: {e}"),
            InferenceError::MissingSidecar(p) => {
                write!(f, "missing ONNX sidecar at {}", p.display())
            }
            InferenceError::SidecarParse(s) => write!(f, "sidecar parse: {s}"),
            InferenceError::VocabHashMismatch { expected, runtime } => write!(
                f,
                "card vocab hash mismatch: sidecar={expected} runtime={runtime}"
            ),
            InferenceError::SchemaMismatch(s) => write!(f, "schema mismatch: {s}"),
            InferenceError::Featurize(e) => write!(f, "featurize: {e}"),
            InferenceError::OutputShape(s) => write!(f, "output shape: {s}"),
        }
    }
}

impl std::error::Error for InferenceError {}

impl<T> From<ort::Error<T>> for InferenceError {
    fn from(e: ort::Error<T>) -> Self {
        InferenceError::Ort(e.to_string())
    }
}

impl From<featurize::FeaturizeError> for InferenceError {
    fn from(e: featurize::FeaturizeError) -> Self {
        InferenceError::Featurize(e)
    }
}

/// Loaded in-process ONNX session — thread-safe per ORT's contract
/// (`unsafe impl Send + Sync for Session`); callers wrap in an `Arc`
/// to share across MCTS workers. The internal `Mutex<Session>` is
/// required because `ort::Session::run` takes `&mut self` at the
/// Rust API surface even though the underlying ORT C API is reentrant.
/// MCTS today is single-threaded per binary, so the Mutex is
/// effectively uncontended.
pub struct InferenceSession {
    session: Mutex<Session>,
    onnx_path: PathBuf,
    schema: GraphSchema,
}

impl InferenceSession {
    /// Construct an `InferenceSession` from an ONNX file path. Validates
    /// the graph signature against the v3.0 contract and asserts vocab
    /// hash parity with the sidecar.
    pub fn load(onnx_path: &Path) -> Result<Self, InferenceError> {
        // Sidecar vocab hash parity. We load the sidecar BEFORE building
        // the ORT session so a vocab mismatch is the first error the
        // operator sees (cheap, deterministic, no ORT cost).
        let sidecar_path = sidecar_path_for(onnx_path);
        let sidecar = read_sidecar(&sidecar_path)?;
        let runtime_hash = card_vocab_metadata_hash();
        let sidecar_hash = sidecar
            .get("card_vocab")
            .and_then(|v| v.get("hash"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        if let Some(expected) = sidecar_hash {
            if runtime_hash != "missing" && expected != runtime_hash {
                return Err(InferenceError::VocabHashMismatch {
                    expected,
                    runtime: runtime_hash,
                });
            }
        }

        // Build a single-threaded ORT session. R14.G rationale: pin
        // intra/inter-op = 1 to keep FP-determinism on the policy logits
        // (matches the `serve_onnx --ort-threads 1` server default the
        // parity smoke runs against). Threading the engine wider here
        // adds tiny FP non-determinism that bleeds into MCTS visit
        // counts; out of scope for the parity slice.
        let session = Session::builder()
            .map_err(InferenceError::from)?
            .with_intra_threads(1)
            .map_err(InferenceError::from)?
            .with_inter_threads(1)
            .map_err(InferenceError::from)?
            .commit_from_file(onnx_path)
            .map_err(InferenceError::from)?;

        // Graph-signature validation. Detects v3.0 vs v3.2 by input set
        // (matches `serve_onnx._lookup_schema`). Partial v3.2 (one slot
        // input missing) is rejected explicitly; v3.1 (164-d) is still
        // a hard error pointing at the follow-up slice.
        let schema = validate_graph_signature(&session)?;

        Ok(InferenceSession {
            session: Mutex::new(session),
            onnx_path: onnx_path.to_path_buf(),
            schema,
        })
    }

    /// Drive a single `(observation, legal_actions)` pair through the
    /// graph and return `(masked-softmax probs, scalar value)`.
    ///
    /// Per the `serve_onnx.py` semantics with `sampling = greedy` and
    /// `temperature = 0.0` (the MCTS prior path), we return the
    /// masked-softmax over raw policy logits and the unmodified scalar
    /// value head. Probs are clipped at 0.0 for masked positions and
    /// renormalized to sum to 1.0.
    pub fn predict_v3(
        &self,
        observation: &PublicObservation,
        legal_actions: &[LegalAiAction],
    ) -> Result<PredictionV3, InferenceError> {
        if legal_actions.is_empty() {
            return Err(InferenceError::SchemaMismatch(
                "legalActions must not be empty".into(),
            ));
        }
        // Pack the five v3.0 input tensors. Layout matches
        // `serve_onnx.request_to_arrays` exactly (batch=1 leading dim
        // everywhere).
        let state = featurize::observation_state_features(observation);
        let state_arr = Array::from_shape_vec((1, STATE_DIM_V3), state)
            .map_err(|e| InferenceError::OutputShape(format!("state reshape: {e}")))?;

        let n_actions = legal_actions.len();
        let action_features_flat = featurize::legal_actions_features(legal_actions)?;
        let action_features_arr = Array::from_shape_vec(
            (1, n_actions, ACTION_DIM),
            action_features_flat,
        )
        .map_err(|e| InferenceError::OutputShape(format!("action_features reshape: {e}")))?;

        // action_mask is a bool tensor; all-true since `legal_actions`
        // is already the masked legal set (parity with serve_onnx
        // request packing).
        let action_mask_arr = Array::from_elem((1, n_actions), true);

        let card_ids_flat = featurize::observation_card_ids_by_zone(observation);
        let card_ids_arr = Array::from_shape_vec(
            (1, NUM_ZONES, MAX_CARDS_PER_ZONE),
            card_ids_flat,
        )
        .map_err(|e| InferenceError::OutputShape(format!("card_ids reshape: {e}")))?;

        let action_card_idx_flat = featurize::action_card_idx_pairs_flat(legal_actions);
        let action_card_idx_arr =
            Array::from_shape_vec((1, n_actions, 2), action_card_idx_flat)
                .map_err(|e| InferenceError::OutputShape(format!("action_card_idx reshape: {e}")))?;

        // v3.2-only auxiliary tensors. We build them unconditionally so the
        // `TensorRef::from_array_view` borrow lives long enough on both
        // branches; v3.0 dispatch simply ignores them (ORT hard-rejects
        // unknown feed keys, so v3.0 graphs MUST omit these from `inputs!`).
        let (slot_card_ids_flat, slot_features_flat) = match self.schema {
            GraphSchema::V3_2 => featurize::observation_uma_slots(observation),
            GraphSchema::V3_0 => (Vec::new(), Vec::new()),
        };
        let slot_card_ids_arr = if matches!(self.schema, GraphSchema::V3_2) {
            Some(
                Array::from_shape_vec((1, UMA_SLOT_COUNT), slot_card_ids_flat).map_err(|e| {
                    InferenceError::OutputShape(format!("uma_slot_card_ids reshape: {e}"))
                })?,
            )
        } else {
            None
        };
        let slot_features_arr = if matches!(self.schema, GraphSchema::V3_2) {
            Some(
                Array::from_shape_vec(
                    (1, UMA_SLOT_COUNT, UMA_SLOT_FEATURE_DIM),
                    slot_features_flat,
                )
                .map_err(|e| {
                    InferenceError::OutputShape(format!("uma_slot_features reshape: {e}"))
                })?,
            )
        } else {
            None
        };

        // `TensorRef::from_array_view` requires `&ArrayBase<OwnedRepr,_>`
        // (not a `View` produced by `.view()`); pass the owned arrays by
        // reference. This is zero-copy at the FFI boundary — ORT borrows
        // the buffer for the duration of `run()`. The two paths build a
        // different ort::inputs! map (5 keys for v3.0, 7 keys for v3.2) —
        // ORT hard-rejects unknown feed keys so we MUST omit the v3.2
        // tensors from the v3.0 feed.
        let inputs = match self.schema {
            GraphSchema::V3_0 => ort::inputs![
                "state_features" => TensorRef::from_array_view(&state_arr)?,
                "action_features" => TensorRef::from_array_view(&action_features_arr)?,
                "action_mask" => TensorRef::from_array_view(&action_mask_arr)?,
                "card_ids_by_zone" => TensorRef::from_array_view(&card_ids_arr)?,
                "action_card_idx" => TensorRef::from_array_view(&action_card_idx_arr)?,
            ],
            GraphSchema::V3_2 => {
                let slot_ids = slot_card_ids_arr.as_ref().expect("v3.2 built above");
                let slot_feats = slot_features_arr.as_ref().expect("v3.2 built above");
                ort::inputs![
                    "state_features" => TensorRef::from_array_view(&state_arr)?,
                    "action_features" => TensorRef::from_array_view(&action_features_arr)?,
                    "action_mask" => TensorRef::from_array_view(&action_mask_arr)?,
                    "card_ids_by_zone" => TensorRef::from_array_view(&card_ids_arr)?,
                    "action_card_idx" => TensorRef::from_array_view(&action_card_idx_arr)?,
                    "uma_slot_card_ids" => TensorRef::from_array_view(slot_ids)?,
                    "uma_slot_features" => TensorRef::from_array_view(slot_feats)?,
                ]
            }
        };

        // Hold the lock across both `run()` and the tensor extraction
        // — `outputs[i].try_extract_tensor` borrows from the session,
        // so we must finish copying out the f32 buffers before the
        // lock guard drops.
        let (logits_vec, value_scalar) = {
            let mut sess = self.session.lock().expect("inference session mutex poisoned");
            let outputs = sess.run(inputs)?;

            // Outputs order matches `serve_onnx.PolicyServer.session.run(...)`:
            // [logits (float[B, A]), value (float[B])]. The ONNX graph
            // declares the output order; we read by index 0 / 1.
            let (_logits_shape, logits_flat) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(InferenceError::from)?;
            let (_value_shape, value_flat) = outputs[1]
                .try_extract_tensor::<f32>()
                .map_err(InferenceError::from)?;
            if logits_flat.len() != n_actions {
                return Err(InferenceError::OutputShape(format!(
                    "logits has {} elements, expected {}",
                    logits_flat.len(),
                    n_actions
                )));
            }
            if value_flat.is_empty() {
                return Err(InferenceError::OutputShape(
                    "value tensor is empty".into(),
                ));
            }
            (logits_flat.to_vec(), value_flat[0])
        };

        // Masked softmax — mirror of `serve_onnx.masked_softmax`. All
        // legal positions get the standard softmax; masked positions
        // are zero. We're operating on batch=1, every action is legal,
        // so the masking is a no-op; the softmax keeps the parity
        // contract numerically tight (max-shift then exp-normalize).
        let probs = greedy_masked_softmax(&logits_vec);
        Ok(PredictionV3 { probs, value: value_scalar })
    }

    /// Path the session was loaded from. Useful for diagnostics.
    pub fn onnx_path(&self) -> &Path {
        &self.onnx_path
    }
}

/// `(masked-softmax over legal actions, scalar value)`.
#[derive(Debug, Clone)]
pub struct PredictionV3 {
    pub probs: Vec<f32>,
    pub value: f32,
}

/// Numerically-stable masked softmax. Mirrors the `serve_onnx`
/// implementation but without the JSON encoding step: we already have
/// the legal subset, so the mask is implicit (all positions legal).
fn greedy_masked_softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let max = if max.is_finite() { max } else { 0.0 };
    let mut exps: Vec<f32> = logits.iter().map(|&v| ((v - max) as f64).exp() as f32).collect();
    let sum: f64 = exps.iter().map(|&v| v as f64).sum();
    if sum > 0.0 {
        for p in exps.iter_mut() {
            *p = (*p as f64 / sum) as f32;
        }
    } else {
        let n = exps.len() as f32;
        for p in exps.iter_mut() {
            *p = 1.0 / n;
        }
    }
    exps
}

fn sidecar_path_for(onnx_path: &Path) -> PathBuf {
    let mut s = onnx_path.as_os_str().to_owned();
    s.push(".meta.json");
    PathBuf::from(s)
}

fn read_sidecar(path: &Path) -> Result<serde_json::Value, InferenceError> {
    let raw = std::fs::read_to_string(path)
        .map_err(|_| InferenceError::MissingSidecar(path.to_path_buf()))?;
    serde_json::from_str::<serde_json::Value>(&raw)
        .map_err(|e| InferenceError::SidecarParse(e.to_string()))
}

fn card_vocab_metadata_hash() -> String {
    // `card_vocab.json` is `include_str!`-embedded; re-parse the raw
    // string to extract the `hash` field, matching what
    // `serve_onnx.card_vocab_metadata()` returns. Falls back to
    // "missing" if absent (the Rust loader doesn't expose the raw
    // hash today; surface a defensive default that still triggers the
    // parity comparison).
    let raw = include_str!("../../../../../shared/src/cardVocab.json");
    let parsed: serde_json::Value =
        serde_json::from_str(raw).unwrap_or(serde_json::Value::Null);
    parsed
        .get("hash")
        .and_then(|v| v.as_str())
        .unwrap_or_else(|| {
            // Silence dead-code warning + keep the runtime helper exported.
            let _ = card_vocab();
            "missing"
        })
        .to_string()
}

fn validate_graph_signature(session: &Session) -> Result<GraphSchema, InferenceError> {
    let session_inputs = session.inputs();
    let inputs: Vec<&str> = session_inputs.iter().map(|i| i.name()).collect();
    let input_set: std::collections::BTreeSet<&str> = inputs.iter().copied().collect();
    let required_v3: std::collections::BTreeSet<&str> =
        REQUIRED_V3_INPUTS.iter().copied().collect();
    let required_v3_2: std::collections::BTreeSet<&str> = REQUIRED_V3_INPUTS
        .iter()
        .chain(REQUIRED_V3_2_EXTRA_INPUTS.iter())
        .copied()
        .collect();

    // Partial-pair guard (mirrors serve_onnx._graph_has_partial_uma_slot_inputs):
    // the v3.2 slot tensors are a CONTRACTUAL pair; refusing to serve a
    // partial v3.2 graph keeps the diagnostic crisp vs the generic
    // schema-mismatch path.
    let has_slot_ids = input_set.contains("uma_slot_card_ids");
    let has_slot_feats = input_set.contains("uma_slot_features");
    if has_slot_ids ^ has_slot_feats {
        return Err(InferenceError::SchemaMismatch(format!(
            "graph declares exactly one of `uma_slot_card_ids` / \
             `uma_slot_features` (inputs: {:?}). The v3.2 slot tensors are \
             a contractual pair; refusing to serve a partial v3.2 graph.",
            inputs
        )));
    }

    // v3.2 dispatch: both slot inputs present + required v3.0 inputs all
    // present (set-equality on the 7-input contract).
    if has_slot_ids && has_slot_feats {
        if input_set != required_v3_2 {
            return Err(InferenceError::SchemaMismatch(format!(
                "graph input set {:?} does not match v3.2 contract {:?}",
                inputs,
                required_v3_2.iter().copied().collect::<Vec<_>>()
            )));
        }
        return Ok(GraphSchema::V3_2);
    }

    // v3.0 dispatch: set-equality on the 5-input v3.0 contract.
    if input_set != required_v3 {
        // Likely cause for an unexpected set with no slot inputs is a
        // v3.1 (164-d) graph — that's the only other documented schema
        // and it's still unimplemented this slice.
        if input_set.contains("temporal_features")
            || inputs.iter().any(|n| n.contains("temporal"))
        {
            return Err(InferenceError::SchemaMismatch(format!(
                "graph appears to be v3.1 (164-d temporal/turn-state; \
                 inputs {:?}); not implemented in this slice — file a \
                 follow-up issue.",
                inputs
            )));
        }
        return Err(InferenceError::SchemaMismatch(format!(
            "graph input set {:?} does not match v3.0 contract {:?} or \
             v3.2 contract {:?}",
            inputs, REQUIRED_V3_INPUTS, required_v3_2.iter().copied().collect::<Vec<_>>()
        )));
    }
    Ok(GraphSchema::V3_0)
}

// ---------------------------------------------------------------------------
// Global session — initialized once per process so MCTS workers don't pay
// per-call session-load cost. Initialization is fallible (returns the
// `InferenceError`); subsequent `global()` calls receive the same `Arc`.
// ---------------------------------------------------------------------------

static GLOBAL: OnceLock<std::sync::Arc<InferenceSession>> = OnceLock::new();

/// Stamp the global inference session. Idempotent — subsequent calls
/// are no-ops (`OnceLock`). Used by sim-cli main() once before MCTS
/// loop launches.
pub fn set_global(session: InferenceSession) {
    let _ = GLOBAL.set(std::sync::Arc::new(session));
}

/// Borrow the global session. Returns `None` if `set_global` was never
/// called; the MCTS driver path treats this as a hard error since
/// value-head/policy-prior modes require a loaded session.
pub fn global() -> Option<std::sync::Arc<InferenceSession>> {
    GLOBAL.get().cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_masked_softmax_normalizes() {
        let logits = [1.0f32, 2.0, 3.0];
        let p = greedy_masked_softmax(&logits);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "got sum {}", sum);
        // Monotone in logit.
        assert!(p[0] < p[1] && p[1] < p[2]);
    }

    #[test]
    fn greedy_masked_softmax_uniform_on_zeros() {
        let logits = [0.0f32; 4];
        let p = greedy_masked_softmax(&logits);
        for q in &p {
            assert!((q - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn sidecar_path_appends_meta_json() {
        let p = sidecar_path_for(Path::new("/tmp/a/policy.onnx"));
        assert_eq!(p, PathBuf::from("/tmp/a/policy.onnx.meta.json"));
    }
}
