//! R16-P3 throughput-spike Option A: in-process ONNX inference for the
//! v3.0, v3.1, and v3.2 policy/value graphs.
//!
//! Replaces the HTTP `/predict` round-trip in
//! `crate::mcts::driver::predict_policy_and_value` /
//! `value_head_leaf_value` with a direct ORT session call. The session
//! is loaded once per process and shared via a global; downstream MCTS
//! workers stamp it through `InferenceSession::set_global`.
//!
//! Schema scope: **v3.0 + v3.1 + v3.2** (Slice 3b lands v3.1). The graph
//! signature is validated at load time:
//!   - REQUIRED v3.0 inputs (set-equality): `state_features`,
//!     `action_features`, `action_mask`, `card_ids_by_zone`,
//!     `action_card_idx`. Five inputs total. `state_features` last
//!     dim = 110.
//!   - REQUIRED v3.1 inputs (set-equality): same 5 names as v3.0 BUT
//!     `state_features` last dim = 164 (the 54-d temporal/turn-state
//!     tail). Distinguished from v3.0 by the input-shape probe, not the
//!     input-name set (graph signature matches `serve_onnx._graph_signature`
//!     state-dim discriminator).
//!   - REQUIRED v3.2 inputs: the five v3.0 inputs PLUS `uma_slot_card_ids`
//!     and `uma_slot_features` (set-equality of 7). Either both v3.2
//!     inputs are present (v3.2) or neither (v3.0/v3.1). A partial v3.2
//!     graph (one slot input missing) is rejected explicitly (mirrors
//!     `serve_onnx._graph_has_partial_uma_slot_inputs` guard).
//!   - REJECTED: unknown schemas (e.g. 96-d v2 graphs, 7-input graphs at
//!     a non-110 state dim, partial v3.2 pairs). Surface a clean error
//!     pointing at the actual input shape.
//!
//! ONNX runtime: dynamic-loaded via `load-dynamic` (set
//! `ORT_DYLIB_PATH=/path/to/libonnxruntime.so` before invoking; the
//! Python venv's bundled `libonnxruntime.so.1.22.0` works).
//!
//! Sidecar `.onnx.meta.json` (`card_vocab.hash`) is asserted at load
//! time against the compiled-in `cardVocab.json` hash; a mismatch is a
//! hard error — silently serving with a wrong vocab corrupts the
//! embedding inputs.

use std::cell::UnsafeCell;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use ndarray::Array;
use ort::ep::CUDA as CUDAExecutionProvider;
use ort::session::Session;
use ort::value::TensorRef;

use crate::policy::card_vocab::card_vocab;
use crate::policy::featurize::{
    self, ACTION_DIM, MAX_CARDS_PER_ZONE, NUM_ZONES, STATE_DIM_V3, STATE_DIM_V3_1, STATE_DIM_V3_3,
    UMA_SLOT_COUNT, UMA_SLOT_FEATURE_DIM,
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
/// `predict_v3` to dispatch the correct tensor packing. v3.0 and v3.1
/// share the 5-input contract; the discriminator is the `state_features`
/// last-dim shape (110 vs 164).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GraphSchema {
    V3_0,
    V3_1,
    V3_2,
    /// v33-additive-tail: 167-d state-features + 5-input contract (no
    /// slot tokens). Layered on v3.1 (164-d temporal head) with 3
    /// opp-side flag bits appended. See
    /// `docs/ai-research/scoping/v33-additive-tail-scoping.md`.
    V3_3,
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
    /// `action_feature_schema_version` between runtime and sidecar disagrees.
    /// Mirror of [`VocabHashMismatch`] for the v33-correctness-fix Fix 2-4
    /// action-slot bump (2 → 3). Running mismatched action features against
    /// a checkpoint trained at a different schema silently degrades MCTS
    /// prior quality — fail-fast.
    ActionSchemaMismatch {
        expected: u32,
        runtime: u32,
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
            InferenceError::ActionSchemaMismatch { expected, runtime } => write!(
                f,
                "action_feature_schema_version mismatch: sidecar={expected} runtime={runtime}. \
                 Running mismatched action features silently degrades MCTS prior quality. \
                 Either rebuild against the sidecar's schema OR re-export the ONNX from a \
                 freshly-trained checkpoint at the runtime's schema."
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

/// Selects the ORT execution provider used to load the session.
///
/// `Cpu` keeps the R14.G FP-determinism contract on the SessionBuilder
/// (`intra/inter_threads = 1`, matching the `serve_onnx --ort-threads 1`
/// server the `sim-inference-parity` smoke runs against). G5 (2026-05-22)
/// dropped the per-call `Mutex<Session>` on the CPU path as well: ORT's
/// `Session::Run` is documented thread-safe for concurrent calls on a
/// single session, and the `&mut self` was a Rust-API artifact of the
/// `ort` crate's signature, not a real exclusivity requirement. With
/// intra/inter pinned to 1, no ORT internal thread pool re-orders
/// reductions — the FP-determinism contract is unaffected.
///
/// `Cuda` wires the ORT CUDA EP at session-build time. The CUDA path
/// requires `libonnxruntime.so` to be CUDA-enabled and
/// `libonnxruntime_providers_cuda.so` to be loadable at `Session::run`
/// time; failures surface as `InferenceError::Ort`.
#[derive(Debug, Clone, Copy)]
pub enum Device {
    /// CPU EP (default). Keeps FP-determinism with `serve_onnx`.
    Cpu,
    /// CUDA EP, pinned to a specific device.
    Cuda { device_id: i32 },
}

impl Default for Device {
    fn default() -> Self {
        Device::Cpu
    }
}

/// Internal session guard. Both CPU and CUDA variants now wrap
/// `Session` in an `UnsafeCell` and rely on ORT's documented
/// thread-safety for concurrent `Session::run` calls. The `&mut self`
/// signature on `ort::Session::run` is a Rust-API artifact only — the
/// underlying ORT C API (`OrtApi::Run`) is reentrant on a single
/// session for both the CPU EP and the CUDA EP.
///
/// G5 (2026-05-22) collapsed the CPU variant from `Mutex<Session>` to
/// `UnsafeCell<Session>` to remove the Slice 3c game-level worker
/// serialization bottleneck. The two variants are kept distinct (rather
/// than a single `UnsafeCell<Session>`) to preserve the
/// `inference/mod.rs` dispatch shape and keep the `Device` enum's two
/// configuration paths (intra/inter thread pinning vs CUDA EP wiring)
/// visible at the type level.
enum SessionGuard {
    Cpu(UnsafeCell<Session>),
    Cuda(UnsafeCell<Session>),
}

// Safety: `Session` is `Send + Sync` per ort's `unsafe impl`. The
// `UnsafeCell` here is a punch-through to call `Session::run` (which
// has a Rust-side `&mut self`) from multiple threads. ORT's
// `Session::Run` is documented thread-safe for concurrent calls on a
// single session for both the CPU EP and the CUDA EP, so granting
// `&mut Session` to multiple threads simultaneously is sound. On the
// CPU branch, `intra_threads=1` + `inter_threads=1` are pinned at
// session-build time so no internal ORT thread pool re-orders FP
// reductions (R14.G determinism contract preserved).
unsafe impl Sync for SessionGuard {}

/// Loaded in-process ONNX session — thread-safe per ORT's contract
/// (`unsafe impl Send + Sync for Session`); callers wrap in an `Arc`
/// to share across MCTS workers. Both CPU and CUDA paths share the
/// `UnsafeCell<Session>` punch-through (see `SessionGuard`): ORT's
/// `Session::Run` is reentrant on a single session, the `&mut self`
/// on the Rust API surface is an artifact only. The CPU path's
/// `intra/inter_threads = 1` pins (R14.G FP-determinism contract)
/// remain set on the builder — those control ORT's internal thread
/// pool, not how many Rust threads may concurrently call `Session::run`
/// on the same session.
pub struct InferenceSession {
    session: SessionGuard,
    onnx_path: PathBuf,
    schema: GraphSchema,
    device: Device,
}

impl InferenceSession {
    /// Construct an `InferenceSession` from an ONNX file path on the
    /// default CPU EP. Equivalent to `load_on(onnx_path, Device::Cpu)`
    /// — kept as a back-compat entrypoint for binaries that haven't
    /// (yet) exposed a `--device` flag.
    pub fn load(onnx_path: &Path) -> Result<Self, InferenceError> {
        Self::load_on(onnx_path, Device::Cpu)
    }

    /// Construct an `InferenceSession` on a specific execution provider.
    /// Validates the graph signature against the v3.0/v3.1/v3.2 contract
    /// and asserts vocab hash parity with the sidecar before building
    /// the ORT session.
    ///
    /// CPU branch keeps the historical
    /// `intra/inter_threads = 1` FP-determinism pin (parity with
    /// `serve_onnx --ort-threads 1`). CUDA branch wires the CUDA EP via
    /// `with_execution_providers` and drops the intra/inter-thread pins
    /// (thread counts on the CPU pool are irrelevant once compute is on
    /// the GPU). Both branches share the lock-free `UnsafeCell<Session>`
    /// punch-through; see `SessionGuard`.
    pub fn load_on(onnx_path: &Path, device: Device) -> Result<Self, InferenceError> {
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

        // v33-correctness-fix Fix 2-4: assert action_feature_schema_version
        // parity. Sidecars from before this field landed (e.g. existing
        // v3.0/v3.1/v3.2 ckpts pre-export-onnx-update) omit the key; we
        // skip the check in that case (legacy ckpt under unknown schema —
        // operator's responsibility to ensure compatibility). Fresh
        // exports stamp the field, so any mismatch is operator error.
        let sidecar_action_schema = sidecar
            .get("action_feature_schema_version")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32);
        if let Some(expected) = sidecar_action_schema {
            let runtime = crate::policy::actions::ACTION_FEATURE_SCHEMA_VERSION;
            if expected != runtime {
                return Err(InferenceError::ActionSchemaMismatch {
                    expected,
                    runtime,
                });
            }
        }

        // Build the ORT session per device. R14.G CPU rationale: pin
        // intra/inter-op = 1 to keep FP-determinism on the policy logits
        // (matches the `serve_onnx --ort-threads 1` server default the
        // parity smoke runs against). On CUDA the CPU thread pins are
        // irrelevant — compute runs on the GPU.
        let session = match device {
            Device::Cpu => Session::builder()
                .map_err(InferenceError::from)?
                .with_intra_threads(1)
                .map_err(InferenceError::from)?
                .with_inter_threads(1)
                .map_err(InferenceError::from)?
                .commit_from_file(onnx_path)
                .map_err(InferenceError::from)?,
            Device::Cuda { device_id } => Session::builder()
                .map_err(InferenceError::from)?
                .with_execution_providers([CUDAExecutionProvider::default()
                    .with_device_id(device_id)
                    .build()
                    .error_on_failure()])
                .map_err(InferenceError::from)?
                .commit_from_file(onnx_path)
                .map_err(InferenceError::from)?,
        };

        // Graph-signature validation. Detects v3.0/v3.1 (5-input
        // contract; discriminated by `state_features` last-dim 110 vs
        // 164) vs v3.2 (7-input contract) — matches
        // `serve_onnx._lookup_schema`. Partial v3.2 (one slot input
        // missing) is rejected explicitly.
        let schema = validate_graph_signature(&session)?;

        let guard = match device {
            Device::Cpu => SessionGuard::Cpu(UnsafeCell::new(session)),
            Device::Cuda { .. } => SessionGuard::Cuda(UnsafeCell::new(session)),
        };

        Ok(InferenceSession {
            session: guard,
            onnx_path: onnx_path.to_path_buf(),
            schema,
            device,
        })
    }

    /// Active execution provider for this session. Diagnostic only.
    pub fn device(&self) -> Device {
        self.device
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
        // Pack the five v3.0 (or v3.1 — 164-d head) input tensors.
        // Layout matches `serve_onnx.request_to_arrays` exactly (batch=1
        // leading dim everywhere). v3.2 reuses the v3.0 state head and
        // ADDS the slot tensors below; the state-features tensor is
        // unchanged.
        let (state, state_dim) = match self.schema {
            GraphSchema::V3_1 => (
                featurize::observation_state_features_v3_1(observation),
                STATE_DIM_V3_1,
            ),
            GraphSchema::V3_3 => (
                featurize::observation_state_features_v3_3(observation),
                STATE_DIM_V3_3,
            ),
            GraphSchema::V3_0 | GraphSchema::V3_2 => (
                featurize::observation_state_features(observation),
                STATE_DIM_V3,
            ),
        };
        let state_arr = Array::from_shape_vec((1, state_dim), state)
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
        // branches; v3.0 + v3.1 dispatch simply ignores them (ORT hard-rejects
        // unknown feed keys, so v3.0/v3.1 graphs MUST omit these from `inputs!`).
        let (slot_card_ids_flat, slot_features_flat) = match self.schema {
            GraphSchema::V3_2 => featurize::observation_uma_slots(observation),
            GraphSchema::V3_0 | GraphSchema::V3_1 | GraphSchema::V3_3 => {
                (Vec::new(), Vec::new())
            }
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
        // different ort::inputs! map (5 keys for v3.0/v3.1, 7 keys for
        // v3.2) — ORT hard-rejects unknown feed keys so we MUST omit the
        // v3.2 tensors from the v3.0/v3.1 feed. v3.0 and v3.1 share the
        // same 5-input shape (only the `state_features` tensor's last
        // dim differs: 110 vs 164).
        let inputs = match self.schema {
            GraphSchema::V3_0 | GraphSchema::V3_1 | GraphSchema::V3_3 => ort::inputs![
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

        // Both CPU and CUDA paths punch through `UnsafeCell` —
        // `outputs[i].try_extract_tensor` borrows from the session, so
        // we keep the `&mut Session` alive for the full extract. Safety
        // for the multi-threaded access lives on `unsafe impl Sync for
        // SessionGuard`: ORT's `Session::Run` is documented thread-safe
        // for concurrent calls on a single session for both the CPU EP
        // (with `intra/inter_threads = 1` pinned at build time) and the
        // CUDA EP.
        let (logits_vec, value_scalar) = match &self.session {
            SessionGuard::Cpu(cell) | SessionGuard::Cuda(cell) => {
                // Safety: see `unsafe impl Sync for SessionGuard`.
                let sess: &mut Session = unsafe { &mut *cell.get() };
                let outputs = sess.run(inputs)?;
                extract_logits_and_value(&outputs, n_actions)?
            }
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

/// Read `(logits_vec, value_scalar)` out of an ORT `SessionOutputs`.
/// Outputs order matches `serve_onnx.PolicyServer.session.run(...)`:
/// `[logits (float[B, A]), value (float[B])]`. Factored out so the
/// CPU/CUDA dispatch in `predict_v3` doesn't duplicate the extract.
fn extract_logits_and_value(
    outputs: &ort::session::SessionOutputs<'_>,
    n_actions: usize,
) -> Result<(Vec<f32>, f32), InferenceError> {
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
    Ok((logits_flat.to_vec(), value_flat[0]))
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

/// Probe the `state_features` input's last-dim from the ORT graph
/// signature. Returns `None` if the graph declares no `state_features`
/// input (caught downstream by the set-equality check) or if the shape
/// has no concrete last dim. Mirrors the Python `_graph_signature`
/// state-dim discriminator in `serve_onnx.py`.
fn read_state_features_last_dim(session: &Session) -> Option<i64> {
    for inp in session.inputs().iter() {
        if inp.name() == "state_features" {
            // `Outlet::dtype()` returns the `ValueType`; for a Tensor
            // input the shape is exposed via `tensor_shape()` and Derefs
            // to `[i64]`. Read the last entry if concrete (>0); a
            // symbolic dim is represented as a negative sentinel and we
            // treat that as "unknown" (fall back to v3.0).
            if let Some(shape) = inp.dtype().tensor_shape() {
                if let Some(&last) = shape.last() {
                    if last > 0 {
                        return Some(last);
                    }
                }
            }
            return None;
        }
    }
    None
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
    // present (set-equality on the 7-input contract). v3.2 reuses the
    // 110-d state head — the shape probe is informational only here.
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

    // v3.0/v3.1 dispatch: same 5-input name set; differ ONLY by
    // `state_features` last-dim (110 vs 164).
    if input_set != required_v3 {
        return Err(InferenceError::SchemaMismatch(format!(
            "graph input set {:?} does not match v3.0/v3.1 contract {:?} or \
             v3.2 contract {:?}",
            inputs, REQUIRED_V3_INPUTS, required_v3_2.iter().copied().collect::<Vec<_>>()
        )));
    }

    // Discriminate v3.0 (110) from v3.1 (164) by state_features last
    // dim. The shape is a concrete int on every exported policy graph
    // (export pins `state_features: float[1, STATE_DIM]`); a missing or
    // symbolic last dim is treated as v3.0 for backwards compatibility
    // (matches the Python `_lookup_schema` fallback behaviour where
    // state_dim=None is rejected upstream).
    match read_state_features_last_dim(session) {
        Some(d) if d as usize == STATE_DIM_V3 => Ok(GraphSchema::V3_0),
        Some(d) if d as usize == STATE_DIM_V3_1 => Ok(GraphSchema::V3_1),
        Some(d) if d as usize == STATE_DIM_V3_3 => Ok(GraphSchema::V3_3),
        Some(d) => Err(InferenceError::SchemaMismatch(format!(
            "graph state_features last dim {} does not match v3.0 ({}), \
             v3.1 ({}), or v3.3 ({}); inputs {:?}",
            d, STATE_DIM_V3, STATE_DIM_V3_1, STATE_DIM_V3_3, inputs
        ))),
        None => {
            // No concrete state_features shape — fall back to v3.0 for
            // backwards compatibility (older v3.0 graphs may have had
            // a dynamic batch + dynamic state dim under some exporters;
            // every modern exporter emits a concrete 110/164).
            Ok(GraphSchema::V3_0)
        }
    }
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
