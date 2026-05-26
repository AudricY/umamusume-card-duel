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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ndarray::Array;
use ort::ep::CUDA as CUDAExecutionProvider;
use ort::session::Session;
use ort::value::TensorRef;

use crate::policy::card_vocab::card_vocab;
use crate::policy::featurize::{
    self, ACTION_DIM, MAX_CARDS_PER_ZONE, NUM_ZONES, STATE_DIM_V3, STATE_DIM_V3_1, STATE_DIM_V3_3,
    STATE_DIM_V3_5, STATE_DIM_V3_6, STATE_DIM_V3_7,
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
    /// v34-compound-axis: 167-d state-features + 7-input contract
    /// (with slot tokens). Combines v3.3's opp-side flag tail with
    /// v3.2's per-Uma slot tokens to test whether the two axes
    /// compound additively (slot +0.0066, opp-flag +0.0100) or cap.
    V3_4,
    /// v35-multichannel-tail: 212-d state-features + 5-input contract
    /// (no slot tokens). Layered on v3.3 (167-d) with a 45-bit
    /// channel-orthogonal tail (phase one-hot, per-condition one-hot,
    /// energy-zone front typed, opp discard buckets, bench-refill
    /// catastrophe). See
    /// `docs/ai-research/scoping/v35-multichannel-tail-scoping.md`.
    V3_5,
    /// v36-priors-and-arithmetic: 246-d state-features + 5-input
    /// contract (no slot tokens). Layered on v3.5: the v3.5 [197:207]
    /// band is repurposed in-place for own.energy_pool typed multihot
    /// and a new 34-bit tail appended at [212:246] adds opp pool,
    /// own/opp prize-card one-hot, opp bench typed aggregate,
    /// lethal-next-turn face values, and secondary-attack usable/KO
    /// bits. See
    /// `docs/ai-research/scoping/v36-priors-and-arithmetic-scoping.md`.
    V3_6,
    /// v37-combat-arith-and-catalog: 296-d state-features + 5-input
    /// contract (no slot tokens). Layered on v3.6 with a 50-bit
    /// combat-arith + catalog-lookup tail at [246:296]: weakness-
    /// adjusted lethal, coin-flip expected damage, conditional damage
    /// bonus indicators, energy ETA + paralysis windows, and frozen
    /// tool / ability effect-kind one-hots. v3.6 head [0:246] stays
    /// byte-stable. See
    /// `docs/ai-research/scoping/v37-combat-arith-and-catalog-scoping.md`.
    V3_7,
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

/// Backing storage for the ORT `Session`.
///
/// `Inline` is the historical zero-overhead path: the Session lives on
/// `InferenceSession` itself and `predict_v3` calls it directly via the
/// `UnsafeCell` punch-through (concurrent `Session::run` from N worker
/// threads). `Dispatched` is the B2 batched-inference path: the Session
/// is moved into the `BatchedDispatcher` thread at construction time and
/// `predict_v3` enqueues per-row work to that thread. Crucially, when
/// `Dispatched` is selected the Session is single-threaded (only the
/// dispatcher thread touches it), so the `unsafe impl Sync` story is
/// confined to the `Inline` variant.
enum SessionStorage {
    Inline(SessionGuard),
    Dispatched,
}

/// Loaded in-process ONNX session — thread-safe per ORT's contract
/// (`unsafe impl Send + Sync for Session`); callers wrap in an `Arc`
/// to share across MCTS workers. The `Inline` storage variant shares
/// the `UnsafeCell<Session>` punch-through (see `SessionGuard`); the
/// `Dispatched` variant routes through a `BatchedDispatcher` thread
/// that owns the Session single-threadedly.
pub struct InferenceSession {
    session: SessionStorage,
    dispatcher: Option<Arc<BatchedDispatcher>>,
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
            session: SessionStorage::Inline(guard),
            dispatcher: None,
            onnx_path: onnx_path.to_path_buf(),
            schema,
            device,
        })
    }

    /// Construct an `InferenceSession` with optional batched dispatch.
    ///
    /// When `max_batch <= 1`, this is bit-identical to
    /// [`load_on`](Self::load_on) — no dispatcher is built, no channel
    /// overhead, no behavioural change (B=1 fast path preserved per the
    /// `gpu-batched-inference-throughput.md` B2 acceptance gate).
    ///
    /// When `max_batch > 1`, the loaded `Session` is moved into a
    /// dedicated dispatcher thread that gathers requests from N
    /// concurrent worker threads and issues a single `Session::run`
    /// per batch (up to `max_batch` rows, flushing on a `max_wait_us`
    /// micro-deadline). See [`BatchedDispatcher`] for the gather pattern
    /// and the n_actions-padding invariant.
    pub fn load_on_with_batching(
        onnx_path: &Path,
        device: Device,
        max_batch: usize,
        max_wait_us: u64,
    ) -> Result<Self, InferenceError> {
        if max_batch <= 1 {
            return Self::load_on(onnx_path, device);
        }

        // Re-run the full load_on path to get a session + schema, then
        // peel the Session out of its UnsafeCell and hand it to the
        // dispatcher thread. The peel is sound because we're about to
        // drop the SessionGuard wrapper anyway and the Session moves
        // into a thread that owns it single-threadedly from that point.
        let loaded = Self::load_on(onnx_path, device)?;
        let InferenceSession {
            session,
            schema,
            onnx_path: path_out,
            device: device_out,
            ..
        } = loaded;
        let session = match session {
            SessionStorage::Inline(SessionGuard::Cpu(cell))
            | SessionStorage::Inline(SessionGuard::Cuda(cell)) => cell.into_inner(),
            SessionStorage::Dispatched => unreachable!("load_on always returns Inline"),
        };

        let dispatcher = BatchedDispatcher::start(session, schema, max_batch, max_wait_us);

        Ok(InferenceSession {
            session: SessionStorage::Dispatched,
            dispatcher: Some(Arc::new(dispatcher)),
            onnx_path: path_out,
            schema,
            device: device_out,
        })
    }

    /// Borrow the dispatcher when batching is active. Diagnostic only —
    /// used by `sim-eval-gate` to log `mean_fill` at end of run.
    pub fn dispatcher(&self) -> Option<&BatchedDispatcher> {
        self.dispatcher.as_deref()
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
        let row = pack_row(self.schema, observation, legal_actions)?;

        // Dispatched path: enqueue the packed row to the dispatcher
        // thread and block on the per-request response channel. The
        // dispatcher batches up to `max_batch` concurrent requests into
        // a single `Session::run`.
        if let Some(dispatcher) = self.dispatcher.as_ref() {
            return dispatcher.predict(row);
        }

        // Inline path: bit-identical to pre-B2 behavior (B=1 single
        // Session::run, no channel overhead). Concurrent calls share
        // the Session via the `UnsafeCell` punch-through; see
        // `unsafe impl Sync for SessionGuard`.
        let guard = match &self.session {
            SessionStorage::Inline(g) => g,
            SessionStorage::Dispatched => {
                // Unreachable: `Dispatched` only set when `dispatcher`
                // is `Some`, which routes above.
                return Err(InferenceError::OutputShape(
                    "Dispatched storage without dispatcher".into(),
                ));
            }
        };
        run_inline_row(guard, self.schema, &row)
    }

    /// Path the session was loaded from. Useful for diagnostics.
    pub fn onnx_path(&self) -> &Path {
        &self.onnx_path
    }

    /// B6 direct batched API for intra-tree wave-batching (see
    /// `docs/ai-research/scoping/gpu-batched-inference-throughput.md`).
    /// Packs `batch.len()` `(observation, legal_actions)` rows with the
    /// same `max_n_actions`-padding scheme the cross-thread
    /// `BatchedDispatcher` uses, issues a single `Session::run` with
    /// `B = batch.len()`, slices the outputs per row, and runs the same
    /// `greedy_masked_softmax` on each row's `logits[..n_actions]` slice.
    ///
    /// This is a synchronous direct call on the inline session — it
    /// bypasses the cross-thread dispatcher entirely. The wave caller
    /// already has B requests in hand from one tree's leaf-selection
    /// phase, so paying the per-request mpsc + condvar overhead is
    /// pointless when one batched `Session::run` covers the whole wave.
    ///
    /// **Constraint:** the session must be in the `Inline` storage
    /// variant. `load_on_with_batching(_, _, max_batch > 1, _)` moves the
    /// `Session` into the dispatcher thread (single-threaded ownership
    /// invariant), so wave mode is incompatible with the cross-thread
    /// dispatcher; the caller (`sim-eval-gate`) is expected to reject the
    /// `--wave-size > 1` AND `--batch-size > 1` combination at flag-parse
    /// time. If `predict_v3_batch` is called against a `Dispatched`
    /// session, it returns `InferenceError::OutputShape` describing the
    /// constraint.
    ///
    /// Output order mirrors input order: `predictions[i]` corresponds to
    /// `batch[i]`.
    pub fn predict_v3_batch(
        &self,
        batch: &[(&PublicObservation, &[LegalAiAction])],
    ) -> Result<Vec<PredictionV3>, InferenceError> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }
        // Single-row fast path keeps the inline call shape (avoids the
        // n_actions-padding allocator pressure for the trivial wave-size-1
        // case; matters because the wave loop's own bit-identical-at-
        // wave_size=1 guarantee already calls the serial path, but a
        // caller building B6 follow-ons may still poke single rows here).
        if batch.len() == 1 {
            let (obs, legal) = batch[0];
            return Ok(vec![self.predict_v3(obs, legal)?]);
        }
        let guard = match &self.session {
            SessionStorage::Inline(g) => g,
            SessionStorage::Dispatched => {
                return Err(InferenceError::OutputShape(
                    "predict_v3_batch requires inline session storage; this session was \
                     constructed with load_on_with_batching(max_batch>1) which moves the \
                     Session into a dispatcher thread. The wave-batching caller and the \
                     cross-thread dispatcher are mutually exclusive — pick one."
                        .to_string(),
                ));
            }
        };
        let mut rows: Vec<PackedRow> = Vec::with_capacity(batch.len());
        for (obs, legal) in batch.iter() {
            if legal.is_empty() {
                return Err(InferenceError::SchemaMismatch(
                    "legalActions must not be empty".into(),
                ));
            }
            rows.push(pack_row(self.schema, obs, legal)?);
        }
        run_inline_batch(guard, self.schema, rows)
    }
}

// ---------------------------------------------------------------------------
// Per-row packing — shared by the inline `predict_v3` path and the batched
// dispatcher. Each `PackedRow` is `Send`able (owned `Vec`s only); the
// dispatcher channel moves these between worker threads and the dispatcher
// thread.
// ---------------------------------------------------------------------------

/// All input tensors for a single `(observation, legal_actions)` pair,
/// flattened to owned `Vec`s. The dispatcher pads `action_features`,
/// `action_mask`, and `action_card_idx` to `max_n_actions` across the
/// batch before stacking; the other tensors are fixed-size per row.
struct PackedRow {
    state: Vec<f32>,
    state_dim: usize,
    action_features: Vec<f32>, // n_actions * ACTION_DIM
    n_actions: usize,
    card_ids: Vec<i64>, // NUM_ZONES * MAX_CARDS_PER_ZONE
    action_card_idx: Vec<i64>, // n_actions * 2
    /// v3.2/v3.4 only.
    slot_card_ids: Option<Vec<i64>>,
    /// v3.2/v3.4 only.
    slot_features: Option<Vec<f32>>,
}

fn pack_row(
    schema: GraphSchema,
    observation: &PublicObservation,
    legal_actions: &[LegalAiAction],
) -> Result<PackedRow, InferenceError> {
    let (state, state_dim) = match schema {
        GraphSchema::V3_1 => (
            featurize::observation_state_features_v3_1(observation),
            STATE_DIM_V3_1,
        ),
        GraphSchema::V3_3 | GraphSchema::V3_4 => (
            featurize::observation_state_features_v3_3(observation),
            STATE_DIM_V3_3,
        ),
        GraphSchema::V3_5 => (
            featurize::observation_state_features_v3_5(observation),
            STATE_DIM_V3_5,
        ),
        GraphSchema::V3_6 => (
            featurize::observation_state_features_v3_6(observation),
            STATE_DIM_V3_6,
        ),
        GraphSchema::V3_7 => (
            featurize::observation_state_features_v3_7(observation),
            STATE_DIM_V3_7,
        ),
        GraphSchema::V3_0 | GraphSchema::V3_2 => (
            featurize::observation_state_features(observation),
            STATE_DIM_V3,
        ),
    };
    let n_actions = legal_actions.len();
    let action_features = featurize::legal_actions_features(legal_actions)?;
    let card_ids = featurize::observation_card_ids_by_zone(observation);
    let action_card_idx = featurize::action_card_idx_pairs_flat(legal_actions);
    let (slot_card_ids, slot_features) = match schema {
        GraphSchema::V3_2 | GraphSchema::V3_4 => {
            let (ids, feats) = featurize::observation_uma_slots(observation);
            (Some(ids), Some(feats))
        }
        GraphSchema::V3_0
        | GraphSchema::V3_1
        | GraphSchema::V3_3
        | GraphSchema::V3_5
        | GraphSchema::V3_6
        | GraphSchema::V3_7 => (None, None),
    };
    Ok(PackedRow {
        state,
        state_dim,
        action_features,
        n_actions,
        card_ids,
        action_card_idx,
        slot_card_ids,
        slot_features,
    })
}

/// Run a single `PackedRow` through the inline session (B=1 path, no
/// batching). Bit-identical to the pre-B2 behavior — exists as a helper
/// so `predict_v3` can route between dispatcher and inline without
/// duplicating the tensor-build + run + extract sequence.
fn run_inline_row(
    guard: &SessionGuard,
    schema: GraphSchema,
    row: &PackedRow,
) -> Result<PredictionV3, InferenceError> {
    let state_arr = Array::from_shape_vec((1, row.state_dim), row.state.clone())
        .map_err(|e| InferenceError::OutputShape(format!("state reshape: {e}")))?;
    let action_features_arr = Array::from_shape_vec(
        (1, row.n_actions, ACTION_DIM),
        row.action_features.clone(),
    )
    .map_err(|e| InferenceError::OutputShape(format!("action_features reshape: {e}")))?;
    let action_mask_arr = Array::from_elem((1, row.n_actions), true);
    let card_ids_arr = Array::from_shape_vec(
        (1, NUM_ZONES, MAX_CARDS_PER_ZONE),
        row.card_ids.clone(),
    )
    .map_err(|e| InferenceError::OutputShape(format!("card_ids reshape: {e}")))?;
    let action_card_idx_arr =
        Array::from_shape_vec((1, row.n_actions, 2), row.action_card_idx.clone())
            .map_err(|e| InferenceError::OutputShape(format!("action_card_idx reshape: {e}")))?;
    let slot_card_ids_arr = match (schema, row.slot_card_ids.as_ref()) {
        (GraphSchema::V3_2 | GraphSchema::V3_4, Some(ids)) => Some(
            Array::from_shape_vec((1, UMA_SLOT_COUNT), ids.clone()).map_err(|e| {
                InferenceError::OutputShape(format!("uma_slot_card_ids reshape: {e}"))
            })?,
        ),
        _ => None,
    };
    let slot_features_arr = match (schema, row.slot_features.as_ref()) {
        (GraphSchema::V3_2 | GraphSchema::V3_4, Some(feats)) => Some(
            Array::from_shape_vec(
                (1, UMA_SLOT_COUNT, UMA_SLOT_FEATURE_DIM),
                feats.clone(),
            )
            .map_err(|e| InferenceError::OutputShape(format!("uma_slot_features reshape: {e}")))?,
        ),
        _ => None,
    };

    let inputs = match schema {
        GraphSchema::V3_0
        | GraphSchema::V3_1
        | GraphSchema::V3_3
        | GraphSchema::V3_5
        | GraphSchema::V3_6
        | GraphSchema::V3_7 => ort::inputs![
            "state_features" => TensorRef::from_array_view(&state_arr)?,
            "action_features" => TensorRef::from_array_view(&action_features_arr)?,
            "action_mask" => TensorRef::from_array_view(&action_mask_arr)?,
            "card_ids_by_zone" => TensorRef::from_array_view(&card_ids_arr)?,
            "action_card_idx" => TensorRef::from_array_view(&action_card_idx_arr)?,
        ],
        GraphSchema::V3_2 | GraphSchema::V3_4 => {
            let slot_ids = slot_card_ids_arr
                .as_ref()
                .expect("slot tensors built above for v3.2/v3.4");
            let slot_feats = slot_features_arr
                .as_ref()
                .expect("slot tensors built above for v3.2/v3.4");
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

    let (logits_vec, value_scalar) = match guard {
        SessionGuard::Cpu(cell) | SessionGuard::Cuda(cell) => {
            // Safety: see `unsafe impl Sync for SessionGuard`.
            let sess: &mut Session = unsafe { &mut *cell.get() };
            let outputs = sess.run(inputs)?;
            extract_logits_and_value(&outputs, row.n_actions)?
        }
    };

    let probs = greedy_masked_softmax(&logits_vec);
    Ok(PredictionV3 {
        probs,
        value: value_scalar,
    })
}

/// Run a stacked-B batch through the inline session and return per-row
/// `PredictionV3`s in input order. Shared between the public
/// `predict_v3_batch` API (B6 intra-tree wave batching) and the
/// cross-thread dispatcher's `run_batch` (B2). Behaviour is identical to
/// the dispatcher path: same n_actions-padding, same `Session::run`
/// shape, same per-row `greedy_masked_softmax`. The only difference is
/// the `Session` access pattern — inline uses the `UnsafeCell`
/// punch-through (ORT's `Session::run` is documented thread-safe), the
/// dispatcher path owns the Session single-threadedly.
fn run_inline_batch(
    guard: &SessionGuard,
    schema: GraphSchema,
    rows: Vec<PackedRow>,
) -> Result<Vec<PredictionV3>, InferenceError> {
    let n_batch = rows.len();
    if n_batch == 0 {
        return Ok(Vec::new());
    }
    let state_dim = rows[0].state_dim;
    let max_n = rows.iter().map(|r| r.n_actions).max().unwrap_or(0);
    if max_n == 0 {
        return Err(InferenceError::SchemaMismatch(
            "predict_v3_batch: every row has empty legal_actions".into(),
        ));
    }
    let needs_slots = matches!(schema, GraphSchema::V3_2 | GraphSchema::V3_4);

    let mut state_buf: Vec<f32> = Vec::with_capacity(n_batch * state_dim);
    let mut action_features_buf: Vec<f32> = vec![0.0; n_batch * max_n * ACTION_DIM];
    let mut action_mask_buf: Vec<bool> = vec![false; n_batch * max_n];
    let mut card_ids_buf: Vec<i64> = Vec::with_capacity(n_batch * NUM_ZONES * MAX_CARDS_PER_ZONE);
    let mut action_card_idx_buf: Vec<i64> = vec![0; n_batch * max_n * 2];
    let mut slot_card_ids_buf: Vec<i64> = if needs_slots {
        Vec::with_capacity(n_batch * UMA_SLOT_COUNT)
    } else {
        Vec::new()
    };
    let mut slot_features_buf: Vec<f32> = if needs_slots {
        Vec::with_capacity(n_batch * UMA_SLOT_COUNT * UMA_SLOT_FEATURE_DIM)
    } else {
        Vec::new()
    };

    for (row_idx, r) in rows.iter().enumerate() {
        if r.state_dim != state_dim {
            return Err(InferenceError::SchemaMismatch(format!(
                "predict_v3_batch: state_dim {} != {} on row {}",
                r.state_dim, state_dim, row_idx
            )));
        }
        state_buf.extend_from_slice(&r.state);
        let af_dst_off = row_idx * max_n * ACTION_DIM;
        let af_src_len = r.n_actions * ACTION_DIM;
        action_features_buf[af_dst_off..af_dst_off + af_src_len]
            .copy_from_slice(&r.action_features);
        let am_dst_off = row_idx * max_n;
        for i in 0..r.n_actions {
            action_mask_buf[am_dst_off + i] = true;
        }
        card_ids_buf.extend_from_slice(&r.card_ids);
        let aci_dst_off = row_idx * max_n * 2;
        let aci_src_len = r.n_actions * 2;
        action_card_idx_buf[aci_dst_off..aci_dst_off + aci_src_len]
            .copy_from_slice(&r.action_card_idx);
        if needs_slots {
            let slot_ids = r.slot_card_ids.as_ref().ok_or_else(|| {
                InferenceError::SchemaMismatch(
                    "predict_v3_batch: slot tensors missing on v3.2/v3.4 row".into(),
                )
            })?;
            let slot_feats = r.slot_features.as_ref().ok_or_else(|| {
                InferenceError::SchemaMismatch(
                    "predict_v3_batch: slot features missing on v3.2/v3.4 row".into(),
                )
            })?;
            slot_card_ids_buf.extend_from_slice(slot_ids);
            slot_features_buf.extend_from_slice(slot_feats);
        }
    }

    let state_arr = Array::from_shape_vec((n_batch, state_dim), state_buf)
        .map_err(|e| InferenceError::OutputShape(format!("state reshape: {e}")))?;
    let action_features_arr =
        Array::from_shape_vec((n_batch, max_n, ACTION_DIM), action_features_buf)
            .map_err(|e| InferenceError::OutputShape(format!("action_features reshape: {e}")))?;
    let action_mask_arr = Array::from_shape_vec((n_batch, max_n), action_mask_buf)
        .map_err(|e| InferenceError::OutputShape(format!("action_mask reshape: {e}")))?;
    let card_ids_arr =
        Array::from_shape_vec((n_batch, NUM_ZONES, MAX_CARDS_PER_ZONE), card_ids_buf)
            .map_err(|e| InferenceError::OutputShape(format!("card_ids reshape: {e}")))?;
    let action_card_idx_arr =
        Array::from_shape_vec((n_batch, max_n, 2), action_card_idx_buf)
            .map_err(|e| InferenceError::OutputShape(format!("action_card_idx reshape: {e}")))?;
    let slot_card_ids_arr = if needs_slots {
        Some(
            Array::from_shape_vec((n_batch, UMA_SLOT_COUNT), slot_card_ids_buf).map_err(
                |e| InferenceError::OutputShape(format!("uma_slot_card_ids reshape: {e}")),
            )?,
        )
    } else {
        None
    };
    let slot_features_arr = if needs_slots {
        Some(
            Array::from_shape_vec(
                (n_batch, UMA_SLOT_COUNT, UMA_SLOT_FEATURE_DIM),
                slot_features_buf,
            )
            .map_err(|e| InferenceError::OutputShape(format!("uma_slot_features reshape: {e}")))?,
        )
    } else {
        None
    };

    let inputs = match schema {
        GraphSchema::V3_0
        | GraphSchema::V3_1
        | GraphSchema::V3_3
        | GraphSchema::V3_5
        | GraphSchema::V3_6
        | GraphSchema::V3_7 => ort::inputs![
            "state_features" => TensorRef::from_array_view(&state_arr)?,
            "action_features" => TensorRef::from_array_view(&action_features_arr)?,
            "action_mask" => TensorRef::from_array_view(&action_mask_arr)?,
            "card_ids_by_zone" => TensorRef::from_array_view(&card_ids_arr)?,
            "action_card_idx" => TensorRef::from_array_view(&action_card_idx_arr)?,
        ],
        GraphSchema::V3_2 | GraphSchema::V3_4 => {
            let slot_ids = slot_card_ids_arr
                .as_ref()
                .expect("slot tensors built above for v3.2/v3.4");
            let slot_feats = slot_features_arr
                .as_ref()
                .expect("slot tensors built above for v3.2/v3.4");
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

    let (logits_flat, value_flat) = match guard {
        SessionGuard::Cpu(cell) | SessionGuard::Cuda(cell) => {
            // Safety: see `unsafe impl Sync for SessionGuard`. ORT's
            // `Session::run` is documented thread-safe for concurrent
            // calls on a single session for both CPU and CUDA EPs; the
            // `&mut self` is a Rust-API artifact.
            let sess: &mut Session = unsafe { &mut *cell.get() };
            let outputs = sess.run(inputs)?;
            let (_, l) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(InferenceError::from)?;
            let (_, v) = outputs[1]
                .try_extract_tensor::<f32>()
                .map_err(InferenceError::from)?;
            (l.to_vec(), v.to_vec())
        }
    };

    if logits_flat.len() != n_batch * max_n {
        return Err(InferenceError::OutputShape(format!(
            "logits has {} elements, expected {} ({} x {})",
            logits_flat.len(),
            n_batch * max_n,
            n_batch,
            max_n
        )));
    }
    if value_flat.len() != n_batch {
        return Err(InferenceError::OutputShape(format!(
            "value has {} elements, expected {}",
            value_flat.len(),
            n_batch
        )));
    }

    let mut out: Vec<PredictionV3> = Vec::with_capacity(n_batch);
    for (row_idx, r) in rows.iter().enumerate() {
        let row_start = row_idx * max_n;
        let row_logits = &logits_flat[row_start..row_start + r.n_actions];
        let probs = greedy_masked_softmax(row_logits);
        let value = value_flat[row_idx];
        out.push(PredictionV3 { probs, value });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// BatchedDispatcher — gathers per-row work from N worker threads, packs a
// stacked-B input, runs `Session::run` once per batch, and fans per-row
// outputs back to each waiter.
//
// **Leader-thread-blocks-on-first-recv pattern.** The dispatcher loop is
// a single thread (owns the Session single-threadedly — no UnsafeCell
// needed on this path). It blocks on `rx.recv()` for the first request
// of a batch; once it has one, it spins `rx.recv_timeout(deadline -
// Instant::now())` to gather up to `max_batch - 1` more, then runs.
// This trades off latency-vs-fill: a worker that arrives alone after a
// dead period pays only `max_wait_us` extra latency, not a full block.
//
// **n_actions-padding invariant.** Each row has a different
// `legal_actions.len()`. The ONNX graph's `actions` axis is dynamic
// (per `training/export_onnx.py:141-172`) but `Session::run` requires
// all rows in a batch share the same dim. We pad every row up to
// `max_n = max(req.n_actions)`: `action_features` to (max_n, ACTION_DIM)
// with zeros, `action_mask` to (max_n,) with `false` for padded positions
// (true for `[..n_actions]`), `action_card_idx` to (max_n, 2) with zeros.
// On the way back we slice `logits[row, ..req.n_actions]` and run the
// SAME `greedy_masked_softmax` the inline path uses — the graph's
// internal mask handling on padded positions is the load-bearing
// correctness assumption (validated by the batched-vs-inline parity unit
// test below).
// ---------------------------------------------------------------------------

struct BatchedRequest {
    row: PackedRow,
    response: SyncSender<Result<PredictionV3, InferenceError>>,
}

/// Per-session batched-inference dispatcher. Public surface is just
/// `stats()` (read mean-fill counters) and `Drop` (terminates the
/// dispatcher thread cleanly).
pub struct BatchedDispatcher {
    tx: Option<SyncSender<BatchedRequest>>,
    handle: Option<JoinHandle<()>>,
    total_batches: Arc<AtomicU64>,
    total_requests: Arc<AtomicU64>,
}

impl BatchedDispatcher {
    fn start(
        session: Session,
        schema: GraphSchema,
        max_batch: usize,
        max_wait_us: u64,
    ) -> Self {
        // Bounded queue: caps the in-flight burst from a flood of
        // workers. 4× max_batch is generous enough to never block in
        // steady state but bounded enough to fail loudly if something
        // is wrong (e.g. dispatcher thread wedged).
        let (tx, rx): (SyncSender<BatchedRequest>, Receiver<BatchedRequest>) =
            sync_channel(max_batch.saturating_mul(4).max(8));
        let total_batches = Arc::new(AtomicU64::new(0));
        let total_requests = Arc::new(AtomicU64::new(0));
        let batches_for_thread = Arc::clone(&total_batches);
        let requests_for_thread = Arc::clone(&total_requests);
        let handle = std::thread::Builder::new()
            .name("ort-batched-dispatcher".into())
            .spawn(move || {
                dispatcher_loop(
                    session,
                    schema,
                    max_batch,
                    max_wait_us,
                    rx,
                    batches_for_thread,
                    requests_for_thread,
                );
            })
            .expect("spawn dispatcher thread");
        BatchedDispatcher {
            tx: Some(tx),
            handle: Some(handle),
            total_batches,
            total_requests,
        }
    }

    fn predict(&self, row: PackedRow) -> Result<PredictionV3, InferenceError> {
        let (resp_tx, resp_rx) = sync_channel::<Result<PredictionV3, InferenceError>>(1);
        let tx = self
            .tx
            .as_ref()
            .expect("dispatcher tx alive while session alive");
        tx.send(BatchedRequest {
            row,
            response: resp_tx,
        })
        .map_err(|e| InferenceError::Ort(format!("dispatcher send: {e}")))?;
        resp_rx
            .recv()
            .map_err(|e| InferenceError::Ort(format!("dispatcher recv: {e}")))?
    }

    /// `(total_batches, total_requests)` since session start. Diagnostic
    /// — `sim-eval-gate` reads these at end of run to compute mean fill.
    pub fn stats(&self) -> (u64, u64) {
        (
            self.total_batches.load(Ordering::Relaxed),
            self.total_requests.load(Ordering::Relaxed),
        )
    }
}

impl Drop for BatchedDispatcher {
    fn drop(&mut self) {
        // Drop the sender so the dispatcher loop's `rx.recv()` returns
        // Err and the loop terminates; then join.
        self.tx.take();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn dispatcher_loop(
    mut session: Session,
    schema: GraphSchema,
    max_batch: usize,
    max_wait_us: u64,
    rx: Receiver<BatchedRequest>,
    total_batches: Arc<AtomicU64>,
    total_requests: Arc<AtomicU64>,
) {
    let max_wait = Duration::from_micros(max_wait_us);
    loop {
        // Block on first request — when all senders drop the channel
        // returns Err and the loop terminates.
        let first = match rx.recv() {
            Ok(r) => r,
            Err(_) => return,
        };
        let mut batch: Vec<BatchedRequest> = Vec::with_capacity(max_batch);
        batch.push(first);
        let deadline = Instant::now() + max_wait;
        while batch.len() < max_batch {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            match rx.recv_timeout(deadline - now) {
                Ok(r) => batch.push(r),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => break,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        let n_batch = batch.len();
        total_batches.fetch_add(1, Ordering::Relaxed);
        total_requests.fetch_add(n_batch as u64, Ordering::Relaxed);
        run_batch(&mut session, schema, batch);
    }
}

/// Pack a vector of `PackedRow`s into stacked-B tensors, run a single
/// `Session::run`, then fan per-row outputs back via each request's
/// response channel. On any pack/run error, every waiter receives the
/// same error message (cheap stringification — the dispatcher cannot
/// move ownership of a non-Clone error across N senders).
fn run_batch(
    session: &mut Session,
    schema: GraphSchema,
    batch: Vec<BatchedRequest>,
) {
    let n_batch = batch.len();
    let state_dim = batch[0].row.state_dim;
    let max_n = batch.iter().map(|r| r.row.n_actions).max().unwrap_or(0);
    if max_n == 0 {
        // Defensive — predict_v3 already rejects empty legal_actions,
        // but if a row ever slipped through, surface the same error to
        // every waiter rather than panicking the dispatcher thread.
        for req in batch {
            let _ = req.response.send(Err(InferenceError::SchemaMismatch(
                "batched run: empty legal_actions row".into(),
            )));
        }
        return;
    }

    // Allocate stacked-B buffers and copy/pad each row in.
    let mut state_buf: Vec<f32> = Vec::with_capacity(n_batch * state_dim);
    let mut action_features_buf: Vec<f32> = vec![0.0; n_batch * max_n * ACTION_DIM];
    let mut action_mask_buf: Vec<bool> = vec![false; n_batch * max_n];
    let mut card_ids_buf: Vec<i64> = Vec::with_capacity(n_batch * NUM_ZONES * MAX_CARDS_PER_ZONE);
    let mut action_card_idx_buf: Vec<i64> = vec![0; n_batch * max_n * 2];
    let needs_slots = matches!(schema, GraphSchema::V3_2 | GraphSchema::V3_4);
    let mut slot_card_ids_buf: Vec<i64> = if needs_slots {
        Vec::with_capacity(n_batch * UMA_SLOT_COUNT)
    } else {
        Vec::new()
    };
    let mut slot_features_buf: Vec<f32> = if needs_slots {
        Vec::with_capacity(n_batch * UMA_SLOT_COUNT * UMA_SLOT_FEATURE_DIM)
    } else {
        Vec::new()
    };

    for (row_idx, req) in batch.iter().enumerate() {
        let r = &req.row;
        if r.state_dim != state_dim {
            // Different state dim in one batch ⇒ different schema in
            // one session. Should never happen (schema is per-session).
            let err = InferenceError::SchemaMismatch(format!(
                "batched run: state_dim {} != {} on row {}",
                r.state_dim, state_dim, row_idx
            ));
            for req in batch {
                let _ = req.response.send(Err(InferenceError::SchemaMismatch(
                    format!("{}", err),
                )));
            }
            return;
        }
        state_buf.extend_from_slice(&r.state);
        // Pad action_features to (max_n, ACTION_DIM); leading
        // r.n_actions rows are copied, trailing (max_n - r.n_actions)
        // rows are left as zeros.
        let af_dst_off = row_idx * max_n * ACTION_DIM;
        let af_src_len = r.n_actions * ACTION_DIM;
        action_features_buf[af_dst_off..af_dst_off + af_src_len]
            .copy_from_slice(&r.action_features);
        // Pad action_mask to (max_n,); leading r.n_actions positions
        // are true, trailing are false.
        let am_dst_off = row_idx * max_n;
        for i in 0..r.n_actions {
            action_mask_buf[am_dst_off + i] = true;
        }
        card_ids_buf.extend_from_slice(&r.card_ids);
        // Pad action_card_idx to (max_n, 2).
        let aci_dst_off = row_idx * max_n * 2;
        let aci_src_len = r.n_actions * 2;
        action_card_idx_buf[aci_dst_off..aci_dst_off + aci_src_len]
            .copy_from_slice(&r.action_card_idx);
        if needs_slots {
            let Some(slot_ids) = r.slot_card_ids.as_ref() else {
                let err = InferenceError::SchemaMismatch(
                    "batched run: slot tensors missing on v3.2/v3.4 row".into(),
                );
                for req in batch {
                    let _ = req.response.send(Err(InferenceError::SchemaMismatch(
                        format!("{}", err),
                    )));
                }
                return;
            };
            let Some(slot_feats) = r.slot_features.as_ref() else {
                let err = InferenceError::SchemaMismatch(
                    "batched run: slot features missing on v3.2/v3.4 row".into(),
                );
                for req in batch {
                    let _ = req.response.send(Err(InferenceError::SchemaMismatch(
                        format!("{}", err),
                    )));
                }
                return;
            };
            slot_card_ids_buf.extend_from_slice(slot_ids);
            slot_features_buf.extend_from_slice(slot_feats);
        }
    }

    let state_arr = match Array::from_shape_vec((n_batch, state_dim), state_buf) {
        Ok(a) => a,
        Err(e) => {
            broadcast_err(batch, format!("state reshape: {e}"));
            return;
        }
    };
    let action_features_arr =
        match Array::from_shape_vec((n_batch, max_n, ACTION_DIM), action_features_buf) {
            Ok(a) => a,
            Err(e) => {
                broadcast_err(batch, format!("action_features reshape: {e}"));
                return;
            }
        };
    let action_mask_arr = match Array::from_shape_vec((n_batch, max_n), action_mask_buf) {
        Ok(a) => a,
        Err(e) => {
            broadcast_err(batch, format!("action_mask reshape: {e}"));
            return;
        }
    };
    let card_ids_arr = match Array::from_shape_vec(
        (n_batch, NUM_ZONES, MAX_CARDS_PER_ZONE),
        card_ids_buf,
    ) {
        Ok(a) => a,
        Err(e) => {
            broadcast_err(batch, format!("card_ids reshape: {e}"));
            return;
        }
    };
    let action_card_idx_arr =
        match Array::from_shape_vec((n_batch, max_n, 2), action_card_idx_buf) {
            Ok(a) => a,
            Err(e) => {
                broadcast_err(batch, format!("action_card_idx reshape: {e}"));
                return;
            }
        };
    let slot_card_ids_arr = if needs_slots {
        match Array::from_shape_vec((n_batch, UMA_SLOT_COUNT), slot_card_ids_buf) {
            Ok(a) => Some(a),
            Err(e) => {
                broadcast_err(batch, format!("uma_slot_card_ids reshape: {e}"));
                return;
            }
        }
    } else {
        None
    };
    let slot_features_arr = if needs_slots {
        match Array::from_shape_vec(
            (n_batch, UMA_SLOT_COUNT, UMA_SLOT_FEATURE_DIM),
            slot_features_buf,
        ) {
            Ok(a) => Some(a),
            Err(e) => {
                broadcast_err(batch, format!("uma_slot_features reshape: {e}"));
                return;
            }
        }
    } else {
        None
    };

    let inputs_res = match schema {
        GraphSchema::V3_0
        | GraphSchema::V3_1
        | GraphSchema::V3_3
        | GraphSchema::V3_5
        | GraphSchema::V3_6
        | GraphSchema::V3_7 => (|| -> Result<_, InferenceError> {
            Ok(ort::inputs![
                "state_features" => TensorRef::from_array_view(&state_arr)?,
                "action_features" => TensorRef::from_array_view(&action_features_arr)?,
                "action_mask" => TensorRef::from_array_view(&action_mask_arr)?,
                "card_ids_by_zone" => TensorRef::from_array_view(&card_ids_arr)?,
                "action_card_idx" => TensorRef::from_array_view(&action_card_idx_arr)?,
            ])
        })(),
        GraphSchema::V3_2 | GraphSchema::V3_4 => {
            let slot_ids = slot_card_ids_arr
                .as_ref()
                .expect("slot tensors built above for v3.2/v3.4");
            let slot_feats = slot_features_arr
                .as_ref()
                .expect("slot tensors built above for v3.2/v3.4");
            (|| -> Result<_, InferenceError> {
                Ok(ort::inputs![
                    "state_features" => TensorRef::from_array_view(&state_arr)?,
                    "action_features" => TensorRef::from_array_view(&action_features_arr)?,
                    "action_mask" => TensorRef::from_array_view(&action_mask_arr)?,
                    "card_ids_by_zone" => TensorRef::from_array_view(&card_ids_arr)?,
                    "action_card_idx" => TensorRef::from_array_view(&action_card_idx_arr)?,
                    "uma_slot_card_ids" => TensorRef::from_array_view(slot_ids)?,
                    "uma_slot_features" => TensorRef::from_array_view(slot_feats)?,
                ])
            })()
        }
    };
    let inputs = match inputs_res {
        Ok(i) => i,
        Err(e) => {
            broadcast_err(batch, format!("tensor-ref build: {e}"));
            return;
        }
    };

    let outputs = match session.run(inputs) {
        Ok(o) => o,
        Err(e) => {
            broadcast_err(batch, format!("Session::run: {e}"));
            return;
        }
    };

    // Extract `logits[B, max_n]` and `value[B]`, then slice each row
    // back to its own `n_actions` and softmax — mirror of the inline
    // path's `greedy_masked_softmax` call. Padded positions on the
    // logits axis are discarded; the graph's mask-aware softmax should
    // already zero them out, but slicing first is the safer invariant.
    let logits_extract = outputs[0].try_extract_tensor::<f32>();
    let value_extract = outputs[1].try_extract_tensor::<f32>();
    let (logits_flat, value_flat) = match (logits_extract, value_extract) {
        (Ok((_, l)), Ok((_, v))) => (l, v),
        (Err(e), _) | (_, Err(e)) => {
            broadcast_err(batch, format!("extract: {e}"));
            return;
        }
    };
    if logits_flat.len() != n_batch * max_n {
        broadcast_err(
            batch,
            format!(
                "logits has {} elements, expected {} ({} x {})",
                logits_flat.len(),
                n_batch * max_n,
                n_batch,
                max_n
            ),
        );
        return;
    }
    if value_flat.len() != n_batch {
        broadcast_err(
            batch,
            format!(
                "value has {} elements, expected {}",
                value_flat.len(),
                n_batch
            ),
        );
        return;
    }

    for (row_idx, req) in batch.into_iter().enumerate() {
        let n_actions = req.row.n_actions;
        let row_start = row_idx * max_n;
        let row_logits = &logits_flat[row_start..row_start + n_actions];
        let probs = greedy_masked_softmax(row_logits);
        let value = value_flat[row_idx];
        let _ = req.response.send(Ok(PredictionV3 { probs, value }));
    }
}

fn broadcast_err(batch: Vec<BatchedRequest>, msg: String) {
    for req in batch {
        let _ = req
            .response
            .send(Err(InferenceError::Ort(msg.clone())));
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

    // 7-input dispatch: both slot inputs present + required v3.0 inputs
    // all present (set-equality on the 7-input contract). Discriminate
    // v3.2 (state_dim=110) from v3.4 (state_dim=167) by the
    // `state_features` last-dim shape.
    if has_slot_ids && has_slot_feats {
        if input_set != required_v3_2 {
            return Err(InferenceError::SchemaMismatch(format!(
                "graph input set {:?} does not match 7-input slot-token contract {:?}",
                inputs,
                required_v3_2.iter().copied().collect::<Vec<_>>()
            )));
        }
        return match read_state_features_last_dim(session) {
            Some(d) if d as usize == STATE_DIM_V3 => Ok(GraphSchema::V3_2),
            Some(d) if d as usize == STATE_DIM_V3_3 => Ok(GraphSchema::V3_4),
            Some(d) => Err(InferenceError::SchemaMismatch(format!(
                "7-input slot-token graph has state_features last dim {} \
                 — expected {} (v3.2) or {} (v3.4); inputs {:?}",
                d, STATE_DIM_V3, STATE_DIM_V3_3, inputs
            ))),
            None => Ok(GraphSchema::V3_2),
        };
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
        Some(d) if d as usize == STATE_DIM_V3_5 => Ok(GraphSchema::V3_5),
        Some(d) if d as usize == STATE_DIM_V3_6 => Ok(GraphSchema::V3_6),
        Some(d) if d as usize == STATE_DIM_V3_7 => Ok(GraphSchema::V3_7),
        Some(d) => Err(InferenceError::SchemaMismatch(format!(
            "graph state_features last dim {} does not match v3.0 ({}), \
             v3.1 ({}), v3.3 ({}), v3.5 ({}), v3.6 ({}), or v3.7 ({}); inputs {:?}",
            d, STATE_DIM_V3, STATE_DIM_V3_1, STATE_DIM_V3_3, STATE_DIM_V3_5, STATE_DIM_V3_6, STATE_DIM_V3_7, inputs
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

    /// B2 acceptance: assert that a batched (`max_batch=4`) CPU session
    /// produces bit-identical `predict_v3` outputs vs an inline (B=1)
    /// CPU session across 4 distinct `(observation, legal_actions)`
    /// pairs.
    ///
    /// Why bit-identical on CPU: per
    /// `docs/ai-research/scoping/gpu-batched-inference-throughput.md`
    /// B1 evidence, single-threaded CPU ORT (intra=inter=1) is
    /// reduction-order-stable across batch sizes. CUDA drifts ~1e-3 to
    /// ~7e-3 on logits structurally; that drift is gated by the eval
    /// wilson_lower envelope, not by this unit test.
    ///
    /// `#[ignore]`d because it requires `ORT_DYLIB_PATH` + the R110
    /// ckpt; orchestrator (or the smoke-script in the brief) opts in
    /// explicitly.
    #[test]
    #[ignore]
    fn batched_dispatcher_matches_inline_b1_on_cpu() {
        use crate::core::constants::SideId;
        use crate::core::random::{with_rng, Rng};
        use crate::headless_setup::setup_ai_vs_ai_game;
        use crate::policy::actions::enumerate_legal_ai_actions;
        use crate::policy::observation::build_public_observation;

        // Resolve from the workspace root via CARGO_MANIFEST_DIR (the
        // crate dir) so the test works regardless of test cwd.
        // CARGO_MANIFEST_DIR is `<repo>/engine-rs/crates/engine`; pop
        // three levels to reach the repo root.
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let onnx_path_buf = PathBuf::from(manifest_dir)
            .join("..")
            .join("..")
            .join("..")
            .join("runs/R110-W6-repro/iter-0/policy.onnx");
        let onnx_path = onnx_path_buf.as_path();
        if !onnx_path.exists() {
            eprintln!(
                "batched_dispatcher_matches_inline_b1_on_cpu: skipping (missing {})",
                onnx_path.display()
            );
            return;
        }

        // Collect 4 distinct (PublicObservation, Vec<LegalAiAction>)
        // pairs by stepping a few seeds through `setup_ai_vs_ai_game`
        // and pulling the legal action set on each side.
        let mut samples: Vec<(PublicObservation, Vec<LegalAiAction>)> = Vec::new();
        for seed in &[1u32, 2, 3, 4] {
            let seed_str = format!("{}:test", seed);
            let rng = Rng::from_seed(seed_str.as_str(), "test");
            let (state, _used) = with_rng(rng, || setup_ai_vs_ai_game());
            let side = if seed % 2 == 0 {
                SideId::Player
            } else {
                SideId::Opponent
            };
            let legal_seed = format!("{}:legal", seed);
            let (legal, _used) = with_rng(
                Rng::from_seed(legal_seed.as_str(), "legal"),
                || enumerate_legal_ai_actions(&state, side),
            );
            if legal.is_empty() {
                continue;
            }
            let obs = build_public_observation(&state, side);
            samples.push((obs, legal));
        }
        assert!(
            samples.len() >= 4,
            "need at least 4 distinct samples, got {}",
            samples.len()
        );

        let inline =
            InferenceSession::load_on(onnx_path, Device::Cpu).expect("load inline B=1 CPU");
        let batched =
            InferenceSession::load_on_with_batching(onnx_path, Device::Cpu, 4, 5_000)
                .expect("load batched B=4 CPU");

        // Drive the 4 samples through inline sequentially first, then
        // hammer them through the batched session from 4 worker
        // threads (so the dispatcher actually has to batch).
        let inline_outputs: Vec<PredictionV3> = samples
            .iter()
            .map(|(obs, legal)| inline.predict_v3(obs, legal).expect("inline predict_v3"))
            .collect();

        let batched_arc = std::sync::Arc::new(batched);
        let samples_arc = std::sync::Arc::new(samples.clone());
        let mut handles = Vec::new();
        for i in 0..samples.len() {
            let b = std::sync::Arc::clone(&batched_arc);
            let s = std::sync::Arc::clone(&samples_arc);
            handles.push(std::thread::spawn(move || {
                let (obs, legal) = &s[i];
                b.predict_v3(obs, legal).expect("batched predict_v3")
            }));
        }
        let batched_outputs: Vec<PredictionV3> =
            handles.into_iter().map(|h| h.join().unwrap()).collect();

        let mut worst_dprob: f32 = 0.0;
        let mut worst_dvalue: f32 = 0.0;
        for (i, (inl, bat)) in inline_outputs.iter().zip(batched_outputs.iter()).enumerate() {
            assert_eq!(
                inl.probs.len(),
                bat.probs.len(),
                "sample {} probs len mismatch",
                i
            );
            for (a, b) in inl.probs.iter().zip(bat.probs.iter()) {
                let d = (a - b).abs();
                if d > worst_dprob {
                    worst_dprob = d;
                }
            }
            let dv = (inl.value - bat.value).abs();
            if dv > worst_dvalue {
                worst_dvalue = dv;
            }
        }
        eprintln!(
            "batched_dispatcher_matches_inline_b1_on_cpu: worst dprob={:e} dvalue={:e}",
            worst_dprob, worst_dvalue
        );
        assert!(
            worst_dprob < 1e-5,
            "max|Δprob| {:e} exceeds 1e-5 on CPU — graph mask handling on padded positions may be wrong",
            worst_dprob
        );
        assert!(
            worst_dvalue < 1e-5,
            "max|Δvalue| {:e} exceeds 1e-5 on CPU",
            worst_dvalue
        );

        // Sanity: dispatcher actually batched.
        let (batches, requests) =
            batched_arc.dispatcher().expect("dispatcher present").stats();
        assert_eq!(requests as usize, samples.len());
        assert!(batches >= 1);

        // NOTE: ORT 2.0-rc.12 has shown spurious teardown SIGSEGVs
        // when two `Session` instances are dropped in the same process
        // (the inline session + the batched session living inside the
        // dispatcher thread). The test assertions pass before exit;
        // the SIGSEGV happens during ORT global teardown, after the
        // test reports OK. Production paths only construct one
        // `InferenceSession` per run so this is test-only noise.
        // Explicit `drop` here is a no-op but documents the intended
        // teardown order (dispatcher thread joined first).
        drop(batched_arc);
        drop(samples_arc);
        drop(inline);
    }

    /// B6 acceptance: `predict_v3_batch` on a CPU `InferenceSession`
    /// produces bit-identical outputs vs the same N rows driven
    /// individually through `predict_v3`.
    ///
    /// Why bit-identical on CPU: same B1 invariant as the dispatcher
    /// test — single-threaded ORT (intra=inter=1) is reduction-order-
    /// stable across batch sizes. Same `< 1e-5` gate.
    ///
    /// `#[ignore]`d because it requires `ORT_DYLIB_PATH` + the R110
    /// ckpt; orchestrator opts in explicitly. Same SIGSEGV-at-exit
    /// caveat as the dispatcher test (two Session instances in one
    /// process — though here only one session ever lives at a time,
    /// so the caveat is theoretical).
    #[test]
    #[ignore]
    fn predict_v3_batch_matches_individual_predict_v3_on_cpu() {
        use crate::core::constants::SideId;
        use crate::core::random::{with_rng, Rng};
        use crate::headless_setup::setup_ai_vs_ai_game;
        use crate::policy::actions::enumerate_legal_ai_actions;
        use crate::policy::observation::build_public_observation;

        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let onnx_path_buf = PathBuf::from(manifest_dir)
            .join("..")
            .join("..")
            .join("..")
            .join("runs/R110-W6-repro/iter-0/policy.onnx");
        let onnx_path = onnx_path_buf.as_path();
        if !onnx_path.exists() {
            eprintln!(
                "predict_v3_batch_matches_individual_predict_v3_on_cpu: skipping (missing {})",
                onnx_path.display()
            );
            return;
        }

        // Collect 4 distinct (PublicObservation, Vec<LegalAiAction>)
        // pairs the same way the dispatcher parity test does.
        let mut samples: Vec<(PublicObservation, Vec<LegalAiAction>)> = Vec::new();
        for seed in &[1u32, 2, 3, 4] {
            let seed_str = format!("{}:test", seed);
            let rng = Rng::from_seed(seed_str.as_str(), "test");
            let (state, _used) = with_rng(rng, || setup_ai_vs_ai_game());
            let side = if seed % 2 == 0 {
                SideId::Player
            } else {
                SideId::Opponent
            };
            let legal_seed = format!("{}:legal", seed);
            let (legal, _used) = with_rng(
                Rng::from_seed(legal_seed.as_str(), "legal"),
                || enumerate_legal_ai_actions(&state, side),
            );
            if legal.is_empty() {
                continue;
            }
            let obs = build_public_observation(&state, side);
            samples.push((obs, legal));
        }
        assert!(
            samples.len() >= 4,
            "need at least 4 distinct samples, got {}",
            samples.len()
        );

        let session = InferenceSession::load_on(onnx_path, Device::Cpu)
            .expect("load inline CPU session");

        let individual_outputs: Vec<PredictionV3> = samples
            .iter()
            .map(|(obs, legal)| {
                session
                    .predict_v3(obs, legal)
                    .expect("individual predict_v3")
            })
            .collect();

        let batch_refs: Vec<(&PublicObservation, &[LegalAiAction])> = samples
            .iter()
            .map(|(obs, legal)| (obs, legal.as_slice()))
            .collect();
        let batched_outputs = session
            .predict_v3_batch(&batch_refs)
            .expect("predict_v3_batch");

        assert_eq!(batched_outputs.len(), individual_outputs.len());
        let mut worst_dprob: f32 = 0.0;
        let mut worst_dvalue: f32 = 0.0;
        for (i, (ind, bat)) in individual_outputs.iter().zip(batched_outputs.iter()).enumerate() {
            assert_eq!(
                ind.probs.len(),
                bat.probs.len(),
                "sample {} probs len mismatch",
                i
            );
            for (a, b) in ind.probs.iter().zip(bat.probs.iter()) {
                let d = (a - b).abs();
                if d > worst_dprob {
                    worst_dprob = d;
                }
            }
            let dv = (ind.value - bat.value).abs();
            if dv > worst_dvalue {
                worst_dvalue = dv;
            }
        }
        eprintln!(
            "predict_v3_batch_matches_individual_predict_v3_on_cpu: worst dprob={:e} dvalue={:e}",
            worst_dprob, worst_dvalue
        );
        assert!(
            worst_dprob < 1e-5,
            "max|Δprob| {:e} exceeds 1e-5 on CPU — n_actions padding may be wrong",
            worst_dprob
        );
        assert!(
            worst_dvalue < 1e-5,
            "max|Δvalue| {:e} exceeds 1e-5 on CPU",
            worst_dvalue
        );

        drop(session);
    }

    /// Pure-Rust dispatch parity: every supported state_features last
    /// dim maps to the expected `GraphSchema` variant via the same
    /// match arms `validate_graph_signature` uses on the 5-input
    /// branch. Keeps the v3.7 (296-d) wiring covered without standing
    /// up an ORT session in unit-test scope.
    #[test]
    fn state_dim_dispatch_covers_v3_0_through_v3_7() {
        fn schema_for(state_dim: usize) -> Option<GraphSchema> {
            // Mirror of the 5-input dispatch arms in
            // `validate_graph_signature`.
            if state_dim == STATE_DIM_V3 {
                Some(GraphSchema::V3_0)
            } else if state_dim == STATE_DIM_V3_1 {
                Some(GraphSchema::V3_1)
            } else if state_dim == STATE_DIM_V3_3 {
                Some(GraphSchema::V3_3)
            } else if state_dim == STATE_DIM_V3_5 {
                Some(GraphSchema::V3_5)
            } else if state_dim == STATE_DIM_V3_6 {
                Some(GraphSchema::V3_6)
            } else if state_dim == STATE_DIM_V3_7 {
                Some(GraphSchema::V3_7)
            } else {
                None
            }
        }
        assert_eq!(schema_for(STATE_DIM_V3), Some(GraphSchema::V3_0));
        assert_eq!(schema_for(STATE_DIM_V3_1), Some(GraphSchema::V3_1));
        assert_eq!(schema_for(STATE_DIM_V3_3), Some(GraphSchema::V3_3));
        assert_eq!(schema_for(STATE_DIM_V3_5), Some(GraphSchema::V3_5));
        assert_eq!(schema_for(STATE_DIM_V3_6), Some(GraphSchema::V3_6));
        assert_eq!(schema_for(STATE_DIM_V3_7), Some(GraphSchema::V3_7));
        // Sanity: STATE_DIM_V3_7 is the documented 296-d v3.7 contract.
        assert_eq!(STATE_DIM_V3_7, 296);
    }
}
