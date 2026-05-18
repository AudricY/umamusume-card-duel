from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from uma_ai.features import (
    ACTION_DIM,
    CARD_ID_SHAPES,
    STATE_DIM,
    STATE_DIM_V2,
    STATE_DIM_V3,
    STATE_DIM_V3_1,
    ZONE_ORDER,
    action_card_idx_pair,
    card_vocab_metadata,
    legal_actions_to_features,
    observation_to_card_ids,
    observation_to_features,
    observation_to_features_v2,
)

# R7.b.2 Phase 3: fixed per-zone width for the embedding inputs — mirrors
# `training/export_onnx.py`'s MAX_CARDS_PER_ZONE so the ORT input shape is
# stable across producers.
MAX_CARDS_PER_ZONE = max(CARD_ID_SHAPES.values())
NUM_ZONES = len(ZONE_ORDER)


class PolicyServer(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        model_path: str,
        provider: str,
        *,
        default_sampling: str = "greedy",
        default_temperature: float | None = None,
        ort_threads: int | None = 1,
        feature_schema: str = "auto",
    ) -> None:
        super().__init__(address, handler)
        preload_cuda_libraries(provider)
        # R14.G: ort_threads controls ORT's intra/inter-op thread pool.
        # Default 1 pins to single-threaded execution, removing the small
        # FP non-determinism that ORT's default parallel reductions can
        # introduce. With R14.B's AsyncLocalStorage fix landing engine
        # determinism at the right layer (the engine RNG state, not the
        # inference reduction order), it becomes safe to pass
        # ``--ort-threads 0`` (let ORT pick) or a specific count to
        # restore concurrent /predict throughput for workloads where the
        # simulator is fast and workers are starved on inference (e.g.
        # value-head-leaf selfplay loops).
        session_options = ort.SessionOptions()
        if ort_threads is not None and ort_threads > 0:
            session_options.intra_op_num_threads = ort_threads
            session_options.inter_op_num_threads = ort_threads
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # else: leave ORT defaults — parallel reductions, multi-threaded
        # within and between operators. Determinism guarantee narrows from
        # bit-exact /predict to bit-exact engine state (engine RNG still
        # deterministic; tiny per-call FP variation in policy logits is
        # absorbed by MCTS's visit counts at simulation budgets > a few
        # sims).
        self.session = ort.InferenceSession(model_path, sess_options=session_options, providers=resolve_providers(provider))
        self.ort_threads = ort_threads
        # Pin the state-feature encoding to the loaded graph. The 96-d
        # production model (`runs/R13-W6-phase-d/iter-2/policy.onnx`, schema
        # v2) has no embedding inputs; the HEAD-trained 110-d v3 graph adds
        # `card_ids_by_zone` / `action_card_idx`. Serving the wrong encoding
        # is a silent correctness bug (ORT also hard-rejects unknown feed
        # keys), so resolve + fail loud at startup rather than per-request.
        self.feature_schema = _resolve_feature_schema(feature_schema, self.session)
        self.runtime_card_vocab = card_vocab_metadata()
        self.expected_card_vocab = load_meta_card_vocab(model_path)
        if self.expected_card_vocab is not None and self.runtime_card_vocab.get("hash") != "missing":
            if self.expected_card_vocab.get("hash") != self.runtime_card_vocab.get("hash"):
                raise RuntimeError(
                    f"Card vocab hash mismatch at serve time: model={self.expected_card_vocab.get('hash')}"
                    f" runtime={self.runtime_card_vocab.get('hash')}"
                )
        # Server-wide defaults injected when /predict bodies omit ``sampling``
        # / ``temperature``. F1/PPO rollout drives this with
        # ``--default-sampling stochastic --default-temperature 1.0`` so the
        # existing greedy TS evaluator transport produces stochastic traces
        # without TS-side changes. See ppo_orchestrator for rationale.
        if default_sampling not in {"greedy", "stochastic"}:
            raise ValueError(f"unknown default sampling mode: {default_sampling}")
        self.default_sampling = default_sampling
        self.default_temperature = default_temperature


def _graph_signature(session: ort.InferenceSession) -> tuple[int | None, bool]:
    """Return (state_features last dim or None, has_embedding_inputs).

    `has_embedding_inputs` is True iff the graph declares a
    `card_ids_by_zone` input (the v3 embedding pass). The state-features
    last dim distinguishes 96-d (v2) from 110-d (v3).
    """
    state_dim: int | None = None
    has_embedding = False
    for inp in session.get_inputs():
        if inp.name == "card_ids_by_zone":
            has_embedding = True
        if inp.name == "state_features":
            shape = inp.shape or []
            if shape:
                last = shape[-1]
                if isinstance(last, int):
                    state_dim = last
    return state_dim, has_embedding


# --- Serving-schema 96/110/164 guard (r16 P1 prerequisite) ----------------
# The feature builder is resolved STRICTLY from the loaded ONNX graph's
# `state_features` last dim against this explicit table. This replaces the
# old binary "96 -> v2 else -> v3" rule, which would have silently paired a
# future 164-d v3.1 graph with the frozen 110-d v3.0 builder once STATE_DIM
# is bumped for P1. Each entry: graph_state_dim -> (schema token, requires
# `card_ids_by_zone` embedding input, human label). The schema token is the
# internal selector consumed by `request_to_arrays` ("v2" = frozen 96-d, no
# embedding feeds; "v3" = frozen 110-d v3.0, embedding feeds). 164 is a
# DISABLED placeholder: it has a named dim contract but no builder until P1
# (STATE_DIM 164 temporal/turn-state features) lands, so it fails fast here.
#
# `_SCHEMA_BUILDER` enumerates only the schemas that have a real builder
# today. `_PLACEHOLDER_DIMS` are known-but-unimplemented dims that must fail
# loud (never silently fall back to v3) so a bump cannot corrupt serving.
_SCHEMA_BY_STATE_DIM: dict[int, tuple[str, bool, str]] = {
    STATE_DIM_V2: ("v2", False, "frozen 96-d v2 (no embedding inputs)"),
    STATE_DIM_V3: ("v3", True, "frozen 110-d v3.0 (embedding inputs)"),
}
_PLACEHOLDER_DIMS: dict[int, str] = {
    STATE_DIM_V3_1: (
        "STATE_DIM=164 v3.1 temporal/turn-state schema (r16 P1) is a "
        "declared-but-unimplemented placeholder: no feature builder exists "
        "yet. Land P1 (features.py STATE_DIM_V3_1 builder + export) before "
        "serving a 164-d graph"
    ),
}


def _resolve_feature_schema(requested: str, session: ort.InferenceSession) -> str:
    """Resolve the serving feature builder STRICTLY by graph state dim.

    Resolution is keyed off the ONNX graph's `state_features` last dim
    against `_SCHEMA_BY_STATE_DIM` (96 -> v2, 110 -> v3.0). Any dim that is
    not a known *implemented* schema fails fast at startup: declared-but-
    unimplemented dims (e.g. the 164-d v3.1 P1 placeholder) raise with a
    specific message; anything else raises generically. The guard never
    silently defaults to v3 — that silent fallback is exactly what would let
    a STATE_DIM 164 bump corrupt 96-d/110-d serving. An explicit
    `v2`/`v3` request is asserted consistent with the resolved schema and
    exits non-zero at startup if not. Exactly one startup line is logged.
    """
    state_dim, has_embedding = _graph_signature(session)

    if state_dim is None:
        sys.stderr.write(
            "[serve_onnx] FATAL: could not read a concrete `state_features` "
            "last dim from the ONNX graph signature; refusing to guess a "
            "feature schema. Re-export the graph with a fixed state dim.\n"
        )
        raise SystemExit(2)

    if state_dim not in _SCHEMA_BY_STATE_DIM:
        placeholder = _PLACEHOLDER_DIMS.get(state_dim)
        if placeholder is not None:
            sys.stderr.write(
                f"[serve_onnx] FATAL: graph state_dim={state_dim} maps to a "
                f"declared-but-unimplemented schema. {placeholder}.\n"
            )
            raise SystemExit(2)
        known = ", ".join(str(d) for d in sorted(_SCHEMA_BY_STATE_DIM))
        sys.stderr.write(
            f"[serve_onnx] FATAL: graph state_dim={state_dim} matches no "
            f"known feature schema (known: {known}). Refusing to serve an "
            f"unrecognized encoding.\n"
        )
        raise SystemExit(2)

    schema, expects_embedding, label = _SCHEMA_BY_STATE_DIM[state_dim]

    # The state dim alone selects the schema, but the embedding-input
    # presence must agree with that schema or the feeds will not match the
    # graph (v3 needs `card_ids_by_zone`; v2 must NOT receive it).
    if has_embedding != expects_embedding:
        sys.stderr.write(
            f"[serve_onnx] FATAL: graph state_dim={state_dim} resolves to "
            f"schema {schema} ({label}) but `card_ids_by_zone` input "
            f"presence={has_embedding} contradicts it "
            f"(expected {expects_embedding}). The ONNX graph signature is "
            f"internally inconsistent; refusing to serve.\n"
        )
        raise SystemExit(2)

    if requested == "auto":
        source = "auto-graph"
    elif requested in {"v2", "v3"}:
        source = "--feature-schema/env"
        if requested != schema:
            sys.stderr.write(
                f"[serve_onnx] FATAL: --feature-schema={requested} is "
                f"inconsistent with the loaded ONNX graph signature "
                f"(state_dim={state_dim}, card_ids_by_zone input="
                f"{has_embedding} -> graph is {schema}). Refusing to "
                f"serve a mismatched encoding.\n"
            )
            raise SystemExit(2)
    else:
        raise ValueError(f"unknown feature schema: {requested!r}")

    print(
        f"[serve_onnx] feature schema = {schema} "
        f"(graph state_dim={state_dim}, {label}) [source: {source}]"
    )
    return schema


def load_meta_card_vocab(model_path: str) -> dict[str, Any] | None:
    sidecar = Path(model_path).with_suffix(Path(model_path).suffix + ".meta.json")
    if not sidecar.exists():
        return None
    payload = json.loads(sidecar.read_text(encoding="utf8"))
    return payload.get("card_vocab")


class Handler(BaseHTTPRequestHandler):
    server: PolicyServer

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_error(404)
            return
        self.respond({"status": "ok", "providers": self.server.session.get_providers()})

    def do_POST(self) -> None:
        if self.path != "/predict":
            self.send_error(404)
            return
        try:
            payload = self.read_json()
            arrays, action_ids = request_to_arrays(payload, self.server.feature_schema)
            logits, value = self.server.session.run(None, arrays)
            mask = arrays["action_mask"]
            # Per-request sampling mode. ``greedy`` is the DAgger gate
            # default; ``stochastic`` is the F1/PPO collector mode that
            # enables non-zero-entropy rollouts and finite importance ratios.
            # ``actionLogProbs`` always reflects the distribution actually
            # used to sample (post-temperature) so trace consumers can use
            # it directly as ``behavior_logp`` for importance weights.
            # Server defaults are consulted only when the request body
            # omits the field. Explicit ``sampling: greedy`` in a body still
            # forces greedy even if the server was started with
            # ``--default-sampling stochastic``. This is the escape hatch
            # the gate-eval path uses to stay greedy under a stochastic PPO
            # collector server (item 18: gate is greedy, collection is
            # stochastic, both share the serve_onnx process).
            sampling = str(payload.get("sampling", self.server.default_sampling)).lower()
            if sampling not in {"greedy", "stochastic"}:
                raise ValueError(f"unknown sampling mode: {sampling}")
            default_temp = self.server.default_temperature
            if "temperature" in payload:
                temperature = float(payload["temperature"])
            elif default_temp is not None:
                temperature = float(default_temp)
            else:
                temperature = 0.0 if sampling == "greedy" else 1.0
            if sampling == "stochastic" and temperature <= 0.0:
                # Stochastic with temp=0 collapses to argmax; treat as greedy
                # to avoid divide-by-zero in the Gumbel softmax.
                sampling = "greedy"
                temperature = 0.0

            if sampling == "greedy":
                log_probs = masked_log_softmax(logits, mask)
                action_probs = masked_softmax(logits, mask)
                selected = logits.argmax(axis=1).astype(int)
            else:
                scaled_logits = logits.astype(np.float64) / temperature
                log_probs = masked_log_softmax(scaled_logits.astype(np.float32), mask)
                action_probs = masked_softmax(scaled_logits.astype(np.float32), mask)
                sampling_seed = payload.get("samplingSeed")
                rng = np.random.default_rng(sampling_seed if sampling_seed is not None else None)
                # Gumbel-max: argmax(log p + Gumbel(0,1)) ~ Categorical(p).
                # Masked positions already have log_p = -1e9, dominating any
                # Gumbel noise so they are never selected.
                gumbel = -np.log(-np.log(rng.uniform(low=1e-12, high=1.0, size=log_probs.shape)))
                selected = (log_probs.astype(np.float64) + gumbel).argmax(axis=1).astype(int)
            selected_log_probs = [
                float(log_probs[row, idx]) for row, idx in enumerate(selected.tolist())
            ]
            response: dict[str, Any] = {
                "logits": logits.tolist(),
                "value": value.tolist(),
                "selectedIndex": selected.tolist(),
                "actionLogProbs": log_probs.tolist(),
                "actionProbs": action_probs.tolist(),
                "selectedLogProb": selected_log_probs,
                "behaviorPolicy": {
                    "kind": sampling,
                    "temperature": temperature,
                },
            }
            if action_ids is not None:
                response["selectedActionId"] = [action_ids[row][index] for row, index in enumerate(selected.tolist())]
            self.respond(response)
        except Exception as exc:
            self.respond({"error": str(exc)}, status=400)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length).decode("utf8"))

    def respond(self, payload: dict[str, Any], *, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def masked_log_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Numerically-stable log-softmax over masked logits.

    Masked positions get -inf log-prob (so e^logp = 0). Item 18: F1's
    importance-sampling ratio is exp(target_logp - behavior_logp); masked
    actions must contribute zero probability to keep the ratio finite.
    """
    masked = np.where(mask, logits.astype(np.float64), -np.inf)
    max_per_row = np.max(masked, axis=1, keepdims=True)
    max_per_row = np.where(np.isfinite(max_per_row), max_per_row, 0.0)
    shifted = masked - max_per_row
    exp_shifted = np.where(mask, np.exp(shifted), 0.0)
    log_sum_exp = np.log(np.sum(exp_shifted, axis=1, keepdims=True) + 1e-30)
    log_probs = shifted - log_sum_exp
    # JSON cannot represent -inf; clamp masked positions to a large negative
    # finite sentinel that is still safe in any downstream exp() (exp(-1e9) ≈ 0).
    log_probs = np.where(mask, log_probs, -1.0e9)
    return log_probs.astype(np.float32)


def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mask-aware softmax that zeroes masked positions exactly."""
    log_probs = masked_log_softmax(logits, mask)
    probs = np.exp(log_probs.astype(np.float64))
    probs = np.where(mask, probs, 0.0)
    return probs.astype(np.float32)


def request_to_arrays(
    payload: dict[str, Any], feature_schema: str = "v3"
) -> tuple[dict[str, np.ndarray], list[list[str]] | None]:
    # v2 = the FROZEN 96-d production encoding (no embedding inputs); v3 =
    # the HEAD-trained 110-d additive embedding encoding. The pinned schema
    # is resolved once at startup (see _resolve_feature_schema); here it
    # only selects the encoder + which feed keys are emitted. ORT hard-
    # rejects unknown feed keys, so v2 MUST omit (not zero) the embedding
    # inputs.
    is_v2 = feature_schema == "v2"
    expected_state_dim = STATE_DIM_V2 if is_v2 else STATE_DIM
    if "observation" in payload and "legalActions" in payload:
        actions = payload["legalActions"]
        if not actions:
            raise ValueError("legalActions must not be empty")
        observation = payload["observation"]
        encode_state = observation_to_features_v2 if is_v2 else observation_to_features
        state_features = encode_state(observation)[None, :]
        action_features = legal_actions_to_features(actions)[None, :, :]
        action_mask = np.ones(action_features.shape[:2], dtype=np.bool_)
        action_ids = [[str(action.get("id", index)) for index, action in enumerate(actions)]]
        if not is_v2:
            # R7.b.2 Phase 3: build the embedding-pass tensors from the JSON
            # observation (Phase 1 emits `cardIdsByZone`) and per-action
            # source/target idx (Phase 1 adds `actionSourceCardIdx` /
            # `actionTargetCardIdx`). Pack into the same fixed shape the
            # ONNX graph expects (NUM_ZONES, MAX_CARDS_PER_ZONE) so ORT's
            # shape inference matches export-time exactly.
            card_id_zones = observation_to_card_ids(observation)
            card_ids_by_zone = np.zeros((1, NUM_ZONES, MAX_CARDS_PER_ZONE), dtype=np.int64)
            for zone_index, zone in enumerate(ZONE_ORDER):
                zone_arr = card_id_zones[zone]
                card_ids_by_zone[0, zone_index, : zone_arr.shape[0]] = zone_arr
            action_card_idx = np.zeros((1, len(actions), 2), dtype=np.int64)
            for action_index, action in enumerate(actions):
                action_card_idx[0, action_index, :] = action_card_idx_pair(action)
    else:
        state_features = np.asarray(payload["state_features"], dtype=np.float32)
        action_features = np.asarray(payload["action_features"], dtype=np.float32)
        action_mask = np.asarray(payload["action_mask"], dtype=np.bool_)
        action_ids = None
        if not is_v2:
            # Raw-arrays callers (training/debug paths) can pre-pack the new
            # tensors too. Default to zero so the embedding pass collapses to
            # the additive-residual null path (verified <1e-6 in Phase 2).
            # Under a v2 pin we never build these — a v2 graph rejects them.
            if "card_ids_by_zone" in payload:
                card_ids_by_zone = np.asarray(payload["card_ids_by_zone"], dtype=np.int64)
            else:
                card_ids_by_zone = np.zeros(
                    (state_features.shape[0], NUM_ZONES, MAX_CARDS_PER_ZONE), dtype=np.int64
                )
            if "action_card_idx" in payload:
                action_card_idx = np.asarray(payload["action_card_idx"], dtype=np.int64)
            else:
                action_card_idx = np.zeros(
                    (action_features.shape[0], action_features.shape[1], 2), dtype=np.int64
                )
    if state_features.ndim != 2:
        raise ValueError("state_features must have shape [batch,state_dim]")
    if state_features.shape[1] != expected_state_dim:
        raise ValueError(
            f"state_features dimension mismatch: got {state_features.shape[1]}, "
            f"expected {expected_state_dim} (feature schema {feature_schema})"
        )
    if action_features.ndim != 3:
        raise ValueError("action_features must have shape [batch,actions,action_dim]")
    if action_features.shape[2] != ACTION_DIM:
        raise ValueError(f"action_features dimension mismatch: got {action_features.shape[2]}, expected {ACTION_DIM}")
    if action_mask.shape != action_features.shape[:2]:
        raise ValueError("action_mask must have shape [batch,actions]")
    feed: dict[str, np.ndarray] = {
        "state_features": state_features.astype(np.float32),
        "action_features": action_features.astype(np.float32),
        "action_mask": action_mask.astype(np.bool_),
    }
    if not is_v2:
        expected_czi = (state_features.shape[0], NUM_ZONES, MAX_CARDS_PER_ZONE)
        if card_ids_by_zone.shape != expected_czi:
            raise ValueError(
                f"card_ids_by_zone shape mismatch: got {card_ids_by_zone.shape}, expected {expected_czi}"
            )
        expected_aci = (action_features.shape[0], action_features.shape[1], 2)
        if action_card_idx.shape != expected_aci:
            raise ValueError(
                f"action_card_idx shape mismatch: got {action_card_idx.shape}, expected {expected_aci}"
            )
        # ORT hard-rejects unknown feed keys, so the v2 (96-d) graph MUST
        # receive ONLY these three; the embedding inputs are omitted, not
        # zeroed.
        feed["card_ids_by_zone"] = card_ids_by_zone.astype(np.int64)
        feed["action_card_idx"] = action_card_idx.astype(np.int64)
    return feed, action_ids


def main() -> None:
    args = parse_args()
    model_path = str(Path(args.model))
    # --ort-threads parsing: "auto" -> None (ORT defaults), int -> pin.
    raw = args.ort_threads
    if raw == "auto":
        ort_threads: int | None = None
    else:
        ort_threads = int(raw)
    server = PolicyServer(
        (args.host, args.port),
        Handler,
        model_path,
        args.provider,
        default_sampling=args.default_sampling,
        default_temperature=args.default_temperature,
        ort_threads=ort_threads,
        feature_schema=args.feature_schema,
    )
    print(json.dumps({
        "status": "serving",
        "host": args.host,
        "port": args.port,
        "model": model_path,
        "providers": server.session.get_providers(),
        "card_vocab": server.runtime_card_vocab,
        "default_sampling": server.default_sampling,
        "default_temperature": server.default_temperature,
        "ort_threads": ort_threads if ort_threads is not None else "auto",
        "feature_schema": server.feature_schema,
    }))
    server.serve_forever()


def resolve_providers(provider: str) -> list[str]:
    available = set(ort.get_available_providers())
    if provider == "cpu":
        return ["CPUExecutionProvider"]
    if provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDAExecutionProvider is not available; install onnxruntime-gpu and CUDA runtime support.")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def preload_cuda_libraries(provider: str) -> None:
    if provider == "cpu":
        return
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return
    preload = getattr(ort, "preload_dlls", None)
    if callable(preload):
        try:
            preload()
        except Exception:
            pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.init()
    except Exception:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve an ONNX candidate policy over HTTP.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--provider", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--default-sampling",
        choices=["greedy", "stochastic"],
        default="greedy",
        help=(
            "Sampling mode injected when a /predict body omits ``sampling``. "
            "F1/PPO rollout sets this to ``stochastic`` so the TS evaluator's "
            "default greedy POST body still produces stochastic traces "
            "without TS-side flag changes."
        ),
    )
    parser.add_argument(
        "--default-temperature",
        type=float,
        default=None,
        help=(
            "Temperature injected when a /predict body omits ``temperature``. "
            "Defaults to 0.0 under greedy and 1.0 under stochastic when not set."
        ),
    )
    parser.add_argument(
        "--feature-schema",
        choices=["auto", "v2", "v3"],
        default=os.environ.get("UMA_FEATURE_SCHEMA", "auto"),
        help=(
            "State-feature encoding pin. 'auto' (default; env "
            "UMA_FEATURE_SCHEMA overrides the default) selects v2 vs v3 "
            "from the loaded ONNX graph signature: v2 = the FROZEN 96-d "
            "production encoding (no embedding inputs, e.g. "
            "runs/R13-W6-phase-d/iter-2/policy.onnx); v3 = the HEAD 110-d "
            "additive embedding encoding. An explicit 'v2'/'v3' is asserted "
            "consistent with the graph and the server exits at startup if "
            "not. 'Promote later' = serve a 110-d model under v3/auto."
        ),
    )
    parser.add_argument(
        "--ort-threads",
        default="1",
        help=(
            "ORT intra/inter-op thread count. Default '1' pins to single-threaded "
            "execution (the legacy R13.W1 behavior, FP-deterministic under "
            "concurrent /predict). Pass 'auto' to let ORT decide (parallel "
            "reductions, higher /predict throughput, microscopic FP non-determinism "
            "that's absorbed by MCTS at >a-few sims). Pass an integer to pin to N."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
