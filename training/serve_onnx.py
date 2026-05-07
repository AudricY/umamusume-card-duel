from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from uma_ai.features import legal_actions_to_features, observation_to_features


class PolicyServer(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        model_path: str,
        provider: str,
    ) -> None:
        super().__init__(address, handler)
        self.session = ort.InferenceSession(model_path, providers=resolve_providers(provider))


class Handler(BaseHTTPRequestHandler):
    server: PolicyServer

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_error(404)
            return
        self.respond({"status": "ok"})

    def do_POST(self) -> None:
        if self.path != "/predict":
            self.send_error(404)
            return
        try:
            payload = self.read_json()
            arrays, action_ids = request_to_arrays(payload)
            logits, value = self.server.session.run(None, arrays)
            selected = logits.argmax(axis=1).astype(int)
            response: dict[str, Any] = {
                "logits": logits.tolist(),
                "value": value.tolist(),
                "selectedIndex": selected.tolist(),
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


def request_to_arrays(payload: dict[str, Any]) -> tuple[dict[str, np.ndarray], list[list[str]] | None]:
    if "observation" in payload and "legalActions" in payload:
        actions = payload["legalActions"]
        if not actions:
            raise ValueError("legalActions must not be empty")
        state_features = observation_to_features(payload["observation"])[None, :]
        action_features = legal_actions_to_features(actions)[None, :, :]
        action_mask = np.ones(action_features.shape[:2], dtype=np.bool_)
        action_ids = [[str(action.get("id", index)) for index, action in enumerate(actions)]]
    else:
        state_features = np.asarray(payload["state_features"], dtype=np.float32)
        action_features = np.asarray(payload["action_features"], dtype=np.float32)
        action_mask = np.asarray(payload["action_mask"], dtype=np.bool_)
        action_ids = None
    if state_features.ndim != 2:
        raise ValueError("state_features must have shape [batch,state_dim]")
    if action_features.ndim != 3:
        raise ValueError("action_features must have shape [batch,actions,action_dim]")
    if action_mask.shape != action_features.shape[:2]:
        raise ValueError("action_mask must have shape [batch,actions]")
    return {
        "state_features": state_features.astype(np.float32),
        "action_features": action_features.astype(np.float32),
        "action_mask": action_mask.astype(np.bool_),
    }, action_ids


def main() -> None:
    args = parse_args()
    model_path = str(Path(args.model))
    server = PolicyServer((args.host, args.port), Handler, model_path, args.provider)
    print(json.dumps({
        "status": "serving",
        "host": args.host,
        "port": args.port,
        "model": model_path,
        "providers": server.session.get_providers(),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve an ONNX candidate policy over HTTP.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--provider", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    main()
