"""Backlog regression smoke for serve_onnx under high worker fan-in.

Root cause this guards (R16-P1 v3.1 ablation iter-0 gate death): the gate
fans in ~24 concurrent workers, but `PolicyServer` inherited
`socketserver.TCPServer.request_queue_size = 5`, so `server_activate()` did
`socket.listen(5)`. When the worker pool launches (24 OS processes opening
connections at once while the single accept thread is busy), the kernel's
completed-connection (accept) queue overflows: dmesg logs "Possible SYN
flooding ... Sending cookies", overflow connections are refused/reset,
workers get ECONNRESET/ECONNREFUSED, the gate exits 1 and writes a bogus
wilson_lower:0.0.

The fix sets `PolicyServer.request_queue_size = 128`, so the actual
`listen()` backlog the kernel applies to the serving socket is 128.

This smoke has two parts:

  1. Backlog assertion (the load-bearing, deterministic proof). It reads
     the *actual kernel listen backlog* of the live serve_onnx socket via
     `ss` (Send-Q of a LISTEN row == the argument passed to listen(); host
     net.core.somaxconn is 4096 so it is not capped). It asserts the
     backlog is large (>= 64, and specifically > the old default of 5).
     This DIRECTLY characterizes both the bug (listen(5)) and the fix
     (listen(128)); verified by temporarily forcing request_queue_size = 5
     (ss then reports Send-Q 5 and this assertion FAILS) vs the shipped 128
     (Send-Q 128, PASS). It is deterministic — it cannot be masked by the
     kernel draining the accept queue quickly the way an in-process connect
     storm can on loopback.

  2. End-to-end concurrent burst (no-regression check). Fires bursts of
     `BURST` simultaneous /health connections under `LOAD` concurrent real
     /predict workers and asserts zero refusals/resets and all 200 — proves
     the larger backlog actually carries 24+ worker fan-in cleanly.

Use training/.venv/bin/python (system python lacks onnxruntime).
"""

from __future__ import annotations

import http.client
import json
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[1]

CANDIDATE_MODELS = [
    REPO_ROOT / "runs" / "R110-W6-repro" / "iter-0" / "policy.onnx",
    REPO_ROOT / "runs" / "R16-P1-v31-ablation" / "loop" / "iter-0" / "policy.onnx",
    REPO_ROOT / "runs" / "R13-W6-phase-d" / "iter-2" / "policy.onnx",
]

BURST = 64  # >> old default backlog of 5; covers 24+ worker gate fan-in
ROUNDS = 4
LOAD = 24  # concurrent persistent /predict workers — live-gate-like pressure
MIN_BACKLOG = 64  # must clear this; old default (5) is far below


def _pick_model() -> Path:
    for p in CANDIDATE_MODELS:
        if p.exists():
            return p
    print(
        "SKIP: no exported policy.onnx found under runs/ "
        f"(looked for {[str(p) for p in CANDIDATE_MODELS]})"
    )
    sys.exit(0)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _raw_array_payload(model: Path) -> dict:
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    dims = {i.name: i.shape for i in sess.get_inputs()}
    state_dim = int(dims["state_features"][-1])
    action_dim = int(dims["action_features"][-1])
    n = 3
    body: dict = {
        "state_features": np.zeros((1, state_dim), dtype=np.float32).tolist(),
        "action_features": np.zeros((1, n, action_dim), dtype=np.float32).tolist(),
        "action_mask": np.ones((1, n), dtype=bool).tolist(),
        "sampling": "greedy",
    }
    if "card_ids_by_zone" in dims:
        z = [int(d) for d in dims["card_ids_by_zone"][1:]]
        body["card_ids_by_zone"] = np.zeros((1, *z), dtype=np.int64).tolist()
        body["action_card_idx"] = np.zeros((1, n, 2), dtype=np.int64).tolist()
    return body


def _wait_healthy(port: int, proc: subprocess.Popen, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"serve_onnx exited early rc={proc.returncode}")
        try:
            urllib.request.urlopen(url, timeout=2).read()
            return
        except Exception:
            time.sleep(0.3)
    raise TimeoutError(f"serve_onnx not healthy on {port}")


def _kernel_listen_backlog(port: int) -> int:
    """Actual listen() backlog of the serving socket (ss Send-Q of LISTEN).

    For a LISTEN socket `ss` reports Recv-Q = current accept-queue depth and
    Send-Q = the backlog argument the app passed to listen(). somaxconn is
    4096 on this host so the value is not capped below request_queue_size.
    """
    out = subprocess.run(
        ["ss", "-tlnH", f"sport = :{port}"],
        capture_output=True, text=True, check=True,
    ).stdout
    for line in out.splitlines():
        parts = line.split()
        # State Recv-Q Send-Q Local Peer ...
        if len(parts) >= 4 and parts[0] == "LISTEN":
            return int(parts[2])
    raise RuntimeError(f"no LISTEN row for port {port} in ss output:\n{out}")


def _load_worker(port: int, body: bytes, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            c.request("POST", "/predict", body=body,
                       headers={"Content-Type": "application/json"})
            c.getresponse().read()
            c.close()
        except Exception:
            time.sleep(0.01)


def _burst(port: int, n: int) -> tuple[int, list[str]]:
    results: list[str] = [""] * n
    gate = threading.Barrier(n + 1)

    def worker(i: int) -> None:
        gate.wait()
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            c.request("GET", "/health")
            r = c.getresponse()
            b = r.read()
            results[i] = "OK" if (r.status == 200 and b"ok" in b) else f"HTTP {r.status}"
            c.close()
        except (ConnectionResetError, ConnectionRefusedError) as exc:
            results[i] = f"REFUSED/RESET: {exc!r}"
        except Exception as exc:  # noqa: BLE001
            results[i] = f"ERR: {exc!r}"

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in ts:
        t.start()
    gate.wait()
    for t in ts:
        t.join()
    ok = sum(1 for r in results if r == "OK")
    return ok, [r for r in results if r != "OK"]


def main() -> int:
    model = _pick_model()
    port = _free_port()
    body = json.dumps(_raw_array_payload(model)).encode()
    print(f"serve_onnx model={model.relative_to(REPO_ROOT)} port={port} "
          f"load={LOAD} burst={BURST}")
    proc = subprocess.Popen(
        [
            sys.executable,
            str(REPO_ROOT / "training" / "serve_onnx.py"),
            "--model", str(model),
            "--port", str(port),
            "--provider", "cpu",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    stop = threading.Event()
    load_threads: list[threading.Thread] = []
    failures: list[str] = []
    try:
        _wait_healthy(port, proc)

        # --- Part 1: deterministic backlog assertion (load-bearing) ---
        backlog = _kernel_listen_backlog(port)
        bl_ok = backlog >= MIN_BACKLOG
        print(f"  kernel listen backlog = {backlog} "
              f"(old default was 5; require >= {MIN_BACKLOG}) "
              f"[{'PASS' if bl_ok else 'FAIL'}]")
        if not bl_ok:
            failures.append(
                f"listen backlog {backlog} < {MIN_BACKLOG} "
                f"(serve_onnx would SYN-flood under 24-worker fan-in)"
            )

        # --- Part 2: end-to-end concurrent burst (no-regression) ---
        for _ in range(LOAD):
            t = threading.Thread(target=_load_worker, args=(port, body, stop),
                                  daemon=True)
            t.start()
            load_threads.append(t)
        time.sleep(1.0)
        for rnd in range(ROUNDS):
            ok, errors = _burst(port, BURST)
            ok_all = ok == BURST
            print(f"  round {rnd + 1}/{ROUNDS}: {ok}/{BURST} ok "
                  f"[{'PASS' if ok_all else 'FAIL'}]")
            if errors:
                for e in errors[:5]:
                    print(f"    - {e}")
                failures.append(f"round {rnd + 1}: {len(errors)} non-OK")
    finally:
        stop.set()
        for t in load_threads:
            t.join(timeout=2)
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    if failures:
        print(f"BACKLOG SMOKE FAIL: {failures}")
        return 1
    print(f"BACKLOG SMOKE PASS: listen backlog {backlog} >= {MIN_BACKLOG}; "
          f"{ROUNDS}x{BURST} concurrent /health under {LOAD}-worker /predict "
          f"load, 0 refusals/resets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
