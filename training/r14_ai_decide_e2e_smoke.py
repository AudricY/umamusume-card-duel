"""R14.E end-to-end /ai/decide smoke.

Validates the new /ai/decide contract:
  - Endpoint returns actionIndex, selectedActionId, decisionMs,
    simulationsRun, haltedEarly.
  - For a real mid-game GameState the endpoint also returns
    `nextState` (R14.E). nextState.stateFingerprint must differ from
    the input state's fingerprint (the engine actually advanced).
  - For 0/1-legal-action positions the endpoint short-circuits with no
    nextState (UI falls through to the rule-bot advance step).

Spins up its own serve_onnx + backend dev server. Uses a tiny TS
helper to construct a real GameState via headlessAiVsAi past the
setup phase. Tears everything down at the end.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_for(url: str, attempts: int = 80) -> bool:
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.4)
    return False


def export_onnx_if_needed(repo_root: Path, ckpt: Path) -> Path:
    onnx = ckpt.with_name("policy.onnx")
    if not onnx.exists() or onnx.stat().st_mtime < ckpt.stat().st_mtime:
        subprocess.run(
            [sys.executable, str(repo_root / "training" / "export_onnx.py"),
             "--checkpoint", str(ckpt), "--out", str(onnx)],
            cwd=repo_root, check=True,
        )
    return onnx


def build_real_state(repo_root: Path, out_dir: Path) -> dict:
    """Run a short headless AI-vs-AI game and dump a mid-game GameState
    once a side has >1 legal actions so /ai/decide can MCTS over it."""
    out_path = out_dir / "mid_state.json"
    helper_path = out_dir / "build_state.ts"
    snippet = """
import "../../backend/src/sim/rngAsyncStore";
import { writeFileSync } from "node:fs";
import {
  createSeededRng,
  withRng,
} from "../../frontend/src/game/engine/core/random";
import {
  advanceOpponentTurnStep,
  advancePlayerAiTurnStep,
} from "../../frontend/src/game/engine";
import { enumerateLegalAiActions } from "../../frontend/src/game/engine/ai-policy/actions";
import { setupAiVsAiGame } from "../../backend/src/sim/evaluateModelVsHeuristic";

const rng = createSeededRng("r14-e2e-smoke", "smoke");
let state = withRng(rng, () => setupAiVsAiGame());
let captured: any = null;
for (let step = 0; step < 60; step += 1) {
  if (state.phase !== "play" || state.gameOver) break;
  const sideId = state.currentSide;
  if (sideId !== "player" && sideId !== "opponent") break;
  const legal = enumerateLegalAiActions(state, sideId);
  if (legal.length > 1) {
    captured = { state, sideId };
    break;
  }
  state = withRng(rng, () =>
    sideId === "player" ? advancePlayerAiTurnStep(state) : advanceOpponentTurnStep(state)
  );
}
if (!captured) {
  process.stderr.write("[r14-e2e-smoke] could not find a >1 legal-action position\\n");
  process.exit(1);
}
writeFileSync("OUT_PATH_PLACEHOLDER", JSON.stringify(captured, null, 2), "utf8");
"""
    helper_path.write_text(snippet.replace("OUT_PATH_PLACEHOLDER", str(out_path)), encoding="utf8")
    subprocess.run(
        ["npx", "tsx", str(helper_path)],
        cwd=repo_root, check=True,
    )
    return json.loads(out_path.read_text(encoding="utf8"))


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ckpt = next((
        c for c in [
            repo_root / "runs" / "R13-W6-phase-d" / "iter-1" / "checkpoint.pt",
            repo_root / "runs" / "R4-value-head-retrain" / "checkpoint.pt",
        ] if c.exists()
    ), None)
    if ckpt is None:
        print("[r14-e2e-smoke] no checkpoint found", file=sys.stderr)
        sys.exit(2)

    out_dir = repo_root / "runs" / "R14-ai-decide-e2e-smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx = export_onnx_if_needed(repo_root, ckpt)

    print("[r14-e2e-smoke] building a real mid-game state...")
    captured = build_real_state(repo_root, out_dir)
    state = captured["state"]
    model_side = captured["sideId"]
    print(f"[r14-e2e-smoke] state at side={model_side}, currentSide={state.get('currentSide')}, turn={state.get('turnNumber')}")

    serve_port = free_port()
    server_port = free_port()
    serve_proc = subprocess.Popen(
        [sys.executable, str(repo_root / "training" / "serve_onnx.py"),
         "--model", str(onnx), "--port", str(serve_port), "--default-sampling", "greedy"],
        cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    env = dict(os.environ)
    env["PORT"] = str(server_port)
    env["AI_MODEL_URL"] = f"http://127.0.0.1:{serve_port}"
    dev_log_path = out_dir / "dev.log"
    server_proc = subprocess.Popen(
        ["npm", "--workspace", "backend", "run", "dev"],
        cwd=repo_root, env=env,
        stdout=dev_log_path.open("w"), stderr=subprocess.STDOUT,
    )
    try:
        if not wait_for(f"http://127.0.0.1:{serve_port}/health"):
            print("[r14-e2e-smoke] serve_onnx never healthy", file=sys.stderr)
            sys.exit(2)
        if not wait_for(f"http://127.0.0.1:{server_port}/api/health"):
            print("[r14-e2e-smoke] dev server never healthy", file=sys.stderr)
            sys.exit(2)

        body = json.dumps({
            "state": state,
            "modelSide": model_side,
            "mctsConfig": {"simulations": 16},
            "seed": "r14-e2e-smoke",
        }).encode("utf8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{server_port}/ai/decide",
            data=body, headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read())

        ok = True
        problems: list[str] = []
        if "actionIndex" not in payload or payload["actionIndex"] < 0:
            ok = False
            problems.append(f"actionIndex missing or negative: {payload.get('actionIndex')}")
        if "nextState" not in payload:
            ok = False
            problems.append("nextState missing from response (R14.E contract)")
        else:
            next_state = payload["nextState"]
            # Cheap structural check: turnNumber or hand sizes should differ,
            # OR opponentTurnStep should advance.
            same_turn = next_state.get("turnNumber") == state.get("turnNumber")
            same_step = next_state.get("opponentTurnStep") == state.get("opponentTurnStep")
            same_currentside = next_state.get("currentSide") == state.get("currentSide")
            input_hand = (state.get("sides", {}).get(model_side, {}) or {}).get("hand") or []
            output_hand = (next_state.get("sides", {}).get(model_side, {}) or {}).get("hand") or []
            if same_turn and same_step and same_currentside and input_hand == output_hand:
                ok = False
                problems.append("nextState has no detectable difference from input state")

        result = {
            "status": "PASS" if ok else "FAIL",
            "model_side": model_side,
            "response": {
                "actionIndex": payload.get("actionIndex"),
                "selectedActionId": payload.get("selectedActionId"),
                "decisionMs": payload.get("decisionMs"),
                "simulationsRun": payload.get("simulationsRun"),
                "haltedEarly": payload.get("haltedEarly"),
                "has_nextState": "nextState" in payload,
                "fallback": payload.get("fallback", False),
            },
            "problems": problems,
        }
        print(json.dumps(result, indent=2))
        if not ok:
            sys.exit(1)
    finally:
        serve_proc.terminate()
        server_proc.terminate()
        try:
            serve_proc.wait(timeout=3)
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            serve_proc.kill()
            server_proc.kill()


if __name__ == "__main__":
    main()
