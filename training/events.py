"""Unified pipeline event stream.

Single append-only ``runs/<run>/events.jsonl`` written by every stage
(rule-bot replay, trace-gen, relabel, mix, train, gate, pool-eval,
calibrate, decision). Schema is intentionally flat so downstream views
(TensorBoard adapters, the Flask observability app) can index by
``iteration`` and ``stage`` without parsing nested manifests.

Concurrency model: POSIX ``O_APPEND`` makes single ``write()`` calls
atomic for sizes under PIPE_BUF (4 KiB on Linux), which covers every
event we emit. The in-process lock additionally guards interleaving
when the same writer instance is shared across threads.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 1


@dataclass
class Event:
    ts: float
    run_id: str
    iteration: int
    stage: str
    event_type: str
    data: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(
            {
                "ts": self.ts,
                "run_id": self.run_id,
                "iteration": self.iteration,
                "stage": self.stage,
                "event_type": self.event_type,
                "schema_version": self.schema_version,
                "data": self.data,
            },
            separators=(",", ":"),
        )


class EventWriter:
    """Append-only JSONL event writer scoped to a run directory.

    Accepts either a directory (then writes ``events.jsonl`` inside it) or
    an explicit ``*.jsonl`` file path. The latter lets a subprocess
    (e.g. ``train_bc.py``) write into the orchestrator's stream without
    having to reconstruct the directory convention.
    """

    def __init__(self, target: Path | str):
        p = Path(target)
        if p.suffix == ".jsonl":
            self.path = p
            self.run_dir = p.parent
        else:
            self.run_dir = p
            self.path = p / "events.jsonl"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = self.run_dir.name
        self._lock = threading.Lock()

    def emit(
        self,
        *,
        iteration: int,
        stage: str,
        event_type: str,
        **data: Any,
    ) -> None:
        event = Event(
            ts=time.time(),
            run_id=self.run_id,
            iteration=iteration,
            stage=stage,
            event_type=event_type,
            data=data,
        )
        line = event.to_json() + "\n"
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())

    def emit_run(self, *, stage: str, event_type: str, **data: Any) -> None:
        """Run-level event (iteration = -1)."""

        self.emit(iteration=-1, stage=stage, event_type=event_type, **data)


def read_events(path: Path | str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def iter_events(path: Path | str) -> Iterator[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
