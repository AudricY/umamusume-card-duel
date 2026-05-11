"""Flask observability dashboard for DAgger sweep runs (Stage 3, research tool).

Reads ``runs/<name>/orchestrator-state.json`` + ``events.jsonl`` (see
``training/events.py`` for schema). Self-contained: single file, no templates
dir. Page auto-refreshes every 10s via ``<meta http-equiv="refresh">``.

CLI:
    training/.venv/bin/python training/observability_app.py \
        --runs-dir runs --host 127.0.0.1 --port 5000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, render_template_string

STAGES = ["trace-gen", "relabel", "mix", "train", "gate", "pool-eval", "decision"]
STALL_S = 300.0  # 5 min


def _load_state(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "orchestrator-state.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _load_events(run_dir: Path) -> list[dict[str, Any]]:
    p = run_dir / "events.jsonl"
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        return out
    return out


def _list_runs(runs_dir: Path) -> list[dict[str, Any]]:
    rows = []
    if not runs_dir.exists():
        return rows
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        has_state = (d / "orchestrator-state.json").exists()
        has_events = (d / "events.jsonl").exists()
        if not (has_state or has_events):
            continue
        state = _load_state(d)
        events = _load_events(d)
        last_ts = events[-1]["ts"] if events else None
        iters = state.get("iterations", []) or []
        rows.append({
            "name": d.name,
            "status": _infer_status(state, events),
            "last_ts": last_ts,
            "last_age_s": (time.time() - last_ts) if last_ts else None,
            "iterations": len(iters),
            "promoted_wilson_lower": state.get("promoted_wilson_lower"),
        })
    return rows


def _infer_status(state: dict[str, Any], events: list[dict[str, Any]]) -> str:
    if state.get("halted"):
        return "halted"
    run_completed = any(
        e.get("stage") == "orchestrator" and e.get("event_type") == "run_completed"
        for e in events
    )
    if run_completed:
        return "done"
    if not events:
        return "idle"
    age = time.time() - events[-1]["ts"]
    if age > STALL_S:
        return "stalled"
    return "running"


def _pipeline_state(events: list[dict[str, Any]]) -> dict[int, dict[str, str]]:
    """Per-iteration map of stage -> 'pending'|'running'|'done'."""
    out: dict[int, dict[str, str]] = {}
    for e in events:
        it = e.get("iteration", -1)
        if it < 0:
            continue
        stage = e.get("stage")
        et = e.get("event_type")
        if stage not in STAGES:
            continue
        out.setdefault(it, {s: "pending" for s in STAGES})
        if et == "started" and out[it][stage] == "pending":
            out[it][stage] = "running"
        elif et in ("completed", "promoted", "rejected"):
            out[it][stage] = "done"
    # Decision stage uses promoted/rejected as terminal events.
    for it, evs in out.items():
        pass
    return out


def _epoch_series(events: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    series: dict[int, list[dict[str, Any]]] = {}
    for e in events:
        if e.get("stage") == "train" and e.get("event_type") == "epoch":
            it = e.get("iteration", -1)
            d = e.get("data", {})
            series.setdefault(it, []).append({
                "epoch": d.get("epoch"),
                "train_loss": d.get("train_loss"),
                "kl_loss": d.get("train_kl_loss", 0.0),
            })
    for it in series:
        series[it].sort(key=lambda r: r["epoch"] or 0)
    return series


def _resolve_run(runs_dir: Path, name: str) -> Path:
    # prevent path traversal
    if "/" in name or ".." in name or name.startswith("."):
        abort(404)
    p = runs_dir / name
    if not p.is_dir():
        abort(404)
    return p


def create_app(runs_dir: Path) -> Flask:
    app = Flask(__name__)
    app.config["RUNS_DIR"] = runs_dir.resolve()

    @app.route("/")
    def index() -> str:
        rows = _list_runs(app.config["RUNS_DIR"])
        return render_template_string(INDEX_HTML, rows=rows, runs_dir=str(app.config["RUNS_DIR"]))

    @app.route("/run/<name>")
    def run_view(name: str) -> str:
        run_dir = _resolve_run(app.config["RUNS_DIR"], name)
        state = _load_state(run_dir)
        events = _load_events(run_dir)
        pipeline = _pipeline_state(events)
        epoch_series = _epoch_series(events)
        last = events[-1] if events else None
        last_age = (time.time() - last["ts"]) if last else None
        iterations = state.get("iterations", []) or []
        # Build wilson trajectory dataset.
        wilson_points = [
            {"x": it.get("iteration"), "y": it.get("wilson_lower"),
             "promoted": bool(it.get("promoted"))}
            for it in iterations if it.get("wilson_lower") is not None
        ]
        # Pool eval matrix (rows=iter, cols=opponent_iter -> wilson_lower)
        pool_rows = []
        for it in iterations:
            cells = {pe.get("opponent_iteration"): pe.get("wilson_lower")
                     for pe in (it.get("pool_evals") or [])}
            if cells:
                pool_rows.append({"iteration": it.get("iteration"), "cells": cells})
        pool_cols = sorted({c for r in pool_rows for c in r["cells"].keys()})
        # Tail of events.
        tail = events[-20:]
        if events:
            t0 = events[0]["ts"]
            tail_view = [{
                "dt": e["ts"] - t0,
                "iteration": e.get("iteration"),
                "stage": e.get("stage"),
                "event_type": e.get("event_type"),
                "data": json.dumps(e.get("data", {}))[:200],
            } for e in tail]
        else:
            tail_view = []
        return render_template_string(
            RUN_HTML,
            name=name,
            state=state,
            status=_infer_status(state, events),
            last=last,
            last_age=last_age,
            iterations=iterations,
            pipeline=pipeline,
            stages=STAGES,
            wilson_points=wilson_points,
            promoted_floor=state.get("promoted_wilson_lower"),
            epoch_series=epoch_series,
            pool_rows=pool_rows,
            pool_cols=pool_cols,
            tail=tail_view,
        )

    @app.route("/run/<name>/api/state.json")
    def api_state(name: str):
        run_dir = _resolve_run(app.config["RUNS_DIR"], name)
        return jsonify(_load_state(run_dir))

    @app.route("/run/<name>/api/events.jsonl")
    def api_events(name: str):
        run_dir = _resolve_run(app.config["RUNS_DIR"], name)
        p = run_dir / "events.jsonl"
        text = p.read_text(encoding="utf-8") if p.exists() else ""
        return text, 200, {"Content-Type": "application/x-ndjson"}

    return app


CSS = """
body{font-family:system-ui,sans-serif;margin:0;padding:1rem;background:#fafafa;color:#222}
h1,h2,h3{margin:.6rem 0}table{border-collapse:collapse;margin:.5rem 0}
th,td{border:1px solid #ddd;padding:.25rem .5rem;font-size:13px;text-align:left}
th{background:#eee}a{color:#2196f3;text-decoration:none}a:hover{text-decoration:underline}
.status-running{color:#2196f3;font-weight:600}.status-done{color:#4caf50;font-weight:600}
.status-halted{color:#f44336;font-weight:600}.status-stalled{color:#ffc107;font-weight:600}
.status-idle{color:#9e9e9e}.pill{display:inline-block;padding:1px 6px;border-radius:8px;font-size:11px;color:#fff}
.pill-promoted{background:#4caf50}.pill-rejected{background:#f44336}
.strip{display:flex;gap:4px;margin:4px 0}.cell{padding:4px 8px;border-radius:3px;font-size:11px;color:#fff;min-width:60px;text-align:center}
.cell-pending{background:#9e9e9e}.cell-running{background:#2196f3}.cell-done{background:#4caf50}
.heat{min-width:42px;text-align:center;font-size:11px;color:#222}
canvas{max-width:600px!important;max-height:300px!important}
.section{background:#fff;padding:.8rem;margin:.6rem 0;border:1px solid #e0e0e0;border-radius:4px}
.small{font-size:11px;color:#666}.mono{font-family:ui-monospace,monospace;font-size:11px}
"""

INDEX_HTML = """<!doctype html><html><head><meta charset=utf-8>
<meta http-equiv=refresh content=10><title>Observability</title><style>{{css}}</style></head>
<body><h1>Sweep runs</h1><div class=small>runs dir: {{ runs_dir }}</div>
<table><tr><th>run</th><th>status</th><th>iters</th><th>last event</th>
<th>promoted floor</th></tr>
{% for r in rows %}<tr>
<td><a href="/run/{{ r.name }}">{{ r.name }}</a></td>
<td class="status-{{ r.status }}">{{ r.status }}</td>
<td>{{ r.iterations }}</td>
<td class=small>{% if r.last_age_s is not none %}{{ '%.0f'|format(r.last_age_s) }}s ago{% else %}-{% endif %}</td>
<td>{% if r.promoted_wilson_lower is not none %}{{ '%.4f'|format(r.promoted_wilson_lower) }}{% else %}-{% endif %}</td>
</tr>{% endfor %}</table>
{% if not rows %}<p>No runs with orchestrator-state.json or events.jsonl found.</p>{% endif %}
</body></html>
""".replace("{{css}}", CSS)


RUN_HTML = """<!doctype html><html><head><meta charset=utf-8>
<meta http-equiv=refresh content=10><title>{{ name }}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>{{css}}</style></head>
<body>
<div><a href="/">&larr; all runs</a></div>
<h1>{{ name }}</h1>
<div class=section><b>status:</b> <span class="status-{{ status }}">{{ status }}</span>
&nbsp;<b>current iteration:</b> {{ last.iteration if last else '-' }}
&nbsp;<b>last stage:</b> {{ last.stage if last else '-' }}/{{ last.event_type if last else '-' }}
&nbsp;<b>last event age:</b> {% if last_age is not none %}{{ '%.0f'|format(last_age) }}s{% else %}-{% endif %}
&nbsp;<b>promoted floor:</b> {% if promoted_floor is not none %}{{ '%.4f'|format(promoted_floor) }}{% else %}-{% endif %}
{% if state.halted %}<br><b style="color:#f44336">HALTED:</b> {{ state.halt_reason }}{% endif %}
</div>

<div class=section><h2>Pipeline state</h2>
{% for it_num, stagemap in pipeline|dictsort %}
<div>iteration {{ it_num }}<div class=strip>
{% for s in stages %}<div class="cell cell-{{ stagemap[s] }}">{{ s }}</div>{% endfor %}
</div></div>
{% else %}<div class=small>no iteration events yet</div>{% endfor %}
</div>

<div class=section><h2>Wilson lower trajectory</h2>
<canvas id=wilson></canvas>
<script>
const wpts={{ wilson_points|tojson }};
const floor={{ promoted_floor|tojson }};
const lbl=wpts.map(p=>p.x);
const dv=wpts.map(p=>p.y);
const colors=wpts.map(p=>p.promoted?'#4caf50':'#f44336');
new Chart(document.getElementById('wilson'),{type:'line',data:{labels:lbl,
datasets:[{label:'wilson_lower',data:dv,borderColor:'#2196f3',pointBackgroundColor:colors,pointRadius:5,fill:false},
floor!==null?{label:'floor',data:lbl.map(_=>floor),borderColor:'#9e9e9e',borderDash:[5,5],pointRadius:0,fill:false}:null
].filter(Boolean)},options:{responsive:false,scales:{y:{title:{display:true,text:'wilson_lower'}}}}});
</script></div>

<div class=section><h2>Per-epoch training loss
{% if not epoch_series %} <span class=small>(no epoch events recorded)</span>{% endif %}
</h2>
<canvas id=trainloss></canvas>
<canvas id=klloss></canvas>
<script>
const series={{ epoch_series|tojson }};
const palette=['#2196f3','#4caf50','#f44336','#ffc107','#9e9e9e','#9c27b0','#00bcd4','#ff9800'];
function mk(canvasId,key){
  const ds=[];let i=0;let maxEp=0;
  for(const [it,arr] of Object.entries(series)){
    ds.push({label:'iter '+it,data:arr.map(r=>r[key]),borderColor:palette[i%palette.length],fill:false,pointRadius:2});
    if(arr.length>maxEp)maxEp=arr.length;i++;}
  const labels=Array.from({length:maxEp},(_,k)=>k+1);
  new Chart(document.getElementById(canvasId),{type:'line',data:{labels,datasets:ds},
    options:{responsive:false,scales:{y:{title:{display:true,text:key}}}}});
}
mk('trainloss','train_loss');mk('klloss','kl_loss');
</script></div>

<div class=section><h2>Decision audit</h2>
<table><tr><th>iter</th><th>decision</th><th>wilson_lower</th><th>reason</th>
<th>consec fails</th><th>matchup violations</th><th>cycling alarm</th><th>halted</th></tr>
{% for it in iterations %}<tr>
<td>{{ it.iteration }}</td>
<td>{% if it.promoted %}<span class="pill pill-promoted">promoted</span>
{% else %}<span class="pill pill-rejected">rejected</span>{% endif %}</td>
<td>{% if it.wilson_lower is not none %}{{ '%.4f'|format(it.wilson_lower) }}{% else %}-{% endif %}</td>
<td class=small>{{ it.decision_reason or '-' }}</td>
<td>{{ it.consecutive_failures }}</td>
<td>{{ (it.matchup_floor_violations or [])|length }}</td>
<td class=small>{{ it.cycling_alarm or '-' }}</td>
<td>{% if it.halted %}yes: {{ it.halt_reason }}{% else %}-{% endif %}</td>
</tr>{% endfor %}</table>
</div>

{% if pool_rows %}<div class=section><h2>Pool eval heatmap</h2>
<table><tr><th>iter \\ opp</th>
{% for c in pool_cols %}<th>{{ c }}</th>{% endfor %}</tr>
{% for r in pool_rows %}<tr><td>{{ r.iteration }}</td>
{% for c in pool_cols %}{% set v=r.cells.get(c) %}
<td class=heat style="background:{% if v is none %}#eee{% elif v<0.3 %}#f44336{% elif v<0.5 %}#ffc107{% else %}#4caf50{% endif %};color:{% if v is none %}#999{% else %}#fff{% endif %}">
{% if v is not none %}{{ '%.2f'|format(v) }}{% else %}-{% endif %}</td>
{% endfor %}</tr>{% endfor %}</table>
</div>{% endif %}

<div class=section><h2>Recent events (last 20)</h2>
<table><tr><th>+t (s)</th><th>iter</th><th>stage</th><th>type</th><th>data</th></tr>
{% for e in tail %}<tr>
<td class=mono>{{ '%.2f'|format(e.dt) }}</td>
<td>{{ e.iteration }}</td><td>{{ e.stage }}</td><td>{{ e.event_type }}</td>
<td class=mono>{{ e.data }}</td>
</tr>{% endfor %}</table></div>

</body></html>
""".replace("{{css}}", CSS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()
    app = create_app(Path(args.runs_dir))
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
