"""Local web server for the 3D physics discovery sandbox.

    python server.py            then open http://127.0.0.1:8000

Pages
  GET  /                      3D sandbox (index.html)
  GET  /static/...            three.js and other assets

API (JSON)
  GET  /api/designs           the sandbox designs and the five team laws
  GET  /api/agents            agents that can run right now
  GET  /api/results           last bench.py result table
  GET  /api/pipeline          last pipeline.py result table (50 CSV datasets)
  GET  /api/truth?law=        hidden law curve (evaluator view only)
  GET  /api/leaderboard       every model, latest 10 autonomous runs per design
  GET  /api/saved_runs        saved autonomous runs that can be replayed
  GET  /api/saved_run?file=   one saved run log, for replay in 3D
  POST /api/session           {design, seed, law}       -> {sid, card}
  POST /api/experiment        {sid, action}             -> observation
  POST /api/submit            {sid, answer}             -> score + formula curve
  POST /api/run               {design, seed, agent, law}-> full agent episode

The physics, the hidden law and the grading stay in Python. The browser only
draws what the server returns.
"""

import json
import os
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import agent as agents
import compare
import grader
from laws import DEFAULT_LAW, LAWS, TEAM_LAWS
from sandbox_env import (DESIGNS, X_LIMIT, ExperimentError, LabSession, static_dataset,
                         static_x0, truth_curve)

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
SESSIONS = {}
LOCK = threading.Lock()
_AGENTS = {"list": None}

TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript",
         ".css": "text/css", ".json": "application/json", ".txt": "text/plain"}


def agent_pool(refresh=False):
    if _AGENTS["list"] is None or refresh:
        _AGENTS["list"] = agents.all_agents()
    return _AGENTS["list"]


def formula_curve(answer, n=121, law=None):
    """Sample the submitted formula so the browser can draw it."""
    try:
        _, f = grader.compile_formula(answer, law)
    except Exception:  # noqa: BLE001
        return None
    xs, fs = [], []
    for i in range(n):
        lo, hi = (-X_LIMIT, X_LIMIT) if law is None or law.is_spring else law.x_range
        x = lo + (hi - lo) * i / (n - 1)
        try:
            v = f(x)
        except Exception:  # noqa: BLE001
            v = None
        xs.append(x)
        fs.append(v)
    return {"x": xs, "F": fs}


LAW_TITLES = {"power_spring": "1 \u00b7 Spring", "kinetic_energy": "2 \u00b7 Kinetic energy",
              "wave_speed": "3 \u00b7 Wave speed", "screened_coulomb": "4 \u00b7 Screened Coulomb",
              "saturating_ohm": "5 \u00b7 Saturating Ohm"}


def law_info(name):
    """What the page needs to draw a law: names, units, range, and a scale for the
    3D column (the largest |y| over the range). Agents never see this."""
    L = LAWS[name]
    lo, hi = L.x_range
    ys = [L.force(lo + (hi - lo) * i / 199) for i in range(200)]
    return {"name": name, "title": LAW_TITLES.get(name, name), "kind": L.kind,
            "x_name": L.x_name, "x_sym": L.x_sym, "x_unit": L.x_unit,
            "y_name": L.y_name, "y_sym": L.y_sym, "y_unit": L.y_unit,
            "x_range": [lo, hi], "y_abs_max": max(abs(y) for y in ys)}


def public_designs():
    keep = ("id", "title", "summary", "strengths", "limits", "tools", "budget",
            "force_noise", "position_noise", "tolerance")
    return [{k: d[k] for k in keep} for d in DESIGNS.values()]


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path):
        real = os.path.realpath(path)
        if not (real.startswith(os.path.realpath(STATIC) + os.sep)
                or real == os.path.join(HERE, "index.html")):
            return self._send(403, {"error": "forbidden"})
        try:
            with open(real, "rb") as f:
                body = f.read()
        except FileNotFoundError:
            return self._send(404, {"error": "not found"})
        self._send(200, body, TYPES.get(os.path.splitext(real)[1], "application/octet-stream"))

    # ------------------------------------------------------------------ GET
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            return self._file(os.path.join(HERE, "index.html"))
        if u.path.startswith("/static/"):
            return self._file(os.path.join(STATIC, u.path[len("/static/"):]))
        if u.path == "/api/designs":
            return self._send(200, {"designs": public_designs(),
                                    "laws": list(LAWS), "default_law": DEFAULT_LAW,
                                    "team_laws": [law_info(n) for n in TEAM_LAWS]})
        if u.path == "/api/agents":
            pool = agent_pool(refresh="refresh" in q)
            return self._send(200, {"agents": [{"name": a.name, "kind": a.kind} for a in pool]})
        if u.path == "/api/results":
            path = os.path.join(HERE, "results", "latest.json")
            if not os.path.exists(path):
                return self._send(200, {})
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return self._send(200, {k: data[k] for k in ("timestamp", "law", "seeds", "summary")})
        if u.path == "/api/pipeline":
            path = os.path.join(HERE, "results", "pipeline_latest.json")
            if not os.path.exists(path):
                return self._send(200, {})
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return self._send(200, {k: data[k] for k in ("timestamp", "law", "summary")})
        if u.path == "/api/leaderboard":
            law = q.get("law", [DEFAULT_LAW])[0]
            return self._send(200, {"law": law,
                                    "rows": compare.leaderboard(os.path.join(HERE, "results"), law=law)})
        if u.path == "/api/saved_runs":
            law = q.get("law", [None])[0] or None
            return self._send(200, {"runs": compare.saved_runs(os.path.join(HERE, "results"), law=law)})
        if u.path == "/api/saved_run":
            name = os.path.basename(q.get("file", [""])[0].replace("\\", "/"))
            path = os.path.join(HERE, "results", "runs", name)
            if not name.endswith(".json") or not os.path.isfile(path):
                return self._send(404, {"error": "saved run not found"})
            with open(path, encoding="utf-8") as f:
                log = json.load(f)
            return self._send(200, {k: log.get(k) for k in
                                    ("agent", "design", "seed", "law", "timestamp", "experiments",
                                     "answer", "result", "seconds", "tokens")}
                              | {"curve": formula_curve(log.get("answer") or "",
                                                        law=LAWS.get(log.get("law")))})
        if u.path == "/api/truth":
            law = q.get("law", [DEFAULT_LAW])[0]
            if law not in LAWS:
                return self._send(400, {"error": "unknown law"})
            return self._send(200, truth_curve(law))
        return self._send(404, {"error": "not found"})

    # ----------------------------------------------------------------- POST
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            p = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {"error": "bad json"})
        u = urlparse(self.path)
        try:
            if u.path == "/api/session":
                lab = LabSession(p.get("design", "A"), int(p.get("seed", 0)),
                                 p.get("law") or DEFAULT_LAW)
                sid = secrets.token_hex(6)
                with LOCK:
                    SESSIONS[sid] = lab
                out = {"sid": sid, "card": lab.card()}
                if lab.design["id"] == "S":
                    out["prompt"] = agents.team_prompt(out["card"])
                    if lab.law.is_spring:
                        rows = static_dataset(lab.law, lab.seed)
                        out["x0"] = static_x0(lab.seed)
                        out["trajectory"] = {"t": [r["time"] for r in rows],
                                             "x": [r["position"] for r in rows]}
                return self._send(200, out)

            if u.path == "/api/experiment":
                lab = self._lab(p)
                return self._send(200, lab.experiment(p.get("action")))

            if u.path == "/api/submit":
                lab = self._lab(p)
                answer = str(p.get("answer", ""))
                result = lab.submit(answer)
                return self._send(200, {"result": result,
                                        "curve": formula_curve(answer, law=lab.law)})

            if u.path == "/api/run":
                name = p.get("agent")
                ag = next((a for a in agent_pool() if a.name == name), None)
                if ag is None:
                    return self._send(400, {"error": f"agent '{name}' not available"})
                if p.get("design", "A") not in ag.designs:
                    return self._send(400, {"error": f"{name} only runs designs {', '.join(ag.designs)}"})
                lab = LabSession(p.get("design", "A"), int(p.get("seed", 0)),
                                 p.get("law") or DEFAULT_LAW)
                try:
                    answer, notes, seconds = agents.timed_run(ag, lab)
                except Exception as exc:  # noqa: BLE001
                    answer, notes, seconds = "", [f"agent error: {exc}"], 0
                result = lab.submit(answer)
                return self._send(200, {
                    "agent": ag.name, "answer": answer, "notes": notes,
                    "seconds": seconds, "tokens": getattr(ag, "tokens", 0),
                    "log": lab.log, "result": result,
                    "curve": formula_curve(answer, law=lab.law),
                })
        except ExperimentError as exc:
            return self._send(400, {"error": str(exc)})
        except KeyError as exc:
            return self._send(400, {"error": f"unknown {exc}"})
        return self._send(404, {"error": "not found"})

    def _lab(self, p):
        with LOCK:
            lab = SESSIONS.get(p.get("sid"))
        if lab is None:
            raise ExperimentError("session expired, start a new one")
        return lab

    def log_message(self, *args):
        pass


def main(port=8000):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Physics discovery sandbox -> http://127.0.0.1:{port}")
    print("Agents available: " + ", ".join(a.name for a in agent_pool()))
    print("Press Ctrl+C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main(int(os.environ.get("SBX_PORT", "8000")))
