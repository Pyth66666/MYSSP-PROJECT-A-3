"""Local web server for the 3D physics discovery sandbox.

    python server.py            then open http://127.0.0.1:8000

Pages
  GET  /                      3D sandbox (index.html)
  GET  /static/...            three.js and other assets

API (JSON)
  GET  /api/designs           the three sandbox designs
  GET  /api/agents            agents that can run right now
  GET  /api/results           last bench.py result table
  GET  /api/pipeline          last pipeline.py result table (50 CSV datasets)
  GET  /api/truth?law=        hidden law curve (evaluator view only)
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
import grader
from laws import DEFAULT_LAW, LAWS
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


def formula_curve(answer, n=121):
    """Sample the submitted formula so the browser can draw it."""
    try:
        _, f = grader.compile_formula(answer)
    except Exception:  # noqa: BLE001
        return None
    xs, fs = [], []
    for i in range(n):
        x = -X_LIMIT + 2 * X_LIMIT * i / (n - 1)
        try:
            v = f(x)
        except Exception:  # noqa: BLE001
            v = None
        xs.append(x)
        fs.append(v)
    return {"x": xs, "F": fs}


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
                                    "laws": list(LAWS), "default_law": DEFAULT_LAW})
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
                    rows = static_dataset(lab.law, lab.seed)
                    out["prompt"] = agents.team_prompt(out["card"])
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
                return self._send(200, {"result": result, "curve": formula_curve(answer)})

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
                    "log": lab.log, "result": result, "curve": formula_curve(answer),
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
