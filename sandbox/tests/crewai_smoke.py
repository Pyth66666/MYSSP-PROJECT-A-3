"""Smoke test for the CrewAI adapter against a fake OpenAI-compatible server.

Checks the plumbing, not model quality:
  - the sandbox data reaches the CrewAI agents' prompts
  - the flow finishes and the evaluator can extract the final formula

Run (needs crewai installed):  python tests/crewai_smoke.py
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents_crewai import CrewAIAgent  # noqa: E402
from sandbox_env import LabSession  # noqa: E402

REQUESTS = []
REPLY = ("Thought: I now can give a great answer\n"
         "Final Answer: H1: F = -k|x|^n sign(x), log-log fit gives k = 5.0, n = 1.5.\n"
         "FINAL FORMULA: $$F(x) = -5|x|^{1.5}\\operatorname{sign}(x)$$\n"
         "BEST: H1 SCORE: 0.9")


class Fake(BaseHTTPRequestHandler):
    def do_GET(self):
        self._json({"data": [{"id": "fake"}], "models": [{"name": "fake:latest"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        REQUESTS.append(body)
        if body.get("stream"):
            chunk = {"id": "1", "object": "chat.completion.chunk", "created": 0, "model": "fake",
                     "choices": [{"index": 0, "delta": {"role": "assistant", "content": REPLY},
                                  "finish_reason": "stop"}]}
            data = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self._json({"id": "1", "object": "chat.completion", "created": 0, "model": "fake",
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": REPLY}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

    def _json(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        ag = CrewAIAgent("crewai-fake", "ollama/fake", f"http://127.0.0.1:{srv.server_port}")
        assert ag.available(), "fake server not seen as available"
        lab = LabSession("S", 0)
        answer, notes = ag.run(lab)
        result = lab.submit(answer)
        prompts = json.dumps(REQUESTS)
        print("LLM calls:", len(REQUESTS))
        print("data in prompts:", "-14.1421" in prompts)
        print("notes:", notes[0])
        print("result:", result["passed"], result["reason"], result["parsed"])
        assert "-14.1421" in prompts, "sandbox data did not reach the agents"
        assert result["passed"], "formula was not extracted"
        print("CREWAI SMOKE TEST OK")
    finally:
        srv.shutdown()
        srv.server_close()


if __name__ == "__main__":
    main()
