"""Adapter: runs the team's CrewAI framework as a benchmark agent.

The framework itself is framework/physics_discovery_flow.py (written by
Yzzzz, kept unchanged). It is a CrewAI Flow with two agents:

    theorist  -> proposes 3-5 candidate laws
    analyst   -> designs tests, scores them 0-1 (SymPy + NumPy tools)
    router    -> score >= 0.8: write report, else refine and retest

What this adapter adds, without editing the framework:
  1. Puts the sandbox data into the flow. The original call only passed the
     sentence "Find the law related to Hooke's Law ...", so the agents had no
     data and the word "Hooke" hinted at the answer.
  2. Asks for a last line "FINAL FORMULA: $$F(x) = ...$$" so the evaluator
     can extract the answer from the long report.
  3. Lets models.json choose the model (it swaps the module's local_llm).

Needs:  pip install crewai. The model is either a local Ollama model
("ollama/<name>") or an OpenAI-compatible API model ("openai/<name>" with
api_key_env naming the environment variable that holds the key).
Supports design S (static dataset) only, because the flow has no way to
call experiments.
"""

import importlib.util
import json
import os
import urllib.request

from agent import Agent, team_prompt

HERE = os.path.dirname(os.path.abspath(__file__))
FLOW_FILE = os.path.join(HERE, "framework", "physics_discovery_flow.py")

FINAL_LINE = ("\n\nIMPORTANT: end the final report with exactly one line of the form\n"
              "FINAL FORMULA: $$F(x) = <your formula in x>$$\n"
              "using only numbers and x (no unknown constants).")

import crewai  # noqa: E402,F401 - ImportError here means "not installed", agent is skipped


def _load_flow_module():
    spec = importlib.util.spec_from_file_location("physics_discovery_flow", FLOW_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _usage(llm):
    """Total tokens and number of model calls made through this LLM object."""
    try:
        u = llm.get_token_usage_summary()
        return int(getattr(u, "total_tokens", 0) or 0), int(getattr(u, "successful_requests", 0) or 0)
    except Exception:  # noqa: BLE001
        return 0, 0


class CrewAIAgent(Agent):
    kind = "crewai"
    designs = ("S",)

    def __init__(self, name, model, base_url, api_key_env=None):
        self.name = name
        self.model = model          # e.g. "ollama/qwen2.5-coder:7b" or "openai/gpt-6-luna"
        self.base_url = base_url    # e.g. "http://localhost:11434"
        self.api_key_env = api_key_env
        self.tokens = 0
        self.calls = 0

    def available(self):
        if self.model.startswith("ollama/"):
            try:
                with urllib.request.urlopen(self.base_url.rstrip("/") + "/api/tags",
                                            timeout=3) as r:
                    tags = json.loads(r.read().decode("utf-8"))
                names = {m.get("name") for m in tags.get("models", [])}
                want = self.model.split("/", 1)[1]
                return want in names or f"{want}:latest" in names
            except Exception:  # noqa: BLE001
                return False
        if self.api_key_env:
            return bool(os.environ.get(self.api_key_env))
        return True

    def run(self, lab):
        if lab.design["id"] not in self.designs:
            raise ValueError("the CrewAI framework only supports the static dataset (design S)")
        from crewai import LLM
        mod = _load_flow_module()
        extra = {"api_key": os.environ[self.api_key_env]} if self.api_key_env else {}
        llm = LLM(model=self.model, base_url=self.base_url, **extra)
        mod.local_llm = llm
        phenomenon = team_prompt(lab.card()) + FINAL_LINE
        flow = mod.PhysicsLawDiscovery()
        report = flow.kickoff(inputs={"phenomenon": phenomenon})
        report = str(report)
        self.tokens, self.calls = _usage(llm)
        branch = "supported" if flow.state.refined_results == "" else "refined"
        return report, [f"CrewAI flow finished ({branch} branch)", report[-300:]]
