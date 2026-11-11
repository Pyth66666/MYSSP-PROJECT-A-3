"""Discovery agents for the physics sandbox.

Every agent has one method:

    answer, notes = agent.run(lab)    # lab is a sandbox_env.LabSession

It may call lab.card() and lab.experiment(...) as often as the budget allows,
then returns its formula as text. The benchmark (bench.py) does the grading,
so an agent can never see the hidden law.

Agents included
---------------
FitAgent     - no AI. Plain curve fitting of F = -k|x|^p sign(x). A strong
               non-LLM baseline: if an LLM cannot beat this, it is not
               "discovering", only guessing.
HookeAgent   - no AI. Assumes the textbook law F = -kx and fits k. Shows what
               happens when a model trusts memory over data.
LLMAgent     - any OpenAI-compatible chat endpoint: Ollama or LM Studio on
               this laptop, or OpenAI / DeepSeek / Gemini APIs.
               Configured in models.json. Standard library only.
               Design S: one prompt (the team prompt) and one answer.
               Designs A-C: a multi-turn experiment loop with JSON actions.
CrewAIAgent  - the team's CrewAI framework (theorist + analyst agents), see
               agents_crewai.py. Design S only.
"""

import json
import math
import os
import time
import urllib.error
import urllib.request

import sandbox_env


# =========================================================== shared helpers
def _plan_positions(n):
    """Spread probe positions over both sides of the rail."""
    base = [3.0, -2.5, 2.0, -1.5, 1.0, -0.6, 2.6, -2.2, 1.4, -0.9, 0.4, -3.0]
    return base[:n]


# The team's prompt (physics law redesign.py / Discord), used for design S.
TEAM_PROMPT = """You are an expert physicist. Below is experimental data collected from an unknown 1D mechanical system.
The table contains position(x) in meters and measured restoring force (F) in Newtons.

{data_string}

Task:
1. Analyze the mathematical relationship between position (x) and force (f)
2. Deduce the exact functional form/formula for F(x)
3. State the derived formula clearly in LaTeX format
"""


# The same prompt for the team's other redesigned laws (kinetic energy, wave
# speed, electrostatics, Ohm's law): only the system and the variable names change.
GENERAL_PROMPT = """You are an expert physicist. Below is experimental data collected from an unknown 1D physical system.
{setup}
The table contains {x_name} ({x_sym}) in {x_unit} and measured {y_name} ({y_sym}) in {y_unit}.

{data_string}

Task:
1. Analyze the mathematical relationship between {x_name} ({x_sym}) and {y_name} ({y_sym})
2. Deduce the exact functional form/formula for {y_sym}({x_sym})
3. State the derived formula clearly in LaTeX format
"""


def format_table(data, variables=None):
    """Same layout as pandas DataFrame.to_string(index=False) in the team script."""
    if variables is None:
        lines = [f"{'position':>9} {'force':>9}"]
        for row in data:
            lines.append(f"{row['x']:>9.4f} {row['F']:>9.4f}")
        return "\n".join(lines)
    xh, yh = variables["x_name"].replace(" ", "_"), variables["y_name"].replace(" ", "_")
    w = max(10, len(xh), len(yh))
    lines = [f"{xh:>{w}} {yh:>{w}}"]
    for row in data:
        lines.append(f"{row['x']:>{w}.4f} {row['y']:>{w}.4f}")
    return "\n".join(lines)


def team_prompt(card):
    if card.get("system") == "general":
        v = card["variables"]
        return GENERAL_PROMPT.format(setup=card["setup"],
                                     data_string=format_table(card["data"], v), **{
                                         k: v[k] for k in ("x_name", "x_sym", "x_unit",
                                                           "y_name", "y_sym", "y_unit")})
    return TEAM_PROMPT.format(data_string=format_table(card["data"]))


def _plan_inputs(card, n):
    """General laws: n inputs spread over the allowed range."""
    lo, hi = card["variables"]["x_range"]
    return [round(lo + (hi - lo) * (i + 0.5) / n, 4) for i in range(n)]


def collect_xf(lab):
    """Spend the budget on experiments and return (x, F) pairs.

    Design S: use the given table. Otherwise uses force probes when available.
    With only 'release', estimates F = m*a from the second difference of the
    recorded positions.
    """
    card = lab.card()
    if "data" in card:
        key = "y" if card.get("system") == "general" else "F"
        return [(r["x"], r[key]) for r in card["data"]], ["used the given dataset"]
    tools = [list(t["action"])[0] for t in card["tools"]]
    pairs, notes = [], []
    budget = card["budget"]
    if "measure" in tools:
        for x in _plan_inputs(card, budget):
            obs = lab.experiment({"measure": {"x": x}})
            pairs.append((obs["x"], obs["y"]))
            notes.append(f"measure x={x:g} -> y={obs['y']:+.4f}")
    elif "measure_force" in tools:
        for x in _plan_positions(budget):
            obs = lab.experiment({"measure_force": {"x": x}})
            pairs.append((obs["x"], obs["F"]))
            notes.append(f"probe x={x:+.2f} -> F={obs['F']:+.3f}")
    else:
        mass = 1.0  # given on the task card
        for x0 in _plan_positions(budget):
            obs = lab.experiment({"release": {"x0": x0}})
            t, xs = obs["t"], obs["x"]
            dt = t[1] - t[0]
            got = 0
            for i in range(1, len(xs) - 1, 3):
                a = (xs[i + 1] - 2 * xs[i] + xs[i - 1]) / (dt * dt)
                if abs(xs[i]) > 0.15:
                    pairs.append((xs[i], mass * a))
                    got += 1
            notes.append(f"release x0={x0:+.2f} -> {got} (x, m*a) points")
    return pairs, notes


def fit_power_law(pairs):
    """Least squares on log|F| = log k + p log|x| (restoring force)."""
    pts = [(math.log(abs(x)), math.log(abs(f))) for x, f in pairs
           if abs(x) > 1e-6 and abs(f) > 1e-9 and (x * f) < 0]
    n = len(pts)
    if n < 2:
        raise ValueError("not enough usable data")
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    p = sxy / sxx
    k = math.exp(my - p * mx)
    return k, p


def fit_power_general(pairs):
    """General laws: y = c * x**p (all inputs positive) or c * |x|**p * sign(x)."""
    pts = [(x, y) for x, y in pairs if abs(x) > 1e-9 and abs(y) > 1e-12]
    if len(pts) < 2:
        raise ValueError("not enough usable data")
    lx = [math.log(abs(x)) for x, _ in pts]
    ly = [math.log(abs(y)) for _, y in pts]
    n = len(pts)
    mx, my = sum(lx) / n, sum(ly) / n
    sxx = sum((a - mx) ** 2 for a in lx)
    p = sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sxx
    k = math.exp(my - p * mx)
    if all(x > 0 for x, _ in pts):
        c = k if sum(y for _, y in pts) >= 0 else -k
        return f"y = {c:.4f}*x**{p:.4f}"
    c = k if sum(x * y for x, y in pts) >= 0 else -k
    return f"y = {c:.4f}*abs(x)**{p:.4f}*sign(x)"


def fit_linear(pairs):
    """Least squares through the origin for F = -k x."""
    sxx = sum(x * x for x, _ in pairs)
    sxf = sum(x * f for x, f in pairs)
    return -sxf / sxx


# ================================================================ agents
class Agent:
    name = "agent"
    kind = "baseline"
    designs = ("S", "A", "B", "C")   # sandbox designs this agent can run

    def run(self, lab):
        raise NotImplementedError


class FitAgent(Agent):
    name = "curve-fit baseline (no AI)"

    def run(self, lab):
        pairs, notes = collect_xf(lab)
        if lab.card().get("system") == "general":
            answer = fit_power_general(pairs)
            notes.append(f"log-log power fit: {answer}")
            return answer, notes
        k, p = fit_power_law(pairs)
        notes.append(f"log-log fit: k={k:.4f}, p={p:.4f}")
        return f"F(x) = -{k:.4f}*abs(x)**{p:.4f}*sign(x)", notes


class HookeAgent(Agent):
    name = "Hooke assumption (no AI)"

    def run(self, lab):
        pairs, notes = collect_xf(lab)
        if lab.card().get("system") == "general":   # the textbook-style linear guess
            c = -fit_linear(pairs)
            notes.append(f"assumed y = c*x, fitted c={c:.4f}")
            return f"y = {c:.4f}*x", notes
        k = fit_linear(pairs)
        notes.append(f"assumed F=-kx, fitted k={k:.4f}")
        return f"F(x) = -{k:.4f}*x", notes


# ------------------------------------------------------------------ LLM
SYSTEM_PROMPT = """You are a physicist running experiments in a simulated laboratory.
You must discover an unknown force law from data. Do not assume any textbook law is correct;
trust only the measurements.

Every reply must end with exactly ONE JSON object on its own line, either
  {"experiment": <one action from the task card>}
or, when you are confident,
  {"answer": "F(x) = <expression in x>"}
Use Python syntax in the answer, e.g. -2*x**3, abs(x), sign(x), sqrt(x).
You may think briefly before the JSON line. Each experiment uses 1 budget."""

RELEASE_STRIDE = 5  # send every 5th recorded point (every 0.1 s) to save tokens


def _compact_observation(obs):
    if "t" in obs:
        t = obs["t"][::RELEASE_STRIDE]
        x = obs["x"][::RELEASE_STRIDE]
        return {"x0": obs["x0"], "dt_s": round(t[1] - t[0], 4),
                "x": [round(v, 4) for v in x], "budget_left": obs["budget_left"]}
    return obs


def _last_json(text):
    """Find the last {...} in the text that parses as JSON."""
    end = text.rfind("}")
    while end != -1:
        depth = 0
        for start in range(end, -1, -1):
            if text[start] == "}":
                depth += 1
            elif text[start] == "{":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:end + 1])
                    except json.JSONDecodeError:
                        break
        end = text.rfind("}", 0, end)
    return None


def _all_json(text):
    """Every JSON object in the text, in the order written."""
    dec, out, i = json.JSONDecoder(), [], 0
    while True:
        i = text.find("{", i)
        if i == -1:
            return out
        try:
            obj, end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict):
            out.append(obj)
        i = end


def _pick_action(text):
    """Return (action, ignored). One action per reply: if a reply holds several,
    the first experiment is run and the rest are ignored, because anything
    written after it was written before seeing its result. Several answers and
    no experiment: the last answer counts (the model revised it)."""
    acts = [o for o in _all_json(text) if "experiment" in o or "answer" in o]
    if not acts:
        return {}, 0
    if len(acts) == 1:
        return acts[0], 0
    exps = [o for o in acts if "experiment" in o]
    if exps:
        return exps[0], len(acts) - 1
    return acts[-1], 0


def _normalise_action(msg):
    """Return (action, corrected). Accepts the flat form small models often write,
    {"experiment": "measure_force", "x": -2}, as {"measure_force": {"x": -2}}.
    It has only one possible meaning, so this fixes a typing slip, not physics."""
    act = msg.get("experiment")
    if isinstance(act, str):
        text = act.strip()
        # The action written as JSON text inside a string.
        if text.startswith("{"):
            try:
                inner = json.loads(text)
            except json.JSONDecodeError:
                inner = None
            if isinstance(inner, dict):
                return inner, True
        # The action written as words: "measure_force -1.5" or "release 2".
        parts = text.replace(":", " ").replace("=", " ").split()
        if len(parts) >= 2 and parts[0] in ("measure_force", "measure", "release"):
            try:
                value = float(parts[-1])
            except ValueError:
                value = None
            if value is not None:
                key = "x0" if parts[0] == "release" else "x"
                return {parts[0]: {key: value}}, True
        args = {k: v for k, v in msg.items() if k != "experiment"}
        for key in ("args", "arguments", "params", "parameters", "input"):
            if isinstance(args.get(key), dict):
                args = args[key]
                break
        return {act: args}, True
    return act, False


def _format_hint(lab):
    tools = getattr(lab, "tools", lab.design["tools"])
    if "measure" in tools:
        return '{"experiment": {"measure": {"x": 1.0}}}'
    if "measure_force" in tools:
        return '{"experiment": {"measure_force": {"x": 1.0}}}'
    return '{"experiment": {"release": {"x0": 1.0}}}'


class LLMAgent(Agent):
    kind = "llm"

    def __init__(self, name, base_url, model, api_key=None, temperature=0.2,
                 max_tokens=2048, timeout=240, extra_system=""):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or "not-needed"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.tokens = 0
        # Extra system text: the experiment objective and, in rsi.py, the
        # strategy notes the agent wrote for itself. Empty = original behaviour.
        self.extra_system = extra_system
        self.transcript = []          # full message list of the last run
        self._token_param = "max_tokens"
        self._send_temperature = True

    def _chat(self, messages):
        """One chat call. Adapts once to newer models that reject max_tokens or a
        custom temperature, and retries rate limits / server errors briefly."""
        for attempt in range(5):
            payload = {"model": self.model, "messages": messages,
                       self._token_param: self.max_tokens}
            if self._send_temperature:
                payload["temperature"] = self.temperature
            req = urllib.request.Request(
                self.base_url + "/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self.api_key}"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:400]
                low = detail.lower()
                if exc.code == 400 and "max_tokens" in low and self._token_param == "max_tokens":
                    self._token_param = "max_completion_tokens"
                    continue
                if exc.code == 400 and "temperature" in low and self._send_temperature:
                    self._send_temperature = False
                    continue
                if exc.code in (429, 500, 502, 503, 504) and attempt < 4:
                    time.sleep(3 * (attempt + 1))
                    continue
                raise RuntimeError(f"HTTP {exc.code} from {self.model}: {detail}") from None
            usage = data.get("usage") or {}
            self.tokens += int(usage.get("total_tokens") or 0)
            return data["choices"][0]["message"].get("content") or ""
        raise RuntimeError(f"{self.model}: too many retries")

    def run(self, lab):
        self.tokens = 0
        self.transcript = []
        if lab.design["id"] == "S":
            prompt = [{"role": "user", "content": team_prompt(lab.card())}]
            reply = self._chat(prompt)
            self.transcript = prompt + [{"role": "assistant", "content": reply}]
            return reply, ["one prompt, one answer (team prompt)", reply[-300:]]
        notes = []
        system = SYSTEM_PROMPT
        if lab.card().get("system") == "general":
            system = system.replace("an unknown force law", "an unknown physical law")
        system += ("\n\n" + self.extra_system if self.extra_system else "")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "Task card:\n" + json.dumps(lab.card(), indent=1)},
        ]
        for _turn in range(lab.design["budget"] + 4):
            reply = self._chat(messages)
            messages.append({"role": "assistant", "content": reply})
            msg, ignored = _pick_action(reply)
            if ignored:
                notes.append(f"one action per reply: {ignored} extra action(s) ignored")
            if "answer" in msg:
                notes.append("answer given")
                self.transcript = messages
                return str(msg["answer"]), notes
            if "experiment" in msg:
                action, corrected = _normalise_action(msg)
                if corrected:
                    notes.append("format corrected: flat action accepted")
                try:
                    obs = _compact_observation(lab.experiment(action))
                    notes.append(f"experiment {json.dumps(action)}")
                except sandbox_env.ExperimentError as exc:
                    obs = {"error": str(exc), "budget_left": lab.budget_left,
                           "correct_format": _format_hint(lab)}
                    notes.append(f"rejected: {exc}")
                if ignored:
                    obs = dict(obs, note="Only one action per reply. The first experiment "
                                         "was run; everything after it was ignored.")
                messages.append({"role": "user", "content": "observation: " + json.dumps(obs)})
                continue
            notes.append("reply had no valid JSON")
            messages.append({"role": "user", "content":
                             "Your reply had no valid JSON object. End with "
                             '{"experiment": ...} or {"answer": "F(x) = ..."}.'})
        # Out of turns: ask for a final answer once.
        messages.append({"role": "user", "content":
                         'No more experiments. Reply now with {"answer": "F(x) = ..."}.'})
        reply = self._chat(messages)
        messages.append({"role": "assistant", "content": reply})
        self.transcript = messages
        msg = _last_json(reply) or {}
        return str(msg.get("answer", reply)), notes + ["forced final answer"]


# ============================================================== registry
HERE = os.path.dirname(os.path.abspath(__file__))


def load_llm_agents(path=None, only_available=True):
    """Read models.json. Skip entries without a key or an unreachable server."""
    path = path or os.environ.get("SBX_MODELS") or os.path.join(HERE, "models.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)
    agents = []
    for e in entries:
        if not e.get("enabled", True):
            continue
        if e.get("type") == "crewai":
            try:
                from agents_crewai import CrewAIAgent
                agent = CrewAIAgent(e["name"], e["model"], e["base_url"])
            except ImportError:
                continue  # crewai not installed
            if only_available and not agent.available():
                continue
            agents.append(agent)
            continue
        key = os.environ.get(e["api_key_env"]) if e.get("api_key_env") else None
        if e.get("api_key_env") and not key:
            continue
        agent = LLMAgent(e["name"], e["base_url"], e["model"], api_key=key,
                         temperature=e.get("temperature", 0.2),
                         max_tokens=e.get("max_tokens", 2048))
        if only_available and not _reachable(agent):
            continue
        agents.append(agent)
    return agents


def _reachable(agent):
    """Server answers AND lists this model (so an Ollama model that was never
    pulled, or a placeholder model id, is skipped instead of failing later)."""
    try:
        req = urllib.request.Request(agent.base_url + "/models",
                                     headers={"Authorization": f"Bearer {agent.api_key}"})
        with urllib.request.urlopen(req, timeout=3) as r:
            data = json.loads(r.read().decode("utf-8") or "{}")
    except Exception:  # noqa: BLE001 - any failure means "not available now"
        return False
    if "data" not in data:
        return True   # server does not list models; try it anyway
    ids = [str(m.get("id", "")) for m in (data.get("data") or []) if isinstance(m, dict)]
    if not ids:
        return False  # server lists no models, e.g. Ollama with nothing pulled
    return any(i == agent.model or i.endswith("/" + agent.model) for i in ids)


def all_agents():
    return [FitAgent(), HookeAgent()] + load_llm_agents()


def timed_run(agent, lab):
    start = time.time()
    answer, notes = agent.run(lab)
    return answer, notes, round(time.time() - start, 2)
