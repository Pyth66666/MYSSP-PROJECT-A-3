"""The physics sandbox: three experiment designs for discovering one hidden law.

Every design hides the same law (laws.py). What changes is HOW the agent is
allowed to look at it:

  S  Static dataset    - the team pipeline: the sandbox saves one simulated run
                         as CSV, the agent gets the (x, F) table in a prompt and
                         answers once. No experiments.
  A  Force probe       - hold the block at a position x, read the force.
  B  Release and watch - release the block from x0, see only its motion x(t).
  C  Noisy lab         - both tools, but the sensors are noisy and the budget
                         is tight. Closest to a real laboratory.

API (the same for every agent, model, CLI or web page):

    lab = LabSession(design="A", seed=0)
    lab.card()                         -> public task description
    lab.experiment({...})              -> one observation (uses 1 budget)
    lab.submit("F(x) = ...")           -> score from the hidden-law grader

The session keeps a log of every experiment, so any run can be replayed and
checked later.
"""

import math
import random

from laws import get_law
import grader

DESIGNS = {
    "S": {
        "id": "S",
        "title": "Static dataset",
        "tools": [],
        "budget": 0,
        "force_noise": 0.0,
        "position_noise": 0.0,
        "tolerance": 0.01,
        "summary": "The team pipeline. The sandbox simulates one release (dt = 0.1 s, 10 s) "
                   "and saves it as CSV. The agent gets 20 (x, F) rows in a prompt and must "
                   "state F(x). No experiments.",
        "strengths": [
            "Matches the agreed pipeline: CSV -> prompt -> agent -> SymPy evaluator",
            "Same input for every model, so results compare directly",
        ],
        "limits": [
            "The agent cannot choose or repeat measurements",
            "Tests formula fitting more than experimental discovery",
        ],
    },
    "A": {
        "id": "A",
        "title": "Force probe",
        "tools": ["measure_force"],
        "budget": 12,
        "force_noise": 0.0,
        "position_noise": 0.0,
        "tolerance": 0.01,
        "summary": "Hold the block at any position x and read the spring force F. "
                   "The most direct view of F(x).",
        "strengths": [
            "Direct (x, F) data, easiest to fit and to score",
            "Shows quickly whether a model can find a non-linear power law",
        ],
        "limits": [
            "Too easy for strong models: almost a curve-fitting exercise",
            "Not how most real experiments work (no motion, no time)",
        ],
    },
    "B": {
        "id": "B",
        "title": "Release and watch",
        "tools": ["release"],
        "budget": 6,
        "force_noise": 0.0,
        "position_noise": 0.0,
        "tolerance": 0.02,
        "summary": "Pull the block to x0 and let go. You only see position over time, "
                   "never the force. The law must be inferred from the motion.",
        "strengths": [
            "Real discovery step: force is never measured directly",
            "Classic trap: the period changes with amplitude, unlike Hooke's law",
        ],
        "limits": [
            "Needs numerical differentiation or a period analysis",
            "Weaker models may give up or guess Hooke's law",
        ],
    },
    "C": {
        "id": "C",
        "title": "Noisy lab",
        "tools": ["measure_force", "release"],
        "budget": 10,
        "force_noise": 0.03,
        "position_noise": 0.005,
        "tolerance": 0.03,
        "summary": "Both tools, but sensors are noisy (3% force noise, 5 mm position "
                   "noise) and the budget is small. Tests planning and robustness.",
        "strengths": [
            "Closest to a real laboratory",
            "Rewards good experiment choice, not only good fitting",
        ],
        "limits": [
            "Scores vary with the noise seed, so it needs several runs",
            "Harder to tell model weakness apart from bad luck",
        ],
    },
}

X_LIMIT = 3.0          # metres, the rail length on each side of x = 0
OBS_DT = 0.02          # seconds between recorded positions in a release
OBS_DURATION = 4.0     # seconds recorded per release
SIM_DT = 0.0005        # internal integration step (velocity Verlet)


STATIC_DT = 0.1        # the team script's time step
STATIC_TIME = 10.0     # the team script's total time
STATIC_ROWS = 20       # rows of (x, F) put into the prompt


def static_x0(seed):
    """Release position for dataset number `seed`. 50 different datasets:
    |x0| from 0.5 m to 3.0 m in 25 steps, each on both sides of 0.
    Seed 0 gives x0 = 2.0 m, the same as the team script."""
    if seed % 50 == 0:
        return 2.0
    i = seed % 50
    mag = 0.5 + 2.5 * (i // 2) / 24
    return round(mag if i % 2 else -mag, 4)


def static_dataset(law, seed):
    """One simulated run, exactly like the team script (physics law redesign.py):
    semi-implicit Euler, v = v + a*dt then x = x + v*dt, starting at rest."""
    x, v = static_x0(seed), 0.0
    rows = []
    for step in range(int(round(STATIC_TIME / STATIC_DT))):
        t = step * STATIC_DT
        f = law.force(x)
        a = f / law.mass
        rows.append({"time": round(t, 2), "position": round(x, 4), "velocity": round(v, 4),
                     "acceleration": round(a, 4), "force": round(f, 4)})
        v = v + a * STATIC_DT
        x = x + v * STATIC_DT
    return rows


def sample_rows(rows, n=STATIC_ROWS):
    """Evenly spaced rows across the whole run (the team script took every 10th row)."""
    if len(rows) <= n:
        return list(rows)
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)]


class ExperimentError(ValueError):
    """Raised for invalid actions or when the budget is used up."""


def simulate_release(law, x0, duration=OBS_DURATION, obs_dt=OBS_DT, sim_dt=SIM_DT):
    """True motion of the block released at rest from x0. Velocity Verlet."""
    m = law.mass
    x, v = float(x0), 0.0
    a = law.force(x) / m
    every = int(round(obs_dt / sim_dt))
    n_obs = int(round(duration / obs_dt))
    ts, xs, vs = [0.0], [x], [0.0]
    for i in range(1, n_obs * every + 1):
        x += v * sim_dt + 0.5 * a * sim_dt * sim_dt
        a_new = law.force(x) / m
        v += 0.5 * (a + a_new) * sim_dt
        a = a_new
        if i % every == 0:
            ts.append(round(i * sim_dt, 6))
            xs.append(x)
            vs.append(v)
    return ts, xs, vs


class LabSession:
    def __init__(self, design="A", seed=0, law=None, dataset=None):
        """dataset: optional rows loaded from a CSV (design S). Default: simulate."""
        self.dataset = dataset
        if design not in DESIGNS:
            raise ExperimentError(f"unknown design '{design}', use one of {list(DESIGNS)}")
        self.design = DESIGNS[design]
        self.seed = int(seed)
        self.law = get_law(law)
        self.log = []          # [{"action": ..., "observation": ...}]
        self.answer = None
        self.result = None

    # ---------------------------------------------------------------- public
    @property
    def budget_left(self):
        return self.design["budget"] - len(self.log)

    def card(self):
        """Everything an agent is allowed to know before experimenting."""
        d = self.design
        tools = []
        if "measure_force" in d["tools"]:
            tools.append({
                "action": {"measure_force": {"x": "<number between -3 and 3>"}},
                "returns": "the spring force F in newtons with the block held at x metres",
            })
        if "release" in d["tools"]:
            tools.append({
                "action": {"release": {"x0": "<number between -3 and 3, not 0>"}},
                "returns": f"block positions x(t) every {OBS_DT} s for {OBS_DURATION} s "
                           "after releasing it at rest from x0 (force is not shown)",
            })
        if d["id"] == "S":
            rows = self.dataset if self.dataset is not None else static_dataset(self.law, self.seed)
            data = [{"x": float(r["position"]), "F": float(r["force"])} for r in sample_rows(rows)]
            return {
                "design": "S",
                "title": d["title"],
                "setup": "Experimental data collected from an unknown 1D mechanical system.",
                "goal": "Find the exact formula for the restoring force F(x) in newtons, x in metres.",
                "tools": [],
                "budget": 0,
                "budget_left": 0,
                "data": data,
                "submit": "One expression for F in terms of x, in LaTeX or Python.",
                "pass_rule": f"exact SymPy match, or normalised RMSE <= {d['tolerance']:.0%} "
                             "on unseen test positions (both signs of x)",
            }
        return {
            "design": d["id"],
            "title": d["title"],
            "setup": ("A block of mass 1 kg slides without friction on a horizontal rail. "
                      "A spring connects it to a wall. x = 0 is the rest position. "
                      "The spring does NOT necessarily follow Hooke's law."),
            "goal": "Find the exact formula for the spring force F(x) in newtons, x in metres.",
            "tools": tools,
            "budget": d["budget"],
            "budget_left": self.budget_left,
            "noise": {"force_relative": d["force_noise"], "position_m": d["position_noise"]},
            "submit": "One expression for F in terms of x, e.g. 'F(x) = -2*x**3' "
                      "or LaTeX '$$F(x) = -2x^{3}$$'. abs(), sign(), sqrt() are allowed.",
            "pass_rule": f"normalised RMSE against the hidden law <= {d['tolerance']:.0%} "
                         "on unseen test positions",
        }

    def experiment(self, action):
        if self.result is not None:
            raise ExperimentError("answer already submitted, start a new session")
        if self.budget_left <= 0:
            raise ExperimentError("experiment budget used up, submit your answer")
        if not isinstance(action, dict) or len(action) != 1:
            raise ExperimentError("action must be one JSON object like "
                                  '{"measure_force": {"x": 1.0}}')
        kind, args = next(iter(action.items()))
        if kind not in self.design["tools"]:
            raise ExperimentError(f"'{kind}' is not available in design "
                                  f"{self.design['id']}, tools: {self.design['tools']}")
        rng = random.Random(f"{self.seed}:{len(self.log)}")
        if kind == "measure_force":
            obs = self._measure_force(args, rng)
        else:
            obs = self._release(args, rng)
        self.log.append({"action": {kind: args}, "observation": obs})
        obs = dict(obs, budget_left=self.budget_left)
        return obs

    def submit(self, answer):
        if self.result is not None:
            return self.result
        self.answer = str(answer)
        self.result = grader.grade(self.answer, self.law,
                                   tolerance=self.design["tolerance"])
        self.result["experiments_used"] = len(self.log)
        return self.result

    # ------------------------------------------------------------ internals
    def _position(self, args, key):
        try:
            x = float(args[key])
        except (KeyError, TypeError, ValueError):
            raise ExperimentError(f"missing number '{key}'")
        if not math.isfinite(x) or abs(x) > X_LIMIT:
            raise ExperimentError(f"{key} must be between -{X_LIMIT} and {X_LIMIT}")
        return x

    def _measure_force(self, args, rng):
        x = self._position(args, "x")
        f = self.law.force(x)
        if self.design["force_noise"]:
            f += rng.gauss(0.0, self.design["force_noise"] * abs(f) + 0.02)
        return {"x": round(x, 6), "F": round(f, 6)}

    def _release(self, args, rng):
        x0 = self._position(args, "x0")
        if abs(x0) < 0.05:
            raise ExperimentError("x0 must be at least 0.05 m away from 0")
        ts, xs, _ = simulate_release(self.law, x0)
        noise = self.design["position_noise"]
        if noise:
            xs = [x + rng.gauss(0.0, noise) for x in xs]
        return {"x0": x0, "t": ts, "x": [round(x, 6) for x in xs]}

    # ------------------------------------------------------ save / replay
    def to_dict(self):
        return {
            "design": self.design["id"],
            "seed": self.seed,
            "law": self.law.name,
            "actions": [entry["action"] for entry in self.log],
            "answer": self.answer,
        }

    @classmethod
    def from_dict(cls, data):
        """Rebuild a session by replaying its actions. Deterministic."""
        lab = cls(design=data["design"], seed=data["seed"], law=data["law"])
        for action in data["actions"]:
            lab.experiment(action)
        if data.get("answer") is not None:
            lab.submit(data["answer"])
        return lab


def truth_curve(law_name=None, n=121):
    """Evaluator-only: the hidden F(x) curve, for the 'reveal' view after grading."""
    law = get_law(law_name)
    xs = [-X_LIMIT + 2 * X_LIMIT * i / (n - 1) for i in range(n)]
    return {"formula": law.formula, "x": xs, "F": [law.force(x) for x in xs]}
