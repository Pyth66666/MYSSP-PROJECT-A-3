"""Shared physics engine for the self-evolving AI physics benchmark.

Design goals
------------
* One ground-truth law module that all sandbox designs share, so the agent is
  always tested against the *same* physics. Only the *visual / interaction*
  layer differs between designs.
* Deterministic: same parameters + same actions -> same trajectory, so scores
  are comparable across models.
* Headless-friendly: the server can simulate without a browser; the browser
  only renders what the server computed.

Laws implemented
----------------
1. Hooke's law (spring):        F = -k x          -> simple harmonic motion
2. Damped spring:                F = -k x - c v
3. Pendulum (small + full):      theta'' = -(g/L) sin(theta)
4. Inclined plane + spring:      a = g sin(alpha) - (k/m) s
"""

import math
import copy


class System:
    """Base class. Subclasses define `accel()` and `observe()`."""

    name = "system"
    state_names = ()
    param_names = ()

    def __init__(self, **params):
        self.p = params
        self.state = self.initial_state()

    def initial_state(self):
        raise NotImplementedError

    def accel(self):
        raise NotImplementedError

    def observe(self):
        """What the agent 'sees'. Plain dict of numbers."""
        raise NotImplementedError

    def apply_action(self, action):
        """Optional: the agent may change parameters / kick the system.

        action is a dict, e.g. {"set": {"k": 80}} or {"push": 1.5}.
        Default: ignore unknown actions.
        """
        if not action:
            return
        if "set" in action:
            for key, val in action["set"].items():
                if key in self.p:
                    self.p[key] = float(val)

    def step(self, dt):
        raise NotImplementedError

    def sample(self, duration=6.0, dt=0.01):
        sim = copy.deepcopy(self)
        t, out = 0.0, []
        n = int(duration / dt)
        for _ in range(n + 1):
            row = {"t": round(t, 4)}
            row.update(sim.observe())
            out.append(row)
            sim.step(dt)
            t += dt
        return out

    def info(self):
        """Derived quantities a good answer might mention (ground truth)."""
        return {}


# --------------------------------------------------------------------------
# Design A: vertical spring-mass (Hooke's law)
# --------------------------------------------------------------------------
class SpringMass(System):
    name = "spring_mass"
    state_names = ("x", "v")
    param_names = ("k", "mass", "damping")

    def __init__(self, k=40.0, mass=1.0, damping=0.0, x0=1.0, v0=0.0):
        self.p = {"k": float(k), "mass": float(mass), "damping": float(damping)}
        self.x0 = float(x0)
        self.v0 = float(v0)
        super().__init__(**self.p)

    def initial_state(self):
        return {"x": self.x0, "v": self.v0}

    @property
    def omega(self):
        return math.sqrt(self.p["k"] / self.p["mass"])

    @property
    def period(self):
        return 2 * math.pi / self.omega if self.omega else float("inf")

    def accel(self):
        k, m, c = self.p["k"], self.p["mass"], self.p["damping"]
        return (-k * self.state["x"] - c * self.state["v"]) / m

    def force(self):
        return -self.p["k"] * self.state["x"]

    def observe(self):
        return {
            "x": self.state["x"],
            "v": self.state["v"],
            "F": self.force(),
        }

    def apply_action(self, action):
        super().apply_action(action)
        if action and "push" in action:
            self.state["v"] += float(action["push"])

    def step(self, dt):
        a = self.accel()
        self.state["v"] += a * dt
        self.state["x"] += self.state["v"] * dt

    def energy(self):
        ke = 0.5 * self.p["mass"] * self.state["v"] ** 2
        pe = 0.5 * self.p["k"] * self.state["x"] ** 2
        return {"kinetic": ke, "potential": pe, "total": ke + pe}

    def info(self):
        return {"omega": self.omega, "period": self.period, "energy": self.energy()}


# --------------------------------------------------------------------------
# Design B: pendulum (nonlinear, small-angle approximation as a trap)
# --------------------------------------------------------------------------
class Pendulum(System):
    name = "pendulum"
    state_names = ("theta", "omega")
    param_names = ("length", "gravity", "damping")

    def __init__(self, length=1.0, gravity=9.81, damping=0.0, theta0=0.6, omega0=0.0):
        self.p = {"length": float(length), "gravity": float(gravity),
                  "damping": float(damping)}
        self.theta0 = float(theta0)
        self.omega0 = float(omega0)
        super().__init__(**self.p)

    def initial_state(self):
        return {"theta": self.theta0, "omega": self.omega0}

    @property
    def small_angle_period(self):
        return 2 * math.pi * math.sqrt(self.p["length"] / self.p["gravity"])

    def accel(self):
        g, L, c = self.p["gravity"], self.p["length"], self.p["damping"]
        th, w = self.state["theta"], self.state["omega"]
        return -(g / L) * math.sin(th) - c * w

    def observe(self):
        g, L = self.p["gravity"], self.p["length"]
        th = self.state["theta"]
        return {
            "theta": th,
            "omega": self.state["omega"],
            "x": L * math.sin(th),
            "y": -L * math.cos(th),
            "tension": self.p.get("mass", 1.0) * (g * math.cos(th) + L * self.state["omega"] ** 2),
        }

    def step(self, dt):
        a = self.accel()
        self.state["omega"] += a * dt
        self.state["theta"] += self.state["omega"] * dt

    def info(self):
        return {"small_angle_period": self.small_angle_period}


# --------------------------------------------------------------------------
# Design C: spring on an inclined plane (3D scene uses same 1D law)
# --------------------------------------------------------------------------
class InclineSpring(System):
    name = "incline_spring"
    state_names = ("s", "v")
    param_names = ("k", "mass", "angle", "gravity", "friction")

    def __init__(self, k=30.0, mass=1.0, angle=20.0, gravity=9.81,
                 friction=0.0, s0=0.5):
        self.p = {"k": float(k), "mass": float(mass), "angle": float(angle),
                  "gravity": float(gravity), "friction": float(friction)}
        self.s0 = float(s0)
        super().__init__(**self.p)

    def initial_state(self):
        return {"s": self.s0, "v": 0.0}

    def accel(self):
        k, m = self.p["k"], self.p["mass"]
        g = self.p["gravity"]
        a = math.radians(self.p["angle"])
        fric = self.p["friction"] * g * math.cos(a)
        s, v = self.state["s"], self.state["v"]
        drive = g * math.sin(a) - (k / m) * s
        fric_term = -math.copysign(fric, v) if abs(v) > 1e-6 else 0.0
        return drive + fric_term

    def observe(self):
        return {"s": self.state["s"], "v": self.state["v"], "a_along": self.accel()}

    def step(self, dt):
        a = self.accel()
        self.state["v"] += a * dt
        self.state["s"] += self.state["v"] * dt

    def info(self):
        a = math.radians(self.p["angle"])
        return {"equilibrium_s": self.p["mass"] * self.p["gravity"] * math.sin(a) / self.p["k"]}


SYSTEMS = {
    "spring_mass": SpringMass,
    "pendulum": Pendulum,
    "incline_spring": InclineSpring,
}


def build(name, **params):
    return SYSTEMS[name](**params)