"""Hidden physical laws used by the sandbox.

RULE: only the sandbox (sandbox_env.py) and the evaluator (grader.py) may
import this file. Agents never see it. They learn about the law only through
the experiments the sandbox lets them run.

To add one of the team's new laws, add one Law(...) entry to LAWS below.

Two kinds of law:
  spring   a block on a spring, F(x) in newtons. All four designs work
           (S, A, B, C), and the 3D page shows it.
  general  any other one-input law y(x), for example the team's redesigned
           kinetic energy, wave speed, electrostatics and Ohm's laws. Designs
           S (fixed table), A (measure y at any x) and C (noisy measurements)
           work. Design B needs a moving block, so it is spring only.

Every law is written with ONE input. Values that the team scripts keep fixed
(the mass, the wavelength, the two charges) are fixed here too and are told
to the agent in the task card, so the agent can only discover the shape in x.
"""

import math


def sign(v):
    return (v > 0) - (v < 0)


class Law:
    def __init__(self, name, force, formula, sympy_expr, mass=1.0, note="", kind="spring",
                 x_name="position", x_sym="x", x_unit="m", x_col="position",
                 y_name="force", y_sym="F", y_unit="N", y_col="force",
                 x_range=(-3.0, 3.0), setup="", source="", given=None):
        self.name = name
        self.force = force      # python function y(x); for springs the force F(x) in N
        self.formula = formula  # human-readable truth, shown only after grading
        self.sympy_expr = sympy_expr  # same law as a SymPy string in x, for exact checks
        self.mass = mass        # kg, the block on the spring (springs only)
        self.note = note
        self.kind = kind        # "spring" or "general", see the top of this file
        self.x_name, self.x_sym, self.x_unit, self.x_col = x_name, x_sym, x_unit, x_col
        self.y_name, self.y_sym, self.y_unit, self.y_col = y_name, y_sym, y_unit, y_col
        self.x_range = x_range  # the input values the agent may use
        self.setup = setup      # what the agent is told about the system (general laws)
        self.source = source    # team file the law comes from
        # Constants the task card tells the agent (regex for the symbol -> value).
        # An answer written with them, such as E_k = m / v, is read with the value.
        self.given = given or []

    @property
    def is_spring(self):
        return self.kind == "spring"

    def test_points(self, n=60):
        """Unseen inputs used by the grader. Springs: both sides of 0, avoiding 0.
        General laws: spread over the whole allowed range, between sample points."""
        if self.is_spring:
            return [s * (0.13 + 0.0475 * i) for i in range(n // 2) for s in (1, -1)]
        lo, hi = self.x_range
        return [lo + (hi - lo) * (i + 0.37) / n for i in range(n)]


LAWS = {
    # Team repo "physics law redesign.py": Hooke's law with x replaced by |x|^1.5
    "power_spring": Law(
        name="power_spring",
        force=lambda x: -5.0 * abs(x) ** 1.5 * sign(x),
        formula="F(x) = -5 * |x|^1.5 * sign(x)",
        sympy_expr="-5*Abs(x)**Rational(3, 2)*sign(x)",
        note="Redesigned Hooke's law (team repo). Non-linear: stiffer when stretched more.",
    ),
    # Normal Hooke's law, kept as a control task the models should find easily
    "hooke": Law(
        name="hooke",
        force=lambda x: -5.0 * x,
        formula="F(x) = -5 * x",
        sympy_expr="-5*x",
        note="Classic Hooke's law. Control task.",
    ),
    # ---- Training laws for rsi.py (self-improvement). The agent may see these
    # laws' true formulas while it improves its strategy. power_spring is NEVER
    # shown to it, so the final test on power_spring stays a fair test.
    "cubic_spring": Law(
        name="cubic_spring",
        force=lambda x: -2.0 * x - 0.5 * x ** 3,
        formula="F(x) = -2*x - 0.5*x^3",
        sympy_expr="-2*x - x**3/2",
        note="Training law: stiffening spring (linear + cubic).",
    ),
    "root_spring": Law(
        name="root_spring",
        force=lambda x: -3.0 * math.sqrt(abs(x)) * sign(x),
        formula="F(x) = -3 * |x|^0.5 * sign(x)",
        sympy_expr="-3*sqrt(Abs(x))*sign(x)",
        note="Training law: softening power law.",
    ),
    "steep_power": Law(
        name="steep_power",
        force=lambda x: -1.2 * abs(x) ** 2.5 * sign(x),
        formula="F(x) = -1.2 * |x|^2.5 * sign(x)",
        sympy_expr="-6*Abs(x)**Rational(5, 2)*sign(x)/5",
        note="Training law: steep power law.",
    ),
    "softening_spring": Law(
        name="softening_spring",
        force=lambda x: -4.0 * x + 0.3 * x ** 3,
        formula="F(x) = -4*x + 0.3*x^3",
        sympy_expr="-4*x + 3*x**3/10",
        note="Training law: softening spring (linear minus cubic).",
    ),

    # ---- The team's redesigned laws 2 to 5 (Ananya, 7 Nov 2026, repo top folder).
    # Same constants as the team scripts. Agents see the setup text, never the formula.
    "kinetic_energy": Law(
        name="kinetic_energy", kind="general",
        force=lambda v: 0.5 * 2.0 ** 2 / v,
        formula="E_k(v) = 0.5 * m^2 / v, with m = 2 kg, so E_k = 2 / v",
        sympy_expr="2/x",
        x_name="speed", x_sym="v", x_unit="m/s", x_col="velocity",
        y_name="kinetic energy", y_sym="E_k", y_unit="J", y_col="kinetic_energy",
        x_range=(0.5, 10.0),
        setup=("A cart of mass m = 2 kg moves along a straight track at speed v (m/s). Its "
               "kinetic energy E_k (J) is measured at different speeds. The energy does NOT "
               "necessarily follow the textbook formula."),
        source="kinetic_energy_law2.py", given=[(r"m", "2")],
        note="Law 2: redesigned kinetic energy. Falls with speed instead of rising.",
    ),
    "wave_speed": Law(
        name="wave_speed", kind="general",
        force=lambda f: 15.0 * math.log(f * 0.5),
        formula="v(f) = c * ln(f * lambda), with c = 15 and lambda = 0.5 m, so v = 15 ln(0.5 f)",
        sympy_expr="15*log(x/2)",
        x_name="frequency", x_sym="f", x_unit="Hz", x_col="frequency",
        y_name="wave speed", y_sym="v", y_unit="m/s", y_col="wavespeed",
        x_range=(2.0, 100.0),
        setup=("A wave travels along a string. The wavelength is fixed at lambda = 0.5 m. The "
               "wave speed v (m/s) is measured at different driving frequencies f (Hz). The "
               "speed does NOT necessarily follow the textbook formula."),
        source="wavespeed_law3.py", given=[(r"\\lambda", "0.5")],
        note="Law 3: redesigned wave speed. Starts at 2 Hz, where f * lambda = 1, so the "
             "speed is never negative.",
    ),
    "screened_coulomb": Law(
        name="screened_coulomb", kind="general",
        force=lambda r: 1.0 * 2.0 * 3.0 / r ** 2 * math.exp(-0.4 * r),
        formula="F(r) = k q1 q2 / r^2 * exp(-alpha r), with k = 1, q1 = 2, q2 = 3, "
                "alpha = 0.4, so F = 6 exp(-0.4 r) / r^2",
        sympy_expr="6*exp(-2*x/5)/x**2",
        x_name="distance", x_sym="r", x_unit="m", x_col="distance",
        y_name="force", y_sym="F", y_unit="N", y_col="force",
        x_range=(0.2, 5.0),
        setup=("Two point charges, q1 = 2 C and q2 = 3 C, are held a distance r (m) apart. In "
               "these units the Coulomb constant is k = 1. The force F (N) between them is "
               "measured at different distances. The force does NOT necessarily follow the "
               "textbook formula."),
        source="electrostatics_law4.py",
        given=[(r"q_\{1\}|q_1|q1", "2"), (r"q_\{2\}|q_2|q2", "3"), (r"k", "1")],
        note="Law 4: Coulomb's law with an exponential decay (a screened force).",
    ),
    "saturating_ohm": Law(
        name="saturating_ohm", kind="general",
        force=lambda V: 5.0 * math.tanh(V / 2.0),
        formula="I(V) = I_max * tanh(V / R), with I_max = 5 A and R = 2 V, so I = 5 tanh(V/2)",
        sympy_expr="5*tanh(x/2)",
        x_name="voltage", x_sym="V", x_unit="V", x_col="voltage",
        y_name="current", y_sym="I", y_unit="A", y_col="current",
        x_range=(-10.0, 10.0),
        setup=("A component is connected to an adjustable voltage source. The current I (A) "
               "through it is measured at different voltages V (V). The current does NOT "
               "necessarily follow the textbook formula."),
        source="ohms_law5.py",
        note="Law 5: redesigned Ohm's law. The current saturates at 5 A.",
    ),
}

DEFAULT_LAW = "power_spring"
TEST_LAW = "power_spring"   # held out from self-improvement
TRAINING_LAWS = ["cubic_spring", "root_spring", "steep_power", "softening_spring"]
TEAM_LAWS = ["power_spring", "kinetic_energy", "wave_speed", "screened_coulomb", "saturating_ohm"]


def get_law(name=None):
    return LAWS[name or DEFAULT_LAW]
