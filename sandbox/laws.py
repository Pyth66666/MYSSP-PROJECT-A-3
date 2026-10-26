"""Hidden physical laws used by the sandbox.

RULE: only the sandbox (sandbox_env.py) and the evaluator (grader.py) may
import this file. Agents never see it. They learn about the law only through
the experiments the sandbox lets them run.

To add one of the team's new laws, add one Law(...) entry to LAWS below.
Nothing else in the sandbox needs to change.
"""

import math


def sign(v):
    return (v > 0) - (v < 0)


class Law:
    def __init__(self, name, force, formula, sympy_expr, mass=1.0, note=""):
        self.name = name
        self.force = force      # python function F(x) in newtons
        self.formula = formula  # human-readable truth, shown only after grading
        self.sympy_expr = sympy_expr  # same law as a SymPy string, for exact checks
        self.mass = mass        # kg, the block on the spring
        self.note = note


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
}

DEFAULT_LAW = "power_spring"


def get_law(name=None):
    return LAWS[name or DEFAULT_LAW]
