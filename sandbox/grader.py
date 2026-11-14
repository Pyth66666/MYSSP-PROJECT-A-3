"""The evaluator: scores a submitted formula against the hidden law.

This is the "extract formula -> check math -> SymPy evaluator" end of the
team pipeline.

1. Pull the formula out of the model's text (plain Python or LaTeX). If the
   text has several formulas, the last one that can be read is used.
2. Turn it into a safe expression (checked node by node, never raw eval).
3. SymPy check: simplify(answer - truth) == 0 with x real -> "exact match".
4. Numeric check on 60 unseen positions on BOTH sides of 0 -> normalised RMSE.
   Pass if exact, or if the error is within the design's tolerance.

Testing both signs matters: -5*x**1.5 matches the law for x > 0 but is wrong
(complex) for x < 0. The team script only tested 0.5..5, so it would pass it.

SymPy is optional. Without it the numeric check still runs.
"""

import ast
import math
import re

FUNCTIONS = {
    "abs": abs,
    "sign": lambda v: float((v > 0) - (v < 0)),
    "sgn": lambda v: float((v > 0) - (v < 0)),
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "ln": math.log,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "tanh": math.tanh,
    "sinh": math.sinh,
    "cosh": math.cosh,
}
CONSTANTS = {"pi": math.pi, "e": math.e}
KNOWN_NAMES = sorted(list(FUNCTIONS) + list(CONSTANTS) + ["x"], key=len, reverse=True)

# Unseen positions, both sides of 0, avoiding 0 itself
TEST_POINTS = [s * (0.13 + 0.0475 * i) for i in range(30) for s in (1, -1)]


# ------------------------------------------------------------------ extract
def _clean(chosen):
    chosen = chosen.replace(r"\approx", "=").replace("≈", "=").replace(r"\simeq", "=")
    chosen = chosen.strip().strip("`").strip()
    while chosen.startswith("**") and chosen.endswith("**") and len(chosen) > 4:
        chosen = chosen[2:-2].strip()   # markdown bold
    if "=" in chosen:
        chosen = chosen.split("=")[-1]
    chosen = chosen.strip().rstrip(".,;:\"'").strip()
    while chosen.endswith("}") and chosen.count("}") > chosen.count("{"):
        chosen = chosen[:-1].rstrip().rstrip("\"'").strip()   # leftover JSON braces
    return chosen


def _states_law(s):
    return "=" in s or r"\approx" in s or "≈" in s


def formula_candidates(text):
    """All plausible formulas in the text, most likely final answer first."""
    text = str(text).strip()
    out = []
    final = re.findall(r"FINAL FORMULA:\s*(.+)", text)
    for f in reversed(final):
        out.append(f.strip().strip("$").strip())
    out.extend(reversed(re.findall(r'"answer"\s*:\s*"([^"]+)"', text)))
    # LaTeX \boxed{...}: the model's own marked answer, last one first
    out.extend(reversed([b for b in _boxed(text) if _states_law(b)]))
    blocks = re.findall(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]|\\\((.+?)\\\)|\$(.+?)\$",
                        text, re.DOTALL)
    pieces = [next(p for p in b if p) for b in blocks]
    # Only maths that states a law counts; a lone \( x \) in a sentence does not.
    out.extend(reversed([p for p in pieces if _states_law(p)]))
    out.extend(reversed([ln for ln in text.splitlines() if _states_law(ln)]))
    out.append(text)
    seen, result = set(), []
    for c in out:
        c = _clean(c)
        if c and "x" in c and c not in seen:
            seen.add(c)
            result.append(c)
    return result or [_clean(text)]


def _boxed(text):
    """Contents of every \\boxed{...} in the text, with nested braces."""
    found = []
    _replace_braced_command(text, r"\boxed", 1, lambda a: found.append(a[0]) or "")
    return found


def _strip_units_and_notes(s):
    """Drop what is not part of the formula: a trailing condition after \\qquad or
    \\quad (for example 'v > 0'), and units written as \\mathrm{..} or \\text{..}."""
    s = re.split(r"\\qquad|\\quad", s)[0].rstrip().rstrip(".,;").rstrip()
    s = re.sub(r"\[\s*\\(?:mathrm|text)\s*\{?[^\]]*\]", "", s)   # E_k[\mathrm J]
    s = re.sub(r"\\(?:mathrm|text|mathit)\s*\{\s*e\s*\}", "e", s)  # Euler e
    s = _replace_braced_command(s, r"\mathrm", 1, lambda a: "")
    s = _replace_braced_command(s, r"\text", 1, lambda a: "")
    s = re.sub(r"\\(?:mathrm|text)\s*[A-Za-z]+", "", s)               # \mathrm J
    return s.rstrip().rstrip(".,;").rstrip()


def extract_formula(text):
    """Return the right-hand side of the most likely final formula."""
    return formula_candidates(text)[0]


# ------------------------------------------------------------ latex -> py
def _replace_braced_command(s, cmd, nargs, build):
    """Replace \\cmd{a}{b} (with nested braces) using build(args)."""
    out, i = "", 0
    while True:
        j = s.find(cmd, i)
        if j == -1:
            return out + s[i:]
        out += s[i:j]
        k = j + len(cmd)
        args = []
        for _ in range(nargs):
            while k < len(s) and s[k] == " ":
                k += 1
            if k >= len(s) or s[k] != "{":
                break
            depth, start = 0, k
            while k < len(s):
                if s[k] == "{":
                    depth += 1
                elif s[k] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            args.append(s[start + 1:k])
            k += 1
        if len(args) != nargs:
            out += cmd
            i = j + len(cmd)
            continue
        out += build(args)
        i = k


def latex_to_python(expr):
    s = expr
    for a, b in [("−", "-"), ("·", "*"), ("×", "*"), ("²", "^2"), ("³", "^3"),
                 ("√", "sqrt"), ("**", "^")]:
        s = s.replace(a, b)
    for junk in [r"\left", r"\right", r"\displaystyle", r"\,", r"\;", r"\!", r"\ "]:
        s = s.replace(junk, "")
    for word in ["sign", "sgn", "abs", "tanh", "sinh", "cosh"]:
        for wrap in [r"\mathrm", r"\operatorname", r"\text", r"\mathit"]:
            s = s.replace(wrap + "{" + word + "}", word)
    s = s.replace(r"\sgn", "sign")
    s = _strip_units_and_notes(s)
    s = s.replace(r"\tfrac", r"\frac")
    s = s.replace(r"\cdot", "*").replace(r"\times", "*")
    s = s.replace(r"\lvert", "|").replace(r"\rvert", "|").replace(r"\vert", "|")
    s = re.sub(r"\\sqrt\[([^\]]+)\]\{", r"\\nroot{\1}{", s)
    s = _replace_braced_command(s, r"\nroot", 2, lambda a: f"(({a[1]})^(1/({a[0]})))")
    s = _replace_braced_command(s, r"\frac", 2, lambda a: f"(({a[0]})/({a[1]}))")
    s = _replace_braced_command(s, r"\dfrac", 2, lambda a: f"(({a[0]})/({a[1]}))")
    s = _replace_braced_command(s, r"\sqrt", 1, lambda a: f"sqrt({a[0]})")
    s = re.sub(r"\\(tanh|sinh|cosh|exp|log|ln|sin|cos|tan|pi)", r"\1", s)
    # |...| -> abs(...), innermost first
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\|([^|]+)\|", r"abs(\1)", s)
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace("^", "**")
    return s


# ------------------------------------------------ implicit multiplication
TOKEN = re.compile(r"\s*(\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+|[A-Za-z_]+|\*\*|[-+*/(),])")


def _tokens(s):
    pos, out = 0, []
    s = s.strip()
    while pos < len(s):
        m = TOKEN.match(s, pos)
        if not m:
            raise ValueError(f"unexpected character '{s[pos]}'")
        tok = m.group(1)
        pos = m.end()
        if tok[0].isalpha() or tok[0] == "_":
            out.extend(_split_name(tok))
        else:
            out.append(tok)
    return out


def _split_name(word):
    """'xabs' -> ['x', 'abs'] so '5x|x|' style input still parses."""
    parts, i = [], 0
    low = word.lower()
    while i < len(low):
        for name in KNOWN_NAMES:
            if low.startswith(name, i):
                parts.append(name)
                i += len(name)
                break
        else:
            raise ValueError(f"unknown symbol '{word}'")
    return parts


def _wrap_bare_calls(toks):
    """'ln x' or 'ln 2' -> 'ln ( x )': a function followed by one plain value."""
    out, i = [], 0
    while i < len(toks):
        tok = toks[i]
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        if tok in FUNCTIONS and nxt is not None and nxt != "(" and \
                (nxt[0].isdigit() or nxt[0] == "." or nxt == "x" or nxt in CONSTANTS):
            out += [tok, "(", nxt, ")"]
            i += 2
            continue
        out.append(tok)
        i += 1
    return out


def add_implicit_multiplication(s):
    toks = _wrap_bare_calls(_tokens(s))
    out = []
    for i, tok in enumerate(toks):
        if out:
            prev = out[-1]
            prev_is_value = (prev[0].isdigit() or prev[0] == "." or prev == ")"
                             or prev in CONSTANTS or prev == "x")
            cur_starts_value = (tok[0].isdigit() or tok[0] == "." or tok == "("
                                or tok in FUNCTIONS or tok in CONSTANTS or tok == "x")
            if prev_is_value and cur_starts_value:
                out.append("*")
        out.append(tok)
    return "".join(out)


# ----------------------------------------------------- the law's own symbol
def map_symbols(text, law):
    """Let the agent write the answer with the law's own input symbol, such as
    E_k = 2/v or I = 5 tanh(V/2): that symbol becomes x before parsing. Letters
    inside words and LaTeX commands (\\frac, \\right, sqrt) are left alone."""
    sym = getattr(law, "x_sym", "x") if law is not None else "x"
    if not sym or sym == "x":
        return str(text)
    return re.sub(r"(?<![A-Za-z_\\])" + re.escape(sym) + r"(?![A-Za-z_])", "x", str(text))


# ------------------------------------------------------------- safe eval
ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name,
                 ast.Load, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div,
                 ast.Pow, ast.USub, ast.UAdd)


def compile_formula(text, law=None):
    """Return (python_expression_string, function f(x)) for the first readable
    candidate formula in the text. With a law, its input symbol is read as x."""
    errors = []
    for cand in formula_candidates(map_symbols(text, law)):
        try:
            return _compile_one(cand)
        except Exception as exc:  # noqa: BLE001 - try the next candidate
            errors.append(str(exc))
    raise ValueError(errors[0] if errors else "no formula found")


def _compile_one(candidate):
    expr = add_implicit_multiplication(latex_to_python(candidate))
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError(f"not allowed in a formula: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in FUNCTIONS \
                and node.id not in CONSTANTS and node.id != "x":
            raise ValueError(f"unknown name '{node.id}'")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name)
                                               and node.func.id in FUNCTIONS):
            raise ValueError("only abs, sign, sqrt, exp, log, sin, cos, tan, tanh, sinh, "
                             "cosh calls allowed")
    code = compile(tree, "<formula>", "eval")
    env = {"__builtins__": {}, **FUNCTIONS, **CONSTANTS}

    def f(x):
        v = eval(code, env, {"x": x})  # noqa: S307 - AST checked above
        if isinstance(v, complex):
            raise ValueError("complex result (e.g. negative number to a fractional power)")
        return float(v)

    return expr, f


# ------------------------------------------------------------- sympy check
def sympy_exact(parsed_expr, law):
    """True if SymPy proves the answer equals the law for real x.
    None if SymPy is not installed or cannot decide."""
    try:
        import sympy as sp
    except ImportError:
        return None
    x = sp.Symbol("x", real=True)
    names = {"x": x, "abs": sp.Abs, "sign": sp.sign, "sgn": sp.sign, "sqrt": sp.sqrt,
             "exp": sp.exp, "log": sp.log, "ln": sp.log, "sin": sp.sin, "cos": sp.cos,
             "tan": sp.tan, "tanh": sp.tanh, "sinh": sp.sinh, "cosh": sp.cosh,
             "pi": sp.pi, "e": sp.E, "Abs": sp.Abs, "Rational": sp.Rational}
    try:
        # parsed_expr already passed the AST whitelist in compile_formula
        ai = sp.nsimplify(sp.sympify(parsed_expr, locals=names), rational=True)
        truth = sp.sympify(law.sympy_expr, locals=names)
        diff = sp.simplify(ai - truth)
        if diff == 0:
            return True
        # split on the sign of x, where Abs and sign become plain powers. A law whose
        # inputs are all positive (speed, frequency, distance) is only checked there.
        p = sp.Symbol("p", positive=True)
        positive_only = getattr(law, "x_range", (-1, 1))[0] > 0
        if positive_only:   # logs may be combined when every input is positive
            return sp.simplify(sp.expand_log(diff.subs(x, p), force=True)) == 0
        return all(sp.simplify(d) == 0 for d in (diff.subs(x, p), diff.subs(x, -p)))
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- grading
def grade(answer_text, law, tolerance=0.01):
    result = {"passed": False, "exact": None, "nrmse": None, "parsed": None,
              "reason": "", "truth": law.formula, "tolerance": tolerance}
    try:
        parsed, f = compile_formula(answer_text, law)
    except Exception as exc:  # noqa: BLE001 - any parse problem is a fail
        result["reason"] = f"could not read formula: {exc}"
        return result
    result["parsed"] = parsed

    err2, ref2 = 0.0, 0.0
    points = law.test_points() if hasattr(law, "test_points") else TEST_POINTS
    for x in points:
        truth = law.force(x)
        try:
            guess = f(x)
        except Exception as exc:  # noqa: BLE001
            result["reason"] = f"formula fails at x = {x:.3f}: {exc}"
            return result
        if not math.isfinite(guess):
            result["reason"] = f"formula is not finite at x = {x:.3f}"
            return result
        err2 += (guess - truth) ** 2
        ref2 += truth ** 2
    nrmse = math.sqrt(err2 / ref2) if ref2 else math.sqrt(err2)
    result["nrmse"] = round(nrmse, 6)
    result["exact"] = sympy_exact(parsed, law) if nrmse < 0.05 else False
    result["passed"] = bool(result["exact"]) or nrmse <= tolerance
    if result["exact"]:
        result["reason"] = "exact SymPy match"
    else:
        result["reason"] = (f"normalised RMSE {nrmse:.2%} "
                            f"{'<=' if result['passed'] else '>'} {tolerance:.0%}")
    return result
