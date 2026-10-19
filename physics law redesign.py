import numpy as np
import pandas as pd

mass = 1.0 #kg
k = 5.0 #spring constant
dt = 0.1 #time step seconds
total_time = 10 #total time seconds

def redesigned_force(x): # x is the position
    return -k * (np.abs(x) ** 1.5) * np.sign(x)

#REDESIGNED PHYSICS LAW: -k * x^1.5


x = 2.0 #position
v = 0.0 #velocity
telemetry_data = []

for step in range(int(total_time/dt)):
    t = step * dt
    f = redesigned_force(x)
    a = f / mass

    telemetry_data.append({
        "time: ", round(t, 2),
        "position: ", round(x, 4),
        "velocity: ", round(v, 4),
        "acceleration: ", round(a, 4),
        "force: ", round(f, 4)
    })
    
    v = v + a * dt
    x = x + v * dt

#creates csv file
df = pd.DataFrame(telemetry_data, columns=['time', 'position', 'velocity', 'acceleration', 'force', 'col5', 'col6', 'col7', 'col8'])
df.to_csv("newphysformula.csv", index=False)

df = pd.read_csv("newphysformula.csv")

df_sample = df.iloc[::10] #takes every tenth line to reduce file size
data_string = df_sample[["position", "force"]].head(20).to_string(index=False) #takes first 20 rows in position/force column



# no expert physicist
# use python to generate data set
# atleast 50 sets of data
# test with ai models (gpt, gemini, calude, deep seek)
# redesgin 3 more laws




#example prompt
prompt = """
You are an expert physicist. Below is experimental data collected from an unknown 1D mechanical system. 
The table contains position(x) in meters and measured restoring force (F) in Newtons.

{data_String}

Task:
1. Analyze the mathematical relationship between position (x) and force (f)
2. Deduce the exact functional form/formula for F(x)
3. State the derived formula clearly in LaTeX format
 """

# NEED TO LINK LLM TO CODE FIRST!!!

#turns formula into algebraic form
import sympy as sp
x_sym = sp.Symbol('x')
real_formula = -5 * (sp.Abs(x_sym)**1.5) * sp.sign(x_sym)


import re
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

#turns text output from llm into algebraic sympy format
def extract_equation(ai_text: str) -> str:
    latex_matches = re.findall(r"\$\$(.*?)\$\$|\$(.*?)\$", ai_text, re.DOTALL)
    extracted = ""
    for match in latex_matches:
        candidate = match[0] or match[1]
        if "F" in candidate or "x" in candidate:
            extracted = candidate
            break

    if not extracted:
        extracted = ai_text
    if "=" in extracted:
        extracted = extracted.split("=")[1]
    
# replace latex symbols with python operators
    extracted = (
        extracted.replace(r"\cdot", "*")
        .replace(r"\times", "*")
        .replace("^", "**")
        .replace(r"\left", "")
        .replace(r"\right", "")
        .strip()
    )

    return extracted


def grade_ai_response(
    ai_text: str, real_formula_expr: sp.Expr, symbol: sp.Symbol
) -> bool:
    raw_expr = extract_equation(ai_text)

    # smart transformations allows 5x to be automatically parsed as 5*x
    transformations = standard_transformations + (
        implicit_multiplication_application,
    )

# method 1: algebraic difference test - result should be 0 if equations are an exact match
    try:
        ai_expr = parse_expr(raw_expr, transformations=transformations)
    except Exception as e:
        print(f"parsing failed! could not parse: '{raw_expr}'. error: {e}")
        return False

    difference = sp.simplify(ai_expr - real_formula_expr) #if they are a match difference = 0
    if difference == 0:
        print("PASS: exact symbolic match")
        return True

  # method 2 backup
  # picks 10 random numbers for position x, plugs those numbers into the both the real equation and the ai equation. 
  # if both formulas produce the exact same  outputs for all 10 test points then pass

    test_points = np.linspace(0.5, 5.0, 10)
    is_equivalent = True

    for p in test_points:
        val_gt = float(real_formula_expr.subs(symbol, p))
        val_ai = float(ai_expr.subs(symbol, p))

        # checks relative difference using a tight tolerance
        if not np.isclose(val_gt, val_ai, rtol=1e-3, atol=1e-3):
            is_equivalent = False
            break

    if is_equivalent:
        print("PASS: Numerical Equivalence Match!")
        return True
    else:
        print(
            f"FAIL: AI output '{ai_expr}' does not match real formula '{real_formula_expr}'"
        )
        return False
    

sample_ai_response = """
#response from llm
"""
    
is_correct = grade_ai_response(
    ai_text=sample_ai_response, real_formula_expr=real_formula, symbol=x
)