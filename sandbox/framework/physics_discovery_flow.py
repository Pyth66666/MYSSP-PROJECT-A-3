import re
import json
import argparse
import subprocess
import sys
from pydantic import BaseModel
from crewai import LLM, Agent, Crew, Task, Process
from crewai.flow.flow import Flow, listen, router, start
from crewai.tools import tool 
import os
import numpy as np
import sympy as sp

os.environ["OPENAI_API_KEY"] = "NA"  # Bypasses CrewAI validation

NO_DATA_RULE = (
    " If the input is missing or empty, output exactly: NO DATA: <source> returned nothing."
    " Estimated, remembered or example values are never a substitute."
)

local_llm = LLM(
    model="ollama/qwen2.5-coder:7b",
    base_url="http://localhost:11434",
)

class DiscoveryState(BaseModel):
    phenomenon: str = ""
    hypotheses: str = ""
    x_value: list[float] = []
    y_value: list[float] = []
    test_results: str = ""
    refined_results: str = ""
    human_input: str = ""

    @property                          # <-- CHANGED: new helper, turns the lists into text
    def data(self) -> str:             
        if not self.x_value or not self.y_value:
            return ""
        return (
                "x (position): " + ",".join(map(str, self.x_value)) + "\n"
                "y (force): " + ",".join(map(str, self.y_value))
            )
    

SANDBOX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATOR = os.path.join(SANDBOX_DIR, "generate_data.py")
DATA_FILE = os.path.join(SANDBOX_DIR, "runs", "output_data.json")


def generate_data(dataset: int = 0, path: str = DATA_FILE) -> str:
    """Run the sandbox data generator, which simulates the spring and writes the JSON packet.

    The generator runs as its own process, so this file never imports the hidden law.
    """
    subprocess.run(
        [sys.executable, GENERATOR, "--json", path, "--dataset", str(dataset)],
        check=True, capture_output=True, text=True,
    )
    return path


def read_data(path: str = DATA_FILE) -> dict:
    """Read the JSON packet written by the data generator.

    Returns {"x_value": [...], "y_value": [...]}. Raises instead of returning empty
    lists, so a failed generator can never silently become "no data".
    """
    with open(path, encoding="utf-8") as f:
        packet = json.load(f)
    x, y = packet.get("x_value"), packet.get("y_value")
    if not x or not y or len(x) != len(y):
        raise ValueError(f"{path} must hold non-empty x_value and y_value lists of equal length.")
    return {"x_value": [float(v) for v in x], "y_value": [float(v) for v in y]}


@tool("SymPy Equation Solver and Simplifier")
def symbolic_math_tool(equation_str: str, solve_for_var: str) -> str:
  """Useful for solving or simplifying physical equations symbolically.
  Provide an equation string (e.g., 'F - m * a = 0' or 'KineticEnergy - 0.5 * m * v**2')
  and the variable you want to solve for (e.g., 'v' or 'a').
  """
  try:
    # Split equation by '=' if provided
    if "=" in equation_str:
      left_str, right_str = equation_str.split("=")
      eq = sp.Eq(sp.sympify(left_str), sp.sympify(right_str))
    else:
      eq = sp.sympify(equation_str)

    target_var = sp.Symbol(solve_for_var)
    solutions = sp.solve(eq, target_var)

    return f"Symbolic solutions for {solve_for_var}: {solutions}"
  except Exception as e:
    return f"Error in symbolic calculation: {str(e)}"

@tool("NumPy Curve Fit Tool")
def numpy_curve_fit_tool(x_values_csv: str, y_values_csv: str, degree: int = 1, model: str = "polynomial") -> str:
  """Fits a curve to data using NumPy.
  Provide comma-separated numbers for x and y.
  model='polynomial': fits a polynomial of the given degree (1 linear, 2 quadratic).
  model='power': fits y = a * x**b (x and y must be positive); the exponent b is found directly.
  Example: x='1,2,3,4', y='2.1,3.9,6.2,8.0', degree=1, model='polynomial'
  Example: x='1,2,3,4', y='2,5.66,10.39,16', model='power'
  """
  try:
    x = np.array([float(i) for i in x_values_csv.split(",")])
    y = np.array([float(i) for i in y_values_csv.split(",")])
 
    if model == "power":
      # FIX 2: power-law fit via log-log linear regression -> y = a * x**b
      if np.any(x <= 0) or np.any(y <= 0):
        return "Error in curve fitting: power model needs all x and y values > 0."
      b, ln_a = np.polyfit(np.log(x), np.log(y), 1)
      a = float(np.exp(ln_a))
      y_pred = a * x ** b
      expression = f"y = {a:.6g} * x^{b:.6g}"
      coeff_text = f"a = {a:.6g}, exponent b = {b:.6g}"
    else:
      coefficients = np.polyfit(x, y, degree)
      y_pred = np.poly1d(coefficients)(x)
      expression = str(np.poly1d(coefficients))
      coeff_text = f"Coefficients (highest degree first): {list(coefficients)}"
 
    # FIX 3: R-squared computed from the residuals of the actual fitted model
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
 
    return (
        f"Model: {model}\n"
        f"{coeff_text}\n"
        f"Mathematical Model Expression: {expression}\n"
        f"R-Squared Accuracy: {round(r_squared, 4)}"
    )
  except Exception as e:
    return f"Error in curve fitting: {str(e)}"
  
def make_theorist() -> Agent:
    return Agent(
        role="Theoretical Physicist",
        goal="Propose, refine and report on candidate physical laws with reasoning.",
        backstory=(
            "You are a rigorous theorist. You work only from what your input contains "
            "and never fabricate data or citations. If your input contains NO DATA, say "
            "in your own words that the source returned nothing; invent nothing."
        ),
        llm=local_llm,
        inject_date=True,
        verbose=True,
    )


def make_analyst() -> Agent:
    return Agent(
        role="Computational Analyst",
        goal="Design tests for each hypothesis, evaluate them, and score each from 0 to 1.",
        backstory=(
            "You design falsifiable tests and analyze rigorously, showing derivations. "
            "You work only from your input; numbers come from explicit derivation, "
            "never guesses, and you flag anything unverified."
        ),
        llm=local_llm,
        tools = [symbolic_math_tool, numpy_curve_fit_tool],
        inject_date=True,
        verbose=True,
    )

class PhysicsLawDiscovery(Flow[DiscoveryState]):

    @start()
    def generate_hypotheses(self):
        result = make_theorist().kickoff(
            f"Discovery aim: {self.state.phenomenon}\n\n"
            "Propose 3-5 candidate laws or mathematical relationships, each with an "
            "equation, variables, assumptions and reasoning.\n"
            "Expected output: a numbered list of hypotheses." + NO_DATA_RULE
        )
        self.state.hypotheses = result.raw
        print(f"Generated hypotheses:\n{self.state.hypotheses}")
        return self.state.hypotheses

    @listen(generate_hypotheses)
    def design_and_test(self):
        analyst = make_analyst()
        design = Task(
            description=(
                f"Hypotheses for {self.state.phenomenon}:\n{self.state.hypotheses}\n\n"
                f"Measured data:\n{self.state.data}\n\n"
                "For each hypothesis define a simulation or analytical test with "
                "predictions, and what result would refute it."
            ),
            expected_output="A test plan per hypothesis with predictions and refutation criteria."
            + NO_DATA_RULE,
            agent=analyst,
        )
        
        analyze = Task(
            description=(
                "Carry out the analytical tests (derivations, limiting cases, dimensional "
                "analysis, consistency with known results) and score each hypothesis from "
                "0 to 1. State what was verified versus unverified. State the reason for each score."
                "End with a line 'BEST: <hypothesis name> SCORE: <number>'."
                "Analyze experimental results using the same methodology and score each hypothesis from 0 to 1. State what was verified versus unverified. "
                "Form physics equations based on the experimental results"

            ),
            expected_output="Per-hypothesis results and scores, ending with 'BEST: <name> SCORE: <number>', analysis of experimental results, and then form Physics equations based on the experimental results with correct units"
            + NO_DATA_RULE,
            agent=analyst,
        )
        
        crew = Crew(agents=[analyst], tasks=[design, analyze],
                    process=Process.sequential, verbose=True)
        
        self.state.test_results = crew.kickoff().raw
        return self.state.test_results

    @router(design_and_test)
    def decide_next_step(self):
        pattern = r"SCORE:\s*(0?\.[89][0-9]*|1(\.0+)?)"
        if re.search(pattern, self.state.test_results):
            return "supported"
        return "refine"

    @listen("supported")
    def write_report(self):
        result = make_theorist().kickoff(
            f"Discovery aim: {self.state.phenomenon}\n\n"
            f"Measured data:\n{self.state.data}\n\n"
            f"Hypotheses:\n{self.state.hypotheses}\n\n"
            f"Tests and scores:\n{self.state.test_results}\n\n"
            "Write the report with sections: Best-supported law, Evidence, Limitations, "
            "Follow-up research. If inputs are missing, state plainly that the source "
            "returned nothing."
        )
        return result.raw

    @listen("refine")
    def refine_and_retest(self):
        theorist, analyst = make_theorist(), make_analyst()
        data = f"Measured data:\n{self.state.data}\n\n"
        tasks = [
            Task(description=(f"Previous hypotheses:\n{self.state.hypotheses}\n\n"
                              f"Previous test results:\n{self.state.test_results}\n\n"
                              "Refine, merge or replace the hypotheses based on where they failed."),
                 expected_output="Refined hypotheses with equations and reasoning." + NO_DATA_RULE,
                 agent=theorist),
            Task(description="Retest the refined hypotheses analytically and score each from 0 to 1, "
                             "stating what was verified versus unverified.",
                 expected_output="Scores per hypothesis." + NO_DATA_RULE,
                 agent=analyst),
            Task(description="If no hypothesis scored well in the last retest, refine again using "
                             "the feedback; otherwise keep the best one unchanged.",
                 expected_output="Final hypotheses." + NO_DATA_RULE,
                 agent=theorist),
            Task(description="Run the final analytical retest, score each hypothesis from 0 to 1, "
                             "and name the best one with its score and remaining weaknesses.",
                 expected_output="Final scores, ending with 'BEST: <name> SCORE: <number>'." + NO_DATA_RULE,
                 agent=analyst),
            Task(description= self.state.data +
                             "Run the final analytical retest against the measured data, score each "
                             "hypothesis from 0 to 1, and name the best one with its score and "
                             "remaining weaknesses.",
                 expected_output="Final scores, ending with 'BEST: <name> SCORE: <number>'." + NO_DATA_RULE,
                 agent=analyst),
        ]
        crew = Crew(agents=[theorist, analyst], tasks=tasks,
                    process=Process.sequential, verbose=True)
        self.state.refined_results = crew.kickoff().raw
        return self.state.refined_results

    @listen(refine_and_retest)
    def write_final_report(self):
        result = make_theorist().kickoff(
            f"Discovery aim: {self.state.phenomenon}\n\n"
            f"Measured data:\n{self.state.data}\n\n"
            f"Initial hypotheses:\n{self.state.hypotheses}\n\n"
            f"Initial tests and scores:\n{self.state.test_results}\n\n"
            f"Refined hypotheses and retest results:\n{self.state.refined_results}\n\n"
            "Write the report with sections: Best-supported law, Evidence, Limitations, "
            "Follow-up research. If support remained weak, say so plainly. If inputs are "
            "missing, state plainly that the source returned nothing."
        )
        return result.raw


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the physics discovery flow on generated data.")
    parser.add_argument("--dataset", type=int, default=0,
                        help="which of the generator's datasets to use (default 0)")
    args = parser.parse_args()

    # Sandbox generator -> JSON packet -> read_data(): no hand-typed data anywhere.
    try:
        data = read_data(generate_data(args.dataset))
    except Exception as e:
        print(f"Data generation failed: {e}")
        print(getattr(e, "stderr", "") or "")
        sys.exit(1)

    try:
        flow = PhysicsLawDiscovery()
        report = flow.kickoff(inputs={
            "phenomenon": "Find the relationship between position and force.",
            "x_value": data["x_value"],
            "y_value": data["y_value"],
        })
        print(report)

    except Exception:
        print(make_theorist().llm.model, make_analyst().llm.model)
        print("LLM calls failed. Check that the LLM server is running and reachable at the specified base_url.")

    


    
