import re
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
    test_results: str = ""
    refined_results: str = ""
    human_input: str = ""

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

@tool("NumPy Polynomial Curve Fit Tool")
def numpy_curve_fit_tool(x_values_csv: str, y_values_csv: str, degree: int = 1) -> str:
  """Calculates the coefficients for a line or curve of best fit using NumPy.
  Provide comma-separated numbers for x and y, and the polynomial degree (1 for linear, 2 for quadratic).
  Example: x='1,2,3,4', y='2.1,3.9,6.2,8.0', degree=1
  """
  try:
    x = np.array([float(i) for i in x_values_csv.split(",")])
    y = np.array([float(i) for i in y_values_csv.split(",")])

    # Fit polynomial curve of best fit
    coefficients = np.polyfit(x, y, degree)

    # Calculate R-squared for accuracy check
    p = np.poly1d(coefficients)
    y_pred = p(x)
    correlation_matrix = np.corrcoef(x, y)
    r_squared = correlation_matrix[0, 1] ** 2

    return (
        f"Curve Fit Coefficients (highest degree first): {list(coefficients)}\n"
        f"Mathematical Model Expression: {coefficients}\n"
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
                "Form physics equations based on the experimental results"

            ),
            expected_output="Per-hypothesis results and scores, ending with 'BEST: <name> SCORE: <number>', and then form Physics equations based on the experimental results with correct units"
            + NO_DATA_RULE,
            agent=analyst,
        )
        
        crew = Crew(agents=[analyst], tasks=[design, analyze],
                    process=Process.sequential, verbose=True)
        
        self.state.test_results = crew.kickoff().raw
        return self.state.test_results

    @router(design_and_test)
    def decide_next_step(self):
        pattern = r"SCORE:\s*(0?\.[89][0-9]*|1(\.0+)?)\s*$"
        if re.search(pattern, self.state.test_results):
            return "supported"
        return "refine"

    @listen("supported")
    def write_report(self):
        result = make_theorist().kickoff(
            f"Discovery aim: {self.state.phenomenon}\n\n"
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
        ]
        crew = Crew(agents=[theorist, analyst], tasks=tasks,
                    process=Process.sequential, verbose=True)
        self.state.refined_results = crew.kickoff().raw
        return self.state.refined_results

    @listen(refine_and_retest)
    def write_final_report(self):
        result = make_theorist().kickoff(
            f"Discovery aim: {self.state.phenomenon}\n\n"
            f"Initial hypotheses:\n{self.state.hypotheses}\n\n"
            f"Initial tests and scores:\n{self.state.test_results}\n\n"
            f"Refined hypotheses and retest results:\n{self.state.refined_results}\n\n"
            "Write the report with sections: Best-supported law, Evidence, Limitations, "
            "Follow-up research. If support remained weak, say so plainly. If inputs are "
            "missing, state plainly that the source returned nothing."
        )
        return result.raw


if __name__ == "__main__":
    try:
        flow = PhysicsLawDiscovery()
        report = flow.kickoff(inputs={
        "phenomenon": "Find the law related to Hooke's Law and the relationship between force, spring constant, and displacement."
    })
        print(report)

    except Exception:
        print(make_theorist().llm.model, make_analyst().llm.model)
        print("LLM calls failed. Check that the LLM server is running and reachable at the specified base_url.")