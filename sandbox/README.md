# Physics Discovery Sandbox (MYSSP A-3)

A virtual physics lab, with a 3D view, for testing whether AI agents can **discover a
hidden force law from data**.

The hidden law is the team's redesigned Hooke's law (`physics law redesign.py`):

    F(x) = -5 * |x|^1.5 * sign(x)

A 1 kg block on a frictionless rail is attached to a wall by a spring that follows this
law. Agents never see the formula. They only get the data.

## The pipeline (as agreed in the team chat)

```
[Sandbox] -> saves CSV data -> [Evaluation pipeline] -> sends prompt + data
          -> [Benchmark agent] -> extracts formula -> checks math -> [SymPy evaluator]
```

| Step | File | What it does |
|---|---|---|
| Sandbox | `generate_data.py` | Simulates the spring exactly like the team script (dt = 0.1 s, 10 s, semi-implicit Euler) and saves **50 CSV datasets** (`data/<law>/dataset_00..49.csv`). Only the release position changes: \|x0\| = 0.5 to 3.0 m, both sides. Dataset 0 is x0 = 2.0 m, the same as the script. |
| Evaluation pipeline | `pipeline.py` | For each CSV: takes 20 evenly spaced (position, force) rows, puts them into the team prompt, sends it to the agent. |
| Benchmark agent | `agent.py`, `agents_crewai.py` | Single-prompt LLM (Ollama or any API), the team's **CrewAI framework**, and two non-AI baselines. |
| SymPy evaluator | `grader.py` | Extracts the formula (LaTeX or Python), checks for an **exact SymPy match**, and measures the error on 60 unseen positions on **both sides** of x = 0. |

The answer to "do we give the AI Hooke's law first?": **no**. The agent gets only the
(x, F) table and must find the new law itself. Giving it Hooke's law would hint at the answer.

## Sandbox design proposals (for the team to compare)

The pipeline above is design **S**. For the "at least three sandbox designs" deliverable
there are three more, where the agent **runs its own experiments** instead of getting a
fixed table. All four hide the same law.

| | S. Static dataset | A. Force probe | B. Release and watch | C. Noisy lab |
|---|---|---|---|---|
| Agent gets | 20 rows of (x, F) from a CSV | holds block at any x, reads F | releases block from x0, sees only x(t) | both tools |
| Experiments | none | 12 | 6 | 10 |
| Noise | none | none | none | 3% force, 5 mm position |
| Pass if error ≤ | 1% (or exact) | 1% | 2% | 3% |
| Tests | fitting a formula to given data | finding a law with chosen data | inferring force from motion | planning + robustness |

## Run it (Windows PowerShell)

Python 3.9+ is enough for the sandbox. Install SymPy for exact matching (`requirements.txt`):

```powershell
cd <this folder>
py -m pip install sympy

py -m unittest discover -s tests     # 20 checks, about 2 seconds, last line: OK
py generate_data.py                  # writes the 50 CSV datasets
py pipeline.py                       # runs the pipeline with every available agent
py bench.py                          # compares agents on designs S, A, B, C
py server.py                         # 3D sandbox at http://127.0.0.1:8000
```

Use `python` instead of `py` if that is what works on your machine. If port 8000 is
taken by another app: `$env:SBX_PORT = "8001"; py server.py`, then open port 8001.

### The 3D page

- Tabs **S, A, B, C** at the top right pick the design.
- **S**: *Play dataset* animates the simulated run. The force chart shows the 20 rows the
  agent receives, and *Prompt sent to the agent* shows the exact prompt.
- **A to C**: move the slider and press *Measure force* or *Release block*.
- Type a formula and press *Submit*, or pick an agent and press *Run*.
- **Evaluator view** draws the hidden law as a dashed line, for explaining results.
  Agents never get it.
- The tables at the bottom show the last `pipeline.py` and `bench.py` results.

## Testing real models (Ollama, free, on the laptop)

```powershell
ollama pull qwen2.5-coder:7b          # the model the CrewAI framework uses
py pipeline.py --list                 # should now list the Ollama agents
py pipeline.py --datasets 10          # start small: 10 datasets
```

`models.json` lists the models. Entries whose model is not pulled, or that need an API key
that is not set, are skipped automatically. For an API model, replace `SET-MODEL-ID` with the
provider's model id and set the key in the terminal, for example
`$env:DEEPSEEK_API_KEY = "..."`. Never put keys in files.

### The team's CrewAI framework

`framework/physics_discovery_flow.py` is Yzzzz's theorist + analyst flow, **unchanged**.
`agents_crewai.py` runs it inside the pipeline:

```powershell
py -m pip install crewai
py pipeline.py --agents "CrewAI framework (qwen2.5-coder:7b)" --datasets 5
```

The adapter does three things without editing the framework: it puts the dataset into the
flow's input, asks for a last line `FINAL FORMULA: $$F(x) = ...$$` so the answer can be
extracted, and lets `models.json` choose the model. Tested with CrewAI 1.15 against a fake
model server (`tests/crewai_smoke.py`): the data reaches the agents and the formula is
extracted. It has **not** been run with a real model yet.

### Terminal agents (Claude Code, Codex, Kilo)

`cli.py` runs any design step by step:

```
py cli.py new --design S --seed 0 --agent-name codex
py cli.py exp <session> '{"release": {"x0": 2.0}}'      (designs A-C)
py cli.py submit <session> "F(x) = ..."
```

Results go to `results/external.tsv`. A coding agent with file access could open `laws.py`
and read the answer, so for real numbers run it where that file is hidden (for example in a
Docker container that only contains `cli.py` access). An agent that has already seen the formula, including
the one that built this sandbox, is not a fair test subject.

## Results so far

No LLM has been run yet: model downloads were not possible from the build machine, so the
first real model run has to happen on the laptop. These are the two non-AI reference agents.

`py pipeline.py`, 50 CSV datasets:

| Agent | Pass | Exact SymPy match | Median error |
|---|---|---|---|
| curve-fit baseline (no AI) | 100% | 0% | 0.01% |
| Hooke assumption (no AI) | 0% | 0% | 28.7% |

`py bench.py`, 3 seeds per design:

| Agent | S | A | B | C |
|---|---|---|---|---|
| curve-fit baseline (no AI) | 100% | 100% | 100% | 100% |
| Hooke assumption (no AI) | 0% | 0% | 0% | 0% |

What this shows:
- The law can be recovered from the data in every design.
- Assuming the textbook law fails every time, so the task tests discovery, not memory.
- The curve-fit baseline gets close (error 0.01%) but never an *exact* match, because it
  reports fitted numbers such as 1.5002. An LLM that states exactly 1.5 can beat it on the
  exact-match column.

## Checks on the team code

Notes from reading the team files, for whoever owns them.

**`physics law redesign.py`**
1. The telemetry rows use `{"time: ", round(t, 2), ...}`, which is a Python *set*, not a dict,
   so the CSV columns come out scrambled. Use `{"time": ..., "position": ...}`.
2. 9 column names are given for 5 values.
3. The prompt has `{data_String}` but is never formatted, so the model would get the literal text.
4. `grade_ai_response(..., symbol=x)` passes the number `x` (the last position), not the
   SymPy symbol `x_sym`.
5. The numeric check only tests x from 0.5 to 5. The answer `-5*x**1.5` is correct for x > 0
   but wrong for x < 0, and that check would pass it. `grader.py` tests both sides.
6. `extract_equation` cannot read `|x|` or `\mathrm{sign}(x)`, the most likely way a model
   writes this law. `grader.py` handles these, plus `\sqrt`, `\frac`, `\text{sgn}`, `\boxed`,
   `\approx` and markdown.
7. SymPy's `x` is not declared real, so `simplify` cannot prove that equivalent forms such as
   `-5*x*sqrt(|x|)` are equal. `grader.py` uses a real symbol and checks each side of 0.

**`framework/physics_discovery_flow.py` (CrewAI)**
1. The example input has no data, only "Find the law related to Hooke's Law...". The agents
   would answer from memory, and "Hooke" hints at the answer. The adapter passes the data
   and the team prompt instead.
2. The analyst's fitting tool only fits polynomials, so it cannot find a power of 1.5 directly
   (a log-log fit would).
3. The tool's R-squared uses `np.corrcoef(x, y)`, which is the linear fit quality, for every
   degree. For degree 2 and up it should compare `y` with `y_pred`.
4. In the refine branch, the later tasks do not include the original data, only earlier results.

## Files

| File | What it does |
|---|---|
| `laws.py` | The hidden laws. Only the sandbox and evaluator read it. Add new laws here. |
| `sandbox_env.py` | The 4 designs, experiment budget, noise, logging, static datasets. |
| `generate_data.py` | Writes the 50 CSV datasets. |
| `pipeline.py` | The evaluation pipeline over the CSV datasets. |
| `grader.py` | Formula extraction + SymPy evaluator. |
| `agent.py` | Baselines and the LLM agent, the team prompt. |
| `agents_crewai.py` | Runs the CrewAI framework as an agent. |
| `framework/` | The team's CrewAI framework (unchanged). |
| `models.json` | Which models to test. |
| `bench.py` | Compares agents across designs S, A, B, C. |
| `cli.py` | Step-by-step terminal version for coding agents. |
| `server.py`, `index.html`, `static/` | The 3D sandbox (three.js is included, works offline). |
| `tests/` | `test_sandbox.py` (20 checks), `crewai_smoke.py` (needs crewai). |

## Adding things

- **New law** (the next 3 redesigns): add one `Law(...)` in `laws.py` with a Python
  function and the same law as a SymPy string, then
  `py generate_data.py --law <name>` and `py pipeline.py --law <name>`.
- **New model**: add an entry to `models.json`.
- **New agent strategy** (reflection, memory, self-evolving): subclass `Agent` in `agent.py`
  with a `run(lab)` method and add it to `all_agents()`.

## Known limits

- The motion is 1D along the rail. The 3D scene shows that motion; it does not add physics.
- Pass tolerances (1%, 2%, 3%) are our choice. Change them in `DESIGNS` in `sandbox_env.py`
  if the team agrees on others.
- Datasets with a small release position (|x0| = 0.5 m) only cover a small range of x, so
  they are harder than dataset 0. That is intended, but worth knowing when reading results.
