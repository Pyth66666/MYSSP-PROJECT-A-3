# Benchmarking Self-Evolving AI Agents for Automated Physics Discovery

A virtual physics lab, with a 3D view, for testing whether LLM agents can **discover counterfactual physics laws** from data rather than recall textbook equations.

Altered physics laws are hidden inside a Python sandbox. An agent gets only data or experiments, submits an equation, and an exact symbolic grader (SymPy) checks whether it recovered the hidden law.

## Contents

- [Why Counterfactual Laws?](#why-counterfactual-laws)
- [Key Findings](#key-findings)
- [Methodology](#methodology)
- [Installation and Quick Start](#installation)
- [Testing Real Models](#testing-real-models)
- [Repository Files](#repository-files)
- [Limitations](#limitations)

## Why Counterfactual Laws?

Benchmarks that ask models to recover Hooke's law or Ohm's law can be passed by memorization, because these equations appear all over textbooks and pre-training data. Here, each familiar law is redesigned so textbook recall fails. A model has to analyze the measurements to succeed. Agents are told that standard physics may not apply, but never see the formula.

| `--law` name | Hidden law | Textbook form | Input range | Designs |
|---|---|---|---|---|
| `power_spring` | F(x) = -5·\|x\|^1.5·sign(x) | F = -kx | -3 to 3 m | S, A, B, C |
| `kinetic_energy` | E_k(v) = 0.5·m²/v, m = 2 kg | E = 0.5 m v² | 0.5 to 10 m/s | S, A, C |
| `wave_speed` | v(f) = 15·ln(0.5·f), λ = 0.5 m | v = fλ | 2 to 100 Hz | S, A, C |
| `screened_coulomb` | F(r) = 6·e^(-0.4r)/r², q₁ = 2, q₂ = 3, k = 1 | F = k q₁q₂/r² | 0.2 to 5 m | S, A, C |
| `saturating_ohm` | I(V) = 5·tanh(V/2) | I = V/R | -10 to 10 V | S, A, C |

Fixed values (mass, wavelength, charges) are told to the agent, so every law has a single input. Design B needs a moving block, so it is spring only.

## Key Findings

- Frontier commercial models (`gpt-6-sol`, `gpt-5.5`, `gpt-5.6-terra`, `deepseek-v4.1-flash`) reached 98-100% pass rates in clean active-probing conditions.
- All four tested open-source models (Qwen2.5:3B, LLaMA 3.2:3B, Gemma 2:2B, Phi-3.5:3.8B) scored 0%.
- 3% sensor noise lowered median commercial pass rates from 98% to 84%.
- Passive observation was much harder than active experimentation: in the spring "release and watch" condition, only `gpt-6-sol` reached 90%.
- Self-written strategy notes did not help `gpt-5.4-nano` (0/10 before and after), while `gpt-5.5` scored 10/10.
- A two-agent CrewAI setup improved `deepseek-v4-flash` (25 to 38 of 50) but not `gpt-6-luna`.
- More tokens did not mean better results: `gpt-6-sol` used fewer median tokens than `gpt-5.4-nano` while passing all 100 runs across Designs A and C.

Full tables and analysis: [docs/results.md](docs/results.md).

## Methodology

The benchmark is a three-component pipeline:

```text
[Sandbox] -> saves CSV data -> [Evaluation pipeline] -> sends prompt + data
          -> [Benchmark agent] -> extracts formula -> checks math -> [SymPy evaluator]
```

### 1. Designing the Counterfactual Laws

Each law was altered with one of five transformation strategies:

| Strategy | Example in this benchmark |
|---|---|
| Altered exponents | Spring: a square relationship becomes a 1.5 power |
| Inverted variable relationships | Kinetic energy: multiplication by v² becomes division by v |
| Logarithmic soft-capping | Wave speed: growth flattens as frequency increases |
| Exponential decay | Screened Coulomb: the effect fades rapidly with distance |
| Smooth saturation ceilings | Ohm's law: tanh makes current approach a limit instead of growing forever |

Input ranges deliberately exclude values that cause division by zero or undefined logarithms.

### 2. Physics Sandbox

The sandbox is written in Python and holds the hidden law. It generates telemetry and saves it as structured CSV files, using two approaches:

- **Time-stepped numerical integration** for dynamic systems (the spring). A 1 kg block on a frictionless rail is attached to a wall by a spring that follows the hidden law. At each time step the sandbox computes force, converts it to acceleration with Newton's second law (a = F/m), and integrates with the semi-implicit Euler (Euler-Cromer) method, using dt = 0.1 s over 10 s:

  $$v_{t+1} = v_t + a_t \Delta t \qquad x_{t+1} = x_t + v_{t+1} \Delta t$$

  Only the release position changes between datasets (|x0| = 0.5 to 3.0 m, both sides).
- **Direct parameter sweeps** for static or state-based laws (kinetic energy, wave speed, screened Coulomb, saturating Ohm). The sandbox builds an evenly spaced input array with `numpy.linspace`, evaluates the hidden function, and writes each result to CSV with Pandas.

### 3. Discovery Agent

The agent receives:

- A short description of the experimental setup.
- Fixed constants and units, and the permitted input range.
- Either a fixed data table or the ability to request measurements.
- A warning that the system does not necessarily follow the textbook law.
- Instructions to submit one final equation in LaTeX.

The agent is never shown the hidden formula. In interactive designs it chooses which experiments to run, observes the results, and decides when to submit.

### 4. Evaluation Pipeline (SymPy)

The grader converts the submitted equation into a symbolic expression and checks it in two layers:

1. **Exact symbolic equivalence.** Subtract the agent's expression from the hidden one and simplify with SymPy. If the difference is zero, the response passes immediately.
2. **Symmetric numerical verification.** If symbolic simplification does not settle it, evaluate both formulas on **60 unseen test points** and compute normalized root mean square error (NRMSE).

The unseen grid checks that the model found the underlying relationship, not just a fit to the points it saw. Test points include positive and negative values, and exclude zero where necessary. The grader accepts LaTeX or Python input, `|x|`, `sign`/`sgn`, `\sqrt`, `\frac`, `\boxed`, `tanh`, `sinh`, `cosh`, and `ln f` without brackets.

### 5. Experimental Designs and Run Counts

| | S. Static dataset | A. Force probe | B. Release and watch | C. Noisy lab |
|---|---|---|---|---|
| Agent gets | 20 rows of (x, F) from a CSV | holds block at any x, reads F | releases block from x0, sees only x(t) | active probing with noise |
| Experiments | none | 12 | 6 | 10 |
| Noise | none | none | none | 3% force, 5 mm position |
| Pass if error ≤ | 1% (or exact) | 1% | 2% | 3% |
| Tests | fitting a formula to given data | finding a law with chosen data | inferring force from motion | planning + robustness |

- Design S tests blind curve fitting from a fixed dataset. Designs A and C test autonomous experimentation. Design B tests inferring a law from a passive trajectory instead of choosing measurements.
- In Designs A, B, and C, each model runs **10 times per law** to account for LLM randomness. In Design S, models are evaluated on **50 fixed datasets per law**.

**Noise model (Design C).** Gaussian noise with a standard deviation of 3% of each measured value. For the spring, additional noise of 0.02 N is applied to force and 5 mm to recorded positions. This tests whether agents can separate a real relationship from random measurement variation.

### 6. Models Evaluated

| Group | Models | How they were run |
|---|---|---|
| Commercial (9) | `gpt-6-sol`, `gpt-6-luna`, `gpt-5.6-terra`, `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`, `deepseek-v4-flash`, `deepseek-v4.1-flash` | RapidScreen API via LiteLLM |
| Open-source (5) | `qwen2.5-coder:7b`, `qwen2.5:3b`, `llama3.2:3b`, `gemma2:2b`, `phi3.5:3.8b` | Locally via Ollama, in early pipeline testing on Hooke's law |
| Non-LLM baselines (2) | Hooke assumption, power-law curve fit | No language model |

**Baselines.** The *Hooke assumption* always outputs the textbook law regardless of the data, which tests whether a model is reasoning instead of defaulting to memory. The *power-law curve fit* fits a power law directly to the data, which separates blind statistical fitting from scientific discovery.

| Baseline | Pass | Exact SymPy match | Median error |
|---|---|---|---|
| Curve fit (no AI) | 100% | 0% | 0.01% |
| Hooke assumption (no AI) | 0% | 0% | 28.7% |

The law is recoverable from the data in every design, and assuming the textbook law fails every time, so the task tests discovery, not memory. The curve fit gets close but never an exact match because it reports fitted numbers such as 1.5002. It solves the two power laws but fails the log, exponential-decay, and tanh laws.

### 7. Recursive Self-Improvement (RSI)

RSI was tested by letting `gpt-5.4-nano` learn from its own earlier attempts, in two phases:

1. **Practice.** The model completes three rounds on one redesigned law, then writes strategy notes on what worked and which mistakes to avoid. In the code (`rsi.py`), new notes are kept only if they score at least as well as the old ones.
2. **Testing.** The model is tested once on an unseen, held-out law (`--test-law`) while using its own notes.

Compared conditions: `gpt-5.4-nano` without notes, `gpt-5.4-nano` with its own notes, and `gpt-5.5` without notes as a large-model baseline. This does not isolate RSI alone, since it also reflects base-model capability, but it asks whether lightweight self-reflection can help a small model approach a much larger one.

In the reported run, the notes did **not** help. They may have reinforced spring-specific habits (measuring at zero, using symmetric positive and negative inputs) that did not transfer to positive-only laws.

### 8. Multi-Agent System (CrewAI)

A two-agent CrewAI system (version 1.15.43) collaborates to discover and validate a law.

| Agent | Role | Responsibility |
|---|---|---|
| **Theorist** | Theoretical Physicist | Proposes, refines, and reports candidate laws with reasoning |
| **Analyst** | Computational Analyst | Designs tests, evaluates hypotheses, runs calculations, and scores candidates from 0 to 1 |

Both agents can use a SymPy solver/simplifier and a NumPy curve-fitting tool. The workflow:

1. The theorist proposes one or more candidate laws.
2. The analyst designs tests and evaluates each hypothesis, scoring it from 0 to 1.
3. A score above 0.8 means the hypothesis is accepted. A score of 0.8 or below sends the agents back to refine it.
4. Once accepted, the system writes a final report with the discovered law and equation.

Because hypotheses are evaluated after experiments are run, wrong hypotheses cost extra tokens and time. The setup is a simple two-agent design and may not represent more sophisticated multi-agent architectures.

### 9. Evaluation Metrics

**Pass rate** is the main metric:

$$\text{Pass rate} = \frac{\text{equations passing the SymPy checks}}{\text{total runs}}$$

A response passes if it is symbolically equivalent to the hidden law or falls within the design's error tolerance.

**Median error** is reported alongside pass rate, because pass rate treats a close approximation and a wildly wrong equation the same. Median is preferred over mean because occasional parsing failures create extreme outliers.

**Milestones** show where models fail in the discovery process:

| Milestone | Meaning |
|---|---|
| **M1: Answer** | The model submitted a formula the grader could read |
| **M2: Shape** | Normalized error on the 60 unseen points is below 100% |
| **M3: Close** | Error is 10% or less |
| **M4: Pass** | Error is within the design-specific tolerance |
| **M5: Exact** | SymPy confirms exact symbolic equivalence with the hidden law |

**Efficiency** is measured by task completion rate, median execution time per run, estimated token use per run, and resource use across designs.

### 10. Logging

Every run is saved as a JSON log with the task objective, each action and result, the full agent conversation, the submitted equation, the grade, execution time, token usage, design, model name, and hidden-law identifier. Design S answers are stored in full so they can be regraded without new API calls.

## Installation

Python 3.9+ is enough for the sandbox (the study used 3.13.1). Install SymPy for exact matching.

```bash
git clone https://github.com/Pyth66666/MYSSP-PROJECT-A-3.git
cd MYSSP-PROJECT-A-3
pip install -r requirements.txt
```

Use `python` instead of `py` in the commands below if that is what works on your machine.

### Quick Start

```powershell
py -m unittest discover -s tests     # 20 checks, about 2 seconds, last line: OK
py generate_data.py                  # writes the 50 CSV datasets
py pipeline.py                       # runs the pipeline with every available agent
py bench.py                          # compares agents on designs S, A, B, C
py server.py                         # 3D sandbox at http://127.0.0.1:8000
```

If port 8000 is taken: `$env:SBX_PORT = "8001"; py server.py`, then open port 8001.

### Other Laws

```powershell
py generate_data.py --law wave_speed                     # 50 datasets for one law
py pipeline.py --law wave_speed --datasets 10            # design S
py experiment.py --law screened_coulomb --design C --agents "RapidScreen gpt-6-sol" --seeds 10
py compare.py --law screened_coulomb                     # leaderboard for that law
py rsi.py --small "RapidScreen gpt-5.4-nano" --big "RapidScreen gpt-6-sol" --design C --test-law saturating_ohm
```

### Autonomous Experiments, Self-Improvement, Leaderboard

| File | What it does |
|---|---|
| `experiment.py` | The agent runs the whole experiment itself from one fixed objective prompt. Full runs go to `results/runs/`, summary rows to `results/experiments.tsv`. |
| `rsi.py` | The self-improvement loop described in the methodology above. |
| `compare.py` | Leaderboard per design from each model's latest runs (`--law`, `--latest`). |

### The 3D Page

- Tabs **S, A, B, C** pick the design.
- **S**: *Play dataset* animates the simulated run. The force chart shows the 20 rows the agent receives, and *Prompt sent to the agent* shows the exact prompt.
- **A to C**: move the slider and press *Measure force* or *Release block*.
- Type a formula and press *Submit*, or pick an agent and press *Run*.
- **Evaluator view** draws the hidden law as a dashed line for explaining results. Agents never get it.
- The page also shows the leaderboard and can replay any saved run.

## Testing Real Models

### Ollama (free, local)

```powershell
ollama pull qwen2.5-coder:7b
py pipeline.py --list                 # should now list the Ollama agents
py pipeline.py --datasets 10          # start small
```

`models.json` lists the models. Entries whose model is not pulled, or whose API key is not set, are skipped automatically. For an API model, replace `SET-MODEL-ID` with the provider's model id and set the key in the terminal (the RapidScreen models share `RAPIDSCREEN_API_KEY`):

```powershell
$env:RAPIDSCREEN_API_KEY = "..."      # macOS/Linux: export RAPIDSCREEN_API_KEY=...
```

Never put keys in files, and never commit raw API logs.

### CrewAI Framework

`framework/physics_discovery_flow.py` is the team's theorist + analyst flow, unchanged. `agents_crewai.py` runs it inside the pipeline by passing in the dataset, asking for a last line `FINAL FORMULA: $$F(x) = ...$$`, and reading the model from `models.json`.

```powershell
py -m pip install crewai
py pipeline.py --agents "CrewAI framework (qwen2.5-coder:7b)" --datasets 5
```

### Terminal Agents (Codex, Kilo)

`cli.py` runs any design step by step:

```text
py cli.py new --design S --seed 0 --agent-name codex
py cli.py exp <session> '{"release": {"x0": 2.0}}'      (designs A-C)
py cli.py submit <session> "F(x) = ..."
```

Results go to `results/external.tsv`. A coding agent with file access could open `laws.py` and read the answer, so for real numbers run it where that file is hidden (for example in a Docker container that only exposes `cli.py`). An agent that has already seen the formula is not a fair test subject.

## Repository Files

| File | What it does |
|---|---|
| `laws.py` | The hidden laws. Only the sandbox and evaluator read it. |
| `sandbox_env.py` | The 4 designs, experiment budget, noise, logging, static datasets. |
| `generate_data.py` | Writes the 50 CSV datasets per law. |
| `pipeline.py` | The evaluation pipeline over the CSV datasets. |
| `grader.py` | Formula extraction + SymPy evaluator. |
| `agent.py` | Baselines, the LLM agent, and the team prompt. |
| `agents_crewai.py` | Runs the CrewAI framework as an agent. |
| `experiment.py`, `rsi.py`, `compare.py` | Autonomous runs, self-improvement, leaderboard. |
| `bench.py` | Compares agents across designs S, A, B, C. |
| `cli.py` | Step-by-step terminal version for coding agents. |
| `models.json` | Which models to test. |
| `server.py`, `index.html`, `static/` | The 3D sandbox (three.js included, works offline). |
| `framework/` | The team's CrewAI framework (unchanged). |
| `tests/` | `test_sandbox.py` (20 checks), `crewai_smoke.py` (needs crewai). |

## Extending

- **New law:** add one `Law(...)` in `laws.py` with a Python function and the same law as a SymPy string, then run `py generate_data.py --law <name>` and `py pipeline.py --law <name>`.
- **New model:** add an entry to `models.json`.
- **New agent strategy** (reflection, memory, self-evolving): subclass `Agent` in `agent.py` with a `run(lab)` method and add it to `all_agents()`.

## Limitations

- Single-input laws only; motion is 1D along the rail. The benchmark does not test multi-variable or coupled systems.
- Simulated data only; no real laboratory measurements.
- No model was trained or fine-tuned. "Self-improvement" means written strategy notes, not parameter updates or reinforcement learning.
- Commercial models were accessed through an API, so versions and behavior may change over time.
- Some Design S evaluations are incomplete because of an API-key change (five models were evaluated only on kinetic energy). Design B covers only the spring law.
- The RSI experiment used few runs, which limits statistical power.
- CrewAI token counts are estimates, because the API does not report token usage to CrewAI.
- API content-filter blocks and strict LaTeX parsing failures were counted as task failures.
- Pass tolerances (1%, 2%, 3%) are our choice. Change them in `DESIGNS` in `sandbox_env.py`.
- Datasets with a small release position (|x0| = 0.5 m) cover only a small range of x, so they are harder. That is intended.

Results and future work: [docs/results.md](docs/results.md). Review notes on the team's original scripts: [docs/team_code_notes.md](docs/team_code_notes.md).

## Acknowledgments

Thanks to mentor **Fan Xiaoyan** and to **MyResearchGuide** for guidance, resources, and programme organization.

## License

MIT License. See [LICENSE](LICENSE).

## Citation

```bibtex
@misc{prasad2026benchmarking,
  title={Benchmarking Self-Evolving AI Agents for Automated Physics Discovery},
  author={Ananya Prasad and Tang Yi Zhe and Muhammad Adief Al Syarif bin Abdullah},
  year={2026},
  howpublished={Malaysia Science Scholar's Programme 2026},
  url={https://github.com/Pyth66666/MYSSP-PROJECT-A-3}
}
```
