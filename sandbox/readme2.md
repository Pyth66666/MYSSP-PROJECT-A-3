# Benchmarking Self-Evolving AI Agents for Automated Physics Discovery

This project evaluates whether large language model (LLM) agents can autonomously discover **counterfactual physics laws** from simulated laboratory data, rather than simply recalling textbook equations from training data.

The system places altered physics laws inside a Python physics sandbox, allows an LLM agent to run virtual experiments, and uses an exact symbolic-grading pipeline to determine whether the agent recovered the hidden law.

**Repository:** https://github.com/Pyth66666/MYSSP-PROJECT-A-3.git

## Overview

Traditional scientific-discovery benchmarks often test whether models can recover well-known laws such as Hooke’s law, Newtonian kinetic energy, Coulomb’s law, or Ohm’s law. Because these equations appear extensively in textbooks and pre-training data, a model may “discover” them through memorization rather than scientific reasoning.

This project addresses that limitation by redesigning familiar laws into **counterfactual** forms. For example, the spring force is no longer linear in displacement, and kinetic energy is no longer proportional to velocity squared. A model that relies on textbook recall should therefore fail, while a model that genuinely analyzes measurements may recover the hidden relationship.

The project evaluates:

- Whether LLM agents can generate hypotheses, design experiments, interpret telemetry, and submit equations.
- How model performance changes under clean, passive, and noisy experimental conditions.
- Whether active experimentation improves discovery compared with receiving a fixed dataset.
- Whether recursive self-improvement through self-written strategy notes helps smaller models.
- Whether a two-agent CrewAI system outperforms a single-prompt agent.
- How accurately, quickly, and efficiently different commercial and open-source models perform.

## Key Findings

- Frontier commercial models substantially outperformed compact open-source models.
- `gpt-6-sol`, `gpt-5.5`, `gpt-5.6-terra`, and `deepseek-v4.1-flash` achieved 98–100% pass rates across the five laws in clean active-probing conditions.
- Under 3% sensor noise, median commercial pass rates decreased from 98% to 84%, showing that noisy data remains a major challenge.
- All four tested open-source models—Qwen2.5:3B, LLaMA 3.2:3B, Gemma 2:2B, and Phi-3.5:3.8B—achieved 0% pass rate in the tested conditions.
- Passive trajectory observation was much harder than active experimentation. In the spring-only “release and watch” condition, only `gpt-6-sol` reached 90%.
- Self-written strategy notes did not improve `gpt-5.4-nano`; performance remained 0/10 before and after the notes, while `gpt-5.5` achieved 10/10.
- The two-agent CrewAI setup improved DeepSeek-V4-Flash from 25 to 38 passes out of 50, but did not improve GPT-6-Luna, suggesting that multi-agent benefits depend on the base model and task.
- Higher token use did not necessarily produce better results. For example, `gpt-6-sol` used fewer median tokens than `gpt-5.4-nano` while passing all 100 runs across Designs A and C.

## Counterfactual Physics Laws

The benchmark uses five single-input laws whose textbook forms have been intentionally altered. Models are told that the system may not follow standard physics and must infer the law from measurements.

| Law | Hidden law | Textbook form | Input range |
|---|---|---|---|
| Spring | `F(x) = -5 * abs(x)**1.5 * sign(x)` | `F = -kx` | `x` from -3 to 3 m |
| Kinetic energy | `E(v) = 2 / v` for `m = 2 kg` | `E_k = 0.5 * m * v^2` | `v` from 0.5 to 10 m/s |
| Wave speed | `v(f) = 15 * log(0.5 * f)` | `v = fλ` | `f` from 2 to 100 Hz |
| Screened Coulomb | `F(r) = 6 * exp(-0.4 * r) / r^2` | `F = k * q1 * q2 / r^2` | `r` from 0.2 to 5 m |
| Saturating Ohm | `I(V) = 5 * tanh(V / 2)` | `I = V / R` | `V` from -10 to 10 V |

The laws were designed using five transformation strategies:

1. **Altered exponents** — for example, changing a squared relationship into a 1.5-power relationship.
2. **Inverted variable relationships** — replacing multiplication with division.
3. **Logarithmic soft-capping** — causing growth to flatten as the input increases.
4. **Exponential decay** — making an effect diminish rapidly with distance.
5. **Smooth saturation ceilings** — using hyperbolic tangent functions so the output approaches a physical limit rather than increasing indefinitely.

Input ranges deliberately exclude values that would cause division by zero or undefined logarithms.

## System Architecture

The project uses a modular three-component pipeline:

```text
┌─────────────────────┐     ┌──────────────────────┐     ┌──────────────────────────┐
│  Physics Sandbox    │ --> │  LLM Discovery Agent │ --> │  SymPy Evaluation Pipeline │
│  Hidden law + data  │     │  Hypotheses, probes, │     │  Symbolic + numerical      │
│  Telemetry CSV      │     │  final equation      │     │  grading, logs, results    │
└─────────────────────┘     └──────────────────────┘     └──────────────────────────┘
```

### 1. Physics Sandbox

The sandbox is implemented in Python 3.13.1 and contains the hidden counterfactual law. It generates experimental telemetry and saves the results as structured CSV files.

Two simulation approaches are used:

- **Time-stepped numerical integration** for dynamic systems, such as the redesigned spring.
- **Direct parameter sweeps** for static or state-based laws, such as kinetic energy, wave speed, screened Coulomb force, and saturating Ohm’s law.

For the spring simulation, the system defines constants such as mass, spring-related parameters, drag, time step, and initial position and velocity. At each time step, the sandbox computes force, converts it to acceleration using Newton’s second law, and integrates motion using the Euler-Cromer method:

\[
v_{t+1} = v_t + a_t \Delta t
\]

\[
x_{t+1} = x_t + v_{t+1} \Delta t
\]

For static laws, the sandbox generates an evenly spaced input array using `numpy.linspace`, evaluates the hidden function, and writes each result to CSV using Pandas.

### 2. LLM Discovery Agent

The agent receives:

- A short description of the experimental setup.
- Fixed constants and units.
- The permitted input range.
- Either a fixed data table or the ability to request measurements.
- A warning that the system does not necessarily follow the textbook law.
- Instructions to submit one final equation in LaTeX format.

The agent is not shown the hidden formula. In interactive designs, it chooses which experiments to run, observes the returned measurements, and decides when to submit its final answer.

### 3. Evaluation Pipeline

The evaluation pipeline is built with Python and SymPy. It converts the model’s submitted equation into a symbolic expression and checks whether it matches the hidden law.

The grading process has two layers:

1. **Exact symbolic equivalence.**  
   The grader subtracts the agent’s expression from the hidden expression and simplifies the result using SymPy. If the difference simplifies to zero, the response passes immediately.

2. **Symmetric numerical verification.**  
   If symbolic simplification does not resolve the expression to zero, the pipeline evaluates both formulas on 60 unseen test points and computes normalized root mean square error (NRMSE).

The unseen test grid is important because it checks whether the model discovered the underlying relationship rather than merely fitting the measurements it observed. Test points include both positive and negative values, while excluding zero where necessary.

## Experimental Designs

The benchmark tests four experimental regimes.

| Design | What the model receives | Measurement limit | Pass threshold |
|---|---|---:|---:|
| **S — Static dataset** | One fixed table containing 50 pre-recorded measurements; no further experiments allowed | One answer | Error ≤ 1% |
| **A — Active probing** | The agent chooses inputs and receives exact outputs, one at a time | 12 measurements | Error ≤ 1% |
| **B — Release and watch** | Spring-only condition; the model observes the motion of a released block but cannot choose inputs | 6 experiments | Error ≤ 2% |
| **C — Noisy lab** | Same active-probing setup as Design A, but every reading contains Gaussian noise | 10 measurements | Error ≤ 3% |

Design S tests blind mathematical curve-fitting from a fixed dataset. Designs A and C test autonomous experimentation. Design B tests whether a model can infer a law from a passive trajectory rather than directly selecting measurements.

In Designs A, B, and C, each model was run 10 times per law to account for randomness in LLM responses. In Design S, models were evaluated across 50 fixed datasets per law.

### Noise Model

Design C adds Gaussian noise with a standard deviation equal to 3% of each measured value. For the spring experiment, additional noise is applied as:

- 0.02 N on force measurements
- 5 mm on recorded positions

This tests whether agents can distinguish a real physical relationship from random measurement variation.

## Models Evaluated

The study evaluated nine commercial cloud models, four small open-source models, and two non-LLM baselines.

### Commercial Models

The commercial models were accessed through the RapidScreen API using LiteLLM.

- `gpt-6-sol`
- `gpt-6-luna`
- `gpt-5.6-terra`
- `gpt-5.5`
- `gpt-5.4`
- `gpt-5.4-mini`
- `gpt-5.4-nano`
- `deepseek-v4-flash`
- `deepseek-v4.1-flash`

### Open-Source Models

The open-source models were run locally through Ollama during early pipeline testing on Hooke’s law.

- `qwen2.5-coder:7b`
- `qwen2.5:3b`
- `llama3.2:3b`
- `gemma2:2b`
- `phi3.5:3.8b`

### Non-LLM Baselines

Two non-language-model baselines provide context for LLM performance:

- **Hooke assumption baseline:** Always outputs standard Hooke’s law, regardless of the data. It tests whether the model is actually reasoning rather than defaulting to a textbook equation.
- **Power-law curve-fit baseline:** Fits a power-law curve directly to the data without language-based reasoning. It separates blind statistical curve-fitting from scientific discovery.

## Recursive Self-Improvement

Recursive self-improvement (RSI) was tested by allowing `gpt-5.4-nano` to learn from its own prior attempts.

The process had two phases:

1. **Practice phase.**  
   The model completed three rounds on one redesigned law. Afterward, it wrote strategy notes describing what worked and which mistakes to avoid.

2. **Testing phase.**  
   The model was then tested on unseen physics laws while using its self-written notes.

The RSI experiment compares:

- `gpt-5.4-nano` without notes
- `gpt-5.4-nano` with its own notes
- `gpt-5.5` without notes, as a large-model baseline

This comparison does not isolate the effect of RSI alone, because it also reflects differences in base model capability. However, it tests whether lightweight self-reflection can help a small model approach the performance of a much larger model.

In the reported experiment, self-written notes did **not** improve performance. The notes may have reinforced strategies suited to the spring law—such as measuring at zero and using symmetric positive and negative inputs—that did not transfer well to positive-only laws.

## Multi-Agent System

A two-agent CrewAI system was built using CrewAI 1.15.43. The agents collaborate to discover and validate a physical law.

| Agent | Role | Responsibility |
|---|---|---|
| **Theorist** | Theoretical Physicist | Proposes, refines, and reports candidate physical laws with reasoning |
| **Analyst** | Computational Analyst | Designs tests, evaluates hypotheses, performs calculations, and scores candidates from 0 to 1 |

The agents have access to:

- A SymPy mathematical solver and simplifier
- A NumPy curve-fitting tool

The workflow is:

1. The theorist proposes one or more candidate laws.
2. The analyst designs tests for each hypothesis.
3. The analyst evaluates the hypothesis and assigns a score from 0 to 1.
4. If the score is above 0.8, the hypothesis is accepted.
5. If it is 0.8 or below, the agents refine the hypothesis and repeat the process.
6. Once accepted, the system writes a final report containing the discovered law and equation.

A limitation is that hypotheses are evaluated after experiments are conducted. Incorrect hypotheses can therefore consume additional tokens and execution time. The setup is also a relatively simple two-agent system and may not represent more sophisticated multi-agent architectures.

## Evaluation Metrics

### Pass Rate

The primary metric is **pass rate**:

\[
\text{Pass rate} = \frac{\text{Number of equations passing the SymPy checks}}{\text{Total number of runs}}
\]

A response passes if it is symbolically equivalent to the hidden law or falls within the numerical error tolerance for its experimental design.

### Median Error

Median error is used alongside pass rate because pass rate alone treats a close approximation and a completely incorrect equation identically. Median error is preferred over mean error because occasional parsing failures can create extreme numerical outliers.

### Milestones

Five milestones measure where models fail in the discovery process:

| Milestone | Meaning |
|---|---|
| **M1 — Answer** | The model submitted a formula that the grader could read |
| **M2 — Shape** | Normalized error on 60 unseen points is below 100% |
| **M3 — Close** | Error is 10% or less |
| **M4 — Pass** | Error is within the design-specific tolerance |
| **M5 — Exact** | SymPy confirms exact symbolic equivalence with the hidden law |

### Efficiency

Efficiency is measured using:

- Task completion rate
- Median execution time per run
- Estimated token consumption per run
- Resource usage across experimental designs

## Results Summary

### Overall Pass Rates

| Model | Design S | Design A | Design B | Design C |
|---|---:|---:|---:|---:|
| `gpt-6-sol` | 94% | 100% | 90% | 100% |
| `gpt-5.5` | 100% | 100% | 10% | 96% |
| `gpt-5.6-terra` | 93% | 100% | 30% | 94% |
| `deepseek-v4.1-flash` | 100% | 98% | 70% | 90% |
| `gpt-6-luna` | 72% | 100% | 40% | 78% |
| `deepseek-v4-flash` | 51% | 86% | 50% | 84% |
| `gpt-5.4` | 100% | 76% | 0% | 40% |
| `gpt-5.4-mini` | 80% | 48% | 0% | 20% |
| `gpt-5.4-nano` | 30% | 30% | 0% | 0% |

### Open-Source Models

| Model | Design S | Design A |
|---|---:|---:|
| `qwen2.5:3b` | 0/50 | 0/50 |
| `llama3.2:3b` | 0/50 | 0/50 |
| `gemma2:2b` | 0/50 | 0/50 |
| `phi3.5:3.8b` | 0/50 | 0/50 |

### Multi-Agent Results

| Model | Setup | Total passes |
|---|---|---:|
| `gpt-6-luna` | Single prompt | 40/50 |
| `gpt-6-luna` | CrewAI, two agents | 40/50 |
| `deepseek-v4-flash` | Single prompt | 25/50 |
| `deepseek-v4-flash` | CrewAI, two agents | 38/50 |

CrewAI improved the weaker DeepSeek model but did not improve GPT-6-Luna. It also substantially increased runtime and estimated token consumption.

## Repository Structure

```text
MYSSP-PROJECT-A-3/
├── README.md
├── requirements.txt
├── src/
│   ├── sandbox/
│   │   ├── laws.py
│   │   ├── spring_simulation.py
│   │   ├── static_sweeps.py
│   │   └── telemetry.py
│   ├── agents/
│   │   ├── single_agent.py
│   │   ├── self_improvement.py
│   │   └── crewai_agents.py
│   ├── evaluation/
│   │   ├── latex_parser.py
│   │   ├── sympy_grader.py
│   │   ├── numerical_checks.py
│   │   └── milestones.py
│   └── analysis/
│       ├── results.py
│       └── plots.py
├── data/
│   ├── design_s/
│   ├── design_a/
│   ├── design_b/
│   └── design_c/
├── logs/
│   ├── runs/
│   └── rsi_notes/
├── notebooks/
├── frontend/
│   └── threejs_replay/
└── results/
    ├── tables/
    └── figures/
```

> Adjust the folder names above to match the final repository layout.

## Installation

### Requirements

- Python 3.13.1 or later
- Pip
- Git
- Ollama, if running local open-source models
- API access for commercial models through LiteLLM / RapidScreen

### Install Dependencies

```bash
git clone [https://github.com/Pyth66666/MYSSP-PROJECT-A-3.git](https://github.com/Pyth66666/MYSSP-PROJECT-A-3.git)
cd MYSSP-PROJECT-A-3

python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

Likely core dependencies include:

```text
numpy
pandas
sympy
matplotlib
litellm
crewai
ollama
streamlit
```

## Configuration

Create a `.env` file in the repository root:

```env
RAPIDSCREEN_API_KEY=your_api_key
OPENAI_API_KEY=your_api_key
DEEPSEEK_API_KEY=your_api_key
```

Do not commit API keys, `.env` files, raw API logs, or model outputs containing sensitive information.

## Running the Benchmark

Example commands may need adjustment to match the final code structure.

### Generate Sandbox Data

```bash
python src/sandbox/static_sweeps.py --law spring --design S
python src/sandbox/spring_simulation.py --design B
```

### Run a Single-Agent Evaluation

```bash
python src/agents/single_agent.py \
  --model gpt-6-sol \
  --design A \
  --law spring \
  --runs 10
```

### Run the Noisy-Lab Benchmark

```bash
python src/agents/single_agent.py \
  --model gpt-5.5 \
  --design C \
  --law kinetic_energy \
  --runs 10
```

### Run Recursive Self-Improvement

```bash
python src/agents/self_improvement.py \
  --model gpt-5.4-nano \
  --practice-law spring \
  --test-laws kinetic_energy,wave_speed,screened_coulomb,saturating_ohm \
  --rounds 3
```

### Run the CrewAI Multi-Agent System

```bash
python src/agents/crewai_agents.py \
  --model deepseek-v4-flash \
  --design S \
  --law spring
```

### Grade Submitted Equations

```bash
python src/evaluation/sympy_grader.py \
  --logs logs/runs \
  --output results/tables/grading_results.csv
```

## Output Format

Every run is saved as a JSON log containing:

- The task objective
- Each action and experimental result
- The full agent conversation
- The submitted final equation
- The grading result
- Execution time
- Token usage
- Experimental design
- Model name
- Hidden law identifier

Design S answers are stored in full so that they can be regraded without making new API calls.

## 3D Visualization

A browser-based Three.js interface can replay saved experimental runs. It visualizes trajectories and tracks per-law leaderboards, making it easier to inspect how an agent explored the sandbox and where its reasoning failed.

## Limitations

- The benchmark uses only single-input laws; it does not test multi-variable or coupled differential-equation systems.
- No real laboratory data are used; all experiments occur in simulation.
- No model was trained or fine-tuned.
- “Self-improvement” refers to written strategy notes, not parameter updates or reinforcement learning.
- Commercial models were accessed through an API, so model versions and behavior may change over time.
- Some Design S evaluations were incomplete because of an API-key change; five models were evaluated only on kinetic energy.
- Design B was tested only on the spring law.
- The RSI experiment used a limited number of runs, reducing statistical power.
- CrewAI token counts are estimates because the API does not report token usage directly to CrewAI.
- API content-filter blocks and strict LaTeX parsing failures were counted as task failures.

## Future Work

Planned extensions include:

- Completing all five laws across all nine commercial models in Design S.
- Testing larger open-weight models, such as 32B to 70B+ parameter models, to identify the point at which open-source models begin discovering laws reliably.
- Extending passive trajectory inference to coupled multi-variable systems.
- Adding tool-augmented reflection, allowing compact models to run regression checks and receive residual feedback.
- Developing hybrid pipelines that use fast single-prompt inference and trigger multi-agent verification only when out-of-sample error exceeds a threshold.
- Testing models without explicit task instructions to better measure fully autonomous scientific behavior.

## Acknowledgments

Thanks to mentor **Fan Xiaoyan** for valuable guidance, and to **MyResearchGuide** for resources and programme organization.

## License

Add an appropriate open-source license before publishing the repository, for example:

```text
MIT License
```

## Citation

If you use this benchmark or build on this work, please cite the original research paper:

```bibtex
@misc{prasad2026benchmarking,
  title={Benchmarking Self-Evolving AI Agents for Automated Physics Discovery},
  author={Ananya Prasad and Tang Yi Zhe and Muhammad Adief Al Syarif bin Abdullah},
  year={2026},
  howpublished={Malaysia Science Scholar’s Programme 2026},
  url={[https://github.com/Pyth66666/MYSSP-PROJECT-A-3](https://github.com/Pyth66666/MYSSP-PROJECT-A-3)}
}
```
