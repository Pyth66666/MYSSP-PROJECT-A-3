# MYSSP Project A-3

Benchmarking AI agents for automated physics discovery.

| Path | What it is |
|---|---|
| `physics law redesign.py` | The redesigned law F(x) = -5·\|x\|^1.5·sign(x), data generation, example prompt and SymPy grading idea. |
| `kinetic_energy_law2.py` | Law 2: redesigned kinetic energy, E_k = 0.5·m²/v. |
| `wavespeed_law3.py` | Law 3: redesigned wave speed, v = c·ln(f·λ). |
| `electrostatics_law4.py` | Law 4: Coulomb's law with an exponential decay, F = k·q₁q₂/r²·e^(−0.4r). |
| `ohms_law5.py` | Law 5: saturating Ohm's law, I = 5·tanh(V/2). |
| `sandbox/` | The sandbox and evaluation pipeline for all five laws: CSV datasets, team prompt, benchmark agents (Ollama, APIs, the CrewAI framework), SymPy evaluator, four sandbox designs, autonomous experiments, a self-improvement loop, a leaderboard and a 3D view. See [`sandbox/README.md`](sandbox/README.md). |

Quick start (Windows PowerShell, from the repo folder):

```powershell
cd sandbox
py -m pip install sympy
py -m unittest discover -s tests
py pipeline.py
py server.py        # then open http://127.0.0.1:8000
```
