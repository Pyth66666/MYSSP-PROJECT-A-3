# MYSSP Project A-3

Benchmarking AI agents for automated physics discovery.

| Path | What it is |
|---|---|
| `physics law redesign.py` | The redesigned law F(x) = -5·\|x\|^1.5·sign(x), data generation, example prompt and SymPy grading idea. |
| `sandbox/` | The sandbox and evaluation pipeline: 50 CSV datasets, team prompt, benchmark agents (Ollama, APIs, the CrewAI framework), SymPy evaluator, four sandbox design proposals and a 3D view. See [`sandbox/README.md`](sandbox/README.md). |

Quick start (Windows PowerShell, from the repo folder):

```powershell
cd sandbox
py -m pip install sympy
py -m unittest discover -s tests
py pipeline.py
py server.py        # then open http://127.0.0.1:8000
```
