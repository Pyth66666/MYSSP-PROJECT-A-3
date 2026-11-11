"""Autonomous experiments: the agent runs the whole experiment by itself.

Every run starts from ONE fixed objective prompt (OBJECTIVE below). The agent
then decides which experiments to do, reads each result, decides the next
step and submits a formula, with nobody stepping in. The sandbox grades the
answer. Each run is saved as a full log, including every model reply.

    python experiment.py --list
    python experiment.py --agents "RapidScreen gpt-5.5" --seeds 10
    python experiment.py --agents "Ollama qwen2.5-coder:7b (single prompt)" --seeds 2
    python experiment.py --design C --seeds 10            # noisy lab
    python experiment.py --notes results/rsi/<run>/best_notes.txt   # with strategy notes

Output (in --out, default results/):
    experiments.tsv        one row per run, appended (history of every run)
    runs/<...>.json        full log of each run: objective, task card, every
                           experiment and observation, the whole conversation,
                           the answer and the grade

Note on seeds: in designs S and A there is no noise, so every seed is the same
task. Repeating it shows how RELIABLE a model is (models answer differently
from run to run). In designs B and C the seed also changes the noise.
"""

import argparse
import csv
import json
import os
import re
import statistics
import time

import agent as agents
from laws import DEFAULT_LAW, LAWS
from sandbox_env import DESIGNS, LabSession

HERE = os.path.dirname(os.path.abspath(__file__))

OBJECTIVE = """OBJECTIVE OF THIS EXPERIMENT
Find the exact formula for the spring force F(x).
You run the experiment yourself: decide which measurements to take, look at
each result, and decide what to do next. Nobody will help you during the run.
When you are confident, submit one formula.
Success means your formula matches the hidden law within the pass rule on
positions you did not measure, including both signs of x."""

def objective_for(law):
    """The fixed objective prompt. Spring laws keep the original text, so new runs
    compare directly with the earlier ones; other laws name their own variables."""
    if law.is_spring:
        return OBJECTIVE
    return OBJECTIVE.replace(
        "Find the exact formula for the spring force F(x).",
        f"Find the exact formula for the {law.y_name} {law.y_sym} as a function of the "
        f"{law.x_name} {law.x_sym}.").replace(
        "including both signs of x.", f"across the whole allowed range of {law.x_sym}.")


FIELDS = ["timestamp", "label", "agent", "design", "law", "seed", "passed", "exact",
          "nrmse", "experiments_used", "seconds", "tokens", "answer", "reason", "log_file"]


def slug(text):
    return re.sub(r"[^A-Za-z0-9.]+", "-", text).strip("-")[:40]


def pick_agents(names):
    """Available agents by name. names = comma list, '' = all available."""
    pool = agents.all_agents()
    if not names:
        return pool
    wanted = [n.strip() for n in names.split(",") if n.strip()]
    found = [a for a in pool if a.name in wanted]
    missing = [n for n in wanted if n not in {a.name for a in found}]
    if missing:
        raise SystemExit(
            "Not available right now: " + ", ".join(missing) +
            "\nCheck the name with --list. API models need their key set in this "
            "window, Ollama models need 'ollama pull'.")
    return found


def run_one(ag, design, law, seed, notes_text, out_dir, label=""):
    """One autonomous run. Returns the summary row; writes the full log."""
    lab = LabSession(design, seed, law)
    is_llm = isinstance(ag, agents.LLMAgent)
    objective = objective_for(lab.law)
    if is_llm:
        extra = objective
        if notes_text:
            extra += "\n\nSTRATEGY NOTES (written by you after earlier experiments)\n" + notes_text
        ag.extra_system = extra
    start = time.time()
    crashed = False
    try:
        answer, run_notes = ag.run(lab)
    except Exception as exc:  # noqa: BLE001 - a crashed agent is a failed run
        answer, run_notes, crashed = f"agent error: {exc}", [f"agent error: {exc}"], True
    seconds = round(time.time() - start, 2)
    if is_llm:
        ag.extra_system = ""
    if crashed:
        result = {"passed": False, "exact": None, "nrmse": None, "parsed": None,
                  "reason": answer, "truth": lab.law.formula,
                  "experiments_used": len(lab.log)}
    else:
        result = lab.submit(answer)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(os.path.join(out_dir, "runs"), exist_ok=True)
    log_name = f"{stamp}_{slug(ag.name)}_{design}_{law}_s{seed}.json"
    log = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "label": label,
        "agent": ag.name,
        "model": getattr(ag, "model", None),
        "design": design, "law": law, "seed": seed,
        "objective": objective if is_llm else None,
        "strategy_notes": notes_text or None,
        "task_card": lab.card(),
        "experiments": lab.log,
        "conversation": list(getattr(ag, "transcript", []) or []),
        "agent_notes": run_notes,
        "answer": answer,
        "result": result,
        "seconds": seconds,
        "tokens": getattr(ag, "tokens", 0),
    }
    with open(os.path.join(out_dir, "runs", log_name), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=1, ensure_ascii=False)
    return {
        "timestamp": log["timestamp"], "label": label, "agent": ag.name,
        "design": design, "law": law, "seed": seed,
        "passed": result["passed"], "exact": result.get("exact"),
        "nrmse": result.get("nrmse"), "experiments_used": len(lab.log),
        "seconds": seconds, "tokens": log["tokens"], "answer": answer,
        "reason": result.get("reason", ""), "log_file": os.path.join("runs", log_name),
    }


def append_rows(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "experiments.tsv")
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore")
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow(dict(r, answer=str(r["answer"]).replace("\n", " ")[:2000]))


def summarise(rows):
    """pass rate, exact rate, median error, mean experiments, tokens."""
    if not rows:
        return {"runs": 0, "pass_rate": 0.0, "exact_rate": 0.0, "median_nrmse": None,
                "mean_experiments": 0.0, "tokens": 0}
    errs = [r["nrmse"] for r in rows if r["nrmse"] is not None]
    return {
        "runs": len(rows),
        "pass_rate": sum(bool(r["passed"]) for r in rows) / len(rows),
        "exact_rate": sum(bool(r["exact"]) for r in rows) / len(rows),
        "median_nrmse": statistics.median(errs) if errs else None,
        "mean_experiments": sum(r["experiments_used"] for r in rows) / len(rows),
        "tokens": sum(int(r["tokens"] or 0) for r in rows),
    }


def run_batch(ag, design, laws, seeds, notes_text, out_dir, label="", verbose=True):
    rows = []
    for law in laws:
        for seed in range(seeds):
            r = run_one(ag, design, law, seed, notes_text, out_dir, label)
            rows.append(r)
            if verbose:
                verdict = "PASS" if r["passed"] else "FAIL"
                err = "-" if r["nrmse"] is None else f"{r['nrmse']:.2%}"
                print(f"  {ag.name[:34]:<34} {design} {law:<16} seed {seed:<2} {verdict}  "
                      f"error {err:>8}  {r['experiments_used']:>2} exp  {r['seconds']:>6}s  "
                      f"{str(r['answer']).splitlines()[0][:60] if r['answer'] else ''}")
    append_rows(rows, out_dir)
    return rows


def print_summary(by_agent):
    print(f"\n{'agent':<40} {'runs':>4} {'pass':>6} {'exact':>6} {'median err':>11} "
          f"{'avg exp':>8} {'tokens':>9}")
    for name, rows in by_agent.items():
        s = summarise(rows)
        med = "-" if s["median_nrmse"] is None else f"{s['median_nrmse']:.2%}"
        print(f"{name[:40]:<40} {s['runs']:>4} {s['pass_rate']:>6.0%} {s['exact_rate']:>6.0%} "
              f"{med:>11} {s['mean_experiments']:>8.1f} {s['tokens']:>9}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--design", default="A", choices=[d for d in DESIGNS if d != "S"])
    p.add_argument("--law", default=DEFAULT_LAW, choices=list(LAWS))
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--agents", default="", help="comma list of agent names (default: all available)")
    p.add_argument("--notes", default="", help="text file with strategy notes to give LLM agents")
    p.add_argument("--out", default=os.path.join(HERE, "results"))
    p.add_argument("--list", action="store_true")
    a = p.parse_args()

    pool = pick_agents(a.agents)
    if a.list:
        for ag in pool:
            print(f"  {ag.kind:<9} {ag.name}")
        return
    notes_text = ""
    if a.notes:
        with open(a.notes, encoding="utf-8") as f:
            notes_text = f.read().strip()
    if a.design == "B" and not LAWS[a.law].is_spring:
        raise SystemExit(f"Design B needs a moving block, so it only works with spring laws. "
                         f"Use --design A or C for '{a.law}'.")
    print(f"Design {a.design}, law {a.law}, {a.seeds} seed(s) per agent. Objective:\n")
    print("  " + objective_for(LAWS[a.law]).replace("\n", "\n  ") + "\n")
    by_agent = {}
    for ag in pool:
        if a.design not in ag.designs:
            print(f"  skip {ag.name}: does not run design {a.design}")
            continue
        by_agent[ag.name] = run_batch(ag, a.design, [a.law], a.seeds, notes_text, a.out,
                                      label="experiment")
    print_summary(by_agent)
    print(f"\nSaved {os.path.join(a.out, 'experiments.tsv')} and one log per run in "
          f"{os.path.join(a.out, 'runs')}")


if __name__ == "__main__":
    main()
