"""Evaluation pipeline (the team diagram):

    [Sandbox] -> saves CSV data -> [Evaluation pipeline] -> sends prompt + data
    -> [Benchmark agent] -> extracts formula -> checks math -> [SymPy evaluator]

    python generate_data.py                       # once: make the 50 CSV datasets
    python pipeline.py --list                     # which agents can run now
    python pipeline.py                            # every available agent, all datasets
    python pipeline.py --agents "Ollama qwen2.5-coder:7b (single prompt)" --datasets 10

For each CSV: take 20 evenly spaced (position, force) rows, put them in the
team prompt, send it to the agent, extract the formula from its answer and
score it with the SymPy evaluator (grader.py).

Output:
    results/pipeline.tsv          one row per (agent, dataset), appended
    results/pipeline_latest.json  last run with answers, read by the 3D viewer
"""

import argparse
import csv
import json
import os
import time

import agent as agents
from generate_data import load_dataset, write_datasets
from laws import DEFAULT_LAW, LAWS
from sandbox_env import LabSession

HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = ["timestamp", "law", "dataset", "x0", "agent", "kind", "passed", "exact", "nrmse",
          "seconds", "tokens", "parsed", "reason", "answer"]


def datasets_for(law, limit):
    folder = os.path.join(HERE, "data", law)
    index_path = os.path.join(folder, "index.csv")
    if not os.path.exists(index_path):
        print(f"No CSV data for '{law}' yet, generating it now...")
        write_datasets(law, 50)
    with open(index_path, newline="", encoding="utf-8") as f:
        index = list(csv.DictReader(f))
    return [(int(r["dataset"]), float(r["x0"]), os.path.join(folder, r["file"]))
            for r in index[:limit]]


def evaluate(ag, law, dataset_id, path):
    rows = load_dataset(path)
    lab = LabSession("S", dataset_id, law, dataset=rows)
    start = time.time()
    try:
        answer, _notes = ag.run(lab)
    except Exception as exc:  # noqa: BLE001 - a crashed agent is a failed run
        answer = f"agent error: {exc}"
    seconds = round(time.time() - start, 2)
    result = lab.submit(answer)
    return answer, result, seconds


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--law", default=DEFAULT_LAW, choices=list(LAWS))
    p.add_argument("--datasets", type=int, default=50, help="how many CSV datasets to use")
    p.add_argument("--agents", default="", help="comma list of agent names (default: all)")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()

    pool = [ag for ag in agents.all_agents() if "S" in ag.designs]
    if a.list:
        for ag in pool:
            print(f"  {ag.kind:<9} {ag.name}")
        print("\nModels in models.json that are not pulled / have no key are not listed.")
        return
    if a.agents:
        wanted = {n.strip().lower() for n in a.agents.split(",")}
        pool = [ag for ag in pool if ag.name.lower() in wanted]
        if not pool:
            raise SystemExit("none of those agents is available, see: python pipeline.py --list")

    sets = datasets_for(a.law, a.datasets)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for ag in pool:
        for dataset_id, x0, path in sets:
            answer, r, seconds = evaluate(ag, a.law, dataset_id, path)
            row = {"timestamp": stamp, "law": a.law, "dataset": dataset_id, "x0": x0,
                   "agent": ag.name, "kind": ag.kind, "passed": r["passed"],
                   "exact": r["exact"], "nrmse": r["nrmse"], "seconds": seconds,
                   "tokens": getattr(ag, "tokens", 0), "parsed": r["parsed"],
                   "reason": r["reason"], "answer": answer}
            rows.append(row)
            print(f"{ag.name:<42} dataset {dataset_id:>2} (x0 {x0:+.2f})  "
                  f"{'PASS' if r['passed'] else 'FAIL'}  {r['reason']}")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    tsv = os.path.join(HERE, "results", "pipeline.tsv")
    new_file = not os.path.exists(tsv)
    with open(tsv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        if new_file:
            w.writeheader()
        w.writerows(rows)

    summary = []
    for name in dict.fromkeys(r["agent"] for r in rows):
        sub = [r for r in rows if r["agent"] == name]
        errs = sorted(r["nrmse"] for r in sub if r["nrmse"] is not None)
        summary.append({
            "agent": name, "datasets": len(sub),
            "pass_rate": sum(r["passed"] for r in sub) / len(sub),
            "exact_rate": sum(bool(r["exact"]) for r in sub) / len(sub),
            "median_nrmse": errs[len(errs) // 2] if errs else None,
            "median_seconds": sorted(r["seconds"] for r in sub)[len(sub) // 2],
        })
    with open(os.path.join(HERE, "results", "pipeline_latest.json"), "w", encoding="utf-8") as f:
        json.dump({"timestamp": stamp, "law": a.law, "summary": summary, "runs": rows}, f, indent=1)

    print(f"\n{'agent':<42} {'pass':>6} {'exact':>6} {'median err':>11}")
    for s in summary:
        med = "-" if s["median_nrmse"] is None else f"{s['median_nrmse']:.2%}"
        print(f"{s['agent']:<42} {s['pass_rate']:>6.0%} {s['exact_rate']:>6.0%} {med:>11}")
    print(f"\nSaved results/pipeline.tsv and results/pipeline_latest.json ({len(rows)} runs)")


if __name__ == "__main__":
    main()
