"""Run every agent on every sandbox design and save one comparison table.

Usage (from the lab folder):
    python bench.py                         # designs S,A,B,C, 3 seeds, all available agents
    python bench.py --designs A,B --seeds 5
    python bench.py --law hooke             # control task with normal Hooke's law
    python bench.py --list                  # show which agents are available

Output:
    results/results.tsv   one row per run, appended (keeps history)
    results/latest.json   the last benchmark, read by the 3D viewer
"""

import argparse
import csv
import json
import os
import time

import agent as agents
from laws import LAWS, DEFAULT_LAW
from sandbox_env import DESIGNS, LabSession

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIELDS = ["timestamp", "law", "design", "seed", "agent", "kind", "passed", "exact", "nrmse",
          "experiments_used", "seconds", "tokens", "answer", "parsed", "reason"]


def run_one(ag, design, seed, law):
    lab = LabSession(design, seed, law)
    try:
        answer, notes, seconds = agents.timed_run(ag, lab)
    except Exception as exc:  # noqa: BLE001 - a crashed agent is a failed run
        answer, notes, seconds = "", [f"agent error: {exc}"], 0.0
    result = lab.submit(answer)
    return {
        "law": law, "design": design, "seed": seed, "agent": ag.name, "kind": ag.kind,
        "passed": result["passed"], "exact": result["exact"], "nrmse": result["nrmse"],
        "experiments_used": result["experiments_used"], "seconds": seconds,
        "tokens": getattr(ag, "tokens", 0), "answer": answer,
        "parsed": result["parsed"], "reason": result["reason"],
        "notes": notes, "log": lab.to_dict()["actions"], "truth": result["truth"],
    }


def summarise(rows):
    table = {}
    for r in rows:
        key = (r["agent"], r["design"])
        t = table.setdefault(key, {"runs": 0, "passed": 0, "nrmse": []})
        t["runs"] += 1
        t["passed"] += int(r["passed"])
        if r["nrmse"] is not None:
            t["nrmse"].append(r["nrmse"])
    out = []
    for (name, design), t in table.items():
        med = sorted(t["nrmse"])[len(t["nrmse"]) // 2] if t["nrmse"] else None
        out.append({"agent": name, "design": design, "runs": t["runs"],
                    "pass_rate": t["passed"] / t["runs"], "median_nrmse": med})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--designs", default="S,A,B,C")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--law", default=DEFAULT_LAW, choices=list(LAWS))
    p.add_argument("--agents", default="", help="comma list of agent names to run (default: all)")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    pool = agents.all_agents()
    if args.list:
        for a in pool:
            print(f"  {a.kind:<9} {a.name}")
        print("\nLLM entries in models.json without an API key or a running server are skipped.")
        return
    if args.agents:
        wanted = {n.strip().lower() for n in args.agents.split(",")}
        pool = [a for a in pool if a.name.lower() in wanted]

    designs = [d.strip().upper() for d in args.designs.split(",") if d.strip()]
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for ag in pool:
        for d in designs:
            if d not in DESIGNS:
                raise SystemExit(f"unknown design {d}")
            if d not in ag.designs:
                continue
            for seed in range(args.seeds):
                row = run_one(ag, d, seed, args.law)
                row["timestamp"] = stamp
                rows.append(row)
                mark = "PASS" if row["passed"] else "FAIL"
                print(f"{ag.name:<30} design {d} seed {seed}  {mark}  {row['reason']}")

    os.makedirs(RESULTS, exist_ok=True)
    tsv = os.path.join(RESULTS, "results.tsv")
    new_file = not os.path.exists(tsv)
    with open(tsv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerows(rows)
    summary = summarise(rows)
    with open(os.path.join(RESULTS, "latest.json"), "w", encoding="utf-8") as f:
        json.dump({"timestamp": stamp, "law": args.law, "seeds": args.seeds,
                   "summary": summary, "runs": rows}, f, indent=1)

    print("\nPass rate by agent and design")
    print(f"  {'agent':<30} {'design':<7} {'pass':>6} {'median NRMSE':>13}")
    for s in summary:
        med = "-" if s["median_nrmse"] is None else f"{s['median_nrmse']:.2%}"
        print(f"  {s['agent']:<30} {s['design']:<7} {s['pass_rate']:>6.0%} {med:>13}")
    print(f"\nSaved results/results.tsv and results/latest.json ({len(rows)} runs)")


if __name__ == "__main__":
    main()
