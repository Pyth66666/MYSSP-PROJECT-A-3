"""Add rows to results/experiments.tsv for saved run logs that have no row.

experiment.py writes the TSV when a whole batch ends. If a batch is stopped
partway (Ctrl+C, laptop asleep), its finished runs are saved in results/runs
but never reach the TSV, so the leaderboard cannot see them. This script adds
them back from their logs. Run it when no experiment is running.

    py repair_tsv.py
"""
import csv, glob, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
TSV = os.path.join(HERE, "results", "experiments.tsv")
with open(TSV, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    fields = reader.fieldnames
    have = {os.path.basename(r["log_file"].replace("\\", "/")) for r in reader}
new = []
for path in sorted(glob.glob(os.path.join(HERE, "results", "runs", "*.json"))):
    name = os.path.basename(path)
    if name in have:
        continue
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    r = d.get("result") or {}
    new.append({"timestamp": d["timestamp"], "label": d.get("label", "experiment"), "agent": d["agent"],
                "design": d["design"], "law": d["law"], "seed": d["seed"], "passed": r.get("passed"),
                "exact": r.get("exact"), "nrmse": r.get("nrmse"),
                "experiments_used": r.get("experiments_used", len(d.get("experiments") or [])),
                "seconds": d.get("seconds"), "tokens": d.get("tokens"), "answer": d.get("answer"),
                "reason": r.get("reason"), "log_file": "runs\\" + name})
with open(TSV, "a", newline="", encoding="utf-8") as f:
    csv.DictWriter(f, fieldnames=fields, delimiter="\t").writerows(new)
print(f"added {len(new)} rows from saved logs")
