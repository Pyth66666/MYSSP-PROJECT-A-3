"""Grade the saved design S answers in results/pipeline.tsv again with the current grader.

Use it after the grader learns to read a new answer style. The models' answers are
not changed and no API call is made, only the grading columns are rewritten. The
old file is kept as results/pipeline_before_regrade.tsv. Run it when pipeline.py
is not running.

    py regrade_pipeline.py              regrade every row
    py regrade_pipeline.py 2026-11-14   regrade rows from that date on
"""
import csv, os, shutil, sys
import grader
from laws import LAWS

HERE = os.path.dirname(os.path.abspath(__file__))
TSV = os.path.join(HERE, "results", "pipeline.tsv")
since = sys.argv[1] if len(sys.argv) > 1 else ""
with open(TSV, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    fields, rows = reader.fieldnames, list(reader)
shutil.copyfile(TSV, os.path.join(HERE, "results", "pipeline_before_regrade.tsv"))
changed = 0
for r in rows:
    if r["timestamp"] < since or r["law"] not in LAWS or r["answer"].startswith("agent error"):
        continue
    g = grader.grade(r["answer"], LAWS[r["law"]], 0.01)
    new = {"passed": str(g["passed"]), "exact": str(g.get("exact")), "nrmse": g.get("nrmse"),
           "parsed": g.get("parsed"), "reason": g["reason"]}
    if str(r["passed"]) != new["passed"]:
        changed += 1
    r.update(new)
with open(TSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    w.writeheader()
    w.writerows(rows)
print(f"regraded, {changed} rows changed pass or fail. Old file: results/pipeline_before_regrade.tsv")
