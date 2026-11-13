"""Leaderboard of every model, from the run history in results/experiments.tsv.

For each model and design it uses only the LATEST runs (default: the 10 most
recent), so older batches, such as runs from before a fix, are not mixed in.
Self-improvement runs (rsi.py) are left out unless --include-rsi is given.

    python compare.py                       # all designs, latest 10 runs each
    python compare.py --design C            # one design
    python compare.py --latest 5            # use the 5 most recent runs
    python compare.py --since "2026-11-01 01:00"

Also writes results/leaderboard.csv (opens in Excel).
"""

import argparse
import csv
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def as_bool(v):
    return str(v).strip().lower() == "true"


def as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def leaderboard(out_dir, law="power_spring", latest=10, designs=None, include_rsi=False):
    """Rows for the leaderboard, best first per design (used by the 3D page)."""
    path = os.path.join(out_dir, "experiments.tsv")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    groups = defaultdict(list)
    for r in rows:
        if r["law"] != law or (designs and r["design"] not in designs):
            continue
        if not include_rsi and r["label"].startswith("rsi"):
            continue
        groups[(r["design"], r["agent"])].append(r)
    table = []
    for (design, agent), rs in groups.items():
        rs = sorted(rs, key=lambda r: r["timestamp"])[-latest:]
        errs = [as_float(r["nrmse"]) for r in rs if as_float(r["nrmse"]) is not None]
        table.append({
            "design": design, "agent": agent, "runs": len(rs),
            "passes": sum(as_bool(r["passed"]) for r in rs),
            "exacts": sum(as_bool(r["exact"]) for r in rs),
            "median_error": statistics.median(errs) if errs else None,
            "avg_experiments": round(sum(int(r["experiments_used"] or 0) for r in rs) / len(rs), 1),
            "avg_seconds": round(sum(as_float(r["seconds"]) or 0 for r in rs) / len(rs), 1),
            "tokens": sum(int(float(r["tokens"] or 0)) for r in rs),
        })
    table.sort(key=lambda t: (t["design"], -t["passes"], -t["exacts"],
                              t["median_error"] if t["median_error"] is not None else 9,
                              t["avg_experiments"]))
    return table


def saved_runs(out_dir, law="power_spring"):
    """Latest saved autonomous run per (law, agent, design, seed), for replay.
    law=None lists every law."""
    path = os.path.join(out_dir, "experiments.tsv")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    seen, out = set(), []
    for r in sorted(rows, key=lambda r: r["timestamp"], reverse=True):
        if (law and r["law"] != law) or r["label"] != "experiment" or not r.get("log_file"):
            continue
        key = (r["law"], r["agent"], r["design"], r["seed"])
        if key in seen:
            continue
        seen.add(key)
        out.append({"file": r["log_file"].replace("\\", "/"), "agent": r["agent"], "law": r["law"],
                    "design": r["design"], "seed": int(r["seed"]),
                    "passed": as_bool(r["passed"]), "nrmse": as_float(r["nrmse"]),
                    "timestamp": r["timestamp"]})
    out.sort(key=lambda r: (r["law"], r["design"], r["agent"], r["seed"]))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--design", default="", help="e.g. A or A,C (default: all)")
    p.add_argument("--law", default="power_spring")
    p.add_argument("--latest", type=int, default=10, help="most recent runs per model and design")
    p.add_argument("--since", default="", help='only runs at or after this time, "YYYY-MM-DD HH:MM"')
    p.add_argument("--include-rsi", action="store_true")
    p.add_argument("--out", default=os.path.join(HERE, "results"))
    a = p.parse_args()

    path = os.path.join(a.out, "experiments.tsv")
    if not os.path.exists(path):
        raise SystemExit(f"No run history yet: {path}")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    designs = {d.strip() for d in a.design.split(",") if d.strip()}
    groups = defaultdict(list)
    for r in rows:
        if r["law"] != a.law:
            continue
        if designs and r["design"] not in designs:
            continue
        if not a.include_rsi and r["label"].startswith("rsi"):
            continue
        if a.since and r["timestamp"] < a.since:
            continue
        groups[(r["design"], r["agent"])].append(r)

    table = []
    for (design, agent), rs in groups.items():
        rs = sorted(rs, key=lambda r: r["timestamp"])[-a.latest:]
        errs = [as_float(r["nrmse"]) for r in rs if as_float(r["nrmse"]) is not None]
        table.append({
            "design": design, "agent": agent, "runs": len(rs),
            "pass_rate": sum(as_bool(r["passed"]) for r in rs) / len(rs),
            "exact_rate": sum(as_bool(r["exact"]) for r in rs) / len(rs),
            "median_error": statistics.median(errs) if errs else None,
            "avg_experiments": sum(int(r["experiments_used"] or 0) for r in rs) / len(rs),
            "avg_seconds": sum(as_float(r["seconds"]) or 0 for r in rs) / len(rs),
            "tokens": sum(int(float(r["tokens"] or 0)) for r in rs),
            "first_run": rs[0]["timestamp"], "last_run": rs[-1]["timestamp"],
        })
    table.sort(key=lambda t: (t["design"], -t["pass_rate"], -t["exact_rate"],
                              t["median_error"] if t["median_error"] is not None else 9,
                              t["avg_experiments"]))

    for design in sorted({t["design"] for t in table}):
        print(f"\nDesign {design}  (law {a.law}, latest {a.latest} runs per model)")
        print(f"{'model':<36} {'runs':>4} {'pass':>6} {'exact':>6} {'median err':>11} "
              f"{'avg exp':>8} {'avg s':>7} {'tokens':>9}")
        for t in table:
            if t["design"] != design:
                continue
            med = "-" if t["median_error"] is None else f"{t['median_error']:.2%}"
            print(f"{t['agent'][:36]:<36} {t['runs']:>4} {t['pass_rate']:>6.0%} {t['exact_rate']:>6.0%} "
                  f"{med:>11} {t['avg_experiments']:>8.1f} {t['avg_seconds']:>7.1f} {t['tokens']:>9}")

    out_csv = os.path.join(a.out, "leaderboard.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]) if table else ["design"])
        w.writeheader()
        w.writerows(table)
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()
