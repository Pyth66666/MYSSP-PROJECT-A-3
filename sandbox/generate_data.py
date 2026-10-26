"""Sandbox step of the team pipeline: simulate the spring and save CSV data.

    python generate_data.py                 # 50 datasets for the redesigned law
    python generate_data.py --n 50 --law hooke

Each dataset is one release of the 1 kg block, simulated exactly like the
team script (physics law redesign.py): dt = 0.1 s for 10 s, semi-implicit
Euler. Only the release position x0 changes between datasets:
|x0| = 0.5 ... 3.0 m on both sides (dataset 0 is x0 = 2.0 m, as in the script).

Output, per law:
    data/<law>/dataset_00.csv ... dataset_49.csv
        columns: time, position, velocity, acceleration, force
    data/<law>/index.csv
        dataset, file, x0, rows, dt
The CSV files never contain the formula.
"""

import argparse
import csv
import os

from laws import DEFAULT_LAW, LAWS, get_law
from sandbox_env import STATIC_DT, static_dataset, static_x0

HERE = os.path.dirname(os.path.abspath(__file__))
COLUMNS = ["time", "position", "velocity", "acceleration", "force"]


def write_datasets(law_name, n):
    law = get_law(law_name)
    folder = os.path.join(HERE, "data", law.name)
    os.makedirs(folder, exist_ok=True)
    index = []
    for i in range(n):
        rows = static_dataset(law, i)
        name = f"dataset_{i:02d}.csv"
        with open(os.path.join(folder, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        index.append({"dataset": i, "file": name, "x0": static_x0(i),
                      "rows": len(rows), "dt": STATIC_DT})
    with open(os.path.join(folder, "index.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "file", "x0", "rows", "dt"])
        w.writeheader()
        w.writerows(index)
    return folder, index


def load_dataset(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=50, help="number of datasets (default 50)")
    p.add_argument("--law", default=DEFAULT_LAW, choices=list(LAWS))
    a = p.parse_args()
    folder, index = write_datasets(a.law, a.n)
    print(f"Wrote {len(index)} datasets ({index[0]['rows']} rows each) to {folder}")


if __name__ == "__main__":
    main()
