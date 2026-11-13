"""Run every model on every law for one design, and pick up where it stopped.

    py run_all.py --design A
    py run_all.py --design C
    py run_all.py --design B      (spring only, B needs a moving block)

A (law, design, model) group counts as done when it has 10 saved runs that
reached the model. Runs that failed only because the internet dropped are not
counted, so they are run again. Before each group the script waits until the
API address can be found, so a sleeping laptop or lost Wi-Fi does not turn
into a wall of failed runs.
"""
import argparse
import glob
import json
import os
import socket
import subprocess
import sys
import time

import agent as agents
from laws import LAWS, TEAM_LAWS

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "results", "runs")
NET_ERRORS = ("urlopen error", "getaddrinfo", "timed out", "Connection", "RemoteDisconnected")


def good_runs(law, design, name):
    n = 0
    for path in glob.glob(os.path.join(RUNS, "*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:  # noqa: BLE001
            continue
        if d.get("label", "experiment") != "experiment":
            continue  # self improvement runs are counted separately
        if d.get("law") != law or d.get("design") != design or d.get("agent") != name:
            continue
        ans = str(d.get("answer") or "")
        if ans.startswith("agent error") and any(e in ans for e in NET_ERRORS):
            continue
        n += 1
    return n


def wait_for_net(host="litellm.rapidscreen.io"):
    while True:
        try:
            socket.getaddrinfo(host, 443)
            return
        except OSError:
            print(time.strftime("%H:%M:%S"), "no internet, waiting 60 s ...", flush=True)
            time.sleep(60)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--design", required=True, choices=["A", "B", "C"])
    p.add_argument("--laws", default="", help="comma list, default: the five team laws")
    p.add_argument("--seeds", type=int, default=10)
    a = p.parse_args()
    laws = [x for x in (a.laws.split(",") if a.laws else TEAM_LAWS) if x]
    if a.design == "B":
        laws = [x for x in laws if LAWS[x].is_spring]
    names = [g.name for g in agents.all_agents() if g.name.startswith("RapidScreen")]
    if not names:
        sys.exit("No RapidScreen models found. Set RAPIDSCREEN_API_KEY in this window first.")
    print("models:", ", ".join(names))
    for rnd in range(3):
        todo = [(L, n) for L in laws for n in names if good_runs(L, a.design, n) < a.seeds]
        print(f"\npass {rnd + 1}: {len(todo)} groups left for design {a.design}", flush=True)
        if not todo:
            break
        for L, n in todo:
            wait_for_net()
            print(f"\n=== {L} design {a.design} {n} ===", flush=True)
            subprocess.run([sys.executable, os.path.join(HERE, "experiment.py"), "--law", L,
                            "--design", a.design, "--agents", n, "--seeds", str(a.seeds)])
    left = [(L, n) for L in laws for n in names if good_runs(L, a.design, n) < a.seeds]
    print("\nALL DONE for design", a.design if not left else f"{a.design}, still missing: {left}")


if __name__ == "__main__":
    main()
