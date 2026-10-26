"""Command-line sandbox for agents that work in a terminal (Claude Code,
Codex, Kilo, or a person). Each command is one step, state is saved to
runs/<session>.json, so the agent can think between steps.

    python cli.py new --design B --seed 0 --agent-name "codex"
    python cli.py exp B-0-1a2b '{"release": {"x0": 2.0}}'
    python cli.py exp B-0-1a2b '{"measure_force": {"x": 1.5}}'   (designs A and C)
    python cli.py submit B-0-1a2b "F(x) = -3*x**3"
    python cli.py show B-0-1a2b

Results of every submit are appended to results/external.tsv, so external
agents land in the same comparison as bench.py.

Fair-play rule: the agent being tested must only use these commands. It must
not open laws.py, grader.py or sandbox_env.py (they contain the hidden law).
"""

import argparse
import csv
import json
import os
import secrets
import sys
import time

from laws import LAWS, DEFAULT_LAW
from sandbox_env import DESIGNS, ExperimentError, LabSession

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")


def _path(sid):
    return os.path.join(RUNS, f"{sid}.json")


def _load(sid):
    try:
        with open(_path(sid), encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        sys.exit(f"no session '{sid}', start one with: python cli.py new")
    return LabSession.from_dict(data), data


def _save(sid, lab, meta):
    os.makedirs(RUNS, exist_ok=True)
    data = lab.to_dict()
    data.update(meta)
    with open(_path(sid), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)


def _print(obj):
    print(json.dumps(obj, indent=1))


def cmd_new(a):
    sid = f"{a.design}-{a.seed}-{secrets.token_hex(2)}"
    lab = LabSession(a.design, a.seed, a.law)
    _save(sid, lab, {"agent_name": a.agent_name, "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    print(f"session: {sid}")
    _print(lab.card())


def cmd_exp(a):
    lab, meta = _load(a.session)
    try:
        action = json.loads(a.action)
        obs = lab.experiment(action)
    except json.JSONDecodeError:
        sys.exit("action must be JSON, e.g. '{\"measure_force\": {\"x\": 1.0}}'")
    except ExperimentError as exc:
        sys.exit(f"rejected: {exc}")
    _save(a.session, lab, {k: meta[k] for k in ("agent_name", "started") if k in meta})
    print(json.dumps(obs))


def cmd_submit(a):
    lab, meta = _load(a.session)
    if lab.result is not None:
        sys.exit("this session was already submitted")
    result = lab.submit(a.formula)
    _save(a.session, lab, {k: meta[k] for k in ("agent_name", "started") if k in meta})
    _print(result)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    tsv = os.path.join(HERE, "results", "external.tsv")
    new_file = not os.path.exists(tsv)
    with open(tsv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        if new_file:
            w.writerow(["timestamp", "session", "agent", "law", "design", "seed",
                        "passed", "nrmse", "experiments_used", "answer"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), a.session,
                    meta.get("agent_name", ""), lab.law.name, lab.design["id"], lab.seed,
                    result["passed"], result["nrmse"], result["experiments_used"], a.formula])


def cmd_show(a):
    lab, meta = _load(a.session)
    print(f"session: {a.session}  agent: {meta.get('agent_name', '')}")
    _print(lab.card())
    for i, entry in enumerate(lab.log):
        print(f"[{i}] {json.dumps(entry['action'])}")
    if lab.result:
        _print(lab.result)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new", help="start a session and print the task card")
    n.add_argument("--design", default="A", choices=list(DESIGNS))
    n.add_argument("--seed", type=int, default=0)
    n.add_argument("--law", default=DEFAULT_LAW, choices=list(LAWS))
    n.add_argument("--agent-name", default="unnamed")
    n.set_defaults(fn=cmd_new)
    e = sub.add_parser("exp", help="run one experiment")
    e.add_argument("session")
    e.add_argument("action", help="JSON action from the task card")
    e.set_defaults(fn=cmd_exp)
    s = sub.add_parser("submit", help="submit the final formula")
    s.add_argument("session")
    s.add_argument("formula")
    s.set_defaults(fn=cmd_submit)
    sh = sub.add_parser("show", help="show a session")
    sh.add_argument("session")
    sh.set_defaults(fn=cmd_show)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
