"""Self-improvement loop (RSI) for a small model, compared with a big model.

What "self-improvement" means here: the model's weights do not change (that is
not possible on a laptop). The small model improves its STRATEGY. After each
round it reads its own runs, writes better strategy notes for itself, and the
notes are kept only if they score at least as well. This is a common, honest
form of recursive self-improvement for agents.

Fair test: the model practises on TRAINING laws (laws.TRAINING_LAWS) and may
see their true formulas after each run. The test law (power_spring) is never
shown, in any prompt. Only the final test uses it.

    python rsi.py --small "Ollama qwen2.5-coder:7b (single prompt)" --big "RapidScreen gpt-5.5"
    python rsi.py --small "RapidScreen gpt-5.4-nano" --big "RapidScreen gpt-5.5" --rounds 3
    python rsi.py ... --rounds 1 --train-seeds 1 --test-seeds 2     # quick first try

Output: results/rsi/<time>/ with the notes from every round, history.json,
final.json, and every run's full log in results/runs/.
"""

import argparse
import json
import os
import time

import agent as agents
import experiment
from laws import LAWS, TEST_LAW, TRAINING_LAWS

HERE = os.path.dirname(os.path.abspath(__file__))

REFLECT_PROMPT = """You are improving your own method for discovering unknown spring force laws.
Below are your latest practice runs. Each shows the true law (revealed after the run),
the measurements you chose, the formula you submitted and the grade.

{runs}

Your current strategy notes:
{notes}

Write NEW strategy notes that will help you on a {target} whose law you have NOT seen.
Rules:
- General advice only: how to choose measurement positions, how to work out the
  functional form and the constants from (x, F) pairs, how to check the answer,
  how to write it in the required JSON format. Do not just list the laws above.
- At most 200 words.
- Put the notes between <notes> and </notes>."""


def describe_run(row, out_dir):
    with open(os.path.join(out_dir, row["log_file"]), encoding="utf-8") as f:
        log = json.load(f)
    pts = []
    for e in log["experiments"]:
        obs = e["observation"]
        if "F" in obs:
            pts.append(f"x={obs['x']:+.3f} F={obs['F']:+.4f}")
        elif "x0" in obs:
            pts.append(f"release x0={obs['x0']:+.2f}")
    err = "-" if row["nrmse"] is None else f"{row['nrmse']:.2%}"
    verdict = "PASS" if row["passed"] else "FAIL"
    answer = str(row["answer"]).splitlines()[0][:200] if row["answer"] else "(none)"
    return (f"- true law: {LAWS[row['law']].formula}\n"
            f"  measurements: {'; '.join(pts) or 'none'}\n"
            f"  your answer: {answer}\n"
            f"  grade: {verdict}, error {err}, {row['experiments_used']} experiments")


def score(rows):
    """Higher is better: pass rate first, then a lower median error."""
    s = experiment.summarise(rows)
    med = s["median_nrmse"] if s["median_nrmse"] is not None else 9.0
    return (round(s["pass_rate"], 6), -round(min(med, 9.0), 6))


def reflect(small, rows, notes, out_dir, test_law=TEST_LAW):
    runs = "\n".join(describe_run(r, out_dir) for r in rows)
    target = "spring" if LAWS[test_law].is_spring else "physical system"
    prompt = REFLECT_PROMPT.format(runs=runs, notes=notes or "(none yet)", target=target)
    # Guard: the held-out test law must never appear in what the model reads.
    assert LAWS[test_law].formula not in prompt, "test law leaked into a prompt"
    reply = small._chat([{"role": "user", "content": prompt}])
    start, end = reply.find("<notes>"), reply.find("</notes>")
    text = reply[start + 7:end] if start != -1 and end > start else reply
    return text.strip()[:1800], prompt, reply


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--small", required=True, help="agent name of the small model")
    p.add_argument("--big", default="", help="agent name of the big model (comparison)")
    p.add_argument("--design", default="A", choices=["A", "B", "C"])
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--train-seeds", type=int, default=1, help="runs per training law per round")
    p.add_argument("--test-seeds", type=int, default=10, help="runs on the test law at the end")
    p.add_argument("--margin", type=float, default=0.10,
                   help="success if small+RSI pass rate >= big pass rate - margin")
    p.add_argument("--test-law", default=TEST_LAW,
                   choices=[n for n in LAWS if n not in TRAINING_LAWS],
                   help="held-out law for the final test, e.g. one of the team's new laws "
                        "to see whether spring practice transfers")
    p.add_argument("--out", default=os.path.join(HERE, "results"))
    a = p.parse_args()

    test_law = a.test_law
    assert test_law not in TRAINING_LAWS
    if a.design == "B" and not LAWS[test_law].is_spring:
        raise SystemExit("Design B only works with spring laws. Use --design A or C.")
    names = a.small + ("," + a.big if a.big else "")
    pool = {ag.name: ag for ag in experiment.pick_agents(names)}
    small, big = pool[a.small], pool.get(a.big)
    for ag in (small, big):
        if ag is not None and not isinstance(ag, agents.LLMAgent):
            raise SystemExit(f"{ag.name} is not an LLM agent")

    run_dir = os.path.join(a.out, "rsi", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    history = []

    def save(name, text):
        with open(os.path.join(run_dir, name), "w", encoding="utf-8") as f:
            f.write(text)

    print(f"Training laws: {', '.join(TRAINING_LAWS)}  |  test law (held out): {test_law}")
    print(f"\nRound 0: {small.name} with no strategy notes")
    rows = experiment.run_batch(small, a.design, TRAINING_LAWS, a.train_seeds, "", a.out, "rsi-round-0")
    best_notes, best_score, best_rows = "", score(rows), rows
    history.append({"round": 0, "accepted": True, "score": best_score, "notes": ""})
    print(f"  training score: pass {best_score[0]:.0%}")

    for rnd in range(1, a.rounds + 1):
        print(f"\nRound {rnd}: {small.name} rewrites its notes")
        try:
            notes, prompt, reply = reflect(small, best_rows, best_notes, a.out, test_law)
        except Exception as exc:  # noqa: BLE001
            print(f"  reflection failed: {exc}")
            history.append({"round": rnd, "accepted": False, "error": str(exc)})
            continue
        save(f"round_{rnd}_notes.txt", notes)
        save(f"round_{rnd}_reflection_prompt.txt", prompt)
        save(f"round_{rnd}_reflection_reply.txt", reply)
        rows = experiment.run_batch(small, a.design, TRAINING_LAWS, a.train_seeds, notes, a.out,
                                    f"rsi-round-{rnd}")
        s = score(rows)
        accepted = s >= best_score
        print(f"  training score: pass {s[0]:.0%} (best so far {best_score[0]:.0%}) -> "
              f"{'KEEP new notes' if accepted else 'keep old notes'}")
        if accepted:
            best_notes, best_score, best_rows = notes, s, rows
        history.append({"round": rnd, "accepted": accepted, "score": s, "notes": notes})
        save("history.json", json.dumps(history, indent=1))

    save("best_notes.txt", best_notes)
    save("history.json", json.dumps(history, indent=1))

    print(f"\nFinal test on the held-out law '{test_law}', {a.test_seeds} run(s) each")
    final = {}
    final["small, no notes"] = experiment.run_batch(small, a.design, [test_law], a.test_seeds, "",
                                                    a.out, "rsi-test-small-base")
    final["small + RSI notes"] = experiment.run_batch(small, a.design, [test_law], a.test_seeds,
                                                      best_notes, a.out, "rsi-test-small-rsi")
    if big is not None:
        final["big, no notes"] = experiment.run_batch(big, a.design, [test_law], a.test_seeds, "",
                                                      a.out, "rsi-test-big")
    experiment.print_summary(final)

    summary = {k: experiment.summarise(v) for k, v in final.items()}
    verdict = None
    if big is not None:
        gap = summary["small + RSI notes"]["pass_rate"] - summary["big, no notes"]["pass_rate"]
        verdict = {"gap_pass_rate": gap, "margin": a.margin, "success": gap >= -a.margin}
        print(f"\nSmall + RSI vs big: pass-rate gap {gap:+.0%} (success if >= -{a.margin:.0%}): "
              f"{'SUCCESS' if verdict['success'] else 'NOT YET'}")
    save("final.json", json.dumps({"small": small.name, "big": big.name if big else None,
                                   "design": a.design, "test_law": test_law,
                                   "training_laws": TRAINING_LAWS, "summary": summary,
                                   "verdict": verdict, "best_notes": best_notes},
                                  indent=1))
    print(f"\nSaved notes, history and final results in {run_dir}")


if __name__ == "__main__":
    main()
