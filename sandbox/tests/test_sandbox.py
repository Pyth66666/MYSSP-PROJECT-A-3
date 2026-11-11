"""Tests for the sandbox. Run from the lab folder:  python -m unittest -v"""

import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent  # noqa: E402
import grader  # noqa: E402
from laws import get_law  # noqa: E402
from sandbox_env import ExperimentError, LabSession, simulate_release  # noqa: E402


class GraderTests(unittest.TestCase):
    law = get_law("power_spring")

    def test_equivalent_forms_pass(self):
        for text in [
            r"$$F(x) = -5 \cdot |x|^{1.5} \cdot \mathrm{sign}(x)$$",
            "F(x) = -5*abs(x)**1.5*sign(x)",
            r"$F = -5x\sqrt{|x|}$",
            r"$$F(x) = -\frac{10}{2}\operatorname{sgn}(x)|x|^{3/2}$$",
            "After fitting, the law is F(x) = -5*x*abs(x)**0.5.",
        ]:
            self.assertTrue(grader.grade(text, self.law)["passed"], text)

    def test_hooke_guess_fails(self):
        self.assertFalse(grader.grade("F(x) = -5*x", self.law)["passed"])

    def test_negative_side_must_work(self):
        r = grader.grade("F(x) = -5*x**1.5", self.law)
        self.assertFalse(r["passed"])
        self.assertIn("complex", r["reason"])

    def test_code_injection_rejected(self):
        for text in ["__import__('os').system('echo hi')", "x.__class__", "[x for x in ()]"]:
            self.assertFalse(grader.grade(text, self.law)["passed"])


class SympyTests(unittest.TestCase):
    law = get_law("power_spring")

    def test_exact_match_reported(self):
        try:
            import sympy  # noqa: F401
        except ImportError:
            self.skipTest("sympy not installed")
        r = grader.grade(r"$$F(x) = -5x\sqrt{|x|}$$", self.law)
        self.assertTrue(r["exact"])
        self.assertEqual(r["reason"], "exact SymPy match")
        self.assertFalse(grader.grade("F(x) = -4.98*sign(x)*abs(x)**1.5", self.law)["exact"])

    def test_last_readable_formula_is_used(self):
        text = "General form $F = -k x^n$. Fitting gives $$F(x) = -5 x |x|^{0.5}$$"
        self.assertTrue(grader.grade(text, self.law)["passed"])

    def test_positive_only_check_is_not_enough(self):
        # matches the law for x > 0, the team script's 0.5..5 check would pass it
        self.assertFalse(grader.grade("F(x) = -5*x**1.5", self.law)["passed"])


class StaticPipelineTests(unittest.TestCase):
    def test_dataset_matches_team_script(self):
        from sandbox_env import static_dataset
        rows = static_dataset(get_law("power_spring"), 0)
        self.assertEqual(len(rows), 100)
        self.assertEqual(rows[0], {"time": 0.0, "position": 2.0, "velocity": 0.0,
                                   "acceleration": -14.1421, "force": -14.1421})
        self.assertEqual(rows[1]["position"], 1.8586)

    def test_fifty_different_datasets(self):
        from sandbox_env import static_x0
        self.assertEqual(len({static_x0(i) for i in range(50)}), 50)

    def test_team_prompt_contains_data(self):
        prompt = agent.team_prompt(LabSession("S", 0).card())
        self.assertIn("unknown 1D mechanical system", prompt)
        self.assertIn("-14.1421", prompt)
        self.assertNotIn("{data", prompt)

    def test_csv_round_trip(self):
        import tempfile
        import generate_data
        with tempfile.TemporaryDirectory() as d:
            old = generate_data.HERE
            generate_data.HERE = d
            try:
                folder, index = generate_data.write_datasets("power_spring", 3)
                rows = generate_data.load_dataset(os.path.join(folder, index[0]["file"]))
            finally:
                generate_data.HERE = old
        a = LabSession("S", 0, dataset=rows).card()["data"]
        b = LabSession("S", 0).card()["data"]
        self.assertEqual(a, b)


class SandboxTests(unittest.TestCase):
    def test_budget_enforced(self):
        lab = LabSession("B", 0)
        for _ in range(lab.design["budget"]):
            lab.experiment({"release": {"x0": 1.0}})
        with self.assertRaises(ExperimentError):
            lab.experiment({"release": {"x0": 1.0}})

    def test_tool_not_in_design(self):
        with self.assertRaises(ExperimentError):
            LabSession("B", 0).experiment({"measure_force": {"x": 1.0}})

    def test_card_hides_the_law(self):
        text = json.dumps(LabSession("C", 0).card())
        self.assertNotIn("1.5", text)
        self.assertNotIn("abs(x)", text)

    def test_replay_is_deterministic(self):
        lab = LabSession("C", 7)
        lab.experiment({"measure_force": {"x": 1.2}})
        lab.experiment({"release": {"x0": -2.0}})
        again = LabSession.from_dict(lab.to_dict())
        self.assertEqual(lab.log, again.log)

    def test_energy_conserved(self):
        law = get_law("power_spring")
        ts, xs, vs = simulate_release(law, 2.0)
        def energy(x, v):
            return 0.5 * v * v + 5.0 * abs(x) ** 2.5 / 2.5
        e0 = energy(xs[0], vs[0])
        self.assertTrue(all(abs(energy(x, v) - e0) / e0 < 1e-4 for x, v in zip(xs, vs)))


class BaselineTests(unittest.TestCase):
    def test_fit_agent_passes_all_designs(self):
        for d in "SABC":
            lab = LabSession(d, 0)
            ans, _ = agent.FitAgent().run(lab)
            self.assertTrue(lab.submit(ans)["passed"], d)

    def test_hooke_agent_fails_redesigned_law(self):
        for d in "SABC":
            lab = LabSession(d, 0)
            ans, _ = agent.HookeAgent().run(lab)
            self.assertFalse(lab.submit(ans)["passed"], d)


class FakeLLM(BaseHTTPRequestHandler):
    """Pretends to be an OpenAI-compatible server: one experiment, then an answer."""
    replies = [
        'I will probe first.\n{"experiment": {"measure_force": {"x": 2.0}}}',
        'Looks like a power law.\n{"answer": "F(x) = -5*abs(x)**1.5*sign(x)"}',
    ]
    calls = 0

    def do_GET(self):
        self._json({"data": [{"id": "fake"}]})

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        text = FakeLLM.replies[min(FakeLLM.calls, len(FakeLLM.replies) - 1)]
        FakeLLM.calls += 1
        self._json({"choices": [{"message": {"content": text}}],
                    "usage": {"total_tokens": 42}})

    def _json(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


class LLMProtocolTest(unittest.TestCase):
    def test_llm_static_single_prompt(self):
        FakeLLM.calls = 1   # skip the experiment reply, answer directly
        srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeLLM)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            ag = agent.LLMAgent("fake", f"http://127.0.0.1:{srv.server_port}/v1", "fake")
            lab = LabSession("S", 0)
            ans, _ = ag.run(lab)
            self.assertTrue(lab.submit(ans)["passed"])
        finally:
            srv.shutdown()
            srv.server_close()

    def test_llm_agent_loop(self):
        FakeLLM.calls = 0
        srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeLLM)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            ag = agent.LLMAgent("fake", f"http://127.0.0.1:{srv.server_port}/v1", "fake")
            lab = LabSession("A", 0)
            ans, notes = ag.run(lab)
            self.assertEqual(len(lab.log), 1)
            self.assertTrue(lab.submit(ans)["passed"])
            self.assertEqual(ag.tokens, 84)
        finally:
            srv.shutdown()
            srv.server_close()


class TeamLawTests(unittest.TestCase):
    """The team's redesigned laws 2 to 5 (kinetic energy, wave speed, screened
    Coulomb, saturating Ohm) work in designs S, A and C, and grade fairly."""
    NEW = {
        "kinetic_energy": (r"E_k(v) = \frac{2}{v}", "E_k = 0.5 * 2 * v**2"),
        "wave_speed": (r"v(f) = 15 \ln(0.5 f)", "v = 0.5 * f"),
        "screened_coulomb": (r"F(r) = \frac{6 e^{-0.4r}}{r^2}", "F = 6 / r**2"),
        "saturating_ohm": (r"I(V) = 5\tanh\left(\frac{V}{2}\right)", "I = V / 2"),
    }

    def test_true_law_exact_and_textbook_fails(self):
        for name, (right, textbook) in self.NEW.items():
            law = get_law(name)
            r = grader.grade(right, law)
            self.assertTrue(r["passed"] and r["exact"], (name, r))
            self.assertFalse(grader.grade(textbook, law)["passed"], name)

    def test_answer_may_use_x_or_the_law_symbol(self):
        law = get_law("kinetic_energy")
        self.assertTrue(grader.grade("y = 2/x", law)["exact"])
        self.assertTrue(grader.grade("E_k = 2/v", law)["exact"])

    def test_bare_function_call(self):
        r = grader.grade(r"v = 15\ln f - 10.397", get_law("wave_speed"))
        self.assertTrue(r["passed"])

    def test_test_points_stay_in_range(self):
        for name in self.NEW:
            law = get_law(name)
            lo, hi = law.x_range
            self.assertTrue(all(lo < x < hi for x in law.test_points()), name)

    def test_design_S_prompt_has_names_not_formula(self):
        lab = LabSession("S", 0, "screened_coulomb")
        prompt = agent.team_prompt(lab.card())
        self.assertIn("distance (r) in m", prompt)
        self.assertEqual(len(lab.card()["data"]), 20)
        for leak in ("exp(", "e^", "-0.4", "alpha", "screened"):
            self.assertNotIn(leak, prompt)

    def test_design_A_measure_and_range(self):
        lab = LabSession("A", 0, "saturating_ohm")
        obs = lab.experiment({"measure": {"x": 2.0}})
        self.assertAlmostEqual(obs["y"], 5 * __import__("math").tanh(1.0), places=5)
        with self.assertRaises(ExperimentError):
            lab.experiment({"measure": {"x": 50}})
        with self.assertRaises(ExperimentError):
            lab.experiment({"measure_force": {"x": 1.0}})

    def test_design_B_is_spring_only(self):
        with self.assertRaises(ExperimentError):
            LabSession("B", 0, "wave_speed")

    def test_curve_fit_finds_the_power_law_only(self):
        ans, _ = agent.FitAgent().run(LabSession("A", 0, "kinetic_energy"))
        self.assertTrue(LabSession("A", 0, "kinetic_energy").submit(ans)["passed"])
        ans, _ = agent.FitAgent().run(LabSession("A", 0, "wave_speed"))
        self.assertFalse(LabSession("A", 0, "wave_speed").submit(ans)["passed"])


if __name__ == "__main__":
    unittest.main()
