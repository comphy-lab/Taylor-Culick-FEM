"""Verification of the universal depth-averaged trajectory against ``log_profile``.

The universal trajectory (state ``zeta = -ln(lambda_z)`` and ``I = int k dy``, explicit
right-hand side) and ``log_profile`` (state ``T``, a root solve for the stretch at every
evaluation, integrated per ``Ec``) are two discretisations of the same depth-averaged
equilibrium, so they must agree on the arrest radius, the edge thickness and the thickness
behind the edge. The runner tests check the run contract without importing pyoomph.
"""

import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from taylor_culick_fem.cases import DEPTH_AVERAGED, case_problems
from taylor_culick_fem.depth_averaged_equilibrium import (
    arrest_log_radius_lower_bound,
    edge_thickness_from_log_radius,
    log_profile,
    scaled_thickness_expansion,
    universal_series_start,
    universal_trajectory,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "simulationCases" / "run_depth_averaged_equilibrium.py"
CASE = ROOT / "simulationCases" / "depth_averaged_equilibrium.json"


class UniversalTrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trajectory = universal_trajectory()

    def test_constant_independent_of_series_start_and_tolerance(self):
        reference = self.trajectory.constant
        for kwargs in ({"far_seed": 3e-4}, {"far_seed": 3e-3},
                       {"rtol": 1e-13, "atol": 1e-15}, {"log_radius_max": 60.0}):
            with self.subTest(**kwargs):
                self.assertAlmostEqual(universal_trajectory(**kwargs).constant, reference, delta=2e-12)
        self.assertGreater(reference, 0)
        self.assertLess(reference, 0.5)

    def test_series_start_lies_on_the_trajectory(self):
        # Integrating from a much smaller start reaches the series values at larger y, with a
        # zeta error that falls as the fifth power of y (the series' truncation order).
        fine = universal_trajectory(far_seed=1e-5, rtol=1e-13, atol=1e-16, log_radius_max=60.0)
        errors = []
        for y in (1e-3, 3e-3, 1e-2):
            zeta, integral = universal_series_start(y)
            errors.append(abs(float(fine.zeta(y)) - zeta))
            self.assertAlmostEqual(float(-fine.tension(y) - y), integral, delta=1e-13)
        self.assertLess(errors[0], 1e-15)
        self.assertLess(errors[2], 1e-11)
        order = math.log(errors[2] / errors[1]) / math.log(1e-2 / 3e-3)
        self.assertGreater(order, 4.5)
        self.assertLess(order, 5.5)

    def test_agrees_with_log_profile(self):
        for ec in (2.0, 1.0, 0.5, 0.3, 0.1, 0.02):
            with self.subTest(ec=ec):
                other = log_profile(ec, rtol=1e-12, atol=1e-14, far_seed=1e-6)
                self.assertAlmostEqual(self.trajectory.arrest_log_radius(ec),
                                       other.log_radius_ratio, delta=1e-9)
                self.assertAlmostEqual(self.trajectory.edge_thickness(ec),
                                       other.edge_thickness, delta=1e-9)
                np.testing.assert_allclose(
                    self.trajectory.thickness_at_distances(ec, [0, 1.4, 1.6, 1.8]),
                    other.thickness_at_distances([0, 1.4, 1.6, 1.8]), rtol=1e-8)

    def test_scaling_with_reference_radius(self):
        np.testing.assert_allclose(self.trajectory.thickness_at_distances(0.3, [2.8, 3.6], r0=2),
                                   self.trajectory.thickness_at_distances(0.3, [1.4, 1.8]),
                                   rtol=1e-12)
        np.testing.assert_allclose(self.trajectory.thickness_at_distances(0.3, [2.8, 3.6], r0=2),
                                   log_profile(0.3, r0=2, rtol=1e-12, atol=1e-14,
                                               far_seed=1e-6).thickness_at_distances([2.8, 3.6]),
                                   rtol=1e-8)

    def test_arrest_identities(self):
        constant = self.trajectory.constant
        for ec in (5.0, 1.0, 0.5, 0.2, 0.05):
            with self.subTest(ec=ec):
                y = self.trajectory.arrest_log_radius(ec)
                self.assertAlmostEqual(float(self.trajectory.tension(y)), -2 / ec, delta=1e-11)
                self.assertGreaterEqual(y, arrest_log_radius_lower_bound(ec) - 1e-12)
                self.assertGreaterEqual(y, 2 / ec - constant - 1e-12)
                self.assertLessEqual(y, 2 / ec)
                # Y = 2/Ec - C + eps(Y): the tail integral against the cancelling difference.
                eps = y - (2 / ec - constant)
                if eps > 1e-6:
                    self.assertAlmostEqual(math.exp(self.trajectory.log_remainder(y)) / eps, 1,
                                           delta=1e-8)
                self.assertAlmostEqual(edge_thickness_from_log_radius(ec, y),
                                       self.trajectory.edge_thickness(ec), delta=1e-11)

    def test_edge_identity_where_the_quartic_cancels(self):
        # At Ec = 0.0887 the two leading quartic terms cancel below the exponential term in
        # floating point; the log-space root must still solve the identity.
        for ec in (0.0887, 0.08869, 0.0888):
            with self.subTest(ec=ec):
                y = self.trajectory.arrest_log_radius(ec)
                self.assertAlmostEqual(edge_thickness_from_log_radius(ec, y),
                                       self.trajectory.edge_thickness(ec), delta=1e-11)

    def test_distances_beyond_the_series_start(self):
        # Far behind a large-Ec edge, and with a coarse series start, the distance mapping
        # continues below far_seed through the far-field series.
        reference = log_profile(10.0, rtol=1e-12, atol=1e-14, far_seed=1e-6)
        np.testing.assert_allclose(self.trajectory.thickness_at_distances(10.0, [10.0, 100.0]),
                                   reference.thickness_at_distances([10.0, 100.0]), rtol=1e-12)
        coarse = universal_trajectory(far_seed=1e-2)
        np.testing.assert_allclose(
            coarse.thickness_at_distances(90.0, [0.5]),
            log_profile(90.0, rtol=1e-12, atol=1e-14, far_seed=1e-6).thickness_at_distances([0.5]),
            rtol=1e-12)
        np.testing.assert_allclose(coarse.thickness_at_distances(1.0, [1.4, 30.0]),
                                   self.trajectory.thickness_at_distances(1.0, [1.4, 30.0]),
                                   rtol=1e-11)
        far = self.trajectory.thickness_at_distances(10.0, [1e4, 1e8])
        self.assertTrue(np.all(far >= 1) and np.all(far - 1 < 1e-12))

    def test_small_ec_needs_no_representable_radius(self):
        ec = 0.002
        y = self.trajectory.arrest_log_radius(ec)
        self.assertGreater(y, math.log(np.finfo(float).max))
        self.assertTrue(math.isfinite(self.trajectory.log_remainder(y)))
        heights = self.trajectory.thickness_at_distances(ec, [1.4, 1.8])
        np.testing.assert_allclose(heights, log_profile(ec).thickness_at_distances([1.4, 1.8]),
                                   rtol=1e-7)

    def test_second_order_expansion_improves_at_small_ec(self):
        constant = self.trajectory.constant
        errors = []
        for ec in (0.1, 0.05):
            exact = ec ** (1 / 3) * float(self.trajectory.thickness_at_distances(ec, [1.6])[0])
            first, second = (abs(float(scaled_thickness_expansion(ec, 1.6, constant, order)) - exact)
                             for order in (1, 2))
            self.assertLess(second, first)
            errors.append(second)
        self.assertLess(errors[1], errors[0])

    def test_invalid_arguments(self):
        for bad in (0, -1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                self.trajectory.arrest_log_radius(bad)
        with self.assertRaises(ValueError):
            self.trajectory.arrest_log_radius(1e-3 * 0.9)  # 2/Ec + 20 beyond the trajectory
        with self.assertRaises(ValueError):
            self.trajectory.arrest_log_radius(2e3)  # edge inside the series start
        with self.assertRaises(ValueError):
            self.trajectory.zeta(self.trajectory.log_radius_max + 1)
        with self.assertRaises(ValueError):
            universal_trajectory(far_seed=0.1)
        with self.assertRaises(ValueError):
            universal_trajectory(log_radius_max=50.0)
        for ds in ([-1], [math.nan], [[1]]):
            with self.assertRaises(ValueError):
                self.trajectory.thickness_at_distances(1.0, ds)
        with self.assertRaises(ValueError):
            self.trajectory.thickness_at_distances(0.0028, [1000.0], r0=1e-308)
        short = universal_trajectory(log_radius_max=60.0)
        self.assertTrue(math.isfinite(short.log_remainder(40.0)))
        for y in (40.01, 59.99, 60.0, 0.5 * short.far_seed):
            with self.assertRaises(ValueError):
                short.log_remainder(y)


class DepthAveragedCaseTests(unittest.TestCase):
    def case(self):
        return json.loads(CASE.read_text(encoding="utf-8"))

    def test_default_case_is_valid(self):
        self.assertEqual(case_problems(self.case(), DEPTH_AVERAGED), [])

    def test_every_problem_is_named(self):
        case = self.case()
        case["physics"]["Ec"] = [1.0, 1.0]
        case["evaluation"]["distances"] = [-1]
        case["solver"]["far_seed"] = 0.5
        problems = case_problems(case, DEPTH_AVERAGED)
        self.assertEqual(len(problems), 3, problems)
        case = self.case()
        case["physics"]["Ec"] = [5e-4]
        self.assertTrue(any("log_radius_max" in p for p in case_problems(case, DEPTH_AVERAGED)))
        case = self.case()
        case["physics"]["r0"] = 1e-308
        case["evaluation"]["distances"] = [1000.0]
        self.assertTrue(any("representable" in p for p in case_problems(case, DEPTH_AVERAGED)))


class DepthAveragedRunnerTests(unittest.TestCase):
    def run_runner(self, *args, cwd=None):
        return subprocess.run([sys.executable, str(RUNNER), *map(str, args)], cwd=cwd,
                              capture_output=True, text=True, check=False)

    def test_check_creates_nothing_from_any_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            for cwd in (Path(tmp), RUNNER.parent):
                done = self.run_runner(CASE, "--out", out, "--check", cwd=cwd)
                self.assertEqual(done.returncode, 0, done.stderr)
            self.assertFalse(out.exists())

    def test_refuses_a_used_output_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "something").write_text("x")
            done = self.run_runner(CASE, "--out", tmp)
            self.assertNotEqual(done.returncode, 0)
            self.assertIn("refused", done.stderr)

    def test_run_writes_manifest_and_matches_the_library(self):
        case = json.loads(CASE.read_text(encoding="utf-8"))
        case["physics"]["Ec"] = [1.0, 0.1, 0.0887]
        case["evaluation"]["distances"] = [1.6]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "case.json"
            path.write_text(json.dumps(case), encoding="utf-8")
            out = Path(tmp) / "run"
            done = self.run_runner(path, "--out", out)
            self.assertEqual(done.returncode, 0, done.stderr)
            manifest = json.loads((out / "run-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["case_id"], case["case_id"])
            self.assertIn("physics.Ec", manifest["case_fields_used"])
            events = [json.loads(line)["event"]
                      for line in (out / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events, ["started", "completed"])
            with (out / "arrest.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            trajectory = universal_trajectory()
            for row in rows:
                ec = float(row["Ec"])
                self.assertEqual(float(row["log_radius_ratio"]), trajectory.arrest_log_radius(ec))
                self.assertEqual(float(row["h_at_1.6"]),
                                 float(trajectory.thickness_at_distances(ec, [1.6])[0]))
            metadata = json.loads((out / "case.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["constant_C"], trajectory.constant)
            self.assertEqual(metadata["status"], "passed")


if __name__ == "__main__":
    unittest.main()
