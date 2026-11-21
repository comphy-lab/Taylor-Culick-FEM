"""Verification of logarithmic coordinates and finite-distance evaluation."""

import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from taylor_culick_fem.depth_averaged_equilibrium import (
    _log_column_stretch, log_profile, profile, shoot,
)


class LogMembraneTests(unittest.TestCase):
    def test_stretch_bracket_when_one_term_underflows(self):
        for tension in (-7.3, -20, -200, -1000):
            z = _log_column_stretch(1000, tension)
            self.assertAlmostEqual(z, -math.log(-tension) / 3, delta=1e-13)

    def test_independent_radial_coordinates_overlap(self):
        # The original r/rho ODE has a distinct state and integration coordinate.
        for ec in (0.3, 0.2, 0.1):
            old = profile(ec)
            new = log_profile(ec)
            self.assertAlmostEqual(new.log_radius_ratio, math.log(old["R"]), delta=3e-6)
            self.assertAlmostEqual(new.edge_thickness, old["lambda_z"][0], delta=3e-6)
            self.assertTrue(new.diagnostics["edge_reached"])
            self.assertLess(abs(new.diagnostics["edge_tension_residual"]), 1e-9)

    def test_finite_distance_matches_dense_radial_solution(self):
        old = profile(0.3)
        sol = shoot(old["A"], dense=True)
        from scipy.optimize import brentq
        distances = [0, 1.4, 1.6, 1.8]
        expected = []
        for distance in distances:
            rho = brentq(lambda rho: sol.sol(rho)[0] - old["R"] - distance,
                         1, sol.t[0])
            r, s = sol.sol(rho)
            from taylor_culick_fem.depth_averaged_equilibrium import radial_stretch
            expected.append((rho / r) / radial_stretch(rho / r, s / r))
        np.testing.assert_allclose(log_profile(0.3).thickness_at_distances(distances),
                                   expected, rtol=3e-6, atol=1e-8)

    def test_small_ec_finite_distances_and_seed_sensitivity(self):
        for ec in (0.15, 0.08, 0.05, 0.025, 0.01):
            new = log_profile(ec)
            heights = new.thickness_at_distances([1.8, 0, 1.4, 1.6, 1.8])
            self.assertTrue(np.all(np.isfinite(heights)))
            self.assertGreater(heights[2], heights[3])
            self.assertGreater(heights[3], heights[0])
            self.assertEqual(heights[0], heights[-1])
            refined = log_profile(ec, far_seed=2e-6, rtol=2e-11, atol=2e-13)
            self.assertAlmostEqual(new.log_radius_ratio, refined.log_radius_ratio, delta=2e-8)
            np.testing.assert_allclose(new.thickness_at_distances([1.4, 1.6, 1.8]),
                                       refined.thickness_at_distances([1.4, 1.6, 1.8]),
                                       rtol=2e-7)

    def test_radius_need_not_be_representable(self):
        new = log_profile(0.002)
        self.assertGreater(new.log_radius_ratio, math.log(np.finfo(float).max))
        self.assertTrue(np.all(np.isfinite(new.thickness_at_distances([1.4, 1.8]))))

    def test_scaling_with_reference_radius(self):
        one, two = log_profile(0.3), log_profile(0.3, r0=2)
        self.assertEqual(one.log_radius_ratio, two.log_radius_ratio)
        np.testing.assert_allclose(one.thickness_at_distances([1.4, 1.8]),
                                   two.thickness_at_distances([2.8, 3.6]), rtol=1e-8)

    def test_invalid_parameters(self):
        for bad in (0, -1, math.nan, math.inf):
            for kwargs in ({"ec": bad}, {"ec": 1, "r0": bad},
                           {"ec": 1, "rtol": bad}, {"ec": 1, "far_seed": bad}):
                with self.assertRaises(ValueError):
                    log_profile(**kwargs)
        with self.assertRaises(ValueError):
            log_profile(1, far_seed=0.1)
        new = log_profile(1)
        for ds in ([-1], [math.nan], [math.inf], [[1]]):
            with self.assertRaises(ValueError):
                new.thickness_at_distances(ds)

    def test_failed_or_truncated_integration_is_rejected(self):
        for success in (False, True):
            fake = SimpleNamespace(success=success, y=np.array([[1.0], [-1.0]]),
                                   t=np.array([10.0]), message="truncated", t_events=[[]])
            with patch("taylor_culick_fem.depth_averaged_equilibrium.solve_ivp", return_value=fake):
                with self.assertRaises(RuntimeError):
                    shoot(1)
                with self.assertRaises(RuntimeError):
                    log_profile(1)


if __name__ == "__main__":
    unittest.main()
