"""Software contracts for retained-state spectrum screening; no pyoomph solve."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix

from taylor_culick_fem.equilibrium_stability import (
    constrained_curvature_screen,
    eigenpair_residuals, matrix_diagnostics, pressure_rows,
    recover_pressure, restore_displacement, retained_nodes, returned_spectrum,
    stationary_correction,
)


class SpectrumTests(unittest.TestCase):
    def test_positive_returned_modes_are_explicit(self):
        report = returned_spectrum([-0.1, 0.2], count=2, shift=0)
        self.assertTrue(report["positive_returned_mode_detected"])
        self.assertFalse(report["spectrum_scope"]["global_stability_established"])

    def test_near_zero_negative_modes_do_not_claim_global_stability(self):
        report = returned_spectrum([-0.1 + 0.03j, -0.2 - 0.03j], count=2, shift=0)
        self.assertEqual(report["leading_returned_eigenvalue"], -0.1)
        self.assertEqual(report["returned_eigenvalues"][0]["imag"], .03)
        self.assertFalse(report["spectrum_scope"]["global_stability_established"])
        self.assertFalse(report["spectrum_scope"]["spectral_completeness"])
        # A remote positive eigenvalue absent from the supplied set cannot be ruled out.
        self.assertIn("azimuthal m>=1", report["spectrum_scope"]["excluded_sectors"])

    def test_nonfinite_spectra_are_rejected(self):
        for values in ([], [np.nan], [complex(-1, np.inf)]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                returned_spectrum(values, count=1, shift=0)

    def test_residual_uses_growth_rate_sign_and_complex_modes(self):
        K = csr_matrix([[1., 2., 0.], [-2., 1., 0.], [0., 0., 0.]])
        M = csr_matrix(np.diag([1., 1., 0.]))
        v = np.array([[1., 1j, 0.]])
        good = eigenpair_residuals(K, M, [-1 - 2j], v, np.array([False, False, True]))[0]
        bad = eigenpair_residuals(K, M, [1 + 2j], v, np.array([False, False, True]))[0]
        self.assertLess(good["relative_action_residual"], 1e-14)
        self.assertGreater(bad["relative_action_residual"], .9)

    def test_constraint_violation_is_reported_separately(self):
        K = csr_matrix([[1., 0.], [2., 0.]])
        M = csr_matrix([[1., 0.], [0., 0.]])
        p = eigenpair_residuals(K, M, [-1], [[1., 0.]], np.array([False, True]))[0]
        self.assertEqual(p["constraint_Kv_norm_per_vector_norm"], 2.)
        self.assertGreater(p["relative_backward_residual"], 0)

    def test_mixed_tangent_symmetry_requires_pressure_row_sign(self):
        K = csr_matrix([[3., 2.], [-2., 0.]])
        M = csr_matrix([[1., 0.], [0., 0.]])
        report = matrix_diagnostics(K, M, [0., .004], np.array([False, True]))
        self.assertGreater(report["raw_tangent_relative_asymmetry"], 0)
        self.assertEqual(report["pressure_sign_corrected_tangent_relative_asymmetry"], 0)
        self.assertEqual(report["max_constraint_residual"], .004)
        self.assertIn("not computed", report["constrained_inertia"])
        with self.assertRaises(ValueError):
            matrix_diagnostics(K, M, [np.nan, 0], np.array([False, True]))


class RetainedStateTests(unittest.TestCase):
    def setUp(self):
        self.ij = np.array([[(i, j) for j in range(3) for i in range(3)]])
        self.Xe = self.ij.astype(float) + [1., 0.]
        self.ue = .1 * self.Xe

    def test_shared_nodes_keep_total_displacement(self):
        lookup = retained_nodes(np.repeat(self.Xe, 2, axis=0), np.repeat(self.ue, 2, axis=0),
                                np.repeat(self.ij, 2, axis=0))
        self.assertEqual(len(lookup), 9)
        np.testing.assert_array_equal(lookup[(1., 0.)], [.1, 0.])

    def test_conflicting_shared_nodes_are_not_averaged(self):
        u = np.repeat(self.ue, 2, axis=0)
        u[1, 0, 0] += 1e-9
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            retained_nodes(np.repeat(self.Xe, 2, axis=0), u, np.repeat(self.ij, 2, axis=0))

    def test_invalid_local_topology_and_nonfinite_states_are_rejected(self):
        ij = self.ij.copy()
        ij[0, 1] = ij[0, 0]
        with self.assertRaisesRegex(ValueError, "distinct"):
            retained_nodes(self.Xe, self.ue, ij)
        u = self.ue.copy()
        u[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            retained_nodes(self.Xe, u, self.ij)

    def test_restoration_assigns_total_displacement_and_refuses_other_grids(self):
        class Node:
            def __init__(self, X):
                self.X, self.values = X, [100., 100.]

            def x(self, i):
                return self.X[i]

            def value(self, i):
                return self.values[i]

            def set_value(self, i, value):
                self.values[i] = value

        class Element:
            def __init__(self, nodes):
                self.nodes = nodes

            def nnode(self):
                return 9

            def node_pt(self, i):
                return self.nodes[i]

            def local_coordinate_of_node(self, i):
                return (i % 3 - 1, i // 3 - 1)

        class Mesh:
            def __init__(self, X, background=False):
                self.background = background
                self._nodes = [Node(x) for x in X]
                if background:
                    for node in self._nodes:
                        node.values += [200., 200.]

            def nodes(self):
                return self._nodes

            def elements(self):
                return [Element(self._nodes)]

            def get_nodal_field_indices(self):
                fields = {"displacement_x": 0, "displacement_y": 1}
                if self.background:
                    fields.update(displacement_background_x=2, displacement_background_y=3)
                return fields

        class Problem:
            def __init__(self, X, background=False):
                self.mesh = Mesh(X, background)

            def get_mesh(self, name):
                return self.mesh

        saved = {"Xe": self.Xe, "ue": self.ue, "ij": self.ij}
        problem = Problem(self.Xe[0])
        receipt = restore_displacement(problem, saved)
        self.assertEqual(receipt["max_nodal_displacement_change"], 0)
        np.testing.assert_array_equal([n.values for n in problem.mesh.nodes()], self.ue[0])
        with self.assertRaisesRegex(ValueError, "connectivity"):
            restore_displacement(Problem(self.Xe[0] + [1e-8, 0]), saved)
        translated = Problem(self.Xe[0], background=True)
        receipt = restore_displacement(translated, saved)
        self.assertIn("zero increment", receipt["representation"])
        np.testing.assert_array_equal([n.values[:2] for n in translated.mesh.nodes()], np.zeros((9, 2)))
        np.testing.assert_array_equal([n.values[2:] for n in translated.mesh.nodes()], self.ue[0])


class StationaryCorrectionTests(unittest.TestCase):
    def correction(self, delta):
        from taylor_culick_fem.equilibrium_geometry import FieldMap

        before = SimpleNamespace(Xe=np.zeros((1, 9, 2)), ij=np.zeros((1, 9, 2)), ue=np.zeros((1, 9, 2)))
        after = SimpleNamespace(Xe=before.Xe.copy(), ij=before.ij.copy(), ue=before.ue + delta)
        node = SimpleNamespace(value=lambda index: delta)
        mesh = SimpleNamespace(nodes=lambda: [node],
                               get_nodal_field_indices=lambda: {"displacement_x": 0, "displacement_y": 1})

        class Problem:
            displacement_background = True
            solved = False

            def solve(self, *, max_newton_iterations):
                self.solved = True
                self.max_iterations = max_newton_iterations

            def get_mesh(self, name):
                return mesh

            def hole_radius(self):
                return 500. + (delta if self.solved else 0.)

            def rim_height(self):
                return 1.4 + (delta if self.solved else 0.)

            def get_residuals(self):
                return [1e-8 if self.solved else 3e-6]

            def get_last_residual_convergence(self):
                return [3e-6, 1e-8]

        problem = Problem()
        with patch.object(FieldMap, "from_problem", side_effect=[before, after]), \
                patch("taylor_culick_fem.equilibrium_geometry.incompressibility_error",
                      side_effect=[.02, .02 + delta]):
            receipt = stationary_correction(problem, max_iterations=20)
        self.assertEqual(problem.max_iterations, 20)
        return receipt

    def test_tiny_corrected_state_is_distinguished_from_retained_coefficients(self):
        receipt = self.correction(1e-10)
        self.assertTrue(receipt["geometry_gate_passed"])
        self.assertFalse(receipt["retained_input_modified"])
        self.assertEqual(receipt["max_increment_nodal_displacement"], 1e-10)
        self.assertIn("corrected discrete equilibrium", receipt["interpretation"])

    def test_correction_cannot_hide_changed_geometry_or_nonfinite_coefficients(self):
        for delta in (1e-7, np.nan):
            with self.subTest(delta=delta):
                self.assertFalse(self.correction(delta)["geometry_gate_passed"])

    def test_correction_requires_translated_trial_space(self):
        with self.assertRaisesRegex(ValueError, "pinned displacement background"):
            stationary_correction(SimpleNamespace(), max_iterations=20)


class PressureRecoveryTests(unittest.TestCase):
    def test_lsqr_column_scaling_recovers_both_pressure_scales_at_same_budget(self):
        class Problem:
            def __init__(self):
                self.dofs = np.array([4., 5., 0., 0.])
                self.B = np.diag([1e-12, 1e12])

            def assemble_jacobian(self, with_residual=True):
                residual = np.concatenate(([1., 2.] + self.B @ self.dofs[2:], [0., 0.]))
                return residual, csr_matrix(np.block([[np.eye(2), self.B], [-self.B.T, np.zeros((2, 2))]]))

            def get_current_dofs(self):
                return self.dofs, None

            def set_current_dofs(self, dofs):
                self.dofs = dofs

        problem = Problem()
        record = recover_pressure(problem, np.array([False, False, True, True]),
                                  tolerance=1e-12, iterations=2, method="lsqr")
        np.testing.assert_array_equal(problem.dofs[:2], [4., 5.])
        np.testing.assert_allclose(problem.dofs[2:], [-1e12, -2e-12], rtol=1e-12, atol=0)
        self.assertTrue(record["converged"])
        self.assertLessEqual(record["iterations"], 2)
        self.assertEqual(record["iteration_limit"], 2)
        self.assertEqual(record["tolerance"], 1e-12)
        self.assertLess(record["max_unscaled_force_residual"], 1e-12)
        self.assertLess(np.max(np.abs(problem.assemble_jacobian()[0])), 1e-12)
        self.assertEqual(record["column_norm_min"], 1e-12)
        self.assertEqual(record["column_norm_max"], 1e12)

    def test_augmented_fit_preserves_unweighted_objective_for_inconsistent_forces(self):
        class Problem:
            def __init__(self):
                self.B = np.array([[1e-10, 1e10], [2e-10, -1e10], [0., 1e10]])
                self.force = np.array([2., -1., 4.])
                self.dofs = np.zeros(5)

            def assemble_jacobian(self, with_residual=True):
                residual = np.concatenate((self.force + self.B @ self.dofs[3:], [0., 0.]))
                tangent = csr_matrix(np.block([[np.eye(3), self.B], [-self.B.T, np.zeros((2, 2))]]))
                return residual, tangent

            def get_current_dofs(self):
                return self.dofs, None

            def set_current_dofs(self, dofs):
                self.dofs = dofs

        problem = Problem()
        mask = np.array([False, False, False, True, True])
        record = recover_pressure(problem, mask, tolerance=1e-12, iterations=2)
        C = problem.B / np.linalg.norm(problem.B, axis=0)
        expected, *_ = np.linalg.lstsq(C, -problem.force, rcond=None)
        np.testing.assert_allclose(problem.dofs[3:] * np.linalg.norm(problem.B, axis=0), expected,
                                   rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(problem.dofs[:3], [0., 0., 0.])
        self.assertTrue(record["converged"])
        self.assertGreater(record["max_unscaled_force_residual"], .1)
        self.assertFalse(record["normal_matrix_formed"])
        self.assertIn("not physical stationarity", record["convergence_scope"])

    def test_rank_deficient_augmented_fit_does_not_change_state(self):
        class Problem:
            def assemble_jacobian(self, with_residual=True):
                B = np.array([[1., 1.], [1., 1.]])
                return np.zeros(4), csr_matrix(np.block([[np.eye(2), B], [-B.T, np.zeros((2, 2))]]))

            def get_current_dofs(self):
                raise AssertionError("a singular direct fit must not access or change the state")

        with self.assertRaises(RuntimeError):
            recover_pressure(Problem(), np.array([False, False, True, True]),
                             tolerance=1e-12, iterations=10)

    def test_zero_or_nonfinite_pressure_columns_fail_before_state_change(self):
        class Problem:
            def __init__(self, entry):
                self.entry = entry

            def assemble_jacobian(self, with_residual=True):
                return np.array([1., 0.]), csr_matrix([[1., self.entry], [-self.entry, 0.]])

            def get_current_dofs(self):
                raise AssertionError("invalid pressure block must not access or change the state")

        for entry in (0., np.inf, np.nan, 1e308):
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                recover_pressure(Problem(entry), np.array([False, True]), tolerance=1e-12, iterations=10)

    def test_missing_pressure_is_recovered_without_moving_displacement(self):
        class Problem:
            def __init__(self):
                self.dofs = np.array([4., 0.])

            def assemble_jacobian(self, with_residual=True):
                return np.array([6. + 2. * self.dofs[1], 0.]), csr_matrix([[3., 2.], [-2., 0.]])

            def get_current_dofs(self):
                return self.dofs, None

            def set_current_dofs(self, dofs):
                self.dofs = dofs

        problem = Problem()
        record = recover_pressure(problem, np.array([False, True]), tolerance=1e-12, iterations=10)
        np.testing.assert_allclose(problem.dofs, [4., -3.])
        self.assertTrue(record["converged"])
        self.assertLess(abs(problem.assemble_jacobian()[0][0]), 1e-12)

    def test_dof_description_does_not_silently_mislabel_unknowns(self):
        class Problem:
            def ndof(self):
                return 2

            def get_dof_description(self):
                return [-1, 0], ["solid/pressure"]

        with self.assertRaisesRegex(ValueError, "incomplete"):
            pressure_rows(Problem())


class ConstrainedCurvatureTests(unittest.TestCase):
    pressure = np.array([False, False, True])

    @staticmethod
    def tangent(first, second):
        # B=[1,0]; ker(B) is the second displacement coordinate. Mixed pressure
        # row has the opposite sign to the displacement/pressure coupling.
        return csr_matrix([[first, 0., -1.], [0., second, 0.], [1., 0., 0.]])

    def test_penalty_stabilises_only_off_constraint_negative_direction(self):
        result = constrained_curvature_screen(self.tangent(-1., 2.), self.pressure)
        self.assertTrue(result["numerical_screen_passed"])
        self.assertFalse(result["rigorous_certificate"])
        self.assertIn("no rigorous norm bound", result["factor_error_bound"])
        self.assertTrue(all(a["passed"] for a in result["attempts"]))
        self.assertIn("discrete", result["scope"])
        self.assertIn("generic LU pivots are not inertia", result["limitations"])

    def test_true_negative_direction_in_constraint_kernel_is_not_certified(self):
        result = constrained_curvature_screen(self.tangent(2., -1.), self.pressure)
        self.assertFalse(result["numerical_screen_passed"])
        self.assertEqual(result["classification"], "numerical constrained-curvature screen unestablished")
        self.assertIn("failure does not establish physical instability or constrained negative curvature",
                      result["limitations"])

    def test_asymmetric_operator_is_rejected(self):
        K = self.tangent(2., 2.).tolil()
        K[0, 1] = .1
        result = constrained_curvature_screen(K.tocsr(), self.pressure)
        self.assertFalse(result["numerical_screen_passed"])
        self.assertEqual(result["attempts"], [])

    def test_dominant_displacement_scale_cannot_hide_wrong_constraint_sign(self):
        K = csr_matrix([[1e16, 0., 1.], [0., 2., 0.], [1., 0., 0.]])
        result = constrained_curvature_screen(K, self.pressure)
        self.assertFalse(result["numerical_screen_passed"])
        self.assertEqual(result["attempts"], [])
        self.assertGreater(result["pressure_coupling_sign_corrected_relative_difference"], .9)

    def test_nontrivial_symmetric_permutation_is_checked_in_factor_actions(self):
        H = np.array([[3., .3, 0., .4], [.3, 2., .1, 0.],
                      [0., .1, 4., .2], [.4, 0., .2, 5.]])
        B = np.array([[1., 0., 0., 0.]])
        K = csr_matrix(np.block([[H, -B.T], [B, np.zeros((1, 1))]]))
        result = constrained_curvature_screen(K, np.array([False] * 4 + [True]))
        self.assertTrue(result["numerical_screen_passed"])
        self.assertTrue(all(a["passed"] for a in result["attempts"]))

    def test_row_pivoting_does_not_become_an_inertia_claim(self):
        fake = SimpleNamespace(perm_r=np.array([1, 0]), perm_c=np.array([0, 1]))
        with patch("scipy.sparse.linalg.splu", return_value=fake):
            result = constrained_curvature_screen(self.tangent(2., 2.), self.pressure)
        self.assertFalse(result["numerical_screen_passed"])
        self.assertTrue(all("row pivoting" in a["reason"] for a in result["attempts"]))

    def test_generic_lu_with_positive_diagonal_is_not_a_symmetric_factor(self):
        fake = SimpleNamespace(perm_r=np.array([0, 1]), perm_c=np.array([0, 1]),
                               L=csr_matrix(np.eye(2)), U=csr_matrix([[1., .1], [0., 1.]]))
        with patch("scipy.sparse.linalg.splu", return_value=fake):
            result = constrained_curvature_screen(self.tangent(2., 2.), self.pressure)
        self.assertFalse(result["numerical_screen_passed"])
        self.assertTrue(all("diag(U)L.T" in a["reason"] for a in result["attempts"]))

    def test_invalid_weights_are_not_accepted(self):
        for weights in ([], [0], [-1], [np.nan], [np.inf], [10, 10]):
            with self.assertRaises(ValueError):
                constrained_curvature_screen(self.tangent(2., 2.), self.pressure, weights=weights)

    def test_zero_constraint_row_is_inconclusive_before_factorisation(self):
        result = constrained_curvature_screen(csr_matrix(np.diag([2., 2., 0.])), self.pressure)
        self.assertFalse(result["numerical_screen_passed"])
        self.assertFalse(result["rigorous_certificate"])
        self.assertEqual(result["attempts"], [])
        self.assertIn("zero constraint row", result["reason"])


if __name__ == "__main__":
    unittest.main()
