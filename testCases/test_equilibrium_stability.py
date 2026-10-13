"""Software contracts for retained-state spectrum screening; no pyoomph solve."""

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from taylor_culick_fem.equilibrium_stability import (
    eigenpair_residuals, matrix_diagnostics, pressure_rows,
    recover_pressure, restore_displacement, retained_nodes, returned_spectrum,
)


class SpectrumTests(unittest.TestCase):
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
            def __init__(self, X):
                self._nodes = [Node(x) for x in X]

            def nodes(self):
                return self._nodes

            def elements(self):
                return [Element(self._nodes)]

            def get_nodal_field_indices(self):
                return {"displacement_x": 0, "displacement_y": 1}

        class Problem:
            def __init__(self, X):
                self.mesh = Mesh(X)

            def get_mesh(self, name):
                return self.mesh

        saved = {"Xe": self.Xe, "ue": self.ue, "ij": self.ij}
        problem = Problem(self.Xe[0])
        receipt = restore_displacement(problem, saved)
        self.assertEqual(receipt["max_nodal_displacement_change"], 0)
        np.testing.assert_array_equal([n.values for n in problem.mesh.nodes()], self.ue[0])
        with self.assertRaisesRegex(ValueError, "connectivity"):
            restore_displacement(Problem(self.Xe[0] + [1e-8, 0]), saved)


class PressureRecoveryTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
