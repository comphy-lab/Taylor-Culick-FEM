"""Retained-state eigenvalue screens; no claim of spectral completeness.

The half-sheet displacement problem represents meridional, axisymmetric,
reflection-even perturbations only. Its pressure mass is singular. Finite
shift-invert searches must not be described as a globally leading spectrum.
"""

from __future__ import annotations

import numpy as np


def spectrum_scope() -> dict:
    """The perturbation space and explicit limits of the existing equations."""
    return {
        "azimuthal_mode": 0,
        "displacement_components": ["radial", "axial"],
        "midplane_parity": "reflection-even: radial even, axial odd",
        "excluded_sectors": ["azimuthal m>=1", "azimuthal displacement/swirl",
                             "reflection-odd/bending"],
        "neutral_modes": "not removed or classified; boundary conditions may exclude rigid modes",
        "flow": "overdamped reference-volume displacement mass weighted by 1/X^2",
        "spectral_completeness": False,
        "global_stability_established": False,
        "interpretation": "negative returned modes are a screen, not proof of stability",
    }


def returned_spectrum(values, *, count: int, shift: float, selector: str = "LM") -> dict:
    """Preserve complex values and label the maximum of only the returned set."""
    values = np.asarray(values, dtype=complex).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("the returned spectrum must be nonempty and finite")
    return {
        "leading_returned_eigenvalue": float(values.real.max()),
        "returned_eigenvalues": [{"real": float(v.real), "imag": float(v.imag)} for v in values],
        "requested_count": int(count), "returned_count": int(len(values)),
        "selector": selector, "shift": float(shift),
        "selection": "largest magnitude of the shift-inverted operator; near the shift",
        "spectrum_scope": spectrum_scope(),
    }


def retained_nodes(Xe, ue, ij) -> dict[tuple[float, float], np.ndarray]:
    """Exact reference-node lookup; reject conflicting shared nodes and bad Q2 data."""
    Xe, ue, ij = np.asarray(Xe), np.asarray(ue), np.asarray(ij)
    if Xe.ndim != 3 or Xe.shape[1:] != (9, 2) or ue.shape != Xe.shape or ij.shape != Xe.shape:
        raise ValueError("retained state must contain Q2 Xe, ue and ij arrays of shape (nel,9,2)")
    if not len(Xe) or not all(np.isfinite(a).all() for a in (Xe, ue, ij)):
        raise ValueError("retained state must be nonempty and finite")
    expected = {(i, j) for i in range(3) for j in range(3)}
    if any(set(map(tuple, element)) != expected for element in ij):
        raise ValueError("retained local indices do not describe nine distinct Q2 nodes")
    nodes = {}
    for X, u in zip(Xe.reshape(-1, 2), ue.reshape(-1, 2)):
        key = tuple(map(float, X))
        if key in nodes and not np.array_equal(nodes[key], u):
            raise ValueError("shared retained nodes have inconsistent total displacement")
        nodes[key] = u.copy()
    return nodes


def restore_displacement(problem, saved) -> dict:
    """Restore total displacement on identical nodes without interpolation or a solve."""
    from taylor_culick_fem.equilibrium_geometry import FieldMap

    lookup = retained_nodes(saved["Xe"], saved["ue"], saved["ij"])
    mesh = problem.get_mesh("solid")
    fields = mesh.get_nodal_field_indices()
    current = FieldMap.from_problem(problem)
    # Element order can differ, but reference nodes and Q2 connectivity must agree.
    topology = lambda X: {frozenset(map(tuple, element)) for element in X}
    if topology(current.Xe) != topology(saved["Xe"]):
        raise ValueError("retained grids/state and rebuilt mesh have different Q2 connectivity")
    nodes = list(mesh.nodes())
    coordinates = {(float(n.x(0)), float(n.x(1))) for n in nodes}
    if coordinates != set(lookup):
        raise ValueError("rebuilt mesh does not have the identical retained reference nodes")
    for node in nodes:
        u = lookup[(float(node.x(0)), float(node.x(1)))]
        if node.x(1) == 0 and u[1] != 0:
            raise ValueError("retained displacement violates the midplane symmetry condition")
        node.set_value(fields["displacement_x"], float(u[0]))
        node.set_value(fields["displacement_y"], float(u[1]))
    after = FieldMap.from_problem(problem)
    restored = retained_nodes(after.Xe, after.ue, after.ij)
    if any(not np.array_equal(restored[X], u) for X, u in lookup.items()):
        raise ValueError("restoration changed retained total nodal displacement")
    return {"method": "exact reference-node and Q2-connectivity match; total displacement",
            "nodes": len(lookup), "elements": len(saved["Xe"]),
            "max_nodal_displacement_change": 0.0,
            "pressure_retained": False, "stationary_solve_performed": False}


def pressure_rows(problem) -> np.ndarray:
    """Pressure constraint rows from the supported global DOF description API."""
    types, names = problem.get_dof_description()
    types = np.asarray(types, dtype=int)
    if len(types) != problem.ndof() or np.any(types < 0) or np.any(types >= len(names)):
        raise ValueError("incomplete or inconsistent global DOF description")
    pressure = np.asarray([names[i].split("/")[-1] == "pressure" for i in types])
    if not pressure.any():
        raise ValueError("pressure constraint DOFs were not identified")
    return pressure


def max_abs(values) -> float:
    values = np.asarray(values)
    return float(np.max(np.abs(values))) if values.size else 0.0


def recover_pressure(problem, pressure, *, tolerance: float, iterations: int) -> dict:
    """Recover missing pressure at fixed displacement, recording least-squares limits.

    The retained state has no pressure. The constitutive residual is affine in
    pressure; solve its displacement rows without moving the retained geometry.
    This is pressure reconstruction, not an equilibrium correction.
    """
    from scipy.sparse.linalg import lsqr

    residual, tangent = problem.assemble_jacobian(with_residual=True)
    residual = np.asarray(residual)
    if max_abs(tangent[pressure][:, pressure].data) != 0:
        raise ValueError("pressure reconstruction requires a zero pressure-pressure block")
    block = tangent[~pressure][:, pressure]
    fit = lsqr(block, -residual[~pressure], atol=tolerance, btol=tolerance, iter_lim=iterations)
    dofs, _ = problem.get_current_dofs()
    updated = np.asarray(dofs, dtype=float).copy()
    updated[pressure] += fit[0]
    if not np.isfinite(updated).all():
        raise ValueError("pressure reconstruction produced non-finite values")
    problem.set_current_dofs(updated)
    return {"method": "fixed-displacement pressure-block least squares",
            "stop_code": int(fit[1]), "iterations": int(fit[2]),
            "residual_norm": float(fit[3]), "normal_residual_norm": float(fit[7]),
            "estimated_condition": float(fit[6]),
            "converged": int(fit[1]) in (0, 1, 2, 4, 5),
            "tolerance": tolerance, "iteration_limit": iterations}


def matrix_diagnostics(tangent, mass, residual, pressure) -> dict:
    """Residual, constraint and sign-corrected variational symmetry diagnostics."""
    from scipy.sparse import diags
    from scipy.sparse.linalg import norm

    residual = np.asarray(residual)
    if not all(np.isfinite(a).all() for a in (tangent.data, mass.data, residual)):
        raise ValueError("assembled matrices and residual must be finite")
    if tangent.shape != mass.shape or tangent.shape != (len(residual), len(residual)):
        raise ValueError("assembled matrix/residual dimensions disagree")
    signs = np.where(pressure, -1.0, 1.0)
    symmetric_tangent = diags(signs) @ tangent
    relative = lambda A: float(norm(A - A.T) / max(float(norm(A)), np.finfo(float).tiny))
    return {
        "finite_values": True, "ndof": len(residual),
        "pressure_dofs": int(np.count_nonzero(pressure)),
        "tangent_nnz": int(tangent.nnz), "mass_nnz": int(mass.nnz),
        "max_equilibrium_residual": max_abs(residual),
        "max_displacement_residual": max_abs(residual[~pressure]),
        "max_constraint_residual": max_abs(residual[pressure]),
        "raw_tangent_relative_asymmetry": relative(tangent),
        "pressure_sign_corrected_tangent_relative_asymmetry": relative(symmetric_tangent),
        "pressure_row_sign": -1, "symmetry_norm": "Frobenius",
        "mass_relative_asymmetry": relative(mass),
        "mass_pressure_row_max": max_abs(mass[pressure].data),
        "mass_pressure_column_max": max_abs(mass[:, pressure].data),
        "constrained_inertia": "not computed: indefinite mixed tangent is not constrained inertia",
        "rightmost_spectrum": "not computed: pressure mass is singular; no unshifted LR claim",
    }


def eigenpair_residuals(tangent, mass, values, vectors, pressure) -> list[dict]:
    """Backward residuals for K v + lambda M v = 0, plus constraint-row residuals."""
    from scipy.sparse.linalg import norm

    values, vectors = np.asarray(values, complex), np.asarray(vectors, complex)
    if vectors.shape != (len(values), tangent.shape[0]):
        raise ValueError("eigenvectors must be returned as rows in the assembled DOF order")
    if not np.isfinite(vectors).all() or not np.isfinite(values).all():
        raise ValueError("eigenpairs must be finite")
    knorm, mnorm = float(norm(tangent)), float(norm(mass))
    records = []
    for value, vector in zip(values, vectors):
        vnorm = float(np.linalg.norm(vector))
        if vnorm == 0:
            raise ValueError("zero eigenvector")
        Kv, Mv = tangent @ vector, mass @ vector
        error = Kv + value * Mv
        scale = (knorm + abs(value) * mnorm) * vnorm
        action_scale = np.linalg.norm(Kv) + abs(value) * np.linalg.norm(Mv)
        records.append({"real": float(value.real), "imag": float(value.imag),
                        "absolute_residual_norm": float(np.linalg.norm(error)),
                        "relative_backward_residual": float(np.linalg.norm(error) / max(scale, np.finfo(float).tiny)),
                        "relative_action_residual": float(np.linalg.norm(error) / max(action_scale, np.finfo(float).tiny)),
                        "constraint_residual_norm_per_vector_norm": float(np.linalg.norm(error[pressure]) / vnorm),
                        "constraint_Kv_norm_per_vector_norm": float(np.linalg.norm(Kv[pressure]) / vnorm)})
    return records
