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
        "positive_returned_mode_detected": bool(np.any(values.real > 0)),
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
    background = "displacement_background_x" in fields
    if background != ("displacement_background_y" in fields):
        raise ValueError("incomplete pinned displacement background")
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
        for axis, value in zip(("x", "y"), u):
            node.set_value(fields["displacement_" + axis], 0.0 if background else float(value))
            if background:
                node.set_value(fields["displacement_background_" + axis], float(value))
    after = FieldMap.from_problem(problem)
    restored = retained_nodes(after.Xe, after.ue, after.ij)
    if any(not np.array_equal(restored[X], u) for X, u in lookup.items()):
        raise ValueError("restoration changed retained total nodal displacement")
    return {"method": "exact reference-node and Q2-connectivity match; total displacement",
            "nodes": len(lookup), "elements": len(saved["Xe"]),
            "max_nodal_displacement_change": 0.0,
            "representation": "pinned Q2 background plus zero increment" if background else "total Q2 displacement",
            "pressure_retained": False, "stationary_solve_performed": False}


def stationary_correction(problem, *, max_iterations: int, geometry_tolerance=1e-8,
                          nodal_tolerance=1e-8) -> dict:
    """Bound one same-model/grid solve in a translated displacement trial space.

    This operates only on the rebuilt problem. The retained input is unchanged;
    the post-correction Hessian belongs to the corrected discrete equilibrium.
    """
    from taylor_culick_fem.equilibrium_geometry import FieldMap, incompressibility_error

    if not getattr(problem, "displacement_background", False):
        raise ValueError("bounded stationary correction requires a pinned displacement background")
    before = FieldMap.from_problem(problem)
    radius, height = problem.hole_radius(), problem.rim_height()
    sampled_j = incompressibility_error(before)
    residual_before = max_abs(problem.get_residuals())
    problem.solve(max_newton_iterations=max_iterations)
    after = FieldMap.from_problem(problem)
    if not np.array_equal(before.Xe, after.Xe) or not np.array_equal(before.ij, after.ij):
        raise ValueError("stationary correction changed the retained reference grid")
    mesh = problem.get_mesh("solid")
    indices = mesh.get_nodal_field_indices()
    maximum_increment = max(abs(node.value(indices["displacement_" + axis]))
                            for node in mesh.nodes() for axis in ("x", "y"))
    record = {
        "method": "one stationary solve at unchanged Ec, regularisation, reference grid and constraints",
        "representation": "saved total Q2 displacement as pinned background plus free Q2 increment",
        "max_residual_before": residual_before, "max_residual_after": max_abs(problem.get_residuals()),
        "relative_R_change": problem.hole_radius() / radius - 1,
        "relative_H_change": problem.rim_height() / height - 1,
        "sampled_J_error_change": incompressibility_error(after) - sampled_j,
        "max_total_nodal_displacement_change": max_abs(after.ue - before.ue),
        "max_increment_nodal_displacement": maximum_increment,
        "geometry_tolerance": geometry_tolerance, "nodal_tolerance": nodal_tolerance,
        "residual_history": list(problem.get_last_residual_convergence()),
        "retained_input_modified": False,
        "interpretation": "screen belongs to corrected discrete equilibrium, not the untouched retained coefficients",
    }
    checks = [record["relative_R_change"], record["relative_H_change"], record["sampled_J_error_change"],
              record["max_total_nodal_displacement_change"], maximum_increment, record["max_residual_after"]]
    record["geometry_gate_passed"] = bool(
        np.isfinite(checks).all() and max(map(abs, checks[:3])) <= geometry_tolerance
        and max(checks[3:5]) <= nodal_tolerance)
    return record


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


def retained_eigen_matrices(problem):
    """Detach complete serial CSR matrices before another native assembly.

    The pinned native binding exposes views of cached C++ matrix arrays. Its
    next eigenproblem assembly deletes those caches. Independent copies of
    data, column indices and row pointers are therefore required for later
    residual analysis, even while the Problem object remains alive.
    """
    from scipy.sparse import csr_matrix

    n, _, nr, mv, mc, mr, _, kr, kv, kc, kp = problem.assemble_eigenproblem_matrices(0)
    if nr != n or kr != n or n != problem.ndof():
        raise ValueError("eigenproblem assembly is not the complete serial DOF system")
    owned = lambda values, columns, starts: csr_matrix(
        (np.array(values, copy=True), np.array(columns, copy=True), np.array(starts, copy=True)),
        shape=(n, n))
    return owned(mv, mc, mr), owned(kv, kc, kp)


def _augmented_pressure_fit(block, rhs, tolerance: float, refinements: int):
    """Unweighted least squares without forming the squared-condition normal matrix."""
    from scipy.sparse import bmat, eye
    from scipy.sparse.linalg import norm, splu

    rows, columns = block.shape
    augmented = bmat([[eye(rows, format="csc"), block], [block.T, None]], format="csc")
    full_rhs = np.concatenate((rhs, np.zeros(columns)))
    factor = splu(augmented)
    solution = factor.solve(full_rhs)
    anorm = float(norm(augmented))
    tiny = np.finfo(float).tiny
    history = []
    for step in range(refinements + 1):
        error = full_rhs - augmented @ solution
        scale = anorm * float(np.linalg.norm(solution)) + float(np.linalg.norm(full_rhs))
        backward = float(np.linalg.norm(error) / max(scale, tiny))
        action = float(np.linalg.norm(error) / max(
            float(np.linalg.norm(augmented @ solution)) + float(np.linalg.norm(full_rhs)), tiny))
        history.append({"refinement": step, "backward_residual": backward,
                        "relative_action_residual": action})
        if not np.isfinite(solution).all() or not np.isfinite([backward, action]).all():
            raise ValueError("augmented pressure reconstruction produced non-finite values")
        if max(backward, action) <= tolerance or step == refinements:
            break
        solution += factor.solve(error)
    return solution[rows:], {
        "driver": "SuperLU on [I C; C.T 0]; C is column-equilibrated B",
        "normal_matrix_formed": False, "augmented_ndof": rows + columns,
        "augmented_nnz": int(augmented.nnz), "refinement_limit": refinements,
        "refinement_history": history, "iterations": len(history) - 1,
        "converged": max(history[-1]["backward_residual"],
                         history[-1]["relative_action_residual"]) <= tolerance,
        "convergence_scope": "augmented linear-system residual, not physical stationarity",
    }


def recover_pressure(problem, pressure, *, tolerance: float, iterations: int,
                     method: str = "augmented", refinements: int = 4) -> dict:
    """Recover missing pressure at fixed displacement, recording least-squares limits.

    The retained state has no pressure. The constitutive residual is affine in
    pressure; solve its displacement rows without moving the retained geometry.
    This is pressure reconstruction, not an equilibrium correction.
    Column equilibration changes the pressure coordinates, not the physical
    least-squares objective: solve (B D^-1) q = -r, then delta_p = D^-1 q.
    No row weighting or physical-residual tolerance change is applied.
    The direct augmented system avoids forming B.T B. LSQR is an explicit
    alternative; neither driver's convergence establishes physical stationarity.
    """
    from scipy.sparse import diags
    from scipy.sparse.linalg import lsqr

    if (method not in ("augmented", "lsqr") or not np.isfinite(tolerance) or tolerance <= 0
            or iterations <= 0 or not isinstance(refinements, int) or refinements < 0):
        raise ValueError("invalid pressure reconstruction method or numerical controls")
    residual, tangent = problem.assemble_jacobian(with_residual=True)
    residual = np.asarray(residual)
    if max_abs(tangent[pressure][:, pressure].data) != 0:
        raise ValueError("pressure reconstruction requires a zero pressure-pressure block")
    block = tangent[~pressure][:, pressure]
    if not np.isfinite(block.data).all() or not np.isfinite(residual).all():
        raise ValueError("pressure reconstruction requires finite force residuals and pressure block")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        column_norms = np.sqrt(np.asarray(block.power(2).sum(axis=0)).ravel())
    if not len(column_norms) or not np.isfinite(column_norms).all() or np.any(column_norms <= 0):
        raise ValueError("pressure block column norms must be nonzero and finite")
    inverse_norms = 1.0 / column_norms
    if not np.isfinite(inverse_norms).all():
        raise ValueError("inverse pressure block column norms must be finite")
    scaled_block = block @ diags(inverse_norms)
    rhs = -residual[~pressure]
    if method == "augmented":
        scaled_increment, driver = _augmented_pressure_fit(scaled_block, rhs, tolerance, refinements)
    else:
        fit = lsqr(scaled_block, rhs, atol=tolerance, btol=tolerance, iter_lim=iterations)
        scaled_increment = fit[0]
        driver = {"driver": "LSQR", "stop_code": int(fit[1]), "iterations": int(fit[2]),
                  "lsqr_residual_norm": float(fit[3]),
                  "lsqr_scaled_normal_residual_norm": float(fit[7]),
                  "estimated_condition": float(fit[6]),
                  "condition_estimate_scope": "column-equilibrated pressure block",
                  "converged": int(fit[1]) in (0, 1, 2, 4, 5), "iteration_limit": iterations,
                  "convergence_scope": "LSQR stop criterion, not physical stationarity"}
    pressure_increment = scaled_increment * inverse_norms
    physical_residual = residual[~pressure] + block @ pressure_increment
    physical_normal_residual = block.T @ physical_residual
    if not np.isfinite(physical_residual).all() or not np.isfinite(physical_normal_residual).all():
        raise ValueError("pressure reconstruction produced non-finite unscaled residuals")
    dofs, _ = problem.get_current_dofs()
    updated = np.asarray(dofs, dtype=float).copy()
    updated[pressure] += pressure_increment
    if not np.isfinite(updated).all():
        raise ValueError("pressure reconstruction produced non-finite values")
    problem.set_current_dofs(updated)
    return {"method": "fixed-displacement pressure-block least squares with column norm scaling",
            **driver,
            "column_norm_min": float(column_norms.min()),
            "column_norm_max": float(column_norms.max()),
            "column_scaling": "q_j = ||B[:,j]||_2 * delta_p_j; no row weighting",
            "residual_norm": float(np.linalg.norm(physical_residual)),
            "max_unscaled_force_residual": max_abs(physical_residual),
            "normal_residual_norm": float(np.linalg.norm(physical_normal_residual)),
            "tolerance": tolerance}


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
    upper = tangent[~pressure][:, pressure]
    lower = tangent[pressure][:, ~pressure]
    return {
        "finite_values": True, "ndof": len(residual),
        "pressure_dofs": int(np.count_nonzero(pressure)),
        "tangent_nnz": int(tangent.nnz), "mass_nnz": int(mass.nnz),
        "max_equilibrium_residual": max_abs(residual),
        "max_displacement_residual": max_abs(residual[~pressure]),
        "max_constraint_residual": max_abs(residual[pressure]),
        "raw_tangent_relative_asymmetry": relative(tangent),
        "pressure_sign_corrected_tangent_relative_asymmetry": relative(symmetric_tangent),
        "displacement_tangent_relative_asymmetry": relative(tangent[~pressure][:, ~pressure]),
        "pressure_coupling_sign_corrected_relative_difference": float(
            norm(upper + lower.T) / max(float(norm(upper) + norm(lower)), np.finfo(float).tiny)),
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


def constrained_curvature_screen(tangent, pressure, *, weights=(10.0, 100.0),
                                 symmetry_tolerance=1e-10, factor_tolerance=1e-9,
                                 probe_tolerance=1e-9, pivot_floor_multiplier=100.0) -> dict:
    """Optional sufficient numerical screen of the discrete constrained energy.

    Let H be the symmetric displacement tangent and B the constraint rows.
    For finite w>0 and positive diagonal W, H+w B.T W B positive definite
    implies v.T H v>0 for every nonzero v in ker(B). Strictly positive
    diagonal displacement congruence and constraint row scaling preserve this
    implication. No null-space basis or eigenvalue completeness is required.

    SuperLU is used only as a candidate symmetric LDL.T factorisation: equal
    row/column permutations, U approximately diag(U)L.T, positive pivots above
    a numerical floor and deterministic residual probes are all required.
    Generic LU pivots are never counted as inertia. These floating-point checks
    are numerical evidence, not rigorous error bounds or a continuum theorem.
    Failure establishes neither a negative constrained direction nor instability.
    """
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import norm, splu

    weights = tuple(float(w) for w in weights)
    thresholds = (symmetry_tolerance, factor_tolerance, probe_tolerance, pivot_floor_multiplier)
    if (not weights or not np.isfinite(weights).all() or any(w <= 0 for w in weights)
            or len(set(weights)) != len(weights)
            or not np.isfinite(thresholds).all() or any(t <= 0 for t in thresholds)):
        raise ValueError("curvature weights and numerical thresholds must be finite and positive")
    K = csr_matrix(tangent, dtype=float)
    pressure = np.asarray(pressure, dtype=bool)
    if (K.shape != (len(pressure), len(pressure)) or not np.isfinite(K.data).all()
            or not pressure.any() or pressure.all()):
        raise ValueError("curvature screen requires a finite mixed displacement/pressure tangent")
    report = {
        "numerical_screen_passed": False, "rigorous_certificate": False,
        "classification": "numerical constrained-curvature screen unestablished",
        "scope": "regularised discrete meridional m=0 reflection-even constrained energy",
        "implication": "H+w B.T W B positive definite implies H positive on ker(B)",
        "limitations": ["floating-point sufficient numerical screen, not rigorous operator error bounds",
                        "no continuum, non-axisymmetric or global nonlinear certificate",
                        "failure does not establish physical instability or constrained negative curvature",
                        "generic LU pivots are not inertia"],
        "factor_error_bound": "no rigorous norm bound established; checked factors may represent a nearby operator",
        "symmetry_tolerance": symmetry_tolerance, "factor_tolerance": factor_tolerance,
        "probe_tolerance": probe_tolerance, "pivot_floor_multiplier": pivot_floor_multiplier,
        "weights": list(weights), "attempts": [],
    }
    tiny = np.finfo(float).tiny
    signed = diags(np.where(pressure, -1.0, 1.0)) @ K
    asymmetry = float(norm(signed - signed.T) / max(float(norm(signed)), tiny))
    report["pressure_sign_corrected_relative_asymmetry"] = asymmetry
    report["pressure_pressure_block_max"] = max_abs(K[pressure][:, pressure].data)
    if asymmetry > symmetry_tolerance or report["pressure_pressure_block_max"] != 0:
        report["reason"] = "variational symmetry or zero pressure-pressure block assumption failed"
        return report
    Hraw = K[~pressure][:, ~pressure]
    B = K[pressure][:, ~pressure]
    upper = K[~pressure][:, pressure]
    report["displacement_tangent_relative_asymmetry"] = float(
        norm(Hraw - Hraw.T) / max(float(norm(Hraw)), tiny))
    report["pressure_coupling_sign_corrected_relative_difference"] = float(
        norm(upper + B.T) / max(float(norm(upper) + norm(B)), tiny))
    if max(report["displacement_tangent_relative_asymmetry"],
           report["pressure_coupling_sign_corrected_relative_difference"]) > symmetry_tolerance:
        report["reason"] = "separate displacement or pressure-coupling variational symmetry failed"
        return report
    H = (Hraw + Hraw.T) * 0.5
    report["energy_tangent"] = "symmetric part of Kuu; skew part contributes zero to quadratic energy"
    report["discarded_skew_relative_norm"] = float(norm(Hraw - H) / max(float(norm(Hraw)), tiny))
    row_max = np.asarray(abs(H).max(axis=1).toarray()).ravel()
    scaling = 1 / np.sqrt(np.maximum(row_max, tiny))
    if not np.isfinite(scaling).all() or np.any(scaling <= 0):
        report["reason"] = "strictly positive finite displacement congruence unavailable"
        return report
    D = diags(scaling)
    Hscaled = D @ H @ D
    C = B @ D
    # Normalise constraints without squaring unscaled extreme coefficients.
    maxima = np.asarray(abs(C).max(axis=1).toarray()).ravel()
    safe = np.where(maxima > 0, maxima, 1.0)
    Cnormal = diags(1 / safe) @ C
    lengths = np.sqrt(np.asarray(Cnormal.power(2).sum(axis=1)).ravel())
    row_scaling = (1 / safe) / np.where(lengths > 0, lengths, 1.0)
    if not np.isfinite(row_scaling).all() or np.any(row_scaling <= 0):
        report["reason"] = "strictly positive finite constraint row scaling unavailable"
        return report
    C = diags(row_scaling) @ C
    penalty = C.T @ C
    report["scaling"] = {"displacement_diagonal_min": float(scaling.min()),
                         "displacement_diagonal_max": float(scaling.max()),
                         "constraint_row_diagonal_min": float(row_scaling.min()),
                         "constraint_row_diagonal_max": float(row_scaling.max()),
                         "zero_constraint_rows": int(np.count_nonzero(maxima == 0)),
                         "constraint_weight": "W=diag(constraint_row_diagonal**2)>0"}
    if report["scaling"]["zero_constraint_rows"]:
        report["reason"] = "zero constraint row invalidates the mixed-constraint screen"
        return report
    for weight in weights:
        attempt = {"weight": weight, "passed": False}
        report["attempts"].append(attempt)
        try:
            A = (Hscaled + weight * penalty).tocsc()
            if not np.isfinite(A.data).all():
                raise ValueError("penalised matrix is non-finite")
            anorm = float(norm(A))
            attempt["relative_asymmetry"] = float(norm(A - A.T) / max(anorm, tiny))
            if attempt["relative_asymmetry"] > symmetry_tolerance:
                raise ValueError("penalised operator symmetry check failed")
            factor = splu(A, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                          options={"SymmetricMode": True, "Equil": False})
            attempt["equal_row_column_permutations"] = bool(np.array_equal(factor.perm_r, factor.perm_c))
            if not attempt["equal_row_column_permutations"]:
                raise ValueError("row pivoting invalidates the candidate symmetric factorisation")
            pivots = factor.U.diagonal()
            floor = pivot_floor_multiplier * np.finfo(float).eps * max(
                1.0, float(np.max(np.asarray(abs(A).sum(axis=1)))))
            attempt.update({"pivot_floor": floor, "minimum_pivot": float(pivots.min()),
                            "finite_factor_values": bool(np.isfinite(factor.L.data).all()
                                                         and np.isfinite(factor.U.data).all())})
            if not attempt["finite_factor_values"] or np.any(pivots <= floor):
                raise ValueError("candidate LDL.T pivots are non-finite or below the positive floor")
            difference = factor.U - diags(pivots) @ factor.L.T
            attempt["U_vs_diagU_Ltranspose_relative_difference"] = float(
                norm(difference) / max(float(norm(factor.U)), tiny))
            if attempt["U_vs_diagU_Ltranspose_relative_difference"] > factor_tolerance:
                raise ValueError("U does not match diag(U)L.T within tolerance")
            # Probe factor actions and solves, never form the full L@U product.
            order = np.argsort(factor.perm_r)
            probes = []
            index = np.arange(A.shape[0], dtype=float) + 1
            for vector in (np.ones(A.shape[0]), np.sin(index), np.cos(index * np.sqrt(2))):
                Av = A @ vector
                factor_action = factor.L @ (factor.U @ vector[order])
                action_error = float(np.linalg.norm(factor_action - Av[order]) / max(
                    float(np.linalg.norm(factor_action) + np.linalg.norm(Av)), tiny))
                solution = factor.solve(vector)
                residual = A @ solution - vector
                backward_error = float(np.linalg.norm(residual) / max(
                    anorm * float(np.linalg.norm(solution)) + float(np.linalg.norm(vector)), tiny))
                solve_action_error = float(np.linalg.norm(residual) / max(
                    float(np.linalg.norm(A @ solution) + np.linalg.norm(vector)), tiny))
                probes.append({"factor_action_relative_residual": action_error,
                               "solve_backward_residual": backward_error,
                               "solve_action_relative_residual": solve_action_error})
            attempt["deterministic_probes"] = probes
            if any(not np.isfinite(list(p.values())).all() or max(p.values()) > probe_tolerance
                   for p in probes):
                raise ValueError("deterministic factor-action or solve residual check failed")
            attempt["passed"] = True
        except (ValueError, RuntimeError) as exc:
            attempt["reason"] = str(exc)
    report["numerical_screen_passed"] = any(a["passed"] for a in report["attempts"])
    if report["numerical_screen_passed"]:
        report["classification"] = "numerical sufficient-curvature screen passed; no rigorous factor-error bound"
    return report
