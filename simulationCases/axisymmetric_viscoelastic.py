"""Axisymmetric co-moving Taylor--Culick retraction with Oldroyd-B elasticity.

This is deliberately a thin extension of :mod:`axisymmetric_comoving`.  The
mesh, ALE kinematics, sharp free-surface traction, tip control and laboratory-
frame velocity are inherited unchanged.  The only new bulk contribution is
the polymer conformation equation and its stress ``G (C - I)``.

The dimensionless modulus used here is ``G = G_dim h0 / sigma``, on the same
full-thickness ``h0`` and surface-tension scale as the inherited problem.

``relaxation_time=math.inf`` is an exact non-relaxing limit: no large finite
surrogate is used.  The incoming film is unstretched (``log(C)=0``), while the
midplane imposes only the tensor symmetry condition.  The free surface has no
conformation boundary condition because it is a material boundary.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from axisymmetric_comoving import AxisymmetricComovingProblem
from pyoomph import DirichletBC

from taylor_culick_fem.viscoelastic import (
    GLambdaOldroydBParameters,
    make_g_lambda_oldroyd_b_equations,
)


class AxisymmetricViscoelasticProblem(AxisymmetricComovingProblem):
    """Co-moving axisymmetric sheet with solvent viscosity and polymer elasticity.

    ``solvent_oh`` is the project's Ohnesorge number, so the viscosity passed
    to Navier--Stokes is ``mu_s = sqrt(2) * solvent_oh``.  ``elastic_modulus``
    is ``G = G_dim h0 / sigma`` and ``relaxation_time`` is ``lambda`` in
    capillary units.
    """

    def __init__(self):
        super().__init__()
        self.solvent_oh = 0.0
        self.elastic_modulus = 1.0
        self.relaxation_time = math.inf
        self.conformation_space = "C2"
        self.conformation_stabilization = "SUPG"
        self.conformation_supg_factor = 1.0
        self.viscoelastic_parameters: GLambdaOldroydBParameters | None = None

    def define_problem(self):
        if not math.isfinite(self.solvent_oh) or self.solvent_oh < 0:
            raise ValueError("solvent_oh must be finite and non-negative")
        if not math.isfinite(self.elastic_modulus) or self.elastic_modulus < 0:
            raise ValueError("elastic_modulus must be finite and non-negative")
        if not (self.relaxation_time > 0):
            raise ValueError("relaxation_time must be positive (or infinity)")

        # NavierStokesEquations in the inherited problem carries the solvent
        # stress only.  Velocities remain laboratory-frame velocities.
        self.mu = math.sqrt(2.0) * self.solvent_oh
        super().define_problem()

        # G=0 is intentionally an exact Newtonian baseline: adding decoupled
        # conformation dofs would change the algebraic system for no physics.
        if self.elastic_modulus == 0:
            self.viscoelastic_parameters = None
            return

        params = GLambdaOldroydBParameters(
            elastic_modulus=self.elastic_modulus,
            relaxation_time=self.relaxation_time,
        )
        self.viscoelastic_parameters = params
        eqs = make_g_lambda_oldroyd_b_equations(
            params,
            formulation="log-conf",
            space=self.conformation_space,
            stabilization=self.conformation_stabilization,
            supg_factor=self.conformation_supg_factor,
            output_conformation=True,
        )

        # Undisturbed material enters through the translating far boundary.
        # In axisymmetry the azimuthal tensor component is named ``aa``.
        eqs += (
            DirichletBC(
                log_conformation_xx=0,
                log_conformation_xy=0,
                log_conformation_yy=0,
                log_conformation_aa=0,
            )
            @ "far"
        )
        # Reflection symmetry forbids only the r-z shear component.  The
        # normal components are evolved, not prescribed.  There is
        # deliberately no conformation condition on ``interface``.
        eqs += DirichletBC(log_conformation_xy=0) @ "symmetry"
        self += eqs @ "liquid"

    def polymer_diagnostic_extrema(self) -> dict[str, float]:
        """Return nodal conformation/stress extrema available for a smoke run.

        The primary unknown is ``Psi=log(C)``.  Reconstructing ``C=exp(Psi)``
        at nodes makes these diagnostics independent of optional VTK local-
        expression output.  For this Oldroyd-B parameterisation
        ``tau_p=G(C-I)`` exactly, including at infinite relaxation time.
        """
        if self.elastic_modulus == 0:
            return {}

        mesh = self.get_mesh("liquid")
        indices = mesh.get_nodal_field_indices()
        names = [
            "log_conformation_xx",
            "log_conformation_xy",
            "log_conformation_yy",
            "log_conformation_aa",
        ]
        missing = [name for name in names if name not in indices]
        if missing:
            raise RuntimeError(f"missing conformation fields: {', '.join(missing)}")

        psi_abs_max = 0.0
        c_min = math.inf
        c_max = -math.inf
        stress_min = math.inf
        stress_max = -math.inf
        for node in mesh.nodes():
            psi = np.array(
                [
                    [node.value(indices[names[0]]), node.value(indices[names[1]]), 0.0],
                    [node.value(indices[names[1]]), node.value(indices[names[2]]), 0.0],
                    [0.0, 0.0, node.value(indices[names[3]])],
                ],
                dtype=float,
            )
            psi_eigenvalues = np.linalg.eigvalsh(psi)
            c_eigenvalues = np.exp(psi_eigenvalues)
            stress_eigenvalues = self.elastic_modulus * (c_eigenvalues - 1.0)
            psi_abs_max = max(psi_abs_max, float(np.max(np.abs(psi_eigenvalues))))
            c_min = min(c_min, float(c_eigenvalues[0]))
            c_max = max(c_max, float(c_eigenvalues[-1]))
            stress_min = min(stress_min, float(stress_eigenvalues[0]))
            stress_max = max(stress_max, float(stress_eigenvalues[-1]))

        return {
            "log_conformation_abs_max": psi_abs_max,
            "conformation_eigenvalue_min": c_min,
            "conformation_eigenvalue_max": c_max,
            "polymer_stress_eigenvalue_min": stress_min,
            "polymer_stress_eigenvalue_max": stress_max,
        }
