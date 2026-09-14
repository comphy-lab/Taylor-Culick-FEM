"""Integration checks for G--lambda constitutive evolution in pyoomph."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("HWLOC_COMPONENTS", "-gl")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest
from pyoomph import Equations, Problem
from pyoomph.equations.viscoelastic import OldroydB, ViscoelasticEquations
from pyoomph.expressions import var, vector
from pyoomph.meshes.simplemeshes import RectangularQuadMesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from taylor_culick_fem.viscoelastic import (
    GLambdaOldroydBParameters,
    make_g_lambda_oldroyd_b_equations,
)


class _ImposedVelocity(Equations):
    def __init__(self, expression: Any) -> None:
        super().__init__()
        self.expression = expression

    def define_fields(self) -> None:
        self.define_field_by_substitution("velocity", self.expression, also_on_interface=True)


class _EvolutionProblem(Problem):
    def __init__(
        self, velocity: Any, constitutive: ViscoelasticEquations, axisymmetric: bool
    ) -> None:
        super().__init__()
        self.velocity = velocity
        self.constitutive = constitutive
        self.axisymmetric = axisymmetric

    def define_problem(self) -> None:
        if self.axisymmetric:
            self.set_coordinate_system("axisymmetric")
            # Keep the element away from r=0, where the analytic axisymmetric
            # decomposition divides by radius.
            self.add_mesh(RectangularQuadMesh(N=1, size=[1, 1], lower_left=[1, 0]))
        else:
            self.add_mesh(RectangularQuadMesh(N=1))
        self.add_equations((_ImposedVelocity(self.velocity) + self.constitutive) @ "domain")


def _adapter_equations(params: GLambdaOldroydBParameters) -> ViscoelasticEquations:
    equations = make_g_lambda_oldroyd_b_equations(
        params, space="C1", stabilization=None, output_conformation=True
    )
    equations.add_polymer_stress_to_momentum = False
    return equations


def _run(
    output_dir: Path,
    velocity: Any,
    equations: ViscoelasticEquations,
    *,
    axisymmetric: bool = False,
    end_time: float = 0.25,
    time_step: float = 0.005,
) -> tuple[np.ndarray, dict[str, str]]:
    with _EvolutionProblem(velocity, equations, axisymmetric) as problem:
        problem.set_output_directory(str(output_dir))
        problem.initialise()
        problem.run(
            end_time,
            startstep=time_step,
            maxstep=time_step,
            temporal_error=None,
            outstep=False,
        )
        mesh = problem.get_mesh("domain")
        indices = mesh.get_nodal_field_indices()
        node = next(iter(mesh.nodes()))

        suffixes = {"rr": "xx", "zz": "yy", "theta_theta": "aa"}
        if not axisymmetric:
            suffixes = {"xx": "xx", "yy": "yy", "zz": "zz"}

        psi = np.zeros((3, 3))
        psi[0, 0] = node.value(indices["log_conformation_xx"])
        psi[1, 1] = node.value(indices["log_conformation_yy"])
        if "log_conformation_aa" in indices:
            psi[2, 2] = node.value(indices["log_conformation_aa"])
        psi[0, 1] = psi[1, 0] = node.value(indices["log_conformation_xy"])
        eigenvalues, eigenvectors = np.linalg.eigh(psi)
        conformation = eigenvectors @ np.diag(np.exp(eigenvalues)) @ eigenvectors.T
        return conformation, suffixes


def test_finite_lambda_matches_builtin_oldroyd_b_evolution(tmp_path: Path) -> None:
    """Verify adapter evolution against pyoomph's viscosity--lambda form."""

    elastic_modulus = 0.3
    relaxation_time = 0.7
    shear_rate = 0.4
    velocity = vector(shear_rate * var("coordinate_y"), 0)

    adapter_C, _ = _run(
        tmp_path / "adapter",
        velocity,
        _adapter_equations(GLambdaOldroydBParameters(elastic_modulus, relaxation_time)),
    )
    builtin_C, _ = _run(
        tmp_path / "builtin",
        velocity,
        ViscoelasticEquations(
            model=OldroydB(),
            relaxation_time=relaxation_time,
            polymer_viscosity=elastic_modulus * relaxation_time,
            formulation="log-conf",
            space="C1",
            stabilization=None,
            add_polymer_stress_to_momentum=False,
            output_conformation=True,
        ),
    )

    np.testing.assert_allclose(adapter_C, builtin_C, rtol=2e-10, atol=2e-12)
    identity = np.identity(3)
    adapter_stress = elastic_modulus * (adapter_C - identity)
    builtin_stress = elastic_modulus * relaxation_time / relaxation_time * (builtin_C - identity)
    np.testing.assert_allclose(adapter_stress, builtin_stress, rtol=2e-10, atol=2e-12)


def test_infinite_lambda_axisymmetric_extension_is_isochoric(tmp_path: Path) -> None:
    """Verify the q=0 axisymmetric elastic evolution and hoop-component mapping."""

    extension_rate = 0.5
    end_time = 0.2
    velocity = vector(
        -extension_rate * var("coordinate_x") / 2,
        extension_rate * var("coordinate_y"),
    )
    conformation, suffixes = _run(
        tmp_path / "axisymmetric-elastic",
        velocity,
        _adapter_equations(GLambdaOldroydBParameters(0.4, math.inf)),
        axisymmetric=True,
        end_time=end_time,
        time_step=0.002,
    )

    # In pyoomph's (r,z) slice, xx=rr, yy=zz and aa=theta-theta.
    assert suffixes == {"rr": "xx", "zz": "yy", "theta_theta": "aa"}
    expected = np.diag(
        [
            math.exp(-extension_rate * end_time),
            math.exp(2 * extension_rate * end_time),
            math.exp(-extension_rate * end_time),
        ]
    )
    np.testing.assert_allclose(conformation, expected, rtol=2e-7, atol=2e-9)
    assert np.linalg.det(conformation) == pytest.approx(1.0, abs=5e-8)
