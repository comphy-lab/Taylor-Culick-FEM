"""Software contracts for axisymmetric active and passive conformation modes."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HWLOC_COMPONENTS", "-gl")
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "simulationCases"))
sys.path.insert(0, str(_ROOT / "src"))

import axisymmetric_viscoelastic as axisymmetric

from taylor_culick_fem.viscoelastic import (
    make_g_lambda_oldroyd_b_equations,
)


def _define_problem(
    monkeypatch,
    *,
    elastic_modulus: float,
    passive: bool,
    relaxation_time: float = float("inf"),
):
    captured: dict[str, object] = {}

    def factory(params, **kwargs):
        equations = make_g_lambda_oldroyd_b_equations(params, **kwargs)
        captured["equations"] = equations
        captured["kwargs"] = kwargs
        return equations

    # The inherited mesh/momentum setup is covered by the existing FEM smoke
    # test. Here the contract under test is the conformation mode wiring.
    monkeypatch.setattr(
        axisymmetric.AxisymmetricComovingProblem, "define_problem", lambda _: None
    )
    monkeypatch.setattr(axisymmetric, "make_g_lambda_oldroyd_b_equations", factory)

    problem = axisymmetric.AxisymmetricViscoelasticProblem()
    problem.elastic_modulus = elastic_modulus
    problem.passive_conformation = passive
    problem.relaxation_time = relaxation_time
    problem.define_problem()
    return problem, captured.get("equations"), captured


def test_active_mode_preserves_polymer_feedback_default(monkeypatch) -> None:
    problem, equations, captured = _define_problem(
        monkeypatch, elastic_modulus=0.4, passive=False
    )

    assert problem.passive_conformation is False
    assert captured["kwargs"]["add_polymer_stress_to_momentum"] is True
    assert equations.add_polymer_stress_to_momentum is True
    assert equations.output_conformation is True


@pytest.mark.parametrize("relaxation_time", [0.001, 1.0, 1000.0])
def test_finite_relaxation_registers_velocity_and_tensor_error(monkeypatch, relaxation_time):
    problem, _, _ = _define_problem(
        monkeypatch, elastic_modulus=1.0, passive=False, relaxation_time=relaxation_time
    )
    equations = problem.additional_equations._children["liquid"]._equation_list
    estimators = [e for e in equations if isinstance(e, axisymmetric.TemporalErrorEstimator)]
    assert len(estimators) == 1
    assert dict(estimators[0].fieldfactors) == {
        "velocity": 1, "log_conformation_xx": 1, "log_conformation_xy": 1,
        "log_conformation_yy": 1, "log_conformation_aa": 1,
    }


def test_zero_modulus_passive_mode_retains_log_conformation_and_boundaries(monkeypatch) -> None:
    problem, equations, captured = _define_problem(
        monkeypatch, elastic_modulus=0.0, passive=True
    )

    assert problem.viscoelastic_parameters is not None
    assert problem.viscoelastic_parameters.elastic_modulus == 0.0
    assert captured["kwargs"]["add_polymer_stress_to_momentum"] is False
    assert equations.add_polymer_stress_to_momentum is False
    assert equations.polymer_viscosity == 0.0
    assert equations.output_conformation is True

    liquid = problem.additional_equations._children["liquid"]
    assert liquid._equation_list == [equations]
    assert set(liquid._children) == {"far", "symmetry"}


def test_zero_modulus_is_exact_newtonian_without_passive_transport(monkeypatch) -> None:
    problem, equations, captured = _define_problem(
        monkeypatch, elastic_modulus=0.0, passive=False
    )

    assert equations is None
    assert captured == {}
    assert problem.viscoelastic_parameters is None


@pytest.mark.parametrize("passive", [False, True])
def test_zero_relaxation_is_exact_newtonian_regardless_of_mode(
    monkeypatch, passive: bool
) -> None:
    problem, equations, captured = _define_problem(
        monkeypatch,
        elastic_modulus=1.0,
        passive=passive,
        relaxation_time=0.0,
    )

    assert equations is None
    assert captured == {}
    assert problem.viscoelastic_parameters is None
    assert problem.polymer_diagnostic_extrema() == {}


@pytest.mark.parametrize("relaxation_time", [-1.0, float("nan")])
def test_invalid_relaxation_time_is_rejected(monkeypatch, relaxation_time: float) -> None:
    problem = axisymmetric.AxisymmetricViscoelasticProblem()
    problem.relaxation_time = relaxation_time

    with pytest.raises(ValueError, match="relaxation_time must be non-negative"):
        problem.define_problem()


def test_passive_zero_stress_diagnostic_still_reports_conformation(monkeypatch) -> None:
    problem, _, _ = _define_problem(monkeypatch, elastic_modulus=0.0, passive=True)

    class Node:
        def value(self, index: int) -> float:
            del index
            return 0.0

    class Mesh:
        def get_nodal_field_indices(self):
            return {
                "log_conformation_xx": 0,
                "log_conformation_xy": 1,
                "log_conformation_yy": 2,
                "log_conformation_aa": 3,
            }

        def nodes(self):
            return [Node()]

    monkeypatch.setattr(problem, "get_mesh", lambda _: Mesh())
    diagnostics = problem.polymer_diagnostic_extrema()

    assert diagnostics["log_conformation_abs_max"] == 0.0
    assert diagnostics["conformation_eigenvalue_min"] == 1.0
    assert diagnostics["conformation_eigenvalue_max"] == 1.0
    assert diagnostics["polymer_stress_eigenvalue_min"] == 0.0
    assert diagnostics["polymer_stress_eigenvalue_max"] == 0.0


def test_passive_mode_requires_a_boolean(monkeypatch) -> None:
    problem = axisymmetric.AxisymmetricViscoelasticProblem()
    problem.passive_conformation = 1

    with pytest.raises(TypeError, match="passive_conformation must be a boolean"):
        problem.define_problem()


def test_rim_height_finds_quadratic_maximum_between_interface_nodes(monkeypatch) -> None:
    class Node:
        def __init__(self, height: float):
            self.height = height

        def x(self, coordinate: int) -> float:
            assert coordinate == 1
            return self.height

    class Element:
        # Deliberately shuffled: the implementation must use local coordinates,
        # not assume endpoint/midside node ordering.
        local = (1.0, -1.0, 0.0)

        def nnode(self):
            return len(self.local)

        def local_coordinate_of_node(self, index):
            return [self.local[index]]

        def node_pt(self, index):
            coordinate = self.local[index]
            return Node(1.0 - (coordinate - 0.25) ** 2)

    class Interface:
        def elements(self):
            return [Element()]

    problem = axisymmetric.AxisymmetricViscoelasticProblem()
    monkeypatch.setattr(problem, "get_mesh", lambda _: Interface())

    assert problem.rim_height() == pytest.approx(2.0)
