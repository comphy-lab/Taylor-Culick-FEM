from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("HWLOC_COMPONENTS", "-gl")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from taylor_culick_fem.viscoelastic import (
    GLambdaOldroydBParameters,
    RelaxationRateOldroydB,
    g_lambda_log_relaxation,
    g_lambda_polymer_stress,
    g_lambda_relaxation,
    make_g_lambda_oldroyd_b_equations,
)


@pytest.mark.parametrize("value", [-1, math.inf, -math.inf, math.nan, True, "bad"])
def test_elastic_modulus_rejects_invalid_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="elastic_modulus"):
        GLambdaOldroydBParameters(value, 1)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -1, -math.inf, math.nan, False, "bad"])
def test_relaxation_time_rejects_invalid_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="relaxation_time"):
        GLambdaOldroydBParameters(1, value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [math.inf, "inf", "Infinity", "+inf", "∞"])
def test_infinite_relaxation_time_has_exact_zero_rate(value: float | str) -> None:
    params = GLambdaOldroydBParameters(2.5, value)  # type: ignore[arg-type]
    assert params.relaxation_rate == 0.0
    assert params.as_json_dict()["relaxation_time"] == "inf"
    assert "Infinity" not in json.dumps(params.as_json_dict())


def test_relaxation_and_log_relaxation_are_identical_representations() -> None:
    C = np.array([[1.8, 0.2, 0.0], [0.2, 1.1, 0.0], [0.0, 0.0, 0.7]])
    g = g_lambda_relaxation(C, 4)
    g_log = g_lambda_log_relaxation(C, 4)
    np.testing.assert_allclose(C @ g_log, g, atol=2e-16)


def test_polymer_stress_is_modulus_times_strain_measure() -> None:
    C = np.array([[1.5, 0.25], [0.25, 0.8]])
    np.testing.assert_allclose(g_lambda_polymer_stress(C, 3.0), 3.0 * (C - np.identity(2)))


def test_constitutive_model_scales_relaxation_but_not_stress() -> None:
    C = np.diag([2.0, 1.5, 0.75])
    identity = np.identity(3)
    model = RelaxationRateOldroydB(0.25)
    np.testing.assert_allclose(
        model.relaxation_matrix(C, C.trace(), identity), 0.25 * (C - identity)
    )
    np.testing.assert_allclose(
        C @ model.log_relaxation_matrix(C, np.linalg.inv(C), C.trace(), identity),
        model.relaxation_matrix(C, C.trace(), identity),
    )
    np.testing.assert_allclose(model.stress_matrix(C, C.trace(), identity), C - identity)


def test_equation_factory_uses_only_finite_base_parameters_at_infinite_lambda() -> None:
    params = GLambdaOldroydBParameters(0.4, math.inf)
    equations = make_g_lambda_oldroyd_b_equations(params)
    assert equations.relaxation_time == 1.0
    assert equations.polymer_viscosity == 0.4
    assert equations.formulation == "log-conf"
    assert equations.stabilization == "SUPG"
    assert isinstance(equations.model, RelaxationRateOldroydB)
    assert equations.model.relaxation_rate == 0.0
