"""Oldroyd-B equations parameterised by elastic modulus and relaxation time.

The pyoomph constitutive API normally uses polymer viscosity
``eta_p = G * lambda``.  That parameterisation is singular at infinite
relaxation time.  This adapter instead keeps ``G`` finite and moves the
inverse relaxation time into the constitutive model:

``upper_convected(C) = -(C - I) / lambda`` and
``tau_p = G * (C - I)``.

Consequently, ``lambda=inf`` is represented exactly by a zero relaxation
rate; symbolic infinity never enters the assembled equations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pyoomph.equations.viscoelastic import (
    ViscoelasticConstitutiveModel,
    ViscoelasticEquations,
)

Formulation = Literal["log-conf", "conformation"]
Stabilization = Literal["SUPG"] | None


def _finite_nonnegative(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite non-negative number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _positive_relaxation_time(value: float | str) -> float:
    if isinstance(value, bool):
        raise TypeError("relaxation_time must be positive or 'inf'")
    if isinstance(value, str):
        normalised = value.strip().casefold()
        if normalised in {"inf", "+inf", "infinity", "+infinity", "∞"}:
            return math.inf
        try:
            result = float(normalised)
        except ValueError as exc:
            raise ValueError("relaxation_time must be positive or 'inf'") from exc
    else:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("relaxation_time must be positive or 'inf'") from exc
    if math.isnan(result) or result <= 0 or result == -math.inf:
        raise ValueError("relaxation_time must be positive or 'inf'")
    return result


@dataclass(frozen=True, slots=True)
class GLambdaOldroydBParameters:
    """Validated ``G``--``lambda`` parameters for an Oldroyd-B material."""

    elastic_modulus: float
    relaxation_time: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "elastic_modulus",
            _finite_nonnegative(self.elastic_modulus, name="elastic_modulus"),
        )
        object.__setattr__(
            self,
            "relaxation_time",
            _positive_relaxation_time(self.relaxation_time),
        )

    @property
    def relaxation_rate(self) -> float:
        """Return ``1/lambda``, exactly zero when ``lambda`` is infinite."""

        if math.isinf(self.relaxation_time):
            return 0.0
        return 1.0 / self.relaxation_time

    def as_json_dict(self) -> dict[str, float | str]:
        """Return metadata without emitting non-standard JSON infinity."""

        relaxation_time: float | str = self.relaxation_time
        if math.isinf(self.relaxation_time):
            relaxation_time = "inf"
        return {
            "elastic_modulus": self.elastic_modulus,
            "relaxation_time": relaxation_time,
        }


class RelaxationRateOldroydB(ViscoelasticConstitutiveModel):
    """Oldroyd-B model whose relaxation function includes ``1/lambda``.

    ``ViscoelasticEquations`` is constructed with a unit reference relaxation
    time, so scaling both relaxation forms here gives the physical decay rate
    while leaving the stress function unscaled.
    """

    def __init__(self, relaxation_rate: float) -> None:
        self.relaxation_rate = _finite_nonnegative(relaxation_rate, name="relaxation_rate")

    def relaxation_matrix(self, C: Any, trace: Any, identity: Any) -> Any:
        del trace
        return self.relaxation_rate * (C - identity)

    def log_relaxation_matrix(self, C: Any, Cinv: Any, trace: Any, identity: Any) -> Any:
        del C, trace
        return self.relaxation_rate * (identity - Cinv)

    def stress_matrix(self, C: Any, trace: Any, identity: Any) -> Any:
        del trace
        return C - identity


def g_lambda_relaxation(
    conformation: ArrayLike, relaxation_time: float | str
) -> NDArray[np.float64]:
    """Evaluate ``g(C) = (C-I)/lambda`` numerically."""

    C = np.asarray(conformation, dtype=float)
    identity = np.identity(_square_matrix_size(C))
    rate = GLambdaOldroydBParameters(0, relaxation_time).relaxation_rate
    return rate * (C - identity)


def g_lambda_log_relaxation(
    conformation: ArrayLike, relaxation_time: float | str
) -> NDArray[np.float64]:
    """Evaluate ``C^-1 g(C)`` numerically."""

    C = np.asarray(conformation, dtype=float)
    identity = np.identity(_square_matrix_size(C))
    rate = GLambdaOldroydBParameters(0, relaxation_time).relaxation_rate
    return rate * (identity - np.linalg.inv(C))


def g_lambda_polymer_stress(conformation: ArrayLike, elastic_modulus: float) -> NDArray[np.float64]:
    """Evaluate ``tau_p = G(C-I)`` numerically."""

    C = np.asarray(conformation, dtype=float)
    identity = np.identity(_square_matrix_size(C))
    G = _finite_nonnegative(elastic_modulus, name="elastic_modulus")
    return G * (C - identity)


def _square_matrix_size(matrix: NDArray[np.float64]) -> int:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("conformation must be a square matrix")
    return matrix.shape[0]


def make_g_lambda_oldroyd_b_equations(
    params: GLambdaOldroydBParameters,
    *,
    formulation: Formulation = "log-conf",
    space: Any = "C2",
    stabilization: Stabilization = "SUPG",
    supg_factor: Any = 1,
    output_conformation: bool = True,
) -> ViscoelasticEquations:
    """Build pyoomph equations for the ``G``--``lambda`` formulation.

    The unit relaxation time is a finite bookkeeping scale.  The constitutive
    model contains the physical inverse relaxation time, including exact zero
    for the elastic limit, while ``polymer_viscosity=G`` produces
    ``tau_p=G(C-I)``.  When SUPG is enabled, the same unit scale regularises
    its intrinsic time at stagnation points.  This changes only the consistent
    numerical stabilisation, not the continuous constitutive equation; SUPG
    sensitivity must therefore be checked before quantitative use.
    """

    if not isinstance(params, GLambdaOldroydBParameters):
        raise TypeError("params must be GLambdaOldroydBParameters")
    return ViscoelasticEquations(
        model=RelaxationRateOldroydB(params.relaxation_rate),
        relaxation_time=1.0,
        polymer_viscosity=params.elastic_modulus,
        formulation=formulation,
        space=space,
        stabilization=stabilization,
        supg_factor=supg_factor,
        output_conformation=output_conformation,
    )
