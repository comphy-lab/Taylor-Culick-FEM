"""Shared equations and utilities for Taylor--Culick FEM simulations."""

from .viscoelastic import (
    GLambdaOldroydBParameters,
    RelaxationRateOldroydB,
    g_lambda_log_relaxation,
    g_lambda_polymer_stress,
    g_lambda_relaxation,
    make_g_lambda_oldroyd_b_equations,
)

__all__ = [
    "GLambdaOldroydBParameters",
    "RelaxationRateOldroydB",
    "g_lambda_log_relaxation",
    "g_lambda_polymer_stress",
    "g_lambda_relaxation",
    "make_g_lambda_oldroyd_b_equations",
]
