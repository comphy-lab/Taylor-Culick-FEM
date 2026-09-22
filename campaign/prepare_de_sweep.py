"""Prepare, but never launch, fixed-modulus Deborah-number cases.

Ec = G_dim*h0/sigma; De = lambda_dim/sqrt(rho*h0**3/sigma).
Oh denotes solvent viscosity: mu_s=sqrt(2)*Oh. The polymer viscosity is
Ec*De, so holding Ec fixed recovers the solvent Newtonian problem at De=0.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DE_VALUES = (0.0, *(10.0 ** (k / 2) for k in range(-6, 7)), math.inf)


def make_case(de: float, *, ec: float = 1.0, oh: float = 1.0) -> dict:
    """Return one independent, initially relaxed, axisymmetric case."""
    if math.isnan(de) or de < 0:
        raise ValueError("De must be non-negative or infinity")
    if not math.isfinite(ec) or ec < 0 or not math.isfinite(oh) or oh < 0:
        raise ValueError("Ec and Oh must be finite and non-negative")
    label = "inf" if math.isinf(de) else f"{de:.8g}"
    newtonian = ec == 0 or de == 0
    return {
        "schema": "pyoomph-case-v1",
        "case_id": f"oh{oh:g}-ec{ec:g}-de{label}-t200",
        "geometry": {"kind": "axisymmetric", "coordinates": ["r", "z"],
                     "frame": "laboratory"},
        "nondimensionalization": {
            "length_scale": "h0", "time_scale": "sqrt(rho*h0^3/sigma)",
            "velocity_scale": "sqrt(sigma/(rho*h0))",
            "Ec": "G_dim*h0/sigma", "De": "lambda_dim/sqrt(rho*h0^3/sigma)",
            "Oh": "mu_s_dim/sqrt(2*rho*sigma*h0)",
        },
        "physics": {
            "Oh": oh, "elastic_modulus": ec,
            "relaxation_time": "inf" if math.isinf(de) else de,
            "constitutive_model": "Newtonian" if newtonian else "Oldroyd-B",
            "stabilization": "supg", "supg_factor": 1.0,
            "passive_conformation": False,
        },
        "mesh": {"length": 250.0, "r0": 1.0, "resolution": 0.06,
                 "far_resolution": 0.25},
        "tolerances": {
            "temporal_error": 1e-3, "max_volume_error": 5e-3,
            "max_tip_pin_residual": 1e-6, "min_radius_increase": 1e-4,
            "min_polymer_stress": 1e-10,
            "max_positive_sponge_power": 1e-12,
        },
        "output": {"cadence": 0.05, "field_stride": 20,
                   "write_states": False, "checkpoint_every": 200},
        "solver": {"max_step": 0.01, "max_newton": 20},
        "arrest": {"opening_ustar": 0.02, "arrest_ustar": 1e-3,
                   "consecutive_samples": 11, "trend_tolerance": 1e-4,
                   "radius_span_tolerance": 1e-3},
        "completion": {"kind": "time", "value": 200.0, "allow_recoil": True},
    }


def prepare(destination: Path) -> list[Path]:
    """Write a new case directory exclusively; existing attempts are untouched."""
    destination.mkdir(parents=True, exist_ok=False)
    paths = []
    for de in DE_VALUES:
        case = make_case(de)
        path = destination / (case["case_id"] + ".json")
        path.write_text(json.dumps(case, indent=2, allow_nan=False) + "\n")
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in prepare(args.output):
        print(path)


if __name__ == "__main__":
    main()
