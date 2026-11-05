"""Contracts for fixed-Ec case preparation; no campaign is launched."""

import json
import math
import sys
from pathlib import Path

import pytest

from taylor_culick_fem.cases import VISCOELASTIC, case_problems

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulationCases"))
from prepare_de_sweep import (
    DE_VALUES,
    FINITE_DE_ERROR_FIELDS,
    FINITE_DE_START_FRACTION,
    make_case,
    prepare,
)


def test_all_cases_reach_driver_without_arrest_stopping(tmp_path):
    from taylor_culick_fem.case_adapter import case_to_driver_args

    paths = prepare(tmp_path / "cases")
    assert len(paths) == 15
    assert len({path.name for path in paths}) == 15
    for path, de in zip(paths, DE_VALUES, strict=True):
        case = json.loads(path.read_text())
        assert case_problems(case, VISCOELASTIC) == []
        args = case_to_driver_args(path, tmp_path / "not-launched")
        assert args.oh == args.elastic_modulus == 1
        assert args.relaxation_time == de
        assert args.t_end == 200
        assert not args.stop_when_arrested
        assert not args.stop_on_height_plateau
        assert args.allow_recoil
        assert not args.passive_conformation
        assert args.length == 250
        finite = 0 < de < math.inf
        # Only an evolved finite-De case opts into the finite-relaxation numerics.
        assert args.temporal_error_fields == (FINITE_DE_ERROR_FIELDS if finite else None)
        assert args.relaxation_start_fraction == (FINITE_DE_START_FRACTION if finite else None)
    assert not (tmp_path / "not-launched").exists()
    with pytest.raises(FileExistsError):
        prepare(tmp_path / "cases")


@pytest.mark.parametrize("ec,de", [(0, 0), (0, 0.001), (0, math.inf), (1, 0)])
def test_zero_endpoint_selects_newtonian_without_passive_transport(ec, de):
    case = make_case(de, ec=ec)
    assert case["physics"]["constitutive_model"] == "Newtonian"
    assert not case["physics"]["passive_conformation"]
    assert "temporal_error_fields" not in case["solver"]
    assert "relaxation_start_fraction" not in case["solver"]
    json.dumps(case, allow_nan=False)


@pytest.mark.parametrize("de", [-1, -math.inf, math.nan])
def test_invalid_de_rejected(de):
    with pytest.raises(ValueError):
        make_case(de)
