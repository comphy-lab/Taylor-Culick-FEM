"""Software tests for serialized viscoelastic production cases."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).parents[1] / "campaign" / "run_viscoelastic_case.py"
_SPEC = importlib.util.spec_from_file_location("run_viscoelastic_case", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
adapter = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(adapter)


def _case(completion: dict) -> dict:
    return {
        "schema": "pyoomph-case-v1",
        "case_id": "oh1-g1-laminf",
        "geometry": {
            "kind": "axisymmetric",
            "coordinates": ["r", "z"],
            "frame": "laboratory",
        },
        "nondimensionalization": {"length": "h0", "time": "capillary"},
        "physics": {
            "Oh": 1.0,
            "elastic_modulus": 1.0,
            "relaxation_time": "inf",
            "stabilization": "supg",
        },
        "tolerances": {
            "temporal_error": 1e-3,
            "max_volume_error": 5e-3,
            "max_tip_pin_residual": 1e-6,
            "min_radius_increase": 1e-4,
            "min_polymer_stress": 1e-10,
        },
        "mesh": {"length": 75.0, "r0": 1.0, "resolution": 0.06, "far_resolution": 0.25},
        "output": {"cadence": 0.05, "write_states": False, "checkpoint_every": 20},
        "solver": {"max_step": 0.01, "max_newton": 20},
        "arrest": {
            "opening_ustar": 0.02,
            "arrest_ustar": 1e-3,
            "consecutive_samples": 11,
            "trend_tolerance": 1e-4,
            "radius_span_tolerance": 1e-3,
        },
        "completion": completion,
    }


def _write_case(tmp_path: Path, case: dict) -> Path:
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case))
    return path


def test_event_case_maps_to_first_turning_point_driver(tmp_path: Path) -> None:
    path = _write_case(
        tmp_path,
        _case({"kind": "event", "value": "first_post_startup_turning_point", "horizon": 25.0}),
    )

    args = adapter.case_to_driver_args(path, tmp_path / "attempt")

    assert args.length == 75.0
    assert args.stop_when_arrested
    assert not args.allow_recoil
    assert args.t_end == 25.0


def test_time_case_explicitly_allows_recoil(tmp_path: Path) -> None:
    case = _case({"kind": "time", "value": 200.0, "allow_recoil": True})
    case["mesh"]["length"] = 100.0
    path = _write_case(tmp_path, case)

    args = adapter.case_to_driver_args(path, tmp_path / "attempt")

    assert args.length == 100.0
    assert not args.stop_when_arrested
    assert args.allow_recoil
    assert args.t_end == 200.0


def test_adapter_rejects_nonlaboratory_axisymmetric_case(tmp_path: Path) -> None:
    case = _case({"kind": "time", "value": 1.0})
    case["geometry"]["frame"] = "co-moving"
    path = _write_case(tmp_path, case)

    with pytest.raises(ValueError, match="axisymmetric laboratory"):
        adapter.case_to_driver_args(path, tmp_path / "attempt")


@pytest.mark.parametrize("payload", [[], {"schema": "pyoomph-case-v1", "geometry": None}])
def test_adapter_rejects_non_object_case_sections(tmp_path: Path, payload) -> None:
    path = _write_case(tmp_path, payload)
    with pytest.raises(TypeError, match="JSON object|geometry must be an object"):
        adapter.case_to_driver_args(path, tmp_path / "attempt")


def test_runner_progress_is_append_only_jsonl(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    adapter.append_progress("started", case="case.json")
    adapter.append_progress("completed", physical_time=2.0, primary_observable=0.1)

    rows = [json.loads(line) for line in (tmp_path / "runtime" / "progress.jsonl").read_text().splitlines()]
    assert [row["event"] for row in rows] == ["started", "completed"]
    assert rows[1]["physical_time"] == 2.0
