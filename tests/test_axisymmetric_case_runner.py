"""Contracts for a provenance-bound fresh axisymmetric case."""
import copy
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("axisymmetric_case_runner", ROOT / "simulationCases/run_axisymmetric_case.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def case():
    return {
        "schema": "pyoomph-case-v1", "case_id": "test",
        "geometry": {"kind": "axisymmetric", "coordinates": ["r", "z"], "frame": "laboratory"},
        "nondimensionalization": {"length_scale": "h0", "time_scale": "sqrt(rho*h0^3/sigma)", "velocity_scale": "sqrt(sigma/(rho*h0))", "Ec": "G_dim*h0/sigma", "De": "lambda_dim/sqrt(rho*h0^3/sigma)", "Oh": "mu_s_dim/sqrt(2*rho*sigma*h0)"},
        "physics": {"Oh": 1, "elastic_modulus": 1, "relaxation_time": 100, "constitutive_model": "Oldroyd-B", "stabilization": "supg", "supg_factor": 1, "passive_conformation": False, "far_field": {"mode": "rest"}},
        "mesh": {"length": 250, "r0": 1, "resolution": .06, "far_resolution": .25},
        "tolerances": {"temporal_error": .001, "max_volume_error": .005, "max_tip_pin_residual": 1e-6, "min_radius_increase": .0001, "min_polymer_stress": 1e-10},
        "output": {"cadence": .05, "field_stride": 20, "write_states": False, "checkpoint_every": 200},
        "solver": {"max_step": .01, "max_newton": 20},
        "arrest": {"opening_ustar": .02, "arrest_ustar": .001, "consecutive_samples": 11, "trend_tolerance": .0001, "radius_span_tolerance": .001},
        "completion": {"kind": "time", "value": 200, "allow_recoil": True},
    }


def test_check_is_solver_free_and_does_not_create_output(tmp_path, case, monkeypatch):
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(case))
    out = tmp_path / "out"
    monkeypatch.delitem(sys.modules, "pyoomph", raising=False)
    assert runner.main([str(case_path), "--out", str(out), "--check"]) == 0
    assert not out.exists() and "pyoomph" not in sys.modules


def test_invalid_or_occupied_run_is_refused(tmp_path, case):
    case_path = tmp_path / "case.json"
    out = tmp_path / "out"
    bad = copy.deepcopy(case)
    bad["physics"]["far_field"] = {"mode": "membrane", "extension": 4000}
    case_path.write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="fully resolved"):
        runner.main([str(case_path), "--out", str(out), "--check"])
    assert not out.exists()
    case_path.write_text(json.dumps(case))
    out.mkdir()
    (out / "existing").write_text("untouched")
    with pytest.raises(SystemExit, match="new or empty"):
        runner.main([str(case_path), "--out", str(out), "--check"])
    assert (out / "existing").read_text() == "untouched"


@pytest.mark.parametrize("status,time,expected", [("passed", 200, 0), ("passed", 199, 1), ("failed", 200, 1), ("raised", 0, None)])
def test_manifest_precedes_output_and_only_endpoint_passes(tmp_path, case, monkeypatch, status, time, expected):
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(case))
    out = tmp_path / "out"
    pyoomph = types.ModuleType("pyoomph")
    pyoomph.__file__ = "fake"
    solvers = types.ModuleType("pyoomph.solvers")
    generic = types.ModuleType("pyoomph.solvers.generic")
    generic.set_default_linear_solver = lambda name: pytest.fail("implicit solver override")
    generic.get_default_linear_solver = lambda: "auto-selected-test-solver"
    adapter = types.ModuleType("run_viscoelastic_case")
    adapter.case_to_driver_args = lambda path, dest: types.SimpleNamespace(serialized_case=path.read_bytes())
    driver = types.ModuleType("run_viscoelastic_axi")

    def solve(args):
        assert sys.argv[1:] == []
        manifest = json.loads((out / "run-manifest.json").read_text())
        assert manifest["case_sha256"] == runner.sha256(case_path)
        assert manifest["linear_solver"] == "auto-selected-test-solver"
        assert (out / "progress.jsonl").is_file()
        if status == "raised":
            raise RuntimeError("solver failure")
        return {"status": status, "completion_reason": "fixed_horizon", "t_reached": time}

    driver.run_case = solve
    for name, module in (("pyoomph", pyoomph), ("pyoomph.solvers", solvers), ("pyoomph.solvers.generic", generic), ("run_viscoelastic_case", adapter), ("run_viscoelastic_axi", driver)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(runner.provenance, "component_commit", lambda root: "a" * 40)
    monkeypatch.setattr(runner.provenance, "pyoomph_revision", lambda: {"version": "test", "url": "test", "commit": "b" * 40})
    original_argv = ["runner", str(case_path), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", original_argv)
    if status == "raised":
        with pytest.raises(RuntimeError, match="solver failure"):
            runner.main(original_argv[1:])
    else:
        assert runner.main(original_argv[1:]) == expected
    assert sys.argv is original_argv
    events = [json.loads(line)["event"] for line in (out / "progress.jsonl").read_text().splitlines()]
    assert events == ["started", "completed" if expected == 0 else "failed"]


def test_explicit_linear_solver_is_applied_and_recorded(tmp_path, case, monkeypatch):
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(case))
    out = tmp_path / "out"
    pyoomph = types.ModuleType("pyoomph")
    solvers = types.ModuleType("pyoomph.solvers")
    generic = types.ModuleType("pyoomph.solvers.generic")
    chosen = []
    generic.set_default_linear_solver = chosen.append
    generic.get_default_linear_solver = lambda: chosen[-1]
    adapter = types.ModuleType("run_viscoelastic_case")
    adapter.case_to_driver_args = lambda path, dest: types.SimpleNamespace(serialized_case=path.read_bytes())
    driver = types.ModuleType("run_viscoelastic_axi")
    driver.run_case = lambda args: {"status": "passed", "completion_reason": "fixed_horizon", "t_reached": 200}
    for name, module in (("pyoomph", pyoomph), ("pyoomph.solvers", solvers), ("pyoomph.solvers.generic", generic), ("run_viscoelastic_case", adapter), ("run_viscoelastic_axi", driver)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(runner.provenance, "component_commit", lambda root: "a" * 40)
    monkeypatch.setattr(runner.provenance, "pyoomph_revision", lambda: {"version": "test", "url": "test", "commit": "b" * 40})
    assert runner.main([str(case_path), "--out", str(out), "--linear-solver", "pardiso"]) == 0
    assert chosen == ["pardiso"]
    assert json.loads((out / "run-manifest.json").read_text())["linear_solver"] == "pardiso"
