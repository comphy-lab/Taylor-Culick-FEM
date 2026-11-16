"""Contracts for a provenance-bound fresh axisymmetric case."""
import copy
import importlib.util
import json
import os
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
    with pytest.raises(SystemExit, match="rest edge or a membrane exterior"):
        runner.main([str(case_path), "--out", str(out), "--check"])
    assert not out.exists()
    case_path.write_text(json.dumps(case))
    out.mkdir()
    (out / "existing").write_text("untouched")
    with pytest.raises(SystemExit, match="new or empty"):
        runner.main([str(case_path), "--out", str(out), "--check"])
    assert (out / "existing").read_text() == "untouched"


def test_endpoint_landing_uses_the_effective_step_bound(case):
    case["completion"]["value"] = 1
    assert runner.endpoint_reached(1.000000095, case)
    assert not runner.endpoint_reached(1.00000012, case)
    assert not runner.endpoint_reached(0.99999999, case)
    case["solver"]["max_step"] = 0.05
    assert runner.endpoint_reached(1.0000004, case)
    assert not runner.endpoint_reached(1.0000006, case)


def test_segment_end_requires_a_checkpoint_and_never_exceeds_horizon(case):
    assert runner.segment_target(case, None) == 200
    assert runner.segment_target(case, 10) == 10
    for invalid in (0, -1, 201, float("nan"), 10.5):
        with pytest.raises(ValueError):
            runner.segment_target(case, invalid)
    case["output"]["checkpoint_every"] = 0
    with pytest.raises(ValueError, match="requires scheduled checkpoints"):
        runner.segment_target(case, 10)


MEMBRANE = {"mode": "membrane", "inner_fraction": 0.5, "extension": 4000.0, "node_spacing": 0.5}


@pytest.mark.parametrize("material", ["Oldroyd-B", "Newtonian"])
def test_membrane_exterior_is_a_valid_case_for_both_branches(case, material):
    case["physics"]["far_field"] = dict(MEMBRANE)
    if material == "Newtonian":
        case["physics"].update(constitutive_model="Newtonian", elastic_modulus=0.0,
                               relaxation_time=0.0, Oh=8.071067811865476)
    assert runner._load_helper("cases").case_problems(case) == []


@pytest.mark.parametrize("change,message", [
    ({"inner_fraction": 1.0}, "strictly inside"), ({"extension": 0.0}, "must be positive"),
    ({"node_spacing": float("nan")}, "finite numbers"), ({"extra": 1}, "exactly mode"),
])
def test_membrane_exterior_settings_are_checked_not_invented(case, change, message):
    case["physics"]["far_field"] = {**MEMBRANE, **change}
    assert any(message in problem for problem in runner._load_helper("cases").case_problems(case))


def test_membrane_restart_evidence_binds_exterior_state_and_history(tmp_path, case):
    case["physics"]["far_field"] = dict(MEMBRANE)
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(case))
    summary, state = write_segment_output(tmp_path / "out", case_path, 10)
    root = state.parent.parent
    sidecar = state.with_suffix(".farfield.npz")
    sidecar.write_bytes(b"exterior state")
    (root / "run_viscoelastic_axi/far-field.txt").write_text("exterior history")
    summary["latest_checkpoint"]["far_field_state"] = {
        "path": str(sidecar.relative_to(root)), "sha256": runner.sha256(sidecar)}
    (root / "case.json").write_text(json.dumps(summary))
    evidence = runner.endpoint_checkpoint(root, case, runner.sha256(case_path), 10)
    assert evidence["far_field_state"]["sha256"] == runner.sha256(sidecar)
    assert evidence["far_field_history"]["path"] == str((root / "run_viscoelastic_axi/far-field.txt").resolve())
    sidecar.write_bytes(b"changed exterior")
    with pytest.raises(ValueError, match="membrane state does not match"):
        runner.endpoint_checkpoint(root, case, runner.sha256(case_path), 10)
    del summary["latest_checkpoint"]["far_field_state"]
    (root / "case.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="no far-field state"):
        runner.endpoint_checkpoint(root, case, runner.sha256(case_path), 10)


@pytest.fixture
def completed_parent(tmp_path, case):
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(case))
    runtime = tmp_path / "parent" / "runtime"
    root = runtime / "axi_case"
    (root / "checkpoints").mkdir(parents=True)
    output = root / "run_viscoelastic_axi"
    output.mkdir()
    (root / "input-case.json").write_bytes(case_path.read_bytes())
    state = root / "checkpoints" / "ckpt_00200.dump"
    state.write_bytes(b"parent checkpoint")
    for name in ("control.txt", "rim-height.txt", "liquid.pvd"):
        (output / name).write_text("parent history")
    manifest = {"schema": "pyoomph-runner-manifest-v1", "case_id": case["case_id"],
                "case_sha256": runner.sha256(case_path), "segment_end": 10,
                "component_commit": "a" * 40, "pyoomph": {"commit": "b" * 40},
                "runner": {"sha256": runner.sha256(Path(runner.__file__))}, "linear_solver": "test",
                "argv": ["runner"], "case_fields_used": ["schema"]}
    (runtime / "run-manifest.json").write_text(json.dumps(manifest))
    metadata = {"status": "passed", "completion_reason": "fixed_horizon", "t_reached": 10,
                "far_boundary_remained_sampled_inflow": True,
                "latest_checkpoint": {"time": 10, "control_index": 200, "field_index": 10,
                                      "boundary_gate_passed": True,
                                      "checkpoint": {"path": "checkpoints/ckpt_00200.dump",
                                                     "sha256": runner.sha256(state)}}}
    (root / "case.json").write_text(json.dumps(metadata))
    (runtime / "progress.jsonl").write_text(
        json.dumps({"event": "started", "case_id": case["case_id"]}) + "\n"
        + json.dumps({"event": "segment_completed", "status": "passed", "physical_time": 10}) + "\n")
    return case_path, runtime, root, state


def write_segment_output(out, case_path, end):
    root = out / "axi_case"
    (root / "checkpoints").mkdir(parents=True)
    output = root / "run_viscoelastic_axi"
    output.mkdir()
    (root / "input-case.json").write_bytes(case_path.read_bytes())
    state = root / "checkpoints" / f"ckpt_{round(end / .05):05d}.dump"
    state.write_bytes(b"state")
    for name in ("control.txt", "rim-height.txt", "liquid.pvd"):
        (output / name).write_text("history")
    summary = {"status": "passed", "completion_reason": "fixed_horizon", "t_reached": end,
               "far_boundary_remained_sampled_inflow": True,
               "latest_checkpoint": {"time": end, "control_index": round(end / .05),
                                     "field_index": round(end / .05) // 20,
                                     "boundary_gate_passed": True,
                                     "checkpoint": {"path": str(state.relative_to(root)),
                                                    "sha256": runner.sha256(state)}}}
    (root / "case.json").write_text(json.dumps(summary))
    return summary, state


@pytest.fixture(params=["Oldroyd-B", "Newtonian"])
def old_fresh_parent(request, completed_parent, case):
    case_path, runtime, root, state = completed_parent
    if request.param == "Newtonian":
        case["physics"].update(constitutive_model="Newtonian", elastic_modulus=0.0,
                               relaxation_time=0.0, Oh=71.71067811865476)
        case_path.write_text(json.dumps(case))
    parent = copy.deepcopy(case)
    parent["completion"]["value"] = 10
    raw = json.dumps(parent).encode()
    (root / "input-case.json").write_bytes(raw)
    manifest_path = runtime / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("segment_end")
    manifest["case_sha256"] = runner.sha256(root / "input-case.json")
    manifest["component_commit"] = runner.MIGRATION_PARENT_COMMIT
    manifest["runner"]["sha256"] = runner.MIGRATION_PARENT_RUNNER
    manifest["pyoomph"]["commit"] = runner.MIGRATION_PYOOMPH_COMMIT
    manifest_path.write_text(json.dumps(manifest))
    (runtime / "progress.jsonl").write_text(
        json.dumps({"event": "started", "case_id": case["case_id"]}) + "\n"
        + json.dumps({"event": "completed", "status": "passed", "physical_time": 10}) + "\n")
    return case_path, runtime, root, state


def test_horizon_migration_requires_explicit_pin_and_preserves_parent(tmp_path, old_fresh_parent):
    case_path, runtime, root, state = old_fresh_parent
    case = json.loads(case_path.read_bytes())
    old_bytes = (root / "input-case.json").read_bytes()
    with pytest.raises(ValueError, match="same case"):
        runner.restart_lineage(state, tmp_path / "out", case, runner.sha256(case_path), 20)
    line = runner.restart_lineage(state, tmp_path / "out", case, runner.sha256(case_path), 20,
                                  migrate_horizon=True)
    assert line["migration"]["parent_horizon"] == 10
    assert line["migration"]["parent_case_sha256"] == runner.sha256(root / "input-case.json")
    assert line["evidence"]["checkpoint"]["sha256"] == runner.sha256(state)
    assert (root / "input-case.json").read_bytes() == old_bytes


@pytest.mark.parametrize("mutation", ["parent_commit", "parent_runner", "parent_pyoomph", "parent_hash",
                                      "parent_segment", "physics", "case_id", "completion_flag",
                                      "smaller_horizon", "failed_parent", "failed_checkpoint", "missing_completion_event"])
def test_horizon_migration_rejects_unproven_or_changed_parent(tmp_path, old_fresh_parent, mutation):
    case_path, runtime, root, state = old_fresh_parent
    case = json.loads(case_path.read_bytes())
    manifest_path = runtime / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "parent_commit": manifest["component_commit"] = "c" * 40
    if mutation == "parent_runner": manifest["runner"]["sha256"] = "c" * 64
    if mutation == "parent_pyoomph": manifest["pyoomph"]["commit"] = "c" * 40
    if mutation == "parent_hash": manifest["case_sha256"] = "c" * 64
    if mutation == "parent_segment": manifest["segment_end"] = 10
    manifest_path.write_text(json.dumps(manifest))
    if mutation == "physics": case["physics"]["Oh"] = 2
    if mutation == "case_id": case["case_id"] = "other"
    if mutation == "completion_flag": case["completion"]["allow_recoil"] = False
    if mutation == "smaller_horizon": case["completion"]["value"] = 5
    if mutation == "missing_completion_event":
        (runtime / "progress.jsonl").write_text(json.dumps({"event": "started", "case_id": case["case_id"]}) + "\n")
    if mutation in ("failed_parent", "failed_checkpoint"):
        meta_path = root / "case.json"
        meta = json.loads(meta_path.read_text())
        if mutation == "failed_parent": meta["status"] = "failed"
        else: meta["latest_checkpoint"]["boundary_gate_passed"] = False
        meta_path.write_text(json.dumps(meta))
    expected = {
        "parent_commit": "pinned completed fresh-run parent", "parent_runner": "pinned completed fresh-run parent",
        "parent_pyoomph": "pinned completed fresh-run parent", "parent_hash": "pinned completed fresh-run parent",
        "parent_segment": "pinned completed fresh-run parent", "physics": "increase only the original case horizon",
        "case_id": "does not bind the same case", "completion_flag": "increase only the original case horizon",
        "smaller_horizon": "increase only the original case horizon", "failed_parent": "did not complete successfully",
        "failed_checkpoint": "no passing endpoint checkpoint", "missing_completion_event": "no matching completion event",
    }
    with pytest.raises(ValueError, match=expected[mutation]):
        runner.restart_lineage(state, tmp_path / "out", case, runner.sha256(case_path), 20,
                               migrate_horizon=True)


@pytest.mark.parametrize("source_changed", [False, True])
@pytest.mark.parametrize("check_only", [False, True])
def test_migrated_runner_binds_old_case_and_new_case_in_manifest(tmp_path, old_fresh_parent, monkeypatch, source_changed, check_only):
    case_path, runtime, root, state = old_fresh_parent
    out = tmp_path / "migrated"
    pyoomph = types.ModuleType("pyoomph")
    solvers = types.ModuleType("pyoomph.solvers")
    generic = types.ModuleType("pyoomph.solvers.generic")
    generic.get_default_linear_solver = lambda: "test"
    generic.set_default_linear_solver = lambda name: None
    adapter = types.ModuleType("run_viscoelastic_case")
    adapter.case_to_driver_args = lambda path, dest: types.SimpleNamespace(serialized_case=path.read_bytes())
    driver = types.ModuleType("run_viscoelastic_axi")
    driver._case_name = lambda *args: "axi_case"
    driver._read_restart = lambda args: None
    def solve(args):
        saved = json.loads((out / "run-manifest.json").read_text())
        assert saved["case_sha256"] == runner.sha256(case_path)
        assert saved["restart_from"]["migration"]["parent_case_sha256"] == runner.sha256(root / "input-case.json")
        assert saved["restart_from"]["migration"]["parent_horizon"] == 10
        assert saved["restart_from"]["parent_manifest"]["sha256"] == runner.sha256(runtime / "run-manifest.json")
        return write_segment_output(out, case_path, 20)[0]
    driver.run_case = solve
    for name, module in (("pyoomph", pyoomph), ("pyoomph.solvers", solvers),
                         ("pyoomph.solvers.generic", generic), ("run_viscoelastic_case", adapter),
                         ("run_viscoelastic_axi", driver)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(runner.provenance, "component_commit", lambda root: "a" * 40)
    monkeypatch.setattr(runner.provenance, "pyoomph_revision", lambda: {"commit": runner.MIGRATION_PYOOMPH_COMMIT})
    monkeypatch.setattr(runner, "migration_source_digest", lambda root: "c" * 64 if source_changed else runner.MIGRATION_SOURCE_DIGEST)
    argv = [str(case_path), "--out", str(out), "--segment-end", "20",
            "--restart-from", str(state), "--migrate-horizon"]
    if check_only:
        argv.append("--check")
    if source_changed:
        with pytest.raises(SystemExit, match="source differs"):
            runner.main(argv)
        assert not out.exists()
        return
    assert runner.main(argv) == 0
    if check_only:
        assert not out.exists()
        return
    assert [json.loads(line)["event"] for line in (out / "progress.jsonl").read_text().splitlines()] == ["started", "segment_completed"]


def test_real_migration_source_digest_pins_baseline_and_detects_input_changes(tmp_path):
    # The pin is the 57a7297 source tree; later solver changes must not
    # silently inherit horizon migration, so the working tree may differ.
    import subprocess
    import tarfile
    baseline_root = tmp_path / "baseline"
    archive = tmp_path / "baseline.tar"
    subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", "-o", str(archive), "57a7297"],
                   check=True)
    with tarfile.open(archive) as stream:
        stream.extractall(baseline_root, filter="data")
    assert runner.migration_source_digest(baseline_root) == runner.MIGRATION_SOURCE_DIGEST
    root = tmp_path / "source"
    for name in ("campaign/run.py", "simulationCases/run_axisymmetric_case.py", "simulationCases/a.py",
                 "src/taylor_culick_fem/module.py", "validation/driver.py", ".python-version", "pyproject.toml", "uv.lock"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    baseline = runner.migration_source_digest(root)
    (root / "simulationCases/run_axisymmetric_case.py").write_text("new runner deliberately excluded")
    assert runner.migration_source_digest(root) == baseline
    for name in ("campaign/run.py", "simulationCases/a.py", "src/taylor_culick_fem/module.py", "validation/driver.py",
                 ".python-version", "pyproject.toml", "uv.lock"):
        path = root / name
        original = path.read_bytes()
        path.write_bytes(original + b"changed")
        assert runner.migration_source_digest(root) != baseline, name
        path.write_bytes(original)
    extra = root / "validation/extra.py"
    extra.write_text("unexpected")
    assert runner.migration_source_digest(root) != baseline
    extra.unlink()
    (root / "uv.lock").unlink()
    with pytest.raises(FileNotFoundError):
        runner.migration_source_digest(root)


def test_horizon_migration_rechecks_source_after_imports(tmp_path, old_fresh_parent, monkeypatch):
    case_path, _, _, state = old_fresh_parent
    out = tmp_path / "migrated"
    pyoomph = types.ModuleType("pyoomph")
    solvers = types.ModuleType("pyoomph.solvers")
    generic = types.ModuleType("pyoomph.solvers.generic")
    generic.get_default_linear_solver = lambda: "test"
    generic.set_default_linear_solver = lambda name: None
    adapter = types.ModuleType("run_viscoelastic_case")
    adapter.case_to_driver_args = lambda path, dest: types.SimpleNamespace(serialized_case=path.read_bytes())
    driver = types.ModuleType("run_viscoelastic_axi")
    driver._read_restart = lambda args: pytest.fail("changed source must be refused before restart")
    driver.run_case = lambda args: pytest.fail("changed source must be refused before the solve")
    for name, module in (("pyoomph", pyoomph), ("pyoomph.solvers", solvers),
                         ("pyoomph.solvers.generic", generic), ("run_viscoelastic_case", adapter),
                         ("run_viscoelastic_axi", driver)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(runner.provenance, "component_commit", lambda root: "a" * 40)
    monkeypatch.setattr(runner.provenance, "pyoomph_revision", lambda: {"commit": runner.MIGRATION_PYOOMPH_COMMIT})
    digests = iter([runner.MIGRATION_SOURCE_DIGEST, "c" * 64])
    monkeypatch.setattr(runner, "migration_source_digest", lambda root: next(digests))
    with pytest.raises(SystemExit, match="source differs"):
        runner.main([str(case_path), "--out", str(out), "--segment-end", "20",
                     "--restart-from", str(state), "--migrate-horizon"])
    assert not out.exists()


@pytest.mark.parametrize("mutation", ["missing_manifest", "foreign_case", "changed_state", "changed_input", "failed_parent", "failed_boundary", "outflow", "wrong_index", "missing_event", "nested_output"])
def test_restart_refuses_unproven_parent_before_output(tmp_path, completed_parent, mutation):
    case_path, runtime, root, state = completed_parent
    out = tmp_path / "next"
    if mutation == "missing_manifest":
        (runtime / "run-manifest.json").unlink()
    elif mutation == "foreign_case":
        case_path.write_text(case_path.read_text().replace('"Oh": 1', '"Oh": 2'))
    elif mutation == "changed_state":
        state.write_bytes(b"changed")
    elif mutation == "changed_input":
        (root / "input-case.json").write_text("{}")
    elif mutation in ("failed_parent", "failed_boundary", "outflow", "wrong_index"):
        metadata_path = root / "case.json"
        metadata = json.loads(metadata_path.read_text())
        if mutation == "failed_parent":
            metadata["status"] = "failed"
        elif mutation == "failed_boundary":
            metadata["latest_checkpoint"]["boundary_gate_passed"] = False
        elif mutation == "outflow":
            metadata["far_boundary_remained_sampled_inflow"] = False
        else:
            metadata["latest_checkpoint"]["control_index"] = 199
        metadata_path.write_text(json.dumps(metadata))
    elif mutation == "missing_event":
        (runtime / "progress.jsonl").write_text('{"event":"started"}\n')
    else:
        out = runtime / "nested"
    with pytest.raises(SystemExit, match="refused"):
        runner.main([str(case_path), "--out", str(out), "--segment-end", "20", "--restart-from", str(state), "--check"])
    assert not out.exists()


@pytest.mark.parametrize("end,event", [(20, "segment_completed"), (200, "completed")])
def test_completed_parent_restart_preserves_case_and_records_lineage(tmp_path, case, completed_parent, monkeypatch, end, event):
    case_path, runtime, root, state = completed_parent
    out = tmp_path / "next"
    pyoomph = types.ModuleType("pyoomph")
    solvers = types.ModuleType("pyoomph.solvers")
    generic = types.ModuleType("pyoomph.solvers.generic")
    generic.get_default_linear_solver = lambda: "test"
    generic.set_default_linear_solver = lambda name: None
    adapter = types.ModuleType("run_viscoelastic_case")
    adapter.case_to_driver_args = lambda path, dest: types.SimpleNamespace(serialized_case=path.read_bytes())
    driver = types.ModuleType("run_viscoelastic_axi")
    driver._case_name = lambda *args: "axi_case"
    def validate(args):
        assert args.serialized_case == case_path.read_bytes()
        if not out.exists():
            assert args.t_end == end
            assert args.restart_evidence["checkpoint"]["sha256"] == runner.sha256(state)
        else:
            assert args.t_end == case["completion"]["value"]
            assert Path(args.restart_evidence["checkpoint"]["path"]).is_file()
    driver._read_restart = validate
    def solve(args):
        manifest = json.loads((out / "run-manifest.json").read_text())
        assert manifest["case_sha256"] == runner.sha256(case_path)
        assert manifest["restart_from"]["parent_manifest"]["sha256"] == runner.sha256(runtime / "run-manifest.json")
        assert manifest["restart_from"]["path"] == str(state)
        if end < case["completion"]["value"]:
            return write_segment_output(out, case_path, end)[0]
        return {"status": "passed", "completion_reason": "fixed_horizon", "t_reached": end,
                "far_boundary_remained_sampled_inflow": True}
    driver.run_case = solve
    for name, module in (("pyoomph", pyoomph), ("pyoomph.solvers", solvers), ("pyoomph.solvers.generic", generic), ("run_viscoelastic_case", adapter), ("run_viscoelastic_axi", driver)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(runner.provenance, "component_commit", lambda root: "a" * 40)
    monkeypatch.setattr(runner.provenance, "pyoomph_revision", lambda: {"version": "test", "url": "test", "commit": "b" * 40})
    assert runner.main([str(case_path), "--out", str(out), "--segment-end", str(end), "--restart-from", str(state)]) == 0
    events = [json.loads(line)["event"] for line in (out / "progress.jsonl").read_text().splitlines()]
    assert events == ["started", event]


@pytest.mark.parametrize("failure", ["driver_lineage", "different_commit", "different_solver"])
def test_restart_preflight_failure_creates_no_output(tmp_path, completed_parent, monkeypatch, failure):
    case_path, runtime, root, state = completed_parent
    out = tmp_path / "next"
    pyoomph = types.ModuleType("pyoomph")
    solvers = types.ModuleType("pyoomph.solvers")
    generic = types.ModuleType("pyoomph.solvers.generic")
    generic.get_default_linear_solver = lambda: "other" if failure == "different_solver" else "test"
    generic.set_default_linear_solver = lambda name: None
    adapter = types.ModuleType("run_viscoelastic_case")
    adapter.case_to_driver_args = lambda path, dest: types.SimpleNamespace(serialized_case=path.read_bytes())
    driver = types.ModuleType("run_viscoelastic_axi")
    def check(args):
        if failure == "driver_lineage":
            raise ValueError("history is incomplete")
    driver._read_restart = check
    driver.run_case = lambda args: pytest.fail("solver must not start")
    for name, module in (("pyoomph", pyoomph), ("pyoomph.solvers", solvers), ("pyoomph.solvers.generic", generic), ("run_viscoelastic_case", adapter), ("run_viscoelastic_axi", driver)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(runner.provenance, "component_commit", lambda root: ("c" if failure == "different_commit" else "a") * 40)
    monkeypatch.setattr(runner.provenance, "pyoomph_revision", lambda: {"version": "test", "url": "test", "commit": "b" * 40})
    with pytest.raises(SystemExit, match="refused"):
        runner.main([str(case_path), "--out", str(out), "--segment-end", "20", "--restart-from", str(state)])
    assert not out.exists()


@pytest.mark.parametrize("mutation", ["valid", "missing", "empty", "changed", "boundary", "outflow", "index", "history", "driver"])
def test_intermediate_completion_requires_restart_ready_checkpoint(tmp_path, case, monkeypatch, mutation):
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(case))
    out = tmp_path / "out"
    pyoomph = types.ModuleType("pyoomph")
    solvers = types.ModuleType("pyoomph.solvers")
    generic = types.ModuleType("pyoomph.solvers.generic")
    generic.get_default_linear_solver = lambda: "test"
    generic.set_default_linear_solver = lambda name: None
    adapter = types.ModuleType("run_viscoelastic_case")
    adapter.case_to_driver_args = lambda path, dest: types.SimpleNamespace(serialized_case=path.read_bytes())
    driver = types.ModuleType("run_viscoelastic_axi")
    driver._case_name = lambda *args: "axi_case"
    def validate(args):
        assert args.t_end == case["completion"]["value"]
        if mutation == "driver":
            raise ValueError("checkpoint history is invalid")
    driver._read_restart = validate
    def solve(args):
        summary, state = write_segment_output(out, case_path, 10)
        if mutation == "missing":
            state.unlink()
        elif mutation == "empty":
            state.write_bytes(b"")
        elif mutation == "changed":
            state.write_bytes(b"changed")
        elif mutation in ("boundary", "outflow", "index"):
            if mutation == "boundary":
                summary["latest_checkpoint"]["boundary_gate_passed"] = False
            elif mutation == "outflow":
                summary["far_boundary_remained_sampled_inflow"] = False
            else:
                summary["latest_checkpoint"]["control_index"] = 199
            (state.parent.parent / "case.json").write_text(json.dumps(summary))
        elif mutation == "history":
            (state.parent.parent / "run_viscoelastic_axi/control.txt").unlink()
        return summary
    driver.run_case = solve
    for name, module in (("pyoomph", pyoomph), ("pyoomph.solvers", solvers), ("pyoomph.solvers.generic", generic), ("run_viscoelastic_case", adapter), ("run_viscoelastic_axi", driver)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(runner.provenance, "component_commit", lambda root: "a" * 40)
    monkeypatch.setattr(runner.provenance, "pyoomph_revision", lambda: {"version": "test", "url": "test", "commit": "b" * 40})
    expected = 0 if mutation == "valid" else 1
    assert runner.main([str(case_path), "--out", str(out), "--segment-end", "10"]) == expected
    events = [json.loads(line) for line in (out / "progress.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == ["started", "segment_completed" if expected == 0 else "failed"]
    if expected:
        assert ("far boundary" if mutation == "outflow" else "not restart-ready") in events[-1]["error"]


@pytest.mark.parametrize("status,target,time,inflow,material,expected", [
    ("passed", 200, 200, True, "Oldroyd-B", 0),
    ("passed", 200, 199, True, "Oldroyd-B", 1),
    ("failed", 200, 200, True, "Oldroyd-B", 1),
    ("raised", 200, 0, True, "Oldroyd-B", None),
    ("passed", 1, 1.000000095, True, "Oldroyd-B", 0),
    ("passed", 1, 0.99999, True, "Oldroyd-B", 1),
    ("passed", 200, 200, False, "Oldroyd-B", 1),
    ("passed", 200, 200, None, "Oldroyd-B", 1),
    ("passed", 200, 200, False, "Newtonian", 0),
])
def test_manifest_precedes_output_and_only_endpoint_passes(tmp_path, case, monkeypatch, status, target, time, inflow, material, expected):
    case["completion"]["value"] = target
    if material == "Newtonian":
        case["physics"].update(constitutive_model="Newtonian", elastic_modulus=0, relaxation_time=0)
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
        return {"status": status, "completion_reason": "fixed_horizon", "t_reached": time,
                "far_boundary_remained_sampled_inflow": inflow}

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
    driver.run_case = lambda args: {"status": "passed", "completion_reason": "fixed_horizon", "t_reached": 200,
                                    "far_boundary_remained_sampled_inflow": True}
    for name, module in (("pyoomph", pyoomph), ("pyoomph.solvers", solvers), ("pyoomph.solvers.generic", generic), ("run_viscoelastic_case", adapter), ("run_viscoelastic_axi", driver)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(runner.provenance, "component_commit", lambda root: "a" * 40)
    monkeypatch.setattr(runner.provenance, "pyoomph_revision", lambda: {"version": "test", "url": "test", "commit": "b" * 40})
    assert runner.main([str(case_path), "--out", str(out), "--linear-solver", "pardiso"]) == 0
    assert chosen == ["pardiso"]
    assert json.loads((out / "run-manifest.json").read_text())["linear_solver"] == "pardiso"


@pytest.mark.skipif(os.environ.get("TAYLOR_CULICK_REAL_FEM") != "1",
                    reason="real FEM smoke requires an admitted compute allocation")
@pytest.mark.parametrize("far_field", ["rest", "membrane"])
@pytest.mark.parametrize("material", ["Oldroyd-B", "Newtonian"])
def test_real_segment_restart_matches_continuous(tmp_path, case, material, far_field):
    """Check the actual runner, binary checkpoint, both constitutive paths and exteriors."""
    import numpy as np
    import xml.etree.ElementTree as ET

    case["mesh"].update(length=3.0, resolution=0.5, far_resolution=0.5)
    case["physics"]["relaxation_time"] = 10.0
    if material == "Newtonian":
        case["physics"].update(constitutive_model="Newtonian", elastic_modulus=0.0,
                               relaxation_time=0.0, Oh=8.071067811865476)
    if far_field == "membrane":
        case["physics"]["far_field"] = {**MEMBRANE, "extension": 20.0}
    case["output"].update(cadence=0.02, checkpoint_every=2, field_stride=1)
    case["completion"]["value"] = 0.08
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(case))
    reference, parent, child = (tmp_path / name for name in ("reference", "parent", "child"))
    assert runner.main([str(case_path), "--out", str(reference)]) == 0
    assert runner.main([str(case_path), "--out", str(parent), "--segment-end", "0.04"]) == 0
    states = list(parent.glob("*/checkpoints/ckpt_00002.dump"))
    assert len(states) == 1
    assert runner.main([str(case_path), "--out", str(child), "--restart-from", str(states[0])]) == 0
    def events(path):
        return [json.loads(line)["event"] for line in (path / "progress.jsonl").read_text().splitlines()]
    # A membrane exterior also reports one far_field_sample per output interval.
    samples = {"rest": 0, "membrane": 2}[far_field]
    for path, last in ((parent, "segment_completed"), (child, "completed")):
        observed = events(path)
        assert [e for e in observed if e != "far_field_sample"] == ["started", last]
        assert observed.count("far_field_sample") == samples
    ref_root, child_root = (next(path.glob("*/run_viscoelastic_axi")) for path in (reference, child))
    files = ("control.txt", "rim-height.txt") + (("far-field.txt",) if far_field == "membrane" else ())
    for filename in files:
        np.testing.assert_allclose(np.loadtxt(child_root / filename), np.loadtxt(ref_root / filename),
                                   rtol=1e-8, atol=1e-9)
    def field_count(root):
        history = root / "liquid-history.pvd"
        return len(ET.parse(history if history.exists() else root / "liquid.pvd").findall("./Collection/DataSet"))
    assert field_count(child_root) == field_count(ref_root) == 4


@pytest.mark.skipif(os.environ.get("TAYLOR_CULICK_REAL_FEM") != "1" or
                    not os.environ.get("TAYLOR_CULICK_MIGRATION_OLD_RUNNER"),
                    reason="migration smoke requires an admitted allocation and pinned old source")
@pytest.mark.parametrize("material", ["Oldroyd-B", "Newtonian"])
def test_real_horizon_migration_matches_continuous(tmp_path, case, material):
    """Exercise actual old-runner final checkpoints and new-runner migration."""
    import numpy as np
    import subprocess
    import xml.etree.ElementTree as ET

    old_runner = Path(os.environ["TAYLOR_CULICK_MIGRATION_OLD_RUNNER"]).resolve(strict=True)
    assert runner.sha256(old_runner) == runner.MIGRATION_PARENT_RUNNER
    case["mesh"].update(length=3.0, resolution=0.5, far_resolution=0.5)
    case["physics"]["relaxation_time"] = 10.0
    if material == "Newtonian":
        case["physics"].update(constitutive_model="Newtonian", elastic_modulus=0.0,
                               relaxation_time=0.0, Oh=8.071067811865476)
    case["output"].update(cadence=0.02, checkpoint_every=2, field_stride=1)
    case["completion"]["value"] = 0.08
    new_case = tmp_path / "new-case.json"
    new_case.write_text(json.dumps(case))
    old_case = copy.deepcopy(case)
    old_case["completion"]["value"] = 0.04
    old_case_path = tmp_path / "old-case.json"
    old_case_path.write_text(json.dumps(old_case))
    reference, parent, child = (tmp_path / name for name in ("reference", "parent", "child"))
    current_runner = ROOT / "simulationCases/run_axisymmetric_case.py"
    def invoke(script, source_case, output, *args):
        subprocess.run([sys.executable, str(script), str(source_case), "--out", str(output), *args],
                       check=True, timeout=600)
    invoke(current_runner, new_case, reference)
    invoke(old_runner, old_case_path, parent)
    states = list(parent.glob("*/checkpoints/ckpt_00002.dump"))
    assert len(states) == 1
    if runner.migration_source_digest(ROOT) != runner.MIGRATION_SOURCE_DIGEST:
        # Later solver changes must not inherit migration: the current runner
        # refuses, before creating output, and the comparison is not possible.
        refused = subprocess.run([sys.executable, str(current_runner), str(new_case), "--out", str(child),
                                  "--restart-from", str(states[0]), "--migrate-horizon"],
                                 capture_output=True, text=True, timeout=600)
        assert refused.returncode != 0
        assert "migration continuation source differs from reviewed baseline" in refused.stderr
        assert not child.exists()
        pytest.skip("current sources differ from the pinned migration baseline")
    invoke(current_runner, new_case, child, "--restart-from", str(states[0]), "--migrate-horizon")
    manifest = json.loads((child / "run-manifest.json").read_text())
    assert manifest["restart_from"]["migration"]["parent_case_sha256"] == runner.sha256(old_case_path)
    assert manifest["restart_from"]["migration"]["parent_horizon"] == 0.04
    assert [json.loads(line)["event"] for line in (parent / "progress.jsonl").read_text().splitlines()] == ["started", "completed"]
    assert [json.loads(line)["event"] for line in (child / "progress.jsonl").read_text().splitlines()] == ["started", "completed"]
    ref_root, child_root = (next(path.glob("*/run_viscoelastic_axi")) for path in (reference, child))
    for filename in ("control.txt", "rim-height.txt"):
        np.testing.assert_allclose(np.loadtxt(child_root / filename), np.loadtxt(ref_root / filename),
                                   rtol=1e-8, atol=1e-9)
    def field_count(path):
        history = path / "liquid-history.pvd"
        return len(ET.parse(history if history.exists() else path / "liquid.pvd").findall("./Collection/DataSet"))
    assert field_count(child_root) == field_count(ref_root) == 4
