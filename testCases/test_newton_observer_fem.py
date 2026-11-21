"""Integration checks for the opt-in Newton observer."""

from __future__ import annotations

import copy
import json

import numpy as np

from testCases.test_viscoelastic_case_adapter import _case, _reference, adapter


def test_observer_records_actual_newton_callbacks(tmp_path):
    from taylor_culick_fem import viscoelastic_driver as driver

    case = _case({"kind": "time", "value": 0.04, "allow_recoil": True})
    case["mesh"].update(length=3.0, resolution=0.5, far_resolution=0.5)
    case["output"].update(cadence=0.02, checkpoint_every=1)
    case["physics"]["far_field"] = {"mode": "membrane", "extension": 20.0, "node_spacing": 0.25}
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(case))
    result = driver.run_case(adapter.case_to_driver_args(input_path, tmp_path / "run", newton_observer=True))
    assert result["status"] == "passed"
    root = tmp_path / "run" / "axi_viscoelastic_Oh_1_G_1_lambda_inf"
    records = [json.loads(line) for line in (root / "newton-observer.jsonl").read_text().splitlines()]
    assert records
    assert any(record["iteration_count"] > 0 for record in records)
    assert any(record["native_residuals"] for record in records)
    assert any(record["temporal_acceptance"] == "accepted" for record in records)
    assert all(record.get("sampled_residual") is None for record in records)


def test_observer_preserves_a_forced_newton_retry(tmp_path, monkeypatch):
    """A one-time abort must still reach the native adaptive retry path."""
    from taylor_culick_fem import viscoelastic_driver as driver
    from taylor_culick_fem.axisymmetric_viscoelastic import AxisymmetricViscoelasticProblem

    original = AxisymmetricViscoelasticProblem.actions_before_newton_convergence_check

    def abort_once(self):
        original(self)
        if not getattr(self, "_test_abort_sent", False) and float(self.time_pt().dt(0)) > 0:
            self._test_abort_sent = True
            self._request_newton_abort("test-only forced Newton retry")

    monkeypatch.setattr(AxisymmetricViscoelasticProblem, "actions_before_newton_convergence_check", abort_once)
    case = _case({"kind": "time", "value": 0.04, "allow_recoil": True})
    case["mesh"].update(length=3.0, resolution=0.5, far_resolution=0.5)
    case["output"].update(cadence=0.02, checkpoint_every=1)
    case["solver"]["initial_guess"] = "extrapolate"
    case["physics"]["far_field"] = {
        "mode": "membrane", "extension": 20.0, "node_spacing": 0.25,
        "interpolation": "linear", "history": "keep",
    }
    case_path = tmp_path / "input.json"
    case_path.write_text(json.dumps(case))
    roots = []
    for name, enabled in (("off", False), ("on", True)):
        result = driver.run_case(adapter.case_to_driver_args(case_path, tmp_path / name, newton_observer=enabled))
        assert result["status"] == "passed"
        roots.append(tmp_path / name / "axi_viscoelastic_Oh_1_G_1_lambda_inf")
    for root in roots:
        log = (root / "run_viscoelastic_axi" / "_pyoomph_logfile.txt").read_text()
        assert "TIMESTEP REJECTED" in log
    for relative in ("run_viscoelastic_axi/control.txt", "run_viscoelastic_axi/rim-height.txt",
                     "run_viscoelastic_axi/far-field.txt", "checkpoints/ckpt_00002.dump"):
        assert (roots[0] / relative).read_bytes() == (roots[1] / relative).read_bytes()
    records = [json.loads(line) for line in (roots[1] / "newton-observer.jsonl").read_text().splitlines()]
    assert any(row["temporal_acceptance"] == "rejected" for row in records)
    assert records[1]["dt"] == records[0]["dt"] / 2


def test_deterministic_retry_remesh_and_restart(tmp_path, monkeypatch):
    """Compare repeated original execution and observation through real solver events."""
    from taylor_culick_fem import viscoelastic_driver as driver
    from taylor_culick_fem.axisymmetric_viscoelastic import AxisymmetricViscoelasticProblem

    base = AxisymmetricViscoelasticProblem
    before_check = base.actions_before_newton_convergence_check
    native_run = base.run
    after_remesh = base.actions_after_remeshing
    after_step = base.actions_after_newton_step
    observed_remeshes = []

    def abort_once(self):
        before_check(self)
        if (getattr(self, "_test_updated", False) and not getattr(self, "_test_abort_sent", False)
                and float(self.time_pt().dt(0)) > 0):
            self._test_abort_sent = True
            self._request_newton_abort("test-only forced Newton retry")

    def mark_update(self):
        after_step(self)
        self._test_updated = True

    def count_remesh(self):
        after_remesh(self)
        self._test_remesh_count = getattr(self, "_test_remesh_count", 0) + 1
        observed_remeshes.append(self._test_remesh_count)

    def run_then_remesh(self, *args, **kwargs):
        result = native_run(self, *args, **kwargs)
        if not getattr(self, "_test_remeshed", False):
            self._test_remeshed = True
            before = getattr(self, "_test_remesh_count", 0)
            self.force_remesh()
            assert self._test_remesh_count > before
        return result

    monkeypatch.setattr(base, "actions_before_newton_convergence_check", abort_once)
    monkeypatch.setattr(base, "actions_after_newton_step", mark_update)
    monkeypatch.setattr(base, "actions_after_remeshing", count_remesh)
    monkeypatch.setattr(base, "run", run_then_remesh)
    case = _case({"kind": "time", "value": 0.04, "allow_recoil": True})
    case["mesh"].update(length=3.0, resolution=0.5, far_resolution=0.5)
    case["output"].update(cadence=0.02, checkpoint_every=1)
    case["solver"].update(initial_guess="extrapolate", interval_start="carry",
                          temporal_error_fields={"X": 1, "U": 1})
    case["physics"]["far_field"] = {"mode": "membrane", "extension": 20.0,
                                   "node_spacing": 0.25, "interpolation": "linear", "history": "keep"}
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(case))
    roots = []
    for name, enabled in (("off1", False), ("off2", False), ("on", True)):
        run_dir = tmp_path / name
        parent = run_dir / "parent" / "axi_viscoelastic_Oh_1_G_1_lambda_inf"
        result = driver.run_case(adapter.case_to_driver_args(input_path, run_dir / "parent", newton_observer=enabled))
        assert result["status"] == "passed"
        output = parent / "run_viscoelastic_axi"
        rows = np.loadtxt(output / "control.txt", ndmin=2)
        continuation = copy.deepcopy(case)
        continuation["completion"]["value"] = 0.08
        continuation["restart"] = {
            **{key: _reference(path) for key, path in {
                "checkpoint": parent / "checkpoints" / "ckpt_00002.dump",
                "input_case": input_path, "metadata": parent / "case.json",
                "control": output / "control.txt", "rim_height": output / "rim-height.txt",
                "pvd": output / "liquid.pvd",
            }.items()},
            "time": float(rows[-1, 0]), "control_index": 2, "field_index": 2,
        }
        next_path = tmp_path / (name + "-next.json")
        next_path.write_text(json.dumps(continuation))
        result = driver.run_case(adapter.case_to_driver_args(next_path, run_dir / "next", newton_observer=enabled))
        assert result["status"] == "passed"
        roots.append(run_dir)
    # Restoring a checkpoint can invoke additional remeshing callbacks.
    assert len(observed_remeshes) >= 6
    for leg in ("parent", "next"):
        suffix = f"{leg}/axi_viscoelastic_Oh_1_G_1_lambda_inf/"
        for relative in ("run_viscoelastic_axi/control.txt", "run_viscoelastic_axi/rim-height.txt",
                         "run_viscoelastic_axi/far-field.txt"):
            files = [root / suffix / relative for root in roots]
            assert all(path.exists() for path in files)
            assert files[0].read_bytes() == files[1].read_bytes()
            assert files[0].read_bytes() == files[2].read_bytes()
        fields = sorted((roots[0] / suffix / "run_viscoelastic_axi/liquid").glob("*.vtu"))
        assert fields
        for field in fields:
            others = [root / suffix / "run_viscoelastic_axi/liquid" / field.name for root in roots[1:]]
            assert all(other.read_bytes() == field.read_bytes() for other in others)
        companions = sorted((roots[0] / suffix / "checkpoints").glob("*.farfield.npz"))
        assert companions
        for companion in companions:
            others = [root / suffix / "checkpoints" / companion.name for root in roots[1:]]
            with np.load(companion) as reference:
                for other in others:
                    with np.load(other) as candidate:
                        assert reference.files == candidate.files
                        for key in reference.files:
                            np.testing.assert_array_equal(reference[key], candidate[key])
        for root in roots:
            log = (root / suffix / "run_viscoelastic_axi/_pyoomph_logfile.txt").read_text()
            assert "TIMESTEP REJECTED" in log
        records = [json.loads(line) for line in (roots[2] / suffix / "newton-observer.jsonl").read_text().splitlines()]
        retry = next(i for i, record in enumerate(records) if record["temporal_acceptance"] == "rejected")
        assert records[retry + 1]["dt"] == records[retry]["dt"] / 2
