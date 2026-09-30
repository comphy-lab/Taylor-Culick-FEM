"""Software tests for serialized viscoelastic production cases."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
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
    assert args.passive_conformation is False


def test_time_case_explicitly_allows_recoil(tmp_path: Path) -> None:
    case = _case({"kind": "time", "value": 200.0, "allow_recoil": True})
    case["mesh"]["length"] = 100.0
    path = _write_case(tmp_path, case)

    args = adapter.case_to_driver_args(path, tmp_path / "attempt")

    assert args.length == 100.0
    assert not args.stop_when_arrested
    assert args.allow_recoil
    assert args.t_end == 200.0


def test_height_plateau_case_maps_explicit_certificate_thresholds(tmp_path: Path) -> None:
    completion = {
        "kind": "height_plateau",
        "horizon": 1000.0,
        "hold": 10.0,
        "minimum_window": 10.0,
        "window_time_fraction": 0.1,
        "slope_window_fraction": 0.005,
        "instantaneous_slope_fraction": 0.02,
        "residual_rms_fraction": 0.005,
        "adjacent_mean_fraction": 0.005,
        "residual_envelope_growth_fraction": 0.05,
    }
    case = _case(completion)
    case["output"]["field_stride"] = 10
    case["output"]["checkpoint_every"] = 20

    args = adapter.case_to_driver_args(
        _write_case(tmp_path, case), tmp_path / "attempt"
    )

    assert args.stop_on_height_plateau
    assert not args.stop_when_arrested
    assert args.t_end == 1000.0
    assert args.height_hold == 10.0
    assert args.height_oscillation_period is None
    assert args.field_stride == 10


def test_supg_factor_is_explicit_and_backwards_compatible(tmp_path: Path) -> None:
    baseline = _case({"kind": "time", "value": 1.0, "allow_recoil": True})
    baseline_args = adapter.case_to_driver_args(
        _write_case(tmp_path, baseline), tmp_path / "baseline"
    )
    assert baseline_args.supg_factor == 1.0

    varied = copy.deepcopy(baseline)
    varied["physics"]["supg_factor"] = 0.5
    varied_path = tmp_path / "varied.json"
    varied_path.write_text(json.dumps(varied))
    varied_args = adapter.case_to_driver_args(varied_path, tmp_path / "varied")
    assert varied_args.supg_factor == 0.5


def test_passive_conformation_maps_and_allows_zero_modulus(tmp_path: Path) -> None:
    case = _case({"kind": "time", "value": 1.0, "allow_recoil": True})
    case["physics"]["elastic_modulus"] = 0.0
    case["physics"]["passive_conformation"] = True

    args = adapter.case_to_driver_args(_write_case(tmp_path, case), tmp_path / "attempt")

    assert args.elastic_modulus == 0.0
    assert args.passive_conformation is True


def test_missing_passive_conformation_rejects_zero_modulus(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case = _case({"kind": "time", "value": 1.0, "allow_recoil": True})
    case["physics"]["elastic_modulus"] = 0.0

    with pytest.raises(SystemExit):
        adapter.case_to_driver_args(_write_case(tmp_path, case), tmp_path / "attempt")
    assert "--elastic-modulus=0 requires --passive-conformation" in capsys.readouterr().err


def test_adapter_rejects_non_boolean_passive_conformation(tmp_path: Path) -> None:
    case = _case({"kind": "time", "value": 1.0, "allow_recoil": True})
    case["physics"]["passive_conformation"] = 1

    with pytest.raises(TypeError, match="passive_conformation must be a boolean"):
        adapter.case_to_driver_args(_write_case(tmp_path, case), tmp_path / "attempt")


def test_passive_zero_stress_diagnostics_do_not_fail_driver(tmp_path, monkeypatch) -> None:
    import run_viscoelastic_axi as driver

    class History:
        def __init__(self, path, header):
            self.path = path
            path.write_text("#" + " ".join(header) + "\n")

        def add_row(self, *row):
            with self.path.open("a") as stream:
                stream.write(" ".join(map(str, row)) + "\n")

    class Problem:
        def __init__(self):
            self.time = 0.0

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def set_output_directory(self, path):
            self.output_dir = Path(path)

        def initialise(self):
            self.output_dir.mkdir()

        def volume(self):
            return 1.0

        def window_position(self):
            return 1.0 + self.time

        def hole_radius(self):
            return self.window_position()

        def rim_height(self):
            return 1.0

        def retraction_speed(self):
            return 0.1

        def expected_volume(self, volume0, radius0):
            del radius0
            return volume0

        def create_text_file_output(self, filename, header):
            return History(self.output_dir / filename, header)

        def run(self, target, **kwargs):
            del kwargs
            self.time = target

        def output(self, **kwargs):
            del kwargs

        def get_current_time(self, **kwargs):
            del kwargs
            return self.time

        def get_output_directory(self):
            return str(self.output_dir)

        def polymer_diagnostic_extrema(self):
            return {
                "log_conformation_abs_max": 0.0,
                "conformation_eigenvalue_min": 1.0,
                "conformation_eigenvalue_max": 1.0,
                "polymer_stress_eigenvalue_min": 0.0,
                "polymer_stress_eigenvalue_max": 0.0,
            }

    monkeypatch.setattr(driver, "AxisymmetricViscoelasticProblem", Problem)
    monkeypatch.setattr(driver, "_enforce_launcher_affinity", dict)
    case = _case({"kind": "time", "value": 0.1, "allow_recoil": True})
    case["physics"].update(elastic_modulus=0.0, passive_conformation=True)
    args = adapter.case_to_driver_args(_write_case(tmp_path, case), tmp_path / "attempt")

    metadata = driver.run_case(args)

    assert metadata["status"] == "passed"
    assert metadata["passive_conformation"] is True
    assert metadata["polymer_diagnostics_available"] is True
    assert metadata["polymer_diagnostics_final"]["polymer_stress_eigenvalue_max"] == 0.0
    assert (
        tmp_path / "attempt" / "axi_viscoelastic_Oh_1_G_0_lambda_inf_passive_conformation"
    ).is_dir()


def test_optional_sponge_maps_to_driver_and_preserves_absent_default(tmp_path: Path) -> None:
    baseline = _case({"kind": "time", "value": 1.0, "allow_recoil": True})
    baseline_path = _write_case(tmp_path, baseline)
    baseline_args = adapter.case_to_driver_args(baseline_path, tmp_path / "baseline")
    assert baseline_args.sponge_start is None
    assert baseline_args.sponge_width is None
    assert baseline_args.sponge_gamma_max is None

    sponge_case = copy.deepcopy(baseline)
    sponge_case["physics"]["sponge"] = {
        "start": 50.0,
        "width": 25.0,
        "gamma_max": 2.0,
        "profile": "quintic",
    }
    sponge_case["mesh"]["length"] = 75.0
    sponge_path = tmp_path / "sponge.json"
    sponge_path.write_text(json.dumps(sponge_case))
    sponge_args = adapter.case_to_driver_args(sponge_path, tmp_path / "sponge")
    assert sponge_args.sponge_start == 50.0
    assert sponge_args.sponge_width == 25.0
    assert sponge_args.sponge_gamma_max == 2.0
    assert sponge_args.sponge_profile == "quintic"


@pytest.mark.parametrize(
    "sponge,match",
    [
        ({"start": 50, "width": 25, "gamma_max": 2}, "requires exactly"),
        ("not-an-object", "must be an object"),
    ],
)
def test_adapter_rejects_invalid_sponge(tmp_path: Path, sponge, match) -> None:
    case = _case({"kind": "time", "value": 1.0, "allow_recoil": True})
    case["physics"]["sponge"] = sponge
    path = _write_case(tmp_path, case)
    with pytest.raises((TypeError, ValueError), match=match):
        adapter.case_to_driver_args(path, tmp_path / "attempt")


def test_adapter_rejects_sponge_extending_beyond_window(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case = _case({"kind": "time", "value": 1.0, "allow_recoil": True})
    case["physics"]["sponge"] = {
        "start": 50,
        "width": 30,
        "gamma_max": 2,
        "profile": "quintic",
    }
    with pytest.raises(SystemExit):
        adapter.case_to_driver_args(_write_case(tmp_path, case), tmp_path / "attempt")
    assert "must not exceed window length" in capsys.readouterr().err


def test_far_field_maps_to_driver_and_preserves_absent_default(tmp_path: Path) -> None:
    baseline = _case({"kind": "time", "value": 1.0, "allow_recoil": True})
    args = adapter.case_to_driver_args(_write_case(tmp_path, baseline), tmp_path / "a")
    assert args.far_field == "rest"
    assert args.far_field_extension is None

    membrane = copy.deepcopy(baseline)
    membrane["physics"]["far_field"] = {"mode": "membrane", "extension": 800.0}
    path = tmp_path / "membrane.json"
    path.write_text(json.dumps(membrane))
    args = adapter.case_to_driver_args(path, tmp_path / "b")
    assert args.far_field == "membrane"
    assert args.far_field_extension == 800.0
    assert args.far_field_inner_fraction is None


@pytest.mark.parametrize(
    "far_field,match",
    [
        ({"mode": "rest", "ramp_time": 1.0}, "takes no other keys"),
        ({"mode": "membrane", "extension": 10.0, "damping": 1.0}, "accepts only"),
        ({"mode": "membrane"}, "requires extension"),
        ({"mode": "sponge"}, "unsupported"),
        ("membrane", "must be an object"),
    ],
)
def test_adapter_rejects_invalid_far_field(tmp_path: Path, far_field, match) -> None:
    case = _case({"kind": "time", "value": 1.0, "allow_recoil": True})
    case["physics"]["far_field"] = far_field
    with pytest.raises((TypeError, ValueError), match=match):
        adapter.case_to_driver_args(_write_case(tmp_path, case), tmp_path / "attempt")


def test_small_fem_membrane_far_field_conserves_volume_and_restarts(tmp_path):
    """A real coarse solve coupled to the far-field membrane, and its restart."""
    import run_viscoelastic_axi as driver

    case = _case({"kind": "time", "value": 0.04, "allow_recoil": True})
    case["mesh"].update(length=3.0, resolution=0.5, far_resolution=0.5)
    case["output"].update(cadence=0.02, checkpoint_every=1)
    case["physics"]["far_field"] = {"mode": "membrane", "extension": 20.0, "node_spacing": 0.25}
    scratch = tmp_path / "project-scratch"
    scratch.mkdir()
    bundle = tmp_path / "attempt-bundle"
    parent_input = scratch / "parent-input.json"
    parent_input.write_text(json.dumps(case))
    parent_metadata = driver.run_case(adapter.case_to_driver_args(parent_input, bundle / "parent"))
    parent = bundle / "parent" / "axi_viscoelastic_Oh_1_G_1_lambda_inf"
    parent_output = parent / "run_viscoelastic_axi"
    far = np.loadtxt(parent_output / "far-field.txt", ndmin=2)
    assert far.shape == (2, len(driver.FAR_FIELD_HEADER))
    assert parent_metadata["far_field"]["mode"] == "membrane"
    # The drive reaches the membrane, and its edge state is a thin film.
    assert np.any(far[:, 1] != 0.0)
    assert np.all(np.abs(far[:, 3] - 1.0) < 0.1)
    assert np.all(np.abs(far[:, 4:7].sum(axis=1)) < 1e-12)
    assert (parent / "checkpoints" / "ckpt_00002.farfield.npz").is_file()
    assert parent_metadata["worst_volume_rel_error"] < 1e-6
    assert parent_metadata["far_field_outer_velocity_max"] < driver.FAR_FIELD_OUTER_TOLERANCE

    rows = np.loadtxt(parent_output / "control.txt", ndmin=2)
    continued = copy.deepcopy(case)
    continued["completion"]["value"] = 0.08
    continued["restart"] = {
        **{
            key: _reference(path)
            for key, path in {
                "checkpoint": parent / "checkpoints" / "ckpt_00002.dump",
                "input_case": parent_input,
                "metadata": parent / "case.json",
                "control": parent_output / "control.txt",
                "rim_height": parent_output / "rim-height.txt",
                "pvd": parent_output / "liquid.pvd",
            }.items()
        },
        "time": float(rows[-1, 0]),
        "control_index": 2,
        "field_index": 2,
    }
    next_input = scratch / "next-input.json"
    next_input.write_text(json.dumps(continued))
    driver.run_case(adapter.case_to_driver_args(next_input, bundle / "next"))
    del continued["restart"]
    reference_input = scratch / "reference-input.json"
    reference_input.write_text(json.dumps(continued))
    driver.run_case(adapter.case_to_driver_args(reference_input, bundle / "reference"))
    suffix = Path("axi_viscoelastic_Oh_1_G_1_lambda_inf/run_viscoelastic_axi")
    for filename in ("control.txt", "far-field.txt"):
        resumed = np.loadtxt(bundle / "next" / suffix / filename)
        reference = np.loadtxt(bundle / "reference" / suffix / filename)
        np.testing.assert_allclose(resumed, reference, rtol=1e-9, atol=1e-10)


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

    rows = [
        json.loads(line)
        for line in (tmp_path / "runtime" / "progress.jsonl").read_text().splitlines()
    ]
    assert [row["event"] for row in rows] == ["started", "completed"]
    assert rows[1]["physical_time"] == 2.0


def _reference(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def restart_case(tmp_path):
    case = _case({"kind": "time", "value": 0.1, "allow_recoil": True})
    parent = tmp_path / "attempt-bundle" / "parent"
    parent.mkdir(parents=True)
    scratch = tmp_path / "project-scratch"
    scratch.mkdir()
    input_case = scratch / "input.json"
    input_case.write_text(json.dumps(case))
    root = parent / "axi_viscoelastic_Oh_1_G_1_lambda_inf"
    checkpoints = root / "checkpoints"
    output = root / "run_viscoelastic_axi"
    checkpoints.mkdir(parents=True)
    output.mkdir()
    (root / "input-case.json").write_bytes(input_case.read_bytes())
    checkpoint = checkpoints / "ckpt_00002.dump"
    checkpoint.write_bytes(b"mock state")
    metadata = root / "case.json"
    metadata.write_text(
        json.dumps(
            {
                "Oh": 1.0,
                "elastic_modulus": 1.0,
                "relaxation_time": "inf",
                "length": 75.0,
                "r0": 1.0,
                "resolution": 0.06,
                "far_resolution": 0.25,
                "output_step": 0.05,
                "max_step": 0.01,
                "stabilization_option": "supg",
                "conformation_supg_factor": 1.0,
                "temporal_error": 0.001,
                "largest_positive_sponge_power": 2.5e-13,
            }
        )
    )
    control = output / "control.txt"
    control.write_text(
        "#t R U ustar volume expected error pin\n"
        "0.05 1.05 0.1 0.07071067811865475 12 12 0 0\n"
        "0.1 1.1 0.1 0.07071067811865475 13 13 0 0\n"
    )
    rim_height = output / "rim-height.txt"
    rim_height.write_text("#t H\n0.05 1.0\n0.1 1.0\n")
    for index in range(2):
        (output / f"liquid_{index:06d}.vtu").write_text("mock field")
    pvd = output / "liquid.pvd"
    pvd.write_text(
        "<VTKFile><Collection>"
        '<DataSet timestep="0.05" file="liquid_000000.vtu"/>'
        '<DataSet timestep="0.1" file="liquid_000001.vtu"/>'
        "</Collection></VTKFile>"
    )
    case["completion"]["value"] = 0.2
    case["restart"] = {
        **{
            key: _reference(path)
            for key, path in {
                "checkpoint": checkpoint,
                "input_case": input_case,
                "metadata": metadata,
                "control": control,
                "rim_height": rim_height,
                "pvd": pvd,
            }.items()
        },
        "time": 0.1,
        "control_index": 2,
        "field_index": 2,
    }
    return case


def test_restart_selector_maps_to_driver_and_validates_prefix(tmp_path, restart_case):
    from run_viscoelastic_axi import _read_restart

    path = _write_case(tmp_path, restart_case)
    args = adapter.case_to_driver_args(path, tmp_path / "next")
    assert args.restart_case == path
    restart = _read_restart(args)
    assert restart["control_index"] == 2
    assert restart["paths"]["input_case"].name == "input-case.json"
    assert restart["paths"]["input_case"].is_relative_to(tmp_path / "attempt-bundle")
    np.testing.assert_allclose(restart["rows"][:, 0], [0.05, 0.1])


def test_restart_snapshot_removes_dependency_on_external_source(tmp_path, restart_case):
    from run_viscoelastic_axi import _read_restart

    source = Path(restart_case["restart"]["input_case"]["path"])
    source.rename(source.with_suffix(".unavailable"))
    args = adapter.case_to_driver_args(_write_case(tmp_path, restart_case), tmp_path / "next")
    restart = _read_restart(args)
    assert restart["paths"]["input_case"].name == "input-case.json"


def test_restart_rejects_external_parent_case_without_snapshot(tmp_path, restart_case):
    from run_viscoelastic_axi import _read_restart

    root = Path(restart_case["restart"]["metadata"]["path"]).parent
    snapshot = root / "input-case.json"
    snapshot.rename(snapshot.with_suffix(".unavailable"))
    args = adapter.case_to_driver_args(_write_case(tmp_path, restart_case), tmp_path / "next")
    with pytest.raises(ValueError, match="verified parent input-case snapshot"):
        _read_restart(args)


def test_restart_identity_rejects_a_changed_conformation_mode(tmp_path, restart_case):
    from run_viscoelastic_axi import _read_restart

    case = copy.deepcopy(restart_case)
    case["physics"]["passive_conformation"] = True
    args = adapter.case_to_driver_args(_write_case(tmp_path, case), tmp_path / "next")

    with pytest.raises(ValueError, match="restart case mismatch: physics"):
        _read_restart(args)


def test_restart_metadata_identity_binds_conformation_mode(tmp_path, restart_case):
    from run_viscoelastic_axi import _read_restart

    case = copy.deepcopy(restart_case)
    # Explicit false is the same backwards-compatible active mode as the
    # parent snapshot, so the metadata identity check is reached below.
    case["physics"]["passive_conformation"] = False
    metadata_path = Path(case["restart"]["metadata"]["path"])
    metadata = json.loads(metadata_path.read_text())
    metadata["passive_conformation"] = True
    metadata_path.write_text(json.dumps(metadata))
    case["restart"]["metadata"] = _reference(metadata_path)
    args = adapter.case_to_driver_args(_write_case(tmp_path, case), tmp_path / "next")

    with pytest.raises(ValueError, match="restart metadata mismatch: passive_conformation"):
        _read_restart(args)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("hash", "hash mismatch"),
        ("time", "selected control row"),
        ("case", "case mismatch"),
        ("metadata", "metadata mismatch"),
        ("index", "positive integer"),
        ("duplicate", "strictly increasing"),
        ("pvd", "PVD and control"),
        ("endpoint", "before t_end"),
    ],
)
def test_restart_rejects_unproven_lineage(tmp_path, restart_case, mutation, match):
    from run_viscoelastic_axi import _read_restart

    case = copy.deepcopy(restart_case)
    if mutation == "hash":
        case["restart"]["checkpoint"]["sha256"] = "0" * 64
    elif mutation == "time":
        case["restart"]["time"] = 0.11
    elif mutation == "case":
        case["mesh"]["length"] = 50
    elif mutation == "metadata":
        ref = case["restart"]["metadata"]
        path = Path(ref["path"])
        payload = json.loads(path.read_text())
        payload["length"] = 50
        path.write_text(json.dumps(payload))
        case["restart"]["metadata"] = _reference(path)
    elif mutation == "index":
        case["restart"]["control_index"] = True
    elif mutation == "duplicate":
        path = Path(case["restart"]["control"]["path"])
        path.write_text(path.read_text().replace("0.1 1.1", "0.05 1.1"))
        case["restart"]["control"] = _reference(path)
    elif mutation == "pvd":
        path = Path(case["restart"]["pvd"]["path"])
        path.write_text(path.read_text().replace('timestep="0.1"', 'timestep="0.11"'))
        case["restart"]["pvd"] = _reference(path)
    else:
        case["completion"]["value"] = 0.1
    args = adapter.case_to_driver_args(_write_case(tmp_path, case), tmp_path / "next")
    with pytest.raises(ValueError, match=match):
        _read_restart(args)


@pytest.mark.parametrize("header_time,header_index", [(0.1, 2), (0.09, 2), (0.1, 1)])
def test_resume_driver_preserves_parent_and_output_lineage(
    tmp_path,
    restart_case,
    monkeypatch,
    header_time,
    header_index,
):
    import shutil
    import xml.etree.ElementTree as ET

    import run_viscoelastic_axi as driver

    references = restart_case["restart"]
    old_hashes = {
        key: _reference(Path(value["path"]))
        for key, value in references.items()
        if isinstance(value, dict)
    }
    outputs, targets = [], []

    class History:
        def __init__(self, path, header):
            self.path = path
            path.write_text("#" + " ".join(header) + "\n")

        def add_row(self, *row):
            with self.path.open("a") as stream:
                stream.write(" ".join(map(str, row)) + "\n")

    class Problem:
        def __init__(self):
            self.time, self._output_step = 0.0, 0

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def set_output_directory(self, path):
            self.output_dir = Path(path)

        def initialise(self):
            self.output_dir.mkdir()

        def _get_time_of_state_file(self, path):
            return header_time, header_index

        def load_state(self, path, ignore_outstep):
            assert not ignore_outstep
            self.time, self._output_step = 0.1, 2

        def get_current_time(self, **_):
            return self.time

        def get_output_directory(self):
            return str(self.output_dir)

        def window_position(self):
            return 1 + self.time

        def hole_radius(self):
            return self.window_position()

        def rim_height(self):
            return 1.0

        def retraction_speed(self):
            return 0.1

        def volume(self):
            return 13.0 if self.time <= 0.1 else self.expected_volume(self.reference, 1)

        def expected_volume(self, volume0, radius0):
            self.reference = volume0
            return volume0 + np.pi / 2 * ((self.window_position() + 75) ** 2 - (radius0 + 75) ** 2)

        def create_text_file_output(self, filename, header):
            return History(self.output_dir / filename, header)

        def run(self, target, **kwargs):
            targets.append(target)
            self.time = target
            # Set by the driver only after run; use the lineage-derived reference.
            self.reference = 13 - np.pi / 2 * ((1.1 + 75) ** 2 - 76**2)

        def output(self, **_):
            outputs.append(self._output_step)
            path = self.output_dir / f"liquid_{self._output_step:06d}.vtu"
            path.write_text("new field")
            pvd = self.output_dir / "liquid.pvd"
            root = (
                ET.parse(pvd).getroot()
                if pvd.exists()
                else ET.fromstring("<VTKFile><Collection/></VTKFile>")
            )
            ET.SubElement(
                root.find("Collection"), "DataSet", timestep=str(self.time), file=path.name
            )
            ET.ElementTree(root).write(pvd)
            self._output_step += 1

        def polymer_diagnostic_extrema(self):
            return {}

    monkeypatch.setattr(driver, "AxisymmetricViscoelasticProblem", Problem)
    monkeypatch.setattr(driver, "copy_state_file", lambda src, dst: shutil.copyfile(src, dst))
    monkeypatch.setattr(driver, "_enforce_launcher_affinity", dict)
    path = _write_case(tmp_path, restart_case)
    validated_bytes = path.read_bytes()
    args = adapter.case_to_driver_args(path, tmp_path / "attempt-bundle" / "next")
    path.write_text("{}")  # A later edit must not replace the validated input bytes.
    if (header_time, header_index) != (0.1, 2):
        with pytest.raises(ValueError, match="checkpoint header"):
            driver.run_case(args)
        assert not targets
        return
    metadata = driver.run_case(args)
    np.testing.assert_allclose(targets, [0.15, 0.2])
    assert outputs == [2, 3]
    assert metadata["outputs_written"] == 4
    assert metadata["largest_positive_sponge_power"] == pytest.approx(2.5e-13)
    root = tmp_path / "attempt-bundle" / "next" / "axi_viscoelastic_Oh_1_G_1_lambda_inf"
    assert (root / "input-case.json").read_bytes() == validated_bytes
    assert metadata["input_case"] == {
        "path": "input-case.json",
        "sha256": hashlib.sha256(validated_bytes).hexdigest(),
    }
    assert metadata["checkpoint_relocation"]["byte_identical"] is True
    output = root / "run_viscoelastic_axi"
    np.testing.assert_allclose(np.loadtxt(output / "control.txt")[:, 0], [0.05, 0.1, 0.15, 0.2])
    np.testing.assert_allclose(np.loadtxt(output / "control-segment.txt")[:, 0], [0.15, 0.2])
    times = [
        float(node.attrib["timestep"])
        for node in ET.parse(output / "liquid-history.pvd").findall("./Collection/DataSet")
    ]
    np.testing.assert_allclose(times, [0.05, 0.1, 0.15, 0.2])
    for key, expected in old_hashes.items():
        assert _reference(Path(references[key]["path"])) == expected

    # All field references survive a whole-ancestor-bundle move; the derived view
    # is deliberately not a standalone archive of the parent VTUs.
    relocated = tmp_path.parent / (tmp_path.name + "-relocated")
    bundle = tmp_path / "attempt-bundle"
    shutil.copytree(bundle, relocated)
    copied = relocated / output.relative_to(bundle)
    for node in ET.parse(copied / "liquid-history.pvd").findall("./Collection/DataSet"):
        assert not Path(node.attrib["file"]).is_absolute()
        assert (copied / node.attrib["file"]).is_file()
    copied_root = relocated / root.relative_to(bundle)
    assert (copied_root / "input-case.json").read_bytes() == validated_bytes
    lineage = json.loads((copied_root / "restart-lineage.json").read_text())
    for value in lineage["parent"].values():
        if isinstance(value, dict):
            assert not Path(value["path"]).is_absolute()
            resolved = (copied_root / value["path"]).resolve()
            assert resolved.is_relative_to(relocated)
            assert _reference(resolved)["sha256"] == value["sha256"]


def test_legacy_restart_without_rim_height_is_rejected(tmp_path, restart_case):
    from run_viscoelastic_axi import _read_restart

    case = copy.deepcopy(restart_case)
    case["restart"].pop("rim_height")
    case["restart"].pop("control_index")
    case["restart"].pop("field_index")
    case["restart"]["output_index"] = 2
    args = adapter.case_to_driver_args(_write_case(tmp_path, case), tmp_path / "next")

    with pytest.raises(ValueError, match="lacks rim-height history"):
        _read_restart(args)


def test_small_fem_checkpoint_continuation_matches_uninterrupted(tmp_path):
    """A short coarse-mesh solve checks the real state/output API, not convergence."""
    import run_viscoelastic_axi as driver

    case = _case({"kind": "time", "value": 0.04, "allow_recoil": True})
    case["mesh"].update(length=3.0, resolution=0.5, far_resolution=0.5)
    case["output"].update(cadence=0.02, checkpoint_every=1)
    scratch = tmp_path / "project-scratch"
    scratch.mkdir()
    bundle = tmp_path / "attempt-bundle"
    parent_input = scratch / "parent-input.json"
    parent_input.write_text(json.dumps(case))
    parent_metadata = driver.run_case(adapter.case_to_driver_args(parent_input, bundle / "parent"))
    parent = bundle / "parent" / "axi_viscoelastic_Oh_1_G_1_lambda_inf"
    assert (parent / "input-case.json").read_bytes() == parent_input.read_bytes()
    assert parent_metadata["input_case"]["sha256"] == _reference(parent_input)["sha256"]
    parent_output = parent / "run_viscoelastic_axi"
    rows = np.loadtxt(parent_output / "control.txt", ndmin=2)
    continued = copy.deepcopy(case)
    continued["completion"]["value"] = 0.08
    continued["restart"] = {
        **{
            key: _reference(path)
            for key, path in {
                "checkpoint": parent / "checkpoints" / "ckpt_00002.dump",
                "input_case": parent_input,
                "metadata": parent / "case.json",
                "control": parent_output / "control.txt",
                "rim_height": parent_output / "rim-height.txt",
                "pvd": parent_output / "liquid.pvd",
            }.items()
        },
        "time": float(rows[-1, 0]),
        "control_index": 2,
        "field_index": 2,
    }
    next_input = scratch / "next-input.json"
    next_input.write_text(json.dumps(continued))
    resumed_metadata = driver.run_case(adapter.case_to_driver_args(next_input, bundle / "next"))
    relocation = resumed_metadata["checkpoint_relocation"]
    assert relocation["source_sha256"] == continued["restart"]["checkpoint"]["sha256"]
    assert relocation["byte_identical"] == (
        relocation["source_sha256"] == relocation["relocated_sha256"]
    )
    del continued["restart"]
    reference_input = scratch / "reference-input.json"
    reference_input.write_text(json.dumps(continued))
    driver.run_case(adapter.case_to_driver_args(reference_input, bundle / "reference"))
    suffix = Path("axi_viscoelastic_Oh_1_G_1_lambda_inf/run_viscoelastic_axi/control.txt")
    resumed = np.loadtxt(bundle / "next" / suffix)
    reference = np.loadtxt(bundle / "reference" / suffix)
    np.testing.assert_allclose(resumed, reference, rtol=1e-9, atol=1e-10)


def test_decoupled_field_restart_matches_uninterrupted_histories(tmp_path):
    """Scalar and quadratic-height histories survive a sparse-field restart."""
    import xml.etree.ElementTree as ET

    import run_viscoelastic_axi as driver

    case = _case({"kind": "time", "value": 0.04, "allow_recoil": True})
    case["mesh"].update(length=3.0, resolution=0.5, far_resolution=0.5)
    case["output"].update(cadence=0.02, field_stride=2, checkpoint_every=2)
    scratch = tmp_path / "project-scratch"
    scratch.mkdir()
    bundle = tmp_path / "attempt-bundle"
    parent_input = scratch / "parent-input.json"
    parent_input.write_text(json.dumps(case))
    parent_metadata = driver.run_case(
        adapter.case_to_driver_args(parent_input, bundle / "parent")
    )
    parent = bundle / "parent" / "axi_viscoelastic_Oh_1_G_1_lambda_inf"
    parent_output = parent / "run_viscoelastic_axi"
    assert parent_metadata["scalar_samples_written"] == 2
    assert parent_metadata["fields_written"] == 1
    assert len(ET.parse(parent_output / "liquid.pvd").findall("./Collection/DataSet")) == 1

    rows = np.loadtxt(parent_output / "control.txt", ndmin=2)
    continued = copy.deepcopy(case)
    continued["completion"]["value"] = 0.08
    continued["restart"] = {
        **{
            key: _reference(path)
            for key, path in {
                "checkpoint": parent / "checkpoints" / "ckpt_00002.dump",
                "input_case": parent_input,
                "metadata": parent / "case.json",
                "control": parent_output / "control.txt",
                "rim_height": parent_output / "rim-height.txt",
                "pvd": parent_output / "liquid.pvd",
            }.items()
        },
        "time": float(rows[-1, 0]),
        "control_index": 2,
        "field_index": 1,
    }
    next_input = scratch / "next-input.json"
    next_input.write_text(json.dumps(continued))
    resumed_metadata = driver.run_case(
        adapter.case_to_driver_args(next_input, bundle / "next")
    )

    del continued["restart"]
    reference_input = scratch / "reference-input.json"
    reference_input.write_text(json.dumps(continued))
    reference_metadata = driver.run_case(
        adapter.case_to_driver_args(reference_input, bundle / "reference")
    )
    suffix = Path("axi_viscoelastic_Oh_1_G_1_lambda_inf/run_viscoelastic_axi")
    for filename in ("control.txt", "rim-height.txt"):
        resumed = np.loadtxt(bundle / "next" / suffix / filename)
        reference = np.loadtxt(bundle / "reference" / suffix / filename)
        np.testing.assert_allclose(resumed, reference, rtol=1e-9, atol=1e-10)
    assert resumed_metadata["scalar_samples_written"] == 4
    assert resumed_metadata["fields_written"] == 2
    assert resumed_metadata["outputs_written"] == 2
    assert reference_metadata["scalar_samples_written"] == 4
    assert reference_metadata["fields_written"] == 2
    history_times = [
        float(node.attrib["timestep"])
        for node in ET.parse(bundle / "next" / suffix / "liquid-history.pvd").findall(
            "./Collection/DataSet"
        )
    ]
    control_times = np.loadtxt(bundle / "next" / suffix / "control.txt")[:, 0]
    np.testing.assert_allclose(history_times, control_times[[1, 3]])


def test_small_fem_passive_zero_modulus_evolves_conformation(tmp_path):
    """A real coarse solve retains conformation at zero momentum feedback."""
    import run_viscoelastic_axi as driver

    case = _case({"kind": "time", "value": 0.02, "allow_recoil": True})
    case["physics"].update(elastic_modulus=0.0, passive_conformation=True)
    case["mesh"].update(length=3.0, resolution=0.5, far_resolution=0.5)
    case["output"].update(cadence=0.02, checkpoint_every=0)
    case["tolerances"]["min_polymer_stress"] = 1e-10
    case["tolerances"]["min_radius_increase"] = 1e-6
    input_path = _write_case(tmp_path, case)

    metadata = driver.run_case(
        adapter.case_to_driver_args(input_path, tmp_path / "attempt")
    )

    root = (
        tmp_path
        / "attempt"
        / "axi_viscoelastic_Oh_1_G_0_lambda_inf_passive_conformation"
    )
    assert metadata["status"] == "passed"
    assert metadata["passive_conformation"] is True
    assert metadata["polymer_diagnostics_available"] is True
    assert metadata["polymer_diagnostics_final"]["log_conformation_abs_max"] > 0
    assert metadata["polymer_diagnostics_final"]["polymer_stress_eigenvalue_min"] == 0
    assert metadata["polymer_diagnostics_final"]["polymer_stress_eigenvalue_max"] == 0
    assert (root / "run_viscoelastic_axi/liquid/liquid_000000.vtu").is_file()


def test_small_fem_membrane_restart_may_lengthen_but_not_shorten_the_extension(tmp_path):
    import run_viscoelastic_axi as driver

    case = _case({"kind": "time", "value": 0.04, "allow_recoil": True})
    case["mesh"].update(length=3.0, resolution=0.5, far_resolution=0.5)
    case["output"].update(cadence=0.02, checkpoint_every=1)
    case["physics"]["far_field"] = {"mode": "membrane", "extension": 20.0, "node_spacing": 0.5}
    scratch = tmp_path / "project-scratch"
    scratch.mkdir()
    bundle = tmp_path / "attempt-bundle"
    parent_input = scratch / "parent-input.json"
    parent_input.write_text(json.dumps(case))
    driver.run_case(adapter.case_to_driver_args(parent_input, bundle / "parent"))
    parent = bundle / "parent" / "axi_viscoelastic_Oh_1_G_1_lambda_inf"
    output = parent / "run_viscoelastic_axi"
    rows = np.loadtxt(output / "control.txt", ndmin=2)

    def restart_case(extension: float) -> dict:
        continued = copy.deepcopy(case)
        continued["completion"]["value"] = 0.08
        continued["physics"]["far_field"]["extension"] = extension
        continued["restart"] = {
            **{
                key: _reference(path)
                for key, path in {
                    "checkpoint": parent / "checkpoints" / "ckpt_00002.dump",
                    "input_case": parent_input,
                    "metadata": parent / "case.json",
                    "control": output / "control.txt",
                    "rim_height": output / "rim-height.txt",
                    "pvd": output / "liquid.pvd",
                }.items()
            },
            "time": float(rows[-1, 0]),
            "control_index": 2,
            "field_index": 2,
        }
        return continued

    shorter = scratch / "shorter.json"
    shorter.write_text(json.dumps(restart_case(10.0)))
    with pytest.raises(ValueError, match="never shorten"):
        driver.run_case(adapter.case_to_driver_args(shorter, bundle / "shorter"))

    longer = scratch / "longer.json"
    longer.write_text(json.dumps(restart_case(60.0)))
    metadata = driver.run_case(adapter.case_to_driver_args(longer, bundle / "longer"))
    extended = metadata["restart_far_field_extended"]
    assert extended["to_outer"] == pytest.approx(1.0 + 3.0 + 60.0)
    assert extended["from_outer"] == pytest.approx(1.0 + 3.0 + 20.0)
    resumed = np.loadtxt(
        bundle / "longer" / "axi_viscoelastic_Oh_1_G_1_lambda_inf" / "run_viscoelastic_axi"
        / "far-field.txt", ndmin=2)
    assert len(resumed) == 4 and np.all(np.isfinite(resumed))
    assert metadata["status"] == "passed"
