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


def test_time_case_explicitly_allows_recoil(tmp_path: Path) -> None:
    case = _case({"kind": "time", "value": 200.0, "allow_recoil": True})
    case["mesh"]["length"] = 100.0
    path = _write_case(tmp_path, case)

    args = adapter.case_to_driver_args(path, tmp_path / "attempt")

    assert args.length == 100.0
    assert not args.stop_when_arrested
    assert args.allow_recoil
    assert args.t_end == 200.0


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
                "pvd": pvd,
            }.items()
        },
        "time": 0.1,
        "output_index": 2,
    }
    return case


def test_restart_selector_maps_to_driver_and_validates_prefix(tmp_path, restart_case):
    from run_viscoelastic_axi import _read_restart

    path = _write_case(tmp_path, restart_case)
    args = adapter.case_to_driver_args(path, tmp_path / "next")
    assert args.restart_case == path
    restart = _read_restart(args)
    assert restart["output_index"] == 2
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
        case["restart"]["output_index"] = True
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
                "pvd": parent_output / "liquid.pvd",
            }.items()
        },
        "time": float(rows[-1, 0]),
        "output_index": 2,
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
