"""Regression tests for the planar validation driver's output cadence."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_MODULE_PATH = Path(__file__).parents[1] / "validationCases" / "savva-bush-planar" / "run_validation.py"
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
_SPEC = importlib.util.spec_from_file_location("run_validation", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
driver = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(driver)


def test_driver_emits_one_field_output_per_scheduled_target(monkeypatch, tmp_path):
    run_calls = []
    run_targets = []
    output_times = []
    history_rows = []

    class History:
        def add_row(self, *row):
            history_rows.append(row)

    class Problem:
        def __init__(self):
            self.time = 0.0
            self.mu = None
            self.domain_length = None
            self.resolution = None

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def initialise(self):
            pass

        def create_text_file_output(self, *_args, **_kwargs):
            return History()

        def run(self, target, **kwargs):
            self.time = float(target)
            run_targets.append(self.time)
            run_calls.append(kwargs)

        def output(self):
            output_times.append(self.time)

        def control_velocity(self):
            return 0.5

        def get_current_time(self, *, as_float):
            assert as_float is True
            return self.time

    monkeypatch.setattr(driver, "PlanarTaylorCulickProblem", Problem)
    monkeypatch.chdir(tmp_path)

    driver.run_case(
        1.0,
        tmp_path / "case",
        resolution=0.5,
        tstar_end=1.0,
        length=3.0,
        temporal_error=1e-3,
        n_outputs=4,
    )

    assert len(run_calls) == len(output_times) == len(history_rows) == 4
    endpoint = driver.tau_vis(driver.viscosity(1.0))
    expected_targets = np.geomspace(endpoint * 1e-4, endpoint, 4)
    np.testing.assert_allclose(run_targets, expected_targets)
    assert output_times == [row[0] for row in history_rows]
    assert all(call["outstep"] is False for call in run_calls)
    assert all(call["out_initially"] is False for call in run_calls)
