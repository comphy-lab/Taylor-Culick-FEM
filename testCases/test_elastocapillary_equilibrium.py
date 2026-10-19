"""The steady elastocapillary equilibrium: case contract, geometry, theory and runner checks.

The solver itself is exercised by ``EQUILIBRIUM_SOLVER_TESTS=1``, which runs a short
continuation in the pinned environment; everything else needs neither pyoomph nor a solve.
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from taylor_culick_fem.cases import (
    EQUILIBRIUM,
    EQUILIBRIUM_FIELDS_USED,
    case_problems,
    driver_arguments,
)
from taylor_culick_fem.depth_averaged_equilibrium import profile, radial_stretch, shoot
from taylor_culick_fem.equilibrium_branch import SegmentSettings
from taylor_culick_fem.equilibrium_geometry import (
    HALF_THICKNESS,
    FieldMap,
    design_grids,
    incompressibility_error,
    radial_nodes,
    reference_grids,
)

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "simulationCases" / "elastocapillary_equilibrium.json"
RUNNER = ROOT / "simulationCases" / "run_elastocapillary_equilibrium.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("run_elastocapillary_equilibrium", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def field_map(grids: dict, displacement) -> FieldMap:
    """Q2 elements of nine-node block grids carrying ``displacement(X)``."""
    Xe, ij = [], []
    local = [(i, j) for j in range(3) for i in range(3)]
    for grid in grids.values():
        for p in range(0, grid.shape[0] - 1, 2):
            for q in range(0, grid.shape[1] - 1, 2):
                Xe.append([grid[p + i, q + j] for i, j in local])
                ij.append(local)
    Xe = np.asarray(Xe, float)
    ue = np.apply_along_axis(displacement, 2, Xe)
    return FieldMap(Xe, ue, np.asarray(ij))


class EquilibriumCaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = json.loads(CASE.read_text(encoding="utf-8"))

    def test_default_case_is_valid(self) -> None:
        self.assertEqual(case_problems(self.case), [])
        self.assertEqual(case_problems(self.case, problem=EQUILIBRIUM), [])

    def test_lagrangian_frame_and_reference_alias_are_equivalent(self) -> None:
        for frame in ("lagrangian", "reference"):
            case = copy.deepcopy(self.case)
            case["geometry"]["frame"] = frame
            self.assertEqual(case_problems(case, problem=EQUILIBRIUM), [])
            self.assertEqual(SegmentSettings.from_case(case), SegmentSettings.from_case(self.case))

    def test_each_defect_is_named(self) -> None:
        defects = {
            "geometry.frame": lambda c: c["geometry"].update(frame="laboratory"),
            "physics.problem": lambda c: c["physics"].update(problem="axisymmetric_viscoelastic"),
            "physics.r0": lambda c: c["physics"].update(r0=0),
            "mesh.elements_per_quarter": lambda c: c["mesh"].update(elements_per_quarter=1.5),
            "mesh.template": lambda c: c["mesh"].update(template="other-body"),
            "mesh.growth": lambda c: c["mesh"].update(growth=0.9),
            "mesh.length": lambda c: c["mesh"].update(length=1.5),
            "continuation.Ec_stop": lambda c: c["continuation"].update(Ec_stop=20.0),
            "continuation.initial_step": lambda c: c["continuation"].update(initial_step=0.05),
            "regularisation.cap_beta": lambda c: c["regularisation"].update(cap_beta=-1),
            "regularisation.volume_penalty": lambda c: c["regularisation"].update(volume_penalty=-1),
            "mesh.columns": lambda c: c["mesh"].update(columns="radial"),
            "mesh.shoulder_angle": lambda c: c["mesh"].update(shoulder_angle=95),
            "mesh.blend": lambda c: c["mesh"].update(blend=0),
            "case.restart": lambda c: c.update(restart={}),
        }
        for name, damage in defects.items():
            case = copy.deepcopy(self.case)
            damage(case)
            with self.subTest(defect=name):
                problems = case_problems(case, problem=EQUILIBRIUM)
                self.assertTrue(problems)
                self.assertTrue(any(name.split(".")[-1] in p for p in problems), problems)

    def test_a_viscoelastic_runner_refuses_it(self) -> None:
        self.assertTrue(case_problems(self.case, problem="axisymmetric_viscoelastic"))
        with self.assertRaises(ValueError):
            driver_arguments(self.case, Path("unused"))

    def test_fields_used_cover_every_field_read(self) -> None:
        settings = SegmentSettings.from_case(self.case)
        self.assertEqual(settings.ec_start, self.case["continuation"]["Ec_start"])
        for section in ("physics", "mesh", "continuation", "solver"):
            for key in self.case[section]:
                if section == "physics" and key == "problem":
                    continue
                with self.subTest(field=f"{section}.{key}"):
                    self.assertIn(f"{section}.{key}", EQUILIBRIUM_FIELDS_USED)
        self.assertIn("regularisation.cap_beta", EQUILIBRIUM_FIELDS_USED)
        self.assertIn("regularisation.volume_penalty", EQUILIBRIUM_FIELDS_USED)
        self.assertEqual(settings.volume_penalty, 0.0)


RELAX = ROOT / "verificationCases" / "equilibrium-attraction" / "relax_to_equilibrium.py"


@unittest.skipUnless(importlib.util.find_spec("pyoomph") is not None, "needs the pinned pyoomph environment")
class CapStiffeningTests(unittest.TestCase):
    def test_boundary_defaults_to_the_cap_end(self) -> None:
        from taylor_culick_fem.elastocapillary_equilibrium import ElastocapillaryEquilibriumProblem

        problem = ElastocapillaryEquilibriumProblem()
        self.assertEqual(problem.cap_stiffening_boundary(), problem.r0 + HALF_THICKNESS)
        problem.r0 = 2.0
        self.assertEqual(problem.cap_stiffening_boundary(), 2.0 + HALF_THICKNESS)
        problem.cap_stiffening_radius = 1.7
        self.assertEqual(problem.cap_stiffening_boundary(), 1.7)


class SegmentTests(unittest.TestCase):
    """run_segment and build_problem with the pyoomph problem replaced by stand-ins."""

    def setUp(self) -> None:
        self.case = json.loads(CASE.read_text(encoding="utf-8"))
        self.settings = SegmentSettings.from_case(self.case)

    def test_a_start_past_the_stop_takes_no_step(self) -> None:
        from taylor_culick_fem import equilibrium_branch as branch

        class Problem:
            def arclength_continuation(self, *args, **kwargs):
                raise AssertionError("a segment already past Ec_stop must not step")

            def deformed_surface(self):
                return np.zeros((2, 4))

        class Field:
            def save(self, path, **kwargs):
                Path(path).write_bytes(b"")

        problem, parameter = Problem(), types.SimpleNamespace(value=math.log(0.5))
        row = {"Ec": 0.5, "R": 3.0, "H": 1.2}
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(branch, "_start", lambda *args: (problem, parameter)), \
                mock.patch.object(branch, "_point", lambda *args: dict(row)), \
                mock.patch.object(branch.FieldMap, "from_problem", lambda p: Field()):
            result = branch.run_segment(self.settings, None, Path(tmp) / "s000", ec_stop=0.6)
        self.assertEqual(result["status"], "ec_stop")
        self.assertEqual(len(result["rows"]), 1)

    def test_the_case_newton_limit_is_the_problem_default(self) -> None:
        from taylor_culick_fem import equilibrium_branch as branch

        class Problem:
            def set_output_directory(self, folder): pass
            def get_mesh(self, name): return None
            def set_linear_solver(self, name): pass
            def set_eigensolver(self, name): pass
            def initialise(self): pass

        fake = types.ModuleType("taylor_culick_fem.elastocapillary_equilibrium")
        fake.ElastocapillaryEquilibriumProblem = Problem
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(sys.modules, {"taylor_culick_fem.elastocapillary_equilibrium": fake}):
            problem = branch.build_problem(self.settings, {}, Path(tmp))
        self.assertEqual(problem.max_newton_iterations, self.case["solver"]["max_newton"])


class RelaxationContractTests(unittest.TestCase):
    def test_fields_used_cover_the_segment_settings(self) -> None:
        """Every case field SegmentSettings.from_case reads is listed in the relaxation manifest."""
        spec = importlib.util.spec_from_file_location("relax_to_equilibrium", RELAX)
        relax = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(relax)
        read: set[str] = set()

        class Recorder(dict):
            def __init__(self, data: dict, prefix: str) -> None:
                super().__init__(data)
                self.prefix = prefix

            def _seen(self, key, value):
                path = f"{self.prefix}{key}"
                read.add(path)
                return Recorder(value, path + ".") if isinstance(value, dict) else value

            def __getitem__(self, key):
                return self._seen(key, super().__getitem__(key))

            def get(self, key, default=None):
                return self._seen(key, super().get(key, default))

        case = json.loads((RELAX.parent / "relaxation.json").read_text(encoding="utf-8"))
        SegmentSettings.from_case(Recorder(case, ""))
        leaves = {path for path in read if not any(other.startswith(path + ".") for other in read)}
        self.assertTrue(leaves)
        self.assertEqual(sorted(leaves - set(relax.CASE_FIELDS_USED)), [])


class ReferenceGeometryTests(unittest.TestCase):
    def test_radial_nodes_grow_and_cap(self) -> None:
        nodes = radial_nodes(1.5, 0.1, 1.2, 100.0, max_spacing=5.0)
        steps = np.diff(nodes)
        self.assertEqual((nodes[0], nodes[-1]), (1.5, 100.0))
        self.assertTrue(np.all(steps > 0))
        self.assertLessEqual(steps.max(), 5.0 * 1.5)
        with self.assertRaises(ValueError):
            radial_nodes(2.0, 0.1, 1.2, 1.0)

    def test_cap_lies_on_the_reference_circle(self) -> None:
        r0 = 1.0
        for quadratic in (False, True):
            grids = reference_grids(r0, 200.0, 4, 1.1, quadratic=quadratic)
            centre = np.array([r0 + HALF_THICKNESS, 0.0])
            arc = np.concatenate([grids["lower"][0, :], grids["upper"][:, -1]])
            with self.subTest(quadratic=quadratic):
                np.testing.assert_allclose(np.hypot(*(arc - centre).T), HALF_THICKNESS, atol=1e-12)
                sheet = grids["sheet"]
                self.assertAlmostEqual(sheet[0, 0, 0], r0 + HALF_THICKNESS)
                self.assertAlmostEqual(sheet[-1, 0, 0], 200.0)
                np.testing.assert_allclose(sheet[:, -1, 1], HALF_THICKNESS)
                np.testing.assert_allclose(sheet[:, 0, 1], 0.0)

    def test_quadratic_grids_carry_mid_nodes(self) -> None:
        linear = reference_grids(1.0, 50.0, 3, 1.1)
        quadratic = reference_grids(1.0, 50.0, 3, 1.1, quadratic=True)
        for name in ("core", "lower", "upper"):
            self.assertEqual(quadratic[name].shape[:2],
                             tuple(2 * (n - 1) + 1 for n in linear[name].shape[:2]))
        np.testing.assert_allclose(quadratic["sheet"][1::2, 0, 0],
                                   0.5 * (quadratic["sheet"][:-1:2, 0, 0] + quadratic["sheet"][2::2, 0, 0]))


class FieldMapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grids = reference_grids(1.0, 20.0, 3, 1.3, quadratic=True)
        self.matrix = np.array([[0.3, 0.1], [-0.05, -0.2]])
        self.shift = np.array([0.7, 0.0])

    def test_affine_field_round_trips(self) -> None:
        fmap = field_map(self.grids, lambda X: self.matrix @ X + self.shift)
        for X in ([1.02, 0.01], [1.4, 0.3], [1.6, 0.45], [7.3, 0.2], [19.0, 0.49]):
            X = np.asarray(X)
            u = fmap.forward(X)
            np.testing.assert_allclose(u, self.matrix @ X + self.shift, atol=1e-10)
            np.testing.assert_allclose(fmap.pullback(X + u), X, atol=1e-10)

    def test_point_outside_the_body_is_refused(self) -> None:
        fmap = field_map(self.grids, lambda X: 0 * X)
        with self.assertRaises(RuntimeError):
            fmap.forward(np.array([0.5, 0.2]))

    def test_incompressibility_error(self) -> None:
        self.assertLess(incompressibility_error(field_map(self.grids, lambda X: 0 * X)), 1e-12)
        stretch = field_map(self.grids, lambda X: np.array([0.1 * X[0], 0.0]))
        # J including the hoop stretch is 1.1^2 everywhere.
        self.assertAlmostEqual(incompressibility_error(stretch), 0.21, places=10)

    def test_fan_columns_are_vertical_on_a_flat_sheet(self) -> None:
        # Undeformed, the surface beyond the cap is flat, so arclength along it equals
        # radial distance and the fan's columns coincide with vertical ones.
        fmap = field_map(self.grids, lambda X: 0 * X)
        vertical = design_grids(fmap, 3, 20.0, growth=1.3, max_spacing=5.0)
        fan = design_grids(fmap, 3, 20.0, growth=1.3, max_spacing=5.0, columns="fan")
        for name in vertical:
            np.testing.assert_allclose(fan[name], vertical[name], atol=1e-9)
        with self.assertRaises(ValueError):
            design_grids(fmap, 3, 20.0, columns="radial")

    def test_save_and_load(self) -> None:
        fmap = field_map(self.grids, lambda X: self.matrix @ X)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.npz"
            fmap.save(path, log_ec=0.5)
            loaded = FieldMap.load(path)
            np.testing.assert_array_equal(loaded.ue, fmap.ue)
            self.assertEqual(float(np.load(path)["log_ec"]), 0.5)


class DepthAveragedTheoryTests(unittest.TestCase):
    def test_radial_stretch_solves_its_equation(self) -> None:
        for c, tension in ((0.5, -1.0), (0.9, 0.0), (1.0, 2.0), (0.05, -0.04)):
            a = radial_stretch(c, tension)
            self.assertAlmostEqual(c * (a - c**2 / a**3), tension, places=10)

    def test_far_field_is_the_lame_field(self) -> None:
        sol = shoot(1.0)
        rho = sol.t[0]
        self.assertAlmostEqual(sol.y[0, 0], rho + 1.0 / rho, places=12)

    def test_hole_radius_and_its_asymptote(self) -> None:
        self.assertAlmostEqual(profile(0.5)["R"], 34.047, delta=0.01)
        # ln R = 2/Ec + const at small Ec; the slope in 1/Ec tends to 2.
        lo, hi = profile(0.25)["R"], profile(0.3)["R"]
        slope = (math.log(lo) - math.log(hi)) / (1 / 0.25 - 1 / 0.3)
        self.assertAlmostEqual(slope, 2.0, delta=0.01)
        self.assertAlmostEqual(math.log(lo) - 2 / 0.25, -0.4725, delta=0.005)

    def test_scales_with_the_edge_radius(self) -> None:
        self.assertAlmostEqual(profile(1.0, r0=2.0)["R"] / profile(1.0)["R"], 2.0, places=6)

    def test_thickness_decays_to_the_sheet(self) -> None:
        prof = profile(0.7)
        self.assertGreater(prof["lambda_z"][0], 1.0)
        self.assertAlmostEqual(prof["lambda_z"][-1], 1.0, places=6)


class EquilibriumRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = load_runner()

    def refused(self, *argv: str) -> str:
        with self.assertRaises(SystemExit) as raised:
            self.runner.main(list(argv))
        return str(raised.exception)

    def test_check_creates_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                self.assertEqual(self.runner.main([str(CASE), "--out", str(out), "--check"]), 0)
            self.assertEqual(json.loads(printed.getvalue())["check"], "passed")
            self.assertFalse(out.exists())

    def test_refusals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            self.assertIn("--ec-stop", self.refused(str(CASE), "--out", str(out), "--ec-stop", "-1"))
            busy = Path(tmp) / "busy"
            busy.mkdir()
            (busy / "x").write_text("", encoding="utf-8")
            self.assertIn("own new output folder", self.refused(str(CASE), "--out", str(busy)))
            stray = Path(tmp) / "state.npz"
            stray.write_bytes(b"")
            self.assertIn("segments/sNNN/state.npz",
                          self.refused(str(CASE), "--out", str(out), "--restart-from", str(stray)))
            self.assertFalse(out.exists())

    def test_reversed_segments_stall(self) -> None:
        """Repeated two-row reversed segments above Ec_stop stop as stalled, not forever."""
        case = json.loads(CASE.read_text(encoding="utf-8"))
        limit = int(case["continuation"]["max_stalled_segments"])
        ec = 2.0 * float(case["continuation"]["Ec_stop"])
        calls: list[Path] = []

        def row(tag: str) -> dict:
            return {"Ec": ec, "R": 2.0, "H": 1.2, "A": 0.0, "leading_eigenvalue": -1.0,
                    "J_err": 0.0, "ndof": 10, "cap_beta": 0.0, "tag": tag}

        def reversed_segment(settings, state, folder, ec_stop):
            calls.append(folder)
            if len(calls) > limit + 2:
                raise AssertionError("the runner kept continuing after reversed segments")
            return {"rows": [row("start"), row("dlog=+0.01000")], "status": "parameter_reversed",
                    "state": folder / "state.npz"}

        def fake_start_run(out, **kwargs):
            out.mkdir(parents=True)
            return "0" * 40

        jit = types.ModuleType("pyoomph.generic.jit_cache")
        jit.set_enabled = lambda enabled: None
        fake = {"pyoomph": types.ModuleType("pyoomph"), "pyoomph.generic": types.ModuleType("pyoomph.generic"),
                "pyoomph.generic.jit_cache": jit}
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(sys.modules, fake), \
                mock.patch("taylor_culick_fem.equilibrium_branch.run_segment", reversed_segment), \
                mock.patch.object(self.runner, "start_run", fake_start_run), \
                mock.patch.object(self.runner, "enforce_launcher_affinity", lambda cpus: None):
            out = Path(tmp) / "out"
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                self.assertEqual(self.runner.main([str(CASE), "--out", str(out)]), 1)
            self.assertEqual(len(calls), limit)
            self.assertEqual(json.loads(printed.getvalue())["status"], "stalled")
            self.assertEqual(json.loads((out / "case.json").read_text())["status"], "stalled")


@unittest.skipUnless(os.environ.get("EQUILIBRIUM_SOLVER_TESTS") == "1"
                     and importlib.util.find_spec("pyoomph") is not None,
                     "set EQUILIBRIUM_SOLVER_TESTS=1 in the pinned pyoomph environment")
class EquilibriumSolverTests(unittest.TestCase):
    def test_short_branch(self) -> None:
        case = json.loads(CASE.read_text(encoding="utf-8"))
        case["mesh"].update(elements_per_quarter=4, length=2000.0, max_spacing=100.0)
        case["continuation"].update(Ec_start=2.0, Ec_stop=1.4, segment_steps=3,
                                    segment_log_radius_growth=0.2)
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "case.json"
            path.write_text(json.dumps(case), encoding="utf-8")
            out = Path(tmp) / "run"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main([str(path), "--out", str(out)]), 0)
            rows = np.genfromtxt(out / "branch.csv", delimiter=",", names=True, dtype=None,
                                 encoding=None)
            self.assertAlmostEqual(float(rows["R"][0]), 1.702385, delta=2e-3)
            self.assertTrue(np.all(rows["leading_eigenvalue"] < 0))
            self.assertTrue(np.all(np.diff(rows["R"]) > 0))
            segments = sorted(p.name for p in (out / "segments").iterdir())
            self.assertGreaterEqual(len(segments), 2)
            events = [json.loads(line)["event"] for line in (out / "progress.jsonl").read_text().splitlines()]
            self.assertEqual((events[0], events[-1]), ("started", "completed"))


if __name__ == "__main__":
    unittest.main()
