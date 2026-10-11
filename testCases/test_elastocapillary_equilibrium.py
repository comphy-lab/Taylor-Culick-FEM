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
import tempfile
import unittest
from pathlib import Path

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

    def test_each_defect_is_named(self) -> None:
        defects = {
            "geometry.frame": lambda c: c["geometry"].update(frame="laboratory"),
            "physics.problem": lambda c: c["physics"].update(problem="axisymmetric_viscoelastic"),
            "physics.r0": lambda c: c["physics"].update(r0=0),
            "mesh.elements_per_quarter": lambda c: c["mesh"].update(elements_per_quarter=1.5),
            "mesh.growth": lambda c: c["mesh"].update(growth=0.9),
            "mesh.length": lambda c: c["mesh"].update(length=1.5),
            "continuation.Ec_stop": lambda c: c["continuation"].update(Ec_stop=20.0),
            "continuation.initial_step": lambda c: c["continuation"].update(initial_step=0.05),
            "regularisation.cap_beta": lambda c: c["regularisation"].update(cap_beta=-1),
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
