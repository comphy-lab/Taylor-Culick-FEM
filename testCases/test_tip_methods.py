"""The two tip methods: the graded face layer and the transient Lagrangian solver.

Case contract, mesh helpers and the default-preserving options need neither pyoomph nor a
solve. ``LAGRANGIAN_SOLVER_TESTS=1`` in the pinned pyoomph environment also runs short solves:
a static equilibrium held at rest, the start of the Ec = 1 opening against the co-moving
ALE solver, the restart of a remeshed run, and the face-layer mesh.
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
from typing import ClassVar

import numpy as np

from taylor_culick_fem.cases import (
    LAGRANGIAN,
    case_problems,
    driver_arguments,
    face_layer_problems,
)
from taylor_culick_fem.equilibrium_geometry import (
    FieldMap,
    design_grids,
    reference_grids,
)

ROOT = Path(__file__).resolve().parents[1]
LAGRANGIAN_CASE = ROOT / "simulationCases" / "axisymmetric_lagrangian.json"
VISCOELASTIC_CASE = ROOT / "simulationCases" / "axisymmetric_viscoelastic.json"
RUNNER = ROOT / "simulationCases" / "run_axisymmetric_lagrangian.py"
SOLVER = (os.environ.get("LAGRANGIAN_SOLVER_TESTS") == "1"
          and importlib.util.find_spec("pyoomph") is not None)


def load_runner():
    spec = importlib.util.spec_from_file_location("run_axisymmetric_lagrangian", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def face_end_index(points, max_slope_degrees=1.0):
    # The mesh module imports pyoomph; this copy of its rule is checked against it below.
    pts = np.asarray(points, dtype=float)
    chords = np.diff(pts, axis=0)
    slopes = np.degrees(np.arctan2(chords[:, 1], chords[:, 0]))
    flat = np.flatnonzero(slopes[2:] < max_slope_degrees)
    index = int(flat[0]) + 2 if flat.size else len(pts) - 3
    return min(index, len(pts) - 3)


def cap_surface(tip=1.0, step=0.0075):
    """Points on the initial rounded cap (radius 1/2, centre on the midplane), tip first."""
    count = max(4, round(0.25 * math.pi / step))
    return [(tip + 0.5 - 0.5 * math.cos(a), 0.5 * math.sin(a))
            for a in np.linspace(0.0, 0.5 * math.pi, count + 1)]


def film_surface(start, end, height, tip, bump=None, first=0.0075):
    """Film points from ``start`` to ``end``: from ``first`` the spacing grows by 1.1 per point
    up to half the graded tip spacing (boundary coordinates include the mid-side nodes, and
    a mesh's boundary spacing has no jumps); ``bump(r)`` is added to ``height``."""
    points, r, step = [], start, first
    while r < end - 1e-12:
        r = min(end, r + min(step, 0.5 * (0.015 + 0.235 * min(1.0, (r - tip) / 6.0))))
        step *= 1.1
        points.append((r, height(r) + (bump(r) if bump else 0.0)))
    return points


def early_surface(length=20.0):
    """The surface at which the 2.5e-4 run failed (t = 0.0025): the cap, then a film carrying
    a 6.5e-6 bump that peaks at r = 1.8, i.e. well beyond the cap end."""
    cap = cap_surface()
    film = film_surface(1.5, 1.0 + length, lambda r: 0.5, 1.0,
                        lambda r: 6.5e-6 * math.exp(-(((r - 1.8) / 0.4) ** 2)))
    return cap + film


def rim_surface(tip=5.0, rim=1.0, length=20.0, step=0.0075):
    """A formed rim: a quarter-circle face of radius ``rim`` from the tip to the rim top, then
    a Gaussian fall to the half-thickness 1/2 of the film, whose width makes the curvature
    continuous (1/rim) at the top."""
    count = round(0.5 * math.pi * rim / step)
    face = [(tip + rim - rim * math.cos(a), rim * math.sin(a))
            for a in np.linspace(0.0, 0.5 * math.pi, count + 1)]
    width = math.sqrt(2.0 * rim * (rim - 0.5))
    back = film_surface(tip + rim, tip + length, lambda r: 0.5 + (rim - 0.5) * math.exp(
        -(((r - tip - rim) / width) ** 2)), tip, first=step)
    return face + back


class FaceLayerCaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = json.loads(VISCOELASTIC_CASE.read_text(encoding="utf-8"))

    def test_absent_layer_leaves_the_driver_arguments_unchanged(self) -> None:
        self.assertNotIn("--face-layer", driver_arguments(self.case, Path("/tmp/x")))

    def test_a_valid_layer_is_passed_to_the_driver(self) -> None:
        self.case["mesh"]["face_layer"] = {"spacing": 1e-3, "growth": 1.2, "thickness": 0.07}
        self.assertEqual(case_problems(self.case), [])
        arguments = driver_arguments(self.case, Path("/tmp/x"))
        value = json.loads(arguments[arguments.index("--face-layer") + 1])
        self.assertEqual(value, {"growth": 1.2, "spacing": 1e-3, "thickness": 0.07})

    def test_tip_diagnostics_are_opt_in(self) -> None:
        self.assertNotIn("--tip-diagnostics", driver_arguments(self.case, Path("/tmp/x")))
        self.case["output"]["tip_diagnostics"] = True
        self.assertIn("--tip-diagnostics", driver_arguments(self.case, Path("/tmp/x")))
        self.case["output"]["tip_diagnostics"] = "yes"
        self.assertIn("output.tip_diagnostics must be a boolean", case_problems(self.case))

    def test_each_layer_defect_is_named(self) -> None:
        cases = {
            "exactly spacing": {"spacing": 1e-3, "growth": 1.2},
            "growth must lie": {"spacing": 1e-3, "growth": 1.3, "thickness": 0.07},
            "spacing must be smaller": {"spacing": 0.1, "growth": 1.1, "thickness": 0.07},
            "finer than mesh.resolution": {"spacing": 0.05, "growth": 1.1, "thickness": 0.2},
            "finite positive": {"spacing": -1.0, "growth": 1.1, "thickness": 0.2},
        }
        for fragment, layer in cases.items():
            with self.subTest(fragment=fragment):
                problems = face_layer_problems(layer, 0.02)
                self.assertTrue(any(fragment in p for p in problems), problems)

    def test_face_end_is_the_rim_top_or_the_cap_end(self) -> None:
        cap = [(1 + 0.5 - 0.5 * math.cos(a), 0.5 * math.sin(a)) for a in np.linspace(0, math.pi / 2, 9)]
        flat = [(1.5 + k, 0.5) for k in range(1, 6)]
        self.assertEqual(face_end_index(cap + flat), 8)
        rim = [(10 + 0.1 * k, 2 * math.sin(math.pi * k / 20)) for k in range(21)] + [(13, 0.5)] * 3
        self.assertEqual(face_end_index(rim), 10)

    def test_face_end_before_a_rim_forms_is_the_cap_end(self) -> None:
        # The 2.5e-4 run remeshed at t = 0.0025 with a 6.5e-6 bump on the film beyond the cap.
        # The face must still end at the cap (r ~ 1.5), not at the bump's crest (r = 1.8).
        surface = early_surface()
        end = face_end_index(surface)
        self.assertLess(abs(surface[end][0] - 1.5), 0.01)
        self.assertGreater(surface[end][1], 0.4999)
        self.assertGreater(max(range(len(surface)), key=lambda i: surface[i][1]), end)

    def test_face_end_of_a_formed_rim_is_its_top(self) -> None:
        surface = rim_surface()
        end = face_end_index(surface)
        self.assertLess(abs(surface[end][0] - 6.0), 0.02)
        self.assertGreater(surface[end][1], 0.9998)

    def test_face_end_leaves_both_splines_a_segment(self) -> None:
        rising = [(1 + 0.1 * k, 0.1 * k) for k in range(8)]
        self.assertEqual(face_end_index(rising), len(rising) - 3)

    @unittest.skipUnless(SOLVER, "set LAGRANGIAN_SOLVER_TESTS=1 in the pinned pyoomph environment")
    def test_the_module_rule_matches(self) -> None:
        from taylor_culick_fem.axisymmetric_comoving import face_end_index as module_rule

        rim = [(10 + 0.1 * k, 2 * math.sin(math.pi * k / 20)) for k in range(21)] + [(13, 0.5)] * 3
        for surface in (rim, early_surface(), rim_surface(), [(1 + 0.1 * k, 0.1 * k) for k in range(8)]):
            self.assertEqual(module_rule(surface), face_end_index(surface))
        with self.assertRaises(ValueError):
            module_rule(rim[:4])


def mesh_quality(msh: Path) -> dict:
    """minSICN, minSJ and the largest interior angle (degrees) of the vertex triangles."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.open(str(msh))
        tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = coords.reshape(-1, 3)
        index = {t: i for i, t in enumerate(tags)}
        _, elements, nodes = gmsh.model.mesh.getElements(2)
        elements = np.asarray(elements[0])
        nodes = np.asarray(nodes[0]).reshape(len(elements), -1)
        sicn = np.asarray(gmsh.model.mesh.getElementQualities(elements, "minSICN"))
        sj = np.asarray(gmsh.model.mesh.getElementQualities(elements, "minSJ"))
    finally:
        gmsh.finalize()
    vertices = np.array([[coords[index[n]][:2] for n in row[:3]] for row in nodes])
    largest = np.zeros(len(vertices))
    for i in range(3):
        u = vertices[:, (i + 1) % 3] - vertices[:, i]
        w = vertices[:, (i + 2) % 3] - vertices[:, i]
        cosine = (u * w).sum(1) / np.linalg.norm(u, axis=1) / np.linalg.norm(w, axis=1)
        largest = np.maximum(largest, np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    edges = np.concatenate([np.linalg.norm(vertices[:, i] - vertices[:, (i + 1) % 3], axis=1)
                            for i in range(3)])
    return {"sicn": float(sicn.min()), "sj": float(sj.min()), "angle": float(largest.max()),
            "shortest": float(edges.min()), "count": len(elements)}


@unittest.skipUnless(SOLVER, "set LAGRANGIAN_SOLVER_TESTS=1 in the pinned pyoomph environment")
class FaceLayerMeshQualityTests(unittest.TestCase):
    """The face-layer mesher on the initial cap and on remeshed (deformed) surfaces.

    Floors, against the defect of 1a89aa9 (spacing 2.5e-4 / 1e-3) in brackets:
    - largest interior angle below 125 degrees: a sliver approaches 180 (179.95 / 178.5);
      layer triangles stay near 90, and the fixed mesher gives at most ~114;
    - minimum scaled Jacobian of the quadratic elements above 0.5 (0.63 / 0.93; the fix
      gives ~0.96, and 0.07 on a large rim before the face spacing was capped);
    - minSICN above 0.1 x spacing / resolution, a tenth of the layer's intended anisotropy
      (0.0024 / 0.05 of it).
    """

    LAYERS: ClassVar[dict[float, float]] = {1e-3: 0.07, 2.5e-4: 0.08}

    def build(self, spacing, surface=None):
        from pyoomph.equations.poisson import PoissonEquation

        from taylor_culick_fem.axisymmetric_comoving import (
            AxisymmetricComovingProblem,
            ComovingAnnularSheetMesh,
        )

        class SurfaceMesh(ComovingAnnularSheetMesh):
            # Remeshes from a given surface, as the remesher does from the deformed one.
            def is_remeshing(self):
                return surface is not None

            def get_boundary_coordinates(self, name, sort_along_axis=None, **kwargs):
                return [list(surface)]

        class MeshOnly(AxisymmetricComovingProblem):
            def define_problem(self):
                self.add_mesh(SurfaceMesh())
                self.add_equations(PoissonEquation(name="u", space="C1") @ "liquid")

        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        problem = MeshOnly()
        problem.ignore_command_line = True
        problem.set_output_directory(tmp)
        problem.X0, problem.window_length = 1.0, 20.0
        problem.resolution, problem.far_resolution = 0.015, 0.25
        problem.face_layer = {"spacing": spacing, "growth": 1.2, "thickness": self.LAYERS[spacing]}
        problem.initialise()
        quality = mesh_quality(Path(tmp) / "_gmsh" / "SurfaceMesh.msh")
        problem.release()
        return quality

    def check(self, quality, spacing):
        self.assertLess(quality["angle"], 125.0, quality)
        self.assertGreater(quality["sj"], 0.5, quality)
        self.assertGreater(quality["sicn"], 0.1 * spacing / 0.015, quality)
        self.assertGreater(quality["shortest"], 0.7 * spacing, quality)
        self.assertLess(quality["shortest"], 1.25 * spacing, quality)

    def test_initial_mesh(self) -> None:
        for spacing in self.LAYERS:
            with self.subTest(spacing=spacing):
                self.check(self.build(spacing), spacing)

    def test_remesh_of_the_failing_early_surface(self) -> None:
        for spacing in self.LAYERS:
            with self.subTest(spacing=spacing):
                self.check(self.build(spacing, early_surface()), spacing)

    def test_remesh_of_a_formed_rim(self) -> None:
        # Radius 4: the face is long and curved, so the face spacing cap matters.
        for spacing in self.LAYERS:
            for rim in (1.0, 4.0):
                with self.subTest(spacing=spacing, rim=rim):
                    surface = rim_surface(tip=3.0 + 2.0 * rim, rim=rim, length=20.0 + 2.0 * rim)
                    self.check(self.build(spacing, surface), spacing)


class LagrangianCaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = json.loads(LAGRANGIAN_CASE.read_text(encoding="utf-8"))

    def test_default_case_is_valid(self) -> None:
        self.assertEqual(self.case["physics"]["problem"], LAGRANGIAN)
        self.assertEqual(case_problems(self.case), [])

    def test_each_defect_is_named(self) -> None:
        edits = {
            'relaxation_time must be "inf"': ("physics", "relaxation_time", 10.0),
            "far_edge": ("physics", "far_edge", "free"),
            "mesh.growth must be a number above 1": ("mesh", "growth", 1.0),
            "solver.dt must be a finite positive": ("solver", "dt", 0.0),
            "whole number of solver.dt": ("output", "cadence", 0.015),
            "completion.kind must be 'time'": ("completion", "kind", "event"),
            "max_spacing must be at least": ("mesh", "max_spacing", 0.1),
            "surface_tension must be 1": ("physics", "surface_tension", 4.0),
            "shoulder_angle must be a number": ("mesh", "shoulder_angle", "bad"),
            "shoulder_angle must be a number of degrees": ("mesh", "shoulder_angle", 95.0),
            "blend must be a positive number": ("mesh", "blend", 0.0),
        }
        for fragment, (section, key, value) in edits.items():
            case = copy.deepcopy(self.case)
            case[section][key] = value
            with self.subTest(fragment=fragment):
                problems = case_problems(case)
                self.assertTrue(any(fragment in p for p in problems), problems)

    def test_seed_needs_paths_and_digests(self) -> None:
        case = copy.deepcopy(self.case)
        case["seed"] = {"state": "s.npz", "grids": "g.npz", "state_sha256": "0" * 64,
                        "grids_sha256": "x"}
        self.assertTrue(any("grids_sha256" in p for p in case_problems(case)))
        case["seed"]["grids_sha256"] = "a" * 64
        self.assertEqual(case_problems(case), [])

    def test_runner_check_needs_no_solver(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(runner.main([str(LAGRANGIAN_CASE), "--out", str(Path(tmp) / "r"), "--check"]), 0)
        self.assertEqual(json.loads(out.getvalue())["check"], "passed")

    def test_a_continuation_that_does_not_advance_is_refused(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "parent"
            (parent / "checkpoints").mkdir(parents=True)
            case_sha = runner.sha256(LAGRANGIAN_CASE)
            manifest = {"case_sha256": case_sha,
                        "argv": [str(RUNNER), str(LAGRANGIAN_CASE), "--out", str(parent)]}
            (parent / "run-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            state = parent / "checkpoints" / "ckpt_00600.npz"
            np.savez(state, time=30.0)
            out = Path(tmp) / "child"
            for extra in ([], ["--t-end", "30"], ["--t-end", "20"]):
                with self.subTest(extra=extra), self.assertRaises(SystemExit) as caught:
                    runner.main([str(LAGRANGIAN_CASE), "--out", str(out), "--restart-from", str(state),
                                 *extra])
                self.assertIn("a continuation must advance", str(caught.exception))
            self.assertFalse(out.exists())
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                self.assertEqual(runner.main([str(LAGRANGIAN_CASE), "--out", str(out), "--restart-from",
                                              str(state), "--t-end", "40", "--check"]), 0)
            self.assertEqual(json.loads(printed.getvalue())["check"], "passed")

    def test_a_seed_must_be_the_case_body(self) -> None:
        runner = load_runner()
        grids = reference_grids(1.0, 300.0, 2, 1.2, None, 30.0, quadratic=True)
        nodes = _field_map(grids, lambda X: np.zeros(2)).Xe
        self.assertEqual(runner.seed_domain_problems(grids, nodes, 1.0, 300.0), [])
        short = reference_grids(1.0, 10.0, 2, 1.2, None, 30.0, quadratic=True)
        problems = runner.seed_domain_problems(short, nodes, 1.0, 300.0)
        self.assertTrue(any("seed.grids ends" in p for p in problems), problems)
        self.assertFalse(any("seed.state" in p for p in problems), problems)
        problems = runner.seed_domain_problems(grids, nodes, 2.0, 300.0)
        self.assertTrue(any("seed.state starts" in p for p in problems), problems)

    def test_a_continuation_resumes_on_the_next_cadence_boundary(self) -> None:
        runner = load_runner()
        # A parent that ended on a shortened interval at t = 0.025 resumes with t = 0.05 next.
        index = runner.cadence_index(0.025, 0.05)
        self.assertAlmostEqual((index + 1) * 0.05, 0.05)
        self.assertEqual(runner.cadence_index(0.05, 0.05), 1)
        self.assertEqual(runner.cadence_index(0.1499999999, 0.05), 3)
        self.assertEqual(runner.cadence_index(30.0, 0.05), 600)

    def test_viscosity_follows_the_project_convention(self) -> None:
        if importlib.util.find_spec("pyoomph") is None:
            self.skipTest("needs pyoomph")
        settings = load_runner().settings_from_case(self.case)
        self.assertEqual(settings.oh, 1.0)
        from taylor_culick_fem.lagrangian_dynamic import LagrangianRetractionProblem

        problem = LagrangianRetractionProblem()
        problem.oh = 0.5
        self.assertEqual(problem.viscosity, math.sqrt(2.0) * 0.5)


@unittest.skipUnless(importlib.util.find_spec("pyoomph") is not None, "needs pyoomph to import")
class LagrangianGeometryTests(unittest.TestCase):
    def test_sweep_radii_hold_then_grow(self) -> None:
        from taylor_culick_fem.lagrangian_dynamic import sweep_radii

        radii = sweep_radii(1.5, 0.02, 3000.0, sweep_end=40.0, sweep_spacing=0.25, growth=1.1,
                            max_spacing=300.0)
        steps = np.diff(radii)
        self.assertEqual(radii[0], 1.5)
        self.assertEqual(radii[-1], 3000.0)
        self.assertTrue(np.all(steps > 0))
        held = (radii[:-1] > 5) & (radii[1:] < 40.0 - 0.25)
        self.assertTrue(np.allclose(steps[held], 0.25))
        # radial_nodes merges a short final interval, so the last one may reach 1.5 times the cap.
        self.assertLessEqual(steps[:-1].max(), 300.0 + 1e-9)
        self.assertLessEqual(steps[-1], 450.0 + 1e-9)
        self.assertTrue(np.all(steps[1:-1] / steps[:-2] <= 1.1 + 1e-9))

    def test_initial_grids_keep_the_static_cap(self) -> None:
        from taylor_culick_fem.lagrangian_dynamic import initial_grids

        grids = initial_grids(1.0, 3000.0, 4, sweep_end=30.0, sweep_spacing=0.25, growth=1.1,
                              max_spacing=300.0)
        static = reference_grids(1.0, 3000.0, 4, 1.1, None, 300.0, quadratic=True)
        for name in ("core", "lower", "upper"):
            np.testing.assert_array_equal(grids[name], static[name])
        np.testing.assert_array_equal(grids["sheet"][0], static["sheet"][0])
        self.assertEqual(grids["sheet"][-1, 0, 0], 3000.0)
        self.assertAlmostEqual(float(grids["sheet"][:, 0, 0].max()), 3000.0)
        self.assertTrue(np.all(np.diff(grids["sheet"][:, 0, 0]) > 0))

    def test_initial_grids_accept_a_spacing_below_the_cap_spacing(self) -> None:
        from taylor_culick_fem.lagrangian_dynamic import initial_grids

        # elements_per_quarter = 8 gives a cap first spacing of 1/32 > max_spacing = 0.02.
        grids = initial_grids(1.0, 4.0, 8, sweep_end=3.0, sweep_spacing=0.01, growth=1.1,
                              max_spacing=0.02)
        static = reference_grids(1.0, 4.0, 8, 1.1, None, 1.0, quadratic=True)
        for name in ("core", "lower", "upper"):
            np.testing.assert_array_equal(grids[name], static[name])
        columns = grids["sheet"][::2, 0, 0]
        self.assertEqual(columns[-1], 4.0)
        self.assertLessEqual(float(np.diff(columns)[:-1].max()), 0.02 + 1e-12)

    def test_advance_lands_on_the_target_with_a_short_last_step(self) -> None:
        from taylor_culick_fem.lagrangian_dynamic import LagrangianRun, LagrangianSettings

        class Clock:
            def __init__(self):
                self.t, self.calls = 0.0, []

            def get_current_time(self, as_float=True):
                return self.t

            def run(self, endtime, startstep, maxstep, outstep):
                # pyoomph steps at startstep without shortening the last step to endtime.
                self.calls.append((endtime, startstep, maxstep))
                while self.t < endtime - 1e-12:
                    self.t += startstep

        run = LagrangianRun(LagrangianSettings(oh=1.0, elastic_modulus=1.0, dt=0.01), "/tmp/unused")
        run.problem = Clock()
        run.advance(0.005)
        self.assertAlmostEqual(run.time(), 0.005, delta=1e-15)
        self.assertEqual(run.problem.calls, [(0.005, 0.005, 0.005)])
        run.advance(0.05)
        self.assertAlmostEqual(run.time(), 0.05, delta=1e-12)
        self.assertEqual(len(run.problem.calls), 3)
        self.assertAlmostEqual(run.problem.calls[1][1], 0.01)
        self.assertAlmostEqual(run.problem.calls[2][1], 0.005, delta=1e-12)
        before = len(run.problem.calls)
        run.advance(0.05)
        self.assertEqual(len(run.problem.calls), before)

    def test_design_grids_default_is_unchanged_by_the_new_option(self) -> None:
        grids = reference_grids(1.0, 300.0, 2, 1.2, None, 30.0, quadratic=True)
        fmap = _field_map(grids, lambda X: np.array([0.3 / X[0], 0.0]))
        default = design_grids(fmap, 2, 300.0, growth=1.2, max_spacing=30.0)
        explicit = design_grids(fmap, 2, 300.0, growth=1.2, max_spacing=30.0, column_radii=None)
        for name in default:
            np.testing.assert_array_equal(default[name], explicit[name])

    def test_state_map_interpolates_every_level(self) -> None:
        from taylor_culick_fem.lagrangian_dynamic import StateMap

        grids = reference_grids(1.0, 50.0, 2, 1.3, None, 10.0, quadratic=True)
        fmap = _field_map(grids, lambda X: np.array([X[0] * 0.01, X[1] * 0.02]))
        levels = np.stack([np.stack([fmap.ue * (k + 1), -fmap.ue]) for k in range(3)])
        state = StateMap(fmap.Xe, fmap.ij, levels)
        X = np.array([7.3, 0.21])
        values = state.values_at(X)
        exact = np.array([X[0] * 0.01, X[1] * 0.02])
        for k in range(3):
            np.testing.assert_allclose(values[k, 0], (k + 1) * exact, atol=1e-12)
            np.testing.assert_allclose(values[k, 1], -exact, atol=1e-12)

    def test_volume_error_vanishes_for_an_isochoric_field(self) -> None:
        from taylor_culick_fem.lagrangian_dynamic import StateMap, state_integrals

        grids = reference_grids(1.0, 20.0, 2, 1.3, None, 5.0, quadratic=True)
        fmap = _field_map(grids, lambda X: np.array([0.0, 0.0]))
        levels = np.zeros((3, 2, *fmap.Xe.shape))
        out = state_integrals(StateMap(fmap.Xe, fmap.ij, levels), 1.0)
        self.assertEqual(out["volume_error"], 0.0)
        self.assertEqual(out["elastic"], 0.0)


def _field_map(grids: dict, displacement) -> FieldMap:
    Xe, ij = [], []
    local = [(i, j) for j in range(3) for i in range(3)]
    for grid in grids.values():
        for p in range(0, grid.shape[0] - 1, 2):
            for q in range(0, grid.shape[1] - 1, 2):
                Xe.append([grid[p + i, q + j] for i, j in local])
                ij.append(local)
    Xe = np.asarray(Xe, float)
    ue = np.array([[displacement(x) for x in element] for element in Xe])
    return FieldMap(Xe, ue, np.asarray(ij))


@unittest.skipUnless(SOLVER, "set LAGRANGIAN_SOLVER_TESTS=1 in the pinned pyoomph environment")
class LagrangianSolverTests(unittest.TestCase):
    def settings(self, **changes):
        from taylor_culick_fem.lagrangian_dynamic import LagrangianSettings

        values = {"oh": 1.0, "elastic_modulus": 1.0, "elements_per_quarter": 4,
                  "length": 3000.0, "max_spacing": 100.0, "dt": 0.01}
        values.update(changes)
        return LagrangianSettings(**values)

    def test_opening_starts_as_the_ale_solver(self) -> None:
        # The co-moving ALE reference (Oh = 1, Ec = 1, tip spacing 0.06): R(0.3) = 1.0371255.
        from taylor_culick_fem.lagrangian_dynamic import LagrangianRun

        with tempfile.TemporaryDirectory() as tmp:
            run = LagrangianRun(self.settings(), tmp)
            run.start_flat()
            run.advance(0.3)
            self.assertAlmostEqual(run.tip()["R"], 1.0371255, delta=1e-4)
            run.release()

    def test_static_equilibrium_is_a_rest_state(self) -> None:
        from taylor_culick_fem.equilibrium_branch import SegmentSettings, build_problem
        from taylor_culick_fem.lagrangian_dynamic import LagrangianRun

        case = json.loads((ROOT / "simulationCases" / "elastocapillary_equilibrium.json").read_text())
        case["mesh"].update(elements_per_quarter=4, length=2000.0, max_spacing=100.0)
        static_settings = SegmentSettings.from_case(case)
        with tempfile.TemporaryDirectory() as tmp:
            grids = reference_grids(1.0, 2000.0, 4, 1.1, None, 100.0, quadratic=True)
            static = build_problem(static_settings, grids, Path(tmp) / "static")
            static.get_global_parameter("log_Ec").value = math.log(2.0)
            static.newton_solver_tolerance = 1e-11
            static.solve(max_newton_iterations=30)
            fmap = FieldMap.from_problem(static)
            radius = static.hole_radius()
            static.release()
            run = LagrangianRun(self.settings(elastic_modulus=2.0, length=2000.0), Path(tmp) / "dyn")
            run.start_static_seed(grids, fmap)
            self.assertAlmostEqual(run.tip()["R"], radius, delta=1e-12)
            run.advance(0.5)
            self.assertLess(abs(run.tip()["R"] / radius - 1), 1e-8)
            self.assertLess(abs(run.tip()["U"]), 1e-7)
            run.release()

    def test_restart_after_remesh_reproduces_the_run(self) -> None:
        from taylor_culick_fem.lagrangian_dynamic import LagrangianRun

        settings = self.settings(elastic_modulus=0.1, remesh_log_radius_growth=0.02)
        with tempfile.TemporaryDirectory() as tmp:
            run = LagrangianRun(settings, Path(tmp) / "a")
            run.start_flat()
            run.advance(0.4)
            record = run.remesh(run.remesh_reason() or "test")
            self.assertEqual(record["R_before"], record["R_after"])
            self.assertLess(abs(record["kinetic_rel_change"]), 1e-4)
            run.checkpoint(Path(tmp) / "ck.npz", output_index=8)
            run.advance(0.5)
            first = run.tip()
            run.release()
            again = LagrangianRun(settings, Path(tmp) / "b")
            again.start_checkpoint(Path(tmp) / "ck.npz")
            again.advance(0.5)
            self.assertLess(abs(again.tip()["R"] / first["R"] - 1), 1e-7)
            again.release()

    def test_face_layer_refines_the_hole_face(self) -> None:
        from taylor_culick_fem.axisymmetric_viscoelastic import AxisymmetricViscoelasticProblem

        with tempfile.TemporaryDirectory() as tmp:
            problem = AxisymmetricViscoelasticProblem()
            problem.ignore_command_line = True
            problem.set_output_directory(tmp)
            problem.solvent_oh, problem.elastic_modulus = 1.0, 0.1
            problem.X0, problem.window_length = 1.0, 20.0
            problem.resolution, problem.far_resolution = 0.06, 0.25
            problem.face_layer = {"spacing": 2e-3, "growth": 1.2, "thickness": 0.05}
            problem.initialise()
            mesh = problem.get_mesh("liquid")
            shortest = math.inf
            for element in mesh.elements():
                xs = [(element.node_pt(k).x(0), element.node_pt(k).x(1)) for k in range(3)]
                for a in range(3):
                    b = (a + 1) % 3
                    shortest = min(shortest, math.dist(xs[a], xs[b]))
            self.assertLess(shortest, 2.5e-3)
            self.assertGreater(shortest, 1.5e-3)
            problem.run(0.02, startstep=0.01, maxstep=0.01, outstep=False)
            self.assertGreater(problem.window_position(), 1.0)
