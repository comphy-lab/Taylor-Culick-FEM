"""Every committed case is valid, and the case validator names each defect."""

from __future__ import annotations

import copy
import itertools
import json
import unittest
from pathlib import Path

from taylor_culick_fem.cases import (
    SCHEMA,
    case_problems,
    continuation_case_problems,
    driver_arguments,
    validate_case,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE = ROOT / "simulationCases" / "axisymmetric_viscoelastic.json"
CASE_FOLDERS = ("simulationCases", "validationCases", "verificationCases", "modelComparisonCases")


def committed_cases() -> list[tuple[Path, dict]]:
    """Every ``pyoomph-case-v1`` file in the case-owning folders; other JSON is skipped."""
    cases = []
    for folder in CASE_FOLDERS:
        for path in sorted((ROOT / folder).rglob("*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and value.get("schema") == SCHEMA:
                cases.append((path, value))
    return cases


class CommittedCaseTests(unittest.TestCase):
    def test_default_case_is_committed(self) -> None:
        self.assertTrue(DEFAULT_CASE.is_file())

    def test_every_committed_case_is_valid(self) -> None:
        cases = committed_cases()
        self.assertTrue(cases)
        for path, case in cases:
            with self.subTest(case=str(path.relative_to(ROOT))):
                self.assertEqual(case_problems(case), [])


class ValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = json.loads(DEFAULT_CASE.read_text(encoding="utf-8"))

    def test_each_defect_is_named(self) -> None:
        defects = {
            "schema": lambda c: c.update(schema="other"),
            "case_id": lambda c: c.update(case_id=""),
            "geometry.kind": lambda c: c["geometry"].update(kind="spherical"),
            "geometry.coordinates": lambda c: c["geometry"].update(
                coordinates=list(reversed(c["geometry"]["coordinates"]))),
            "geometry.frame": lambda c: c["geometry"].update(frame="rotating"),
            "nondimensionalization": lambda c: c.update(nondimensionalization={}),
            "physics.Oh": lambda c: c["physics"].update(Oh=-1),
            "physics.relaxation_time": lambda c: c["physics"].update(relaxation_time="never"),
            "physics.far_field": lambda c: c["physics"].update(far_field={"mode": "radial"}),
            "mesh.length": lambda c: c["mesh"].update(length=0),
            "output.write_states": lambda c: c["output"].update(write_states=True),
            "completion": lambda c: c["completion"].update(kind="steady"),
            "case.restart": lambda c: c.update(restart={}),
        }
        for name, spoil in defects.items():
            case = copy.deepcopy(self.case)
            spoil(case)
            with self.subTest(defect=name):
                problems = case_problems(case)
                self.assertTrue(any(name in problem for problem in problems), problems)
                with self.assertRaises(SystemExit):
                    validate_case(case)

    def test_every_problem_is_reported_together(self) -> None:
        problems = case_problems({"schema": "other", "geometry": {}, "physics": {}})
        for field in ("schema", "case_id", "geometry.kind", "geometry.frame",
                      "nondimensionalization", "physics"):
            self.assertTrue(any(field in problem for problem in problems), (field, problems))

    def test_membrane_far_field_needs_active_infinite_relaxation(self) -> None:
        case = copy.deepcopy(self.case)
        case["physics"]["far_field"] = {"mode": "membrane", "extension": 600}
        case["physics"]["relaxation_time"] = 10.0
        self.assertTrue(any("membrane requires active elasticity" in p for p in case_problems(case)))
        case["physics"]["relaxation_time"] = "inf"
        case["physics"]["passive_conformation"] = True
        self.assertTrue(any("membrane requires active elasticity" in p for p in case_problems(case)))

    def test_a_non_object_is_refused(self) -> None:
        self.assertEqual(case_problems([]), ["case must be a JSON object"])



def fast_numerics(case: dict) -> dict:
    """``case`` with every optional continuation numeric set to a valid non-default value."""
    case = copy.deepcopy(case)
    case["physics"]["far_field"] = {
        "mode": "membrane", "extension": 600.0, "interpolation": "linear", "history": "keep",
        "grading": {"near_length": 200.0, "growth": 1.002},
        "auto_extend": {"tail_fraction": 0.2, "factor": 2.0},
    }
    case["physics"]["relaxation_time"] = "inf"
    case["physics"].pop("passive_conformation", None)
    case["solver"].update(interval_start="carry", initial_guess="extrapolate",
                          temporal_error_fields={"X": 1.0, "U": 1.0}, newton_floor_max_dt=0.025)
    return case


class ContinuationNumericsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = fast_numerics(json.loads(DEFAULT_CASE.read_text(encoding="utf-8")))

    def test_the_optional_numerics_are_accepted_and_mapped(self) -> None:
        self.assertEqual(case_problems(self.case), [])
        arguments = driver_arguments(self.case, Path("/tmp/out"))
        pairs = dict(itertools.pairwise(arguments))
        self.assertEqual(pairs["--interval-start"], "carry")
        self.assertEqual(pairs["--initial-guess"], "extrapolate")
        self.assertEqual(json.loads(pairs["--temporal-error-fields"]), {"U": 1.0, "X": 1.0})
        self.assertEqual(pairs["--newton-floor-max-dt"], "0.025")
        self.assertEqual(pairs["--far-field-interpolation"], "linear")
        self.assertEqual(pairs["--far-field-history"], "keep")
        self.assertEqual(json.loads(pairs["--far-field-grading"]), {"growth": 1.002, "near_length": 200.0})
        self.assertEqual(json.loads(pairs["--far-field-auto-extend"]), {"factor": 2.0, "tail_fraction": 0.2})
        # Absent, they take the defaults that predate them.
        plain = json.loads(DEFAULT_CASE.read_text(encoding="utf-8"))
        pairs = dict(itertools.pairwise(driver_arguments(plain, Path("/tmp/out"))))
        self.assertEqual((pairs["--interval-start"], pairs["--initial-guess"]), ("legacy", "previous"))
        self.assertNotIn("--temporal-error-fields", pairs)

    def test_each_numerics_defect_is_named(self) -> None:
        defects = {
            "solver.interval_start": lambda c: c["solver"].update(interval_start="adaptive"),
            "solver.initial_guess": lambda c: c["solver"].update(initial_guess="predictor"),
            "solver.temporal_error_fields": lambda c: c["solver"].update(temporal_error_fields={}),
            "weight X and U equally": lambda c: c["solver"].update(temporal_error_fields={"X": 1, "U": 2}),
            "solver.newton_floor_max_dt": lambda c: c["solver"].update(newton_floor_max_dt=-1),
            "far_field.interpolation": lambda c: c["physics"]["far_field"].update(interpolation="cubic"),
            "far_field.history": lambda c: c["physics"]["far_field"].update(history="drop"),
            "far_field.grading": lambda c: c["physics"]["far_field"].update(
                grading={"near_length": 200, "growth": float("inf")}),
            "far_field.auto_extend": lambda c: c["physics"]["far_field"].update(
                auto_extend=["tail_fraction", "factor"]),
        }
        for name, spoil in defects.items():
            case = copy.deepcopy(self.case)
            spoil(case)
            with self.subTest(defect=name):
                problems = case_problems(case)
                self.assertTrue(any(name in problem for problem in problems), problems)

    def test_a_continuation_may_change_only_numerics(self) -> None:
        parent = json.loads(DEFAULT_CASE.read_text(encoding="utf-8"))
        parent["physics"]["far_field"] = {"mode": "membrane", "extension": 600.0}
        self.assertEqual(continuation_case_problems(parent, parent), ([], []))
        child = fast_numerics(parent)
        child["output"]["cadence"] = 2 * parent["output"]["cadence"]
        child["solver"]["max_step"] = 2 * parent["solver"]["max_step"]
        child["tolerances"]["temporal_error"] = 1e-6
        child["physics"]["far_field"]["extension"] = 1200.0
        problems, changed = continuation_case_problems(child, parent)
        self.assertEqual(problems, [])
        self.assertEqual(changed, sorted([
            "output.cadence", "solver.max_step", "solver.interval_start", "solver.initial_guess",
            "solver.temporal_error_fields", "solver.newton_floor_max_dt", "tolerances.temporal_error",
            "physics.far_field.extension", "physics.far_field.interpolation",
            "physics.far_field.history", "physics.far_field.grading", "physics.far_field.auto_extend"]))
        for path, spoil in {
            "physics.Oh": lambda c: c["physics"].update(Oh=2.0),
            "mesh.resolution": lambda c: c["mesh"].update(resolution=0.03),
            "completion.value": lambda c: c["completion"].update(value=999.0),
            "case_id": lambda c: c.update(case_id="other"),
        }.items():
            spoilt = copy.deepcopy(child)
            spoil(spoilt)
            with self.subTest(change=path):
                problems, _ = continuation_case_problems(spoilt, parent)
                self.assertTrue(any(path in problem for problem in problems), problems)
        shorter = copy.deepcopy(child)
        shorter["physics"]["far_field"]["extension"] = 300.0
        problems, _ = continuation_case_problems(shorter, parent)
        self.assertTrue(any("never shorten" in problem for problem in problems), problems)
        problems, _ = continuation_case_problems(
            {**copy.deepcopy(parent), "completion": {**parent["completion"], "value": 1.0}}, parent)
        self.assertTrue(any("--t-end" in problem for problem in problems), problems)

if __name__ == "__main__":
    unittest.main()
