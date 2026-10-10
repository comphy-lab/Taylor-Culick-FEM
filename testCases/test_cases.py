"""Every committed case is valid, and the case validator names each defect."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from taylor_culick_fem.cases import SCHEMA, case_problems, validate_case

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


if __name__ == "__main__":
    unittest.main()
