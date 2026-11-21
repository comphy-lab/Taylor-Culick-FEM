"""Bounded callback overhead on the existing small membrane-coupled FEM fixture.

Run this test on an otherwise idle, explicitly pinned core.  It times only
Newton callbacks and iterations after the JIT-compiled problem has started;
it does not include compilation or claim to predict production-run overhead.
"""

from __future__ import annotations

import json
from time import perf_counter_ns

import numpy as np

from testCases.test_viscoelastic_case_adapter import _case, adapter


def test_observer_newton_iteration_overhead(tmp_path, monkeypatch):
    """Measure inclusive observer hooks against their actual native iterations.

    Timed superclass calls separate existing solver-hook work from the extra
    observer adapter.  An iteration runs from its before-step callback through
    the next before-step callback, or through after-solve for the last update.
    This includes the native post-update residual assembly.  The observer's
    per-solve setup and finalisation are charged to the first and last updates,
    respectively, rather than disappearing from the result.

    This is a deliberately conservative small-mesh screen.  Scheduling noise,
    tracing wrappers and timer calls are included in the measurements.  A pass
    is not evidence of the overhead on the four production meshes.
    """
    from taylor_culick_fem import viscoelastic_driver as driver
    from taylor_culick_fem.axisymmetric_viscoelastic import AxisymmetricViscoelasticProblem
    from taylor_culick_fem.newton_observer_problem import ObservedAxisymmetricViscoelasticProblem

    base = AxisymmetricViscoelasticProblem
    observed = ObservedAxisymmetricViscoelasticProblem
    phases = (
        "actions_before_newton_solve",
        "actions_before_newton_step",
        "actions_after_newton_step",
        "actions_after_newton_solve",
        "actions_after_transient_solve",
    )
    # If row sampling is added through this callback, time its residual
    # assembly too.  Otherwise the inherited native hook is not observer work.
    if "actions_before_newton_convergence_check" in observed.__dict__:
        phases += ("actions_before_newton_convergence_check",)
    native_ns = {phase: 0 for phase in phases}
    samples = []
    pending = None
    setup_ns = 0
    solve_id = 0
    disabled = []

    def native_wrapper(phase, original):
        def wrapped(self, *args, **kwargs):
            start = perf_counter_ns()
            try:
                return original(self, *args, **kwargs)
            finally:
                native_ns[phase] += perf_counter_ns() - start
        return wrapped

    def observed_wrapper(phase, original):
        def wrapped(self, *args, **kwargs):
            nonlocal pending, setup_ns, solve_id
            start = perf_counter_ns()
            if phase == "actions_before_newton_step":
                if pending is not None:
                    pending["iteration_ns"] = (start - pending.pop("start_ns")
                                               + pending.pop("setup_ns"))
                    samples.append(pending)
                observer = self._newton_observer
                pending = {
                    "solve": solve_id,
                    "details": bool(observer is not None and observer.wants_details),
                    "start_ns": start,
                    "observer_ns": setup_ns,
                    "setup_ns": setup_ns,
                }
                setup_ns = 0
            before_native = native_ns[phase]
            try:
                return original(self, *args, **kwargs)
            finally:
                end = perf_counter_ns()
                # The timer-wrapper cost is included.  Clamp only sub-clock
                # rounding, and never let a negative value improve the gate.
                extra = max(0, end - start - (native_ns[phase] - before_native))
                if phase == "actions_before_newton_solve":
                    solve_id += 1
                    setup_ns = extra
                elif phase in ("actions_before_newton_step", "actions_after_newton_step",
                                "actions_before_newton_convergence_check"):
                    if pending is not None:
                        pending["observer_ns"] += extra
                elif phase == "actions_after_newton_solve" and pending is not None:
                    pending["observer_ns"] += extra
                    pending["iteration_ns"] = (end - pending.pop("start_ns")
                                               + pending.pop("setup_ns"))
                    samples.append(pending)
                    pending = None
                elif phase == "actions_after_transient_solve" and samples:
                    if samples[-1]["solve"] == solve_id:
                        samples[-1]["observer_ns"] += extra
                        samples[-1]["iteration_ns"] += extra
                observer = self._newton_observer
                if observer is None or not observer.enabled:
                    disabled.append(phase)
        return wrapped

    # Both wrappers retain the actual callbacks and execute each exactly once.
    # Resolve originals before patching either class: the observed class calls
    # its superclass with super(), which then reaches the timed native method.
    originals = {phase: (getattr(base, phase), getattr(observed, phase)) for phase in phases}
    for phase, (native, instrumented) in originals.items():
        monkeypatch.setattr(base, phase, native_wrapper(phase, native))
        monkeypatch.setattr(observed, phase, observed_wrapper(phase, instrumented))

    case = _case({"kind": "time", "value": 0.08, "allow_recoil": True})
    case["mesh"].update(length=3.0, resolution=0.5, far_resolution=0.5)
    case["output"].update(cadence=0.02, checkpoint_every=1)
    case["tolerances"]["temporal_error"] = 1.0
    case["solver"].update(initial_guess="extrapolate", interval_start="carry",
                          temporal_error_fields={"X": 1, "U": 1})
    case["physics"]["far_field"] = {
        "mode": "membrane", "extension": 20.0, "node_spacing": 0.25,
        "interpolation": "linear", "history": "keep",
    }
    case_path = tmp_path / "input.json"
    case_path.write_text(json.dumps(case))
    result = driver.run_case(adapter.case_to_driver_args(
        case_path, tmp_path / "run", newton_observer=True
    ))
    assert result["status"] == "passed"
    assert not disabled, f"observer disabled during timing: {disabled}"
    root = tmp_path / "run" / "axi_viscoelastic_Oh_1_G_1_lambda_inf"
    log = (root / "run_viscoelastic_axi" / "_pyoomph_logfile.txt").read_text()
    assert "TIMESTEP REJECTED" not in log, "retry/rollback timing requires its own accounting"
    assert pending is None and len(samples) >= 8, "insufficient complete iteration samples"
    assert any(row["details"] for row in samples), "no fully sampled solve was measured"

    # Only callback time is charged; native work in the same hooks is removed.
    # Setup is measured outside the iteration and added to both numerator and
    # denominator; finalisation is charged the same way above.
    for row in samples:
        assert row["iteration_ns"] > 0
        assert 0 <= row["observer_ns"] <= row["iteration_ns"]
    ratios = np.asarray([row["observer_ns"] / row["iteration_ns"] for row in samples])
    report = {
        "iteration_count": len(samples),
        "detailed_iteration_count": sum(row["details"] for row in samples),
        "median_fraction": float(np.median(ratios)),
        "p90_fraction": float(np.quantile(ratios, 0.9)),
        "max_fraction": float(np.max(ratios)),
        "aggregate_fraction": sum(row["observer_ns"] for row in samples)
        / sum(row["iteration_ns"] for row in samples),
        "limit_per_iteration": 0.05,
        "samples": samples,
    }
    (tmp_path / "newton-observer-overhead.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Newton observer overhead:", json.dumps({k: v for k, v in report.items() if k != "samples"}))
    assert report["max_fraction"] <= report["limit_per_iteration"], report
