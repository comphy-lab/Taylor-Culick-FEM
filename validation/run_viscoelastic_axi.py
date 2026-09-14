"""Run one bounded axisymmetric viscoelastic Taylor--Culick case.

Time is the physical capillary time used by the FEM problem.  This driver does
not use the viscous ``t*`` clock, which is undefined when the solvent
Ohnesorge number is zero.  ``lambda=inf`` selects the exact non-relaxing
Oldroyd-B/Kelvin--Voigt limit.

The modulus is non-dimensionalised as ``G = G_dim h0 / sigma`` using the full
sheet thickness, consistently with the inherited FEM problem.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "simulationCases"))

from axisymmetric_viscoelastic import AxisymmetricViscoelasticProblem
from planar_comoving import V_TAYLOR_CULICK


def _positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def _nonnegative_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return value


def _relaxation_time(text: str) -> float:
    if text.strip().lower() in {"inf", "+inf", "infinity", "+infinity"}:
        return math.inf
    return _positive_float(text)


def _parameter_label(value: float) -> str:
    return "inf" if math.isinf(value) else f"{value:g}"


def _case_name(oh: float, elastic_modulus: float, relaxation_time: float) -> str:
    return (
        f"axi_viscoelastic_Oh_{oh:g}_G_{elastic_modulus:g}_"
        f"lambda_{_parameter_label(relaxation_time)}"
    )


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _assert_finite(label: str, values) -> None:
    array = np.asarray(list(values), dtype=float)
    if array.size and not np.all(np.isfinite(array)):
        raise RuntimeError(f"non-finite {label}: {array!r}")


def _enforce_launcher_affinity() -> dict[str, object]:
    """Confine auxiliary threads to the launcher's Linux CPU affinity.

    MPI initialisation creates an auxiliary ``fuse`` thread which widens its
    own affinity after inheriting the launcher's mask.  On Linux, restore every
    extant thread to the main thread's mask before any FEM work begins.  On
    other platforms this is a reported no-op; their launchers own affinity.
    """

    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return {"supported": False, "repaired_threads": 0}
    allowed = os.sched_getaffinity(0)
    task_root = Path("/proc/self/task")
    if not task_root.is_dir():
        return {"supported": False, "repaired_threads": 0}
    repaired = 0
    checked = 0
    for task in task_root.iterdir():
        try:
            tid = int(task.name)
            checked += 1
            if os.sched_getaffinity(tid) != allowed:
                os.sched_setaffinity(tid, allowed)
                if os.sched_getaffinity(tid) != allowed:
                    raise RuntimeError(f"thread {tid} rejected the launcher CPU mask")
                repaired += 1
        except ProcessLookupError:
            # A short-lived library thread exited between enumeration and the
            # affinity calls; it cannot escape the later computation.
            continue
    return {
        "supported": True,
        "allowed_cpus": sorted(allowed),
        "checked_threads": checked,
        "repaired_threads": repaired,
    }


def run_case(args: argparse.Namespace) -> dict:
    affinity = _enforce_launcher_affinity()
    run_dir = args.results.resolve() / _case_name(
        args.oh, args.elastic_modulus, args.relaxation_time
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    relaxation_json: float | str = (
        "inf" if math.isinf(args.relaxation_time) else args.relaxation_time
    )
    metadata = {
        "case": "axisymmetric_viscoelastic_comoving",
        "Oh": args.oh,
        "mu_s": math.sqrt(2.0) * args.oh,
        "elastic_modulus": args.elastic_modulus,
        "elastic_modulus_definition": "G_dim*h0/sigma",
        "relaxation_time": relaxation_json,
        "constitutive_model": "Oldroyd-B",
        "polymer_stress": "G*(C-I)",
        "conformation_formulation": "log-conf",
        "stabilization_option": args.stabilization,
        "conformation_stabilization": "SUPG" if args.stabilization == "supg" else None,
        "coordinate_system": "axisymmetric",
        "velocity_frame": "laboratory",
        "time_scale": "capillary",
        "t_end": args.t_end,
        "resolution": args.resolution,
        "far_resolution": args.far_resolution,
        "length": args.length,
        "r0": args.r0,
        "n_outputs": args.n_outputs,
        "fixed_step": args.fixed_step,
        "fixed_step_semantics": "maximum-step cap after smaller BDF startup",
        "max_step": args.max_step,
        "temporal_error": None if args.fixed_step is not None else args.temporal_error,
        "write_states": False,
        "V_TC": V_TAYLOR_CULICK,
        "launcher_affinity": affinity,
        "status": "initialising",
    }
    _write_json(run_dir / "case.json", metadata)

    previous_cwd = Path.cwd()
    os.chdir(run_dir)
    try:
        with AxisymmetricViscoelasticProblem() as problem:
            problem.solvent_oh = args.oh
            problem.elastic_modulus = args.elastic_modulus
            problem.relaxation_time = args.relaxation_time
            problem.conformation_stabilization = "SUPG" if args.stabilization == "supg" else None
            problem.X0 = args.r0
            problem.window_length = args.length
            problem.resolution = args.resolution
            problem.far_resolution = args.far_resolution
            problem.max_newton_iterations = args.max_newton
            problem.initialise()
            # This is a bounded smoke, not a resumable production run.  The
            # pyoomph default writes a full state at every internal step,
            # turning a sub-second check into thousands of redundant files.
            problem.write_states = False

            volume0 = problem.volume()
            radius0 = problem.window_position()
            history = problem.create_text_file_output(
                "control.txt",
                header=[
                    "t",
                    "R",
                    "U",
                    "ustar",
                    "volume",
                    "volume_expected",
                    "volume_rel_error",
                    "tip_pin_residual",
                ],
            )
            out_times = np.linspace(args.t_end / args.n_outputs, args.t_end, args.n_outputs)
            previous_target = 0.0
            worst_volume_error = 0.0
            worst_tip_pin = 0.0
            minimum_speed = math.inf
            last_row: list[float] | None = None
            diagnostics: dict[str, float] = {}

            for target in out_times:
                interval = float(target - previous_target)
                if args.fixed_step is not None:
                    # This is a maximum-step cap, not a promise that every
                    # internal step is identical: BDF starts smaller to build
                    # its history, and output landing can add a short sliver.
                    maxstep = min(args.fixed_step, interval)
                    startstep = min(maxstep, max(interval * 1e-2, 1e-6))
                else:
                    maxstep = (
                        min(args.max_step, interval) if args.max_step is not None else interval
                    )
                    startstep = min(maxstep, max(interval * 1e-2, 1e-6))
                problem.run(
                    float(target),
                    startstep=startstep,
                    maxstep=maxstep,
                    temporal_error=None if args.fixed_step is not None else args.temporal_error,
                )
                now = float(problem.get_current_time(as_float=True))
                radius = problem.window_position()
                speed = problem.retraction_speed()
                volume = problem.volume()
                expected = problem.expected_volume(volume0, radius0)
                volume_error = abs(volume - expected) / max(abs(expected), 1e-300)
                tip_pin = abs(radius - problem.hole_radius())
                last_row = [
                    now,
                    radius,
                    speed,
                    speed / V_TAYLOR_CULICK,
                    volume,
                    expected,
                    volume_error,
                    tip_pin,
                ]
                _assert_finite("control row", last_row)
                history.add_row(*last_row)
                diagnostics = problem.polymer_diagnostic_extrema()
                _assert_finite("polymer diagnostics", diagnostics.values())
                worst_volume_error = max(worst_volume_error, volume_error)
                worst_tip_pin = max(worst_tip_pin, tip_pin)
                minimum_speed = min(minimum_speed, speed)
                previous_target = float(target)

            assert last_row is not None
            radius_final = last_row[1]
            radius_increase = radius_final - radius0
            failures = []
            if radius_increase < args.min_radius_increase:
                failures.append(
                    f"hole growth {radius_increase:.6e} is below {args.min_radius_increase:.6e}"
                )
            # At the far edge u=0 and the mesh moves with +U, so the relative
            # normal velocity is -U.  Psi=0 is an inflow condition only while
            # U>=0; elastic recoil would turn that edge into outflow and make
            # the strong conformation condition invalid.
            if args.max_recoil_speed is not None and minimum_speed < -args.max_recoil_speed:
                failures.append(f"far boundary ceased to be inflow (minimum U={minimum_speed:.6e})")
            if last_row[2] <= args.min_final_speed:
                failures.append(
                    f"final retraction speed {last_row[2]:.6e} is not above "
                    f"{args.min_final_speed:.6e}"
                )
            if worst_volume_error > args.max_volume_error:
                failures.append(
                    f"window-volume relative error {worst_volume_error:.3e} exceeds "
                    f"{args.max_volume_error:.3e}"
                )
            if worst_tip_pin > args.max_tip_pin_residual:
                failures.append(
                    f"tip-pin residual {worst_tip_pin:.3e} exceeds {args.max_tip_pin_residual:.3e}"
                )
            if diagnostics:
                stress_abs_max = max(
                    abs(diagnostics["polymer_stress_eigenvalue_min"]),
                    abs(diagnostics["polymer_stress_eigenvalue_max"]),
                )
                if diagnostics["conformation_eigenvalue_min"] <= 0:
                    failures.append("conformation tensor lost positive definiteness")
                if stress_abs_max < args.min_polymer_stress:
                    failures.append(
                        f"polymer stress {stress_abs_max:.6e} is below "
                        f"{args.min_polymer_stress:.6e}"
                    )

            metadata.update(
                {
                    "status": "failed" if failures else "passed",
                    "R_initial": radius0,
                    "R_final": radius_final,
                    "R_increase": radius_increase,
                    "minimum_U": minimum_speed,
                    "far_boundary_remained_inflow": minimum_speed >= 0,
                    "U_final": last_row[2],
                    "ustar_final": last_row[3],
                    "worst_volume_rel_error": worst_volume_error,
                    "worst_tip_pin_residual": worst_tip_pin,
                    "polymer_diagnostics_available": bool(diagnostics),
                    "polymer_diagnostics_final": diagnostics,
                    "failure_reasons": failures,
                }
            )
    except BaseException as exc:
        metadata["status"] = "error"
        metadata["error_type"] = type(exc).__name__
        metadata["error"] = str(exc)
        _write_json(run_dir / "case.json", metadata)
        raise
    finally:
        os.chdir(previous_cwd)

    _write_json(run_dir / "case.json", metadata)
    if metadata["failure_reasons"]:
        raise RuntimeError("; ".join(metadata["failure_reasons"]))
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oh",
        type=_nonnegative_float,
        default=0.0,
        help="solvent Ohnesorge number; mu_s=sqrt(2)*Oh",
    )
    parser.add_argument(
        "--elastic-modulus",
        type=_nonnegative_float,
        default=1.0,
        help="dimensionless polymer modulus G=G_dim*h0/sigma",
    )
    parser.add_argument(
        "--relaxation-time",
        type=_relaxation_time,
        default=math.inf,
        help="dimensionless lambda, or 'inf' for no relaxation",
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="output root; one canonical case directory is created below it",
    )
    parser.add_argument(
        "--t-end", type=_positive_float, default=0.2, help="end time in capillary units"
    )
    parser.add_argument("--resolution", type=_positive_float, default=0.25)
    parser.add_argument("--far-resolution", type=_positive_float, default=0.5)
    parser.add_argument(
        "--length",
        type=_positive_float,
        default=8.0,
        help="fixed radial length of the co-moving window",
    )
    parser.add_argument("--r0", type=_positive_float, default=1.0, help="initial hole radius")
    parser.add_argument("--n-outputs", type=int, default=5)
    parser.add_argument(
        "--stabilization",
        choices=("supg", "none"),
        default="supg",
        help="conformation stabilisation: SUPG or plain Galerkin",
    )
    parser.add_argument(
        "--fixed-step",
        type=_positive_float,
        default=None,
        help="disable temporal adaptivity and cap maxstep after BDF startup",
    )
    parser.add_argument(
        "--max-step",
        type=_positive_float,
        default=None,
        help="maximum adaptive step (default: each output interval)",
    )
    parser.add_argument("--temporal-error", type=_positive_float, default=1e-3)
    parser.add_argument("--max-newton", type=int, default=20)
    parser.add_argument("--max-volume-error", type=_positive_float, default=5e-3)
    parser.add_argument("--max-tip-pin-residual", type=_positive_float, default=1e-6)
    parser.add_argument("--min-radius-increase", type=_nonnegative_float, default=1e-4)
    parser.add_argument(
        "--max-recoil-speed",
        type=_nonnegative_float,
        default=None,
        help="optional failure threshold for transient inward U; always recorded",
    )
    parser.add_argument("--min-final-speed", type=_nonnegative_float, default=0.0)
    parser.add_argument("--min-polymer-stress", type=_nonnegative_float, default=1e-10)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.n_outputs < 1:
        parser.error("--n-outputs must be at least 1")
    if args.max_newton < 1:
        parser.error("--max-newton must be at least 1")
    if args.length <= args.resolution:
        parser.error("--length must exceed --resolution")
    metadata = run_case(args)
    print(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
