#!/usr/bin/env python3
"""axisymmetric_lagrangian_retraction: the opening hole in a Kelvin--Voigt sheet, Lagrangian.

With an infinite relaxation time the elastic sheet of the dynamic runs is a Kelvin--Voigt
neo-Hookean solid. This runner solves the same equations (inertia, solvent viscosity
``sqrt(2) Oh``, polymer stress ``Ec (B - I)``, surface tension) in the reference
configuration, so ``B = F F^T`` is exact and no conformation is transported
(``taylor_culick_fem.lagrangian_dynamic``). The mesh is redesigned in the deformed state and
pulled back to the reference body as the hole opens; every remap is recorded.

    python simulationCases/run_axisymmetric_lagrangian.py <case> --out <folder> [--t-end T]

Output: ``run-manifest.json`` first, ``progress.jsonl``, the live metadata ``case.json``,
``control.txt`` (t, R, U, u* = U / V_TC, full rim height H), ``diagnostics.jsonl`` (volume
error, energies, hoop force near the tip), ``remaps.jsonl``, ``retries.jsonl`` (every step whose
Newton solve failed and was retaken in halves, ``solver.max_step_halvings``) and
``checkpoints/``. A continuation (``--restart-from <run>/checkpoints/ckpt_NNNNN.npz``) uses the
same case, writes to its own new folder and may change only ``--t-end`` and
``--newton-tolerance-safety`` (the Newton acceptance as a multiple of the ``eps r^2/dt``
round-off floor, in place of ``solver.newton_tolerance_safety``). A case may seed the run at
rest from a static equilibrium on its own grids (``seed``). ``--check`` stops after the checks
that need neither pyoomph nor the solver.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from taylor_culick_fem.cases import (
    LAGRANGIAN,
    LAGRANGIAN_FIELDS_USED,
    MAX_STEP_HALVINGS,
    validate_case,
)
from taylor_culick_fem.run_contract import (
    add_contract_options,
    append_progress,
    check_restart,
    continuation_problems,
    enforce_launcher_affinity,
    output_problems,
    read_case,
    sha256,
    start_run,
    write_json_atomic,
)

CASE_FIELDS_USED = LAGRANGIAN_FIELDS_USED
UNITS = ("lengths on the full sheet thickness h0, time on sqrt(rho h0^3/sigma), velocity on "
         "sqrt(sigma/(rho h0)); Ec = G h0/sigma; coded viscosity sqrt(2) Oh")
V_TC = math.sqrt(2.0)


def build_parser() -> argparse.ArgumentParser:
    parser = add_contract_options(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    parser.add_argument("--t-end", type=float, default=None, help="replace completion.value")
    parser.add_argument("--newton-tolerance-safety", type=float, default=None,
                        help="accept a Newton solve below this multiple of the eps*r^2/dt "
                             "round-off floor, in place of solver.newton_tolerance_safety "
                             "(at least 1)")
    return parser


#: Options a continuation may change from its parent run's.
CONTINUATION_OPTIONS = ("t_end", "newton_tolerance_safety")


def option_problems(args: argparse.Namespace) -> list[str]:
    problems = []
    if args.t_end is not None and (not math.isfinite(args.t_end) or args.t_end <= 0):
        problems.append("--t-end must be finite and positive")
    safety = args.newton_tolerance_safety
    if safety is not None and (not math.isfinite(safety) or safety < 1):
        problems.append("--newton-tolerance-safety must be finite and at least 1")
    return problems


def advance_logging_retries(run, target: float, stream) -> None:
    """``run.advance(target)``, writing every retry it records to ``stream`` (JSON Lines).

    The records are written even when the interval fails, so a step that fails after its last
    halving leaves its retry history; the failure is then re-raised unchanged.
    """
    logged = len(run.retries)
    try:
        run.advance(target)
    finally:
        for record in run.retries[logged:]:
            stream.write(json.dumps(record) + "\n")
        if len(run.retries) > logged:
            stream.flush()


def settings_from_case(case: dict, newton_tolerance_safety: float | None = None):
    """The solver settings of ``case``; a given ``newton_tolerance_safety`` replaces the case's."""
    from taylor_culick_fem.lagrangian_dynamic import LagrangianSettings

    physics, mesh, remesh, solver = case["physics"], case["mesh"], case["remesh"], case["solver"]
    linear = solver.get("linear_solver", "default")
    return LagrangianSettings(
        oh=float(physics["Oh"]), elastic_modulus=float(physics["elastic_modulus"]),
        r0=float(physics["r0"]), surface_tension=float(physics["surface_tension"]),
        far_edge=str(physics.get("far_edge", "robin")),
        elements_per_quarter=int(mesh["elements_per_quarter"]), length=float(mesh["length"]),
        growth=float(mesh["growth"]), max_spacing=float(mesh["max_spacing"]),
        sweep_spacing=float(mesh["sweep_spacing"]), sweep_ahead=float(mesh["sweep_ahead"]),
        sweep_ahead_fraction=float(mesh["sweep_ahead_fraction"]),
        columns=str(mesh.get("columns", "fan")), shoulder_angle=float(mesh.get("shoulder_angle", 10.0)),
        blend=float(mesh.get("blend", 4.0)),
        remesh_log_radius_growth=float(remesh["log_radius_growth"]),
        remesh_front_margin=float(remesh["front_margin"]),
        dt=float(solver["dt"]), max_newton=int(solver["max_newton"]),
        newton_tolerance=float(solver["newton_tolerance"]),
        newton_tolerance_safety=float(solver["newton_tolerance_safety"]
                                      if newton_tolerance_safety is None else newton_tolerance_safety),
        linear_solver=None if linear == "default" else linear,
        max_step_halvings=int(solver.get("max_step_halvings", MAX_STEP_HALVINGS)),
    )


def _seed(case: dict) -> tuple[dict, object]:
    """Grids and static state of the case's seed, after checking their hashes and Ec."""
    import numpy as np

    from taylor_culick_fem.equilibrium_geometry import FieldMap

    seed = case["seed"]
    for key in ("state", "grids"):
        if sha256(Path(seed[key])) != seed[f"{key}_sha256"]:
            raise SystemExit(f"refused: seed.{key} does not match seed.{key}_sha256")
    saved = np.load(seed["state"])
    ec = math.exp(float(saved["log_ec"]))
    if not math.isclose(ec, case["physics"]["elastic_modulus"], rel_tol=1e-12):
        raise SystemExit(f"refused: the seed is the equilibrium at Ec = {ec!r}, not the case's "
                         f"{case['physics']['elastic_modulus']!r}")
    if float(saved["cap_beta"]) if "cap_beta" in saved.files else 0.0:
        raise SystemExit("refused: the seed carries cap stiffening the transient solver omits")
    grids = dict(np.load(seed["grids"]))
    problems = seed_domain_problems(grids, saved["Xe"], float(case["physics"]["r0"]),
                                    float(case["mesh"]["length"]), saved["ij"])
    if problems:
        raise SystemExit("refused: " + "; ".join(problems))
    return grids, FieldMap(saved["Xe"], saved["ue"], saved["ij"])


def seed_domain_problems(grids: dict, state_nodes, r0: float, length: float,
                         state_ij=None) -> list[str]:
    """The seed's reference body must be the case's: tip at ``r0``, far end at ``length``.

    Both the grids and the state's element nodes are checked against the body, and the
    state's elements against the grids, so a state from another mesh of the same body is
    caught as well.
    """
    import numpy as np

    problems = []
    for name, points in (("seed.grids", np.concatenate([np.reshape(g, (-1, 2)) for g in grids.values()])),
                         ("seed.state", np.reshape(np.asarray(state_nodes, float), (-1, 2)))):
        low, high = float(points[:, 0].min()), float(points[:, 0].max())
        if not math.isclose(low, r0, rel_tol=1e-12, abs_tol=1e-12):
            problems.append(f"{name} starts at reference radius {low!r}, not physics.r0 = {r0!r}")
        if not math.isclose(high, length, rel_tol=1e-12):
            problems.append(f"{name} ends at reference radius {high!r}, not mesh.length = {length!r}")
    # Same bounds are not the same mesh: the state's elements must be exactly the grids'
    # (two grid intervals per element side). Element corners are grid points; edge nodes on
    # the curved cap are moved onto the arc, so only the corners are compared. Shared block
    # nodes are merged on coordinates rounded to 11 decimals, as ReferenceSheetMesh merges
    # them, so both sets are keyed the same way.
    blocks = [np.asarray(g, float) for g in grids.values()]
    expected = sum(((g.shape[0] - 1) // 2) * ((g.shape[1] - 1) // 2) for g in blocks)
    elements = np.reshape(np.asarray(state_nodes, float), (-1, 9, 2))
    local = np.reshape(np.asarray(state_ij), (-1, 9, 2)) if state_ij is not None \
        else np.tile([(i, j) for j in range(3) for i in range(3)], (len(elements), 1, 1))
    corner = np.isin(local[..., 0], (0, 2)) & np.isin(local[..., 1], (0, 2))
    grid_corners = np.unique(np.round(np.concatenate([g[::2, ::2].reshape(-1, 2) for g in blocks]), 11),
                             axis=0)
    state_corners = np.unique(np.round(elements[corner], 11), axis=0)
    if len(elements) != expected or not np.array_equal(grid_corners, state_corners):
        problems.append(f"seed.state has {len(elements)} elements on {len(state_corners)} corner nodes, "
                        f"but seed.grids define {expected} on {len(grid_corners)}: a state from "
                        "another mesh")
    return problems


def cadence_index(time_value: float, cadence: float) -> int:
    """Index of the last cadence boundary at or before ``time_value`` (within round-off)."""
    ratio = time_value / cadence
    nearest = round(ratio)
    return nearest if abs(ratio - nearest) <= 1e-6 else math.floor(ratio)


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw)
    case_path = args.case.resolve()
    case = read_case(case_path)
    validate_case(case, problem=LAGRANGIAN)
    problems = option_problems(args) + output_problems(args.out)
    if problems:
        raise SystemExit("refused: " + "; ".join(problems))
    case_sha256 = sha256(case_path)
    restart = None
    if args.restart_from is not None:
        if args.restart_from.suffix != ".npz" or args.restart_from.parent.name != "checkpoints":
            raise SystemExit("refused: --restart-from must name <run>/checkpoints/ckpt_NNNNN.npz")
        parent, restart = check_restart(args.restart_from, args.out, case_sha256)
        args.parent_run = parent
        changed = continuation_problems(parser, args, parent, CONTINUATION_OPTIONS)
        if changed:
            raise SystemExit("refused: a continuation may change only --t-end and "
                             "--newton-tolerance-safety: " + "; ".join(changed))
        import numpy as np

        with np.load(args.restart_from) as saved:
            restart_time = float(saved["time"])
        end = args.t_end if args.t_end is not None else float(case["completion"]["value"])
        if end <= restart_time + 1e-9 * max(1.0, end):
            raise SystemExit(f"refused: the checkpoint is at t = {restart_time!r}, not before the "
                             f"end time {end!r}; a continuation must advance")
    if args.check:
        print(json.dumps({"check": "passed", "case_id": case["case_id"], "case_sha256": case_sha256}))
        return 0
    launch_cpus = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
    try:
        import pyoomph
    except ImportError as exc:
        raise SystemExit("refused: pyoomph is not importable; use the pinned environment "
                         "(pyproject.toml, uv.lock)") from exc
    from taylor_culick_fem.lagrangian_dynamic import (
        LagrangianRun,
        StateMap,
        hoop_bands,
        state_integrals,
    )
    affinity = enforce_launcher_affinity(launch_cpus)

    settings = settings_from_case(case, args.newton_tolerance_safety)
    t_end = args.t_end if args.t_end is not None else float(case["completion"]["value"])
    out = args.out.resolve()
    commit = start_run(out, root=ROOT, runner=Path(__file__), argv=raw, case_path=case_path,
                       case=case, case_sha256=case_sha256, case_fields_used=CASE_FIELDS_USED,
                       units=UNITS, restart=restart, pyoomph_module=getattr(pyoomph, "__file__", None))
    (out / "input-case.json").write_bytes(case_path.read_bytes())
    append_progress(out, "started", case_id=case["case_id"], component_commit=commit,
                    launcher_affinity=affinity)
    run = LagrangianRun(settings, out / "pyoomph")
    cadence = float(case["output"]["cadence"])
    every = int(case["output"]["checkpoint_every"])
    diagnostics_every = int(case["output"]["diagnostics_every"])
    status = "running"
    meta = {"case_id": case["case_id"], "status": status, "t_end": t_end, "V_TC": V_TC,
            "viscosity": math.sqrt(2.0) * settings.oh, "component_commit": commit,
            "newton_tolerance_safety": settings.newton_tolerance_safety,
            "max_step_halvings": settings.max_step_halvings}
    started = time.time()
    try:
        if restart is not None:
            info = run.start_checkpoint(args.restart_from.resolve())
            # The next output is the next cadence boundary after the checkpoint time, even when
            # the parent ended on a shortened interval between boundaries.
            index = cadence_index(float(info["time"]), cadence)
            meta["restart"] = {"time": info["time"], "output_index": index,
                               "pinned_mismatch": info["pinned_mismatch"]}
        elif "seed" in case:
            grids, static = _seed(case)
            info = run.start_static_seed(grids, static)
            index = 0
            meta["seed"] = {"pinned_mismatch": info["pinned_mismatch"]}
        else:
            run.start_flat()
            index = 0
        meta["ndof_start"] = int(run.problem.ndof())
        write_json_atomic(out / "case.json", meta)
        (out / "checkpoints").mkdir()
        control = (out / "control.txt").open("a", encoding="utf-8")
        control.write("# t R U ustar H wall_seconds\n")
        diagnostics = (out / "diagnostics.jsonl").open("a", encoding="utf-8")
        remaps = (out / "remaps.jsonl").open("a", encoding="utf-8")
        retries = (out / "retries.jsonl").open("a", encoding="utf-8")

        def record_diagnostics(now: float, tip: dict) -> None:
            state = StateMap.from_problem(run.problem)
            mesh = run.problem.get_mesh("solid/interface")
            stretch = max((nd.x(0) + nd.value(run.problem.nodal_fields()["displacement_x"])) / nd.x(0)
                          for nd in mesh.nodes())
            row = {"t": now, "R": tip["R"], "tip_R0_effective": tip["R"] / stretch,
                   **state_integrals(state, settings.elastic_modulus),
                   "hoop_bands": hoop_bands(state, settings.elastic_modulus, tip["R"]),
                   "ndof": int(run.problem.ndof()), "mesh": run.mesh_index,
                   "wall_seconds": time.time() - started}
            diagnostics.write(json.dumps(row) + "\n")
            diagnostics.flush()

        now = run.time()
        tip = run.tip()
        if restart is None:
            control.write(f"{now:.10g} {tip['R']:.15g} {tip['U']:.15g} {tip['U'] / V_TC:.15g} "
                          f"{run.problem.rim_height():.15g} 0\n")
            record_diagnostics(now, tip)
        while now < t_end - 1e-9 * max(1.0, t_end):
            index += 1
            target = min(round(index * cadence, 12), t_end)
            advance_logging_retries(run, target, retries)
            now = run.time()
            tip = run.tip()
            height = run.problem.rim_height()
            values = [now, tip["R"], tip["U"], tip["U"] / V_TC, height]
            if not all(math.isfinite(v) for v in values):
                raise RuntimeError(f"non-finite control row at t = {now}")
            control.write(" ".join(f"{v:.15g}" for v in values) + f" {time.time() - started:.1f}\n")
            control.flush()
            if diagnostics_every and index % diagnostics_every == 0:
                record_diagnostics(now, tip)
            reason = run.remesh_reason()
            if reason is not None:
                remaps.write(json.dumps(run.remesh(reason)) + "\n")
                remaps.flush()
            if every and index % every == 0:
                path = out / "checkpoints" / f"ckpt_{index:05d}.npz"
                run.checkpoint(path, output_index=index)
                append_progress(out, "checkpoint", path=str(path.relative_to(out)), physical_time=now,
                                R=tip["R"], U=tip["U"])
        status = "passed"
        path = out / "checkpoints" / f"ckpt_{index:05d}.npz"
        if not path.exists():
            run.checkpoint(path, output_index=index)
    except BaseException as exc:
        meta["status"] = "failed"
        write_json_atomic(out / "case.json", meta)
        append_progress(out, "failed", error_type=type(exc).__name__, error=str(exc)[:500])
        raise
    meta.update(status=status, t=now, R=tip["R"], U=tip["U"], remaps=len(run.remaps),
                retries=len(run.retries), wall_seconds=time.time() - started)
    write_json_atomic(out / "case.json", meta)
    append_progress(out, "completed", status=status, physical_time=now, primary_observable=tip["R"])
    print(json.dumps(meta, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
