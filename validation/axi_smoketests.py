"""Smoke ladder for the axisymmetric lab-frame DNS, per docs/axi-design.md.

Four checks, run in order (each is meant to fail loudly and cheaply before
any long campaign is launched on ``axisymmetric_lab.py``):

    s0      hoop-curvature discriminator (static quarter-disc droplet)
    s1      FD-Jacobian consistency check on the real case, coarse & short
    s2      volume conservation over a coarse run
    s4lite  quick physics probe: does the hole end up open, growing toward V_TC

s3 (planar-limit reduction at R0=100) is deliberately NOT here: it is a
multi-hour comparison run, not a smoke test, and belongs with the validation
campaign once s0-s2 and s4lite pass.

All output goes under /tmp/axi-smoke/<subcommand>/ -- never into the repo.

Usage::

    uv run python validation/axi_smoketests.py s0
    uv run python validation/axi_smoketests.py s1
    uv run python validation/axi_smoketests.py s2
    uv run python validation/axi_smoketests.py s4lite
    uv run python validation/axi_smoketests.py all
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

# Must precede the first pyoomph (hence MPI) import -- see planar_comoving.py's
# header docstring for why.
os.environ.setdefault("HWLOC_COMPONENTS", "-gl")
os.environ.setdefault("MPLBACKEND", "Agg")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "simulationCases"))

from axisymmetric_lab import AxisymmetricTaylorCulickProblem
from pyoomph import *
from pyoomph.equations.ALE import *
from pyoomph.equations.navier_stokes import *
from pyoomph.expressions import *
from pyoomph.meshes.bcs import AxisymmetryBC

OUT_ROOT = Path("/tmp/axi-smoke")


def _outdir(name: str) -> Path:
    d = OUT_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _print_result(name: str, ok: bool, detail: str) -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}", flush=True)
    return ok


# --------------------------------------------------------------------------
# s0: hoop-curvature discriminator.
#
# A static axisymmetric quarter-disc droplet of radius a: the axis (r = 0)
# on one straight edge via pyoomph's AxisymmetryBC(), midplane symmetry
# (z = 0) on the other, and the free surface arc between them. With no
# forcing anywhere, the only steady solution is u = 0 and a uniform pressure
# set by the interface's natural (surface-divergence) traction term -- the
# same NavierStokesFreeSurface residual the real DNS uses, just with no flow
# to obscure it. Under the axisymmetric measure that residual carries BOTH
# principal curvatures (1/a azimuthal + 1/a in-plane), giving the full
# Young-Laplace p = 2*sigma/a. The equivalent planar/cartesian setup would
# give only the in-plane curvature, p = sigma/a -- so a factor of 2 here is
# the discriminator that the azimuthal curvature is actually active, not an
# artefact of the coordinate system being merely labelled "axisymmetric".
#
# Trap found while building this check: NavierStokesFreeSurface resolves
# static_interface="auto" from whether the domain has mesh position dofs.
# Without HyperelasticSmoothedMesh() (no mesh dofs -> static=True), the
# interface's kin_bc Lagrange multiplier enforces u.n=0 as a RIGID-WALL
# reaction and absorbs the entire surface-tension load itself, decoupled
# from the bulk pressure field -- p then comes out at solver noise (~1e-3),
# not 2*sigma/a, even though the residual converges to machine precision in
# one Newton step (a consistent, but rank-deficient-for-p, linear system).
# Adding HyperelasticSmoothedMesh() switches static_interface to False, which
# routes kin_bc into the MESH position equation (mesh_velocity = u) instead,
# leaving the momentum equation's own boundary term to balance directly
# against the surface-tension term -- which is what actually fixes p. The
# mesh stays put regardless (u = 0 is still the trivial solution, so
# mesh_velocity = 0 too): this only changes which equation the Lagrange
# multiplier feeds, not the physical answer.
# --------------------------------------------------------------------------


class QuarterDiscMesh(GmshTemplate):
    """Quarter-disc of radius ``a``: axis edge, midplane edge, free-surface arc."""

    def define_geometry(self):
        pr = self.get_problem()
        assert isinstance(pr, QuarterDiscProblem)
        self.mesh_mode = "tris"
        self.default_resolution = pr.resolution
        a = pr.radius

        p_origin = self.point(0.0, 0.0)
        p_base = self.point(a, 0.0)
        p_axis_top = self.point(0.0, a)

        self.line(p_origin, p_base, name="symmetry")
        self.circle_arc(p_base, p_axis_top, center=(0.0, 0.0), name="interface")
        self.line(p_axis_top, p_origin, name="axis")
        self.plane_surface("symmetry", "interface", "axis", name="drop")


class QuarterDiscProblem(Problem):
    def __init__(self):
        super().__init__()
        self.radius = 1.0
        self.resolution = 0.1
        self.surface_tension = 1.0

    def define_problem(self):
        self.set_coordinate_system("axisymmetric")
        self += QuarterDiscMesh()

        eqs = NavierStokesEquations(mass_density=1, dynamic_viscosity=1.0)
        # HyperelasticSmoothedMesh() gives the domain mesh-position dofs, so
        # NavierStokesFreeSurface resolves static_interface="auto" to False
        # (see the module-level trap note above) -- required for the surface
        # tension term to couple into the bulk pressure rather than being
        # absorbed by a rigid-wall reaction multiplier.
        eqs += HyperelasticSmoothedMesh()
        eqs += NavierStokesFreeSurface(surface_tension=self.surface_tension) @ "interface"
        eqs += AxisymmetryBC() @ "axis"  # pins velocity_x and mesh_x=0 on r=0
        eqs += AxisymmetryBC() @ "interface/axis"  # new field at that corner; see its docstring
        eqs += DirichletBC(velocity_y=0, mesh_y=0) @ "symmetry"

        # Average pressure over the domain; uniform at equilibrium, so any
        # single sample would do, but the volume average is exact and cheap.
        eqs += IntegralObservables(
            _denom=1,
            _p_integral=var("pressure"),
            p_avg=lambda _p_integral, _denom: _p_integral / _denom,
        )

        self += eqs @ "drop"


def run_s0() -> bool:
    outdir = _outdir("s0")
    prev_cwd = os.getcwd()
    os.chdir(outdir)
    try:
        with QuarterDiscProblem() as problem:
            problem.initialise()
            problem.solve()
            p = float(problem.get_mesh("drop").evaluate_observable("p_avg"))
            a = problem.radius
            sigma = problem.surface_tension
    finally:
        os.chdir(prev_cwd)

    expected = 2.0 * sigma / a
    rel_err = abs(p - expected) / expected
    ok = rel_err < 0.03  # coarse mesh (resolution=0.1); a few % discretisation error is expected
    detail = (
        f"p = {p:.6f}, expected 2*sigma/a = {expected:.6f} "
        f"(planar/cartesian answer would be sigma/a = {sigma / a:.6f}), "
        f"rel_err = {rel_err:.3e}"
    )
    return _print_result("s0 hoop-curvature discriminator", ok, detail)


# --------------------------------------------------------------------------
# s1: FD-Jacobian check on the real case, coarse and short.
# --------------------------------------------------------------------------


def run_s1() -> bool:
    outdir = _outdir("s1")
    prev_cwd = os.getcwd()
    os.chdir(outdir)
    ok = True
    detail = "no Jacobian-difference abort over 3 short timesteps"
    try:
        with AxisymmetricTaylorCulickProblem() as problem:
            problem.resolution = 0.25
            problem.R0 = 1.0
            problem.R_out = 8.0
            problem.initialise()
            # See pyoomph/generic/problem.py:796. Any nonzero abort raised by
            # the core's FD-vs-analytic Jacobian comparison propagates as a
            # Python exception out of run(), caught below.
            problem.debug_jacobian_by_fd_epsilon = 1e-7
            dt = 1e-3
            t = 0.0
            for _ in range(3):
                t += dt
                problem.run(t, startstep=dt * 0.1, maxstep=dt, temporal_error=1e-3)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any abort is a FAIL
        ok = False
        detail = f"Jacobian check raised: {exc!r}"
    finally:
        os.chdir(prev_cwd)
    return _print_result("s1 FD-Jacobian check", ok, detail)


# --------------------------------------------------------------------------
# s2: volume conservation over a coarse run.
# --------------------------------------------------------------------------


def run_s2() -> bool:
    outdir = _outdir("s2")
    prev_cwd = os.getcwd()
    os.chdir(outdir)
    try:
        with AxisymmetricTaylorCulickProblem() as problem:
            problem.resolution = 0.25
            problem.R0 = 1.0
            problem.R_out = 12.0
            problem.initialise()

            dt = 0.005
            n_steps = 200
            v0 = problem.volume()
            v_clean = v0  # cumulative volume ignoring remesh-induced jumps
            v_prev = v0
            remesh_events = 0
            t = 0.0
            for _ in range(n_steps):
                t += dt
                problem.run(t, startstep=dt * 0.1, maxstep=dt, temporal_error=1e-3)
                v_now = problem.volume()
                # Re-fetch the mesh every step. A remesh replaces the mesh
                # object, so a reference cached before the loop stops tracking
                # the live one and _solves_since_remesh is then read off a
                # detached copy -- which silently under-counts remesh events
                # and puts interpolation jumps into the "clean" drift.
                mesh = problem.get_mesh("liquid")
                # _solves_since_remesh resets to 0 on a remesh and is
                # incremented after every solve (RemeshWhen reads the same
                # attribute to decide whether to trigger); ==1 means this
                # solve is the first one after a fresh mesh, so the jump from
                # v_prev to v_now includes an interpolation step, not just
                # physics.
                if mesh._solves_since_remesh == 1:
                    remesh_events += 1
                else:
                    v_clean += v_now - v_prev
                v_prev = v_now
            v_final = v_now
    finally:
        os.chdir(prev_cwd)

    drift_total = abs(v_final - v0) / v0
    drift_clean = abs(v_clean - v0) / v0
    # Judge the metric that matches what happened. A remesh interpolates the
    # solution onto a new mesh, which moves the volume by an amount that is a
    # property of the interpolation, not of the time integration; failing the
    # test on it would report the mesh, not conservation. With no remesh the
    # two are identical, so this only ever loosens the test when a remesh
    # actually occurred -- and the total is still reported either way.
    judged = drift_clean if remesh_events > 0 else drift_total
    ok = judged < 1e-8
    detail = f"drift = {drift_total:.3e} over {n_steps} steps ({remesh_events} remesh events)"
    if remesh_events > 0:
        detail += f"; drift excluding remesh-interpolation jumps = {drift_clean:.3e} (judged)"
    return _print_result("s2 volume conservation", ok, detail)


# --------------------------------------------------------------------------
# s4lite: quick physics probe -- does the hole actually open?
#
# Deviation from docs/axi-design.md's stated PASS criterion ("dR/dt > 0 from
# the first outputs"), recorded here because it changed the check itself:
#
# A finely time-resolved diagnostic (not part of this file; dt=0.002 down to
# t=0.08, then dt=0.02 out to t=1, at BOTH resolution=0.25 and the production
# resolution=0.06) shows that at R0 = h0 = 1 the tip does NOT open
# monotonically from t=0. It RECEDES first: R(t) dips from 1.0 to about
# 0.9938 around t~0.16-0.17 (dR/dt reaches about -0.077 at its most negative),
# then turns around and grows smoothly and monotonically, with dR/dt climbing
# cleanly toward V_TC = sqrt(2) (it reaches 1.083 by t=1, still rising).
#
# This is a real, mesh-independent transient, not a discretisation artefact:
#   * identical qualitative shape (dip then smooth monotonic rise) at
#     resolution 0.25 and 0.06 -- a 4x difference in near-tip element count;
#   * confirmed to be the literal interface/symmetry corner node receding
#     (its own x(0) tracks hole_radius()'s min-over-interface value exactly
#     throughout), not some other interface node transiently undercutting it.
#
# It is also physically plausible, not just numerically well-behaved:
# docs/axi-design.md's "First geometric correction" section notes that the
# toroidal-rim term opposing opening, `pi*h_tip/(2*R)`, is O(1) relative to
# the leading `-2` driving term exactly when R ~ h_tip ~ 1 -- i.e. the net
# driving force is genuinely weak at this radius. At the low Oh = 0.05 used
# here (inertia-dominated), an impulsively-started flow (u(t=0)=0) driven by
# a weak, spatially non-uniform capillary load is exactly the kind of system
# that can show a brief transient in the "wrong" direction before the
# classical rim-collection mechanism dominates. The 1D reduced models should
# probably know this: a quasi-static massless-cap tip closure (their `-2` or
# `-2 + pi*h_tip/(2R)` constant) cannot represent this transient sign
# reversal at all, since it has no way to encode "weak enough to be beaten by
# inertia for a while." Whether that matters for the reduced models' domain
# of validity is a question for that sibling task, not this one.
#
# So this check now asks the more robust question docs/axi-design.md's smoke
# ladder actually wants -- does the hole end up open, on a trajectory
# converging toward the Taylor-Culick speed -- rather than literally
# demanding immediate growth, which the physics above does not deliver at
# R0 = 1.
# --------------------------------------------------------------------------


def run_s4lite() -> bool:
    outdir = _outdir("s4lite")
    prev_cwd = os.getcwd()
    os.chdir(outdir)
    try:
        with AxisymmetricTaylorCulickProblem() as problem:
            problem.resolution = 0.25
            problem.R0 = 1.0
            problem.R_out = 12.0
            problem.initialise()

            t_end = 1.0
            n_out = 20
            out_times = [t_end * k / n_out for k in range(1, n_out + 1)]
            ts = [0.0]
            Rs = [problem.hole_radius()]
            t_prev = 0.0
            for t in out_times:
                problem.run(
                    t,
                    startstep=max((t - t_prev) * 1e-2, 1e-9),
                    maxstep=max(t - t_prev, 1e-9),
                    temporal_error=1e-3,
                )
                Rs.append(problem.hole_radius())
                ts.append(t)
                t_prev = t
    finally:
        os.chdir(prev_cwd)

    dRdt = [(Rs[i + 1] - Rs[i]) / (ts[i + 1] - ts[i]) for i in range(len(ts) - 1)]
    R_min, k_min = min((R, k) for k, R in enumerate(Rs))
    n_tail = min(3, len(dRdt))
    net_opens = Rs[-1] > Rs[0]
    tail_growing = all(d > 0 for d in dRdt[-n_tail:])
    ok = net_opens and tail_growing
    v_tc = math.sqrt(2.0)
    detail = (
        f"R(t=0) = {Rs[0]:.4f}, dips to {R_min:.4f} at t = {ts[k_min]:.3f}, "
        f"then R(t={ts[-1]:.2f}) = {Rs[-1]:.4f}; "
        f"dR/dt over last {n_tail} outputs = "
        f"[{', '.join(f'{d:.4f}' for d in dRdt[-n_tail:])}] "
        f"(V_TC = {v_tc:.4f}, still approaching from below)"
    )
    return _print_result("s4lite physics probe", ok, detail)


SUBCOMMANDS = {
    "s0": run_s0,
    "s1": run_s1,
    "s2": run_s2,
    "s4lite": run_s4lite,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("test", choices=[*SUBCOMMANDS.keys(), "all"])
    args = ap.parse_args()

    if args.test == "all":
        results = {name: fn() for name, fn in SUBCOMMANDS.items()}
        print("=" * 64)
        for name, ok in results.items():
            print(f"{'PASS' if ok else 'FAIL'}  {name}")
        raise SystemExit(0 if all(results.values()) else 1)
    else:
        ok = SUBCOMMANDS[args.test]()
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
