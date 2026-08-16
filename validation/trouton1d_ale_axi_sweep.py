#!/usr/bin/env python3
"""Driver for TC1D-ALE-AXI: one case per directory, fixed timestep, geometric output.

Deliberately the same shape as ``validation/trouton1d_ale_sweep.py`` (the
planar ALE driver -- read that module's docstring first, most of its
reasoning carries over unchanged): ``maxstep`` fixed for the whole run
because pyoomph's ``temporal_error`` estimator is inert on this problem
family (it reports ``GLOBAL TEMPORAL ERROR 0`` and asks to grow ``dt`` by a
factor of 1000 on every step, so it never binds); geometric output times,
because ``u*`` moves through most of its range in the first per cent of the
run; ``outstep=False``, because ``Problem.run`` defaults ``outstep`` to True
whenever ``temporal_error`` is passed, which is one profile file per INTERNAL
timestep.

COLUMN LAYOUT, AND WHY IT IS NOT PLANAR'S WITH ONE COLUMN RENAMED. Planar's
table carries ``xc``/``Xc``/``Xc_pred``/``Xc_int`` (Ahsan et al.'s nondim
variables, ``X = x/l_I`` with ``l_I = 4 Oh_SB``) and a column named ``R``
holding the LATE-TIME COEFFICIENT ``(1+T)^2 (1 - u*)``. Two things do not
carry over to axi: there is no published axisymmetric analogue of Ahsan et
al.'s stretching map, so the ``l_I``-based columns have nothing to convert
into; and axi's own late-time law is different in form (``docs/axi-design.md``
check 8: ``(1 - u*) R -> const``, no ``(1+T)^2``) and uses the letter ``R``
for the PHYSICAL hole radius throughout the design note. Re-using ``R`` for
the late-time coefficient here, as planar does, would collide with the
solver's own primary output. This driver therefore:

  * keeps ``R`` as the physical hole radius (planar's ``xc``, renamed to its
    axi role, per the task spec -- read straight off ``edge_position()``,
    a mesh degree of freedom, not a time integral of the tip velocity);
  * renames planar's late-time-coefficient column to ``late_coeff`` and
    changes its formula to ``R * (1 - u*)`` (no ``(1+T)^2``);
  * drops ``Xc``/``Xc_pred``/``Xc_int``/``T`` outright;
  * adds ``h_tipode``, integrating docs/axi-design.md check 3's exact
    identity ``dh_tip/dt = 1/(2 mu) - h_tip u_tip / (2 R)`` alongside the run
    using the solver's OWN ``u_tip``/``R`` at each output leg (Heun/RK2
    between output times -- see ``_step_h_tipode`` below). This is the axi
    analogue of planar's ``h_tip_exact`` column, which has no closed form
    here in general (only the two asymptotic limits recorded in
    ``docs/axi-design.md``), so it is measured by integrating the ODE rather
    than by a formula.

NEWTON TOLERANCE. ``newton_solver_tolerance=1e-6`` is passed to every
``run()`` call by default (``--newton-tol`` to change it). See the "TRAP HIT
WHILE WRITING THIS" note at the end of ``trouton1d_ale_axi.py``'s module
docstring: pyoomph's default absolute tolerance (1e-8) is unreachable at
plainly-converged Newton states once the axisymmetric measure scales
residual/Jacobian magnitudes by ``R0`` (or, at fixed ``R0``, once the tiny
impulsive-start timestep does) -- this is a roundoff-floor effect, verified
by FD-Jacobian comparison and by reproducing the identical plateau with the
axi-only terms switched off, not a formulation bug. 1e-6 is still nine
orders of magnitude tighter than the ``O(1)`` physical scale of the problem.

DOMAIN SIZING. Two growth regimes bound the edge displacement beyond ``r0``
(``docs/axi-design.md`` checks 6 and 8): the small-hole exponential
``R(t*) = R0 exp(t*/2)`` (S&B 3.30, valid for ``R0* = R0/Oh_SB << 1``) and the
planar-limit asymptote ``dR/dt -> V_TC``, i.e. a linear displacement
``V_TC * t = mu * t*/sqrt(2)`` (valid for ``R0* >> 1``). The default ``--l0``
takes the larger of the two estimates plus headroom, exactly as planar takes
one estimate; unlike planar, a single formula cannot cover the full ``R0*``
range, so both are computed and maxed. As in planar this is a starting point,
not a promise: every output leg checks ``far_field_drift`` and the run warns
if the truncation was reached.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "simulationCases"))
from trouton1d_ale_axi import Trouton1DALEAxiProblem  # noqa: E402

V_TC = math.sqrt(2.0)


def _step_h_tipode(h_prev: float, u_prev: float, r_prev: float,
                    u_now: float, r_now: float, dt_seg: float, mu: float) -> float:
    """One Heun (RK2) step of dh_tip/dt = 1/(2 mu) - h_tip*u_tip/(2*r_tip).

    docs/axi-design.md, "Internal exact checks" #3: this is the axi analogue
    of planar's closed-form ``h_tip_exact = 1 + t*/4``, integrated rather
    than evaluated because there is no closed form away from the two
    asymptotic limits. Uses the solver's OWN ``u_tip``/``r_tip`` at both ends
    of the leg, as the design note specifies, not a separately-modelled R(t).
    """
    def dhdt(h: float, u: float, r: float) -> float:
        return 1.0 / (2.0 * mu) - h * u / (2.0 * r)

    if dt_seg <= 0.0:
        return h_prev
    k1 = dhdt(h_prev, u_prev, r_prev)
    h_pred = h_prev + dt_seg * k1
    k2 = dhdt(h_pred, u_now, r_now)
    return h_prev + 0.5 * dt_seg * (k1 + k2)


def run_case(oh_sb: float, tstar_end: float, nsteps: int, n_outputs: int,
             outroot: Path, r0: float, l0: float | None, nelem: int,
             grade: float, temporal_error: float = 1e-3, drift_tol: float = 1e-4,
             max_newton: int | None = None, gcn: bool = False,
             ale: bool = True, tip_azimuthal_correction: bool = False,
             newton_tol: float = 1e-6,
             profiles: bool = False,
             output_spacing: str = "geom") -> dict:
    if output_spacing not in ("geom", "uniform"):
        raise ValueError(f"output_spacing must be 'geom' or 'uniform', "
                         f"got {output_spacing!r}")
    outdir = outroot
    outdir.mkdir(parents=True, exist_ok=True)
    os.chdir(outdir)

    with Trouton1DALEAxiProblem() as p:
        # PYOOMPH READS sys.argv, AND ARGPARSE ABBREVIATES. Problem.initialise
        # calls parse_known_args() on the real command line, which carries
        # --outdir; prefix matching means a caller's --out (or --outd, --outdi)
        # is accepted as that flag and OVERWRITES the output directory set
        # here, for every problem built in the process. A single-case script
        # never notices because the two paths coincide; a driver running
        # several cases in one process silently writes them all into one
        # directory. trouton1d_axi_sweep.py refuses the command line for this
        # reason and the ALE sibling must do the same.
        p.ignore_command_line = True
        p.Oh_SB = oh_sb
        p.r0 = r0
        p.nelem = nelem
        p.grade = grade
        p.ale = ale
        p.tip_azimuthal_correction = tip_azimuthal_correction
        t_end = tstar_end * p.tau_vis

        if l0 is None:
            # Small-hole exponential estimate (S&B 3.30). The exponential law
            # is an EARLY-TIME statement: it holds only while the edge speed
            # R/(2 tau_vis) is below V_TC, i.e. until R reaches the crossover
            # radius sqrt(2)*mu; past that the edge cannot outrun V_TC. Taking
            # the raw exponential at large tstar_end therefore requests an
            # astronomically long domain (r0*e^{t*/2} ~ 1e21 at t* = 100) and
            # the resulting mesh coordinates blow the first Newton solve up to
            # O(1e34) residuals -- which is how this bound was discovered.
            disp_exp = r0 * (math.exp(min(tstar_end, 700.0) / 2.0) - 1.0)
            # Exponential-phase displacement can never exceed reaching the
            # crossover radius; after that, growth is bounded by V_TC.
            disp_exp_bound = max(0.0, math.sqrt(2.0) * p.mu - r0)
            # Planar-limit linear estimate: dR/dt -> V_TC, i.e. displacement
            # V_TC * t = mu * t*/sqrt(2) (dimension-consistent with planar's
            # own xc_end = l_I * tstar_end/4 = Oh_SB * tstar_end).
            disp_lin = p.mu * tstar_end / math.sqrt(2.0)
            disp_est = max(disp_lin, min(disp_exp, disp_exp_bound + disp_lin))
            l0 = disp_est + max(0.2 * disp_est, 30.0 * max(1.0, p.mu))
        p.l0 = l0
        p.set_output_directory(str(outdir))
        p.initialise()
        p.write_states = False

        dt = t_end / nsteps
        meta = {
            "code": "TC1D-ALE-AXI", "Oh_SB": oh_sb, "mu": p.mu,
            "tau_vis": p.tau_vis, "R0": r0, "R0_star": p.r0_star,
            "tstar_end": tstar_end, "t_end": t_end,
            "l0": p.l0, "nelem": p.nelem, "grade": p.grade,
            "ale": p.ale, "tip_azimuthal_correction": p.tip_azimuthal_correction,
            "nsteps": nsteps, "dt": dt, "n_outputs": n_outputs,
            "scheme": p.default_timestepping_scheme,
            "newton_tol": newton_tol,
            "tip_cell_initial": p.tip_cell_size(),
            "mass_initial": p.total_mass(),
            "V_TC": V_TC,
        }
        print(f"[ale-axi oh={oh_sb:g} R0={r0:g}] mu={p.mu:.6g} R0*={p.r0_star:.4g} "
              f"l0={p.l0:.6g} N={p.nelem} grade={p.grade:g} t_end={t_end:.6g} "
              f"dt={dt:.6g} tip_cell={meta['tip_cell_initial']:.3e}", flush=True)

        history = p.create_text_file_output(
            "control1d_ale_axi.txt",
            header=["t", "tstar", "R", "u_tip", "ustar", "h_tip", "h_tipode",
                    "late_coeff", "mass", "affinity", "roughness",
                    "drift_h", "drift_u", "tip_cell"])

        if output_spacing == "uniform":
            out_times = np.linspace(t_end / n_outputs, t_end, n_outputs)
        elif n_outputs == 1:
            out_times = np.array([t_end])
        else:
            out_times = np.geomspace(t_end * 1e-4, t_end, n_outputs)

        prev = 0.0
        u_prev = 0.0
        r_prev = r0
        h_tipode = 1.0                    # h_tip(0) = 1, the IC
        worst_drift = 0.0
        worst_affinity = 0.0
        worst_rough = 0.0
        rows = []
        t0 = time.time()
        for t in out_times:
            # Tiny first step only: the impulsively loaded edge has a sqrt(t)
            # start-up singularity, exactly as planar. maxstep is identical
            # on every leg.
            start = t_end * 1e-6 if prev == 0.0 else min(dt, t - prev)
            extra = {}
            if max_newton is not None:
                extra["max_newton_iterations"] = max_newton
            if gcn:
                extra["globally_convergent_newton"] = True
            p.run(t, startstep=start, maxstep=dt, temporal_error=temporal_error,
                  outstep=False, out_initially=False,
                  newton_solver_tolerance=newton_tol, **extra)
            if profiles:
                p.output()
            now = float(p.get_current_time(as_float=True))
            tstar = now / p.tau_vis
            u = p.tip_velocity()
            ustar = u / V_TC
            R = p.edge_position()
            h_tipode = _step_h_tipode(h_tipode, u_prev, r_prev, u, R,
                                      now - prev, p.mu)
            late_coeff = R * (1.0 - ustar)
            dh, du = p.far_field_drift()
            aff = p.mesh_affinity_residual()
            rough = p.thickness_roughness()
            h_tip = p.tip_thickness()
            history.add_row(now, tstar, R, u, ustar, h_tip, h_tipode,
                            late_coeff, p.total_mass(), aff, rough, dh, du,
                            p.tip_cell_size())
            rows.append([tstar, R, ustar, h_tip, h_tipode, late_coeff])
            worst_drift = max(worst_drift, abs(dh), abs(du))
            worst_affinity = max(worst_affinity, aff)
            worst_rough = max(worst_rough, rough)
            prev, u_prev, r_prev = now, u, R

        tab = np.array(rows)
        tstar_reached = float(p.get_current_time(as_float=True)) / p.tau_vis
        h_tip_final = p.tip_thickness()
        meta.update({
            "wall_s": time.time() - t0,
            "tstar_reached": tstar_reached,
            "R_final": p.edge_position(),
            "ustar_final": p.tip_velocity() / V_TC,
            "h_tip_final": h_tip_final,
            "h_tipode_final": h_tipode,
            "h_tipode_rel_err": abs(h_tip_final / h_tipode - 1.0) if h_tipode != 0 else float("nan"),
            "late_coeff_final": float(tab[-1, 5]),
            "mass_final": p.total_mass(),
            "mass_rel_drift": abs(p.total_mass() / meta["mass_initial"] - 1.0),
            "worst_far_field_drift": worst_drift,
            "worst_mesh_affinity": worst_affinity,
            "worst_thickness_roughness": worst_rough,
            "tip_cell_final": p.tip_cell_size(),
        })
        meta["domain_ok"] = bool(worst_drift <= drift_tol)
        if not meta["domain_ok"]:
            print(f"[ale-axi oh={oh_sb:g} R0={r0:g}] WARNING: far-field drift "
                  f"{worst_drift:.3e} exceeds {drift_tol:g}; the disturbance "
                  f"reached the truncation. Re-run with a larger --l0.",
                  flush=True)
        (outdir / "case1d_ale_axi.json").write_text(json.dumps(meta, indent=2))
        for tq in (0.5, 1, 2, 5, 10, 20, 50):
            if tq <= tstar_reached + 1e-9:
                i = int(np.argmin(np.abs(tab[:, 0] - tq)))
                print(f"  t*={tab[i,0]:8.4f} R={tab[i,1]:10.5f} "
                      f"u*={tab[i,2]:.7f} h_tip={tab[i,3]:.5f} "
                      f"h_tipode={tab[i,4]:.5f} late_coeff={tab[i,5]:.5f}",
                      flush=True)
        print(f"[ale-axi oh={oh_sb:g} R0={r0:g}] done: u*({tstar_end:g}) = "
              f"{meta['ustar_final']:.10f} R={meta['R_final']:.6f} "
              f"affinity={worst_affinity:.2e} rough={worst_rough:.2e} "
              f"mass_drift={meta['mass_rel_drift']:.2e} "
              f"h_tipode_rel_err={meta['h_tipode_rel_err']:.2e} "
              f"drift={worst_drift:.2e} wall={meta['wall_s']:.0f}s", flush=True)
        return meta


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oh", type=float, default=1.0, help="Oh_SB (Savva-Bush)")
    ap.add_argument("--r0", type=float, default=5.0, help="initial hole radius R0")
    ap.add_argument("--tstar-end", type=float, required=True)
    ap.add_argument("--nsteps", type=int, required=True,
                    help="fixed dt = t_end/nsteps; run twice with 2x for the order")
    ap.add_argument("--nelem", type=int, default=800)
    ap.add_argument("--grade", type=float, default=2.0)
    ap.add_argument("--l0", type=float, default=None)
    ap.add_argument("--n-outputs", type=int, default=300)
    ap.add_argument("--temporal-error", type=float, default=1e-3)
    ap.add_argument("--drift-tol", type=float, default=1e-4)
    ap.add_argument("--newton-tol", type=float, default=1e-6,
                    help="absolute Newton tolerance; the pyoomph default "
                         "(1e-8) is unreachable at this solver's natural "
                         "roundoff floor, see trouton1d_ale_axi.py")
    ap.add_argument("--max-newton", type=int, default=None)
    ap.add_argument("--gcn", action="store_true")
    ap.add_argument("--tip-azimuthal-correction", action="store_true",
                    help="add the O(h/R) toroidal-rim tip term "
                         "(docs/axi-design.md, 'Tip closure')")
    ap.add_argument("--output-spacing", choices=("geom", "uniform"),
                    default="geom")
    ap.add_argument("--profiles", action="store_true",
                    help="also write an h/u profile per output leg")
    ap.add_argument("--no-ale", dest="ale", action="store_false",
                    help="drop the ALE correction: WRONG on a moving mesh, "
                         "kept to measure how large the correction is")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    for name, value in (("--oh", a.oh), ("--r0", a.r0), ("--tstar-end", a.tstar_end),
                        ("--nsteps", a.nsteps), ("--n-outputs", a.n_outputs),
                        ("--grade", a.grade), ("--drift-tol", a.drift_tol),
                        ("--newton-tol", a.newton_tol)):
        if value <= 0:
            ap.error(f"{name} must be positive, got {value!r}")
    if a.nelem < 2:
        ap.error(f"--nelem must be at least 2, got {a.nelem!r}")
    if a.l0 is not None and a.l0 <= 0:
        ap.error(f"--l0 must be positive, got {a.l0!r}")

    run_case(a.oh, a.tstar_end, a.nsteps, a.n_outputs, a.out.resolve(),
             a.r0, a.l0, a.nelem, a.grade, a.temporal_error, a.drift_tol,
             a.max_newton, a.gcn, a.ale, a.tip_azimuthal_correction,
             a.newton_tol, a.profiles, output_spacing=a.output_spacing)


if __name__ == "__main__":
    main()
