#!/usr/bin/env python3
"""Internal exact checks for the axisymmetric mass-coordinate solver.

These are the tests listed under "Internal exact checks" in
``the axisymmetric design note (axi-design.md)``, made runnable. Each subcommand prints one or more
``PASS``/``FAIL`` lines carrying the measured numbers, and the process exit
code is non-zero if any check failed.

    imports        the module imports and the field spaces are what is claimed
    metric         w_m - 1/h vanishes per cell to round-off
    mass           the metric-implied global radius identity closes
    planar-limit   axi -> planar as 1/R0             [THE GATE]
    tipthickness   the solver tracks the tip-closure identity
    exponential    high-Oh small-hole growth rate is 1/(2 tau_vis)
    dissipation    the depth-averaged dissipation is positive by construction
    all            every check above, in that order

``planar-limit`` is the gate. It is the only check that compares against an
independently validated solver rather than against the formulation's own
internal structure, so nothing else here is evidence of correctness until it
passes: a metric constraint can be satisfied exactly by a wrong metric, and a
positive-definite dissipation stays positive under a wrong prefactor.

Run order matters for cost, not for correctness -- ``metric`` and ``mass`` are
seconds, ``planar-limit`` is the expensive one, and running it first would
mean paying for it before knowing the cheap structural checks hold.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "src"))

import trouton1d_axi_sweep as axi  # noqa: E402
import trouton1d_sweep as planar  # noqa: E402

#: Default run root. Simulation output never belongs in the repository.
OUTROOT = Path("/tmp/axi-1d-mass/checks")


def report(name: str, ok: bool, msg: str) -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {msg}", flush=True)
    return ok


# ---------------------------------------------------------------------------
# 0. imports
# ---------------------------------------------------------------------------

def check_imports() -> bool:
    from taylor_culick_fem.trouton1d_axi import (  # noqa: F401
        AxiTipFlux,
        AxiTroutonSheet,
        Trouton1DAxiProblem,
        auto_mass,
    )
    ok = True
    ok &= report("imports.capillarity-guard",
                 _raises(lambda: AxiTroutonSheet(mu=1.0,
                                                 with_capillarity=True)),
                 "with_capillarity=True rejected with a ValueError")
    p = Trouton1DAxiProblem()
    p.Oh, p.r0 = 1.0, 5.0
    ok &= report("imports.units",
                 abs(p.mu - math.sqrt(2.0)) < 1e-15
                 and abs(p.tau_vis - math.sqrt(2.0) / 2) < 1e-15,
                 f"Oh=1 -> mu={p.mu:.10f}, tau_vis={p.tau_vis:.10f}, "
                 f"R0*={p.r0_star:g}")
    m = auto_mass(5.0, 0.0)
    ok &= report("imports.auto_mass", abs(m - 0.5 * (125.0 ** 2 - 25.0)) < 1e-9,
                 f"auto_mass(r0=5, t_end=0) = {m:.4f} = (125^2 - 5^2)/2")
    return ok


def _raises(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    return False


# ---------------------------------------------------------------------------
# short reference run, shared by metric / mass / dissipation
# ---------------------------------------------------------------------------

_SHORT = {"oh": 1.0, "tstar_end": 5.0, "nsteps": 400, "n_outputs": 25,
          "r0": 5.0, "mass": 3000.0, "nelem": 250}


def _run_short(name: str, probe):
    return axi.run_case(outroot=OUTROOT / name, tag=name, probe=probe, **_SHORT)


# ---------------------------------------------------------------------------
# 1. metric consistency
# ---------------------------------------------------------------------------

def check_metric() -> bool:
    worst = {"defect": 0.0}

    def probe(p, row):
        worst["defect"] = max(worst["defect"], p.metric_defect())

    meta = _run_short("metric", probe)
    d = worst["defect"]
    return report("metric", d < 1e-10,
                  f"max over cells and outputs of |w_m - 1/h| = {d:.3e} "
                  f"(tolerance 1e-10); final value {meta['metric_defect_final']:.3e}, "
                  f"R0=5, Oh=1, t*=5, N=250")


# ---------------------------------------------------------------------------
# 2. metric-implied global identity, and far-field h
# ---------------------------------------------------------------------------

def check_mass() -> bool:
    worst = {"abs": 0.0, "rel": 0.0, "dh": 0.0}

    def probe(p, row):
        c = p.metric_closure()
        r_max2 = 2.0 * p.nodal_profile()[2][-1]
        worst["abs"] = max(worst["abs"], abs(c))
        worst["rel"] = max(worst["rel"], abs(c) / r_max2)
        worst["dh"] = max(worst["dh"], p.far_field_drift()[0])

    _run_short("mass", probe)
    ok_closure = worst["rel"] < 1e-9
    ok_far = worst["dh"] < 1e-10
    ok = report("mass.closure", ok_closure,
                f"max |r(M)^2 - R^2 - 2 int dm/h| = {worst['abs']:.3e}, "
                f"relative to r(M)^2 = {worst['rel']:.3e} (tolerance 1e-9). "
                "Mass itself is exact by construction: M does not move in this "
                "coordinate, so this is the metric identity instead")
    ok &= report("mass.far-field-h", ok_far,
                 f"max |h_lastcell - 1| = {worst['dh']:.3e} (tolerance 1e-10)")
    return ok


# ---------------------------------------------------------------------------
# 3. planar limit -- THE GATE
# ---------------------------------------------------------------------------

_PL = {"oh": 1.0, "tstar_end": 10.0, "nsteps": 1500, "n_outputs": 60,
       "planar_mass": 120.0, "planar_nelem": 400, "grade": 2.0}

#: The gate's asymptotic-order threshold, applied to the LARGEST pair only.
_PL_ORDER_MIN = 0.75


def check_planar_limit(r0_list=(50.0, 100.0, 200.0, 400.0)) -> bool:
    """Axi at large R0 must track the planar solver with an O(1/R0) error.

    The two runs are matched where it matters and only where it matters. Same
    Oh, same t_end, same ``nsteps`` and therefore the same dt on every step,
    and the same ``n_outputs``, so both drivers sample the identical geometric
    time grid and the comparison is row by row with no interpolation. The
    meshes are matched by TIP CELL IN RADIUS, not by element count: the axi
    domain is a mass, so an equal N would put a systematically different cell
    at the tip and contaminate the difference with a discretisation offset
    that does not scale with 1/R0.

    WHERE THE ASYMPTOTIC REGIME STARTS, AND WHY THE GATE IS ON THE LARGEST
    PAIR. The film occupies ``R0 <= r <= R0 + L`` with ``L`` the validated
    planar domain of 120 thicknesses, so the sub-dominant term in the
    difference is governed by ``L/R0``, not by the small parameter ``h0/R0``
    that governs the physics. At the bottom of the ladder ``L/R0`` is of order
    one -- the far end of the domain sits at several times the hole radius --
    and the O(1/R0^2) term is therefore not small there however clean the
    leading behaviour is. Shortening ``L`` would suppress it, at the cost of
    replacing it with the finite-window clamp error that is the planar
    solver's own documented dominant error. So the ladder is run long enough
    to reach the asymptotic pair and the LINEARITY is asserted there, while
    the MAGNITUDE bound is asserted at every rung. ``R0 * max|du*|``
    approaching a constant from below across the ladder is the same statement
    in a form that is read off directly.

    ``R0 = 1e6`` is deliberately not in the ladder. The flux slot carries
    ``r^2 h^2 u_m``; at that radius it would carry ~1e12 scale factors and the
    measurement would be of conditioning, not of the metric terms.
    """
    L, N = _PL["planar_mass"], _PL["planar_nelem"]
    pm = planar.run_case(_PL["oh"], _PL["tstar_end"], _PL["nsteps"],
                         _PL["n_outputs"], 1e-3, OUTROOT / "planar_ref",
                         L, N, 1e-4)
    ref = np.loadtxt(OUTROOT / "planar_ref" / "control1d.txt")
    t_ref, ustar_ref = ref[:, 1], ref[:, 3]
    print(f"       planar reference: u*({_PL['tstar_end']:g}) = "
          f"{pm['ustar_final']:.10f}, drift {pm['worst_far_field_drift']:.2e}",
          flush=True)

    errs = []
    for r0 in r0_list:
        mass = 0.5 * ((r0 + L) ** 2 - r0 * r0)
        nelem = round(N * math.sqrt(mass / (r0 * L)))
        m = axi.run_case(_PL["oh"], _PL["tstar_end"], _PL["nsteps"],
                         _PL["n_outputs"], OUTROOT / f"planar_limit_R{r0:g}",
                         r0=r0, mass=mass, nelem=nelem, grade=_PL["grade"],
                         tag=f"planar-limit R0={r0:g}")
        s = m["series"]
        assert np.allclose(s["tstar"], t_ref, rtol=1e-10), \
            "output time grids diverged; the comparison would need interpolation"
        e = float(np.max(np.abs(s["ustar"] - ustar_ref)))
        errs.append(e)
        print(f"       R0={r0:g}: M={mass:.5g} N={nelem} "
              f"max|du*|={e:.4e}  R0*max|du*|={e * r0:.4f}  "
              f"drift={m['worst_far_field_drift']:.2e}", flush=True)

    ok = True
    for r0, e in zip(r0_list, errs, strict=True):
        ok &= report(f"planar-limit.magnitude.R0={r0:g}", e < 2.0 / r0,
                     f"max|u*_axi - u*_planar| = {e:.4e} against the 2/R0 "
                     f"bound {2.0 / r0:.4e}; R0 max|du*| = {e * r0:.4f}")
    order = [math.log(errs[i] / errs[i + 1])
             / math.log(r0_list[i + 1] / r0_list[i])
             for i in range(len(errs) - 1)]
    ok &= report("planar-limit.monotone",
                 all(errs[i] > errs[i + 1] for i in range(len(errs) - 1)),
                 "the error decreases at every rung of the ladder: "
                 + ", ".join(f"{e:.4e}" for e in errs))
    ok &= report(f"planar-limit.linearity.{r0_list[-2]:g}->{r0_list[-1]:g}",
                 order[-1] >= _PL_ORDER_MIN,
                 f"observed order in 1/R0 on the asymptotic pair = "
                 f"{order[-1]:.3f} (threshold {_PL_ORDER_MIN:g}); full "
                 "sequence " + ", ".join(f"{o:.3f}" for o in order))
    return ok


# ---------------------------------------------------------------------------
# 4 & 5. high-Oh small-hole run, shared
# ---------------------------------------------------------------------------

#: Oh = 100 with R0 = 5 puts R0* = R0/(Oh h0) = 0.05 << 1, which is the
#: regime in which Savva & Bush's exponential opening (their 3.30) holds. t*=4
#: multiplies R by exp(2) ~ 7.4 if that law is right, which is the "several
#: fold" the tip-thickness check needs. The domain is sized by hand rather than
#: by auto_mass: the viscous influence length is ~mu = 141 thicknesses here, so
#: r_max = 250 covers both the travel and the disturbance.
_HIGH = {"oh": 100.0, "tstar_end": 4.0, "nsteps": 2000, "n_outputs": 60,
         "r0": 5.0, "mass": 0.5 * (250.0 ** 2 - 5.0 ** 2), "nelem": 800}

_high_cache: dict[str, dict] = {}


def _run_high() -> dict:
    if "meta" not in _high_cache:
        cells = []

        def probe(p, row):
            cells.append(p.tip_thickness())

        m = axi.run_case(outroot=OUTROOT / "high_oh", tag="high-Oh",
                         probe=probe, **_HIGH)
        m["h_tip_cell"] = np.asarray(cells)
        _high_cache["meta"] = m
    return _high_cache["meta"]


def check_tipthickness() -> bool:
    m = _run_high()
    s = m["series"]
    h_cell, h_ode = m["h_tip_cell"], s["h_tipode"]
    rel = np.abs(h_cell - h_ode) / np.abs(h_ode)
    late = s["tstar"] > 0.5
    worst_all, worst_late = float(rel.max()), float(rel[late].max())
    at_worst = float(s["tstar"][int(np.argmax(rel))])
    grew = float(s["R"][-1] / _HIGH["r0"])
    # The first cell is one cell wide, so it is an average over that cell
    # rather than a point value and exact agreement is not expected; what is
    # being tested is that the solver and the closure identity describe the
    # same rim, not that they agree to round-off.
    ok = report("tipthickness", worst_late < 1e-2,
                f"max relative deviation of the first-cell h from the closure "
                f"identity = {worst_late:.3e} for t* > 0.5, {worst_all:.3e} "
                f"over the whole run (attained at t* = {at_worst:.3g}), "
                f"tolerance 1e-2; R grew {grew:.2f}-fold to {s['R'][-1]:.3f}; "
                f"final h_tip cell {h_cell[-1]:.6f} vs identity "
                f"{h_ode[-1]:.6f}")
    print(f"       context: planar identity 1 + t*/4 = "
          f"{1 + s['tstar'][-1] / 4:.4f}, late-time axi limit t*/6 = "
          f"{s['tstar'][-1] / 6:.4f} (an asymptote, not reached at t* = "
          f"{s['tstar'][-1]:.2f})", flush=True)
    return ok


# ---------------------------------------------------------------------------
# 5. exponential early growth at high Oh, small hole
# ---------------------------------------------------------------------------

def check_exponential() -> bool:
    m = _run_high()
    s = m["series"]
    tau, mu = m["tau_vis"], m["mu"]
    target = 1.0 / (2.0 * tau)          # = 1/mu

    def fit(lo: float, hi: float) -> tuple[float, int]:
        w = (s["t"] >= lo * tau) & (s["t"] <= hi * tau)
        if w.sum() < 4:
            return float("nan"), int(w.sum())
        slope = np.polyfit(s["t"][w], np.log(s["R"][w]), 1)[0]
        return float(slope), int(w.sum())

    # The similarity solution is an EARLY-TIME law that the impulsive start
    # has to settle onto: R(0) is at rest, so the first fraction of tau_vis is
    # a start-up transient with a rate below the asymptote. The reported
    # window excludes it; the others are printed so the sensitivity to that
    # choice is visible rather than hidden in the pass.
    windows = [(0.0, 1.0), (0.1, 1.0), (0.2, 1.0), (0.3, 1.0), (0.2, 0.6)]
    fits = {w: fit(*w) for w in windows}
    slope, npts = fits[(0.2, 1.0)]
    err = abs(slope - target) / target
    ok = report("exponential", err < 0.05,
                f"d(ln R)/dt = {slope:.6e} on t/tau_vis in [0.2, 1] "
                f"({npts} points) against 1/(2 tau_vis) = {target:.6e}, "
                f"relative error {err * 100:.2f}% (tolerance 5%); "
                f"Oh={m['Oh']:g}, R0={m['r0']:g}, R0*={m['r0_star']:g}, "
                f"mu={mu:.4f}")
    for w, (sl, n) in fits.items():
        print(f"       window t/tau_vis in [{w[0]:g}, {w[1]:g}]: "
              f"rate {sl:.6e} ({sl / target:.4f} x target, {n} points)",
              flush=True)
    return ok


# ---------------------------------------------------------------------------
# 6. dissipation positivity
# ---------------------------------------------------------------------------

def check_dissipation() -> bool:
    rec: list[tuple[float, float, float, float, float]] = []

    def probe(p, row):
        lo, total = p.dissipation()
        _, u, _ = p.nodal_profile()
        ke = 0.0
        for j, (ml, mr, h) in enumerate(p.cell_profile()):
            a, b = u[j], u[j + 1]
            ke += (mr - ml) * (a * a + a * b + b * b) / 6.0
        rec.append((row["t"], lo, total, ke, 2.0 * row["R"] * row["u_tip"]))

    _run_short("dissipation", probe)
    arr = np.asarray(rec)
    t, lo, diss, ke, power = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    worst = float(lo.min())
    ok = report("dissipation.positivity",
                worst >= 0.0 and np.all(np.isfinite(arr)),
                f"min over cells and outputs of "
                f"2 mu h [u_r^2 + (u/r)^2 + (u_r + u/r)^2] = {worst:.3e}; "
                f"all {arr.size} recorded values finite. Negative here means "
                "the hoop remainder has been restructured into sigma_thth/r "
                "or has the wrong sign")

    # The same weak form with v = u gives d(KE)/dt = 2 R u_tip - D exactly,
    # so this is a genuine budget rather than a restatement. It is reported
    # and not gated: the time derivative is a finite difference on a
    # geometric output cadence, which is the accuracy limit.
    dke = np.gradient(ke, t)
    resid = dke - (power - diss)
    scale = np.maximum(np.abs(power), 1e-30)
    mid = slice(1, -1)
    worst_bal = float(np.max(np.abs(resid[mid]) / scale[mid]))
    print(f"       energy budget d(KE)/dt = 2 R u_tip - D: worst relative "
          f"residual {worst_bal:.3e} over interior outputs (finite-difference "
          f"derivative on a geometric cadence; diagnostic only). Final: "
          f"KE={ke[-1]:.6g}, D={diss[-1]:.6g}, tip power={power[-1]:.6g}",
          flush=True)
    return ok


CHECKS = {
    "imports": check_imports,
    "metric": check_metric,
    "mass": check_mass,
    "planar-limit": check_planar_limit,
    "tipthickness": check_tipthickness,
    "exponential": check_exponential,
    "dissipation": check_dissipation,
}
ORDER = ["imports", "metric", "mass", "planar-limit", "tipthickness",
         "exponential", "dissipation"]


def main() -> None:
    global OUTROOT
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("check", choices=ORDER + ["all"])
    ap.add_argument("--out", type=Path, default=OUTROOT,
                    help="run directory root (never inside the repository)")
    args = ap.parse_args()

    # HIDE THE COMMAND LINE FROM pyoomph. Problem.initialise parses the real
    # sys.argv with argparse's prefix matching active, and one of its flags is
    # --outdir, so this script's own --out would be accepted as that flag and
    # would override every per-case output directory in the process. The axi
    # driver sets ignore_command_line for the same reason, but the planar
    # reference run below goes through trouton1d_sweep, which does not, so the
    # argv is emptied here as well.
    sys.argv = sys.argv[:1]

    OUTROOT = args.out.resolve()
    OUTROOT.mkdir(parents=True, exist_ok=True)

    names = ORDER if args.check == "all" else [args.check]
    results = {}
    for n in names:
        print(f"\n===== {n} =====", flush=True)
        results[n] = bool(CHECKS[n]())

    print("\n===== summary =====")
    for n, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
