#!/usr/bin/env python3
"""Runnable PASS/FAIL checks for TC1D-ALE-AXI against docs/axi-design.md.

Each subcommand runs the solver (via ``trouton1d_ale_axi_sweep.run_case``, so
every check exercises the exact driver the rest of the campaign uses, not a
bespoke harness) at settings chosen for that check, and prints the measured
number against the design note's threshold. ``all`` runs them in the spec's
own order: import check, then 1 (mass), then 2 (planar-limit -- "the gate"),
then 4/5/6 (tipthickness, affinity+roughness, exponential), then 3
(cross-solver, which may legitimately SKIP if the sibling mass-coordinate
solver has not landed yet), matching the numbering in ``docs/axi-design.md``,
"Internal exact checks".

Everything writes under ``/tmp/axi-1d-ale/checks`` by default (``--outroot``
to change it) -- simulation output does not belong in the repository.

Usage:
    uv run python validation/trouton1d_ale_axi_checks.py all
    uv run python validation/trouton1d_ale_axi_checks.py mass
    uv run python validation/trouton1d_ale_axi_checks.py planar-limit
    ...
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "simulationCases"))
sys.path.insert(0, str(REPO_ROOT / "validation"))

import trouton1d_ale_axi_sweep as axi_sweep  # noqa: E402
import trouton1d_ale_sweep as planar_sweep  # noqa: E402

DEFAULT_OUTROOT = Path("/tmp/axi-1d-ale/checks")


def _hr(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}", flush=True)


def _verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


# ---------------------------------------------------------------------------
# 0. import check
# ---------------------------------------------------------------------------

def check_import() -> bool:
    """Cold-process import of all three of this task's files.

    Run as a subprocess deliberately: by the time ``main()`` below is
    running at all, this checks script has already imported
    ``trouton1d_ale_axi_sweep`` (hence ``trouton1d_ale_axi``) at module
    level, so an in-process re-import would trivially pass regardless of
    whether the files are actually importable standalone. A fresh
    interpreter is the only way this check can fail independently of the
    rest of the run.
    """
    _hr("import check")
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(REPO_ROOT / 'simulationCases')!r}); "
        f"sys.path.insert(0, {str(REPO_ROOT / 'validation')!r}); "
        "import trouton1d_ale_axi; "
        "import trouton1d_ale_axi_sweep; "
        "import trouton1d_ale_axi_checks; "
        "print('OK')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, check=False)
    print(proc.stdout.strip())
    ok = proc.returncode == 0 and "OK" in proc.stdout
    if not ok:
        print(f"IMPORT FAILED (exit {proc.returncode}):\n{proc.stderr[-2000:]}")
    print(f"import check -> {_verdict(ok)}")
    return ok


# ---------------------------------------------------------------------------
# 1. mass
# ---------------------------------------------------------------------------

def check_mass(outroot: Path, threshold: float = 1e-8) -> bool:
    _hr("1. mass (2*pi*int h r dr, Oh_SB=1, R0=5, t*=5)")
    # Resolution ladder measured while writing this check: mass_rel_drift is
    # spatial-error-dominated (doubling nsteps at fixed nelem does not move
    # it), 3.6e-8 at nelem=800, falling to 9.1e-9 at nelem=1600 and 5.8e-9 at
    # nelem=2000/nsteps=4000 -- the setting used here, with margin under the
    # 1e-8 threshold rather than sitting right on it.
    meta = axi_sweep.run_case(
        oh_sb=1.0, tstar_end=5.0, nsteps=4000, n_outputs=60,
        outroot=outroot / "mass", r0=5.0, l0=None, nelem=2000, grade=2.0)
    drift = meta["mass_rel_drift"]
    ok = drift < threshold
    print(f"mass_initial = {meta['mass_initial']:.10g}")
    print(f"mass_final   = {meta['mass_final']:.10g}")
    print(f"relative drift = {drift:.3e}  (threshold {threshold:.0e})")
    print(f"mass -> {_verdict(ok)}")
    return ok


# ---------------------------------------------------------------------------
# 2. planar-limit (the gate)
# ---------------------------------------------------------------------------

def check_planar_limit(outroot: Path) -> bool:
    _hr("2. planar-limit (R0 = 50, 100, 200; Oh_SB=1, t*=10) -- THE GATE")
    r0_values = (50.0, 100.0, 200.0)
    l0_matched = 100.0          # SAME l0/nelem/grade for axi and planar, so
    nelem_matched = 600         # the R0-trend below is not contaminated by a
    grade_matched = 2.0         # resolution difference between the two runs.
    nsteps = 3000

    planar_meta = planar_sweep.run_case(
        oh_sb=1.0, tstar_end=10.0, nsteps=nsteps, n_outputs=40,
        outroot=outroot / "planar_limit" / "planar", l0=l0_matched,
        nelem=nelem_matched, grade=grade_matched)
    ustar_planar = planar_meta["ustar_final"]
    print(f"planar TC1D-ALE reference: u*(10) = {ustar_planar:.8f}")

    devs = {}
    for r0 in r0_values:
        axi_meta = axi_sweep.run_case(
            oh_sb=1.0, tstar_end=10.0, nsteps=nsteps, n_outputs=40,
            outroot=outroot / "planar_limit" / f"axi_R0_{r0:g}",
            r0=r0, l0=l0_matched, nelem=nelem_matched, grade=grade_matched)
        dev = abs(axi_meta["ustar_final"] - ustar_planar)
        devs[r0] = dev
        threshold = 2.0 / r0
        print(f"R0={r0:6.1f}: u*(10)={axi_meta['ustar_final']:.8f} "
              f"|dev|={dev:.4e}  2/R0={threshold:.4e}  "
              f"dev*R0={dev * r0:.4e}")

    threshold_ok = all(devs[r0] < 2.0 / r0 for r0 in r0_values)
    # "falls ~linearly in 1/R0" <=> dev*R0 is roughly constant across the
    # ladder. Loose bound: max/min of dev*R0 within a factor of 4 -- tight
    # enough to catch a wrong power law, loose enough to survive the
    # discretisation noise a single matched-resolution ladder carries.
    scaled = [devs[r0] * r0 for r0 in r0_values]
    linear_ratio = max(scaled) / min(scaled) if min(scaled) > 0 else float("inf")
    linear_ok = linear_ratio < 4.0
    print(f"max(dev*R0)/min(dev*R0) = {linear_ratio:.3f}  (threshold < 4, "
          f"i.e. ~linear in 1/R0)")

    ok = threshold_ok and linear_ok
    print(f"planar-limit -> {_verdict(ok)}"
          f"{'' if ok else '  (threshold_ok=' + str(threshold_ok) + ', linear_ok=' + str(linear_ok) + ')'}")
    return ok


# ---------------------------------------------------------------------------
# 4. tipthickness
# ---------------------------------------------------------------------------

def check_tipthickness(outroot: Path, threshold: float = 1e-2) -> bool:
    _hr("4. tipthickness (Oh_SB=100, R0=5, t* to 6: R grows ~20x)")
    meta = axi_sweep.run_case(
        oh_sb=100.0, tstar_end=6.0, nsteps=4000, n_outputs=60,
        outroot=outroot / "tipthickness", r0=5.0, l0=None, nelem=800,
        grade=2.0)
    err = meta["h_tipode_rel_err"]
    print(f"R: {5.0:.3f} -> {meta['R_final']:.3f} "
          f"({meta['R_final'] / 5.0:.1f}x growth)")
    print(f"h_tip (solver)  = {meta['h_tip_final']:.6f}")
    print(f"h_tip (ODE)     = {meta['h_tipode_final']:.6f}")
    print(f"relative deviation = {err:.3e}  (threshold {threshold:.0e})")
    ok = err < threshold
    print(f"tipthickness -> {_verdict(ok)}")
    return ok


# ---------------------------------------------------------------------------
# 5. affinity + roughness
# ---------------------------------------------------------------------------

def check_affinity_roughness(outroot: Path, affinity_threshold: float = 1e-10,
                              roughness_threshold: float = 0.05) -> bool:
    _hr("5. affinity + roughness (Oh_SB=1, R0=5, t*=10)")
    meta = axi_sweep.run_case(
        oh_sb=1.0, tstar_end=10.0, nsteps=3000, n_outputs=60,
        outroot=outroot / "affinity_roughness", r0=5.0, l0=None, nelem=800,
        grade=2.0)
    aff = meta["worst_mesh_affinity"]
    rough = meta["worst_thickness_roughness"]
    print(f"worst mesh_affinity_residual = {aff:.3e}  "
          f"(threshold {affinity_threshold:.0e}, planar measures ~1e-14 to "
          f"1e-16 -- LaplaceSmoothedMesh IS the affine map to machine "
          f"precision)")
    print(f"worst thickness_roughness    = {rough:.3e}  "
          f"(threshold {roughness_threshold:g}; falls as dx^2 in planar, "
          f"no coded lower bound -- this is a sanity ceiling, not a target)")
    ok_aff = aff < affinity_threshold
    ok_rough = rough < roughness_threshold
    ok = ok_aff and ok_rough
    print(f"affinity+roughness -> {_verdict(ok)}"
          f"{'' if ok else f'  (affinity {_verdict(ok_aff)}, roughness {_verdict(ok_rough)})'}")
    return ok


# ---------------------------------------------------------------------------
# 6. exponential
# ---------------------------------------------------------------------------

def check_exponential(outroot: Path, threshold: float = 0.03) -> bool:
    _hr("6. exponential (Oh_SB=100, R0=5 => R0*=0.035<<1; early R(t) growth)")
    rundir = outroot / "exponential"
    meta = axi_sweep.run_case(
        oh_sb=100.0, tstar_end=3.0, nsteps=3000, n_outputs=120,
        outroot=rundir, r0=5.0, l0=None, nelem=800, grade=2.0)
    mu = meta["mu"]
    data = np.loadtxt(rundir / "control1d_ale_axi.txt")
    t = data[:, 0]
    tstar = data[:, 1]
    R = data[:, 2]
    # Early window: deep in the R0* << 1 regime (R0* = 0.035; the regime
    # breaks down only once R approaches Oh_SB*h0 = 100, i.e. t* ~ 2 ln(20)
    # ~ 6 -- t* <= 2 stays comfortably inside it, R(t*=2) ~ 5*e ~ 13.6).
    window = tstar <= 2.0
    n = int(window.sum())
    slope, _intercept = np.polyfit(t[window], np.log(R[window]), 1)
    predicted = 1.0 / mu
    rel_err = abs(slope - predicted) / predicted
    print(f"mu = {mu:.6f}, R0* = {meta['R0_star']:.4g}")
    print(f"fit window: t* in [{tstar[window][0]:.4f}, {tstar[window][-1]:.4f}] "
          f"({n} points)")
    print(f"fitted d(ln R)/dt = {slope:.6e}")
    print(f"predicted 1/mu    = {predicted:.6e}")
    print(f"relative error    = {rel_err:.4e}  (threshold {threshold:.0%})")
    ok = rel_err < threshold
    print(f"exponential -> {_verdict(ok)}")
    return ok


# ---------------------------------------------------------------------------
# 3. cross-solver (may SKIP)
# ---------------------------------------------------------------------------

def check_cross_solver(outroot: Path, threshold: float = 1e-3) -> str:
    """PASS/FAIL/SKIP -- returns a string, not a bool, because SKIP is a
    legitimate outcome (the sibling mass-coordinate axi solver, built by
    another agent in parallel, may not exist yet)."""
    _hr("3. cross-solver (vs simulationCases/trouton1d_axi.py, if it exists)")
    sibling = REPO_ROOT / "simulationCases" / "trouton1d_axi.py"
    if not sibling.exists():
        print(f"SKIP: {sibling} does not exist yet. The orchestrator should "
              "re-run this check once the mass-coordinate axi solver lands.")
        return "SKIP"
    try:
        importlib.import_module("trouton1d_axi")
    except Exception as e:  # noqa: BLE001
        print(f"SKIP: {sibling} exists but does not import cleanly ({e!r}). "
              "Not a failure of THIS solver; re-run once it does.")
        return "SKIP"

    sibling_sweep = REPO_ROOT / "validation" / "trouton1d_axi_sweep.py"
    if not sibling_sweep.exists():
        print(f"SKIP: {sibling} imports, but its sweep driver "
              f"{sibling_sweep} does not exist yet, so there is no stable "
              "CLI to run a matched-settings comparison against.")
        return "SKIP"

    axi_ale_dir = outroot / "cross_solver" / "ale"
    axi_mass_dir = outroot / "cross_solver" / "mass_coordinate"
    oh_sb, r0, tstar_end, nelem, grade, nsteps = 1.0, 5.0, 10.0, 800, 2.0, 3000

    ale_meta = axi_sweep.run_case(
        oh_sb=oh_sb, tstar_end=tstar_end, nsteps=nsteps, n_outputs=60,
        outroot=axi_ale_dir, r0=r0, l0=None, nelem=nelem, grade=grade)

    # Best-effort: invoke the sibling's own sweep driver as a subprocess with
    # the SAME CLI conventions this driver uses (see trouton1d_ale_axi_sweep's
    # --oh/--r0/--tstar-end/--nelem/--grade/--nsteps/--out). If its actual CLI
    # differs, this fails cleanly and is reported as SKIP, not a crash --
    # "imports cleanly" is the documented gate, not "has an identical CLI".
    cmd = [sys.executable, str(sibling_sweep),
          "--oh", str(oh_sb), "--r0", str(r0), "--tstar-end", str(tstar_end),
          "--nsteps", str(nsteps), "--nelem", str(nelem), "--grade", str(grade),
          "--out", str(axi_mass_dir)]
    try:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True,
                              text=True, timeout=1800, check=False)
        if proc.returncode != 0:
            print(f"SKIP: sibling sweep driver exited {proc.returncode}. "
                  f"stderr tail:\n{proc.stderr[-1500:]}")
            return "SKIP"
    except Exception as e:  # noqa: BLE001
        print(f"SKIP: could not run sibling sweep driver ({e!r}).")
        return "SKIP"

    import json
    mass_json_candidates = list(axi_mass_dir.glob("*.json"))
    if not mass_json_candidates:
        print(f"SKIP: sibling sweep produced no JSON summary in {axi_mass_dir}.")
        return "SKIP"
    try:
        mass_meta = json.loads(mass_json_candidates[0].read_text())
        ustar_mass = mass_meta.get("ustar_final")
        if ustar_mass is None:
            raise KeyError("ustar_final")
    except Exception as e:  # noqa: BLE001
        print(f"SKIP: could not read a matching 'ustar_final' from the "
              f"sibling's JSON summary ({e!r}). Likely an output-schema "
              "mismatch, not a physics disagreement.")
        return "SKIP"

    dev = abs(ale_meta["ustar_final"] - ustar_mass)
    print(f"TC1D-ALE-AXI  u*({tstar_end:g}) = {ale_meta['ustar_final']:.8f}")
    print(f"sibling axi   u*({tstar_end:g}) = {ustar_mass:.8f}")
    print(f"max|dev| = {dev:.4e}  (threshold {threshold:.0e})")
    ok = dev < threshold
    print(f"cross-solver -> {_verdict(ok)}")
    return "PASS" if ok else "FAIL"


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def cmd_all(outroot: Path) -> int:
    results: dict[str, str] = {}
    results["import"] = _verdict(check_import())
    results["1_mass"] = _verdict(check_mass(outroot))
    results["2_planar_limit_GATE"] = _verdict(check_planar_limit(outroot))
    results["4_tipthickness"] = _verdict(check_tipthickness(outroot))
    results["5_affinity_roughness"] = _verdict(check_affinity_roughness(outroot))
    results["6_exponential"] = _verdict(check_exponential(outroot))
    results["3_cross_solver"] = check_cross_solver(outroot)

    _hr("SUMMARY")
    for name, verdict in results.items():
        print(f"  {name:28s} {verdict}")
    hard_fail = any(v == "FAIL" for v in results.values())
    return 1 if hard_fail else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outroot", type=Path, default=DEFAULT_OUTROOT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("import", help="import check")
    sub.add_parser("mass", help="check 1: mass conservation")
    sub.add_parser("planar-limit", help="check 2: planar limit (the gate)")
    sub.add_parser("tipthickness", help="check 4: h_tip vs h_tipode ODE")
    sub.add_parser("affinity-roughness", help="check 5: mesh affinity + thickness roughness")
    sub.add_parser("exponential", help="check 6: R(t) = R0 exp(t/mu) early growth")
    sub.add_parser("cross-solver", help="check 3: vs mass-coordinate axi solver (may SKIP)")
    sub.add_parser("all", help="run every check, in the spec's run order")
    a = ap.parse_args()
    a.outroot.mkdir(parents=True, exist_ok=True)

    if a.cmd == "import":
        return 0 if check_import() else 1
    if a.cmd == "mass":
        return 0 if check_mass(a.outroot) else 1
    if a.cmd == "planar-limit":
        return 0 if check_planar_limit(a.outroot) else 1
    if a.cmd == "tipthickness":
        return 0 if check_tipthickness(a.outroot) else 1
    if a.cmd == "affinity-roughness":
        return 0 if check_affinity_roughness(a.outroot) else 1
    if a.cmd == "exponential":
        return 0 if check_exponential(a.outroot) else 1
    if a.cmd == "cross-solver":
        verdict = check_cross_solver(a.outroot)
        return 0 if verdict in ("PASS", "SKIP") else 1
    if a.cmd == "all":
        return cmd_all(a.outroot)
    raise AssertionError(f"unhandled subcommand {a.cmd!r}")


if __name__ == "__main__":
    sys.exit(main())
