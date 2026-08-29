#!/usr/bin/env python3
"""Emit the job list for the TC1D-FEM / ABDS-CN / theory validation campaign.

Writes a newline-delimited list of shell commands to stdout, one per case, for
``xargs -P``. Every command is self-contained and re-runnable, which is what
MANIFEST.md quotes. Nothing here runs a solver; it only decides parameters, so
the parameter policy is in one readable place.

RESOLUTION POLICY

  TC1D-FEM   dT = 5e-4 uniformly (nsteps = Tend/5e-4). BDF2 is second order in
             dt and the published convergence study (docs/planar-campaign/README.md) puts
             u*(30) at 3.9e-8 from the dt -> 0 limit at dT = 1.2e-3, so 5e-4 is
             comfortably inside the plateau. Mesh: the driver's auto_nelem
             holds the graded tip cell at the value validated by the 1D mesh
             study (M/N^2 = 7.5e-4), floor 400 elements.
  ABDS-CN    N = 4x and dT = 4x finer than the paper's own stated numerics
             ("typically 1000 grid points ... time step generally of order
             1e-4", appendix D). Their static-frame grid is uniform in
             X' = (L-X)/(L-Xc), so the PHYSICAL spacing is ~L/N while the
             free-edge diffusive layer is O(sqrt(T)) wide: resolving early time
             needs L/N << sqrt(T), which is why the early-time windows below
             carry a much larger N than the late-time ones at the same L. That
             is a property of their scheme, not of the port.

WINDOW SPLICING. Both codes write outputs geometrically in T for the early-time
sections, but a single run cannot span T = 1e-4 to 1 with a timestep that
resolves both ends. Each early-time case is therefore two runs -- [0, 1e-2] and
[0, 1] -- spliced at T = 1e-2, the same two-window construction
``ahsan_thinfilm._ahsan_inf_reference`` already uses.
"""
from __future__ import annotations

import math

ROOT = "$ROOT"
TC = ("$UV run python validation/ahsan_campaign.py tcfem")
AB = ("$UV run python validation/ahsan_campaign.py abds")

# TC1D-FEM timestep, in their T units.
DT_TC = 5e-4


def nsteps(Tend, dT=DT_TC):
    return int(round(Tend / dT))


jobs: list[str] = []


def tcfem(tag, out, L, Tend, *, spacing="geom", nout=200, prof=(), oh=1.0,
          dT=DT_TC, nelem=None, t0frac=1e-4):
    c = (f"{TC} --oh {oh:g} --L {L:g} --Tend {Tend:g} "
         f"--nsteps {nsteps(Tend, dT)} --n-outputs {nout} --spacing {spacing} "
         f"--t0-frac {t0frac:g} --out {ROOT}/{out}/case_{tag} --tag {tag}")
    if nelem:
        c += f" --nelem {nelem}"
    if prof:
        c += " --profile-T " + ",".join(f"{v:g}" for v in prof)
    jobs.append(c)


def abds_fin(tag, out, L, Tend, N, NT, *, nframe=4000, prof=()):
    c = (f"{AB} --mode fin --L {L:g} --Tend {Tend:g} --N {N} --NT {NT} "
         f"--nframe {nframe} --out {ROOT}/{out}/case_{tag} --tag {tag}")
    if prof:
        c += " --profile-T " + ",".join(f"{v:g}" for v in prof)
    jobs.append(c)


def abds_inf(tag, out, Tend, dx, dt, Lhat, *, nframe=4000, prof=()):
    c = (f"{AB} --mode inf --Tend {Tend:g} --dx {dx:g} --dt {dt:g} "
         f"--Lhat {Lhat:g} --nframe {nframe} --out {ROOT}/{out}/case_{tag} "
         f"--tag {tag}")
    if prof:
        c += " --profile-T " + ",".join(f"{v:g}" for v in prof)
    jobs.append(c)


# ===========================================================================
# SECTION 1 -- very early time, T << 1, finite L. Test of eq (5.10).
# ===========================================================================
# Their fig. 3 insets use L = 0.1, 1, 20, so those are the three values.
SEC1 = "sec1-early"
for L in (0.1, 1.0, 20.0):
    tag = f"L{L:g}".replace(".", "p")
    # window A: T in [1e-4, 1e-2];  window B: T in [1e-2, 1]
    # The graded mesh puts the tip cell at M/N^2; at L = 0.1 (M = 0.4) the
    # 400-element floor already gives 2.5e-6, far finer than the 7.5e-4 the
    # mesh study validated, so only the long domain needs extra elements.
    ne = 800 if L >= 10 else None
    tcfem(f"tcfem_{tag}_wA", SEC1, L, 1e-2, nout=80, t0frac=1e-2,
          dT=2.5e-7, nelem=ne)
    tcfem(f"tcfem_{tag}_wB", SEC1, L, 1.0, nout=120, t0frac=1e-2,
          dT=2.5e-5, nelem=ne)
    # ABDS-CN needs L/N << sqrt(T); window A reaches T = 1e-4, sqrt(T) = 1e-2.
    # dX = L/N must sit well inside the O(sqrt(T)) free-edge layer at the
    # BOTTOM of each window: sqrt(1e-4) = 1e-2 for window A, sqrt(1e-2) = 1e-1
    # for window B, and 10 cells across the layer is the target.
    NA = max(2000, int(math.ceil(L / 1e-3)))
    abds_fin(f"abds_{tag}_wA", SEC1, L, 1e-2, NA, 100000, nframe=4000)
    NB = max(2000, int(math.ceil(L / 2.5e-3)))
    abds_fin(f"abds_{tag}_wB", SEC1, L, 1.0, NB, 200000, nframe=4000)

# ===========================================================================
# SECTION 2 -- late time, L - Xc << 1. Test of eq (6.10).
# ===========================================================================
# Tend = max(40, 4L) puts L - Xc = L/(1+T) at ~0.12-0.25 for every L, so the
# small parameter of their expansion is comparable across the set rather than
# varying by a factor of 40 with L.
SEC2 = "sec2-late-finite"
for L in (1.0, 5.0, 20.0, 40.0):
    tag = f"L{L:g}"
    Tend = max(40.0, 4.0 * L)
    prof = (0.5 * Tend, Tend)
    tcfem(f"tcfem_{tag}", SEC2, L, Tend, spacing="uniform", nout=4000,
          prof=prof)
    abds_fin(f"abds_{tag}", SEC2, L, Tend, 4000, int(Tend / 2.5e-5),
             nframe=4000, prof=prof)
# One case pushed an order of magnitude further in T, to show the RATE at which
# (6.10) is approached rather than only that it is approached.
tcfem("tcfem_L5_deep", SEC2, 5.0, 400.0, spacing="uniform", nout=4000)
abds_fin("abds_L5_deep", SEC2, 5.0, 400.0, 2000, 8000000, nframe=4000)
# Their fig. 2(a-d): the finite-sheet profiles at L = 0.1 and L = 1.
tcfem("tcfem_fig2_L0p1", SEC2, 0.1, 1.0, spacing="uniform", nout=1000,
      prof=(0.2, 0.4, 0.6, 0.8, 1.0))
tcfem("tcfem_fig2_L1", SEC2, 1.0, 1.8, spacing="uniform", nout=1800,
      prof=(0.3, 0.6, 0.9, 1.2, 1.5, 1.8))
abds_fin("abds_fig2_L0p1", SEC2, 0.1, 1.0, 4000, 200000, nframe=1000,
         prof=(0.2, 0.4, 0.6, 0.8, 1.0))
abds_fin("abds_fig2_L1", SEC2, 1.0, 1.8, 4000, 360000, nframe=1800,
         prof=(0.3, 0.6, 0.9, 1.2, 1.5, 1.8))

# ===========================================================================
# SECTION 3 -- the main regime, L >> T >> 1. Tests (7.11), (3.10a), (7.13),
# (7.14), (7.17).
# ===========================================================================
SEC3 = "sec3-main"
PROF3 = (1.0, 2.0, 5.0, 10.0, 12.0, 20.0, 28.0, 40.0, 50.0)
# L = 200 at Tend = 50 gives Tc ~ L = 200, i.e. 4x beyond the window, and the
# far-end thickness drift is the check that this is enough.
tcfem("tcfem_Linf", SEC3, 200.0, 50.0, nout=400, prof=PROF3)
# dt refinement twin: same everything, half the timestep.
tcfem("tcfem_Linf_dt2", SEC3, 200.0, 50.0, nout=400, dT=DT_TC / 2)
# mesh refinement twin
tcfem("tcfem_Linf_mesh2", SEC3, 200.0, 50.0, nout=400, nelem=2066)
# Oh-independence: the reduced problem has NO parameter but L, so u*(t*) from
# TC1D-FEM must be independent of Oh_SB at fixed L. Oh_SB = 0.25 and 4 against
# the baseline 1, i.e. a factor of 16 (1.2 decades), which moves the code-unit
# mass M = 4 Oh_SB L, the viscosity and the timescale together.
for oh in (0.25, 4.0):
    tcfem(f"tcfem_L5_oh{oh:g}".replace(".", "p"), SEC3, 5.0, 40.0,
          spacing="uniform", nout=4000, oh=oh)
# ABDS-CN, both of their schemes.
abds_inf("abds_Linf", SEC3, 50.0, 0.01, 2.5e-5, 80.0, nframe=4000, prof=PROF3)
abds_inf("abds_Linf_coarse", SEC3, 50.0, 0.02, 5e-5, 80.0, nframe=4000)
abds_fin("abds_L200", SEC3, 200.0, 50.0, 8000, 1000000, nframe=4000,
         prof=PROF3)

# ===========================================================================
# SECTION 4 -- their fig. 5: Tc(L) and Delta Tc(L).
# ===========================================================================
# Their fig. 5 spans L ~ 5 to 50 in ~19 log-spaced points; the brief asks for 3
# to 50. Tend = max(40, 3L) keeps the deceleration peak (at T ~ L) and its full
# width inside the record for every L.
SEC4 = "sec4-fig5"
L4 = [3.0, 3.7, 4.6, 5.7, 7.0, 8.7, 10.7, 13.2, 16.3, 20.2, 25.0,
      30.9, 38.2, 47.2, 50.0]
for L in L4:
    tag = f"L{L:g}".replace(".", "p")
    Tend = max(40.0, 3.0 * L)
    # 6000 uniform samples -> dT_out <= 0.025, and Delta T_c ~ 2.5, so the
    # half-maximum crossing is located to ~1% of the width.
    tcfem(f"tcfem_{tag}", SEC4, L, Tend, spacing="uniform", nout=6000)
    abds_fin(f"abds_{tag}", SEC4, L, Tend, 2000, int(Tend / 5e-5),
             nframe=6000)

# EVERY case must own its --out directory. The runner deletes that directory
# after promoting the case's CSVs, so a job whose --out is a bare section
# directory deletes the whole section -- which is exactly what two `abds
# --mode inf` lines did on the first full run, taking a 9-minute finished run
# and three in-flight ones with them. Assert it here rather than trust three
# f-strings to stay in step.
import re as _re  # noqa: E402

_seen = {}
for _j in jobs:
    _o = _re.search(r"--out (\S+)", _j).group(1)
    _t = _re.search(r"--tag (\S+)", _j).group(1)
    assert _o.endswith("/case_" + _t), f"--out is not case-private: {_o}"
    assert _o not in _seen, f"--out collides between {_t} and {_seen[_o]}"
    _seen[_o] = _t

if __name__ == "__main__":
    print("\n".join(jobs))
