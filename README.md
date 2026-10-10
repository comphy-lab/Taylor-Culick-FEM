# Taylor-Culick-FEM

Sharp-interface finite-element simulations of Taylor–Culick sheet retraction,
solved in a frame co-moving with the retracting rim. Built on
[pyoomph](https://github.com/comphy-lab/pyoomph) (a Python front-end to
oomph-lib), pinned to the immutable CoMPhy release `comphy-2026.09.07.1`
(commit `a186e07e`).

## Problem

A liquid sheet of thickness `h0`, density `rho`, surface tension `sigma` and
dynamic viscosity `mu` retracts from a free edge. Lengths are scaled on the
**full** thickness `h0`, velocities on `sqrt(sigma / (rho h0))` and times on
`sqrt(rho h0^3 / sigma)`, so `rho = sigma = h0 = 1` and the half-thickness is
`0.5`. The Ohnesorge number is `Oh = mu / sqrt(2 h0 rho sigma)` (the Savva &
Bush convention, the only `Oh` in this repository); the coded viscosity is
`mu = sqrt(2) Oh`. The Taylor–Culick speed is

```
V_TC = sqrt(2 sigma / (rho h0))  ->  sqrt(2) = 1.41421...
```

and results are reported as `u* = U / V_TC`, where `U` is the tip speed.

Three configurations share one formulation:

1. **Planar sheet.** A semi-infinite two-dimensional sheet retracts into a rim.
   There is no steady state: the rim collects mass and grows while the tip
   speed approaches `V_TC`.
2. **Axisymmetric hole.** A circular hole of initial radius `R0 = h0` opens in a
   flat sheet. The azimuthal curvature adds a correction that decays as
   `h0/R`, so `V_TC` is approached only asymptotically.
3. **Elastic axisymmetric hole.** The sheet carries the extra stress of an
   Oldroyd-B liquid in the `G`–`lambda` parameterisation,
   `tau_p = G (A - I)`, `upper-convected(A) = -(A - I)/lambda`, with
   elastocapillary number `Ec = G h0 / sigma`. `lambda = inf` removes the
   relaxation term exactly, giving a neo-Hookean solid with a Newtonian solvent
   viscosity (the Kelvin–Voigt limit).

## Layout

| Path | Contents |
|---|---|
| `src/taylor_culick_fem/` | Library code: the problem definitions (planar, axisymmetric laboratory and co-moving, elastic axisymmetric, one-dimensional Trouton models), the independent comparison solvers, the elastic driver, the far-field membrane and absorbing layer, the case validator (`cases.py`), restart lineage (`lineage.py`) and the run-provenance helper (`provenance.py`). |
| `simulationCases/` | The entry point `run_axisymmetric_viscoelastic.py`, its default case `axisymmetric_viscoelastic.json` and the case-schema table. |
| `validationCases/<study>/` | Comparisons with independent published numerical solutions: their drivers, digitised reference data with its extraction record, and the comparison scripts. |
| `verificationCases/<study>/` | Comparisons between independent discretisations in this repository, and solver construction checks. |
| `modelComparisonCases/<study>/` | Comparisons between the two-dimensional solution and the one-dimensional reduced model. |
| `postProcess/` | Tools that act on run outputs: histories, analysis and animations. |
| `testCases/` | Software tests. |

Simulation output stays outside this repository. A published figure states the
solver commit and the pinned pyoomph version that produced it; reproducing it
means rerunning the simulation with that commit and version.

## Method

The interface is a sharp free surface on a moving (ALE) mesh with no gas
phase. The rim speed `U` is a global unknown fixed by a Lagrange multiplier
that pins the rim in the computational window, so the mesh covers a window of
length `L` around the rim rather than the whole retraction path. The window is
implemented as ALE mesh motion with the physical radius retained in the
axisymmetric metric, because a radial translation is not a Galilean
transformation. Time integration is BDF2 with adaptive steps controlled by a
temporal error tolerance; the mesh is rebuilt with Gmsh when its quality
degrades, and linear systems are solved with PARDISO.

The elastic case solves the conformation in log form with streamline-upwind
stabilisation. At the window's far edge the film is either held at rest, or
coupled to a one-dimensional axisymmetric membrane with the same constitutive
law that carries the sheet beyond the window.

## Running

[uv](https://docs.astral.sh/uv/) owns the environment. Python is pinned by
`.python-version` and dependencies are locked in `uv.lock`; `uv sync` performs a
source build of pyoomph (CMake compiles the vendored oomph-lib and builds CLN and
GiNaC), which needs a C/C++ toolchain, CMake, MPI and the Python headers. pyoomph
is pinned to a release commit in `pyproject.toml`; change the pin only
deliberately, to another release tag, and commit the updated `uv.lock` with it.

```bash
uv sync
uv run python simulationCases/run_axisymmetric_viscoelastic.py \
    simulationCases/axisymmetric_viscoelastic.json --out <new folder>
uv run python simulationCases/run_axisymmetric_viscoelastic.py \
    simulationCases/axisymmetric_viscoelastic.json --out <new folder> --check
uv run python -m pytest testCases
```

The runner validates the case, writes `run-manifest.json` before any other
output and refuses an output folder that already holds anything. A continuation
uses the same case with `--restart-from <run>/checkpoints/ckpt_NNNNN.dump` into
its own new folder. `simulationCases/README.md` lists the case fields; each
study folder's README says how to run that study. Set `HWLOC_COMPONENTS=-gl`
where hwloc could probe an unresponsive graphics display while MPI starts.

## Reference

- Basilisk implementation of the same problem:
  [comphy-lab/Taylor-Culick-ViscoElastic](https://github.com/comphy-lab/Taylor-Culick-ViscoElastic)
- Taylor, G. I. (1959) *Proc. R. Soc. A* **253**, 313.
- Culick, F. E. C. (1960) *J. Appl. Phys.* **31**, 1128.
- Savva, N. & Bush, J. W. M. (2009) *J. Fluid Mech.* **626**, 211.
