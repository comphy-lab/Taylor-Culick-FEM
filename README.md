# Taylor-Culick-FEM

Sharp-interface finite-element study of Taylor–Culick sheet retraction, solved in
a frame co-moving with the retracting rim.

Built on [pyoomph](https://github.com/comphy-lab/pyoomph) (a Python front-end to
oomph-lib), tracked against the CoMPhy fork's `develop` branch.

> The documentation index is [`docs/README.md`](docs/README.md).
> Inertial FEM versus Basilisk (`Oh < 0.1`) is
> [`docs/validation-inertial/`](docs/validation-inertial/) (planar closed).
> The viscous Trouton/Ahsan note is
> [`docs/validation-viscous/`](docs/validation-viscous/).
> The full planar Newtonian write-up, including the `√2` convention mismatch
> and the comparison against Savva & Bush (2009), is
> [`docs/planar-campaign/`](docs/planar-campaign/). That write-up is intended
> as an extension of Appendix A of
> [Sanjay *et al.*, JFM 948 (2022) A14](https://arxiv.org/abs/2202.13400) to
> arbitrary `Oh`, with FEM in place of Basilisk.

## Why FEM here

The existing Basilisk study of this problem
([comphy-lab/Taylor-Culick-ViscoElastic](https://github.com/comphy-lab/Taylor-Culick-ViscoElastic))
is a volume-of-fluid, two-phase calculation on a domain 100 sheet-thicknesses
wide. Two things motivate the FEM version:

1. **True free surface.** pyoomph resolves the interface sharply with a moving
   (ALE) mesh and no gas phase, so there is no density/viscosity ratio, no
   diffuse interface, and no spurious-current budget to defend.
2. **Co-moving frame.** The rim velocity is promoted to a *global unknown* `U`
   fixed by a Lagrange multiplier that pins the rim in place. The mesh then only
   has to cover a window around the rim rather than the whole retraction path,
   which is what forced the large Basilisk domain. The recovered control `U(t)`
   is the physical observable: it should approach the Taylor–Culick velocity.

## Physical setup

Non-dimensionalised on the sheet's **full** thickness `h0`, the liquid density
`rho`, and surface tension `sigma`:

| Scale | Definition |
|---|---|
| length | `h0` |
| velocity | `sqrt(sigma / (rho h0))` |
| time | `sqrt(rho h0^3 / sigma)` |

so `rho = sigma = h0 = 1` and the half-thickness is `0.5`. The single
dimensionless group is the Ohnesorge number `Oh = mu / sqrt(2 h0 rho sigma)`
(Savva & Bush convention -- the only `Oh` in this repository), matched to the
Basilisk study at coded viscosity `mu = 0.05`, i.e. `Oh = 0.035`.

The target is the Taylor–Culick velocity

```
V_TC = sqrt(2 sigma / (rho h0))  ->  sqrt(2) = 1.41421...
```

## Cases

### 1. Planar (`simulationCases/planar_comoving.py`) — validation

A semi-infinite two-dimensional sheet retracting into a rim. The control
velocity must approach `sqrt(2)`.

There is **no true steady state**, even in the co-moving frame: the rim
collects mass at rate `rho h V` and grows without bound, which is exactly how
the classical result arises (`d(mV)/dt = 2 sigma` with `m = rho h V t` gives
constant `V` and growing `m`). A stationary solve is therefore ill-posed and
diverges; the driver attempts one only as an optional extra.

Run it first, because it can be checked against published data. Savva & Bush,
JFM **626** (2009), figure 6 gives the edge velocity against `t* = t/tau_vis`.
At `Oh = 0.5` this case gives

| | `U/V_TC` at `t* = 30` |
|---|---|
| this code | **0.9297** |
| Savva & Bush fig. 6 | ~0.93 |

Two convention traps in that comparison: `Oh = mu/sqrt(2 H rho gamma)` carries
a `sqrt(2)` that the coded viscosity does not (coded `mu = 0.05` is
`Oh = 0.035`), and
`tau_vis = mu H/(2 gamma)` is the *viscous* clock, onto which their curves
collapse only for `Oh >~ 0.5`. At lower `Oh` the approach is inertial and set
by rim growth in capillary time, so `t*` is not a useful target.

### 2. Axisymmetric (`simulationCases/axisymmetric_comoving.py`) — the physical problem

A hole opening in a flat sheet: the rim is a torus expanding radially, matching
the Basilisk configuration. Here `U(t)` approaches `sqrt(2)` only
**asymptotically**, because the azimuthal curvature contributes a correction
that decays as `h0/R`. There is no exact steady state, so convergence of the
control is the result rather than a fixed point.

Note that a radial translation is *not* a Galilean transformation in
axisymmetry — the `1/r` metric terms do not translate. The co-moving frame is
therefore implemented as ALE mesh motion with the true physical radius retained
in the metric, not as a change of variables. Schematics and the constraint
equations are in
[`docs/axi-comoving-window.html`](docs/axi-comoving-window.html).

## Layout

```
simulationCases/        runnable problem definitions (2D co-moving, 1D Trouton)
validation/             validation drivers, convergence studies, digitised data
postProcess/            analysis and figure generation
campaign/               the exact scripts used to run the Oh sweep on Stokes
docs/                   public write-up tree (index, viscous, inertial, planar campaign, axi solvers)
```

Simulation output does not live here. Runs write to the project run root on the
compute host; custody is recorded in the project tracker.

## Environment

Python is pinned by `.python-version` and dependencies are locked in
`uv.lock`. [uv](https://docs.astral.sh/uv/) owns the environment:

```bash
uv sync                    # build the environment (compiles pyoomph from source)
uv run python -m pyoomph check all
uv run python simulationCases/planar_comoving.py
```

`uv sync` performs a **source build** of pyoomph: CMake compiles the vendored
oomph-lib and downloads/builds CLN and GiNaC. Expect the first build to take a
while. System prerequisites on Ubuntu:

```bash
sudo apt-get install -y build-essential cmake python3-dev pkg-config \
    libopenmpi-dev wget bzip2 patch flex bison
```

To move to a newer pyoomph `develop` commit:

```bash
uv sync --upgrade-package pyoomph
```

### Python version

Pinned to **3.13**. pyoomph itself supports 3.10–3.14 (it ships an abi3 wheel
covering 3.12+; the 3.10–3.13 trove classifiers in its `pyproject.toml` are
stale). The pin is set by the *runtime* dependencies — `gmsh` and `mkl` are the
components most likely to lag on a brand-new CPython — and by the fact that the
pyoomph devcontainer targets this range. The `<3.14` upper bound in
`pyproject.toml` is deliberate; raise it only after checking the whole
dependency set resolves and `pyoomph check all` passes.

## Reference

- Basilisk implementation: [comphy-lab/Taylor-Culick-ViscoElastic](https://github.com/comphy-lab/Taylor-Culick-ViscoElastic)
- Taylor, G. I. (1959) *Proc. R. Soc. A* **253**, 313.
- Culick, F. E. C. (1960) *J. Appl. Phys.* **31**, 1128.
