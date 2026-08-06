# Taylor-Culick-FEM

Sharp-interface finite-element study of Taylor–Culick sheet retraction, solved in
a frame co-moving with the retracting rim.

Built on [pyoomph](https://github.com/comphy-lab/pyoomph) (a Python front-end to
oomph-lib), tracked against the CoMPhy fork's `develop` branch.

## Why FEM here

The existing Basilisk study of this problem
([comphy-lab/Elastic-Taylor-Culick](https://github.com/comphy-lab/Elastic-Taylor-Culick))
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
dimensionless group is the Ohnesorge number `Oh = mu / sqrt(rho sigma h0)`,
matched to the Basilisk study at `Oh = 0.05`.

The target is the Taylor–Culick velocity

```
V_TC = sqrt(2 sigma / (rho h0))  ->  sqrt(2) = 1.41421...
```

## Cases

### 1. Planar (`cases/planar_comoving.py`) — validation

A semi-infinite two-dimensional sheet retracting into a rim. In the co-moving
frame this problem has a **genuine steady state**, and the control velocity must
equal `sqrt(2)` exactly. That makes it a sharp test of the Lagrange-multiplier
constraint: any error in the constraint shows up directly as an offset in `U`,
with no asymptotics to hide behind.

Run it first. If `U` does not sit on `sqrt(2)`, nothing downstream is
trustworthy.

### 2. Axisymmetric (`cases/axisymmetric_comoving.py`) — the physical problem

A hole opening in a flat sheet: the rim is a torus expanding radially, matching
the Basilisk configuration. Here `U(t)` approaches `sqrt(2)` only
**asymptotically**, because the azimuthal curvature contributes a correction
that decays as `h0/R`. There is no exact steady state, so convergence of the
control is the result rather than a fixed point.

Note that a radial translation is *not* a Galilean transformation in
axisymmetry — the `1/r` metric terms do not translate. The co-moving frame is
therefore implemented as ALE mesh motion with the true physical radius retained
in the metric, not as a change of variables.

## Layout

```
cases/                  runnable problem definitions
src/taylor_culick_fem/  shared equations, meshes, diagnostics
postProcess/            analysis and figures
tests/                  regression tests
```

## Environment

Python is pinned by `.python-version` and dependencies are locked in
`uv.lock`. [uv](https://docs.astral.sh/uv/) owns the environment:

```bash
uv sync                    # build the environment (compiles pyoomph from source)
uv run python -m pyoomph check all
uv run python cases/planar_comoving.py
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

- Basilisk implementation: [comphy-lab/Elastic-Taylor-Culick](https://github.com/comphy-lab/Elastic-Taylor-Culick)
- Taylor, G. I. (1959) *Proc. R. Soc. A* **253**, 313.
- Culick, F. E. C. (1960) *J. Appl. Phys.* **31**, 1128.
