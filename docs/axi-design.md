# Axisymmetric Taylor-Culick: design note

Feature branch `kelvin/axi-taylor-culick`. Scope: viscous (Newtonian)
axisymmetric hole-opening variants of the depth-averaged solvers (mass
coordinate and ALE) and of the 2D sharp-interface DNS. The guiding choice,
set by Vatsal, is a **metric formulation**: the grids stay one- and
two-dimensional exactly as in the planar codes, and axisymmetry enters through
the `r`-weighted measure and the hoop terms of the governing equations.

Reference: Savva & Bush, JFM 626 (2009), section 3 and the Appendix — their
numerical model *is* the axisymmetric system; the planar case is the aside.
Equation numbers below are theirs. All forms below were re-derived from first
principles and checked term-by-term against their (3.15)-(3.16); the planar
limit reproduces the residuals coded in `simulationCases/trouton1d_pyoomph.py`
and `trouton1d_ale.py` exactly.

## Non-dimensionalisation

Unchanged from planar: `rho = sigma = h0 = 1` with `h0` the FULL thickness,
`V_TC = sqrt(2)`, coded viscosity `mu = sqrt(2) * Oh_SB`, viscous clock
`tau_vis = mu/2`, `t* = t/tau_vis`. One new parameter enters: the initial hole
radius `R0` (in units of `h0`). The Savva-Bush axisymmetric self-similarity
variable at high Oh is `R0* = R0/(Oh_SB h0)` — curves with equal `H Oh/R0`
collapse (their section 3.3.1), which is a validation lever.

## Governing equations (Eulerian, membrane form)

Radial coordinate `r`, hole edge `R(t)`, film on `r > R`. Depth-averaged
radial velocity `u(r,t)`, full thickness `h(r,t)`. Plane-stress (sigma_zz = 0)
viscous membrane stresses:

    sigma_rr     = 2 mu (2 u_r + u/r)
    sigma_thth   = 2 mu (u_r + 2 u/r)
    N_rr = h sigma_rr,   N_thth = h sigma_thth

    h_t + (1/r) (r u h)_r = 0                                   (mass, = 3.15)
    h (u_t + u u_r) = dN_rr/dr + (N_rr - N_thth)/r              (momentum)

Expanding the momentum RHS gives

    4 nu h (u_rr + u_r/r - u/r^2) + 2 nu h_r (2 u_r + u/r),

which is exactly S&B (3.16) multiplied through by `h` (their form
`(4nu/h)[((h/r)(ur)_r)_r - u h_r/(2r)]` — verified by expansion). Two terms
are new relative to planar: the metric weight in the mass flux and stress
divergence, and the hoop stress `N_thth/r`. Both vanish as `h0/R -> 0`;
the planar operator is the exact `R0 -> infinity` limit.

The distributed capillary term `(gamma/rho) kappa_r` (S&B 3.16, with kappa
from their 3.5 including the azimuthal `h_r/(2r)` component) stays **off** by
default, exactly as in planar: the tip closure below carries the capillary
driving, and a bulk `kappa_r` would double-count it. The `f = h^2` azimuthal
curvature form, S&B (A1), is recorded in `trouton1d_pyoomph.py` for the day
capillarity is revisited.

## Tip closure

Leading order, unchanged from planar: the massless-cap force balance
`N_rr(R) = -2 gamma_total = -2`, i.e. `2 mu h (2 u_r + u/r)|_tip = -2`.
S&B's axisymmetric jump condition (3.20) is the same `[u_r] = -1/4` as the
planar (2.22), so `TipFlux`/`TipClosure` keep their constant `-2` — under an
`r`-weighted measure the boundary term automatically carries the factor
`r_tip = R`, giving the per-radian pull `2 gamma R`.

First geometric correction (optional, behind a flag, default off): the
toroidal rim surface opposes opening with force `gamma pi h_tip / (2R)` per
unit edge length, so

    N_rr(R) = -2 + pi h_tip / (2 R) .

Derived two ways (surface-energy release rate of the cap, and integrating the
azimuthal Laplace pressure `gamma/R` over the cap arc `pi h_tip/2`); it is the
O(h/R) effect S&B neglect in their early-time analysis (their 3.31 requires
`H/R0 << 1`). With `R0 = h0 = 1` this correction is O(1) at t = 0, so the
flag exists to *measure* its importance against the DNS, not to assert it.

## Mass-coordinate form (axi TC1D-FEM)

Per-radian mass coordinate `dm = r h dr`, tip at `m = 0`, fixed domain
`[0, M]`. New relative to planar: the radius `r(m,t)` must be carried as an
unknown, because it appears in every residual. This removes the planar
pathology where a carried `x` had dead Jacobian columns — here `r` is fully
coupled. With `d/dr = r h d/dm`:

    r_t|_m = u                          (pointwise ODE, C1 field)
    h_t|_m = -h (r h u_m + u/r)         (pointwise in h: D0 survives)
    u_t|_m = d/dm (r h sigma_rr) - sigma_thth / r
    sigma_rr   = 2 mu (2 r h u_m + u/r)
    sigma_thth = 2 mu (r h u_m + 2 u/r)

Weak form (test v on C1, measure stays cartesian in `m`; the metric is
explicit):

    int u_t v dm = - int (r h sigma_rr) v_m dm - int (sigma_thth/r) v dm
                   + 2 R v(0)          [tip: -(r h sigma_rr)|_0 = +2R]

Far field `m = M`: `u = 0` Dirichlet (h has no nodal value on D0, as in
planar). Consistency identity: `r_m = 1/(r h)` follows from mass
conservation; enforcing `r_t = u` and checking `r_m - 1/(rh)` drift is a
free domain diagnostic, as is total mass (exact by construction).

Staggering carries over untouched: `h` on D0, `u` on C1, `r` on C1, no
midside dofs anywhere. `with_capillarity=True` stays rejected for D0 `h`.

Initial condition: `h = 1`, `u = 0`, `r(m) = sqrt(R0^2 + 2m)` (the exact map
for uniform h). This is the canonical-rupture analogue; `R0 = 0` is excluded
because the hole then has zero edge circumference and the `u/r`, `1/r` terms
are singular at `m = 0` — a point hole does not open in this model
(Taylor & Michael 1973 closure aside; S&B section 3 opens with this).

## ALE physical-coordinate form (axi TC1D-ALE)

Domain `r in [R(t), R_out]`, `LaplaceSmoothedMesh` affine stretch — which is
precisely S&B's mapping (A2). Residuals (measure `r dr`; either via the
axisymmetric coordinate system or an explicit weight — decided at
implementation against pyoomph's 1D-mesh support for coordinate systems):

    mass:      weak( h_t|ALE + (u - w) h_r + h (u_r + u/r), ht )
    momentum:  weak( h (u_t|ALE + (u - w) u_r), ut )
               + weak( N_rr, d(ut)/dr )        [flux slot, N_rr = 2 mu h (2u_r + u/r)]
               + weak( N_thth / r, ut )        [hoop term, explicit]
    tip:       TipClosure -2 (natural term picks up r_tip from the measure)
    kinematic: EnforcedBC(mesh_x = mesh_velocity - u) @ tip   (unchanged)
    far:       DirichletBC(u = 0, mesh_x pinned)              (unchanged)

Note the flux slot holds the FULL `N_rr` including its `u/r` part; only the
hoop term lives outside the integration by parts. C1/C1 co-location stays
safe (no midside nodes on a linear line mesh).

## Internal exact checks (tests)

1. **Planar-limit regression**: axi with `R0 ~ 100` must track the planar
   solver over an early window with `O(1/R0)` error; the error must fall
   linearly in `1/R0` across `R0 = {50, 100, 200}`. (Do not run `R0 = 1e6`:
   the `r^2 h^2 u_m` flux then carries ~1e12 scale factors; conditioning,
   not physics, would be measured.)
2. **Tip-thickness identity**: the closure implies, exactly,
   `dh_tip/dt = 1/(2 mu) - h_tip u_tip / (2 r_tip)`; planar limit recovers
   `h_tip = 1 + t*/4`. Integrable alongside the run and comparable to the
   solver's own tip state — the axi analogue of the `h_tip_exact` column.
3. **Mass conservation**: exact in `m`; measured drift in ALE.
4. **High-Oh early growth**: `R(t) = R0 exp(t / (2 tau_vis))` (S&B 3.30) at
   `Oh_SB >> 1`, `t <~ tau_vis` — measured exponential rate.
5. **Self-similarity**: matched `Oh h0/R0` at high Oh collapses `u*(t*)`
   (S&B section 3.3.1).
6. **Late-time**: `dR/dt -> sqrt(2)` with a correction decaying in `h0/R`;
   the planar late-time coefficients (3/2, `sqrt(Oh/t*)` law, 0.98073) are
   planar-only and are NOT axi targets (`docs/validation-1d`, and the axi
   system admits a purely viscous solution).

## 2D DNS (lab-frame axisymmetric ALE on a graded annulus)

Primary design: **lab frame, fixed outer boundary, graded annular mesh**.
pyoomph's built-in `axisymmetric` coordinate system is already the metric
formulation: the mesh stays 2D in `(r, z)`, the measure gains `2 pi r`, the
operators gain the hoop terms, and `NavierStokesFreeSurface`'s
surface-divergence form carries BOTH principal curvatures with no curvature
code written anywhere. Relative to `planar_comoving.py`:

- Domain `r in [R(t), R_out]`, midplane `z = 0`. The hole behind the rim is
  not meshed — the liquid domain *shrinks* as the hole opens. The quiescent
  far film ahead of the disturbance needs only ~2 elements across the half
  thickness (size ~0.25), so a graded mesh out to `R_out ~ 400` costs less
  than the planar `L = 160` campaign runs already completed.
- Lab-frame velocities: far field quiescent, all-plain
  `DirichletBC(mesh_x = R_out, mesh_y=True, velocity_x=0, velocity_y=0)`.
  No `-U` inflow, no bulkforce (a radial translation is not Galilean, and
  none is needed), no global dofs, no `EnforcedDirichlet`, no tip pinning
  constraint. Neither recorded pyoomph trap can fire.
- `R(t)` is a pure observable: the tip node's `mesh_x` at each output;
  `dR/dt` by differentiation.
- Mesh smoothing stays `HyperelasticSmoothedMesh()` with its default
  cartesian metric (deliberate in pyoomph; do not "fix" it).
- Initial geometry: annular translate of the planar mesh — tip `(R0, 0)`,
  cap semicircle radius `1/2` centred `(R0 + 1/2, 0)`, shoulder
  `(R0 + 1/2, 1/2)`, film to `(R_out, 1/2)`; `R0 = 1` matches the Basilisk
  reference IC (`hole0 = 1`, cap radius `h0/2`) exactly.
- Exact global volume conservation (no flux through any boundary) is a free,
  sharp correctness diagnostic — encode it as a monitored observable.
- `R_out` sizing reuses the planar disturbance-footprint rule
  (`run_validation.py`), plus the full tip travel.

Fallback, only if `t_end` must grow beyond the graded-annulus budget: a
moving window `[X(t), X(t) + L]` with one bookkeeping global dof `X` slaved
to the tip via `WeakContribution(mesh_x - X, Xtest,
coordinate_system=cartesian)` and `EnforcedDirichlet(mesh_x = X + L)` at the
far boundary. This reintroduces the finite-window clamp (the dominant planar
error) and both recorded traps; it is not the default. The shifted-metric
co-moving variant (measure depending on a global unknown) was assessed and
rejected: implementable, but it routes the azimuthal curvature — the very
quantity under study — through unexercised custom machinery.

Smoke ladder before any long run (in order, all must pass):

1. **S0 hoop-curvature discriminator**: static quarter-disc droplet, axis at
   `r = 0`, stationary solve; require `p = 2 sigma/a` (axisymmetric) vs
   `sigma/a` (planar) — a factor-2 check that the azimuthal curvature is on.
2. **S1 FD Jacobian**: `debug_jacobian_by_fd_epsilon` over a few steps of
   the real case at coarse resolution.
3. **S2 volume conservation**: relative drift < 1e-8 over a coarse run.
4. **S3 planar reduction**: `R0 = 100`, matched Oh; `dR/dt` must track the
   validated planar `U(t)` to `O(h0/R)`.
5. **S4 physics**: `R0 = 1`; `dR/dt > 0` from `t = 0+`, approaching
   `sqrt(2)` from below with a residual decaying like `1/R`.

## Validation targets (external)

- Basilisk `Elastic-Taylor-Culick/simulationCases/TaylorCulick.c` run
  Newtonian (`G1 = 0`): same IC, `dR/dt -> sqrt(2)`.
- 2024 campaign CSV `Ec0.000_v1202_2024.csv` (t, rTip, vTip to t = 300,
  plateau 1.4134): direct vTip(t) comparison for the DNS.
- Sanjay, Sen, Kant & Lohse, JFM 948 (2022) / arXiv:2202.13400 (axisymmetric,
  `We_f -> 1` plateau; note their `Oh` carries the `2 gamma` convention:
  `Oh_paper = Oh_code/sqrt(2)`).
- Savva & Bush axisymmetric figures (section 3.3) — to be digitised; no
  axisymmetric digitised curves exist in `validation/data/` yet.
- 1D-vs-DNS: matched `R0 = 1`, matched `Oh_SB`, compare `R(t)` and
  `u*(t*)`; expect the leading early-time discrepancy to be the O(h/R)
  azimuthal edge correction, measurable via its flag.

## File structure

Sibling files, not flags (the planar classes already carry `staggered`,
`variable`, `with_capillarity`; the DNS changes are structural):

    simulationCases/trouton1d_axi.py       # mass-coordinate axi (D0/C1 + r on C1)
    simulationCases/trouton1d_ale_axi.py   # ALE axi
    simulationCases/axisymmetric_lab.py    # 2D DNS, lab-frame graded annulus
    validation/trouton1d_axi_sweep.py
    validation/trouton1d_ale_axi_sweep.py
    validation/run_validation_axi.py       # mirrors run_validation.py columns
    validation/axi_smoketests.py           # smoke ladder S0-S2 (+ quick S4 probe)

`TipFlux`/`TipClosure` are imported from the planar modules (geometry-
independent constants). Output tables keep the planar column prefix and
append new columns (`compare_1d_2d.py` indexes positionally).
