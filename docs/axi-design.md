# Axisymmetric Taylor-Culick: design note

Formulation authority for the four axisymmetric solvers. The inertial
FEM–Basilisk *campaign* is `docs/validation-inertial/`; the solver
construction checks are `docs/axi-validation/`. This file is the design
note those documents point at.

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
`[0, M]`. New relative to planar: the radial metric must be carried as an
unknown, because it appears in every residual. Carry `w = r^2/2` on C1 (not
`r` itself), constrained by

    w_m = 1/h        (both sides cell-constant under the staggering,
                      so the discrete radial map is EXACTLY consistent
                      with the discrete mass distribution, cell by cell)

with `r = sqrt(2w)`, plus one global scalar `R(t)`:

    partial_t(R) - u|_{m=0} = 0        (point contribution at the tip)
    w|_{m=0} - R^2/2 = 0               (EnforcedBC — the value depends on
                                        the unknown R; never DirichletBC)

This supersedes evolving `r_t = u` pointwise, which is only consistent in
the continuum. Governing residuals (`rho = gamma_surface = 1`):

    h_t|_m = -h (r h u_m + u/r)         (pointwise in h: D0 survives)
    u_t|_m = d/dm (r h sigma_rr) - sigma_thth / r
    sigma_rr   = 2 mu (2 r h u_m + u/r)
    sigma_thth = 2 mu (r h u_m + 2 u/r)

Momentum weak form (test v on C1, cartesian measure in `m`, metric
explicit), expanded so that the ONLY reciprocal is one guarded `1/r^2`
(implementing `sigma_thth/r` literally introduces a spurious `1/r` that
cancels analytically but not numerically):

    int u_t v dm
    + int [ 4 mu r^2 h^2 u_m + 2 mu h u ] v_m dm     [flux slot]
    + int [ 2 mu h u_m + 4 mu u / r^2 ] v dm          [hoop remainder]
    - 2 R v(0)  = 0                                   [tip pull, nonlinear in R]

Tip term sign convention matches planar `TipFlux` (value -2, natural
boundary term of the flux slot), multiplied by the unknown `R`. Far field
`m = M`: `u = 0` Dirichlet (h has no nodal value on D0, as in planar).

Staggering carries over untouched: `h` on D0, `u` and `w` on C1, no midside
dofs anywhere. `with_capillarity=True` stays rejected for D0 `h`; if
capillarity is ever wanted in axi, switch to `f = h^2` at the same time
(S&B A1; both curvature terms bounded at the tip, azimuthal part -> 1/r).

Initial condition: `h = 1`, `u = 0`, `w(m) = (R0^2 + 2m)/2` (exact map for
uniform h), `R = R0`. This is the canonical-rupture analogue; `R0 = 0` is
excluded on four independent grounds: S&B's early solution R(t) = R0
exp(t/2 tau_vis) is linear in R0 (a zero hole never opens); the mass map
degenerates (dr/dm -> infinity); the edge force and rim mass both vanish
(0/0 tip acceleration); and the azimuthal curvature 1/R diverges — below
the Taylor & Michael (1973) threshold at R ~ h0 the hole closes. Production
1D runs should use `R0 >= 5` (S&B's own smallest is 5H); `R0 ~ 1` sits at
~2x the closing threshold and is reserved for the DNS comparison, where the
volume-matched radius below applies.

Conditioning watch: the flux slot carries `r^2` (a 1e4-1e6 spread across a
long domain), on top of the planar `1/dm^2` stiffness. If Newton degrades
as R grows, scale the momentum test space or non-dimensionalise `r` by
`R(t)` before reaching for smaller timesteps.

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

1. **Metric consistency**: `w_m - 1/h = 0` per cell at machine precision
   (the structural advantage of the staggering — if this drifts, the
   discretisation of `w` is wrong).
2. **Planar-limit regression**: axi with `R0 ~ 100` must track the planar
   solver over an early window with `O(1/R0)` error; the error must fall
   linearly in `1/R0` across `R0 = {50, 100, 200}`. (Do not run `R0 = 1e6`:
   the `r^2 h^2 u_m` flux then carries ~1e12 scale factors; conditioning,
   not physics, would be measured.)
3. **Tip-thickness identity**: the closure implies, exactly,
   `dh_tip/dt = 1/(2 mu) - h_tip u_tip / (2 r_tip)`; planar limit recovers
   `h_tip = 1 + t*/4`, and the late-time axi limit is `h_tip -> t*/6`
   (azimuthal stretching thins the rim by exactly 2/3 relative to planar —
   the headline non-planar prediction, derived independently twice).
   Integrate the ODE alongside the run using the solver's own `u_tip`,
   `r_tip`; this is the axi analogue of the `h_tip_exact` column.
4. **Mass conservation**: exact in `m`; measured drift in ALE
   (`2 pi int h r dr`).
5. **Dissipation positivity**: the depth-averaged dissipation density
   `2 mu h [u_r^2 + (u/r)^2 + (u_r + u/r)^2]` is a positive-definite form
   (eigenvalues 1/2, 3/2); assemble it as a diagnostic — any hoop-term sign
   error makes it go negative immediately.
6. **High-Oh early growth**: `R(t) = R0 exp(t / (2 tau_vis))` (S&B 3.30) at
   `Oh_SB >> 1` — valid only in the small-hole regime
   `R0* = R0/(Oh_SB h0) << 1`. The opposite regime `R0* >> 1` follows the
   planar `sqrt(t*/pi)` start instead; note the Basilisk-matching DNS point
   (`Oh_SB = 0.035`, `R0 = 1`) has `R0* = 28` — neither closed form is a
   clean target there, so use `R0*` as a regime label when choosing tests.
7. **Self-similarity**: matched `Oh h0/R0` at high Oh collapses `u*(t*)`
   (S&B section 3.3.1).
8. **Late-time**: `dR/dt -> sqrt(2)` with a correction decaying like
   `ell/R`, `ell = max(h0, Oh_SB h0)` — i.e. `1/t`, which DOMINATES the
   planar `t^-2` deficit. Test `(1 - u*) R -> const`. The planar late-time
   coefficients (3/2, `sqrt(Oh/t*)` law, 0.98073) are planar-only and must
   never be fitted to axi data.

## 1D-vs-DNS comparison protocol

The DNS IC (cap of radius 1/2 on a hole of radius `R0 = 1`) carries cap
volume the clean-hole 1D IC does not. Initialise the 1D solver at the
volume-matched radius

    R_eq^2 = r_s^2 - pi a r_s / 2 + 2 a^2 / 3,   r_s = R0 + a,  a = 1/2
    (R0 = 1  ->  R_eq = 1.1129)

and compare `dR/dt` rather than `R` (the 1D tip is the matched edge; the
DNS hole radius is the innermost interface point — an irreducible O(h/2)
offset). Expect the 1D model to miss the first few `tau_vis` at `R0 ~ 1`:
the azimuthal curvature is an O(1) fraction of the edge force there and the
default closure omits it — that is precisely what the optional
`-2 + pi h_tip/(2R)` tip-correction flag exists to measure.

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

The window formulation (`simulationCases/axisymmetric_comoving.py`) is a
first-class sibling of the lab annulus (`simulationCases/axisymmetric_lab.py`).
One bookkeeping global dof `X` is slaved to the tip by
`WeakContribution(mesh_x - X, Xtest, coordinate_system=cartesian)`. The far
edge is an `EnforcedBC` residual `r_far - (X + L) = 0`, not
`EnforcedDirichlet(mesh_x = X + L)`. The window still imposes a finite-domain
clamp, so it is not a substitute for checking that `L` contains the
disturbance footprint. The shifted-metric co-moving variant (measure depending
on a global unknown) was assessed and rejected: implementable, but it routes
the azimuthal curvature — the very quantity under study — through unexercised
custom machinery. A physical account, with schematics and the constraint
equations, is [`axi-comoving-window.html`](axi-comoving-window.html).

Smoke ladder before any long run (in order, all must pass):

1. **S0 hoop-curvature discriminator**: static quarter-disc droplet, axis at
   `r = 0`, stationary solve; require `p = 2 sigma/a` (axisymmetric) vs
   `sigma/a` (planar) — a factor-2 check that the azimuthal curvature is on.
2. **S1 FD Jacobian**: `debug_jacobian_by_fd_epsilon` over a few steps of
   the real case at coarse resolution.
3. **S2 volume conservation**: relative drift < 1e-6 over a coarse run.
   The drift is spatial-error-dominated at this resolution (measured
   ~1e-8), so the bound tests conservation rather than the mesh.
4. **S3 planar reduction**: `R0 = 100`, matched Oh; `dR/dt` must track the
   validated planar `U(t)` to `O(h0/R)`.
5. **S4 physics**: `R0 = 1`; `dR/dt > 0` from `t = 0+`, approaching
   `sqrt(2)` from below with a residual decaying like `1/R`.

## Validation targets (external)

- Basilisk `Taylor-Culick-ViscoElastic/simulationCases/TaylorCulick.c` run
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
    simulationCases/axisymmetric_comoving.py  # 2D DNS, translating window
    validation/trouton1d_axi_sweep.py
    validation/trouton1d_ale_axi_sweep.py
    validation/trouton1d_axi_checks.py     # mass-coordinate check suite
    validation/trouton1d_ale_axi_checks.py # ALE check suite
    validation/run_validation_axi.py       # mirrors run_validation.py columns
    validation/run_validation_axi_comoving.py  # co-moving window driver
    validation/axi_smoketests.py           # smoke ladder S0-S2 (+ quick S4 probe)

`TipFlux`/`TipClosure` are imported from the planar modules (geometry-
independent constants). Output tables keep the planar column prefix and
append new columns (`compare_1d_2d.py` indexes positionally).

## Formulation audits (2026-08-15)

Two independent source-level audits, run after the validation gates passed.

**Against Basilisk's `axi.h` (the metric-formulation reference).** Every
axisymmetric term Basilisk adds by hand — the `lambda` hoop-viscosity patch
in `viscosity.h`, the three separate `#if AXI` azimuthal-curvature additions
in `curvature.h`, the bespoke `tau_qq` hoop-stress field and its momentum
forcing in `log-conform.h` — corresponds one-to-one to terms pyoomph's
single generic tensor layer produces automatically (`vector_gradient`'s
`u_r/r` entry, `tensor_divergence`'s `(T_rr - T_thth)/r`, the surface
divergence). No physics is missing on the FEM side; when the viscoelastic
extension comes, the polymeric hoop stress arrives through `div(T)` with no
bespoke module. Candidates for the residual 1.6-1.8% offset against the
2024 Basilisk curves, ranked, all on the VOF side: height-function
curvature noise plus the explicit `Delta/2` floor on the azimuthal
curvature; the hoop viscous term being an acknowledged after-the-fact
solver patch; quadtree coarse/fine transitions at the rim; and the
two-phase ambient (`rho2 = 1e-3`) our single-phase formulation excludes.
The metric factors themselves are exact and ruled out.

**Against oomph-lib's hand-written axisymmetric elements (the FEM ground
truth).** The C residuals pyoomph generated for `axisymmetric_lab.py` are
term-for-term identical to `AxisymmetricNavierStokesEquations` (momentum
including the `u_r/r^2` hoop viscous and `p psi/r` hoop pressure terms,
continuity r-factors, ALE corrections, `Gamma = 1` stress-divergence form)
and to the axisymmetric `FluidInterfaceElement` (surface-divergence
capillary term with the `+psi/r` azimuthal-curvature injection), up to a
global 2*pi and a global sign. The omitted end-of-interface co-normal term
`[sigma m.v]` lands entirely in pinned equations at both ends — discarded
correctly, not lost. The setup matches pyoomph's canonical
`rayleigh_plateau.py` moving-mesh example, minus `AxisymmetryBC`, which the
annulus structurally never needs. The absolute Newton tolerance behind the
`eps R^2/dt` floor (hit three times in this strand) is inherited from
oomph-lib's `Problem::newton_solve` — no relative test exists upstream
either — so the per-step retune in `actions_before_newton_solve` is the
correct local remedy and a relative-tolerance option is upstream work.
