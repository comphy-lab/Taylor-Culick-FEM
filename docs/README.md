# Taylor–Culick FEM documentation

Public write-up of the sharp-interface FEM campaign. Simulation output does
not live here. Runs sit on the compute host; custody is in the project tracker.

The split is by question, not by geometry.

| document | remit | status |
|---|---|---|
| [`validation-viscous/`](validation-viscous/) | One-dimensional Trouton (long-wave) planar sheet in the viscous limit, checked against Ahsan *et al.* and against independent implementations of the same reduced model, including the solvers written here. Formerly `validation-1d/`. | settled |
| [`validation-inertial/`](validation-inertial/) | Two-dimensional inertial retraction, `Oh < 0.1`. Planar FEM against Basilisk, and the same comparison for the axisymmetric hole. | planar closed; axi running |
| [`planar-campaign/`](planar-campaign/) | Full planar Newtonian write-up at arbitrary Ohnesorge number (`planar-campaign.tex`). Method, co-moving frame, Savva & Bush, 1D versus 2D, and what the Oh sweep does and does not establish. | planar inertial closed; moderate-Oh material as written |
| [`axi-validation/`](axi-validation/) | Construction and checks of the four axisymmetric solvers (two depth-averaged, two free-surface Navier–Stokes). Solver V&V across viscous and inertial axi formulations. The inertial campaign comparison is [`validation-inertial/`](validation-inertial/). | settled as a solver note |
| [`axi-design.md`](axi-design.md) | Design authority for the axi formulations. | living design note |
| [`axi-comoving-window.html`](axi-comoving-window.html) | Self-contained explainer of the translating axi window. | living explainer |

`validation-1d` was renamed because the note is about the viscous-limit reduced
model, not about being one-dimensional as an end in itself. `axi-validation`
was not renamed to `validation-inertial`. That note covers viscous and
inertial axi solvers together, and renaming it would collapse two different
jobs into one label.

The leftover top-level `docs/README.md` and `docs/figures/` were the planar
Newtonian campaign write-up. They are still needed; they now live in
[`planar-campaign/`](planar-campaign/). `axi-design.md` and
`axi-comoving-window.html` stay at this level because several solvers and the
explainer point at them.

Start with [`validation-inertial/`](validation-inertial/) for the FEM–Basilisk
inertial comparison, and with [`validation-viscous/`](validation-viscous/) for
the Trouton/Ahsan work.
