# Axisymmetric construction checks

A ladder of short checks on the axisymmetric laboratory-frame solver
(`src/taylor_culick_fem/axisymmetric_lab.py`), each meant to fail loudly and cheaply
before a long run:

| Check | What it tests |
|---|---|
| `s0` | the hoop-curvature term, on a static quarter-disc droplet |
| `s1` | finite-difference Jacobian consistency on the real case, coarse and short |
| `s2` | volume conservation over a coarse run |
| `s4lite` | a quick physics probe: the hole opens and grows toward `V_TC` |

```bash
uv run python verificationCases/axisymmetric-construction/axi_smoketests.py all
```
